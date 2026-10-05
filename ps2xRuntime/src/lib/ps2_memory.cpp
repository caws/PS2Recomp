#include "runtime/ps2_pipe_capture.h"
#include "runtime/ps2_gs_pipeline.h"
bool ps2xVif1DirectSpanQuiescent();
extern "C" int ps2xGsGifParseQuiescent();
#include "runtime/ps2_memory.h"
#include "runtime/ps2_address.h"
#include "ipu/ps2x_ipu.h" // cont.245: the PCSX2 IPU (registers 0x10002000, FIFOs 0x10007000, DMA ch3/4)
#include "runtime/gs/gs_frontend.h"

// Display-flip presentation snapshot (cont.165): defined in gs_cpu_backend.cpp; called from the
// GS privileged-register write paths below when DISPFB1/2 changes (the guest's frame flip),
// with the PRE-write register pair (the completed frame's display state).
void ps2xGsNotifyDisplayFlip(uint64_t preFlipDispfb1, uint64_t preFlipDispfb2);
#include "ps2_log.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>   // cont.357f: std::align_val_t for the page-aligned GS VRAM
#include <limits>
#include <stdexcept>
#include <cstdio>

// ★ cont.344 PS2X_GIF_TRACE=<flip> [+ PS2X_GIF_TRACE_W, default 2] (default OFF, read-only): one line per
// GIF/VIF1 DMA kick, per packet handed to the arbiter (path, bytes, IMAGE?, held by MSKPATH3?), per
// MSKPATH3 set/clear and per masked-FIFO release, inside that flip window -- the ORDER of the texture
// stream's uploads against the draws and the mask, which the pagewatch/pixwatch pair could only show
// for the draws and the landed uploads.
extern "C" unsigned long long ps2xGsPerfFlips();
static const unsigned long long s_gifTraceFlip = []
{ const char *e = std::getenv("PS2X_GIF_TRACE"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
static const unsigned long long s_gifTraceW = []
{ const char *e = std::getenv("PS2X_GIF_TRACE_W"); unsigned long long v = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 2ull; return v ? v : 2ull; }();
bool ps2xGifTraceOn()
{
    if (s_gifTraceFlip == 0ull) return false;
    const unsigned long long f = ps2xGsPerfFlips();
    return f >= s_gifTraceFlip && f < s_gifTraceFlip + s_gifTraceW;
}

#include <algorithm>
#include <string>
#include <vector>

namespace
{
    inline void inRange(uint32_t offset, size_t bytes, size_t regionSize, const char *op, uint32_t address)
    {
        if (static_cast<uint64_t>(offset) + static_cast<uint64_t>(bytes) > static_cast<uint64_t>(regionSize))
        {
            throw std::runtime_error(std::string(op) + " out-of-bounds at address: 0x" + std::to_string(address));
        }
    }

    template <typename T>
    inline T loadScalar(const uint8_t *base, uint32_t offset, size_t regionSize, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        T value{};
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }

    template <typename T>
    inline void storeScalar(uint8_t *base, uint32_t offset, size_t regionSize, T value, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        std::memcpy(base + offset, &value, sizeof(T));
    }

    inline bool isGsPrivReg(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE);
    }

    inline bool isIoRegister(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_IO_BASE, PS2_IO_SIZE);
    }

    inline uint64_t *gsRegPtr(GSRegisters &gs, uint32_t addr)
    {
        // Support both 64-bit base offsets and +4 dword aliases.
        uint32_t off = (addr - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        switch (off)
        {
        case 0x0000:
            return &gs.pmode;
        case 0x0010:
            return &gs.smode1;
        case 0x0020:
            return &gs.smode2;
        case 0x0030:
            return &gs.srfsh;
        case 0x0040:
            return &gs.synch1;
        case 0x0050:
            return &gs.synch2;
        case 0x0060:
            return &gs.syncv;
        case 0x0070:
            return &gs.dispfb1;
        case 0x0080:
            return &gs.display1;
        case 0x0090:
            return &gs.dispfb2;
        case 0x00A0:
            return &gs.display2;
        case 0x00B0:
            return &gs.extbuf;
        case 0x00C0:
            return &gs.extdata;
        case 0x00D0:
            return &gs.extwrite;
        case 0x00E0:
            return &gs.bgcolor;
        // CSR (offset 0x1000) is intentionally not handled here: it is
        // std::atomic<uint64_t> and no longer converts to uint64_t*. Callers must
        // check for offset 0x1000 themselves and go through writeCsrHalf/
        // writeCsrFull/gs.csr.load() instead of gsRegPtr().
        case 0x1010:
            return &gs.imr;
        case 0x1040:
            return &gs.busdir;
        case 0x1080:
            return &gs.siglblid;
        default:
            return nullptr;
        }
    }

    constexpr uint32_t kGsCsrRegOffset = 0x1000u;

    // Atomically apply a 32-bit write to one half (off=0 low dword, off=4 high
    // dword) of the GS CSR register. Bits 0..1 of the low dword (SIGNAL/FINISH) are
    // write-one-to-clear; everything else is a plain merge. Uses compare_exchange
    // so the whole read-modify-write is a single atomic step -- this register is
    // also touched by the vsync worker (FIELD bit) and the GIF (SIGNAL/FINISH) on
    // other threads, so a load-then-store here would race with them.
    inline void writeCsrHalf(std::atomic<uint64_t> &csr, uint32_t off, uint32_t value)
    {
        constexpr uint32_t kW1cMask = 0x3u;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            if (off == 0u)
            {
                uint32_t oldLow = static_cast<uint32_t>(expected & 0xFFFFFFFFull);
                uint32_t mergedLow = (oldLow & kW1cMask) | (value & ~kW1cMask);
                desired = (expected & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(mergedLow);
                desired &= ~static_cast<uint64_t>(value & kW1cMask);
            }
            else
            {
                uint64_t mask = 0xFFFFFFFFull << (off * 8u);
                desired = (expected & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
            }
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    // Same as writeCsrHalf but for a full 64-bit CSR write (bits 0..1 are still
    // write-one-to-clear against the current value).
    inline void writeCsrFull(std::atomic<uint64_t> &csr, uint64_t value)
    {
        constexpr uint64_t kW1cMask = 0x3ull;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            desired = (expected & kW1cMask) | (value & ~kW1cMask);
            desired &= ~(value & kW1cMask);
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    constexpr std::array<uint32_t, 4> kEeTimerBases = {
        0x10000000u,
        0x10000800u,
        0x10001000u,
        0x10001800u,
    };
    constexpr uint32_t kEeTimerCountOffset = 0x00u;
    constexpr uint32_t kEeTimerModeOffset = 0x10u;
    constexpr uint32_t kEeTimerCompareOffset = 0x20u;
    constexpr uint32_t kEeTimerHoldOffset = 0x30u;
    constexpr uint32_t kEeTimerModeClksMask = 0x3u;
    constexpr uint32_t kEeTimerModeConfigMask = 0x3FFu;
    constexpr uint32_t kEeTimerModeStatusMask = 0xC00u;
    constexpr uint32_t kEeTimerModeZret = 1u << 6;
    constexpr uint32_t kEeTimerModeCue = 1u << 7;
    constexpr uint32_t kEeTimerModeCmpe = 1u << 8;
    constexpr uint32_t kEeTimerModeOvfe = 1u << 9;
    constexpr uint32_t kEeTimerModeEquf = 1u << 10;
    constexpr uint32_t kEeTimerModeOvff = 1u << 11;
    constexpr uint64_t kEeClockHz = 294912000ull;
    constexpr std::array<uint64_t, 4> kEeTimerClockHz = {
        147456000ull,
        9216000ull,
        576000ull,
        15734ull,
    };

    inline bool decodeEeTimerRegister(uint32_t address, size_t &timerIndex, uint32_t &offset)
    {
        for (size_t index = 0; index < kEeTimerBases.size(); ++index)
        {
            const uint32_t candidateOffset = address - kEeTimerBases[index];
            if (candidateOffset == kEeTimerCountOffset ||
                candidateOffset == kEeTimerModeOffset ||
                candidateOffset == kEeTimerCompareOffset ||
                (index < 2u && candidateOffset == kEeTimerHoldOffset))
            {
                timerIndex = index;
                offset = candidateOffset;
                return true;
            }
        }
        return false;
    }

    constexpr uint64_t ticksUntilMatch(uint32_t count, uint32_t target)
    {
        const uint32_t distance = (target - count) & 0xFFFFu;
        return distance == 0u ? 0x10000ull : static_cast<uint64_t>(distance);
    }

    struct DmaTagView
    {
        uint16_t qwc = 0;
        uint8_t id = 0;
        bool irq = false;
        uint32_t addr = 0;
        uint32_t upper = 0;
    };

    inline DmaTagView decodeDmaTag(uint64_t tag)
    {
        DmaTagView out{};
        out.qwc = static_cast<uint16_t>(tag & 0xFFFFu);
        out.id = static_cast<uint8_t>((tag >> 28u) & 0x7u);
        out.irq = ((tag >> 31u) & 0x1ull) != 0ull;
        out.addr = static_cast<uint32_t>((tag >> 32u) & 0x7FFFFFFFu);
        out.upper = static_cast<uint32_t>((tag >> 16u) & 0xFFFFu);
        return out;
    }

    inline uint32_t gifTagNloop(uint64_t tagLo)
    {
        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }

    inline uint8_t gifTagFlg(uint64_t tagLo)
    {
        return static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
    }

    inline uint32_t gifTagNreg(uint64_t tagLo)
    {
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        return nreg == 0u ? 16u : nreg;
    }

}

// Helpers for GS VRAM addressing (PSMCT32 path).
static inline uint32_t gs_vram_offset(uint32_t basePage, uint32_t x, uint32_t y, uint32_t fbw)
{
    // basePage is in 2048-byte units; fbw is in blocks of 64 pixels.
    uint32_t strideBytes = fbw * 64 * 4;
    return basePage * 2048 + y * strideBytes + x * 4;
}

PS2Memory::PS2Memory()
    : m_rdram(nullptr), m_scratchpad(nullptr), iop_ram(nullptr), m_seenGifCopy(false), m_gsVRAM(nullptr)
{
    ps2SetScratchpadHostPtr(nullptr);
}

PS2Memory::~PS2Memory()
{
    if (m_rdram)
    {
        delete[] m_rdram;
        m_rdram = nullptr;
    }

    if (m_scratchpad)
    {
        ps2SetScratchpadHostPtr(nullptr);
        delete[] m_scratchpad;
        m_scratchpad = nullptr;
    }

    if (m_gsVRAM)
    {
        // cont.357f: must match the aligned allocation above -- mixing aligned new[] with plain
        // delete[] is undefined behaviour, not a leak.
        if (m_gsVRAMAlign) ::operator delete[](m_gsVRAM, std::align_val_t(m_gsVRAMAlign));
        else delete[] m_gsVRAM;
        m_gsVRAM = nullptr;
    }

    if (m_vu1Code)
    {
        delete[] m_vu1Code;
        m_vu1Code = nullptr;
    }
    if (m_vu1Data)
    {
        delete[] m_vu1Data;
        m_vu1Data = nullptr;
    }
    if (m_vu0Code)
    {
        delete[] m_vu0Code;
        m_vu0Code = nullptr;
    }
    if (m_vu0Data)
    {
        delete[] m_vu0Data;
        m_vu0Data = nullptr;
    }

    if (iop_ram)
    {
        delete[] iop_ram;
        iop_ram = nullptr;
    }
}

bool PS2Memory::initialize(size_t ramSize)
{
    auto cleanup = [this]()
    {
        delete[] m_rdram;
        delete[] m_scratchpad;
        delete[] iop_ram;
        // cont.357f: the GS VRAM may be an ALIGNED allocation; free it the way it was made.
        if (m_gsVRAMAlign) ::operator delete[](m_gsVRAM, std::align_val_t(m_gsVRAMAlign));
        else delete[] m_gsVRAM;
        delete[] m_vu0Code;
        delete[] m_vu0Data;
        delete[] m_vu1Code;
        delete[] m_vu1Data;
        m_rdram = nullptr;
        m_scratchpad = nullptr;
        ps2SetScratchpadHostPtr(nullptr);
        iop_ram = nullptr;
        m_gsVRAM = nullptr;
        m_vu0Code = nullptr;
        m_vu0Data = nullptr;
        m_vu1Code = nullptr;
        m_vu1Data = nullptr;
    };

    cleanup();
    m_seenGifCopy = false;
    m_dmaStartCount.store(0, std::memory_order_relaxed);
    m_gifCopyCount.store(0, std::memory_order_relaxed);
    m_gsWriteCount.store(0, std::memory_order_relaxed);
    m_vifWriteCount.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_completedDmacMutex);
        m_completedDmacCauses.clear();
    }
    m_codeRegions.clear();
    m_path3Masked = false;
    m_path3MaskedFifo.clear();
    m_vif1PendingPath2ImageQwc = 0u;
    m_vif1PendingPath2DirectHl = false;
    {
        extern void ps2xVif1ResetDirectSpan();
        ps2xVif1ResetDirectSpan();
    }
    resetEeTimers();

    try
    {
        // Allocate main RAM
        m_rdram = new uint8_t[ramSize];
        std::memset(m_rdram, 0, ramSize);

        // Allocate scratchpad
        m_scratchpad = new uint8_t[PS2_SCRATCHPAD_SIZE];
        std::memset(m_scratchpad, 0, PS2_SCRATCHPAD_SIZE);
        ps2SetScratchpadHostPtr(m_scratchpad);

        // Initialize EE TLB entries (R5900 has 48 entries).
        m_tlbEntries.assign(48, TLBEntry{0, 0, 0, false});

        // Allocate IOP RAM
        iop_ram = new uint8_t[2 * 1024 * 1024]; // 2MB

        // Initialize IOP RAM with zeros
        std::memset(iop_ram, 0, 2 * 1024 * 1024);

        // Initialize I/O registers
        m_ioRegisters.clear();

        // Initialize GS registers
        memset(&gs_regs, 0, sizeof(gs_regs));
        // memset zero-fills std::atomic<uint64_t>::csr's bytes, which is not itself
        // a guaranteed-valid atomic store; make the zero-initialization explicit.
        gs_regs.csr.store(0);
        gs_regs.dispfb1 = (0ULL << 0) | (10ULL << 9) | (0ULL << 15) | (0ULL << 32) | (0ULL << 43);
        gs_regs.display1 = (0ULL << 0) | (0ULL << 12) | (0ULL << 23) | (0ULL << 27) | (639ULL << 32) | (447ULL << 44);
        gs_regs.dispfb2 = gs_regs.dispfb1;
        gs_regs.display2 = gs_regs.display1;

        // ★★★★ cont.357f GS VRAM IS PAGE-ALIGNED (PS2X_GS_VRAM_ALIGN, default 4096; "=0" restores the
        // plain new[]). `new uint8_t[4 MB]` lands 16 bytes past a page on glibc, and EVERY structure
        // the GS writes is naturally 64/256-byte sized -- a CT32 column is 64 bytes, a block 256 --
        // so a misaligned base makes each of those straddle TWO cache lines. Measured directly: with
        // the plain allocation, all 81,130,211 of cont.357e's row-pair column writes failed a
        // 64-byte alignment test ([gs2:upblock] ntUnaligned), i.e. not one of them was a single-line
        // write. The env gate exists so the A/B runs on ONE binary with the flag flipped.
        // ⚠ MEASURED NULL FOR SPEED: aligned vs plain is 64.2 s vs 64.3 s over the matched-scene
        // window (n=2, ranges 62.5-66.0 vs 64.2-64.4) and upMs is unchanged (4,332/4,342 vs
        // 4,257/4,356). Straddling costs nothing here because consecutive columns are CONTIGUOUS,
        // so a 16-byte shift changes the lines touched by one per RUN, not per write. It is kept
        // because it is correct hygiene and because PS2X_GS_UPSTREAM's non-temporal stores REQUIRE
        // it: unaligned, every one of those 81,130,211 columns was rejected by the alignment test.
        {
            const char *av = std::getenv("PS2X_GS_VRAM_ALIGN");
            m_gsVRAMAlign = (av && av[0]) ? static_cast<size_t>(std::strtoul(av, nullptr, 0)) : 4096u;
            if (m_gsVRAMAlign < alignof(std::max_align_t) || (m_gsVRAMAlign & (m_gsVRAMAlign - 1u)) != 0u)
                m_gsVRAMAlign = 0u;   // 0 = plain new[], and anything not a sane power of two falls back to it
            m_gsVRAM = m_gsVRAMAlign
                           ? static_cast<uint8_t *>(::operator new[](PS2_GS_VRAM_SIZE, std::align_val_t(m_gsVRAMAlign)))
                           : new uint8_t[PS2_GS_VRAM_SIZE];
        }
        std::memset(m_gsVRAM, 0, PS2_GS_VRAM_SIZE);

        m_vu0Code = new uint8_t[PS2_VU0_CODE_SIZE];
        m_vu0Data = new uint8_t[PS2_VU0_DATA_SIZE];
        std::memset(m_vu0Code, 0, PS2_VU0_CODE_SIZE);
        std::memset(m_vu0Data, 0, PS2_VU0_DATA_SIZE);

        m_vu1Code = new uint8_t[PS2_VU1_CODE_SIZE];
        m_vu1Data = new uint8_t[PS2_VU1_DATA_SIZE];
        std::memset(m_vu1Code, 0, PS2_VU1_CODE_SIZE);
        std::memset(m_vu1Data, 0, PS2_VU1_DATA_SIZE);
        markVU0CodeModified();
        markVU1CodeModified();

        // Initialize VIF registers
        memset(&vif0_regs, 0, sizeof(vif0_regs));
        memset(&vif1_regs, 0, sizeof(vif1_regs));

        // Initialize DMA registers
        memset(dma_regs, 0, sizeof(dma_regs));

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error initializing PS2 memory: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void PS2Memory::resetEeTimers() noexcept
{
    m_eeTimers = {};
    m_eeTimerPendingCycles = 0ull;   // cont.317
    m_eeTimerLatchedInterrupts = 0u;
}

uint32_t PS2Memory::advanceEeTimersFlushAndTake() noexcept
{
    // cont.317: batch (see the header); the accumulate/early-return half is inline there.
    flushEeTimers();
    const uint32_t latched = m_eeTimerLatchedInterrupts;
    m_eeTimerLatchedInterrupts = 0u;
    return latched;
}

void PS2Memory::flushEeTimers() noexcept
{
    const uint64_t eeCycles = m_eeTimerPendingCycles;
    m_eeTimerPendingCycles = 0ull;
    if (eeCycles == 0u)
    {
        return;
    }

    uint32_t interruptMask = 0u;
    for (size_t index = 0; index < m_eeTimers.size(); ++index)
    {
        EeTimer &timer = m_eeTimers[index];
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t wholeSeconds = eeCycles / kEeClockHz;
        const uint64_t remainingCycles = eeCycles % kEeClockHz;
        const uint64_t scaled = remainingCycles * clockHz + timer.clockRemainder;
        const uint64_t ticks = wholeSeconds * clockHz + scaled / kEeClockHz;
        timer.clockRemainder = scaled % kEeClockHz;
        if (ticks == 0u)
        {
            continue;
        }

        const uint32_t oldCount = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(oldCount, compare);
        const uint64_t overflowDistance = 0x10000ull - oldCount;
        const bool zeroReturn = (timer.mode & kEeTimerModeZret) != 0u;
        const bool compareReached = ticks >= compareDistance;
        bool overflowReached = false;

        if (zeroReturn)
        {
            overflowReached = ticks >= overflowDistance && overflowDistance <= compareDistance;
            if (compareReached)
            {
                const uint64_t remaining = ticks - compareDistance;
                timer.count = compare == 0u
                                  ? static_cast<uint32_t>(remaining & 0xFFFFu)
                                  : static_cast<uint32_t>(remaining % compare);
            }
            else
            {
                timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
            }
        }
        else
        {
            overflowReached = ticks >= overflowDistance;
            timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
        }

        if (compareReached && (timer.mode & kEeTimerModeCmpe) != 0u && (timer.mode & kEeTimerModeEquf) == 0u)
        {
            timer.mode |= kEeTimerModeEquf;
            interruptMask |= 1u << index;
        }
        if (overflowReached && (timer.mode & kEeTimerModeOvfe) != 0u && (timer.mode & kEeTimerModeOvff) == 0u)
        {
            timer.mode |= kEeTimerModeOvff;
            interruptMask |= 1u << index;
        }
    }
    m_eeTimerLatchedInterrupts |= interruptMask;
}

uint64_t PS2Memory::cyclesUntilNextEeTimerInterrupt() noexcept
{
    flushEeTimers(); // cont.317: the deadline is computed from exact counts
    uint64_t nearest = std::numeric_limits<uint64_t>::max();
    for (const EeTimer &timer : m_eeTimers)
    {
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint32_t count = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(count, compare);
        const uint64_t overflowDistance = 0x10000ull - count;
        uint64_t eventTicks = std::numeric_limits<uint64_t>::max();

        if ((timer.mode & kEeTimerModeCmpe) != 0u &&
            (timer.mode & kEeTimerModeEquf) == 0u)
        {
            eventTicks = compareDistance;
        }
        const bool overflowCanOccur = (timer.mode & kEeTimerModeZret) == 0u ||
                                      overflowDistance <= compareDistance;
        if (overflowCanOccur &&
            (timer.mode & kEeTimerModeOvfe) != 0u &&
            (timer.mode & kEeTimerModeOvff) == 0u)
        {
            eventTicks = std::min(eventTicks, overflowDistance);
        }
        if (eventTicks == std::numeric_limits<uint64_t>::max())
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t numerator = eventTicks * kEeClockHz - timer.clockRemainder;
        const uint64_t cycles = (numerator + clockHz - 1u) / clockHz;
        nearest = std::min(nearest, std::max<uint64_t>(1u, cycles));
    }
    return nearest;
}

bool PS2Memory::isScratchpad(uint32_t address) const
{
    return ps2IsScratchpadAddress(address);
}

uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit)
{
    return const_cast<uint8_t *>(static_cast<const PS2Memory *>(this)->mapVuMemory(physAddr, size, offset, limit));
}

const uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const
{
    auto mapRange = [&](uint32_t base, uint32_t rangeSize, const uint8_t *ptr) -> const uint8_t *
    {
        if (!ptr || physAddr < base)
        {
            return nullptr;
        }
        const uint32_t local = physAddr - base;
        if (local >= rangeSize || size > (rangeSize - local))
        {
            return nullptr;
        }
        offset = local;
        limit = rangeSize;
        return ptr;
    };

    if (const uint8_t *ptr = mapRange(PS2_VU0_CODE_BASE, PS2_VU0_CODE_SIZE, m_vu0Code))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU0_DATA_BASE, PS2_VU0_DATA_SIZE, m_vu0Data))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU1_CODE_BASE, PS2_VU1_CODE_SIZE, m_vu1Code))
    {
        return ptr;
    }
    return mapRange(PS2_VU1_DATA_BASE, PS2_VU1_DATA_SIZE, m_vu1Data);
}

uint32_t PS2Memory::translateAddress(uint32_t virtualAddress)
{
    if (isScratchpad(virtualAddress))
    {
        return ps2ScratchpadOffset(virtualAddress);
    }

    // EE uncached aliases of main RAM (per PS2 memory map):
    //   0x20000000-0x3FFFFFFF -> 32MB mirror of RDRAM
    // This includes the accelerated window rooted at 0x30100000.
    if (Ps2IsUncachedRamMirrorAddress(virtualAddress))
    {
        return virtualAddress & PS2_RAM_MASK;
    }

    // KSEG0/KSEG1 direct-mapped window.
    if (Ps2IsKseg01Address(virtualAddress))
    {
        return Ps2DirectMappedPhysicalAddress(virtualAddress);
    }

    // In this runtime, low segments are treated as physical-style addresses already.
    if (virtualAddress < 0x80000000)
    {
        return virtualAddress;
    }

    // KSEG2/KSEG3 are TLB mapped.
    if (Ps2IsKseg23Address(virtualAddress))
    {
        for (const auto &entry : m_tlbEntries)
        {
            if (entry.valid)
            {
                // PageMask uses bits [24:13]. Build an address-level mask (plus 4KB base page bits).
                const uint32_t mask = entry.mask & 0x01FFE000u;
                const uint32_t compareMask = ~(mask | 0xFFFu);
                if ((virtualAddress & compareMask) == (entry.vpn & compareMask))
                {
                    // TLB hit
                    const uint32_t pageOffsetMask = mask | 0xFFFu;
                    const uint32_t physBase = entry.pfn << 12;
                    return physBase | (virtualAddress & pageOffsetMask);
                }
            }
        }
        throw std::runtime_error("TLB miss for address: 0x" + std::to_string(virtualAddress));
    }

    return virtualAddress;
}

bool PS2Memory::tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    const TLBEntry &entry = m_tlbEntries[index];
    vpn = entry.vpn;
    pfn = entry.pfn;
    mask = entry.mask;
    valid = entry.valid;
    return true;
}

bool PS2Memory::tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid)
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    TLBEntry &entry = m_tlbEntries[index];
    entry.vpn = vpn & 0xFFFFF000u;
    entry.pfn = pfn & 0x000FFFFFu;
    entry.mask = mask & 0x01FFE000u;
    entry.valid = valid;
    return true;
}

int32_t PS2Memory::tlbProbe(uint32_t vpn) const
{
    const uint32_t normalizedVpn = vpn & 0xFFFFF000u;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlbEntries.size()); ++i)
    {
        const TLBEntry &entry = m_tlbEntries[i];
        if (!entry.valid)
        {
            continue;
        }

        const uint32_t mask = entry.mask & 0x01FFE000u;
        const uint32_t compareMask = ~(mask | 0xFFFu);
        if ((normalizedVpn & compareMask) == (entry.vpn & compareMask))
        {
            return static_cast<int32_t>(i);
        }
    }

    return -1;
}

uint8_t PS2Memory::read8(uint32_t address)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return m_scratchpad[physAddr];
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return m_rdram[physAddr];
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (ps2gs::threaded() && (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        ps2gs::fence(1); // cont.317 stage 2: the consumer owns VU1 memory
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
    {
        (void)vuLimit;
        return vuMem[vuOffset];
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 3) * 8;
        return static_cast<uint8_t>((value >> shift) & 0xFF);
    }

    return 0;
}

uint16_t PS2Memory::read16(uint32_t address)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read16 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read16 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (ps2gs::threaded() && (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        ps2gs::fence(1); // cont.317 stage 2: the consumer owns VU1 memory
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
    {
        return loadScalar<uint16_t>(vuMem, vuOffset, vuLimit, "read16 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 2) * 8;
        return static_cast<uint16_t>((value >> shift) & 0xFFFF);
    }

    return 0;
}

uint32_t PS2Memory::read32(uint32_t address)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            uint64_t val = gs_regs.csr.load();
            return (uint32_t)(val >> (off * 8));
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        if (!reg)
            return 0;
        uint64_t val = *reg;
        return (uint32_t)(val >> (off * 8));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read32 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read32 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (ps2gs::threaded() && (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        ps2gs::fence(1); // cont.317 stage 2: the consumer owns VU1 memory
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
    {
        return loadScalar<uint32_t>(vuMem, vuOffset, vuLimit, "read32 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        return readIORegister(physAddr);
    }

    return 0;
}

uint64_t PS2Memory::read64(uint32_t address)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return gs_regs.csr.load();
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        return reg ? *reg : 0;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read64 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read64 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (ps2gs::threaded() && (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        ps2gs::fence(1); // cont.317 stage 2: the consumer owns VU1 memory
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
    {
        return loadScalar<uint64_t>(vuMem, vuOffset, vuLimit, "read64 vu", address);
    }

    // cont.247: the IPU register block is read 64-bit by libmpeg (`ld` IPU_CMD / IPU_TOP: bit 63 = BUSY,
    // bits 0-31 = the VDEC/FDEC result) -- PCSX2 HwRead.cpp _hwRead64 page 0x02 -> ipuRead64. The
    // generic compose below would return the stale m_ioRegisters slots (the seam never writes them).
    if (address >= 0x10002000u && address <= 0x10002030u && ps2x_ipu::enabled())
    {
        return ps2x_ipu::read64(address);
    }

    // 64-bit IO read: compose from the two adjacent 32-bit IO register slots
    // to avoid any side-effects from read32 handlers.
    if (isIoRegister(address))
    {
        uint32_t lo = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t hi = m_ioRegisters.count(address + 4) ? m_ioRegisters[address + 4] : 0u;
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return (uint64_t)read32(address) | ((uint64_t)read32(address + 4) << 32);
}

__m128i PS2Memory::read128(uint32_t address)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "read128 scratchpad", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]));
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "read128 rdram", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]));
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (ps2gs::threaded() && (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        ps2gs::fence(1); // cont.317 stage 2: the consumer owns VU1 memory
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
    {
        inRange(vuOffset, sizeof(__m128i), vuLimit, "read128 vu", address);
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(vuMem + vuOffset));
    }

    if (physAddr == 0x10007000u && ps2x_ipu::enabled())
    {
        // cont.245: the IPU output FIFO (a 128-bit register).
        alignas(16) uint32_t qw[4] = {0, 0, 0, 0};
        ps2x_ipu::readFifoOut(qw);
        syncIpuDmaChannels();
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(qw));
    }
    // 128-bit reads are primarily for quad-word loads in the EE, which are only valid for RAM areas
    // Return zeroes for unsupported areas
    return _mm_setzero_si128();
}

void PS2Memory::write8(uint32_t address, uint8_t value)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        m_scratchpad[physAddr] = value;
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        m_rdram[physAddr] = value;
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
        {
            (void)vuLimit;
            if (ps2gs::producerSide() && (vuMem == m_vu1Data || vuMem == m_vu1Code)) // cont.317 stage 1
            {
                // the consumer performs the store (stage 2: the only writer of VU1 memory)
                ps2gs::enqueueVu1Mem(this, vuMem == m_vu1Code, vuOffset, vuMem + vuOffset, 1u);
                ps2gs::pump();
            }
            else
                vuMem[vuOffset] = value;
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // IO registers - handle byte writes by modifying the appropriate byte in the word
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 3) * 8;
        uint32_t mask = ~(0xFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write16(uint32_t address, uint16_t value)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write16 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        storeScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write16 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
        {
            if (ps2gs::producerSide() && (vuMem == m_vu1Data || vuMem == m_vu1Code)) // cont.317 stage 1
            {
                // the consumer performs the store (stage 2: the only writer of VU1 memory)
                ps2gs::enqueueVu1Mem(this, vuMem == m_vu1Code, vuOffset, vuMem + vuOffset, sizeof(uint16_t));
                ps2gs::pump();
            }
            else
                storeScalar<uint16_t>(vuMem, vuOffset, vuLimit, value, "write16 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 2) * 8;
        uint32_t mask = ~(0xFFFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write32(uint32_t address, uint32_t value)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 of the low dword are write-one-to-clear status bits.
            // Done as a single atomic RMW -- see writeCsrHalf's comment.
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            uint64_t mask = 0xFFFFFFFFULL << (off * 8);
            uint64_t newVal = (*reg & ~mask) | ((uint64_t)value << (off * 8));
            const bool changed = *reg != newVal;
            // DISPFB1/2 flip = the guest just finished a frame in the buffer the PRE-write
            // registers point at -- snapshot it for presentation (cont.165 menu-text fix).
            const uint64_t preFb1 = gs_regs.dispfb1, preFb2 = gs_regs.dispfb2;
            *reg = newVal;
            if (changed && (regOff == 0x0070u || regOff == 0x0090u))
                ps2xGsNotifyDisplayFlip(preFb1, preFb2);
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write32 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        // Check if this might be code modification
        markModified(address, 4);

        storeScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write32 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
        {
            if (ps2gs::producerSide() && (vuMem == m_vu1Data || vuMem == m_vu1Code)) // cont.317 stage 1
            {
                // the consumer performs the store (stage 2: the only writer of VU1 memory)
                ps2gs::enqueueVu1Mem(this, vuMem == m_vu1Code, vuOffset, vuMem + vuOffset, sizeof(uint32_t));
                ps2gs::pump();
            }
            else
                storeScalar<uint32_t>(vuMem, vuOffset, vuLimit, value, "write32 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        writeIORegister(physAddr, value);
    }
}

void PS2Memory::write64(uint32_t address, uint64_t value)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 are write-one-to-clear status bits. Done as a single
            // atomic RMW -- see writeCsrFull's comment.
            writeCsrFull(gs_regs.csr, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const bool changed = *reg != value;
            const uint64_t preFb1 = gs_regs.dispfb1, preFb2 = gs_regs.dispfb2;
            *reg = value;
            if (changed && (regOff == 0x0070u || regOff == 0x0090u))
                ps2xGsNotifyDisplayFlip(preFb1, preFb2); // display flip -> presentation snapshot (cont.165)
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write64 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 8);
        storeScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write64 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
        {
            if (ps2gs::producerSide() && (vuMem == m_vu1Data || vuMem == m_vu1Code)) // cont.317 stage 1
            {
                // the consumer performs the store (stage 2: the only writer of VU1 memory)
                ps2gs::enqueueVu1Mem(this, vuMem == m_vu1Code, vuOffset, vuMem + vuOffset, sizeof(uint64_t));
                ps2gs::pump();
            }
            else
                storeScalar<uint64_t>(vuMem, vuOffset, vuLimit, value, "write64 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // cont.247: a 64-bit IPU register write is one command (PCSX2 HwWrite.cpp -> ipuWrite64), not two
        // 32-bit stores (the second would land on the reserved +4 slot).
        if (address >= 0x10002000u && address <= 0x10002030u && ps2x_ipu::enabled())
        {
            ps2x_ipu::write64(address, value);
            syncIpuDmaChannels();
            return;
        }
        write32(address, (uint32_t)value);
        write32(address + 4, (uint32_t)(value >> 32));
    }
}

void PS2Memory::write128(uint32_t address, __m128i value)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "write128 scratchpad", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]), value);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 16);
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "write128 rdram", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]), value);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
        {
            inRange(vuOffset, sizeof(__m128i), vuLimit, "write128 vu", address);
            if (ps2gs::producerSide() && (vuMem == m_vu1Data || vuMem == m_vu1Code)) // cont.317 stage 1
            {
                // the consumer performs the store (stage 2: the only writer of VU1 memory)
                ps2gs::enqueueVu1Mem(this, vuMem == m_vu1Code, vuOffset, vuMem + vuOffset, 16u);
                ps2gs::pump();
            }
            else
                _mm_storeu_si128(reinterpret_cast<__m128i *>(vuMem + vuOffset), value);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (physAddr == 0x10007010u && ps2x_ipu::enabled())
    {
        // cont.245: the IPU input FIFO (a 128-bit register); the libipu SETIQ/SETVQ path writes it.
        alignas(16) uint32_t qw[4];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(qw), value);
        ps2x_ipu::writeFifoIn(qw);
        syncIpuDmaChannels();
        return;
    }
    if (isIoRegister(physAddr))
    {
        // Non-RAM 128-bit stores are modeled as two 64-bit stores.
        uint64_t lo = _mm_extract_epi64(value, 0);
        uint64_t hi = _mm_extract_epi64(value, 1);

        write64(address, lo);
        write64(address + 8, hi);
    }
}

bool PS2Memory::writeIORegister(uint32_t address, uint32_t value)
{
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        flushEeTimers(); // cont.317: apply the cycles that precede this write first
        EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            timer.count = value & 0xFFFFu;
            timer.clockRemainder = 0u;
            break;
        case kEeTimerModeOffset:
        {
            const uint32_t previousMode = timer.mode;
            const uint32_t status = (previousMode & kEeTimerModeStatusMask) &~(value & kEeTimerModeStatusMask);
            timer.mode = (value & kEeTimerModeConfigMask) | status;
            if (((previousMode ^ timer.mode) & (kEeTimerModeClksMask | kEeTimerModeCue)) != 0u)
            {
                timer.clockRemainder = 0u;
            }
            break;
        }
        case kEeTimerCompareOffset:
            timer.compare = value & 0xFFFFu;
            break;
        case kEeTimerHoldOffset:
            timer.hold = value & 0xFFFFu;
            break;
        default:
            return false;
        }
        return true;
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from write8/16/32/64 today since those all funnel IO
        // register writes through addresses in PS2_IO_BASE's range, which is
        // disjoint from PS2_GS_PRIV_REG_BASE; kept correct for direct callers.
        m_ioRegisters[address] = value;
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t mask = 0xFFFFFFFFull << (off * 8u);
            const uint64_t newVal = (*reg & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
            const bool changed = *reg != newVal;
            const uint64_t preFb1 = gs_regs.dispfb1, preFb2 = gs_regs.dispfb2;
            *reg = newVal;
            if (changed && (regOff == 0x0070u || regOff == 0x0090u))
                ps2xGsNotifyDisplayFlip(preFb1, preFb2); // display flip -> presentation snapshot (cont.165)
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (ps2x_ipu::enabled())
        {
            ps2x_ipu::write32(address, value);
            syncIpuDmaChannels();
            return true;
        }
        if (address == 0x10002010)
        {
            m_ioRegisters[address] = value & ~(1u << 31);
            if (value & (1u << 30))
            {
                m_ioRegisters[0x10002000] = 0;
                m_ioRegisters[0x10002020] = 0;
                m_ioRegisters[0x10002030] = 0;
            }
        }
        else
        {
            m_ioRegisters[address] = value;
        }
        return true;
    }

    if (address == 0x1000E010u)
    {
        const uint32_t current = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t status = current & 0x3FFu;
        uint32_t mask = (current >> 16) & 0x3FFu;

        // D_STAT low bits are W1C status, high bits [16..25] toggle masks on write-one.
        status &= ~(value & 0x3FFu);
        mask ^= ((value >> 16) & 0x3FFu);

        uint32_t next = (current & ~((0x3FFu) | (0x3FFu << 16) | (1u << 31)));
        next |= status | (mask << 16);
        if ((status & mask) != 0u)
            next |= (1u << 31);
        m_ioRegisters[address] = next;
        return true;
    }

    if (!ps2gs::consumerSide()) // cont.317 stage 2: the consumer re-enters for the VIF1 block; the map is the EE's
        m_ioRegisters[address] = value;

    if (address >= 0x10003C00u && address < 0x10003E00u)
    {
        if (ps2gs::producerSide()) // cont.317 stage 1: VIF1 register-block writes are ring commands
        {
            ps2gs::enqueueVifReg(this, address, value);
            ps2gs::pump();
            return true;
        }
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        // PS2X_VIF_CTLLOG: log VIF1 control-register writes (FBRST/STAT/ERR/MARK) -- which
        // reset mechanism does the game use between frame chains? (cont.156 span-carry hunt)
        {
            static const bool s_ctlLog = []
            { const char *e = std::getenv("PS2X_VIF_CTLLOG"); return e && e[0] && e[0] != '0'; }();
            if (s_ctlLog && address != 0x10003C00u + 0x100u)
            {
                static unsigned long s_n = 0;
                ++s_n;
                if (s_n <= 60u || (s_n % 512u) == 0u)
                    std::fprintf(stderr, "[vif1:ctl] #%lu addr=%08X val=%08X\n",
                                 s_n, address, value);
            }
        }
        switch (address)
        {
        case 0x10003C10u:     // VIF1_FBRST
            if (value & 0x1u) // RST
            {
                std::memset(&vif1_regs, 0, sizeof(vif1_regs));
                m_vif1PendingPath2ImageQwc = 0u;
                m_vif1PendingPath2DirectHl = false;
                {
                    extern void ps2xVif1ResetDirectSpan();
                    ps2xVif1ResetDirectSpan();
                }
            }
            if (value & 0x8u) // STC
            {
                vif1_regs.stat &= ~((1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13));
            }
            break;
        case 0x10003C30u:
            vif1_regs.mark = value & 0xFFFFu;
            vif1_regs.stat &= ~(1u << 6); // clear MRK flag on CPU write
            break;
        case 0x10003C40u:
            vif1_regs.cycle = value & 0xFFFFu;
            break;
        case 0x10003C50u:
            vif1_regs.mode = value & 0x3u;
            break;
        case 0x10003C60u:
            vif1_regs.num = value & 0xFFu;
            break;
        case 0x10003C70u:
            vif1_regs.mask = value;
            break;
        case 0x10003C80u:
            vif1_regs.code = value;
            break;
        case 0x10003C90u:
            vif1_regs.itops = value & 0x3FFu;
            break;
        case 0x10003CA0u:
            vif1_regs.base = value & 0x3FFu;
            break;
        case 0x10003CB0u:
            vif1_regs.ofst = value & 0x3FFu;
            break;
        case 0x10003CC0u:
            vif1_regs.tops = value & 0x3FFu;
            break;
        case 0x10003CD0u:
            vif1_regs.itop = value & 0x3FFu;
            break;
        case 0x10003CE0u:
            vif1_regs.top = value & 0x3FFu;
            break;
        default:
            break;
        }

        return true;
    }

    if (address >= 0x10003800u && address < 0x10003A00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10008000 && address < 0x1000F000)
    {
        // cont.247: a CHCR write with STR CLEAR on an IPU channel stops it in place (PCSX2 Dmac.cpp
        // DmaExec: only STR drops, MADR/QWC/TADR keep the progress). libipu's sceIpuStopDMA does exactly
        // this to park the bitstream chain on D4 while the CSC borrows the channel, then reads the
        // channel back; before this the write only landed in m_ioRegisters and the seam kept the chain
        // running, so a later FIFO underrun would have refilled the CSC input from the bitstream.
        if ((address == 0x1000B000u || address == 0x1000B400u) && (value & 0x100u) == 0u && ps2x_ipu::enabled())
        {
            const int ch = (address == 0x1000B000u) ? 3 : 4;
            uint32_t chcr = value;
            ps2x_ipu::dmaWriteStopped(ch, &chcr);
            m_ioRegisters[address] = chcr;
            syncIpuDmaChannels();
            return true;
        }
        if ((address & 0xFF) == 0x00 && (value & 0x100))
        {
            const auto dctrlIt = m_ioRegisters.find(0x1000E000u);
            const bool dmacEnabled = (dctrlIt == m_ioRegisters.end()) || ((dctrlIt->second & 0x1u) != 0u);
            if (!dmacEnabled)
            {
                return true;
            }

            const uint32_t channelBase = address & 0xFFFFFF00;
            const uint32_t madr = m_ioRegisters[channelBase + 0x10];
            const uint32_t qwc = m_ioRegisters[channelBase + 0x20];
            m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);

            if ((channelBase == 0x1000A000u || channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                (m_gsVRAM || channelBase == 0x10008000u))
            {
                auto enqueueTransfer = [&](uint32_t srcAddr, uint32_t qwCount)
                {
                    if (qwCount == 0)
                        return;
                    const bool scratch = isScratchpad(srcAddr);
                    PendingTransfer pt;
                    pt.fromScratchpad = scratch;
                    pt.srcAddr = srcAddr;
                    pt.qwc = qwCount;
                    if (channelBase == 0x1000A000u)
                        m_pendingGifTransfers.push_back(pt);
                    else if (channelBase == 0x10009000u)
                        m_pendingVif1Transfers.push_back(pt);
                    else if (channelBase == 0x10008000u)
                        m_pendingVif0Transfers.push_back(pt);
                };

                uint32_t chcr = value;
                uint32_t mode = (chcr >> 2) & 0x3;
                if (ps2xGifTraceOn())
                    std::fprintf(stderr, "[gif:trace] flip=%llu DMA kick ch=%s mode=%u qwc=%u madr=0x%x masked=%d\n",
                                 ps2xGsPerfFlips(), (channelBase == 0x1000A000u) ? "GIF" : (channelBase == 0x10009000u) ? "VIF1" : "VIF0",
                                 mode, qwc, madr, m_path3Masked ? 1 : 0);

                if (mode == 0 && qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }
                else if (mode == 1)
                {
                    uint32_t tagAddr = m_ioRegisters[channelBase + 0x30];
                    uint32_t asr0 = m_ioRegisters[channelBase + 0x40];
                    uint32_t asr1 = m_ioRegisters[channelBase + 0x50];
                    uint32_t asp = (chcr >> 4) & 0x3u;
                    const bool tieEnabled = (chcr & (1u << 7)) != 0u;
                    const bool tteEnabled = (chcr & (1u << 6)) != 0u; // Tag Transfer Enable
                    // Cycle 76: 4096 was severing real chains mid-walk ([DMA:chain] end=maxtags
                    // with strip refs still pending), silently dropping every descriptor after
                    // the cut -- i.e. the TAIL of a long display list. PCSX2 walks a source
                    // chain until its END/REFE tag (`DMAC` chain handlers in pcsx2/Dmac.cpp /
                    // hwDmac*), with no arbitrary tag ceiling; the cap here exists only as a
                    // runaway guard against a malformed self-referencing chain, so raise it far
                    // above any legitimate display list rather than letting it truncate one.
                    // PS2X_DMA_MAXCHAINTAGS overrides it for A/B.
                    static const int kMaxChainTags = [] {
                        if (const char *e = std::getenv("PS2X_DMA_MAXCHAINTAGS"))
                        {
                            const int v = std::atoi(e);
                            if (v > 0)
                                return v;
                        }
                        return 1 << 20;
                    }();
                    // ★ rotk row 254: a VIF1 chain is built in a buffer recycled from the pipeline ring's pool
                    // (it travels into the ring by move and comes back after execution), so it keeps its
                    // capacity: no reallocation chain, no page faults on fresh multi-MB memory. perf (LBR,
                    // GameThread, 60-fps fight): this insert path was 5.4% of all samples. PS2X_DMA_CHAINPOOL=0 = off.
                    static const bool s_chainPool = []
                    { const char *e = std::getenv("PS2X_DMA_CHAINPOOL"); return !(e && e[0] == '0'); }();
                    std::vector<uint8_t> chainBuf = (s_chainPool && channelBase == 0x10009000u && ps2gs::threaded())
                                                        ? ps2gs::takeBuffer() : std::vector<uint8_t>();

                    auto appendData = [&](uint32_t srcAddr, uint32_t qwCount)
                    {
                        const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
                        uint32_t bytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
                        const bool scratch = isScratchpad(srcAddr);
                        uint32_t src = 0;
                        src = translateAddress(srcAddr);
                        const uint8_t *base2;
                        uint32_t maxSz2;
                        if (scratch)
                        {
                            base2 = m_scratchpad;
                            maxSz2 = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            base2 = m_rdram;
                            maxSz2 = PS2_RAM_SIZE;
                        }

                        while (bytes > 0)
                        {
                            if (src >= maxSz2)
                                src = 0;
                            uint32_t chunk = bytes;
                            if (src + chunk > maxSz2)
                                chunk = maxSz2 - src;
                            if (chunk == 0)
                                break;
                            chainBuf.insert(chainBuf.end(), base2 + src, base2 + src + chunk);
                            bytes -= chunk;
                            src += chunk;
                        }
                    };

                    // ★ Cycle-86 fix (PS2X_DMA_TTEQW, default ON, `=0` legacy): embed a TTE tag as a
                    // FULL 16-byte qword [0,0,vif0,vif1] instead of the bare 8-byte upper half. The
                    // EE DMAC transfers the whole 128-bit tag slot (the VIF reads the lower half as
                    // two NOPs -- PCSX2 Vif1_Dma.cpp `masked_tag._u64[0] = 0`); a bare 8-byte embed
                    // shifts the stream phase by HALF A QWORD per tag, which word-granular UNPACK
                    // streams tolerate but qword-granular DIRECT framing does not. Measured
                    // (run_c86A): a data word at half-phase pos=2104 parsed as DIRECT imm=0
                    // (=65536 qw) and swallowed the remaining 199 KB of the level chain raw -- the
                    // entire level texture-desync family. The game's dual-purpose tag words
                    // ([FLUSH][DIRECT] that also decode as a harmless GIF nloop=0 tag) only work
                    // with parity-preserving 16-byte slots.
                    static const bool s_tteQw = [] {
                        const char *e = std::getenv("PS2X_DMA_TTEQW");
                        return !(e && e[0] == '0'); // default ON (correctness fix)
                    }();

                    auto appendCompactVif1TagData = [&](uint32_t localTagAddr, uint32_t qwCount)
                    {
                        uint32_t tagPhys = 0u;
                        const bool tagScratch = isScratchpad(localTagAddr);
                        tagPhys = translateAddress(localTagAddr);

                        const uint8_t *localBase = tagScratch ? m_scratchpad : m_rdram;
                        const uint32_t localMax = tagScratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                        if (tagPhys + 16u > localMax)
                            return;

                        // VIF packet helpers embed 8 bytes of VIF stream in the DMAtag's upper half.
                        if (s_tteQw)
                            chainBuf.insert(chainBuf.end(), 8u, uint8_t(0));
                        chainBuf.insert(chainBuf.end(), localBase + tagPhys + 8u, localBase + tagPhys + 16u);
                        appendData(localTagAddr + 16u, qwCount);
                    };

                    int tagsProcessed = 0;
                    uint32_t lastTagUpper = (chcr >> 16) & 0xFFFFu;

                    while (tagsProcessed < kMaxChainTags)
                    {
                        const uint32_t currentTagAddr = tagAddr;
                        const bool tagInSPR = isScratchpad(tagAddr);
                        uint32_t physTag = 0;
                        try
                        {
                            physTag = translateAddress(tagAddr);
                        }
                        catch (...)
                        {
                            break;
                        }
                        const uint8_t *tagBase;
                        uint32_t tagMax;
                        if (tagInSPR)
                        {
                            tagBase = m_scratchpad;
                            tagMax = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            tagBase = m_rdram;
                            tagMax = PS2_RAM_SIZE;
                        }
                        if (physTag + 16 > tagMax)
                            break;

                        const uint8_t *tp = tagBase + physTag;
                        uint64_t tag = loadScalar<uint64_t>(tp, 0, 16, "dma chain tag", tagAddr);
                        uint16_t tagQwc = static_cast<uint16_t>(tag & 0xFFFF);
                        uint32_t id = static_cast<uint32_t>((tag >> 28) & 0x7);
                        const bool irq = ((tag >> 31) & 0x1ull) != 0ull;
                        uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFFF);
                        lastTagUpper = static_cast<uint32_t>((tag >> 16) & 0xFFFFu);
                        ++tagsProcessed;

                        uint32_t dataAddr = 0;
                        bool hasPayload = (tagQwc > 0);
                        bool endChain = false;

                        switch (id)
                        {
                        case 0:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        case 1:
                            dataAddr = tagAddr + 16;
                            tagAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                            break;
                        case 2:
                            dataAddr = tagAddr + 16;
                            tagAddr = addr;
                            break;
                        case 3:
                        case 4:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            break;
                        case 5:
                            dataAddr = tagAddr + 16;
                            {
                                const uint32_t retAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                                if (asp == 0u)
                                {
                                    asr0 = retAddr;
                                    asp = 1u;
                                }
                                else if (asp == 1u)
                                {
                                    asr1 = retAddr;
                                    asp = 2u;
                                }
                            }
                            tagAddr = addr;
                            break;
                        case 6:
                            dataAddr = tagAddr + 16;
                            if (asp == 2u)
                            {
                                tagAddr = asr1;
                                asp = 1u;
                            }
                            else if (asp == 1u)
                            {
                                tagAddr = asr0;
                                asp = 0u;
                            }
                            else
                            {
                                endChain = true;
                            }
                            break;
                        case 7:
                            dataAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        default:
                            hasPayload = false;
                            endChain = true;
                            break;
                        }

                        const bool compactVifLocalTag =
                            (channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                            (id == 1u || id == 2u || id == 5u || id == 6u || id == 7u);
                        if (compactVifLocalTag)
                            appendCompactVif1TagData(currentTagAddr, 0u);

                        // ★ Cycle 84 FIX: VIF source-chain ref tags (id 3/4) with TTE carry 2 VIF
                        // codes in the DMAtag's upper 64 bits -- for LOTR's display list that is the
                        // `DIRECT imm=qwc` that WRAPS the ref'd GIF packet (probe: qwc==imm exactly).
                        // Hardware sends those codes to VIF before the ref'd data (TTE, CHCR bit 6);
                        // we were dropping them, feeding raw GIF data into the VIF command parser ->
                        // the GIF parse desync that leaves level textures untextured (cont.137). This
                        // is the compact-tag embed we already do for id 1/2/5/6/7, extended to refs.
                        // PCSX2 sends qword-0 (the tag's VIF codes) when TTE is set for every chain
                        // tag. Kill switch PS2X_DMA_REFVIF=0.
                        static const bool s_refVif = [] {
                            const char *e = std::getenv("PS2X_DMA_REFVIF");
                            return !(e && e[0] == '0'); // default ON (correctness fix)
                        }();
                        if (s_refVif && tteEnabled &&
                            (channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                            (id == 3u || id == 4u) && (physTag + 16u <= tagMax))
                        {
                            if (s_tteQw)
                                chainBuf.insert(chainBuf.end(), 8u, uint8_t(0)); // qword parity (see PS2X_DMA_TTEQW above)
                            chainBuf.insert(chainBuf.end(), tp + 8, tp + 16);
                        }

                        if (hasPayload)
                        {
                            if (compactVifLocalTag)
                                appendData(currentTagAddr + 16u, tagQwc);
                            else
                                appendData(dataAddr, tagQwc);
                        }
                        if (irq && tieEnabled)
                            endChain = true;
                        // PS2X_VIF_SEQLOG: end-of-chain forensics for big VIF1 chains -- which
                        // tag ended the walk (id/qwc/irq) and how many tags were processed.
                        if (endChain || tagsProcessed >= kMaxChainTags)
                        {
                            static const bool s_endLog = []
                            { const char *e = std::getenv("PS2X_VIF_SEQLOG"); return e && e[0] && e[0] != '0'; }();
                            if (s_endLog && chainBuf.size() > 3000000u)
                                std::fprintf(stderr,
                                             "[vif1:chainend] bytes=%zu tags=%d lastTag{id=%u qwc=%u irq=%d addr=%08X} tie=%d\n",
                                             chainBuf.size(), tagsProcessed, id, (unsigned)tagQwc,
                                             irq ? 1 : 0, addr, tieEnabled ? 1 : 0);
                        }
                        if (endChain)
                            break;
                    }

                    m_ioRegisters[channelBase + 0x30] = tagAddr;
                    m_ioRegisters[channelBase + 0x40] = asr0;
                    m_ioRegisters[channelBase + 0x50] = asr1;
                    chcr = (chcr & ~(0x3u << 4)) | ((asp & 0x3u) << 4);
                    chcr = (chcr & 0x0000FFFFu) | (lastTagUpper << 16);
                    m_ioRegisters[channelBase + 0x00] = chcr;

                    if (!chainBuf.empty())
                    {
                        PendingTransfer pt;
                        pt.fromScratchpad = false;
                        pt.srcAddr = 0;
                        pt.qwc = 0;
                        pt.chainData = std::move(chainBuf);
                        if (channelBase == 0x1000A000)
                        {
                            m_pendingGifTransfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10009000u)
                        {
                            m_pendingVif1Transfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10008000u)
                        {
                            m_pendingVif0Transfers.push_back(std::move(pt));
                        }
                    }
                    // else if (channelBase == 0x10009000u)
                    // {

                    // }
                }
                else if (qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }

                const bool autoProcessTransfers =
                    (channelBase == 0x1000A000u) ? (m_gifPacketCallback || m_gifArbiter != nullptr) : true;
                if (autoProcessTransfers)
                {
                    processPendingTransfers();
                }
            }
            // ---- cont.245: IPU DMA: ch3 fromIPU @0x1000B000, ch4 toIPU @0x1000B400 ----
            // The PCSX2 IPU port owns these channels: the toIPU chain feeds the decoder's input FIFO
            // and the fromIPU normal-mode transfer drains its output FIFO into RAM. Both run to
            // quiescence inside this write (or inside the IPU command write that later produces the
            // output); syncIpuDmaChannels mirrors the channel back and raises its DMAC interrupt.
            else if ((channelBase == 0x1000B000u || channelBase == 0x1000B400u) && ps2x_ipu::enabled())
            {
                const int ch = (channelBase == 0x1000B000u) ? 3 : 4;
                uint32_t chcr = value;
                uint32_t madrV = m_ioRegisters[channelBase + 0x10];
                uint32_t qwcV = m_ioRegisters[channelBase + 0x20];
                uint32_t tadrV = m_ioRegisters[channelBase + 0x30];
                uint32_t asr0V = m_ioRegisters[channelBase + 0x40];
                uint32_t asr1V = m_ioRegisters[channelBase + 0x50];
                bool finished = false;
                ps2x_ipu::dmaKick(ch, &chcr, &madrV, &qwcV, &tadrV, &asr0V, &asr1V, &finished);
                m_ioRegisters[channelBase + 0x00] = chcr;
                m_ioRegisters[channelBase + 0x10] = madrV;
                m_ioRegisters[channelBase + 0x20] = qwcV;
                m_ioRegisters[channelBase + 0x30] = tadrV;
                m_ioRegisters[channelBase + 0x40] = asr0V;
                m_ioRegisters[channelBase + 0x50] = asr1V;
                if (finished)
                {
                    static constexpr uint32_t D_STAT_REG = 0x1000E010u;
                    uint32_t dstat = m_ioRegisters.count(D_STAT_REG) ? m_ioRegisters[D_STAT_REG] : 0u;
                    dstat |= (1u << ch);
                    const uint32_t status = dstat & 0x3FFu;
                    const uint32_t mask = (dstat >> 16) & 0x3FFu;
                    if ((status & mask) != 0u)
                        dstat |= (1u << 31);
                    else
                        dstat &= ~(1u << 31);
                    m_ioRegisters[D_STAT_REG] = dstat;
                    queueCompletedDmacCause(static_cast<uint32_t>(ch));
                }
                syncIpuDmaChannels(); // the other channel may have moved too
            }
            // ---- Scratchpad (SPR) DMA: ch8 fromSPR @0x1000D000, ch9 toSPR @0x1000D400 ----
            // Copies between the 16KB EE scratchpad (SADR) and main RAM (MADR), mirroring PCSX2's
            // SPR.cpp: NORMAL (_SPR0chain/_SPR1chain), CHAIN (fromSPR = destination chain whose tags
            // are read from the scratchpad at SADR, IDs CNTS/CNT/END -- _dmaSPR0; toSPR = source chain
            // through TADR with the standard tag IDs -- _dmaSPR1 + Hw.cpp hwDmacSrcChain) and INTERLEAVE
            // (D_SQWC TQWC/SQWC -- _SPR0interleave/_SPR1interleave). SADR wraps inside the 16KB
            // (memcpy_to_spr/memcpy_from_spr). The whole transfer completes inside this write.
            //
            // History: normal mode arrived first because LOTR builds each text GIF packet in the scratchpad
            // and copies it out with a fromSPR DMA (13F030; strings over ~11 glyphs were dropped while the
            // channel was a no-op). cont.247: libmpeg's motion compensation fetches the REFERENCE
            // macroblocks with a toSPR CHAIN (dmaRefImage 0x116538: CHCR 0x105, QWC 0, TADR -> a REF/REFE
            // tag list it writes with `sd`, SADR = the scratchpad slot); with only normal mode the QWC-0
            // kick copied nothing and every P/B picture was predicted from stale scratchpad data (the
            // I-frames, which need no references, decoded right).
            else if (channelBase == 0x1000D000u || channelBase == 0x1000D400u)
            {
                static const bool s_sprLog = []
                { const char *e = std::getenv("PS2X_SPR_LOG"); return e && e[0] && e[0] != '0'; }();
                const bool fromSpr = (channelBase == 0x1000D000u);
                const uint32_t mode = (value >> 2) & 0x3u;
                const bool tie = (value & (1u << 7)) != 0u;
                const bool tte = (value & (1u << 6)) != 0u;
                uint32_t sadr = m_ioRegisters[channelBase + 0x80];
                uint32_t madrV = madr;
                uint32_t qwcV = qwc & 0xFFFFu;
                uint32_t tadrV = m_ioRegisters[channelBase + 0x30];
                uint32_t asr0 = m_ioRegisters[channelBase + 0x40];
                uint32_t asr1 = m_ioRegisters[channelBase + 0x50];
                uint32_t asp = (value >> 4) & 0x3u;
                uint32_t chcrOut = value;
                if (s_sprLog)
                    std::fprintf(stderr, "[spr] %s kick chcr=0x%08x mode=%u madr=0x%08x qwc=%u tadr=0x%08x sadr=0x%08x\n",
                                 fromSpr ? "fromSPR" : "toSPR", value, mode, madrV, qwcV, tadrV, sadr);
                if (m_scratchpad && m_rdram)
                {
                    // PCSX2 SPR.cpp SPRdmaGetAddr: the 0x70000000 mirror addresses the scratchpad, everything
                    // else is main RAM (VU memory is not modelled here -> skipped).
                    auto hostPtr = [&](uint32_t guest, uint32_t bytes) -> uint8_t *
                    {
                        if ((guest & 0x70000000u) == 0x70000000u)
                            return m_scratchpad + (guest & 0x3FF0u);
                        uint32_t phys = 0;
                        try { phys = translateAddress(guest); } catch (...) { return nullptr; }
                        if (phys >= PS2_RAM_SIZE || bytes > PS2_RAM_SIZE - phys) return nullptr;
                        return m_rdram + phys;
                    };
                    // PCSX2 memcpy_to_spr / memcpy_from_spr: SADR wraps at 16KB.
                    auto copyToSpr = [&](uint32_t dst, const uint8_t *src, uint32_t size)
                    {
                        dst &= PS2_SCRATCHPAD_SIZE - 1u;
                        if (dst + size >= PS2_SCRATCHPAD_SIZE)
                        {
                            const uint32_t first = PS2_SCRATCHPAD_SIZE - dst;
                            std::memcpy(m_scratchpad + dst, src, first);
                            std::memcpy(m_scratchpad, src + first, size - first);
                        }
                        else
                            std::memcpy(m_scratchpad + dst, src, size);
                    };
                    auto copyFromSpr = [&](uint8_t *dst, uint32_t src, uint32_t size)
                    {
                        src &= PS2_SCRATCHPAD_SIZE - 1u;
                        if (src + size >= PS2_SCRATCHPAD_SIZE)
                        {
                            const uint32_t first = PS2_SCRATCHPAD_SIZE - src;
                            std::memcpy(dst, m_scratchpad + src, first);
                            std::memcpy(dst + first, m_scratchpad, size - first);
                        }
                        else
                            std::memcpy(dst, m_scratchpad + src, size);
                    };
                    // One block of qwcV qwords between MADR and SADR (SPR1transfer / _SPR0chain body).
                    auto transferBlock = [&]()
                    {
                        if (qwcV == 0u) return;
                        const uint32_t bytes = qwcV * 16u;
                        uint8_t *ram = hostPtr(madrV, bytes);
                        if (ram)
                        {
                            if (fromSpr) copyFromSpr(ram, sadr, bytes);
                            else copyToSpr(sadr, ram, bytes);
                        }
                        else if (s_sprLog)
                            std::fprintf(stderr, "[spr] WARN madr 0x%08x (%u qw) not addressable; skipped\n", madrV, qwcV);
                        madrV += bytes;
                        sadr = (sadr + bytes) & (PS2_SCRATCHPAD_SIZE - 1u);
                        qwcV = 0u;
                    };
                    if (mode == 0u)
                    {
                        transferBlock();
                    }
                    else if (mode == 1u)
                    {
                        // A kick with QWC > 0 in chain mode resumes an interrupted block first (PCSX2 _dmaSPR0/_dmaSPR1).
                        transferBlock();
                        bool done = false;
                        int guard = 0;
                        while (!done && ++guard < (1 << 16))
                        {
                            uint32_t tag[4] = {0, 0, 0, 0};
                            if (fromSpr)
                            {
                                // Destination chain: the tag lives in the scratchpad at SADR (_dmaSPR0).
                                std::memcpy(tag, m_scratchpad + (sadr & 0x3FF0u), 16);
                                sadr = (sadr + 16u) & (PS2_SCRATCHPAD_SIZE - 1u);
                                qwcV = tag[0] & 0xFFFFu;
                                madrV = tag[1];
                                chcrOut = (chcrOut & 0xFFFFu) | (tag[0] & 0xFFFF0000u);
                                const uint32_t id = (tag[0] >> 28) & 0x7u;
                                if (s_sprLog)
                                    std::fprintf(stderr, "[spr]   dst tag %08x_%08x id=%u qwc=%u madr=0x%08x sadr=0x%08x\n", tag[1], tag[0], id, qwcV, madrV, sadr);
                                switch (id)
                                {
                                case 0: // CNTS
                                case 1: // CNT
                                    break;
                                case 7: // END
                                    done = true;
                                    break;
                                default:
                                    if (s_sprLog) std::fprintf(stderr, "[spr] WARN fromSPR chain: unexpected tag id %u -> ending\n", id);
                                    done = true;
                                    break;
                                }
                                transferBlock();
                            }
                            else
                            {
                                // Source chain: the tag is at TADR (_dmaSPR1 + hwDmacSrcChain).
                                const uint8_t *tp = hostPtr(tadrV, 16u);
                                if (!tp)
                                {
                                    if (s_sprLog) std::fprintf(stderr, "[spr] WARN toSPR chain: tadr 0x%08x not addressable -> ending\n", tadrV);
                                    break;
                                }
                                std::memcpy(tag, tp, 16);
                                qwcV = tag[0] & 0xFFFFu;
                                chcrOut = (chcrOut & 0xFFFFu) | (tag[0] & 0xFFFF0000u);
                                const uint32_t id = (tag[0] >> 28) & 0x7u;
                                const uint32_t addr = tag[1];
                                if (s_sprLog)
                                    std::fprintf(stderr, "[spr]   src tag %08x_%08x id=%u qwc=%u tadr=0x%08x sadr=0x%08x\n", tag[1], tag[0], id, qwcV, tadrV, sadr);
                                if (tte)
                                    copyToSpr(sadr, tp, 16u), sadr = (sadr + 16u) & (PS2_SCRATCHPAD_SIZE - 1u); // SPR1transfer(ptag, 1)
                                bool cntTag = false;
                                switch (id)
                                {
                                case 0: // REFE
                                    madrV = addr; tadrV += 16u; done = true; break;
                                case 1: // CNT
                                    madrV = tadrV + 16u; cntTag = true; break;
                                case 2: // NEXT
                                    madrV = tadrV + 16u; tadrV = addr; break;
                                case 3: // REF
                                case 4: // REFS
                                    madrV = addr; tadrV += 16u; break;
                                case 5: // CALL
                                {
                                    madrV = tadrV + 16u;
                                    const uint32_t ret = madrV + qwcV * 16u;
                                    if (asp == 0u) { asr0 = ret; asp = 1u; }
                                    else if (asp == 1u) { asr1 = ret; asp = 2u; }
                                    tadrV = addr;
                                    break;
                                }
                                case 6: // RET
                                    madrV = tadrV + 16u;
                                    if (asp == 2u) { tadrV = asr1; asp = 1u; }
                                    else if (asp == 1u) { tadrV = asr0; asp = 0u; }
                                    else done = true;
                                    break;
                                case 7: // END
                                default:
                                    madrV = tadrV + 16u; done = true; break;
                                }
                                transferBlock();
                                if (cntTag) tadrV = madrV; // hwDmacSrcTadrInc: the next tag follows the data
                            }
                            if (tie && (tag[0] & 0x80000000u)) done = true; // TIE && tag.IRQ
                        }
                        chcrOut = (chcrOut & ~0x30u) | (asp << 4);
                    }
                    else
                    {
                        // INTERLEAVE: D_SQWC.TQWC qwords transferred, then SQWC qwords of RAM skipped, per row.
                        const uint32_t sqwcReg = m_ioRegisters.count(0x1000E030u) ? m_ioRegisters[0x1000E030u] : 0u;
                        uint32_t tqwc = (sqwcReg >> 16) & 0xFFu;
                        const uint32_t skip = sqwcReg & 0xFFu;
                        uint32_t remaining = qwcV;
                        if (tqwc == 0u) tqwc = remaining;
                        if (s_sprLog) std::fprintf(stderr, "[spr]   interleave tqwc=%u sqwc=%u\n", tqwc, skip);
                        while (remaining > 0u)
                        {
                            qwcV = std::min(tqwc, remaining);
                            remaining -= qwcV;
                            const uint32_t block = qwcV;
                            transferBlock();          // advances madr by block*16
                            madrV += skip * 16u;      // + the skipped qwords ((sqwc + qwc) * 16 in PCSX2)
                            (void)block;
                        }
                        qwcV = 0u;
                    }
                }
                // Completion: STR clears, the registers show the transfer's end (PCSX2 SPRFROMinterrupt /
                // SPRTOinterrupt: `spr0ch.chcr.STR = false; hwDmacIrq(DMAC_FROM_SPR)`).
                m_ioRegisters[channelBase] = chcrOut & ~0x100u;
                m_ioRegisters[channelBase + 0x10] = madrV;
                m_ioRegisters[channelBase + 0x20] = 0u;
                m_ioRegisters[channelBase + 0x30] = tadrV;
                m_ioRegisters[channelBase + 0x40] = asr0;
                m_ioRegisters[channelBase + 0x50] = asr1;
                m_ioRegisters[channelBase + 0x80] = sadr;
                {
                    const uint32_t ch = fromSpr ? 8u : 9u;
                    static constexpr uint32_t D_STAT_REG = 0x1000E010u;
                    uint32_t dstat = m_ioRegisters.count(D_STAT_REG) ? m_ioRegisters[D_STAT_REG] : 0u;
                    dstat |= (1u << ch);
                    const uint32_t status = dstat & 0x3FFu;
                    const uint32_t mask = (dstat >> 16) & 0x3FFu;
                    if ((status & mask) != 0u) dstat |= (1u << 31); else dstat &= ~(1u << 31);
                    m_ioRegisters[D_STAT_REG] = dstat;
                    queueCompletedDmacCause(ch);
                }
            }
        }
        return true;
    }

    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10000200 && address < 0x10000300)
        {
            return true;
        }
        if (address >= 0x10000000 && address < 0x10000100)
        {
            return true;
        }
    }

    return false;
}

// ---- EE subsystem profile (PS2X_EE_PROF, default OFF; read-only). cont.252: VU0 (16.7 ms/frame)
// and VU1 (11.9 ms) have hard numbers from dedicated ns counters, but VIF1 (~9 ms) and DMA (~6 ms)
// were only ever PROFILE-SHARE estimates -- and this project has mis-priced a subsystem from leaf
// share before (cont.234 put VU0 at "~1%"; the slow knob later showed 28%). These are the same kind
// of direct ns accumulators that produced the trustworthy VU0/VU1 figures.
// NESTING: gif is called from inside vif1 (Path2) and from inside the DMA drain (Path3); vif1 is
// called from inside the DMA drain. All three totals are INCLUSIVE, so the report also prints the
// derived exclusive split. Call counts are printed so the clock overhead (2 steady_clock reads,
// ~40 ns/call) can be bounded rather than guessed.
std::atomic<unsigned long long> g_eeProfDmaNs{0}, g_eeProfDmaCalls{0};
std::atomic<unsigned long long> g_eeProfVif1Ns{0}, g_eeProfVif1Calls{0}, g_eeProfVif1Bytes{0};
std::atomic<unsigned long long> g_eeProfGifNs{0}, g_eeProfGifCalls{0}, g_eeProfGifBytes{0};
extern const bool g_eeProf;
static const bool g_gifCensusPkt = []
{ const char *e = std::getenv("PS2X_GIF_CENSUS"); return e && e[0] && e[0] != '0'; }();
extern const bool g_eeProf = []
{ const char *e = std::getenv("PS2X_EE_PROF"); return e && e[0] && e[0] != '0'; }();
static const int g_eeProfEvery = []
{ const char *e = std::getenv("PS2X_EE_PROF_EVERY"); const int v = e && e[0] ? std::atoi(e) : 10; return v > 0 ? v : 10; }();

static void eeProfReport()
{
    // Wall-clock gated, never call-gated (cont.251 lesson: a count gate never fired and a
    // timeout(1) kill skips any end-of-run report).
    static std::chrono::steady_clock::time_point s_t0{}, s_last{};
    const auto now = std::chrono::steady_clock::now();
    if (s_t0.time_since_epoch().count() == 0)
    {
        s_t0 = s_last = now;
        std::fprintf(stderr, "[ee:prof] active (PS2X_EE_PROF=1, every %ds)\n", g_eeProfEvery);
        return;
    }
    if (now - s_last < std::chrono::seconds(g_eeProfEvery))
    {
        return;
    }
    s_last = now;
    const double wall = std::chrono::duration_cast<std::chrono::nanoseconds>(now - s_t0).count() / 1e9;
    const double dma = double(g_eeProfDmaNs.load(std::memory_order_relaxed)) / 1e9;
    const double vif = double(g_eeProfVif1Ns.load(std::memory_order_relaxed)) / 1e9;
    const double gif = double(g_eeProfGifNs.load(std::memory_order_relaxed)) / 1e9;
    const unsigned long long dc = g_eeProfDmaCalls.load(std::memory_order_relaxed);
    const unsigned long long vc = g_eeProfVif1Calls.load(std::memory_order_relaxed);
    const unsigned long long gc = g_eeProfGifCalls.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[ee:prof] wall=%.1fs | INCLUSIVE dma=%.2fs (%.1f%%) vif1=%.2fs (%.1f%%) gif=%.2fs (%.1f%%)"
                 " | EXCLUSIVE dma-only=%.2fs vif1-only=%.2fs gif=%.2fs\n",
                 wall, dma, 100.0 * dma / wall, vif, 100.0 * vif / wall, gif, 100.0 * gif / wall,
                 dma - vif > 0 ? dma - vif : 0.0, vif - gif > 0 ? vif - gif : 0.0, gif);
    std::fprintf(stderr,
                 "[ee:prof] calls dma=%llu vif1=%llu gif=%llu | per-call dma=%.1fus vif1=%.1fus gif=%.1fus"
                 " | bytes vif1=%llu gif=%llu | clock-overhead<=%.2fs\n",
                 dc, vc, gc,
                 dc ? dma * 1e6 / dc : 0.0, vc ? vif * 1e6 / vc : 0.0, gc ? gif * 1e6 / gc : 0.0,
                 g_eeProfVif1Bytes.load(std::memory_order_relaxed),
                 g_eeProfGifBytes.load(std::memory_order_relaxed),
                 double(dc + vc + gc) * 40.0 / 1e9);
}

void PS2Memory::processPendingTransfers()
{
    if (ps2pipe::capEnabled()) // cont.317 pipeline oracle: start at a quiescent pass boundary
        ps2pipe::capPassBegin(*this, m_vif1PendingPath2ImageQwc == 0u && !m_path3Masked && m_path3MaskedFifo.empty() &&
                                         ps2xVif1DirectSpanQuiescent() && ps2xGsGifParseQuiescent() != 0);
    struct PipeCapPassEnd { ~PipeCapPassEnd() { ps2pipe::capPassEnd(); } } pipeCapPassEnd__;
    const auto eeProfT0 = g_eeProf ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    struct EeProfDmaScope
    {
        std::chrono::steady_clock::time_point t0;
        ~EeProfDmaScope()
        {
            if (!g_eeProf) return;
            g_eeProfDmaNs.fetch_add(static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
            g_eeProfDmaCalls.fetch_add(1, std::memory_order_relaxed);
            eeProfReport();
        }
    } eeProfDmaScope{eeProfT0};
    const bool hadGif = !m_pendingGifTransfers.empty();
    for (size_t idx = 0; idx < m_pendingGifTransfers.size(); ++idx)
    {
        auto &p = m_pendingGifTransfers[idx];
        if (!p.chainData.empty())
        {
            m_seenGifCopy = true;
            m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
            submitGifPacket(GifPathId::Path3, p.chainData.data(), static_cast<uint32_t>(p.chainData.size()), false);
        }
        else if (p.qwc > 0)
        {
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            uint32_t srcPhys = 0;
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    m_seenGifCopy = true;
                    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
                    submitGifPacket(GifPathId::Path3, m_scratchpad + srcPhys, chunk, false);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    m_seenGifCopy = true;
                    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
                    submitGifPacket(GifPathId::Path3, m_rdram + srcPhys, chunk, false);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingGifTransfers.clear();

    const bool hadVif0 = !m_pendingVif0Transfers.empty();
    for (auto &p : m_pendingVif0Transfers)
    {
        if (!p.chainData.empty())
        {
            processVIF0Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif0Transfers.clear();

    const bool hadVif1 = !m_pendingVif1Transfers.empty();
    for (auto &p : m_pendingVif1Transfers)
    {
        if (!p.chainData.empty())
        {
            // ★ rotk row 254: the chain buffer is already a private copy -- hand it to the ring by move
            // (processVIF1Data(ptr) would copy it again: `takePayload`, 2.6% of all samples).
            static const bool s_vif1Move = []
            { const char *e = std::getenv("PS2X_DMA_CHAINPOOL"); return !(e && e[0] == '0'); }();
            if (s_vif1Move && ps2gs::producerSide())
            {
                ps2gs::enqueueVif1Owned(this, std::move(p.chainData));
                ps2gs::pump();
            }
            else
                processVIF1Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif1Transfers.clear();

    if (ps2gs::producerSide()) // cont.317 stage 1: the end-of-pass drain is a ring command
    {
        ps2gs::enqueueDrain(this);
        ps2gs::pump();
    }
    else if (m_gifArbiter)
        m_gifArbiter->drain();

    static constexpr uint32_t GIF_CHANNEL = 0x1000A000;
    static constexpr uint32_t VIF0_CHANNEL = 0x10008000;
    static constexpr uint32_t VIF1_CHANNEL = 0x10009000;
    static constexpr uint32_t D_STAT = 0x1000E010u;

    auto raiseDStatChannel = [&](uint32_t channelBit)
    {
        uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
        dstat |= (1u << channelBit);

        const uint32_t status = dstat & 0x3FFu;
        const uint32_t mask = (dstat >> 16) & 0x3FFu;
        if ((status & mask) != 0u)
            dstat |= (1u << 31);
        else
            dstat &= ~(1u << 31);

        m_ioRegisters[D_STAT] = dstat;
    };

    if (hadGif)
    {
        raiseDStatChannel(2u); // GIF channel
        queueCompletedDmacCause(2u);
        m_ioRegisters[GIF_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[GIF_CHANNEL + 0x20] = 0;
    }
    if (hadVif0)
    {
        raiseDStatChannel(0u); // VIF0 channel
        queueCompletedDmacCause(0u);
        m_ioRegisters[VIF0_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF0_CHANNEL + 0x20] = 0;
    }
    if (hadVif1)
    {
        raiseDStatChannel(1u); // VIF1 channel
        queueCompletedDmacCause(1u);
        m_ioRegisters[VIF1_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF1_CHANNEL + 0x20] = 0;
    }
}

void PS2Memory::queueCompletedDmacCause(uint32_t cause)
{
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    m_completedDmacCauses.push_back(cause);
}

void PS2Memory::syncIpuDmaChannels()
{
    static constexpr uint32_t D_STAT_REG = 0x1000E010u;
    for (int ch = 3; ch <= 4; ++ch)
    {
        const uint32_t channelBase = (ch == 3) ? 0x1000B000u : 0x1000B400u;
        uint32_t chcr = 0, madrV = 0, qwcV = 0, tadrV = 0;
        bool finished = false;
        ps2x_ipu::channelState(ch, &chcr, &madrV, &qwcV, &tadrV, &finished);
        m_ioRegisters[channelBase + 0x00] = chcr;
        m_ioRegisters[channelBase + 0x10] = madrV;
        m_ioRegisters[channelBase + 0x20] = qwcV;
        m_ioRegisters[channelBase + 0x30] = tadrV;
        if (finished)
        {
            uint32_t dstat = m_ioRegisters.count(D_STAT_REG) ? m_ioRegisters[D_STAT_REG] : 0u;
            dstat |= (1u << ch);
            const uint32_t status = dstat & 0x3FFu;
            const uint32_t mask = (dstat >> 16) & 0x3FFu;
            if ((status & mask) != 0u)
                dstat |= (1u << 31);
            else
                dstat &= ~(1u << 31);
            m_ioRegisters[D_STAT_REG] = dstat;
            queueCompletedDmacCause(static_cast<uint32_t>(ch));
        }
    }
}

std::vector<uint32_t> PS2Memory::consumeCompletedDmacCauses()
{
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    std::vector<uint32_t> causes;
    causes.swap(m_completedDmacCauses);
    return causes;
}


void PS2Memory::flushMaskedPath3Packets(bool drainImmediately)
{
    if (ps2xGifTraceOn() && !m_path3Masked && !m_path3MaskedFifo.empty())
        std::fprintf(stderr, "[gif:trace] flip=%llu PATH3 masked-FIFO release: %zu packets\n", ps2xGsPerfFlips(), m_path3MaskedFifo.size());
    if (m_path3Masked || m_path3MaskedFifo.empty())
        return;

    auto emit = [&](const uint8_t *packetData, uint32_t packetSize)
    {
        if (m_gifArbiter)
            m_gifArbiter->submit(GifPathId::Path3, packetData, packetSize, false);
        else if (m_gifPacketCallback)
            m_gifPacketCallback(packetData, packetSize);
    };

    for (const auto &packet : m_path3MaskedFifo)
    {
        if (packet.size() >= 16u)
            emit(packet.data(), static_cast<uint32_t>(packet.size()));
    }
    m_path3MaskedFifo.clear();

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately, bool path2DirectHl)
{
    struct EeProfGifScope
    {
        std::chrono::steady_clock::time_point t0;
        uint32_t bytes;
        ~EeProfGifScope()
        {
            if (!g_eeProf) return;
            g_eeProfGifNs.fetch_add(static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
            g_eeProfGifCalls.fetch_add(1, std::memory_order_relaxed);
            g_eeProfGifBytes.fetch_add(bytes, std::memory_order_relaxed);
        }
    } eeProfGifScope{g_eeProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, sizeBytes};

    // cont.253 PS2X_GIF_CENSUS, submission side: is any of the ~11 MB/frame of GIF traffic a
    // byte-identical RESUBMISSION? This codebase has had exactly that bug before (the XGKICK
    // double-parse drew every kicked packet twice, cont.230), so it is worth one cheap check.
    // FNV-1a over the packet, compared against the previous packet on this path and against a
    // 64-entry ring of recent hashes.
    if (g_gifCensusPkt && data && sizeBytes)
    {
        static std::atomic<unsigned long long> s_pkts{0}, s_dupPrev{0}, s_dupRing{0}, s_bytes{0}, s_dupBytes{0};
        static thread_local uint64_t s_prevHash[4] = {0, 0, 0, 0};
        static thread_local uint64_t s_ring[64] = {};
        static thread_local unsigned s_ringPos = 0;
        uint64_t h = 1469598103934665603ull;
        const uint32_t step = sizeBytes > 4096u ? (sizeBytes / 4096u) : 1u;   // sample huge packets
        for (uint32_t i = 0; i < sizeBytes; i += step)
        {
            h ^= data[i];
            h *= 1099511628211ull;
        }
        h ^= static_cast<uint64_t>(sizeBytes) * 1099511628211ull;
        const uint32_t path = static_cast<uint32_t>(pathId) & 3u;
        s_pkts.fetch_add(1, std::memory_order_relaxed);
        s_bytes.fetch_add(sizeBytes, std::memory_order_relaxed);
        if (s_prevHash[path] == h)
        {
            s_dupPrev.fetch_add(1, std::memory_order_relaxed);
            s_dupBytes.fetch_add(sizeBytes, std::memory_order_relaxed);
        }
        else
        {
            for (unsigned k = 0; k < 64u; ++k)
                if (s_ring[k] == h) { s_dupRing.fetch_add(1, std::memory_order_relaxed); break; }
        }
        s_prevHash[path] = h;
        s_ring[s_ringPos++ & 63u] = h;
        static std::chrono::steady_clock::time_point s_last{};
        const auto now = std::chrono::steady_clock::now();
        if (s_last.time_since_epoch().count() == 0) s_last = now;
        else if (now - s_last >= std::chrono::seconds(10))
        {
            s_last = now;
            const unsigned long long n = s_pkts.load(std::memory_order_relaxed);
            const unsigned long long dp = s_dupPrev.load(std::memory_order_relaxed);
            const unsigned long long dr = s_dupRing.load(std::memory_order_relaxed);
            const unsigned long long by = s_bytes.load(std::memory_order_relaxed);
            const unsigned long long dby = s_dupBytes.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                         "[gif:dup] packets=%llu bytes=%.2fGB | dup-of-previous=%llu (%.2f%%, %.2fGB) "
                         "| repeat-in-last-64=%llu (%.2f%%) | mean-packet=%.0fB\n",
                         n, double(by) / 1073741824.0, dp, n ? 100.0 * double(dp) / double(n) : 0.0,
                         double(dby) / 1073741824.0, dr, n ? 100.0 * double(dr) / double(n) : 0.0,
                         n ? double(by) / double(n) : 0.0);
        }
    }
    if (!data || sizeBytes < 16)
        return;

    if (ps2xGifTraceOn())
    {
        uint64_t tagLo = 0; std::memcpy(&tagLo, data, sizeof(tagLo));
        const unsigned flg = static_cast<unsigned>((tagLo >> 58) & 0x3u);
        std::fprintf(stderr, "[gif:trace] flip=%llu packet path=%d bytes=%u flg=%u%s masked=%d producer=%d\n",
                     ps2xGsPerfFlips(), static_cast<int>(pathId), sizeBytes, flg, (flg >= 2u) ? " IMAGE" : "",
                     m_path3Masked ? 1 : 0, ps2gs::producerSide() ? 1 : 0);
    }
    if (pathId == GifPathId::Path3)
    {
        if (ps2gs::producerSide()) // cont.317 stage 1: external path 3 goes through the ring
        {
            ps2gs::enqueueGif3(this, data, sizeBytes, drainImmediately);
            ps2gs::pump();
            return;
        }
        if (m_path3Masked)
        {
            m_path3MaskedFifo.emplace_back(data, data + sizeBytes);
            return;
        }
        flushMaskedPath3Packets(false);
    }

    if (m_gifArbiter)
        m_gifArbiter->submit(pathId, data, sizeBytes, path2DirectHl);
    else if (m_gifPacketCallback)
        m_gifPacketCallback(data, sizeBytes);

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount)
{
    if (!m_rdram || qwCount == 0)
        return;
    const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
    uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
    uint32_t bytesLeft = sizeBytes;
    while (bytesLeft >= 16)
    {
        if (srcPhysAddr >= PS2_RAM_SIZE)
            srcPhysAddr = 0;
        uint32_t chunk = bytesLeft;
        if (srcPhysAddr + chunk > PS2_RAM_SIZE)
            chunk = PS2_RAM_SIZE - srcPhysAddr;
        if (chunk == 0)
            break;

        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
        submitGifPacket(GifPathId::Path3, m_rdram + srcPhysAddr, chunk);

        bytesLeft -= chunk;
        srcPhysAddr += chunk;
    }
}

void PS2Memory::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_gifArbiter)
        submitGifPacket(GifPathId::Path3, data, sizeBytes);
    else if (m_gifPacketCallback && data && sizeBytes >= 16)
        m_gifPacketCallback(data, sizeBytes);
}

bool PS2Memory::tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    auto loadDmaTagAt = [&](uint32_t guestAddr, DmaTagView &out) -> bool
    {
        const uint8_t *ptr = nullptr;
        if (!resolveContiguous(guestAddr, 16u, ptr))
            return false;
        out = decodeDmaTag(loadScalar<uint64_t>(ptr, 0u, 16u, "native gif dma tag", guestAddr));
        return true;
    };

    auto decodeSetupPayload = [&](const uint8_t *payload, uint64_t (&regs)[4]) -> bool
    {
        const uint64_t tagLo = loadScalar<uint64_t>(payload, 0u, 80u, "native gif setup tag", 0u);
        const uint64_t tagHi = loadScalar<uint64_t>(payload, 8u, 80u, "native gif setup regs", 0u);
        if (gifTagNloop(tagLo) != 4u ||
            gifTagFlg(tagLo) != GIF_FMT_PACKED ||
            gifTagNreg(tagLo) != 1u ||
            (tagHi & 0xFull) != 0x0Eull)
        {
            return false;
        }

        static constexpr uint8_t kExpectedRegs[4] = {
            GS_REG_BITBLTBUF,
            GS_REG_TRXPOS,
            GS_REG_TRXREG,
            GS_REG_TRXDIR,
        };

        uint32_t offset = 16u;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            regs[i] = loadScalar<uint64_t>(payload, offset, 80u, "native gif setup value", 0u);
            const uint64_t reg = loadScalar<uint64_t>(payload, offset + 8u, 80u, "native gif setup register", 0u);
            if ((reg & 0xFFu) != kExpectedRegs[i])
                return false;
            offset += 16u;
        }

        const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
        const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
        const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
        return trxdirMode == 0u && rrw != 0u && rrh != 0u;
    };

    DmaTagView setupTag{};
    if (!loadDmaTagAt(tadr, setupTag) ||
        setupTag.id != 1u ||
        setupTag.qwc != 5u ||
        setupTag.irq)
    {
        return false;
    }

    const uint8_t *setupPayload = nullptr;
    const uint32_t setupPayloadAddr = tadr + 16u;
    if (!resolveContiguous(setupPayloadAddr, 5u * 16u, setupPayload))
        return false;

    uint64_t setupRegs[4] = {};
    if (!decodeSetupPayload(setupPayload, setupRegs))
        return false;

    uint32_t imageTagDmaAddr = setupPayloadAddr + 5u * 16u;
    DmaTagView imageTagDma{};
    if (!loadDmaTagAt(imageTagDmaAddr, imageTagDma) ||
        imageTagDma.id != 1u ||
        imageTagDma.qwc != 1u ||
        imageTagDma.irq)
    {
        return false;
    }

    const uint8_t *imageGifTag = nullptr;
    if (!resolveContiguous(imageTagDmaAddr + 16u, 16u, imageGifTag))
        return false;

    const uint64_t imageTagLo = loadScalar<uint64_t>(imageGifTag, 0u, 16u, "native gif image tag", imageTagDmaAddr + 16u);
    if (gifTagFlg(imageTagLo) != GIF_FMT_IMAGE)
        return false;

    const uint32_t imageQwc = gifTagNloop(imageTagLo);
    if (imageQwc == 0u)
        return false;

    const uint64_t imageBytes64 = static_cast<uint64_t>(imageQwc) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);

    const uint32_t payloadTagAddr = imageTagDmaAddr + 32u;
    DmaTagView payloadTag{};
    if (!loadDmaTagAt(payloadTagAddr, payloadTag) ||
        payloadTag.qwc != imageQwc ||
        payloadTag.irq)
    {
        return false;
    }

    uint32_t imageDataAddr = 0u;
    uint32_t finalTadr = payloadTagAddr;
    uint32_t lastTagUpper = payloadTag.upper;
    if (payloadTag.id == 3u || payloadTag.id == 4u)
    {
        imageDataAddr = payloadTag.addr;
        const uint32_t terminalTagAddr = payloadTagAddr + 16u;
        DmaTagView terminalTag{};
        if (!loadDmaTagAt(terminalTagAddr, terminalTag) ||
            terminalTag.qwc != 0u ||
            terminalTag.irq ||
            (terminalTag.id != 0u && terminalTag.id != 7u))
        {
            return false;
        }
        finalTadr = (terminalTag.id == 0u) ? (terminalTagAddr + 16u) : terminalTagAddr;
        lastTagUpper = terminalTag.upper;
    }
    else if (payloadTag.id == 7u)
    {
        imageDataAddr = payloadTagAddr + 16u;
        finalTadr = payloadTagAddr;
    }
    else
    {
        return false;
    }

    const uint8_t *imageData = nullptr;
    if (!resolveContiguous(imageDataAddr, imageBytes, imageData))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
    gs.uploadImageNative(setupRegs[0], setupRegs[1], setupRegs[2], setupRegs[3], imageData, imageBytes);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = finalTadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (lastTagUpper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

bool PS2Memory::tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    const uint8_t *tagPtr = nullptr;
    if (!resolveContiguous(tadr, 16u, tagPtr))
        return false;

    const DmaTagView tag = decodeDmaTag(loadScalar<uint64_t>(tagPtr, 0u, 16u, "native packed gif dma tag", tadr));
    if (tag.id != 7u || tag.qwc == 0u || tag.irq)
        return false;

    const uint64_t payloadBytes64 = static_cast<uint64_t>(tag.qwc) * 16ull;
    if (payloadBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);

    const uint8_t *payload = nullptr;
    if (!resolveContiguous(tadr + 16u, payloadBytes, payload))
        return false;
    if (!gs.processNativePackedGIFPacket(payload, payloadBytes))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = tadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (tag.upper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

int PS2Memory::pollDmaRegisters()
{
    return 0;
}

uint32_t PS2Memory::readIORegister(uint32_t address)
{
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        flushEeTimers(); // cont.317: the guest reads an exact count
        const EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            return timer.count & 0xFFFFu;
        case kEeTimerModeOffset:
            return timer.mode & (kEeTimerModeConfigMask | kEeTimerModeStatusMask);
        case kEeTimerCompareOffset:
            return timer.compare & 0xFFFFu;
        case kEeTimerHoldOffset:
            return timer.hold & 0xFFFFu;
        default:
            return 0u;
        }
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from read8/16/32/64 today, same reasoning as the write
        // path above; kept correct for direct callers.
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return static_cast<uint32_t>((gs_regs.csr.load() >> (off * 8u)) & 0xFFFFFFFFull);
        }
        if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            return static_cast<uint32_t>((*reg >> (off * 8u)) & 0xFFFFFFFFull);
        }
        return 0u;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (ps2x_ipu::enabled())
            return ps2x_ipu::read32(address);
        uint32_t val = 0;
        switch (address)
        {
        case 0x10002000:
            val = m_ioRegisters[address];
            break;
        case 0x10002010:
            val = m_ioRegisters[address] & ~(1u << 31);
            break;
        case 0x10002020:
        case 0x10002030:
            val = m_ioRegisters[address];
            break;
        default:
            val = 0;
            break;
        }
        return val;
    }
    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10008000 && address < 0x1000F000)
        {
            if ((address & 0xFFFFFB00u) == 0x1000B000u && ps2x_ipu::enabled())
            {
                // cont.247: MADR/QWC/TADR of channel 3/4 are the seam's live values -- libipu's
                // sceIpuStopDMA reads them right after the stop to rewind the chain by IFC+FP qwords.
                syncIpuDmaChannels();
                if ((address & 0xFF) != 0x00)
                    return m_ioRegisters[address];
                // cont.245: the IPU channels report their REAL state (a fromIPU DMA stays armed until
                // the IDEC that fills it runs; the game polls STR after that command).
                const uint32_t v = m_ioRegisters[address];
                return ps2x_ipu::dmaRunning(address == 0x1000B000u ? 3 : 4) ? (v | 0x100u) : (v & ~0x100u);
            }
            if ((address & 0xFF) == 0x00)
            {
                uint32_t channelStatus = m_ioRegisters[address] & ~0x100u;
                m_ioRegisters[address] = channelStatus;
                return channelStatus;
            }
        }

        if (address >= 0x10000200 && address < 0x10000300)
        {
            return 0;
        }

        if (address >= 0x1000F200 && address <= 0x1000F260)
        {
            if (address == 0x1000F230)
            {
                return 0x60000;
            }
            if (address == 0x1000F240)
            {
                return 0xF0000002;
            }
            return 0;
        }
    }

    auto it = m_ioRegisters.find(address);
    if (it != m_ioRegisters.end())
    {
        return it->second;
    }

    return 0;
}

void PS2Memory::registerCodeRegion(uint32_t start, uint32_t end)
{
    if (end <= start)
    {
        std::cerr << "Ignoring invalid code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    if ((end - start) > PS2_RAM_SIZE)
    {
        std::cerr << "Ignoring oversized code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    for (const auto &existing : m_codeRegions)
    {
        if (existing.start == start && existing.end == end)
        {
            return;
        }
    }

    CodeRegion region;
    region.start = start;
    region.end = end;

    size_t sizeInWords = (end - start + 3u) / 4u;
    region.modified.resize(sizeInWords, false);

    m_codeRegions.push_back(region);
    RUNTIME_LOG("Registered code region: " << std::hex << start << " - " << end << std::dec);
}

bool PS2Memory::isAddressInRegion(uint32_t address, const CodeRegion &region)
{
    return (address >= region.start && address < region.end);
}

bool PS2Memory::isCodeAddress(uint32_t address) const
{
    for (const auto &region : m_codeRegions)
    {
        if (address >= region.start && address < region.end)
        {
            return true;
        }
    }
    return false;
}

void PS2Memory::markModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = true;
                RUNTIME_LOG("Marked code at " << std::hex << addr << std::dec << " as modified");
            }
        }
    }
}

bool PS2Memory::isCodeModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return false;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (const auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size() && region.modified[bitIndex])
            {
                return true; // Found modified code
            }
        }
    }

    return false; // No modifications found
}

void PS2Memory::clearModifiedFlag(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = false;
            }
        }
    }
}
