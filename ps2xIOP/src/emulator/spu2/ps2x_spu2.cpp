// ps2x SPU2 port (rotk row 273): the seam -- what PCSX2's pcsx2/SPU2/spu2.cpp and IopDma.cpp/IopIrq.cpp do around the
// SPU2 core, against the emulated IOP instead of PCSX2's. The wrappers below are copied from spu2.cpp (master e487304)
// with the host-stream parts replaced by a ring buffer.
#include "ps2x_spu2.h"

#include "SPU2/defs.h"
#include "SPU2/regs.h"
#include "SPU2/spu2.h"
#include "IopCounters.h"
#include "IopDma.h"
#include "IopHw.h"
#include "R3000A.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// ---- the PCSX2 globals the core links against -------------------------------------------------------------------
Ps2xSpu2IopRegs psxRegs;
Ps2xSpu2Counter psxCounters[8];
s32 psxNextDeltaCounter = 0;
u32 psxNextStartCounter = 0;
Ps2xSpu2DmaRegs ps2xSpu2Dma;
u64 lClocks = 0;
const StereoOut32 StereoOut32::Empty(0, 0);
float DCFilterIn[2], DCFilterOut[2];

namespace
{
    ps2x::spu2::Host *s_host = nullptr;

    // Output ring: 1 s of 48 kHz stereo. The mixer (IOP side) writes, the host audio device (another thread) reads.
    constexpr size_t kRingFrames = 48000u;
    std::mutex s_ringMutex;
    std::vector<int16_t> s_ring(kRingFrames * 2u);
    size_t s_ringRead = 0u, s_ringWrite = 0u, s_ringFill = 0u;
    std::atomic<uint64_t> s_mixed{0u};

    // PS2X_SPU2_LOG=1 (diagnostic, default OFF): counters of DMA starts/completions, IRQ 9 and register writes,
    // printed every 2 s of host time and on the first few DMAs.
    struct Stats
    {
        uint64_t dmaStart[2] = {}, dmaDone[2] = {}, irq9 = 0, writes = 0, keyOnWrites = 0;
    } s_stats;
    bool logOn()
    {
        static const bool on = [] { const char *e = std::getenv("PS2X_SPU2_LOG"); return e && e[0] == '1'; }();
        return on;
    }
    void logTick()
    {
        if (!logOn())
            return;
        static auto last = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(2))
            return;
        last = now;
        std::fprintf(stderr, "[spu2:log] cycle=%u dma4 %llu/%llu dma7 %llu/%llu irq9=%llu writes=%llu kon=%llu mixed=%llu\n",
                     psxRegs.cycle, (unsigned long long)s_stats.dmaStart[0], (unsigned long long)s_stats.dmaDone[0],
                     (unsigned long long)s_stats.dmaStart[1], (unsigned long long)s_stats.dmaDone[1],
                     (unsigned long long)s_stats.irq9, (unsigned long long)s_stats.writes,
                     (unsigned long long)s_stats.keyOnWrites, (unsigned long long)s_mixed.load());
    }

    // PS2X_SPU2_DUMP=<path> (diagnostic, default OFF): every mixed sample appended to <path> as raw 48 kHz s16le stereo.
    FILE *dumpFile()
    {
        static FILE *f = [] {
            const char *path = std::getenv("PS2X_SPU2_DUMP");
            FILE *out = (path && *path) ? std::fopen(path, "wb") : nullptr;
            if (out)
                std::fprintf(stderr, "[spu2] dumping the mix to %s (raw s16le, 48000 Hz, stereo)\n", path);
            return out;
        }();
        return f;
    }

    // spu2.cpp DCFilter (a DC-blocking high-pass, "some games pause voices with the volume left on").
    void DCFilter(float *input)
    {
        float output[2];
        output[0] = (input[0] - DCFilterIn[0] + ((0.995f * DCFilterOut[0])));
        output[1] = (input[1] - DCFilterIn[1] + ((0.995f * DCFilterOut[1])));
        DCFilterIn[0] = input[0];
        DCFilterIn[1] = input[1];
        DCFilterOut[0] = output[0];
        DCFilterOut[1] = output[1];
        input[0] = output[0];
        input[1] = output[1];
    }

    int16_t toS16(float value)
    {
        const float scaled = value * 32767.0f;
        return static_cast<int16_t>(scaled > 32767.0f ? 32767 : scaled < -32768.0f ? -32768 : static_cast<int>(scaled));
    }
}

// spu2.cpp spu2Output: the same conversion + DC filter; the chunk goes to the ring instead of an AudioStream.
void spu2Output(StereoOut32 out)
{
    float conv[2];
    conv[0] = static_cast<float>(clamp_mix(out.Left)) / INT16_MAX;
    conv[1] = static_cast<float>(clamp_mix(out.Right)) / INT16_MAX;
    DCFilter(conv);
    const int16_t frame[2] = {toS16(conv[0]), toS16(conv[1])};
    if (FILE *f = dumpFile())
        std::fwrite(frame, sizeof(frame), 1u, f);
    {
        std::lock_guard<std::mutex> lock(s_ringMutex);
        s_ring[s_ringWrite * 2u] = frame[0];
        s_ring[s_ringWrite * 2u + 1u] = frame[1];
        s_ringWrite = (s_ringWrite + 1u) % kRingFrames;
        if (s_ringFill == kRingFrames)
            s_ringRead = (s_ringRead + 1u) % kRingFrames;   // full: drop the oldest frame
        else
            ++s_ringFill;
    }
    s_mixed.fetch_add(1u, std::memory_order_relaxed);
}

// ---- PCSX2 IopIrq.cpp / IopDma.cpp, against the emulated IOP --------------------------------------------------------
void spu2Irq()
{
    ++s_stats.irq9;
    if (s_host)
        s_host->spu2Interrupt();
}

void spu2DMA4Irq()
{
    ++s_stats.dmaDone[0];
    SPU2interruptDMA4();
    if (s_host)
        s_host->spu2DmaComplete(0);
}

void spu2DMA7Irq()
{
    ++s_stats.dmaDone[1];
    SPU2interruptDMA7();
    if (s_host)
        s_host->spu2DmaComplete(1);
}

void *iopPhysMem(u32 address)
{
    return s_host ? s_host->iopRam() + (address & 0x1FFFFFu) : nullptr;
}

// ---- spu2.cpp, the parts that are not the host stream ------------------------------------------------------------
u32 SPU2::GetConsoleSampleRate()
{
    return SAMPLE_RATE;
}

bool SPU2::IsRunningPSXMode()
{
    return false;
}

void SPU2readDMA4Mem(u16 *pMem, u32 size) // size in 16-bit units
{
    TimeUpdate(psxRegs.cycle);
    Cores[0].DoDMAread(pMem, size);
}

void SPU2writeDMA4Mem(u16 *pMem, u32 size)
{
    TimeUpdate(psxRegs.cycle);
    Cores[0].DoDMAwrite(pMem, size);
}

void SPU2interruptDMA4()
{
    if (Cores[0].DmaMode)
        Cores[0].Regs.STATX |= 0x80;
    Cores[0].Regs.STATX &= ~0x400;
    Cores[0].TSA = Cores[0].ActiveTSA;
}

void SPU2interruptDMA7()
{
    if (Cores[1].DmaMode)
        Cores[1].Regs.STATX |= 0x80;
    Cores[1].Regs.STATX &= ~0x400;
    Cores[1].TSA = Cores[1].ActiveTSA;
}

void SPU2readDMA7Mem(u16 *pMem, u32 size)
{
    TimeUpdate(psxRegs.cycle);
    Cores[1].DoDMAread(pMem, size);
}

void SPU2writeDMA7Mem(u16 *pMem, u32 size)
{
    TimeUpdate(psxRegs.cycle);
    Cores[1].DoDMAwrite(pMem, size);
}

u16 SPU2read(u32 rmem)
{
    u16 ret = 0xDEAD;
    u32 core = 0;
    const u32 mem = rmem & 0xFFFF;
    u32 omem = mem;

    if (mem & 0x400)
    {
        omem ^= 0x400;
        core = 1;
    }

    if (omem == 0x1f9001AC)
    {
        Cores[core].ActiveTSA = Cores[core].TSA;
        for (int i = 0; i < 2; i++)
        {
            if (Cores[i].IRQEnable && (Cores[i].IRQA == Cores[core].ActiveTSA))
                SetIrqCall(i);
        }
        ret = Cores[core].DmaRead();
    }
    else
    {
        TimeUpdate(psxRegs.cycle);

        if (rmem >> 16 == 0x1f80)
            ret = Cores[0].ReadRegPS1(rmem);
        else if (mem >= 0x800)
            ret = spu2Ru16(mem);
        else
            ret = *(regtable[(mem >> 1)]);
    }
    return ret;
}

void SPU2write(u32 rmem, u16 value)
{
    ++s_stats.writes;
    if (((rmem & 0x3FF) == 0x1A0 || (rmem & 0x3FF) == 0x1A2) && value != 0)   // KON0/KON1 (regs.h REG_S_KON)
        ++s_stats.keyOnWrites;
    TimeUpdate(psxRegs.cycle);
    if (rmem >> 16 == 0x1f80)
        Cores[0].WriteRegPS1(rmem, value);
    else
        SPU2_FastWrite(rmem, value);
}

// ---- the seam's API ------------------------------------------------------------------------------------------------
namespace ps2x::spu2
{
    void attach(Host *host)
    {
        s_host = host;
    }

    void reset(uint64_t iopCycle)
    {
        psxRegs.cycle = static_cast<u32>(iopCycle);
        lClocks = psxRegs.cycle;
        // spu2.cpp SPU2::InternalReset(false)
        spu2Mix = MULTI_ISA_SELECT(spu2Mix);
        ReverbDownsample = MULTI_ISA_SELECT(ReverbDownsample);
        ReverbUpsample = MULTI_ISA_SELECT(ReverbUpsample);
        std::memset(spu2regs, 0, 0x010000);
        std::memset(_spu2mem, 0, 0x200000);
        std::memset(_spu2mem + 0x2800, 7, 0x10); // from BIOS reversal. Locks the voices so they don't run free.
        std::memset(_spu2mem + 0xe870, 7, 0x10); // Loop which gets left over by the BIOS, Megaman X7 relies on it being there.
        std::memset(DCFilterIn, 0, sizeof(DCFilterIn));
        std::memset(DCFilterOut, 0, sizeof(DCFilterOut));
        Spdif.Info = 0;
        Cores[0].Init(0);
        Cores[1].Init(1);
        // IopCounters.cpp: counter 6 is the SPU2's, rate 768.
        psxCounters[6].rate = 768;
        psxCounters[6].deltaCycles = psxCounters[6].rate;
        psxCounters[6].startCycle = psxRegs.cycle;
        ps2xSpu2Dma = {};
        std::lock_guard<std::mutex> lock(s_ringMutex);
        s_ringRead = s_ringWrite = s_ringFill = 0u;
    }

    void setCycle(uint64_t iopCycle)
    {
        psxRegs.cycle = static_cast<u32>(iopCycle);
    }

    void advance(uint64_t iopCycle)
    {
        psxRegs.cycle = static_cast<u32>(iopCycle);
        TimeUpdate(psxRegs.cycle);
        logTick();
    }

    uint16_t read16(uint32_t address)
    {
        return SPU2read(address);
    }

    void write16(uint32_t address, uint16_t value)
    {
        SPU2write(address, value);
    }

    // IopDma.cpp psxDmaGeneric.
    void dmaStart(int core, uint32_t madr, uint32_t bcr, uint32_t chcr)
    {
        const int size = (bcr >> 16) * (bcr & 0xFFFF);
        dmaMadr(core) = madr;
        ++s_stats.dmaStart[core ? 1 : 0];
        if (logOn() && s_stats.dmaStart[0] + s_stats.dmaStart[1] <= 12)
            std::fprintf(stderr, "[spu2:log] DMA%d start madr=%x bcr=%x chcr=%x tsa=%x dmamode=%d admas=%x\n", core ? 7 : 4,
                         madr, bcr, chcr, Cores[core].TSA, Cores[core].DmaMode, Cores[core].AutoDMACtrl);
        switch (chcr)
        {
        case 0x01000201: // cpu to spu2 transfer
            if (core == 1)
                SPU2writeDMA7Mem(static_cast<u16 *>(iopPhysMem(madr)), size * 2);
            else
                SPU2writeDMA4Mem(static_cast<u16 *>(iopPhysMem(madr)), size * 2);
            break;
        case 0x01000200: // spu2 to cpu transfer
            if (core == 1)
                SPU2readDMA7Mem(static_cast<u16 *>(iopPhysMem(madr)), size * 2);
            else
                SPU2readDMA4Mem(static_cast<u16 *>(iopPhysMem(madr)), size * 2);
            break;
        default:
            std::fprintf(stderr, "[spu2] DMA %d - SPU unknown chcr %x madr %x bcr %x\n", core ? 7 : 4, chcr, madr, bcr);
            break;
        }
    }

    uint32_t &dmaMadr(int core)
    {
        return core ? ps2xSpu2Dma.dma7Madr : ps2xSpu2Dma.dma4Madr;
    }

    uint32_t &dmaTadr(int core)
    {
        return core ? ps2xSpu2Dma.dma7Tadr : ps2xSpu2Dma.dma4Tadr;
    }

    size_t takeSamples(int16_t *destination, size_t maxFrames)
    {
        std::lock_guard<std::mutex> lock(s_ringMutex);
        size_t frames = 0u;
        while (frames < maxFrames && s_ringFill)
        {
            destination[frames * 2u] = s_ring[s_ringRead * 2u];
            destination[frames * 2u + 1u] = s_ring[s_ringRead * 2u + 1u];
            s_ringRead = (s_ringRead + 1u) % kRingFrames;
            --s_ringFill;
            ++frames;
        }
        return frames;
    }

    uint64_t samplesMixed()
    {
        return s_mixed.load(std::memory_order_relaxed);
    }
}
