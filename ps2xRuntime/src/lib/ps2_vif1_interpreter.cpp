// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"
#include "runtime/ps2_pipe_capture.h"
#include "runtime/ps2_gs_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
// cont.344 PS2X_GIF_TRACE (ps2_memory.cpp) and the flip counter (gs_cpu_backend.cpp).
bool ps2xGifTraceOn();
extern "C" unsigned long long ps2xGsPerfFlips();
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    // PS2X_VIF_STRIPLOG=1 (diagnostic, default OFF). Cycle 76: the FMV's 4th movie column quad
    // never reaches GS; the arbiter is submitted only columns 0/1/2 on PATH2, so the shortfall is
    // at or upstream of VIF1. Scan a buffer for movie-strip quad tags and report their columns.
    //
    // ⚠ A strip packet is TWO tags -- [A+D setup: nloop=1 flg=0 nreg=1][quad: nloop=4 flg=0
    // nreg=3] -- so the quad tag is NOT at offset 0. Two earlier probes checked only offset 0,
    // matched nothing, and produced two WRONG eliminations. This walks every tag. Validate any
    // "found nothing" result against a known-positive before believing it.
    void vifScanStripColumns(const char *what, const uint8_t *data, uint32_t sizeBytes)
    {
        static const bool on = [] {
            const char *e = std::getenv("PS2X_VIF_STRIPLOG");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        if (!on || !data || sizeBytes < 32u)
            return;

        int cols[8];
        int nCols = 0;
        uint32_t off = 0u;
        while (off + 16u <= sizeBytes && nCols < 8)
        {
            uint64_t tlo = 0, thi = 0;
            std::memcpy(&tlo, data + off, 8);
            std::memcpy(&thi, data + off + 8, 8);
            const uint32_t nl = static_cast<uint32_t>(tlo & 0x7FFFu);
            const uint8_t fl = static_cast<uint8_t>((tlo >> 58) & 0x3u);
            uint32_t nr = static_cast<uint32_t>((tlo >> 60) & 0xFu);
            if (nr == 0u)
                nr = 16u;
            const uint64_t payload = (fl == 0u)   ? (uint64_t)nr * nl * 16ull
                                     : (fl == 1u) ? (((uint64_t)nr * nl + 1ull) >> 1) * 16ull
                                                  : (uint64_t)nl * 16ull;
            if (fl == 0u && nl == 4u && nr == 3u)
            {
                int slot = -1;
                for (uint32_t r = 0; r < nr; ++r)
                {
                    const uint8_t rd = static_cast<uint8_t>((thi >> (r * 4)) & 0xFu);
                    if (rd == 0x05u || rd == 0x04u)
                        slot = static_cast<int>(r);
                }
                const uint32_t voff = off + 16u + static_cast<uint32_t>(slot < 0 ? 0 : slot) * 16u;
                if (slot >= 0 && voff + 16u <= sizeBytes)
                {
                    uint64_t vlo = 0;
                    std::memcpy(&vlo, data + voff, 8);
                    const float vx = static_cast<float>(vlo & 0xFFFFu) / 16.0f;
                    if (vx >= 1700.0f && vx <= 2400.0f)
                        cols[nCols++] = static_cast<int>((vx - 1792.0f) / 128.0f);
                }
            }
            if (payload == 0ull && fl != 0u)
                break;
            off += 16u + static_cast<uint32_t>(payload);
        }

        if (nCols > 0)
        {
            static uint64_t n = 0;
            ++n;
            if (n <= 60ull || (n % 500ull) == 0ull)
            {
                std::fprintf(stderr, "[VIF:strips] #%llu %s size=%u count=%d cols=",
                             (unsigned long long)n, what, sizeBytes, nCols);
                for (int i = 0; i < nCols; ++i)
                    std::fprintf(stderr, "%d%s", cols[i], (i + 1 < nCols) ? "," : "");
                std::fprintf(stderr, "\n");
            }
        }
    }

    constexpr uint8_t kGifFmtImage = 2u;

    // PS2X_VIF_DIRECTCARRY=0 kills the cycle-85 fix: consume a pending Path2 image continuation
    // INSIDE the DIRECT handler (payload-aligned) instead of raw-eating stream bytes at the top
    // of the parse loop. With PS2X_DMA_REFVIF embedding TTE tag words, the stream between DIRECT
    // payloads is VIF COMMANDS (masked-tag NOPs + the ref's own DIRECT wrapper); the top-of-loop
    // raw consume ate those embedded words as pixels -- one garbage qword per ref tag = the FMV
    // dot lattice (cycle 85, run_c85C). On hardware the pending-image state lives in the GIF
    // (current tag's remaining nloop, PCSX2 Gif_Path::gifTag / Gif_Unit), and the VIF stays in
    // command state -- this reproduces that: commands parse, DIRECT payloads feed the image.
    // Measured stream shape (run_c85D chain #3): [NOP NOP NOP DIRECT imm=1][GIFtag IMAGE
    // nloop=0x7fff] then a ref of qwc=0x7fff raw pixels whose own DIRECT wrapper rides in its
    // TTE tag words -- the game DEPENDS on the embed, and every continuation is DIRECT-wrapped.
    const bool s_vifDirectCarry = [] {
        const char *e = std::getenv("PS2X_VIF_DIRECTCARRY");
        return !(e != nullptr && e[0] == '0'); // default ON (correctness fix)
    }();

    // PS2X_VIF_DIRECTSPAN=0 kills the cycle-86 fix: a DIRECT whose declared qword count exceeds
    // the current chain buffer keeps consuming from the NEXT buffer(s). VIF is a state machine --
    // PCSX2's `vif1.tag.size` persists across DMA transfers -- and the game issues DIRECT imm=0
    // (= 65536 qw) spans that our per-buffer clamp cut short, leaving the remainder to be
    // tag-parsed raw at the next entry (the 223984-B DESYNC packets, run_c86B). TU-static: one VIF1.
    const bool s_vifDirectSpan = [] {
        const char *e = std::getenv("PS2X_VIF_DIRECTSPAN");
        return !(e != nullptr && e[0] == '0'); // default ON (correctness fix)
    }();

    // PS2X_GIF_RESUME (default ON) moves split-tag handling into the GS frontend's per-path
    // resumable parser: it consumes a spilled tag's tail from the path's NEXT packet directly,
    // so the VIF-side image-continuation carry (which prepends a synthesized IMAGE tag) must
    // stand down or the resuming parser reads the synthetic tag as payload.
    const bool s_gifResumeOwnsSplits = [] {
        const char *e = std::getenv("PS2X_GIF_RESUME");
        return !(e && e[0] == '0');
    }();
    uint32_t g_vif1DirectSpanRemainQw = 0u;
    struct VifCmdRingEnt { uint32_t pos; uint32_t bufSize; uint32_t cmd; };
    VifCmdRingEnt g_vifCmdRing[32] = {};
    uint32_t g_vifCmdRingIdx = 0u;
    uint32_t g_vifEntrySpanQw = 0u;
    bool g_vif1DirectSpanHl = false;

    uint32_t gifImageQwcFromTag(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint64_t tagLo = 0u;
        std::memcpy(&tagLo, data, sizeof(tagLo));
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
        if (flg != kGifFmtImage && flg != 3u) // flg=3 "Disable" == IMAGE2
            return 0u;

        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }

    // Cycle 81/82: gifImageQwcFromTag reads ONLY the first tag, so it misses an IMAGE tag that
    // sits BEHIND a BITBLTBUF/TRXREG/TRXDIR setup tag -- the normal texture-upload shape. When
    // that image spans the DIRECT boundary the carry-over is never set and the NEXT DIRECT is
    // submitted RAW starting mid-image, which processGIFPacket misreads as a bogus PACKED tag
    // (measured 100% malformed level transfers, cont.137). Walk the DIRECT payload's GIFtags
    // exactly as processGIFPacket does and return how many image qwords extend PAST this payload
    // (0 if it ends on a tag boundary). Advances match processGIFPacket: PACKED nloop*nreg*16,
    // REGLIST ceil(nloop*nreg/2)*16 (8B entries padded to qword), IMAGE nloop*16.
    uint32_t directTrailingImageQwc(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data)
            return 0u;
        uint32_t off = 0u;
        while (off + 16u <= sizeBytes)
        {
            uint64_t lo = 0u;
            std::memcpy(&lo, data + off, sizeof(lo));
            off += 16u;
            const uint8_t flg = static_cast<uint8_t>((lo >> 58) & 0x3u);
            const uint32_t nloop = static_cast<uint32_t>(lo & 0x7FFFu);
            uint32_t nreg = static_cast<uint32_t>((lo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg == 0u) // PACKED
            {
                off += nloop * nreg * 16u;
            }
            else if (flg == 1u) // REGLIST
            {
                off += ((nloop * nreg + 1u) / 2u) * 16u;
            }
            else // IMAGE (flg=2) or flg=3 "Disable" == IMAGE2 -- both consume nloop qwords
            {
                const uint32_t need = nloop * 16u;
                if (off + need > sizeBytes)
                {
                    const uint32_t spill = nloop - (sizeBytes - off) / 16u;
                    if (flg == 3u)
                    {
                        static unsigned long s_flg3SpillN = 0;
                        ++s_flg3SpillN;
                        if (s_flg3SpillN <= 32u || (s_flg3SpillN % 1024u) == 0u)
                            std::fprintf(stderr, "[vif:flg3spill] #%lu nloop=%u spill=%u pkt=%u off=%u\n",
                                         s_flg3SpillN, nloop, spill, sizeBytes, off);
                    }
                    return spill; // image spills past this payload
                }
                off += need;
            }
            if (off > sizeBytes) // a PACKED/REGLIST tag overran -> malformed payload, stop
                break;
        }
        return 0u;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;
            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }
            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3u) & ~3u;

            if (m_vu0Data && pos + totalBytes <= sizeBytes && vl == 0u)
            {
                uint32_t vuAddr = static_cast<uint32_t>(imm & 0x3FFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr = (vuAddr + (vif0_regs.tops & 0x3FFu)) & 0x3FFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec = (cl >= wl) ? ((vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu)
                                                  : ((vuAddr + writeIndex) & 0x3FFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(m_vu0Data + destOff), _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            break;
        }
    }
}

// External-linkage reset for the TU-static DIRECT-span state (called from the VIF1 reset
// sites in ps2_memory.cpp; declared extern there -- no header change).
bool ps2xVif1DirectSpanQuiescent() { return g_vif1DirectSpanRemainQw == 0u; }
void ps2xVif1ResetDirectSpan()
{
    g_vif1DirectSpanRemainQw = 0u;
    g_vif1DirectSpanHl = false;
}


// ★★★ cont.317 PS2X_VIF_FASTUNPACK (default ON; `=0` restores the generic write loop): the VIF1 UNPACK
// write loop, one SSE vector per write instead of a per-field runtime switch. perf on the live fight
// (EE thread, 7 raster workers) had processVIF1Data at 14.7% self, the largest symbol left, with the
// samples in the generic loop below (per-vector `memcpy` of the old line, per-field mask/mode
// branches). The vector-weighted census (PS2X_VIF_CENSUS `[vif:shape]`, 658 M vectors / 232 s):
//   24.1%  V4-8  zext, no mask, mode 0, cl=wl=1                  (colours)
//   24.1%  V3-32 mask 0x0080 (W = COL[0]), mode 0, cl=wl=1        (positions)
//   23.9%  V2-16 sext, mask 0x1090 (Z = ROW[2], W = COL[0]), cl=wl=1 (texcoords)
//   17%    V4-16 sext, mode 1 (ROW add), cl=wl=1 and cl=3/wl=2
//    4%    V4-8 zext, cl=3/wl=1; the rest under 2% each; fill (cl<wl) = 0, mode 2/3 = 0.
// Shape, decided ONCE per UNPACK: per write-cycle lane selects (data / ROW / COL[c] / keep) and
// the COL broadcast are precomputed from MASK for cyclePos 0..3, and each vector is decoded with
// one SSE conversion (pmovzx/pmovsx or a straight load). Everything the generic loop does is
// mirrored rule for rule so the result is bit-identical:
//   * lanes above the format's component count carry the OLD VU line (the generic loop's
//     `decompressed[c] = lanes[c]`), S formats broadcast the scalar to all four;
//   * mode 1 adds ROW to every lane before the select (the generic loop adds to writeVal for every
//     field whose mask spec is 0, including the preserved lanes);
//   * mask spec 3 keeps the old lane (the generic loop's `continue`);
//   * mode 2 (ROW update), mode 3, V4-5, fill cycles (cl<wl) and out-of-buffer sources fall back.
// Source bytes are read with exact-size copies (never a wide load past the DMA buffer's end).
// PCSX2 reference: x86/Vif_UnpackSSE.cpp -- the same per-shape structure (xPMOVXX conversions,
// doMaskWrite's precomputed row/col selects, mode add before the mask); its arithmetic is identical
// here because the operations are exact integer moves and adds. PS2X_VIF_FASTVERIFY=1 (default OFF)
// runs BOTH paths on every unpack and compares the written VU lines (the oracle: it prints and
// counts mismatches).
static const bool s_vifFastUnpack = []
{ const char *e = std::getenv("PS2X_VIF_FASTUNPACK"); return !(e && e[0] == '0'); }();
// ★ cont.324d PS2X_VIF_V45 (default 1): the V4-5 unpack's 5->8-bit colour expansion (see the decode site).
static const bool s_vifV45 = []
{ const char *e = std::getenv("PS2X_VIF_V45"); return !(e && e[0] == '0'); }();
static const bool s_vifFastVerify = []
{ const char *e = std::getenv("PS2X_VIF_FASTVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_vifFastVerifyChecks = 0ull, g_vifFastVerifyMismatches = 0ull, g_vifFastTaken = 0ull, g_vifFastFallback = 0ull;

static inline bool vifFastUnpackRun(uint8_t *vu1Data, const VIFRegisters &regs, const uint8_t *srcBase,
                                    uint32_t vl, uint32_t vn, bool zeroExtend, bool maskEnable,
                                    uint32_t cl, uint32_t wl, uint32_t writeVectorCount, uint32_t vuAddr)
{
    const uint32_t mode = regs.mode & 3u;
    if (cl < wl || mode > 1u || vl == 3u)
        return false;
    const uint32_t components = vn + 1u;
    const uint32_t bytesPerVector = (vl == 0u) ? components * 4u : (vl == 1u) ? components * 2u : components;

    // Per write-cycle selects (cyclePos capped at 3 for both the mask nibble and COL, as the generic loop does).
    alignas(16) uint32_t selData[4][4], selRow[4][4], selCol[4][4], colVec[4][4];
    const uint32_t cycles = (wl < 4u) ? wl : 4u;
    for (uint32_t c = 0; c < cycles; ++c)
    {
        const uint32_t mc = (c > 3u) ? 3u : c;
        for (uint32_t f = 0; f < 4u; ++f)
        {
            const uint32_t spec = maskEnable ? ((regs.mask >> (((mc * 4u) + f) * 2u)) & 3u) : 0u;
            selData[c][f] = (spec == 0u) ? 0xFFFFFFFFu : 0u;
            selRow[c][f] = (spec == 1u) ? 0xFFFFFFFFu : 0u;
            selCol[c][f] = (spec == 2u) ? 0xFFFFFFFFu : 0u;
            colVec[c][f] = regs.col[mc];
        }
    }
    // Lanes the format actually carries (the rest keep the old VU line). S formats (vn == 0)
    // BROADCAST the scalar to all four lanes in the generic loop, so they carry every lane.
    alignas(16) const uint32_t compMaskArr[4] = {0xFFFFFFFFu, (vn == 0u || components > 1u) ? 0xFFFFFFFFu : 0u,
                                                 (vn == 0u || components > 2u) ? 0xFFFFFFFFu : 0u,
                                                 (vn == 0u || components > 3u) ? 0xFFFFFFFFu : 0u};
    const __m128i compMask = _mm_load_si128(reinterpret_cast<const __m128i *>(compMaskArr));
    const __m128i rowVec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(regs.row));
    const bool anyCol = maskEnable && (regs.mask & 0xAAAAAAAAu) != 0u; // any spec with bit 1 set (2 or 3)
    (void)anyCol;

    const uint8_t *src = srcBase;
    for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
    {
        const uint32_t cyclePos = writeIndex % wl;
        uint32_t destVec;
        if (cl >= wl)
            destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
        else
            destVec = (vuAddr + writeIndex) & 0x3FFu;
        uint8_t *dst = vu1Data + destVec * 16u;
        const __m128i old = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst));

        // ---- decode: 4 lanes, exact-size source read ----
        __m128i dec;
        if (vn == 0u)
        {
            uint32_t v = 0u;
            if (vl == 0u) { std::memcpy(&v, src, 4u); }
            else if (vl == 1u) { uint16_t r = 0; std::memcpy(&r, src, 2u); v = zeroExtend ? uint32_t(r) : uint32_t(int32_t(int16_t(r))); }
            else { const uint8_t r = src[0]; v = zeroExtend ? uint32_t(r) : uint32_t(int32_t(int8_t(r))); }
            dec = _mm_set1_epi32(static_cast<int>(v));
        }
        else if (vl == 0u)
        {
            // Exact-size loads with COMPILE-TIME sizes: a runtime-sized memcpy per vector is a libc
            // call (it showed as 5% memmove in the first cut).
            alignas(16) uint8_t tmp[16] = {0};
            switch (components)
            {
            case 2: std::memcpy(tmp, src, 8u); break;
            case 3: std::memcpy(tmp, src, 12u); break;
            default: std::memcpy(tmp, src, 16u); break;
            }
            dec = _mm_load_si128(reinterpret_cast<const __m128i *>(tmp));
        }
        else if (vl == 1u)
        {
            alignas(16) uint8_t tmp[8] = {0};
            switch (components)
            {
            case 2: std::memcpy(tmp, src, 4u); break;
            case 3: std::memcpy(tmp, src, 6u); break;
            default: std::memcpy(tmp, src, 8u); break;
            }
            const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(tmp));
            dec = zeroExtend ? _mm_cvtepu16_epi32(h) : _mm_cvtepi16_epi32(h);
        }
        else
        {
            uint32_t tmp = 0u;
            switch (components)
            {
            case 2: std::memcpy(&tmp, src, 2u); break;
            case 3: std::memcpy(&tmp, src, 3u); break;
            default: std::memcpy(&tmp, src, 4u); break;
            }
            const __m128i b = _mm_cvtsi32_si128(static_cast<int>(tmp));
            dec = zeroExtend ? _mm_cvtepu8_epi32(b) : _mm_cvtepi8_epi32(b);
        }
        src += bytesPerVector;

        // ---- the generic loop's rules, per lane ----
        dec = _mm_blendv_epi8(old, dec, compMask);            // missing lanes = old VU line
        if (mode == 1u)
            dec = _mm_add_epi32(dec, rowVec);                    // ROW add on every data lane
        const uint32_t sc = (cyclePos > 3u) ? 3u : cyclePos;
        const __m128i sd = _mm_load_si128(reinterpret_cast<const __m128i *>(selData[sc]));
        const __m128i sr = _mm_load_si128(reinterpret_cast<const __m128i *>(selRow[sc]));
        const __m128i scl = _mm_load_si128(reinterpret_cast<const __m128i *>(selCol[sc]));
        const __m128i cv = _mm_load_si128(reinterpret_cast<const __m128i *>(colVec[sc]));
        __m128i out = old;                                       // spec 3 (and nothing else) keeps old
        out = _mm_blendv_epi8(out, dec, sd);
        out = _mm_blendv_epi8(out, rowVec, sr);
        out = _mm_blendv_epi8(out, cv, scl);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), out);
    }
    return true;
}

// ★ rotk 2026-10-03 (row 252) the TEMPLATED unpack, PS2X_VIF_FASTUNPACK=2 (the default; =1 = the cont.317 loop
// above, =0 = the generic loop). Measured on the 60-fps fight (EUR first_fight.pad, PS2X_VIF_PROF): the cont.317
// write loop was 29.9 s of the GsPipeline thread's 190 s -- 75 M unpacks x 36.7 vectors, ~11 ns per vector. Per
// vector it paid a 32-bit divide and modulo (`writeIndex / wl`, `% wl`, though 93% of vectors have cl = wl = 1),
// a runtime switch on the component count, and four blends. Here the shape is a template (VL, VN, sign, mode),
// the write cycles are walked as nested loops (no divide), and the mask is PCSX2's doMaskWrite form,
// precomputed per write cycle: out = (data & D) | (old & K) | C, C = the ROW/COL constants
// (x86/Vif_UnpackSSE.cpp VifUnpackSSE_Simple::doMaskWrite + nVifMask). Rules are those of the cont.317 loop,
// so the result is bit-identical (PS2X_VIF_FASTVERIFY compares it with the generic loop): lanes the format
// does not carry keep the OLD line, S formats broadcast, mode 1 adds ROW before the select, the mask
// nibble and COL index cap at cycle 3. Same fallbacks: cl < wl, mode 2/3, V4-5.
namespace
{
    struct VifTplMasks
    {
        __m128i d[4], k[4], c[4]; // data select, keep-old select, ROW/COL constant -- per write cycle 0..3
    };

    template <uint32_t VL, uint32_t VN, bool ZEXT>
    static inline __m128i vifTplDecode(const uint8_t *src)
    {
        constexpr uint32_t comps = VN + 1u;
        if constexpr (VN == 0u)
        {
            uint32_t v;
            if constexpr (VL == 0u) { std::memcpy(&v, src, 4u); }
            else if constexpr (VL == 1u) { uint16_t r; std::memcpy(&r, src, 2u); v = ZEXT ? uint32_t(r) : uint32_t(int32_t(int16_t(r))); }
            else { const uint8_t r = src[0]; v = ZEXT ? uint32_t(r) : uint32_t(int32_t(int8_t(r))); }
            return _mm_set1_epi32(static_cast<int>(v));
        }
        else if constexpr (VL == 0u)
        {
            if constexpr (comps == 4u)
                return _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
            else if constexpr (comps == 2u)
                return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));
            else
            {
                alignas(16) uint8_t tmp[16] = {0};
                std::memcpy(tmp, src, 12u);
                return _mm_load_si128(reinterpret_cast<const __m128i *>(tmp));
            }
        }
        else if constexpr (VL == 1u)
        {
            uint64_t raw = 0u;
            std::memcpy(&raw, src, comps * 2u);
            const __m128i h = _mm_cvtsi64_si128(static_cast<long long>(raw));
            return ZEXT ? _mm_cvtepu16_epi32(h) : _mm_cvtepi16_epi32(h);
        }
        else
        {
            uint32_t raw = 0u;
            std::memcpy(&raw, src, comps);
            const __m128i b = _mm_cvtsi32_si128(static_cast<int>(raw));
            return ZEXT ? _mm_cvtepu8_epi32(b) : _mm_cvtepi8_epi32(b);
        }
    }

    template <uint32_t VL, uint32_t VN, bool ZEXT, bool MODE1>
    static inline void vifTplWrite(uint8_t *vu1Data, uint32_t destVec, const uint8_t *src, __m128i d, __m128i k,
                                   __m128i c, __m128i row, __m128i comp)
    {
        uint8_t *dst = vu1Data + destVec * 16u;
        const __m128i old = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst));
        __m128i dec = vifTplDecode<VL, VN, ZEXT>(src);
        if constexpr (VN == 1u || VN == 2u)
            dec = _mm_blendv_epi8(old, dec, comp); // lanes the format does not carry = the old line
        if constexpr (MODE1)
            dec = _mm_add_epi32(dec, row);
        const __m128i out = _mm_or_si128(_mm_or_si128(_mm_and_si128(dec, d), _mm_and_si128(old, k)), c);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), out);
    }

    template <uint32_t VL, uint32_t VN, bool ZEXT, bool MODE1>
    static void vifTplRun(uint8_t *vu1Data, const uint8_t *src, uint32_t n, uint32_t vuAddr, uint32_t cl, uint32_t wl,
                          const VifTplMasks &m, __m128i row, __m128i comp)
    {
        constexpr uint32_t bpv = (VL == 0u ? 4u : VL == 1u ? 2u : 1u) * (VN + 1u);
        if (wl == 1u)
        {
            const __m128i d = m.d[0], k = m.k[0], c = m.c[0];
            uint32_t dest = vuAddr;
            for (uint32_t i = 0; i < n; ++i, dest += cl, src += bpv)
                vifTplWrite<VL, VN, ZEXT, MODE1>(vu1Data, dest & 0x3FFu, src, d, k, c, row, comp);
            return;
        }
        uint32_t i = 0, blockBase = vuAddr;
        while (i < n)
        {
            for (uint32_t cyc = 0; cyc < wl && i < n; ++cyc, ++i, src += bpv)
            {
                const uint32_t sc = (cyc > 3u) ? 3u : cyc;
                vifTplWrite<VL, VN, ZEXT, MODE1>(vu1Data, (blockBase + cyc) & 0x3FFu, src, m.d[sc], m.k[sc], m.c[sc], row, comp);
            }
            blockBase += cl;
        }
    }

    using VifTplFn = void (*)(uint8_t *, const uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, const VifTplMasks &,
                              __m128i, __m128i);

    template <uint32_t VL, uint32_t VN>
    constexpr VifTplFn vifTplPick(bool zext, bool mode1)
    {
        return zext ? (mode1 ? &vifTplRun<VL, VN, true, true> : &vifTplRun<VL, VN, true, false>)
                    : (mode1 ? &vifTplRun<VL, VN, false, true> : &vifTplRun<VL, VN, false, false>);
    }

    static VifTplFn vifTplSelect(uint32_t vl, uint32_t vn, bool zext, bool mode1)
    {
        switch (vl * 4u + vn)
        {
        case 0: return vifTplPick<0, 0>(zext, mode1);
        case 1: return vifTplPick<0, 1>(zext, mode1);
        case 2: return vifTplPick<0, 2>(zext, mode1);
        case 3: return vifTplPick<0, 3>(zext, mode1);
        case 4: return vifTplPick<1, 0>(zext, mode1);
        case 5: return vifTplPick<1, 1>(zext, mode1);
        case 6: return vifTplPick<1, 2>(zext, mode1);
        case 7: return vifTplPick<1, 3>(zext, mode1);
        case 8: return vifTplPick<2, 0>(zext, mode1);
        case 9: return vifTplPick<2, 1>(zext, mode1);
        case 10: return vifTplPick<2, 2>(zext, mode1);
        case 11: return vifTplPick<2, 3>(zext, mode1);
        default: return nullptr;
        }
    }
}

static const int s_vifFastUnpackKind = []
{ const char *e = std::getenv("PS2X_VIF_FASTUNPACK"); return (e && e[0] == '0') ? 0 : (e && e[0] == '1') ? 1 : 2; }();

static inline bool vifFastUnpackRunTpl(uint8_t *vu1Data, const VIFRegisters &regs, const uint8_t *srcBase,
                                       uint32_t vl, uint32_t vn, bool zeroExtend, bool maskEnable,
                                       uint32_t cl, uint32_t wl, uint32_t writeVectorCount, uint32_t vuAddr)
{
    const uint32_t mode = regs.mode & 3u;
    if (cl < wl || mode > 1u || vl == 3u)
        return false;
    const VifTplFn fn = vifTplSelect(vl, vn, zeroExtend, mode == 1u);
    if (!fn)
        return false;

    VifTplMasks m;
    const uint32_t cycles = (wl < 4u) ? wl : 4u;
    for (uint32_t cyc = 0; cyc < cycles; ++cyc)
    {
        alignas(16) uint32_t d[4], k[4], c[4];
        for (uint32_t f = 0; f < 4u; ++f)
        {
            const uint32_t spec = maskEnable ? ((regs.mask >> (((cyc * 4u) + f) * 2u)) & 3u) : 0u;
            d[f] = (spec == 0u) ? 0xFFFFFFFFu : 0u;
            k[f] = (spec == 3u) ? 0xFFFFFFFFu : 0u;
            c[f] = (spec == 1u) ? regs.row[f] : (spec == 2u) ? regs.col[cyc] : 0u;
        }
        m.d[cyc] = _mm_load_si128(reinterpret_cast<const __m128i *>(d));
        m.k[cyc] = _mm_load_si128(reinterpret_cast<const __m128i *>(k));
        m.c[cyc] = _mm_load_si128(reinterpret_cast<const __m128i *>(c));
    }
    alignas(16) const uint32_t compArr[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, (vn >= 2u) ? 0xFFFFFFFFu : 0u,
                                             (vn >= 3u) ? 0xFFFFFFFFu : 0u};
    const __m128i comp = _mm_load_si128(reinterpret_cast<const __m128i *>(compArr));
    const __m128i row = _mm_loadu_si128(reinterpret_cast<const __m128i *>(regs.row));
    fn(vu1Data, srcBase, writeVectorCount, vuAddr, cl, wl, m, row, comp);
    return true;
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

// cont.252 PS2X_EE_PROF accumulators (defined in ps2_memory.cpp; extern here so no header that the
// generated code includes has to change -- that would recompile all ~100 unity units).
extern std::atomic<unsigned long long> g_eeProfVif1Ns, g_eeProfVif1Calls, g_eeProfVif1Bytes;
extern const bool g_eeProf;


// ---- cont.257 PS2X_VIF_PROF (default OFF, read-only): the INTRA-VIF1 breakdown.
// WHY: cont.252's [ee:prof] `vif1` accumulator wraps the WHOLE of processVIF1Data -- and MSCAL/
// MSCALF/MSCNT run the VU1 microprogram SYNCHRONOUSLY inside it (ps2_runtime.cpp
// setVu1MscalCallback -> m_vu1.execute(...)), while DIRECT submits GIF packets from inside it too.
// So the reported "vif1-only = 19.0 ms/frame" is VIF1 commands + VU1 + Path2 GIF COMBINED, and the
// companion "guest code + VU0 + VU1 = 24.7 ms" line -- derived as EE-busy minus vif1-inclusive --
// double-counts VU1. This splits the command loop by class so VIF1's OWN cost is measured, not
// derived. Nesting: unpackWr < unpack; mscal is entirely VU1; direct is mostly submitGifPacket.
// `other` is derived (total - the four classes) and carries the decode/loop overhead.
// Only the EE thread runs processVIF1Data, so the counters are plain scalars (no atomics).
enum VifProfCls { VP_TOTAL = 0, VP_UNPACK, VP_UNPACKWR, VP_DIRECT, VP_MPG, VP_MSCAL, VP_N };
static const bool g_vifProf = []
{ const char *e = std::getenv("PS2X_VIF_PROF"); return e && e[0] && e[0] != '0'; }();
// ⚠ PS2X_VIF_ABL bit 1 skips the UNPACK write loop. It is NOT a valid ceiling ablation and must
// never be read as one (cont.257, measured): with no unpack writes the VU1 microprograms run on
// stale VU data memory, loop on garbage counters and burn their whole 65536-cycle budget -- mscal
// went 4.97us -> 103.32us per call (20x) and the run made 787 frames instead of 5041. It CHANGES
// the workload instead of removing work (the cont.231 "ablation path drift" trap). Kept only to
// reproduce that result; use PS2X_VIF_SLOW2 to size the unpack path.
static const unsigned g_vifAbl = []
{ const char *e = std::getenv("PS2X_VIF_ABL"); return e && e[0] ? (unsigned)std::atoi(e) : 0u; }();
// ★ PS2X_VIF_SLOW2=1 (default OFF): the BIT-EXACT 2x-slow knob for the UNPACK write loop -- the
// doctrine's sizing tool (as PS2X_VU0_SLOW2 sized VU0 at 28% in cont.250). Runs the write loop
// twice over the same source; the wall-clock delta is what removing the path entirely would buy.
// Idempotent, hence bit-exact, for VIF modes 0/1/3: a rep only rewrites the values the previous
// rep wrote (write-protected fields keep their lane, un-decoded components re-read what they
// already held). Mode 2 writes back into vif1_regs.row, so it is NOT idempotent and is excluded
// -- this game issues mode 2 exactly zero times (cont.257 census, 59.4M unpacks).
static const bool g_vifSlow2 = []
{ const char *e = std::getenv("PS2X_VIF_SLOW2"); return e && e[0] && e[0] != '0'; }();
static const int g_vifProfEvery = []
{ const char *e = std::getenv("PS2X_VIF_PROF_EVERY"); const int v = e && e[0] ? std::atoi(e) : 10; return v > 0 ? v : 10; }();
static unsigned long long g_vifProfNs[VP_N] = {0}, g_vifProfN[VP_N] = {0};
static void vifProfReport();

struct VifProfScope
{
    std::chrono::steady_clock::time_point t0;
    int cls;
    explicit VifProfScope(int c) : cls(c)
    {
        if (g_vifProf)
            t0 = std::chrono::steady_clock::now();
    }
    ~VifProfScope()
    {
        if (!g_vifProf)
            return;
        g_vifProfNs[cls] += static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
        ++g_vifProfN[cls];
        if (cls == VP_TOTAL)
            vifProfReport();
    }
};

// Wall-clock gated, never call-gated (cont.251 lesson: a timeout(1) kill skips any end-of-run report).
static void vifProfReport()
{
    static std::chrono::steady_clock::time_point s_t0{}, s_last{};
    const auto now = std::chrono::steady_clock::now();
    if (s_t0.time_since_epoch().count() == 0)
    {
        s_t0 = s_last = now;
        std::fprintf(stderr, "[vif:prof] active (PS2X_VIF_PROF=1, every %ds, PS2X_VIF_ABL=%u)\n",
                     g_vifProfEvery, g_vifAbl);
        return;
    }
    if (now - s_last < std::chrono::seconds(g_vifProfEvery))
        return;
    s_last = now;
    const double wall = std::chrono::duration_cast<std::chrono::nanoseconds>(now - s_t0).count() / 1e9;
    const double tot = double(g_vifProfNs[VP_TOTAL]) / 1e9;
    const double unp = double(g_vifProfNs[VP_UNPACK]) / 1e9;
    const double uwr = double(g_vifProfNs[VP_UNPACKWR]) / 1e9;
    const double dir = double(g_vifProfNs[VP_DIRECT]) / 1e9;
    const double mpg = double(g_vifProfNs[VP_MPG]) / 1e9;
    const double msc = double(g_vifProfNs[VP_MSCAL]) / 1e9;
    const double oth = tot - unp - dir - mpg - msc;
    std::fprintf(stderr,
                 "[vif:prof] wall=%.1fs total=%.2fs (%.1f%% of wall) | unpack=%.2fs (%.1f%%, writeLoop=%.2fs)"
                 " direct=%.2fs (%.1f%%) mpg=%.2fs mscal/VU1=%.2fs (%.1f%%) other=%.2fs (%.1f%%)\n",
                 wall, tot, 100.0 * tot / wall,
                 unp, tot > 0 ? 100.0 * unp / tot : 0.0, uwr,
                 dir, tot > 0 ? 100.0 * dir / tot : 0.0,
                 mpg, msc, tot > 0 ? 100.0 * msc / tot : 0.0,
                 oth, tot > 0 ? 100.0 * oth / tot : 0.0);
    std::fprintf(stderr,
                 "[vif:prof] n total=%llu unpack=%llu direct=%llu mpg=%llu mscal=%llu | per-call"
                 " unpack=%.0fns direct=%.0fns mscal=%.2fus | clock-overhead<=%.2fs\n",
                 g_vifProfN[VP_TOTAL], g_vifProfN[VP_UNPACK], g_vifProfN[VP_DIRECT],
                 g_vifProfN[VP_MPG], g_vifProfN[VP_MSCAL],
                 g_vifProfN[VP_UNPACK] ? unp * 1e9 / double(g_vifProfN[VP_UNPACK]) : 0.0,
                 g_vifProfN[VP_DIRECT] ? dir * 1e9 / double(g_vifProfN[VP_DIRECT]) : 0.0,
                 g_vifProfN[VP_MSCAL] ? msc * 1e6 / double(g_vifProfN[VP_MSCAL]) : 0.0,
                 double(g_vifProfN[VP_TOTAL] + g_vifProfN[VP_UNPACK] + g_vifProfN[VP_UNPACKWR] +
                        g_vifProfN[VP_DIRECT] + g_vifProfN[VP_MPG] + g_vifProfN[VP_MSCAL]) * 45.0 / 1e9);
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (ps2gs::producerSide()) // cont.317 stage 1: the ring (docs/vu1-thread-design.md)
    {
        ps2gs::enqueueVif1(this, data, sizeBytes);
        ps2gs::pump();
        return;
    }
    struct EeProfVifScope
    {
        std::chrono::steady_clock::time_point t0;
        uint32_t bytes;
        ~EeProfVifScope()
        {
            if (!g_eeProf) return;
            g_eeProfVif1Ns.fetch_add(static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
            g_eeProfVif1Calls.fetch_add(1, std::memory_order_relaxed);
            g_eeProfVif1Bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
    } eeProfVifScope{g_eeProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, sizeBytes};
    VifProfScope vifProfTotal(VP_TOTAL);
    // Count the movie strips PRESENT IN THE INPUT BUFFER, before any VIF parsing. If this shows
    // 4 while the arbiter is submitted 3, VIF drops one; if it shows 3, the DMA that filled this
    // buffer is already short and the search moves upstream to the chain walker.
    vifScanStripColumns("INPUT", data, sizeBytes);

    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;

    // Cycle 86 (PS2X_VIF_DIRECTSPAN): an open DIRECT span from the previous buffer consumes
    // FIRST -- this data is mid-span Path2 GIF payload, never VIF commands. Delivery mirrors the
    // DIRECT handler: pending-image continuation (payload-aligned, cycle 85) first, then the
    // remainder submitted for tag parse with carry detection.
    g_vifEntrySpanQw = g_vif1DirectSpanRemainQw;
    // PS2X_VIF_SEQLOG: one line per VIF1 delivery -- size, entry span, first 16 bytes.
    // Ordering forensics (cont.156): does the ~900KB GIF continuation of the trailing
    // DIRECT imm=0 arrive as the NEXT delivery (hardware order) or displaced?
    {
        static const bool s_seqLog = []
        { const char *e = std::getenv("PS2X_VIF_SEQLOG"); return e && e[0] && e[0] != '0'; }();
        if (s_seqLog)
        {
            static unsigned long s_n = 0;
            ++s_n;
            char hb[40];
            for (int k = 0; k < 12 && k < (int)sizeBytes; ++k)
                std::snprintf(hb + k * 2, 3, "%02X", data[k]);
            std::fprintf(stderr, "[vif1:seq] #%lu size=%u entrySpan=%u head=%s\n",
                         s_n, sizeBytes, g_vifEntrySpanQw, hb);
        }
    }
    if (s_vifDirectSpan && g_vif1DirectSpanRemainQw != 0u)
    {
        const uint32_t spanQw = std::min<uint32_t>(g_vif1DirectSpanRemainQw, sizeBytes / 16u);
        if (spanQw != 0u)
        {
            static const bool s_spanLog = [] {
                const char *e = std::getenv("PS2X_VIF_DCLOG");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            if (s_spanLog)
            {
                static uint64_t s_n = 0ull;
                if (++s_n <= 200ull || (s_n % 1000ull) == 0ull)
                    std::fprintf(stderr,
                                 "[VIF:dspan] #%llu remain=%u chunk=%u buf=%u pendImg=%u\n",
                                 (unsigned long long)s_n, g_vif1DirectSpanRemainQw, spanQw,
                                 sizeBytes, m_vif1PendingPath2ImageQwc);
            }
            uint32_t imgQw = 0u;
            if (s_vifDirectCarry && m_vif1PendingPath2ImageQwc != 0u)
            {
                imgQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, spanQw);
                uint32_t left = imgQw;
                uint32_t off = 0u;
                while (left > 0u)
                {
                    const uint32_t chunk = std::min<uint32_t>(left, 0x7FFFu);
                    const bool final = (m_vif1PendingPath2ImageQwc == chunk);
                    std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunk) * 16u, 0u);
                    const uint64_t imageTag =
                        static_cast<uint64_t>(chunk) |
                        (final ? (1ull << 15) : 0ull) |
                        (static_cast<uint64_t>(kGifFmtImage) << 58);
                    std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
                    std::memcpy(imagePacket.data() + 16u, data + off,
                                static_cast<size_t>(chunk) * 16u);
                    submitGifPacket(GifPathId::Path2, imagePacket.data(),
                                    static_cast<uint32_t>(imagePacket.size()), true,
                                    m_vif1PendingPath2DirectHl);
                    m_vif1PendingPath2ImageQwc -= chunk;
                    left -= chunk;
                    off += chunk * 16u;
                }
                if (m_vif1PendingPath2ImageQwc == 0u)
                    m_vif1PendingPath2DirectHl = false;
            }
            const uint32_t restQw = spanQw - imgQw;
            if (restQw > 0u)
            {
                const uint8_t *restData = data + static_cast<size_t>(imgQw) * 16u;
                submitGifPacket(GifPathId::Path2, restData, restQw * 16u, true,
                                g_vif1DirectSpanHl);
                const uint32_t carry =
                    s_gifResumeOwnsSplits ? 0u : directTrailingImageQwc(restData, restQw * 16u);
                if (carry != 0u)
                {
                    m_vif1PendingPath2ImageQwc = carry;
                    m_vif1PendingPath2DirectHl = g_vif1DirectSpanHl;
                }
            }
            g_vif1DirectSpanRemainQw -= spanQw;
            pos = spanQw * 16u;
        }
    }

    while (pos + 4 <= sizeBytes)
    {
        // With PS2X_VIF_DIRECTCARRY the pending image is consumed inside the DIRECT handler
        // (payload-aligned); this position-blind raw consume would eat the REFVIF-embedded tag
        // words as pixels. Kept for the legacy (=0) path only.
        if (!s_vifDirectCarry && m_vif1PendingPath2ImageQwc != 0u)
        {
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, availableQw);

            // PS2X_VIF_IMGCONSUME=1 (probe, cycle 82b). Disambiguate fix #1's failure: DETECTION
            // (wrong carry amount) vs CONSUMPTION (this synthesized-image path). Log each consume:
            // pending, availableQw, chunkQw, and the 16 bytes at data+pos being fed as image -- if
            // they are NOT plausible image pixels (e.g. they look like a GIFtag / VIF command), the
            // carry amount was wrong and we are eating real stream (detection bug). Correlate with
            // TRXORIGIN: if the synthesized transfers are OK, consumption works and detection is the
            // culprit; if they are BAD, the consumption/transfer-state continuation is itself broken.
            static const bool s_imgConsumeLog = [] {
                const char *e = std::getenv("PS2X_VIF_IMGCONSUME");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            if (s_imgConsumeLog)
            {
                static uint64_t s_n = 0ull;
                if (++s_n <= 60ull || (s_n % 500ull) == 0ull)
                {
                    char hb[52];
                    hb[0] = '\0';
                    for (int k = 0; k < 16 && (pos + k) < sizeBytes; ++k)
                        std::snprintf(hb + k * 3, 4, "%02x ", data[pos + k]);
                    std::fprintf(stderr,
                                 "[VIF:imgconsume] #%llu pending=%u availQw=%u chunkQw=%u pos=%u/%u | img: %s\n",
                                 (unsigned long long)s_n, m_vif1PendingPath2ImageQwc, availableQw,
                                 chunkQw, pos, sizeBytes, hb);
                }
            }

            std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
            const uint64_t imageTag =
                static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                (static_cast<uint64_t>(kGifFmtImage) << 58);
            std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
            std::memcpy(imagePacket.data() + 16u, data + pos, static_cast<size_t>(chunkQw) * 16u);
            submitGifPacket(GifPathId::Path2,
                            imagePacket.data(),
                            static_cast<uint32_t>(imagePacket.size()),
                            true,
                            m_vif1PendingPath2DirectHl);

            pos += chunkQw * 16u;
            m_vif1PendingPath2ImageQwc -= chunkQw;
            if (m_vif1PendingPath2ImageQwc == 0u)
            {
                m_vif1PendingPath2DirectHl = false;
            }
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;
        // Command ring (cont.156): last 32 VIF1 commands, dumped by the directbig probe to find
        // where the command-stream walk desyncs into data (phantom DIRECTHL family).
        g_vifCmdRing[g_vifCmdRingIdx & 31u] = {pos - 4u, sizeBytes, cmd};
        ++g_vifCmdRingIdx;
        // One-shot big-buffer capture (PS2X_VIF_BUFDUMP=<minBytes>): raw buffer + every decoded
        // command, for offline re-walks (test size formulas without rebuilds).
        {
            static const uint32_t s_bufDumpMin = [] {
                const char *e = std::getenv("PS2X_VIF_BUFDUMP");
                return (e && e[0]) ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 0u;
            }();
            static FILE *s_bufTrace = nullptr;
            static const uint8_t *s_bufTarget = nullptr;
            if (s_bufDumpMin != 0u && sizeBytes >= s_bufDumpMin && !s_bufTarget)
            {
                if (FILE *bf = std::fopen("tmp/vifbuf.bin", "wb"))
                {
                    std::fwrite(data, 1, sizeBytes, bf);
                    std::fclose(bf);
                }
                s_bufTrace = std::fopen("tmp/viftrace.txt", "w");
                s_bufTarget = data;
                std::fprintf(stderr, "[VIF:bufdump] captured %u bytes -> tmp/vifbuf.bin\n", sizeBytes);
            }
            if (s_bufTrace && s_bufTarget == data)
            {
                std::fprintf(s_bufTrace, "%u %08X\n", pos - 4u, cmd);
                if (pos + 4u > sizeBytes)
                {
                    std::fclose(s_bufTrace);
                    s_bufTrace = nullptr;
                }
            }
        }

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (ps2xGifTraceOn()) std::fprintf(stderr, "[gif:trace] flip=%llu MSKPATH3 %s (was %d)\n", ps2xGsPerfFlips(), m_path3Masked ? "SET" : "CLEAR", wasMasked ? 1 : 0);
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            VifProfScope vps__(VP_MSCAL);
            uint32_t startPC = (uint32_t)imm * 8u;

            // Values visible to the VU program for this MSCAL.
            // DobieStation semantics: ITOP = ITOPS; TOP = current TOPS;
            // then TOPS/DBF are prepared for the next buffer.
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, runTop, runItop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            VifProfScope vps__(VP_MSCAL);
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            VifProfScope vps__(VP_MPG);
            uint32_t destAddr = (uint32_t)imm * 8u;
            // VIF MPG semantics: NUM==0 means 256 instructions (2048 bytes).
            // MPG payload is instruction-packed and should not be QW-aligned.
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                    markVU1CodeModified();
                }
                else
                {
                    // Same silent-skip-on-span shape as the (falsified) unpack case — measure it.
                    static unsigned long s_mpgSpan = 0;
                    ++s_mpgSpan;
                    if (s_mpgSpan <= 12u || (s_mpgSpan % 256u) == 0u)
                        std::fprintf(stderr, "[vif:MPG-SPAN] n=%lu dest=0x%x bytes=%u avail=%u\n",
                                     s_mpgSpan, destAddr, copyBytes, sizeBytes - pos);
                }
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            VifProfScope vps__(VP_DIRECT);
            uint32_t qwCount = imm;
            // ★ cont.156: DIRECT imm=0 consumes NOTHING in this engine's chain-flattened
            // delivery. The game's frame chains carry a terminal-and-mid-list [NOP][DIRECT
            // imm=0] embed followed by MORE embed-wrapped mini-DIRECTs and real VIF commands
            // (STCYCL/UNPACK matrix uploads); with imm=0 -> 0 the captured 3.7 MB level chain
            // parses PERFECTLY at both layers (VIF walk consumes the buffer exactly, 0 unknown
            // opcodes; the concatenated 3.16 MB GIF stream = 184 tags, 0 implausible, 0
            // overrun -- offline proof, vifbuf_bad.bin). With the nominal 65536-qw reading it
            // swallowed ~1 MB of the chain (embeds decode as GIF NOPs so it half-worked,
            // until UNPACK float payloads got read as tags -> the phantom-PACKED garbage
            // register writes). PCSX2 uses 65536 on the raw DMA stream; our flattened
            // delivery differs structurally. PS2X_VIF_IMM0DIRECT=65536 restores the old read.
            static const uint32_t s_imm0Direct = [] {
                const char *e = std::getenv("PS2X_VIF_IMM0DIRECT");
                return e && e[0] ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 0u;
            }();
            if (qwCount == 0)
                qwCount = s_imm0Direct;
            const uint32_t declaredQw = qwCount;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            // PS2X_VIF_DIRECTBIG=<qw> (diagnostic, default OFF). Cycle 77: the malformed post-FMV
            // BITBLTBUF/TRXREG come from 164 KB - 902 KB buffers that are NOT GIF data (one holds
            // a 4x4 float matrix, another a pixel gradient) and whose byte 0 is already not a
            // GIFtag. This is the only site that can submit a slab that large: `imm == 0` means
            // 65536 qw (1 MB) -- PCSX2-faithful (`Vif_Codes.cpp:148`,
            // `vif1.tag.size = vifImm ? (vifImm*4) : (65536*4)`) -- clamped here to the rest of
            // the buffer. So the question is whether we are REACHING this command on data that
            // was never a VIF command stream. Dump the command in context: if the preceding bytes
            // do not look like VIF commands, the chain buffer is carrying non-VIF payload and the
            // desync is upstream, in what the chain walker concatenated.
            static const uint32_t s_directBig = [] {
                const char *e = std::getenv("PS2X_VIF_DIRECTBIG");
                return (e != nullptr && e[0] != '\0') ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 0u;
            }();
            if (s_directBig != 0u && qwCount >= s_directBig)
            {
                static uint64_t s_n = 0ull;
                if (++s_n <= 30ull)
                {
                    std::fprintf(stderr,
                                 "[VIF:directbig] #%llu imm=%u qw=%u (%u B) pos=%u/%u trunc=%d hl=%d\n",
                                 (unsigned long long)s_n, (unsigned)imm, qwCount, qwCount * 16u,
                                 pos, sizeBytes, truncated ? 1 : 0,
                                 (opcode == VIF_DIRECTHL) ? 1 : 0);
                    const uint32_t from = (pos >= 32u) ? (pos - 32u) : 0u;
                    for (uint32_t o = from; o < pos + 32u && o < sizeBytes; o += 16u)
                    {
                        std::fprintf(stderr, "    %s +%06X  ", (o < pos) ? "pre " : "pay ", o);
                        for (uint32_t b = 0; b < 16u && (o + b) < sizeBytes; ++b)
                            std::fprintf(stderr, "%02X", data[o + b]);
                        std::fprintf(stderr, "\n");
                    }
                    // Retroactive bad-buffer dump: the EXACT instance that produced this
                    // phantom, for offline ground-truth walks. One-shot.
                    {
                        static bool s_badDumped = false;
                        if (!s_badDumped && std::getenv("PS2X_VIF_BUFDUMP"))
                        {
                            s_badDumped = true;
                            if (FILE *bf = std::fopen("tmp/vifbuf_bad.bin", "wb"))
                            {
                                std::fwrite(data, 1, sizeBytes, bf);
                                std::fclose(bf);
                            }
                            std::fprintf(stderr,
                                         "[VIF:baddump] wrote %u bytes, phantom at pos=%u entrySpanQw=%u\n",
                                         sizeBytes, pos, g_vifEntrySpanQw);
                        }
                    }
                    std::fprintf(stderr, "    ring (oldest->newest):\n");
                    for (uint32_t k = 0; k < 32u; ++k)
                    {
                        const VifCmdRingEnt &e = g_vifCmdRing[(g_vifCmdRingIdx + k) & 31u];
                        if (e.bufSize == 0u)
                            continue;
                        std::fprintf(stderr, "      pos=%u/%u cmd=%08X op=%02X num=%u imm=%u\n",
                                     e.pos, e.bufSize, e.cmd,
                                     (e.cmd >> 24) & 0x7Fu, (e.cmd >> 16) & 0xFFu, e.cmd & 0xFFFFu);
                    }
                }
            }

            if (qwCount > 0)
            {
                const bool directHl = (opcode == VIF_DIRECTHL);

                // ★ Cycle-85 fix (PS2X_VIF_DIRECTCARRY): a pending Path2 image continuation is
                // satisfied by THIS DIRECT's payload -- the game wraps every image chunk in its
                // ref tag's TTE DIRECT (measured, run_c85D chain #3), so the payload IS the
                // continuation. Synthesize the IMAGE-tagged packet exactly like the legacy
                // top-of-loop consume did, but payload-aligned: the embedded tag words upstream
                // were already parsed as VIF commands and never leak into the image.
                uint32_t imgQw = 0u;
                if (s_vifDirectCarry && m_vif1PendingPath2ImageQwc != 0u)
                {
                    imgQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, qwCount);
                    // PS2X_VIF_DCLOG=1 (probe, cycle 86, no behaviour change): each payload-aligned
                    // consume with its pending/span/truncation context -- correlates a consume with
                    // the gifwalk/TRXORIGIN records that follow it.
                    static const bool s_dcLog = [] {
                        const char *e = std::getenv("PS2X_VIF_DCLOG");
                        return e != nullptr && e[0] != '\0' && e[0] != '0';
                    }();
                    if (s_dcLog)
                    {
                        static uint64_t s_n = 0ull;
                        if (++s_n <= 200ull || (s_n % 1000ull) == 0ull)
                            std::fprintf(stderr,
                                         "[VIF:dcarry] #%llu pending=%u dirQw=%u imgQw=%u pos=%u/%u trunc=%d\n",
                                         (unsigned long long)s_n, m_vif1PendingPath2ImageQwc,
                                         qwCount, imgQw, pos, sizeBytes, truncated ? 1 : 0);
                    }
                    uint32_t left = imgQw;
                    uint32_t off = 0u;
                    while (left > 0u)
                    {
                        // nloop is 15-bit; a bigger chunk must be split or the tag under-declares.
                        const uint32_t chunk = std::min<uint32_t>(left, 0x7FFFu);
                        const bool final = (m_vif1PendingPath2ImageQwc == chunk);
                        std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunk) * 16u, 0u);
                        const uint64_t imageTag =
                            static_cast<uint64_t>(chunk) |
                            (final ? (1ull << 15) : 0ull) |
                            (static_cast<uint64_t>(kGifFmtImage) << 58);
                        std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
                        std::memcpy(imagePacket.data() + 16u, data + pos + off,
                                    static_cast<size_t>(chunk) * 16u);
                        submitGifPacket(GifPathId::Path2, imagePacket.data(),
                                        static_cast<uint32_t>(imagePacket.size()), true,
                                        m_vif1PendingPath2DirectHl);
                        m_vif1PendingPath2ImageQwc -= chunk;
                        left -= chunk;
                        off += chunk * 16u;
                    }
                    if (m_vif1PendingPath2ImageQwc == 0u)
                        m_vif1PendingPath2DirectHl = false;
                }
                const uint32_t restQw = qwCount - imgQw;
                const uint8_t *restData = data + pos + static_cast<size_t>(imgQw) * 16u;

                if (restQw > 0u)
                {
                // PS2X_GS_IMGCONT (cycle 83): sample the GS pending-image BEFORE this DIRECT so we
                // can tell whether THIS DIRECT opened a fresh image that overran (before==0,
                // after>0) vs a stale/pre-existing incomplete transfer (before>0) -- the latter must
                // NOT trigger a carry or we false-fire on geometry DIRECTs (cycle 89 over-fired).
                static const int s_imgCont = [] {
                    const char *e = std::getenv("PS2X_GS_IMGCONT");
                    return (e && e[0] && e[0] != '0') ? 1 : 0;
                }();
                const uint32_t pendBefore =
                    (s_imgCont && m_gsPendingImageFn) ? m_gsPendingImageFn() : 0u;
                vifScanStripColumns("DIRECT", restData, restQw * 16u);
                submitGifPacket(GifPathId::Path2, restData, restQw * 16, true, directHl);

                // Image-continuation carry-over. OLD detection (gifImageQwcFromTag) reads only the
                // FIRST tag of the payload and misses an IMAGE tag behind a setup tag (cycle 81);
                // NEW walks the payload's tags (directTrailingImageQwc). PS2X_GS_DIRECTIMGFIX=1
                // selects NEW (default OFF -- unproven until the continuation is verified raw).
                static const int s_dirImgFix = [] {
                    const char *e = std::getenv("PS2X_GS_DIRECTIMGFIX");
                    return (e && e[0] && e[0] != '0') ? 1 : 0;
                }();
                const uint32_t oldImageQw = gifImageQwcFromTag(restData, restQw * 16u);
                const uint32_t oldCarry =
                    (oldImageQw > (restQw - 1u)) ? (oldImageQw - (restQw - 1u)) : 0u;
                const uint32_t newCarry =
                    s_gifResumeOwnsSplits ? 0u : directTrailingImageQwc(restData, restQw * 16u);

                // PS2X_GS_DIRECTIMGLOG=1 (probe, no behaviour change). Where NEW disagrees with OLD,
                // dump qwCount, first-tag flg, both carries, AND the 16 bytes FOLLOWING this DIRECT
                // (pos+qwCount*16): if the opcode byte (offset +3) is 0x50/0x51 the continuation is
                // another DIRECT command (my raw-image assumption is WRONG); otherwise it is raw
                // image data (assumption holds). This is the cycle-82 verify-before-fix gate.
                static const bool s_dirImgLog = [] {
                    const char *e = std::getenv("PS2X_GS_DIRECTIMGLOG");
                    return e && e[0] && e[0] != '0';
                }();
                if (s_dirImgLog && newCarry != oldCarry)
                {
                    static uint64_t s_n = 0ull;
                    if (++s_n <= 60ull || (s_n % 500ull) == 0ull)
                    {
                        uint64_t t0 = 0u;
                        std::memcpy(&t0, restData, sizeof(t0));
                        const uint32_t nextOff = pos + qwCount * 16u;
                        char nb[64];
                        nb[0] = '\0';
                        if (nextOff + 16u <= sizeBytes)
                            for (int k = 0; k < 16; ++k)
                                std::snprintf(nb + k * 3, 4, "%02x ", data[nextOff + k]);
                        else
                            std::snprintf(nb, sizeof(nb), "(end of buffer)");
                        std::fprintf(stderr,
                                     "[VIF:dirimg] #%llu qw=%u firstFlg=%u oldCarry=%u newCarry=%u "
                                     "trunc=%d | next: %s\n",
                                     (unsigned long long)s_n, qwCount,
                                     (unsigned)((t0 >> 58) & 0x3u), oldCarry, newCarry,
                                     truncated ? 1 : 0, nb);
                    }
                }

                // Carry ONLY when THIS DIRECT opened a fresh image that overran its payload
                // (pendBefore==0 && pendAfter>0). A pre-existing incomplete transfer (pendBefore>0)
                // is either already being carried or is stale garbage -- either way, do not fire a
                // new carry on it. Reliable and false-positive-free at the SOURCE, unlike cycle 82's
                // re-walk and cycle 89's stale-transfer over-fire. Default OFF pending validation.
                uint32_t carry;
                if (s_gifResumeOwnsSplits)
                {
                    carry = 0u; // the frontend's resumable parser owns split tags
                }
                else if (s_imgCont)
                {
                    const uint32_t pendAfter =
                        m_gsPendingImageFn ? m_gsPendingImageFn() : 0u;
                    carry = (pendBefore == 0u && pendAfter > 0u) ? pendAfter : 0u;
                }
                else
                {
                    // Under DIRECTCARRY the tag-walking detector is the correct one: it finds an
                    // IMAGE tag sitting BEHIND setup tags (the normal texture-upload shape the
                    // first-tag-only detector misses -- cycle 81's diagnosed gap, the level's
                    // un-consumed 233 KB continuation family).
                    carry = (s_vifDirectCarry || s_dirImgFix) ? newCarry : oldCarry;
                }
                if (carry != 0u)
                {
                    m_vif1PendingPath2ImageQwc = carry;
                    m_vif1PendingPath2DirectHl = directHl;
                }
                } // restQw > 0
            }

            pos += qwCount * 16;
            if (truncated)
            {
                // Cycle 86 (PS2X_VIF_DIRECTSPAN): the un-delivered remainder of this DIRECT
                // continues at the next processVIF1Data entry instead of being forgotten.
                // ★ cont.156 refinement: ONLY for imm != 0. The game ends EVERY frame chain
                // with a terminal [NOP][DIRECT imm=0] tag embed followed by nothing -- on
                // hardware that construct is benign, and arming a 65536-qw span from it made
                // the carry eat the first ~900 KB of every subsequent frame chain (the
                // phantom-DIRECTHL corruption cascade). The original imm=0-span evidence
                // (run_c86A) was itself the half-phase misparse artifact that PS2X_DMA_TTEQW
                // later fixed. Sized DIRECTs (imm != 0) still span (the FMV's 32767-qw
                // uploads legitimately split across chain buffers). PS2X_VIF_IMM0SPAN=1
                // restores the old behavior.
                static const bool s_imm0Span = [] {
                    const char *e = std::getenv("PS2X_VIF_IMM0SPAN");
                    return e && e[0] && e[0] != '0';
                }();
                if (s_vifDirectSpan && (imm != 0u || s_imm0Span))
                {
                    g_vif1DirectSpanRemainQw = declaredQw - qwCount;
                    g_vif1DirectSpanHl = (opcode == VIF_DIRECTHL);
                    {
                        static const bool s_spanSetLog = []
                        { const char *e = std::getenv("PS2X_VIF_CTLLOG"); return e && e[0] && e[0] != '0'; }();
                        if (s_spanSetLog)
                        {
                            static unsigned long s_n = 0;
                            ++s_n;
                            if (s_n <= 60u || (s_n % 256u) == 0u)
                                std::fprintf(stderr,
                                             "[vif1:spanset] #%lu remain=%u declared=%u fed=%u pos=%u/%u imm=%u\n",
                                             s_n, g_vif1DirectSpanRemainQw, declaredQw, qwCount,
                                             pos, sizeBytes, (unsigned)imm);
                        }
                    }
                }
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            VifProfScope vps__(VP_UNPACK);
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            // VIF1 unpack census (PS2X_VIF_CENSUS, default OFF): aggregate combo counters,
            // printed every 8192 unpacks — which unpack shapes the game actually uses.
            {
                static const bool s_vifCensus = []
                { const char *e = std::getenv("PS2X_VIF_CENSUS"); return e && e[0] && e[0] != '0'; }();
                if (s_vifCensus)
                {
                    static unsigned long s_n = 0, s_byVlVn[16] = {0}, s_mask = 0, s_fill = 0, s_skip = 0,
                                         s_mode[4] = {0};
                    ++s_n;
                    ++s_byVlVn[(vl & 3u) * 4u + (vn & 3u)];
                    if (maskEnable)
                        ++s_mask;
                    if (cl < wl)
                        ++s_fill;
                    else if (cl > wl)
                        ++s_skip;
                    ++s_mode[vif1_regs.mode & 3u];
                    // cont.317: VECTOR-weighted census of the full unpack shape. Calls are not
                    // cost; the write loop runs per vector, so this is what a fast path must serve.
                    {
                        static std::mutex s_shapeMutex;
                        static std::unordered_map<uint32_t, std::pair<unsigned long long, unsigned long long>> s_shape; // key -> (vectors, calls)
                        static unsigned long long s_vecTotal = 0ull;
                        const uint32_t key = (vl & 3u) | ((vn & 3u) << 2) | ((maskEnable ? 1u : 0u) << 4) |
                                             ((vif1_regs.mode & 3u) << 5) | ((cl < wl ? 1u : cl > wl ? 2u : 0u) << 7) |
                                             (((imm & 0x4000u) ? 1u : 0u) << 9) | ((cl & 0xFu) << 10) | ((wl & 0xFu) << 14) |
                                             ((maskEnable ? (vif1_regs.mask & 0xFFFFu) : 0u) << 18);
                        std::lock_guard<std::mutex> lock(s_shapeMutex);
                        auto &e = s_shape[key];
                        e.first += writeVectorCount;
                        ++e.second;
                        s_vecTotal += writeVectorCount;
                        if ((s_n % 65536u) == 0u)
                        {
                            std::vector<std::pair<uint32_t, std::pair<unsigned long long, unsigned long long>>> v(s_shape.begin(), s_shape.end());
                            std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.first > b.second.first; });
                            std::fprintf(stderr, "[vif:shape] vectors=%llu calls=%lu | vl.vn mask mode cyc(0=cl==wl,1=fill,2=skip) zext cl wl mask16   vectors%%  calls  vec/call\n", s_vecTotal, s_n);
                            for (size_t i = 0; i < v.size() && i < 14u; ++i)
                            {
                                const uint32_t k = v[i].first;
                                std::fprintf(stderr, "[vif:shape]  %u.%u   %u    %u    %u          %u   %2u %2u  %04x   %6.2f%% %8llu %7.1f\n",
                                             k & 3u, (k >> 2) & 3u, (k >> 4) & 1u, (k >> 5) & 3u, (k >> 7) & 3u, (k >> 9) & 1u,
                                             (k >> 10) & 0xFu, (k >> 14) & 0xFu, (k >> 18) & 0xFFFFu,
                                             100.0 * double(v[i].second.first) / double(s_vecTotal), v[i].second.second,
                                             double(v[i].second.first) / double(v[i].second.second));
                            }
                        }
                    }
                    {
                        static unsigned long s_spanned = 0;
                        uint32_t sourceVectorCountPre = writeVectorCount;
                        if (cl < wl)
                        {
                            const uint32_t fb = writeVectorCount / wl;
                            uint32_t rem = writeVectorCount % wl;
                            if (rem > cl)
                                rem = cl;
                            sourceVectorCountPre = fb * cl + rem;
                        }
                        uint32_t tb = sourceVectorCountPre * bytesPerVector;
                        tb = (tb + 3u) & ~3u;
                        if (pos + tb > sizeBytes)
                        {
                            ++s_spanned;
                            if (s_spanned <= 12u || (s_spanned % 512u) == 0u)
                                std::fprintf(stderr,
                                             "[vif:unpack-SPAN] n=%lu payload=%u avail=%u vl=%u vn=%u num=%u addr=0x%x cl=%u wl=%u\n",
                                             s_spanned, tb, sizeBytes - pos, vl, vn,
                                             static_cast<unsigned>(num), static_cast<unsigned>(imm & 0x3FFu), cl, wl);
                        }
                    }
                    if ((s_n % 8192u) == 0u)
                    {
                        std::fprintf(stderr,
                                     "[vif:unpack] n=%lu mask=%lu fill=%lu skip=%lu mode{0:%lu 1:%lu 2:%lu 3:%lu} "
                                     "vl.vn{", s_n, s_mask, s_fill, s_skip, s_mode[0], s_mode[1], s_mode[2], s_mode[3]);
                        for (unsigned i = 0; i < 16u; ++i)
                            if (s_byVlVn[i])
                                std::fprintf(stderr, " %u.%u:%lu", i / 4u, i % 4u, s_byVlVn[i]);
                        std::fprintf(stderr, " } row=%08x/%08x/%08x/%08x cl=%u wl=%u num=%u addr=0x%x\n",
                                     vif1_regs.row[0], vif1_regs.row[1], vif1_regs.row[2], vif1_regs.row[3],
                                     cl, wl, static_cast<unsigned>(num), static_cast<unsigned>(imm & 0x3FFu));
                    }
                }
            }

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;

            // Failing-buffer watch (part of PS2X_VIF_CENSUS): the level-era kick-stub kicks qw
            // 0x10 and finds garbage — log every unpack whose dest range covers it, plus bump a
            // generation counter the kick-drop dump reads (extern, defined here).
            {
                extern unsigned long g_vifQw10WriteGen;
                static const bool s_vifCensus2 = []
                { const char *e = std::getenv("PS2X_VIF_CENSUS"); return e && e[0] && e[0] != '0'; }();
                if (s_vifCensus2)
                {
                    const uint32_t destEnd = vuAddr + writeVectorCount;
                    if (vuAddr <= 0x10u && destEnd > 0x10u)
                    {
                        ++g_vifQw10WriteGen;
                        static unsigned long s_w10 = 0;
                        ++s_w10;
                        if (s_w10 <= 20u || (s_w10 % 4096u) == 0u)
                        {
                            uint32_t s0 = 0, s1 = 0;
                            if (pos + 8u <= sizeBytes)
                            {
                                std::memcpy(&s0, data + pos, 4u);
                                std::memcpy(&s1, data + pos + 4u, 4u);
                            }
                            const uint8_t *rdramBase = m_rdram;
                            long srcRam = -1;
                            if (rdramBase && data >= rdramBase && data < rdramBase + PS2_RAM_SIZE)
                                srcRam = static_cast<long>(data - rdramBase) + static_cast<long>(pos);
                            std::fprintf(stderr,
                                         "[vif:w10] n=%lu gen=%lu vl=%u vn=%u num=%u addr=0x%x(+tops=%u) mode=%u mask=%u cl=%u wl=%u src0=%08x %08x srcRam=0x%lx\n",
                                         s_w10, g_vifQw10WriteGen, vl, vn, static_cast<unsigned>(num),
                                         vuAddr, (imm & 0x8000u) ? 1u : 0u, vif1_regs.mode & 3u,
                                         maskEnable ? 1u : 0u, cl, wl, s0, s1, srcRam);
                        }
                    }
                }
            }

            if (m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes && !(g_vifAbl & 1u))
            {
                VifProfScope vpsw__(VP_UNPACKWR);
                const uint8_t *srcBase = data + pos;
                // cont.317: the SSE write loop first (see vifFastUnpackRun); the generic loop
                // below runs only when the shape is not served, or under PS2X_VIF_FASTVERIFY as
                // the oracle (it re-runs the generic loop on a scratch copy and compares).
                bool tookFast = false;
                bool verifyPending = false;
                alignas(16) static thread_local uint8_t s_verifyRef[PS2_VU1_DATA_SIZE];
                alignas(16) static thread_local uint8_t s_verifyFast[PS2_VU1_DATA_SIZE];
                if (s_vifFastUnpack && !g_vifSlow2)
                {
                    if (s_vifFastVerify)
                        std::memcpy(s_verifyRef, m_vu1Data, PS2_VU1_DATA_SIZE);
                    tookFast = (s_vifFastUnpackKind == 2)
                                   ? vifFastUnpackRunTpl(m_vu1Data, vif1_regs, srcBase, vl, vn, zeroExtend, maskEnable,
                                                         cl, wl, writeVectorCount, vuAddr)
                                   : vifFastUnpackRun(m_vu1Data, vif1_regs, srcBase, vl, vn, zeroExtend, maskEnable,
                                                      cl, wl, writeVectorCount, vuAddr);
                    if (tookFast) ++g_vifFastTaken; else ++g_vifFastFallback;
                    if (tookFast && s_vifFastVerify)
                    {
                        // Oracle: keep the fast result, restore the pre-unpack image, let the generic
                        // loop below run on it, then compare the two VU images.
                        std::memcpy(s_verifyFast, m_vu1Data, PS2_VU1_DATA_SIZE);
                        std::memcpy(m_vu1Data, s_verifyRef, PS2_VU1_DATA_SIZE);
                        tookFast = false;
                        verifyPending = true;
                    }
                }
                // PS2X_VIF_SLOW2: repeat the (idempotent) write loop to size the path. Mode 2 is
                // excluded because it accumulates into vif1_regs.row.
                const unsigned repN__ = tookFast ? 0u : ((g_vifSlow2 && (vif1_regs.mode & 3u) != 2u) ? 2u : 1u);
                for (unsigned rep__ = 0; rep__ < repN__; ++rep__)
                {
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        ++srcIndex;
                        decoded = true;
                    }

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                decompressed[c] = extend8(srcVec[c]);
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: RGBA5551 in one 16-bit value. ★ cont.324d PS2X_VIF_V45 (default 1; "=0" = the old
                        // decode): the hardware expands the 5-bit fields to 8-bit colour and the alpha bit to 0x80 --
                        // PCSX2 Vif_Unpack.cpp UNPACK_V4_5: X=(d&0x1f)<<3, Y=(d&0x3e0)>>2, Z=(d&0x7c00)>>7,
                        // W=(d&0x8000)>>8 (mode is ignored for V4-5, "act as mode==0 always"). The old decode
                        // handed VU1 the RAW 5-bit values and a 0/1 alpha: every colour 8x too dark, alpha never
                        // 0x80. This game issues it unmasked, mode 0, ~327k unpacks per 150 s of the fight.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        if (s_vifV45)
                        {
                            decompressed[0] = (packed & 0x1Fu) << 3;
                            decompressed[1] = (packed & 0x3E0u) >> 2;
                            decompressed[2] = (packed & 0x7C00u) >> 7;
                            decompressed[3] = (packed & 0x8000u) >> 8;
                        }
                        else
                        {
                            decompressed[0] = packed & 0x1Fu;
                            decompressed[1] = (packed >> 5) & 0x1Fu;
                            decompressed[2] = (packed >> 10) & 0x1Fu;
                            decompressed[3] = (packed >> 15) & 0x01u;
                        }
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t mode = vif1_regs.mode & 3u;
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                }
                }
                if (verifyPending)
                {
                    ++g_vifFastVerifyChecks;
                    if (std::memcmp(m_vu1Data, s_verifyFast, PS2_VU1_DATA_SIZE) != 0)
                    {
                        ++g_vifFastVerifyMismatches;
                        if (g_vifFastVerifyMismatches <= 8ull)
                        {
                            uint32_t first = 0;
                            while (first < PS2_VU1_DATA_SIZE && m_vu1Data[first] == s_verifyFast[first]) ++first;
                            uint32_t g = 0, f = 0;
                            std::memcpy(&g, m_vu1Data + (first & ~3u), 4u);
                            std::memcpy(&f, s_verifyFast + (first & ~3u), 4u);
                            std::fprintf(stderr, "[vif:fastverify] MISMATCH #%llu qw %u lane %u: generic=%08x fast=%08x | vl=%u vn=%u mask=%u(%08x) mode=%u cl=%u wl=%u zext=%d num=%u addr=0x%x\n",
                                         g_vifFastVerifyMismatches, first / 16u, (first % 16u) / 4u, g, f,
                                         vl, vn, maskEnable ? 1u : 0u, vif1_regs.mask, vif1_regs.mode & 3u, cl, wl,
                                         zeroExtend ? 1 : 0, writeVectorCount, vuAddr);
                        }
                    }
                    if ((g_vifFastVerifyChecks % 1000000ull) == 0ull)
                        std::fprintf(stderr, "[vif:fastverify] checks=%llu mismatches=%llu taken=%llu fallback=%llu\n",
                                     g_vifFastVerifyChecks, g_vifFastVerifyMismatches, g_vifFastTaken, g_vifFastFallback);
                }
            }
            pos += totalBytes;

            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}

// Definition for the qw-0x10 write-generation watch (read by the VU1 kick-drop dump).
unsigned long g_vifQw10WriteGen = 0;
