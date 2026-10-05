// cont.317 — GS-pipeline capture/replay oracle. See include/runtime/ps2_pipe_capture.h and
// docs/vu1-thread-design.md (stage 0).
//
// Capture (EE thread only): PS2X_PIPE_CAP=<seconds after start> arms it; the first
// processPendingTransfers pass at or after that time whose state is QUIESCENT (no open VIF DIRECT
// span, no pending path-2 image, path 3 unmasked with an empty masked FIFO, the GIF parser idle on
// every path) writes the header -- the VU1 code and data images, the VIF1 registers -- and then
// every pipeline input in program order: VIF1 data chunks, path-3 packets (with their
// drain-immediately flag), the explicit end-of-pass drains, VIF1 register-block writes and guest
// stores into VU1 memory, for PS2X_PIPE_CAP_N passes (default 400).
//
// Replay: PS2X_PIPE_BENCH=<file> [PS2X_PIPE_BENCH_REPS=n]: restore the header, feed the events
// through the real PS2Memory / VIF1 interpreter / microVU / arbiter / GS frontend, hash every
// packet the arbiter hands the GS (path id, size, bytes; FNV-1a 64) and the VU1 data image after
// the pass, print per-rep wall time, and exit. Rep 0 starts from a fresh GS frontend; later reps
// start from whatever GS state rep 0 left (the parse of a split IMAGE can depend on it), so the
// ORACLE is rep 0's hash and the BENCH is the rep timing. Run with PS2X_GS_NORASTER=1 to time the
// pipeline alone.
#include "runtime/ps2_pipe_capture.h"
#include "runtime/ps2_gs_pipeline.h"
#include "ps2_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void ps2xVif1ResetDirectSpan();
extern "C" int ps2xGsGifParseQuiescent();
extern "C" void ps2xGsGifParseReset();
extern "C" void (*ps2xGifArbiterTap)(uint32_t pathId, const uint8_t *data, uint32_t size);
extern "C" void ps2xGifArbiterDrainCurrent();

namespace
{
    enum : uint32_t { kEvVif1 = 1, kEvGif3 = 2, kEvDrain = 3, kEvVifReg = 4, kEvVu1Mem = 5, kEvPassEnd = 6, kEvGsReg = 7, kEvGsClear = 8 };
    constexpr char kMagic[4] = {'P', 'I', 'P', 'C'};
    constexpr uint32_t kVersion = 1u;

    struct CapState
    {
        bool checked = false;
        double armAtSec = -1.0;
        unsigned passesWanted = 400u;
        std::string file;
        std::FILE *f = nullptr;
        bool recording = false;
        bool done = false;
        unsigned passesRecorded = 0u;
        unsigned long long events = 0ull, bytes = 0ull;
        std::chrono::steady_clock::time_point t0;
        unsigned notQuiescent = 0u;
    } g_cap;

    void capInit()
    {
        if (g_cap.checked) return;
        g_cap.checked = true;
        const char *e = std::getenv("PS2X_PIPE_CAP");
        if (!e || !e[0]) return;
        g_cap.armAtSec = std::atof(e);
        const char *n = std::getenv("PS2X_PIPE_CAP_N");
        if (n && n[0]) g_cap.passesWanted = static_cast<unsigned>(std::strtoul(n, nullptr, 10));
        const char *fp = std::getenv("PS2X_PIPE_CAP_FILE");
        g_cap.file = (fp && fp[0]) ? fp : "tmp/pipecap.bin";
        g_cap.t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[pipe:cap] armed at +%.0fs, %u passes -> %s\n", g_cap.armAtSec, g_cap.passesWanted, g_cap.file.c_str());
    }
    void put(const void *p, size_t n) { if (g_cap.f) std::fwrite(p, 1, n, g_cap.f); }
    void putU32(uint32_t v) { put(&v, 4); }
    void event(uint32_t kind, uint32_t a, uint32_t b, const void *data, uint32_t size)
    {
        if (!g_cap.recording) return;
        putU32(kind); putU32(a); putU32(b); putU32(size);
        if (size) put(data, size);
        ++g_cap.events; g_cap.bytes += size + 16u;
    }
    uint64_t fnv(uint64_t h, const void *p, size_t n)
    {
        const uint8_t *b = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    }
    uint64_t g_hash = 1469598103934665603ull;
    unsigned long long g_hashPkts = 0ull, g_hashBytes = 0ull;
    void tap(uint32_t pathId, const uint8_t *data, uint32_t size)
    {
        g_hash = fnv(g_hash, &pathId, 4); g_hash = fnv(g_hash, &size, 4); g_hash = fnv(g_hash, data, size);
        ++g_hashPkts; g_hashBytes += size;
    }
}

namespace ps2pipe
{
    bool capEnabled() { capInit(); return g_cap.armAtSec >= 0.0 && !g_cap.done; }

    void capPassBegin(PS2Memory &mem, bool quiescent)
    {
        if (!capEnabled() || g_cap.recording) return;
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_cap.t0).count();
        if (now < g_cap.armAtSec) return;
        if (!quiescent)
        {
            if (++g_cap.notQuiescent <= 5u || (g_cap.notQuiescent % 500u) == 0u)
                std::fprintf(stderr, "[pipe:cap] pass not quiescent (#%u), waiting\n", g_cap.notQuiescent);
            return;
        }
        g_cap.f = std::fopen(g_cap.file.c_str(), "wb");
        if (!g_cap.f) { std::fprintf(stderr, "[pipe:cap] cannot open %s\n", g_cap.file.c_str()); g_cap.done = true; return; }
        put(kMagic, 4); putU32(kVersion);
        put(mem.getVU1Code(), PS2_VU1_CODE_SIZE);
        put(mem.getVU1Data(), PS2_VU1_DATA_SIZE);
        put(&mem.vif1_regs, sizeof(VIFRegisters));
        putU32(mem.isPath3Masked() ? 1u : 0u);
        g_cap.recording = true;
        std::fprintf(stderr, "[pipe:cap] START at +%.1fs (after %u non-quiescent passes)\n", now, g_cap.notQuiescent);
    }
    void capPassEnd()
    {
        if (!g_cap.recording) return;
        event(kEvPassEnd, 0u, 0u, nullptr, 0u);
        if (++g_cap.passesRecorded >= g_cap.passesWanted)
        {
            std::fclose(g_cap.f); g_cap.f = nullptr; g_cap.recording = false; g_cap.done = true;
            std::fprintf(stderr, "[pipe:cap] wrote %s: %u passes, %llu events, %.1f MB\n",
                         g_cap.file.c_str(), g_cap.passesRecorded, g_cap.events, double(g_cap.bytes) / 1048576.0);
        }
    }
    void capVif1(const uint8_t *data, uint32_t size) { event(kEvVif1, 0u, 0u, data, size); }
    void capGif3(const uint8_t *data, uint32_t size, bool drain) { event(kEvGif3, drain ? 1u : 0u, 0u, data, size); }
    void capDrain() { event(kEvDrain, 0u, 0u, nullptr, 0u); }
    void capVifReg(uint32_t addr, uint32_t value) { event(kEvVifReg, addr, value, nullptr, 0u); }
    void capVu1Mem(bool code, uint32_t offset, const void *data, uint32_t size) { event(kEvVu1Mem, offset, code ? 1u : 0u, data, size); }
    void capGsReg(uint8_t addr, uint64_t value) { event(kEvGsReg, addr, 0u, &value, 8u); }
    void capGsClear(uint32_t contextIndex, uint32_t rgba) { event(kEvGsClear, contextIndex, rgba, nullptr, 0u); }

    bool benchRun(PS2Runtime &rt)
    {
        const char *file = std::getenv("PS2X_PIPE_BENCH");
        if (!file || !file[0]) return false;
        const char *r = std::getenv("PS2X_PIPE_BENCH_REPS");
        const unsigned reps = (r && r[0]) ? static_cast<unsigned>(std::strtoul(r, nullptr, 10)) : 3u;
        std::FILE *f = std::fopen(file, "rb");
        if (!f) { std::fprintf(stderr, "[pipebench] cannot open %s\n", file); return true; }
        std::fseek(f, 0, SEEK_END); const long len = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> blob(static_cast<size_t>(len));
        if (std::fread(blob.data(), 1, blob.size(), f) != blob.size()) { std::fprintf(stderr, "[pipebench] short read\n"); std::fclose(f); return true; }
        std::fclose(f);
        if (blob.size() < 4 + 4 + PS2_VU1_CODE_SIZE + PS2_VU1_DATA_SIZE + sizeof(VIFRegisters) + 4 || std::memcmp(blob.data(), kMagic, 4) != 0)
        { std::fprintf(stderr, "[pipebench] bad capture\n"); return true; }
        size_t off = 8;
        const uint8_t *hdrCode = blob.data() + off; off += PS2_VU1_CODE_SIZE;
        const uint8_t *hdrData = blob.data() + off; off += PS2_VU1_DATA_SIZE;
        VIFRegisters hdrRegs; std::memcpy(&hdrRegs, blob.data() + off, sizeof(hdrRegs)); off += sizeof(hdrRegs);
        uint32_t masked = 0; std::memcpy(&masked, blob.data() + off, 4); off += 4;
        const size_t eventsStart = off;
        PS2Memory &mem = rt.memory();
        ps2xGifArbiterTap = &tap;
        std::fprintf(stderr, "[pipebench] %s: %.1f MB, reps=%u, path3Masked=%u\n", file, double(blob.size()) / 1048576.0, reps, masked);
        for (unsigned rep = 0; rep < reps; ++rep)
        {
            std::memcpy(mem.getVU1Code(), hdrCode, PS2_VU1_CODE_SIZE);
            std::memcpy(mem.getVU1Data(), hdrData, PS2_VU1_DATA_SIZE);
            mem.markVU1CodeModified();
            mem.vif1_regs = hdrRegs;
            ps2xVif1ResetDirectSpan();
            if (rep == 0) ps2xGsGifParseReset();
            g_hash = 1469598103934665603ull; g_hashPkts = 0ull; g_hashBytes = 0ull;
            unsigned long long nVif = 0, nGif = 0, nPass = 0;
            const auto t0 = std::chrono::steady_clock::now();
            size_t p = eventsStart;
            while (p + 16 <= blob.size())
            {
                uint32_t kind, a, b, size;
                std::memcpy(&kind, blob.data() + p, 4); std::memcpy(&a, blob.data() + p + 4, 4);
                std::memcpy(&b, blob.data() + p + 8, 4); std::memcpy(&size, blob.data() + p + 12, 4);
                p += 16;
                const uint8_t *data = blob.data() + p;
                if (p + size > blob.size()) break;
                p += size;
                switch (kind)
                {
                case kEvVif1: mem.processVIF1Data(data, size); ++nVif; break;
                case kEvGif3: mem.submitGifPacket(GifPathId::Path3, data, size, a != 0u); ++nGif; break;
                case kEvDrain: ps2xGifArbiterDrainCurrent(); break;
                case kEvVifReg: mem.writeIORegister(a, b); break;
                case kEvGsReg: if (size == 8u) { uint64_t v; std::memcpy(&v, data, 8u); rt.gs().writeRegister(static_cast<uint8_t>(a), v); } break;
                case kEvGsClear: rt.gs().clearFramebufferContext(a, b); break;
                case kEvVu1Mem:
                    if (b) { if (a + size <= PS2_VU1_CODE_SIZE) { std::memcpy(mem.getVU1Code() + a, data, size); mem.markVU1CodeModified(); } }
                    else { if (a + size <= PS2_VU1_DATA_SIZE) std::memcpy(mem.getVU1Data() + a, data, size); }
                    break;
                case kEvPassEnd: ++nPass; break;
                default: break;
                }
            }
            ps2gs::fence(); // stage 2: the consumer must have executed everything before we hash
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const uint64_t vuHash = fnv(1469598103934665603ull, mem.getVU1Data(), PS2_VU1_DATA_SIZE);
            std::fprintf(stderr, "[pipebench] rep %u: %.1f ms | passes=%llu vif1=%llu gif3=%llu | packets=%llu bytes=%.1fMB pktHash=%016llx vu1DataHash=%016llx\n",
                         rep, ms, nPass, nVif, nGif, g_hashPkts, double(g_hashBytes) / 1048576.0,
                         static_cast<unsigned long long>(g_hash), static_cast<unsigned long long>(vuHash));
        }
        ps2xGifArbiterTap = nullptr;
        std::fflush(nullptr);
        return true;
    }
}
