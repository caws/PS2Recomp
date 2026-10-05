// cont.317 — the GS pipeline ring (see include/runtime/ps2_gs_pipeline.h and
// docs/vu1-thread-design.md). PS2X_GS_PIPELINE: 0 direct, 1 ring consumed synchronously (stage 1),
// 2 ring consumed by the pipeline thread (stage 2). The capture oracle's taps live on the PRODUCER
// side, in the enqueue functions, so a capture records what the EE handed the pipeline and a
// replay re-enters through the ring like the game's own traffic.
//
// Stage 2 threading model: exactly ONE consumer executes the commands, in FIFO order, by calling
// the runtime's own entry points (processVIF1Data, submitGifPacket(Path3), writeIORegister for the
// VIF1 block, the arbiter drain, the VU1 memory store, the backend's OnDisplayFlip) with the
// thread-local consumer flag set. Everything those bodies touch -- the VIF1 registers, VU1 memory,
// the mask FIFO, the arbiter, the GS frontend's parse state, microVU's VU1 unit -- is thereby owned
// by the consumer; the EE reaches it only through fence(). Back-pressure: the producer blocks
// while more than kMaxPendingBytes are queued (a frame's VIF1 traffic is ~10 MB).
#include "runtime/ps2_gs_pipeline.h"
#include "runtime/ps2_pipe_capture.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "ps2_runtime.h"
#include "ThreadNaming.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" void ps2xGifArbiterDrainCurrent();
extern "C" void ps2xGsSetGifPath(uint32_t path); // gs_frontend.cpp

namespace
{
    enum class Kind : uint32_t { Vif1, Gif3, Drain, VifReg, Vu1Mem, Flip, GsReg, GsClear };
    struct Command
    {
        Kind kind;
        PS2Memory *mem;
        GSCpuBackend *backend;
        uint64_t a, b;
        std::vector<uint8_t> payload;
        GS *gs = nullptr; // cont.318: GsReg / GsClear target
    };
    const int s_mode = []
    {
        const char *e = std::getenv("PS2X_GS_PIPELINE");
        // ★ cont.319: DEFAULT 2 (was 1 since stage 2 landed). Gate that flipped it: on build 751 (the
        // four-wide shade, raster -15% instructions) four alternating 150 s recorded fights binned by
        // prims/frame put mode 2 ahead in every populated bin (20k prims/frame: 16.71 vs 13.74 fps,
        // n=6 each; 24k: 15.07 vs 13.93; 12k: 16.85 vs 14.51), 0 not-found / 0 swallowed-unwind in
        // all four, plus a 600 s with-save soak in mode 2. cont.318's hygiene items: applyGsDispEnv
        // needs no ring (it fires no flip and never touches the draw context -- progress.md cont.319).
        if (!e || !e[0]) return 2;
        const int v = std::atoi(e);
        return v < 0 ? 0 : v > 2 ? 2 : v;
    }();
    // cont.317 stage-2 finding (build 741): with 64 MB the EE ran ~9 frames ahead of the display
    // (the ring's 6 frames plus the GS queue's 3) and blocked 28% of wall in back-pressure with
    // one whole VIF1 transfer (~30 ms of consumer work) per wake-up, losing a vblank per block:
    // 1723 vs 2104 presents / 149 s against the synchronous ring. The frame is raster-bound (the GS
    // coordinator ~60 ms/frame), so the pipeline thread cannot buy frames until that drops; keep
    // the EE close to the display -- two frames of VIF1 traffic -- and release a command's bytes
    // when the consumer DEQUEUES it, so the producer wakes at command granularity, not after the
    // consumer's whole run of it.
    // ★ cont.318 PS2X_PIPE_CAP_MB (default 16): the back-pressure cap. WHY a knob: with the raster
    // balanced (fork row 79) mode 2 was still -20% presents and its raster +55% SLOWER PER PRIMITIVE,
    // not explained by thread count (6-thread pool) or placement (a CPU freed for the consumer) --
    // the remaining structural difference is that up to 16 MB of copied pipeline data (> the host's
    // 12 MB L3) sits between producer and consumer, so what the raster touches arrives cold.
    const size_t kMaxPendingBytes = []
    { const char *e = std::getenv("PS2X_PIPE_CAP_MB"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 16ul; if (v < 1ul) v = 1ul; if (v > 256ul) v = 256ul; return static_cast<size_t>(v) << 20; }();

    std::mutex g_m;
    std::condition_variable g_cvWork, g_cvIdle;
    std::deque<Command> g_queue;
    size_t g_pendingBytes = 0u;
    bool g_busy = false;          // the consumer is executing a command
    bool g_stop = false;
    std::thread g_thread;
    bool g_threadStarted = false;
    thread_local bool t_inConsumer = false;
    bool g_pumping = false;       // stage 1 re-entry guard
    unsigned long long g_fences = 0ull, g_fenceWaitNs = 0ull, g_backpressureNs = 0ull, g_commands = 0ull;
    unsigned long long g_fenceSite[8] = {}, g_fenceSiteNs[8] = {};
    std::chrono::steady_clock::time_point g_statT0{};

    // ★ cont.322 PS2X_PIPE_POOL (default 1): payload buffers are RECYCLED between commands instead of
    // being a fresh std::vector per command. WHY: a VIF1 command is a whole DMA transfer -- 5.5 MB on
    // average on the Helm's Deep fight (26 GB over a 240 s run, ~110 MB/s) -- so every push was a
    // multi-megabyte malloc (mmap or heap growth) page-faulted on the EE thread DURING the copy, and
    // every dequeue a free/munmap/trim on the pipeline thread. Symbolised profile of the EE thread
    // (build 772, light regime): libc memmove 11.5% + kernel 15% of its on-CPU time; the pipeline
    // thread carried the matching free side. PCSX2's MTGS copies into a FIXED ring buffer (GS/MTGS.cpp
    // RingBufferSize: the copy stays, no allocation per packet) -- same model here: the copy stays
    // (the guest may overwrite the source), the allocation goes. Buffers come back from the consumer
    // with their capacity, best-fit on take; the pool is bounded by PS2X_PIPE_POOL_MB (default 64) and
    // 32 entries. `=0` = the old per-command vector (A/B). Counters ride the [gs:pipeline] line.
    const bool s_pool = []
    { const char *e = std::getenv("PS2X_PIPE_POOL"); return !(e && e[0] == '0'); }();
    const size_t kPoolMaxBytes = []
    { const char *e = std::getenv("PS2X_PIPE_POOL_MB"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 64ul; if (v < 1ul) v = 1ul; if (v > 1024ul) v = 1024ul; return static_cast<size_t>(v) << 20; }();
    constexpr size_t kPoolMaxEntries = 32u;
    std::mutex g_poolM;
    std::vector<std::vector<uint8_t>> g_pool;
    size_t g_poolBytes = 0u;
    unsigned long long g_poolHits = 0ull, g_poolMisses = 0ull, g_poolDrops = 0ull;

    std::vector<uint8_t> takePayload(const uint8_t *data, size_t size)
    {
        std::vector<uint8_t> v;
        if (s_pool && size != 0u)
        {
            std::lock_guard<std::mutex> lk(g_poolM);
            size_t best = g_pool.size();
            for (size_t i = 0; i < g_pool.size(); ++i)
                if (g_pool[i].capacity() >= size && (best == g_pool.size() || g_pool[i].capacity() < g_pool[best].capacity()))
                    best = i;
            if (best != g_pool.size())
            {
                v = std::move(g_pool[best]);
                g_pool[best] = std::move(g_pool.back());
                g_pool.pop_back();
                g_poolBytes -= v.capacity();
                ++g_poolHits;
            }
            else
                ++g_poolMisses;
        }
        v.assign(data, data + size); // capacity retained: one copy, no zero-fill, no allocation on a hit
        return v;
    }
    void recyclePayload(std::vector<uint8_t> &&v)
    {
        if (!s_pool || v.capacity() < 4096u) return;
        std::lock_guard<std::mutex> lk(g_poolM);
        if (g_pool.size() >= kPoolMaxEntries || g_poolBytes + v.capacity() > kPoolMaxBytes)
        {
            // full: keep the LARGER buffers -- evict the smallest if this one is bigger, else drop it
            size_t small = g_pool.size();
            for (size_t i = 0; i < g_pool.size(); ++i)
                if (small == g_pool.size() || g_pool[i].capacity() < g_pool[small].capacity()) small = i;
            if (small == g_pool.size() || g_pool[small].capacity() >= v.capacity() ||
                g_poolBytes - g_pool[small].capacity() + v.capacity() > kPoolMaxBytes)
            { ++g_poolDrops; return; }
            g_poolBytes -= g_pool[small].capacity();
            g_pool[small] = std::move(g_pool.back());
            g_pool.pop_back();
        }
        v.clear();
        g_poolBytes += v.capacity();
        g_pool.push_back(std::move(v));
    }

    // ---- ★ rotk 2026-10-03 (row 253) stage 3: the GS parse on its own thread, PS2X_GS_SPLIT=1 (default OFF) ----
    // WHY: at fps 60 the pipeline thread is the pole (~32 ms of a 38.6 ms frame on the heavy fight; perf:
    // VU1 JIT ~46%, VIF1 ~19%, GIF/GS register handling + vertexKick + buildDrawBatch ~20%). The GS half
    // depends on nothing the VU half reads back (the VIF1/VU1 side never reads GS frontend state; the GS
    // side never touches VU1), so it can run one FIFO behind it. PCSX2 has the same cut: the MTVU thread
    // runs VU1 and hands each XGKICK packet on (FinishGSPacketMTVU -> gsPackQueue) to the MTGS thread,
    // which owns the GS. Order: the pipeline thread is the only producer, in program order -- the
    // arbiter's sorted drain (packet + path, then the end-of-drain) and the ring's GS commands
    // (GsReg / GsClear / Flip), which used to execute on it. Fences: a guest READ of VU1 memory waits
    // for the pipeline thread only (the per-frame `vu1read` fence); every other site (GS reset,
    // readback, local->host, shutdown, the capture oracle) waits for both. The parse thread counts as
    // a consumer (its fences return at once: it runs inside the stream).
    enum class PKind : uint32_t { Packet, End, GsReg, GsClear, Flip };
    struct PItem
    {
        PKind kind;
        uint32_t path;
        std::vector<uint8_t> data;
        GS *gs;
        GSCpuBackend *backend;
        uint64_t a, b;
    };
    const bool s_split = []
    { const char *e = std::getenv("PS2X_GS_SPLIT"); return e && e[0] && e[0] != '0'; }();
    // PS2X_GS_SPLIT_CAP_MB (default 8): back-pressure between the two threads (the pipeline thread waits).
    const size_t kParseMaxPendingBytes = []
    { const char *e = std::getenv("PS2X_GS_SPLIT_CAP_MB"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 8ul; if (v < 1ul) v = 1ul; if (v > 256ul) v = 256ul; return static_cast<size_t>(v) << 20; }();
    GS *g_parseGs = nullptr;
    std::mutex g_pm;
    std::condition_variable g_pcvWork, g_pcvIdle;
    std::deque<PItem> g_pq;
    size_t g_pPending = 0u;
    bool g_pBusy = false, g_pStop = false, g_pStarted = false;
    std::thread g_pThread;
    thread_local bool t_isPipelineThread = false;
    unsigned long long g_pItems = 0ull, g_pBusyNs = 0ull, g_pBackpressureNs = 0ull, g_pBackpressureHits = 0ull,
                       g_pMaxPending = 0ull, g_pFences = 0ull, g_pFenceNs = 0ull, g_pBatches = 0ull;

    bool splitOn() { return s_mode == 2 && s_split && g_parseGs != nullptr; }
    void pinConsumerThread();

    void parseExecute(PItem &it)
    {
        switch (it.kind)
        {
        case PKind::Packet:
            ps2xGsSetGifPath(it.path);
            g_parseGs->processGIFPacket(it.data.data(), static_cast<uint32_t>(it.data.size()));
            break;
        case PKind::End: g_parseGs->endPacket(); break;
        case PKind::GsReg: if (it.gs) it.gs->writeRegister(static_cast<uint8_t>(it.a), it.b); break;
        case PKind::GsClear: if (it.gs) it.gs->clearFramebufferContext(static_cast<uint32_t>(it.a), static_cast<uint32_t>(it.b)); break;
        case PKind::Flip: if (it.backend) it.backend->OnDisplayFlip(it.a, it.b); break;
        }
    }

    void parseLoop()
    {
        ThreadNaming::SetCurrentThreadName("GsParse");
        pinConsumerThread();
        t_inConsumer = true;
        std::deque<PItem> batch;
        std::unique_lock<std::mutex> lk(g_pm);
        for (;;)
        {
            g_pcvWork.wait(lk, [] { return g_pStop || !g_pq.empty(); });
            if (g_pStop && g_pq.empty())
                break;
            batch.swap(g_pq); // everything published so far, one lock
            g_pBusy = true;
            g_pPending = 0u;
            g_pcvIdle.notify_all(); // a producer blocked on the cap re-checks
            lk.unlock();
            const auto t0 = std::chrono::steady_clock::now();
            const size_t n = batch.size();
            for (PItem &it : batch)
                parseExecute(it);
            batch.clear(); // frees the packets outside the lock
            const unsigned long long ns = static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            lk.lock();
            g_pBusyNs += ns;
            g_pItems += n;
            ++g_pBatches;
            g_pBusy = false;
            if (g_pq.empty())
                g_pcvIdle.notify_all();
        }
    }

    // The pipeline thread (the only producer) collects items here and publishes them in batches: a wake-up
    // per XGKICK packet (~230k/s on the fight) cost more than it moved. Published at 64 items / 256 KB and
    // ALWAYS before the pipeline thread finishes a ring command (consumerLoop), so a fence that has seen the
    // pipeline thread idle sees every item in the parse FIFO.
    std::vector<PItem> g_pLocal;
    size_t g_pLocalBytes = 0u;
    void pflush()
    {
        if (g_pLocal.empty())
            return;
        if (!g_pStarted)
        {
            g_pStarted = true;
            g_pThread = std::thread(parseLoop);
        }
        const size_t bytes = g_pLocalBytes;
        std::unique_lock<std::mutex> lk(g_pm);
        if (g_pPending != 0u && g_pPending + bytes > kParseMaxPendingBytes)
        {
            ++g_pBackpressureHits;
            const auto t0 = std::chrono::steady_clock::now();
            g_pcvIdle.wait(lk, [&] { return g_pPending == 0u || g_pPending + bytes <= kParseMaxPendingBytes; });
            g_pBackpressureNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        }
        g_pPending += bytes;
        if (g_pPending > g_pMaxPending) g_pMaxPending = g_pPending;
        for (PItem &it : g_pLocal)
            g_pq.push_back(std::move(it));
        lk.unlock();
        g_pLocal.clear();
        g_pLocalBytes = 0u;
        g_pcvWork.notify_one();
    }
    void ppush(PItem &&it)
    {
        g_pLocalBytes += it.data.size();
        g_pLocal.push_back(std::move(it));
        if (g_pLocal.size() >= 64u || g_pLocalBytes >= (256u << 10))
            pflush();
    }

    // The EE waits for the parse thread to drain (called after the pipeline thread is idle, so nothing new arrives).
    void parseFence()
    {
        std::unique_lock<std::mutex> lk(g_pm); // (g_pStarted is the pipeline thread's; an unstarted queue is empty)
        if (g_pq.empty() && !g_pBusy) return;
        ++g_pFences;
        const auto t0 = std::chrono::steady_clock::now();
        g_pcvIdle.wait(lk, [] { return g_pq.empty() && !g_pBusy; });
        g_pFenceNs += static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }

    void execute(Command &c)
    {
        if (splitOn()) // stage 3: the GS commands follow the packets to the parse thread, in order
        {
            switch (c.kind)
            {
            case Kind::GsReg: ppush(PItem{PKind::GsReg, 0u, {}, c.gs, nullptr, c.a, c.b}); return;
            case Kind::GsClear: ppush(PItem{PKind::GsClear, 0u, {}, c.gs, nullptr, c.a, c.b}); return;
            case Kind::Flip: ppush(PItem{PKind::Flip, 0u, {}, nullptr, c.backend, c.a, c.b}); return;
            default: break;
            }
        }
        switch (c.kind)
        {
        case Kind::Vif1: c.mem->processVIF1Data(c.payload.data(), static_cast<uint32_t>(c.payload.size())); break;
        case Kind::Gif3: c.mem->submitGifPacket(GifPathId::Path3, c.payload.data(), static_cast<uint32_t>(c.payload.size()), c.a != 0u); break;
        case Kind::Drain: ps2xGifArbiterDrainCurrent(); break;
        case Kind::VifReg: c.mem->writeIORegister(static_cast<uint32_t>(c.a), static_cast<uint32_t>(c.b)); break;
        case Kind::GsReg: if (c.gs) c.gs->writeRegister(static_cast<uint8_t>(c.a), c.b); break;
        case Kind::GsClear: if (c.gs) c.gs->clearFramebufferContext(static_cast<uint32_t>(c.a), static_cast<uint32_t>(c.b)); break;
        case Kind::Vu1Mem:
            if (c.b) { if (c.a + c.payload.size() <= PS2_VU1_CODE_SIZE) { std::memcpy(c.mem->getVU1Code() + c.a, c.payload.data(), c.payload.size()); c.mem->markVU1CodeModified(); } }
            else { if (c.a + c.payload.size() <= PS2_VU1_DATA_SIZE) std::memcpy(c.mem->getVU1Data() + c.a, c.payload.data(), c.payload.size()); }
            break;
        case Kind::Flip: if (c.backend) c.backend->OnDisplayFlip(c.a, c.b); break;
        }
    }

    // ★ cont.322 PS2X_PIPE_CPUS: the consumer thread's CPU list. WHY: the thread is created from the
    // EE thread (ensureThread() runs on the first push), so it INHERITED the EE's affinity mask -- and
    // the auto plan pins the EE to ONE logical CPU (`[affinity] plan: EE=0 GS=4,1,5,2,6,3,7`), so the
    // EE and the pipeline had been time-slicing a single logical CPU since stage 2 landed: on the
    // recorded fight (/proc schedstat, build 772) each spent 32-46% of the wall RUNNABLE-BUT-WAITING
    // for CPU 0, the coordinator waited for items up to 30% of the light-scene frames, and 2-vblank
    // frames were 43-76%; with the mask opened from outside (taskset, A/B/A/B in one run, matched
    // 9.6k prims/frame) the run-delay fell to 4-8%, 2-vblank frames rose to 85-93% and presents went
    // 208 -> 260 per 10 s. Selector: `gs` (DEFAULT) = the raster pool's list (the EE keeps its CPU to
    // itself), `ee` = the EE's list (the old inherited behaviour, for A/B), `all` = every online CPU
    // (no pin), or an explicit list "1-3,5". PCSX2 reference: VMManager::SetEmuThreadAffinities gives
    // the EE, the VU (MTVU) thread and the GS thread DISTINCT processors (s_processor_list[0..2]) and
    // the software-renderer threads the rest; with EnableThreadPinning off it clears every mask
    // (SetAffinity(0)). Neither mode ever shares one logical CPU between the EE and the VU thread.
    void pinConsumerThread()
    {
        const char *e = std::getenv("PS2X_PIPE_CPUS");
        std::string sel = (e && e[0]) ? e : "gs";
        std::string list;
        const char *how = sel.c_str();
        auto allCpus = []
        {
            const unsigned n = std::thread::hardware_concurrency();
            return "0-" + std::to_string(n > 0u ? n - 1u : 0u);
        };
        if (sel == "gs" || sel == "ee")
        {
            const char *env = std::getenv(sel == "gs" ? "PS2X_GS_CPUS" : "PS2X_EE_CPUS");
            if (env && env[0])
                list = env;
            else if (ThreadNaming::AutoCpuPinEnabled() && ThreadNaming::AutoCpuPlan().valid)
                list = sel == "gs" ? ThreadNaming::AutoCpuPlan().gs : ThreadNaming::AutoCpuPlan().ee;
            else
                list = allCpus(); // nothing else is pinned: undo the inherited mask
        }
        else if (sel == "all")
            list = allCpus();
        else
            list = sel;
        const bool ok = ThreadNaming::PinCurrentThreadToList(list.c_str(), "PS2X_PIPE_CPUS");
        std::fprintf(stderr, "[affinity] pipeline consumer: PS2X_PIPE_CPUS=%s -> %s%s\n", how, list.c_str(), ok ? "" : " (not pinned)");
    }

    void consumerLoop()
    {
        ThreadNaming::SetCurrentThreadName("GsPipeline");
        pinConsumerThread();
        t_inConsumer = true;
        t_isPipelineThread = true;
        std::unique_lock<std::mutex> lk(g_m);
        for (;;)
        {
            g_cvWork.wait(lk, [] { return g_stop || !g_queue.empty(); });
            if (g_stop && g_queue.empty())
                break;
            Command c = std::move(g_queue.front());
            g_queue.pop_front();
            g_busy = true;
            g_pendingBytes -= c.payload.size(); // released at dequeue: the producer's accounting, not the data's lifetime
            g_cvIdle.notify_all();              // a producer blocked on the cap re-checks its predicate
            lk.unlock();
            execute(c);
            pflush(); // stage 3: publish this command's GS items before the command counts as done (fences)
            recyclePayload(std::move(c.payload));
            lk.lock();
            g_busy = false;
            ++g_commands;
            if (g_queue.empty())
                g_cvIdle.notify_all();
        }
    }

    void ensureThread()
    {
        if (g_threadStarted) return;
        g_threadStarted = true;
        g_thread = std::thread(consumerLoop);
    }

    unsigned long long g_backpressureHits = 0ull, g_pushes = 0ull, g_maxPending = 0ull, g_maxDepth = 0ull;
    std::chrono::steady_clock::time_point g_pushStatT0{};
    void push(Command &&c)
    {
        if (s_mode == 2)
        {
            ensureThread();
            const size_t bytes = c.payload.size();
            std::unique_lock<std::mutex> lk(g_m);
            ++g_pushes;
            // cont.318: an EMPTY queue always admits the push, whatever its size -- with a cap below the
            // largest single command (PS2X_PIPE_CAP_MB=2 hung at boot) the predicate could never hold.
            if (g_pendingBytes != 0u && g_pendingBytes + bytes > kMaxPendingBytes)
            {
                ++g_backpressureHits;
                const auto t0 = std::chrono::steady_clock::now();
                g_cvIdle.wait(lk, [&] { return g_pendingBytes == 0u || g_pendingBytes + bytes <= kMaxPendingBytes; });
                g_backpressureNs += static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            }
            g_pendingBytes += bytes;
            if (g_pendingBytes > g_maxPending) g_maxPending = g_pendingBytes;
            g_queue.push_back(std::move(c));
            if (g_queue.size() > g_maxDepth) g_maxDepth = g_queue.size();
            // every 10 s from the producer: the ring's shape (the stage-2 diagnosis line)
            const auto now = std::chrono::steady_clock::now();
            if (g_pushStatT0.time_since_epoch().count() == 0) g_pushStatT0 = now;
            if (now - g_pushStatT0 >= std::chrono::seconds(10))
            {
                g_pushStatT0 = now;
                std::fprintf(stderr, "[gs:pipeline] pushes=%llu commands=%llu depth=%zu (max %llu) pending=%.1fMB (max %.1fMB) | backpressure hits=%llu %.0fms | fences=%llu %.0fms | pool hits=%llu misses=%llu drops=%llu held=%.1fMB | parse items=%llu batches=%llu busy=%.0fms bp hits=%llu %.0fms maxPending=%.1fMB fences=%llu %.0fms\n",
                             g_pushes, g_commands, g_queue.size(), g_maxDepth, double(g_pendingBytes) / 1048576.0, double(g_maxPending) / 1048576.0,
                             g_backpressureHits, double(g_backpressureNs) * 1e-6, g_fences, double(g_fenceWaitNs) * 1e-6,
                             g_poolHits, g_poolMisses, g_poolDrops, double(g_poolBytes) / 1048576.0,
                             g_pItems, g_pBatches, double(g_pBusyNs) * 1e-6, g_pBackpressureHits, double(g_pBackpressureNs) * 1e-6,
                             double(g_pMaxPending) / 1048576.0, g_pFences, double(g_pFenceNs) * 1e-6);
            }
            lk.unlock();
            g_cvWork.notify_one();
        }
        else
        {
            g_queue.push_back(std::move(c));
        }
    }
}

namespace ps2gs
{
    int mode() { return s_mode; }
    void setParseGs(GS *gs) { g_parseGs = gs; }
    bool splitSink() { return t_isPipelineThread && splitOn(); }
    void splitPacket(uint32_t path, std::vector<uint8_t> &&data)
    {
        PItem it{PKind::Packet, path, std::move(data), nullptr, nullptr, 0u, 0u};
        ppush(std::move(it));
    }
    void splitEnd() { ppush(PItem{PKind::End, 0u, {}, nullptr, nullptr, 0u, 0u}); }
    bool threaded() { return s_mode == 2; }
    bool producerSide() { return s_mode != 0 && !t_inConsumer; }
    bool consumerSide() { return t_inConsumer; }

    void enqueueVif1(PS2Memory *mem, const uint8_t *data, uint32_t size)
    {
        ps2pipe::capVif1(data, size);
        push(Command{Kind::Vif1, mem, nullptr, 0u, 0u, takePayload(data, size)});
    }
    void enqueueVif1Owned(PS2Memory *mem, std::vector<uint8_t> &&data)
    {
        ps2pipe::capVif1(data.data(), static_cast<uint32_t>(data.size()));
        push(Command{Kind::Vif1, mem, nullptr, 0u, 0u, std::move(data)});
    }
    std::vector<uint8_t> takeBuffer()
    {
        // the LARGEST pooled buffer: a chain's size is unknown until it is walked, and the big ones are the
        // ones whose growth cost (reallocation + page faults on fresh memory) this removes
        std::vector<uint8_t> v;
        if (!s_pool) return v;
        std::lock_guard<std::mutex> lk(g_poolM);
        size_t best = g_pool.size();
        for (size_t i = 0; i < g_pool.size(); ++i)
            if (best == g_pool.size() || g_pool[i].capacity() > g_pool[best].capacity())
                best = i;
        if (best == g_pool.size()) { ++g_poolMisses; return v; }
        v = std::move(g_pool[best]);
        g_pool[best] = std::move(g_pool.back());
        g_pool.pop_back();
        g_poolBytes -= v.capacity();
        ++g_poolHits;
        v.clear();
        return v;
    }
    void enqueueGif3(PS2Memory *mem, const uint8_t *data, uint32_t size, bool drainImmediately)
    {
        ps2pipe::capGif3(data, size, drainImmediately);
        push(Command{Kind::Gif3, mem, nullptr, drainImmediately ? 1u : 0u, 0u, takePayload(data, size)});
    }
    void enqueueDrain(PS2Memory *mem)
    {
        ps2pipe::capDrain();
        push(Command{Kind::Drain, mem, nullptr, 0u, 0u, {}});
    }
    void enqueueGsReg(GS *gs, uint8_t addr, uint64_t value)
    {
        ps2pipe::capGsReg(addr, value);
        Command c{Kind::GsReg, nullptr, nullptr, addr, value, {}};
        c.gs = gs;
        push(std::move(c));
    }
    void enqueueGsClear(GS *gs, uint32_t contextIndex, uint32_t rgba)
    {
        ps2pipe::capGsClear(contextIndex, rgba);
        Command c{Kind::GsClear, nullptr, nullptr, contextIndex, rgba, {}};
        c.gs = gs;
        push(std::move(c));
    }
    void enqueueVifReg(PS2Memory *mem, uint32_t addr, uint32_t value)
    {
        ps2pipe::capVifReg(addr, value);
        push(Command{Kind::VifReg, mem, nullptr, addr, value, {}});
    }
    void enqueueVu1Mem(PS2Memory *mem, bool code, uint32_t offset, const void *data, uint32_t size)
    {
        ps2pipe::capVu1Mem(code, offset, data, size);
        const uint8_t *p = static_cast<const uint8_t *>(data);
        push(Command{Kind::Vu1Mem, mem, nullptr, offset, code ? 1u : 0u, takePayload(p, size)});
    }
    void enqueueFlip(GSCpuBackend *backend, uint64_t fb1, uint64_t fb2)
    {
        push(Command{Kind::Flip, nullptr, backend, fb1, fb2, {}});
    }
    void pump()
    {
        if (s_mode == 2)
            return; // the consumer thread was notified by push()
        if (g_pumping) return; // a consumer-side re-entry (e.g. an unmask flush) must not recurse
        g_pumping = true;
        t_inConsumer = true;
        while (!g_queue.empty())
        {
            Command c = std::move(g_queue.front());
            g_queue.pop_front();
            execute(c);
            recyclePayload(std::move(c.payload));
        }
        t_inConsumer = false;
        g_pumping = false;
    }
    void fence(int site)
    {
        if (s_mode != 2) { pump(); return; }
        if (t_inConsumer) return; // the consumer itself never waits on the ring
        // MEASUREMENT ONLY (rotk 2026-10-03): PS2X_GS_FENCE_VU1READ=0 skips the VU1-read fence, so an EE read of
        // VU1 memory races the consumer. Sizes what that fence costs; not a fix (default 1 = fence).
        static const bool s_vu1ReadFence = [] { const char *e = std::getenv("PS2X_GS_FENCE_VU1READ"); return !(e && e[0] == '0'); }();
        if (site == 1 && !s_vu1ReadFence) return;
        // stage 3: every site but the VU1 read also needs the GS parse drained (after the pipeline thread,
        // which is its only producer, has gone idle).
        struct ParseFenceAfter { int site; ~ParseFenceAfter() { if (site != 1 && splitOn()) parseFence(); } } parseFenceAfter{site};
        std::unique_lock<std::mutex> lk(g_m);
        if (g_queue.empty() && !g_busy) return;
        ++g_fences;
        const auto t0 = std::chrono::steady_clock::now();
        g_cvIdle.wait(lk, [] { return g_queue.empty() && !g_busy; });
        const unsigned long long ns = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        g_fenceWaitNs += ns;
        const int s = (site >= 0 && site < 8) ? site : 0;
        ++g_fenceSite[s]; g_fenceSiteNs[s] += ns;
        // every 10 s: who fences, how often, how long (the stage-2 diagnosis line)
        if (g_statT0.time_since_epoch().count() == 0) g_statT0 = t0;
        if (t0 - g_statT0 >= std::chrono::seconds(10))
        {
            g_statT0 = t0;
            std::fprintf(stderr, "[gs:pipeline] commands=%llu fences=%llu wait=%.0fms backpressure=%.0fms | sites: vu1read n=%llu %.0fms | gsreset n=%llu %.0fms | gsdbg n=%llu %.0fms | l2h n=%llu %.0fms | other n=%llu %.0fms\n",
                         g_commands, g_fences, double(g_fenceWaitNs) * 1e-6, double(g_backpressureNs) * 1e-6,
                         g_fenceSite[1], double(g_fenceSiteNs[1]) * 1e-6, g_fenceSite[2], double(g_fenceSiteNs[2]) * 1e-6,
                         g_fenceSite[3], double(g_fenceSiteNs[3]) * 1e-6, g_fenceSite[4], double(g_fenceSiteNs[4]) * 1e-6,
                         g_fenceSite[0], double(g_fenceSiteNs[0]) * 1e-6);
        }
    }
    void shutdown()
    {
        if (s_mode != 2 || !g_threadStarted) return;
        fence();
        {
            std::lock_guard<std::mutex> lk(g_m);
            g_stop = true;
        }
        g_cvWork.notify_all();
        if (g_thread.joinable()) g_thread.join();
        g_threadStarted = false;
        if (g_pStarted)
        {
            {
                std::lock_guard<std::mutex> plk(g_pm);
                g_pStop = true;
            }
            g_pcvWork.notify_all();
            if (g_pThread.joinable()) g_pThread.join();
            g_pStarted = false;
            std::fprintf(stderr, "[gs:pipeline] parse items=%llu busy=%.1fms backpressure=%.1fms fences=%llu %.1fms\n",
                         g_pItems, double(g_pBusyNs) * 1e-6, double(g_pBackpressureNs) * 1e-6, g_pFences, double(g_pFenceNs) * 1e-6);
        }
        std::fprintf(stderr, "[gs:pipeline] commands=%llu fences=%llu fenceWait=%.1fms backpressure=%.1fms\n",
                     g_commands, g_fences, double(g_fenceWaitNs) * 1e-6, double(g_backpressureNs) * 1e-6);
    }
}
