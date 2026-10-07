#include "runtime/ee_scheduler.h"

#include "ps2_log.h"
#include "ps2_runtime_macros.h"
#include "runtime/ps2_dbcman_hle.h"

#include <algorithm>
#include <unordered_map>
#include <array>
#include <atomic>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

// ---- Host-pump run-to-completion scope (PS2X_HOSTPUMP_RTC, default ON; =0 restores the
// unwind-on-checkpoint behaviour). Game-side HLE hooks drive guest call chains to completion
// with manual dispatch pumps (rfn = lookupFunction(pc); rfn(...)). The checkpoint latch
// (m_checkpointPending, cleared only by the scheduler run loop) makes every pumped dispatch
// execute a SINGLE basic block once a deadline has passed, so bounded pumps abort guest chains
// MID-FLIGHT, leaving guest state (parse cursors, in-flight records, rings) half-updated.
// Neither real EE hardware nor PCSX2 ever abandons a call chain: PCSX2 runs event tests BETWEEN
// recompiled blocks (pcsx2/R5900.cpp cpuEventTest) and each block/chain runs to completion —
// interrupts are delivered between blocks, never by unwinding architectural state mid-chain.
// While the scope is open, checkpointDue() reports false (cycles still accrue, so timers and
// vblank deadlines latch and fire the moment the scope closes; stop-requests still preempt).
// External linkage on purpose: game overrides declare `extern void ps2xBeginHostPump();`.
static std::atomic<uint32_t> s_hostPumpDepth{0};
static const bool s_hostPumpRtc = []
{ const char *e = std::getenv("PS2X_HOSTPUMP_RTC"); return !(e && e[0] == '0'); }();
void ps2xBeginHostPump() noexcept { s_hostPumpDepth.fetch_add(1u, std::memory_order_relaxed); }
void ps2xEndHostPump() noexcept { s_hostPumpDepth.fetch_sub(1u, std::memory_order_relaxed); }

// ---- Wall-clock vblank floor (PS2X_VBLANK_WALLCLOCK, default ON; =0 restores pure
// cycle-gating). Scheduled deadlines require BOTH deadlineCycle <= m_eeCycle AND
// hostDeadline <= now; guest cycles accrue only kGuestDispatchCycles per backward edge, so a
// busy guest under-credits the emulated clock and vblank collapses (measured ~1.5 ticks/s vs
// 50). PCSX2 reference: pcsx2/Counters.cpp fires VSyncStart/hwIntcIrq(INTC_VBLANK_S) from
// cycle counters credited per executed block — effective vblank pacing tracks real time. The
// floor advances the EE clock to a deadline whose HOST time has arrived (the same catch-up
// waitForEvent() already performs when all threads are idle).
static const bool s_vblankWallclock = []
{ const char *e = std::getenv("PS2X_VBLANK_WALLCLOCK"); return !(e && e[0] == '0'); }();

// ---- cont.322e PS2X_VIRTUAL_TIME (default OFF): DETERMINISTIC REPLAY. The scheduler's deadlines carry a
// cycle (m_eeCycle, credited per guest backward edge) and a host time; normally both must have
// arrived, the host floor paces the vblank ladder at real time, and the idle wait sleeps until the
// host deadline. In virtual time the host clock is never consulted: a deadline is due when its
// cycle has arrived, the idle wait jumps the cycle clock to the next deadline instead of sleeping,
// and COP0 Count is the cycle clock. Every vblank then lands at the same guest instant on every
// run, the pad script (indexed by pad reads = vblank callbacks) lands on the same game frames, and
// the recorded fight replays the SAME SCENE regardless of host speed -- which is what an A/B needs
// (cont.322: every live pair diverged into different fights; prim-matched bins were the
// substitute). The game runs as fast as the host allows at a constant one vblank per frame; the
// metric is wall seconds per fixed guest-frame interval. NOT for play (the clock is not real time).
// PCSX2 reference: pcsx2/Counters.cpp -- counters and vsync are credited from EE cycles
// (cpuRegs.cycle), the frame limiter only THROTTLES; with the limiter off PCSX2 is deterministic by
// construction. Ours is host-timed because the EE runs untimed; this mode restores cycle gating.
static const bool s_virtualTime = []
{ const char *e = std::getenv("PS2X_VIRTUAL_TIME"); return e && e[0] && e[0] != '0'; }();
bool ps2xVirtualTimeEnabled() noexcept { return s_virtualTime; } // for the HLE stubs that expose a host clock (sceCdReadClock)

// ---- Invocation-lifecycle diagnostic (PS2X_INVOKE_LOG, default OFF). Traces guest-invocation
// pushes/dispatches/completions and the run-loop's skip paths, to diagnose invocation-stack
// pileups (the "EE invocation stack space exhausted" abort). Read-only.
static const bool s_invokeLog = []
{ const char *e = std::getenv("PS2X_INVOKE_LOG"); return e && e[0] && e[0] != '0'; }();
static int s_invokeLogBudget = 4000;


// ---- cont.258 PS2X_EE_GUESTPROF (default OFF, read-only): direct ns for GUEST CODE.
// cont.250 concluded "the transliterated guest code is NOT a performance factor" from PROFILE LEAF
// SHARE (the hot guest functions showed ~16% inclusive but ~zero leaf). This project has twice been
// wrong pricing a subsystem off share -- cont.234 put VU0 at "~1%" and the 2x-slow knob later said
// 28%, and cont.252 found the EE-thread sampling itself biased -- so guest code is the last EE item
// never measured directly. This wraps the recompiled-function call in the EE dispatch loop.
// NESTING: every DMA drain path (ps2_memory.cpp:1706 CHCR write, ps2_runtime.cpp:2468
// kickGifDmaChainFromMMIO, the Kernel/Stubs HLE) is reached FROM guest code, and VU0 runs on COP2
// instructions inside it, so guest-inclusive CONTAINS vif1-inclusive (which contains VU1 and GIF)
// and VU0. Derive: guest-exclusive = guestIncl - [ee:prof]vif1 - [vu0:microvu]ns.
// Depth-guarded: an override may pump the dispatch loop re-entrantly, and only the outermost entry
// may accumulate or the nested time is counted twice.
static const bool g_guestProf = []
{ const char *e = std::getenv("PS2X_EE_GUESTPROF"); return e && e[0] && e[0] != '0'; }();
static const int g_guestProfEvery = []
{ const char *e = std::getenv("PS2X_EE_GUESTPROF_EVERY"); const int v = e && e[0] ? std::atoi(e) : 10; return v > 0 ? v : 10; }();
static unsigned long long g_guestProfNs = 0ull, g_guestProfCalls = 0ull, g_guestProfNested = 0ull;
#ifdef PS2X_GPR_COUNT
// cont.259: definition for the compile-time GPR-write counter in ps2_runtime_macros.h.
unsigned long long g_ps2GprWrites = 0ull;
#endif
static thread_local int g_guestProfDepth = 0;

// Wall-clock gated, never call-gated (cont.251 lesson: a timeout(1) kill skips any end-of-run report).
static void guestProfReport()
{
    static std::chrono::steady_clock::time_point s_t0{}, s_last{};
    const auto now = std::chrono::steady_clock::now();
    if (s_t0.time_since_epoch().count() == 0)
    {
        s_t0 = s_last = now;
        std::fprintf(stderr, "[ee:guest] active (PS2X_EE_GUESTPROF=1, every %ds)\n", g_guestProfEvery);
        return;
    }
    if (now - s_last < std::chrono::seconds(g_guestProfEvery))
        return;
    s_last = now;
    const double wall = std::chrono::duration_cast<std::chrono::nanoseconds>(now - s_t0).count() / 1e9;
    const double g = double(g_guestProfNs) / 1e9;
    std::fprintf(stderr,
                 "[ee:guest] wall=%.1fs guestIncl=%.2fs (%.1f%% of wall) calls=%llu nested=%llu"
                 " | per-call=%.2fus | clock-overhead<=%.2fs\n",
                 wall, g, 100.0 * g / wall, g_guestProfCalls, g_guestProfNested,
                 g_guestProfCalls ? g * 1e6 / double(g_guestProfCalls) : 0.0,
                 double(g_guestProfCalls) * 45.0 / 1e9);
#ifdef PS2X_GPR_COUNT
    std::fprintf(stderr, "[ee:guest] gprWr=%llu (%.1f M/s)  <-- PS2X_GPR_COUNT build, NOT a timing build\n",
                 g_ps2GprWrites, double(g_ps2GprWrites) / wall / 1e6);
#endif
}

struct GuestProfScope
{
    std::chrono::steady_clock::time_point t0;
    bool top;
    GuestProfScope() : top(g_guestProfDepth++ == 0)
    {
        if (g_guestProf && top)
            t0 = std::chrono::steady_clock::now();
    }
    ~GuestProfScope()
    {
        --g_guestProfDepth;
        if (!g_guestProf)
            return;
        if (!top)
        {
            ++g_guestProfNested;
            return;
        }
        g_guestProfNs += static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
        ++g_guestProfCalls;
        guestProfReport();
    }
};

// ---- Scheduler pacing instrument (PS2X_SCHED_PACE, default OFF; 1 = counters + periodic
// [sched:pace] line, 2 = also a detail line per long wait). cont.251: the gdb sampling profiles
// (tmp/prof250_gdb.log 9/33, tmp/prof251_gdb.log 18/77) put the EE executor thread inside
// waitForEvent()'s timed wait for 23-27% of its samples, but a sampling share cannot say WHICH
// deadline it slept to, how long it actually slept, or why no guest thread was runnable. This
// measures exactly that. READ-ONLY: it only reads state waitForEvent() already computes, plus
// steady_clock, and prints. waitForEvent() runs solely on the EE executor thread, so plain
// scalars need no atomics.
static const int s_schedPace = []
{ const char *e = std::getenv("PS2X_SCHED_PACE"); return e && e[0] ? std::atoi(e) : 0; }();

namespace
{
    struct SchedPaceStats
    {
        unsigned long long calls = 0;      // waitForEvent() entries
        unsigned long long early = 0;      // returned at once: an event was already queued / stop
        unsigned long long stall = 0;      // no deadline at all -> untimed wait (a real stall)
        unsigned long long timed = 0;      // deadline-bounded wait_until
        unsigned long long signaled = 0;   // woken by an event before the deadline
        unsigned long long timedout = 0;   // deadline reached -> cycle catch-up + checkpoint
        unsigned long long overdue = 0;    // deadline already past on entry (no real sleep)
        unsigned long long srcVBlankStart = 0;
        unsigned long long srcVBlankEnd = 0;
        unsigned long long srcAlarm = 0;
        unsigned long long srcOther = 0;   // Stop / Dmac / ExternalWake sitting in m_deadlines
        unsigned long long srcTimer = 0;   // the EE timer deadline beat every scheduled event
        unsigned long long reqNs = 0;      // Sigma requested (hostDeadline - now), overdue counted 0
        unsigned long long sleptNs = 0;    // Sigma actually measured around the wait
        unsigned long long stallNs = 0;    // Sigma of the untimed-wait path
        unsigned long long catchupCycles = 0;  // Sigma cycles credited by the timeout catch-up
        std::array<unsigned long long, 7> bucket{}; // <1us,<10us,<100us,<1ms,<5ms,<20ms,>=20ms
        // Why was nothing runnable? Wait-reason census of every non-Dormant guest thread, taken
        // only on waits that requested >= kWhyThresholdNs so it cannot perturb the common path.
        std::array<unsigned long long, 7> whyReason{}; // indexed by EeWaitReason
        unsigned long long whyRunnableIsh = 0;  // threads Ready/Running at that moment (should be 0)
        unsigned long long whySuspended = 0;
        unsigned long long whyWaits = 0;        // how many waits contributed a census
        std::chrono::steady_clock::time_point t0{};
        std::chrono::steady_clock::time_point lastReport{};
        uint64_t tick0 = 0;
        bool started = false;
    };
    SchedPaceStats g_pace;
    constexpr unsigned long long kWhyThresholdNs = 500000ull; // 0.5 ms
    // Report on a WALL-CLOCK interval, not a call count: waitForEvent() turned out to be entered
    // only a few thousand times in a 300 s run, so a count-gated line never printed and the
    // timeout(1) kill skips the end-of-run() report entirely (cont.251, first measurement attempt).
    const int g_paceEverySec = []
    { const char *e = std::getenv("PS2X_SCHED_PACE_EVERY"); const int v = e && e[0] ? std::atoi(e) : 10; return v > 0 ? v : 10; }();

    size_t schedPaceBucket(unsigned long long ns)
    {
        if (ns < 1000ull) return 0;
        if (ns < 10000ull) return 1;
        if (ns < 100000ull) return 2;
        if (ns < 1000000ull) return 3;
        if (ns < 5000000ull) return 4;
        if (ns < 20000000ull) return 5;
        return 6;
    }

    void schedPaceReport(uint64_t vsyncTick)
    {
        const SchedPaceStats &s = g_pace;
        const double wallNs = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s.t0).count());
        const double idleNs = static_cast<double>(s.sleptNs + s.stallNs);
        const uint64_t ticks = vsyncTick > s.tick0 ? vsyncTick - s.tick0 : 0u;
        std::fprintf(stderr,
                     "[sched:pace] calls=%llu early=%llu timed=%llu stall=%llu | signaled=%llu timeout=%llu overdue=%llu"
                     " | src vbs=%llu vbe=%llu alarm=%llu other=%llu timer=%llu\n",
                     s.calls, s.early, s.timed, s.stall, s.signaled, s.timedout, s.overdue,
                     s.srcVBlankStart, s.srcVBlankEnd, s.srcAlarm, s.srcOther, s.srcTimer);
        std::fprintf(stderr,
                     "[sched:pace] slept=%.3fs stall=%.3fs req=%.3fs wall=%.3fs IDLE=%.1f%% | mean-slept=%.1fus"
                     " | vblanks=%llu idle/vblank=%.2fms | catchup=%llu cyc\n",
                     double(s.sleptNs) / 1e9, double(s.stallNs) / 1e9, double(s.reqNs) / 1e9, wallNs / 1e9,
                     wallNs > 0.0 ? 100.0 * idleNs / wallNs : 0.0,
                     s.timed ? double(s.sleptNs) / double(s.timed) / 1e3 : 0.0,
                     (unsigned long long)ticks, ticks ? idleNs / double(ticks) / 1e6 : 0.0,
                     s.catchupCycles);
        std::fprintf(stderr,
                     "[sched:pace] slept-buckets <1us=%llu <10us=%llu <100us=%llu <1ms=%llu <5ms=%llu <20ms=%llu >=20ms=%llu\n",
                     s.bucket[0], s.bucket[1], s.bucket[2], s.bucket[3], s.bucket[4], s.bucket[5], s.bucket[6]);
        std::fprintf(stderr,
                     "[sched:pace] why (census over %llu long waits): none=%llu sleep=%llu sema=%llu evf=%llu vsync=%llu"
                     " ext=%llu mpeg=%llu | ready/running=%llu suspended=%llu\n",
                     s.whyWaits, s.whyReason[0], s.whyReason[1], s.whyReason[2], s.whyReason[3],
                     s.whyReason[4], s.whyReason[5], s.whyReason[6], s.whyRunnableIsh, s.whySuspended);
    }
} // namespace

namespace
{
    constexpr int KE_OK = 0;
    constexpr int KE_ERROR = -1;
    constexpr int KE_ILLEGAL_PRIORITY = -403;
    constexpr int KE_ILLEGAL_THID = -406;
    constexpr int KE_UNKNOWN_THID = -407;
    constexpr int KE_UNKNOWN_SEMID = -408;
    constexpr int KE_UNKNOWN_EVFID = -409;
    constexpr int KE_DORMANT = -413;
    constexpr int KE_NOT_DORMANT = -414;
    constexpr int KE_NOT_SUSPEND = -415;
    constexpr int KE_NOT_WAIT = -416;
    constexpr int KE_RELEASE_WAIT = -418;
    constexpr int KE_SEMA_ZERO = -419;
    constexpr int KE_SEMA_OVF = -420;
    constexpr int KE_EVF_COND = -421;
    constexpr int KE_WAIT_DELETE = -425;

    constexpr uint32_t WEF_OR = 0x01u;
    constexpr uint32_t WEF_CLEAR = 0x10u;
    constexpr uint32_t WEF_CLEAR_ALL = 0x20u;
    // ★ cont.232: the vblank period follows the VIDEO MODE the guest selects with SetGsCrt, as PCSX2's
    // Counters.cpp GetVerticalFrequency() does: PAL 50.00 Hz (49.76 non-interlaced), NTSC 59.94
    // (59.82 non-interlaced), SDTV 480p 59.94, VESA/HDTV 60.00, and 60.00 until SetGsCrt has run. It
    // was a fixed 16667 us (60.00 Hz) for every game: a PAL title that counts vsyncs for its game clock
    // (LOTR: dt = ticks x 1/50, clamped) ran ~1.2x fast, and its pad/vsync callbacks fired 60x/s.
    // PS2X_VBLANK_HZ=<float> overrides (A/B; 0/unset = follow SetGsCrt).
    std::atomic<uint32_t> g_vblankPeriodUs{16667u};
    std::atomic<bool> g_vblankFromSetGsCrt{false}; // SetGsCrt seen: its mode wins over the SMODE1 fallback
    // The console's boot-time mode: a real machine's OSD has already run SetGsCrt for its region before
    // the game starts, and a game that never programs the CRTC itself (LOTR: no SetGsCrt caller, SMODE1
    // stays 0) simply inherits it. Set from the ELF name (SLES/SCES/SLED = PAL) by PS2Runtime::loadELF,
    // overridden by PS2X_REGION=PAL|NTSC.
    std::atomic<bool> g_defaultRegionPal{false};
    constexpr auto kVBlankDuration = std::chrono::microseconds(500);
    std::chrono::microseconds vblankPeriod() { return std::chrono::microseconds(g_vblankPeriodUs.load(std::memory_order_relaxed)); }
    constexpr uint64_t kAlarmTickMicroseconds = 64u;
    constexpr uint32_t kDebugPublishDispatchInterval = 4096u;

    constexpr uint64_t microsecondsToEeCycles(uint64_t microseconds)
    {
        return (microseconds * EeScheduler::kEeClockHz + 999999ull) / 1000000ull;
    }

    std::chrono::nanoseconds eeCyclesToHostDuration(uint64_t cycles)
    {
        constexpr uint64_t kNanosecondsPerSecond = 1000000000ull;
        const uint64_t wholeSeconds = cycles / EeScheduler::kEeClockHz;
        const uint64_t remainingCycles = cycles % EeScheduler::kEeClockHz;
        const uint64_t remainingNanoseconds = (remainingCycles * kNanosecondsPerSecond + EeScheduler::kEeClockHz - 1u) / EeScheduler::kEeClockHz;
        return std::chrono::seconds(wholeSeconds) + std::chrono::nanoseconds(remainingNanoseconds);
    }

    uint64_t vblankPeriodCycles() { return microsecondsToEeCycles(g_vblankPeriodUs.load(std::memory_order_relaxed)); }
    constexpr uint64_t kVBlankDurationCycles = microsecondsToEeCycles(500u);
    constexpr uint64_t kAlarmTickCycles = microsecondsToEeCycles(kAlarmTickMicroseconds);

    template <typename Map>
    int allocatePositiveId(int &nextId, const Map &objects)
    {
        const int first = std::max(1, nextId);
        int candidate = first;
        do
        {
            if (!objects.contains(candidate))
            {
                nextId = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
                return candidate;
            }
            candidate = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
        } while (candidate != first);
        return 0;
    }
}

EeScheduler::EeScheduler(PS2Runtime &runtime)
    : m_runtime(runtime)
{
}

EeScheduler::~EeScheduler()
{
    requestStop();
}

void EeScheduler::reset(uint8_t *rdram, const R5900Context &mainContext)
{
    m_executorThread = std::this_thread::get_id();
    m_rdram = rdram;
    m_readyQueues = {};
    m_threads.clear();
    m_semaphores.clear();
    m_eventFlags.clear();
    m_alarms.clear();
    m_intcHandlers.clear();
    m_dmacHandlers.clear();
    m_nextThreadId = kFirstThreadId;
    m_nextInvocationThreadId = -1;
    m_nextSemaphoreId = 1;
    m_nextEventFlagId = 1;
    m_nextAlarmId = 1;
    m_nextIntcHandlerId = 1;
    m_nextDmacHandlerId = 1;
    m_intcHeadOrder = 0;
    m_intcTailOrder = 1000;
    m_dmacHeadOrder = 0;
    m_dmacTailOrder = 1000;
    m_enabledIntcMask = 0xFFFFFFFFu;
    m_enabledDmacMask = 0xFFFFFFFFu;
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    m_insideInterrupt = false;
    m_pendingEeTimerInterrupts = 0u;
    m_eeCycle = 0u;
    m_sliceEndCycle = kDefaultTimeSliceCycles;
    m_stopRequested.store(false, std::memory_order_release);
    m_checkpointPending.store(false, std::memory_order_release);
    m_debugPublishCountdown = 0u;
    {
        std::lock_guard lock(m_eventMutex);
        m_events.clear();
        m_deadlines.clear();
        m_pendingInvocations.clear();
        m_pendingSifCommands = 0u;
    }
    m_eventSequence = 0;
    m_invocationSequence = 0;
    m_vsyncTick = 0;
    m_vsyncFlagAddress = 0;
    m_vsyncTickAddress = 0;
    m_gsVSyncCallback = 0;
    m_gsVSyncCallbackGp = 0;
    m_gsVSyncCallbackSp = 0;
    m_runtime.memory().gs().vsyncTick.store(0u, std::memory_order_release);
    m_runtime.memory().resetEeTimers();

    GuestThread main{};
    main.id = kMainThreadId;
    main.context = mainContext;
    main.entry = mainContext.pc;
    // $sp is live execution state, not the stable initial stack descriptor
    // returned by ReferThreadStatus. SetupThread records that metadata.
    main.stack = 0u;
    main.gp = getRegU32(&mainContext, 28);
    main.initialPriority = 0;
    main.currentPriority = 0;
    main.status = EeThreadStatus::Ready;
    m_threads.emplace(main.id, std::move(main));
    m_readyQueues[0].push_back(kMainThreadId);
    scheduleEvent(m_eeCycle + vblankPeriodCycles(),
                  std::chrono::steady_clock::now() + vblankPeriod(),
                  EeEvent{EeEventType::VBlankStart, 0, 0});
    publishSnapshot();
}

// cont.232: called by the SetGsCrt syscall HLE (Kernel/Syscalls/System.cpp) with the raw arguments.
// The PCSX2 mapping (R5900OpcodeImpl.cpp SYSCALL SetGsCrt -> gsSetVideoMode; Counters.cpp
// GetVerticalFrequency): mode 0/2 = NTSC, 1/3 = PAL, 0x50 = SDTV 480p (59.94), 0x51/0x52 = HDTV
// 1080i/720p (60.00), 0x53 = SDTV 576p (60.00), 0x1A..0x4B = VESA (60.00); non-interlaced NTSC/PAL
// run 0.11 / 0.24 Hz slower. Takes effect at the next VBlankStart scheduling.
void ps2xSetGsVideoMode(uint32_t interlaced, uint32_t mode) noexcept
{
    // PCSX2 reads the LOW BYTE of each argument (cpuRegs.GPR.n.a1.UC[0]): the SDK prototype is
    // SetGsCrt(short, short, short) and callers leave the upper halves undefined.
    const uint32_t rawI = interlaced, rawM = mode;
    interlaced &= 0xFFu;
    mode &= 0xFFu;
    static bool s_first = true;
    if (s_first)
    {
        s_first = false;
        std::fprintf(stderr, "[vblank] SetGsCrt called: a0=0x%x a1=0x%x (interlaced=%u mode=0x%x)\n", rawI, rawM, interlaced, mode);
    }
    double hz = 60.0;
    const char *name = "unknown->60.00";
    switch (mode)
    {
    case 0x0: case 0x2: hz = interlaced ? 59.94 : 59.82; name = "NTSC"; break;
    case 0x1: case 0x3: hz = interlaced ? 50.00 : 49.76; name = "PAL"; break;
    case 0x50: hz = 59.94; name = "SDTV 480p"; break;
    case 0x51: hz = 60.00; name = "HDTV 1080i"; break;
    case 0x52: hz = 60.00; name = "HDTV 720p"; break;
    case 0x53: hz = 60.00; name = "SDTV 576p"; break;
    default: hz = 60.00; name = (mode >= 0x1A && mode <= 0x4B) ? "VESA" : "unknown->60.00"; break;
    }
    static const double s_override = []
    { const char *e = std::getenv("PS2X_VBLANK_HZ"); return e && e[0] ? std::atof(e) : 0.0; }();
    if (s_override > 0.0)
        hz = s_override;
    const uint32_t us = static_cast<uint32_t>(1000000.0 / hz + 0.5);
    g_vblankFromSetGsCrt.store(true, std::memory_order_relaxed);
    const uint32_t old = g_vblankPeriodUs.exchange(us, std::memory_order_relaxed);
    if (old != us)
        std::fprintf(stderr, "[vblank] SetGsCrt interlaced=%u mode=0x%x -> %s %.2f Hz (%u us/vblank; was %u us)%s\n",
                     interlaced, mode, name, hz, us, old, s_override > 0.0 ? " [PS2X_VBLANK_HZ override]" : "");
}

// cont.232: games that program the CRTC themselves (LOTR never calls SetGsCrt) leave the mode in the GS
// privileged register SMODE1: CMOD (bits 13-14) = 2 NTSC / 3 PAL (GS User's Manual; PCSX2 GSRegs.h
// GSRegSMODE1.CMOD), SMODE2.INT = interlaced. Evaluated at every VBlankStart while SetGsCrt has not
// spoken; same rates as above.
void ps2xSetDefaultVideoRegion(bool pal) noexcept
{
    g_defaultRegionPal.store(pal, std::memory_order_relaxed);
}

void EeScheduler::refreshVblankFromSmode1()
{
    if (g_vblankFromSetGsCrt.load(std::memory_order_relaxed))
        return;
    const uint64_t smode1 = m_runtime.memory().gs().smode1;
    const uint64_t smode2 = m_runtime.memory().gs().smode2;
    static uint64_t s_lastSmode1 = ~0ull, s_lastSmode2 = ~0ull;
    if (smode1 != s_lastSmode1 || smode2 != s_lastSmode2)
    {
        std::fprintf(stderr, "[vblank] CRTC regs: SMODE1=0x%llx SMODE2=0x%llx (vsync #%llu)\n",
                     (unsigned long long)smode1, (unsigned long long)smode2, (unsigned long long)m_vsyncTick);
        s_lastSmode1 = smode1; s_lastSmode2 = smode2;
    }
    const uint32_t cmod = static_cast<uint32_t>((smode1 >> 13) & 0x3ull);
    // SMODE2 unset (0) means the game left the console's interlaced default; treat as interlaced.
    const bool interlaced = smode2 == 0ull || (smode2 & 0x1ull) != 0ull;
    static const int s_regionEnv = []
    {
        const char *e = std::getenv("PS2X_REGION");
        if (!e || !e[0]) return -1;
        return (e[0] == 'P' || e[0] == 'p') ? 1 : 0;
    }();
    const bool defPal = s_regionEnv >= 0 ? (s_regionEnv == 1) : g_defaultRegionPal.load(std::memory_order_relaxed);
    double hz = 60.0;
    const char *name = "unknown->60.00";
    if (cmod == 3u) { hz = interlaced ? 50.00 : 49.76; name = "PAL (SMODE1)"; }
    else if (cmod == 2u) { hz = interlaced ? 59.94 : 59.82; name = "NTSC (SMODE1)"; }
    else if (defPal) { hz = interlaced ? 50.00 : 49.76; name = "PAL (console default)"; }
    else { hz = interlaced ? 59.94 : 59.82; name = "NTSC (console default)"; }
    static const double s_override = []
    { const char *e = std::getenv("PS2X_VBLANK_HZ"); return e && e[0] ? std::atof(e) : 0.0; }();
    if (s_override > 0.0)
        hz = s_override;
    const uint32_t us = static_cast<uint32_t>(1000000.0 / hz + 0.5);
    const uint32_t old = g_vblankPeriodUs.exchange(us, std::memory_order_relaxed);
    static bool s_firstEval = true;
    if (old != us || s_firstEval)
        std::fprintf(stderr, "[vblank] SMODE1=0x%llx CMOD=%u SMODE2=0x%llx -> %s %.2f Hz (%u us/vblank; was %u us)%s\n",
                     (unsigned long long)smode1, cmod, (unsigned long long)smode2, name, hz, us, old,
                     s_override > 0.0 ? " [PS2X_VBLANK_HZ override]" : "");
    s_firstEval = false;
}

// ---- Stack-guard diagnostic (PS2X_STACKGUARD, default OFF; cont.248). The "EE scheduler missing-target"
// race resumes a guest context whose saved ra/s-registers were overwritten on ITS OWN stack while it was not
// running (crash sp always inside the main thread's frames, 0x10fca0..0x10feb0 on LOTR). After every dispatch
// the guard snapshots [sp, sp+0x400) of the context that just ran, keyed by (thread, invocation depth); when
// that same key is dispatched again after OTHER keys ran in between, it compares the region and, on a change,
// prints the differing words and the intervening dispatches (thread, depth, kind, pc) -- the writer's identity.
namespace
{
struct StackGuardSnap
{
    uint32_t sp = 0;
    unsigned long seq = 0;
    std::array<uint8_t, 0x400> bytes{};
};
struct StackGuardDispatch
{
    int tid = 0;
    size_t depth = 0;
    int kind = -1;
    uint32_t pc = 0;
    // cont.249: sp/ra of the context AS DISPATCHED. The ww2/ww4/ww7 failure is a stack-pointer DESYNC
    // (FUN_0015bba0's epilogue ran on FUN_00159a30's frame), so the ring has to show where sp jumps.
    uint32_t sp = 0;
    uint32_t ra = 0;
};
const bool s_stackGuard = []
{ const char *e = std::getenv("PS2X_STACKGUARD"); return e && e[0] && e[0] != '0'; }();
std::unordered_map<uint64_t, StackGuardSnap> s_stackSnaps;
std::array<StackGuardDispatch, 256> s_stackRing{};
unsigned long s_stackSeq = 0;
uint64_t s_stackLastKey = ~0ull;
int s_stackReports = 0;

inline uint64_t stackGuardKey(int tid, size_t depth)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(tid)) << 32u) | static_cast<uint32_t>(depth);
}

void stackGuardCheckBefore(const uint8_t *rdram, int tid, size_t depth, int kind, const R5900Context &ctx)
{
    const uint64_t key = stackGuardKey(tid, depth);
    const uint32_t sp = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx.r[29]));
    if (key != s_stackLastKey)
    {
        const auto it = s_stackSnaps.find(key);
        if (it != s_stackSnaps.end() && it->second.sp == sp && (sp & 0x01FFFFFFu) + 0x400u <= 0x02000000u && s_stackReports < 6)
        {
            const uint8_t *now = rdram + (sp & 0x01FFFFFFu);
            int diffs = 0;
            for (uint32_t off = 0; off < 0x400u; off += 4u)
            {
                if (std::memcmp(now + off, it->second.bytes.data() + off, 4) != 0)
                {
                    if (diffs == 0)
                    {
                        ++s_stackReports;
                        std::fprintf(stderr, "[stackguard] thread %d depth %zu (pc=0x%x sp=0x%x) resumes with its stack CHANGED while it was not running (snap seq %lu, now %lu):\n",
                                     tid, depth, ctx.pc, sp, it->second.seq, s_stackSeq);
                    }
                    if (++diffs <= 24)
                    {
                        uint32_t o, n;
                        std::memcpy(&o, it->second.bytes.data() + off, 4);
                        std::memcpy(&n, now + off, 4);
                        std::fprintf(stderr, "[stackguard]   sp+0x%03x (0x%x): 0x%08x -> 0x%08x\n", off, sp + off, o, n);
                    }
                }
            }
            if (diffs > 0)
            {
                std::fprintf(stderr, "[stackguard]   %d words changed; dispatches since the snapshot (oldest first):\n", diffs);
                const unsigned long from = it->second.seq + 1;
                for (unsigned long q = from; q <= s_stackSeq && q > s_stackSeq - 256; ++q)
                {
                    const StackGuardDispatch &d = s_stackRing[q % 256];
                    std::fprintf(stderr, "[stackguard]     #%lu thread %d depth %zu kind %d pc=0x%x sp=0x%x ra=0x%x\n", q, d.tid, d.depth, d.kind, d.pc, d.sp, d.ra);
                }
            }
        }
    }
    ++s_stackSeq;
    g_ps2DispatchSeq = s_stackSeq;
    s_stackRing[s_stackSeq % 256] = StackGuardDispatch{tid, depth, kind, ctx.pc, sp,
                                                       static_cast<uint32_t>(_mm_cvtsi128_si32(ctx.r[31]))};
    s_stackLastKey = key;
}

void stackGuardSnapAfter(const uint8_t *rdram, int tid, size_t depth, const R5900Context &ctx)
{
    const uint32_t sp = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx.r[29]));
    if ((sp & 0x01FFFFFFu) + 0x400u > 0x02000000u)
    {
        return;
    }
    StackGuardSnap &snap = s_stackSnaps[stackGuardKey(tid, depth)];
    snap.sp = sp;
    snap.seq = s_stackSeq;
    std::memcpy(snap.bytes.data(), rdram + (sp & 0x01FFFFFFu), 0x400u);
}
} // namespace

void EeScheduler::run()
{
    assertExecutor();
    m_running.store(true, std::memory_order_release);

    while (!m_stopRequested.load(std::memory_order_acquire))
    {
        processPendingEvents();
        if (m_stopRequested.load(std::memory_order_acquire))
        {
            break;
        }

        if (m_currentThreadId == 0)
        {
            GuestThread *next = selectReady();
            if (!next && m_pendingInvocations.empty())
            {
                if (s_stackGuard)
                {
                    // cont.248: an idle scheduler with a stalled heartbeat = every guest thread blocked on something nobody
                    // signals. Dump the thread table (rate-limited) so the log names the wait.
                    static auto s_lastIdleDump = std::chrono::steady_clock::now() - std::chrono::seconds(10);
                    const auto now = std::chrono::steady_clock::now();
                    if (now - s_lastIdleDump >= std::chrono::seconds(3))
                    {
                        s_lastIdleDump = now;
                        std::fprintf(stderr, "[stackguard] IDLE: no ready thread, no pending invocation; vsyncTick=%llu threads=%zu\n",
                                     (unsigned long long)m_vsyncTick, m_threads.size());
                        for (const auto &[id, th] : m_threads)
                        {
                            int waitId = -1;
                            if (th.wait.reason == EeWaitReason::Semaphore) waitId = std::get<EeSemaphoreWait>(th.wait.payload).id;
                            else if (th.wait.reason == EeWaitReason::EventFlag) waitId = std::get<EeEventFlagWait>(th.wait.payload).id;
                            std::fprintf(stderr, "[stackguard]   thread %d status=%d prio=%d wait=%d id=%d base-pc=0x%x sp=0x%x invocations=%zu",
                                         id, static_cast<int>(th.status), th.currentPriority, static_cast<int>(th.wait.reason), waitId,
                                         th.context.pc, static_cast<uint32_t>(_mm_cvtsi128_si32(th.context.r[29])), th.invocations.size());
                            for (const GuestInvocation &inv : th.invocations)
                                std::fprintf(stderr, " [kind %d pc=0x%x ra=0x%x]", static_cast<int>(inv.kind), inv.context.pc,
                                             static_cast<uint32_t>(_mm_cvtsi128_si32(inv.context.r[31])));
                            std::fprintf(stderr, "\n");
                        }
                    }
                }
                copyMainContextToRuntime();
                publishIdleDebugContext();
                publishSnapshot();
                waitForEvent();
                continue;
            }
            if (next)
            {
                makeRunning(*next);
            }
            else
            {
                GuestThread *owner = &acquireInvocationThread();
                GuestInvocation invocation = std::move(m_pendingInvocations.front());
                m_pendingInvocations.pop_front();
                if (invocation.kind == GuestInvocationKind::SifCommand && m_pendingSifCommands)
                    --m_pendingSifCommands;
                owner->status = EeThreadStatus::Running;
                m_currentThreadId = owner->id;
                renewTimeSlice();
                if (getRegU32(&invocation.context, 29) == 0u)
                {
                    SET_GPR_U32(&invocation.context, 29, invocationStackTop());
                }
                if (s_invokeLog && s_invokeLogBudget > 0)
                {
                    --s_invokeLogBudget;
                    std::fprintf(stderr, "[eeinv] push-idle kind=%d pc=0x%x thr=%d depth=%zu\n",
                                 static_cast<int>(invocation.kind), invocation.context.pc,
                                 owner->id, owner->invocations.size());
                }
                owner->invocations.push_back(std::move(invocation));
            }
        }

        GuestThread *running = currentThread();
        assert(running != nullptr);
        if (running->resumeCompletion)
        {
            auto completion = std::move(running->resumeCompletion);
            running->resumeCompletion = {};
            try
            {
                completion(running->activeContext());
            }
            catch (const EeDispatcherTransfer &)
            {
            }
            if (m_currentThreadId == 0)
            {
                continue;
            }
        }
        R5900Context &context = running->activeContext();
        if (m_debugPublishCountdown == 0u)
        {
            copyMainContextToRuntime();
            publishSnapshot();
            m_debugPublishCountdown = kDebugPublishDispatchInterval - 1u;
        }
        else
        {
            --m_debugPublishCountdown;
        }

        publishDebugContext(context);

        if (context.pc == 0u)
        {
            if (!running->invocations.empty())
            {
                GuestInvocation completed = std::move(running->invocations.back());
                running->invocations.pop_back();
                if (s_invokeLog && s_invokeLogBudget > 0)
                {
                    --s_invokeLogBudget;
                    std::fprintf(stderr, "[eeinv] done kind=%d thr=%d depth->%zu\n",
                                 static_cast<int>(completed.kind), running->id,
                                 running->invocations.size());
                }
                if (completed.onComplete)
                {
                    try
                    {
                        completed.onComplete(completed.context, running->activeContext());
                    }
                    catch (const EeDispatcherTransfer &)
                    {
                    }
                }
                continue;
            }
            makeDormant(*running);
            m_currentThreadId = 0;
            copyMainContextToRuntime();
            publishIdleDebugContext();
            publishSnapshot();
            continue;
        }

        if (!m_pendingInvocations.empty() && running->invocations.empty())
        {
            GuestInvocation invocation = std::move(m_pendingInvocations.front());
            m_pendingInvocations.pop_front();
            if (invocation.kind == GuestInvocationKind::SifCommand && m_pendingSifCommands)
                --m_pendingSifCommands;
            if (getRegU32(&invocation.context, 29) == 0u)
            {
                SET_GPR_U32(&invocation.context, 29, invocationStackTop());
            }
            if (s_invokeLog && s_invokeLogBudget > 0)
            {
                --s_invokeLogBudget;
                std::fprintf(stderr, "[eeinv] push-run kind=%d pc=0x%x thr=%d depth=%zu basepc=0x%x\n",
                             static_cast<int>(invocation.kind), invocation.context.pc,
                             running->id, running->invocations.size(), running->context.pc);
            }
            running->invocations.push_back(std::move(invocation));
            continue;
        }

        if (!m_runtime.hasFunction(context.pc))
        {
            if (!running->invocations.empty())
            {
                context.pc = 0u;
            }
            else
            {
                m_runtime.reportMissingFunction(m_rdram,
                                                &context,
                                                context.pc,
                                                context.pc,
                                                PS2Runtime::GuestBranchKind::DirectJump,
                                                "EE scheduler");
                if (s_stackGuard)
                {
                    // cont.248: the resumed context is corrupt -- dump the guest stack around its sp (the frames the
                    // corrupted saved registers were loaded from) and the last dispatches, so the overwritten frame
                    // and the writer can be identified from the log alone.
                    const uint32_t sp = static_cast<uint32_t>(_mm_cvtsi128_si32(context.r[29]));
                    std::fprintf(stderr, "[stackguard] CRASH thread %d depth %zu pc=0x%x sp=0x%x -- stack words [sp-0x40, sp+0x300):\n",
                                 running->id, running->invocations.size(), context.pc, sp);
                    for (uint32_t a = sp - 0xA0u; a < sp + 0x300u; a += 0x20u)
                    {
                        if ((a & 0x01FFFFFFu) + 0x20u > 0x02000000u) break;
                        uint32_t w[8];
                        std::memcpy(w, m_rdram + (a & 0x01FFFFFFu), sizeof w);
                        std::fprintf(stderr, "[stackguard]   %08x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                                     a, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                    }
                    // cont.249: the saved-ra slot the failing `jr ra` read lives in the frame BELOW the
                    // reported sp (the delay slot already popped it), so name every writer down there.
                    ps2WriteWatchDumpRange(sp - 0xA0u, sp + 0x20u);
                    ps2WriteWatchDump();
                    std::fprintf(stderr, "[stackguard]   last dispatches (oldest first):\n");
                    for (unsigned long q = (s_stackSeq > 199 ? s_stackSeq - 199 : 1); q <= s_stackSeq; ++q)
                    {
                        const StackGuardDispatch &d = s_stackRing[q % 256];
                        std::fprintf(stderr, "[stackguard]     #%lu thread %d depth %zu kind %d pc=0x%x sp=0x%x ra=0x%x\n", q, d.tid, d.depth, d.kind, d.pc, d.sp, d.ra);
                    }
                }
                makeDormant(*running);
                m_currentThreadId = 0;
            }
            continue;
        }
        PS2Runtime::RecompiledFunction function = m_runtime.lookupFunction(context.pc);

        if (checkpointDue(kGuestDispatchCycles))
        {
            if (s_invokeLog)
            {
                static unsigned long s_cpSkips = 0;
                if ((++s_cpSkips % 200000UL) == 1UL)
                    std::fprintf(stderr, "[eeinv] cp-skip n=%lu thr=%d pc=0x%x depth=%zu\n",
                                 s_cpSkips, running->id, context.pc, running->invocations.size());
            }
            continue;
        }
        if (s_invokeLog && !running->invocations.empty() && s_invokeLogBudget > 0)
        {
            --s_invokeLogBudget;
            std::fprintf(stderr, "[eeinv] disp thr=%d pc=0x%x depth=%zu\n",
                         running->id, context.pc, running->invocations.size());
        }

        // ★ cont.232: COP0 Count. The recompiled `mfc0 $rt, Count` reads ctx->cop0_count and nothing
        // advanced it (frozen at 0 since the start: the game's Count-based stopwatch/streaming timers
        // measured nothing). PCSX2: Count = cpuRegs.cycle, the EE cycle counter at 294.912 MHz, which at
        // full speed is wall time (COP0.cpp / Counters.cpp). Here the EE runs untimed, so Count is derived
        // from the host clock at the EE rate, refreshed on every 8th dispatch (a clock read is ~20 ns; a
        // guest frame is thousands of dispatches, so the granularity is far below a frame). A guest
        // busy-wait on Count inside ONE recompiled function (no dispatch) would still spin -- none does
        // today, or it would already hang on the frozen value.
        {
            static uint32_t s_countRefresh = 0;
            if ((++s_countRefresh & 7u) == 0u)
            {
                if (s_virtualTime)
                    context.cop0_count = static_cast<uint32_t>(m_eeCycle); // cont.322e: the cycle clock IS the time
                else
                {
                static const auto s_t0 = std::chrono::steady_clock::now();
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s_t0).count();
                context.cop0_count = static_cast<uint32_t>((static_cast<__int128>(ns) * kEeClockHz) / 1000000000ll);
                }
            }
        }
        if (s_stackGuard)
        {
            stackGuardCheckBefore(m_rdram, running->id, running->invocations.size(),
                                  running->invocations.empty() ? -1 : static_cast<int>(running->invocations.back().kind), context);
        }
        try
        {
            m_insideInterrupt = !running->invocations.empty() && running->invocations.back().kind == GuestInvocationKind::Interrupt;
            m_guestExecuting.store(true, std::memory_order_release);
            {
                GuestProfScope guestProf__;
                function(m_rdram, &context, &m_runtime);
            }
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
            if (s_stackGuard && m_currentThreadId == running->id)
            {
                stackGuardSnapAfter(m_rdram, running->id, running->invocations.size(), running->activeContext());
            }
        }
        catch (const EeDispatcherTransfer &)
        {
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
        }
        catch (...)
        {
            m_guestExecuting.store(false, std::memory_order_release);
            m_running.store(false, std::memory_order_release);
            publishSnapshot();
            throw;
        }

        processPendingEvents();
        if (m_rescheduleRequested && m_currentThreadId != 0)
        {
            GuestThread *preempted = currentThread();
            assert(preempted != nullptr);
            enqueueReady(*preempted, !m_timeSliceExpired);
            m_currentThreadId = 0;
            m_rescheduleRequested = false;
            m_timeSliceExpired = false;
        }
    }

    m_guestExecuting.store(false, std::memory_order_release);
    m_running.store(false, std::memory_order_release);
    if (s_schedPace && g_pace.started)
    {
        std::fprintf(stderr, "[sched:pace] FINAL\n");
        schedPaceReport(m_vsyncTick);
    }
    copyMainContextToRuntime();
    publishSnapshot();
}

void EeScheduler::requestStop()
{
    m_stopRequested.store(true, std::memory_order_release);
    m_checkpointPending.store(true, std::memory_order_release);
    m_eventCv.notify_all();
}

void EeScheduler::postEvent(EeEvent event)
{
    if (event.type == EeEventType::Stop)
    {
        requestStop();
        return;
    }

    {
        std::lock_guard lock(m_eventMutex);
        m_events.push_back(event);
        m_checkpointPending.store(true, std::memory_order_release);
    }
    m_eventCv.notify_one();
}

bool EeScheduler::checkpointDueSlow() noexcept
{
    // cont.317 (cont.): the cycles were accounted by the inline checkpointDue(); this is the rest of
    // the previous body, unchanged.

    // Host-pump scope: report no checkpoint so pumped guest chains run to completion (see the
    // block comment at the top of this file). Cycles above still accrue; a due deadline fires
    // as soon as the scope closes. Stop-requests still preempt.
    if (s_hostPumpRtc && s_hostPumpDepth.load(std::memory_order_relaxed) != 0u &&
        !m_stopRequested.load(std::memory_order_acquire))
    {
        // rotk row 269: a SIF command handler is an interrupt on hardware -- it preempts the running code, pumped
        // chain or not. A guest chain that waits on an IOP reply inside a host pump (rotk: the level-load drain)
        // would otherwise wait forever: the run loop cannot deliver an invocation until the chain ends. So deliver
        // the queued SIF handlers here, at this block boundary. Only when the game forwards SIF commands to the
        // emulated IOP (setIopSifCommandForwarding); nothing else queues SifCommand invocations.
        if (m_pendingSifCommands != 0u && !m_deliveringSifCommands && m_runtime.iopSifCommandForwarding())
        {
            m_deliveringSifCommands = true;
            (void)m_runtime.runPendingSifCommandHandlers(m_runtime.memory().getRDRAM());
            m_deliveringSifCommands = false;
        }
        return false;
    }

    if (m_checkpointPending.load(std::memory_order_acquire) ||
        m_stopRequested.load(std::memory_order_acquire))
    {
        m_checkpointEpoch.fetch_add(1u, std::memory_order_acq_rel);
        return true;
    }

    const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    if (nextEventCycle != 0u && m_eeCycle >= nextEventCycle)
    {
        m_checkpointPending.store(true, std::memory_order_release);
        m_checkpointEpoch.fetch_add(1u, std::memory_order_acq_rel);
        return true;
    }

    if (m_eeCycle < m_sliceEndCycle)
    {
        return false;
    }

    const GuestThread *running = currentThread();
    if (running != nullptr && hasReadyAtOrAbovePriority(running->currentPriority))
    {
        m_rescheduleRequested = true;
        m_timeSliceExpired = true;
        m_checkpointEpoch.fetch_add(1u, std::memory_order_acq_rel);
        return true;
    }

    renewTimeSlice();
    return false;
}

bool EeScheduler::isExecutingGuest() const noexcept
{
    return m_guestExecuting.load(std::memory_order_acquire);
}

void EeScheduler::setupCurrentThread(uint32_t stack, uint32_t stackSize, uint32_t gp)
{
    assertExecutor();
    GuestThread *target = currentThread();
    if (!target)
    {
        return;
    }

    target->stack = stack;
    target->stackSize = stackSize;
    target->gp = gp;
    publishSnapshot();
}

int EeScheduler::createThread(const EeThreadCreateParams &params)
{
    assertExecutor();
    if (params.priority < 1 || params.priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }

    const int id = allocateThreadId();
    if (id == 0)
    {
        return KE_ERROR;
    }

    GuestThread thread{};
    thread.id = id;
    thread.entry = params.entry;
    thread.stack = params.stack;
    thread.stackSize = params.stackSize;
    thread.gp = params.gp;
    thread.attr = params.attr;
    thread.option = params.option;
    thread.initialPriority = params.priority;
    thread.currentPriority = params.priority;
    thread.status = EeThreadStatus::Dormant;
    m_threads.emplace(id, std::move(thread));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteThread(int id, uint32_t &ownedStack)
{
    assertExecutor();
    ownedStack = 0;
    if (id <= kMainThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    auto it = m_threads.find(id);
    if (it == m_threads.end())
    {
        return KE_UNKNOWN_THID;
    }
    if (it->second.status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }
    if (it->second.ownsStack)
    {
        ownedStack = it->second.stack;
    }
    m_threads.erase(it);
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::startThread(int id, uint32_t arg, const R5900Context &caller, bool interruptSafe)
{
    assertExecutor();
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }

    target->context = R5900Context{};
    target->context.pc = target->entry;
    target->arg = arg;
    target->suspendCount = 0;
    target->wakeupCount = 0;
    target->wait = {};
    SET_GPR_U32(&target->context, 4, arg);
    SET_GPR_U32(&target->context, 28, target->gp != 0u ? target->gp : getRegU32(&caller, 28));
    const uint32_t stackTop = target->stack != 0u
                                  ? (target->stack + target->stackSize) & ~0xFu
                                  : getRegU32(&caller, 29);
    SET_GPR_U32(&target->context, 29, stackTop);
    SET_GPR_U32(&target->context, 31, 0u);
    enqueueReady(*target);
    requestPreemptionIfHigher(*target, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

[[noreturn]] void EeScheduler::exitCurrent(bool deleteThreadRecord)
{
    assertExecutor();
    GuestThread *exiting = currentThread();
    assert(exiting != nullptr);
    const int id = exiting->id;
    const uint32_t ownedStack = deleteThreadRecord && exiting->ownsStack ? exiting->stack : 0u;
    makeDormant(*exiting);
    m_currentThreadId = 0;
    if (deleteThreadRecord && id != kMainThreadId)
    {
        m_threads.erase(id);
    }
    if (ownedStack != 0u)
    {
        m_runtime.guestFree(ownedStack);
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::terminateThread(int id, uint32_t &ownedStack, bool interruptSafe)
{
    assertExecutor();
    ownedStack = 0;
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if (target->ownsStack)
    {
        ownedStack = target->stack;
        target->ownsStack = false;
    }
    makeDormant(*target);
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::suspendThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }

    ++target->suspendCount;
    switch (target->status)
    {
    case EeThreadStatus::Running:
        target->status = EeThreadStatus::Suspended;
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
        break;
    case EeThreadStatus::Ready:
        removeReady(*target);
        target->status = EeThreadStatus::Suspended;
        break;
    case EeThreadStatus::Waiting:
        target->status = EeThreadStatus::WaitingSuspended;
        break;
    case EeThreadStatus::WaitingSuspended:
    case EeThreadStatus::Suspended:
        break;
    case EeThreadStatus::Dormant:
        break;
    }
    if (interruptSafe && m_insideInterrupt)
    {
        m_rescheduleRequested = true;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::resumeThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->suspendCount == 0)
    {
        return KE_NOT_SUSPEND;
    }
    --target->suspendCount;
    if (target->suspendCount != 0)
    {
        return KE_OK;
    }
    if (target->status == EeThreadStatus::WaitingSuspended)
    {
        target->status = EeThreadStatus::Waiting;
    }
    else if (target->status == EeThreadStatus::Suspended)
    {
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::sleepCurrent()
{
    assertExecutor();
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (self->wakeupCount != 0u)
    {
        --self->wakeupCount;
        setReturnS32(&self->activeContext(), KE_OK);
        return;
    }
    blockCurrent(EeWaitState{EeWaitReason::Sleep, std::monostate{}});
}

int EeScheduler::wakeupThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if ((target->status == EeThreadStatus::Waiting || target->status == EeThreadStatus::WaitingSuspended) &&
        target->wait.reason == EeWaitReason::Sleep)
    {
        makeReady(*target, KE_OK, interruptSafe);
    }
    else
    {
        ++target->wakeupCount;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::cancelWakeup(int id)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    const int old = static_cast<int>(target->wakeupCount);
    target->wakeupCount = 0;
    publishSnapshot();
    return old;
}

int EeScheduler::changePriority(int id, int priority, bool interruptSafe, int &oldPriority)
{
    assertExecutor();
    if (priority < 1 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    oldPriority = target->currentPriority;
    if (target->status == EeThreadStatus::Ready)
    {
        removeReady(*target);
        target->currentPriority = priority;
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    else
    {
        target->currentPriority = priority;
        if (target->status == EeThreadStatus::Running)
        {
            for (int p = 0; p < target->currentPriority; ++p)
            {
                if (!m_readyQueues[p].empty())
                {
                    m_rescheduleRequested = true;
                    break;
                }
            }
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::rotateReadyQueue(int priority, bool interruptSafe)
{
    assertExecutor();
    if (priority == 0)
    {
        const GuestThread *self = currentThread();
        priority = self ? self->currentPriority : 0;
    }
    if (priority < 0 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }

    GuestThread *self = currentThread();
    if (self && self->currentPriority == priority)
    {
        enqueueReady(*self);
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
    }
    else
    {
        auto &queue = m_readyQueues[priority];
        if (queue.size() > 1u)
        {
            const int head = queue.front();
            queue.pop_front();
            queue.push_back(head);
        }
    }
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::releaseWait(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Waiting && target->status != EeThreadStatus::WaitingSuspended)
    {
        return KE_NOT_WAIT;
    }
    removeFromWaitObject(*target);
    makeReady(*target, KE_RELEASE_WAIT, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::transferIfRequested(bool interruptSafe)
{
    assertExecutor();
    if (interruptSafe || m_insideInterrupt || !m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId != 0)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        enqueueReady(*self, true);
        m_currentThreadId = 0;
    }
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::createSemaphore(int initCount, int maxCount, uint32_t attr, uint32_t option)
{
    assertExecutor();
    if (maxCount <= 0 || initCount < 0 || initCount > maxCount)
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextSemaphoreId, m_semaphores);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeSemaphore semaphore{};
    semaphore.id = id;
    semaphore.count = initCount;
    semaphore.maxCount = maxCount;
    semaphore.initCount = initCount;
    semaphore.attr = attr;
    semaphore.option = option;
    m_semaphores.emplace(id, std::move(semaphore));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_semaphores.find(id);
    if (it == m_semaphores.end())
    {
        return KE_UNKNOWN_SEMID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_semaphores.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return id;
}

int EeScheduler::signalSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (!object->waiters.empty())
    {
        const int waiterId = object->waiters.front();
        object->waiters.pop_front();
        GuestThread *waiter = thread(waiterId);
        assert(waiter != nullptr);
        makeReady(*waiter, id, interruptSafe);
        publishSnapshot();
        return id;
    }
    if (object->count == object->maxCount)
    {
        return KE_SEMA_OVF;
    }
    ++object->count;
    publishSnapshot();
    return id;
}

int EeScheduler::pollSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (object->count == 0)
    {
        return KE_SEMA_ZERO;
    }
    --object->count;
    publishSnapshot();
    return id;
}

void EeScheduler::waitSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), KE_UNKNOWN_SEMID);
        return;
    }
    if (object->count != 0)
    {
        --object->count;
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), id);
        publishSnapshot();
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    object->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::Semaphore, EeSemaphoreWait{id}});
}

int EeScheduler::createEventFlag(uint32_t initialBits, uint32_t attr, uint32_t option)
{
    assertExecutor();
    const int id = allocatePositiveId(m_nextEventFlagId, m_eventFlags);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeEventFlag flag{};
    flag.id = id;
    flag.attr = attr;
    flag.option = option;
    flag.initBits = initialBits;
    flag.bits = initialBits;
    m_eventFlags.emplace(id, std::move(flag));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteEventFlag(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_eventFlags.find(id);
    if (it == m_eventFlags.end())
    {
        return KE_UNKNOWN_EVFID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_eventFlags.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::setEventFlag(int id, uint32_t bits, bool interruptSafe)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits |= bits;
    finishEventWaiters(*flag, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::clearEventFlag(int id, uint32_t mask)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits &= mask;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::pollEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t &observedBits)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    if (!eventCondition(flag->bits, bits, mode))
    {
        return KE_EVF_COND;
    }
    observedBits = flag->bits;
    if ((mode & WEF_CLEAR_ALL) != 0u)
    {
        flag->bits = 0;
    }
    else if ((mode & WEF_CLEAR) != 0u)
    {
        flag->bits &= ~bits;
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::waitEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t resultAddress)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (!flag)
    {
        setReturnS32(&self->activeContext(), KE_UNKNOWN_EVFID);
        return;
    }
    if (eventCondition(flag->bits, bits, mode))
    {
        const uint32_t observed = flag->bits;
        writeGuestU32(resultAddress, observed);
        if ((mode & WEF_CLEAR_ALL) != 0u)
        {
            flag->bits = 0;
        }
        else if ((mode & WEF_CLEAR) != 0u)
        {
            flag->bits &= ~bits;
        }
        setReturnS32(&self->activeContext(), KE_OK);
        publishSnapshot();
        return;
    }
    flag->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::EventFlag,
                             EeEventFlagWait{id, bits, mode, resultAddress}});
}

int EeScheduler::setAlarm(uint16_t ticks,
                          uint32_t handler,
                          uint32_t argument,
                          uint32_t gp,
                          uint32_t sp)
{
    assertExecutor();
    if (handler == 0u || !m_runtime.hasFunction(handler))
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextAlarmId, m_alarms);
    if (id == 0)
    {
        return KE_ERROR;
    }
    m_alarms.emplace(id, EeAlarm{id, ticks, handler, argument, gp, sp});
    const uint64_t tickCount = ticks == 0u ? 1u : static_cast<uint64_t>(ticks);
    scheduleEvent(m_eeCycle + tickCount * kAlarmTickCycles,
                  std::chrono::steady_clock::now() + std::chrono::microseconds(tickCount * kAlarmTickMicroseconds),
                  EeEvent{EeEventType::Alarm, static_cast<uint32_t>(id), 0});
    return id;
}

int EeScheduler::cancelAlarm(int id)
{
    assertExecutor();
    if (m_alarms.erase(id) == 0u)
    {
        return KE_ERROR;
    }
    {
        std::lock_guard lock(m_eventMutex);
        std::erase_if(m_deadlines, [id](const ScheduledEvent &scheduled)
                      { return scheduled.event.type == EeEventType::Alarm &&
                               scheduled.event.id == static_cast<uint32_t>(id); });
        updateNextDeadline();
    }
    return KE_OK;
}

// ---- Invocation coalescing (PS2X_INVOKE_COALESCE = the maximum number of IDENTICAL event-edge
// invocations (same kind + entry pc) allowed pending / stacked un-started; default 32; =8 was the
// cont.232 default, =1 coalesces at one, the original; =0 never coalesces -- and =0 still ABORTS
// with "EE invocation stack space exhausted" during the level load, which is why the cap exists).
// ★ cont.251: the cap is a CLOCK CAP -- any frame longer than it silently loses the guest's vblank
// ticks. Measured on LOTR (Helm's Deep, 300 s, [hero:dt] reported dt vs the real-time 50 Hz ladder):
// at cap=8 the reported dt tracks reality exactly to 10 vblanks and then SATURATES at 10.1 however
// long the frame really was (11,12,...,21 vblanks all report ~10.1), leaving the game's simulation
// clock at 0.844x real over the run and 0.59x through the heavy cutscene phase, whose frames run
// 10-16 vblanks. At cap=32 reported dt tracks actual 1:1 out to 16 and the clock is 1.000x. This is
// the SAME defect cont.232 found at cap=1 (16% of ticks lost, clock 0.84x) -- raising 1 -> 8 fixed
// gameplay frames (5-9 vblanks) but not heavy scenes, so the whole-run figure came back to 0.844x.
// Hardware/PCSX2 say the count should never be capped at all (see below); 32 covers a 640 ms frame,
// far worse than anything measured, while staying well under the depth-64 arena limit the guard is
// for. Validated on BOTH memory-card states: fresh card -> Helm's Deep, save present -> Pat01.
// Why a cap and not one: on the PS2 the vblank interrupt preempts the EE within cycles (PCSX2
// Counters.cpp VSyncStart -> hwIntcIrq(INTC_VBLANK_S) -> R5900 cpuTestINTCInts takes it at the
// next instruction), so a game whose frame takes N vblanks still runs its handler N times and its
// sceGsSyncVCallback counter advances by N. This EE cannot take an interrupt mid-function: the N
// vblank deadlines come due in ONE processDueDeadlines pass at the next checkpoint. Coalescing them
// at one delivered ONE callback per frame however long the frame took -- measured on LOTR (cont.232):
// 5 vblanks per frame counted as 4, 16% of the game's vsync ticks lost in the level, the game's
// clock (frame dt = counter delta x 1/50) running 0.84x wall. Delivering the backlog reproduces the
// hardware count; the cap keeps the guard this mechanism exists for: a single long guest dispatch
// (>1 s -- e.g. a host-pump chain with checkpoints suppressed) backlogs 60+ vblank deadlines whose
// invocations the run loop stacks before dispatching, and the per-depth invocation stacks exhaust
// the 1 MB async arena at depth 64 ("EE invocation stack space exhausted"). INTC itself latches ONE
// status bit per cause, so a genuinely stalled EE (interrupts masked) would see one interrupt, not
// 60 -- the cap models that stall too. Payload-carrying kinds are never coalesced.
static const int s_invokeCoalesceCap = []
{
    const char *e = std::getenv("PS2X_INVOKE_COALESCE");
    if (!e || !e[0])
    {
        return 32;
    }
    const int v = std::atoi(e);
    return v < 0 ? 32 : v;
}();

void EeScheduler::queueInvocation(GuestInvocation invocation)
{
    assertExecutor();
    if (s_invokeCoalesceCap > 0 &&
        (invocation.kind == GuestInvocationKind::GsCallback ||
         invocation.kind == GuestInvocationKind::Interrupt ||
         invocation.kind == GuestInvocationKind::Alarm))
    {
        const auto duplicate = [&invocation](const GuestInvocation &other)
        {
            return other.kind == invocation.kind && other.context.pc == invocation.context.pc;
        };
        int identical = 0;
        for (const GuestInvocation &pending : m_pendingInvocations)
        {
            if (duplicate(pending) && ++identical >= s_invokeCoalesceCap)
            {
                return;
            }
        }
        for (const auto &[threadId, thread] : m_threads)
        {
            for (const GuestInvocation &stacked : thread.invocations)
            {
                if (duplicate(stacked) && ++identical >= s_invokeCoalesceCap)
                {
                    return;
                }
            }
        }
    }
    invocation.sequence = ++m_invocationSequence;
    if (invocation.kind == GuestInvocationKind::SifCommand)
        ++m_pendingSifCommands;
    m_pendingInvocations.push_back(std::move(invocation));
    m_checkpointPending.store(true, std::memory_order_release);
}

bool EeScheduler::takePendingInvocation(GuestInvocationKind kind, GuestInvocation &out)
{
    assertExecutor();
    for (auto it = m_pendingInvocations.begin(); it != m_pendingInvocations.end(); ++it)
    {
        if (it->kind == kind)
        {
            out = std::move(*it);
            m_pendingInvocations.erase(it);
            if (kind == GuestInvocationKind::SifCommand && m_pendingSifCommands)
                --m_pendingSifCommands;
            return true;
        }
    }
    return false;
}

[[noreturn]] void EeScheduler::invokeCurrent(GuestInvocation invocation)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    if (getRegU32(&invocation.context, 29) == 0u)
    {
        SET_GPR_U32(&invocation.context, 29, invocationStackTop());
    }
    invocation.sequence = ++m_invocationSequence;
    owner->invocations.push_back(std::move(invocation));
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

[[noreturn]] void EeScheduler::invokeCurrentSequence(std::vector<GuestInvocation> invocations)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    assert(!invocations.empty());
    for (auto it = invocations.rbegin(); it != invocations.rend(); ++it)
    {
        if (getRegU32(&it->context, 29) == 0u)
        {
            SET_GPR_U32(&it->context, 29, invocationStackTop());
        }
        it->sequence = ++m_invocationSequence;
        owner->invocations.push_back(std::move(*it));
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

bool EeScheduler::hasInvocation(GuestInvocationKind kind, uint64_t tag) const
{
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        return false;
    }
    return std::any_of(owner->invocations.begin(), owner->invocations.end(),
                       [kind, tag](const GuestInvocation &invocation)
                       {
                           return invocation.kind == kind && invocation.tag == tag;
                       });
}

uint32_t EeScheduler::invocationStackTop()
{
    assertExecutor();
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        throw std::logic_error("EE invocation stack requested without a current guest context");
    }
    const size_t depth = owner ? owner->invocations.size() : 0u;
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(owner->id)) << 32u) |
                         static_cast<uint32_t>(depth);
    const auto existing = m_invocationStackTops.find(key);
    if (existing != m_invocationStackTops.end())
    {
        return existing->second;
    }
    constexpr uint32_t kInvocationStackSize = 0x4000u;
    const uint32_t top = m_runtime.reserveAsyncCallbackStack(kInvocationStackSize, 16u);
    if (top == 0u)
    {
        // Error-path diagnostic only: dump the runaway invocation stack so the piling kind is
        // identifiable from the crash log (kind/tag/pc per level, innermost last).
        std::fprintf(stderr,
                     "[ee-invoke:EXHAUSTED] thread=%d depth=%zu base-pc=0x%x — stacked invocations:\n",
                     owner->id, depth, owner->context.pc);
        for (size_t i = 0; i < owner->invocations.size(); ++i)
        {
            const GuestInvocation &inv = owner->invocations[i];
            std::fprintf(stderr, "  [%zu] kind=%d tag=0x%llx pc=0x%x ra=0x%x\n",
                         i, static_cast<int>(inv.kind),
                         static_cast<unsigned long long>(inv.tag),
                         inv.context.pc,
                         static_cast<uint32_t>(_mm_cvtsi128_si32(inv.context.r[31])));
        }
        throw std::runtime_error("EE invocation stack space exhausted");
    }
    m_invocationStackTops.emplace(key, top);
    return top;
}

int EeScheduler::addIrqHandler(bool dmac,
                               uint32_t cause,
                               uint32_t handler,
                               bool append,
                               uint32_t argument,
                               uint32_t gp,
                               uint32_t sp)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    int &nextId = dmac ? m_nextDmacHandlerId : m_nextIntcHandlerId;
    const int id = allocatePositiveId(nextId, handlers);
    if (id == 0)
    {
        return KE_ERROR;
    }
    int &head = dmac ? m_dmacHeadOrder : m_intcHeadOrder;
    int &tail = dmac ? m_dmacTailOrder : m_intcTailOrder;
    handlers.emplace(id,
                     EeIrqHandler{id,
                                  cause,
                                  handler,
                                  argument,
                                  gp,
                                  sp,
                                  true,
                                  append ? ++tail : --head});
    return id;
}

int EeScheduler::removeIrqHandler(bool dmac, uint32_t cause, int id)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end() && it->second.cause == cause)
    {
        handlers.erase(it);
    }
    return KE_OK;
}

int EeScheduler::setIrqHandlerEnabled(bool dmac, int id, bool enabled)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end())
    {
        it->second.enabled = enabled;
    }
    return KE_OK;
}

int EeScheduler::setIrqCauseEnabled(bool dmac, uint32_t cause, bool enabled)
{
    assertExecutor();
    if (cause < 32u)
    {
        uint32_t &mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
        if (enabled)
        {
            mask |= 1u << cause;
        }
        else
        {
            mask &= ~(1u << cause);
        }
    }
    return KE_OK;
}

void EeScheduler::dispatchIrq(bool dmac, uint32_t cause)
{
    assertExecutor();
    const uint32_t mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
    if (cause < 32u && (mask & (1u << cause)) == 0u)
    {
        return;
    }
    const auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    std::vector<EeIrqHandler> matching;
    for (const auto &[id, handler] : handlers)
    {
        (void)id;
        if (handler.enabled && handler.cause == cause && handler.handler != 0u &&
            m_runtime.hasFunction(handler.handler))
        {
            matching.push_back(handler);
        }
    }
    std::sort(matching.begin(), matching.end(), [](const EeIrqHandler &left, const EeIrqHandler &right)
              { return left.order < right.order; });
    for (const EeIrqHandler &handler : matching)
    {
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Interrupt;
        invocation.context.pc = handler.handler;
        SET_GPR_U32(&invocation.context, 4, cause);
        SET_GPR_U32(&invocation.context, 5, handler.argument);
        SET_GPR_U32(&invocation.context, 28, handler.gp);
        // ★ cont.248: INTC/DMAC handlers run on the per-thread invocation arena (sp 0 -> invocationStackTop()
        // at the push), like the vsync callback (setGsVSyncCallback discards the registered sp, upstream #184).
        // `handler.sp` is the REGISTERING thread's live $sp at the AddIntcHandler/AddDmacHandler syscall -- an
        // init-time position inside that thread's stack. Running a handler's frames downward from there lands
        // inside the interrupted thread's LIVE frames whenever it is deeper at the checkpoint, overwriting an
        // ancestor's saved ra/s-registers: the resumed context then returns into garbage -- the
        // "[guest-branch:missing-target] ... EE scheduler pc=0x3 ra=0x3 s1=0x3 sp=0x10fcd0" race (1 run in 3 at
        // 50 Hz, open since cont.232). The hardware kernel's exception dispatch saves the interrupted context and
        // runs INTC/DMAC handlers in its own interrupt context, never inside a thread's frames (PCSX2 runs that
        // kernel from the BIOS), so nothing a handler does may depend on the registering thread's stack.
        SET_GPR_U32(&invocation.context, 29, 0u);
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
    }
}

void EeScheduler::setVSyncFlag(uint32_t flagAddress, uint32_t tickAddress)
{
    assertExecutor();
    m_vsyncFlagAddress = flagAddress;
    m_vsyncTickAddress = tickAddress;
    writeGuestU32(flagAddress, 0u);
    if (tickAddress != 0u)
    {
        const uint32_t physical = tickAddress & 0x1FFFFFFFu;
        if (m_rdram && physical <= PS2_RAM_SIZE - sizeof(uint64_t))
        {
            const uint64_t zero = 0u;
            std::memcpy(m_rdram + physical, &zero, sizeof(zero));
        }
    }
}

uint64_t EeScheduler::currentVSyncTick() const noexcept
{
    return m_vsyncTick;
}

uint32_t EeScheduler::setGsVSyncCallback(uint32_t callback, uint32_t gp, uint32_t sp)
{
    assertExecutor();
    (void)sp;
    const uint32_t previous = m_gsVSyncCallback;
    m_gsVSyncCallback = callback;
    m_gsVSyncCallbackGp = gp;
    m_gsVSyncCallbackSp = 0u;
    return previous;
}

[[noreturn]] void EeScheduler::waitVSync(uint64_t afterTick, int fixedResult, std::function<void(R5900Context &)> completion)
{
    blockCurrent(EeWaitState{
        EeWaitReason::VSync,
        EeVSyncWait{afterTick, fixedResult},
        std::move(completion)});
}

void EeScheduler::completeVSync(uint64_t tick)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status == EeThreadStatus::Waiting || candidate.status == EeThreadStatus::WaitingSuspended) &&
            candidate.wait.reason == EeWaitReason::VSync &&
            std::get<EeVSyncWait>(candidate.wait.payload).afterTick < tick)
        {
            completed.push_back(id);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        const EeVSyncWait wait = std::get<EeVSyncWait>(waiter->wait.payload);
        const int result = wait.fixedResult >= 0
                               ? wait.fixedResult
                               : static_cast<int>((tick - 1u) & 1u);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

void EeScheduler::completeExternalWait(uint32_t type, uint64_t token, int result)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status != EeThreadStatus::Waiting && candidate.status != EeThreadStatus::WaitingSuspended) ||
            (candidate.wait.reason != EeWaitReason::External &&
             candidate.wait.reason != EeWaitReason::Mpeg))
        {
            continue;
        }
        const auto &external = std::get<EeExternalWait>(candidate.wait.payload);
        if (external.type == type && external.token == token)
        {
            completed.push_back(id);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

[[noreturn]] void EeScheduler::waitExternal(EeWaitReason reason,
                                            uint32_t type,
                                            uint64_t token,
                                            std::function<void(R5900Context &)> completion)
{
    EeWaitState wait{reason, EeExternalWait{type, token}, std::move(completion)};
    blockCurrent(std::move(wait));
}

GuestThread *EeScheduler::thread(int id)
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

const GuestThread *EeScheduler::thread(int id) const
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

EeSemaphore *EeScheduler::semaphore(int id)
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

const EeSemaphore *EeScheduler::semaphore(int id) const
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

EeEventFlag *EeScheduler::eventFlag(int id)
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

const EeEventFlag *EeScheduler::eventFlag(int id) const
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

GuestThread *EeScheduler::currentThread()
{
    return thread(m_currentThreadId);
}

const GuestThread *EeScheduler::currentThread() const
{
    return thread(m_currentThreadId);
}

int EeScheduler::currentThreadId() const noexcept
{
    return m_currentThreadId;
}

R5900Context *EeScheduler::currentContext()
{
    GuestThread *self = currentThread();
    return self ? &self->activeContext() : nullptr;
}

uint8_t *EeScheduler::rdram() const noexcept
{
    return m_rdram;
}

void EeScheduler::bindMainContextForSyscall(R5900Context &ctx, uint8_t *rdram)
{
    if (m_executorThread == std::thread::id{})
    {
        reset(rdram, ctx);
        GuestThread *main = selectReady();
        assert(main != nullptr);
        makeRunning(*main);
        return;
    }
    assertExecutor();
    m_rdram = rdram;
    if (m_currentThreadId == 0)
    {
        GuestThread *main = thread(kMainThreadId);
        assert(main != nullptr);
        assert(main->status == EeThreadStatus::Ready);
        removeReady(*main);
        makeRunning(*main);
    }
}

EeKernelSnapshot EeScheduler::snapshot() const
{
    std::lock_guard lock(m_snapshotMutex);
    return m_snapshot;
}

void EeScheduler::publishSnapshot()
{
    EeKernelSnapshot next{};
    next.sequence = ++m_snapshotSequence;
    next.eeCycle = m_eeCycle;
    next.sliceEndCycle = m_sliceEndCycle;
    next.nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    next.runningThreadId = m_currentThreadId;
    next.threads.reserve(m_threads.size());
    for (const auto &[id, item] : m_threads)
    {
        if (id < 0)
        {
            continue;
        }
        EeThreadSnapshot snapshot{};
        snapshot.id = id;
        const R5900Context &context = item.activeContext();
        snapshot.pc = context.pc;
        snapshot.ra = getRegU32(&context, 31);
        snapshot.sp = getRegU32(&context, 29);
        snapshot.contextGp = getRegU32(&context, 28);
        snapshot.entry = item.entry;
        snapshot.stack = item.stack;
        snapshot.stackSize = item.stackSize;
        snapshot.gp = item.gp;
        snapshot.initialPriority = item.initialPriority;
        snapshot.currentPriority = item.currentPriority;
        snapshot.status = item.status;
        snapshot.waitReason = item.wait.reason;
        snapshot.waitId = waitObjectId(item.wait);
        snapshot.suspendCount = item.suspendCount;
        snapshot.wakeupCount = item.wakeupCount;
        snapshot.invocationDepth = static_cast<uint32_t>(item.invocations.size());
        next.threads.push_back(snapshot);
    }
    std::sort(next.threads.begin(), next.threads.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.semaphores.reserve(m_semaphores.size());
    for (const auto &[id, item] : m_semaphores)
    {
        next.semaphores.push_back(EeSemaphoreSnapshot{id,
                                                      item.count,
                                                      item.maxCount,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.semaphores.begin(), next.semaphores.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.eventFlags.reserve(m_eventFlags.size());
    for (const auto &[id, item] : m_eventFlags)
    {
        next.eventFlags.push_back(EeEventFlagSnapshot{id,
                                                      item.bits,
                                                      item.initBits,
                                                      item.attr,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.eventFlags.begin(), next.eventFlags.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    {
        std::lock_guard lock(m_snapshotMutex);
        m_snapshot = std::move(next);
    }
}

void EeScheduler::assertExecutor() const
{
    assert(m_executorThread == std::this_thread::get_id());
}

int EeScheduler::allocateThreadId()
{
    for (int attempts = 0; attempts <= kLastThreadId - kFirstThreadId; ++attempts)
    {
        const int candidate = m_nextThreadId;
        m_nextThreadId = candidate == kLastThreadId ? kFirstThreadId : candidate + 1;
        if (!m_threads.contains(candidate))
        {
            return candidate;
        }
    }
    return 0;
}

GuestThread &EeScheduler::acquireInvocationThread()
{
    for (auto &[id, candidate] : m_threads)
    {
        if (id < 0 && candidate.status == EeThreadStatus::Dormant && candidate.invocations.empty())
        {
            return candidate;
        }
    }

    GuestThread dispatcher{};
    dispatcher.id = m_nextInvocationThreadId--;
    dispatcher.initialPriority = 0;
    dispatcher.currentPriority = 0;
    dispatcher.status = EeThreadStatus::Dormant;
    return m_threads.emplace(dispatcher.id, std::move(dispatcher)).first->second;
}

void EeScheduler::enqueueReady(GuestThread &item, bool front)
{
    assert(item.currentPriority >= 0 && item.currentPriority < kPriorityCount);
    item.status = EeThreadStatus::Ready;
    auto &queue = m_readyQueues[item.currentPriority];
    if (front)
    {
        queue.push_front(item.id);
    }
    else
    {
        queue.push_back(item.id);
    }
}

void EeScheduler::removeReady(GuestThread &item)
{
    if (item.status != EeThreadStatus::Ready)
    {
        return;
    }
    auto &queue = m_readyQueues[item.currentPriority];
    auto it = std::find(queue.begin(), queue.end(), item.id);
    assert(it != queue.end());
    queue.erase(it);
}

GuestThread *EeScheduler::selectReady()
{
    for (auto &queue : m_readyQueues)
    {
        if (queue.empty())
        {
            continue;
        }
        const int id = queue.front();
        queue.pop_front();
        GuestThread *selected = thread(id);
        assert(selected != nullptr);
        assert(selected->status == EeThreadStatus::Ready);
        return selected;
    }
    return nullptr;
}

void EeScheduler::makeRunning(GuestThread &item)
{
    assert(m_currentThreadId == 0);
    assert(item.status == EeThreadStatus::Ready);
    item.status = EeThreadStatus::Running;
    m_currentThreadId = item.id;
    renewTimeSlice();
}

void EeScheduler::makeDormant(GuestThread &item)
{
    removeReady(item);
    removeFromWaitObject(item);
    item.status = EeThreadStatus::Dormant;
    item.wait = {};
    item.resumeCompletion = {};
    item.suspendCount = 0;
    item.wakeupCount = 0;
    item.invocations.clear();
}

void EeScheduler::removeFromWaitObject(GuestThread &item)
{
    const int id = item.id;
    if (item.wait.reason == EeWaitReason::Semaphore)
    {
        const int objectId = std::get<EeSemaphoreWait>(item.wait.payload).id;
        if (EeSemaphore *object = semaphore(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    else if (item.wait.reason == EeWaitReason::EventFlag)
    {
        const int objectId = std::get<EeEventFlagWait>(item.wait.payload).id;
        if (EeEventFlag *object = eventFlag(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    item.wait = {};
}

void EeScheduler::blockCurrent(EeWaitState wait)
{
    GuestThread *self = currentThread();
    assert(self != nullptr);
    self->wait = std::move(wait);
    self->status = self->suspendCount == 0 ? EeThreadStatus::Waiting : EeThreadStatus::WaitingSuspended;
    m_currentThreadId = 0;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

void EeScheduler::makeReady(GuestThread &item, int result, bool interruptSafe)
{
    auto completion = std::move(item.wait.completion);
    item.wait = {};
    setReturnS32(&item.activeContext(), result);
    item.resumeCompletion = std::move(completion);
    if (item.suspendCount != 0)
    {
        item.status = EeThreadStatus::Suspended;
        return;
    }
    enqueueReady(item);
    requestPreemptionIfHigher(item, interruptSafe);
}

void EeScheduler::requestPreemptionIfHigher(const GuestThread &readyThread, bool interruptSafe)
{
    const GuestThread *running = currentThread();
    if (!running || readyThread.currentPriority >= running->currentPriority)
    {
        return;
    }
    m_rescheduleRequested = true;
    if (interruptSafe || m_insideInterrupt)
    {
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::applyPendingPreemption()
{
    if (!m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId == 0)
    {
        m_rescheduleRequested = false;
        m_timeSliceExpired = false;
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    enqueueReady(*self, !m_timeSliceExpired);
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
}

void EeScheduler::processPendingEvents()
{
    assertExecutor();
    processDueDeadlines();
    const uint32_t timerInterrupts = m_pendingEeTimerInterrupts;
    m_pendingEeTimerInterrupts = 0u;
    for (uint32_t timer = 0u; timer < 4u; ++timer)
    {
        if ((timerInterrupts & (1u << timer)) != 0u)
        {
            dispatchIrq(false, 9u + timer);
        }
    }
    std::deque<EeEvent> pending;
    {
        std::lock_guard lock(m_eventMutex);
        pending.swap(m_events);
    }
    for (const EeEvent &event : pending)
    {
        processEvent(event);
    }

    {
        std::lock_guard lock(m_eventMutex);
        const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
        const bool cycleEventDue = nextEventCycle != 0u && m_eeCycle >= nextEventCycle;
        const bool pendingWork = !m_events.empty() || cycleEventDue || m_stopRequested.load(std::memory_order_acquire);
        m_checkpointPending.store(pendingWork, std::memory_order_release);
    }
    applyPendingPreemption();
}

void EeScheduler::processDueDeadlines()
{
    for (;;)
    {
        std::vector<ScheduledEvent> due;
        std::chrono::steady_clock::time_point pacingDeadline{};
        {
            std::unique_lock lock(m_eventMutex);
            if (s_virtualTime)
            {
                // cycle gating only: everything whose cycle has arrived is due now, nothing waits on the host
                auto firstFutureV = std::partition(m_deadlines.begin(), m_deadlines.end(),
                                                   [this](const ScheduledEvent &item) { return item.deadlineCycle <= m_eeCycle; });
                if (firstFutureV == m_deadlines.begin()) { updateNextDeadline(); return; }
                due.insert(due.end(), std::make_move_iterator(m_deadlines.begin()), std::make_move_iterator(firstFutureV));
                m_deadlines.erase(m_deadlines.begin(), firstFutureV);
                updateNextDeadline();
            }
            else
            {
            const auto now = std::chrono::steady_clock::now();
            if (s_vblankWallclock)
            {
                // Wall-clock floor (see the block comment at the top of this file): a deadline
                // whose HOST time has arrived catches the under-credited EE clock up to its
                // cycle, so the pass below picks it up. Only the scheduler thread writes
                // m_eeCycle; we are that thread.
                for (const ScheduledEvent &item : m_deadlines)
                {
                    if (item.hostDeadline <= now && item.deadlineCycle > m_eeCycle)
                    {
                        m_eeCycle = item.deadlineCycle;
                    }
                }
            }
            for (const ScheduledEvent &item : m_deadlines)
            {
                if (item.deadlineCycle <= m_eeCycle &&
                    (pacingDeadline == std::chrono::steady_clock::time_point{} ||
                     item.hostDeadline < pacingDeadline))
                {
                    pacingDeadline = item.hostDeadline;
                }
            }

            if (pacingDeadline == std::chrono::steady_clock::time_point{})
            {
                updateNextDeadline();
                return;
            }

            if (now < pacingDeadline)
            {
                m_eventCv.wait_until(lock, pacingDeadline, [this]()
                                     { return !m_events.empty() ||
                                              m_stopRequested.load(std::memory_order_acquire); });
                if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
                {
                    updateNextDeadline();
                    return;
                }
            }

            const auto pacedNow = std::chrono::steady_clock::now();
            auto firstFuture = std::partition(m_deadlines.begin(), m_deadlines.end(),
                                              [this, pacedNow](const ScheduledEvent &item)
                                              { return item.deadlineCycle <= m_eeCycle &&
                                                       item.hostDeadline <= pacedNow; });
            due.insert(due.end(),
                       std::make_move_iterator(m_deadlines.begin()),
                       std::make_move_iterator(firstFuture));
            m_deadlines.erase(m_deadlines.begin(), firstFuture);
            updateNextDeadline();
            }
        }

        std::sort(due.begin(), due.end(), [](const ScheduledEvent &left, const ScheduledEvent &right)
                  {
                      if (left.deadlineCycle != right.deadlineCycle)
                      {
                          return left.deadlineCycle < right.deadlineCycle;
                      }
                      if (left.event.type != right.event.type)
                      {
                          return left.event.type < right.event.type;
                      }
                      if (left.event.id != right.event.id)
                      {
                          return left.event.id < right.event.id;
                      }
                      return left.sequence < right.sequence; });

        if (due.empty())
        {
            return;
        }

        for (ScheduledEvent &scheduled : due)
        {
            if (scheduled.event.type == EeEventType::VBlankStart)
            {
                refreshVblankFromSmode1(); // cont.232: the period follows the guest's CRTC mode
                scheduleEvent(scheduled.deadlineCycle + kVBlankDurationCycles,
                              scheduled.hostDeadline + kVBlankDuration,
                              EeEvent{EeEventType::VBlankEnd, 0, m_vsyncTick + 1u});
                scheduleEvent(scheduled.deadlineCycle + vblankPeriodCycles(),
                              scheduled.hostDeadline + vblankPeriod(),
                              EeEvent{EeEventType::VBlankStart, 0, 0});
            }
            processEvent(scheduled.event);
        }
    }
}

void EeScheduler::processEvent(const EeEvent &event)
{
    switch (event.type)
    {
    case EeEventType::Stop:
        requestStop();
        break;
    case EeEventType::VBlankStart:
        ++m_vsyncTick;
        m_runtime.syncIopToVblank(vblankPeriodCycles());   // rotk row 274: the IOP keeps console time
        m_runtime.memory().gs().vsyncTick.store(m_vsyncTick, std::memory_order_release);
        if ((m_vsyncTick & 1u) != 0u)
        {
            m_runtime.memory().gs().csr.fetch_or(0x2000ull, std::memory_order_acq_rel);
        }
        else
        {
            m_runtime.memory().gs().csr.fetch_and(~0x2000ull, std::memory_order_acq_rel);
        }
        writeGuestU32(m_vsyncFlagAddress, 1u);
        if (m_vsyncTickAddress != 0u)
        {
            const uint32_t physical = m_vsyncTickAddress & 0x1FFFFFFFu;
            if (m_rdram && physical <= PS2_RAM_SIZE - sizeof(uint64_t))
            {
                std::memcpy(m_rdram + physical, &m_vsyncTick, sizeof(m_vsyncTick));
            }
        }
        m_vsyncFlagAddress = 0u;
        m_vsyncTickAddress = 0u;
        // Per-vblank DBCMAN pad delivery (LOTR pad2/libdbc HLE). The frequency is load-bearing:
        // it re-asserts the per-port work table each tick, which the game wipes during its BSS
        // load; too-rare delivery reads as "controller removed" and the game self-exits.
        // (Re-homed here from the old Interrupt.cpp vblank worker, removed by the EE-scheduler
        // refactor #184.)
        ps2_dbcman_hle::deliverPadData(m_rdram, &m_runtime);
        completeVSync(m_vsyncTick);
        if (m_gsVSyncCallback != 0u && m_runtime.hasFunction(m_gsVSyncCallback))
        {
            GuestInvocation invocation{};
            invocation.kind = GuestInvocationKind::GsCallback;
            invocation.context.pc = m_gsVSyncCallback;
            SET_GPR_U32(&invocation.context, 4, static_cast<uint32_t>(m_vsyncTick));
            SET_GPR_U32(&invocation.context, 28, m_gsVSyncCallbackGp);
            SET_GPR_U32(&invocation.context, 29, m_gsVSyncCallbackSp);
            SET_GPR_U32(&invocation.context, 31, 0u);
            queueInvocation(std::move(invocation));
        }
        dispatchIrq(false, 2u);
        break;
    case EeEventType::ExternalWake:
        completeExternalWait(event.id, event.value, KE_OK);
        break;
    case EeEventType::VBlankEnd:
        dispatchIrq(false, 3u);
        break;
    case EeEventType::Dmac:
        break;
    case EeEventType::Alarm:
    {
        auto it = m_alarms.find(static_cast<int>(event.id));
        if (it == m_alarms.end())
        {
            break;
        }
        const EeAlarm alarm = it->second;
        m_alarms.erase(it);
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Alarm;
        invocation.context.pc = alarm.handler;
        SET_GPR_U32(&invocation.context, 4, static_cast<uint32_t>(alarm.id));
        SET_GPR_U32(&invocation.context, 5, static_cast<uint32_t>(alarm.ticks));
        SET_GPR_U32(&invocation.context, 6, alarm.argument);
        SET_GPR_U32(&invocation.context, 28, alarm.gp);
        SET_GPR_U32(&invocation.context, 29, 0u);   // cont.248: the arena, not the registering thread's $sp (see dispatchIrq)
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
        break;
    }
    }
}

void EeScheduler::finishEventWaiters(EeEventFlag &flag, bool interruptSafe)
{
    for (auto it = flag.waiters.begin(); it != flag.waiters.end();)
    {
        GuestThread *waiter = thread(*it);
        assert(waiter != nullptr);
        const EeEventFlagWait wait = std::get<EeEventFlagWait>(waiter->wait.payload);
        if (!eventCondition(flag.bits, wait.bits, wait.mode))
        {
            ++it;
            continue;
        }
        const uint32_t observed = flag.bits;
        writeGuestU32(wait.resultAddress, observed);
        if ((wait.mode & WEF_CLEAR_ALL) != 0u)
        {
            flag.bits = 0;
        }
        else if ((wait.mode & WEF_CLEAR) != 0u)
        {
            flag.bits &= ~wait.bits;
        }
        it = flag.waiters.erase(it);
        makeReady(*waiter, KE_OK, interruptSafe);
    }
}

bool EeScheduler::eventCondition(uint32_t current, uint32_t requested, uint32_t mode)
{
    return (mode & WEF_OR) != 0u ? (current & requested) != 0u
                                 : (current & requested) == requested;
}

int EeScheduler::waitObjectId(const EeWaitState &wait)
{
    switch (wait.reason)
    {
    case EeWaitReason::Semaphore:
        return std::get<EeSemaphoreWait>(wait.payload).id;
    case EeWaitReason::EventFlag:
        return std::get<EeEventFlagWait>(wait.payload).id;
    default:
        return 0;
    }
}

void EeScheduler::writeGuestU32(uint32_t address, uint32_t value)
{
    if (address == 0u)
    {
        return;
    }
    const uint32_t physical = address & 0x1FFFFFFFu;
    if (!m_rdram || physical > PS2_RAM_SIZE - sizeof(value))
    {
        return;
    }
    std::memcpy(m_rdram + physical, &value, sizeof(value));
}

void EeScheduler::waitForEvent()
{
    std::unique_lock lock(m_eventMutex);
    if (s_schedPace)
    {
        if (!g_pace.started)
        {
            g_pace.started = true;
            g_pace.t0 = std::chrono::steady_clock::now();
            g_pace.lastReport = g_pace.t0;
            g_pace.tick0 = m_vsyncTick;
            std::fprintf(stderr, "[sched:pace] active (PS2X_SCHED_PACE=%d)\n", s_schedPace);
        }
        ++g_pace.calls;
    }
    if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
    {
        if (s_schedPace)
        {
            ++g_pace.early;
        }
        return;
    }
    const uint64_t timerCycles = m_runtime.memory().cyclesUntilNextEeTimerInterrupt();
    const bool hasTimerDeadline = timerCycles != std::numeric_limits<uint64_t>::max();
    if (m_deadlines.empty() && !hasTimerDeadline)
    {
        // No deadline of any kind: this is a STALL, not pacing -- nothing in the emulator is
        // scheduled to wake the guest, so only an external post can. Timed separately.
        const auto stallT0 = s_schedPace ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
        m_eventCv.wait(lock, [this]()
                       { return !m_events.empty() || m_stopRequested.load(std::memory_order_acquire); });
        if (s_schedPace)
        {
            ++g_pace.stall;
            g_pace.stallNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - stallT0).count());
        }
        return;
    }

    uint64_t deadlineCycle = 0u;
    auto hostDeadline = std::chrono::steady_clock::time_point::max();
    EeEventType paceSource = EeEventType::Stop;
    bool paceHaveSource = false;
    if (!m_deadlines.empty())
    {
        const auto next = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                           [](const ScheduledEvent &left, const ScheduledEvent &right)
                                           {
                                               if (left.deadlineCycle != right.deadlineCycle)
                                               {
                                                   return left.deadlineCycle < right.deadlineCycle;
                                               }
                                               return left.sequence < right.sequence;
                                           });
        deadlineCycle = next->deadlineCycle;
        hostDeadline = next->hostDeadline;
        paceSource = next->event.type;
        paceHaveSource = true;
    }
    bool paceTimerWon = false;
    if (hasTimerDeadline)
    {
        const auto timerHostDeadline = std::chrono::steady_clock::now() + eeCyclesToHostDuration(timerCycles);
        const bool timerWins = s_virtualTime ? (!paceHaveSource || m_eeCycle + timerCycles < deadlineCycle)
                                             : (timerHostDeadline < hostDeadline);
        if (timerWins)
        {
            deadlineCycle = m_eeCycle + timerCycles;
            hostDeadline = timerHostDeadline;
            paceTimerWon = true;
        }
    }
    // cont.322e virtual time: with a cycle deadline ahead, do not sleep -- return at once and let the
    // catch-up below advance the cycle clock to it (a pending event still wins, as before). With no
    // deadline at all the wait is unchanged: only another thread's event can wake us.
    if (s_virtualTime && (paceHaveSource || paceTimerWon))
        hostDeadline = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point paceT0{};
    if (s_schedPace)
    {
        ++g_pace.timed;
        if (paceTimerWon)
        {
            ++g_pace.srcTimer;
        }
        else if (paceHaveSource)
        {
            switch (paceSource)
            {
            case EeEventType::VBlankStart: ++g_pace.srcVBlankStart; break;
            case EeEventType::VBlankEnd: ++g_pace.srcVBlankEnd; break;
            case EeEventType::Alarm: ++g_pace.srcAlarm; break;
            default: ++g_pace.srcOther; break;
            }
        }
        paceT0 = std::chrono::steady_clock::now();
        const auto reqNs = std::chrono::duration_cast<std::chrono::nanoseconds>(hostDeadline - paceT0).count();
        if (reqNs <= 0)
        {
            ++g_pace.overdue;
        }
        else
        {
            g_pace.reqNs += static_cast<unsigned long long>(reqNs);
            if (static_cast<unsigned long long>(reqNs) >= kWhyThresholdNs)
            {
                // Why is nothing runnable? Census the guest threads' wait reasons. Only on long
                // waits, so the common overdue/short path pays nothing.
                ++g_pace.whyWaits;
                for (const auto &entry : m_threads)
                {
                    const GuestThread &th = entry.second;
                    switch (th.status)
                    {
                    case EeThreadStatus::Running:
                    case EeThreadStatus::Ready:
                        ++g_pace.whyRunnableIsh;
                        break;
                    case EeThreadStatus::Suspended:
                    case EeThreadStatus::WaitingSuspended:
                        ++g_pace.whySuspended;
                        break;
                    case EeThreadStatus::Waiting:
                    {
                        const size_t r = static_cast<size_t>(th.wait.reason);
                        if (r < g_pace.whyReason.size())
                        {
                            ++g_pace.whyReason[r];
                        }
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
        }
    }

    const bool signaled = m_eventCv.wait_until(lock, hostDeadline, [this]()
                                               { return !m_events.empty() ||
                                                        m_stopRequested.load(std::memory_order_acquire); });
    if (s_schedPace)
    {
        const auto sleptNs = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - paceT0).count());
        g_pace.sleptNs += sleptNs;
        ++g_pace.bucket[schedPaceBucket(sleptNs)];
        if (signaled)
        {
            ++g_pace.signaled;
        }
        else
        {
            ++g_pace.timedout;
            g_pace.catchupCycles += deadlineCycle > m_eeCycle ? deadlineCycle - m_eeCycle : 0u;
        }
        if (s_schedPace >= 2 && sleptNs >= kWhyThresholdNs)
        {
            std::fprintf(stderr, "[sched:wait] slept=%.3fms src=%s signaled=%d deadlineCycle=%llu eeCycle=%llu tick=%llu\n",
                         double(sleptNs) / 1e6,
                         paceTimerWon ? "eetimer" : (paceHaveSource ? (paceSource == EeEventType::VBlankStart ? "vbstart"
                                                                      : paceSource == EeEventType::VBlankEnd ? "vbend"
                                                                      : paceSource == EeEventType::Alarm ? "alarm"
                                                                                                         : "other")
                                                                   : "none"),
                         signaled ? 1 : 0, (unsigned long long)deadlineCycle, (unsigned long long)m_eeCycle,
                         (unsigned long long)m_vsyncTick);
        }
        const auto paceNow = std::chrono::steady_clock::now();
        if (paceNow - g_pace.lastReport >= std::chrono::seconds(g_paceEverySec))
        {
            g_pace.lastReport = paceNow;
            schedPaceReport(m_vsyncTick);
        }
    }
    if (!signaled)
    {
        const uint64_t elapsed = deadlineCycle > m_eeCycle ? deadlineCycle - m_eeCycle : 0u;
        lock.unlock();
        uint64_t remaining = elapsed;
        while (remaining > 0u)
        {
            const uint32_t step = static_cast<uint32_t>(std::min<uint64_t>(remaining, std::numeric_limits<uint32_t>::max()));
            accountCycles(step);
            remaining -= step;
        }
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::scheduleEvent(uint64_t deadlineCycle,
                                std::chrono::steady_clock::time_point hostDeadline,
                                EeEvent event)
{
    {
        std::lock_guard lock(m_eventMutex);
        m_deadlines.push_back(ScheduledEvent{deadlineCycle, hostDeadline, event, ++m_eventSequence});
        updateNextDeadline();
    }
    m_eventCv.notify_one();
}

void EeScheduler::updateNextDeadline()
{
    if (m_deadlines.empty())
    {
        m_nextDeadlineCycle.store(0u, std::memory_order_release);
        return;
    }
    const auto it = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                     [](const ScheduledEvent &left, const ScheduledEvent &right)
                                     {
                                         if (left.deadlineCycle != right.deadlineCycle)
                                         {
                                             return left.deadlineCycle < right.deadlineCycle;
                                         }
                                         return left.sequence < right.sequence;
                                     });
    m_nextDeadlineCycle.store(it->deadlineCycle, std::memory_order_release);
}

bool EeScheduler::hasReadyAtOrAbovePriority(int priority) const
{
    const int last = std::clamp(priority, 0, kPriorityCount - 1);
    for (int p = 0; p <= last; ++p)
    {
        if (!m_readyQueues[static_cast<size_t>(p)].empty())
        {
            return true;
        }
    }
    return false;
}

void EeScheduler::renewTimeSlice()
{
    m_sliceEndCycle = m_eeCycle + kDefaultTimeSliceCycles;
    m_timeSliceExpired = false;
}

void EeScheduler::copyMainContextToRuntime()
{
    const GuestThread *main = thread(kMainThreadId);
    if (main)
    {
        m_runtime.m_cpuContext = main->context;
    }
}

void EeScheduler::publishDebugContext(const R5900Context &context)
{
    m_runtime.m_debugPc.store(context.pc, std::memory_order_relaxed);
    m_runtime.m_debugRa.store(getRegU32(&context, 31), std::memory_order_relaxed);
    m_runtime.m_debugSp.store(getRegU32(&context, 29), std::memory_order_relaxed);
    m_runtime.m_debugGp.store(getRegU32(&context, 28), std::memory_order_relaxed);
}

void EeScheduler::publishIdleDebugContext()
{
    // Temporary IRQ/RPC/alarm invocations deliberately return to PC=0. Once
    // the scheduler is idle, show a real EE thread context instead of leaving
    // the debugger pinned to that completed dispatcher frame.
    const GuestThread *selected = thread(kMainThreadId);
    if (!selected)
    {
        for (const auto &[id, candidate] : m_threads)
        {
            if (id > 0 && candidate.status != EeThreadStatus::Dormant)
            {
                selected = &candidate;
                break;
            }
        }
    }

    if (selected)
    {
        publishDebugContext(selected->activeContext());
    }
}
