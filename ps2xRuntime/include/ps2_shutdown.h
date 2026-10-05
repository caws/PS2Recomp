#ifndef PS2_SHUTDOWN_H
#define PS2_SHUTDOWN_H

// Bounded shutdown for the runner (2026-10-02). Before this, two exits could hang for ever:
//  - a stop request (window close, or SIGTERM -- SDL turns SIGTERM/SIGINT into a quit event) ended the
//    present loop, which then JOINED the EE thread; an EE thread that never reached its stop check
//    (guest state already corrupt) blocked the join, so `timeout` could not end the process;
//  - an unrecoverable missing branch target was logged once and the guest kept running on corrupt state
//    (the default policy resumed the caller), growing in memory until the machine locked up.
// Now every stop arms a watchdog: if the process has not exited `PS2X_SHUTDOWN_GRACE_MS` (default 5000)
// after the stop, it flushes the logs and _Exit()s with the recorded exit code.
//
// Exit codes: 0 = normal, 3 = unrecoverable missing branch target, 128+N = termination signal N.
//
// Header-only and kept out of ps2_runtime.h on purpose: ps2_runtime.h is included by every generated
// unit, so a change there rebuilds the whole game.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace ps2x::shutdown
{
    constexpr int kExitMissingTarget = 3;

    inline std::atomic<int> &exitCodeSlot()
    {
        static std::atomic<int> s_code{0};
        return s_code;
    }

    // The process exit code. The FIRST nonzero cause wins; a later one (the signal that arrives while a
    // fatal stop is already draining) does not overwrite it.
    inline int exitCode() { return exitCodeSlot().load(std::memory_order_acquire); }
    inline void setExitCode(int code)
    {
        int expected = 0;
        exitCodeSlot().compare_exchange_strong(expected, code, std::memory_order_acq_rel);
    }

    // Set by the runner's main(): an unrecoverable missing branch target ends the process (exit 3).
    // Off by default so the unit tests, which drive PS2Runtime directly, keep the old Stop semantics.
    inline std::atomic<bool> &fatalMissingTargetSlot()
    {
        static std::atomic<bool> s_fatal{false};
        return s_fatal;
    }
    inline bool fatalMissingTarget() { return fatalMissingTargetSlot().load(std::memory_order_acquire); }
    inline void setFatalMissingTarget(bool on) { fatalMissingTargetSlot().store(on, std::memory_order_release); }

    // PS2X_SHUTDOWN_GRACE_MS (default 5000): how long a requested stop may take before the watchdog
    // forces the exit. 0 = exit at once.
    inline int graceMs()
    {
        static const int s_ms = []
        {
            const char *e = std::getenv("PS2X_SHUTDOWN_GRACE_MS");
            if (!e || !e[0])
                return 5000;
            const long v = std::strtol(e, nullptr, 10);
            return v < 0 ? 5000 : static_cast<int>(v);
        }();
        return s_ms;
    }

    // Arm the exit watchdog (once per process; later calls are no-ops). Thread-safe; NOT for use inside
    // a signal handler.
    inline void armWatchdog(const char *reason)
    {
        static std::atomic<bool> s_armed{false};
        if (s_armed.exchange(true, std::memory_order_acq_rel))
            return;
        const int ms = graceMs();
        std::fprintf(stderr, "[shutdown] stop requested (%s); forcing exit in %d ms if still running\n",
                     reason ? reason : "?", ms);
        std::thread([ms, reason]
                    {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            const int code = exitCode();
            std::fprintf(stderr, "[shutdown] still running %d ms after the stop (%s) -- forcing exit %d\n",
                         ms, reason ? reason : "?", code);
            std::cout.flush();
            std::cerr.flush();
            std::fflush(nullptr);
            std::_Exit(code); })
            .detach();
    }
}

#endif
