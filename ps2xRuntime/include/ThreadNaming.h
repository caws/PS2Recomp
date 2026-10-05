#pragma once

#include <string_view>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WINAPI
#define WINAPI __stdcall
#endif

extern "C"
{
    typedef void *HANDLE;
    typedef void *HMODULE;
    typedef const wchar_t *PCWSTR;
    typedef long HRESULT;

    __declspec(dllimport) HMODULE WINAPI GetModuleHandleW(const wchar_t *lpModuleName);
    __declspec(dllimport) void *WINAPI GetProcAddress(HMODULE hModule, const char *lpProcName);
    __declspec(dllimport) HANDLE WINAPI GetCurrentThread(void);
}
#elif defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <vector>
#endif

namespace ThreadNaming
{
    inline void SetCurrentThreadName(std::string_view name)
    {
#if defined(_WIN32) 
        using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);

        HMODULE kernel32 = ::GetModuleHandleW(L"Kernel32.dll");
        auto setThreadDescription =
            reinterpret_cast<SetThreadDescriptionFn>(::GetProcAddress(kernel32, "SetThreadDescription"));

        if (setThreadDescription)
        {
            std::wstring wname(name.begin(), name.end());
            setThreadDescription(::GetCurrentThread(), wname.c_str());
        }

#elif defined(__APPLE__)
        pthread_setname_np(name.data());

#elif defined(__linux__)
        pthread_setname_np(pthread_self(), name.data());
#endif
    }

    // ---- cont.230 CPU pinning ------------------------------------------------------------------
    // WHY: the guest EE thread is the long pole (96% busy, VU1 inside it) and on an SMT host the OS
    // scheduler puts a ~45%-busy raster band thread on its sibling, so it runs at a fraction of a
    // dedicated core's speed and with less turbo headroom. Measured on a 4-core/8-thread i7-1165G7
    // at matched guest work with ALTERNATING on/off runs (build 378, level era): VU1 50-54 ns/pair
    // pinned vs 55-56 unpinned (~7%), wall 2-5% less for the same guest frames. (A first, non-
    // alternating comparison read +38%; that was host clock drift between runs -- the laptop's
    // powersave/balance_power clock moves run to run -- so only alternating pairs are trusted.)
    // PCSX2 has no equivalent default; this is a host-scheduling decision, not emulation semantics.
    //
    // Resolution, per role (EE = guest thread, GS = raster worker + band threads):
    //   1. explicit list in PS2X_EE_CPUS / PS2X_GS_CPUS ("0", "1,2,3", "1-3,5-7") always wins;
    //   2. PS2X_CPU_PIN=0 disables the automatic plan (explicit lists still apply);
    //   3. otherwise (default "auto") the plan below, derived from sysfs topology: the EE thread on
    //      ONE logical CPU of the lowest physical core, the raster pool on EVERY OTHER logical CPU
    //      -- including the EE's SMT sibling since cont.317. Needs >= 3 physical cores (pinning the
    //      EE alone on a 2-core box would starve the raster); else no pinning.
    // ★★ cont.317: the premise above ("the guest EE thread is the long pole, 96% busy") is no longer
    //    true -- perf + schedstat on the Helm's Deep fight (build 726) had three raster workers at
    //    90-92% and the EE at 30-38%: the RASTER is the pole. So the pool now takes every logical
    //    CPU but the EE's own (7 of 8 here) and gsCpus carries that count for the raster default:
    //    the fight capture benches 335 -> 268 ms (-20%, three alternating pairs, non-overlapping)
    //    with 7 threads on that pool vs 3 on the old one. The EE loses ~7% to its sibling
    //    (cont.230's number) out of a 60% idle budget.
    // Linux only; a no-op elsewhere.
    struct CpuPlan
    {
        bool valid = false;
        unsigned physicalCores = 0; // total physical cores seen
        std::string ee;             // CPU list for the EE thread
        std::string gs;             // CPU list for the raster pool
        unsigned gsCpus = 0;        // cont.317: how many logical CPUs are in `gs` (the raster default)
    };

    inline const CpuPlan &AutoCpuPlan()
    {
        static const CpuPlan plan = []
        {
            CpuPlan p;
#if defined(__linux__)
            std::map<long, std::vector<int>> cores; // core_id -> logical cpus (ascending)
            for (int cpu = 0; cpu < 1024; ++cpu)
            {
                std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id");
                if (!f)
                    break;
                long core = -1;
                f >> core;
                if (core < 0)
                    break;
                cores[core].push_back(cpu);
            }
            p.physicalCores = static_cast<unsigned>(cores.size());
            if (cores.size() < 3u)
                return p;
            // Lowest physical core = the one containing the lowest logical cpu.
            auto lowest = cores.begin();
            for (auto it = cores.begin(); it != cores.end(); ++it)
                if (it->second.front() < lowest->second.front())
                    lowest = it;
            const int eeCpu = lowest->second.front();
            p.ee = std::to_string(eeCpu);
            for (auto it = cores.begin(); it != cores.end(); ++it)
            {
                for (int cpu : it->second)
                {
                    if (cpu == eeCpu)
                        continue; // cont.317: the EE's sibling joins the pool; only the EE's own CPU is reserved
                    p.gs += (p.gs.empty() ? "" : ",") + std::to_string(cpu);
                    ++p.gsCpus;
                }
            }
            p.valid = !p.gs.empty();
#endif
            return p;
        }();
        return plan;
    }

    inline bool AutoCpuPinEnabled()
    {
        static const bool on = []
        { const char *e = std::getenv("PS2X_CPU_PIN"); return !(e && e[0] == '0'); }();
        return on;
    }

    // Pin the calling thread to a CPU list ("0", "1,2,3", "1-3,5-7"). Returns false when nothing
    // was pinned (empty list, parse failure, or not Linux).
    inline bool PinCurrentThreadToList(const char *list, const char *what)
    {
#if defined(__linux__)
        if (!list || !list[0])
            return false;
        cpu_set_t set;
        CPU_ZERO(&set);
        int count = 0;
        const char *q = list;
        while (*q)
        {
            char *end = nullptr;
            long a = std::strtol(q, &end, 10);
            if (end == q) break;
            long b = a;
            if (*end == '-') { const char *r = end + 1; b = std::strtol(r, &end, 10); if (end == r) break; }
            for (long c = a; c <= b && c < CPU_SETSIZE; ++c)
                if (c >= 0) { CPU_SET(static_cast<int>(c), &set); ++count; }
            q = (*end == ',') ? end + 1 : end;
        }
        if (count == 0)
            return false;
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        {
            std::fprintf(stderr, "[affinity] %s=%s: pthread_setaffinity_np failed\n", what, list);
            return false;
        }
        return true;
#else
        (void)list; (void)what;
        return false;
#endif
    }

    enum class CpuRole { EE, GS };

    // Pin the calling thread for its role -- explicit env list, else the automatic plan.
    inline void PinCurrentThreadForRole(CpuRole role)
    {
        const char *envName = role == CpuRole::EE ? "PS2X_EE_CPUS" : "PS2X_GS_CPUS";
        const char *e = std::getenv(envName);
        if (e && e[0])
        {
            PinCurrentThreadToList(e, envName);
            return;
        }
        if (!AutoCpuPinEnabled())
            return;
        const CpuPlan &plan = AutoCpuPlan();
        if (!plan.valid)
            return;
        PinCurrentThreadToList(role == CpuRole::EE ? plan.ee.c_str() : plan.gs.c_str(),
                               role == CpuRole::EE ? "auto EE" : "auto GS");
    }

    // ★ cont.318 PS2X_GS_PIN1TO1 (default OFF): pin raster participant `index` to ONE logical CPU of
    // its role's list (the index-th entry, wrapping) instead of the whole set. WHY: the pool is a
    // shared affinity SET, so the OS may stack two of the seven raster threads on one core and leave
    // another CPU idle (cont.317's bimodal T=4 bench); every run then ends at the barrier waiting for
    // the stacked pair. Ownership of rows is unchanged, so this is a scheduling knob only.
    inline bool OneToOnePinEnabled()
    {
        static const bool on = []
        { const char *e = std::getenv("PS2X_GS_PIN1TO1"); return e && e[0] && e[0] != '0'; }();
        return on;
    }
    inline void PinCurrentThreadForRoleIndex(CpuRole role, unsigned index)
    {
        if (!OneToOnePinEnabled())
        {
            PinCurrentThreadForRole(role);
            return;
        }
        const char *envName = role == CpuRole::EE ? "PS2X_EE_CPUS" : "PS2X_GS_CPUS";
        const char *e = std::getenv(envName);
        std::string list;
        if (e && e[0])
            list = e;
        else if (AutoCpuPinEnabled() && AutoCpuPlan().valid)
            list = role == CpuRole::EE ? AutoCpuPlan().ee : AutoCpuPlan().gs;
        if (list.empty())
            return;
        std::vector<int> cpus;
        const char *q = list.c_str();
        while (*q)
        {
            char *end = nullptr;
            long a = std::strtol(q, &end, 10);
            if (end == q) break;
            long b = a;
            if (*end == '-') { const char *r = end + 1; b = std::strtol(r, &end, 10); if (end == r) break; }
            for (long c = a; c <= b; ++c)
                if (c >= 0) cpus.push_back(static_cast<int>(c));
            q = (*end == ',') ? end + 1 : end;
        }
        if (cpus.empty())
            return;
        const std::string one = std::to_string(cpus[index % cpus.size()]);
        PinCurrentThreadToList(one.c_str(), role == CpuRole::EE ? "1to1 EE" : "1to1 GS");
    }

    // Kept for callers that pass an env name directly (same resolution as PinCurrentThreadForRole).
    inline void PinCurrentThreadFromEnv(const char *envName)
    {
        PinCurrentThreadForRole(std::string_view(envName) == "PS2X_GS_CPUS" ? CpuRole::GS : CpuRole::EE);
    }
}
