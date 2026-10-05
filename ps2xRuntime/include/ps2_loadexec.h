#ifndef PS2_LOADEXEC_H
#define PS2_LOADEXEC_H

// LoadExecPS2 / ExecPS2 (EE syscalls 6 / 7) for a static recompilation (row 248, 2026-10-03).
//
// On the console the kernel resets the EE, loads the named ELF and starts it with the given arguments. A static
// recomp contains ONE program, so the runner can only (a) start itself again with new arguments, or (b) let the
// game decide what an exec of some other program means here (rotk USA: the online lobby ONLINE.ELF -> the game's
// own co-op page in the launcher). PCSX2 runs the real kernel's LoadExecPS2 (R5900OpcodeImpl.cpp only observes
// syscall 7 for its debugger), so there is no HLE to mirror; the argument convention follows the kernel's: the
// arguments AFTER argv[0], argv[0] = the path.
//
//   - a handler the game registered (static init) runs first; it may restart() or simply return false;
//   - default: a path naming the running ELF restarts the process with PS2X_GUEST_ARGS = the new arguments
//     (row 246 delivers them to the guest) and the launcher skipped;
//   - anything else: logged, and the runner stops cleanly instead of spinning in the guest's exec loop.
//
// Header-only and kept out of ps2_runtime.h (every generated unit includes that one).

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace ps2x::loadexec
{
    // (path as the guest wrote it, e.g. "cdrom0:\\ONLINEFE\\ONLINE.ELF;1", the arguments after argv[0]).
    // Return true when handled (normally by calling restart(), which does not return on success).
    using Handler = bool (*)(const std::string &path, const std::vector<std::string> &args);

    inline Handler &handlerSlot() { static Handler s_h = nullptr; return s_h; }
    inline bool setHandler(Handler h) { handlerSlot() = h; return true; }   // bool: usable in a static initializer

    inline char **&hostArgvSlot() { static char **s_argv = nullptr; return s_argv; }
    inline void setHostArgv(char **argv) { hostArgvSlot() = argv; }        // main() calls this

    // "cdrom0:\\DIR\\NAME.ELF;1" -> "NAME.ELF"
    inline std::string baseName(const std::string &path)
    {
        std::string s = path;
        if (const size_t semi = s.rfind(';'); semi != std::string::npos) s.resize(semi);
        if (const size_t sep = s.find_last_of("\\/:"); sep != std::string::npos) s = s.substr(sep + 1);
        return s;
    }

    inline std::string joinArgs(const std::vector<std::string> &args)
    {
        std::string out;
        for (const std::string &a : args) { if (!out.empty()) out.push_back(' '); out += a; }
        return out;
    }

    // Start this executable again with the same host arguments and the given environment changes ({name, value};
    // an empty value unsets). Returns only on failure.
    inline bool restart(const std::vector<std::pair<std::string, std::string>> &env)
    {
        char **argv = hostArgvSlot();
        if (!argv) { std::fprintf(stderr, "[loadexec] no host argv recorded; cannot restart\n"); return false; }
        for (const auto &kv : env)
        {
#if defined(_WIN32)
            _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
            if (kv.second.empty()) unsetenv(kv.first.c_str()); else setenv(kv.first.c_str(), kv.second.c_str(), 1);
#endif
        }
        std::fflush(stdout);
        std::fflush(stderr);
#if defined(__linux__)
        std::error_code ec;   // exec the resolved path so the process keeps the game's name (as ps2_settings.cpp does)
        const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
        execv(!ec && !self.empty() ? self.c_str() : "/proc/self/exe", argv);
        std::fprintf(stderr, "[loadexec] restart failed: %s\n", std::strerror(errno));
#else
        std::fprintf(stderr, "[loadexec] restart is not implemented on this platform\n");
#endif
        return false;
    }

    // Restart into the game itself with new guest arguments (launcher skipped, settings re-applied).
    inline bool restartGuest(const std::vector<std::string> &args)
    {
        return restart({{"PS2X_GUEST_ARGS", joinArgs(args)}, {"PS2X_LAUNCHER", "0"}, {"PS2X_SETTINGS_APPLIED", ""}});
    }
}

#endif
