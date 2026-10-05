#include "ps2_runtime.h"
#include "ps2_settings.h"
#include "ps2_launcher.h"
#include "ps2_user_dir.h"
#include "games_database.h"
#include "ps2_shutdown.h"
#include "ps2_loadexec.h"
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
#include "ps2_debug_panel.h"
#endif

#ifdef _DEBUG
#include "ps2_log.h"
#endif

#include <iostream>
#include <string>
#include <filesystem>
#include <exception>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <fstream>

#if !defined(_WIN32) && !defined(PLATFORM_VITA)
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <thread>
#include <unistd.h>
#define PS2X_POSIX_TERM_SIGNALS 1
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#include <unistd.h>
#include <thread>
#include <cstdio>
#include <cstring>
#endif

namespace
{
#if defined(PS2X_POSIX_TERM_SIGNALS)
    // 2026-10-02: SIGTERM/SIGINT end the runner. SDL's own handler only queued a quit event, and the
    // run loop's EE-thread join could then wait for ever (a stuck guest), so `timeout` never ended it.
    // The handler only records the signal (async-signal-safe); a watcher thread requests the stop and
    // arms the exit watchdog (ps2_shutdown.h). A second signal at least 1 s after the first exits at once;
    // an earlier repeat is the same request (GNU `timeout` signals the child AND its process group).
    std::atomic<int> g_termSignal{0};
    std::atomic<long long> g_termSignalNs{0};

    long long monotonicNs()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts); // async-signal-safe
        return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    }

    extern "C" void onTermSignal(int sig)
    {
        const long long now = monotonicNs();
        long long first = 0;
        if (!g_termSignalNs.compare_exchange_strong(first, now))
        {
            if (now - first >= 1000000000LL)
                _exit(128 + sig);
            return;
        }
        g_termSignal.store(sig);
        static const char kMsg[] = "[shutdown] termination signal received -- stopping\n";
        const ssize_t ignored = write(STDERR_FILENO, kMsg, sizeof(kMsg) - 1);
        (void)ignored;
    }

    // Installed AFTER runtime.initialize(): SDL_Init installs its handlers only over SIG_DFL, and ours
    // must be the one that runs.
    void installTermSignalHandlers(PS2Runtime &runtime)
    {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = onTermSignal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGINT, &sa, nullptr);
        std::thread([&runtime]
                    {
            while (g_termSignal.load(std::memory_order_acquire) == 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const int sig = g_termSignal.load(std::memory_order_acquire);
            ps2x::shutdown::setExitCode(128 + sig);
            runtime.requestStop();
            ps2x::shutdown::armWatchdog(sig == SIGINT ? "SIGINT" : "SIGTERM"); })
            .detach();
    }
#endif

#if defined(__ANDROID__)
    int g_logcatPipeFds[2]{-1, -1};
    std::thread g_logcatThread;

    void stopLogcatRedirect()
    {
        std::fflush(stdout);
        std::fflush(stderr);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        if (g_logcatPipeFds[1] >= 0)
        {
            close(g_logcatPipeFds[1]);
            g_logcatPipeFds[1] = -1;
        }
        if (g_logcatThread.joinable())
        {
            g_logcatThread.join();
        }
    }

    void redirectStdioToLogcat()
    {
        if (pipe(g_logcatPipeFds) != 0)
        {
            return;
        }

        setvbuf(stdout, nullptr, _IOLBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        dup2(g_logcatPipeFds[1], STDOUT_FILENO);
        dup2(g_logcatPipeFds[1], STDERR_FILENO);

        g_logcatThread = std::thread([]()
                                     {
                                         FILE *reader = fdopen(g_logcatPipeFds[0], "r");
                                         if (!reader)
                                         {
                                             return;
                                         }
                                         char line[1024];
                                         while (fgets(line, sizeof(line), reader))
                                         {
                                             size_t len = std::strlen(line);
                                             if (len > 0 && line[len - 1] == '\n')
                                             {
                                                 line[len - 1] = '\0';
                                             }
                                             __android_log_write(ANDROID_LOG_INFO, "ps2x", line);
                                         }
                                         fclose(reader);
                                         g_logcatPipeFds[0] = -1;
                                     });
        if (std::atexit(stopLogcatRedirect) != 0)
        {
            close(STDOUT_FILENO);
            close(STDERR_FILENO);
            close(g_logcatPipeFds[1]);
            g_logcatPipeFds[1] = -1;
            if (g_logcatThread.joinable())
            {
                g_logcatThread.join();
            }
        }
    }
#endif

    void setupTerminateLogger() // to help on release build crashs
    {
        std::set_terminate([]()
                           {
                               std::cerr << "[terminate] unhandled exception" << std::endl;
                               const std::exception_ptr ep = std::current_exception();
                               if (ep)
                               {
                                   try
                                   {
                                       std::rethrow_exception(ep);
                                   }
                                   catch (const std::system_error &e)
                                   {
                                       std::cerr << "[terminate] std::system_error code=" << e.code().value()
                                                 << " category=" << e.code().category().name()
                                                 << " message=" << e.what() << std::endl;
                                   }
                                   catch (const std::exception &e)
                                   {
                                       std::cerr << "[terminate] std::exception: " << e.what() << std::endl;
                                   }
                                   catch (...)
                                   {
                                       std::cerr << "[terminate] non-std exception" << std::endl;
                                   }
                               }
                               std::abort(); });
    }

    std::string normalizeGameId(const std::string &folderName)
    {
        std::string result = folderName;

        size_t underscore = result.find('_');
        if (underscore != std::string::npos)
            result[underscore] = '-';

        size_t dot = result.find('.');
        if (dot != std::string::npos)
            result.erase(dot, 1);

        std::ranges::transform(result, result.begin(), [](unsigned char character)
                               { return static_cast<char>(std::toupper(character)); });

        return result;
    }

#if !defined(PLATFORM_VITA) && !defined(__ANDROID__)
    // 2026-10-02: with no argument, boot the disc the way a PS2 does -- read SYSTEM.CNF and take the
    // executable its BOOT2 line names (`BOOT2 = cdrom0:\SLES_520.17;1` -> SLES_520.17). Looked for in
    // $PS2_GAMEFILES, then <exe dir>/gamefiles/ (the layout scripts/build.sh produces), so a player can
    // start the game by launching the executable alone. Generic: no game's file name is hardcoded.
    std::filesystem::path bootElfFromDisc(char *argv[])
    {
        std::vector<std::filesystem::path> dirs;
        if (const char *e = std::getenv("PS2_GAMEFILES"); e && e[0])
            dirs.emplace_back(e);
        const std::filesystem::path exeDir = ps2x::userdir::executableDirectory(argv);
        if (!exeDir.empty())
            dirs.push_back(exeDir / "gamefiles");
        for (const auto &dir : dirs)
        {
            for (const char *cnfName : {"SYSTEM.CNF", "system.cnf"})
            {
                std::ifstream cnf(dir / cnfName);
                if (!cnf)
                    continue;
                std::string line;
                while (std::getline(cnf, line))
                {
                    const size_t eq = line.find('=');
                    if (eq == std::string::npos)
                        continue;
                    std::string key = line.substr(0, eq);
                    key.erase(std::remove_if(key.begin(), key.end(), [](unsigned char c) { return std::isspace(c); }), key.end());
                    if (key != "BOOT2")
                        continue;
                    std::string value = line.substr(eq + 1);
                    const size_t sep = value.find_last_of("\\/:");
                    if (sep != std::string::npos)
                        value = value.substr(sep + 1);
                    if (const size_t semi = value.find(';'); semi != std::string::npos)
                        value.resize(semi);
                    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); }), value.end());
                    if (!value.empty() && std::filesystem::exists(dir / value))
                        return dir / value;
                }
            }
        }
        return {};
    }
#endif

    std::filesystem::path getExecutablePath(int argc, char *argv[])
    {
        if (argc >= 2 && argv[1] && argv[1][0] != '\0')
        {
            std::cout << "Using argv boot path" << std::endl;
            return std::filesystem::path(argv[1]);
        }
#if !defined(PLATFORM_VITA) && !defined(__ANDROID__)
        if (const std::filesystem::path disc = bootElfFromDisc(argv); !disc.empty())
        {
            std::cout << "Using the disc's boot file (SYSTEM.CNF): " << disc.string() << std::endl;
            return disc;
        }
#endif
#if defined(PS2X_DEFAULT_BOOT_ELF)
        std::cout << "Using default boot file" << std::endl;
        const std::filesystem::path configuredPath = std::filesystem::path(PS2X_DEFAULT_BOOT_ELF);
#if defined(PLATFORM_VITA)
        return configuredPath;
#endif
        if (configuredPath.is_absolute())
        {
            return configuredPath;
        }
        return (std::filesystem::current_path() / configuredPath).lexically_normal();
#else
        throw std::runtime_error("No game found. Put the disc files in gamefiles/ beside the executable, or pass the guest ELF as argv[1].");
#endif
    }
}

int main(int argc, char *argv[])
{
#if defined(__ANDROID__)
    redirectStdioToLogcat();
#endif
    setupTerminateLogger();
    ps2x::loadexec::setHostArgv(argv);   // row 248: LoadExecPS2 restarts the runner with these

    try
    {
        std::filesystem::path pathObj = getExecutablePath(argc, argv);

        std::string filePathStr = pathObj.string();
        std::string elfName = pathObj.filename().string();
        std::string normalizedId = normalizeGameId(elfName);

        std::string windowTitle = "PS2-Recomp | ";
        const char *gameName = getGameName(normalizedId);

#if !defined(PLATFORM_VITA)
        if (const ps2x::launcher::GameInfo *info = ps2x::launcher::gameInfo(); info && info->title)
        {
            windowTitle = info->title;   // the game named itself (ps2_launcher.h)
        }
        else if (gameName)
        {
            windowTitle += std::string(gameName) + " | " + elfName;
        }
        else
#endif
        {
            windowTitle += elfName;
        }

        // The pre-game launcher (ps2_launcher.h), then the settings layer (ps2_settings.h): resolve
        // <exe dir>/config/settings.ini into env flags and re-exec once so static-init readers see
        // them. After the launcher the re-exec is unconditional -- its window must not linger. Both
        // precede anything that reads a flag at runtime.
        bool ranLauncher = false;
#if !defined(PLATFORM_VITA) && !defined(__ANDROID__)
        if (ps2x::launcher::shouldShow(argv))
        {
            if (ps2x::launcher::run(argv, windowTitle, pathObj) == ps2x::launcher::Result::Quit)
                std::_Exit(0);
            ranLauncher = true;
        }
#endif
        ps2x::settings::applyAtStartup(argc, argv, ranLauncher);

        PS2Runtime runtime;
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
        // This hook is to prevent leak rlimgui deps to recompiler etc
        PS2DebugPanel debugPanel;
        runtime.setDebugUiCallbacks(
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                static_cast<PS2DebugPanel *>(userData)->initialize();
            },
            [](PS2Runtime &rt, void *userData)
            {
                static_cast<PS2DebugPanel *>(userData)->draw(rt);
            },
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                static_cast<PS2DebugPanel *>(userData)->shutdown();
            },
            &debugPanel);
#endif
        if (!runtime.initialize(windowTitle.c_str()))
        {
            std::cerr << "Failed to initialize PS2 runtime" << std::endl;
            return 1;
        }

        if (!runtime.loadELF(filePathStr))
        {
            std::cerr << "Failed to load ELF file: " << filePathStr << std::endl;
            return 1;
        }

        // PS2X_MISSING_TARGET (default "stop"): what an unrecoverable missing branch target does.
        //   stop     = log it and END the process, exit 3 (ps2_shutdown.h; the guest state is corrupt)
        //   continue = the old default: log once and resume the caller (the call is skipped) -- debug only,
        //              this is the path that ran away in memory (2026-10-01)
        //   skip     = SkipCallDebug, an escape hatch that hides guest bugs
        {
            const char *e = std::getenv("PS2X_MISSING_TARGET");
            const std::string mode = (e && e[0]) ? e : "stop";
            if (mode == "continue")
                runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::ContinueToTarget);
            else if (mode == "skip")
                runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::SkipCallDebug);
            else
            {
                runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::Stop);
                ps2x::shutdown::setFatalMissingTarget(true);
            }
        }
#if defined(PS2X_POSIX_TERM_SIGNALS)
        installTermSignalHandlers(runtime);
#endif

        runtime.run();

#ifdef _DEBUG
        ps2_log::print_saved_location();
#endif
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(ps2x::shutdown::exitCode());
    }
    catch (const std::exception &e)
    {
        std::cerr << "[main] fatal exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[main] fatal exception: unknown" << std::endl;
    }

    std::cout.flush();
    std::cerr.flush();
    std::_Exit(1);
}
