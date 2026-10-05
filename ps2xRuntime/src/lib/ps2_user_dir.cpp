#include "ps2_user_dir.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <system_error>

namespace ps2x::userdir
{
    namespace
    {
        std::filesystem::path selfPath(char **argv)
        {
            std::error_code ec;
#if defined(__linux__)
            const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
            if (!ec && !self.empty()) return self;
#endif
            if (argv && argv[0])
            {
                const std::filesystem::path p = std::filesystem::absolute(argv[0], ec);
                if (!ec) return p;
            }
            return {};
        }

        std::filesystem::path envPath(const char *name)
        {
            const char *e = std::getenv(name);
            return (e && e[0]) ? std::filesystem::path(e) : std::filesystem::path{};
        }

        std::filesystem::path platformConfigRoot()
        {
#if defined(_WIN32)
            return envPath("APPDATA");
#elif defined(__APPLE__)
            const std::filesystem::path home = envPath("HOME");
            return home.empty() ? home : home / "Library" / "Application Support";
#else
            const std::filesystem::path xdg = envPath("XDG_CONFIG_HOME");
            if (!xdg.empty() && xdg.is_absolute()) return xdg; // the XDG spec ignores a relative value
            const std::filesystem::path home = envPath("HOME");
            return home.empty() ? home : home / ".config";
#endif
        }

        // Copy `from` to `to` unless `to` already exists. Files and directory trees alike.
        bool copyIfMissing(const std::filesystem::path &from, const std::filesystem::path &to)
        {
            std::error_code ec;
            if (!std::filesystem::exists(from, ec) || std::filesystem::exists(to, ec))
                return false;
            std::filesystem::create_directories(to.parent_path(), ec);
            std::filesystem::copy(from, to,
                                  std::filesystem::copy_options::recursive |
                                      std::filesystem::copy_options::skip_existing,
                                  ec);
            if (ec)
            {
                std::fprintf(stderr, "[userdir] could not copy %s -> %s: %s\n", from.string().c_str(),
                             to.string().c_str(), ec.message().c_str());
                return false;
            }
            return true;
        }
    }

    std::filesystem::path executableDirectory(char **argv)
    {
        const std::filesystem::path self = selfPath(argv);
        return self.empty() ? self : self.parent_path();
    }

    std::filesystem::path directory(char **argv)
    {
        const std::filesystem::path fromEnv = envPath("PS2X_USER_DIR");
        if (!fromEnv.empty()) return fromEnv;
        const std::filesystem::path exeDir = executableDirectory(argv);
        return exeDir.empty() ? exeDir : exeDir / "config";
    }

    void migrateLegacy(char **argv)
    {
        if (!envPath("PS2X_USER_DIR").empty()) return;
        const std::filesystem::path dst = directory(argv);
        const std::filesystem::path root = platformConfigRoot();
        const std::filesystem::path self = selfPath(argv);
        if (dst.empty() || root.empty() || self.empty()) return;
        const std::filesystem::path src = root / self.stem();
        std::error_code ec;
        if (std::filesystem::equivalent(src, dst, ec)) return;
        for (const char *item : {"settings.ini", "mc0", "mc1"})
        {
            if (copyIfMissing(src / item, dst / item))
                std::fprintf(stderr, "[userdir] copied %s -> %s (the old copy is left in place and no longer read)\n",
                             (src / item).string().c_str(), (dst / item).string().c_str());
        }
    }
}
