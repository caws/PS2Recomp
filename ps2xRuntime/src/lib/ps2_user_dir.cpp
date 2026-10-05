#include "ps2_user_dir.h"

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
}
