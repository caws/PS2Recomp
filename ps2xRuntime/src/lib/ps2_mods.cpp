#include "ps2_mods.h"
#include "ps2_settings.h"
#include "ps2_user_dir.h"

#include <algorithm>
#include <cstdlib>

namespace ps2x::mods
{
    namespace
    {
        std::string &registeredDirectory()   // function-local: safe to fill from any static initializer
        {
            static std::string rel;
            return rel;
        }
    }

    bool registerDirectory(const char *relativeToExe)
    {
        registeredDirectory() = relativeToExe ? relativeToExe : "";
        return true;
    }

    std::filesystem::path modsDirectory(char **argv)
    {
        // <exe dir>/mods: installed content beside the executable, next to config/ (the HD pack reads
        // <mods>/hd/assets). A folder there wins; else the folder the game registered, relative to the
        // executable (ps2_mods.h); else <exe dir>/mods again, so a "not found" message names it.
        const std::filesystem::path exeDir = userdir::executableDirectory(argv);
        if (exeDir.empty()) return {};
        const std::filesystem::path beside = exeDir / "mods";
        std::error_code ec;
        if (std::filesystem::is_directory(beside, ec)) return beside;
        const std::string &rel = registeredDirectory();
        if (!rel.empty())
        {
            const std::filesystem::path p = (exeDir / rel).lexically_normal();
            if (std::filesystem::is_directory(p, ec)) return p;
        }
        return beside;
    }

    std::vector<ModInfo> scan(char **argv)
    {
        std::vector<ModInfo> out;
        const std::filesystem::path root = modsDirectory(argv);
        std::error_code ec;
        if (root.empty() || !std::filesystem::is_directory(root, ec)) return out;
        for (const auto &entry : std::filesystem::directory_iterator(root, ec))
        {
            if (!entry.is_directory(ec)) continue;
            ModInfo m;
            m.id = entry.path().filename().string();
            m.dir = entry.path();
            m.name = m.id;
            const std::filesystem::path ini = entry.path() / "mod.ini";
            if (std::filesystem::exists(ini, ec))
            {
                // readValues gives "section.key"; everything a mod declares is under [mod].
                const auto v = settings::readValues(ini);
                auto get = [&v](const char *k) -> std::string
                {
                    auto it = v.find(std::string("mod.") + k);
                    return it == v.end() ? std::string{} : it->second;
                };
                m.hasIni = true;
                if (!get("name").empty()) m.name = get("name");
                m.description = get("description");
                m.author = get("author");
                m.version = get("version");
            }
            // The switch is found by convention, not declared in the folder: option mods.<id>.
            for (const settings::Setting &st : settings::schema())
            {
                if (settings::key(st) == "mods." + m.id) m.setting = "mods." + m.id;
                if (std::string(st.section) == "mods." + m.id) m.options.push_back(settings::key(st));
            }
            out.push_back(std::move(m));
        }
        std::sort(out.begin(), out.end(), [](const ModInfo &a, const ModInfo &b) { return a.name < b.name; });
        return out;
    }
}
