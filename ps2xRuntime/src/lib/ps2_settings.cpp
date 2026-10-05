#include "ps2_settings.h"
#include "ps2_user_dir.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#if defined(__linux__)
#include <unistd.h>
#endif

// See ps2_settings.h for the model. Implementation notes:
// - The registries are function-local statics, so registration from any TU's static initializer is
//   safe regardless of initialization order.
// - Everything here runs before the runtime exists: plain stdio only, no runtime logging.

namespace ps2x::settings
{
    namespace
    {
        std::vector<Setting> &settingsRegistry()
        {
            static std::vector<Setting> v;
            return v;
        }

        std::vector<Preset> &presetRegistry()
        {
            static std::vector<Preset> v;
            return v;
        }

        std::map<std::string, std::string> &aliasRegistry()   // old "section.key" -> current
        {
            static std::map<std::string, std::string> v;
            return v;
        }

        std::string trim(const std::string &s)
        {
            size_t a = 0, b = s.size();
            while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
            while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
            return s.substr(a, b - a);
        }

        std::string lower(std::string s)
        {
            for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        void setEnv(const char *name, const char *value, bool overwrite)
        {
#if defined(_WIN32)
            if (!overwrite && std::getenv(name)) return;
            _putenv_s(name, value);
#else
            setenv(name, value, overwrite ? 1 : 0);
#endif
        }

        std::string fullKey(const Setting &s) { return std::string(s.section) + "." + s.key; }

        // Sections in the order a player reads them; anything else follows alphabetically. A mod's
        // own options ([mods.<id>]) follow the [mods] switches.
        int sectionRank(const std::string &section)
        {
            static const char *kOrder[] = {"display", "audio", "input", "enhancements", "mods"};
            for (int i = 0; i < 5; ++i)
                if (section == kOrder[i]) return i;
            if (section.rfind("mods.", 0) == 0) return 5;
            return 100;
        }

        std::vector<Setting> sortedSettings()
        {
            std::vector<Setting> v = settingsRegistry();
            std::stable_sort(v.begin(), v.end(), [](const Setting &a, const Setting &b)
                             {
                                 const int ra = sectionRank(a.section), rb = sectionRank(b.section);
                                 if (ra != rb) return ra < rb;
                                 return std::strcmp(a.section, b.section) < 0;
                             });
            return v;
        }

        // Validate and normalise one value. Returns false (with a reason) when it does not fit.
        bool normalise(const Setting &s, const std::string &raw, std::string &out, std::string &why)
        {
            const std::string v = lower(trim(raw));
            switch (s.kind)
            {
            case Kind::Bool:
                if (v == "true" || v == "1" || v == "on" || v == "yes") { out = "true"; return true; }
                if (v == "false" || v == "0" || v == "off" || v == "no") { out = "false"; return true; }
                why = "expected true or false";
                return false;
            case Kind::Int:
            {
                char *end = nullptr;
                const long n = std::strtol(v.c_str(), &end, 10);
                if (v.empty() || !end || *end != '\0' || n < s.minInt || n > s.maxInt)
                {
                    why = "expected an integer from " + std::to_string(s.minInt) + " to " + std::to_string(s.maxInt);
                    return false;
                }
                out = std::to_string(n);
                return true;
            }
            case Kind::Choice:
            {
                std::stringstream ss(s.choices ? s.choices : "");
                std::string c;
                while (std::getline(ss, c, '|'))
                    if (v == c) { out = v; return true; }
                why = std::string("expected one of: ") + (s.choices ? s.choices : "");
                return false;
            }
            }
            return false;
        }

        std::filesystem::path executableDirectory(char **argv)
        {
#if defined(__linux__)
            std::error_code ec;
            const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
            if (!ec && !self.empty()) return self.parent_path();
#endif
            if (argv && argv[0])
            {
                std::error_code ec;
                const std::filesystem::path p = std::filesystem::absolute(argv[0], ec);
                if (!ec) return p.parent_path();
            }
            return {};
        }

        std::string allowedText(const Setting &s)
        {
            switch (s.kind)
            {
            case Kind::Bool: return "true | false";
            case Kind::Int: return std::to_string(s.minInt) + ".." + std::to_string(s.maxInt);
            case Kind::Choice:
            {
                std::string t = s.choices ? s.choices : "";
                std::string out;
                for (char c : t) out += (c == '|') ? std::string(" | ") : std::string(1, c);
                return out;
            }
            }
            return {};
        }

        // The file: every option present with its description. Options in `values` are written
        // active; everything else stays commented at its default -- so the first-run file (no values)
        // changes NOTHING and still documents every choice.
        bool writeFile(const std::filesystem::path &file, const std::map<std::string, std::string> &values)
        {
            std::error_code ec;
            std::filesystem::create_directories(file.parent_path(), ec);
            std::ofstream out(file);
            if (!out) return false;
            out << "; Settings. Lines starting with ';' are comments: remove the ';' in front of an option\n"
                   "; and change its value to use it. Commented options are shown at their default.\n"
                   "; An environment variable for the same flag always wins over this file.\n"
                   "; Order of precedence: environment > this file > preset > built-in default.\n\n";
            if (auto it = values.find("preset"); it != values.end())
                out << "preset = " << it->second << "\n";
            else
                out << "; preset = original\n";
            for (const Preset &p : presetRegistry())
                out << ";   " << p.name << ": " << p.description << "\n";
            std::string section;
            for (const Setting &s : sortedSettings())
            {
                if (section != s.section)
                {
                    section = s.section;
                    out << "\n[" << section << "]\n";
                }
                out << "; " << s.description << " (" << allowedText(s) << ")\n";
                if (auto it = values.find(fullKey(s)); it != values.end())
                    out << s.key << " = " << it->second << "\n";
                else
                    out << "; " << s.key << " = " << s.defaultValue << "\n";
            }
            return static_cast<bool>(out);
        }

        using Ini = std::map<std::string, std::pair<std::string, int>>;   // "section.key" -> (value, line)

        bool readIni(const std::filesystem::path &file, Ini &ini)
        {
            std::ifstream in(file);
            if (!in) return false;
            std::string line, section;
            int lineNo = 0;
            while (std::getline(in, line))
            {
                ++lineNo;
                // A comment runs from ';' or '#' to the end of the line (values never need either).
                const size_t cut = line.find_first_of(";#");
                if (cut != std::string::npos) line.resize(cut);
                line = trim(line);
                if (line.empty()) continue;
                if (line.front() == '[' && line.back() == ']')
                {
                    section = lower(trim(line.substr(1, line.size() - 2)));
                    continue;
                }
                const size_t eq = line.find('=');
                if (eq == std::string::npos)
                {
                    std::fprintf(stderr, "[settings] WARNING %s:%d: not 'key = value', ignored: %s\n",
                                 file.string().c_str(), lineNo, line.c_str());
                    continue;
                }
                const std::string key = lower(trim(line.substr(0, eq)));
                std::string k = section.empty() ? key : section + "." + key;
                // A renamed option keeps working under its old name (registerAlias); a line using the
                // current name wins over one using the old name.
                if (auto a = aliasRegistry().find(k); a != aliasRegistry().end())
                {
                    if (ini.count(a->second)) continue;
                    k = a->second;
                }
                ini[k] = {trim(line.substr(eq + 1)), lineNo};
            }
            return true;
        }
    }

    bool registerSetting(const Setting &setting)
    {
        settingsRegistry().push_back(setting);
        return true;
    }

    bool registerAlias(const char *oldKey, const char *newKey)
    {
        aliasRegistry()[oldKey] = newKey;
        return true;
    }

    bool registerPreset(const Preset &preset)
    {
        presetRegistry().push_back(preset);
        return true;
    }

    void applyAtStartup(int argc, char **argv, bool forceReexec)
    {
        (void)argc;
        // Second start after our own re-exec: everything is already in the environment.
        if (const char *done = std::getenv("PS2X_SETTINGS_APPLIED"); done && done[0] == '1')
            return;

        const char *override = std::getenv("PS2X_SETTINGS");
        if (!enabled())
        {
            std::fprintf(stderr, "[settings] disabled (PS2X_SETTINGS=0)\n");
            return;
        }
        const std::filesystem::path file = settingsFile(argv);
        if (file.empty())
        {
            std::fprintf(stderr, "[settings] no executable directory; settings not read\n");
            return;
        }

        Ini ini;
        if (!readIni(file, ini))
        {
            if (!override)
            {
                writeFile(file, {});
                std::fprintf(stderr, "[settings] created %s (every option commented = all defaults)\n", file.string().c_str());
            }
            else
                std::fprintf(stderr, "[settings] WARNING cannot read %s; settings not applied\n", file.string().c_str());
            return;
        }

        // The preset, if any, supplies values the file does not.
        const Preset *preset = nullptr;
        std::string presetName = "original";
        if (auto it = ini.find("preset"); it != ini.end())
        {
            presetName = lower(it->second.first);
            ini.erase(it);
        }
        for (const Preset &p : presetRegistry())
            if (presetName == p.name) preset = &p;
        if (!preset && presetName != "original")
            std::fprintf(stderr, "[settings] WARNING unknown preset '%s' ignored\n", presetName.c_str());

        std::fprintf(stderr, "[settings] file=%s preset=%s\n", file.string().c_str(), preset ? preset->name : "original");

        // A mod's own options ([mods.<id>], ps2_mods.h) apply only while its switch mods.<id> is on,
        // and then ALL of them, unset ones at their registered default: those defaults are the mod's,
        // while the flags' built-in defaults are "mod off". Resolved before the loop, which consumes
        // the file's entries. The switch is read from file > preset > default (an environment flag
        // still wins over every option it maps, as everywhere).
        auto modOf = [](const Setting &s) -> std::string
        {
            const std::string sec = s.section;
            return sec.rfind("mods.", 0) == 0 ? sec.substr(5) : std::string{};
        };
        std::map<std::string, bool> modOn;   // id -> switch; a mod with no registered switch counts as on
        for (const Setting &s : sortedSettings())
        {
            if (std::string(s.section) != "mods") continue;
            const std::string k = fullKey(s);
            std::string raw = s.defaultValue;
            if (auto it = ini.find(k); it != ini.end()) raw = it->second.first;
            else if (preset)
                for (const auto &kv : preset->values)
                    if (kv.first == k) raw = kv.second;
            std::string v, why;
            modOn[s.key] = normalise(s, raw, v, why) && v == "true";
        }

        bool exported = false;
        for (const Setting &s : sortedSettings())
        {
            const std::string k = fullKey(s);
            const std::string mod = modOf(s);
            const bool modOff = !mod.empty() && modOn.count(mod) && !modOn[mod];
            std::string raw, source;
            if (auto it = ini.find(k); it != ini.end())
            {
                raw = it->second.first;
                source = "file:" + std::to_string(it->second.second);
                ini.erase(it);
            }
            else if (preset)
            {
                for (const auto &kv : preset->values)
                    if (kv.first == k) { raw = kv.second; source = std::string("preset ") + preset->name; }
            }
            if (source.empty() && !mod.empty() && !modOff)
            {
                raw = s.defaultValue;
                source = "mod " + mod + " on, default";
            }
            if (modOff)
            {
                if (!source.empty())
                    std::fprintf(stderr, "[settings] %s (%s): mod %s is off, not applied\n", k.c_str(), source.c_str(), mod.c_str());
                continue;
            }
            if (source.empty()) continue;   // not set anywhere: the flag's own default applies

            std::string value, why;
            if (!normalise(s, raw, value, why))
            {
                std::fprintf(stderr, "[settings] WARNING %s = '%s' (%s): %s -- ignored\n",
                             k.c_str(), raw.c_str(), source.c_str(), why.c_str());
                continue;
            }
            EnvList env;
            s.map(value, env);
            for (const auto &[name, v] : env)
            {
                if (const char *cur = std::getenv(name.c_str()))
                {
                    std::fprintf(stderr, "[settings] %s = %s (%s) -> %s: environment wins (%s=%s)\n",
                                 k.c_str(), value.c_str(), source.c_str(), name.c_str(), name.c_str(), cur);
                    continue;
                }
                setEnv(name.c_str(), v.c_str(), false);
                exported = true;
                std::fprintf(stderr, "[settings] %s = %s (%s) -> %s=%s\n",
                             k.c_str(), value.c_str(), source.c_str(), name.c_str(), v.c_str());
            }
        }
        for (const auto &kv : ini)
            std::fprintf(stderr, "[settings] WARNING %s:%d: unknown option '%s' ignored\n",
                         file.string().c_str(), kv.second.second, kv.first.c_str());

        if (!exported && !forceReexec) return;

        // Re-exec once so every reader -- including static initializers that already ran in THIS
        // process -- sees the final environment.
        setEnv("PS2X_SETTINGS_APPLIED", "1", true);
        std::fflush(stdout);
        std::fflush(stderr);
#if defined(__linux__)
        // exec the RESOLVED path, not the /proc/self/exe link: the kernel names the process after the
        // path's basename, and "exe" would break `pgrep -x <game>` and every tool that finds the runner.
        std::error_code ec;
        const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
        execv(!ec && !self.empty() ? self.c_str() : "/proc/self/exe", argv);
        std::fprintf(stderr, "[settings] WARNING re-exec failed (%s); options read at static init may not apply\n",
                     std::strerror(errno));
#else
        std::fprintf(stderr, "[settings] WARNING no re-exec on this platform; options read at static init may not apply\n");
#endif
    }

    bool enabled()
    {
        const char *e = std::getenv("PS2X_SETTINGS");
        return !(e && e[0] == '0' && e[1] == '\0');
    }

    std::filesystem::path settingsFile(char **argv)
    {
        if (const char *e = std::getenv("PS2X_SETTINGS"); e && e[0] && !(e[0] == '0' && e[1] == '\0'))
            return e;
        // <exe dir>/config/ (PS2X_USER_DIR overrides), ps2_user_dir.h.
        const std::filesystem::path dir = userdir::directory(argv);
        return dir.empty() ? std::filesystem::path{} : dir / "settings.ini";
    }

    std::vector<Setting> schema() { return sortedSettings(); }
    const std::vector<Preset> &presets() { return presetRegistry(); }
    std::string key(const Setting &s) { return fullKey(s); }

    std::map<std::string, std::string> readValues(const std::filesystem::path &file)
    {
        Ini ini;
        std::map<std::string, std::string> out;
        if (!readIni(file, ini)) return out;
        for (const auto &kv : ini) out[kv.first] = kv.second.first;
        return out;
    }

    bool writeValues(const std::filesystem::path &file, const std::map<std::string, std::string> &values)
    {
        return writeFile(file, values);
    }

    bool validate(const Setting &s, const std::string &raw, std::string &normalised)
    {
        std::string why;
        return normalise(s, raw, normalised, why);
    }

    std::string environmentOverride(const Setting &s)
    {
        EnvList env;
        std::string v;
        std::string why;
        if (!normalise(s, s.defaultValue, v, why)) v = s.defaultValue;
        s.map(v, env);
        for (const auto &[name, value] : env)
            if (const char *cur = std::getenv(name.c_str()))
                return name + "=" + cur;
        return {};
    }

    // ---- the runtime's own options (PS2X_*). The game registers its own beside its code. ----
    namespace
    {
        const bool s_regRuntime = []
        {
            registerPreset({"original", "the game as it shipped: every option at its default", {}});
            registerSetting({"display", "renderer", Kind::Choice, "gl|cpu", 0, 0, "gl",
                             "renderer: gl = GPU, cpu = the software reference",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_GS_RENDERER", v}); }});
            registerSetting({"display", "render_scale", Kind::Int, nullptr, 1, 8, "1",
                             "internal resolution multiplier (gl renderer)",
                             [](const std::string &v, EnvList &o)
                             {
                                 o.push_back({"PS2X_GS_SCALE", v});
                                 if (v != "1") o.push_back({"PS2X_GS_PRESENT_HIRES", "1"});   // show it at that size
                             }});
            registerSetting({"display", "window_scale", Kind::Int, nullptr, 1, 4, "1",
                             "initial window size as a multiple of the PS2 display",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_WINDOW_SCALE", v}); }});
            registerSetting({"audio", "enabled", Kind::Bool, nullptr, 0, 0, "true",
                             "sound output",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_AUDIO", v == "true" ? "1" : "0"}); }});
            registerSetting({"input", "keyboard", Kind::Choice, "auto|pad|keys|off", 0, 0, "auto",
                             "player 2: auto = keyboard when a gamepad has player 1, pad = second gamepad only, "
                             "keys = both players on one keyboard, off = no player 2",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_PAD2", v}); }});
            registerSetting({"launcher", "show", Kind::Bool, nullptr, 0, 0, "true",
                             "show the settings launcher before the game starts (PS2X_LAUNCHER=1 brings it back)",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_LAUNCHER", v == "true" ? "1" : "0"}); }});
            // The HD texture pack is a mod (<exe dir>/mods/hd): its switch lives in [mods] under its
            // folder name, which is how the launcher's Mods screen finds it (ps2_mods.h).
            registerSetting({"mods", "hd", Kind::Bool, nullptr, 0, 0, "false",
                             "HD textures: replace textures from the mods folder's hd/assets (PS2X_GS_TEXPACK)",
                             [](const std::string &v, EnvList &o) { o.push_back({"PS2X_GS_TEXPACK", v == "true" ? "1" : "0"}); }});
            registerAlias("enhancements.hd_textures", "mods.hd");
            return true;
        }();
    }
}
