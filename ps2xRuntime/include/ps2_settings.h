#ifndef PS2_SETTINGS_H
#define PS2_SETTINGS_H

// The player-facing settings layer (rotk platform-plan §3.1, cont.363).
//
// A CURATED set of options in `settings.ini` in the user directory (ps2_user_dir.h: `<exe dir>/config/`,
// or PS2X_USER_DIR), each mapped onto one or more of the
// environment flags the runtime and the game already read. The flags stay the ground truth: an
// explicitly exported flag always wins over the file, so every harness script and A/B keeps working.
//
// Precedence, per option: explicit env  >  settings.ini  >  preset  >  (nothing -- the flag's own
// built-in default applies).
//
// Applied by RE-EXEC: many flags are read by static initializers before main() runs, so a setenv()
// from inside the running process can come too late (the cont.359 trap). applyAtStartup() reads the
// file, exports what it resolved (never overwriting), and execv()s the same binary once; the second
// start sees final values everywhere.
//
// Registration is static-init, from any translation unit: the runtime registers its PS2X_* options,
// the GAME registers its own (LOTR_* etc.) beside the code that reads them, so the engine stays
// game-agnostic. Everything is registered before main(), which is when applyAtStartup() runs.
//
// A mod's switch is the option `mods.<id>` and its own options live in section `mods.<id>`
// ([mods.fourplayer] players = 4): those apply only while the switch is on, and then all of them,
// unset ones at their registered default (ps2_mods.h; the launcher's Mods screen edits both).
//
//   PS2X_SETTINGS=<path>  read that file instead      PS2X_SETTINGS=0  disable the layer (harness)

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ps2x::settings
{
    using EnvList = std::vector<std::pair<std::string, std::string>>;

    enum class Kind
    {
        Bool,    // true | false             (also accepted: 1/0, on/off, yes/no)
        Int,     // an integer in [minInt, maxInt]
        Choice,  // one of `choices`, '|'-separated
    };

    struct Setting
    {
        const char *section;       // INI section, e.g. "display"
        const char *key;           // key inside it, e.g. "widescreen"
        Kind kind;
        const char *choices;       // Choice: "gl|cpu"; unused otherwise
        int minInt, maxInt;        // Int only
        const char *defaultValue;  // what the flag does when unset -- documentation for the file
        const char *description;   // one line, written into the first-run file
        // value (already validated; a Bool arrives as "true"/"false") -> the env flags it sets
        void (*map)(const std::string &value, EnvList &out);
    };

    struct Preset
    {
        const char *name;
        const char *description;
        std::vector<std::pair<std::string, std::string>> values;   // {"section.key", "value"}
    };

    bool registerSetting(const Setting &setting);
    bool registerPreset(const Preset &preset);
    // An option was renamed: files that still use `oldKey` keep working ("section.key" -> "section.key").
    bool registerAlias(const char *oldKey, const char *newKey);

    // Call before anything reads a flag. May not return: it re-execs once when it exported anything,
    // or always when `forceReexec` (the launcher ran in this process and its window must not linger).
    void applyAtStartup(int argc, char **argv, bool forceReexec = false);

    // ---- for UIs over the same table (the launcher; later an in-game menu) ----
    bool enabled();                                              // false under PS2X_SETTINGS=0
    std::filesystem::path settingsFile(char **argv);             // empty = no location
    std::vector<Setting> schema();                               // display order
    const std::vector<Preset> &presets();
    std::string key(const Setting &s);                           // "section.key"
    std::map<std::string, std::string> readValues(const std::filesystem::path &file);   // raw, + "preset"
    bool writeValues(const std::filesystem::path &file, const std::map<std::string, std::string> &values);
    bool validate(const Setting &s, const std::string &raw, std::string &normalised);
    std::string environmentOverride(const Setting &s);           // "NAME=value" if the env pins it, else ""
}

#endif
