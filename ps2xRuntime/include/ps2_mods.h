#ifndef PS2_MODS_H
#define PS2_MODS_H

// Mod discovery for the launcher's Mods screen: every folder `<exe dir>/mods/<id>/` is a mod, and an
// optional `mod.ini` DESCRIBES it -- description only, never settings:
//
//     [mod]
//     name        = Widescreen
//     description = One or two sentences a player reads before switching it on.
//     author      = ...
//     version     = ...
//
// All settings stay in config/settings.ini. A mod is switched by the settings option `mods.<id>`
// (section [mods], key = its folder name), which the mod registers beside its code; so env > file >
// preset > default and the re-exec apply unchanged. No such option = the mod is always active.
// A mod's own options are the settings registered in section `mods.<id>` (ps2_settings.h): the Mods
// screen shows them on the mod's options page, and they apply only while the mod is switched on.
// Mods are still compiled into the game (the game repo's mods/ tree); this only reads metadata.

#include <filesystem>
#include <string>
#include <vector>

namespace ps2x::mods
{
    struct ModInfo
    {
        std::string id;            // the folder name
        std::string name;          // [mod] name, else the folder name
        std::string description;
        std::string author;
        std::string version;
        std::string setting;       // "mods.<id>" when that option exists; empty = not switchable
        std::vector<std::string> options;   // its own options: "mods.<id>.<key>", in registration order
        bool hasIni = false;       // false: the folder has no mod.ini
        std::filesystem::path dir;
    };

    // Where the mods live (the Mods screen's folders + the HD pack's default <mods>/hd/assets):
    //   1. <exe dir>/mods, when it is a folder (a packaged game ships its mods beside the binary);
    //   2. else <exe dir>/<the path a game registered> (registerDirectory), when that is a folder;
    //   3. else <exe dir>/mods (so "no mods found" names the expected place). Empty if the exe is unknown.
    std::filesystem::path modsDirectory(char **argv = nullptr);
    // A game whose built binary does not sit beside its mods/ (e.g. one repo for several discs, each
    // binary in its own disc folder) names where they are, RELATIVE to the executable's folder, e.g.
    // "../../mods". Call it during static init (static const bool s = registerDirectory(...)); the last
    // call wins. Returns true.
    bool registerDirectory(const char *relativeToExe);
    std::vector<ModInfo> scan(char **argv);              // sorted by name
}

#endif
