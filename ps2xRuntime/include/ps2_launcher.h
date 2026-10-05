#ifndef PS2_LAUNCHER_H
#define PS2_LAUNCHER_H

// The pre-game launcher, shaped like the reference projects' (Zelda 64: Recompiled's launcher.rml):
// a title screen showing the game's name, the disc status and a short menu -- Start game (or "Disc
// not found" with instructions), Settings, Controls, Exit -- with the build version in a corner.
// Settings is a sub-screen over the settings schema (ps2_settings_menu.h); Mods lists the mods
// (ps2_mods.h) with their switches, and a switched-on mod's own options nested under it;
// Controls lists the bindings. It runs in its own short-lived raylib window; on Start it writes settings.ini and main()
// re-execs into the game (ps2x::settings::applyAtStartup(..., forceReexec = true)), so the game never
// shares a window or an initialisation with the launcher.
//
// Shown unless: the settings layer is off (PS2X_SETTINGS=0 -- the replay harness), this is the
// re-exec'd second start, PS2X_LAUNCHER=0, or settings.ini says `[launcher] show = false`.
// PS2X_LAUNCHER=1 forces it.

#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ps2x::launcher
{
    enum class Result
    {
        Play,
        Quit,
    };

    // The GAME names itself (static init, like its settings rows): the title screen and the game
    // window use it instead of the ELF name. `version` is shown in a corner (e.g. the build tag).
    struct GameInfo
    {
        const char *title;
        const char *version;
    };
    bool registerGameInfo(const GameInfo &info);
    const GameInfo *gameInfo();   // nullptr until a game registers

    bool shouldShow(char **argv);
    Result run(char **argv, const std::string &windowTitle, const std::filesystem::path &elfPath);

    // ---- Game pages (row 249) -------------------------------------------------------------------------------
    // A game can add its own screen to the launcher (rotk USA: "Online co-op", in place of the original's online
    // lobby program). The page is DECLARATIVE: rows the launcher draws and navigates in its own style, and a
    // callback run once per frame that reads/changes them and decides what happens next -- the game needs no
    // window or input code. It appears in the main menu under `menuLabel`; PS2X_LAUNCHER_PAGE=<id> opens the
    // launcher directly on it (Back then goes to the main menu).
    struct PageRow
    {
        enum class Kind { Choice, Text, Button, Label };
        Kind kind = Kind::Label;
        std::string label;
        std::vector<std::string> choices;   // Choice: the values, Left/Right steps through them
        int choice = 0;                     // Choice: the selected index
        std::string text;                   // Text: the value (Enter edits it: typing, Backspace, Ctrl+V)
        size_t maxLen = 64;                 // Text: the longest value accepted
        std::string help;                   // shown beside the list while the row is selected
        bool visible = true;                // hidden rows are skipped and not drawn
        bool enabled = true;                // a disabled row is drawn dimmed and cannot change
    };
    struct PageEvent
    {
        int activated = -1;     // index of the Button row pressed this frame, -1 = none
        bool back = false;      // Esc / B this frame (the page decides: cancel something, or leave)
    };
    struct PageResult
    {
        enum class Kind { Stay, Back, Play };
        Kind kind = Kind::Stay;
        // Play: environment set before the game starts ({name, value}; empty value = unset), e.g. PS2X_GUEST_ARGS.
        std::vector<std::pair<std::string, std::string>> env;
    };
    struct Page
    {
        std::string id;          // PS2X_LAUNCHER_PAGE value
        std::string menuLabel;   // its entry in the main menu
        std::string title;       // the page heading
        std::function<void(std::vector<PageRow> &rows)> init;   // build the rows (each time the page opens)
        // Every frame: update rows / status (the line under the list) and decide.
        std::function<PageResult(std::vector<PageRow> &rows, const PageEvent &ev, std::string &status)> update;
    };
    bool registerPage(const Page &page);   // static init; returns true so it can initialise a static
}

#endif
