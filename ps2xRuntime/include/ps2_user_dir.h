#ifndef PS2_USER_DIR_H
#define PS2_USER_DIR_H

// Where a player's own state lives: settings.ini and the memory cards (mc0/, mc1/).
//
// Default = `<exe dir>/config/`, beside the executable we build (user, 2026-10-01: "always in the config
// folder next to the executable"). PS2X_USER_DIR=<dir> overrides it. (2026-09-25 to 2026-10-01 the
// default was the platform config folder -- ~/.config/<exe name> -- with `portable.txt` opting into
// this layout; that mode is gone and migrateLegacy() brings its data back.)
//
// Game data (the disc in gamefiles/) and mods (<exe dir>/mods/) stay beside the executable too.

#include <filesystem>

namespace ps2x::userdir
{
    // The directory holding the running executable (empty if it cannot be determined).
    std::filesystem::path executableDirectory(char **argv = nullptr);

    // The user-state directory (see above). Not created here.
    std::filesystem::path directory(char **argv = nullptr);

    // One-time move BACK from the platform folder ($XDG_CONFIG_HOME|~/.config/<exe name>, %APPDATA%\<exe
    // name>, ~/Library/Application Support/<exe name>): COPY its settings.ini, mc0/ and mc1/ into
    // directory() for whichever of them directory() does not have yet. Never deletes or overwrites; the
    // old copies are left in place (and are no longer read). Idempotent, so safe on every start.
    void migrateLegacy(char **argv = nullptr);
}

#endif
