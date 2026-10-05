#ifndef PS2_USER_DIR_H
#define PS2_USER_DIR_H

// Where a player's own state lives: settings.ini and the memory cards (mc0/, mc1/).
//
// Default = `<exe dir>/config/`, beside the executable we build (user, 2026-10-01: "always in the config
// folder next to the executable"). PS2X_USER_DIR=<dir> overrides it.
//
// Game data (the disc in gamefiles/) and mods (<exe dir>/mods/) stay beside the executable too.

#include <filesystem>

namespace ps2x::userdir
{
    // The directory holding the running executable (empty if it cannot be determined).
    std::filesystem::path executableDirectory(char **argv = nullptr);

    // The user-state directory (see above). Not created here.
    std::filesystem::path directory(char **argv = nullptr);
}

#endif
