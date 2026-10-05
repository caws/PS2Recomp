#pragma once

// MOD: the subtitles the ORIGINAL game is missing. LOTR_SUBS_EXTRA, default OFF.
//
// The PS2 release subtitles most content but drops specific spoken lines (user, 2026-09-20,
// confirmed on real hardware -- so it is the shipped game's defect, not ours). This mod shows a
// subtitle for lines we list, using the GAME'S OWN timed-text display. Default OFF = the original's
// gaps, faithfully. See README.md in this folder.
//
// Hook BODIES only -- the registerFunction call lives in mods/register_mods.cpp.

#include <cstdint>

class PS2Runtime;
struct R5900Context;

namespace lotr::mods::subtitles
{

    // ---- hook bodies (registered from mods/register_mods.cpp) ----
    void hook_1c9aa0_subs(uint8_t* rdram, R5900Context* ctx, PS2Runtime* runtime);
    void hook_203c60_subs(uint8_t* rdram, R5900Context* ctx, PS2Runtime* runtime);
    void hook_15ee80_subs(uint8_t* rdram, R5900Context* ctx, PS2Runtime* runtime);
    void hook_15eeb0_subs(uint8_t* rdram, R5900Context* ctx, PS2Runtime* runtime);
}
