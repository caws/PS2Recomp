#pragma once
// ★★★★ HD texture pack: substitute a higher-resolution image for a decoded PS2 texture.
//
// ★ DEFAULT OFF: unset or `=0` is the vanilla game. `PS2X_GS_TEXPACK=1` enables the pack at its
// default location (`<exe dir>/mods/hd/assets`); `PS2X_GS_TEXPACK=<dir>` enables it at that
// directory instead. A replacement is looked up by the
// texture's CONTENT HASH (`GSCpuBackend::glTexContentHash`) as `<dir>/<016llx>.png`, which is the
// same name `PS2X_GS_GLR_DUMPTEX` writes into its filenames — that is what joins the two halves of
// the workflow.
//
// It works without touching geometry because the GL shader samples NORMALISED
// (`texture(uTex, st)`), so a denser image at the same logical extent lands exactly where the
// original did. See the pack's own README for the two conventions a replacement must keep:
// RGB carries no colour for alpha-only art, and ★ ALPHA IS IN THE PS2 RANGE (0x80 = fully opaque,
// NOT 255) — these bytes go straight into the decoded RGBA the GL path uploads, unscaled.
#include <cstdint>
#include <vector>

namespace ps2x::texpack
{
    bool enabled();

    // ★★★★★ The lookup key is a hash of the DECODED PIXELS, not of VRAM.
    //
    // `GSCpuBackend::glTexContentHash` hashes VRAM PAGES, and a texture's pages hold more than that
    // texture -- so the SAME art hashes differently depending on what else happens to be resident.
    // Measured: the main menu and gameplay draw a byte-identical 256x256 glyph sheet (identical
    // alpha planes, identical generated PNG) under two different VRAM hashes, `a48e9830c112884c`
    // and `0e4513951db40344`. A pack keyed on VRAM therefore MISSES half its own art and needs a
    // duplicate file per occurrence, with no way to enumerate them up front.
    //
    // Keying on the decoded RGBA makes one file serve every occurrence, because the key is what the
    // texture LOOKS LIKE.
    uint64_t contentKey(const uint8_t *rgba, uint32_t w, uint32_t h);

    // Replace one mip level. `baseW/baseH` are level 0's dimensions (the scale factor is derived
    // from them, so every level of a chain keeps its ratio); `w`,`h`,`rgba` are this level's
    // decoded pixels and are overwritten on a hit. Returns true if a replacement was applied.
    bool substitute(uint64_t hash, uint32_t baseW, uint32_t baseH,
                    uint32_t &w, uint32_t &h, std::vector<uint8_t> &rgba);
}
