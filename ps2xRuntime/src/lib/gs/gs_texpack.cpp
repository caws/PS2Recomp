#include "runtime/gs/gs_texpack.h"
#include "ps2_mods.h"

#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// raylib is the runtime's host layer and is already linked; LoadImage/ImageResize are pure CPU
// (stb_image / stb_image_resize) and need no GL context, so this stays usable from the GS worker
// threads. The GS backend itself deliberately does not include raylib, which is why this lives in
// its own translation unit.
#include "raylib.h"

namespace ps2x::texpack
{
    namespace
    {
        struct Variant { uint32_t w = 0, h = 0; std::vector<uint8_t> rgba; };
        struct Entry
        {
            bool present = false;          // a file was found and decoded
            uint32_t w = 0, h = 0;         // the replacement's own dimensions
            std::vector<uint8_t> rgba;     // level-0 pixels, RGBA8, alpha in the PS2 0..128 range
            std::map<uint64_t, Variant> scaled;   // key = (w << 32) | h  -- per-level resamples
        };

        std::mutex g_mutex;
        std::unordered_map<uint64_t, Entry> g_cache;   // negative entries are cached too
        unsigned long g_hits = 0, g_files = 0;

        // ★★★★★ DEFAULT OFF -- a bare run is the VANILLA game (user, 2026-09-21). An enhancement
        // is opted INTO, never opted out of, and that has to hold for a renderer-side mod exactly
        // as it holds for a guest-hook one (`mods/README.md` rule 3).
        //
        //   PS2X_GS_TEXPACK unset  -> OFF, original textures                       (the default)
        //   PS2X_GS_TEXPACK=1      -> ON at the default location, <mods>/hd/assets (ps2_mods.h)
        //   PS2X_GS_TEXPACK=<dir>  -> ON at that directory instead (a pack kept elsewhere)
        //   PS2X_GS_TEXPACK=0      -> OFF
        //
        // The default LOCATION is `<mods>/hd/assets`, `<mods>` = ps2x::mods::modsDirectory():
        // `<exe dir>/mods` -- the same `<exe dir>` anchor the runtime uses for `config/` (cont.346q) --
        // or, when that folder does not exist, the one the game registered relative to the executable
        // (row 245). So `=1` works in a package (ship the folder beside the binary) and in a game repo
        // whose binaries sit elsewhere (rotk_recomp: regions/<r>/, mods/ at the root).
        //
        // ⚠ Why it is no longer "on whenever the folder exists": that made the answer to "is the
        // pack live?" depend on whether the player had ever dropped a file into a directory the
        // repo itself ships, so a fresh clone and a working clone behaved differently with the same
        // command line -- and the only symptom is art that is subtly not the game's own.
        //
        // ⚠ An EXPLICIT request that finds no directory SAYS SO, once. Staying silent would be
        // indistinguishable from "the pack is on and this texture simply has no replacement", which
        // is the normal and correct state for a pack that only covers a few textures.
        // ⓘ An empty-but-present directory is still not an error: nothing matches, nothing is
        // replaced, and the game renders exactly as it does without the pack.
        const std::string &packDirStr()
        {
            static const std::string dir = []() -> std::string {
                const char *e = std::getenv("PS2X_GS_TEXPACK");
                if (!e || !e[0]) return {};                            // ★ default: OFF (vanilla)
                if (e[0] == '0' && e[1] == '\0') return {};            // explicitly off
                std::filesystem::path p;
                if (e[0] == '1' && e[1] == '\0')                       // opt in at the default place
                {
                    // <mods>/hd/assets, <mods> = ps2x::mods::modsDirectory(): <exe dir>/mods, or the
                    // folder the game registered relative to the executable (ps2_mods.h, row 245).
                    const std::filesystem::path mods = ps2x::mods::modsDirectory();
                    if (mods.empty())
                    {
                        std::fprintf(stderr, "[texpack] PS2X_GS_TEXPACK=1 but the executable path"
                                             " is unknown -- pack DISABLED\n");
                        return {};
                    }
                    p = mods / "hd" / "assets";
                }
                else
                {
                    p = std::filesystem::path(e);                      // explicit directory
                }
                std::error_code ec2;
                if (!std::filesystem::is_directory(p, ec2))
                {
                    std::fprintf(stderr, "[texpack] pack directory not found: %s -- pack DISABLED\n",
                                 p.string().c_str());
                    return {};
                }
                return p.string();
            }();
            return dir;
        }

        const char *packDir()
        {
            const std::string &d = packDirStr();
            return d.empty() ? nullptr : d.c_str();
        }
    }

    bool enabled() { return packDir() != nullptr; }

    uint64_t contentKey(const uint8_t *rgba, uint32_t w, uint32_t h)
    {
        // FNV-1a over the decoded pixels, with the dimensions mixed in so two different textures
        // that happen to share a byte pattern at different sizes cannot collide.
        uint64_t k = 1469598103934665603ull;
        auto mix = [&k](uint64_t v) { k ^= v; k *= 1099511628211ull; };
        mix(w); mix(h);
        if (rgba)
        {
            const size_t n = (size_t)w * h * 4u;
            for (size_t i = 0; i < n; ++i) { k ^= rgba[i]; k *= 1099511628211ull; }
        }
        return k;
    }

    bool substitute(uint64_t hash, uint32_t baseW, uint32_t baseH,
                    uint32_t &w, uint32_t &h, std::vector<uint8_t> &rgba)
    {
        const char *dir = packDir();
        if (!dir || !baseW || !baseH || !w || !h) return false;

        std::lock_guard<std::mutex> lk(g_mutex);

        auto it = g_cache.find(hash);
        if (it == g_cache.end())
        {
            Entry e;
            static bool announced = false;
            if (!announced)
            {
                announced = true;
                const char *how = std::getenv("PS2X_GS_TEXPACK");
                std::fprintf(stderr, "[texpack] pack directory: %s  (PS2X_GS_TEXPACK=%s)\n", dir,
                             how ? how : "?");
            }
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%016llx.png", dir, (unsigned long long)hash);
            if (FILE *probe = std::fopen(path, "rb"))
            {
                std::fclose(probe);
                Image img = LoadImage(path);
                if (img.data && img.width > 0 && img.height > 0)
                {
                    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
                    e.present = true;
                    e.w = (uint32_t)img.width;
                    e.h = (uint32_t)img.height;
                    e.rgba.assign((const uint8_t *)img.data,
                                  (const uint8_t *)img.data + (size_t)img.width * img.height * 4u);
                    ++g_files;
                    std::fprintf(stderr, "[texpack] loaded %016llx  %ux%u  (replacing %ux%u, x%.2f)\n",
                                 (unsigned long long)hash, e.w, e.h, baseW, baseH,
                                 double(e.w) / double(baseW));
                }
                if (img.data) UnloadImage(img);
            }
            it = g_cache.emplace(hash, std::move(e)).first;
        }
        Entry &e = it->second;
        if (!e.present) return false;

        // Every level of a mip chain keeps its ratio to level 0, so a chain stays consistent
        // instead of level 0 jumping 4x while level 1 stays put (which would break trilinear).
        const double scale = double(e.w) / double(baseW);
        uint32_t tw = (uint32_t)(double(w) * scale + 0.5);
        uint32_t th = (uint32_t)(double(h) * scale + 0.5);
        if (tw < 1u) tw = 1u;
        if (th < 1u) th = 1u;

        if (tw == e.w && th == e.h)
        {
            w = e.w; h = e.h; rgba = e.rgba;
        }
        else
        {
            const uint64_t key = (uint64_t(tw) << 32) | th;
            auto vit = e.scaled.find(key);
            if (vit == e.scaled.end())
            {
                Image img = {};
                img.data = std::malloc(e.rgba.size());
                if (!img.data) return false;
                std::memcpy(img.data, e.rgba.data(), e.rgba.size());
                img.width = (int)e.w; img.height = (int)e.h; img.mipmaps = 1;
                img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
                ImageResize(&img, (int)tw, (int)th);
                Variant v;
                v.w = (uint32_t)img.width; v.h = (uint32_t)img.height;
                v.rgba.assign((const uint8_t *)img.data,
                              (const uint8_t *)img.data + (size_t)img.width * img.height * 4u);
                UnloadImage(img);
                vit = e.scaled.emplace(key, std::move(v)).first;
            }
            w = vit->second.w; h = vit->second.h; rgba = vit->second.rgba;
        }

        if (++g_hits <= 3u || (g_hits % 2000u) == 0u)
            std::fprintf(stderr, "[texpack] substitution #%lu  %016llx -> %ux%u  (%lu file(s) loaded)\n",
                         g_hits, (unsigned long long)hash, w, h, g_files);
        return true;
    }
}
