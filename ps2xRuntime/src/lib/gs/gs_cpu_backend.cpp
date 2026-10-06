#include "runtime/ps2_gs_pipeline.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/gs/gs_texpack.h"   // cont.355e: the HD texture pack
#include "ps2_log.h"
#include "ThreadNaming.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <tuple>
#include <type_traits>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <functional>   // cont.358e: the cluster census union-find
#include <map>
#include <set>
#include <mutex>
#include <vector>
#include <smmintrin.h> // cont.231: SSE4.1 bilinear filter (bilinear4)

// PS2X_HAS_GS_GPU_DEVICE comes from CMake (desktop=1; Vita/Android=0, no EGL there).
#if !defined(PS2X_HAS_GS_GPU_DEVICE)
#define PS2X_HAS_GS_GPU_DEVICE 0
#endif
#if PS2X_HAS_GS_GPU_DEVICE
#include "runtime/gs/gs_gpu_device.h"
#endif

using namespace GSInternal;

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    __attribute__((always_inline)) inline u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    __attribute__((always_inline)) inline uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    __attribute__((always_inline)) inline int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    __attribute__((always_inline)) inline bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    // cont.231 PS2X_GS_FASTCOMBINE (default ON; "=0" restores the scalar switch): the MODULATE texture
    // combine as one 16-bit SSE multiply/shift/saturate -- (t*v)>>7 is at most 508, exact in 16 bits, and
    // packus_epi16 is clampU8. Same bytes as the scalar code (the bench hash is the oracle).
    static const bool s_gsFastCombine = []
    { const char *e = std::getenv("PS2X_GS_FASTCOMBINE"); return !(e && e[0] == '0'); }();

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    __attribute__((always_inline)) inline TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        // ★ cont.231: MODULATE (this game's combine for all but one state combo) in four 16-bit
        // lanes: t*v <= 65025 is exact in u16, >>7 <= 508, packus saturates to 255 exactly as
        // clampU8 does. The alpha lane is the same expression when TCC is set; otherwise va.
        if (s_gsFastCombine && tex.tfx == 0)
        {
            const uint32_t tp = static_cast<uint32_t>(tr) | (static_cast<uint32_t>(tg) << 8) |
                                (static_cast<uint32_t>(tb) << 16) | (static_cast<uint32_t>(ta) << 24);
            const uint32_t vp = static_cast<uint32_t>(vr) | (static_cast<uint32_t>(vg) << 8) |
                                (static_cast<uint32_t>(vb) << 16) | (static_cast<uint32_t>(va) << 24);
            const __m128i t16 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(static_cast<int>(tp)));
            const __m128i v16 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(static_cast<int>(vp)));
            __m128i m = _mm_srli_epi16(_mm_mullo_epi16(t16, v16), 7);
            m = _mm_packus_epi16(m, m);
            const uint32_t packed = static_cast<uint32_t>(_mm_cvtsi128_si32(m));
            out.r = static_cast<uint8_t>(packed & 0xFFu);
            out.g = static_cast<uint8_t>((packed >> 8) & 0xFFu);
            out.b = static_cast<uint8_t>((packed >> 16) & 0xFFu);
            out.a = textureHasAlpha ? static_cast<uint8_t>(packed >> 24) : va;
            return out;
        }

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    // TODO: clut cache
    uint32_t resolveClutIndex(uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm)
    {
        uint32_t clutIndex = static_cast<uint32_t>(index);

        // CSM2 addresses the source directly through TEXCLUT. CSA is required
        // to be zero there, so it must not offset the source coordinates.
        if (csm != 0u)
            return (sourcePsm == GS_PSM_T4 ||
                    sourcePsm == GS_PSM_T4HH ||
                    sourcePsm == GS_PSM_T4HL)
                       ? (clutIndex & 0x0Fu)
                       : clutIndex;

        const bool is16BitClut = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
        const uint32_t csaMask = is16BitClut ? 0x1Fu : 0x0Fu;
        const uint32_t clutIndexMask = is16BitClut ? 0x1FFu : 0x0FFu;
        const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;

        switch (sourcePsm)
        {
        case GS_PSM_T4:
        case GS_PSM_T4HH:
        case GS_PSM_T4HL:
            clutIndex = clutBase + (clutIndex & 0x0Fu);
            break;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            clutIndex = clutBase + clutIndex;
            break;
        default:
            return clutIndex;
        }

        return swizzleClutIndexCSM1(clutIndex & clutIndexMask);
    }

    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(static_cast<int>(std::lround(top + (bottom - top) * fy)));
    }

    // ★★ cont.231: the bilinear filter for all four channels at once, BIT-IDENTICAL to four
    // lerpChannel calls. Each SSE lane performs exactly the scalar sequence in single precision --
    // (float)c00 + ((float)c10 - (float)c00) * fx, then top + (bottom - top) * fy -- and per-lane SSE
    // arithmetic is IEEE-identical to scalar (the build is SSE4.1 without FMA, so the scalar code
    // could not have been contracted either). The final std::lround (half away from zero) is done
    // exactly without libm: every lane is in [0,255] (a lerp of bytes by a weight in [0,1) cannot
    // leave the interval of its endpoints under round-to-nearest), so lround(v) == floor(v + 0.5),
    // and v + 0.5 is EXACT in double (a float add can round up at the .5 boundary -- v = 0.5 - 2^-25
    // gives 1.0f -- which is why the round went through double). packus keeps clampU8's guard.
    // The bench profile (cont.231) had __lroundf at 5.5% and lerpChannel at 5.5% of the whole raster.
    // ★ cont.234, MEASURED AND REVERTED -- do not re-attempt. The cont.234 bench profile put 6 of
    // ~48 busy samples on the six round-chain lines below, so the double round trip was replaced by
    // an exact single-precision equivalent: n = trunc(v + 0.5f) is either floor(v+0.5) or one more
    // (a float add can round UP across an integer boundary but never down), and `n - 0.5` is exactly
    // representable for n in [0,256], so `v < n - 0.5` corrects it exactly. That form was verified
    // over ALL 1,132,396,545 floats in [0,255] (0 mismatches, boundary case included) and held the
    // bench hash on all 8 runs -- and it measured DEAD NEUTRAL: 309.2/309.9 vs 309.5/308.1 ms at one
    // thread, 127.3/127.0 vs 127.1/127.5 at three, alternating on one binary. Nine ALU ops came out
    // of the per-texel tail and nothing moved, because this loop is bound by the texel gathers, not
    // by its arithmetic. Kept as the double version: two code paths and an env flag are not worth 0%.
    // cont.324: the filter on ALREADY-UNPACKED float texels (the four-wide lin tap reads its palette as
    // floats, ResolvedDraw::clutF); bilinear4 below is the u32 form = unpack + this, unchanged.
    __attribute__((always_inline)) inline uint32_t bilinear4Lerp(__m128 p00, __m128 p10, __m128 p01, __m128 p11, float fx, float fy)
    {
        const __m128 vfx = _mm_set1_ps(fx), vfy = _mm_set1_ps(fy);
        const __m128 top = _mm_add_ps(p00, _mm_mul_ps(_mm_sub_ps(p10, p00), vfx));
        const __m128 bot = _mm_add_ps(p01, _mm_mul_ps(_mm_sub_ps(p11, p01), vfx));
        const __m128 v = _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bot, top), vfy));
        const __m128d half = _mm_set1_pd(0.5);
        const __m128d lo = _mm_add_pd(_mm_cvtps_pd(v), half);
        const __m128d hi = _mm_add_pd(_mm_cvtps_pd(_mm_movehl_ps(v, v)), half);
        __m128i i = _mm_unpacklo_epi64(_mm_cvttpd_epi32(lo), _mm_cvttpd_epi32(hi));
        i = _mm_packus_epi32(i, i);
        i = _mm_packus_epi16(i, i);
        return static_cast<uint32_t>(_mm_cvtsi128_si32(i));
    }
    __attribute__((always_inline)) inline uint32_t bilinear4(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, float fx, float fy)
    {
        const auto unpack = [](uint32_t c) {
            return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(c))));
        };
        return bilinear4Lerp(unpack(c00), unpack(c10), unpack(c01), unpack(c11), fx, fy);
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

// ---- Off-thread rasterization (PS2X_GS_THREAD, default OFF — cont.161 structural
// experiment). Profiling measured GS rasterization at 24-40% of EE-thread time by scene,
// all reached synchronously under the VU1 XGKICK drain. With the flag ON, Submit/
// BeginTransfer/UploadImage enqueue self-contained copies to a worker thread (strict
// FIFO, so draw->upload->draw ordering is preserved), and every read-side entry point
// (ReadVram/WriteVram/Snapshot/transfer state/local-to-host/Present/Clear) drains the
// queue first. Sync()/Flush() drain — the frontend already calls Flush+Sync(Finish)
// BEFORE latching the CSR FINISH bit, so guest-visible completion timing stays faithful
// (the interface was designed for async backends: "GPU backends may wait on fences
// here"). Backpressure: bounded items + upload bytes, producer blocks when full.
// Default ON since cont.162: validated by the 300s soak (RSS plateau ~197MB, 57.3k
// draws/s sustained with zero degradation, 0 degen/kick-drops/reserved, gameplay frames
// pixel-perfect) on top of the cont.161 A/B (+28% draws, MOVIE-END -59s). =0 reverts to
// the synchronous legacy paths.
static const bool s_gsThread = []
{ const char *e = std::getenv("PS2X_GS_THREAD"); return !(e && e[0] == '0'); }();
// ★★ cont.230 PS2X_GS_DRAWRUN (default ON, "=0" restores one queue item per primitive): the EE thread
// enqueued every primitive individually -- a mutex, a condition-variable predicate, a ~400-byte item
// copy and a notify per draw, ~40k times a frame (GSCpuBackend::Submit 2.5% self + the lock leaves in
// the cont.230 profile). Draws now accumulate in m_pendingRun and go to the worker as one DrawRun item
// at the end of each GIF packet / arbiter drain, before any non-draw operation (transfer, upload,
// sync, present, VRAM read/write, snapshot -- everything funnels through enqueueWork/drainQueue), or
// at kPendingRunMax. Order is therefore preserved by construction; the worker already rasterises
// consecutive draws as one run.
static const bool s_gsDrawRun = []
{ const char *e = std::getenv("PS2X_GS_DRAWRUN"); return !(e && e[0] == '0'); }();
static constexpr size_t kPendingRunMax = 1024; // the cont.230 default; the live size is s_gsRunMax
// ★ cont.231 (build 402) PS2X_GS_RUNMAX (default 4096 = the worker's coalescing cap) and
// PS2X_GS_RUNFLUSH_PACKET (default OFF): the EE used to flush a DrawRun item at the end of EVERY GIF
// packet (~3600 per level frame, ~20 primitives each) and the worker merged them back into ~4096-
// primitive runs by COPYING every batch (3.4-6.6 ms of a ~110 ms frame, all of the between-run gap
// that was not barrier variance -- build 401's timers). Draws now accumulate to RUNMAX primitives
// (or the next non-draw item / drain / flip, which flush as before), so a run arrives as ONE vector
// the worker swaps in without copying, and the queue carries ~70 items per frame instead of ~3600.
// The raster is the wall (the queue never empties), so the latency of a bigger flush costs nothing.
static const size_t s_gsRunMax = []
{ const char *e = std::getenv("PS2X_GS_RUNMAX"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 4096ul; if (v < 16ul) v = 16ul; if (v > 4096ul) v = 4096ul; return static_cast<size_t>(v); }();
static const bool s_gsRunFlushPacket = []
{ const char *e = std::getenv("PS2X_GS_RUNFLUSH_PACKET"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_drawRunItems = 0, g_drawRunBatches = 0;
static constexpr size_t kGsQueueMaxItems = 65536;
static constexpr size_t kGsQueueMaxUploadBytes = 96u * 1024u * 1024u;

// ---- Display-flip presentation snapshot (PS2X_GS_FLIPSNAP, default ON; =0 reverts to the pure
// per-Sync capture). Root cause (cont.165, menu-text RE): the menu redraws its DISPLAY buffer
// every frame -- clear -> scene composite -> cover into the BACK buffer, then flips, then draws
// the UI TEXT directly onto the new FRONT buffer (real scanout picks that up mid-frame). So the
// frame-complete state exists only shortly AFTER each DISPFB flip, and mid-frame captures (the
// old per-Sync snapshot, or the live no-drain present) showed post-clear black or the pre-cover
// scene, never the text. Mechanism: the DISPFB1/2 write ARMS a capture; the worker consumes it
// at its first idle moment after at least one post-flip draw executed (the front-buffer text
// batch). The per-Sync capture remains as a fallback for eras with no flips for >400ms.
static const bool s_gsFlipSnap = []
{ const char *e = std::getenv("PS2X_GS_FLIPSNAP"); return !(e && e[0] == '0'); }();
// ★★ cont.231 PS2X_GS_FLIPASYNC (default ON; "=0" restores the drained flip): the flip snapshot is
// taken by the RASTER WORKER when it reaches a FlipSnapshot item enqueued at the flip -- the same
// FIFO point the drained copy captured (everything submitted before the flip, the UI text included,
// nothing after), so the cont.165 text-in-frame property holds -- and the EE thread no longer parks
// until the raster queue is empty. With microVU the EE spent 49% of the level frame in that drain
// (cont.231 profile) while the raster threads were busy; the raster is now cheaper than the EE's own
// work, so overlapping them makes the frame EE-bound instead of EE-plus-raster-tail.
static const bool s_gsFlipAsync = []
{ const char *e = std::getenv("PS2X_GS_FLIPASYNC"); return !(e && e[0] == '0'); }();
// ★ cont.231 PS2X_GS_FASTCOORD (default ON; "=0" restores DrawPrimitive per item): inside a band run
// the coordinator used to reach its own rows through the whole DrawPrimitive preamble -- some twenty
// diagnostic flag checks, the capture hook, the GPU-verify hooks and a 64-bit modulo for the
// throughput print -- before the same RasterBand() call the band threads make directly. Build 399's
// per-item timers: coordinator raster 105 ms per level frame vs ~96 ms per band, bands idle ~25 ms
// each; that ~130 ns per primitive was the pool's long pole. With no diagnostic armed the
// coordinator now calls RasterBand() directly, keeping only the capture hook (it early-returns
// unless armed) and the throughput count. Same pixels: the preamble writes nothing.
static const bool s_gsFastCoord = []
{ const char *e = std::getenv("PS2X_GS_FASTCOORD"); return !(e && e[0] == '0'); }();
// ★ cont.232 PS2X_GS_FIELD_BOB (default OFF; "=1" restores the cont.231 presentation): in FIELD mode
// (SMODE2 INT=1, FFMD=0) the guest framebuffer is full-height and the CRT scans out alternate lines
// per field. The presenter used to line-double ONE field per vsync parity (PCSX2's BobTFF/BFF
// option), so every snapshot was decoded twice -- once per parity -- and the image bobbed by a line.
// The default now presents the full frame, which is PCSX2's Automatic/default behaviour for FFMD=0:
// GSRenderer::Merge leaves mode = -1 for a non-FFMD game and GSDevice::Interlace's default branch
// presents the merged texture as-is (m_current = m_merge); GSPCRTCRegs' framebufferRect reads the
// full renderHeight and is halved only when FFMD && INT. One decode per snapshot.
static const bool s_gsFieldBob = []
{ const char *e = std::getenv("PS2X_GS_FIELD_BOB"); return e && e[0] && e[0] != '0'; }();
// Present() renders the frame-complete copy while it is at most this old; older = live VRAM.
static constexpr uint64_t kPresentCopyFreshMs = 500u;
// cont.231 PS2X_GS_FASTUPLOAD (default ON): see UploadImageUnlocked.
static const bool s_gsFastUpload = []
{ const char *e = std::getenv("PS2X_GS_FASTUPLOAD"); return !(e && e[0] == '0'); }();
// cont.231 PS2X_GS_UPLOADVERIFY (default OFF): after every fast-path upload chunk, re-walk the chunk
// through the generic per-pixel read and compare each pixel's stored value with the value the
// chunk carried -- the bench hash cannot see uploads (the capture replays draws), so this is the
// oracle for the CT32 row-group path. Prints the first mismatches and a tally every 4096 chunks.
static const bool s_gsUploadVerify = []
{ const char *e = std::getenv("PS2X_GS_UPLOADVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_upVerifyChunks = 0, g_upVerifyPixels = 0, g_upVerifyMismatch = 0;
// cont.231 PS2X_GS_PRESENT_CACHE (default ON): Present() returns the previously decoded frame when
// the frame-complete snapshot (its capture sequence) and every register the decode reads (the
// display registers, the vsync field parity, the context frames, the preferred source) are unchanged
// since the last present -- the presenter re-presents the same guest frame ~6x at 45 Hz, each time
// copying the 4 MB snapshot and decoding it (32% of a core in the cont.231 live profile).
static const bool s_gsPresentCache = []
{ const char *e = std::getenv("PS2X_GS_PRESENT_CACHE"); return !(e && e[0] == '0'); }();
// CLUT shadow (default ON; =0 disables) -- see the m_clutShadow member comment (cont.165).
static const bool s_gsClutShadow = []
{ const char *e = std::getenv("PS2X_GS_CLUTSHADOW"); return !(e && e[0] == '0'); }();
// ---- GPU present decode (cont.166, phase 0 of the GPU-backend arc; desktop only) ----
// PS2X_GS_GPU_PRESENT (default OFF): decode the presented display rects with the GPU
// compute kernel (EGL surfaceless GL 4.3 -- gs_gpu_device.cpp) instead of the per-pixel
// CPU walk in CopyFrameToHostRgbaCpu; the CPU path remains the automatic fallback
// (device init failure, non-CT display psm, non-owner thread). PS2X_GS_GPU_VERIFY
// (default OFF; implies attempting the GPU decode): run BOTH decodes per rect and
// byte-compare -- the CPU result stays authoritative. Same shadow-verify rollout
// pattern PS2X_GS_THREAD used.
static const bool s_gsGpuPresent = []
{ const char *e = std::getenv("PS2X_GS_GPU_PRESENT"); return e && e[0] && e[0] != '0'; }();
static const bool s_gsGpuVerify = []
{ const char *e = std::getenv("PS2X_GS_GPU_VERIFY"); return e && e[0] && e[0] != '0'; }();
// PS2X_GS_GPU_BATCHPERF (default OFF; implies the mirror): run the batch accumulator and the
// GPU tile rasterizer FOR REAL -- no seeding, no compares -- and time both rasterizers, so the
// question "would the GPU actually be faster?" gets an answer WITHOUT flipping authority.
// The CPU stays authoritative for what the game sees; the GPU renders the same primitives into
// the mirror in parallel. This is also exactly the no-verify batch mode the authority flip
// needs, so it is not throwaway scaffolding.
static const bool s_gsGpuBatchPerf = []
{ const char *e = std::getenv("PS2X_GS_GPU_BATCHPERF"); return e && e[0] && e[0] != '0'; }();
#if PS2X_HAS_GS_GPU_DEVICE
// cont.329 phase 3b: GL texture decode volatility (see the instrument at the decode site).
static unsigned long long g_glTexNew = 0, g_glTexRegen = 0, g_glTexTexels = 0;
// Decodes avoided because the generation moved but the CONTENT did not -- the whole point of 3b.
static unsigned long long g_glTexHashSaved = 0;
// Textured draws that could not be given a texture, by reason (see the decode site).
static unsigned long long g_glTexRejZero = 0, g_glTexRejBig = 0, g_glTexDrawsTme = 0, g_glTexDrawsPlain = 0;
// ★★ cont.332: `g_glTexNew`/`g_glTexRegen` used to be incremented for every textured DRAW,
// outside the `needDecode` branch whose two outcomes they name, so `[gs2:glrtex] decodes=`
// printed 103,571,456 against `new=202` and read as a cache that never hits -- when it was
// serving 99.95% of draws. They now count DECODES; draws keep their own counter.
extern std::atomic<float> g_gsPresentAspect; // cont.332d, defined near ps2xGsPresentAspect()
static unsigned long long g_glTexDraws = 0;
// ★★★ cont.332 PS2X_GS_GLR_DECVERIFY census: of the decodes that re-decode a texture we have
// already shipped, how many produce BYTE-IDENTICAL texels? Those are the ones the page-level
// invalidation got wrong.
static unsigned long long g_glTexDecSame = 0, g_glTexDecDiff = 0, g_glTexDecFirst = 0;
// Draws promoted to sample a render target instead of decoding stale memory (phase 4b).
static unsigned long long g_glTexFromTarget = 0;
// cont.329 phase 4e: Z distribution census (PS2X_GS_GLR_ZCENSUS).
static const bool g_glZCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_ZCENSUS"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_glZN = 0, g_glZHist[10] = {}, g_glZMsk0 = 0, g_glZMsk1 = 0;
static float g_glZMin = 1e9f, g_glZMax = -1e9f;
static const bool g_glPrimCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_PRIMCENSUS"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_glPrimSeen[8] = {}, g_glPrimDropped[8] = {}, g_glPrimN = 0;
static const bool g_glVtxCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_VTXCENSUS"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_glVtxN = 0, g_glVtxOff = 0, g_glVtxNaN = 0, g_glVtxLum[8] = {};
static float g_glVtxMinX = 1e9f, g_glVtxMaxX = -1e9f, g_glVtxMinY = 1e9f, g_glVtxMaxY = -1e9f;
// ★★★★★ cont.332l PS2X_GS_GLR_KEYCENSUS (default OFF, read-only): does the GL texture cache key
// COLLIDE? The key is an XOR of shifted TEX0 fields and `cbp << 44` overlaps `tbp << 40`, so two
// different (tbp, cbp) pairs can land on the same key -- and then one texture's upload overwrites
// the other's content in the same GL object. The character-select draw samples a texture whose
// fixed texels do NOT match its own decode (cont.332l), which is exactly that symptom. This
// records the distinct TEX0 tuples seen per key and prints every key that has more than one.
// ★ cont.332l PS2X_GS_GLR_TEXKEY (default 1 = the injective key): `=0` restores the old XOR key
// so the A/B is ONE BINARY with a flag, not two builds.
// ★★★★ cont.332m PS2X_GS_WRAPCENSUS (default OFF, read-only): how many DRAWS use each PS2
// CLAMP mode pair. The GL path collapses four modes (REPEAT / CLAMP / REGION_CLAMP /
// REGION_REPEAT) to two, and REGION_REPEAT is `u = (u & MINU) | MAXU` -- not expressible as a GL
// wrap mode at all. A draw whose coordinates leave [0,1] then samples a different texel in each
// renderer, which is what the character-select speckle turned out to be.
static const bool g_gsWrapCensus = []
{ const char *e = std::getenv("PS2X_GS_WRAPCENSUS"); return e && e[0] && e[0] != '0'; }();
static std::atomic<unsigned long long> g_wrapPair[16];
static std::atomic<unsigned long long> g_wrapDraws{0};
static const bool g_glTexKeyPacked = []
{ const char *e = std::getenv("PS2X_GS_GLR_TEXKEY"); return !(e && e[0] == '0'); }();
static const bool g_glKeyCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_KEYCENSUS"); return e && e[0] && e[0] != '0'; }();
struct GlKeyTuple { uint32_t tbp, tbw, psm, tw, th, cbp; };
static std::unordered_map<uint64_t, std::vector<GlKeyTuple>> g_glKeyMap;
static std::mutex g_glKeyMutex;
static unsigned long long g_glKeyCollisions = 0;
static const int g_glZMode = []
{ const char *e = std::getenv("PS2X_GS_GLR_ZMODE"); return e && e[0] ? std::atoi(e) : 1; }();
// Default 1 (mask to the format's width) -- measured better than clamping (distinct colours per
// matched frame 10236 -> 13036) and it is what the hardware stores. Mode 2 (scale by 2^32) was
// measured WORSE (9554) and is kept only so the comparison can be repeated.
static double g_glZSum = 0.0;
// ★★ cont.329g THE PIPELINE CENSUS (PS2X_GS_GLR_CENSUS=1, default OFF) -- the BACKEND half.
// The device half counts what survives inside GL; this half counts what the TRANSLATOR hands it,
// stage by stage, so the two logs bracket the whole path: guest primitive -> translated primitive
// -> vertex -> state group -> flushed batch -> (device) draw call -> depth-passed fragment.
// The one number here that no existing instrument reports is STRANDED GEOMETRY: batches are
// flushed on a vertex threshold or a target change and NOTHING flushes them at a frame boundary,
// so whatever is still in the accumulator when the presenter resolves the frame is drawn into the
// NEXT one. Counting it says how much of each frame that is.
// ⚠ COUNTING ONLY -- no behaviour change; these are plain counters on the single submitting
// thread (glSubmitRun already assumes that: m_glPendVerts is unsynchronised).
static const bool g_glCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_CENSUS"); return e && e[0] && e[0] != '0'; }();
// ★★★ cont.329h PS2X_GS_GLR_TGTSPLIT (default ON; `=0` restores the old behaviour for an A/B):
// split a run on its RENDER TARGET before translating it, as RasterRunFanOut does for the CPU
// rasterizer. Measured on the deterministic replay: 4,323 runs per 200 s mixed two framebuffers
// and 7,977,252 primitives were drawn into the wrong one -- including a post-process chain whose
// invert/fill/blur passes landed on the VISIBLE frame and washed it to white (presented-frame
// luma 169-199 against the software path's 34-52; with the split, 10-35).
// ★★★ cont.329i, both default OFF while they are being measured:
//   PS2X_GS_GLR_SRCTEXA (default ON) -- a promoted CT24 target-as-source takes its alpha from
//                            TEXA, expanded by AEM, exactly as the CPU sampler does. A 24-bit
//                            source HAS no alpha; ours read the target's own (1.0 from the clear,
//                            never written) and the composite's blend Cd*(1-As) therefore wiped
//                            the whole visible buffer BLACK. Measured at matched guest flips:
//                            luma 22.8 -> 32.2, 10.5 -> 19.7, 31.2 -> 42.5, 35.4 -> 44.2,
//                            45.4 -> 52.7 against the CPU's 41.3 / 33.8 / 51.6 / 50.4 / 56.0,
//                            and distinct colours +25%. `=0` restores the old behaviour.
//   PS2X_GS_GLR_SRCNEUTRAL-- a promoted PALETTED target-as-source (the T4HL channel shuffle)
//                            samples neutral white. DIAGNOSTIC: it sizes that composite's
//                            contribution, it does not implement it.
// ★★★ cont.329j PS2X_GS_GLR_MIPS, BACKEND HALF (default ON): decode and ship the game's
// own mip chain. 74.4% of this game's textured draws carry TEX1 MMIN=5 (trilinear) with MXL=3, and
// the GL path sampled level 0 for every one of them while the CPU rasterizer sampled the level the
// GS selects -- the largest remaining sampling difference between the two renderers.
// ⚠ cont.332h: the DEVICE half (gs_gpu_device.cpp `glrMips`, which selects the mipmap MIN filter)
// read this SAME env var with the OPPOSITE default until cont.332h, so shipping the chain bought
// nothing at pure defaults. Both halves are default ON now; keep them in step.
// ★★★ cont.329j PS2X_GS_GLR_PALTGT (default OFF = the fix): =1 restores promoting a PALETTED
// target-as-source read to sampling the GL target, which cannot express a CLUT lookup of the
// buffer's alpha nibble.
// ★★★ cont.329k PS2X_GS_GLR_TGTFRESH (default ON = the fix): promote a target-as-source read only
// while the GL target is still FRESHER than VRAM. Compared by the page-generation sum over the
// target's first pages, which uploads and transfers bump.
static const bool s_glTgtFresh = []
{ const char *e = std::getenv("PS2X_GS_GLR_TGTFRESH"); return !(e && e[0] == '0'); }();
static const bool s_glPalTgt = []
{ const char *e = std::getenv("PS2X_GS_GLR_PALTGT"); return e && e[0] && e[0] != '0'; }();
// ★★★ cont.329k PS2X_GS_GLR_CLUTGEN (default ON = the fix): fold the CLUT's page generation into
// the decoded-texture cache's generation, so a palette re-upload actually triggers the content
// check. Without it a palette-animated texture is frozen at its first decode.
static const bool s_glClutGen = []
{ const char *e = std::getenv("PS2X_GS_GLR_CLUTGEN"); return !(e && e[0] == '0'); }();
// ★★★ cont.332 PS2X_GS_GLR_DECVERIFY (default OFF -- it hashes every decoded texture, ~90 KB per
// decode, so it is a diagnostic, not production): hash the DECODED RGBA and compare it with what
// this cache key produced last time. `hash` is taken over whole 8 KB VRAM pages, so a write
// anywhere in a shared page invalidates a texture whose own texels did not move; this counts how
// often that happens. PCSX2 keeps its hash per SOURCE and its dirty rects per TARGET
// (GSTextureCache::Source::m_hash / Target::m_dirty) for exactly this reason.
static const bool s_glDecVerify = []
{ const char *e = std::getenv("PS2X_GS_GLR_DECVERIFY"); return e && e[0] && e[0] != '0'; }();
// ★★★★ cont.332g PS2X_GS_GLR_FLATTEX (default **ON**; "=0" disables): a DETECTOR for the reported
// missing-face / missing-chest-texture glitch, which the user reports is INTERMITTENT -- so looking
// at frames cannot settle it (a clean run proves nothing) and a single-run A/B is meaningless.
// A texture that decodes UNIFORM is what a blank face or a floating pale quad looks like, so count
// them: how many decodes come out flat, how many distinct texture keys, and -- the question that
// matters -- how many of those keys are ever re-decoded into something non-flat afterwards. If a
// flat decode is cached and never recovers, the invalidation missed the upload that filled it, and
// that is the bug. Default ON because it is a live-correctness detector and the cost is a STRIDED
// sample (every 64th texel) per decode, not a full pass; logging is capped at 32 lines.
// ⚠ If a glitchy run reports ZERO flat decodes, the texture is fine and the fault is on the DRAW
// side (collapsed UVs sampling one texel look identical on screen) -- the census discriminates.
static const bool s_glFlatTex = []
{ const char *e = std::getenv("PS2X_GS_GLR_FLATTEX"); return !(e && e[0] == '0'); }();
static unsigned long long g_glFlatDecodes = 0, g_glFlatKeys = 0, g_glFlatRecovered = 0;
// ★★★ cont.329k PS2X_GS_GLR_UVPROBE (default OFF): print the texture state and the EMITTED
// texture coordinates of the first wide textured draws. The pre-rendered background screens come
// out TILED 4x horizontally in GL -- each strip sampling the whole image instead of its own slice
// -- which is a texture-coordinate scaling bug the gameplay comparison never showed.
static const int s_glUvProbe = []
{ const char *e = std::getenv("PS2X_GS_GLR_UVPROBE"); return (e && e[0]) ? std::atoi(e) : 0; }();
static int g_glUvShown = 0;
static const int s_glUvProbeTW = []
{ const char *e = std::getenv("PS2X_GS_GLR_UVPROBETW"); return (e && e[0]) ? std::atoi(e) : 0; }();
static const float s_glUvProbeW = []
{ const char *e = std::getenv("PS2X_GS_GLR_UVPROBEW"); return (e && e[0]) ? float(std::atof(e)) : 200.0f; }();
static const bool s_glMips = []
{ const char *e = std::getenv("PS2X_GS_GLR_MIPS"); return !(e && e[0] == '0'); }();
static const bool s_glSrcTexa = []
{ const char *e = std::getenv("PS2X_GS_GLR_SRCTEXA"); return !(e && e[0] == '0'); }();
static const bool s_glSrcNeutral = []
{ const char *e = std::getenv("PS2X_GS_GLR_SRCNEUTRAL"); return e && e[0] && e[0] != '0'; }();
static const bool s_glTgtSplit = []
{ const char *e = std::getenv("PS2X_GS_GLR_TGTSPLIT"); return !(e && e[0] == '0'); }();
// ★★★ cont.330 PS2X_GS_GLR_TEXTRACE=<tbp> (default 0 = OFF, read-only diagnostic): trace ONE
// texture's cache decisions, and every VRAM mutation that touches its pages, for the FMV 4x-tiling
// bug (cont.329k). The movie draws four 128px quads that all sample tbp=12291, and the run log
// shows the GL renderer decoded that texture ONCE across 32 movie frames -- so either nothing
// bumps its page generation, or the generation moves but the content hash does not. Those two
// have completely different fixes (a missing bump at a writer, versus the picture not living at
// the pages we hash), and inference cannot separate them: this prints the three quantities that
// actually decide it -- the summed page generation, the content hash, and whether a decode ran --
// plus the writer of any mutation landing in the same pages.
// ★★★ cont.330 PS2X_GS_GLR_TGTSEED (default **OFF** -- INCOMPLETE, see below): seed a GL render
// target from VRAM, per dirty PAGE, before the target is drawn into. See GsGlTargetSeed in
// gs_gpu_device.h for why -- a GL target holds only what GL has drawn into it, so a guest UPLOAD
// into the target's pages is invisible to it.
//
// ⚠ DEFAULT OFF because it does not yet work end to end, and it is not free.
//   - The BACKEND half is verified: `[gs2:glrseed/t]` reports target=0x180 h=512 fbw=8 psm=0x00,
//     114 dirty pages, 114 rects, and 222,888 of 233,472 decoded texels NON-ZERO -- i.e. the
//     per-page dirty set and the VRAM decode are both right and carry real picture data.
//   - The DEVICE half does not land: with this ON, `PS2X_GS_GLR_DUMPTGT` still dumps target 0x180
//     as pure black (mean 0.0) on every dump. The blit in glrApplyTargetSeeds is the suspect --
//     framebuffer completeness / read-buffer state / the destination row flip are all unverified.
//   - And it is NOT sufficient for the bug it was built for: an A/B in one binary
//     (PS2X_GS_GLR_TGTSEED=1 vs =0, deterministic replay, matched guest flips) leaves the FMV
//     screens byte-for-byte equally tiled at flips 950 and 1050, so a second, independent cause
//     produces the visible tiling.
//   - Cost when on: ~114 page decodes (64x32 texels each) per seed event.
// ★★★★ cont.331 PS2X_GS_GLR_AUTHORITY (default OFF, phase A of the authority model): apply a
// target's pending dirty VRAM rects when the target is USED -- specifically at the present resolve
// -- not only when the guest draws into it. Without this a target that the guest rarely draws into
// is never brought up to date, which is exactly why the FMV (macroblock uploads into the display
// buffer, a fade drawn over it) presents black under the GL renderer. See progress.md cont.330h.
// ★★★★★ cont.331q DEFAULT ON. It could not be measured honestly before: the GL path was
// ignoring FRAME.FBMSK, so a depth-only FBMSK=ffffffff quad zeroed the target every frame and every
// seed died no matter where it was placed. With that fixed (gs_gpu_device.cpp, fork row 134), a
// seed path is the ONLY thing that puts the background picture into a GL frame -- at defaults the
// whole frame measures ~19.7 = the clear colour plus glyphs, i.e. NO BACKGROUND AT ALL, against the
// CPU reference's ~93.
static const bool s_glAuthority = []
{ const char *e = std::getenv("PS2X_GS_GLR_AUTHORITY"); return !(e && e[0] == '0'); }();
static unsigned long long g_glPresentSeedEvents = 0, g_glPresentSeedRects = 0;
// ★ cont.331: WHY the present-time seed declined, counted per reason. A bare "events=0" cannot
// distinguish "nothing was dirty" from "the path is gated off somewhere" -- and this arc has
// already lost six cycles to instruments whose silence read like a result.
static unsigned long long g_psNoVram = 0, g_psSeqSame = 0, g_psNoRects = 0, g_psNoH = 0, g_psCalls = 0;
// ★ cont.331d: the parameters the present-time seed DECODES with, for direct comparison against
// the [gs2:mutcensus] "WROTE psm/bw" lines.
static uint32_t g_psLastFbp = 0xFFFFFFFFu, g_psLastFbw = 0, g_psLastPsm = 0xFFu;
// ★★★★ cont.331i PS2X_GS_GLR_FLIPRESOLVE (default OFF): present the PRE-FLIP display buffer, as
// the CPU path's flip snapshot does. cont.165: this game issues 128 glyph draws per inter-flip
// window, ALWAYS AFTER the flip, onto the buffer that is being displayed. The CPU captures VRAM at
// flip N paired with the PRE-FLIP registers, so the presented frame includes glyphs drawn during
// the window that just ended. The GL path instead resolves the LIVE display register, which after
// the flip already points at the NEXT buffer -- so the glyphs are never presented. That is the
// user-reported "press start button" bug (cont.331h): UI text missing, everything else exact.
// ★★★★ cont.345 PS2X_GS_GLR_SNAPRESOLVE (default ON; =0 restores the live resolve): the GL display
// target is resolved BY THE WORKER at the flip-snapshot marker (FIFO behind everything the guest
// submitted before the flip, ahead of its next frame) and filed under the snapshot's seq; the
// presenter then shows that frame. See GsGpuPresentDevice::SnapshotFrameGl.
static const bool s_glSnapResolve = []
{ const char *e = std::getenv("PS2X_GS_GLR_SNAPRESOLVE"); return !(e && e[0] == '0'); }();
static const bool s_glFlipResolve = []
{ const char *e = std::getenv("PS2X_GS_GLR_FLIPRESOLVE"); return e && e[0] && e[0] != '0'; }();
static std::atomic<uint32_t> g_glPreFlipFbp{0xFFFFFFFFu};
static std::atomic<unsigned long long> g_glFlipResolveUsed{0};
// ★★★★★ cont.331q DEFAULT ON, and this RETRACTS the cont.331o note that stood here.
// cont.331o reverted this to OFF because turning it on took the title screen from whole-frame mean
// 92.5 to 1.0, and blamed the frame's own `Cd*(1-As)` darken pass. That was wrong. The real cause
// was that the GL path ignored FRAME.FBMSK, so a full-screen DEPTH-ONLY quad (`FBMSK=ffffffff`,
// which writes no colour at all on hardware) was drawn at full strength and ZEROED the target --
// one draw BEFORE the darken pass, which is also why cont.331p's darken-pass ablation changed
// nothing. Seeding early is the physically correct order (the guest's uploaded background, then its
// draws composited on top), and with the mask honoured it is what the frame wants: seeds land at
// 92.4734, the depth-only quad leaves them untouched, and the UI glyphs compose on top to 93.4685
// against the CPU reference's 93.5.
//
// Measured over 270 densely captured flips (PS2X_GS_PRESENT_SAVE_BYFLIP=10), seeds ON vs OFF:
// mean whole-frame advantage +37.6, and only 2 of 270 samples favour OFF -- both single-sample
// ERA-BOUNDARY transients (flip 950 is the EA intro ending one 10-flip sample early; its
// neighbours tie at 940 and are both black at 960). ⚠ An earlier single-flip read of that same
// boundary looked like a real regression at flip 900 (34.08 vs 50.15) and was NOT: the EA intro is
// a host-paced FMV, so a guest flip index does not pin the movie frame, and a re-measure put seeds
// ON *ahead* at that flip (60.24 vs 53.23). Sample an era, never a flip.
static const bool s_glTgtSeed = []
{ const char *e = std::getenv("PS2X_GS_GLR_TGTSEED"); return !(e && e[0] == '0'); }();
static const bool s_glSeedTest = []
{ const char *e = std::getenv("PS2X_GS_GLR_SEEDTEST"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_glSeedEvents = 0, g_glSeedPages = 0, g_glSeedSkippedNoH = 0;
static const uint32_t s_glTexTrace = []
{ const char *e = std::getenv("PS2X_GS_GLR_TEXTRACE"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
// ★★★ cont.330e PS2X_GS_GLR_ASPROBE=<tbp> (default 0 = OFF, read-only): for the first
// PS2X_GS_GLR_ASPROBEN draws sampling that TBP, print EVERY input to the draw's colour
// contribution at once -- blend selectors, TEXA, TCC/TFX, the promotion's mode, and the decoded
// texture's ALPHA distribution. cont.330d eliminated the texture SOURCE (the draw contributes
// nothing whatever it resolves to), which leaves the alpha: Cs*As + Cd*(1-As) with As=0 is
// invisible for any Cs. Four single-variable knobs each produced a null; this prints the whole
// decision in one line instead.
static const uint32_t s_glAsProbe = []
{ const char *e = std::getenv("PS2X_GS_GLR_ASPROBE"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static const int s_glAsProbeN = []
{ const char *e = std::getenv("PS2X_GS_GLR_ASPROBEN"); return (e && e[0]) ? std::atoi(e) : 12; }();
static int g_glAsShown = 0;
static uint32_t g_glAsMin = 255, g_glAsMax = 0;
static unsigned long long g_glAsSum = 0, g_glAsN = 0;
// ★★★ cont.330e PS2X_GS_GLR_TEXAEM (default OFF = today's behaviour): apply the FULL TEXA
// rule to a CT24/CT16 decode, as the CPU sampler does (gs_cpu_backend.cpp applyTexa: CT24 alpha is
// `(aem && rgb==0) ? 0 : TA0`, CT16 picks TA1 when the high bit is set). The GL decode currently
// writes TA0 unconditionally and ignores AEM entirely -- a KNOWN approximation (cont.329j's census
// counts `texa-aem-draws`). Default OFF so the A/B is one binary; PCSX2 GSLocalMemory's
// ReadTexel/Expand paths apply the same rule.
static const bool s_glTexAem = []
{ const char *e = std::getenv("PS2X_GS_GLR_TEXAEM"); return e && e[0] && e[0] != '0'; }();
// ★★★ cont.330e PS2X_GS_GLR_DUMPTEX=<dir> + PS2X_GS_GLR_DUMPTEXTBP=<tbp> (default OFF):
// write the DECODED RGBA the GL path is about to upload, as a PPM, for the first
// PS2X_GS_GLR_DUMPTEXN decodes of that TBP. Two cycles have now been spent re-interpreting probe
// TEXT about this texture; a picture of what GL actually binds settles it in one look. Alpha is
// written as a second, greyscale PPM because the whole question is whether As kills the draw.
static const char *const s_glDumpTexDir = std::getenv("PS2X_GS_GLR_DUMPTEX");
static const uint32_t s_glDumpTexTbp = []
{ const char *e = std::getenv("PS2X_GS_GLR_DUMPTEXTBP"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static const int s_glDumpTexN = []
{ const char *e = std::getenv("PS2X_GS_GLR_DUMPTEXN"); return (e && e[0]) ? std::atoi(e) : 4; }();
// cont.332m: 0 = any CLUT (the old behaviour); otherwise only the draw whose TEX0.CBP matches.
static const uint32_t s_glDumpTexCbp = []
{ const char *e = std::getenv("PS2X_GS_GLR_DUMPTEXCBP"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static int g_glDumpTexShown = 0;
// ★★★ cont.355d PS2X_GS_GLR_DUMPTEXANY=1: dump EVERY distinct texture of the probe flip window
// instead of one chosen TBP, de-duplicated by CONTENT HASH and capped at DUMPTEXN. Picking a
// texture to replace previously meant guessing a tbp from the FRAMEHIST census, which reports only
// the first TEX0 of each draw STATE -- so most of a frame's textures were invisible. An HD pack
// needs the inventory, not a guess.
static const bool s_glDumpTexAny = []
{ const char *e = std::getenv("PS2X_GS_GLR_DUMPTEXANY"); return e && e[0] && e[0] != '0'; }();
static uint64_t g_glDumpTexSeen[256];
static unsigned g_glDumpTexSeenN = 0;
static bool glDumpTexFirstSighting(uint64_t h)
{
    for (unsigned i = 0; i < g_glDumpTexSeenN; ++i)
        if (g_glDumpTexSeen[i] == h) return false;
    if (g_glDumpTexSeenN < 256u) g_glDumpTexSeen[g_glDumpTexSeenN++] = h;
    return true;
}
// ★★★ cont.330e PS2X_GS_PROBEFLIP=<n> [+ PS2X_GS_PROBEFLIPW=<w>, default 2]: restrict the
// per-draw probes (ASPROBE, UVPROBE, DUMPTEX) to guest DISPLAY FLIPS [n, n+w). THREE conclusions
// this cycle were wrong because a capped probe sampled the BOOT era while the screen under
// investigation is at flip 950: "the strips all sample s in [0..1]" (UVPROBE's first 60), "GL's
// texture is black" (DUMPTEX's first 4), and a width gate that matched nothing at all. A probe
// must be aimed at the instant being investigated, not at the first N draws of the run.
static const unsigned long long s_probeFlip = []
{ const char *e = std::getenv("PS2X_GS_PROBEFLIP"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
// ★★★★ cont.330g PS2X_GS_GLR_FRAMEHIST=1 (default OFF, read-only): a CENSUS of the draw
// state over EVERY draw inside the probe flip window, instead of the first N. The first-N sample
// has now produced four wrong readings in this arc; "what do the first 24 draws do" and "what is
// this frame made of" are different questions, and a frame here carries thousands of draws.
// Keyed by the state that decides whether a fragment survives, with the geometry area so a state
// that covers the screen is distinguishable from one that covers nothing.
struct FrameHistRow
{
    uint32_t ate = 0, atst = 0, aref = 0, afail = 0, zte = 0, ztst = 0, abe = 0, fbmsk = 0;
    uint32_t tme = 0, tgt = 0, tfx = 0, tcc = 0, ba = 0, bb = 0, bc = 0, bd = 0;
    // ★★★ cont.355b: the HUD counter-scale for anamorphic widescreen needs to know WHICH draws are
    // 2D screen-space and how wide they are -- a full-screen fade must NOT be counter-scaled, an
    // icon must. FST is the GS's own 2D signal (1 = UV/texel coords, no perspective; 0 = STQ), and
    // the width range separates "covers the raster" from "covers an element". `tbp` is the first
    // TEX0 seen in the row, i.e. a concrete texture to aim PS2X_GS_GLR_DUMPTEX at.
    uint32_t fst = 0, prim = 0, tbp = 0;
    double wMin = 1e30, wMax = 0.0;
    // ★★★★★ cont.356: FST alone does NOT classify 2D here. The rich-HUD census proved every
    // pictorial HUD element (portrait, badges, bars, icons) is FST=0 and only the TEXT is FST=1, so
    // the widescreen HUD counter-scale cannot be built on FST. Two columns decide what replaces it:
    //   q      -- the perspective term. The GS samples at (S/Q, T/Q), so a SCREEN-SPACE draw carries
    //             q == 1 exactly while a depth-placed billboard does not. ⚠ It is MEANINGLESS when
    //             TME=0: no texture lookup happens, so the vertex keeps whatever RGBAQ last held.
    //   x/y    -- the SCREEN bbox, i.e. vertex - (XYOFFSET>>4), which is where a draw actually lands.
    //             ⚠ GSVertex carries GS COORDINATES: unconverted, this game's full-screen sprite
    //             reads [1792..2304] and a screen-rect test returns a confident zero (cont.331j built
    //             a conclusion on exactly that null). The offset used is printed beside the rect so
    //             the unit is visible in the log rather than assumed.
    double qMin = 1e30, qMax = -1e30;
    double xMin = 1e30, xMax = -1e30, yMin = 1e30, yMax = -1e30;
    int ofx = 0, ofy = 0;
    unsigned long long draws = 0;
    double area = 0.0;
};
static FrameHistRow g_frameHist[48];
static unsigned g_frameHistN = 0;
static unsigned long long g_frameHistDropped = 0;
// ★★★★★ cont.356 PS2X_GS_HUD_ASPECT (default ON) -- the 2D/HUD half of anamorphic widescreen.
// Default ON is safe because the correction is IDENTITY unless the picture is being presented at a
// ratio wider than the game's own (PS2X_WINDOW_ASPECT), so at pure defaults nothing changes on any
// title. `=0` disables the classifier outright, which is the A/B control.
// ★★★★★ cont.356c PS2X_GS_PERSPCOUNT=1 (default OFF, read-only): PER-FLIP counts of the signal the
// full-screen 2D correction would key on. The proposal is "this frame drew no perspective geometry
// => the full-screen composite holds 2D art => counter-scale it", and two censuses separate cleanly
// (title screen 0 perspective draws over 100 consecutive flips; level select 11,562). But a census
// samples a window, and the failure this must rule out is a ONE-FRAME aspect pop on a frame that
// merely happens to emit no perspective draws -- a pause, a loading screen, a cross-fade. So count
// every flip of a whole run and read the DISTRIBUTION rather than argue about it.
static const bool s_perspCount = []
{ const char *e = std::getenv("PS2X_GS_PERSPCOUNT"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_pcDraws = 0, g_pcPersp = 0, g_pcFull2d = 0, g_pcNarrow2d = 0;
// ★★★★★ cont.356d PS2X_GS_HUD_ASPECT_FULL (default OFF): extend the aspect correction to the
// FULL-SCREEN 2D composite -- the title/menu artwork, which the width gate deliberately excludes
// because that same gate protects fades and the gameplay scene composite. Per-draw GS state cannot
// tell "target holding 2D art" from "target holding the 3D scene" (measured: identical signatures),
// so the discriminator is the FRAME: a frame that drew no perspective geometry is a 2D screen.
//
// Measured over 16,982 flips (cont.356c): zero flips carry 1-10 perspective draws, gameplay never
// fires, and the only defect is a 12-flip firing during a screen TRANSITION. A symmetric debounce of
// PS2X_GS_HUD_ASPECT_FULL_DEBOUNCE flips (default 16) removes it -- 6 state switches in a whole run,
// shortest era 0.94 s. ⚠ It costs ~0.32 s of lag at screen entry, during which the art is still
// stretched.
// cont.356g: 0 = OFF (default), 1 = DECLARED-only, 2 = AUTO (the heuristic, a bring-up tool).
enum { kFullOff = 0, kFullDeclared = 1, kFullAuto = 2 };
// ★★★★★ cont.359: read LAZILY, on first use -- never at static-initialisation time. A game's
// override may DERIVE this variable (LOTR's widescreen mod setenv()s it from LOTR_WIDESCREEN in
// registerMods, cont.358o), and a namespace-scope static read it before main(), i.e. before any
// override had run: the derived value was silently ignored and every declared 2D screen stayed
// stretched (measured: 0 `[gs2:hudfull]` transitions in every derived run from build 1141 on; the
// same binary with the variable set explicitly engages at flip 13). ps2xWindowAspect() already
// reads its variable this way, which is why the presentation aspect survived the same derivation.
// Both callers run per flip / per draw, i.e. long after overrides are registered.
static int hudFullMode()
{
    static const int m = []
    {
        const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL");
        if (!e || !e[0] || (e[0] == '0' && e[1] == '\0')) return int(kFullOff);
        if (e[0] == 'a' || e[0] == 'A') return int(kFullAuto);
        return int(kFullDeclared);
    }();
    return m;
}
static bool hudAspectFullOn() { return hudFullMode() != kFullOff; }
// What the GAME says it is showing. Unknown until an override declares, so a title with no table
// gets no correction at all -- the engine ships a mechanism, never a policy.
static std::atomic<unsigned> g_declaredKind{0u};
void ps2xGsDeclareScreen(unsigned kind)
{
    g_declaredKind.store(kind, std::memory_order_relaxed);
}
// ★★★★★ cont.358: the game's HUD anchor hook (see gs_cpu_backend.h for the contract). Null until
// a game installs one, and null is the shipped state on every title -- the engine's own default
// (the screen centre) is then used, which is the pre-cont.358 behaviour exactly.
static std::atomic<Ps2xHudAnchorFn> g_hudAnchorFn{nullptr};
void ps2xGsSetHudAnchorFn(Ps2xHudAnchorFn fn)
{
    g_hudAnchorFn.store(fn, std::memory_order_release);
}
// PS2X_GS_HUD_ANCHOR (default ON): honour an installed hook. `=0` ignores it without uninstalling,
// so the A/B control is the SAME BINARY with the flag off -- never a second build.
static const bool s_hudAnchorOn = []
{ const char *e = std::getenv("PS2X_GS_HUD_ANCHOR"); return !(e && e[0] == '0' && e[1] == '\0'); }();
// ★★★★ cont.358c PS2X_GS_HUDDUMP=1 (read-only, default OFF): one line PER CLASSIFIED DRAW inside
// the PS2X_GS_PROBEFLIP window. FRAMEHIST aggregates per STATE, so a layer drawn as many small
// primitives -- this game's text is ~30 glyph draws per flip -- reports ONE union bbox spanning the
// screen, which is precisely the information an anchor policy cannot use (cont.358b: a per-draw
// anchor rule built on that aggregate tore a text block in half). This prints the individual rects.
// ⚠ Bounded, AND IT REPORTS ITS BOUND: the summary says total= vs shown= and marks a truncated dump,
// so a capped sample can never be read as a census.
static const bool s_hudDump = []
{ const char *e = std::getenv("PS2X_GS_HUDDUMP"); return e && e[0] && e[0] != '0'; }();
// ★ cont.358q PS2X_GS_HUDDUMP_TBPMIN (default 0 = dump everything): only dump a draw whose texture
// base is >= this, OR which the policy actually anchored. It exists because the useful draws are
// rare and the useless ones are not: a capture aimed at co-op gameplay spent its whole 400,000-line
// budget on MENU TEXT (tbp=8192, every one declining) and stopped at flip 2251, long before the
// scene it was meant to record. Setting 12000 here drops the shared font atlases and keeps the HUD
// ART -- with its verdict either way, which is what a flicker diagnosis needs -- so one budget now
// covers a whole play session and no flip number has to be guessed in advance.
static const uint32_t s_hudDumpTbpMin = []
{ const char *e = std::getenv("PS2X_GS_HUDDUMP_TBPMIN");
  return uint32_t((e && e[0]) ? std::strtoul(e, nullptr, 10) : 0ul); }();
// ★ cont.358w PS2X_GS_HUDDUMP_YMIN (default 0 = off): also keep a draw whose top edge is at or
// below this y, whatever its texture and whatever the policy decided. TBPMIN keeps the HUD ART and
// anything ANCHORED -- which silently hides the case that matters here: a draw that DECLINED simply
// vanishes from the dump, so "no verdict changes" was being read off a capture that could not
// contain a text draw losing its anchor. Setting 400 keeps the whole HUD band either way.
static const int s_hudDumpYMin = []
{ const char *e = std::getenv("PS2X_GS_HUDDUMP_YMIN");
  return int((e && e[0]) ? std::strtol(e, nullptr, 10) : 0l); }();
static const unsigned long long s_hudDumpMax = []
{ const char *e = std::getenv("PS2X_GS_HUDDUMP_MAX");
  const unsigned long long v = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 4000ull; return v ? v : 4000ull; }();
static std::atomic<unsigned long long> g_hudDumpShown{0}, g_hudDumpTotal{0};
static bool g_hudDumpReported = false;

// ★★★★★ cont.358e PS2X_GS_HUDCLUSTER=1 (read-only, default OFF): group each flip's CLASSIFIED
// draws into connected blocks and report them. This is the de-risking measurement for a
// cluster-based anchor policy, and it is deliberately a CENSUS ONLY -- nothing is applied.
//
// WHY. cont.358d killed the per-draw policy: a draw cannot know where its string ends, so any
// geometric rule cuts strings at its own boundary (measured: one atlas in both the anchored and the
// declined set, the anchored run stopping exactly at the region edge). The fix is to answer per
// BLOCK, so every draw of one UI element shares one anchor and a string cannot be split by
// construction. That only works if proximity clustering actually recovers UI structure on this
// game -- which is a question to MEASURE before building on, not to assume (the mistake cont.358c
// made twice).
//
// METHOD. Union-find over the flip's classified bboxes, unioning any two whose rects come within
// PS2X_GS_HUDCLUSTER_GAP pixels (default 24; glyphs of a word sit 3-16 px apart on this game).
// ⚠ The threshold errs LARGE on purpose: over-merging yields one big cluster whose only sensible
// anchor is "decline", i.e. today's behaviour, while under-merging is the tear we already have.
static const bool s_hudCluster = []
{ const char *e = std::getenv("PS2X_GS_HUDCLUSTER"); return e && e[0] && e[0] != '0'; }();
static const float s_hudClusterGap = []
{ const char *e = std::getenv("PS2X_GS_HUDCLUSTER_GAP"); const double v = (e && e[0]) ? std::atof(e) : 24.0;
  // ★ cont.358m: 24 is load-bearing for CO-OP. The two players' HUDs sit ~34 raster px apart, so a
  // gap much above that would FUSE them into one block taking a single anchor. 24 keeps them
  // distinct and still holds one player's block together (cont.358e sweep: gameplay resolves as
  // exactly 2 blocks at every gap from 16 to 160). Raising this without re-checking co-op is a trap.

  return float(v >= 0.0 ? v : 24.0); }();
struct HudClusterDraw { float x0, y0, x1, y1; uint32_t tbp; };
static std::mutex g_hudClusterMx;
static std::vector<HudClusterDraw> g_hudClusterDraws;

// ★★★★★ cont.358f: the LIVE cluster map -- the census promoted to a feature. Accumulated over a
// flip's classified draws, clustered at the flip, and published for the NEXT flip to look up.
// Gap: PS2X_GS_HUDCLUSTER_GAP, default 24. ⚠ An earlier revision of this comment claimed 128 and
// justified it from a torn-line detector whose 200 px proximity window was itself wrong (real glyph
// gaps here are 3-16 px, so it was flagging separate labels, not split strings -- cont.358i). The
// code always read 24. What actually constrains this value is CO-OP: the two players' HUDs are
// ~34 raster px apart and must not fuse into one block.
struct HudClusterRect { float x0, y0, x1, y1; unsigned id, n;
                        // cont.358j: the distinct texture bases seen in this run (8 = "8 or more")
                        unsigned tbpN = 0; uint32_t tbp[8] = {0,0,0,0,0,0,0,0}; };
// cont.358g: how many classified draws found NO block, within the probe window.
static std::atomic<unsigned long long> g_hudNoCluster{0};
// ★★★★★ cont.358h: THE PER-FRAME GATE -- the property that makes tearing impossible.
// Tearing requires two draws IN ONE FRAME to receive different answers. Every per-draw rescue we
// tried (nearest-cluster, region containment) preserved that possibility and duly tore a string.
// So the decision is made ONCE PER FRAME: the map is used for the whole frame or for none of it.
//   * a frame in which every classified draw was contained in a block  -> the map DESCRIBES the
//     layout, and the next frame may anchor
//   * a frame with even ONE uncontained draw -> the layout has moved out from under the map (the
//     select screens animate), so the next frame declines ENTIRELY and looks exactly as it does
//     with the feature off
// The worst case is therefore "this screen gets no enhancement", never a split string. Gameplay's
// map covers every draw every flip (measured 40/40), so it anchors continuously.
static std::atomic<unsigned long long> g_hudFrameSeen{0}, g_hudFrameMiss{0};
static std::atomic<bool> g_hudGate{false};          // may THIS frame use the map?
static std::atomic<unsigned long long> g_hudGateOn{0}, g_hudGateOff{0};

// ★★★★★ cont.358i — GROUPING BY DRAW ORDER, which is what finally removes the lag from the
// GROUPING (it remains only in the geometry, where it is harmless).
//
// WHY POSITION FAILED, four times. Every position-based scheme groups frame N's draws using frame
// N-1's rectangles, and on an animated screen those rectangles no longer describe the layout: a
// line that clusters as ONE block within its own flip gets split across two stale blocks. Measured
// (cont.358h): clustering flip 1074's own draws puts the two glyphs that tear into a single centred
// cluster which would decline; using flip 1073's published map splits them. The per-frame coverage
// gate cannot see this -- it asks whether every draw is inside SOME block, not whether the blocks
// still mean anything.
//
// WHY ORDER WORKS. Classified draws arrive in EMISSION order, and a string is a consecutive run of
// them sharing a texture atlas. A run boundary -- "the atlas changed, or this draw is further than
// the gap from the previous one" -- is decidable THE MOMENT THE DRAW ARRIVES, with no knowledge of
// the future and no reference to the previous frame. So every glyph of a string provably gets the
// same run index, and a string cannot be split however the screen animates.
//
// The run's GEOMETRY still comes from the previous frame's run of the same index, because a run's
// extent is unknown until it ends. That is safe where position was not: animation moves coordinates
// but does not reorder the strings a screen emits. The guard is structural rather than spatial --
// if the frame's RUN COUNT differs from the previous frame's, the whole next frame declines.
static std::mutex g_hudRunMx;
static unsigned g_runIdx = 0;                    // this frame's current run
static bool     g_runHavePrev = false;           // is there a previous draw in this frame?
static float    g_runPx0 = 0, g_runPy0 = 0, g_runPx1 = 0, g_runPy1 = 0;
static uint32_t g_runPrevAtlas = 0xFFFFFFFFu;
static std::vector<HudClusterRect> g_runThis;    // per-run bbox, this frame
static std::vector<HudClusterRect> g_runPrev;    // per-run bbox, previous frame
static size_t   g_runPrevCount = 0;
static std::atomic<unsigned long long> g_runSplitFrames{0};
static std::mutex g_hudMapMx;
static std::vector<HudClusterRect> g_hudMap;       // the PREVIOUS flip's blocks
static std::vector<HudClusterDraw> g_hudPending;   // this flip's classified draws, accumulating

// Group `v` into connected blocks: two rects union when they come within `g` px in BOTH axes.
static std::vector<HudClusterRect> hudBuildClusters(const std::vector<HudClusterDraw> &v, float g)
{
    std::vector<HudClusterRect> out;
    const size_t n = v.size();
    if (!n) return out;
    std::vector<size_t> parent(n);
    for (size_t i = 0; i < n; ++i) parent[i] = i;
    std::function<size_t(size_t)> find = [&](size_t a)
    { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (v[i].x0 - g <= v[j].x1 && v[j].x0 - g <= v[i].x1 &&
                v[i].y0 - g <= v[j].y1 && v[j].y0 - g <= v[i].y1)
            {
                const size_t a = find(i), b = find(j);
                if (a != b) parent[a] = b;
            }
    std::map<size_t, HudClusterRect> acc;
    for (size_t i = 0; i < n; ++i)
    {
        const size_t r = find(i);
        auto it = acc.find(r);
        if (it == acc.end())
            acc.emplace(r, HudClusterRect{v[i].x0, v[i].y0, v[i].x1, v[i].y1, 0u, 1u});
        else
        {
            HudClusterRect &c = it->second;
            c.x0 = std::min(c.x0, v[i].x0); c.y0 = std::min(c.y0, v[i].y0);
            c.x1 = std::max(c.x1, v[i].x1); c.y1 = std::max(c.y1, v[i].y1);
            ++c.n;
        }
        {   // cont.358l: the distinct textures in this block -- how a policy NAMES it
            HudClusterRect &c = acc.find(r)->second;
            bool known = false;
            for (unsigned t = 0; t < c.tbpN; ++t) if (c.tbp[t] == v[i].tbp) { known = true; break; }
            if (!known && c.tbpN < 8u) c.tbp[c.tbpN++] = v[i].tbp;
        }
    }
    unsigned id = 1u;   // ★ 1-based: 0 is reserved for "unknown", which a policy must decline
    for (auto &kv : acc) { kv.second.id = id++; out.push_back(kv.second); }
    return out;
}

// Cluster the flip's classified draws and print one line per block. Called at the display flip.
static void dumpHudClusters(unsigned long long flip)
{
    std::vector<HudClusterDraw> v;
    {
        std::lock_guard<std::mutex> lk(g_hudClusterMx);
        v.swap(g_hudClusterDraws);
    }
    if (v.empty())
    {
        std::fprintf(stderr, "[gs2:hudclust] flip=%llu  no classified draws\n", flip);
        return;
    }
    const size_t n = v.size();
    std::vector<size_t> parent(n);
    for (size_t i = 0; i < n; ++i) parent[i] = i;
    std::function<size_t(size_t)> find = [&](size_t a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    const float g = s_hudClusterGap;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
        {
            // Rects within `g` of each other in BOTH axes (an inflated-rect overlap test).
            if (v[i].x0 - g <= v[j].x1 && v[j].x0 - g <= v[i].x1 &&
                v[i].y0 - g <= v[j].y1 && v[j].y0 - g <= v[i].y1)
            {
                const size_t a = find(i), b = find(j);
                if (a != b) parent[a] = b;
            }
        }
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t i = 0; i < n; ++i) groups[find(i)].push_back(i);
    std::fprintf(stderr, "[gs2:hudclust] flip=%llu  draws=%zu -> %zu clusters (gap=%.0f)\n",
                 flip, n, groups.size(), double(g));
    size_t k = 0;
    for (const auto &kv : groups)
    {
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        std::set<uint32_t> tbps;
        for (size_t i : kv.second)
        {
            x0 = std::min(x0, v[i].x0); y0 = std::min(y0, v[i].y0);
            x1 = std::max(x1, v[i].x1); y1 = std::max(y1, v[i].y1);
            tbps.insert(v[i].tbp);
        }
        char tb[160]; int off = 0; tb[0] = '\0';
        for (uint32_t t : tbps) { off += std::snprintf(tb + off, sizeof(tb) - size_t(off), "%s%u", off ? "," : "", t);
                                  if (off >= int(sizeof(tb)) - 8) break; }
        std::fprintf(stderr, "[gs2:hudclust]   #%zu n=%-4zu x=[%.0f..%.0f] y=[%.0f..%.0f] w=%.0f h=%.0f tbps={%s}\n",
                     k++, kv.second.size(), x0, x1, y0, y1, x1 - x0, y1 - y0, tb);
    }
}

// The engine's default anchor, as the GsGlVertex fixed point: 32768 -> (32768-1)/65534 = exactly
// 0.5, the screen centre. Declining draws and hook-less games keep this value.
static constexpr uint16_t kHudAnchorCentre = 32768u;
// Encode a normalised [0,1] anchor into that fixed point. Out-of-range values are CLAMPED rather
// than rejected: a game asking to anchor beyond an edge means the edge.
static inline uint16_t hudAnchorEncode(float a)
{
    if (!(a >= 0.0f)) return kHudAnchorCentre;          // also catches NaN
    if (a > 1.0f) a = 1.0f;
    return static_cast<uint16_t>(a * 65534.0f + 1.0f);
}
// ★★★★★ cont.356f: the debounce is ASYMMETRIC, and a frame with no evidence HOLDS rather than
// votes. A symmetric 16-flip debounce shipped a visible artefact (user report): leaving a 2D screen
// for gameplay left the verdict ON for 16 flips, so the 3D scene was presented PILLARBOXED and then
// snapped to full width -- "loads compressed, then stretches". The directions are NOT equally bad: a
// false ON pillarboxes a 3D scene and is glaring, a late ON merely leaves 2D art stretched a moment
// longer, which is what happens with the feature off anyway. So OFF is immediate on real evidence of
// 3D, ON is debounced. Simulated over 21,389 recorded flips: symmetric 16/16 gives 45 frames of
// wrongly-pillarboxed 3D; ON=4 / OFF=1 / hold / minUI=8 gives ZERO, at a cost of 9 frames of
// briefly-stretched 2D across the whole run.
static const unsigned s_hudFullDebounce = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_DEBOUNCE");
  const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 4; return unsigned(v > 0 ? v : 4); }();
static const unsigned s_hudFullOffDebounce = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_OFFDEB");
  const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 1; return unsigned(v > 0 ? v : 1); }();
// A frame must carry real screen-space UI to VOTE for a 2D screen. Without this, near-empty
// transition frames (2-10 draws, a composite, no UI) voted 2D and produced a 12-flip false ON.
// cont.356l: flips to keep a 2D screen corrected after its declaration turns 3D, for the handover
// window in which the game has requested the next screen but is still drawing the current one. A
// frame that draws perspective ends it immediately, so the bound only matters when nothing 3D is
// being drawn at all.
static const unsigned s_hudFullHold = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_HOLD");
  const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 8; return unsigned(v >= 0 ? v : 8); }();
// ★ cont.356o: flips a STRICT declaration must be stable before it engages. The screen request
// arrives a few frames BEFORE the game stops drawing the previous screen (measured at 3 flips for
// the title handover), and a strict entry has no evidence gate to hold it back -- so it would
// pillarbox the OUTGOING screen for those frames, seen as 'the screen changes a bit before moving
// to the save list'. Plain 2D needs no such delay: its persp == 0 gate cannot pass while the old
// 3D screen is still drawing.
static const unsigned s_hudFullOnDelay = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_ONDELAY");
  const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 8; return unsigned(v >= 0 ? v : 8); }();
// ★ cont.362: ...or engage on the first flip that draws NO perspective geometry, whichever comes
// first. That flip is direct evidence the outgoing 3D screen has stopped -- the very thing the
// delay waits for -- so it cannot pillarbox the outgoing screen the delay exists to protect.
// Measured on the id=10 -> 7 (load game) handover: the backdrop draws persp=26 for 3 flips, then
// the new screen is black for 2 and FADES IN over 2 more, all of which the fixed 8-flip delay left
// stretched (user-reported: 'a split second of stretched background'). With this it engages on
// the first black flip. PS2X_GS_HUD_ASPECT_FULL_ONEVIDENCE=0 restores the delay alone.
static const bool s_hudFullOnEvidence = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_ONEVIDENCE"); return !(e && e[0] == '0'); }();
static const unsigned s_hudFullMinUi = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FULL_MINUI");
  const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 8; return unsigned(v >= 0 ? v : 8); }();
// Per-flip accumulators, and the DEBOUNCED verdict the next flip's draws are classified against.
// The verdict necessarily lags by one flip: a frame's perspective total is not known until the
// frame ends, and the composite is drawn inside it. With a 16-flip debounce that is immaterial.
static unsigned long long g_fsPersp = 0, g_fsFull2d = 0, g_fsNarrow2d = 0;
static std::atomic<bool> g_fs2dActive{false};
static bool g_fsLastRaw = false;
static unsigned g_fsRun = 0;
static unsigned long long g_fsApplied = 0;   // draws the full-screen arm actually tagged
static const bool s_hudAspect = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT"); return !(e && e[0] == '0' && e[1] == '\0'); }();
// The full-screen exclusion: a 2D draw wider than this fraction of the scissor is a fade/composite
// and must not be counter-scaled. 0.9 measured; a game whose HUD spans the width would raise it.
static const double s_hudAspectFrac = []
{ const char *e = std::getenv("PS2X_GS_HUD_ASPECT_FRAC"); double v = (e && e[0]) ? std::atof(e) : 0.9;
  return (v > 0.0 && v <= 1.0) ? v : 0.9; }();
static const bool s_frameHist = []
{ const char *e = std::getenv("PS2X_GS_GLR_FRAMEHIST"); return e && e[0] && e[0] != '0'; }();
static const unsigned long long s_probeFlipW = []
{ const char *e = std::getenv("PS2X_GS_PROBEFLIPW"); unsigned long long v = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 2ull; return v ? v : 2ull; }();

// ★★★ cont.355b: FRAMEHIST is dumped from here, not from its original site inside
// CopyFrameToHostRgba -- that is a CPU-ARM path the GL renderer never takes, so with
// PS2X_GS_RENDERER=gl (our default) the census recorded every draw and printed NOTHING. An
// instrument has to live where BOTH arms reach it; the flip counter is the renderer-agnostic clock
// (cont.331s). Fired once, on the first flip past the probe window.
static bool g_frameHistDumped = false;
static void dumpFrameHist()
{
    if (!s_frameHist || !g_frameHistN) return;
    unsigned long long tot = 0; double totA = 0.0;
    for (unsigned r = 0; r < g_frameHistN; ++r)
    { tot += g_frameHist[r].draws; totA += g_frameHist[r].area; }
    std::fprintf(stderr,
                 "[gs2:framehist] EVERY draw in the probe flip window: %llu draws, %.0f px, "
                 "%u distinct states%s\n", tot, totA, g_frameHistN,
                 g_frameHistDropped ? " (⚠ table full, some states dropped)" : "");
    for (unsigned r = 0; r < g_frameHistN; ++r)
    {
        const FrameHistRow &q = g_frameHist[r];
        std::fprintf(stderr,
                     "[gs2:framehist]   draws=%-7llu %5.1f%%  px=%-12.0f %5.1f%%  "
                     "TEST{ate=%u atst=%u aref=%3u afail=%u zte=%u ztst=%u} "
                     "abe=%u ABCD=%u%u%u%u FBMSK=%08x tme=%u tgt=%u tfx=%u tcc=%u "
                     "| FST=%u prim=%u w=[%.0f..%.0f] tbp=%u "
                     "q=[%.4f..%.4f] screen x=[%.0f..%.0f] y=[%.0f..%.0f] (gs-ofs %d,%d)\n",
                     q.draws, 100.0 * double(q.draws) / double(tot ? tot : 1),
                     q.area, 100.0 * q.area / (totA > 0.0 ? totA : 1.0),
                     q.ate, q.atst, q.aref, q.afail, q.zte, q.ztst,
                     q.abe, q.ba, q.bb, q.bc, q.bd, q.fbmsk, q.tme, q.tgt,
                     q.tfx, q.tcc, q.fst, q.prim,
                     (q.wMax > 0.0 ? q.wMin : 0.0), q.wMax, q.tbp,
                     (q.qMax > -1e29 ? q.qMin : 0.0), (q.qMax > -1e29 ? q.qMax : 0.0),
                     (q.xMax > -1e29 ? q.xMin : 0.0), (q.xMax > -1e29 ? q.xMax : 0.0),
                     (q.yMax > -1e29 ? q.yMin : 0.0), (q.yMax > -1e29 ? q.yMax : 0.0),
                     q.ofx, q.ofy);
    }
}
// ★★★★ cont.330e PS2X_GS_GLR_TEXSPLIT (default ON = the FIX; =0 restores the bug for an A/B).
// The device applies ALL of a batch's texture uploads BEFORE any draw in that batch, into ONE GL
// object per key ("contents replaced in place" -- an epoch in the key was measured at 5866 objects
// and 77% of the device thread, so it is not an option). The pre-rendered/FMV screens re-upload a
// DIFFERENT 128x512 slice to the SAME tbp before each of four strip draws, so all four ended up
// sampling the LAST slice -- the 4x tiling. Fix: if an arriving upload targets a key that a group
// ALREADY PENDING in this batch binds, flush the batch first, so those earlier draws keep the
// contents they were built against. Mirrors PCSX2 invalidating a source on an overlapping transfer
// with the draw queue flushed first (GSTextureCache::InvalidateVideoMem).
static const bool s_glTexSplit = []
{ const char *e = std::getenv("PS2X_GS_GLR_TEXSPLIT"); return !(e && e[0] == '0'); }();
// ★★★★★ cont.331k PS2X_GS_GLR_FLIPFLUSH (default ON; =0 restores the old behaviour for an A/B).
// THE GUEST FRAME BOUNDARY IS A FLUSH POINT. The GL path coalesces draws into m_glPendVerts and
// ships a batch only on a 512-vertex threshold, a target change or a texture overwrite -- nothing
// flushed on a frame boundary. In the menu/title era a whole guest frame is ~209 vertices, so most
// of a frame's geometry was still pending when that frame was resolved: the corrected stranded
// census (see g_glPendGlobal) measured 898,200 of 1,552,008 vertices -- 58%, mean 121 per resolve
// -- left in the accumulator AT the resolve. With nothing shipped, glrDrawsSinceResolve stays 0,
// the device's frame cache hands back the previous frame, and the era presents as exactly black
// (measured: whole-frame mean 0.00 from flip 1400 on with the authority seed off).
// The CPU path has had the matching rule since cont.165 -- OnDisplayFlip drains, because this game
// issues its UI text right at the end of a frame and a snapshot without the drain captures the
// frame WITHOUT its text. This is the same rule for the GL accumulator: the flip marker is already
// ordered after the pending DrawRun, so flushing there ships the frame's tail before it resolves.
static const bool s_glFlipFlush = []
{ const char *e = std::getenv("PS2X_GS_GLR_FLIPFLUSH"); return !(e && e[0] == '0'); }();
static unsigned long long g_glFlipFlushes = 0, g_glFlipFlushVerts = 0, g_glFlipFlushEmpty = 0;
static unsigned long long g_glcFlushTexOverwrite = 0;
// ★★★★ cont.331j: m_glPendCount is a MEMBER, and the present path runs on a DIFFERENT object --
// `thread_local GSCpuBackend snapshotBackend` in PresentFromLocalMemory. So the "STRANDED at
// resolve" census read an instance that never accumulates a vertex and has reported
// `events=0/16128 verts=0` in every run since cont.329g: a NULL that reads exactly like a result,
// the same class of defect as cont.331e's four. The worker instance publishes its pending count
// here, and the census reads this instead.
static std::atomic<unsigned long long> g_glPendGlobal{0};
static unsigned long long g_glTexTraceDraws = 0, g_glTexTraceMut = 0;
// ★★★★★ cont.357c PS2X_GS_UPCENSUS=1 (default OFF, read-only) -- a per-TRANSFER census of
// host->local VRAM uploads, keyed by a HASH OF THE PAYLOAD.
//
// Why it exists: the HD texture pack (cont.355b) substitutes a DECODED TEXTURE and is keyed by the
// decoded pixels, so art that never becomes a texture is invisible to it. The title screen's
// artwork is one of those -- it is written into VRAM by a raw transfer and PRESENTED from the
// framebuffer, so no draw and no texture decode ever carries it (cont.356). Before anything can
// substitute on the upload side, three things have to be MEASURED rather than assumed:
//   1. what a surface's unit of work IS -- one transfer, or thousands of row chunks (the FMV
//      uploads its picture as 16x16 macroblocks, and a per-chunk key would be useless);
//   2. whether a surface's payload hash is STABLE run to run (it should be -- the bytes come from
//      EE memory -- but "should" is a hypothesis);
//   3. how many distinct surfaces a screen uploads, i.e. how large a pack's namespace would be.
//
// The key deliberately mirrors ps2x::texpack::contentKey -- FNV-1a with the dimensions mixed in --
// so an upload key and a texture key are the same kind of name, and both are computable offline
// from a dump.
// ⚠ It hashes the PAYLOAD (the transfer's own bytes, in the order the guest sent them), NEVER the
// VRAM pages it lands on. A page carries more than one surface, which is exactly the trap that
// makes a VRAM-keyed pack silently half-work (cont.355b).
// ⚠ It counts local->local blits too, because a null on the host->local side would otherwise be
// uninterpretable: "the art does not arrive this way" and "the art arrives by a route this probe
// does not watch" read identically (cont.346o / cont.355's kRpcSRData guess).
static const bool s_upCensus = []
{ const char *e = std::getenv("PS2X_GS_UPCENSUS"); return e && e[0] && e[0] != '0'; }();
// ★★★★ PS2X_GS_UPCENSUS_MIN=<bytes> (default 65536): the size above which a transfer earns a ROW
// in the table. Measured on the first run of this instrument (build 1102, save path): the intro
// FMV uploads its picture as ~1 KB macroblocks and minted **1,064,543 distinct payload keys** in
// one run, so an ungated table is nothing but movie frames -- it filled its 4,096 rows before the
// title screen was ever reached. Art surfaces are three orders of magnitude larger (the font atlas
// is 232 KB, a full-width CT32 surface ~900 KB), so ONE number separates them cleanly.
// ⚠ Below-gate transfers are COUNTED, never dropped in silence: `small=`/`smallbytes=` on the
// header line say exactly how much of the traffic the gate is hiding. Set it to 0 to see them all.
static const uint32_t s_upCensusMin = []
{ const char *e = std::getenv("PS2X_GS_UPCENSUS_MIN"); return e && e[0] ? (uint32_t)std::strtoul(e, nullptr, 0) : 65536u; }();
// ★★★ cont.357d PS2X_GS_UPABL=1 (default OFF, ABLATION ONLY -- deliberately produces WRONG
// output): consume a large host->local chunk WITHOUT writing VRAM. It exists to SIZE what the
// upload path costs before a dedup is designed, per the ablate-before-optimizing rule: cont.357c's
// census says this game re-uploads ~2.4 MB of BYTE-IDENTICAL texture every flip, and
// [gsgpu:handoff] upMs says the upload path is ~7.5 ms/flip -- but share of the frame is not the
// critical path, and only removing the work measures what removing it would buy.
// PS2X_GS_UPABL_MIN (default 65536) is the chunk size it applies from, so CLUTs and small state
// uploads still land and the run is otherwise normal apart from stale texels.
// ⚠ NOT a fix and never a default: the textures are simply missing their content.
static const bool s_upAbl = []
{ const char *e = std::getenv("PS2X_GS_UPABL"); return e && e[0] && e[0] != '0'; }();
static const uint32_t s_upAblMin = []
{ const char *e = std::getenv("PS2X_GS_UPABL_MIN"); return e && e[0] ? (uint32_t)std::strtoul(e, nullptr, 0) : 65536u; }();

// ★★★★★ cont.357d PS2X_GS_UPDEDUP (default OFF; 1 = skip, 2 = VERIFY ONLY): do not re-write VRAM
// with bytes it already holds.
//
// WHAT IT IS FOR. This game rebuilds a VIF1 display list every frame and re-sends its textures
// inside it (guest 0x13F210 kicks D1_CHCR=0x145 in chain mode over a double-buffered tag chain).
// cont.357c's census measured the consequence: 5 CT32 payloads, ~2.4 MB per flip, re-uploaded to
// the SAME destination rect with BYTE-IDENTICAL content, thousands of times per run. Ablating the
// upload writes (PS2X_GS_UPABL) measured what that costs: the matched-scene A/B went 65.9 s ->
// 61.9 s over a fixed 1500-frame window, i.e. ~6% of wall, with upMs 9,542 -> 361.
//
// THE SKIP CONDITION, and why each part is load-bearing:
//   1. the incoming chunk's bytes are IDENTICAL to what this exact destination rect was last
//      written with (a memcmp against a stored copy -- not a hash, so there is no collision risk
//      to reason about, and a linear memcmp is ~1/20th the cost of the swizzled scatter write);
//   2. NOTHING has written those VRAM pages since. Non-rasterizer writers (uploads, blits, clears,
//      pokes, reset) all bump g_gs2PageGen, so the sum over the covered pages, snapshotted at the
//      same point in this function both times, must equal the stored sum plus this transfer's OWN
//      bump. Bumps only ever increase, so any other writer makes the sum too large and the entry
//      is simply not used -- the failure direction is a missed skip, never a wrong one.
//   3. ★ the CPU rasterizer must not be running (s_gsSkipCpuRaster). Its pixel writes deliberately
//      do NOT bump g_gs2PageGen -- cont.250 measured per-draw page bumping at 5.7x on the bench --
//      so under PS2X_GS_RENDERER=cpu condition 2 cannot see a draw that overwrote the region. The
//      dedup is therefore INERT in the CPU arm rather than unsound in it. Giving it a per-draw
//      invalidation (the gs2DrawCoversPage/texStompSlots shape) is the way to lift that, and is
//      deliberately not attempted here.
//
// =2 is the ORACLE, built before trusting =1: at every would-skip it reads the destination back
// through m_readVramFuncs and compares it to the payload, counts ok/bad, and then writes normally.
// Behaviour is unchanged at =2, so a run of it proves the skip would have been safe WITHOUT
// shipping the skip. [gs2:updedup] reports both.
// ★★★★★ cont.357e PS2X_GS_UPBLOCK (DEFAULT ON; "=0" restores the scalar group path): write a CT32
// upload a ROW PAIR at a time as four contiguous 16-byte stores, instead of sixteen scattered
// dword stores.
//
// MEASURED (fight replay, 200 s): the upload path goes 2.0 -> 4.8 GB/s, `upMs` 9,587/9,262 ->
// 4,132/4,166, and the matched-scene A/B (vtab.sh, 2 pairs, fixed 1500-frame window) goes
// 65.8 s -> 63.0 s = 4.3% of wall, ranges non-overlapping (A 62.8-63.2, B 64.1-67.6). That is ~70%
// of the ceiling PS2X_GS_UPABL measured for removing the writes entirely (6.1%).
// BIT-EXACT: PS2X_GS_UPLOADVERIFY re-read 1,593,940,016 pixels through the generic read over a
// 120 s run with this path taking 16,895,787 row pairs -- 0 mismatches, 0 wrap fallbacks.
//
// WHY IT IS EXACT. In the CT32 column layout, rows (y, y+1) of one 8-pixel group occupy ONE
// CONTIGUOUS 64-BYTE COLUMN: ColumnTable32 gives the even row words 0,1,4,5,8,9,12,13 and the odd
// row 2,3,6,7,10,11,14,15 of that column -- the two rows interleaved at 64-BIT (2-pixel)
// granularity, nothing more. So the whole transform is two unpacks per 32 bytes.
// ★ This is PCSX2's own shape: GSBlock::WriteColumn32 (GS/GSBlock.h) loads two source rows and
// applies GSVector4i::sw64, which is precisely _mm_unpacklo_epi64/_mm_unpackhi_epi64 here. Our
// ColumnTable32 and PCSX2's agree because they describe the same hardware; the address below is
// still taken from OUR traits' Address(), never hand-copied, so the path is exact by construction,
// and PS2X_GS_UPLOADVERIFY re-checks every pixel through the generic read.
// ⚠ The wrap guard is load-bearing: WriteAt masks each pixel address with (MEMORY_SIZE - 4), so a
// write running past 4 MB continues at block 0 -- which is a FRAMEBUFFER (row 194 lost a day to
// exactly that). A 64-byte store cannot express that, so a group whose column would cross the end
// falls back to the scalar path, which keeps the hardware's wrap.
static const bool s_upBlock = []
{ const char *e = std::getenv("PS2X_GS_UPBLOCK"); return !(e && e[0] == '0'); }();
static unsigned long long g_upBlockPairs = 0ull, g_upBlockWrapFalls = 0ull;
// ★★★ cont.357f PS2X_GS_UPSTREAM=1 (default OFF, EXPERIMENT): use non-temporal stores for the
// row-pair write. A column is 64 contiguous bytes at a 64-byte-aligned address, i.e. exactly one
// cache line, so _mm_stream_si128 can write it without the read-for-ownership the normal store
// pays. The counter-argument is real and is why this is a flag and not a change: the texture
// decode reads these bytes back almost immediately, and a streaming store leaves them out of cache.
// Measured either way, never argued. An sfence is issued at the end of the chunk, because the
// decode may run on another thread and non-temporal stores are not ordered by the usual rules.
//
// ★ THE MEASUREMENT. The upload path itself gets unambiguously faster: upMs 2,910 -> 2,111 ms over
// a fixed 4,000-frame window, 16 runs per arm, non-overlapping (2,757-3,054 vs 2,033-2,190) =
// -27.5% of the path, 4.8 -> 6.4 GB/s. Bit-exact: PS2X_GS_UPLOADVERIFY 1,176,610,352 pixels /
// 0 mismatches with ntLines=80,398,723, and ntUnaligned/wrapFallbacks are 0 over 32 runs.
//
// ★★★★ AND IT STAYS OFF, because cont.357g SETTLED the wall question the only way it can be:
// BY SIZING IT. 8 drift-cancelling ABBA blocks (32 runs, 2.7 h) put the wall effect at
//     window [1000,2500]  -0.43 s = -0.65%   95% CI [-1.77, +0.91] s   p=0.45
//     window [1000,5000]  +0.18 s = +0.14%   95% CI [-1.97, +2.32] s   p=0.84
//     frames reached/200 s  +19 frames = +0.4%                         p=0.56
// -- i.e. ZERO, and cont.357f's "-1.2%, pooled n=4" was drift: its A runs happened to sit at the
// ends of a chain on a box whose clock decays (i7-1165G7, 400 MHz-4.7 GHz laptop part).
// ⚠ THE DECISIVE NUMBER IS NOT THE p-VALUE, IT IS THE CEILING. The 799 ms this removes is 0.56% of
// that window's 143.6 s even if EVERY saved millisecond were on the critical path -- while the
// measured per-block noise (sd 2.42 s) puts the 95% CI half-width at 1.41% of wall with n=8.
// The ceiling is 2.5-4x BELOW the noise floor, so resolving it here needs n=57-137 blocks
// (230-550 runs, 13-32 h). cont.357f's own remedy ("a longer chain, n >= 6 per arm") could never
// have worked, and nobody should spend the box on it again: this knob's wall effect is
// STRUCTURALLY UNMEASURABLE on this host, and bounded at ~0.5% by construction.
// ⇒ Default OFF is then the conservative call, not a verdict on the work: shipping it would take on
// the cache risk in the paragraph above (the decode reads these bytes back, and a host with a
// different cache/core count could pay more for the readback than the store saves) in exchange for
// at most ~0.5% that no measurement on this box can see. Flip it only with a host where the
// ceiling clears the noise -- or when perf counters are available (cycles for fixed guest work
// cancels the clock drift; blocked here, /proc/sys/kernel/perf_event_paranoid = 4).
static const bool s_upStream = []
{ const char *e = std::getenv("PS2X_GS_UPSTREAM"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_upStreamLines = 0ull, g_upStreamUnaligned = 0ull;
static const int s_upDedup = []
{ const char *e = std::getenv("PS2X_GS_UPDEDUP"); return e && e[0] ? std::atoi(e) : 0; }();
static const uint32_t s_upDedupMin = []
{ const char *e = std::getenv("PS2X_GS_UPDEDUP_MIN"); return e && e[0] ? (uint32_t)std::strtoul(e, nullptr, 0) : 65536u; }();
struct UpDedupEntry
{
    uint64_t key = 0;
    std::vector<uint8_t> payload;   // the exact bytes this rect was last written with
    uint64_t genSum = 0;            // sum of g_gs2PageGen over the covered pages, at the check point
    unsigned long long lastUse = 0; // for eviction
    bool valid = false;
};
static constexpr size_t kUpDedupMax = 12u;              // the census sees 5 distinct payloads
static constexpr size_t kUpDedupBytesMax = 12u << 20;   // hard cap on what the copies may hold
static std::mutex g_upDedupMutex;
static std::vector<UpDedupEntry> g_upDedupEntries;
static size_t g_upDedupBytes = 0;
static unsigned long long g_upDedupClock = 0;
static unsigned long long g_ddChunks = 0, g_ddHits = 0, g_ddBytesSaved = 0, g_ddStores = 0,
                          g_ddGenMiss = 0, g_ddCmpMiss = 0, g_ddVerifyOk = 0, g_ddVerifyBad = 0;
static void upDedupReport()
{
    std::lock_guard<std::mutex> lk(g_upDedupMutex);
    if (g_ddChunks == 0ull) return;
    std::fprintf(stderr, "[gs2:updedup] mode=%d chunks=%llu hits=%llu (%.1f%%) savedMB=%.1f stores=%llu"
                 " miss: gen=%llu cmp=%llu | verify ok=%llu BAD=%llu | entries=%zu %.1fMB (gate %u)\n",
                 s_upDedup, g_ddChunks, g_ddHits,
                 g_ddChunks ? 100.0 * double(g_ddHits) / double(g_ddChunks) : 0.0,
                 double(g_ddBytesSaved) / 1048576.0, g_ddStores, g_ddGenMiss, g_ddCmpMiss,
                 g_ddVerifyOk, g_ddVerifyBad, g_upDedupEntries.size(),
                 double(g_upDedupBytes) / 1048576.0, s_upDedupMin);
}
struct UpCensusRow
{
    uint32_t dbp = 0, dbw = 0, psm = 0, dsax = 0, dsay = 0, rrw = 0, rrh = 0;
    unsigned long long bytes = 0, chunks = 0, seen = 0;
    unsigned long long firstFlip = 0, lastFlip = 0;
};
// ⓘ Threading: the table is filled from the GS worker (or the EE thread at PS2X_GS_THREAD=0) and
// read by OnDisplayFlip, so the MAP is mutex-guarded. The plain counters below are deliberately
// not -- they are an instrument's totals, a torn read costs one line of a report, and locking the
// upload path would perturb the very traffic being measured.
static std::mutex g_upMutex;                            // OnDisplayFlip reports; the GS worker fills
static std::unordered_map<uint64_t, UpCensusRow> g_upRows;
static unsigned long long g_upXfers = 0, g_upChunks = 0, g_upBytes = 0;
static unsigned long long g_upTornOpen = 0;             // a transfer replaced before it completed
static unsigned long long g_upRowsDropped = 0;          // distinct keys past the cap (never silent)
static unsigned long long g_upFirstPrinted = 0;
static unsigned long long g_upBlits = 0, g_upBlitPixels = 0;
static unsigned long long g_upSmall = 0, g_upSmallBytes = 0;   // below the gate, counted not hidden
static constexpr size_t kUpRowsMax = 16384;             // the FMV alone mints ~1 M (see the gate above)
// The in-flight transfer. Chunks of one transfer arrive strictly in order on one thread (the GS
// worker, or the EE thread when PS2X_GS_THREAD=0), so a single running hash is enough.
static bool g_upOpen = false;
static uint64_t g_upHash = 0;
static UpCensusRow g_upCur;
static inline void upCensusMix(uint64_t v) { g_upHash ^= v; g_upHash *= 1099511628211ull; }
static void upCensusChunk(const uint8_t *data, uint32_t sizeBytes, bool first, uint32_t totalBytes,
                          uint32_t dbp, uint32_t dbw, uint32_t psm, uint32_t dsax, uint32_t dsay,
                          uint32_t rrw, uint32_t rrh, unsigned long long flip)
{
    if (first)
    {
        if (g_upOpen) ++g_upTornOpen;
        // The gate is applied HERE, before any hashing -- the FMV moves ~10 GB of payload through
        // this path in a 220 s run, and hashing every byte of it only to discard the row is pure
        // cost. It admits on EITHER the declared size or this first chunk's real byte count.
        // ⚠ Why both: the declared size is derived from bitsPerPixel(), which is the STORAGE width,
        // and for the formats that live in another surface's unused bits (PSMT4HL/T4HH/T8H) that is
        // not the width the payload arrives in -- measured, this game's 256x232 PSMT4HL font atlas
        // declares 4 bpp (29 KB) and delivers 237,568 bytes, 32 bits per pixel. Estimating alone
        // silently dropped the single texture a pack is most likely to want. Real bytes cannot lie,
        // so whichever number is larger decides.
        if (totalBytes < s_upCensusMin && sizeBytes < s_upCensusMin)
        {
            g_upOpen = false;
            ++g_upSmall;
            g_upSmallBytes += totalBytes > sizeBytes ? totalBytes : sizeBytes;
            return;
        }
        g_upOpen = true;
        g_upHash = 1469598103934665603ull;
        upCensusMix(rrw); upCensusMix(rrh); upCensusMix(psm);
        g_upCur = UpCensusRow{};
        g_upCur.dbp = dbp; g_upCur.dbw = dbw; g_upCur.psm = psm;
        g_upCur.dsax = dsax; g_upCur.dsay = dsay; g_upCur.rrw = rrw; g_upCur.rrh = rrh;
        g_upCur.firstFlip = flip;
    }
    if (!g_upOpen || !data) return;
    for (uint32_t i = 0; i < sizeBytes; ++i) { g_upHash ^= data[i]; g_upHash *= 1099511628211ull; }
    g_upCur.bytes += sizeBytes;
    ++g_upCur.chunks;
    ++g_upChunks;
    g_upBytes += sizeBytes;
}
static void upCensusFinish(unsigned long long flip)
{
    if (!g_upOpen) return;
    g_upOpen = false;
    ++g_upXfers;
    const uint64_t key = g_upHash;
    std::lock_guard<std::mutex> lk(g_upMutex);
    auto it = g_upRows.find(key);
    if (it == g_upRows.end())
    {
        if (g_upRows.size() >= kUpRowsMax) { ++g_upRowsDropped; return; }
        g_upCur.seen = 1;
        g_upCur.lastFlip = flip;
        g_upRows.emplace(key, g_upCur);
        // One line per DISTINCT surface, the first time it is seen. Bounded -- but the bound is
        // announced, because a truncating instrument reads exactly like a null (cont.331b).
        if (++g_upFirstPrinted <= 64ull)
            std::fprintf(stderr, "[gs2:upnew] %016llx flip=%llu dbp=%u(page %u) dbw=%u psm=0x%02x "
                         "pos=(%u,%u) %ux%u bytes=%llu chunks=%llu%s\n",
                         (unsigned long long)key, flip, g_upCur.dbp, g_upCur.dbp / 32u, g_upCur.dbw,
                         g_upCur.psm, g_upCur.dsax, g_upCur.dsay, g_upCur.rrw, g_upCur.rrh,
                         g_upCur.bytes, g_upCur.chunks,
                         g_upFirstPrinted == 64ull ? "   <= further first-sightings suppressed" : "");
        return;
    }
    ++it->second.seen;
    it->second.lastFlip = flip;
    it->second.bytes += g_upCur.bytes;
    it->second.chunks += g_upCur.chunks;
}
static void upCensusReport(unsigned long long flip)
{
    std::lock_guard<std::mutex> lk(g_upMutex);
    std::vector<std::pair<uint64_t, UpCensusRow>> rows(g_upRows.begin(), g_upRows.end());
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b)
              { return a.second.bytes > b.second.bytes; });
    std::fprintf(stderr, "[gs2:upcensus] flip=%llu xfers=%llu chunks=%llu bytes=%llu distinct=%zu"
                 " dropped=%llu torn=%llu blits=%llu blitpx=%llu small=%llu smallbytes=%llu (gate %u)\n",
                 flip, g_upXfers, g_upChunks, g_upBytes, g_upRows.size(), g_upRowsDropped,
                 g_upTornOpen, g_upBlits, g_upBlitPixels, g_upSmall, g_upSmallBytes, s_upCensusMin);
    const size_t n = rows.size() < 16u ? rows.size() : 16u;
    for (size_t i = 0; i < n; ++i)
    {
        const UpCensusRow &r = rows[i].second;
        std::fprintf(stderr, "[gs2:upcensus]   %016llx  %5ux%-5u psm=0x%02x dbp=%-6u(page %-3u) "
                     "dbw=%-2u pos=(%u,%u) seen=%-6llu chunks/xfer=%-5.1f KB/xfer=%-8.1f flips %llu..%llu\n",
                     (unsigned long long)rows[i].first, r.rrw, r.rrh, r.psm, r.dbp, r.dbp / 32u,
                     r.dbw, r.dsax, r.dsay, r.seen,
                     r.seen ? double(r.chunks) / double(r.seen) : 0.0,
                     r.seen ? double(r.bytes) / double(r.seen) / 1024.0 : 0.0,
                     r.firstFlip, r.lastFlip);
    }
}
// ★★★★ cont.331b PS2X_GS_GLR_MUTCENSUS=1 (default OFF, read-only): an UNCAPPED census of every
// non-draw VRAM write, by SITE and by PAGE, accumulated only inside the probe flip window. The
// existing [gs2:textrace/mut] printer stops after 200 events -- which is how cont.330h came to
// read the run's first 200 mutations (early boot) as "the FMV writes the display buffer". A
// truncating instrument is indistinguishable from a null, and this arc has now paid for that
// seven times. This one counts everything and prints the page RANGES it saw per site.
static inline bool probeFlipOk(); // defined once g_perfFlips exists
static const bool s_mutCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_MUTCENSUS"); return e && e[0] && e[0] != '0'; }();
enum { kMutSites = 4 };
static const char *const kMutSiteName[kMutSites] = { "upload(host->local)", "poke(WriteVram)",
                                                     "blit(local->local)", "clear(framebuffer)" };
static unsigned long long g_mutSiteCount[kMutSites] = {};
// ★ cont.331d: the census needs the FORMAT the write used, not just where it landed -- the whole
// question is whether the present-time seed decodes with the same psm/bw the burst wrote.
struct MutFmtRow { uint32_t site = 0, psm = 0, bw = 0; unsigned long long n = 0; };
static MutFmtRow g_mutFmt[24];
static unsigned g_mutFmtN = 0;
static uint64_t g_mutSitePages[kMutSites][8] = {};
static unsigned long long g_mutCensusTotal = 0;
static inline int mutSiteIndex(const char *site)
{
    if (!site) return -1;
    if (site[0] == 'u') return 0;
    if (site[0] == 'p') return 1;
    if (site[0] == 'b') return 2;
    if (site[0] == 'c') return 3;
    return -1;
}
// The watched texture's page span: TBP counts 256-byte blocks, a page is 32 blocks, and the cache
// hashes at most 64 pages from the base (see the generation sum in the decode path below).
static inline bool glTexTraceWatches(uint32_t firstPage, uint32_t lastPage)
{
    if (s_glTexTrace == 0u)
        return false;
    const uint32_t p0 = s_glTexTrace >> 5;
    const uint32_t p1 = p0 + 64u;
    return !(lastPage < p0 || firstPage > p1);
}
// ★★★★ cont.331j PS2X_GS_PAGEWATCH=<p0>:<p1> (default OFF, read-only): every NON-DRAW VRAM write
// whose page span intersects [p0,p1], by site, uncapped. The companion to ROWCENSUS: that one
// proved NO primitive on EITHER renderer covers the "press start button" rectangle (0 hits in
// 176k primitives, cpu and gl alike), so the text must reach VRAM by a transfer, a blit, a poke or
// a clear -- and this names which, with the rect and the format it used. [gs2:mutcensus] could not
// answer it: it is gated to the probe flip window and reports only a page RANGE per site.
// Page arithmetic: a GS page is 64x32 px at 32bpp, so a 512-px-wide display buffer at fbp=P holds
// display rows y..y+31 in pages P + (y/32)*8 .. +7. Rows 438..465 of fbp=0 are pages 104..119.
static unsigned g_pwP0 = 0, g_pwP1 = 0;
static std::atomic<unsigned long long> g_pwFlip{0};
static const bool s_pageWatch = []
{
    const char *e = std::getenv("PS2X_GS_PAGEWATCH");
    // Only the exact string "0" disables: a range like "0:300" begins with '0' and the
    // earlier test read it as OFF -- silence indistinguishable from a null result.
    if (!e || !e[0] || (e[0] == '0' && e[1] == 0))
        return false;
    char *q = nullptr;
    g_pwP0 = static_cast<unsigned>(std::strtoul(e, &q, 10));
    g_pwP1 = (q && *q == ':') ? static_cast<unsigned>(std::strtoul(q + 1, &q, 10)) : g_pwP0;
    std::fprintf(stderr,
                 "[gs2:pagewatch] ARMED pages=%u..%u -- every non-draw VRAM write intersecting "
                 "them, by site, uncapped counting (first 128 printed in full).\n",
                 g_pwP0, g_pwP1);
    return true;
}();
// Optional row filter: the display buffer's pages carry 32 rows each, so a page test alone still
// matches the whole-screen FMV uploads. PS2X_GS_PAGEWATCH_Y=<y0>:<y1> additionally requires the
// write's DESTINATION rect to touch those rows -- which is what separates "the movie repainted the
// screen" from "something wrote the text band".
static int g_pwY0 = 0, g_pwY1 = 1 << 20;
static const bool s_pageWatchY = []
{
    const char *e = std::getenv("PS2X_GS_PAGEWATCH_Y");
    if (!e || !e[0])
        return false;
    char *q = nullptr;
    g_pwY0 = static_cast<int>(std::strtol(e, &q, 10));
    g_pwY1 = (q && *q == ':') ? static_cast<int>(std::strtol(q + 1, &q, 10)) : g_pwY0;
    std::fprintf(stderr, "[gs2:pagewatch] ROW filter y=%d..%d also required.\n", g_pwY0, g_pwY1);
    return true;
}();
static unsigned long long g_pwHit[4] = {}, g_pwShown = 0, g_pwTotal = 0;
static unsigned long long g_pwFirstFlip[4] = {}, g_pwLastFlip[4] = {};
static void pageWatchReport(unsigned long long flip)
{
    static const char *const kName[4] = { "upload(host->local)", "poke(WriteVram)",
                                          "blit(local->local)", "clear(framebuffer)" };
    std::fprintf(stderr, "[gs2:pagewatch] flip=%llu pages=%u..%u | writes intersecting=%llu\n",
                 flip, g_pwP0, g_pwP1, g_pwTotal);
    for (unsigned i = 0; i < 4u; ++i)
        std::fprintf(stderr, "[gs2:pagewatch]   %-20s events=%-10llu flips=%llu..%llu\n",
                     kName[i], g_pwHit[i], g_pwFirstFlip[i], g_pwLastFlip[i]);
}
// Report a non-draw VRAM mutation that lands in the watched texture's pages. `site` names the
// writer so a missing bump is attributed to a function, not to "somewhere".
static inline void glTexTraceMutation(const char *site, uint32_t firstPage, uint32_t lastPage,
                                      uint32_t psm, uint32_t bw, uint32_t x0, uint32_t y0,
                                      uint32_t x1, uint32_t y1, bool bumped)
{
    // ★★★★ cont.331j PAGEWATCH: this funnel is the single point every non-draw VRAM write passes
    // through, so one test here covers uploads, pokes, blits and clears on BOTH renderers.
    // cont.341: with PS2X_GS_PROBEFLIP set, watch only inside its window -- the 512-line print
    // cap was otherwise spent on the boot era's uploads before the level had even loaded.
    if (s_pageWatch && (s_probeFlip == 0ull || probeFlipOk()) &&
        !(lastPage < g_pwP0 || firstPage > g_pwP1) &&
        !(s_pageWatchY && (int(y1) < g_pwY0 || int(y0) > g_pwY1)))
    {
        const int pi = mutSiteIndex(site);
        if (pi >= 0)
        {
            const unsigned long long f = g_pwFlip.load(std::memory_order_relaxed);
            ++g_pwTotal;
            if (g_pwHit[pi]++ == 0ull) g_pwFirstFlip[pi] = f;
            g_pwLastFlip[pi] = f;
            if (g_pwShown < 512ull || (g_pwTotal % 4096ull) == 0ull)
            {
                ++g_pwShown;
                std::fprintf(stderr,
                             "[gs2:pagewatch] #%llu flip=%llu site=%s pages=%u..%u psm=0x%02x bw=%u "
                             "rect=(%u,%u)..(%u,%u) gen-bumped=%d\n",
                             g_pwShown, f, site, firstPage, lastPage, psm, bw, x0, y0, x1, y1,
                             bumped ? 1 : 0);
            }
        }
    }
    // ★★★★ cont.331b: the UNCAPPED census first, before any print cap can hide it.
    if (s_mutCensus && probeFlipOk())
    {
        const int si = mutSiteIndex(site);
        if (si >= 0)
        {
            ++g_mutSiteCount[si];
            ++g_mutCensusTotal;
            constexpr uint32_t kPagesHere = 512u; // kGs2Pages, declared later in this TU
            unsigned fr = 0;
            for (; fr < g_mutFmtN; ++fr)
                if (g_mutFmt[fr].site == uint32_t(si) && g_mutFmt[fr].psm == psm &&
                    g_mutFmt[fr].bw == bw) break;
            if (fr == g_mutFmtN && g_mutFmtN < 24u)
            { g_mutFmt[g_mutFmtN].site = uint32_t(si); g_mutFmt[g_mutFmtN].psm = psm;
              g_mutFmt[g_mutFmtN].bw = bw; ++g_mutFmtN; }
            if (fr < 24u) ++g_mutFmt[fr].n;
            const uint32_t lo = firstPage < kPagesHere ? firstPage : kPagesHere - 1u;
            const uint32_t hi = lastPage < kPagesHere ? lastPage : kPagesHere - 1u;
            for (uint32_t pg = lo; pg <= hi; ++pg)
                g_mutSitePages[si][pg >> 6] |= (1ull << (pg & 63));
        }
    }
    if (!glTexTraceWatches(firstPage, lastPage))
        return;
    ++g_glTexTraceMut;
    if (g_glTexTraceMut > 200ull) // keep counting, stop printing: pokes arrive one pixel at a time
        return;
    std::fprintf(stderr,
                 "[gs2:textrace/mut] #%llu site=%s pages=%u..%u psm=0x%02x bw=%u rect=(%u,%u)..(%u,%u) "
                 "gen-bumped=%d\n",
                 g_glTexTraceMut, site, firstPage, lastPage, psm, bw, x0, y0, x1, y1, bumped ? 1 : 0);
}
static unsigned long long g_glcRuns = 0, g_glcPrimIn = 0, g_glcPrimZero = 0, g_glcPrimDrop = 0,
                          g_glcPrimXlat = 0, g_glcVerts = 0, g_glcGroups = 0, g_glcTexUp = 0;
static unsigned long long g_glcFlushThreshold = 0, g_glcFlushTargetChange = 0,
                          g_glcBatchesOut = 0, g_glcVertsOut = 0, g_glcRunsNoVerts = 0;
static unsigned long long g_glcResolveSamples = 0, g_glcStrandedVerts = 0,
                          g_glcStrandedEvents = 0, g_glcStrandedMax = 0;
// ★★ cont.329g: the geometry's OWN screen area, computed from the guest coordinates before any
// transform of ours touches them. The device counts fragments that passed the depth test; this
// counts the fragments the primitives should PRODUCE. The two together separate the only two
// explanations for an overdraw figure: the scene really is drawn that many times over (both
// renderers would agree), or our vertex transform inflates the geometry (only ours does).
// Sprites are exact (an axis-aligned rect); triangles are the exact signed area of the triangle.
// Both are clamped to the scissor's area, since a primitive cannot cover more than the scissor.
// ★★ cont.329h: the FBMSK census. "FBMSK is 'A' on 100% of draws" was a per-draw frequency over
// a window; a handful of full-screen post-process quads can be 0.001% of draws and the whole
// picture. Counted by VALUE, with the screen area each value covers.
// ★★★ cont.329h: runs that carry draws for MORE THAN ONE render target. RasterRunFanOut splits
// a run on (fbp, fbw, psm, zbp, zpsm) -- but that split sits BELOW the GL renderer's early return,
// so the GL path never sees it and glSubmitRun assigns the WHOLE batch to items[0]'s target.
static unsigned long long g_glcMixedRuns = 0, g_glcMixedItems = 0, g_glcSplitRuns = 0;
// ★★ cont.329j: how much of the scene is FOGGED (PRIM.FGE) -- the GL renderer implemented no fog.
static unsigned long long g_glcFgeDraws = 0, g_glcFgeVerts = 0, g_glcNoFgeDraws = 0;
// Distribution of the per-vertex fog value on FGE draws: F=255 means "no fog at this vertex", so
// a scene whose F is pinned at 255 makes a correct fog implementation INERT -- which is a very
// different thing from the implementation not running.
static unsigned long long g_glcFogHist[8] = {}, g_glcFogN = 0;
// ★★★ cont.329k: the PS2 blend factor can EXCEED 1.0. As/Ad/FIX are divided by 128, so an alpha of
// 0xFF is a factor of 1.99 -- and `((A-B)*C)>>7 + D` is evaluated at that strength before the
// per-channel clamp at 255. GL's fixed-function factors are clamped to [0,1] for a normalised
// target, so every such draw comes out WEAKER, which compresses the set of reachable colours.
// Counted over the vertices of ALPHA-BLENDED draws whose C selector is As.
static unsigned long long g_glcAsHist[9] = {}, g_glcAsN = 0, g_glcAsOver = 0;
// ★★ cont.329j: TEX1's mip state. The CPU rasterizer picks a PER-TRIANGLE mip level and samples
// that level's own tbp/size; the GL path calls resolveTexMip(st, 0) and always samples level 0.
// If this game's draws carry MXL > 0 with a mipmapping MMIN, that is a real sampling difference
// and a candidate for the remaining colour-variety gap. Counted by (MMIN, MXL).
static unsigned long long g_glcMip[8][8] = {};
static unsigned long long g_glcMipN = 0;
// ★★ cont.329j: the state the GL renderer IGNORES or approximates, counted so the next fix is
// chosen by frequency rather than by guess. COLCLAMP=0 means blend results WRAP mod 256 instead of
// clamping (GL always clamps); wrap modes 2/3 are REGION_CLAMP/REGION_REPEAT, which the group
// collapses to CLAMP; the texture decode applies TEXA to CT24 only, without AEM, and nothing at
// all to CT16/CT16S.
static unsigned long long g_glcColclamp[2] = {}, g_glcWrap[4] = {}, g_glcFba[2] = {}, g_glcPabe[2] = {};
static unsigned long long g_glcTexPsm[64] = {}, g_glcTexPsmN = 0;
static unsigned long long g_glcAemTex = 0;
// ★★★ cont.329i: WHO WRITES EACH TARGET, AND WHO READS IT.
// With the cont.329h split in, 27 M vertices per run go to scratch target fbp=0x180, which is never
// resolved and cleared once -- so the background is drawn there and never comes back, and the scene
// reads dark. Two tables answer both halves: every distinct DRAW TARGET (fbp/fbw/psm + extent), and
// every distinct TEXTURE SOURCE that points into VRAM at or above the first scratch page, with
// whether the target-as-source promotion fired for it.
struct GlcTgtRow { uint32_t fbp = 0, fbw = 0, psm = 0, maxX = 0, maxY = 0; unsigned long long draws = 0, verts = 0;
                   uint32_t zbp = 0, zpsm = 0; int vminX = 1 << 30, vminY = 1 << 30, vmaxX = -(1 << 30), vmaxY = -(1 << 30);
                   // ★ cont.330d: how many of this target's draws are UNTEXTURED -- a blanking
                   // sprite and a sprite that samples something are different bugs -- and how its
                   // draws split by FRAME.FBMSK (all = 0xffffffff = writes no colour at all).
                   unsigned long long noTex = 0, mskAll = 0, mskNone = 0, mskOther = 0; };
struct GlcSrcRow { uint32_t tbp = 0, tbw = 0, psm = 0, tw = 0, th = 0, dstFbp = 0; unsigned long long draws = 0, promoted = 0, tcc = 0, tfx = 0;
                   // ★ cont.330d: what this source needs in order to DECODE (a paletted source is
                   // nothing without its CLUT) and which part of it the draw actually samples.
                   uint32_t cbp = 0, cpsm = 0, csa = 0, csm = 0, fst = 0;
                   int uMin = 1 << 30, vMin = 1 << 30, uMax = -(1 << 30), vMax = -(1 << 30); };
static GlcTgtRow g_glcTgt[16];
static GlcSrcRow g_glcSrc[96];
static unsigned g_glcTgtN = 0, g_glcSrcN = 0;
// ★ cont.330d: a silent table overflow would read as "that source does not exist", so count it.
static unsigned long long g_glcSrcDropped = 0;
// ★★★ cont.329k THE PAGE MAP (`PS2X_GS_GLR_PAGEMAP`, read-only): which VRAM pages the game
// RENDERS into, and which it READS as textures. The range ablation showed pages 0x180..0x1FF are
// this game's texture memory -- and 9.4 M draws per run render INTO fbp=0x180, i.e. it renders its
// textures. The GL renderer never writes m_vram, so every read of a page it rendered decodes stale
// bytes. The pages in BOTH columns are exactly what is being served stale.
static unsigned long long g_pgWrite[512] = {}, g_pgRead[512] = {};
// ★★★ cont.329k THE STALE-READ COUNT. A page the GL renderer has DRAWN into holds its pixels only
// in a GL target -- m_vram never sees them -- so a later texture read of that page decodes stale
// bytes. This counts exactly that: reads whose page span intersects pages already rendered, with
// the worst offenders by (tbp, size, psm) so the fix has a list instead of a theory.
static bool g_pgRendered[512] = {};
static unsigned long long g_staleDraws = 0, g_freshDraws = 0;
struct StaleRow { uint32_t tbp = 0, tw = 0, th = 0, psm = 0; unsigned long long draws = 0; };
static StaleRow g_stale[24];
static unsigned g_staleN = 0;
static const bool g_glPageMap = []
{ const char *e = std::getenv("PS2X_GS_GLR_PAGEMAP"); return e && e[0] && e[0] != '0'; }();
static const bool g_glTgtSrcCensus = []
{ const char *e = std::getenv("PS2X_GS_GLR_TGTSRC"); return e && e[0] && e[0] != '0'; }();
// ★★★ cont.330d: the source table above lists only sources at/above page 0x100 -- which is exactly
// why the full-screen sprite that blacks out target 0x180 never appeared in it: its source sits
// BELOW that page. Set this to a target's fbp (e.g. PS2X_GS_GLR_TGTSRC_DST=0x180) and every
// textured draw into THAT target is recorded too, unfiltered by page. Unset (the default) leaves
// the census byte-identical to before.
static const uint32_t s_glTgtSrcDst = []
{ const char *e = std::getenv("PS2X_GS_GLR_TGTSRC_DST"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0xFFFFFFFFu; }();
// ★★★ cont.330d PS2X_GS_GLR_NOPROMOTE=<page>|all (default unset = OFF, DIAGNOSTIC):
// refuse the target-as-source promotion for that target key, so the read falls back to the
// ordinary VRAM decode. Target 0x180 takes 1786 draws per run and EVERY ONE IS UNTEXTURED -- its
// real content arrives by host->local UPLOAD, which lands in m_vram and never in the GL target.
// So promoting `tbp=12288` (512x512 CT24, 891/893 promoted) samples a target that holds only the
// guest's black fills, while VRAM holds the picture. This is the A/B for that: unset = today.
static const uint32_t s_glNoPromote = []
{
    const char *e = std::getenv("PS2X_GS_GLR_NOPROMOTE");
    if (!e || !e[0]) return 0xFFFFFFFFu;                 // unset: promote exactly as before
    if (e[0] == 'a' || e[0] == 'A') return 0xFFFFFFFEu;  // "all": refuse every promotion
    return uint32_t(std::strtoul(e, nullptr, 0));        // a single target key (page)
}();
static unsigned long long g_glPromoteRefused = 0;
static unsigned long long g_glcFbmskN[8] = {};
static double g_glcFbmskArea[8] = {};
static uint32_t g_glcFbmskVal[8] = {};
static unsigned g_glcFbmskUsed = 0;
static unsigned long long g_glcNSprite = 0, g_glcNTri = 0;
static double g_glcAreaSprite = 0.0, g_glcAreaTri = 0.0, g_glcAreaClipped = 0.0;

static GsGpuPresentDevice *gs2GpuDevice()
{
    static GsGpuPresentDevice s_device; // deliberately never torn down (see its dtor note)
    if (!s_device.EnsureInitialized())
        return nullptr;
    // Arm GPU timing once, before any raster job can be posted (perf mode only).
    static const bool s_armed = []
    {
        if (s_gsGpuBatchPerf)
            return true;
        return false;
    }();
    static bool s_done = false;
    if (s_armed && !s_done)
    {
        s_done = true;
        s_device.EnablePerfTiming();
    }
    return &s_device;
}
#endif
static unsigned long g_gpuVerifyCompares = 0;
static unsigned long g_gpuVerifyMismatchRects = 0;
static unsigned long long g_gpuVerifyMismatchPixels = 0;
// ---- Phase 1 (cont.167): GPU VRAM mirror of host->local uploads ----
// PS2X_GS_GPU_XFER (default OFF): post every CT32 upload chunk to the device's
// persistent 4MB mirror SSBO (CT32 = 99.9997% of this game's transfers per the run171
// census) via the swizzle kernel. PS2X_GS_GPU_XFERVERIFY=<N> (default OFF; implies
// XFER): every Nth chunk, read the CPU-authoritative post-upload values and have the
// device compare its mirror on-GPU ([gsgpu:xverify]); CPU stays authoritative.
//
// cont.168 (phase 1b, mirror COMPLETENESS): the swizzle kernel is no longer the only
// mirror path. Every other VRAM writer -- non-CT32 uploads (the T4HL font atlas),
// local->local blits, framebuffer clears, direct WriteVram -- now mirrors through
// mirrorPatchRect(): read the authoritative bytes back out of CPU VRAM and copy them
// verbatim into the mirror at the same offset. Slower per byte than a kernel, but these
// paths are rare AND it is bit-exact by construction (nothing to get wrong), so the only
// remaining mirror gap is the rasterizer's own pixel writes -- which is phase 2 itself.
static const bool s_gsGpuXfer = []
{ const char *e = std::getenv("PS2X_GS_GPU_XFER"); return e && e[0] && e[0] != '0'; }();
static const int s_gsGpuXferVerify = []
{
    const char *e = std::getenv("PS2X_GS_GPU_XFERVERIFY");
    if (!e || !e[0] || e[0] == '0')
        return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 16;
}();
// PS2X_GS_GPU_MIRRORVERIFY=<ms> (default OFF; implies XFER): at most every <ms>
// milliseconds, at a display flip (drained + locked = a coherent CPU VRAM state), compare
// the ENTIRE 4MB mirror against CPU VRAM on the GPU and report which 8KB pages are stale
// ([gsgpu:mverify]). This is the completeness gate for phase 1b and the shadow gate for
// phase 2: once the rasterizer writes into the mirror, stale-words should go to 0.
static const int s_gsGpuMirrorVerifyMs = []
{
    const char *e = std::getenv("PS2X_GS_GPU_MIRRORVERIFY");
    if (!e || !e[0] || e[0] == '0')
        return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 1000;
}();
static unsigned long g_gpuXferSkipped = 0; // uploads the swizzle kernel cannot take (raw-patched)
// Bumped by every VRAM mutation that is NOT a draw (host->local upload, local->local blit,
// clear, direct poke). A GPU draw BATCH cannot span one of these: the batch is rasterized
// at flush time, so an upload that lands mid-batch would be visible to every primitive in
// it, while the oracle only shows it to the primitives that follow. The batch therefore
// has to be flushed before each one -- and how often that happens decides whether batching
// is viable at all, which is what PS2X_GS_BATCHCENSUS's "eff" run lengths measure.
static unsigned long g_vramMutationSeq = 0;
// Counted here (rather than with the other batch tallies) because the eager-flush
// sites sit above them in this file.
static unsigned long g_bvFlushExplicit = 0;
// Perf-mode tallies (PS2X_GS_GPU_BATCHPERF); worker-thread only.
// PS2X_GS_NORASTER (default OFF): ABLATION -- skip the CPU rasterizer entirely (the frame is
// garbage; that is the point). It answers the only question that decides whether the authority
// flip is worth building: if raster were FREE, how much faster would the game actually run? The
// GS worker is a separate thread from the EE, so a faster rasterizer only helps if the worker is
// the limiter; this measures that ceiling directly instead of assuming it.
static const bool s_gsNoRaster = []
{ const char *e = std::getenv("PS2X_GS_NORASTER"); return e && e[0] && e[0] != '0'; }();
// ★ cont.329 PS2X_GS_RENDERER (default "cpu"): selects which renderer consumes draws.
//   cpu = the bit-exact CPU rasterizer -- the only validated path, and the default.
//   gl  = the GL renderer arc (engine docs/gl-renderer.md).
// PHASE 1a is the SEAM ONLY: the CPU pixel work is skipped and nothing draws in its place yet.
// The point of landing it alone is to prove the switch reaches EVERY path into the pixel loop.
// There are three, and two of them are easy to miss (they were called out by the cont.329 survey):
//   1. RasterRunFanOut()            -- the banded/multithreaded pool, the normal path;
//   2. Submit()'s inline call       -- PS2X_GS_THREAD=0, straight to DrawPrimitive;
//   3. workerLoop()'s DrawRun arm   -- when the band pool is empty (PS2X_GS_RASTER_THREADS<=1).
// 2 and 3 both funnel through DrawPrimitive(), so guarding RasterRunFanOut + DrawPrimitive covers
// all three -- which is exactly where PS2X_GS_NORASTER already sits (cont.231 learned that the
// hard way: it guarded only DrawPrimitive, so with the band pool on it silently drew two of every
// three bands and read as "the raster is not the wall").
// WHY an env var rather than a build flag: one binary tests both renderers (working-rules.md).
// ★★★★★ cont.332n THE GL RENDERER IS NOW THE DEFAULT (user's call, 2026-09-17). It is faster
// (game fps 23.8 vs the CPU rasterizer's 16.1 on the same replay, and it holds the game's own
// 2-vblank cadence on 90% of frames against 35%), and after cont.332l's texture-key fix it is also
// MORE correct on characters: the CPU path paints ~10 wrong texels per frame that GL does not (the
// PS2X_GS_QUADTEX defect below). `PS2X_GS_RENDERER=cpu` restores the CPU rasterizer, which stays
// the fallback AND the differential oracle for the native-lift arc -- it is never retired.
static const bool s_gsRendererGl = []
{
    const char *e = std::getenv("PS2X_GS_RENDERER");
    const bool cpu = e && (e[0] == 'c' || e[0] == 'C');
    const bool gl = !cpu;
    // Announce ONLY when non-default, so an unset run's log stays byte-comparable with every
    // log taken before this change (the phase-1 gate is "unset == today").
    if (cpu)
        std::fprintf(stderr, "[gs2:renderer] PS2X_GS_RENDERER=%s -- the CPU rasterizer is active "
                             "(the GL renderer is the default since cont.332n).\n", e);
    else if (e)
        std::fprintf(stderr, "[gs2:renderer] PS2X_GS_RENDERER=%s -- GL renderer active (phase 1): "
                             "the CPU pixel loop is SKIPPED and the frame is presented from a GL "
                             "colour+depth target, which is CLEARED but not yet drawn into. "
                             "PS2X_GS_SCALE sets the internal resolution multiplier. "
                             "Unset, or =cpu, restores the CPU rasterizer.\n", e);
    return gl;
}();
// The ablation and the GL seam both skip the CPU pixel loop; they differ only in intent, and in
// what is expected to appear on screen afterwards. Same-TU ordered init, so this is well defined.
// ★★★ cont.329k PS2X_GS_GLR_VRAMFEED (default OFF, DIAGNOSTIC, slow): with the GL renderer on,
// ALSO run the CPU rasterizer, purely so m_vram keeps receiving the game's draw writes. The GL
// renderer never writes VRAM, so anything the game RENDERS and then SAMPLES as a texture decodes
// stale bytes -- and this game renders into pages 0x180..0x1FF, which the range ablation showed is
// its texture memory. If the GL picture's colour variety jumps with this on, stale textures ARE
// the remaining gap and the fix is a real Source-from-Target; if it does not move, they are not.
static const bool s_glVramFeed = []
{ const char *e = std::getenv("PS2X_GS_GLR_VRAMFEED"); return e && e[0] && e[0] != '0'; }();
static const bool s_gsSkipCpuRaster = s_gsNoRaster || (s_gsRendererGl && !s_glVramFeed);
// PS2X_GS_THRUPUT (default OFF): count + report primitive throughput and nothing else. The
// clean baseline for the ablation A/B -- measuring "raster on" with the GPU path also running
// would compare against an inflated baseline.
static const bool s_gsThruput = []
{ const char *e = std::getenv("PS2X_GS_THRUPUT"); return e && e[0] && e[0] != '0'; }();
// ★★★ cont.331s PS2X_GS_CLOCK_FLIPS (default 32; 0 = off, and it only prints when
// PS2X_GS_THRUPUT is on): THE WALL CLOCK ON A CADENCE BOTH RENDERERS TURN. [gsgpu:thruput] is
// emitted from the CPU rasterizer's primitive counter, which the GL seam never increments, so a
// `PS2X_GS_RENDERER=gl` run carries NO wall clock -- vtwin.py / fpswin.py (wall seconds per fixed
// guest-frame window, the only trustworthy speed metric here) read an empty file, i.e. the two
// arms of the A/B that decides the GL arc were not instrumented the same way. This line is
// emitted from the display flip, which both renderers bump, so both arms carry the SAME clock.
static const unsigned long long s_gsClockFlips = []
{ const char *e = std::getenv("PS2X_GS_CLOCK_FLIPS"); return (e && e[0]) ? std::strtoull(e, nullptr, 0) : 32ull; }();
// ★★ cont.329h PER-DRAW-CALL COMPARISON. `PS2X_GS_DRAWCMP=<primitive count>` (default 0 = OFF)
// arms at a cumulative PRIMITIVE count and then snapshots the DRAW TARGET at every run boundary.
// ⚠ The primitive counter, not the display flip, is the alignment clock. Arming on a flip caught
// **35 of ~24,000 primitives** of that frame: the flip counter is advanced by the presenter while
// the GS worker is drawing whatever is in its queue, so "while flips == N" is a wall-clock slice
// of worker time, not a frame of guest work. `g_perfCpuPrims` is incremented on the same
// primitives in both renderers (cont.329g measured both runs hitting 500,000 at flip 6060/6061),
// so snapshot k of a CPU run and snapshot k of a GL run hold the same primitives by construction. The cont.329g census proved the two renderers receive, translate, ship, draw and
// COVER the same primitives, so what is left is the colour those fragments produce -- and that is a
// question about a particular draw, not about the frame. Both renderers pass through the worker
// loop's Draw/DrawRun arm in the same order, so snapshot N of a CPU run and snapshot N of a GL run
// are the same set of primitives and can be differenced pixel for pixel offline.
// `PS2X_GS_DRAWCMP_DIR=<dir>` writes each snapshot as a PPM; `PS2X_GS_DRAWCMP_MAX` caps them (64).
static const unsigned long long s_drawCmpPrim = []
{ const char *e = std::getenv("PS2X_GS_DRAWCMP"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
static const char *const s_drawCmpDir = std::getenv("PS2X_GS_DRAWCMP_DIR");
// How many primitives after the arm point to trace DRAW CALL BY DRAW CALL on the device.
static const unsigned long long s_drawCmpTrace = []
{ const char *e = std::getenv("PS2X_GS_DRAWCMP_TRACE"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
static const unsigned s_drawCmpMax = []
{ const char *e = std::getenv("PS2X_GS_DRAWCMP_MAX"); const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 64; return (unsigned)(v < 1 ? 1 : v); }();
// The presenter's current display size, so both renderers snapshot the same rect (the GL resolve
// and the CPU decode both anchor at the origin, and the height ALTERNATES 512/511 every frame).
static uint32_t g_drawCmpW = 0, g_drawCmpH = 0;
static unsigned long long g_drawCmpPrims = 0;
static unsigned g_drawCmpN = 0;
static unsigned long long g_perfCpuNs = 0, g_perfCpuPrims = 0;
static unsigned g_perfPrintCountdown = 500000u; // build 400: no per-primitive 64-bit modulo
static unsigned long long g_perfPresentsCached = 0; // cont.231 PS2X_GS_PRESENT_CACHE hits (main thread)
static std::atomic<unsigned long long> g_perfPresents{0};
// cont.230: display FLIPS are the guest's own frame count (one OnDisplayFlip per guest frame), so
// flips per wall second at matched work is the frame rate the player sees; `presents` is only
// the host presenter's cadence, which can outrun or lag the guest.
static std::atomic<unsigned long long> g_perfFlips{0};
// cont.344: the display-flip counter for traces outside this file (ps2_memory / vif1).
extern "C" unsigned long long ps2xGsPerfFlips() { return g_perfFlips.load(std::memory_order_relaxed); }
// ★ cont.331q: expose the guest flip counter to the GL device thread (see gs_gpu_device.h).
unsigned long long gs2CurrentFlip()
{
    return g_perfFlips.load(std::memory_order_relaxed);
}
// Unset (0) = fire anywhere, as before. Set = only inside the requested flip window.
static inline bool probeFlipOk()
{
    if (s_probeFlip == 0ull)
        return true;
    const unsigned long long f = g_perfFlips.load(std::memory_order_relaxed);
    return f >= s_probeFlip && f < s_probeFlip + s_probeFlipW;
}

// ★★★★ cont.331j PS2X_GS_ROWCENSUS=<y0>:<y1>[:<x0>:<x1>] (default OFF, read-only diagnostic).
// Count and describe EVERY primitive whose screen-space bounding box intersects that rectangle,
// at the ONE seam both renderers pass through: the top of RasterRunFanOut and of DrawPrimitive,
// ABOVE the PS2X_GS_RENDERER guard. Every census this arc has built so far lived either inside
// glSubmitRun (GL only) or on the non-draw VRAM-mutation sites, so cont.331i's two findings --
// "the 74 draws in the window cannot paint bright text" and "zero non-draw VRAM writes in the
// window" -- were measured on two different populations, NEITHER of which covers what the CPU
// renderer draws. This reports the SAME quantity on both renderers (cont.331e's own rule), so
// PS2X_GS_RENDERER=cpu and =gl can be compared directly. Default y=438..465 x=133..384 is the
// missing "press start button" (cont.331h).
struct RowCensusRow
{
    uint32_t fbp = 0, fbw = 0, fpsm = 0, fbmsk = 0;
    uint32_t prim = 0, tme = 0, tbp = 0, tbw = 0, tpsm = 0, tw = 0, th = 0, cbp = 0, tfx = 0, tcc = 0;
    uint32_t abe = 0, ba = 0, bb = 0, bc = 0, bd = 0, bfix = 0;
    uint32_t ate = 0, atst = 0, aref = 0, afail = 0, zte = 0, ztst = 0;
    uint32_t vr = 0, vg = 0, vb = 0, va = 0;
    int bx0 = 1 << 29, bx1 = -(1 << 29), by0 = 1 << 29, by1 = -(1 << 29);
    unsigned long long draws = 0, firstFlip = 0, lastFlip = 0;
};
static int s_rowCenY0 = 0, s_rowCenY1 = -1, s_rowCenX0 = 0, s_rowCenX1 = 1 << 20;
// ★★★★ cont.331j PS2X_GS_DRAWPAGES=<p0>:<p1> switches the census predicate from the SCREEN rect to
// the draw's DESTINATION VRAM PAGES. A draw into a different FRAME.FBP can write the very bytes the
// display buffer shows at y=438..465 while its own y is 0..27 -- a screen-rect test is blind to
// that, and would report the honest-looking null this census just produced. Page arithmetic mirrors
// the ClearFramebuffer census site (FBP counts 8 KB pages; linear byte estimate).
static unsigned g_drawPg0 = 0, g_drawPg1 = 0;
static const bool s_drawPages = []
{
    const char *e = std::getenv("PS2X_GS_DRAWPAGES");
    // Only the exact string "0" disables: a range like "0:300" begins with '0' and the
    // earlier test read it as OFF -- silence indistinguishable from a null result.
    if (!e || !e[0] || (e[0] == '0' && e[1] == 0))
        return false;
    char *q = nullptr;
    g_drawPg0 = static_cast<unsigned>(std::strtoul(e, &q, 10));
    g_drawPg1 = (q && *q == ':') ? static_cast<unsigned>(std::strtoul(q + 1, &q, 10)) : g_drawPg0;
    std::fprintf(stderr, "[gs2:rowcensus] DRAWPAGES predicate: destination pages %u..%u "
                         "(the screen-rect test is bypassed).\n", g_drawPg0, g_drawPg1);
    return true;
}();
static const bool s_rowCensus = []
{
    const char *e = std::getenv("PS2X_GS_ROWCENSUS");
    // Only the exact string "0" disables: a range like "0:300" begins with '0' and the
    // earlier test read it as OFF -- silence indistinguishable from a null result.
    if (!e || !e[0] || (e[0] == '0' && e[1] == 0))
        return false;
    char *p = nullptr;
    s_rowCenY0 = static_cast<int>(std::strtol(e, &p, 10));
    s_rowCenY1 = (p && *p == ':') ? static_cast<int>(std::strtol(p + 1, &p, 10)) : s_rowCenY0;
    if (p && *p == ':')
    {
        s_rowCenX0 = static_cast<int>(std::strtol(p + 1, &p, 10));
        s_rowCenX1 = (p && *p == ':') ? static_cast<int>(std::strtol(p + 1, &p, 10)) : (1 << 20);
    }
    std::fprintf(stderr,
                 "[gs2:rowcensus] ARMED y=%d..%d x=%d..%d -- every primitive whose bbox intersects "
                 "this rect, counted at the renderer-INDEPENDENT seam (RasterRunFanOut / "
                 "DrawPrimitive, above the PS2X_GS_RENDERER guard). Identical under =cpu and =gl.\n",
                 s_rowCenY0, s_rowCenY1, s_rowCenX0, s_rowCenX1);
    return true;
}();
static RowCensusRow g_rowCensus[64];
static unsigned g_rowCensusN = 0;
static std::mutex g_rowCensusMu;
static std::atomic<unsigned long long> g_rowCensusSeen{0};
static unsigned long long g_rowCensusHit = 0, g_rowCensusDropped = 0;
static unsigned g_rowCensusShown = 0;
// Recursion guard: RasterRunFanOut splits a run and re-enters itself, and calls DrawPrimitive for
// the diagnostic / raw-split arms. Counting only at depth 0 notes every primitive exactly once,
// whichever of the two entry points the run took.
static thread_local unsigned t_rowCensusDepth = 0;
struct RowCensusDepth
{
    unsigned was;
    RowCensusDepth() : was(t_rowCensusDepth++) {}
    ~RowCensusDepth() { --t_rowCensusDepth; }
    bool top() const { return was == 0u; }
};
static void rowCensusNote(const GSPrimitiveBatch &it)
{
    const unsigned n = it.vertexCount < 3u ? it.vertexCount : 3u;
    if (n == 0u)
        return;
    g_rowCensusSeen.fetch_add(1, std::memory_order_relaxed);
    float fx0 = it.vertices[0].x, fx1 = fx0, fy0 = it.vertices[0].y, fy1 = fy0;
    for (unsigned k = 1; k < n; ++k)
    {
        const GSVertex &v = it.vertices[k];
        if (v.x < fx0) fx0 = v.x;
        if (v.x > fx1) fx1 = v.x;
        if (v.y < fy0) fy0 = v.y;
        if (v.y > fy1) fy1 = v.y;
    }
    const GSDrawState &st = it.state;
    // ★★★★ cont.331j CORRECTION: GSVertex.x/y are GS coordinates, NOT screen pixels -- the raster
    // subtracts XYOFFSET>>4 (here 1792 for a 512-px buffer centred in the 2048 GS plane, so a
    // full-screen sprite reads as [1792..2304]). The first cut of this census compared 435..465
    // against those numbers and reported an honest-looking ZERO. Screen space is what the caller
    // asks about, so convert, exactly as RasterSprite/RasterTriangle do.
    const int ofx = static_cast<int>(st.context.xyoffset.ofx) >> 4;
    const int ofy = static_cast<int>(st.context.xyoffset.ofy) >> 4;
    const int bx0 = static_cast<int>(std::floor(fx0)) - ofx, bx1 = static_cast<int>(std::ceil(fx1)) - ofx;
    const int by0 = static_cast<int>(std::floor(fy0)) - ofy, by1 = static_cast<int>(std::ceil(fy1)) - ofy;
    const bool hitRect = s_rowCensus && !(by1 < s_rowCenY0 || by0 > s_rowCenY1 ||
                                          bx1 < s_rowCenX0 || bx0 > s_rowCenX1);
    uint32_t pLo = 0u, pHi = 0u;
    bool hitPage = false;
    if (s_drawPages)
    {
        // A draw into a SHIFTED FRAME.FBP writes the very bytes the display shows lower down, while
        // its own y stays small -- invisible to the screen-rect test. Page span from FBP (8 KB
        // pages) plus the linear byte estimate the ClearFramebuffer census site uses.
        const uint32_t fbw = st.context.frame.fbw ? st.context.frame.fbw : 1u;
        const uint32_t bpp = bitsPerPixel(st.context.frame.psm) ? bitsPerPixel(st.context.frame.psm) : 32u;
        const int ly0 = by0 < 0 ? 0 : by0, ly1 = by1 < 0 ? 0 : by1;
        pLo = st.context.frame.fbp +
              static_cast<uint32_t>((uint64_t(ly0) * fbw * 64ull * bpp) / 65536ull);
        pHi = st.context.frame.fbp +
              static_cast<uint32_t>((uint64_t(ly1 + 1) * fbw * 64ull * bpp) / 65536ull) + 1u;
        hitPage = !(pHi < g_drawPg0 || pLo > g_drawPg1);
    }
    if (!hitRect && !hitPage)
        return;
    const uint64_t alpha = st.context.alpha;
    RowCensusRow k{};
    k.fbp = st.context.frame.fbp; k.fbw = st.context.frame.fbw;
    k.fpsm = st.context.frame.psm; k.fbmsk = st.context.frame.fbmsk;
    k.prim = static_cast<uint32_t>(st.prim.type);
    k.tme = st.prim.tme ? 1u : 0u;
    k.tbp = st.context.tex0.tbp0; k.tbw = st.context.tex0.tbw; k.tpsm = st.context.tex0.psm;
    k.tw = st.context.tex0.tw; k.th = st.context.tex0.th; k.cbp = st.context.tex0.cbp;
    k.tfx = st.context.tex0.tfx; k.tcc = st.context.tex0.tcc;
    k.abe = st.prim.abe ? 1u : 0u;
    k.ba = uint32_t(alpha & 3u); k.bb = uint32_t((alpha >> 2) & 3u);
    k.bc = uint32_t((alpha >> 4) & 3u); k.bd = uint32_t((alpha >> 6) & 3u);
    k.bfix = uint32_t((alpha >> 32) & 0xFFu);
    k.ate = uint32_t(st.context.test & 1u);
    k.atst = uint32_t((st.context.test >> 1) & 7u);
    k.aref = uint32_t((st.context.test >> 4) & 0xFFu);
    k.afail = uint32_t((st.context.test >> 12) & 3u);
    k.zte = uint32_t((st.context.test >> 16) & 1u);
    k.ztst = uint32_t((st.context.test >> 17) & 3u);
    k.vr = it.vertices[0].r; k.vg = it.vertices[0].g;
    k.vb = it.vertices[0].b; k.va = it.vertices[0].a;
    const unsigned long long flip = g_perfFlips.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_rowCensusMu);
    ++g_rowCensusHit;
    unsigned r = 0;
    for (; r < g_rowCensusN; ++r)
    {
        const RowCensusRow &q = g_rowCensus[r];
        if (q.fbp == k.fbp && q.fbw == k.fbw && q.fpsm == k.fpsm && q.fbmsk == k.fbmsk &&
            q.prim == k.prim && q.tme == k.tme && q.tbp == k.tbp && q.tpsm == k.tpsm &&
            q.tw == k.tw && q.th == k.th && q.cbp == k.cbp && q.tfx == k.tfx && q.tcc == k.tcc &&
            q.abe == k.abe && q.ba == k.ba && q.bb == k.bb && q.bc == k.bc && q.bd == k.bd &&
            q.ate == k.ate && q.atst == k.atst && q.aref == k.aref && q.afail == k.afail &&
            q.zte == k.zte && q.ztst == k.ztst)
            break;
    }
    if (r == g_rowCensusN)
    {
        if (g_rowCensusN >= 64u) { ++g_rowCensusDropped; return; }
        g_rowCensus[g_rowCensusN] = k;
        g_rowCensus[g_rowCensusN].firstFlip = flip;
        ++g_rowCensusN;
    }
    RowCensusRow &q = g_rowCensus[r];
    ++q.draws;
    q.lastFlip = flip;
    q.vr = k.vr; q.vg = k.vg; q.vb = k.vb; q.va = k.va;
    if (bx0 < q.bx0) q.bx0 = bx0;
    if (bx1 > q.bx1) q.bx1 = bx1;
    if (by0 < q.by0) q.by0 = by0;
    if (by1 > q.by1) q.by1 = by1;
    if (g_rowCensusShown < 96u)
    {
        ++g_rowCensusShown;
        std::fprintf(stderr,
                     "[gs2:rowcensus] HIT #%u flip=%llu prim=%u n=%u bbox=[%d..%d]x[%d..%d] "
                     "FRAME{fbp=0x%x fbw=%u psm=0x%02x fbmsk=%08x} tme=%u "
                     "TEX0{tbp=%u tbw=%u psm=0x%02x tw=%u th=%u cbp=%u tfx=%u tcc=%u} "
                     "abe=%u ABCD=%u%u%u%u FIX=%u TEST{ate=%u atst=%u aref=%u afail=%u zte=%u ztst=%u} "
                     "vtx-rgba=(%u,%u,%u,%u) why=%s dstpages=%u..%u\n",
                     g_rowCensusShown, flip, k.prim, n, bx0, bx1, by0, by1,
                     k.fbp, k.fbw, k.fpsm, k.fbmsk, k.tme,
                     k.tbp, k.tbw, k.tpsm, k.tw, k.th, k.cbp, k.tfx, k.tcc,
                     k.abe, k.ba, k.bb, k.bc, k.bd, k.bfix,
                     k.ate, k.atst, k.aref, k.afail, k.zte, k.ztst,
                     k.vr, k.vg, k.vb, k.va,
                     (hitRect && hitPage) ? "rect+page" : (hitRect ? "rect" : "page"), pLo, pHi);
    }
}
// Always prints, hits or none -- a silent instrument is indistinguishable from a null result,
// which this arc has now paid for seven times (cont.331e MEASUREMENT RULES).
static void rowCensusReport(unsigned long long flip)
{
    std::lock_guard<std::mutex> lk(g_rowCensusMu);
    std::fprintf(stderr,
                 "[gs2:rowcensus] flip=%llu y=%d..%d x=%d..%d | primitives seen=%llu, INTERSECTING "
                 "THE RECT=%llu in %u distinct states%s\n",
                 flip, s_rowCenY0, s_rowCenY1, s_rowCenX0, s_rowCenX1,
                 g_rowCensusSeen.load(std::memory_order_relaxed), g_rowCensusHit, g_rowCensusN,
                 g_rowCensusDropped ? " (⚠ table full, states dropped)" : "");
    for (unsigned r = 0; r < g_rowCensusN; ++r)
    {
        const RowCensusRow &q = g_rowCensus[r];
        std::fprintf(stderr,
                     "[gs2:rowcensus]   draws=%-9llu flips=%llu..%llu bbox=[%d..%d]x[%d..%d] "
                     "FRAME{fbp=0x%x fbw=%u psm=0x%02x fbmsk=%08x} prim=%u tme=%u "
                     "TEX0{tbp=%u tbw=%u psm=0x%02x tw=%u th=%u cbp=%u tfx=%u tcc=%u} "
                     "abe=%u ABCD=%u%u%u%u FIX=%u TEST{ate=%u atst=%u aref=%u afail=%u zte=%u ztst=%u} "
                     "vtx-rgba=(%u,%u,%u,%u)\n",
                     q.draws, q.firstFlip, q.lastFlip, q.bx0, q.bx1, q.by0, q.by1,
                     q.fbp, q.fbw, q.fpsm, q.fbmsk, q.prim, q.tme,
                     q.tbp, q.tbw, q.tpsm, q.tw, q.th, q.cbp, q.tfx, q.tcc,
                     q.abe, q.ba, q.bb, q.bc, q.bd, q.bfix,
                     q.ate, q.atst, q.aref, q.afail, q.zte, q.ztst,
                     q.vr, q.vg, q.vb, q.va);
    }
}
static std::chrono::steady_clock::time_point g_perfStart = std::chrono::steady_clock::now();
static // cont.207: how often the band pool actually engaged. A fan-out rate near zero means the
// heuristic is rejecting everything; a high rate with no speedup means the handoff dominates.
std::atomic<unsigned long long> g_gs2BandFan{0};
std::atomic<unsigned long long> g_gs2BandDirect{0};
// cont.231 (build 398) band-handoff instruments, always on (a few atomics per RUN, not per pixel):
// runs / prims per run, the coordinator's wait for its bands at the end of each run, the bands'
// wait for the next run (summed over band threads), upload pixels + time, present-cache hits.
std::atomic<unsigned long long> g_gs2RunCount{0}, g_gs2RunPrims{0};
std::atomic<unsigned long long> g_gs2RawTexItems{0}, g_gs2RawSplits{0}; // cont.321b PS2X_GS_RAWSPLIT
static const int s_gsRawSplit = []
{ const char *e = std::getenv("PS2X_GS_RAWSPLIT"); return (e && e[0]) ? std::atoi(e) : 2; }();
// PS2X_GS_RAWSPLIT_LOG=1: describe the hazardous draws (the first 48, then every 2000th).
static const bool s_gsRawSplitLog = []
{ const char *e = std::getenv("PS2X_GS_RAWSPLIT_LOG"); return e && e[0] && e[0] != '0'; }();
// PS2X_GS_DYNNOTE (default 1): the dynamic claim loop's write notes for skipped primitives (cont.321b); =0 = the
// build-764..766 behaviour (stale cached textures possible) -- the cost A/B only.
static const bool s_gsDynNote = []
{ const char *e = std::getenv("PS2X_GS_DYNNOTE"); return !(e && e[0] == '0'); }();
std::atomic<unsigned long long> g_gs2CoordWaitNs{0}, g_gs2BandIdleNs{0};
// ★ cont.318 per-participant barrier attribution (index 0 = the coordinator): busy ns inside the run
// raster walk, ns spent finished-but-waiting at the barrier (max end - own end), how often this
// participant was the LAST to finish, CPU migrations seen between runs, and the CPU last observed.
static constexpr unsigned kGs2BandMax = 8;
std::atomic<unsigned long long> g_gs2BandBusyNs[kGs2BandMax], g_gs2BandBarrierIdleNs[kGs2BandMax],
    g_gs2BandLast[kGs2BandMax], g_gs2BandMig[kGs2BandMax], g_gs2BandRunsTimed{0};
std::atomic<int> g_gs2BandCpu[kGs2BandMax];
std::atomic<unsigned> g_gs2BandN{0};
double g_gs2BandShareView[kGs2BandMax] = {}; // the coordinator's adaptive shares, for the print only
std::atomic<unsigned long long> g_gs2UploadPixels{0}, g_gs2UploadNs{0};
// build 399: where the worker's time goes per queue item -- dequeue+coalesce (item pop to fan-out
// start), the coordinator's own raster share (fan-out start to its band done), the tail wait (its
// band done to all bands done) and the teardown (all done to the item's end), plus the coalesced
// batch copies.
std::atomic<unsigned long long> g_gs2ItemDequeueNs{0}, g_gs2CoordRasterNs{0}, g_gs2ItemTailNs{0};
std::atomic<unsigned long long> g_gs2CoalesceBatches{0}, g_gs2ItemsDraw{0}, g_gs2ItemsOther{0}, g_gs2ItemOtherNs{0};
thread_local std::chrono::steady_clock::time_point t_itemStart{};
// build 401: the rest of the worker's gap -- the coalescing copy, the presenter-priority yield loop,
// the wait for an empty queue, the teardown (item end -> next pop), and queue items vs sub-runs.
std::atomic<unsigned long long> g_gs2CoalesceNs{0}, g_gs2YieldNs{0}, g_gs2QueueWaitNs{0}, g_gs2TeardownNs{0}, g_gs2QueueItems{0};
// cont.371: the producer side of this queue (GsPipeline parked in enqueueWork on a full queue) and the
// worker's time inside draw runs (the GL renderer's submit: glSubmitRun + target seeds). Always on:
// two clock reads per blocked enqueue / per run. Printed on [gsgpu:items] as prodWaitMs / runMs.
std::atomic<unsigned long long> g_gs2ProdWaitNs{0}, g_gs2ProdWaits{0}, g_gs2RunNs{0};
// cont.317: EE-side drain waits per GSSyncReason (0 Finish, 1 LocalToHost, 2 Presentation, 3 DebugReadback, 4 Reset), and Finish drains skipped.
std::atomic<unsigned long long> g_gs2DrainNs[8]{}, g_gs2DrainCount[8]{}, g_gs2DrainSkipped{0};
// build 402: do the draws around an upload TOUCH its pages? (decides whether uploads must be full
// barriers in a per-band design). Per upload: its VRAM page range; per run: the union of its
// textured batches' texture page ranges and its frame/z page ranges. RAW = batches of the run AFTER an
// upload whose texture pages intersect one of the last 4 uploads; WAR = uploads whose pages intersect
// the PREVIOUS run's texture union. Ranges are conservative (page granularity, tw/th rounded up).
// ★ cont.232 updep v2 (PS2X_GS_UPDEP=1, default OFF): the PAGE-EXACT upload/draw dependency census the
// per-band run-queue design is gated on (cont.231 §12: the first version's linear page RANGES spanned
// whole 64-wide columns of up to 2016 rows -- a column is one page per 32 rows at a stride of dbw
// pages, so a range of ~500 pages "covered" most of VRAM and ~64k of ~70k batches per frame read as
// RAW). VRAM is 512 pages of 8 KB, so a page SET is a 512-bit mask: an upload chunk sets the pages of
// the rows it writes (page geometry per psm), a textured batch sets its TEX0 footprint (+ the CLUT
// page), and a run ORs its batches. Reported per throughput print:
//   RAW  = a run reads pages one of the last four upload chunks wrote (the natural order: upload,
//          then draw -- the dispatcher applies the upload BEFORE appending later runs, so it is safe
//          in the design and only counted for scale);
//   WAR_k = an upload chunk / transfer writes pages one of the last k runs READ (k = 1, 2, 4): in the
//          per-band design that upload must wait for those runs to finish on every band -- the join
//          the design wants to avoid. warXfer(1) over xfers is the number that decides it.
static const bool s_gsUpDep = []
{ const char *e = std::getenv("PS2X_GS_UPDEP"); return e && e[0] && e[0] != '0'; }();
struct Gs2PageMask
{
    uint64_t w[8] = {};
    void set(uint32_t page) { page &= 511u; w[page >> 6] |= 1ull << (page & 63u); }
    void clear() { for (auto &x : w) x = 0ull; }
    void orWith(const Gs2PageMask &o) { for (int i = 0; i < 8; ++i) w[i] |= o.w[i]; }
    bool intersects(const Gs2PageMask &o) const
    { for (int i = 0; i < 8; ++i) if (w[i] & o.w[i]) return true; return false; }
};
// Page geometry per pixel format (ps2tek "GS local memory": CT32/CT24/Z32/Z24 64x32, CT16/CT16S/
// Z16/Z16S 64x64, T8/T8H 128x64, T4/T4HL/T4HH 128x128).
static inline void gs2PageGeom(uint8_t psm, uint32_t &pw, uint32_t &ph)
{
    switch (psm)
    {
    case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S: pw = 64u; ph = 64u; break;
    case GS_PSM_T8: case GS_PSM_T8H: pw = 128u; ph = 64u; break;
    case GS_PSM_T4: case GS_PSM_T4HL: case GS_PSM_T4HH: pw = 128u; ph = 128u; break;
    default: pw = 64u; ph = 32u; break;                                                     // CT32/CT24/Z32/Z24
    }
}
// Pages of the rectangle [x0,x1]x[y0,y1] (inclusive, pixels) of a buffer at basePage with width bw64
// (in 64-pixel units, as TBW/DBW/FBW) in format psm.
static void gs2RectPages(Gs2PageMask &m, uint32_t basePage, uint32_t bw64, uint8_t psm,
                         uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    uint32_t pw, ph; gs2PageGeom(psm, pw, ph);
    const uint32_t pagesPerRow = std::max<uint32_t>(1u, std::max<uint32_t>(bw64, 1u) * 64u / pw);
    for (uint32_t py = y0 / ph; py <= y1 / ph; ++py)
        for (uint32_t px = x0 / pw; px <= x1 / pw; ++px)
            m.set(basePage + py * pagesPerRow + px);
}
static void gs2TexPageMask(Gs2PageMask &m, const GSTex0Reg &t)
{
    const uint32_t tw = 1u << std::min<uint32_t>(t.tw, 10u), th = 1u << std::min<uint32_t>(t.th, 10u);
    gs2RectPages(m, t.tbp0 / 32u, t.tbw, t.psm, 0u, 0u, tw - 1u, th - 1u);
    if (t.psm == GS_PSM_T4 || t.psm == GS_PSM_T8 || t.psm == GS_PSM_T8H || t.psm == GS_PSM_T4HL || t.psm == GS_PSM_T4HH)
        m.set(t.cbp / 32u);
}
std::atomic<unsigned long long> g_gs2UpDep_chunks{0}, g_gs2UpDep_xfers{0}, g_gs2UpDep_runs{0},
    g_gs2UpDep_rawRuns{0}, g_gs2UpDep_rawBatches{0}, g_gs2UpDep_warChunk[3]{}, g_gs2UpDep_warXfer[3]{};
thread_local Gs2PageMask t_upMasks[4];        // the last four upload chunks (RAW test)
thread_local unsigned t_upMaskIdx = 0;
thread_local Gs2PageMask t_runMasks[4];       // the last four runs' texture pages (WAR tests)
thread_local unsigned t_runMaskIdx = 0;
thread_local unsigned t_runMasksValid = 0;
thread_local bool t_xferOpen = false;         // per-transfer WAR flags, finalized at the next transfer
thread_local unsigned t_xferWar = 0;
thread_local std::chrono::steady_clock::time_point t_itemEnd{};
// ---- cont.317 PS2X_GS_KEYCENSUS (default OFF, read-only): rasterizer TIME per (geometry key,
// pipe class). The cont.261 census counted DRAWS (keys 22/23 = 89.7%) and the selector seam was
// built on it; a perf sample of the live fight put key 20 (textured, bilinear, FLAT, no fog) on
// the SCALAR path at 36% of raster time. Draws are not pixels: this counts rdtsc ticks inside
// drawTriangleFastDispatch per key|pipe across all raster threads, plus draws and the bbox area,
// and prints the table with every [gsgpu:thruput] line. Idle-path cost when OFF: one predictable
// branch per triangle.
static const bool s_gsKeyCensus = []
{ const char *e = std::getenv("PS2X_GS_KEYCENSUS"); return e && e[0] && e[0] != '0'; }();
static constexpr int kGsKeySlots = 2048;
static std::atomic<unsigned long long> g_gsKeyTicks[kGsKeySlots], g_gsKeyDraws[kGsKeySlots], g_gsKeyArea[kGsKeySlots];
// slot = key(5 bits) | pipeClass<<5 (2 bits) | texClass<<7 (0 = PSMT4, 1 = PSMT8, 2 = CT32/CT24/CT16, 3 = other/untextured)
//      | primType<<9 (cont.318: 0 = triangle, 1 = sprite, 2 = line -- sprites and lines were uncounted before)
static const char *const kGsKeyTypeName[4] = {"tri", "spr", "lin", "?"};
static const char *const kGsKeyPipeName[4] = {"generic", "stdalph", "addalph", "?"};
static const char *const kGsKeyTexClassName[4] = {"T4", "T8", "CT", "oth"};
// Generic-pipe draws only: which pixel-plan shapes carry the time? sig = key | plan fields.
static std::mutex g_gsKeySigMutex;
static std::unordered_map<uint64_t, std::pair<unsigned long long, unsigned long long>> g_gsKeySig; // sig -> (ticks, draws)
static void gs2PrintKeySigCensus(unsigned long long totalTicks)
{
    std::lock_guard<std::mutex> lock(g_gsKeySigMutex);
    std::vector<std::pair<uint64_t, std::pair<unsigned long long, unsigned long long>>> v(g_gsKeySig.begin(), g_gsKeySig.end());
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.first > b.second.first; });
    std::fprintf(stderr, "[gs:keysig] generic-pipe plans by ticks: typ key abe pabe ABCD fix ztst zmsk ate atst aref fail fbmsk date fpsm  ticks%%    draws\n");
    for (size_t n = 0; n < v.size() && n < 14u; ++n)
    {
        const uint64_t g = v[n].first;
        std::fprintf(stderr, "[gs:keysig] %s %3u  %u    %u   %u%u%u%u  %3u    %u    %u   %u   %u   %3u   %u    %u     %u    %2u  %6.2f%% %8llu\n",
                     ((g >> 61) & 1u) ? "spr" : "tri", unsigned(g & 31u), unsigned((g >> 5) & 1u), unsigned((g >> 6) & 1u),
                     unsigned((g >> 7) & 3u), unsigned((g >> 9) & 3u), unsigned((g >> 11) & 3u), unsigned((g >> 13) & 3u),
                     unsigned((g >> 15) & 255u), unsigned((g >> 23) & 3u), unsigned((g >> 25) & 1u), unsigned((g >> 26) & 1u),
                     unsigned((g >> 27) & 7u), unsigned((g >> 30) & 255u), unsigned((g >> 38) & 7u), unsigned((g >> 41) & 1u),
                     unsigned((g >> 42) & 1u), unsigned((g >> 43) & 63u),
                     100.0 * double(v[n].second.first) / double(totalTicks), v[n].second.second);
    }
}
static void gs2PrintKeyCensus()
{
    unsigned long long total = 0ull;
    for (int i = 0; i < kGsKeySlots; ++i) total += g_gsKeyTicks[i].load(std::memory_order_relaxed);
    if (total == 0ull) return;
    static int order[kGsKeySlots]; for (int i = 0; i < kGsKeySlots; ++i) order[i] = i;
    std::sort(order, order + kGsKeySlots, [](int a, int b) { return g_gsKeyTicks[a].load() > g_gsKeyTicks[b].load(); });
    std::fprintf(stderr, "[gs:keycensus] typ key  tme fst lin iip fge pipe    tex   ticks%%   draws      bbox-px  ticks/bbox-px\n");
    for (int n = 0; n < kGsKeySlots && n < 24; ++n)
    {
        const int i = order[n];
        const unsigned long long t = g_gsKeyTicks[i].load(), d = g_gsKeyDraws[i].load(), a = g_gsKeyArea[i].load();
        if (t == 0ull) break;
        const unsigned key = static_cast<unsigned>(i & 31);
        std::fprintf(stderr, "[gs:keycensus] %s %3u   %u   %u   %u   %u   %u  %-7s %-4s %6.2f%% %8llu %12llu %8.1f\n",
                     kGsKeyTypeName[(i >> 9) & 3], key, (key >> 4) & 1u, (key >> 3) & 1u, (key >> 2) & 1u, (key >> 1) & 1u, key & 1u,
                     kGsKeyPipeName[(i >> 5) & 3], kGsKeyTexClassName[(i >> 7) & 3], 100.0 * double(t) / double(total), d, a,
                     a ? double(t) / double(a) : 0.0);
    }
    gs2PrintKeySigCensus(total);
}
extern "C" unsigned long long ps2xGsPathPrims[4];
namespace { void gs2TexCachePrint(); } // cont.320, defined with the texture cache below
namespace { void gs2TriCensusDump(const char *where); } // cont.329g: dumped periodically now
void gs2PrintRasterThroughput()
{
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_perfStart).count();
    std::fprintf(stderr,
                 "[gsgpu:thruput] %llu prims in %.1f s = %.0f prims/s | presents=%llu = "
                 "%.2f fps (raster %s) flips=%llu p1=%llu p2=%llu p3=%llu\n",
                 g_perfCpuPrims, sec, sec > 0.0 ? double(g_perfCpuPrims) / sec : 0.0,
                 g_perfPresents.load(),
                 sec > 0.0 ? double(g_perfPresents.load()) / sec : 0.0,
                 s_gsRendererGl ? "GL-seam" : (s_gsNoRaster ? "ABLATED" : "on"), g_perfFlips.load(),
                 ps2xGsPathPrims[1], ps2xGsPathPrims[2], ps2xGsPathPrims[3]);
    gs2TexCachePrint(); // cont.320
    gs2TriCensusDump("live"); // cont.329g: periodic, so a timeout-killed run still reports
    if (s_gsKeyCensus)
        gs2PrintKeyCensus();
    const unsigned long long fan = g_gs2BandFan.load(), dir = g_gs2BandDirect.load();
    if (fan + dir != 0ull)
        std::fprintf(stderr, "[gsgpu:bands] fanOut=%llu direct=%llu (%.1f%% fanned)\n",
                     fan, dir, 100.0 * double(fan) / double(fan + dir));
    const unsigned long long runs = g_gs2RunCount.load(), rp = g_gs2RunPrims.load();
    std::fprintf(stderr,
                 "[gsgpu:handoff] runs=%llu prims/run=%.0f coordWait=%.0fms bandIdle=%.0fms upPix=%llu upMs=%.0f presCached=%llu rawTex=%llu rawSplits=%llu\n",
                 runs, runs ? double(rp) / double(runs) : 0.0,
                 double(g_gs2CoordWaitNs.load()) * 1e-6, double(g_gs2BandIdleNs.load()) * 1e-6,
                 g_gs2UploadPixels.load(), double(g_gs2UploadNs.load()) * 1e-6, g_perfPresentsCached,
                 g_gs2RawTexItems.load(), g_gs2RawSplits.load());
    if (s_upDedup)
        upDedupReport();
    if (s_upBlock)
        std::fprintf(stderr, "[gs2:upblock] rowPairs=%llu wrapFallbacks=%llu ntLines=%llu ntUnaligned=%llu\n",
                     g_upBlockPairs, g_upBlockWrapFalls, g_upStreamLines, g_upStreamUnaligned);
    if (const unsigned long long timed = g_gs2BandRunsTimed.load(); timed != 0ull)
    {
        const unsigned n = g_gs2BandN.load();
        std::fprintf(stderr, "[gsgpu:bandbal] runs=%llu N=%u", timed, n);
        for (unsigned i = 0; i < n && i < kGs2BandMax; ++i)
            std::fprintf(stderr, " | b%u busy=%.0fms idle=%.0fms last=%llu mig=%llu cpu=%d share=%.3f", i,
                         double(g_gs2BandBusyNs[i].load()) * 1e-6, double(g_gs2BandBarrierIdleNs[i].load()) * 1e-6,
                         g_gs2BandLast[i].load(), g_gs2BandMig[i].load(), g_gs2BandCpu[i].load(), g_gs2BandShareView[i]);
        std::fprintf(stderr, "\n");
    }
    if (s_gsUpDep)
        std::fprintf(stderr,
                     "[gsgpu:updep2] chunks=%llu xfers=%llu warChunk(k=1/2/4)=%llu/%llu/%llu warXfer(k=1/2/4)=%llu/%llu/%llu"
                     " runs=%llu rawRuns=%llu rawBatches=%llu\n",
                     g_gs2UpDep_chunks.load(), g_gs2UpDep_xfers.load(),
                     g_gs2UpDep_warChunk[0].load(), g_gs2UpDep_warChunk[1].load(), g_gs2UpDep_warChunk[2].load(),
                     g_gs2UpDep_warXfer[0].load(), g_gs2UpDep_warXfer[1].load(), g_gs2UpDep_warXfer[2].load(),
                     g_gs2UpDep_runs.load(), g_gs2UpDep_rawRuns.load(), g_gs2UpDep_rawBatches.load());
    std::fprintf(stderr,
                 "[gsgpu:items] draw=%llu other=%llu dequeue=%.0fms coordRaster=%.0fms tail=%.0fms otherMs=%.0f coalesced=%llu"
                 " qitems=%llu coalesceMs=%.0f yieldMs=%.0f qwaitMs=%.0f teardownMs=%.0f prodWaitMs=%.0f prodWaits=%llu runMs=%.0f\n",
                 g_gs2ItemsDraw.load(), g_gs2ItemsOther.load(), double(g_gs2ItemDequeueNs.load()) * 1e-6,
                 double(g_gs2CoordRasterNs.load()) * 1e-6, double(g_gs2ItemTailNs.load()) * 1e-6,
                 double(g_gs2ItemOtherNs.load()) * 1e-6, g_gs2CoalesceBatches.load(),
                 g_gs2QueueItems.load(), double(g_gs2CoalesceNs.load()) * 1e-6, double(g_gs2YieldNs.load()) * 1e-6,
                 double(g_gs2QueueWaitNs.load()) * 1e-6, double(g_gs2TeardownNs.load()) * 1e-6,
                 double(g_gs2ProdWaitNs.load()) * 1e-6, g_gs2ProdWaits.load(), double(g_gs2RunNs.load()) * 1e-6);
    std::fprintf(stderr, "[gsgpu:drain] finish n=%llu %.0fms | l2h n=%llu %.0fms | dbg n=%llu %.0fms | reset n=%llu %.0fms | finish-skipped=%llu\n",
                 g_gs2DrainCount[0].load(), double(g_gs2DrainNs[0].load()) * 1e-6,
                 g_gs2DrainCount[1].load(), double(g_gs2DrainNs[1].load()) * 1e-6,
                 g_gs2DrainCount[3].load(), double(g_gs2DrainNs[3].load()) * 1e-6,
                 g_gs2DrainCount[4].load(), double(g_gs2DrainNs[4].load()) * 1e-6,
                 g_gs2DrainSkipped.load());
}
// Mipmapping (PS2X_GS_MIPMAP, default ON; =0 restores level-0-only sampling). The frontend
// stores MIPTBP1/2 + TEX1 but the sampler always read level 0 — distant minified surfaces
// alias into the diagonal moiré banding seen on the level terrain (the texture is clean in
// VRAM; verified via PS2X_GS_VRAMDUMP + offline decode, cont.155). Per-triangle LOD is an
// approximation (real GS is per-pixel Q-derived) but kills the aliasing.
static const bool s_gsMipmap = []
{ const char *e = std::getenv("PS2X_GS_MIPMAP"); return !(e && e[0] == '0'); }();

// ---- Phase 2a (cont.169): per-primitive GPU raster shadow verify ----
// PS2X_GS_GPU_RASTERVERIFY=<N> (default OFF; implies the mirror): every Nth supported
// primitive is rasterized on the GPU into a mirror seeded with the CPU's pre-draw bytes,
// then byte-compared against the CPU's post-draw VRAM ([gsgpu:raster]). One primitive at a
// time on purpose: it isolates the PIXEL PIPELINE from tiling and submission order.
static const int s_gsGpuRasterVerify = []
{
    const char *e = std::getenv("PS2X_GS_GPU_RASTERVERIFY");
    if (!e || !e[0] || e[0] == '0')
        return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 2048;
}();
// PS2X_GS_GPU_RASTERTEXSEED (default ON): before rastering a sampled primitive, also copy
// its texture + CLUT pages into the mirror. See the call site -- with it OFF the verify
// additionally measures how often the game samples a RENDER TARGET (pixels only the
// rasterizer produces, which the mirror cannot have until phase 2 writes draws too).
// PS2X_GS_GPU_BATCHVERIFY=<N> (default OFF; implies the mirror, and takes precedence over
// the per-primitive gate): accumulate primitives into batches and rasterize each one with
// the TILED entry point, shadow-verifying every Nth batch against the CPU oracle over the
// whole 4MB. This is the ordering gate -- a batch of thousands of primitives matching the
// oracle byte-for-byte is exactly the proof that per-tile submission order is right.
static const int s_gsGpuBatchVerify = []
{
    const char *e = std::getenv("PS2X_GS_GPU_BATCHVERIFY");
    if (!e || !e[0] || e[0] == '0')
        return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 64;
}();
// Cap a batch so one enormous run cannot make the bin arrays or the seed/compare unbounded.
static const int s_gsGpuBatchMax = []
{
    const char *e = std::getenv("PS2X_GS_GPU_BATCHMAX");
    const int n = e ? std::atoi(e) : 0;
    return (n > 0) ? n : 4096;
}();

static const bool s_gsGpuRasterTexSeed = []
{ const char *e = std::getenv("PS2X_GS_GPU_RASTERTEXSEED"); return !(e && e[0] == '0'); }();
// cont.233: PS2X_GS_HWRASTER_CENSUS=1 (default OFF) tallies state runs + conflict layers per batch.
static const bool s_gsHwCensus = []
{ const char *e = std::getenv("PS2X_GS_HWRASTER_CENSUS"); return e && e[0] && e[0] != '0'; }();
// ---- Phase 2b step 2 scouting (cont.171): PS2X_GS_BATCHCENSUS (default OFF) ----
// A tile rasterizer is only worth building if the game actually SUBMITS in batches. Two
// numbers decide the design: how many consecutive primitives share one render target
// (screen-space tiles are only VRAM-disjoint within a single target -- change fbp/fbw and
// two different tiles can alias the same VRAM word, which would break tile independence),
// and how many tiles a primitive's bbox covers (the per-tile work amplification).
static const bool s_gsBatchCensus = []
{ const char *e = std::getenv("PS2X_GS_BATCHCENSUS"); return e && e[0] && e[0] != '0'; }();
// True when anything should be feeding the device's VRAM mirror this run.
static inline bool gs2MirrorWanted()
{
    return s_gsGpuXfer || s_gsGpuXferVerify > 0 || s_gsGpuMirrorVerifyMs > 0 ||
           s_gsGpuRasterVerify > 0 || s_gsGpuBatchVerify > 0 || s_gsGpuBatchPerf;
}
namespace
{
    // Defined further down, beside the counters it prints (mirrorPatchRect's attribution).
    void gs2PrintMirrorPatch();
    // cont.233 hardware-raster census counters (defined beside the batch tallies).
    extern unsigned long g_bvStateRuns, g_bvLayerSum, g_bvLayerMax;
    // Defined beside the raster-verify rejection tallies: primitives the GPU path REJECTED so far
    // (every reason but kRvOk) -- the GPU bench reports the per-rep delta.
    unsigned long gs2RvRejectedTotal();
    // Defined with the state census; the raster verify tags each compare with it so a
    // divergence names the draw-state combo that produced it.
    uint64_t stateCensusKey(const GSPrimitiveBatch &batch);

    // DrawTriangle's per-triangle LOD, extracted verbatim so the CPU oracle and the
    // phase-2a prim builder cannot drift apart.
    uint32_t gs2TriangleMip(const GSPrimitiveBatch &batch, float denom);
}
// PS2X_GS_FONTTRACE (env, default OFF): LOTR menu-text RE (cont.165). Traces (a) every transfer
// whose destination is the font-CLUT blocks 12288-12290 with its first payload words, and (b)
// every draw that samples a T4HL/T4HH texture with its CLUT at those blocks. Diagnostic for the
// CLUT-RGB-zeroed-in-VRAM corruption (source payloads are white+alpha, VRAM ends up alpha-only).
static const bool g_gs2FontTrace = []
{ const char *e = std::getenv("PS2X_GS_FONTTRACE"); return e && e[0] && e[0] != '0'; }();
// ★★★★★ cont.332h PS2X_GS_PIXWATCH=<x>:<y> (default OFF, read-only): WHO PAINTS THIS PIXEL.
// Every route to "which draw produced these pixels" has failed on this defect -- the row census
// reports bbox INTERSECTION (24.9 M draws sampling the census's top texture were killed and the
// pixels barely moved), DRAWCMP's snapshots do not align between the two renderers (they split runs
// differently), and PS2X_GS_GLR_ORDER cannot be aimed at a gameplay flip. This answers it directly
// on the TRUSTED side: for one screen pixel, log every write that survives every test, with the
// full TEX0 of the draw that made it. Generalises the hardcoded (128,428) font probe below.
// Armed => the per-pixel diagnostic path (gs2PerPixelDiag) forces the scalar loops, exactly as
// FONTTRACE does, because the four-wide loops bypass this write site entirely.
static const int s_pixWatchX = []
{ const char *e = std::getenv("PS2X_GS_PIXWATCH"); return (e && e[0]) ? std::atoi(e) : -1; }();
static const int s_pixWatchY = []
{ const char *e = std::getenv("PS2X_GS_PIXWATCH");
  const char *c = e ? std::strchr(e, ':') : nullptr; return c ? std::atoi(c + 1) : -1; }();
static const unsigned long s_pixWatchN = []
{ const char *e = std::getenv("PS2X_GS_PIXWATCH_N"); return (e && e[0]) ? std::strtoul(e, nullptr, 10) : 64ul; }();
static const bool s_pixWatch = (s_pixWatchX >= 0 && s_pixWatchY >= 0);
static unsigned long g_pixWatchShown = 0;
// The single condition every fast-path gate tests: a per-pixel diagnostic is armed, so the
// specialised loops (which never reach writePixelT) must not be taken.
// cont.341: PIXWATCH forces the scalar loops only INSIDE the probe-flip window (PS2X_GS_PROBEFLIP
// set) -- forcing them for the whole run made a 6,600-flip route take >380 s and never reach the
// watched flip. With no window set, the old whole-run behaviour is kept.
static inline bool gs2PerPixelDiag()
{
    return g_gs2FontTrace || (s_pixWatch && (s_probeFlip == 0ull || probeFlipOk()));
}
static unsigned long g_gs2FontPixWrites = 0; // frameWrite count, read per-draw by [gs2:fontwrites]
static unsigned long g_gs2FontDraws = 0;     // total T4HL/HH cbp-12288 glyph draws (uncapped)
static std::atomic<uint32_t> g_gs2LastFontFbp{0xFFFFu}; // FRAME.fbp of the most recent glyph draw
static std::atomic<GSCpuBackend *> g_gs2ActiveBackend{nullptr};
static std::atomic<uint64_t> g_gs2LastFlipSnapMs{0};
static uint64_t gs2NowMs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

thread_local GSCpuBackend::ResolvedDraw GSCpuBackend::m_draw{};

void GSCpuBackend::workerLoop()
{
    // Reused across iterations so a run drain costs no allocation in steady state.
    static constexpr size_t kMaxRunBatches = 4096;
    std::vector<GSPrimitiveBatch> runBatches;
    runBatches.reserve(kMaxRunBatches);
    ThreadNaming::PinCurrentThreadForRoleIndex(ThreadNaming::CpuRole::GS, 0u); // cont.230 (auto plan / PS2X_GS_CPUS); cont.318 PS2X_GS_PIN1TO1 -> one CPU
    for (;;)
    {
        WorkItem item;
        {
            const auto tq0 = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lk(m_queueMutex);
            m_queueCv.wait(lk, [&] { return m_workerStop || !m_queue.empty(); });
            if (m_workerStop && m_queue.empty())
                return;
            item = std::move(m_queue.front());
            m_queue.pop_front();
            t_itemStart = std::chrono::steady_clock::now();
            g_gs2QueueWaitNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_itemStart - tq0).count()), std::memory_order_relaxed);
            if (t_itemEnd.time_since_epoch().count() != 0)
                g_gs2TeardownNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tq0 - t_itemEnd).count()), std::memory_order_relaxed);
            g_gs2QueueItems.fetch_add(1, std::memory_order_relaxed);
            if (item.kind == WorkItem::Kind::Upload)
                m_queueUploadBytes -= item.bytes.size();
            m_queuePrims -= queuePrimsOf(item);
            // ★ cont.209: drain the RUN of consecutive Draw items behind this one, so the band
            // fan-out is paid once for the whole run instead of once per primitive. Transfers and
            // uploads are left in the queue and end the run, which makes them natural barriers.
            const auto tc0 = std::chrono::steady_clock::now();
            runBatches.clear();
            if (item.kind == WorkItem::Kind::DrawRun)
            {
                // cont.230: the run arrives pre-assembled; still coalesce whatever draw items follow.
                // The swap hands the item's vector to runBatches and the worker's previous one back to
                // the item; that one (a reserved buffer) goes to the pool for the next flush.
                runBatches.swap(item.run);
                recycleRunVectorLocked(std::move(item.run));
            }
            else if (item.kind == WorkItem::Kind::Draw && !m_bandThreads.empty())
            {
                runBatches.push_back(item.batch);
            }
            if (!runBatches.empty() && !m_bandThreads.empty())
            {
                while (!m_queue.empty() && runBatches.size() < kMaxRunBatches)
                {
                    WorkItem &next = m_queue.front();
                    // ★ Count BEFORE recycling: the recycle clears the vector, and the first build
                    // of this loop counted after it (0 instead of N), so m_queuePrims drifted up
                    // until enqueueWork's backpressure predicate could never hold again -- the EE
                    // thread parked forever at the end of the movie (build 376, e23/e25/e26/e27).
                    const size_t nextPrims = queuePrimsOf(next);
                    if (next.kind == WorkItem::Kind::Draw)
                        runBatches.push_back(next.batch);
                    else if (next.kind == WorkItem::Kind::DrawRun &&
                             runBatches.size() + next.run.size() <= kMaxRunBatches)
                    {
                        g_gs2CoalesceBatches.fetch_add(next.run.size(), std::memory_order_relaxed);
                        runBatches.insert(runBatches.end(), next.run.begin(), next.run.end());
                        recycleRunVectorLocked(std::move(next.run));
                    }
                    else
                        break;
                    m_queuePrims -= nextPrims;
                    m_queue.pop_front();
                }
            }
            m_workerBusy = true;
            g_gs2CoalesceNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tc0).count()), std::memory_order_relaxed);
        }
        // Fair handoff to the presenter (cont.163): std::mutex has no fairness, and a
        // saturated worker reacquires m_mutex so hot that the main thread's snapshot
        // lock() can starve for MINUTES (measured: 3 presents in 10 min, frozen black
        // window, dead input pump). Yield until a waiting presenter has taken its turn.
        if (m_presentPriority.load(std::memory_order_acquire))
        {
            const auto ty0 = std::chrono::steady_clock::now();
            while (m_presentPriority.load(std::memory_order_acquire))
                std::this_thread::yield();
            g_gs2YieldNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ty0).count()), std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const bool isRun = item.kind == WorkItem::Kind::Draw || item.kind == WorkItem::Kind::DrawRun;
            const auto tr0 = isRun ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            struct RunClock
            {
                bool on; std::chrono::steady_clock::time_point t0;
                ~RunClock()
                {
                    if (on)
                        g_gs2RunNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
                }
            } runClock{isRun, tr0};
            switch (item.kind)
            {
            case WorkItem::Kind::Draw:
                if (m_vram)
                {
                    if (!runBatches.empty())
                    {
#if PS2X_HAS_GS_GPU_DEVICE
                        // ★★★ cont.329k VRAMFEED diagnostic: GL draws AND the CPU rasterizer runs,
                        // purely to keep m_vram fresh. Submitted HERE, at the top of the run, so
                        // the fan-out's own target-split recursion cannot double-submit it.
                        if (s_gsRendererGl && s_glVramFeed)
                            glSubmitRun(runBatches.data(), runBatches.size());
#endif
                        RasterRunFanOut(runBatches.data(), runBatches.size());
                        drawCmpCheckpoint(runBatches.data(), runBatches.size());
                    }
                    else if (item.batch.vertexCount != 0u)
                    {
                        DrawPrimitive(item.batch);
                        drawCmpCheckpoint(&item.batch, 1u);
                    }
                }
                break;
            case WorkItem::Kind::DrawRun:
                if (m_vram)
                {
#if PS2X_HAS_GS_GPU_DEVICE
                    if (s_gsRendererGl && s_glVramFeed)
                        glSubmitRun(runBatches.data(), runBatches.size());
#endif
                    if (!m_bandThreads.empty())
                        RasterRunFanOut(runBatches.data(), runBatches.size());
                    else
                        for (const GSPrimitiveBatch &b : runBatches)
                            if (b.vertexCount != 0u)
                                DrawPrimitive(b);
                    drawCmpCheckpoint(runBatches.data(), runBatches.size());
                }
                break;
            case WorkItem::Kind::Transfer:
                BeginTransferUnlocked(item.transfer);
                g_gs2ItemsOther.fetch_add(1, std::memory_order_relaxed);
                g_gs2ItemOtherNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_itemStart).count()), std::memory_order_relaxed);
                break;
            case WorkItem::Kind::Upload:
                if (s_gsUpDep && m_transferState.direction == 0u && m_transfer.trxreg.rrw != 0u)
                {
                    // cont.232 updep v2: the chunk's page set (its rows, the transfer's full column
                    // span -- the walker may start mid-row; page granularity absorbs that).
                    const uint32_t rrw = m_transfer.trxreg.rrw;
                    const uint32_t bpp = bitsPerPixel(m_transfer.bitbltbuf.dpsm);
                    const uint32_t pixHere = bpp ? static_cast<uint32_t>(item.bytes.size()) * 8u / bpp : static_cast<uint32_t>(item.bytes.size()) / 4u;
                    const uint32_t rowsHere = (pixHere + rrw - 1u) / rrw + 1u;
                    const uint32_t y0 = m_transferState.y, y1 = m_transferState.y + rowsHere;
                    Gs2PageMask &um = t_upMasks[t_upMaskIdx++ & 3u];
                    um.clear();
                    gs2RectPages(um, m_transfer.bitbltbuf.dbp / 32u, m_transfer.bitbltbuf.dbw, m_transfer.bitbltbuf.dpsm,
                                 m_transfer.trxpos.dsax, y0, m_transfer.trxpos.dsax + rrw - 1u, y1);
                    g_gs2UpDep_chunks.fetch_add(1, std::memory_order_relaxed);
                    if (m_transferState.copiedPixels == 0u)
                    {
                        if (t_xferOpen)
                            for (unsigned k = 0; k < 3u; ++k)
                                if (t_xferWar & (1u << k)) g_gs2UpDep_warXfer[k].fetch_add(1, std::memory_order_relaxed);
                        t_xferOpen = true; t_xferWar = 0u;
                        g_gs2UpDep_xfers.fetch_add(1, std::memory_order_relaxed);
                    }
                    // WAR against the last 1 / 2 / 4 runs' texture pages
                    const unsigned ks[3] = {1u, 2u, 4u};
                    for (unsigned j = 0; j < 3u; ++j)
                    {
                        bool war = false;
                        for (unsigned b = 0; b < std::min(ks[j], t_runMasksValid) && !war; ++b)
                            war = um.intersects(t_runMasks[(t_runMaskIdx - 1u - b) & 3u]);
                        if (war) { g_gs2UpDep_warChunk[j].fetch_add(1, std::memory_order_relaxed); t_xferWar |= 1u << j; }
                    }
                }
                UploadImageUnlocked(item.bytes.data(), static_cast<uint32_t>(item.bytes.size()));
                g_gs2ItemsOther.fetch_add(1, std::memory_order_relaxed);
                g_gs2ItemOtherNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_itemStart).count()), std::memory_order_relaxed);
                break;
            case WorkItem::Kind::FlipSnapshot:
#if PS2X_HAS_GS_GPU_DEVICE
                // ★★★★★ cont.331k: ship the frame's tail BEFORE the frame is resolved (see
                // s_glFlipFlush). This marker is enqueued by OnDisplayFlip and lands after the
                // pending DrawRun, so it is the FIFO frame boundary on the worker thread -- the
                // only thread that may touch m_glPendVerts.
                if (s_gsRendererGl && s_glFlipFlush)
                {
                    const size_t pend = m_glPendVerts.size();
                    if (pend)
                    { ++g_glFlipFlushes; g_glFlipFlushVerts += pend; glFlushPending(); }
                    else
                        ++g_glFlipFlushEmpty;
                    if (((g_glFlipFlushes + g_glFlipFlushEmpty) % 512ull) == 0ull)
                        std::fprintf(stderr,
                                     "[gs2:flipflush] frame-boundary flushes=%llu verts=%llu "
                                     "(mean %.1f) | boundaries with nothing pending=%llu\n",
                                     g_glFlipFlushes, g_glFlipFlushVerts,
                                     g_glFlipFlushes ? double(g_glFlipFlushVerts) / double(g_glFlipFlushes) : 0.0,
                                     g_glFlipFlushEmpty);
                }
#endif
                // cont.231: the flip snapshot, at the FIFO point the drained flip used to capture.
                // m_mutex is held here (the raster is quiescent between items); pair the copy with
                // the pre-flip display registers exactly as OnDisplayFlip's drained path does.
                if (m_vram && m_vramSize != 0u)
                {
                    uint64_t seqNow = 0;
                    {
                        std::lock_guard<std::mutex> plock(m_presentCopyMutex);
                        m_presentCopy.assign(m_vram, m_vram + m_vramSize);
                        seqNow = ++m_presentCopySeq;
                        m_presentCopyHasRegs = true;
                        m_presentCopyDispfb1 = item.flipFb1;
                        m_presentCopyDispfb2 = item.flipFb2;
                        const uint64_t now = gs2NowMs();
                        m_presentCopyTickMs.store(now, std::memory_order_release);
                        g_gs2LastFlipSnapMs.store(now, std::memory_order_release);
                        // rotk row 253 PS2X_GS_FLIPHASH=1 (diagnostic, default OFF): a hash of the WHOLE VRAM at every
                        // flip snapshot -- the FIFO point of the flip, so it is independent of presentation pacing and
                        // pipeline depth. On the CPU renderer under PS2X_VIRTUAL_TIME=1 the sequence is deterministic:
                        // two pipeline layouts (PS2X_GS_SPLIT=0/1) must print the same hashes flip for flip.
                        static const bool s_flipHash = [] { const char *e = std::getenv("PS2X_GS_FLIPHASH"); return e && e[0] && e[0] != '0'; }();
                        if (s_flipHash)
                        {
                            uint64_t h = 1469598103934665603ull;
                            const uint64_t *w = reinterpret_cast<const uint64_t *>(m_presentCopy.data());
                            const size_t nw = m_presentCopy.size() / 8u;
                            for (size_t i = 0; i < nw; ++i)
                                h = (h ^ w[i]) * 1099511628211ull;
                            std::fprintf(stderr, "[gs:fliphash] seq=%llu fb1=%016llx hash=%016llx\n",
                                         static_cast<unsigned long long>(seqNow), static_cast<unsigned long long>(item.flipFb1),
                                         static_cast<unsigned long long>(h));
                        }
                    }
#if PS2X_HAS_GS_GPU_DEVICE
                    // ★★★★ cont.345: the GL arm's snapshot of the same instant (see s_glSnapResolve).
                    glSnapshotDisplayTargets(item.flipFb1, item.flipFb2, seqNow);
#endif
                }
                break;
            }
        }
        t_itemEnd = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(m_queueMutex);
            m_workerBusy = false;
            m_queueIdleCv.notify_all();
        }
    }
}

// cont.230: backpressure counts PRIMITIVES, not items -- a DrawRun item carries up to kPendingRunMax
// draws, so an item-count bound alone would let the EE run 1024x further ahead of the rasterizer
// (memory growth + a present lag of seconds). The bound is the same 65536 primitives as before.
size_t GSCpuBackend::queuePrimsOf(const WorkItem &item)
{
    return item.kind == WorkItem::Kind::DrawRun ? item.run.size()
         : item.kind == WorkItem::Kind::Draw    ? 1u
                                                : 0u;
}

// Caller holds m_queueMutex. Keeps a few reserved vectors so flushPendingDraws() never allocates in
// steady state (a 1024 x 312-byte run is 320 KB, above glibc's initial mmap threshold).
void GSCpuBackend::recycleRunVectorLocked(std::vector<GSPrimitiveBatch> &&v)
{
    if (v.capacity() >= s_gsRunMax && m_runPool.size() < kRunPoolMax)
    {
        v.clear();
        m_runPool.push_back(std::move(v));
    }
}

void GSCpuBackend::enqueueWork(WorkItem &&item)
{
    // cont.230: a non-draw item must land AFTER every draw submitted before it.
    if (item.kind == WorkItem::Kind::Transfer || item.kind == WorkItem::Kind::Upload ||
        item.kind == WorkItem::Kind::FlipSnapshot)
        flushPendingDraws();
    const size_t prims = queuePrimsOf(item);
    std::unique_lock<std::mutex> lk(m_queueMutex);
    auto hasSpace = [&]
    { return m_queue.size() < kGsQueueMaxItems &&
             (m_queuePrims == 0u || m_queuePrims + prims <= kGsQueueMaxItems) &&
             m_queueUploadBytes < kGsQueueMaxUploadBytes; };
    if (!hasSpace())
    {
        const auto tw0 = std::chrono::steady_clock::now();
        m_queueIdleCv.wait(lk, hasSpace);
        g_gs2ProdWaitNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw0).count()), std::memory_order_relaxed);
        g_gs2ProdWaits.fetch_add(1ull, std::memory_order_relaxed);
    }
    if (item.kind == WorkItem::Kind::Upload)
        m_queueUploadBytes += item.bytes.size();
    m_queuePrims += prims;
    m_queue.push_back(std::move(item));
    m_queueCv.notify_one();
}

void GSCpuBackend::drainQueue() const
{
    if (!s_gsThread)
        return;
    flushPendingDraws(); // cont.230: pending draws are part of "everything submitted so far"
    std::unique_lock<std::mutex> lk(m_queueMutex);
    m_queueIdleCv.wait(lk, [&] { return m_queue.empty() && !m_workerBusy; });
}

// ---- cont.208: deterministic rasterizer capture + replay bench --------------------------------
// WHY: every speed metric available in a live run is confounded. Wall time for a fixed guest-frame
// interval swung 7x on an IDENTICAL config (cont.207) because the game is real-time driven, and
// cumulative ns/pair is era-diluted. Optimising the rasterizer against those is guesswork.
//
// This records a real primitive stream plus the exact VRAM it draws into, then replays it
// off-line. Same work every run => the timing is repeatable to ~1%, and — just as valuable — the
// replay is a PIXEL-EXACT correctness oracle: run the same capture at 1 thread and at N and compare
// the VRAM hash. Banded rasterization either produces byte-identical output or it does not.
//
//   PS2X_GS_RASTERCAP=<afterPrims>   capture once the run has drawn this many primitives
//   PS2X_GS_RASTERCAP_N=<count>      how many consecutive batches to record (default 20000)
//   PS2X_GS_RASTERCAP_FILE=<path>    output (default <cwd>/tmp/rastercap.bin)
//   PS2X_GS_RASTERBENCH=<path>       replay that capture at startup, print, and exit
//   PS2X_GS_RASTERBENCH_REPS=<n>     replay repetitions (default 3)
namespace
{
    struct RasterCapHeader
    {
        char magic[4];      // "GSRC"
        uint32_t version;   // 1
        uint32_t vramSize;
        uint32_t batchCount;
    };
    static_assert(std::is_trivially_copyable<GSPrimitiveBatch>::value,
                  "GSPrimitiveBatch must be trivially copyable to dump it raw");

    const char *rasterCapFile()
    {
        static std::string p = []
        {
            const char *e = std::getenv("PS2X_GS_RASTERCAP_FILE");
            return std::string(e && e[0] ? e : "tmp/rastercap.bin");
        }();
        return p.c_str();
    }
    const unsigned long long s_rasterCapAfter = []
    {
        const char *e = std::getenv("PS2X_GS_RASTERCAP");
        return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    const uint32_t s_rasterCapN = []
    {
        const char *e = std::getenv("PS2X_GS_RASTERCAP_N");
        return static_cast<uint32_t>((e && e[0]) ? std::strtoul(e, nullptr, 10) : 20000ul);
    }();
    const char *s_rasterBench = std::getenv("PS2X_GS_RASTERBENCH");
    // Replay only the first N primitives, so a hash divergence can be BISECTED to the exact
    // primitive that introduces it.
    const uint32_t s_rasterBenchLimit = []
    {
        const char *e = std::getenv("PS2X_GS_RASTERBENCH_LIMIT");
        return static_cast<uint32_t>((e && e[0]) ? std::strtoul(e, nullptr, 10) : 0ul);
    }();
    const unsigned s_rasterBenchReps = []
    {
        const char *e = std::getenv("PS2X_GS_RASTERBENCH_REPS");
        unsigned n = static_cast<unsigned>((e && e[0]) ? std::strtoul(e, nullptr, 10) : 3ul);
        return n < 1u ? 1u : n;
    }();
    // ★ cont.232 PS2X_GS_RASTERBENCH_GPU=1: replay the capture through the GPU tile rasterizer instead
    // (asynchronous submission, one readback at the end -- see rasterBenchRun).
    const bool s_rasterBenchGpu = []
    { const char *e = std::getenv("PS2X_GS_RASTERBENCH_GPU"); return e && e[0] && e[0] != '0'; }();

    std::vector<GSPrimitiveBatch> g_capBatches;
    std::vector<uint8_t> g_capVram;
    bool g_capArmed = false;
    bool g_capDone = false;

    uint64_t vramHash(const uint8_t *p, size_t n)
    {
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < n; i += 8)
        {
            uint64_t w = 0;
            std::memcpy(&w, p + i, (n - i >= 8) ? 8 : (n - i));
            h ^= w;
            h *= 0x100000001b3ull;
        }
        return h;
    }
}

// Called from DrawPrimitive. `vram` is the live VRAM at the moment the first batch is recorded.
void GSCpuBackend::rasterCapRecord(const GSPrimitiveBatch &batch)
{
    // ★★ cont.230: PS2X_GS_RASTERCAP_TRIGGER=<path> arms the capture when that FILE APPEARS
    // instead of at a guessed primitive count. An artifact you have to catch by eye cannot be
    // pinned to a prim number in advance -- the player sees it, touches the file, and the very
    // next primitives are recorded. Setting the trigger makes PS2X_GS_RASTERCAP a floor only.
    static const char *const s_capTrigger = std::getenv("PS2X_GS_RASTERCAP_TRIGGER");
    if (g_capDone || m_vram == nullptr)
        return;
    if (s_capTrigger == nullptr && s_rasterCapAfter == 0ull)
        return;
    if (!g_capArmed)
    {
        if (g_perfCpuPrims < s_rasterCapAfter)
            return;
        if (s_capTrigger != nullptr)
        {
            // stat() at most once per 512 primitives -- this runs on the draw path.
            static unsigned long s_poll = 0;
            if ((++s_poll & 511ul) != 0ul)
                return;
            if (::access(s_capTrigger, F_OK) != 0)
                return;
            std::fprintf(stderr, "[gs2:rastercap] TRIGGER seen (%s)\n", s_capTrigger);
        }
        // Snapshot VRAM BEFORE the first recorded batch draws into it, so the replay starts from
        // exactly the state this stream expects.
        g_capVram.assign(m_vram, m_vram + m_vramSize);
        g_capBatches.reserve(s_rasterCapN);
        g_capArmed = true;
        std::fprintf(stderr, "[gs2:rastercap] armed at %llu prims, vram=%u bytes\n",
                     g_perfCpuPrims, m_vramSize);
    }
    g_capBatches.push_back(batch);
    if (g_capBatches.size() < s_rasterCapN)
        return;

    RasterCapHeader hdr{{'G', 'S', 'R', 'C'}, 1u, m_vramSize,
                        static_cast<uint32_t>(g_capBatches.size())};
    std::FILE *f = std::fopen(rasterCapFile(), "wb");
    if (f != nullptr)
    {
        std::fwrite(&hdr, sizeof(hdr), 1, f);
        std::fwrite(g_capVram.data(), 1, g_capVram.size(), f);
        std::fwrite(g_capBatches.data(), sizeof(GSPrimitiveBatch), g_capBatches.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "[gs2:rastercap] wrote %s: %u batches, %u vram bytes\n",
                     rasterCapFile(), hdr.batchCount, hdr.vramSize);
    }
    else
    {
        std::fprintf(stderr, "[gs2:rastercap] FAILED to open %s\n", rasterCapFile());
    }
    g_capDone = true;
    g_capBatches.clear();
    g_capBatches.shrink_to_fit();
    g_capVram.clear();
    g_capVram.shrink_to_fit();
}


// ---- Texture-sample census (PS2X_GS_TEXCENSUS, default OFF; cont.227) ----------------------
// cont.129's "22810 PSMT4 draws vs 636 CT32" counts DRAWS. A devirtualised sampler has to be
// justified by the TEXELS it serves, and by whether the CLUT parameters hold still long enough
// for a per-draw palette decode to amortise -- so this counts per SampleTexture call and per
// texel fetch. Counters are non-atomic on purpose (they sit on a per-pixel path): the census
// forces the band fan-out off, exactly as PS2X_GS_CENSUS2 does, so only one thread touches them.
static const bool s_gsTexCensus = []
{ const char *e = std::getenv("PS2X_GS_TEXCENSUS"); return e && e[0] && e[0] != '0'; }();

// ★★★ cont.227 DECODED CLUT (PS2X_GS_FASTCLUT, default ON, "=0" restores the per-texel VRAM
// read). The census above measured, on the 20,000-prim capture: 83.0% of ALL texel fetches are
// PSMT4, every one of them takes a CLUT lookup, 100% of those are cpsm=CT32 / CSM1, and the CLUT
// configuration changes only 201 times in 20,000 primitives (19,059 lookups per switch). Each
// lookup ran resolveClutIndex + a swizzled ReadCT32 through the 128 KB page table + a std::map
// shadow probe + applyTexa. Decoding the 16 entries once per DrawPrimitive replaces all of that
// with one L1-resident array load. PCSX2 parity: GS/GSClut.cpp GSClut::Read32 expands the palette
// into m_buff32 on a dirty TEX0/TEXA and the sampler indexes that buffer, never VRAM.
static const bool s_gsFastClut = []
{ const char *e = std::getenv("PS2X_GS_FASTCLUT"); return !(e && e[0] == '0'); }();
// ★★★ cont.228 DEVIRTUALISED TEXTURE PATH (PS2X_GS_FASTTEX, default ON, "=0" restores the
// per-texel prologue + the m_draw.texRead function pointer). Two changes, both the cont.225 move
// of REMOVING indirect calls and recomputation: (a) the SampleTexture prologue -- MIPTBP decode,
// texW/texH, the six CLAMP fields -- is resolved once per primitive instead of per texel;
// (b) a PSMT4 texel is read by calling PixelStorageTraits<P4> directly so the whole swizzled
// lookup inlines, and its CLUT hit comes from the cont.227 decoded palette. The census says the
// target is narrow enough to be worth specialising: 83.0% of texel fetches are PSMT4.
// cont.231 PS2X_GS_FASTLERP (default ON; "=0" restores the scalar per-channel lerpChannel path):
// the SSE 4-channel bilinear filter with an exact integer round (bilinear4) plus the per-pixel wrap
// hoist. Both paths produce identical bytes (the bench VRAM hash is the oracle).
static const bool s_gsFastLerp = []
{ const char *e = std::getenv("PS2X_GS_FASTLERP"); return !(e && e[0] == '0'); }();
// ★★ cont.234 PS2X_GS_RASTERABL=<bits> (default 0). ABLATION ONLY, NEVER a correctness run: the
// output is garbage and the bench VRAM hash will not match. It answers "what is the pixel loop
// actually bound by", after the FASTROUND experiment showed nine ALU ops could leave the per-texel
// tail with no effect at all. Both bits are placed on the SINGLE funnel each path shares --
// sampleTextureT and writePixelT -- because an ablation checked on only one of two live paths
// silently stops ablating and reads as "no effect" (cont.231 §2, the PS2X_GS_NORASTER lesson).
//   1 = the whole texture path (addressing, wrap, VRAM gather, CLUT) returns a constant
//   2 = the frame/Z read-modify-write is skipped entirely
//   16 = (cont.319b) DrawSprite draws nothing -- sizes the sprite paths (34.6% of raster ticks, all generic).
//        Bit 16 alone leaves the four-wide triangle paths engaged (their guards test bits 0-3 only), so the
//        triangles are measured as shipped.
//   8 = (cont.264) the per-pixel perspective divide 1/q becomes an rcpss APPROXIMATION -- sizes
//       the divide inside cont.255's undecomposed "skeleton", and stands in for what a 4-wide SoA
//       loop would buy there (divps does four lanes for about the cost of one divss)
// A bit 4 (swizzled frame/Z addressing replaced by a trivial linear address, reads and writes still
// happening in the same number) was measured and REMOVED -- it cost a ternary on the hottest path in
// the rasterizer and its answer is now known: 314.3 -> 284.7 ms at one thread (9.4%) and 128.5 ->
// 125.1 at three (2.6%), and that figure is an OVER-estimate because a linear address also hands the
// loop better locality than the real page/block/column swizzle. So the column-stepped addressing
// lever is capped at ~3% of the raster in the live 3-thread configuration.
static const unsigned s_gsRasterAbl = []
{ const char *e = std::getenv("PS2X_GS_RASTERABL"); return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 0u; }();
// cont.319b PS2X_GS_ZTE (default ON): TEST.ZTE=0 reads as "depth test off, Z not written", PCSX2's
// GSRendererSW rule -- see makePixelPlan. "=0" = the pre-cont.319b reading (ZTST applied regardless).
static const bool s_gsZteAlways = []
{ const char *e = std::getenv("PS2X_GS_ZTE"); return !(e && e[0] == '0'); }();
// ★★★★ cont.343b PS2X_GS_COLCLAMP (default ON; "=0" restores the old saturation): COLCLAMP.CLAMP=0
// means the blend result WRAPS modulo 256 instead of saturating -- PCSX2 GSDrawScanline.cpp
// `if (sel.colclamp == 0) { rb &= 0x00ff; ga &= 0x00ff; }` before the pack. This game's character
// shadow is a per-vertex shadow VOLUME drawn into fbp 0x180 with colclamp=0, ABCD=0221 FIX=128 and
// ZTST=GREATER: light-facing faces add (0,1,1), extruded back faces add (0,255,255) = -1 mod 256, so
// the value left behind is the stencil count and the composite (CT24 read, TEXA aem=1) darkens only
// where it is non-zero. Saturating left every covered pixel non-zero, so the whole sheared volume
// footprint darkened the wall and floor -- the user's "shadow cast on the cave wall" (2026-09-18).
// A colclamp=0 draw takes the generic scalar pixel path (none of the four-wide classes wraps).
static const bool s_gsColClamp = []
{ const char *e = std::getenv("PS2X_GS_COLCLAMP"); return !(e && e[0] == '0'); }();
// ★★★ cont.319b PS2X_GS_SPRITE4 (default ON, "=0" restores the per-pixel sprite loop): sprites four
// pixels at a time. The key census on the recorded Helm's Deep fight charged SPRITES 34% of all
// raster ticks -- full-screen darken passes, the clears, the fills, the T4 particles -- every one of
// them on the fully generic per-pixel path (virtual sampler, scalar combine, writePixelDispatch's
// four-way branch, the generic writePixelT). A sprite is an axis-aligned rectangle: t/v is constant
// per row, s/u is linear in x, the colour, z and fog are the second vertex's, so the four-wide row
// loop is the triangle loop without the edge test, and the write is writePixelQuad with a
// ZTST=ALWAYS class and the two sprite blend classes. Every lane evaluates the scalar sprite
// expression with the same operands in the same order; the capture hashes are the gate.
// Value = a bitmask of the classes served (bisect handle): 1 = kPipeDarken (FST nearest), 2 = kPipeStdAlpha
// (bilinear), 4 = kPipeOpaque (untextured), 8 = kPipeSubFix (cont.321: the untextured ZTE=0 full-screen pair,
// ABCD=0122 FIX -- 13.4% of the live fight's raster ticks on build 764). Unset / "1" (any non-numeric) = all; "0" = none.
static const unsigned s_gsSprite4 = []
{
    const char *e = std::getenv("PS2X_GS_SPRITE4");
    if (!e || !e[0]) return 15u;
    char *end = nullptr;
    const unsigned long v = std::strtoul(e, &end, 0);
    if (end == e) return 15u;
    return v == 1ul && e[1] == 0 ? 15u : static_cast<unsigned>(v);
}();
// cont.231 PS2X_GS_FASTPIX (default ON; "=0" restores the generic row loop): per-state specialised
// triangle pixel loop -- texturing mode, shading and fog resolved at compile time, the texture sampler
// and the pixel writer inlined into it. Same arithmetic as the generic loop (the bench VRAM hash is the
// oracle); it engages only for the CT32/Z24 target and only when no loop-body diagnostic is armed.
static const bool s_gsFastPix = []
{ const char *e = std::getenv("PS2X_GS_FASTPIX"); return !(e && e[0] == '0'); }();
static const bool s_gsFastTex = []
{ const char *e = std::getenv("PS2X_GS_FASTTEX"); return !(e && e[0] == '0'); }();
// ★★★ cont.317 PS2X_GS_FASTP8 (default ON): the PSMT8 twin of the cont.227/228 PSMT4 fast path --
// a 256-entry decoded palette per draw (same key / VRAM-generation / stomp reuse rule) and the
// devirtualised PixelStorageTraits<P8> tap, in the scalar sampler and the four-wide prologue.
// The cont.227 census on the 20,000-prim capture saw T8 at 0.01% of texels and left it on the
// per-texel LookupCLUT path; the cont.317 TIME census of the live Helm's Deep fight put T8 draws
// at 50% of all raster ticks (key 20 generic T8 = 42%, key 22 stdalph T8 = 8%): the terrain.
// `=0` restores the per-texel path. Correctness gate: PS2X_GS_CLUTVERIFY=1 (the same oracle,
// now covering the P8 tap) + the bench hash.
static const bool s_gsFastP8 = []
{ const char *e = std::getenv("PS2X_GS_FASTP8"); return !(e && e[0] == '0'); }();
// ---- Triangle coverage census (PS2X_GS_TRICENSUS, default OFF; cont.229) ------------------
// DrawTriangle walks the FULL bounding box of every triangle and runs the three-edge test per
// pixel. The incremental-DDA fix for that is NOT bit-exact (accumulated float error moves edge
// coverage), and the pixel-exact VRAM hash is this project's correctness gate -- so before
// designing anything, count how much of the walk is actually wasted. Non-atomic; the census
// forces the band fan-out off exactly as PS2X_GS_CENSUS2 does.
static const bool s_gsTriCensus = []
{ const char *e = std::getenv("PS2X_GS_TRICENSUS"); return e && e[0] && e[0] != '0'; }();
// ---- Degenerate-geometry census (PS2X_GS_GEOCENSUS, default OFF; cont.230) -----------------
// Is the geometry ALREADY WRONG when it reaches the GS? The rasterizer is the least likely source
// of the on-screen streaks -- the GPU-mirror arc shadow-verified the whole pixel pipeline against
// this oracle (0 divergent words over 199,673 range compares) and every cont.227-229 change is
// hash-identical -- so this counts what the VU/VIF path actually HANDS us. A triangle crossing the
// near plane without being clipped projects to enormous screen coordinates and rasterises as a
// long streak across the frame, which is exactly the artifact class being chased.
static const bool s_gsGeoCensus = []
{ const char *e = std::getenv("PS2X_GS_GEOCENSUS"); return e && e[0] && e[0] != '0'; }();
// ---- Big-primitive detector (PS2X_GS_BIGPRIM=<percent>, default OFF; cont.230) -------------
// The user's screenshots show large flat DARK wedges extruded from character silhouettes and thin
// geometry, and frames where the whole scene goes black while the HUD stays perfect. That is the
// shape of a shadow/volume pass being drawn VISIBLY instead of only masking. Rather than guess
// which state makes it invisible on hardware (DATE, FBMSK and the blend equation are all already
// implemented), make the offending primitive report ITSELF: log the complete draw state of any
// primitive covering more than <percent> of the scissor.
static const int s_gsBigPrim = []
{ const char *e = std::getenv("PS2X_GS_BIGPRIM"); return e ? std::atoi(e) : 0; }();
// PS2X_GS_DROPBIG=<percent> (default 0 = OFF) -- DIAGNOSTIC ABLATION, never a fix. Skips
// untextured, alpha-blended, near-black primitives covering >= <percent> of the scissor. If the
// blacked-out frames come back with the scene visible, that identifies the culprit draw exactly
// instead of arguing about it.
static const int s_gsDropBig = []
{ const char *e = std::getenv("PS2X_GS_DROPBIG"); return e ? std::atoi(e) : 0; }();

namespace
{
    unsigned long long g_bigPrimHits = 0;
    void gs2BigPrimReport(const char *kind, const GSDrawState &state,
                          int minX, int maxX, int minY, int maxY,
                          unsigned r, unsigned g, unsigned b, unsigned a,
                          const uint32_t *clut, bool clutValid,
                          double zv0, double zv1, double zv2)
    {
        const auto &ctx = state.context;
        const double sw = std::max(1.0, (double)ctx.scissor.x1 - (double)ctx.scissor.x0);
        const double sh = std::max(1.0, (double)ctx.scissor.y1 - (double)ctx.scissor.y0);
        const double cover = 100.0 * (((double)maxX - minX) * ((double)maxY - minY)) / (sw * sh);
        if (cover < (double)s_gsBigPrim)
            return;
        // ★ Dedupe by draw state + target, otherwise one repeated effect floods the log and hides
        // every other big draw (the first pass reported the same 512x512 render-to-texture 40
        // times). "Identify before you fix": a big primitive into an OFF-SCREEN target is not
        // what the player sees.
        {
            // ★ Dedupe REPORTS but keep COUNTS: a state that draws once during a scripted
            // fade and a state that draws every frame look identical in a deduped list, and
            // that difference is the whole question here.
            static std::vector<std::pair<uint64_t, unsigned long long>> tally;
            static unsigned long long calls = 0;
            uint64_t k = (uint64_t)ctx.frame.fbp * 1000003ull;
            k ^= (uint64_t)ctx.frame.fbmsk * 31ull;
            k ^= (uint64_t)ctx.alpha * 131ull;
            k ^= (uint64_t)ctx.test * 17ull;
            k ^= (uint64_t)ctx.tex0.psm * 7919ull;
            k ^= (uint64_t)ctx.scissor.x1 * 65537ull;
            k ^= (uint64_t)ctx.scissor.y1 * 257ull;
            k ^= (uint64_t)(state.prim.abe ? 2u : 0u) | (state.prim.tme ? 1u : 0u);
            bool known = false;
            for (auto &e : tally)
                if (e.first == k) { ++e.second; known = true; break; }
            if (!known && tally.size() < 64u)
                tally.push_back({k, 1ull});
            if ((++calls % 20000ull) == 0ull)
            {
                std::fprintf(stderr, "[gs2:bigtally] bigDraws=%llu distinctStates=%zu counts:",
                             calls, tally.size());
                for (auto &e : tally)
                    std::fprintf(stderr, " %llu", e.second);
                std::fprintf(stderr, "\n");
            }
            if (known) return;
        }
        ++g_bigPrimHits;
        std::fprintf(stderr,
                     "[gs2:bigprim] #%llu %s cover=%.1f%% bbox=[%d,%d]x[%d,%d]"
                     " fbp=%u fbw=%u scis=[%u,%u]x[%u,%u] | tme=%u abe=%u"
                     " iip=%u fst=%u | rgba=(%u,%u,%u,%u) | fbmsk=%08x fpsm=0x%x zmsk=%u"
                     " zpsm=0x%x | test=%016llx alpha=%016llx | tex0psm=0x%x tfx=%u tcc=%u"
                     " | fba=%llu pabe=%u | z=(%.0f,%.0f,%.0f)\n",
                     g_bigPrimHits, kind, cover, minX, maxX, minY, maxY,
                     ctx.frame.fbp, ctx.frame.fbw,
                     ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1,
                     state.prim.tme ? 1u : 0u, state.prim.abe ? 1u : 0u,
                     state.prim.iip ? 1u : 0u, state.prim.fst ? 1u : 0u,
                     r, g, b, a, ctx.frame.fbmsk, ctx.frame.psm, ctx.zbuf.zmask ? 1u : 0u,
                     ctx.zbuf.psm,
                     (unsigned long long)ctx.test, (unsigned long long)ctx.alpha,
                     ctx.tex0.psm, ctx.tex0.tfx, ctx.tex0.tcc,
                     (unsigned long long)ctx.fba, state.pabe ? 1u : 0u,
                     (double)zv0, (double)zv1, (double)zv2);
        // ★★ cont.230: the decoded palette. With tfx=MODULATE and tcc=1 the fragment alpha is
        // (texAlpha * vertexAlpha) >> 7, and this draw's vertex alpha is 128 (= 1.0 on PS2), so
        // the quad's opacity IS the CLUT alpha. Print it so "the alpha is wrong" stops being a
        // hypothesis. PS2 alpha is 0-128 nominal (0x80 = fully opaque); anything above 0x80 is an
        // over-range value, which is exactly the class of bug that would turn a faint haze opaque.
        if (clutValid && clut != nullptr)
        {
            std::fprintf(stderr, "[gs2:bigprim]    cbp=%u cpsm=0x%x csa=%u csm=%u palette:",
                         ctx.tex0.cbp, ctx.tex0.cpsm, ctx.tex0.csa, ctx.tex0.csm);
            for (int i = 0; i < 16; ++i)
                std::fprintf(stderr, " %02x:%08x", i, clut[i]);
            std::fprintf(stderr, "\n");
            unsigned aMin = 255, aMax = 0, aOver = 0;
            for (int i = 0; i < 16; ++i)
            {
                const unsigned av = (clut[i] >> 24) & 0xFFu;
                aMin = std::min(aMin, av);
                aMax = std::max(aMax, av);
                if (av > 0x80u) ++aOver;
            }
            std::fprintf(stderr,
                         "[gs2:bigprim]    palette alpha: min=%u max=%u over0x80=%u\n",
                         aMin, aMax, aOver);
        }
    }
}

namespace
{
    struct Gs2GeoCensus
    {
        unsigned long long tris = 0;
        unsigned long long nonFinite = 0;      // any NaN/Inf vertex coordinate
        unsigned long long magBucket[5] = {};  // max(|x|,|y|): <1k, <10k, <100k, <1e6, >=1e6
        unsigned long long overScissor2 = 0;   // unclipped bbox > 2x the scissor extent
        unsigned long long overScissor8 = 0;
        unsigned long long overScissor64 = 0;  // streak candidates
        unsigned long long thin = 0;           // aspect >= 64:1 and long side > 256 px
        unsigned long long tinyQ = 0;          // non-FST prim with |q| < 1e-3 on a vertex
        unsigned long long dumped = 0;
        double worstMag = 0.0;
        // ★★ cont.230: vertex-COLOUR distribution. With TFX=MODULATE the fragment is
        // (texel * vertex) >> 7, so the vertex colour IS the lighting -- and a light texture
        // rendering near-black means dark vertex colours, not a texture problem. Positions were
        // censused; colours never were, and they come from the same VU1 program.
        unsigned long long lumBucket[8] = {};
        double lumSum = 0.0;
        unsigned long long lumN = 0;
    };
    Gs2GeoCensus g_geoCensus;

    void gs2GeoCensusDump(const char *where)
    {
        if (!s_gsGeoCensus)
            return;
        const Gs2GeoCensus &c = g_geoCensus;
        const double t = c.tris ? static_cast<double>(c.tris) : 1.0;
        std::fprintf(stderr,
                     "[gs2:geocensus/%s] tris=%llu nonFinite=%llu tinyQ=%llu(%.3f%%) worstMag=%.4g\n",
                     where, c.tris, c.nonFinite, c.tinyQ, 100.0 * static_cast<double>(c.tinyQ) / t,
                     c.worstMag);
        std::fprintf(stderr,
                     "[gs2:geocensus/%s] maxCoord <1k=%.3f%% <10k=%.4f%% <100k=%.4f%% <1e6=%.4f%%"
                     " >=1e6=%.4f%%\n", where,
                     100.0 * static_cast<double>(c.magBucket[0]) / t,
                     100.0 * static_cast<double>(c.magBucket[1]) / t,
                     100.0 * static_cast<double>(c.magBucket[2]) / t,
                     100.0 * static_cast<double>(c.magBucket[3]) / t,
                     100.0 * static_cast<double>(c.magBucket[4]) / t);
        std::fprintf(stderr,
                     "[gs2:geocensus/%s] overScissor 2x=%.4f%% 8x=%.4f%% 64x=%.4f%% | thin=%.4f%%\n",
                     where,
                     100.0 * static_cast<double>(c.overScissor2) / t,
                     100.0 * static_cast<double>(c.overScissor8) / t,
                     100.0 * static_cast<double>(c.overScissor64) / t,
                     100.0 * static_cast<double>(c.thin) / t);
        const double ln = c.lumN ? (double)c.lumN : 1.0;
        std::fprintf(stderr, "[gs2:geocensus/%s] vertexLum mean=%.1f | buckets(0-31,..,224+):",
                     where, c.lumSum / ln);
        for (int i = 0; i < 8; ++i)
            std::fprintf(stderr, " %.1f%%", 100.0 * (double)c.lumBucket[i] / ln);
        std::fprintf(stderr, "\n");
    }
}
// ★★★ cont.229 ANALYTIC ROW SPAN (PS2X_GS_FASTTRI, default ON, "=0" restores the full-bbox walk).
// The census above measured 85.45% of all edge tests as pure waste: 42.79% of scanlines cover
// NOTHING, and lead+trail waste is another 48.6%. The textbook fix -- incremental DDA edge
// functions -- is NOT bit-exact (accumulated float error moves coverage at the edges), and the
// pixel-exact VRAM hash is this project's correctness gate. So the per-pixel predicate is left
// EXACTLY as it was, and we only skip pixels PROVEN unable to pass it: each of the three edge
// functions is affine in x, so each admits a half-line, and their intersection is computed in
// DOUBLE precision then widened by a slack far larger than any float/double discrepancy plus a
// whole-pixel margin. Skipping pixels that could never pass cannot change a single output bit.
static const bool s_gsFastTri = []
{ const char *e = std::getenv("PS2X_GS_FASTTRI"); return !(e && e[0] == '0'); }();
// PS2X_GS_SPANVERIFY=1 (default OFF): walk the FULL bbox anyway and report any covered pixel that
// fell outside the computed span -- the oracle that the narrowing is a superset. PS2X_GS_SPANMUTATE
// shrinks the span by that many pixels per side, to prove the oracle can fail.
static const bool s_gsSpanVerify = []
{ const char *e = std::getenv("PS2X_GS_SPANVERIFY"); return e && e[0] && e[0] != '0'; }();
static const int s_gsSpanMutate = []
{ const char *e = std::getenv("PS2X_GS_SPANMUTATE"); return e ? std::atoi(e) : 0; }();

namespace
{
    struct Gs2SpanVerify
    {
        unsigned long long rows = 0;
        unsigned long long escaped = 0;   // covered pixels found OUTSIDE the span
        unsigned long long skipped = 0;   // pixels the span removed from the walk
    };
    Gs2SpanVerify g_spanVerify;

    void gs2SpanVerifyDump(const char *where)
    {
        if (!s_gsSpanVerify)
            return;
        std::fprintf(stderr, "[gs2:spanverify/%s] rows=%llu skipped=%llu ESCAPED=%llu%s\n",
                     where, g_spanVerify.rows, g_spanVerify.skipped, g_spanVerify.escaped,
                     s_gsSpanMutate ? " (MUTATED)" : "");
    }
}

namespace
{
    struct Gs2TriCensus
    {
        unsigned long long tris = 0;        // DrawTriangle calls that reached the walk
        unsigned long long rows = 0;        // scanlines walked
        unsigned long long emptyRows = 0;   // ... covering no pixel at all
        unsigned long long tested = 0;      // inner-loop iterations (edge tests)
        unsigned long long covered = 0;     // pixels that passed
        unsigned long long lead = 0;        // tested before the first covered pixel of a row
        unsigned long long trail = 0;       // tested after the last covered pixel of a row
        unsigned long long bboxW = 0;       // summed row widths, for a mean
        unsigned long long walked = 0;      // summed WALKED row widths (span-narrowed)
        // ★★ cont.329g: sprites, so this census reports the WHOLE frame's coverage rather than
        // its triangles only. A sprite's covered area is its scissor- and band-clipped rect, known
        // once per draw -- no per-pixel cost. This is the CPU-side counterpart of the GL device's
        // GL_SAMPLES_PASSED, and the number the GL renderer's fragment count has to be judged
        // against: "18.6x the frame area" means nothing until the software path's own is known.
        unsigned long long sprites = 0;
        unsigned long long spritePixels = 0;
        // cont.327: lane utilisation of the four-wide groups as walked (4x1, x aligned to 4) and of a 4x2 BLOCK
        // (two consecutive rows of the same triangle, x aligned to 4: the block spans the UNION of the two walked
        // spans) -- the sizing of an 8-lane rasterizer on this geometry. lanes4x1 = 4 * groups per row;
        // lanes4x2 = 8 * groups per row PAIR (rows paired by y parity within a triangle; an unpaired last row
        // counts as a block with one empty row). `covered` is the same numerator for both.
        unsigned long long lanes4x1 = 0, lanes4x2 = 0, pairs = 0, pairRowsEmptyOne = 0;
        int pendY = -2, pendX0 = 0, pendX1 = -1; bool pend = false; // the pending even row of the current triangle
    };
    Gs2TriCensus g_triCensus;

    void gs2TriCensusDump(const char *where)
    {
        if (!s_gsTriCensus)
            return;
        const Gs2TriCensus &c = g_triCensus;
        const double tested = c.tested ? static_cast<double>(c.tested) : 1.0;
        const double rows = c.rows ? static_cast<double>(c.rows) : 1.0;
        std::fprintf(stderr,
                     "[gs2:tricensus/%s] tris=%llu rows=%llu (%.1f/tri, empty=%.2f%%)"
                     " meanRowW=%.1f\n",
                     where, c.tris, c.rows,
                     static_cast<double>(c.rows) / (c.tris ? static_cast<double>(c.tris) : 1.0),
                     100.0 * static_cast<double>(c.emptyRows) / rows,
                     static_cast<double>(c.bboxW) / rows);
        std::fprintf(stderr, "[gs2:tricensus/%s] lanes: 4x1=%llu (covered %.1f%% of lanes) 4x2=%llu (covered %.1f%%; blocks=%llu, %.1f%% with one empty row) -> 4x2 does %.2fx the 4x1 blocks' lane-work\n",
                     where, c.lanes4x1, 100.0 * static_cast<double>(c.covered) / (c.lanes4x1 ? static_cast<double>(c.lanes4x1) : 1.0),
                     c.lanes4x2, 100.0 * static_cast<double>(c.covered) / (c.lanes4x2 ? static_cast<double>(c.lanes4x2) : 1.0),
                     c.pairs, 100.0 * static_cast<double>(c.pairRowsEmptyOne) / (c.pairs ? static_cast<double>(c.pairs) : 1.0),
                     (c.lanes4x1 ? static_cast<double>(c.lanes4x2) / static_cast<double>(c.lanes4x1) : 0.0));
        std::fprintf(stderr, "[gs2:tricensus/%s] meanWalkedW=%.2f (%.1f%% of bbox row)\n",
                     where, static_cast<double>(c.walked) / rows,
                     100.0 * static_cast<double>(c.walked) /
                         (c.bboxW ? static_cast<double>(c.bboxW) : 1.0));
        std::fprintf(stderr,
                     "[gs2:tricensus/%s] COVERAGE tri-covered=%llu sprites=%llu sprite-covered=%llu"
                     " TOTAL=%llu\n",
                     where, c.covered, c.sprites, c.spritePixels, c.covered + c.spritePixels);
        std::fprintf(stderr,
                     "[gs2:tricensus/%s] tested=%llu covered=%llu (%.2f%%)"
                     " | WASTED lead=%.2f%% trail=%.2f%% total=%.2f%%\n",
                     where, c.tested, c.covered, 100.0 * static_cast<double>(c.covered) / tested,
                     100.0 * static_cast<double>(c.lead) / tested,
                     100.0 * static_cast<double>(c.trail) / tested,
                     100.0 * static_cast<double>(c.tested - c.covered) / tested);
    }
}
// PS2X_GS_CLUTVERIFY=1 (default OFF): differential oracle -- every cached hit is compared against
// a live LookupCLUT of the same texel. The bench's VRAM hash already proves pixel-equality, but it
// cannot distinguish "never sampled differently" from "correct", and per the cont.216 lesson a
// verifier that runs the candidate on a COPY proves computation, not execution. This one runs on
// the REAL sampling path. PS2X_GS_CLUTMUTATE=<n> injects a known defect to prove it can fail:
// 1 = never re-decode (a stale cache), 2 = index rotated by one entry.
static const bool s_gsClutVerify = []
{ const char *e = std::getenv("PS2X_GS_CLUTVERIFY"); return e && e[0] && e[0] != '0'; }();
static const int s_gsClutMutate = []
{ const char *e = std::getenv("PS2X_GS_CLUTMUTATE"); return e ? std::atoi(e) : 0; }();
// ★ One flag for BOTH diagnostics, so the PSMT4 fast path carries exactly ONE predicted branch.
// Measured, not assumed: inlining the verify block into the sampling lambda cost ~5% (73.3 -> 78.0
// ms at 1T) -- it is called four times per bilinear pixel, and the extra live state stopped it
// inlining. Everything diagnostic now lives behind this flag in a noinline function.
static const bool s_gsClutDiag = s_gsClutVerify || s_gsClutMutate != 0;

// ---- PS2X_GS_TEXVERIFY (default OFF): the differential oracle for the decoded-texture cache,
// built BEFORE the cache itself. cont.250 sections 21-22 lost three builds because the oracle came
// last and then had defects of its own; the resume rule is to build the oracle first and prove the
// OFF case on the SAME binary.
// What it compares: the production texel fetch (samplePointWrapped: m_draw.texRead / the
// devirtualised P4 traits + the decoded CLUT + applyTexa) against an INDEPENDENT decode built the
// way the cache will fill itself -- GSMem::ReadP4/ReadP8/ReadCT24/ReadCT32 into a LINEAR
// [v*texW+u] array, then CLUT + TEXA. Two independent implementations, so a disagreement is real;
// decoding by calling production per texel would be tautological and would prove nothing.
// It therefore tests exactly what cont.250 section 19 left open (CT24/T8/T4 each independently
// wrong on the GPU path): decode, addressing, AND the linear indexing/dimension handling.
// PS2X_GS_TEXMUTATE=<n> injects a known defect so the oracle is proven able to fail:
// 1 = index the linear array transposed, 2 = drop the TEXA transform, 3 = rotate the CLUT index.
// ⚠ Per the cont.227 measurement (inlining the CLUT verify into the sampling lambda cost ~5%),
// everything here sits behind ONE flag in a noinline lambda, never in the inlined fast path.
static const int s_gsTexVerifyMode = []
{ const char *e = std::getenv("PS2X_GS_TEXVERIFY"); return e ? std::atoi(e) : 0; }();
static const bool s_gsTexVerify = s_gsTexVerifyMode != 0;
// ★ PS2X_GS_TEXVERIFY=2: on every MISMATCH, decode that ONE texel FRESH (at sample time, from
// current VRAM) and report which of the two explanations it is. O(1) per check -- an earlier
// attempt re-decoded the whole texture per check, which is O(checks x texels) and unrunnable.
//   fresh == production, cached != production  ->  STALENESS: the fill was correct when made and
//        VRAM changed under it. The decoders are fine; the CACHE needs invalidation.
//   fresh != production                        ->  a real DECODE fault.
static const bool s_gsTexFresh = s_gsTexVerifyMode >= 2;
static const int s_gsTexMutate = []
{ const char *e = std::getenv("PS2X_GS_TEXMUTATE"); return e ? std::atoi(e) : 0; }();
static const bool s_gsTexDiag = s_gsTexVerify || s_gsTexMutate != 0;

// ★★★ cont.268 PS2X_GS_TEXWRAP (default ON; "=0" restores the per-texel dispatch) -- hoist the
// texture WRAP MODE out of the per-texel path. `wrapTextureCoordinate` is a `switch (mode & 3)` on
// `m_draw.texWrapU/V`, which is loop-invariant for a whole primitive, yet it is called FOUR TIMES
// PER PIXEL under bilinear (and it inlines as a compare chain -- 25 `$0x3` compare sites in the hot
// symbol, no jump table). PS2X_GS_TEXCENSUS on the level capture says the game only ever uses two
// pairs: REPEAT/REPEAT 96.15% and CLAMP/CLAMP 3.85%. So one predicted branch per pixel replaces the
// four dispatches, with each arm the case body copied verbatim -- bit-exact by construction.
// This is the cont.262 PIPESPEC move (an explicit list of measured-common cases, not a cross
// product) one level down, in the sampler.
static const bool s_gsTexWrap = []
{ const char *e = std::getenv("PS2X_GS_TEXWRAP"); return !(e && e[0] == '0'); }();

// ★★★ cont.268 PS2X_GS_TEXTAP (default ON; "=0" restores the general per-tap path) -- hoist the
// per-texel DIAGNOSTIC GATES. Every tap runs `if (s_gsTexCensus)`, `if (s_gsTexDiag)` and
// `if (s_gsClutDiag)` -- three static loads plus branches, twelve per pixel under bilinear -- all
// on globals fixed at startup, plus `if (fastP4)` on a per-draw value. When none is armed and the
// draw is on the devirtualised PSMT4 path (82.56% of this game's texels, cont.250 census), the tap
// is just the inlined swizzled address plus two loads; select that once per pixel instead.
static const bool s_gsTexTap = []
{ const char *e = std::getenv("PS2X_GS_TEXTAP"); return !(e && e[0] == '0'); }();

namespace
{
    struct Gs2TexVerify
    {
        unsigned long long checks = 0, mismatches = 0, fills = 0, fillTexels = 0, skipped = 0;
        unsigned long long checkByPsm[64] = {};
        unsigned long long missByPsm[64] = {};
        unsigned long long staleByPsm[64] = {};      // fresh decode == production -> the fill went STALE
        unsigned long long decodeBadByPsm[64] = {};  // fresh decode != production -> a real decode fault
    };
    Gs2TexVerify g_texVerify;

    void gs2TexVerifyDump(const char *where)
    {
        if (!s_gsTexDiag)
            return;
        const Gs2TexVerify &v = g_texVerify;
        std::fprintf(stderr,
                     "[gs2:texverify] %s checks=%llu MISMATCHES=%llu (%.4f%%) | fills=%llu texels=%llu"
                     " | skipped(unsupported)=%llu\n",
                     where, v.checks, v.mismatches,
                     v.checks ? 100.0 * double(v.mismatches) / double(v.checks) : 0.0,
                     v.fills, v.fillTexels, v.skipped);
        for (int i = 0; i < 64; ++i)
        {
            if (!v.checkByPsm[i])
                continue;
            std::fprintf(stderr,
                         "[gs2:texverify]    psm=0x%02x checks=%llu mismatch=%llu%s%s\n",
                         i, v.checkByPsm[i], v.missByPsm[i],
                         v.missByPsm[i] ? "   <-- WRONG" : "   ok",
                         (v.staleByPsm[i] || v.decodeBadByPsm[i])
                             ? (v.decodeBadByPsm[i] ? "   [DECODE FAULT]" : "   [STALE, decoders OK]")
                             : "");
            if (v.staleByPsm[i] || v.decodeBadByPsm[i])
                std::fprintf(stderr, "[gs2:texverify]        -> stale=%llu decode-fault=%llu\n",
                             v.staleByPsm[i], v.decodeBadByPsm[i]);
        }
    }
} // namespace

namespace
{
    struct Gs2ClutVerify
    {
        unsigned long long checks = 0;
        unsigned long long mismatches = 0;
        unsigned long long resolves = 0;     // resolveDraw calls with a T4-family texture
        unsigned long long decodes = 0;      // palette decodes actually performed
        unsigned long long keyMiss = 0;      // ... because the CLUT descriptor changed
        unsigned long long genMiss = 0;      // ... because VRAM was written outside the raster
        unsigned long long stompMiss = 0;    // ... because a draw covered the CLUT page
        unsigned long long stompsArmed = 0;  // draws that armed clutStomped
    };
    Gs2ClutVerify g_clutVerify;

    void gs2ClutVerifyDump(const char *where)
    {
        if (!s_gsClutVerify && !s_gsTexCensus)
            return;
        std::fprintf(stderr,
                     "[gs2:clutverify/%s] resolves=%llu decodes=%llu (keyMiss=%llu genMiss=%llu"
                     " stompMiss=%llu) stompsArmed=%llu checks=%llu mismatches=%llu%s\n",
                     where, g_clutVerify.resolves, g_clutVerify.decodes, g_clutVerify.keyMiss,
                     g_clutVerify.genMiss, g_clutVerify.stompMiss, g_clutVerify.stompsArmed,
                     g_clutVerify.checks, g_clutVerify.mismatches,
                     s_gsClutMutate ? " (MUTATED)" : "");
    }
}

// ★★ cont.227b: process-wide VRAM generation. Bumped by every writer that is NOT the
// rasterizer's own pixel path -- uploads, transfers, WriteVram pokes, reset -- so a cached
// palette decoded before one of them is re-decoded after it. The rasterizer's own writes are
// covered separately by clutStomped (a per-draw page test), because tracking them here would
// mean an atomic bump per PIXEL. PCSX2 splits the same way: GSClut::InvalidateRange is called
// from the transfer/draw invalidation paths, not from the scanline writers.
static std::atomic<uint64_t> g_gs2VramGen{0};

// ★★★ cont.250: PER-PAGE VRAM generation -- the foundation a persistent TEXTURE cache needs, and the
// piece g_gs2VramGen above cannot provide. That counter is process-wide, so ANY upload invalidates
// EVERY cached entry at once. For a 16-entry palette that is free (re-decoding is trivial); for
// textures it would mean re-decoding up to 128 entries of up to 1 MB each on every transfer, which is
// how the cont.250 §15 linear-decode spike ended up SLOWER than no cache at all.
// The split is preserved exactly as documented above, and as PCSX2 splits it: non-rasterizer writers
// (uploads, transfers, pokes, reset) bump pages here, while the rasterizer's own pixel writes are
// folded in ONCE PER DRAW -- never per pixel -- over the same two page ranges gs2DrawCoversPage()
// tests. Nothing consumes this yet; it is behaviour-neutral until a cache reads it.
static constexpr uint32_t kGs2Pages = 512u; // 4 MB VRAM / 8 KB per page
static std::atomic<uint32_t> g_gs2PageGen[kGs2Pages];
static std::atomic<unsigned long long> g_gs2PageBumps{0}; // volume probe: is per-draw bumping affordable?
// ★★★★ cont.371 PS2X_GS_GLR_SCANCACHE (default ON; =0 restores the per-page walks for an A/B).
// Every page generation only ever moves by +1 (the bump helpers below are its only writers), so a
// running sum per CHUNK of 64 pages, bumped alongside, equals the sum of its pages. The GL renderer
// walks page windows per DRAW -- gs2TargetGenSum (128 pages) and the seed scan (a whole target) --
// and on the Helm's Deep fight those walks were ~55-60% of the GS worker thread, which is the
// thread that sets the frame rate in heavy scenes (cont.371 perf + prodWaitMs). With the chunk
// sums an aligned 64-page run costs one load instead of 64, and a window whose sum has not moved
// has had no bump at all. Exact: same values, same comparisons, no state skipped that could change.
static constexpr uint32_t kGs2ChunkPages = 64u;
static constexpr uint32_t kGs2Chunks = kGs2Pages / kGs2ChunkPages;
static std::atomic<unsigned long long> g_gs2PageGenChunk[kGs2Chunks];
static const bool s_glScanCache = []
{ const char *e = std::getenv("PS2X_GS_GLR_SCANCACHE"); return !(e && e[0] == '0'); }();
// =2: the oracle -- every chunked sum is recomputed page by page, and every memo skip re-reads the
// whole range read-only; any disagreement is counted (printed on [gs2:glrseed]). Diagnostic only.
static const bool s_glScanVerify = []
{ const char *e = std::getenv("PS2X_GS_GLR_SCANCACHE"); return e && (e[0] == '2' || e[0] == '3'); }();
// =3: MUTATION -- the memo ignores the generation sum (a deliberately WRONG skip), to prove the
// oracle above can fail. Never for play.
static const bool s_glScanMutate = []
{ const char *e = std::getenv("PS2X_GS_GLR_SCANCACHE"); return e && e[0] == '3'; }();
static std::atomic<unsigned long long> g_glScanSumChecks{0}, g_glScanSumBad{0};

static inline void gs2BumpPageRange(uint32_t first, uint32_t last)
{
    if (first >= kGs2Pages)
        return;
    last = std::min(last, kGs2Pages - 1u);
    for (uint32_t i = first; i <= last; ++i)
        g_gs2PageGen[i].fetch_add(1u, std::memory_order_relaxed);
    for (uint32_t c = first / kGs2ChunkPages; c <= last / kGs2ChunkPages; ++c)
    {
        const uint32_t lo = std::max(first, c * kGs2ChunkPages);
        const uint32_t hi = std::min(last, c * kGs2ChunkPages + kGs2ChunkPages - 1u);
        g_gs2PageGenChunk[c].fetch_add(hi - lo + 1u, std::memory_order_relaxed);
    }
    g_gs2PageBumps.fetch_add(last - first + 1u, std::memory_order_relaxed);
}

// Sum of a page-generation array over `count` pages from `base` (wrapping at kGs2Pages), using the
// chunk sums for every whole aligned chunk inside the window and the pages themselves at its edges.
template <typename PageT>
static inline uint64_t gs2WindowGenSum(const std::atomic<PageT> *pages,
                                       const std::atomic<unsigned long long> *chunks,
                                       uint32_t base, uint32_t count)
{
    uint64_t g = 0;
    uint32_t i = 0;
    while (i < count)
    {
        const uint32_t p = (base + i) & (kGs2Pages - 1u);
        if (s_glScanCache && (p % kGs2ChunkPages) == 0u && count - i >= kGs2ChunkPages)
        {
            g += chunks[p / kGs2ChunkPages].load(std::memory_order_relaxed);
            i += kGs2ChunkPages;
        }
        else
        {
            g += pages[p].load(std::memory_order_relaxed);
            ++i;
        }
    }
    if (s_glScanVerify)
    {
        uint64_t ref = 0;
        for (uint32_t k = 0; k < count; ++k)
            ref += pages[(base + k) & (kGs2Pages - 1u)].load(std::memory_order_relaxed);
        g_glScanSumChecks.fetch_add(1ull, std::memory_order_relaxed);
        if (ref != g)
            g_glScanSumBad.fetch_add(1ull, std::memory_order_relaxed);
    }
    return g;
}

// Whole-VRAM writers (reset / init / bench restore) and, for now, the transfer paths whose exact
// range is not yet threaded through: identical semantics to the global counter, tightened later.
static inline void gs2BumpAllPages() { gs2BumpPageRange(0u, kGs2Pages - 1u); }

// ★★★ cont.330 TARGET-FROM-VRAM SEEDING: a SECOND per-page counter, bumped ONLY by the writers
// that deliver content with an exact rectangle -- host->local uploads and local->local blits.
// g_gs2PageGen above cannot serve this: clears and pokes bump it WHOLE-VRAM (gs2BumpAllPages), so
// every page of every render target would read as dirty every frame, and seeding a page from VRAM
// that GL (not the guest) last wrote would overwrite GL-rendered pixels with stale bytes. Keeping
// the content-delivery writers on their own counter makes "this page was WRITTEN BY THE GUEST
// since we last seeded it" answerable exactly, which is the per-rect dirty tracking PCSX2 does
// with GSTextureCache::InvalidateVideoMem.
static std::atomic<uint32_t> g_gs2PageXferGen[kGs2Pages];
static std::atomic<unsigned long long> g_gs2PageXferGenChunk[kGs2Chunks]; // cont.371, see g_gs2PageGenChunk
// ★★★★★ cont.331l PAGE OWNERSHIP: WHOEVER WROTE LAST OWNS THE PAGE.
// The authority seed (phase A) copies VRAM into a GL target whenever a page's transfer generation
// moved -- unconditionally, at the present resolve, i.e. AFTER the frame's own draws have been
// shipped. The device applies a batch's target seeds before its draws, so the later seed batch
// overwrote everything the frame drew: with cont.331k's frame-boundary flush landing the glyphs,
// `AUTHORITY=1` still presented the title screen WITHOUT "press start button" (text band 46.9 with
// the seed off, 17.4 with it on, byte-identical to the pre-flush build).
// The rule that fixes it is the authority model's own, on the clock the seed already uses: record,
// per page, the value of `g_gs2PageXferGen[pg]` in force when a GL draw last wrote that page. If
// that stamp still equals the page's current generation, GL drew AFTER the newest transfer there
// and owns the page; if the generation has moved on, VRAM is newer and the page is seeded. Uploads
// therefore keep refreshing the background every frame -- they bump the generation before the
// frame's draws -- while the frame's own geometry keeps its pages until the next upload arrives.
// The stamp starts at a generation no page can hold, so "never drawn" is not "drawn at gen 0".
// Only tracked when the authority model is on, so an unset run pays nothing.
static std::atomic<uint32_t> g_glPageDrawAtXfer[kGs2Pages];
static const bool g_glPageDrawAtXferInit = []
{
    for (uint32_t i = 0; i < kGs2Pages; ++i)
        g_glPageDrawAtXfer[i].store(0xFFFFFFFFu, std::memory_order_relaxed);
    return true;
}();
static unsigned long long g_glSeedOwnedSkips = 0;
// ★★★★★ cont.331p PS2X_GS_GLR_DROPTBP=<tbp> (default 0 = OFF, ABLATION + CENSUS): count -- and with
// PS2X_GS_GLR_DROPTBP_KILL=1 also SKIP -- every GL draw that samples that texture base. Built for
// the title screen's darken pass (`tbp=12288`, `Cd*(1-As)` with As = TEXA/128 = 0.5 exactly), which
// cont.331o showed drives the display target from a clear-colour max of 36 down to max=2 -- thirty-
// six halved four times -- while a correctly-landed seed reads max=238. Two questions in one run:
// how many times does the DEVICE see this pass per frame (the seam census already proves both
// renderers are SUBMITTED the same primitives, so a difference here is below the seam), and does the
// seeded picture survive once the pass is gone.
static const uint32_t s_glDropTbp = []
{ const char *e = std::getenv("PS2X_GS_GLR_DROPTBP"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static const bool s_glDropTbpKill = []
{ const char *e = std::getenv("PS2X_GS_GLR_DROPTBP_KILL"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_glDropTbpSeen = 0, g_glDropTbpKilled = 0;
// ★★★★★ cont.331n PS2X_GS_GLR_SRCSEED (default OFF): seed a target when it is used as a TEXTURE
// SOURCE. Phase A seeds a target when it is used as a DRAW DESTINATION and at the PRESENT RESOLVE;
// the third way a target is used is as a source, and that one was never covered. cont.331m's trace
// is the evidence: 3,120 of 3,123 samples of this game's composite sprite promote `tbp=12288` to
// the GL target at 0x180 with `gen{stamped=483 live=483}` -- `tgtFresh` believes the target is the
// freshest copy -- while the movie picture sits in VRAM (the 3 samples that fall through to the
// VRAM decode hash non-zero) and the target never received it. Seeding on use closes that.
// The seed goes out as its OWN batch for the SOURCE key, submitted before the batch that samples
// it, with the pending batch flushed first so nothing is reordered past it.
static std::unordered_map<uint64_t, unsigned long long> g_glSrcSeedSeq; // worker thread only
static unsigned long long g_srcSeedCalls = 0, g_srcSeedSeqSame = 0, g_srcSeedNoRects = 0,
                          g_srcSeedNoH = 0, g_srcSeedEvents = 0, g_srcSeedRects = 0;
// ★★★★★ cont.331n: page OWNERSHIP is a rule about the DISPLAY target at present time -- "do not
// overwrite what this frame just drew". A SOURCE seed is the opposite question: "does VRAM hold
// bytes this target has never received?" Running both through the one page filter made the source
// seed emit `no-rects=3057` out of 3,057 calls, because the guest's own full-screen draws into the
// texture buffer claim every page. thread_local rather than a member or a plain static: the seed is
// reached from the worker thread (source) AND the present thread (display, on the snapshot
// instance), so the two must not see each other's setting.
static thread_local bool t_seedHonourOwn = true;
static const bool s_glSrcSeed = []
{ const char *e = std::getenv("PS2X_GS_GLR_SRCSEED"); return e && e[0] && e[0] != '0'; }();
// ⚠ DEFAULT OFF -- MEASURED, AND NOT A NET WIN ON ITS OWN (cont.331l). With `PAGEOWN=1` the title
// screen gets its "press start button" back (text band 17.4 -> 44-47) and LOSES its background
// (whole-frame mean 92.5 -> 1.2), because this game composites the picture by DRAWING a full-screen
// sprite into the display buffer every frame: that draw stamps all 128 of the buffer's pages as
// GL-owned, so the uploaded movie picture is never seeded again. Page granularity cannot tell
// "GL drew real content here" from "GL drew a sprite that sampled an empty target here" -- the
// sprite's own source (texture memory at page 0x180) is a GL target that is starved the same way.
// Keep as the measured half of the answer; the other half is the SOURCE side (a sampled target
// must be fed from VRAM, or refuse promotion when VRAM is fresher), which is the next step.
static const bool s_glPageOwn = []
{ const char *e = std::getenv("PS2X_GS_GLR_PAGEOWN"); return e && e[0] && e[0] != '0'; }();
// ★★★★ cont.331: the PRESENT path runs on a DIFFERENT GSCpuBackend instance from the one that
// renders draws (the snapshot backend -- which is also why the device's present numbering repeats).
// So m_glTargetH, which the draw path fills, is EMPTY there, and the present-time seed declined
// 1027/1536 calls on "no target height" while looking like a plain null. Target metadata has to be
// visible wherever a target is USED, so the height lives in a global keyed by target page.
static std::atomic<uint32_t> g_glTargetHByPage[kGs2Pages];
// ★★★★ cont.331: and the SEED WATERMARK has the same problem as the height. m_glTargetSeedGen is a
// member, the present-side snapshot backend does not persist across presents, so every present-time
// call hit the emitter's "first observation only records a baseline" rule and emitted nothing --
// 1261 zero-rect calls that read exactly like "nothing was dirty". The watermark is therefore
// global and keyed by VRAM PAGE: the last transfer generation already seeded into whatever target
// owns that page. (Targets here do not share pages -- fbp=0x0 owns 0..111, fbp=0x80 owns 128..239.)
static std::atomic<uint32_t> g_glSeedSeenByPage[kGs2Pages];
static std::atomic<uint32_t> g_glSeedSeenInit[kGs2Pages];
// One bump per TRANSFER (not per page): the cheap gate that keeps the per-page scan off the
// per-draw path entirely when nothing has been transferred since the last check.
static std::atomic<unsigned long long> g_gs2XferSeq{0};
static inline void gs2BumpXferPage(uint32_t page)
{
    if (page < kGs2Pages)
    {
        g_gs2PageXferGen[page].fetch_add(1u, std::memory_order_relaxed);
        g_gs2PageXferGenChunk[page / kGs2ChunkPages].fetch_add(1u, std::memory_order_relaxed);
    }
}
static inline void gs2BumpXferPageRange(uint32_t first, uint32_t last)
{
    if (first >= kGs2Pages)
        return;
    last = std::min(last, kGs2Pages - 1u);
    for (uint32_t i = first; i <= last; ++i)
        g_gs2PageXferGen[i].fetch_add(1u, std::memory_order_relaxed);
    for (uint32_t c = first / kGs2ChunkPages; c <= last / kGs2ChunkPages; ++c)
    {
        const uint32_t lo = std::max(first, c * kGs2ChunkPages);
        const uint32_t hi = std::min(last, c * kGs2ChunkPages + kGs2ChunkPages - 1u);
        g_gs2PageXferGenChunk[c].fetch_add(hi - lo + 1u, std::memory_order_relaxed);
    }
}

// ★★★ cont.329k: the page-generation fingerprint of a GL target's pages (see m_glTargetGen).
// ★★★★ cont.331m PS2X_GS_GLR_TGTFRESH_PAGES (default 128, =32 restores the old window): the
// fingerprint covered only the target's FIRST 32 PAGES. A 512x512 32-bit render target is 128
// pages, so three quarters of it were outside the window -- and this game's movie uploads into
// texture memory at 0x180 land on pages 384..449 and 446..487, i.e. largely outside pages
// 384..415. Any VRAM write there was invisible to `tgtFresh`, so a draw sampling that target was
// promoted to a GL target that had never received the uploaded picture. 128 pages is one full
// 512x512 buffer; both sides of the comparison read the same window, so widening it is consistent.
static const uint32_t s_glTgtFreshPages = []
{
    const char *e = std::getenv("PS2X_GS_GLR_TGTFRESH_PAGES");
    const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 128;
    return static_cast<uint32_t>(v < 1 ? 1 : (v > 512 ? 512 : v));
}();
static uint64_t gs2TargetGenSum(uint64_t pageBase)
{
    return gs2WindowGenSum(g_gs2PageGen, g_gs2PageGenChunk, uint32_t(pageBase), s_glTgtFreshPages);
}


namespace
{
    uint64_t gs2ClutCacheKey(uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t srcPsm)
    {
        return static_cast<uint64_t>(cbp) | (static_cast<uint64_t>(cpsm) << 16) |
               (static_cast<uint64_t>(csm) << 24) | (static_cast<uint64_t>(csa) << 28) |
               (static_cast<uint64_t>(srcPsm) << 36);
    }

    // Does this draw's frame or Z target cover `page`? Conservative by construction: PageId is
    // monotonic in x and y, and dividing by the SMALLEST page extent (64x32, PSMCT32) over-counts
    // pages for every wider format, so the range is a superset of what the draw can touch.
    bool gs2DrawCoversPage(const GSContext &ctx, uint32_t fbp, uint32_t zbp, uint32_t page)
    {
        const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
        const uint32_t span = (static_cast<uint32_t>(ctx.scissor.y1) >> 5) * fbw +
                              (static_cast<uint32_t>(ctx.scissor.x1) >> 6);
        const uint32_t f0 = fbp / GSMem::BLOCKS_PER_PAGE;
        const uint32_t z0 = zbp / GSMem::BLOCKS_PER_PAGE;
        return (page >= f0 && page <= f0 + span) || (page >= z0 && page <= z0 + span);
    }
}

namespace
{
    struct Gs2TexCensus
    {
        unsigned long long calls = 0;    // SampleTexture entries
        unsigned long long linear = 0;   // ... bilinear (4 texel fetches each)
        unsigned long long mipped = 0;   // ... mip > 0
        unsigned long long fst = 0;      // ... FST (u/v) rather than STQ
        unsigned long long texels = 0;   // samplePoint() fetches
        unsigned long long texelByPsm[64] = {};
        unsigned long long clutCalls = 0;
        unsigned long long clutByCpsm[64] = {};
        unsigned long long clutCsm2 = 0;
        unsigned long long clutShadowHit = 0;
        unsigned long long wrap[4][4] = {}; // [wms][wmt], per call
        unsigned long long clutSwitch = 0;  // call-to-call change of the CLUT tuple
        uint64_t lastClutKey = ~0ull;
        std::unordered_map<uint64_t, unsigned long long> clutKeys; // tuple -> lookups
        // ★ cont.250: per-TEXTURE reuse. A decoded-texture cache only pays if a texture is SAMPLED
        // more often than it has TEXELS (the CLUT cache wins because 16 entries serve ~2410 lookups
        // per switch). key -> {samples, texel count}.
        std::unordered_map<uint64_t, std::pair<unsigned long long, uint32_t>> texKeys;
        // ★ cont.320: reuse PER UPLOAD GENERATION. cont.250's 3.03x above decodes each distinct
        // texture ONCE for the whole run, but 77-94% of uploads are per-frame re-uploads of the
        // same TEX0 (cont.271 census), and a decoded-texture cache must re-decode after every
        // upload that touches the texture's pages. So: a census-owned PRECISE per-page generation
        // (the production g_gs2PageGen is bumped whole-VRAM by every upload, useless here), bumped
        // by each upload chunk's exact page footprint (gs2RectPages, the updep geometry) and by a
        // local-to-local copy's destination; a per-draw signature over the texture's pages
        // (texGensIdx: what an INDEX cache -- linear P4/P8 indices, CLUT applied at sample time as
        // fastP4 already does -- would key on) and over texture + CLUT page (texGensFull: a
        // post-CLUT RGBA cache); samples per (texture, signature) = samples per DECODE.
        struct GenEnt { unsigned long long samples = 0; uint32_t texels = 0; uint64_t key = 0; };
        std::unordered_map<uint64_t, GenEnt> texGensIdx;
        std::unordered_map<uint64_t, GenEnt> texGensFull;
        std::unordered_map<uint64_t, unsigned> gensPerTex; // texture key -> distinct idx signatures
        uint32_t pageGen[512] = {};
        uint64_t curSigIdx = 0, curSigFull = 0;
        unsigned long long upChunks = 0, upXfers = 0, upPages = 0, l2lXfers = 0, bumpAllPoke = 0, bumpAllReset = 0;
        // ★ cont.320 (build 759): the same at 256-byte BLOCK granularity (16384 blocks), because a
        // texture smaller than a page (T4: 128x128 texels) shares its page with neighbours whose
        // uploads invalidate it FALSELY at page granularity. The block index is format-independent
        // (Address() / PixelsPerBlock()), so a CT32 upload into a T4-read texture resolves exactly.
        // Only the C32/Z32/P4/P8 tables are exported; the 16-bit family falls back to whole pages.
        std::unordered_map<uint64_t, GenEnt> texGensBlk;
        std::unordered_map<uint64_t, unsigned> gensPerTexBlk;
        uint32_t blockGen[16384] = {};
        uint64_t curSigBlk = 0;
        unsigned long long blkUnsupported = 0;
        struct TexInfo { uint32_t tbp0 = 0, tbw = 0, cbp = 0; uint8_t psm = 0, tw = 0, th = 0; };
        std::unordered_map<uint64_t, TexInfo> texInfo;
    };
    Gs2TexCensus g_texCensus;

    void gs2TexCensusBumpAll(unsigned long long &ctr)
    {
        if (!s_gsTexCensus) return;
        for (auto &g : g_texCensus.pageGen) ++g;
        for (auto &g : g_texCensus.blockGen) ++g;
        ++ctr;
    }
    void gs2TexCensusBumpMask(const Gs2PageMask &m)
    {
        for (int w = 0; w < 8; ++w)
        {
            uint64_t b = m.w[w];
            while (b)
            {
                const int i = __builtin_ctzll(b);
                b &= b - 1;
                ++g_texCensus.pageGen[w * 64 + i];
                ++g_texCensus.upPages;
            }
        }
    }
    uint64_t gs2TexCensusSig(const Gs2PageMask &m)
    {
        uint64_t h = 0xCBF29CE484222325ull;
        for (int w = 0; w < 8; ++w)
        {
            uint64_t b = m.w[w];
            while (b)
            {
                const int i = __builtin_ctzll(b);
                b &= b - 1;
                const unsigned p = static_cast<unsigned>(w * 64 + i);
                h ^= (static_cast<uint64_t>(g_texCensus.pageGen[p]) << 16) | p;
                h *= 0x100000001B3ull;
            }
        }
        return h;
    }
    struct Gs2BlkMask
    {
        uint64_t w[256] = {};
        void set(uint32_t b) { b &= 16383u; w[b >> 6] |= 1ull << (b & 63u); }
    };
    // The 256-byte block holding pixel (x, y) of a buffer at block address `base`, width bw (64-px
    // units), in format psm -- or ~0u when the format's table is not exported.
    uint32_t gs2BlockOf(uint8_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
    {
        using namespace GSMem;
        switch (psm)
        {
        case GS_PSM_CT32: case GS_PSM_CT24: case GS_PSM_T8H: case GS_PSM_T4HL: case GS_PSM_T4HH:
            return PixelStorageTraits<C32>::Address(PageTableC32, base, bw, x, y) / static_cast<uint32_t>(PixelStorageTraits<C32>::PixelsPerBlock());
        case GS_PSM_Z32: case GS_PSM_Z24:
            return PixelStorageTraits<Z32>::Address(PageTableZ32, base, bw, x, y) / static_cast<uint32_t>(PixelStorageTraits<Z32>::PixelsPerBlock());
        case GS_PSM_T8:
            return PixelStorageTraits<P8>::Address(PageTableP8, base, bw, x, y) / static_cast<uint32_t>(PixelStorageTraits<P8>::PixelsPerBlock());
        case GS_PSM_T4:
            return PixelStorageTraits<P4>::Address(PageTableP4, base, bw, x, y) / static_cast<uint32_t>(PixelStorageTraits<P4>::PixelsPerBlock());
        default:
            return ~0u;
        }
    }
    // Blocks of the pixel rectangle [x0,x1]x[y0,y1]: one gs2BlockOf per block-sized tile (the
    // block's pixel extent per format is its ColumnExtent: C32-family 8x8, 16-bit 16x8, P8 16x16,
    // P4 32x16), stepping to the next tile boundary so unaligned edges are covered.
    void gs2RectBlocks(Gs2BlkMask &m, uint8_t psm, uint32_t base, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
    {
        uint32_t bwPx = 8u, bhPx = 8u;
        switch (psm)
        {
        case GS_PSM_T8: bwPx = 16u; bhPx = 16u; break;
        case GS_PSM_T4: bwPx = 32u; bhPx = 16u; break;
        case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S: bwPx = 16u; bhPx = 8u; break;
        default: break;
        }
        if (gs2BlockOf(psm, base, bw, x0, y0) == ~0u)
        {
            // unsupported format: every block of every page the rectangle touches
            Gs2PageMask pm; gs2RectPages(pm, base / 32u, bw, psm, x0, y0, x1, y1);
            for (int w = 0; w < 8; ++w) { uint64_t b = pm.w[w]; while (b) { const int i = __builtin_ctzll(b); b &= b - 1;
                for (uint32_t k = 0; k < 32u; ++k) m.set(static_cast<uint32_t>(w * 64 + i) * 32u + k); } }
            ++g_texCensus.blkUnsupported;
            return;
        }
        for (uint32_t y = y0; y <= y1; y = (y / bhPx + 1u) * bhPx)
            for (uint32_t x = x0; x <= x1; x = (x / bwPx + 1u) * bwPx)
                m.set(gs2BlockOf(psm, base, bw, x, y));
    }
    void gs2TexCensusBumpBlk(const Gs2BlkMask &m)
    {
        for (int w = 0; w < 256; ++w) { uint64_t b = m.w[w]; while (b) { const int i = __builtin_ctzll(b); b &= b - 1; ++g_texCensus.blockGen[w * 64 + i]; } }
    }
    uint64_t gs2TexCensusSigBlk(const Gs2BlkMask &m)
    {
        uint64_t h = 0xCBF29CE484222325ull;
        for (int w = 0; w < 256; ++w)
        {
            uint64_t b = m.w[w];
            while (b)
            {
                const int i = __builtin_ctzll(b); b &= b - 1;
                const unsigned p = static_cast<unsigned>(w * 64 + i);
                h ^= (static_cast<uint64_t>(g_texCensus.blockGen[p]) << 16) | p;
                h *= 0x100000001B3ull;
            }
        }
        return h;
    }

    uint64_t gs2ClutKey(uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t srcPsm)
    {
        return static_cast<uint64_t>(cbp) | (static_cast<uint64_t>(cpsm) << 16) |
               (static_cast<uint64_t>(csm) << 24) | (static_cast<uint64_t>(csa) << 28) |
               (static_cast<uint64_t>(srcPsm) << 36);
    }

    const char *gs2PsmName(unsigned p)
    {
        switch (p)
        {
        case GS_PSM_CT32: return "CT32";
        case GS_PSM_CT24: return "CT24";
        case GS_PSM_CT16: return "CT16";
        case GS_PSM_CT16S: return "CT16S";
        case GS_PSM_T8: return "T8";
        case GS_PSM_T4: return "T4";
        case GS_PSM_T8H: return "T8H";
        case GS_PSM_T4HL: return "T4HL";
        case GS_PSM_T4HH: return "T4HH";
        case GS_PSM_Z32: return "Z32";
        case GS_PSM_Z24: return "Z24";
        case GS_PSM_Z16: return "Z16";
        case GS_PSM_Z16S: return "Z16S";
        default: return "?";
        }
    }

    void gs2TexCensusDump(const char *where)
    {
        if (!s_gsTexCensus)
            return;
        const Gs2TexCensus &c = g_texCensus;
        const double calls = c.calls ? static_cast<double>(c.calls) : 1.0;
        const double texels = c.texels ? static_cast<double>(c.texels) : 1.0;
        std::fprintf(stderr,
                     "[gs2:texcensus/%s] calls=%llu linear=%.2f%% mip=%.2f%% fst=%.2f%%"
                     " texels=%llu (%.3f/call)\n",
                     where, c.calls, 100.0 * static_cast<double>(c.linear) / calls,
                     100.0 * static_cast<double>(c.mipped) / calls,
                     100.0 * static_cast<double>(c.fst) / calls,
                     c.texels, static_cast<double>(c.texels) / calls);
        for (unsigned i = 0; i < 64u; ++i)
            if (c.texelByPsm[i] != 0ull)
                std::fprintf(stderr, "[gs2:texcensus/%s]   texel psm 0x%02x %-5s %12llu  %6.2f%%\n",
                             where, i, gs2PsmName(i), c.texelByPsm[i],
                             100.0 * static_cast<double>(c.texelByPsm[i]) / texels);
        const double clut = c.clutCalls ? static_cast<double>(c.clutCalls) : 1.0;
        {
            unsigned long long texSamples = 0, texDecode = 0;
            unsigned long long payers = 0; // textures sampled more than they have texels
            for (const auto &kv : c.texKeys)
            {
                texSamples += kv.second.first;
                texDecode += kv.second.second;
                if (kv.second.first > kv.second.second) ++payers;
            }
            std::fprintf(stderr,
                         "[gs2:texcensus/%s] DISTINCT TEXTURES=%zu samples=%llu decodeTexels=%llu"
                         " reuse=%.2fx payers=%llu (%.1f%% of textures sampled more than they have texels)\n",
                         where, c.texKeys.size(), texSamples, texDecode,
                         texDecode ? double(texSamples) / double(texDecode) : 0.0, payers,
                         c.texKeys.empty() ? 0.0 : 100.0 * double(payers) / double(c.texKeys.size()));
        }
        // ★ cont.320: the gen-aware report (see the struct comment). Each (texture, signature)
        // pair is ONE decode a page-generation-invalidated cache would perform; the histogram
        // bins pairs by samples/texels, weighted by samples (what share of the gathers a cache
        // would serve at that amortisation) and by decode texels (where the decode work goes).
        {
            auto report = [&](const char *what, const std::unordered_map<uint64_t, Gs2TexCensus::GenEnt> &m) {
                static const double kEdge[8] = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e300};
                unsigned long long bySamples[8] = {}, byTexels[8] = {}, pairs8[8] = {};
                unsigned long long samples = 0, texels = 0;
                for (const auto &kv : m)
                {
                    const double r = kv.second.texels ? double(kv.second.samples) / double(kv.second.texels) : 1e300;
                    unsigned bin = 0; while (bin < 7u && r >= kEdge[bin]) ++bin;
                    bySamples[bin] += kv.second.samples; byTexels[bin] += kv.second.texels; ++pairs8[bin];
                    samples += kv.second.samples; texels += kv.second.texels;
                }
                std::fprintf(stderr, "[gs2:texcensus/%s] GEN-AWARE %s: pairs=%zu decodeTexels=%llu samples=%llu reuse/decode=%.2fx\n",
                             where, what, m.size(), texels, samples, texels ? double(samples) / double(texels) : 0.0);
                static const char *kName[8] = {"<0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", ">=16"};
                for (unsigned b = 0; b < 8u; ++b)
                    if (pairs8[b])
                        std::fprintf(stderr, "[gs2:texcensus/%s]   %s reuse %-8s pairs=%8llu samples=%6.2f%% decodeTexels=%6.2f%%\n",
                                     where, what, kName[b], pairs8[b],
                                     samples ? 100.0 * double(bySamples[b]) / double(samples) : 0.0,
                                     texels ? 100.0 * double(byTexels[b]) / double(texels) : 0.0);
            };
            report("idx", c.texGensIdx);
            report("full", c.texGensFull);
            report("blk", c.texGensBlk);
            // per-texture: gens (page / block), samples, decode cost, reuse per generation, share of
            // generations with reuse >= 4x -- is the reuse a stable property of the TEXTURE (a TEX0-keyed
            // predictor would work) and is the low-reuse decode mass false page sharing?
            struct Row { uint64_t key; unsigned gensP, gensB, hiB; unsigned long long samples, texels; };
            std::unordered_map<uint64_t, Row> rows;
            for (const auto &kv : c.texGensBlk)
            {
                Row &r = rows[kv.second.key];
                r.key = kv.second.key; ++r.gensB; r.samples += kv.second.samples; r.texels = kv.second.texels;
                if (kv.second.samples >= 4ull * kv.second.texels) ++r.hiB;
            }
            std::vector<Row> v; v.reserve(rows.size());
            unsigned long long totS = 0, totD = 0;
            for (auto &kv : rows)
            {
                auto it = c.gensPerTex.find(kv.first); kv.second.gensP = it == c.gensPerTex.end() ? 0u : it->second;
                totS += kv.second.samples; totD += static_cast<unsigned long long>(kv.second.gensB) * kv.second.texels;
                v.push_back(kv.second);
            }
            auto printRows = [&](const char *title, size_t n) {
                std::fprintf(stderr, "[gs2:texcensus/%s] TOP TEXTURES by %s: psm  size     tbp0  cbp  gensPage gensBlk samples%% decode%% reuse/gen hi%%\n", where, title);
                for (size_t i = 0; i < n && i < v.size(); ++i)
                {
                    const Row &r = v[i]; const auto ti = c.texInfo.find(r.key);
                    const Gs2TexCensus::TexInfo inf = ti == c.texInfo.end() ? Gs2TexCensus::TexInfo{} : ti->second;
                    const unsigned long long dec = static_cast<unsigned long long>(r.gensB) * r.texels;
                    std::fprintf(stderr, "[gs2:texcensus/%s]   %-5s %4ux%-4u %5u %5u %8u %7u %7.2f%% %7.2f%% %8.2fx %5.1f%%\n",
                                 where, gs2PsmName(inf.psm), 1u << inf.tw, 1u << inf.th, inf.tbp0, inf.cbp, r.gensP, r.gensB,
                                 totS ? 100.0 * double(r.samples) / double(totS) : 0.0, totD ? 100.0 * double(dec) / double(totD) : 0.0,
                                 dec ? double(r.samples) / double(dec) : 0.0, r.gensB ? 100.0 * double(r.hiB) / double(r.gensB) : 0.0);
                }
            };
            std::sort(v.begin(), v.end(), [](const Row &a, const Row &b) { return (unsigned long long)a.gensB * a.texels > (unsigned long long)b.gensB * b.texels; });
            printRows("DECODE COST (blk gens x texels)", 24);
            std::sort(v.begin(), v.end(), [](const Row &a, const Row &b) { return a.samples > b.samples; });
            printRows("SAMPLES", 12);
            std::fprintf(stderr, "[gs2:texcensus/%s] GEN-AWARE blk unsupported-format rects=%llu\n", where, c.blkUnsupported);
            unsigned long long g1 = 0, g10 = 0, g100 = 0, gMore = 0; unsigned gMax = 0;
            for (const auto &kv : c.gensPerTex)
            {
                if (kv.second <= 1u) ++g1; else if (kv.second <= 10u) ++g10; else if (kv.second <= 100u) ++g100; else ++gMore;
                gMax = std::max(gMax, kv.second);
            }
            std::fprintf(stderr, "[gs2:texcensus/%s] GEN-AWARE textures by distinct idx generations: 1:%llu 2-10:%llu 11-100:%llu >100:%llu max=%u"
                                 " | uploads xfers=%llu chunks=%llu pages=%llu l2l=%llu bumpAll poke=%llu reset=%llu\n",
                         where, g1, g10, g100, gMore, gMax, c.upXfers, c.upChunks, c.upPages, c.l2lXfers, c.bumpAllPoke, c.bumpAllReset);
        }
        std::fprintf(stderr,
                     "[gs2:texcensus/%s] clut lookups=%llu (%.2f%% of texels) csm2=%.2f%%"
                     " shadowHit=%llu | configs=%zu switches=%llu = %.1f lookups/switch\n",
                     where, c.clutCalls, 100.0 * clut / texels,
                     100.0 * static_cast<double>(c.clutCsm2) / clut, c.clutShadowHit,
                     c.clutKeys.size(), c.clutSwitch,
                     static_cast<double>(c.clutCalls) / (c.clutSwitch ? static_cast<double>(c.clutSwitch) : 1.0));
        for (unsigned i = 0; i < 64u; ++i)
            if (c.clutByCpsm[i] != 0ull)
                std::fprintf(stderr, "[gs2:texcensus/%s]   clut cpsm 0x%02x %-5s %12llu  %6.2f%%\n",
                             where, i, gs2PsmName(i), c.clutByCpsm[i],
                             100.0 * static_cast<double>(c.clutByCpsm[i]) / clut);
        static const char *kWrap[4] = {"REPEAT", "CLAMP", "RGN_CLAMP", "RGN_REPEAT"};
        for (unsigned u = 0; u < 4u; ++u)
            for (unsigned v = 0; v < 4u; ++v)
                if (c.wrap[u][v] != 0ull)
                    std::fprintf(stderr, "[gs2:texcensus/%s]   wrap %-10s/%-10s %12llu  %6.2f%%\n",
                                 where, kWrap[u], kWrap[v], c.wrap[u][v],
                                 100.0 * static_cast<double>(c.wrap[u][v]) / calls);
    }
}

// ★★★ cont.320 PS2X_GS_TEXCACHE (default ON since build 764; "=0" restores the swizzled tap): the DECODED-INDEX TEXTURE CACHE, the
// row-84 per-draw hoist one level down -- the texel GATHER. The cont.320 reuse census (row 85)
// measured, per upload generation on the recorded Helm's Deep fight: 1.68x samples per decoded
// texel for an unconditional cache, BUT bimodal and stable per TEX0 -- (texture, generation) pairs
// reused >= 4x carry 61% of all gathers for 5.8% of the decode, the 256x256 stream textures sit at
// 0.1-1.6x and hold half of it. So: decode a level only for a key whose reuse HISTORY says it pays.
//   - per raster thread, like the CLUT cache (every thread walks the same run in the same order and
//     applies the same tests, so no atomics on the hot path -- cont.250 sections 16/21);
//   - keyed on the RESOLVED level (tbp/tbw/psm/w/h; each mip is its own entry), PSMT4/PSMT8 with a
//     decoded palette only (texFastP4/P8), REPEAT/REPEAT or CLAMP/CLAMP wrap (the linear index
//     needs u < w, v < h), <= 64K texels and <= 16 pages;
//   - validity = the exact per-page UPLOAD generation g_gs2PageGen over the level's footprint (made
//     precise here: UploadImageUnlocked bumps the chunk's pages, not the whole VRAM) + a per-draw
//     STOMP test (noteDrawWrites: the draw's frame/Z page ranges vs each cached level's pages, the
//     CLUT stomp generalised; a stomped or self-covered level is served swizzled for that draw);
//   - the PREDICTOR: per key, taps served per generation / texels, EMA over generations; a level is
//     decoded when the EMA >= PS2X_GS_TEXCACHE_THR (default 4 = the census's >=4x bucket; 0 = decode
//     on first touch, the unconditional variant). Unknown keys learn for one generation first.
//   - the tap: clut[lin[v*w+u]] in place of clut[PixelStorageTraits<>::Read(...)] -- the same index
//     byte, so bit-exact by construction; the fill IS the swizzled read, once per texel.
// Oracles: the bench hash (four-wide arm + stomps), PS2X_GS_TEXVERIFY on a live fight (the scalar
// arm serves the cache too, so the independent decode checks it), PS2X_GS_TEXCACHE_MUTATE 1 = ignore
// upload generations / 2 = ignore stomps (TEXVERIFY must then report staleness).
static const bool s_gsTexCache = []
{ const char *e = std::getenv("PS2X_GS_TEXCACHE"); return !(e && e[0] == '0'); }();
static const float s_gsTexCacheThr = []
{ const char *e = std::getenv("PS2X_GS_TEXCACHE_THR"); return e ? static_cast<float>(std::atof(e)) : 4.0f; }();
static const int s_gsTexCacheMutate = []
{ const char *e = std::getenv("PS2X_GS_TEXCACHE_MUTATE"); return e ? std::atoi(e) : 0; }();
namespace
{
    constexpr int kTexSlots = 16;
    constexpr uint32_t kTexMaxPages = 16u;
    constexpr uint32_t kTexMaxTexels = 65536u;
    struct TexSlot
    {
        uint64_t key = ~0ull;
        uint32_t nPages = 0u;
        uint32_t pages[kTexMaxPages] = {};
        uint32_t gens[kTexMaxPages] = {};
        uint32_t w = 0u, h = 0u;
        bool stomped = false;
        uint64_t lastUse = 0u;
        std::vector<uint8_t> idx;
    };
    struct TexPred
    {
        uint64_t sig = 0u;
        unsigned long long taps = 0u;
        float ema = -1.0f; // < 0 = no completed generation yet
        uint32_t gens = 0u;
        uint32_t texels = 0u;
    };
    struct TexCacheStats
    {
        unsigned long long drawsLin = 0, drawsSwz = 0, drawsDecl = 0, drawsInel = 0, fills = 0, fillTexels = 0,
                           invGen = 0, invStomp = 0, tapsLin = 0, tapsSwz = 0;
        unsigned pending = 0;
    };
    // cont.322c: a per-key census of the draws the cache does NOT serve (ineligible, or served live) with
    // their swizzled tap counts -- printed by gs2TexLiveCensusPrint() at the end of a bench run. Reasons:
    // 1 format (not P4/P8), 2 wrap class 2 (mixed / region modes), 3 size (> kTexMaxTexels), 4 too many
    // pages, 5 the draw writes the pages it samples, 6 a stomped slot served live, 7 declined by the predictor.
    struct TexLiveEntry { unsigned long long taps = 0, draws = 0; uint32_t tbp = 0, w = 0, h = 0; uint8_t psm = 0, reason = 0, wrapU = 0, wrapV = 0; };
    thread_local std::unordered_map<uint64_t, TexLiveEntry> t_texLive;
    thread_local TexLiveEntry *t_texLivePending = nullptr;
    // ★ cont.322c PS2X_GS_TEXCACHE_SLOTS (default 16 = the former constant): the per-thread slot count. WHY a
    // knob: the bench census (build 777) showed the cache REFILLING whole textures every rep -- 810 fills per
    // 3 reps at thr 4, 2160 at thr 0, with 141 live keys against 16 slots -- so the "fill cost" measured in
    // cont.320b/322c was thrash, not first-touch decode. A 256x256 slot is 64 KB; 128 slots = 8 MB per thread.
    const int s_gsTexSlots = []
    { const char *e = std::getenv("PS2X_GS_TEXCACHE_SLOTS"); const int v = e && e[0] ? std::atoi(e) : kTexSlots; return v < 1 ? 1 : v > 4096 ? 4096 : v; }();
    thread_local std::vector<TexSlot> t_texSlots(static_cast<size_t>(s_gsTexSlots));
    thread_local std::unordered_map<uint64_t, TexPred> t_texPreds;
    thread_local uint64_t t_texClock = 0u;
    thread_local TexCacheStats t_texStats;
    // cont.321b: the page-interval UNION of this thread's live slots, so noteDrawWrites can reject a draw
    // in a handful of compares. It is now called for every primitive the dynamic claim loop skips (see
    // bandDynClaimLoop), i.e. ~all of a run's primitives per group, so the full slot walk must be rare.
    thread_local uint32_t t_texUnionLo = ~0u, t_texUnionHi = 0u;
    thread_local unsigned long long t_texFillSeq = 0ull; // monotonic (t_texStats.fills is flushed to zero periodically)
    thread_local bool t_texUnionDirty = true;
    inline void texUnionRefresh()
    {
        uint32_t lo = ~0u, hi = 0u;
        for (const TexSlot &sl : t_texSlots)
        {
            if (sl.key == ~0ull || sl.stomped) continue;
            for (uint32_t i = 0; i < sl.nPages; ++i) { if (sl.pages[i] < lo) lo = sl.pages[i]; if (sl.pages[i] > hi) hi = sl.pages[i]; }
        }
        t_texUnionLo = lo; t_texUnionHi = hi; t_texUnionDirty = false;
    }
    // ★ cont.324b PS2X_GS_STOMPUNION (default 1; "=0" = the plain slot walk): the per-draw stomp walk over the cached texture
    // levels, SHARED by resolveDraw (every draw) and noteDrawWrites (the band-skipped draws), behind cont.321b's union
    // rejection. resolveDraw had the plain walk -- 16 slots x their pages x gs2DrawCoversPage on every primitive -- and the
    // cont.324 annotate put 76% of resolveDraw (2.5% of the Helm's bench) on it. Exact: the union is the min/max page over the
    // live, un-stomped slots (refreshed whenever a slot changes), the draw covers two page intervals, so no overlap with the
    // union = no slot page can be covered = the walk would mark nothing.
    static const bool s_gsStompUnion = []
    { const char *e = std::getenv("PS2X_GS_STOMPUNION"); return !(e && e[0] == '0'); }();
    inline void texStompSlots(const GSContext &ctx, uint32_t fbp, uint32_t zbp)
    {
        if (s_gsStompUnion)
        {
            if (t_texUnionDirty) texUnionRefresh();
            if (t_texUnionLo > t_texUnionHi) return;
            const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
            const uint32_t span = (static_cast<uint32_t>(ctx.scissor.y1) >> 5) * fbw + (static_cast<uint32_t>(ctx.scissor.x1) >> 6);
            const uint32_t f0 = fbp / GSMem::BLOCKS_PER_PAGE, z0 = zbp / GSMem::BLOCKS_PER_PAGE;
            if (!((f0 <= t_texUnionHi && f0 + span >= t_texUnionLo) || (z0 <= t_texUnionHi && z0 + span >= t_texUnionLo)))
                return;
        }
        for (TexSlot &sl : t_texSlots)
        {
            if (sl.key == ~0ull || sl.stomped) continue;
            for (uint32_t i = 0; i < sl.nPages; ++i)
                if (gs2DrawCoversPage(ctx, fbp, zbp, sl.pages[i])) { sl.stomped = true; t_texUnionDirty = true; break; }
        }
    }
    std::atomic<unsigned long long> g_texCacheDrawsLin{0}, g_texCacheDrawsSwz{0}, g_texCacheDrawsDecl{0}, g_texCacheDrawsInel{0},
        g_texCacheFills{0}, g_texCacheFillTexels{0}, g_texCacheInvGen{0}, g_texCacheInvStomp{0}, g_texCacheTapsLin{0}, g_texCacheTapsSwz{0};
    void texCacheFlushStats(bool force)
    {
        TexCacheStats &t = t_texStats;
        if (!force && ++t.pending < 1024u) return;
        t.pending = 0;
        g_texCacheDrawsLin.fetch_add(t.drawsLin, std::memory_order_relaxed); g_texCacheDrawsSwz.fetch_add(t.drawsSwz, std::memory_order_relaxed);
        g_texCacheDrawsDecl.fetch_add(t.drawsDecl, std::memory_order_relaxed); g_texCacheDrawsInel.fetch_add(t.drawsInel, std::memory_order_relaxed);
        g_texCacheFills.fetch_add(t.fills, std::memory_order_relaxed); g_texCacheFillTexels.fetch_add(t.fillTexels, std::memory_order_relaxed);
        g_texCacheInvGen.fetch_add(t.invGen, std::memory_order_relaxed); g_texCacheInvStomp.fetch_add(t.invStomp, std::memory_order_relaxed);
        g_texCacheTapsLin.fetch_add(t.tapsLin, std::memory_order_relaxed); g_texCacheTapsSwz.fetch_add(t.tapsSwz, std::memory_order_relaxed);
        t = TexCacheStats{};
    }
    void gs2TexLiveCensusPrint(const char *what)
    {
        std::vector<const TexLiveEntry *> v;
        unsigned long long tot = 0ull;
        for (const auto &kv : t_texLive) { v.push_back(&kv.second); tot += kv.second.taps; }
        std::sort(v.begin(), v.end(), [](const TexLiveEntry *a, const TexLiveEntry *b) { return a->taps > b->taps; });
        std::fprintf(stderr, "[gs2:texlive] %s: %zu keys, swizzled taps=%llu (this thread) | reason 1 fmt 2 wrap 3 size 4 pages 5 self-write 6 stomped 7 declined\n", what, v.size(), tot);
        size_t shown = 0;
        for (const TexLiveEntry *e : v)
        {
            if (shown++ >= 16) break;
            std::fprintf(stderr, "[gs2:texlive]   %5.1f%%  taps=%llu draws=%llu reason=%u psm=0x%02x tbp=%u %ux%u wrap=%u/%u\n",
                         tot ? 100.0 * double(e->taps) / double(tot) : 0.0, e->taps, e->draws, e->reason, e->psm, e->tbp, e->w, e->h, e->wrapU, e->wrapV);
        }
    }
    void gs2TexCachePrint()
    {
        if (!s_gsTexCache) return;
        texCacheFlushStats(true); // this thread's share; workers flush every 1024 draws
        const unsigned long long lin = g_texCacheTapsLin.load(), swz = g_texCacheTapsSwz.load();
        std::fprintf(stderr, "[gs2:texcache] thr=%.1f draws lin=%llu swz=%llu declined=%llu inel=%llu | fills=%llu texels=%llu | inval gen=%llu stomp=%llu | taps lin=%llu (%.1f%%) swz=%llu\n",
                     static_cast<double>(s_gsTexCacheThr), g_texCacheDrawsLin.load(), g_texCacheDrawsSwz.load(), g_texCacheDrawsDecl.load(), g_texCacheDrawsInel.load(),
                     g_texCacheFills.load(), g_texCacheFillTexels.load(), g_texCacheInvGen.load(), g_texCacheInvStomp.load(),
                     lin, (lin + swz) ? 100.0 * double(lin) / double(lin + swz) : 0.0, swz);
    }
}

// ★ cont.322c PS2X_GS_TEXFILL (default 1): the decoded-index cache FILL walks VRAM one 256-byte BLOCK at
// a time instead of resolving every texel through PixelStorageTraits::Read (PageId divide/multiply, the
// three-level page table, the bit maths -- ~20 instructions per texel). WHY: on the Helm's Deep bench
// the swizzled taps inside the four-wide loops are ~16% of the instructions (PixelStorageTraits::Read
// 15.3% + the tap lambdas 3.5%), and decoding EVERY texture (`PS2X_GS_TEXCACHE_THR=0`) cost exactly
// what those taps saved (+0.6% net, 54.08 vs 53.77 G): the per-texel fill was as expensive as the tap.
// A block's INTERNAL layout is the same for every block (the column swizzle), so it is one table:
// the page table's block 0 gives, for texel (x, y) of a block, its pixel index within the block --
// derived from the SAME tables the per-texel read uses, so the two agree by construction (the
// `PS2X_GS_TEXVERIFY` oracle and the capture hashes are the gates). The block's byte base comes from
// the same Address() call the per-texel read makes, once per block. PCSX2 reference: GS/GSBlock.h
// `ReadBlock4P` / `ReadBlock8` -- the software renderer's texture cache (GSTextureCacheSW) unswizzles
// whole blocks into a linear buffer and never samples swizzled memory per texel; its kernels are the
// shuffle form of this table walk (`=0` = the per-texel fill, A/B).
static const bool s_gsTexFillBlock = []
{ const char *e = std::getenv("PS2X_GS_TEXFILL"); return !(e && e[0] == '0'); }();
namespace
{
    struct Gs2BlockOrder
    {
        uint16_t p4[16 * 32]; // (y, x) of a PSMT4 block (32x16) -> pixel index within the block (0..511)
        uint16_t p8[16 * 16]; // (y, x) of a PSMT8 block (16x16) -> pixel index within the block (0..255)
    };
    const Gs2BlockOrder &gs2BlockOrder()
    {
        static const Gs2BlockOrder o = []
        {
            Gs2BlockOrder r{};
            for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 32; ++x)
                    r.p4[y * 32 + x] = static_cast<uint16_t>(GSMem::PageTableP4[0][y][x] % GSMem::PixelStorageTraits<GSMem::P4>::PixelsPerBlock());
            for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 16; ++x)
                    r.p8[y * 16 + x] = static_cast<uint16_t>(GSMem::PageTableP8[0][y][x] % GSMem::PixelStorageTraits<GSMem::P8>::PixelsPerBlock());
            return r;
        }();
        return o;
    }
    void gs2TexFillBlocks(uint8_t *dst, uint32_t w, uint32_t h, uint32_t tbp, uint32_t tbw, bool p4, uint8_t *vram)
    {
        const Gs2BlockOrder &ord = gs2BlockOrder();
        const uint32_t bw = p4 ? 32u : 16u, bh = 16u;
        alignas(16) uint8_t tmp[32 * 16];
        for (uint32_t by = 0; by < h; by += bh)
        {
            for (uint32_t bx = 0; bx < w; bx += bw)
            {
                const uint32_t idx = p4 ? GSMem::PixelStorageTraits<GSMem::P4>::Address(GSMem::PageTableP4, tbp, tbw, bx, by)
                                        : GSMem::PixelStorageTraits<GSMem::P8>::Address(GSMem::PageTableP8, tbp, tbw, bx, by);
                const uint32_t cw = std::min(bw, w - bx), ch = std::min(bh, h - by);
                const bool whole = (cw == bw && ch == bh);
                uint8_t *d = whole ? dst + static_cast<size_t>(by) * w + bx : tmp;
                const uint32_t pitch = whole ? w : bw;
                if (p4)
                {
                    if ((idx & 511u) != 0u) // not block-aligned (never for a block-addressed texture): the exact path
                    {
                        for (uint32_t y = 0; y < ch; ++y)
                            for (uint32_t x = 0; x < cw; ++x)
                                dst[static_cast<size_t>(by + y) * w + bx + x] = static_cast<uint8_t>(GSMem::PixelStorageTraits<GSMem::P4>::Read(GSMem::PageTableP4, vram, tbp, tbw, bx + x, by + y) & 15u);
                        continue;
                    }
                    const uint8_t *src = vram + ((idx >> 1) & (GSMem::MEMORY_SIZE - 1u)); // 256-aligned, so the block fits below MEMORY_SIZE
                    const uint16_t *o = ord.p4;
                    for (uint32_t y = 0; y < 16; ++y, d += pitch, o += 32)
                        for (uint32_t x = 0; x < 32; ++x)
                        {
                            const uint32_t n = o[x];
                            d[x] = static_cast<uint8_t>((src[n >> 1] >> ((n & 1u) << 2)) & 15u);
                        }
                }
                else
                {
                    if ((idx & 255u) != 0u)
                    {
                        for (uint32_t y = 0; y < ch; ++y)
                            for (uint32_t x = 0; x < cw; ++x)
                                dst[static_cast<size_t>(by + y) * w + bx + x] = static_cast<uint8_t>(GSMem::PixelStorageTraits<GSMem::P8>::Read(GSMem::PageTableP8, vram, tbp, tbw, bx + x, by + y) & 255u);
                        continue;
                    }
                    const uint8_t *src = vram + (idx & (GSMem::MEMORY_SIZE - 1u)); // 256-aligned
                    const uint16_t *o = ord.p8;
                    for (uint32_t y = 0; y < 16; ++y, d += pitch, o += 16)
                        for (uint32_t x = 0; x < 16; ++x)
                            d[x] = src[o[x]];
                }
                if (!whole)
                    for (uint32_t y = 0; y < ch; ++y)
                        std::memcpy(dst + static_cast<size_t>(by + y) * w + bx, tmp + y * bw, cw);
            }
        }
    }
}

void GSCpuBackend::noteDrawWrites(const GSContext &ctx)
{
    const uint32_t fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    // The CLUT stomp test (resolveDraw's, for the band-skipped draws that never reach it): a draw
    // this thread does not rasterize still overwrites, on other threads' rows, the page a cached
    // palette came from.
    if (m_draw.clutKey != ~0ull && !m_draw.clutStomped && gs2DrawCoversPage(ctx, fbp, zbp, m_draw.clutPage))
    {
        m_draw.clutStomped = true;
        if (s_gsClutVerify || s_gsTexCensus)
            ++g_clutVerify.stompsArmed;
    }
    if (!s_gsTexCache || s_gsTexCacheMutate == 2)
        return;
    // cont.321b: the cheap rejection -- the draw's frame/Z page intervals (gs2DrawCoversPage's own
    // superset) against the union of the live slots' pages -- now inside texStompSlots (cont.324b),
    // which resolveDraw shares.
    texStompSlots(ctx, fbp, zbp);
}

// ★★ cont.329j PS2X_GS_T4HLPROBE (default OFF, CPU path): what does the ~200,000-draw
// page-0x100 PSMT4HL composite actually SAMPLE? That read is PCSX2's "channel shuffle" -- a 4-bit
// index taken from the ALPHA byte of a render target, through the CLUT -- and the GL renderer
// cannot reproduce it. Before building it, measure whether the thing it samples VARIES: if the
// decoded texture is one colour, the whole feature is a uniform tint and cannot be the source of
// the remaining per-pixel colour differences on lit surfaces.
// ★★★ cont.329j PS2X_GS_SKIPZTEX (default OFF, CPU path, DIAGNOSTIC ABLATION): skip any draw
// whose TEXTURE is the Z BUFFER (tex page == ZBUF.zbp). This game samples page 0x100 -- which the
// layout census shows IS the Z buffer -- as PSMT4HL, a 4-bit index from the spare byte of PSMZ24
// through the CLUT, on ~200,000 draws. The GL renderer keeps depth in a GL renderbuffer and cannot
// express that read at all, so this ablation removes the same thing from the REFERENCE: if the
// CPU frame then looks like the GL frame, that read is the remaining colour-variety gap.
// ★★★ cont.329j PS2X_GS_SKIPTGT=<page> (default 0 = OFF, CPU path, DIAGNOSTIC ABLATION): skip
// every draw whose FRAME.fbp is that page. The GL renderer never writes m_vram, so anything the
// game RENDERS and then SAMPLES as a texture is stale there by construction. Target 0x180 takes
// 9.4 M draws per run and most of this game's texture sources sit inside its page range, so this
// removes the same content from the REFERENCE: if the CPU frame then moves toward the GL frame,
// render-to-texture is the remaining colour-variety gap.
static const uint32_t s_skipTgtPage = []
{ const char *e = std::getenv("PS2X_GS_SKIPTGT"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
// ★★★ cont.329k PS2X_GS_SKIPTEXPAGE=<page> [+ PS2X_GS_SKIPTEXPAGES=<count>, default 1] (CPU path,
// DIAGNOSTIC ABLATION): skip every draw whose TEXTURE lies in that page range. cont.329j showed
// that ablating the draws INTO target 0x180 costs the reference 7,239 distinct colours -- this
// separates the two ways that content can reach the frame: the ~4,000 CT24 composites that sample
// page 0x180 itself, or the millions of ordinary texture reads from pages INSIDE its range (which
// would mean the game renders its textures there, and the GL path -- which never writes m_vram --
// samples stale bytes for every one of them).
// ★★★ cont.330e PS2X_GS_SKIPTEXTBP=<tbp> (default 0 = OFF, CPU path, DIAGNOSTIC ABLATION):
// skip every draw whose texture TBP0 is EXACTLY this. PS2X_GS_SKIPTEXPAGE is page-granular, and
// the two draws that decide the FMV screen live in the SAME page (0x180): the four 128x512 strips
// at tbp=12291 and the 512x512 composite at tbp=12288. Only an exact-TBP ablation can tell them
// apart -- run it on the REFERENCE (the CPU renderer, which draws the screen correctly) and see
// which one carries the picture.
static const uint32_t s_skipTexTbp = []
{ const char *e = std::getenv("PS2X_GS_SKIPTEXTBP"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static const uint32_t s_skipTexPage = []
{ const char *e = std::getenv("PS2X_GS_SKIPTEXPAGE"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 0u; }();
static const uint32_t s_skipTexPages = []
{ const char *e = std::getenv("PS2X_GS_SKIPTEXPAGES"); return (e && e[0]) ? uint32_t(std::strtoul(e, nullptr, 0)) : 1u; }();
static const bool s_skipZTex = []
{ const char *e = std::getenv("PS2X_GS_SKIPZTEX"); return e && e[0] && e[0] != '0'; }();
static const bool s_t4hlProbe = []
{ const char *e = std::getenv("PS2X_GS_T4HLPROBE"); return e && e[0] && e[0] != '0'; }();

void GSCpuBackend::texCacheResolve(const GSDrawState &state)
{
    if (s_t4hlProbe && state.prim.tme && (m_draw.texTbp >> 5) >= 0x100u &&
        (state.context.tex0.psm == GS_PSM_T4HL || state.context.tex0.psm == GS_PSM_T4HH ||
         state.context.tex0.psm == GS_PSM_T8H))
    {
        static unsigned long long n = 0;
        if ((n++ % 4096ull) == 0ull)
        {
            // Sample a coarse grid of the decoded texture and count distinct results.
            unsigned seenIdx = 0, seenCol = 0;
            bool idxSeen[256] = {}, colSeen[256] = {};
            uint32_t cols[256]; unsigned nc = 0;
            const int w = m_draw.texW, h = m_draw.texH;
            for (int y = 0; y < h; y += 8)
                for (int x = 0; x < w; x += 8)
                {
                    const uint32_t raw = ReadVramUnlocked(state.context.tex0.psm, m_draw.texTbp,
                                                          m_draw.texTbw, u32(x), u32(y));
                    const uint32_t idx = raw & m_draw.clutMask;
                    if (idx < 256u && !idxSeen[idx]) { idxSeen[idx] = true; ++seenIdx; }
                    const uint32_t c = m_draw.clut[idx];
                    bool have = false;
                    for (unsigned k = 0; k < nc; ++k) if (cols[k] == c) { have = true; break; }
                    if (!have && nc < 256u) { cols[nc++] = c; ++seenCol; }
                }
            (void)colSeen;
            std::fprintf(stderr,
                         "[gs2:t4hl] draw %llu tbp=%u (page 0x%x) psm=0x%02x %dx%d -> distinct "
                         "INDICES=%u distinct COLOURS=%u (grid of %d samples) first=%08x\n",
                         n, m_draw.texTbp, m_draw.texTbp >> 5, state.context.tex0.psm, w, h,
                         seenIdx, seenCol, (w / 8) * (h / 8), nc ? cols[0] : 0u);
        }
    }
    // Close the previous key's tap count into its predictor entry (the same key continues below).
    if (m_draw.texPred != nullptr)
    {
        static_cast<TexPred *>(m_draw.texPred)->taps += m_draw.texTaps;
        if (m_draw.texLin) t_texStats.tapsLin += m_draw.texTaps; else t_texStats.tapsSwz += m_draw.texTaps;
    }
    if (t_texLivePending != nullptr) { t_texLivePending->taps += m_draw.texTaps; t_texLivePending = nullptr; }
    m_draw.texTaps = 0u;
    m_draw.texLin = nullptr;
    m_draw.texPred = nullptr;
    auto liveNote = [&](uint8_t reason, uint32_t tbpv, uint32_t wv, uint32_t hv, uint8_t psmv)
    {
        const uint64_t k = (static_cast<uint64_t>(reason) << 56) ^ (static_cast<uint64_t>(tbpv) << 40) ^ (static_cast<uint64_t>(psmv) << 24) ^
                           (static_cast<uint64_t>(wv) << 12) ^ static_cast<uint64_t>(hv) ^ (static_cast<uint64_t>(m_draw.texWrapU) << 52) ^ (static_cast<uint64_t>(m_draw.texWrapV) << 54);
        TexLiveEntry &e = t_texLive[k];
        ++e.draws; e.tbp = tbpv; e.w = wv; e.h = hv; e.psm = psmv; e.reason = reason; e.wrapU = static_cast<uint8_t>(m_draw.texWrapU); e.wrapV = static_cast<uint8_t>(m_draw.texWrapV);
        t_texLivePending = &e;
    };
    if (!s_gsTexCache) return;
    texCacheFlushStats(false);
    const bool p4 = m_draw.texFastP4, p8 = m_draw.texFastP8;
    const uint32_t w = static_cast<uint32_t>(m_draw.texW), h = static_cast<uint32_t>(m_draw.texH);
    const auto &ctx = state.context;
    const uint8_t psm = static_cast<uint8_t>(ctx.tex0.psm & 0x3Fu);
    if (!(p4 || p8) || m_draw.texWrapClass == 2 || w * h > kTexMaxTexels || w == 0u || h == 0u)
    {
        ++t_texStats.drawsInel;
        liveNote(!(p4 || p8) ? 1 : m_draw.texWrapClass == 2 ? 2 : 3, m_draw.texTbp, w, h, psm);
        return;
    }
    const uint32_t tbp = m_draw.texTbp, tbw = std::max<uint32_t>(m_draw.texTbw, 1u);
    const uint64_t key = (static_cast<uint64_t>(tbp) << 40) ^ (static_cast<uint64_t>(tbw) << 32) ^
                         (static_cast<uint64_t>(psm) << 24) ^ (static_cast<uint64_t>(w) << 12) ^ static_cast<uint64_t>(h);
    // the level's pages and their current upload generations
    Gs2PageMask pm;
    gs2RectPages(pm, tbp / 32u, tbw, psm, 0u, 0u, w - 1u, h - 1u);
    uint32_t pages[kTexMaxPages], gens[kTexMaxPages], n = 0u;
    for (int wi = 0; wi < 8; ++wi)
    {
        uint64_t b = pm.w[wi];
        while (b)
        {
            const int i = __builtin_ctzll(b); b &= b - 1;
            if (n == kTexMaxPages) { ++t_texStats.drawsInel; liveNote(4, tbp, w, h, psm); return; }
            pages[n] = static_cast<uint32_t>(wi * 64 + i);
            gens[n] = g_gs2PageGen[pages[n]].load(std::memory_order_relaxed);
            ++n;
        }
    }
    uint64_t sig = 0xCBF29CE484222325ull;
    for (uint32_t i = 0; i < n; ++i) { sig ^= (static_cast<uint64_t>(gens[i]) << 16) | pages[i]; sig *= 0x100000001B3ull; }
    // the predictor: a new generation of this key closes the previous one's reuse into the EMA
    TexPred &pred = t_texPreds[key];
    if (pred.sig != sig)
    {
        if (pred.gens != 0u && pred.texels != 0u)
        {
            const float reuse = static_cast<float>(pred.taps) / static_cast<float>(pred.texels);
            pred.ema = pred.ema < 0.0f ? reuse : 0.5f * (pred.ema + reuse);
        }
        pred.taps = 0u; pred.sig = sig; pred.texels = w * h; ++pred.gens;
    }
    m_draw.texPred = &pred;
    // the slot
    TexSlot *slot = nullptr, *lru = &t_texSlots[0];
    for (TexSlot &sl : t_texSlots)
    {
        if (sl.key == key) { slot = &sl; break; }
        if (sl.key == ~0ull) { lru = &sl; }
        else if (lru->key != ~0ull && sl.lastUse < lru->lastUse) lru = &sl;
    }
    if (slot != nullptr)
    {
        bool valid = !slot->stomped;
        if (valid && s_gsTexCacheMutate != 1)
            for (uint32_t i = 0; i < n; ++i) if (slot->gens[i] != gens[i]) { valid = false; break; }
        if (!valid)
        {
            if (slot->stomped) ++t_texStats.invStomp; else ++t_texStats.invGen;
            const bool wasStomped = slot->stomped;
            slot->key = ~0ull; slot->stomped = false; t_texUnionDirty = true;
            if (wasStomped) { ++t_texStats.drawsSwz; liveNote(6, tbp, w, h, psm); return; } // served live this draw; the next one refills
            slot = nullptr;
        }
    }
    if (slot == nullptr)
    {
        if (s_gsTexCacheThr > 0.0f && pred.ema < s_gsTexCacheThr) { ++t_texStats.drawsDecl; liveNote(7, tbp, w, h, psm); return; }
        // a draw that writes the pages it samples is served live (the texture unit reads VRAM)
        const uint32_t fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp), zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
        for (uint32_t i = 0; i < n; ++i)
            if (gs2DrawCoversPage(ctx, fbp, zbp, pages[i])) { ++t_texStats.drawsSwz; liveNote(5, tbp, w, h, psm); return; }
        slot = lru;
        slot->key = key; slot->nPages = n; slot->w = w; slot->h = h; slot->stomped = false; t_texUnionDirty = true;
        for (uint32_t i = 0; i < n; ++i) { slot->pages[i] = pages[i]; slot->gens[i] = gens[i]; }
        slot->idx.resize(static_cast<size_t>(w) * h);
        uint8_t *dst = slot->idx.data();
        if (s_gsTexFillBlock)
            gs2TexFillBlocks(dst, w, h, tbp, tbw, p4, m_vram);
        else if (p4)
            for (uint32_t v = 0; v < h; ++v)
                for (uint32_t u = 0; u < w; ++u)
                    dst[v * w + u] = static_cast<uint8_t>(GSMem::PixelStorageTraits<GSMem::P4>::Read(GSMem::PageTableP4, m_vram, tbp, tbw, u, v) & 15u);
        else
            for (uint32_t v = 0; v < h; ++v)
                for (uint32_t u = 0; u < w; ++u)
                    dst[v * w + u] = static_cast<uint8_t>(GSMem::PixelStorageTraits<GSMem::P8>::Read(GSMem::PageTableP8, m_vram, tbp, tbw, u, v) & 255u);
        ++t_texStats.fills; t_texStats.fillTexels += w * h; ++t_texFillSeq;
    }
    slot->lastUse = ++t_texClock;
    m_draw.texLin = slot->idx.data();
    m_draw.texLinW = w;
    ++t_texStats.drawsLin;
}

// Replay the capture off-line and report. Deterministic: identical work every run.
void GSCpuBackend::rasterBenchRun()
{
    std::FILE *f = std::fopen(s_rasterBench, "rb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "[gs2:rasterbench] cannot open %s\n", s_rasterBench);
        return;
    }
    RasterCapHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1 || std::memcmp(hdr.magic, "GSRC", 4) != 0)
    {
        std::fprintf(stderr, "[gs2:rasterbench] bad header in %s\n", s_rasterBench);
        std::fclose(f);
        return;
    }
    std::vector<uint8_t> vram0(hdr.vramSize);
    std::vector<GSPrimitiveBatch> batches(hdr.batchCount);
    if (std::fread(vram0.data(), 1, hdr.vramSize, f) != hdr.vramSize ||
        std::fread(batches.data(), sizeof(GSPrimitiveBatch), hdr.batchCount, f) != hdr.batchCount)
    {
        std::fprintf(stderr, "[gs2:rasterbench] short read\n");
        std::fclose(f);
        return;
    }
    std::fclose(f);

    if (s_rasterBenchLimit != 0u && s_rasterBenchLimit < batches.size())
        batches.resize(s_rasterBenchLimit);
    // PS2X_GS_RASTERBENCH_DUMP=<1-based index>: describe one primitive, for diagnosing a hash
    // divergence bisected down to a single draw.
    if (const char *d = std::getenv("PS2X_GS_RASTERBENCH_DUMP"))
    {
        const size_t idx = static_cast<size_t>(std::strtoul(d, nullptr, 10));
        for (size_t k = (idx > 3 ? idx - 3 : 1); k <= idx + 1 && k <= batches.size(); ++k)
        {
            const GSPrimitiveBatch &b = batches[k - 1];
            const auto &c = b.state.context;
            float ylo = b.vertices[0].y, yhi = b.vertices[0].y;
            for (uint32_t i = 1; i < b.vertexCount && i < 3u; ++i)
            {
                if (b.vertices[i].y < ylo) ylo = b.vertices[i].y;
                if (b.vertices[i].y > yhi) yhi = b.vertices[i].y;
            }
            std::fprintf(stderr,
                         "[gs2:dump] #%zu prim=%u n=%u tme=%u abe=%u | frame{fbp=%u fbw=%u psm=0x%x msk=%x}"
                         " zbuf{zbp=%u psm=0x%x} scis{%u-%u,%u-%u} ofy=%d yRange=%.1f..%.1f"
                         " tex0{tbp0=%u tbw=%u psm=0x%x cbp=%u tw=%u th=%u} test=%llx alpha=%llx\n",
                         k, static_cast<unsigned>(b.state.prim.type), b.vertexCount,
                         b.state.prim.tme ? 1u : 0u, b.state.prim.abe ? 1u : 0u,
                         c.frame.fbp, c.frame.fbw, c.frame.psm, c.frame.fbmsk,
                         c.zbuf.zbp, c.zbuf.psm,
                         c.scissor.x0, c.scissor.x1, c.scissor.y0, c.scissor.y1,
                         c.xyoffset.ofy >> 4, ylo, yhi,
                         c.tex0.tbp0, c.tex0.tbw, c.tex0.psm, c.tex0.cbp, c.tex0.tw, c.tex0.th,
                         static_cast<unsigned long long>(c.test),
                         static_cast<unsigned long long>(c.alpha));
        }
    }
    std::vector<uint8_t> vram(hdr.vramSize);
    double best = 1e30;
    uint64_t hash = 0;
#if PS2X_HAS_GS_GPU_DEVICE
    // ★ cont.232 PS2X_GS_RASTERBENCH_GPU=1: the same capture through the GPU tile rasterizer,
    // submitted ASYNCHRONOUSLY. The only GPU figure on record (cont.174: 14.4 us/prim) had a
    // glFinish per batch; this is the number the authority flip would actually get. Per rep: one
    // 4 MB patch seeds the mirror with the capture's VRAM; every primitive goes through the batch
    // accumulator exactly as the flip would run it (host-side state dedup + 16x16 binning, flushed
    // at 4096 prims / 128 states / a target change / a RAW hazard; NO verify, NO per-batch finish);
    // then one synchronous readback of the whole mirror ends the clock and yields the hash to
    // compare with the CPU replay's. "submit" = the host side alone (state build + binning: what a
    // flipped authority still pays on the GS worker); "total" = submit + the GPU + the readback.
    if (s_rasterBenchGpu)
    {
        GsGpuPresentDevice *dev = gs2GpuDevice();
        if (dev == nullptr)
        {
            std::fprintf(stderr, "[gs2:rasterbench-gpu] no GPU device\n");
            return;
        }
        std::memcpy(vram.data(), vram0.data(), hdr.vramSize);
        m_vram = vram.data(); // the conversion's range checks / CLUT shadow read the CPU copy
        m_vramSize = hdr.vramSize;
        std::vector<uint8_t> gpuOut;
        double bestSubmit = 1e30, bestTotal = 1e30, bestGpu = 1e30;
        for (unsigned rep = 0; rep < s_rasterBenchReps; ++rep)
        {
            const unsigned long rejBefore = gs2RvRejectedTotal();
            unsigned long long gpuNs0 = 0, gpuPrims0 = 0;
            unsigned long gpuDisp0 = 0;
            dev->PerfStats(gpuNs0, gpuPrims0, gpuDisp0); // PS2X_GS_GPU_BATCHPERF=1: per-batch finish-timed GPU ns
            dev->MirrorPatchRaw(0u, std::vector<uint8_t>(vram0)); // FIFO: lands before the batches
            const auto t0 = std::chrono::steady_clock::now();
            for (const GSPrimitiveBatch &b : batches)
                if (b.vertexCount != 0u)
                {
                    // The conversion reads the RESOLVED target (m_draw.fbp/fbw/zbp), which the live
                    // path sets in DrawPrimitive/RasterBand before every primitive. (Without this every
                    // primitive carried one stale target: 0 target flushes, a wrong image, aliasing.)
                    resolveDraw(b.state);
                    gpuBatchAppend(b);
                }
            gpuBatchFlush();
            const auto t1 = std::chrono::steady_clock::now();
            dev->MirrorReadback(0u, hdr.vramSize, gpuOut); // waits for every queued job
            const auto t2 = std::chrono::steady_clock::now();
            const double submitMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double totalMs = std::chrono::duration<double, std::milli>(t2 - t0).count();
            unsigned long long gpuNs1 = 0, gpuPrims1 = 0;
            unsigned long gpuDisp1 = 0;
            dev->PerfStats(gpuNs1, gpuPrims1, gpuDisp1);
            const double gpuMs = double(gpuNs1 - gpuNs0) * 1e-6;
            // Rep 0 pays the shader compile (seconds); the BEST figures exclude it when there is
            // more than one rep.
            if (rep > 0u || s_rasterBenchReps == 1u)
            {
                bestSubmit = std::min(bestSubmit, submitMs);
                bestTotal = std::min(bestTotal, totalMs);
                bestGpu = std::min(bestGpu, gpuMs);
            }
            hash = gpuOut.size() == hdr.vramSize ? vramHash(gpuOut.data(), hdr.vramSize) : 0ull;
            const unsigned long rejected = gs2RvRejectedTotal() - rejBefore;
            std::fprintf(stderr,
                         "[gs2:rasterbench-gpu] rep %u: submit %.2f ms  total %.2f ms  gpu %.2f ms  vramHash=%016llx"
                         " rejected=%lu\n",
                         rep, submitMs, totalMs, gpuMs, static_cast<unsigned long long>(hash), rejected);
        }
        std::fprintf(stderr,
                     "[gs2:rasterbench-gpu] BEST submit %.2f ms, total %.2f ms, gpu %.2f ms for %u prims = %.0f ns/prim"
                     " (total) | vramHash=%016llx\n",
                     bestSubmit, bestTotal, bestGpu, static_cast<uint32_t>(batches.size()),
                     bestTotal * 1e6 / double(batches.size() ? batches.size() : 1),
                     static_cast<unsigned long long>(hash));
        if (s_gsHwCensus)
            std::fprintf(stderr, "[gsgpu:hwcensus] reps=%u state-runs=%lu layers{sum=%lu max=%lu} (totals over all reps)\n",
                         s_rasterBenchReps, g_bvStateRuns, g_bvLayerSum, g_bvLayerMax);
        gs2PrintMirrorPatch(); // the [gsgpu:batch] flush breakdown + rejection reasons
        if (const char *pp = std::getenv("PS2X_GS_RASTERBENCH_PPM"))
            if (gpuOut.size() == hdr.vramSize)
                for (uint32_t fbpPage : {0u, 128u, 384u})
                {
                    char path[600];
                    std::snprintf(path, sizeof(path), "%s_gpu_fbp%u.ppm", pp, fbpPage);
                    std::FILE *g = std::fopen(path, "wb");
                    if (g == nullptr)
                        continue;
                    const uint32_t blk = GSInternal::framePageBaseToBlock(fbpPage);
                    std::fprintf(g, "P6\n512 512\n255\n");
                    for (uint32_t yy = 0; yy < 512u; ++yy)
                        for (uint32_t xx = 0; xx < 512u; ++xx)
                        {
                            const uint32_t px = GSMem::ReadCT32(gpuOut.data(), blk, 8u, xx, yy);
                            std::fputc(px & 0xFFu, g);
                            std::fputc((px >> 8) & 0xFFu, g);
                            std::fputc((px >> 16) & 0xFFu, g);
                        }
                    std::fclose(g);
                    std::fprintf(stderr, "[gs2:rasterbench-gpu] wrote %s\n", path);
                }
        return;
    }
#endif
    for (unsigned rep = 0; rep < s_rasterBenchReps; ++rep)
    {
        std::memcpy(vram.data(), vram0.data(), hdr.vramSize);
        g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.227b: VRAM restored
        m_vram = vram.data();
        m_vramSize = hdr.vramSize;
        const auto t0 = std::chrono::steady_clock::now();
        if (!m_bandThreads.empty())
        {
            // cont.209: one fan-out for the WHOLE run.
            RasterRunFanOut(batches.data(), batches.size());
        }
        else
        {
            for (const GSPrimitiveBatch &b : batches)
                if (b.vertexCount != 0u)
                    DrawPrimitive(b);
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        if (ms < best)
            best = ms;
        hash = vramHash(vram.data(), hdr.vramSize);
        std::fprintf(stderr, "[gs2:rasterbench] rep %u: %.2f ms  vramHash=%016llx\n",
                     rep, ms, static_cast<unsigned long long>(hash));
    }
    // ★★ cont.230: PS2X_GS_RASTERBENCH_PPM=<prefix> decodes the replayed framebuffers to images.
    // The capture replays deterministically, so this is the artifact frozen in a file -- which is
    // what makes a bisect possible at all. Uses the engine's own verified CT32 addressing rather
    // than re-deriving the swizzle.
    if (const char *pp = std::getenv("PS2X_GS_RASTERBENCH_PPM"))
    {
        for (uint32_t fbpPage : {0u, 128u, 384u})
        {
            char path[600];
            std::snprintf(path, sizeof(path), "%s_fbp%u.ppm", pp, fbpPage);
            std::FILE *g = std::fopen(path, "wb");
            if (g == nullptr)
                continue;
            const uint32_t blk = GSInternal::framePageBaseToBlock(fbpPage);
            std::fprintf(g, "P6\n512 512\n255\n");
            for (uint32_t yy = 0; yy < 512u; ++yy)
                for (uint32_t xx = 0; xx < 512u; ++xx)
                {
                    const uint32_t px = GSMem::ReadCT32(vram.data(), blk, 8u, xx, yy);
                    std::fputc(px & 0xFFu, g);
                    std::fputc((px >> 8) & 0xFFu, g);
                    std::fputc((px >> 16) & 0xFFu, g);
                }
            std::fclose(g);
            std::fprintf(stderr, "[gs2:rasterbench] wrote %s\n", path);
        }
    }
    if (const char *vo = std::getenv("PS2X_GS_RASTERBENCH_VRAMOUT"))
    {
        std::FILE *g = std::fopen(vo, "wb");
        if (g != nullptr) { std::fwrite(vram.data(), 1, vram.size(), g); std::fclose(g); }
    }
    std::fprintf(stderr,
                 "[gs2:rasterbench] BEST %.2f ms for %u prims = %.0f ns/prim"
                 " | threads=%s minpix=%s runs=%llu | vramHash=%016llx\n",
                 best, static_cast<uint32_t>(batches.size()),
                 best * 1e6 / double(batches.size() ? batches.size() : 1),
                 std::to_string(m_bandCount).c_str(),
                 std::getenv("PS2X_GS_BAND_MINPIX") ? std::getenv("PS2X_GS_BAND_MINPIX") : "default",
                 m_bandRuns, static_cast<unsigned long long>(hash));
    gs2TexCachePrint();
    gs2TexLiveCensusPrint("bench");
    gs2TexCensusDump("bench");
    gs2ClutVerifyDump("bench");
    gs2TexVerifyDump("bench");
    gs2TriCensusDump("bench");
    gs2SpanVerifyDump("bench");
    gs2GeoCensusDump("bench");
}


GSCpuBackend::~GSCpuBackend()
{
    GSCpuBackend *self = this;
    g_gs2ActiveBackend.compare_exchange_strong(self, nullptr, std::memory_order_release,
                                               std::memory_order_relaxed);
    if (m_worker.joinable())
    {
        {
            std::lock_guard<std::mutex> lk(m_queueMutex);
            m_workerStop = true;
        }
        m_queueCv.notify_all();
        m_worker.join();
    }
    stopBandPool();
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    if (s_gsThread)
    {
        m_worker = std::thread(&GSCpuBackend::workerLoop, this);
        startBandPool();
    }
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            // Undefined PSM codes decode as PSMCT32 on real GS hardware -- PCSX2
            // GS/GSLocalMemory.cpp fills every undefined m_psm[64] slot with the PSMCT32
            // layout. A Null fallback makes any texture declared with an undefined psm read
            // as 0 (fully transparent): LOTR ROTK's movie sprites carry a literal TEX0
            // psm=0x20 (hardcoded immediate @0x1fa3d4), so with ReadNull the whole FMV is
            // invisible. (Same fix as the pre-refactor ReadVram/WriteVram patch, cont.116e.)
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        }
    }
    Reset();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.227b: invalidate cached CLUTs
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
    // Display-flip snapshot routing: FIRST-initialized backend wins. The presentation path
    // Initializes a thread_local scratch backend over the VRAM snapshot on every present --
    // that one must never capture flips (its "VRAM" is the copy itself).
    GSCpuBackend *expected = nullptr;
    g_gs2ActiveBackend.compare_exchange_strong(expected, this, std::memory_order_release,
                                               std::memory_order_relaxed);
    // cont.208: replay a capture instead of running the game. Deterministic and fast, so raster
    // changes can be measured (and pixel-compared) in seconds rather than 7-minute noisy runs.
    if (s_rasterBench != nullptr && s_rasterBench[0] != '\0' && expected == nullptr)
    {
        rasterBenchRun();
        std::fflush(nullptr);
        _exit(0);
    }
}

void GSCpuBackend::Reset()
{
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.227b: invalidate cached CLUTs
    // cont.320: only the game's raster backend counts -- the presenter Initializes a thread_local
    // scratch backend over its VRAM snapshot on EVERY present, and that reset must not touch the
    // census (build 757 measured per-FRAME reuse because of it: reset=768 vs presents=772).
    {
        GSCpuBackend *active = g_gs2ActiveBackend.load(std::memory_order_acquire);
        if (active == nullptr || active == this)
            gs2TexCensusBumpAll(g_texCensus.bumpAllReset);
    }
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    m_clutShadow.clear();
}

void GSCpuBackend::flushPendingDraws() const
{
    if (m_pendingRun.empty())
        return;
    WorkItem item;
    item.kind = WorkItem::Kind::DrawRun;
    item.run.swap(m_pendingRun);
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        if (!m_runPool.empty())
        {
            m_pendingRun.swap(m_runPool.back());
            m_runPool.pop_back();
        }
    }
    m_pendingRun.clear();
    if (m_pendingRun.capacity() < s_gsRunMax)
        m_pendingRun.reserve(s_gsRunMax);
    ++g_drawRunItems;
    g_drawRunBatches += item.run.size();
    const_cast<GSCpuBackend *>(this)->enqueueWork(std::move(item)); // the queue is mutable state behind const read-side entries
}

void GSCpuBackend::LoadClut(const GSTex0Reg &, const GSTexClutReg &)
{
}

void GSCpuBackend::EndPacket()
{
    if (s_gsThread && s_gsDrawRun && s_gsRunFlushPacket)
        flushPendingDraws();
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (s_gsThread)
    {
        if (batch.vertexCount == 0u)
            return;
        if (s_gsDrawRun)
        {
            m_pendingRun.push_back(batch);
            if (m_pendingRun.size() >= s_gsRunMax)
                flushPendingDraws();
            return;
        }
        WorkItem item;
        item.kind = WorkItem::Kind::Draw;
        item.batch = batch;
        enqueueWork(std::move(item));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    DrawPrimitive(batch);
}

void GSCpuBackend::Flush()
{
    // Submission-only (cont.163): for the strict-FIFO worker, submission already happened
    // at enqueue time, and EVERY frontend Flush() is immediately followed by a
    // Sync(reason) that carries the completion semantics (Finish/Reset drain;
    // Presentation deliberately does not). Draining here made the PRESENT path drain too
    // (gdb: main thread parked in drainQueue under latchHostPresentationFrame's Flush,
    // frontend:548) — during continuously-producing eras (boot/menu, no vsync gaps) the
    // queue never empties, the main thread froze for minutes, the window went
    // "not responding", input died, and mutter eventually killed the client.
}

void GSCpuBackend::TextureFlush()
{
    // CPU texture reads are coherent with local memory, and the worker queue is strict
    // FIFO (an upload queued before a draw executes before it), so no drain is needed.
}

void GSCpuBackend::Sync(GSSyncReason reason)
{
    // Presentation must NOT drain (cont.163): with the EE producing draws continuously,
    // a full drain can starve the MAIN thread for long stretches — and the main thread is
    // the only one pumping X events, so input (and the present cadence itself) dies
    // (gdb: main thread parked on the drain condvar at the menu; zero _glfwInputKey hits
    // across a verified 30s key hold). Presenting without a drain is also the
    // hardware-faithful reading: real GS scanout reads local memory AS IT IS at that
    // instant. Completion-semantics reasons (Finish before the CSR latch, Reset,
    // DebugReadback) still drain.
    if (reason == GSSyncReason::Presentation)
        return;
    // ★★★ cont.317 PS2X_GS_FINISH_DRAIN (default 0): a FINISH no longer drains the raster queue.
    // The [ee:prof]/[sched:pace] split of the EE thread at 7 workers (build 735, 230 s fight) was
    // 17.5% vsync sleep, ~19% inside the GIF submit, and the raster workers 30% idle -- the two
    // sides each ~45 ms of a 66 ms frame with a serialisation point between them: every frame's
    // FINISH came through here and waited for the WHOLE queued backlog before the EE could go on
    // (the drain existed for the per-Sync VRAM capture below, which is skipped anyway while flip
    // snapshots are live). PCSX2 (Gif.cpp gsHandler, case FINISH): CSR.FINISH is latched when the
    // EE side PARSES the packet -- the MTGS render thread is not waited for -- so the game's
    // FINISH wait means "submitted", not "rendered". Same model here: Finish drains only when
    // the capture below will actually run (no live flips) or when `=1` restores the old drain.
    // VRAM ordering is unaffected: every EE-side VRAM effect is a queued item (strict FIFO) or a
    // LocalToHost/DebugReadback/Reset sync, which still drain. The drain wait is counted per
    // reason and printed with [gsgpu:items] (drainMs=...).
    static const int s_gsFinishDrain = []
    { const char *e = std::getenv("PS2X_GS_FINISH_DRAIN"); return (e && e[0]) ? std::atoi(e) : 0; }();
    const bool flipsLive = s_gsFlipSnap && gs2NowMs() - g_gs2LastFlipSnapMs.load(std::memory_order_acquire) < 400u;
    const bool captureWillRun = !flipsLive;
    if (reason != GSSyncReason::Finish || s_gsFinishDrain != 0 || captureWillRun)
    {
        const auto td0 = std::chrono::steady_clock::now();
        drainQueue();
        const unsigned long long ns = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - td0).count());
        const int r = static_cast<int>(reason) & 7;
        g_gs2DrainNs[r].fetch_add(ns, std::memory_order_relaxed);
        g_gs2DrainCount[r].fetch_add(1ull, std::memory_order_relaxed);
    }
    else
    {
        g_gs2DrainSkipped.fetch_add(1ull, std::memory_order_relaxed);
    }
    // Frame-boundary presentation snapshot (cont.164): with the no-drain present, live
    // VRAM can LAG the display registers by the queued backlog, so presents showed stale
    // texture-arena frames OVER the menu (user-observed). A drained Sync is the
    // hardware-meaningful "caught up" moment — the queue is empty, VRAM reflects
    // everything the guest submitted before this Sync (FINISH fires per frame chain).
    // Copy VRAM here; Present() prefers this frame-complete copy while fresh and falls
    // back to live VRAM when an era doesn't Sync.
    // cont.165: while display-flip snapshots are live (a flip in the last 400ms), ordinary
    // Syncs must NOT capture -- a mid-frame capture would overwrite the frame-complete flip
    // snapshot with a post-clear/mid-scene state (that was the menu-text eraser). When an era
    // stops flipping for >400ms, this per-Sync capture resumes as the fallback.
    if (s_gsFlipSnap && gs2NowMs() - g_gs2LastFlipSnapMs.load(std::memory_order_acquire) < 400u)
        return;
    if (s_gsThread && m_vram && m_vramSize != 0u)
    {
        uint64_t seqNow = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::lock_guard<std::mutex> plock(m_presentCopyMutex);
            m_presentCopy.assign(m_vram, m_vram + m_vramSize);
            seqNow = ++m_presentCopySeq;
            m_presentCopyHasRegs = false; // a Sync capture pairs with the LIVE display registers
            m_presentCopyTickMs.store(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count()),
                std::memory_order_release);
        }
#if PS2X_HAS_GS_GPU_DEVICE
        // cont.345: the GL snapshot of the same instant (the queue is drained here, so the device
        // resolve lands behind everything submitted so far), paired with the LIVE registers.
        // Outside m_mutex: the device enqueue can wait on backpressure.
        glSnapshotDisplayTargets(g_ps2xGsLiveDispfb1.load(std::memory_order_acquire),
                                 g_ps2xGsLiveDispfb2.load(std::memory_order_acquire), seqNow);
#endif
    }
}

void GSCpuBackend::OnDisplayFlip(uint64_t preFlipDispfb1, uint64_t preFlipDispfb2)
{
    // ★ cont.331i: remember the buffer that was on screen for the window just ending -- the one
    // the post-flip glyph draws land in (see s_glFlipResolve).
    if (s_glFlipResolve)
        g_glPreFlipFbp.store(static_cast<uint32_t>(preFlipDispfb1 & 0x1FFu), std::memory_order_relaxed);
    // The guest just flipped the display. The buffer the PRE-flip registers point to holds the
    // last fully completed frame (its UI text was drawn onto it right after ITS flip -- proven
    // cont.165: 128 glyph draws land in every inter-flip window, always AFTER the flip). Capture
    // VRAM together with those pre-flip registers; Present() renders the pair. No drain: the
    // pre-flip buffer's work executed a whole frame ago, and any concurrently-executing worker
    // items only target the new back buffer.
    if (!m_vram || m_vramSize == 0u)
        return;
    g_perfFlips.fetch_add(1, std::memory_order_relaxed);
    // cont.358e: cluster THIS flip's classified draws and report the blocks. Per flip, not per
    // window -- a window would smear a moving element across its whole path.
    if (s_hudCluster && probeFlipOk())
        dumpHudClusters(g_perfFlips.load(std::memory_order_relaxed));
    // cont.358f: cluster this flip's classified draws and publish them for the NEXT flip's lookups.
    // Only while a game actually has an anchor policy installed -- otherwise nothing accumulates.
    if (g_hudAnchorFn.load(std::memory_order_acquire) != nullptr)
    {
        std::vector<HudClusterDraw> cur;
        {
            std::lock_guard<std::mutex> lk(g_hudMapMx);
            cur.swap(g_hudPending);
        }
        // ★★★★★ cont.358g: NEVER PUBLISH AN EMPTY MAP OVER A GOOD ONE. The classified HUD draws
        // do not arrive on every flip -- the census shows them alternating (flip 9000 none, 9001
        // fifty) -- so rebuilding unconditionally replaced a valid map with an empty one every
        // other flip, every draw then found no cluster, declined, and the HUD snapped back to
        // centre for that frame. That is the one-frame-on/one-frame-off FLICKER the user reported.
        // A flip that classified nothing carries no information about the layout, so it must leave
        // the previous map standing rather than erase it.
        // ★★★★★ cont.358h: the gate for the NEXT frame -- was the frame just ended fully described
        // by the map? One uncontained draw means the layout moved out from under it, so the next
        // frame declines in full rather than letting each draw improvise (which is what tore).
        const unsigned long long seen = g_hudFrameSeen.exchange(0, std::memory_order_relaxed);
        const unsigned long long miss = g_hudFrameMiss.exchange(0, std::memory_order_relaxed);
        if (seen > 0ull) ((miss == 0ull) ? g_hudGateOn : g_hudGateOff).fetch_add(1, std::memory_order_relaxed);
        g_hudGate.store(true, std::memory_order_relaxed);
        if (!cur.empty())
        {
            std::vector<HudClusterRect> built = hudBuildClusters(cur, s_hudClusterGap);
            std::lock_guard<std::mutex> lk(g_hudMapMx);
            g_hudMap.swap(built);
        }
    }
    // cont.331s: the renderer-agnostic wall clock (see s_gsClockFlips).
    if (s_gsThruput && s_gsClockFlips != 0ull)
    {
        const unsigned long long cf = g_perfFlips.load(std::memory_order_relaxed);
        if ((cf % s_gsClockFlips) == 0ull)
            std::fprintf(stderr, "[gsgpu:clock] t=%.3f s flips=%llu presents=%llu renderer=%s\n",
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - g_perfStart).count(),
                         cf, g_perfPresents.load(), s_gsRendererGl ? "gl" : "cpu");
    }
    // ★★★★★ cont.356 PS2X_GS_VRAMRECT=<dir>: decode a VRAM REGION to PPM, aimed by PS2X_GS_PROBEFLIP.
    // PS2X_GS_VRAMDUMP already dumps local memory, but it is pinned to a fixed DRAW count (420000,
    // or the era gate + 30000) and the screen under investigation draws ~37 primitives per flip --
    // the title screen comes and goes thousands of draws before either trigger. Aim it the way every
    // other probe here is aimed, and decode through the backend's OWN ReadVram so the GS swizzle is
    // the runtime's implementation rather than a second one written offline (which would make a
    // garbage result ambiguous between "wrong address" and "wrong swizzle").
    // PS2X_GS_VRAMRECT_TBP (default 12288) takes a comma list of TBP0 values; each is decoded as
    // PSMCT32 at bw 8/4/2/1, because the upload census shows this art arriving in bw=2 and bw=1
    // STRIPS, so the width it should be read at is exactly what is in question.
    if (m_vram)
    {
        static bool s_vramRectDone = false;
        const char *vrDir = std::getenv("PS2X_GS_VRAMRECT");
        if (vrDir && vrDir[0] && !s_vramRectDone && s_probeFlip != 0ull &&
            g_perfFlips.load(std::memory_order_relaxed) >= s_probeFlip)
        {
            s_vramRectDone = true;
            // ★ cont.356b: a SPEC list, because guessing the format cost four runs. Each entry is
            // tbp:psm:bw:w:h (psm and bw default to the GS meaning of the composite this was built
            // for). PS2X_GS_VRAMRECT_TBP is still honoured as a shorthand for the old bw sweep.
            //   PS2X_GS_VRAMRECT_SPEC=12288:1:8:512:512,9237:20:1:256:256
            // ⚠ Reads WRAP: ReadAt masks the byte address to VRAM size, so a read that runs past
            // 4 MB silently continues at block 0 -- which is FRAMEBUFFER 0 and, on a menu screen,
            // holds the very image being looked for. That produced a false confirmation here. Any
            // entry whose extent leaves VRAM is clamped and SAID SO rather than quietly aliased.
            const char *specList = std::getenv("PS2X_GS_VRAMRECT_SPEC");
            const char *tbpList = std::getenv("PS2X_GS_VRAMRECT_TBP");
            std::string spec;
            if (specList && specList[0])
                spec = specList;
            else
            {
                const std::string t = (tbpList && tbpList[0]) ? tbpList : "12288";
                size_t p0 = 0;
                while (p0 <= t.size())
                {
                    const size_t c = t.find(',', p0);
                    const std::string one = t.substr(p0, c == std::string::npos ? std::string::npos : c - p0);
                    p0 = (c == std::string::npos) ? t.size() + 1 : c + 1;
                    if (one.empty()) continue;
                    for (unsigned bw : {8u, 4u, 2u, 1u})
                    {
                        if (!spec.empty()) spec += ',';
                        spec += one + ":0:" + std::to_string(bw) + ":" +
                                std::to_string(bw * 64u) + ":512";
                    }
                }
            }
            size_t pos = 0;
            while (pos <= spec.size())
            {
                const size_t comma = spec.find(',', pos);
                const std::string one = spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                                                    : comma - pos);
                pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
                if (one.empty()) continue;
                uint32_t tbp = 0, psm = 0, bw = 8, w = 512, h = 512;
                {
                    std::string f[5]; int nf = 0; size_t q0 = 0;
                    while (nf < 5 && q0 <= one.size())
                    {
                        const size_t c = one.find(':', q0);
                        f[nf++] = one.substr(q0, c == std::string::npos ? std::string::npos : c - q0);
                        if (c == std::string::npos) break;
                        q0 = c + 1;
                    }
                    if (nf > 0 && !f[0].empty()) tbp = uint32_t(std::strtoul(f[0].c_str(), nullptr, 0));
                    if (nf > 1 && !f[1].empty()) psm = uint32_t(std::strtoul(f[1].c_str(), nullptr, 0));
                    if (nf > 2 && !f[2].empty()) bw  = uint32_t(std::strtoul(f[2].c_str(), nullptr, 0));
                    if (nf > 3 && !f[3].empty()) w   = uint32_t(std::strtoul(f[3].c_str(), nullptr, 0));
                    if (nf > 4 && !f[4].empty()) h   = uint32_t(std::strtoul(f[4].c_str(), nullptr, 0));
                }
                if (!bw) bw = 1;
                if (!w) w = bw * 64u;
                if (!h) h = 512u;
                // Blocks consumed per row at this width, for 32-bit-slot formats (C32/C24 share a
                // layout). 64 px per block row * bw -> bw blocks per 64-px row chunk.
                const uint32_t kBlocksTotal = m_vramSize / 256u;
                const uint32_t rowBlocks = (w * 4u) / 256u ? (w * 4u) / 256u : 1u;
                uint32_t hMax = (tbp < kBlocksTotal) ? ((kBlocksTotal - tbp) / rowBlocks) : 0u;
                const bool clamped = (h > hMax);
                if (clamped) h = hMax;
                if (h == 0u)
                {
                    std::fprintf(stderr, "[gs2:vramrect] SKIP tbp=%u: starts at/after the end of "
                                 "VRAM (%u blocks) -- a read there would WRAP to block 0\n",
                                 tbp, kBlocksTotal);
                    continue;
                }
                char path[512];
                std::snprintf(path, sizeof(path), "%s/vram_tbp%u_psm%02x_bw%u_%ux%u.ppm",
                              vrDir, tbp, psm, bw, w, h);
                FILE *f = std::fopen(path, "wb");
                if (!f) continue;
                std::fprintf(f, "P6\n%u %u\n255\n", w, h);
                for (uint32_t yy = 0; yy < h; ++yy)
                    for (uint32_t xx = 0; xx < w; ++xx)
                    {
                        const uint32_t c = ReadVramUnlocked(psm, tbp, bw, xx, yy);
                        std::fputc(int(c & 0xFFu), f);
                        std::fputc(int((c >> 8) & 0xFFu), f);
                        std::fputc(int((c >> 16) & 0xFFu), f);
                    }
                std::fclose(f);
                std::fprintf(stderr, "[gs2:vramrect] wrote %s at flip %llu%s\n", path,
                             (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                             clamped ? "  (HEIGHT CLAMPED to stay inside VRAM)" : "");
            }
        }
    }
    // ★★★★★ cont.356d: fold this flip's counts into the debounced 2D-screen verdict.
    if (hudAspectFullOn())
    {
        static unsigned long long s_lastFsFlip = ~0ull;
        const unsigned long long fnow = g_perfFlips.load(std::memory_order_relaxed);
        if (fnow != s_lastFsFlip)
        {
            s_lastFsFlip = fnow;
            // -1 = no evidence (HOLD), 0 = this frame drew 3D, 1 = this frame is a 2D screen.
            // Holding on no evidence is what lets OFF be immediate: the single-flip EMPTY frames
            // inside a menu era would otherwise each read as a vote for 3D and drop the verdict.
            const int vote = (g_fsPersp > 0ull) ? 0
                           : ((g_fsFull2d > 0ull && g_fsNarrow2d >= s_hudFullMinUi) ? 1 : -1);
            const bool raw = (vote == 1);
            // ★ cont.356g DECLARED mode: the table says WHEN, and evidence of 3D says STOP. A frame
            // that drew perspective geometry cancels the 2D state immediately, with no threshold and
            // no debounce -- "did this frame draw perspective geometry" is hardware truth, not a
            // tuned count, so it is the one part of the heuristic that transfers to any game. It
            // also makes a mis-curated table entry self-correcting within a frame instead of
            // pillarboxing a 3D scene indefinitely.
            if (hudFullMode() == kFullDeclared)
            {
                const unsigned dk = g_declaredKind.load(std::memory_order_relaxed);
                // ★ cont.356l: HOLD the 2D state briefly after the declaration turns 3D. The game
                // requests the NEXT screen a few frames before it stops drawing the current one
                // (measured: the title screen is still drawing its own UI, narrow2d=32, for 3
                // flips after id=17 is requested), so an instant drop un-corrects a screen that is
                // still on display -- reported as 'loses the correction just before I am sent to
                // the next one'. ⚠ EVIDENCE STILL WINS: a frame that actually draws perspective
                // ends the hold at once, so this cannot bring back the 'loads compressed, then
                // stretches' artefact, which was a real 3D scene being pillarboxed.
                // 3 = STRICT: the curator has looked at this screen and it is flat, so the
                // perspective gate is skipped. A screen can draw decorative geometry through a
                // perspective projection and still be a wholly flat 4:3 layout -- the gate cannot
                // tell that from a real 3D scene, and only someone looking at it can.
                static unsigned s_hold = 0u;
                const bool decl2d = (dk == 2u || dk == 3u);
                if (decl2d) s_hold = s_hudFullHold;           // re-arm while a 2D screen is declared
                else if (s_hold && g_fsPersp == 0ull) --s_hold;  // coasting through the handover
                else s_hold = 0u;                            // real 3D, or the hold ran out
                const bool held = (!decl2d && s_hold > 0u && g_fsPersp == 0ull);
                // A strict entry waits until its declaration has been stable for ONDELAY flips, so
                // the outgoing screen is gone before the pillarbox appears.
                static unsigned s_onDelay = 0u;
                if (dk == 3u)
                {
                    if (s_hudFullOnEvidence && g_fsPersp == 0ull) s_onDelay = s_hudFullOnDelay; // cont.362
                    else if (s_onDelay < s_hudFullOnDelay) ++s_onDelay;
                }
                else            s_onDelay = 0u;
                const bool strictReady = (dk == 3u && s_onDelay >= s_hudFullOnDelay);
                const bool want = strictReady || ((dk == 2u) && (g_fsPersp == 0ull)) || held;
                if (want != g_fs2dActive.load(std::memory_order_relaxed))
                {
                    g_fs2dActive.store(want, std::memory_order_relaxed);
                    std::fprintf(stderr, "[gs2:hudfull] flip=%llu 2D-screen=%s (declared=%u persp=%llu)\n",
                                 fnow, want ? "ON" : "off",
                                 g_declaredKind.load(std::memory_order_relaxed), g_fsPersp);
                }
            }
            else
            {
            // AUTO: the heuristic decides, and disagrees loudly with any declaration -- that is what
            // makes it a validator for a table rather than a second opinion nobody reads.
            {
                // ⚠ EDGE-TRIGGERED. A declaration is STICKY -- nothing re-declares when the game
                // leaves a menu for gameplay -- so a level-triggered log fired on every flip
                // afterwards (measured: 16,867 lines in one run, noise rather than signal).
                const unsigned d = g_declaredKind.load(std::memory_order_relaxed);
                static int s_lastDis = -1;
                const int dis = (d != 0u && ((d == 2u) != raw)) ? 1 : 0;
                const bool edge = (dis != s_lastDis);
                s_lastDis = dis;
                if (dis && edge)
                    std::fprintf(stderr, "[gs2:hudfull-disagree] flip=%llu declared=%s heuristic=%s "
                                 "(persp=%llu full2d=%llu narrow2d=%llu)\n", fnow,
                                 d == 2u ? "2D" : "3D", raw ? "2D" : "3D",
                                 g_fsPersp, g_fsFull2d, g_fsNarrow2d);
            }
            // Per-flip trace of the APPLY, not just the verdict: a state machine that flips
            // correctly while tagging nothing looks identical to one that works (cont.330).
            if (s_perspCount || g_fsApplied)
                std::fprintf(stderr, "[gs2:hudfull-apply] flip=%llu active=%d persp=%llu full2d=%llu "
                             "narrow2d=%llu vote=%d tagged=%llu\n",
                             fnow, g_fs2dActive.load(std::memory_order_relaxed) ? 1 : 0,
                             g_fsPersp, g_fsFull2d, g_fsNarrow2d, vote, g_fsApplied);
            g_fsApplied = 0ull;
            if (vote >= 0)
            {
                if (raw == g_fsLastRaw) { if (g_fsRun < 0xFFFFFFFFu) ++g_fsRun; }
                else                    { g_fsLastRaw = raw; g_fsRun = 1u; }
                const unsigned need = raw ? s_hudFullDebounce : s_hudFullOffDebounce;
                if (raw != g_fs2dActive.load(std::memory_order_relaxed) && g_fsRun >= need)
                {
                    g_fs2dActive.store(raw, std::memory_order_relaxed);
                    std::fprintf(stderr, "[gs2:hudfull] flip=%llu 2D-screen=%s (persp=%llu full2d=%llu "
                                 "narrow2d=%llu)\n", fnow, raw ? "ON" : "off",
                                 g_fsPersp, g_fsFull2d, g_fsNarrow2d);
                }
            }
            }
            g_fsPersp = g_fsFull2d = g_fsNarrow2d = 0ull;
            g_fsApplied = 0ull;   // declared mode never reaches the else-branch reset
        }
    }
    // ★★★★★ cont.356c: one line per flip. `persp` is the discriminator; `full2d` is the draw the
    // correction would act on. The pair "persp=0 with full2d>0" is exactly when the heuristic fires.
    if (s_perspCount)
    {
        static unsigned long long s_lastFlip = ~0ull;
        const unsigned long long fnow = g_perfFlips.load(std::memory_order_relaxed);
        if (fnow != s_lastFlip)
        {
            s_lastFlip = fnow;
            std::fprintf(stderr, "[gs2:perspcount] flip=%llu draws=%llu persp=%llu full2d=%llu narrow2d=%llu%s\n",
                         fnow, g_pcDraws, g_pcPersp, g_pcFull2d, g_pcNarrow2d,
                         (g_pcPersp == 0ull && g_pcFull2d > 0ull) ? "  <= WOULD FIRE" : "");
            g_pcDraws = g_pcPersp = g_pcFull2d = g_pcNarrow2d = 0;
        }
    }
    // cont.355b: the FST/width census, dumped where BOTH renderers reach it (see dumpFrameHist).
    if (s_frameHist && !g_frameHistDumped &&
        g_perfFlips.load(std::memory_order_relaxed) > s_probeFlip + s_probeFlipW)
    {
        g_frameHistDumped = true;
        dumpFrameHist();
    }
    // cont.358c: the HUDDUMP summary, on the same flip-counter trigger. Printed even when nothing
    // matched -- a silent instrument reads exactly like a null (the cont.331j rule).
    if (s_hudDump && !g_hudDumpReported &&
        g_perfFlips.load(std::memory_order_relaxed) > s_probeFlip + s_probeFlipW)
    {
        g_hudDumpReported = true;
        const unsigned long long tot = g_hudDumpTotal.load(std::memory_order_relaxed);
        const unsigned long long shown = tot < s_hudDumpMax ? tot : s_hudDumpMax;
        std::fprintf(stderr, "[gs2:huddump] CLASSIFIED draws in the probe window: total=%llu shown=%llu"
                     " no-cluster=%llu | frames gated ON=%llu OFF=%llu (run-count changed %llu)%s\n",
                     tot, shown, (unsigned long long)g_hudNoCluster.load(std::memory_order_relaxed),
                     (unsigned long long)g_hudGateOn.load(std::memory_order_relaxed),
                     (unsigned long long)g_hudGateOff.load(std::memory_order_relaxed),
                     (unsigned long long)g_runSplitFrames.load(std::memory_order_relaxed),
                     (tot > shown) ? "  ⚠ TRUNCATED -- raise PS2X_GS_HUDDUMP_MAX, this is NOT a census" : "");
    }
    // ★★★★ cont.331j ROWCENSUS report -- driven off the flip counter, which both renderers bump,
    // and printed even when nothing matched (a silent instrument reads exactly like a null).
    if (s_rowCensus || s_drawPages)
    {
        const unsigned long long rcFlip = g_perfFlips.load(std::memory_order_relaxed);
        if ((rcFlip % 200ull) == 0ull)
            rowCensusReport(rcFlip);
    }
    // cont.357c: driven off the flip counter (both renderers bump it) and printed periodically --
    // a census that only dumps at exit prints nothing when a run is killed by `timeout` (cont.329g).
    if (s_upCensus)
    {
        const unsigned long long uf = g_perfFlips.load(std::memory_order_relaxed);
        static unsigned long long s_upLast = ~0ull;
        if (uf != s_upLast && (uf % 100ull) == 0ull) { s_upLast = uf; upCensusReport(uf); }
    }
    if (s_pageWatch)
    {
        const unsigned long long pwF = g_perfFlips.load(std::memory_order_relaxed);
        g_pwFlip.store(pwF, std::memory_order_relaxed);
        if ((pwF % 200ull) == 0ull)
            pageWatchReport(pwF);
    }
    // cont.231: hand the snapshot to the worker (see s_gsFlipAsync). The whole-mirror verify needs
    // the DRAINED flip (it compares quiescent VRAM), so it keeps the old path.
    if (s_gsFlipAsync && s_gsThread && s_gsGpuMirrorVerifyMs <= 0)
    {
        WorkItem item;
        item.kind = WorkItem::Kind::FlipSnapshot;
        item.flipFb1 = preFlipDispfb1;
        item.flipFb2 = preFlipDispfb2;
        enqueueWork(std::move(item)); // flushes the pending DrawRun first: the marker lands after it
        // Arm the per-Sync capture gate now (as the drained flip did), not when the copy lands.
        g_gs2LastFlipSnapMs.store(gs2NowMs(), std::memory_order_release);
        return;
    }
    // The frame's tail (the UI text) is enqueued microseconds before the flip -- without a
    // drain the snapshot races the worker and captures the frame WITHOUT its text (proven
    // cont.165). This drain is bounded by one frame's tail at flip cadence (a few per second),
    // unlike the per-present drain that starved the main thread (cont.163).
    drainQueue();
#if PS2X_HAS_GS_GPU_DEVICE
    // ★ cont.331k: the same frame boundary on the DRAINED flip path (PS2X_GS_FLIPASYNC=0). The
    // drain has already parked the worker, so the accumulator is quiescent here.
    if (s_gsRendererGl && s_glFlipFlush)
    {
        const size_t pend = m_glPendVerts.size();
        if (pend)
        { ++g_glFlipFlushes; g_glFlipFlushVerts += pend; glFlushPending(); }
        else
            ++g_glFlipFlushEmpty;
    }
#endif
    {
        uint64_t seqNow = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::lock_guard<std::mutex> plock(m_presentCopyMutex);
            m_presentCopy.assign(m_vram, m_vram + m_vramSize);
            seqNow = ++m_presentCopySeq;
            m_presentCopyHasRegs = true;
            m_presentCopyDispfb1 = preFlipDispfb1;
            m_presentCopyDispfb2 = preFlipDispfb2;
            const uint64_t now = gs2NowMs();
            m_presentCopyTickMs.store(now, std::memory_order_release);
            g_gs2LastFlipSnapMs.store(now, std::memory_order_release);
        }
#if PS2X_HAS_GS_GPU_DEVICE
        glSnapshotDisplayTargets(preFlipDispfb1, preFlipDispfb2, seqNow); // cont.345, drained path
#endif
    }
#if PS2X_HAS_GS_GPU_DEVICE
    // cont.168 full-mirror shadow verify (PS2X_GS_GPU_MIRRORVERIFY=<ms>, default OFF).
    // A drained flip is the one moment CPU VRAM is quiescent and coherent, so it is the
    // only honest place to compare the whole mirror. The device runs it behind every
    // pending mirror job and this call waits for the answer, so what it reports is a real
    // gap: a VRAM writer with no mirror path (today: the rasterizer -- phase 2), never a
    // race. Rate-limited because it moves 4MB to the GPU per run.
    if (s_gsGpuMirrorVerifyMs > 0 && m_vram && m_vramSize != 0u)
    {
        static uint64_t s_lastMv = 0;
        const uint64_t nowMv = gs2NowMs();
        if (nowMv - s_lastMv >= static_cast<uint64_t>(s_gsGpuMirrorVerifyMs))
        {
            s_lastMv = nowMv;
            std::lock_guard<std::mutex> lock(m_mutex);
            if (s_gsGpuBatchVerify > 0 && !m_gpuBatch.prims.empty())
    { ++g_bvFlushExplicit; gpuBatchFlush(); }
            mirrorFlushPokes(); // everything written so far must be IN the mirror first
            if (GsGpuPresentDevice *dev = gs2GpuDevice())
                dev->MirrorVerifyFull(m_vram, m_vramSize, m_drawPages.data());
            m_drawPages.fill(0u); // next window starts with no draws to excuse
            gs2PrintMirrorPatch(); // which writer fed the mirror since the last report
        }
    }
#endif
    if (g_gs2FontTrace)
    {
        static unsigned long s_nf = 0;
        static unsigned long s_lastFontDraws = 0;
        const unsigned long fontDrawsNow = g_gs2FontDraws;
        ++s_nf;
        if (s_nf <= 32u || (s_nf % 256u) == 0u)
            std::fprintf(stderr,
                         "[gs2:flipsnap] #%lu captured pre-flip frame (glyphDraws+%lu since prev flip, "
                         "last glyph fbp=%u, preFb1.fbp=%u preFb2.fbp=%u)\n",
                         s_nf, fontDrawsNow - s_lastFontDraws,
                         g_gs2LastFontFbp.load(std::memory_order_relaxed),
                         static_cast<uint32_t>(preFlipDispfb1 & 0x1FFu),
                         static_cast<uint32_t>(preFlipDispfb2 & 0x1FFu));
        s_lastFontDraws = fontDrawsNow;
    }
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    // Ordered wrt queued work: drain, then write (single guest producer, so nothing can
    // slip in between).
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    ++g_vramMutationSeq;
    if (s_gsGpuBatchVerify > 0 && !m_gpuBatch.prims.empty())
    { ++g_bvFlushExplicit; gpuBatchFlush(); }
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.320: was per pixel in WriteVramUnlocked
    gs2TexCensusBumpAll(g_texCensus.bumpAllPoke); // cont.320: a guest poke, rare -- counted
    if (s_glTexTrace != 0u || s_mutCensus || s_pageWatch) // cont.331j: PAGEWATCH needs these sites too
        glTexTraceMutation("poke(WriteVram)", base / 32u, base / 32u, psm, bw, x, y, x, y, true);
    WriteVramUnlocked(psm, base, bw, x, y, value);
#if PS2X_HAS_GS_GPU_DEVICE
    // Host-side pokes (LOTR's movie frame blit, the CLUT guard, font diagnostics) mutate
    // VRAM outside every transfer path -- mirror them too, or the mirror drifts. They
    // come one pixel at a time and by the hundred thousand, so accumulate a dirty box
    // and mirror it as a single rect (see mirrorFlushPokes).
    if (gs2MirrorWanted())
    {
        size_t slot = kPokeBoxes;
        size_t freeSlot = kPokeBoxes;
        for (size_t i = 0; i < kPokeBoxes; ++i)
        {
            if (!m_pokeBox[i].used)
            {
                if (freeSlot == kPokeBoxes)
                    freeSlot = i;
                continue;
            }
            if (m_pokeBox[i].psm == psm && m_pokeBox[i].bp == base && m_pokeBox[i].bw == bw)
            {
                slot = i;
                break;
            }
        }
        if (slot == kPokeBoxes)
        {
            // No live box for this destination: take a free slot, or evict slot 0 (all
            // slots busy means more than kPokeBoxes destinations are interleaving, which
            // no observed path does -- evicting is correct, just less efficient).
            slot = freeSlot;
            if (slot == kPokeBoxes)
            {
                mirrorFlushPokeSlot(0);
                slot = 0;
            }
            m_pokeBox[slot] = PokeBox{psm, base, bw, x, y, x, y, true};
        }
        else
        {
            PokeBox &b = m_pokeBox[slot];
            b.x0 = std::min(b.x0, x);
            b.y0 = std::min(b.y0, y);
            b.x1 = std::max(b.x1, x);
            b.y1 = std::max(b.y1, y);
        }
        // Bound the box so one stray far-away poke cannot turn the next flush into a
        // whole-VRAM copy. A movie strip (128x448 = 57k px) stays well under this.
        const PokeBox &b = m_pokeBox[slot];
        if (static_cast<uint64_t>(b.x1 - b.x0 + 1u) * (b.y1 - b.y0 + 1u) > 512ull * 1024ull)
            mirrorFlushPokeSlot(slot);
    }
#endif
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    // cont.320: no generation bump per PIXEL any more -- every caller bumps once at its top
    // (UploadImageUnlocked: the chunk's exact pages; PerformLocalToLocalTransfer, ClearFramebuffer,
    // WriteVram: whole VRAM). The old per-pixel gs2BumpAllPages() was 512 atomics per pixel on the
    // generic upload path (every PSMT8 upload).
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

#if PS2X_HAS_GS_GPU_DEVICE
namespace
{
    // Page extent in PIXELS per psm -- the same table PixelStorageTraits<>::PageExtent()
    // encodes (ps2_gs_memory.h). Undefined psm codes fall through to the CT32 layout,
    // matching the psm-handler table's undefined-slot fallback (PCSX2 GSLocalMemory.cpp).
    void gs2PageExtent(uint32_t psm, uint32_t &pw, uint32_t &ph)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            pw = 64u;
            ph = 64u;
            break;
        case GS_PSM_T8:
            pw = 128u;
            ph = 64u;
            break;
        case GS_PSM_T4:
            pw = 128u;
            ph = 128u;
            break;
        default:
            pw = 64u;
            ph = 32u;
            break;
        }
    }

    // Extra pages a rect can spill past its own last page. Address() adds
    // (bp%32 + BlockId)*PixelsPerBlock, so when bp is NOT page-aligned a pixel can land up
    // to ~63/32 of a page beyond its page -- but when bp%32 == 0 the offset is bounded by
    // PixelsPerPage and there is NO spill at all. Making this conditional is not just an
    // optimization: a fixed +2 pushed a 512x512 texture based at the LAST page group past
    // the top of VRAM, so the range was rejected and the phase-2a texture seed silently
    // did not happen (cont.170 -- it looked like a sampler bug for one sprite combo).
    uint32_t gs2PageSpill(uint32_t bp) { return (bp % 32u == 0u) ? 0u : 2u; }

    // The single contiguous VRAM byte range a rect occupies, in whole 8KB pages. Fails
    // rather than wrapping, so callers that need ONE range (the phase-2a raster verify)
    // can simply skip.
    bool gs2PageRange(uint32_t psm, uint32_t bp, uint32_t bw,
                      uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                      uint32_t vramSize, uint32_t &off, uint32_t &len)
    {
        if (x1 < x0 || y1 < y0)
            return false;
        uint32_t pw = 64u, ph = 32u;
        gs2PageExtent(psm & 0x3Fu, pw, ph);
        const uint32_t stride = (std::max<uint32_t>(bw, 1u) * 64u) / pw;
        const uint64_t basePage = bp / 32u;
        const uint64_t first = basePage + static_cast<uint64_t>(y0 / ph) * stride + (x0 / pw);
        const uint64_t last = basePage + static_cast<uint64_t>(y1 / ph) * stride + (x1 / pw) +
                              gs2PageSpill(bp);
        const uint64_t start = first * 8192ull;
        const uint64_t end = (last + 1ull) * 8192ull;
        if (start >= end || end > vramSize)
            return false;
        off = static_cast<uint32_t>(start);
        len = static_cast<uint32_t>(end - start);
        return true;
    }

    // Every psm whose page layout is PageTableC32 with a 32-bit pixel stride. For all of
    // these, pixel (x,y) resolves to the SAME word address as CT32 at (x,y) -- the
    // sub-word formats (CT24 keeps the top byte, T8H/T4HL/T4HH occupy bit fields of it)
    // differ only in which bits of that word they touch. So reading the finished word
    // back as CT32 and re-writing it as CT32 reproduces the CPU result exactly, which
    // lets the existing CT32 swizzle kernel mirror all of them.
    bool gs2IsC32Layout(uint32_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_CT24:
        case GS_PSM_Z32:
        case GS_PSM_Z24:
        case GS_PSM_T8H:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return true;
        default:
            // Undefined psm codes decode as CT32 (the psm-handler table's fallback), and
            // that is exactly the C32 layout.
            return psm != GS_PSM_CT16 && psm != GS_PSM_CT16S && psm != GS_PSM_Z16 &&
                   psm != GS_PSM_Z16S && psm != GS_PSM_T8 && psm != GS_PSM_T4;
        }
    }

    // mirrorPatchRect attribution ([gsgpu:mpatch]): which VRAM writer is feeding the
    // mirror, and at what cost. Written only from the GS worker / drained callers.
    struct Gs2MirrorPatchStats
    {
        unsigned long n = 0;
        unsigned long long words = 0;
        unsigned long rawN = 0;
        unsigned long long rawBytes = 0;
    };
    Gs2MirrorPatchStats g_mpStats[4]; // indexed by GSCpuBackend::MirrorPatchSource
    const char *const kMpSourceName[4] = {"upload", "l2l", "clear", "poke"};

    // Phase-2b batch tallies, printed with the other GPU counters.
    unsigned long g_bvBatches = 0, g_bvVerified = 0, g_bvFlushTarget = 0, g_bvFlushMutation = 0;
    unsigned long g_bvFlushCap = 0, g_bvFlushStates = 0, g_bvFlushHazard = 0;
    unsigned long g_bvReplays = 0, g_bvReplayTilingBug = 0, g_bvReplayPrimBug = 0;
    unsigned long long g_bvPrims = 0, g_bvBinEntries = 0;
    unsigned long g_bvMaxPrims = 0, g_bvMaxStates = 0;
    // cont.233 hardware-raster census (PS2X_GS_HWRASTER_CENSUS=1): consecutive same-state
    // runs (one draw each if the state became a per-draw uniform) and the conflict LAYERS a
    // batch needs if ordering came from barriers between non-overlapping draws (8x8 tiles).
    unsigned long g_bvStateRuns = 0, g_bvLayerSum = 0, g_bvLayerMax = 0;
    unsigned long long g_bvVerifiedPrims = 0;
    unsigned long g_bvSkippedBatches = 0;
    unsigned long g_bvVerifiedMaxPrims = 0;

    // Phase-2a/2b coverage: of the primitives the verify SAMPLED, how many did the GPU
    // path actually accept, and why were the rest turned away? Without this, "0 mismatches"
    // could just mean "we verified almost nothing".
    enum Gs2RvReject { kRvOk = 0, kRvPrimType, kRvVerts, kRvFeature, kRvDegenerate,
                       kRvOffscreen, kRvRange, kRvNoDevice, kRvReasonCount };
    unsigned long g_rvStats[kRvReasonCount] = {};
    unsigned long gs2RvRejectedTotal()
    {
        unsigned long n = 0;
        for (unsigned i = 0; i < kRvReasonCount; ++i)
            if (i != kRvOk)
                n += g_rvStats[i];
        return n;
    }
    const char *const kRvName[kRvReasonCount] = {"ok", "primtype", "verts", "feature",
                                                 "degenerate", "offscreen", "range", "nodevice"};

    void gs2PrintMirrorPatch()
    {
        std::fprintf(stderr, "[gsgpu:mpatch]");
        for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, " %s{n=%lu kernel-words=%llu raw=%lu raw-bytes=%llu}",
                         kMpSourceName[i], g_mpStats[i].n, g_mpStats[i].words,
                         g_mpStats[i].rawN, g_mpStats[i].rawBytes);
        std::fprintf(stderr, "\n");
        if (s_gsGpuBatchPerf || s_gsSkipCpuRaster)
            gs2PrintRasterThroughput();
        if (s_gsGpuBatchPerf)
        {
            if (GsGpuPresentDevice *dev = gs2GpuDevice())
            {
                unsigned long long gns = 0, gp = 0;
                unsigned long gd = 0;
                dev->PerfStats(gns, gp, gd);
                const double cpuMs = double(g_perfCpuNs) / 1e6;
                const double gpuMs = double(gns) / 1e6;
                std::fprintf(stderr,
                             "[gsgpu:perf] CPU raster %.0f ms over %llu prims (%.2f us/prim) | "
                             "GPU raster %.0f ms over %llu prims in %lu dispatches (%.2f us/prim) "
                             "| speedup %.2fx\n",
                             cpuMs, g_perfCpuPrims,
                             g_perfCpuPrims ? double(g_perfCpuNs) / 1e3 / double(g_perfCpuPrims) : 0.0,
                             gpuMs, gp, gd,
                             gp ? double(gns) / 1e3 / double(gp) : 0.0,
                             gpuMs > 0.0 ? cpuMs / gpuMs : 0.0);
            }
        }
        if (s_gsGpuBatchVerify > 0 || s_gsGpuBatchPerf || s_rasterBenchGpu)
            std::fprintf(stderr,
                         "[gsgpu:batch] batches=%lu verified=%lu prims=%llu bin-entries=%llu "
                         "mean-prims=%.1f max-prims=%lu max-states=%lu | verified-prims=%llu "
                         "verified-max=%lu unverifiable=%lu | replay{n=%lu tiling=%lu prim=%lu} | flush{target=%lu "
                         "mutation=%lu cap=%lu states=%lu hazard=%lu explicit=%lu}\n",
                         g_bvBatches, g_bvVerified, g_bvPrims, g_bvBinEntries,
                         g_bvBatches ? double(g_bvPrims) / double(g_bvBatches) : 0.0,
                         g_bvMaxPrims, g_bvMaxStates, g_bvVerifiedPrims,
                         g_bvVerifiedMaxPrims, g_bvSkippedBatches,
                         g_bvReplays, g_bvReplayTilingBug, g_bvReplayPrimBug,
                         g_bvFlushTarget, g_bvFlushMutation,
                         g_bvFlushCap, g_bvFlushStates, g_bvFlushHazard,
                         g_bvFlushExplicit);
        if (s_gsGpuRasterVerify > 0 || s_gsGpuBatchVerify > 0)
        {
            unsigned long total = 0;
            for (int i = 0; i < kRvReasonCount; ++i)
                total += g_rvStats[i];
            std::fprintf(stderr, "[gsgpu:rvcover] sampled=%lu", total);
            for (int i = 0; i < kRvReasonCount; ++i)
                if (g_rvStats[i] != 0u)
                    std::fprintf(stderr, " %s=%lu", kRvName[i], g_rvStats[i]);
            std::fprintf(stderr, " (accepted %.1f%%)\n",
                         total ? 100.0 * double(g_rvStats[kRvOk]) / double(total) : 0.0);
        }
        if (s_gsGpuRasterVerify > 0)
        {
            if (GsGpuPresentDevice *dev = gs2GpuDevice())
            {
                unsigned long prims = 0, verifies = 0, mismRanges = 0;
                unsigned long long mismWords = 0;
                dev->RasterStats(prims, verifies, mismRanges, mismWords);
                std::fprintf(stderr,
                             "[gsgpu:raster] prims=%lu ranges-compared=%lu mismatch-ranges=%lu "
                             "mismatch-words=%llu\n",
                             prims, verifies, mismRanges, mismWords);
            }
        }
    }
} // namespace

// Mirror the authoritative CPU bytes for a just-written rect (cont.168, phase 1b).
//
// Two routes. For the C32-layout psm family the destination words are read back as CT32
// and pushed through the phase-1 swizzle KERNEL -- exact, and it moves 4 bytes per pixel
// actually written. Everything else falls back to a whole-8KB-PAGE raw copy: PageId() is
// monotonic in x and y, so the rect's pages span [page(x0,y0), page(x1,y1)], and a pixel
// address can spill up to two pages past its own page when bp is not page-aligned
// (Address() adds (bp%32 + BlockId)*PixelsPerBlock, at most 63/32 of a page) -- hence the
// +2 page tail. Over-copying is harmless (every byte copied is authoritative), but at 3
// pages minimum it is far too coarse for small rects, which is why the kernel route
// exists: run175a measured 266k page-copies/s (6.5 GB/s) with raw as the only route.
// Shared by both prim builders: every field that does not depend on the primitive's
// SHAPE. Kept in one place so the triangle and sprite paths cannot describe the same GS
// state differently. `mip` is the host-resolved LOD (always 0 for sprites -- DrawSprite
// calls SampleTexture without a mip argument).
void GSCpuBackend::gs2FillCommonPrimState(const GSPrimitiveBatch &batch, GsGpuState &p, uint32_t mip)
{
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;
    // m_draw was resolved at DrawPrimitive entry for this very batch.
    p.fbp = m_draw.fbp; p.fbw = m_draw.fbw;
    p.fpsm = ctx.frame.psm; p.fbmsk = ctx.frame.fbmsk;
    p.zbp = m_draw.zbp; p.zpsm = ctx.zbuf.psm; p.zmsk = ctx.zbuf.zmask ? 1u : 0u;
    p.scissorX0 = ctx.scissor.x0; p.scissorY0 = ctx.scissor.y0;
    p.scissorX1 = ctx.scissor.x1; p.scissorY1 = ctx.scissor.y1;

    p.primType = st.prim.type;
    p.iip = st.prim.iip ? 1u : 0u;
    p.tme = st.prim.tme ? 1u : 0u;
    p.fge = st.prim.fge ? 1u : 0u;
    p.fst = st.prim.fst ? 1u : 0u;
    p.abe = st.prim.abe ? 1u : 0u;

    p.ate = static_cast<uint32_t>(ctx.test & 1u);
    p.atst = static_cast<uint32_t>((ctx.test >> 1) & 7u);
    p.aref = static_cast<uint32_t>((ctx.test >> 4) & 0xFFu);
    p.afail = static_cast<uint32_t>((ctx.test >> 12) & 3u);
    p.zte = static_cast<uint32_t>((ctx.test >> 16) & 1u);
    p.ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    p.blendA = static_cast<uint32_t>(ctx.alpha & 3u);
    p.blendB = static_cast<uint32_t>((ctx.alpha >> 2) & 3u);
    p.blendC = static_cast<uint32_t>((ctx.alpha >> 4) & 3u);
    p.blendD = static_cast<uint32_t>((ctx.alpha >> 6) & 3u);
    p.blendFix = static_cast<uint32_t>((ctx.alpha >> 32) & 0xFFu);
    p.fba = static_cast<uint32_t>(ctx.fba & 1u);
    p.fogR = st.fogR; p.fogG = st.fogG; p.fogB = st.fogB;

    // Mip is resolved by the CALLER (the shader must not re-derive a log2): fold the
    // level's base/width/size and region bounds in, and hand over the FST texel divisor.
    uint32_t mipTbp = ctx.tex0.tbp0, mipTbw = ctx.tex0.tbw;
    int texW = st.textureWidth, texH = st.textureHeight;
    if (mip > 0u)
    {
        const uint64_t m1 = ctx.miptbp1, m2 = ctx.miptbp2;
        switch (mip)
        {
        case 1: mipTbp = static_cast<uint32_t>(m1 & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 14) & 0x3Fu); break;
        case 2: mipTbp = static_cast<uint32_t>((m1 >> 20) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 34) & 0x3Fu); break;
        case 3: mipTbp = static_cast<uint32_t>((m1 >> 40) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 54) & 0x3Fu); break;
        case 4: mipTbp = static_cast<uint32_t>(m2 & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 14) & 0x3Fu); break;
        case 5: mipTbp = static_cast<uint32_t>((m2 >> 20) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 34) & 0x3Fu); break;
        default: mipTbp = static_cast<uint32_t>((m2 >> 40) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 54) & 0x3Fu); break;
        }
        texW = std::max(1, texW >> mip);
        texH = std::max(1, texH >> mip);
    }
    p.tbp = mipTbp; p.tbw = mipTbw; p.tpsm = ctx.tex0.psm;
    p.tcc = ctx.tex0.tcc; p.tfx = ctx.tex0.tfx;
    p.cbp = ctx.tex0.cbp; p.cpsm = ctx.tex0.cpsm; p.csm = ctx.tex0.csm; p.csa = ctx.tex0.csa;
    p.texW = static_cast<uint32_t>(texW); p.texH = static_cast<uint32_t>(texH);
    p.linear = st.linearFilter ? 1u : 0u;
    p.fstDiv = 16.0f * static_cast<float>(1u << mip);
    p.wms = static_cast<uint32_t>(ctx.clamp & 3u);
    p.wmt = static_cast<uint32_t>((ctx.clamp >> 2) & 3u);
    p.minU = static_cast<uint32_t>(((ctx.clamp >> 4) & 0x3FFu) >> mip);
    p.maxU = static_cast<uint32_t>(((ctx.clamp >> 14) & 0x3FFu) >> mip);
    p.minV = static_cast<uint32_t>(((ctx.clamp >> 24) & 0x3FFu) >> mip);
    p.maxV = static_cast<uint32_t>(((ctx.clamp >> 34) & 0x3FFu) >> mip);
    p.ta0 = st.texa.ta0; p.ta1 = st.texa.ta1; p.aem = st.texa.aem ? 1u : 0u;

    if (s_gsClutShadow && st.prim.tme && ctx.tex0.csm == 0u && ctx.tex0.cpsm == GS_PSM_CT32)
    {
        auto it = m_clutShadow.find(ctx.tex0.cbp);
        if (it != m_clutShadow.end())
        {
            p.clutShadowHave = 1u;
            std::memcpy(p.clutShadow, it->second.data(), sizeof(p.clutShadow));
        }
    }

}

// Seed the mirror with the CPU's PRE-draw bytes for everything this primitive will read
// or write, then post the raster. FIFO order guarantees the patches land first.
void GSCpuBackend::rasterSeedAndPost(const GsGpuState &st, GsGpuGeom &&g,
                                     const RasterVerifyCtx &out)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev)
        return;
    mirrorFlushPokes();
    // Seed: the mirror's frame/Z pages become the CPU's PRE-draw bytes.
    dev->MirrorPatchRaw(out.frameOff,
                        std::vector<uint8_t>(m_vram + out.frameOff, m_vram + out.frameOff + out.frameLen));
    if (out.zSeed)
        dev->MirrorPatchRaw(out.zOff,
                            std::vector<uint8_t>(m_vram + out.zOff, m_vram + out.zOff + out.zLen));
    // Seed the TEXTURE and CLUT too (PS2X_GS_GPU_RASTERTEXSEED, default ON). The mirror is
    // complete for every VRAM writer EXCEPT the rasterizer, so a draw that samples a
    // RENDER TARGET -- which this game does -- would read stale texels and diverge for a
    // reason that has nothing to do with the pixel pipeline under test. Seeding isolates
    // the pipeline; it becomes unnecessary once phase 2 makes the GPU write draws too.
    // Turning it OFF is the measurement of how much render-to-texture actually happens.
    if (s_gsGpuRasterTexSeed && st.tme != 0u)
    {
        uint32_t tOff = 0, tLen = 0;
        if (st.texW != 0u && st.texH != 0u &&
            gs2PageRange(st.tpsm, st.tbp, st.tbw, 0u, 0u, st.texW - 1u, st.texH - 1u, m_vramSize, tOff, tLen))
            dev->MirrorPatchRaw(tOff, std::vector<uint8_t>(m_vram + tOff, m_vram + tOff + tLen));
        uint32_t cOff = 0, cLen = 0;
        const bool paletted = st.tpsm == GS_PSM_T4 || st.tpsm == GS_PSM_T8 || st.tpsm == GS_PSM_T4HL ||
                              st.tpsm == GS_PSM_T4HH || st.tpsm == GS_PSM_T8H;
        if (paletted && gs2PageRange(st.cpsm, st.cbp, 1u, 0u, 0u, 15u, 15u, m_vramSize, cOff, cLen))
            dev->MirrorPatchRaw(cOff, std::vector<uint8_t>(m_vram + cOff, m_vram + cOff + cLen));
    }
    GsGpuBatch batch;
    batch.states.push_back(st);
    g.stateIndex = 0u;
    batch.prims.push_back(std::move(g));
    batch.vram = m_vram; batch.vramSize = m_vramSize; // cont.250 PS2X_GS_LINTEX decode source
    dev->RasterBatch(std::move(batch)); // a batch of one -- same shader, same code path
}

// Phase 2b (cont.170): the SPRITE path. DrawSprite is not a degenerate triangle -- it
// takes colour/Z/fog flat from v1 (ignoring IIP), truncates v1.z with no +0.5, walks an
// axis-aligned scissor-clipped rect, and derives UV from a screen-space lerp between two
// corners (with an FST fixed-point round trip). Everything downstream of the texel is the
// same WritePixel, so only the setup differs.
bool GSCpuBackend::rasterVerifyBeginSprite(const GSPrimitiveBatch &batch, RasterVerifyCtx &out,
                                           GsGpuState *bs, GsGpuGeom *bg)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev)
        return false;
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];

    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);
    const int unclippedX0 = x0, unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return ++g_rvStats[kRvOffscreen], false; // DrawSprite renders nothing either
    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    if (drawX1 < drawX0 || drawY1 < drawY0)
        return ++g_rvStats[kRvOffscreen], false;

    const uint32_t fbp = m_draw.fbp, fbw = m_draw.fbw, zbp = m_draw.zbp;
    const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    const bool zWritten = !ctx.zbuf.zmask;
    const bool zUsed = zWritten || ztst == 2u || ztst == 3u;
    const bool buildOnly = (bs != nullptr && bg != nullptr);
    if (!buildOnly &&
        !gs2PageRange(ctx.frame.psm, fbp, fbw, static_cast<uint32_t>(drawX0), static_cast<uint32_t>(drawY0),
                      static_cast<uint32_t>(drawX1), static_cast<uint32_t>(drawY1), m_vramSize,
                      out.frameOff, out.frameLen))
        return ++g_rvStats[kRvRange], false;
    if (!buildOnly && zUsed &&
        !gs2PageRange(ctx.zbuf.psm, zbp, fbw, static_cast<uint32_t>(drawX0), static_cast<uint32_t>(drawY0),
                               static_cast<uint32_t>(drawX1), static_cast<uint32_t>(drawY1), m_vramSize,
                               out.zOff, out.zLen))
        return ++g_rvStats[kRvRange], false;
    out.zSeed = zUsed;
    out.zWritten = zWritten;

    GsGpuState stt;
    GsGpuGeom g;
    gs2FillCommonPrimState(batch, stt, /*mip*/ 0u); // sprites never mipmap (SampleTexture mip=0)
    stt.primType = GS_PRIM_SPRITE;
    g.minX = static_cast<uint32_t>(drawX0); g.minY = static_cast<uint32_t>(drawY0);
    g.maxX = static_cast<uint32_t>(drawX1); g.maxY = static_cast<uint32_t>(drawY1);
    g.unclipX0 = static_cast<uint32_t>(unclippedX0);
    g.unclipY0 = static_cast<uint32_t>(unclippedY0);
    g.spriteW = std::max(1.0f, static_cast<float>(spanX));
    g.spriteH = std::max(1.0f, static_cast<float>(spanY));
    g.spriteZ = static_cast<uint32_t>(v1.z);
    g.rgba1 = static_cast<uint32_t>(v1.r) | (static_cast<uint32_t>(v1.g) << 8) |
              (static_cast<uint32_t>(v1.b) << 16) | (static_cast<uint32_t>(v1.a) << 24);
    g.fog1 = v1.fog;
    if (st.prim.tme)
    {
        if (st.prim.fst)
        {
            g.su0 = static_cast<float>(v0.u >> 4); g.sv0 = static_cast<float>(v0.v >> 4);
            g.su1 = static_cast<float>(v1.u >> 4); g.sv1 = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q), q1 = fabsQ(v1.q);
            const float tw = static_cast<float>(st.textureWidth);
            const float th = static_cast<float>(st.textureHeight);
            g.su0 = (v0.s / q0) * tw; g.sv0 = (v0.t / q0) * th;
            g.su1 = (v1.s / q1) * tw; g.sv1 = (v1.t / q1) * th;
        }
    }

    if (buildOnly)
    {
        // Batch mode: hand back the built primitive; the accumulator owns seeding/posting.
        *bs = stt;
        *bg = std::move(g);
        ++g_rvStats[kRvOk];
        return true;
    }
    // Tag = the census key, so a mismatch names the draw-state combo directly.
    out.tag = stateCensusKey(batch);
    out.active = true;
    ++g_rvStats[kRvOk];
    rasterSeedAndPost(stt, std::move(g), out);
    return true;
}

// Phase 2a (cont.169). Returns true when the primitive was handed to the GPU and the
// caller must call rasterVerifyEnd() after the CPU draw.
bool GSCpuBackend::rasterVerifyBegin(const GSPrimitiveBatch &batch, RasterVerifyCtx &out,
                                     GsGpuState *bs, GsGpuGeom *bg)
{
    out = RasterVerifyCtx{};
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev || !m_vram || m_vramSize == 0u)
        return ++g_rvStats[kRvNoDevice], false;
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;

    // Coverage gate: triangles only for now, and none of the features the shader does not
    // implement. The run171 census says this game never uses DATE/PABE/CSM2 at all, so
    // this excludes nothing real -- but a gate beats silently "verifying" a path the
    // shader ignores.
    const bool isSprite = st.prim.type == GS_PRIM_SPRITE;
    const bool isTriangle = st.prim.type == GS_PRIM_TRIANGLE ||
                            st.prim.type == GS_PRIM_TRISTRIP ||
                            st.prim.type == GS_PRIM_TRIFAN;
    if (!isSprite && !isTriangle)
        return ++g_rvStats[kRvPrimType], false;
    if (batch.vertexCount < (isSprite ? 2u : 3u))
        return ++g_rvStats[kRvVerts], false;
    if (((ctx.test >> 14) & 0x1u) != 0u || st.pabe)
        return ++g_rvStats[kRvFeature], false;
    if (st.prim.tme && ctx.tex0.csm != 0u)
        return ++g_rvStats[kRvFeature], false;
    if (isSprite)
        return rasterVerifyBeginSprite(batch, out, bs, bg);

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    const float fx0 = v0.x - static_cast<float>(ofx), fy0 = v0.y - static_cast<float>(ofy);
    const float fx1 = v1.x - static_cast<float>(ofx), fy1 = v1.y - static_cast<float>(ofy);
    const float fx2 = v2.x - static_cast<float>(ofx), fy2 = v2.y - static_cast<float>(ofy);

    // Exactly DrawTriangle's setup, so the GPU sees the identical bbox and weights.
    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));
    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);
    const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return ++g_rvStats[kRvDegenerate], false; // the oracle draws nothing either

    const uint32_t fbp = m_draw.fbp;
    const uint32_t fbw = m_draw.fbw;
    const uint32_t zbp = m_draw.zbp;
    const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    const bool zWritten = !ctx.zbuf.zmask;
    const bool zUsed = zWritten || ztst == 2u || ztst == 3u; // 2/3 READ the Z buffer
    const bool buildOnly = (bs != nullptr && bg != nullptr);
    if (!buildOnly &&
        !gs2PageRange(ctx.frame.psm, fbp, fbw, static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                      static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY), m_vramSize,
                      out.frameOff, out.frameLen))
        return ++g_rvStats[kRvRange], false;
    if (!buildOnly && zUsed &&
        !gs2PageRange(ctx.zbuf.psm, zbp, fbw, static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                               static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY), m_vramSize,
                               out.zOff, out.zLen))
        return ++g_rvStats[kRvRange], false;
    out.zSeed = zUsed;
    out.zWritten = zWritten;

    GsGpuState stt;
    GsGpuGeom g;
    g.x0 = fx0; g.y0 = fy0; g.x1 = fx1; g.y1 = fy1; g.x2 = fx2; g.y2 = fy2;
    auto splitDouble = [](double d, uint32_t &lo, uint32_t &hi)
    {
        uint64_t bits = 0;
        std::memcpy(&bits, &d, sizeof(bits));
        lo = static_cast<uint32_t>(bits);
        hi = static_cast<uint32_t>(bits >> 32);
    };
    splitDouble(v0.z, g.z0lo, g.z0hi);
    splitDouble(v1.z, g.z1lo, g.z1hi);
    splitDouble(v2.z, g.z2lo, g.z2hi);
    auto packRgba = [](const GSVertex &v)
    {
        return static_cast<uint32_t>(v.r) | (static_cast<uint32_t>(v.g) << 8) |
               (static_cast<uint32_t>(v.b) << 16) | (static_cast<uint32_t>(v.a) << 24);
    };
    g.rgba0 = packRgba(v0); g.rgba1 = packRgba(v1); g.rgba2 = packRgba(v2);
    g.q0 = v0.q; g.q1 = v1.q; g.q2 = v2.q;
    g.s0 = v0.s; g.s1 = v1.s; g.s2 = v2.s;
    g.t0 = v0.t; g.t1 = v1.t; g.t2 = v2.t;
    g.uv0 = v0.u | (static_cast<uint32_t>(v0.v) << 16);
    g.uv1 = v1.u | (static_cast<uint32_t>(v1.v) << 16);
    g.uv2 = v2.u | (static_cast<uint32_t>(v2.v) << 16);
    g.fog0 = v0.fog; g.fog1 = v1.fog; g.fog2 = v2.fog;

    g.winding = (denom < 0.0f) ? -1.0f : 1.0f;
    g.invAbsDenom = 1.0f / std::fabs(denom);
    g.minX = static_cast<uint32_t>(minX); g.minY = static_cast<uint32_t>(minY);
    g.maxX = static_cast<uint32_t>(maxX); g.maxY = static_cast<uint32_t>(maxY);

    // Same LOD the oracle will resolve for this triangle (shared helper, cannot drift).
    gs2FillCommonPrimState(batch, stt, gs2TriangleMip(batch, denom));
    stt.primType = st.prim.type;

    // Tag = the census key, so a mismatch names the draw-state combo directly.
    if (buildOnly)
    {
        // Batch mode: hand back the built primitive; the accumulator owns seeding/posting.
        *bs = stt;
        *bg = std::move(g);
        ++g_rvStats[kRvOk];
        return true;
    }
    // Tag = the census key, so a mismatch names the draw-state combo directly.
    out.tag = stateCensusKey(batch);
    out.active = true;
    ++g_rvStats[kRvOk];
    rasterSeedAndPost(stt, std::move(g), out);
    return true;
}

void GSCpuBackend::rasterVerifyEnd(const RasterVerifyCtx &ctx)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!ctx.active || !dev || !m_vram)
        return;
    dev->MirrorVerifyRange(ctx.frameOff,
                           std::vector<uint8_t>(m_vram + ctx.frameOff, m_vram + ctx.frameOff + ctx.frameLen),
                           ctx.tag);
    if (ctx.zWritten)
        dev->MirrorVerifyRange(ctx.zOff,
                               std::vector<uint8_t>(m_vram + ctx.zOff, m_vram + ctx.zOff + ctx.zLen),
                               ctx.tag);
}

namespace
{
    double gs2UnpackDouble(uint32_t lo, uint32_t hi)
    {
        const uint64_t bits = uint64_t(lo) | (uint64_t(hi) << 32);
        double d = 0.0;
        std::memcpy(&d, &bits, sizeof(d));
        return d;
    }

    uint64_t gs2StateBytesHash(const GsGpuState &st)
    {
        // GsGpuState is a POD of 4-byte scalars with no padding, so its bytes ARE its value.
        // cont.233: FNV-1a over WORDS, not bytes -- the byte loop was 48% of the host submit
        // thread in the hardware-raster bench (gdb profile hw428sub2), run once per primitive.
        static_assert(sizeof(GsGpuState) % 4u == 0u, "GsGpuState must be whole words");
        const uint32_t *w = reinterpret_cast<const uint32_t *>(&st);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(GsGpuState) / 4u; ++i)
            h = (h ^ w[i]) * 1099511628211ull;
        return h;
    }
}

// Append one primitive to the current batch, flushing first if it cannot join.
void GSCpuBackend::gpuBatchAppend(const GSPrimitiveBatch &batch)
{
    GsGpuState stt;
    GsGpuGeom g;
    RasterVerifyCtx unused;
    if (!rasterVerifyBegin(batch, unused, &stt, &g))
    {
        // The oracle still draws this primitive, so the batch is no longer a faithful
        // whole -- unless it draws nothing at all (a degenerate triangle or a fully
        // off-screen sprite), which is the only rejection that costs nothing.
        if (g_rvStats[kRvDegenerate] != m_lastDegen || g_rvStats[kRvOffscreen] != m_lastOffscr)
        {
            m_lastDegen = g_rvStats[kRvDegenerate];
            m_lastOffscr = g_rvStats[kRvOffscreen];
            return;
        }
        ++m_gpuBatch.skipped;
        return;
    }

    // --- read-after-write hazard check (see GpuBatchAccum::writtenPages) ---
    auto pageSpan = [&](uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0,
                        uint32_t x1, uint32_t y1, uint32_t &p0, uint32_t &p1) -> bool
    {
        uint32_t off = 0, len = 0;
        if (!gs2PageRange(psm, bp, bw, x0, y0, x1, y1, m_vramSize, off, len))
            return false;
        p0 = off / 8192u;
        p1 = (off + len - 1u) / 8192u;
        return true;
    };
    auto anyMarked = [&](uint32_t p0, uint32_t p1) -> bool
    {
        for (uint32_t q = p0; q <= p1 && q < 512u; ++q)
            if (m_gpuBatch.writtenPages[q >> 6] & (1ull << (q & 63)))
                return true;
        return false;
    };
    auto mark = [&](uint32_t p0, uint32_t p1)
    {
        for (uint32_t q = p0; q <= p1 && q < 512u; ++q)
            m_gpuBatch.writtenPages[q >> 6] |= (1ull << (q & 63));
    };
    bool hazard = false;
    if (!m_gpuBatch.prims.empty() && stt.tme != 0u)
    {
        uint32_t p0 = 0, p1 = 0;
        if (!pageSpan(stt.tpsm, stt.tbp, stt.tbw, 0u, 0u,
                      stt.texW ? stt.texW - 1u : 0u, stt.texH ? stt.texH - 1u : 0u, p0, p1))
            hazard = true; // unknown extent: be conservative
        else if (anyMarked(p0, p1))
            hazard = true;
        const bool paletted = stt.tpsm == GS_PSM_T4 || stt.tpsm == GS_PSM_T8 ||
                              stt.tpsm == GS_PSM_T4HL || stt.tpsm == GS_PSM_T4HH ||
                              stt.tpsm == GS_PSM_T8H;
        if (!hazard && paletted)
        {
            if (!pageSpan(stt.cpsm, stt.cbp, 1u, 0u, 0u, 15u, 15u, p0, p1))
                hazard = true;
            else if (anyMarked(p0, p1))
                hazard = true;
        }
    }
    if (hazard)
    {
        ++g_bvFlushHazard;
        gpuBatchFlush();
    }

    const uint64_t targetKey = uint64_t(stt.fbp) | (uint64_t(stt.fbw) << 20) |
                               (uint64_t(stt.fpsm) << 26) | (uint64_t(stt.zbp) << 32) |
                               (uint64_t(stt.zpsm) << 52);
    GpuBatchAccum &acc = m_gpuBatch;
    if (!acc.prims.empty())
    {
        if (targetKey != acc.targetKey)                { ++g_bvFlushTarget;   gpuBatchFlush(); }
        else if (g_vramMutationSeq != acc.mutSeq)      { ++g_bvFlushMutation; gpuBatchFlush(); }
        else if (acc.prims.size() >= size_t(s_gsGpuBatchMax)) { ++g_bvFlushCap; gpuBatchFlush(); }
    }
    if (acc.prims.empty())
    {
        acc.targetKey = targetKey;
        acc.mutSeq = g_vramMutationSeq;
        acc.maxX = acc.maxY = 0;
        acc.skipped = 0;
        acc.writtenPages.fill(0ull);
        acc.srcPrims.clear();
        // Seed the WHOLE mirror from CPU VRAM at batch start when this batch will be
        // verified: only some batches are, so the mirror otherwise lacks every earlier
        // draw. After this the mirror == CPU VRAM, and the batch is the only difference.
        static unsigned long s_bvN = 0;
        acc.verifying = s_gsGpuBatchVerify > 0 &&
                        ((++s_bvN % static_cast<unsigned long>(s_gsGpuBatchVerify)) == 0u);
        if (acc.verifying)
        {
            if (GsGpuPresentDevice *dev = gs2GpuDevice())
            {
                mirrorFlushPokes();
                acc.preVram.assign(m_vram, m_vram + m_vramSize);
                dev->MirrorPatchRaw(0u, std::vector<uint8_t>(acc.preVram));
            }
        }
    }

    // Dedup the state (measured: mean 3.9 distinct states per run, max 61 -- a linear scan
    // over <=128 entries is cheaper than a hash map at this size). cont.233: consecutive
    // primitives almost always share a state, so compare against the LAST one first and hash
    // only on a miss.
    uint32_t idx = 0xFFFFFFFFu;
    if (acc.lastStateIdx < acc.states.size() &&
        std::memcmp(&acc.states[acc.lastStateIdx], &stt, sizeof(GsGpuState)) == 0)
        idx = acc.lastStateIdx;
    uint64_t hh = 0;
    if (idx == 0xFFFFFFFFu)
    {
        hh = gs2StateBytesHash(stt);
        for (size_t i = 0; i < acc.stateHashes.size(); ++i)
            if (acc.stateHashes[i] == hh)
            {
                idx = static_cast<uint32_t>(i);
                break;
            }
    }
    if (idx == 0xFFFFFFFFu)
    {
        if (acc.states.size() >= 128u)
        {
            ++g_bvFlushStates;
            gpuBatchFlush();
            return gpuBatchAppend(batch); // re-append into the fresh batch
        }
        idx = static_cast<uint32_t>(acc.states.size());
        acc.states.push_back(stt);
        acc.stateHashes.push_back(hh);
    }
    g.stateIndex = idx;
    if (acc.prims.empty() || idx != acc.lastStateIdx)
        ++g_bvStateRuns;
    acc.lastStateIdx = idx;
    if (acc.verifying)
        acc.srcPrims.push_back(batch);
    {
        // Record what this primitive writes, for the next primitive's hazard check.
        uint32_t p0 = 0, p1 = 0;
        if (pageSpan(stt.fpsm, stt.fbp, stt.fbw, g.minX, g.minY, g.maxX, g.maxY, p0, p1))
            mark(p0, p1);
        if (stt.zmsk == 0u &&
            pageSpan(stt.zpsm, stt.zbp, stt.fbw, g.minX, g.minY, g.maxX, g.maxY, p0, p1))
            mark(p0, p1);
    }
    acc.maxX = std::max(acc.maxX, g.maxX);
    acc.maxY = std::max(acc.maxY, g.maxY);
    acc.prims.push_back(std::move(g));
}

// Bin the accumulated primitives into 16x16 screen tiles and hand the batch to the device.
// Binning is done HERE, on the host, for two reasons: it is trivially cheap at these sizes,
// and it is naturally in SUBMISSION ORDER -- which a GPU atomic append would not be, and
// per-pixel submission order is the entire correctness requirement.
void GSCpuBackend::gpuBatchFlush()
{
    GpuBatchAccum &acc = m_gpuBatch;
    if (acc.prims.empty())
    {
        acc.states.clear();
        acc.stateHashes.clear();
        return;
    }
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (dev)
    {
        GsGpuBatch b;
        b.states = std::move(acc.states);
        b.prims = std::move(acc.prims);
        b.tilesX = (acc.maxX >> 4) + 1u;
        b.tilesY = (acc.maxY >> 4) + 1u;
        if (s_gsHwCensus)
        {
            const uint32_t tw = (acc.maxX >> 3) + 1u, th = (acc.maxY >> 3) + 1u;
            std::vector<uint32_t> layer(size_t(tw) * th, 0u);
            uint32_t maxLayer = 0;
            for (const GsGpuGeom &g : b.prims)
            {
                const uint32_t tx0 = g.minX >> 3, ty0 = g.minY >> 3;
                const uint32_t tx1 = std::min<uint32_t>(g.maxX >> 3, tw - 1u);
                const uint32_t ty1 = std::min<uint32_t>(g.maxY >> 3, th - 1u);
                uint32_t L = 0;
                for (uint32_t ty = ty0; ty <= ty1; ++ty)
                    for (uint32_t tx = tx0; tx <= tx1; ++tx)
                        L = std::max(L, layer[size_t(ty) * tw + tx]);
                ++L;
                for (uint32_t ty = ty0; ty <= ty1; ++ty)
                    for (uint32_t tx = tx0; tx <= tx1; ++tx)
                        layer[size_t(ty) * tw + tx] = L;
                maxLayer = std::max(maxLayer, L);
            }
            g_bvLayerSum += maxLayer;
            g_bvLayerMax = std::max<unsigned long>(g_bvLayerMax, maxLayer);
        }
        // cont.233: the hardware path needs no bins (the GPU rasterizes); the device bins
        // itself if it has to fall back to the tile kernel.
        const bool hostBins = !dev->HardwareRasterRequested();
        const size_t tiles = hostBins ? size_t(b.tilesX) * b.tilesY : 0u;
        std::vector<uint32_t> counts(tiles + 1u, 0u);
        auto tileSpan = [&](const GsGpuGeom &g, uint32_t &tx0, uint32_t &ty0, uint32_t &tx1, uint32_t &ty1)
        {
            tx0 = g.minX >> 4; ty0 = g.minY >> 4;
            tx1 = std::min<uint32_t>(g.maxX >> 4, b.tilesX - 1u);
            ty1 = std::min<uint32_t>(g.maxY >> 4, b.tilesY - 1u);
        };
        if (hostBins)
        {
            for (const GsGpuGeom &g : b.prims)
            {
                uint32_t tx0, ty0, tx1, ty1;
                tileSpan(g, tx0, ty0, tx1, ty1);
                for (uint32_t ty = ty0; ty <= ty1; ++ty)
                    for (uint32_t tx = tx0; tx <= tx1; ++tx)
                        ++counts[size_t(ty) * b.tilesX + tx];
            }
            b.binOffsets.resize(tiles + 1u);
            uint32_t running = 0;
            for (size_t t = 0; t < tiles; ++t)
            {
                b.binOffsets[t] = running;
                if (counts[t] != 0u)
                    b.tileList.push_back(static_cast<uint32_t>(t));
                running += counts[t];
            }
            b.binOffsets[tiles] = running;
            b.binEntries.resize(running);
            std::vector<uint32_t> cursor(b.binOffsets.begin(), b.binOffsets.end() - 1);
            for (size_t i = 0; i < b.prims.size(); ++i)
            {
                uint32_t tx0, ty0, tx1, ty1;
                tileSpan(b.prims[i], tx0, ty0, tx1, ty1);
                for (uint32_t ty = ty0; ty <= ty1; ++ty)
                    for (uint32_t tx = tx0; tx <= tx1; ++tx)
                        b.binEntries[cursor[size_t(ty) * b.tilesX + tx]++] = static_cast<uint32_t>(i);
            }
        }

        ++g_bvBatches;
        // Perf mode has no verify path to hang the report off, so print periodically here.
        if (s_gsGpuBatchPerf && (g_bvBatches % 2000u) == 0u)
            gs2PrintMirrorPatch();
        g_bvPrims += b.prims.size();
        g_bvBinEntries += b.binEntries.size();
        g_bvMaxPrims = std::max<unsigned long>(g_bvMaxPrims, (unsigned long)b.prims.size());
        g_bvMaxStates = std::max<unsigned long>(g_bvMaxStates, (unsigned long)b.states.size());
        // Only verify a batch the GPU rendered in full (see GpuBatchAccum::skipped).
        // Descriptor snapshot for the log line below (b is moved into the device call).
        const size_t verifyStates = b.states.size();
        const uint32_t verifyTilesX = b.tilesX, verifyTilesY = b.tilesY;
        const uint32_t verifyFbp = b.states[0].fbp, verifyFbw = b.states[0].fbw;
        const uint32_t verifyFpsm = b.states[0].fpsm, verifyZbp = b.states[0].zbp;
        unsigned long verifyOverwide = 0;
        uint32_t verifyMaxScisX = 0;
        for (const GsGpuState &st : b.states)
            verifyMaxScisX = std::max(verifyMaxScisX, st.scissorX1);
        for (const GsGpuGeom &g : b.prims)
            if (g.maxX >= b.states[g.stateIndex].fbw * 64u)
                ++verifyOverwide;
        const bool verify = acc.verifying && acc.skipped == 0u;
        // Keep copies for the replay A/B (b is moved into the device call below).
        std::vector<GsGpuState> replayStates;
        std::vector<GsGpuGeom> replayPrims;
        std::vector<GSPrimitiveBatch> replaySrc;
        if (verify)
        {
            replayStates = b.states;
            replayPrims = b.prims;
            replaySrc = acc.srcPrims;
        }
        if (acc.verifying && acc.skipped != 0u)
            ++g_bvSkippedBatches;
        const size_t verifyPrims = b.prims.size();
        b.vram = m_vram; b.vramSize = m_vramSize; // cont.250 PS2X_GS_LINTEX decode source
        dev->RasterBatch(std::move(b));
        if (verify)
        {
            // Whole-VRAM compare: the mirror was seeded at batch start and the batch is the
            // only thing that touched it since, so it must now equal CPU VRAM exactly --
            // no draw-page mask, nothing excused.
            ++g_bvVerified;
            g_bvVerifiedPrims += verifyPrims;
            // Log the batch right before its compare, so a failing [gsgpu:mverify] line is
            // always preceded by the descriptor of the batch that produced it. `overwide`
            // is the tile-disjointness check: screen tiles only map to disjoint VRAM while
            // x stays inside the buffer (fbw*64) -- past that, PageId wraps a pixel into
            // the next page row, where a DIFFERENT tile can alias the same word.
            std::fprintf(stderr,
                         "[gsgpu:bverify] #%lu prims=%zu states=%zu tiles=%ux%u bbox=(%u,%u) "
                         "target{fbp=%u fbw=%u fpsm=0x%x zbp=%u} overwide=%lu maxScisX=%u\n",
                         g_bvVerified, verifyPrims, verifyStates, verifyTilesX, verifyTilesY,
                         acc.maxX, acc.maxY, verifyFbp, verifyFbw, verifyFpsm, verifyZbp,
                         verifyOverwide, verifyMaxScisX);
            g_bvVerifiedMaxPrims = std::max<unsigned long>(g_bvVerifiedMaxPrims,
                                                           (unsigned long)verifyPrims);
            const bool tiledOk = dev->MirrorVerifyFull(m_vram, m_vramSize, nullptr);
            if (!tiledOk)
            {
                // A/B: replay the SAME primitives, in order, one batch of one each -- the
                // single-primitive entry point that cont.169/170 proved bit-exact. If this
                // passes, the divergence is in tiling/ordering; if it fails too, some
                // primitive in here shades wrong and the per-primitive sampling missed it.
                ++g_bvReplays;
                dev->MirrorPatchRaw(0u, std::vector<uint8_t>(acc.preVram));
                for (const GsGpuGeom &one : replayPrims)
                {
                    GsGpuBatch sb;
                    sb.states.push_back(replayStates[one.stateIndex < replayStates.size()
                                                         ? one.stateIndex : 0]);
                    GsGpuGeom g1 = one;
                    g1.stateIndex = 0u;
                    sb.prims.push_back(g1);
                    sb.vram = m_vram; sb.vramSize = m_vramSize; // cont.250 PS2X_GS_LINTEX decode source
                    dev->RasterBatch(std::move(sb)); // tileList empty => per-prim path
                }
                const bool serialOk = dev->MirrorVerifyFull(m_vram, m_vramSize, nullptr);
                if (serialOk)
                    ++g_bvReplayTilingBug;
                else
                {
                    ++g_bvReplayPrimBug;
                    // The bad primitive's state is one of these. The whole game is only 49
                    // draw-state combos, so dumping the batch's (<=31) states is enough to
                    // spot the one the shader gets wrong.
                    if (g_bvReplayPrimBug <= 2u)
                        for (size_t si = 0; si < replayStates.size(); ++si)
                        {
                            const GsGpuState &q = replayStates[si];
                            unsigned long uses = 0;
                            for (const GsGpuGeom &gg : replayPrims)
                                if (gg.stateIndex == si)
                                    ++uses;
                            std::fprintf(stderr,
                                         "[gsgpu:bstate] s%zu uses=%lu prim=%u iip=%u tme=%u fge=%u "
                                         "fst=%u abe=%u | f{psm=0x%x msk=%08x} z{psm=0x%x msk=%u} "
                                         "test{ate=%u atst=%u aref=%u afail=%u ztst=%u} "
                                         "blend{%u%u%u%u fix=%u} tex{psm=0x%x tw=%u th=%u lin=%u "
                                         "tfx=%u tcc=%u cpsm=0x%x csm=%u csa=%u wms=%u wmt=%u "
                                         "minU=%u maxU=%u minV=%u maxV=%u} texa{%u,%u,aem=%u} "
                                         "fba=%u shadow=%u\n",
                                         si, uses, q.primType, q.iip, q.tme, q.fge, q.fst, q.abe,
                                         q.fpsm, q.fbmsk, q.zpsm, q.zmsk,
                                         q.ate, q.atst, q.aref, q.afail, q.ztst,
                                         q.blendA, q.blendB, q.blendC, q.blendD, q.blendFix,
                                         q.tpsm, q.texW, q.texH, q.linear, q.tfx, q.tcc,
                                         q.cpsm, q.csm, q.csa, q.wms, q.wmt,
                                         q.minU, q.maxU, q.minV, q.maxV,
                                         q.ta0, q.ta1, q.aem, q.fba, q.clutShadowHave);
                        }
                }
                std::fprintf(stderr,
                             "[gsgpu:bverify]   REPLAY verdict: serial=%s => %s\n",
                             serialOk ? "OK" : "FAIL",
                             serialOk ? "TILING/ORDERING" : "PER-PRIMITIVE");
                if (!serialOk && replaySrc.size() == replayPrims.size() &&
                    g_bvReplayPrimBug <= 2u)
                {
                    m_gpuBatch.srcPrims = replaySrc; // gpuBisect reads from the accumulator
                    gpuBisectFailingBatch(replayStates, replayPrims);
                }
            }
        }
    }
    acc.states.clear();
    acc.stateHashes.clear();
    acc.prims.clear();
    acc.preVram.clear();
    acc.srcPrims.clear();
    acc.stateHashes.shrink_to_fit();
    acc.verifying = false;
}

// Binary-search a failing batch for the ONE primitive whose GPU shading differs from the
// oracle. For a prefix length k: re-run the oracle over prims [0..k) into a scratch VRAM
// starting from the batch's pre-draw bytes, replay the same prefix on the GPU from the same
// starting point, and compare. The smallest failing k names primitive k-1.
void GSCpuBackend::gpuBisectFailingBatch(const std::vector<GsGpuState> &states,
                                         const std::vector<GsGpuGeom> &prims)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    const GpuBatchAccum &acc = m_gpuBatch;
    if (!dev || acc.preVram.size() != m_vramSize || acc.srcPrims.size() != prims.size())
        return;

    static uint32_t s_lastFirstWord = 0, s_lastStaleWords = 0;
    auto prefixMatches = [&](size_t k) -> bool
    {
        // Oracle: scratch = preVram, then draw the first k primitives into it.
        m_bisectScratch = acc.preVram;
        uint8_t *saved = m_vram;
        m_vram = m_bisectScratch.data();
        m_rawDraw = true;
        for (size_t i = 0; i < k; ++i)
            DrawPrimitive(acc.srcPrims[i]);
        m_rawDraw = false;
        m_vram = saved;

        // GPU: mirror = preVram, then replay the same k primitives one at a time.
        dev->MirrorPatchRaw(0u, std::vector<uint8_t>(acc.preVram));
        for (size_t i = 0; i < k; ++i)
        {
            GsGpuBatch sb;
            sb.states.push_back(states[prims[i].stateIndex < states.size()
                                           ? prims[i].stateIndex : 0]);
            GsGpuGeom g1 = prims[i];
            g1.stateIndex = 0u;
            sb.prims.push_back(g1);
            sb.vram = m_vram; sb.vramSize = m_vramSize; // cont.250 PS2X_GS_LINTEX decode source
            dev->RasterBatch(std::move(sb));
        }
        return dev->MirrorVerifyFull(m_bisectScratch.data(), m_vramSize, nullptr, /*quiet*/ true,
                                     &s_lastFirstWord, &s_lastStaleWords);
    };

    if (prefixMatches(prims.size()))
    {
        std::fprintf(stderr, "[gsgpu:bisect] full prefix MATCHES -- not reproducible here\n");
        return;
    }
    size_t lo = 0, hi = prims.size(); // invariant: [0..lo) matches, [0..hi) does not
    while (hi - lo > 1)
    {
        const size_t mid = lo + (hi - lo) / 2;
        if (prefixMatches(mid))
            lo = mid;
        else
            hi = mid;
    }
    const size_t bad = hi - 1;
    const GsGpuGeom &g = prims[bad];
    const GsGpuState &q = states[g.stateIndex < states.size() ? g.stateIndex : 0];
    std::fprintf(stderr,
                 "[gsgpu:bisect] CULPRIT prim %zu of %zu | type=%u rect=(%u,%u)-(%u,%u) "
                 "winding=%.1f invAbsDenom=%.9g\n"
                 "[gsgpu:bisect]   v0=(%.6f,%.6f) v1=(%.6f,%.6f) v2=(%.6f,%.6f)\n"
                 "[gsgpu:bisect]   q=(%.9g,%.9g,%.9g) s=(%.9g,%.9g,%.9g) t=(%.9g,%.9g,%.9g)\n"
                 "[gsgpu:bisect]   z=(%.1f,%.1f,%.1f) rgba=(%08x,%08x,%08x) fog=(%u,%u,%u)\n"
                 "[gsgpu:bisect]   state: f{psm=0x%x fbp=%u fbw=%u msk=%08x} z{psm=0x%x bp=%u msk=%u} "
                 "test{ate=%u atst=%u aref=%u afail=%u ztst=%u} blend{%u%u%u%u fix=%u abe=%u} "
                 "tex{psm=0x%x tbp=%u tbw=%u tw=%u th=%u lin=%u tfx=%u tcc=%u cbp=%u cpsm=0x%x "
                 "csa=%u wms=%u wmt=%u} texa{%u,%u,aem=%u} iip=%u fge=%u fst=%u\n",
                 bad, prims.size(), q.primType, g.minX, g.minY, g.maxX, g.maxY,
                 g.winding, g.invAbsDenom,
                 g.x0, g.y0, g.x1, g.y1, g.x2, g.y2,
                 g.q0, g.q1, g.q2, g.s0, g.s1, g.s2, g.t0, g.t1, g.t2,
                 gs2UnpackDouble(g.z0lo, g.z0hi), gs2UnpackDouble(g.z1lo, g.z1hi),
                 gs2UnpackDouble(g.z2lo, g.z2hi),
                 g.rgba0, g.rgba1, g.rgba2, g.fog0, g.fog1, g.fog2,
                 q.fpsm, q.fbp, q.fbw, q.fbmsk, q.zpsm, q.zbp, q.zmsk,
                 q.ate, q.atst, q.aref, q.afail, q.ztst,
                 q.blendA, q.blendB, q.blendC, q.blendD, q.blendFix, q.abe,
                 q.tpsm, q.tbp, q.tbw, q.texW, q.texH, q.linear, q.tfx, q.tcc,
                 q.cbp, q.cpsm, q.csa, q.wms, q.wmt, q.ta0, q.ta1, q.aem,
                 q.iip, q.fge, q.fst);

    // ★ WHERE does the prefix actually diverge? cont.172 assumed it was inside the
    // culprit's rect and hit a contradiction (culprit-alone from a matching state showed no
    // difference). Print the failing word and decide the question instead of assuming it.
    {
        (void)prefixMatches(bad + 1u);
        const uint32_t w = s_lastFirstWord, stale = s_lastStaleWords;
        const uint32_t byteOff = w * 4u;
        // Is that word inside the culprit's frame rect, or its Z rect, or neither?
        uint32_t fOff = 0, fLen = 0, zOff = 0, zLen = 0;
        const bool haveF = gs2PageRange(q.fpsm, q.fbp, q.fbw, g.minX, g.minY, g.maxX, g.maxY,
                                        m_vramSize, fOff, fLen);
        const bool haveZ = gs2PageRange(q.zpsm, q.zbp, q.fbw, g.minX, g.minY, g.maxX, g.maxY,
                                        m_vramSize, zOff, zLen);
        const bool inF = haveF && byteOff >= fOff && byteOff < fOff + fLen;
        const bool inZ = haveZ && byteOff >= zOff && byteOff < zOff + zLen;
        // How many of the stale words are accounted for by the culprit's own rect? Compare
        // the two buffers over the rect directly -- if the whole-4MB stale count is much
        // larger than what the rect explains, the bisect's k is naming the primitive that
        // TRIGGERS the divergence, not the one that contains it.
        // Capture the BEFORE state too (prefix [0..bad), which matched exactly), so each
        // failing pixel can be read three ways. If gpu-after == before, the GPU REJECTED
        // the pixel (alpha or Z test) where the CPU accepted it; if it differs from both,
        // the GPU accepted it and shaded it differently. That single distinction picks the
        // next move, and nothing so far has actually established which it is.
        (void)prefixMatches(bad);
        const std::vector<uint8_t> beforeCpu = m_bisectScratch;
        std::vector<uint8_t> beforeGpu;
        dev->MirrorReadback(0u, m_vramSize, beforeGpu);
        (void)prefixMatches(bad + 1u);
        std::vector<uint8_t> gpuNow;
        dev->MirrorReadback(0u, m_vramSize, gpuNow);
        if (beforeCpu.size() == m_vramSize && beforeGpu.size() == m_vramSize &&
            gpuNow.size() == m_vramSize)
        {
            unsigned shown3 = 0;
            for (uint32_t y = g.minY; y <= g.maxY && shown3 < 4u; ++y)
                for (uint32_t x = g.minX; x <= g.maxX && shown3 < 4u; ++x)
                {
                    const uint32_t cB = GSMem::ReadCT32(const_cast<uint8_t *>(beforeCpu.data()), q.fbp, q.fbw, x, y);
                    const uint32_t gB = GSMem::ReadCT32(beforeGpu.data(), q.fbp, q.fbw, x, y);
                    const uint32_t cA = GSMem::ReadCT32(m_bisectScratch.data(), q.fbp, q.fbw, x, y);
                    const uint32_t gA = GSMem::ReadCT32(gpuNow.data(), q.fbp, q.fbw, x, y);
                    const uint32_t czB = GSMem::ReadZ24(const_cast<uint8_t *>(beforeCpu.data()), q.zbp, q.fbw, x, y);
                    const uint32_t gzB = GSMem::ReadZ24(beforeGpu.data(), q.zbp, q.fbw, x, y);
                    const uint32_t czA = GSMem::ReadZ24(m_bisectScratch.data(), q.zbp, q.fbw, x, y);
                    const uint32_t gzA = GSMem::ReadZ24(gpuNow.data(), q.zbp, q.fbw, x, y);
                    if (cA == gA && czA == gzA)
                        continue;
                    ++shown3;
                    std::fprintf(stderr,
                                 "[gsgpu:bisect]   (%u,%u) rgb before c=%08x g=%08x -> after c=%08x g=%08x "
                                 "%s | z before c=%06x g=%06x -> after c=%06x g=%06x %s\n",
                                 x, y, cB, gB, cA, gA,
                                 (gA == gB) ? "[GPU REJECTED]" : "[gpu wrote]",
                                 czB, gzB, czA, gzA,
                                 (gzA == gzB) ? "[GPU REJECTED]" : "[gpu wrote]");
                }
        }
        unsigned rectDiff = 0;
        if (gpuNow.size() == m_vramSize)
            for (uint32_t y = g.minY; y <= g.maxY; ++y)
                for (uint32_t x = g.minX; x <= g.maxX; ++x)
                {
                    if (GSMem::ReadCT32(m_bisectScratch.data(), q.fbp, q.fbw, x, y) !=
                        GSMem::ReadCT32(gpuNow.data(), q.fbp, q.fbw, x, y))
                        ++rectDiff;
                    if (GSMem::ReadZ24(m_bisectScratch.data(), q.zbp, q.fbw, x, y) !=
                        GSMem::ReadZ24(gpuNow.data(), q.zbp, q.fbw, x, y))
                        ++rectDiff;
                }
        std::fprintf(stderr,
                     "[gsgpu:bisect]   WHERE: first stale word=%u (byte 0x%06x, page %u), "
                     "stale-words=%u | culprit frame[0x%06x+0x%x] %s, z[0x%06x+0x%x] %s | "
                     "culprit rect explains %u word(s)\n",
                     w, byteOff, byteOff / 8192u, stale,
                     fOff, fLen, inF ? "INSIDE" : "outside",
                     zOff, zLen, inZ ? "INSIDE" : "outside", rectDiff);
    }

    // DETERMINISM FIRST. The coverage probe below re-runs the same prefix; if the GPU
    // answer is not stable run-to-run, every "which primitive" conclusion is meaningless
    // and the real bug is a race, not arithmetic. Cheap to ask, so ask before theorising.
    {
        char rk[8] = {}, rk1[8] = {};
        for (int t = 0; t < 5; ++t)
        {
            rk[t] = prefixMatches(bad) ? 'M' : 'x';
            rk1[t] = prefixMatches(bad + 1u) ? 'M' : 'x';
        }
        std::fprintf(stderr,
                     "[gsgpu:bisect]   determinism: prefix[0..%zu)=%s prefix[0..%zu)=%s "
                     "(M=match, x=differ; unstable => a RACE, not arithmetic)\n",
                     bad, rk, bad + 1u, rk1);
    }

    // Coverage masks: which pixels does each side actually WRITE for the culprit alone?
    // Run the prefix without it, snapshot both, then apply just the culprit and diff. This
    // separates "shades differently" from "covers differently" with no shader changes.
    {
        (void)prefixMatches(bad); // leaves scratch = CPU[0..bad), mirror = GPU[0..bad)
        const std::vector<uint8_t> cpuBefore = m_bisectScratch;
        std::vector<uint8_t> gpuBefore;
        dev->MirrorReadback(0u, m_vramSize, gpuBefore);

        m_rawDraw = true;
        uint8_t *saved = m_vram;
        m_vram = m_bisectScratch.data();
        DrawPrimitive(acc.srcPrims[bad]);
        m_vram = saved;
        m_rawDraw = false;

        GsGpuBatch sb;
        sb.states.push_back(q);
        GsGpuGeom g1 = g;
        g1.stateIndex = 0u;
        sb.prims.push_back(g1);
        sb.vram = m_vram; sb.vramSize = m_vramSize; // cont.250 PS2X_GS_LINTEX decode source
        dev->RasterBatch(std::move(sb));
        std::vector<uint8_t> gpuAfter;
        dev->MirrorReadback(0u, m_vramSize, gpuAfter);

        if (cpuBefore.size() == m_vramSize && gpuBefore.size() == m_vramSize &&
            gpuAfter.size() == m_vramSize)
        {
            unsigned cpuN = 0, gpuN = 0;
            std::fprintf(stderr, "[gsgpu:bisect]   coverage (C=cpu only, G=gpu only, "
                                 "*=both, .=neither):\n");
            for (uint32_t y = g.minY; y <= g.maxY; ++y)
            {
                char row[80];
                unsigned n = 0;
                for (uint32_t x = g.minX; x <= g.maxX && n < 72u; ++x)
                {
                    const bool cw =
                        GSMem::ReadCT32(m_bisectScratch.data(), q.fbp, q.fbw, x, y) !=
                            GSMem::ReadCT32(const_cast<uint8_t *>(cpuBefore.data()), q.fbp, q.fbw, x, y) ||
                        GSMem::ReadZ24(m_bisectScratch.data(), q.zbp, q.fbw, x, y) !=
                            GSMem::ReadZ24(const_cast<uint8_t *>(cpuBefore.data()), q.zbp, q.fbw, x, y);
                    const bool gw =
                        GSMem::ReadCT32(gpuAfter.data(), q.fbp, q.fbw, x, y) !=
                            GSMem::ReadCT32(gpuBefore.data(), q.fbp, q.fbw, x, y) ||
                        GSMem::ReadZ24(gpuAfter.data(), q.zbp, q.fbw, x, y) !=
                            GSMem::ReadZ24(gpuBefore.data(), q.zbp, q.fbw, x, y);
                    cpuN += cw ? 1u : 0u;
                    gpuN += gw ? 1u : 0u;
                    row[n++] = (cw && gw) ? '*' : (cw ? 'C' : (gw ? 'G' : '.'));
                }
                row[n] = 0;
                std::fprintf(stderr, "[gsgpu:bisect]     y=%3u %s\n", y, row);
            }
            std::fprintf(stderr, "[gsgpu:bisect]   cpu wrote %u px, gpu wrote %u px\n", cpuN, gpuN);
        }
    }

    // With the culprit isolated, compare the ACTUAL pixels: re-run both sides over the
    // prefix ending at the culprit, read the mirror back, and diff the primitive's bbox.
    // A whole-pixel difference points at a decision (alpha/Z test) flipping; a small
    // channel difference points at arithmetic.
    (void)prefixMatches(bad + 1u);
    std::vector<uint8_t> gpuVram;
    dev->MirrorReadback(0u, m_vramSize, gpuVram);
    if (gpuVram.size() != m_vramSize)
        return;
    unsigned shown = 0, differing = 0;
    for (uint32_t y = g.minY; y <= g.maxY; ++y)
        for (uint32_t x = g.minX; x <= g.maxX; ++x)
        {
            const uint32_t c = GSMem::ReadCT32(m_bisectScratch.data(), q.fbp, q.fbw, x, y);
            const uint32_t v = GSMem::ReadCT32(gpuVram.data(), q.fbp, q.fbw, x, y);
            const uint32_t cz = GSMem::ReadZ24(m_bisectScratch.data(), q.zbp, q.fbw, x, y);
            const uint32_t vz = GSMem::ReadZ24(gpuVram.data(), q.zbp, q.fbw, x, y);
            if (c == v && cz == vz)
                continue;
            ++differing;
            if (shown < 8u)
            {
                ++shown;
                // Recompute the barycentric weights and the interpolated vertex alpha two
                // ways: exactly as the oracle does, and with the multiply-adds CONTRACTED
                // into fma(). GLSL is allowed to contract; the oracle (built -msse4.1) is
                // not. If int(alpha) differs between the two, that contraction is the whole
                // divergence -- and near ATST's zero crossing it flips a decision, not a LSB.
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                const float w0 = (((g.y1 - g.y2) * (px - g.x2) + (g.x2 - g.x1) * (py - g.y2)) *
                                  g.winding) * g.invAbsDenom;
                const float w1 = (((g.y2 - g.y0) * (px - g.x2) + (g.x0 - g.x2) * (py - g.y2)) *
                                  g.winding) * g.invAbsDenom;
                const float w2 = 1.0f - w0 - w1;
                const float w0f = (std::fma(g.y1 - g.y2, px - g.x2,
                                            (g.x2 - g.x1) * (py - g.y2)) * g.winding) * g.invAbsDenom;
                const float w1f = (std::fma(g.y2 - g.y0, px - g.x2,
                                            (g.x0 - g.x2) * (py - g.y2)) * g.winding) * g.invAbsDenom;
                const float w2f = 1.0f - w0f - w1f;
                const float a0 = static_cast<float>((g.rgba0 >> 24) & 0xFFu);
                const float a1 = static_cast<float>((g.rgba1 >> 24) & 0xFFu);
                const float a2 = static_cast<float>((g.rgba2 >> 24) & 0xFFu);
                const float ia = a0 * w0 + a1 * w1 + a2 * w2;
                const float iaf = std::fma(a0, w0f, std::fma(a1, w1f, a2 * w2f));
                std::fprintf(stderr,
                             "[gsgpu:bisect]   pixel (%u,%u) cpu=%08x gpu=%08x | z cpu=%06x gpu=%06x\n"
                             "[gsgpu:bisect]     w=(%.9g,%.9g,%.9g) alpha=%.9g -> %d | "
                             "fma w=(%.9g,%.9g,%.9g) alpha=%.9g -> %d%s\n",
                             x, y, c, v, cz, vz,
                             w0, w1, w2, ia, static_cast<int>(ia),
                             w0f, w1f, w2f, iaf, static_cast<int>(iaf),
                             (static_cast<int>(ia) != static_cast<int>(iaf)) ? "  <== DIFFERS" : "");
            }
        }
    std::fprintf(stderr, "[gsgpu:bisect]   %u of %u bbox pixels differ\n", differing,
                 (g.maxX - g.minX + 1u) * (g.maxY - g.minY + 1u));
}

void GSCpuBackend::markDrawPages(uint32_t psm, uint32_t bp, uint32_t bw,
                                 uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
    if (x1 < x0 || y1 < y0)
        return;
    uint32_t pw = 64u, ph = 32u;
    gs2PageExtent(psm & 0x3Fu, pw, ph);
    const uint32_t stride = (std::max<uint32_t>(bw, 1u) * 64u) / pw;
    const uint64_t basePage = bp / 32u;
    const uint64_t first = basePage + static_cast<uint64_t>(y0 / ph) * stride + (x0 / pw);
    const uint64_t last = basePage + static_cast<uint64_t>(y1 / ph) * stride + (x1 / pw) +
                          gs2PageSpill(bp); // intra-page block spill; see gs2PageSpill
    for (uint64_t p = first; p <= last; ++p)
        m_drawPages[static_cast<size_t>(p % m_drawPages.size())] = 1u; // guest pages wrap at 4MB
}

void GSCpuBackend::mirrorFlushPokeSlot(size_t i)
{
    if (i >= kPokeBoxes || !m_pokeBox[i].used)
        return;
    const PokeBox b = m_pokeBox[i];
    m_pokeBox[i].used = false; // cleared FIRST: mirrorPatchRect(Poke) must not re-enter
    mirrorPatchRect(b.psm, b.bp, b.bw, b.x0, b.y0, b.x1, b.y1, MirrorPatchSource::Poke);
}

void GSCpuBackend::mirrorFlushPokes()
{
    for (size_t i = 0; i < kPokeBoxes; ++i)
        mirrorFlushPokeSlot(i);
}

void GSCpuBackend::mirrorPatchRect(uint32_t psm, uint32_t bp, uint32_t bw,
                                   uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                                   MirrorPatchSource src)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev || !m_vram || m_vramSize == 0u || x1 < x0 || y1 < y0)
        return;
    // Device FIFO order must equal VRAM write order: any other mirror op has to land
    // behind the pokes that preceded it.
    if (src != MirrorPatchSource::Poke)
        mirrorFlushPokes();
    auto &st = g_mpStats[static_cast<int>(src) & 3];
    ++st.n;

    const uint32_t rectW = x1 - x0 + 1u;
    const uint32_t rectH = y1 - y0 + 1u;
    const uint64_t pixels = static_cast<uint64_t>(rectW) * rectH;
    // Above this the page copy is the cheaper transport (a full-screen clear touches
    // ~140 pages either way, and the per-pixel readback is pure overhead).
    constexpr uint64_t kKernelPixelCap = 64u * 1024u;
    if (gs2IsC32Layout(psm & 0x3Fu) && pixels != 0u && pixels <= kKernelPixelCap)
    {
        GsGpuUploadChunk c;
        c.dbp = bp;
        c.dbw = std::max<uint32_t>(bw, 1u);
        c.dsax = x0;
        c.dsay = y0;
        c.rrw = rectW;
        c.startPixel = 0u;
        c.payload.resize(static_cast<size_t>(pixels));
        for (uint64_t i = 0; i < pixels; ++i)
            c.payload[static_cast<size_t>(i)] =
                ReadVramUnlocked(GS_PSM_CT32, bp, c.dbw,
                                 x0 + static_cast<uint32_t>(i % rectW),
                                 y0 + static_cast<uint32_t>(i / rectW));
        st.words += pixels;
        dev->MirrorUpload(std::move(c));
        return;
    }

    uint32_t pw = 64u, ph = 32u;
    gs2PageExtent(psm & 0x3Fu, pw, ph);
    const uint32_t stride = (std::max<uint32_t>(bw, 1u) * 64u) / pw;
    const uint64_t basePage = bp / 32u;
    const uint64_t firstPage = basePage + static_cast<uint64_t>(y0 / ph) * stride + (x0 / pw);
    const uint64_t lastPage = basePage + static_cast<uint64_t>(y1 / ph) * stride + (x1 / pw) +
                              gs2PageSpill(bp);

    const uint64_t start = firstPage * 8192ull;
    const uint64_t end = (lastPage + 1ull) * 8192ull;
    if (start >= end)
        return;
    // Guest addresses wrap at 4MB (PixelStorageTraits<>::Write masks with MEMORY_SIZE-1),
    // so a range running off the top continues at 0. Patch it as two spans.
    auto patch = [&](uint64_t from, uint64_t to)
    {
        if (from >= to)
            return;
        const uint32_t off = static_cast<uint32_t>(from);
        const uint32_t len = static_cast<uint32_t>(std::min<uint64_t>(to, m_vramSize) - from);
        if (off >= m_vramSize || len == 0u)
            return;
        ++st.rawN;
        st.rawBytes += len;
        dev->MirrorPatchRaw(off, std::vector<uint8_t>(m_vram + off, m_vram + off + len));
    };
    if (end <= m_vramSize)
    {
        patch(start, end);
        return;
    }
    if (start >= m_vramSize || (end - start) >= m_vramSize)
    {
        // Wrapped past itself (or started out of range): the whole mirror is the only
        // conservative answer. Rare enough to be worth counting if it ever shows up.
        static unsigned long s_whole = 0;
        if ((++s_whole % 1024u) == 1u)
            std::fprintf(stderr, "[gsgpu:patch] whole-VRAM patch #%lu (psm=0x%x bp=%u bw=%u "
                                 "rect=%u,%u..%u,%u)\n",
                         s_whole, psm, bp, bw, x0, y0, x1, y1);
        patch(0u, m_vramSize);
        return;
    }
    patch(start, m_vramSize);
    patch(0u, end - m_vramSize);
}
#endif

void GSCpuBackend::SnapshotVramNoDrain(std::vector<uint8_t> &out) const
{
    // Presentation-path snapshot: consistent (worker holds m_mutex per item) but not
    // drained — see the Sync(Presentation) comment. The priority flag makes the worker
    // yield before its next item so this lock() wins promptly (bounded by one item).
    m_presentPriority.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(m_mutex);
    m_presentPriority.store(false, std::memory_order_release);
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

// ---- Level-era GS census (PS2X_GS_CENSUS2, default OFF): uncapped AGGREGATE counters printed
// every N events, so the population is measurable in any era (a capped trace exhausts before the
// level — the old rule-11/16 trap). Draw side: total/textured split + per-psm counts of textured
// draws + the last textured-draw descriptor. Upload side (see UploadImage): transfer count/bytes,
// per-dpsm counts, dims-sanity (the cycle-73 malformed discriminator), last descriptor.
static const bool s_gsCensus2 = []
{ const char *e = std::getenv("PS2X_GS_CENSUS2"); return e && e[0] && e[0] != '0'; }();
void ps2xGsNotifyDisplayFlip(uint64_t preFlipDispfb1, uint64_t preFlipDispfb2)
{
    if (!s_gsFlipSnap)
        return;
    if (GSCpuBackend *b = g_gs2ActiveBackend.load(std::memory_order_acquire))
    {
        // cont.317 stage 2: the flip snapshot must land AFTER the packets submitted before it, so
        // under the pipeline thread it is a ring command executed in order by the consumer.
        if (ps2gs::threaded() && ps2gs::producerSide())
            ps2gs::enqueueFlip(b, preFlipDispfb1, preFlipDispfb2);
        else
            b->OnDisplayFlip(preFlipDispfb1, preFlipDispfb2);
    }
}
bool GSCpuBackend::PresentSourceFresh(uint64_t &outSeq) const
{
    outSeq = m_presentCopySeq.load(std::memory_order_acquire);
    const uint64_t tick = m_presentCopyTickMs.load(std::memory_order_acquire);
    // cont.345: in GL mode the copy never goes stale for the presenter (see Present()).
    return tick != 0u && ((s_gsRendererGl && s_glSnapResolve) || gs2NowMs() - tick <= kPresentCopyFreshMs);
}
// ★★★★ cont.332d: the aspect the presenter should draw the frame at, derived from the DISPLAY
// registers of the last present (0 = not known yet, draw it pixel-square as before).
std::atomic<float> g_gsPresentAspect{0.0f};
// ★★★★★ cont.356e: is the CURRENT frame a 2D screen (title/menu art presented straight from the
// framebuffer)? Debounced verdict of the frame-level classifier, false unless PS2X_GS_HUD_ASPECT_FULL
// is on. Two consumers: the presenter pillarboxes such a frame at its natural aspect instead of
// stretching it, and the GL device suppresses the per-draw HUD counter-scale so the same pixels are
// not corrected twice.
bool ps2xGsIs2dScreen()
{
    return g_fs2dActive.load(std::memory_order_relaxed);
}

float ps2xGsPresentAspect()
{
    return g_gsPresentAspect.load(std::memory_order_relaxed);
}

bool ps2xGsTakeHiresPresentFrame(std::vector<uint8_t> &out, uint32_t &w, uint32_t &h)
{
#if PS2X_HAS_GS_GPU_DEVICE
    // Only ask when the GL renderer is the one drawing -- gs2GpuDevice() spins up an EGL context
    // on first call, and the CPU rasterizer has no scene target to read.
    if (!s_gsRendererGl)
        return false;
    if (GsGpuPresentDevice *dev = gs2GpuDevice())
        return dev->TakeHiresPresentFrame(out, w, h);
#else
    (void)out; (void)w; (void)h;
#endif
    return false;
}

bool ps2xGsPresentSourceFresh(uint64_t &outSeq)
{
    outSeq = 0u;
    if (GSCpuBackend *b = g_gs2ActiveBackend.load(std::memory_order_acquire))
        return b->PresentSourceFresh(outSeq);
    return false;
}
namespace
{
    struct Gs2DrawStats
    {
        unsigned long total = 0, tme = 0;
        unsigned long psmT4 = 0, psmT8 = 0, psmCt32 = 0, psmCt16 = 0, psmOther = 0;
        uint32_t lastTbp0 = 0, lastPsm = 0, lastTw = 0, lastTh = 0, lastTfx = 0, lastFbp = 0;
    } g_gs2Draw;

// ★ cont.230 instrument fix: the documented contract is "without PS2X_GS_ERA_FILE the gate is
// always open", but this started false and was only ever opened inside the PS2X_GS_CENSUS2 block
// -- so every era-gated dump (PS2X_GS_PRESENT_SAVE, VRAMDUMP) silently required CENSUS2, which
// forces the band pool off and makes the game too slow to reach the level era at all. Default it
// open unless an era file was actually requested.
static bool g_gs2EraOpen = (std::getenv("PS2X_GS_ERA_FILE") == nullptr);
static unsigned long g_gs2EraOpenDraw = 0;
static bool g_gs2DegenReset = false;

    // ---- Draw-state census (PS2X_GS_STATECENSUS, default OFF; cont.166 GPU-arc phase 0) ----
    // Enumerates the DISTINCT draw-state combinations the game submits -- the coverage list
    // the future GPU raster ubershader must honor (anything absent from it can take a loud
    // fallback path). Key = one packed uint64, bit layout mirroring THIS backend's own
    // register decodes (TEST bits from passesAlphaTest/classifyAlphaTest/DATE/ZTST, CLAMP
    // wms/wmt = bits 0-3, ALPHA A/B/C/D = bits 0-7). Counted per DrawPrimitive on the
    // raster thread (worker-serialized, so no lock); cumulative sorted dump every ~10s.
    // Bit 63 = the PS2X_GS_ERA_FILE era gate -- run with PS2X_GS_CENSUS2=1 as well, the
    // gate only advances inside its block.
    static const bool s_gsStateCensus = []
    { const char *e = std::getenv("PS2X_GS_STATECENSUS"); return e && e[0] && e[0] != '0'; }();
    static std::unordered_map<uint64_t, uint64_t> g_stateCensus;
    static unsigned long g_stateCensusN = 0;
    static uint64_t g_stateCensusLastDumpMs = 0;

    // Per-triangle mip level: LOD from the texel-area / pixel-area ratio, clamped to
    // TEX1.MXL. Extracted from DrawTriangle (cont.169) so the CPU oracle and the phase-2a
    // GPU prim builder resolve the SAME level -- a drift here would show up as texture
    // mismatches that look like a sampler bug.
    uint32_t gs2TriangleMip(const GSPrimitiveBatch &batch, float denom)
    {
        const GSDrawState &state = batch.state;
        const auto &ctx = state.context;
        if (!s_gsMipmap || !state.prim.tme)
            return 0u;
        const uint32_t mxl = static_cast<uint32_t>((ctx.tex1 >> 2) & 0x7u);
        if (mxl == 0u)
            return 0u;
        const float tw = static_cast<float>(state.textureWidth);
        const float th = static_cast<float>(state.textureHeight);
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const GSVertex &v2 = batch.vertices[2];
        float tu0, tv0, tu1, tv1, tu2, tv2;
        if (state.prim.fst)
        {
            tu0 = v0.u / 16.0f; tv0 = v0.v / 16.0f;
            tu1 = v1.u / 16.0f; tv1 = v1.v / 16.0f;
            tu2 = v2.u / 16.0f; tv2 = v2.v / 16.0f;
        }
        else
        {
            const float q0 = fabsQ(v0.q), q1 = fabsQ(v1.q), q2 = fabsQ(v2.q);
            tu0 = v0.s / q0 * tw; tv0 = v0.t / q0 * th;
            tu1 = v1.s / q1 * tw; tv1 = v1.t / q1 * th;
            tu2 = v2.s / q2 * tw; tv2 = v2.t / q2 * th;
        }
        const float texArea = std::fabs((tu1 - tu0) * (tv2 - tv0) - (tu2 - tu0) * (tv1 - tv0));
        const float scrArea = std::fabs(denom);
        if (scrArea <= 0.0f || texArea <= scrArea)
            return 0u;
        const float ratio = texArea / scrArea;
        const int lod = static_cast<int>(0.5f * std::log2(ratio) + 0.25f);
        return static_cast<uint32_t>(std::clamp(lod, 0, static_cast<int>(std::min(mxl, 6u))));
    }

    // Phase-2b scouting census (PS2X_GS_BATCHCENSUS). Run = a maximal span of consecutive
    // primitives sharing one (fbp,fbw,fpsm,zbp,zpsm) render target.
    struct Gs2BatchCensus
    {
        uint64_t curKey = ~0ull;
        unsigned long curRun = 0;
        unsigned long runs = 0;
        unsigned long long prims = 0;
        unsigned long long tilesTotal = 0;   // 64x64 tiles touched, summed over prims
        unsigned long long pixelsTotal = 0;  // bbox pixels, summed over prims
        unsigned long long tiles16Total = 0; // same, for 16x16 tiles
        unsigned long runHist[20] = {};      // log2 buckets of run length
        unsigned long tileHist[20] = {};     // log2 buckets of 64x64 tiles per prim
        unsigned long tile16Hist[20] = {};   // log2 buckets of 16x16 tiles per prim
        unsigned long maxRun = 0;
        // "eff" = the run a GPU batch could ACTUALLY take: broken by a target change AND
        // by any non-draw VRAM mutation (see g_vramMutationSeq).
        unsigned long lastMutSeq = 0;
        unsigned long effRun = 0, effRuns = 0, effMax = 0;
        unsigned long effHist[20] = {};
        unsigned long long effPrims = 0;
        uint64_t lastDumpMs = 0;
        // Distinct draw STATES within the current run -- sizes the dedup'd state table the
        // device needs (shipping ~200B of GSDrawState per primitive is not an option).
        static constexpr unsigned kStateSlots = 4096;
        uint64_t stateSlot[kStateSlots] = {};
        unsigned long statesInRun = 0;
        unsigned long statesTotal = 0;
        unsigned long stateHist[20] = {};
        unsigned long maxStates = 0;
    } g_batch;

    // FNV-1a over exactly the state the rasterizer consumes (the GsGpuState fields).
    uint64_t gs2DrawStateHash(const GSDrawState &st)
    {
        const GSContext &c = st.context;
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
        mix(c.frame.fbp); mix(c.frame.fbw); mix(c.frame.psm); mix(c.frame.fbmsk);
        mix(c.zbuf.zbp); mix(c.zbuf.psm); mix(c.zbuf.zmask ? 1u : 0u);
        mix(c.scissor.x0); mix(c.scissor.x1); mix(c.scissor.y0); mix(c.scissor.y1);
        mix(c.xyoffset.ofx); mix(c.xyoffset.ofy);
        mix(c.test); mix(c.alpha); mix(c.fba); mix(c.clamp); mix(c.tex1);
        mix(c.miptbp1); mix(c.miptbp2);
        mix(c.tex0.tbp0); mix(c.tex0.tbw); mix(c.tex0.psm); mix(c.tex0.tw); mix(c.tex0.th);
        mix(c.tex0.tcc); mix(c.tex0.tfx); mix(c.tex0.cbp); mix(c.tex0.cpsm);
        mix(c.tex0.csm); mix(c.tex0.csa);
        mix(st.texa.ta0); mix(st.texa.ta1); mix(st.texa.aem ? 1u : 0u);
        mix(st.fogR); mix(st.fogG); mix(st.fogB);
        mix(st.textureWidth); mix(st.textureHeight); mix(st.linearFilter ? 1u : 0u);
        const GSPrimReg &pr = st.prim;
        mix((pr.type) | (pr.iip << 3) | (pr.tme << 4) | (pr.fge << 5) |
            (pr.abe << 6) | (pr.fst << 7) | (pr.ctxt << 8));
        return h ? h : 1ull;
    }

    unsigned gs2Log2Bucket(unsigned long v)
    {
        unsigned b = 0;
        while (v > 1u && b < 19u) { v >>= 1; ++b; }
        return b;
    }

    void gs2PrintBatchCensus()
    {
        const Gs2BatchCensus &c = g_batch;
        std::fprintf(stderr,
                     "[gs2:batch] prims=%llu runs=%lu mean-run=%.1f max-run=%lu | "
                     "tiles/prim 64=%.2f 16=%.2f | mean-bbox-px=%.0f | "
                     "states/run mean=%.1f max=%lu | EFF-batch mean=%.1f max=%lu\n",
                     c.prims, c.runs,
                     c.runs ? double(c.prims) / double(c.runs) : 0.0, c.maxRun,
                     c.prims ? double(c.tilesTotal) / double(c.prims) : 0.0,
                     c.prims ? double(c.tiles16Total) / double(c.prims) : 0.0,
                     c.prims ? double(c.pixelsTotal) / double(c.prims) : 0.0,
                     c.runs ? double(c.statesTotal) / double(c.runs) : 0.0, c.maxStates,
                     c.effRuns ? double(c.effPrims) / double(c.effRuns) : 0.0, c.effMax);
        std::fprintf(stderr, "[gs2:batch]   run-len 2^n:");
        for (unsigned i = 0; i < 20u; ++i)
            if (c.runHist[i])
                std::fprintf(stderr, " %u:%lu", i, c.runHist[i]);
        std::fprintf(stderr, "\n[gs2:batch]   tiles64/prim 2^n:");
        for (unsigned i = 0; i < 20u; ++i)
            if (c.tileHist[i])
                std::fprintf(stderr, " %u:%lu", i, c.tileHist[i]);
        std::fprintf(stderr, "\n[gs2:batch]   tiles16/prim 2^n:");
        for (unsigned i = 0; i < 20u; ++i)
            if (c.tile16Hist[i])
                std::fprintf(stderr, " %u:%lu", i, c.tile16Hist[i]);
        std::fprintf(stderr, "\n[gs2:batch]   EFF-batch 2^n:");
        for (unsigned i = 0; i < 20u; ++i)
            if (c.effHist[i])
                std::fprintf(stderr, " %u:%lu", i, c.effHist[i]);
        std::fprintf(stderr, "\n[gs2:batch]   states/run 2^n:");
        for (unsigned i = 0; i < 20u; ++i)
            if (c.stateHist[i])
                std::fprintf(stderr, " %u:%lu", i, c.stateHist[i]);
        std::fprintf(stderr, "\n");
    }

    void gs2BatchCensusRecord(const GSPrimitiveBatch &batch, uint32_t fbp, uint32_t fbw, uint32_t zbp)
    {
        Gs2BatchCensus &c = g_batch;
        const GSContext &ctx = batch.state.context;
        const uint64_t key = uint64_t(fbp) | (uint64_t(fbw) << 20) |
                             (uint64_t(ctx.frame.psm & 0x3Fu) << 26) |
                             (uint64_t(zbp) << 32) | (uint64_t(ctx.zbuf.psm & 0x3Fu) << 52);
        if (key != c.curKey)
        {
            if (c.curRun != 0u)
            {
                ++c.runs;
                ++c.runHist[gs2Log2Bucket(c.curRun)];
                if (c.curRun > c.maxRun)
                    c.maxRun = c.curRun;
                c.statesTotal += c.statesInRun;
                ++c.stateHist[gs2Log2Bucket(c.statesInRun ? c.statesInRun : 1u)];
                if (c.statesInRun > c.maxStates)
                    c.maxStates = c.statesInRun;
            }
            c.curKey = key;
            c.curRun = 0u;
            c.statesInRun = 0u;
            std::memset(c.stateSlot, 0, sizeof(c.stateSlot));
        }
        // Distinct-state count for this run (open addressing; saturates at the table size,
        // which is itself the answer we care about -- "does it fit in a small table?").
        {
            const uint64_t h = gs2DrawStateHash(batch.state);
            unsigned idx = unsigned(h % Gs2BatchCensus::kStateSlots);
            for (unsigned probe = 0; probe < 64u; ++probe)
            {
                uint64_t &slot = c.stateSlot[(idx + probe) % Gs2BatchCensus::kStateSlots];
                if (slot == h)
                    break;
                if (slot == 0ull)
                {
                    slot = h;
                    ++c.statesInRun;
                    break;
                }
            }
        }
        ++c.curRun;
        ++c.prims;

        // effective batch run: also broken by any interleaved non-draw VRAM mutation
        const bool mutated = (g_vramMutationSeq != c.lastMutSeq);
        c.lastMutSeq = g_vramMutationSeq;
        if (mutated || c.curRun == 1u)
        {
            if (c.effRun != 0u)
            {
                ++c.effRuns;
                ++c.effHist[gs2Log2Bucket(c.effRun)];
                if (c.effRun > c.effMax)
                    c.effMax = c.effRun;
            }
            c.effRun = 0u;
        }
        ++c.effRun;
        ++c.effPrims;

        // bbox -> tile count (64x64 screen tiles), scissor-clipped like the raster is
        const int ofx = ctx.xyoffset.ofx >> 4, ofy = ctx.xyoffset.ofy >> 4;
        int minX = INT32_MAX, minY = INT32_MAX, maxX = INT32_MIN, maxY = INT32_MIN;
        for (uint8_t i = 0; i < batch.vertexCount && i < batch.vertices.size(); ++i)
        {
            const int vx = int(batch.vertices[i].x) - ofx, vy = int(batch.vertices[i].y) - ofy;
            minX = std::min(minX, vx); minY = std::min(minY, vy);
            maxX = std::max(maxX, vx); maxY = std::max(maxY, vy);
        }
        minX = std::max(minX, int(ctx.scissor.x0)); minY = std::max(minY, int(ctx.scissor.y0));
        maxX = std::min(maxX, int(ctx.scissor.x1)); maxY = std::min(maxY, int(ctx.scissor.y1));
        if (minX <= maxX && minY <= maxY)
        {
            const unsigned long tiles = (unsigned long)((maxX >> 6) - (minX >> 6) + 1) *
                                        (unsigned long)((maxY >> 6) - (minY >> 6) + 1);
            const unsigned long tiles16 = (unsigned long)((maxX >> 4) - (minX >> 4) + 1) *
                                          (unsigned long)((maxY >> 4) - (minY >> 4) + 1);
            c.tilesTotal += tiles;
            c.tiles16Total += tiles16;
            c.pixelsTotal += (unsigned long long)(maxX - minX + 1) * (maxY - minY + 1);
            ++c.tileHist[gs2Log2Bucket(tiles)];
            ++c.tile16Hist[gs2Log2Bucket(tiles16)];
        }

        if ((c.prims & 1023u) != 0u)
            return;
        const uint64_t now = gs2NowMs();
        if (c.lastDumpMs == 0u) { c.lastDumpMs = now; return; }
        if (now - c.lastDumpMs < 10000u) return;
        c.lastDumpMs = now;
        gs2PrintBatchCensus();
    }

    uint64_t stateCensusKey(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &st = batch.state;
        const GSContext &ctx = st.context;
        uint64_t k = 0;
        k |= static_cast<uint64_t>(st.prim.type & 7u);
        k |= static_cast<uint64_t>(st.prim.iip ? 1u : 0u) << 3;
        k |= static_cast<uint64_t>(st.prim.tme ? 1u : 0u) << 4;
        k |= static_cast<uint64_t>(st.prim.fge ? 1u : 0u) << 5;
        k |= static_cast<uint64_t>(st.prim.abe ? 1u : 0u) << 6;
        k |= static_cast<uint64_t>(st.prim.fst ? 1u : 0u) << 7;
        k |= static_cast<uint64_t>(ctx.frame.psm & 0x3Fu) << 8;
        const uint32_t msk = ctx.frame.fbmsk;
        k |= (msk == 0u ? 0ull : msk == 0xFF000000u ? 1ull : msk == 0x00FFFFFFu ? 2ull : 3ull) << 14;
        k |= static_cast<uint64_t>(ctx.zbuf.psm & 0xFu) << 16;
        k |= static_cast<uint64_t>(ctx.zbuf.zmask ? 1u : 0u) << 20;
        const uint64_t test = ctx.test;
        k |= (test & 1u) << 21;         // ATE
        k |= ((test >> 1) & 7u) << 22;  // ATST
        k |= ((test >> 12) & 3u) << 25; // AFAIL
        k |= ((test >> 14) & 1u) << 27; // DATE
        k |= ((test >> 15) & 1u) << 28; // DATM
        k |= ((test >> 16) & 1u) << 29; // ZTE
        k |= ((test >> 17) & 3u) << 30; // ZTST
        k |= (ctx.alpha & 0xFFu) << 32; // blend A/B/C/D
        if (st.prim.tme)
        {
            k |= static_cast<uint64_t>(ctx.tex0.psm & 0x3Fu) << 40;
            k |= static_cast<uint64_t>(ctx.tex0.cpsm & 0xFu) << 46;
            k |= static_cast<uint64_t>(ctx.tex0.csm & 1u) << 50;
            k |= static_cast<uint64_t>(ctx.tex0.tfx & 3u) << 51;
            k |= static_cast<uint64_t>(ctx.tex0.tcc & 1u) << 53;
            k |= static_cast<uint64_t>(st.linearFilter ? 1u : 0u) << 54;
            k |= (ctx.clamp & 0xFu) << 55; // WMS | WMT<<2
        }
        k |= static_cast<uint64_t>(st.pabe ? 1u : 0u) << 59;
        k |= (st.dthe & 1u) << 60;
        k |= (st.colclamp & 1u) << 61;
        k |= static_cast<uint64_t>((st.scanmsk & 3u) != 0u ? 1u : 0u) << 62;
        k |= static_cast<uint64_t>(g_gs2EraOpen ? 1u : 0u) << 63;
        return k;
    }

    void stateCensusDump()
    {
        std::vector<std::pair<uint64_t, uint64_t>> rows(g_stateCensus.begin(), g_stateCensus.end());
        std::sort(rows.begin(), rows.end(),
                  [](const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b)
                  { return a.second > b.second; });
        std::fprintf(stderr, "[gs2:sc] === draws=%lu combos=%zu (desc, top 96) ===\n",
                     g_stateCensusN, rows.size());
        static const char *kPrimNames[8] = {"pt", "ln", "ls", "tri", "ts", "tf", "spr", "?"};
        static const char *kMskNames[4] = {"0", "A", "RGB", "mix"};
        size_t shown = 0;
        for (const auto &row : rows)
        {
            if (++shown > 96u)
                break;
            const uint64_t k = row.first;
            char tex[96] = "un-tex";
            if ((k >> 4) & 1u)
                std::snprintf(tex, sizeof(tex),
                              "tpsm=0x%02x cpsm=0x%x csm=%u tfx=%u tcc=%u lin=%u wms=%u wmt=%u",
                              static_cast<unsigned>((k >> 40) & 0x3Fu),
                              static_cast<unsigned>((k >> 46) & 0xFu),
                              static_cast<unsigned>((k >> 50) & 1u),
                              static_cast<unsigned>((k >> 51) & 3u),
                              static_cast<unsigned>((k >> 53) & 1u),
                              static_cast<unsigned>((k >> 54) & 1u),
                              static_cast<unsigned>((k >> 55) & 3u),
                              static_cast<unsigned>((k >> 57) & 3u));
            std::fprintf(stderr,
                         "[gs2:sc] n=%llu era=%u key=%016llx %s iip=%u fge=%u abe=%u fst=%u | "
                         "fpsm=0x%02x msk=%s zpsm=0x%x zmsk=%u | ate=%u atst=%u afail=%u date=%u "
                         "datm=%u zte=%u ztst=%u | abcd=%u%u%u%u | %s | pabe=%u dthe=%u cclamp=%u smsk=%u\n",
                         static_cast<unsigned long long>(row.second),
                         static_cast<unsigned>(k >> 63),
                         static_cast<unsigned long long>(k),
                         kPrimNames[k & 7u],
                         static_cast<unsigned>((k >> 3) & 1u), static_cast<unsigned>((k >> 5) & 1u),
                         static_cast<unsigned>((k >> 6) & 1u), static_cast<unsigned>((k >> 7) & 1u),
                         static_cast<unsigned>((k >> 8) & 0x3Fu),
                         kMskNames[(k >> 14) & 3u],
                         static_cast<unsigned>((k >> 16) & 0xFu), static_cast<unsigned>((k >> 20) & 1u),
                         static_cast<unsigned>((k >> 21) & 1u), static_cast<unsigned>((k >> 22) & 7u),
                         static_cast<unsigned>((k >> 25) & 3u), static_cast<unsigned>((k >> 27) & 1u),
                         static_cast<unsigned>((k >> 28) & 1u), static_cast<unsigned>((k >> 29) & 1u),
                         static_cast<unsigned>((k >> 30) & 3u),
                         static_cast<unsigned>((k >> 32) & 3u), static_cast<unsigned>((k >> 34) & 3u),
                         static_cast<unsigned>((k >> 36) & 3u), static_cast<unsigned>((k >> 38) & 3u),
                         tex,
                         static_cast<unsigned>((k >> 59) & 1u), static_cast<unsigned>((k >> 60) & 1u),
                         static_cast<unsigned>((k >> 61) & 1u), static_cast<unsigned>((k >> 62) & 1u));
        }
    }

    void stateCensusRecord(const GSPrimitiveBatch &batch)
    {
        ++g_stateCensusN;
        ++g_stateCensus[stateCensusKey(batch)];
        if ((g_stateCensusN & 255u) != 0u)
            return;
        const uint64_t now = gs2NowMs();
        if (g_stateCensusLastDumpMs == 0u)
        {
            g_stateCensusLastDumpMs = now;
            return;
        }
        if (now - g_stateCensusLastDumpMs < 10000u)
            return;
        g_stateCensusLastDumpMs = now;
        stateCensusDump();
    }
    struct Gs2XferStats
    {
        unsigned long n = 0;
        unsigned long long bytes = 0;
        unsigned long ct32 = 0, t8 = 0, t4 = 0, ct16 = 0, other = 0, badDims = 0;
        // Texture-arena slice (blocks >= 12288 — the level texture arena; the movie's 16x16
        // CT32 tiles also live there BY DESIGN, so split tile vs non-tile):
        unsigned long arenaTile = 0, arenaNonTile = 0;
        uint32_t lastNtDbp = 0, lastNtDpsm = 0, lastNtRrw = 0, lastNtRrh = 0;
        uint32_t lastDbp = 0, lastDbw = 0, lastDpsm = 0, lastRrw = 0, lastRrh = 0;
    } g_gs2Xfer;

    void gs2PrintXfer()
    {
        std::fprintf(stderr,
                     "[gs2:xfer] n=%lu bytes=%llu dpsm{ct32:%lu t8:%lu t4:%lu ct16:%lu other:%lu} "
                     "badDims=%lu arena{tile:%lu nonTile:%lu lastNt(dbp=%u dpsm=0x%x rr=%ux%u)} "
                     "last(dbp=%u dbw=%u dpsm=0x%x rr=%ux%u)\n",
                     g_gs2Xfer.n, g_gs2Xfer.bytes, g_gs2Xfer.ct32, g_gs2Xfer.t8, g_gs2Xfer.t4,
                     g_gs2Xfer.ct16, g_gs2Xfer.other, g_gs2Xfer.badDims,
                     g_gs2Xfer.arenaTile, g_gs2Xfer.arenaNonTile,
                     g_gs2Xfer.lastNtDbp, g_gs2Xfer.lastNtDpsm, g_gs2Xfer.lastNtRrw, g_gs2Xfer.lastNtRrh,
                     g_gs2Xfer.lastDbp, g_gs2Xfer.lastDbw, g_gs2Xfer.lastDpsm,
                     g_gs2Xfer.lastRrw, g_gs2Xfer.lastRrh);
    }
}

// ---- cont.207: banded parallel rasterization -------------------------------------------
// The cont.206 profile found ONE rasterizer thread at 99% with the EE blocked on it 71% of the
// time, on a 4-core box using ~1.4 cores. Raster ablation measured 3.50x available in heavy
// scenes, so most of that is reachable simply by splitting the work across cores.
//
// The split is by SCANLINE BAND: each row has exactly ONE owning thread, so
//   * no two threads ever write the same pixel -> no VRAM lock, and
//   * every pixel is still written by primitives in submission order within its band,
// which is what makes this order-preserving rather than merely parallel. Transfers, uploads and
// presents stay on the coordinator, so they are natural barriers.
// The row->owner map started as one CONTIGUOUS strip per thread (cont.207/209) and is now
// STRIPED -- ((y >> shift) % N) == k, cont.210 below -- because contiguous strips left threads
// idle whenever the geometry concentrated in a few rows.
//
// PS2X_GS_RASTER_THREADS=N. ★ cont.210: the pool is now ON BY DEFAULT, one thread per logical
// core (capped at 8). Banding is PIXEL-EXACT -- the cont.208 bench's VRAM hash is byte-identical
// at every thread count and stripe size -- and it measured 3.2x on the 20,000-primitive capture
// and 2.4x on matched rasterizer work in the real game (188 vs 77 kprim/s in the level era).
// PS2X_GS_RASTER_THREADS=1 restores the single-threaded path exactly.
static const unsigned s_gsRasterThreads = []
{
    const char *e = std::getenv("PS2X_GS_RASTER_THREADS");
    unsigned n;
    if (e && e[0])
        n = static_cast<unsigned>(std::atoi(e));
    else if (ThreadNaming::AutoCpuPinEnabled() && ThreadNaming::AutoCpuPlan().valid)
    {
        // ★ cont.230: with the EE thread pinned to its own physical core (ThreadNaming.h), one
        // raster thread per REMAINING physical core measured best -- 3 threads 58.7 s vs 6 threads
        // 62.1 s for the same guest work (build 377, e31/e30, consecutive runs); more threads only
        // add SMT contention and the raster keeps up with the EE at that count (1 thread does not:
        // e32). Unpinned, fewer threads were faster too (8/4/2: 78.7/75.4/73.2 s).
        // ★★ cont.317: that was measured when the EE was the long pole and the row loop was 37%
        // dearer. With the raster the pole (three workers at 90%+, EE at 30-38%), one worker per
        // logical CPU of the pool -- every CPU but the EE's own, 7 of 8 here -- benches the Helm's
        // Deep fight capture at 268 vs 335 ms (-20%, three alternating pairs, non-overlapping); 6 on
        // the old 3-core pool was -11% in one series and +0% in another, and 4 was bimodal because
        // a shared affinity set lets the OS stack two workers on one core. PS2X_GS_RASTER_THREADS
        // still overrides; PS2X_GS_CPUS still overrides the pool.
        n = ThreadNaming::AutoCpuPlan().gsCpus;
    }
    else
    {
        n = std::thread::hardware_concurrency();
        if (n == 0u) n = 1u; // unknowable core count: stay single-threaded
    }
    if (n < 1u) n = 1u;
    if (n > 8u) n = 8u;
    return n;
}();
// ★ The fan-out threshold must be on PIXELS, not rows. A condvar handoff costs a few microseconds;
// a tall but thin primitive has almost no work in it, so splitting it is pure loss. (Measured: an
// 8-ROW threshold made the whole run SLOWER than single-threaded.)
// VRAM is at most 2048 scanlines; bands are fixed strips of that for a whole run.
static constexpr int kBandScreenRows = 2048;
static const int s_gsBandMinPix = []
{
    const char *e = std::getenv("PS2X_GS_BAND_MINPIX");
    return (e && e[0]) ? std::atoi(e) : 1024; // measured optimum (cont.208 bench)
}();

// ★★ cont.210: STRIPED bands. Contiguous strips bought only 1.25x at 4 threads against 2.1x at 8
// -- that gap is load IMBALANCE, not a threading limit: the geometry concentrates in a subset of
// rows, so one strip carries most of the pixels while its neighbours idle. Ownership becomes
//     ((y >> shift) % N) == index
// i.e. thread k takes every Nth GROUP of 2^shift rows, which interleaves the busy rows across all
// threads no matter where they are.
//
// ★ This mirrors PCSX2's software renderer exactly (GS/Renderers/SW/GSRasterizer.cpp): its ctor
// builds `m_scanline[i] = (i % threads) == id` and every rasterizer path tests
// `m_scanline[top >> m_thread_height]`, with compute_best_thread_height() returning 4 (16-row
// groups) by default and the comment "ideal value between 3 and 5". Groups rather than single rows
// matter for a swizzled framebuffer: a 32-bit GS column is 8x2 pixels in one 64-byte line, so a
// 1-row interleave would put two threads in the same cache line on every write.
//
// PS2X_GS_BAND_STRIPE = log2(rows per stripe unit), default 2 (=4 rows). PCSX2 defaults to 4
// (16 rows); we MEASURED the sweep on the cont.208 bench at 8 threads and 4 rows wins:
//   1 row 38.2 ms | 2 rows 26.2 | 4 rows 26.3 | 8 rows 29.9 | 16 rows 30.3 | 32 rows 39.7 |
//   64 rows 41.0 (= the contiguous 40.5, as it must be: 64x8 covers the whole framebuffer)
// Both tails are explained: too large stops interleaving the busy rows, and 1 row splits a
// 32-bit GS column (8x2 pixels in one 64-byte line) across two threads -- false sharing on every
// write. 4 rows is the smallest unit that keeps whole columns per thread.
// A NEGATIVE value restores cont.209's contiguous strips, for A/B.
static const int s_gsBandStripeShift = []
{
    const char *e = std::getenv("PS2X_GS_BAND_STRIPE");
    int v = (e && e[0]) ? std::atoi(e) : 2;
    if (v > 11) v = 11; // VRAM is 2048 rows: a larger unit would hand band 0 everything
    return v;
}();
static constexpr int kBandRowSpace = 2048;

// This thread's row-ownership table. Non-null ONLY while a fan-out is in flight, so the direct
// (unfanned) path still draws every row. One byte load per scanline -- no division in the loop.
static thread_local const uint8_t *s_bandOwnRows = nullptr;

namespace
{
    // ★ cont.231 PS2X_GS_BAND_W0 / PS2X_GS_BAND_W (defaults 1 / 1 = the cont.210 round-robin): stripe
    // WEIGHTS. The coordinator (band 0) also runs every primitive's prologue and every upload and
    // transfer serially, so with equal shares it is the long pole and the helpers idle at the end of
    // every run (24% of their time in the cont.231 live profile). A cycle of W0 + (count-1)*W stripe
    // units hands the first W0 to band 0 and W to each helper. Ownership is a pure partition of rows:
    // any weights produce byte-identical VRAM (the cont.208 hash oracle), so this is a live env knob.
    const unsigned s_gsBandW0 = []
    { const char *e = std::getenv("PS2X_GS_BAND_W0"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 1ul; return v ? static_cast<unsigned>(v) : 1u; }();
    const unsigned s_gsBandW = []
    { const char *e = std::getenv("PS2X_GS_BAND_W"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 1ul; return v ? static_cast<unsigned>(v) : 1u; }();
    // Ownership depends only on (index, count), both fixed for a thread's lifetime, so build once.
    const uint8_t *gs2BandRowTable(unsigned index, unsigned count)
    {
        static thread_local std::array<uint8_t, kBandRowSpace> tbl{};
        static thread_local unsigned built = ~0u;
        const unsigned key = (index << 8) | count;
        if (built != key)
        {
            const unsigned cycle = s_gsBandW0 + (count - 1u) * s_gsBandW;
            for (int y = 0; y < kBandRowSpace; ++y)
            {
                const unsigned pos = static_cast<unsigned>(y >> s_gsBandStripeShift) % cycle;
                const unsigned owner = pos < s_gsBandW0 ? 0u : 1u + (pos - s_gsBandW0) / s_gsBandW;
                tbl[static_cast<size_t>(y)] = (owner == index) ? 1u : 0u;
            }
            built = key;
        }
        return tbl.data();
    }
    // RAII: an early return inside a rasterizer can never leave the mask armed for the direct path.
    struct BandRowScope
    {
        explicit BandRowScope(const uint8_t *rows) { s_bandOwnRows = rows; }
        ~BandRowScope() { s_bandOwnRows = nullptr; }
        BandRowScope(const BandRowScope &) = delete;
        BandRowScope &operator=(const BandRowScope &) = delete;
    };
    inline bool gs2BandStriped() { return s_gsBandStripeShift >= 0; }

    // ★ cont.318 PS2X_GS_BAND_ADAPT=<alpha %> (default 0 = OFF; 30 = a 30% step per run): ADAPTIVE
    // stripe weights. WHY: with fixed round-robin ownership every run ends at the barrier waiting for
    // the slowest participant -- the one sharing a core with the EE thread or stacked with another
    // worker by the OS -- and cont.317 measured `tail` at ~11 ms/frame with workers idle ~30%. After
    // each run the coordinator knows every participant's busy time; it re-weights the shares so that
    // finish times equalise (share_i <- share_i * mean_rate / rate_i, smoothed) and lays the units out
    // by smooth weighted round-robin (deficit credit), so the interleave stays fine at every scale.
    // Ownership is still a pure partition of rows -> byte-identical VRAM at any weights (the cont.208
    // hash oracle), exactly as the static W0/W knobs.
    const unsigned s_gsBandAdapt = []
    { const char *e = std::getenv("PS2X_GS_BAND_ADAPT"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 0ul; return v > 100ul ? 100u : static_cast<unsigned>(v); }();
    const uint8_t *gs2BandRowTableFrom(const uint8_t *ownerUnits, uint64_t seq, unsigned index)
    {
        static thread_local std::array<uint8_t, kBandRowSpace> tbl{};
        static thread_local uint64_t builtSeq = ~0ull;
        static thread_local unsigned builtIdx = ~0u;
        if (builtSeq != seq || builtIdx != index)
        {
            for (int y = 0; y < kBandRowSpace; ++y)
                tbl[static_cast<size_t>(y)] = (ownerUnits[static_cast<unsigned>(y) >> s_gsBandStripeShift] == index) ? 1u : 0u;
            builtSeq = seq;
            builtIdx = index;
        }
        return tbl.data();
    }

    // ★ cont.318 PS2X_GS_BAND_SPIN=<microseconds> (default 0 = sleep immediately, the cont.207
    // behaviour): before sleeping on the condition variable, a helper polls m_bandSeq and the
    // coordinator polls m_bandPending for up to this long. WHY: the per-participant barrier
    // attribution ([gsgpu:bandbal], build 743) put ~0.6 ms of idle per ~3.2 ms run on EVERY
    // participant, evenly, insensitive to stripe size / pinning / adaptive weights -- i.e. wake-up
    // latency and scheduling jitter, not imbalance. PCSX2 does the same on both sides of its
    // rasterizer job queues: Threading::WorkSema::WaitForWorkWithSpin / WaitForEmptyWithSpin spin
    // SPIN_TIME_NS (50 us, WAIT_SPIN_MICROSECONDS) in ShortSpin() ~500 ns pause steps
    // (common/Semaphore.cpp, common/HostSys.cpp). Ownership/order untouched -> pixel-exact.
    const unsigned long long s_gsBandSpinNs = []
    { const char *e = std::getenv("PS2X_GS_BAND_SPIN"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 0ul; if (v > 100000ul) v = 100000ul; return static_cast<unsigned long long>(v) * 1000ull; }();
    // ★ cont.318 PS2X_GS_BAND_SKIP (default ON, "=0" restores): a participant skips a primitive
    // OUTRIGHT -- no resolveDraw, no edge setup, no CLUT decode -- when none of the rows its
    // conservative vertex extent can touch is owned by this participant. WHY: in the striped run
    // fan-out every participant walked every primitive of the run and only the per-row check
    // inside the rasterizer filtered rows, so a ~6-row triangle paid its whole prologue on all seven
    // threads to draw on one or two. PCSX2 pushes a draw only to the workers whose row blocks its
    // bbox intersects (GSRasterizerList::Queue: m_scanline[top..bottom]). The extent here is the
    // rasterizers' own floor(min y - ofy) .. ceil(max y - ofy) widened by one row each side, so a
    // skipped primitive had no owned row to draw: byte-identical VRAM.
    const bool s_gsBandSkip = []
    { const char *e = std::getenv("PS2X_GS_BAND_SKIP"); return !(e && e[0] == '0'); }();
    inline bool gs2BandOwnsAnyRow(const GSPrimitiveBatch &b, const uint8_t *rows)
    {
        if (rows == nullptr || b.vertexCount == 0u)
            return true;
        const int ofy = b.state.context.xyoffset.ofy >> 4;
        float mn = b.vertices[0].y, mx = mn;
        for (uint32_t i = 1; i < b.vertexCount; ++i)
        {
            const float y = b.vertices[i].y;
            if (y < mn) mn = y;
            if (y > mx) mx = y;
        }
        if (!(mn <= mx)) // NaN: let the rasterizer decide
            return true;
        const float fmn = std::floor(mn) - static_cast<float>(ofy) - 1.0f;
        const float fmx = std::ceil(mx) - static_cast<float>(ofy) + 1.0f;
        if (fmn < 0.0f && fmx > static_cast<float>(kBandRowSpace)) // spans everything: someone owns a row
            return true;
        int y0 = fmn < 0.0f ? 0 : (fmn >= static_cast<float>(kBandRowSpace) ? kBandRowSpace - 1 : static_cast<int>(fmn));
        int y1 = fmx < 0.0f ? 0 : (fmx >= static_cast<float>(kBandRowSpace) ? kBandRowSpace - 1 : static_cast<int>(fmx));
        // The scissor bounds every rasterizer's rows too; clip the probe to it (still conservative:
        // a row outside the scissor is never drawn by anyone).
        const auto &sc = b.state.context.scissor;
        if (static_cast<int>(sc.y0) > y0) y0 = static_cast<int>(sc.y0);
        if (static_cast<int>(sc.y1) < y1) y1 = static_cast<int>(sc.y1);
        for (int y = y0; y <= y1; ++y)
            if (rows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
                return true;
        return false;
    }

    // ★ cont.318 PS2X_GS_BAND_DYN=<rows per claim> (default 16; "=0" = the static striped ownership):
    // DYNAMIC row-group claiming inside a run. WHY: [gsgpu:bandbal] (builds 743/744) showed every
    // participant idle ~0.6 ms of each ~3.2 ms run at the barrier, evenly spread, and unmoved by
    // stripe size (1/2/3), 1:1 pinning, adaptive weights, or spin-before-sleep -- random per-run
    // execution jitter (SMT co-scheduling, interrupts, other threads on a 4-core/8-thread host),
    // which NO static partition can absorb. Here the run's rows are cut into contiguous groups of
    // this many rows (aligned to multiples of it, so the 4-row GS column of the cont.210 stripe
    // argument is never split when the value is a multiple of 4), and participants claim the next
    // group from an atomic counter until none is left: the finish spread collapses to ~one group's
    // work. Each group is drawn by exactly one thread, every primitive in submission order, clipped
    // to the group by the rasterizers' own [bandY0, bandY1] clip (the cont.209 contiguous mode's
    // oracle-verified path), so VRAM is byte-identical. Points are drawn by the coordinator in a
    // pre-pass exactly as the fast coordinator path always did (RasterBand never drew them).
    const int s_gsBandDynRows = []
    // Measured (build 745, 7 threads): Helm's capture static 252-268 ms -> 8 rows 234 / 16 rows 235 /
    // 32 rows 241 / 64 rows 281-299 (too coarse: 7 groups); live 150 s fights, barrier idle per
    // participant 24 s static -> 12 s at 16 rows -> 5.6 s at 8 rows, but 8 rows costs +5% busy in
    // setup replication, so the two tie on busy+idle and 16 burns less CPU (the EE shares a core).
    { const char *e = std::getenv("PS2X_GS_BAND_DYN"); long v = e ? std::strtol(e, nullptr, 10) : 16l; if (v < 0) v = 0; if (v > 512) v = 512; return static_cast<int>(v); }();
    // Conservative row extent of a batch for the dynamic claim: floor(min y) - ofy - 1 .. ceil(max y)
    // - ofy + 1, clipped to the scissor (no rasterizer draws outside it). Empty (y0 > y1) for points
    // and degenerate batches.
    inline void gs2BandExtent(const GSPrimitiveBatch &b, int32_t &outY0, int32_t &outY1)
    {
        outY0 = 1; outY1 = 0;
        if (b.vertexCount == 0u || b.state.prim.type == GS_PRIM_POINT)
            return;
        const int ofy = b.state.context.xyoffset.ofy >> 4;
        float mn = b.vertices[0].y, mx = mn;
        for (uint32_t i = 1; i < b.vertexCount; ++i)
        {
            const float y = b.vertices[i].y;
            if (y < mn) mn = y;
            if (y > mx) mx = y;
        }
        const auto &sc = b.state.context.scissor;
        if (!(mn <= mx)) { outY0 = static_cast<int32_t>(sc.y0); outY1 = static_cast<int32_t>(sc.y1); return; } // NaN: whole scissor
        const float fmn = std::floor(mn) - static_cast<float>(ofy) - 1.0f;
        const float fmx = std::ceil(mx) - static_cast<float>(ofy) + 1.0f;
        int y0 = fmn < -1.0f ? -1 : (fmn > 4096.0f ? 4096 : static_cast<int>(fmn));
        int y1 = fmx < -1.0f ? -1 : (fmx > 4096.0f ? 4096 : static_cast<int>(fmx));
        if (static_cast<int>(sc.y0) > y0) y0 = static_cast<int>(sc.y0);
        if (static_cast<int>(sc.y1) < y1) y1 = static_cast<int>(sc.y1);
        outY0 = y0; outY1 = y1;
    }
}

namespace
{
    // Conservative scanline extent of a batch, used only to decide whether to fan out.
    int gs2BandMinY(const GSPrimitiveBatch &batch)
    {
        const int ofy = batch.state.context.xyoffset.ofy >> 4;
        float m = batch.vertices[0].y;
        for (uint32_t i = 1; i < batch.vertexCount && i < 3u; ++i)
            if (batch.vertices[i].y < m) m = batch.vertices[i].y;
        return static_cast<int>(std::floor(m)) - ofy;
    }
    // Bounding-box width, for the pixel-count fan-out heuristic.
    int gs2BandWidth(const GSPrimitiveBatch &batch)
    {
        float lo = batch.vertices[0].x, hi = batch.vertices[0].x;
        for (uint32_t i = 1; i < batch.vertexCount && i < 3u; ++i)
        {
            if (batch.vertices[i].x < lo) lo = batch.vertices[i].x;
            if (batch.vertices[i].x > hi) hi = batch.vertices[i].x;
        }
        const int w = static_cast<int>(std::ceil(hi)) - static_cast<int>(std::floor(lo)) + 1;
        return w > 0 ? w : 1;
    }
    int gs2BandMaxY(const GSPrimitiveBatch &batch)
    {
        const int ofy = batch.state.context.xyoffset.ofy >> 4;
        float m = batch.vertices[0].y;
        for (uint32_t i = 1; i < batch.vertexCount && i < 3u; ++i)
            if (batch.vertices[i].y > m) m = batch.vertices[i].y;
        return static_cast<int>(std::ceil(m)) - ofy;
    }
}

void GSCpuBackend::resolveDraw(const GSDrawState &state)
{
    const auto &ctx = state.context;
    m_draw.frameRead = m_readVramFuncs[ctx.frame.psm & 0x3Fu];
    m_draw.frameWrite = m_writeVramFuncs[ctx.frame.psm & 0x3Fu];
    m_draw.zRead = m_readVramFuncs[ctx.zbuf.psm & 0x3Fu];
    m_draw.zWrite = m_writeVramFuncs[ctx.zbuf.psm & 0x3Fu];
    m_draw.texRead = m_readVramFuncs[ctx.tex0.psm & 0x3Fu];
    m_draw.fastCt32Z24 = (ctx.frame.psm & 0x3Fu) == GS_PSM_CT32 &&
                         (ctx.zbuf.psm & 0x3Fu) == GS_PSM_Z24;
    m_draw.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    m_draw.fbw = std::max<u32>(ctx.frame.fbw, 1u);
    m_draw.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);

    // ★★ cont.227: decode the palette ONCE for this draw. T4 family only: 16 entries against the
    // census's 191 CLUT lookups per draw is an 8% decode cost for a 100% saving, and it is
    // amortised even on a tiny primitive. T8 (256 entries) stays on the per-texel path -- it is
    // 0.01% of this game's texels, so caching it would be an unmeasured risk for no measured win.
    // Re-decoding on EVERY draw is deliberate: a scene draw can stomp the CLUT blocks it is
    // sampling from (that is exactly why the cont.165 CLUT shadow exists), and a cross-draw reuse
    // key would have to model that invalidation. PCSX2 does model it
    // (GSClut::InvalidateRange(start, end, is_draw)); re-decoding per draw makes it moot.
    m_draw.texMip = ~0u; // cont.228: force resolveTexMip on this draw's first texel fetch
    m_draw.clutValid = false;
    // cont.320: texLin/texPred are carried into the next texCacheResolve on purpose -- it closes the
    // previous key's tap count (linear vs swizzled) before deciding this draw; texLin is only read
    // after resolveTexMip, which every textured draw runs first (texMip = ~0u above).
    if (s_gsFastClut && state.prim.tme && m_vram != nullptr)
    {
        const uint8_t tpsm = static_cast<uint8_t>(ctx.tex0.psm & 0x3Fu);
        const bool t8 = (tpsm == GS_PSM_T8) && s_gsFastP8;
        if (tpsm == GS_PSM_T4 || tpsm == GS_PSM_T4HL || tpsm == GS_PSM_T4HH || t8)
        {
            // cont.317: PSMT8 decodes all 256 entries. A CT32 palette of 256 is 16x16 pixels and a
            // CT16 one at most 32 rows of 16, both inside the ONE page at cbp, so the single-page
            // stomp test below still covers every entry.
            const uint32_t entries = t8 ? 256u : 16u;
            const auto &tex = ctx.tex0;
            const uint64_t key = gs2ClutCacheKey(tex.cbp, tex.cpsm, tex.csm, tex.csa, tpsm);
            const uint64_t gen = g_gs2VramGen.load(std::memory_order_relaxed);
            // All 16 CSM1 entries are read at (x < 64, y = 0) with bw = 1, and PageId is
            // base/32 + (y/pageH)*bw + x/pageW, so they occupy exactly the ONE page holding cbp.
            const uint32_t page = tex.cbp / static_cast<uint32_t>(GSMem::BLOCKS_PER_PAGE);
            const bool keyMiss = key != m_draw.clutKey;
            const bool genMiss = gen != m_draw.clutGen;
            // PS2X_GS_CLUTMUTATE=1: never re-decode once a palette exists -- exactly the
            // staleness the key/generation/stomp tests exist to prevent.
            const bool reuse = (s_gsClutMutate == 1 && m_draw.clutKey != ~0ull) ||
                               (!keyMiss && !genMiss && !m_draw.clutStomped);
            if (!reuse)
            {
                for (uint32_t i = 0; i < entries; ++i)
                {
                    const uint32_t c = LookupCLUT(state, static_cast<uint8_t>(i), tex.cbp, tex.cpsm,
                                                  tex.csm, tex.csa, tpsm);
                    m_draw.clut[i] = c;
                    // cont.324: the float mirror the four-wide lin tap loads directly -- bilinear4's own
                    // unpack (cvtepu8 -> cvtepi32_ps), once per entry instead of once per texel.
                    _mm_store_ps(m_draw.clutF[i], _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(c)))));
                }
                m_draw.clutKey = key;
                m_draw.clutGen = gen;
                m_draw.clutPage = page;
                m_draw.clutStomped = false;
            }
            m_draw.clutMask = entries - 1u;
            m_draw.clutValid = true;
            if (s_gsClutVerify || s_gsTexCensus)
            {
                ++g_clutVerify.resolves;
                if (!reuse)
                {
                    ++g_clutVerify.decodes;
                    if (keyMiss) ++g_clutVerify.keyMiss;
                    else if (genMiss) ++g_clutVerify.genMiss;
                    else ++g_clutVerify.stompMiss;
                }
            }
        }
    }

    // ★ The stomp test runs for EVERY draw, not just the textured ones. A draw that samples no
    // texture at all can still write over the page a CACHED palette came from, and the next
    // indexed draw would then reuse a stale one. Marking after the palette has been resolved is
    // deliberate: the current primitive samples the CLUT the texture unit already latched, and
    // only the NEXT read sees the change -- PCSX2's InvalidateRange(.., is_draw = true).
    if (m_draw.clutKey != ~0ull && !m_draw.clutStomped &&
        gs2DrawCoversPage(ctx, m_draw.fbp, m_draw.zbp, m_draw.clutPage))
    {
        m_draw.clutStomped = true;
        if (s_gsClutVerify || s_gsTexCensus)
            ++g_clutVerify.stompsArmed;
    }
    // cont.320: the same test over the cached texture levels (the CLUT test above is unchanged;
    // noteDrawWrites repeats it only for the band-skipped draws that never reach this function).
    if (s_gsTexCache && s_gsTexCacheMutate != 2)
        texStompSlots(ctx, m_draw.fbp, m_draw.zbp); // cont.324b: behind the union rejection (was the plain walk, 2.5% of the bench)

    // ★★★ cont.250 MEASURED AND REVERTED: bumping the pages a draw covers, once per DRAW, cost
    // 127 -> 725 ms on the CPU bench (5.7x) -- a full-screen draw spans ~120 pages per range, two
    // ranges, times 120k primitives = ~29 M contended atomic RMWs per rep across the band threads.
    // clutStomped survives at per-draw granularity because it is ONE page COMPARISON, not 240 atomic
    // increments. The rasterizer's half must therefore accumulate a page RANGE in cheap local state
    // and publish it ONCE PER BATCH -- not per primitive. Until that exists, only the non-rasterizer
    // writers feed g_gs2PageGen, so it is not yet sufficient for render-to-texture invalidation and
    // no cache may rely on it alone. See progress.md cont.250 §16.
}

// ★★ cont.228: the mip-dependent half of the resolved texture state. Everything here was
// recomputed on EVERY texel fetch even though the mip is constant for a primitive.
void GSCpuBackend::resolveTexMip(const GSDrawState &state, uint32_t mip)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    uint32_t mipTbp = tex.tbp0;
    uint32_t mipTbw = tex.tbw;
    int texW = state.textureWidth;
    int texH = state.textureHeight;
    if (mip > 0u)
    {
        const uint64_t m1 = ctx.miptbp1;
        const uint64_t m2 = ctx.miptbp2;
        switch (mip)
        {
        case 1: mipTbp = static_cast<uint32_t>(m1 & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 14) & 0x3Fu); break;
        case 2: mipTbp = static_cast<uint32_t>((m1 >> 20) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 34) & 0x3Fu); break;
        case 3: mipTbp = static_cast<uint32_t>((m1 >> 40) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m1 >> 54) & 0x3Fu); break;
        case 4: mipTbp = static_cast<uint32_t>(m2 & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 14) & 0x3Fu); break;
        case 5: mipTbp = static_cast<uint32_t>((m2 >> 20) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 34) & 0x3Fu); break;
        default: mipTbp = static_cast<uint32_t>((m2 >> 40) & 0x3FFFu); mipTbw = static_cast<uint32_t>((m2 >> 54) & 0x3Fu); break;
        }
        texW = std::max(1, texW >> mip);
        texH = std::max(1, texH >> mip);
    }
    const uint64_t clamp = ctx.clamp;
    m_draw.texTbp = mipTbp;
    m_draw.texTbw = mipTbw;
    m_draw.texW = texW;
    m_draw.texH = texH;
    m_draw.texWf = static_cast<float>(texW);
    m_draw.texHf = static_cast<float>(texH);
    // 16 << mip is a power of two, so the reciprocal is exact and this multiply is bit-identical
    // to the division it replaces.
    m_draw.texFstScale = 1.0f / (16.0f * static_cast<float>(1u << mip));
    m_draw.texWrapU = static_cast<uint8_t>(clamp & 0x3u);
    m_draw.texWrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    if (g_gsWrapCensus)
    {
        g_wrapPair[(m_draw.texWrapU & 3u) * 4u + (m_draw.texWrapV & 3u)].fetch_add(1, std::memory_order_relaxed);
        const unsigned long long n = g_wrapDraws.fetch_add(1, std::memory_order_relaxed) + 1ull;
        if ((n % 2000000ull) == 0ull)
        {
            std::fprintf(stderr, "[gs2:wrapcensus] draws=%llu | ", n);
            static const char *kName[4] = {"REPEAT", "CLAMP", "RGNCLAMP", "RGNREPEAT"};
            for (unsigned u = 0; u < 4; ++u)
                for (unsigned v = 0; v < 4; ++v)
                {
                    const unsigned long long c = g_wrapPair[u * 4 + v].load(std::memory_order_relaxed);
                    if (c) std::fprintf(stderr, "%s/%s=%.1f%% ", kName[u], kName[v], 100.0 * double(c) / double(n));
                }
            std::fprintf(stderr, "\n");
        }
    }
    m_draw.texMinU = static_cast<uint16_t>(((clamp >> 4) & 0x3FFu) >> mip);
    m_draw.texMaxU = static_cast<uint16_t>(((clamp >> 14) & 0x3FFu) >> mip);
    m_draw.texMinV = static_cast<uint16_t>(((clamp >> 24) & 0x3FFu) >> mip);
    m_draw.texMaxV = static_cast<uint16_t>(((clamp >> 34) & 0x3FFu) >> mip);
    // Only plain PSMT4 takes the inlined read: T4HL/T4HH are the SAME 4-bit index but live in the
    // high nibbles of a CT32 page (ReadP4HL/HH use PageTableC32), and the census saw zero of them.
    m_draw.texFastP4 = s_gsFastTex && m_draw.clutValid &&
                       static_cast<uint8_t>(tex.psm & 0x3Fu) == GS_PSM_T4;
    // cont.317: plain PSMT8 the same way (T8H lives in CT32 high bytes -- ReadP8H -- and is not served).
    m_draw.texFastP8 = s_gsFastTex && s_gsFastP8 && m_draw.clutValid &&
                       static_cast<uint8_t>(tex.psm & 0x3Fu) == GS_PSM_T8;
    // ★★ cont.268: hoist the sampler's loop-invariant decisions one level further -- out of
    // sampleTextureT (2.59 M calls per bench replay) and into the per-draw resolve. See the
    // PS2X_GS_TEXWRAP / PS2X_GS_TEXTAP comments for what each selects and why it is bit-exact.
    m_draw.texWrapClass = (!s_gsTexWrap)                                        ? 2
                          : (m_draw.texWrapU == 0u && m_draw.texWrapV == 0u)    ? 0
                          : (m_draw.texWrapU == 1u && m_draw.texWrapV == 1u)    ? 1
                                                                                : 2;
    m_draw.texWrapMaskU = static_cast<uint32_t>(m_draw.texW - 1);
    m_draw.texWrapMaskV = static_cast<uint32_t>(m_draw.texH - 1);
    m_draw.texTapFast = s_gsTexTap && (m_draw.texFastP4 || m_draw.texFastP8) &&
                        !s_gsTexCensus && !s_gsClutDiag && !s_gsTexDiag;
    m_draw.texMip = mip;
    if (s_gsTexCensus)
    {
        // cont.320: this draw's page-generation signature over the resolved level's footprint
        // (the mip level's own base/width) and, for the "full" variant, the CLUT page too.
        Gs2PageMask tm;
        gs2RectPages(tm, mipTbp / 32u, std::max<uint32_t>(mipTbw, 1u), static_cast<uint8_t>(tex.psm & 0x3Fu),
                     0u, 0u, static_cast<uint32_t>(texW - 1), static_cast<uint32_t>(texH - 1));
        g_texCensus.curSigIdx = gs2TexCensusSig(tm) ^ (static_cast<uint64_t>(mip) << 60);
        const uint8_t tpsm = static_cast<uint8_t>(tex.psm & 0x3Fu);
        Gs2PageMask cm;
        if (tpsm == GS_PSM_T4 || tpsm == GS_PSM_T8 || tpsm == GS_PSM_T8H || tpsm == GS_PSM_T4HL || tpsm == GS_PSM_T4HH)
            cm.set(tex.cbp / 32u);
        g_texCensus.curSigFull = g_texCensus.curSigIdx ^ (gs2TexCensusSig(cm) * 0x2545F4914F6CDD1Dull);
        Gs2BlkMask bm;
        gs2RectBlocks(bm, tpsm, mipTbp, std::max<uint32_t>(mipTbw, 1u), 0u, 0u, static_cast<uint32_t>(texW - 1), static_cast<uint32_t>(texH - 1));
        g_texCensus.curSigBlk = gs2TexCensusSigBlk(bm) ^ (static_cast<uint64_t>(mip) << 60);
    }
    texCacheResolve(state); // cont.320: hit / fill / swizzled for this level
}

// ★★ cont.209: BATCH-LEVEL banding. Bands here are FIXED SCREEN STRIPS for the whole run, not
// per-primitive extents, so the one fan-out is amortised over every primitive in the run --
// including the ~95% that were too small for per-primitive fan-out to touch (cont.208 measured
// that cap at ~1.2x).
//
// Ordering is preserved by the same invariant as before: each pixel belongs to exactly ONE band,
// and the thread owning that band walks the run in submission order, so the write sequence for any
// given pixel is identical to single-threaded. Transfers/uploads/presents end a run and are
// therefore natural barriers.
// ★★ Banding by scanline is sound only while a row maps to a UNIQUE VRAM address. That holds
// within one render-target configuration, but NOT across a change of it: cont.209 measured a hash
// divergence bisected to a full-screen sprite whose Z base (zbp=256) is the *frame* base of the
// primitive before it (fbp=256). Two formats over the same memory make row y in one alias a
// different row in the other, so two bands can collide. Runs are therefore split whenever the
// frame or Z target changes, and each sub-run is banded on its own. Target changes are rare
// compared to primitive count, so the amortisation this whole design exists for survives.
#if PS2X_HAS_GS_GPU_DEVICE
// ★ cont.329 phase 2a (the GL renderer arc): turn a run of GSPrimitiveBatch into the flat vertex
// stream + state groups the GL renderer consumes. Untextured this phase; textures are phase 3.
//
// What this does NOT do yet, deliberately: no render-target tracking (every draw goes into the one
// GL target -- phase 4), no blending, no alpha test. The gate for this phase is recognisable
// geometry, not a correct picture.
// ★★★ cont.330 TARGET-FROM-VRAM SEEDING (PS2X_GS_GLR_TGTSEED, default ON).
// A GL render target held only what GL had DRAWN into it. Nothing ever copied guest VRAM back in,
// so a guest UPLOAD into pages the target occupies was invisible to the GL copy -- which is the
// FMV / pre-rendered background 4x-tiling bug (cont.330): the movie picture is uploaded into pages
// 0x180+, the composite samples that address, GL promotes it to the target, and the target does
// not have the picture. PCSX2 handles exactly this with per-Target dirty rects flushed into the
// target before use (GSTextureCache::InvalidateVideoMem / GSTextureCache::Target::Update).
//
// Granularity is ONE PAGE, and that is what makes it safe. Pages 0x180..0x1FF are BOTH this
// target's memory and the game's texture memory, so a whole-target seed would overwrite
// GL-rendered pixels with VRAM bytes GL never wrote. Per page, the rule is exactly the hardware's:
// a page the GUEST wrote since we last seeded it really does hold the guest's bytes now, and a
// page it did not write keeps whatever GL rendered there.
//
// ⚠ Called from the per-draw path, so the common case must be free: the caller gates on the global
// transfer sequence and only reaches here when something was actually transferred.
void GSCpuBackend::glSeedTargetFromVram(const GSDrawState &st, uint64_t targetKey, GsGlBatch &b)
{
    glSeedTargetFromVramParams(st.context.frame.fbp, st.context.frame.fbw, st.context.frame.psm,
                               targetKey, b);
}

// ★★★★ cont.371 (PS2X_GS_GLR_SCANCACHE): the seed scan's memo, per target. After a complete scan
// of a page range, every page in it is either up to date (seen == gen, initialised) or GL-owned
// (drawAtXfer == gen, PAGEOWN only). Neither can turn dirty again without its transfer generation
// moving: the watermark writers only ever store the page's CURRENT generation, and ownership is
// only ever stamped TO the current generation. So while the range's generation sum is unchanged
// (generations only grow, so an equal sum means no bump at all), a rescan emits no rect and changes
// no state -- which is exactly what cont.331o's unconsumed one-shot made every draw after an
// unrelated upload do. Keyed by target; the page range is part of the entry because the target's
// height grows. Only for the owner-honouring scan (draw + present paths), never the source seed.
// thread_local: the scan state it summarises is global, and each thread keeps its own memo.
namespace
{
    struct SeedScanMemo
    {
        uint32_t basePage = 0, nPages = 0;
        uint64_t xferSum = ~0ull;
    };
    thread_local std::unordered_map<uint64_t, SeedScanMemo> t_seedScanMemo;
    unsigned long long g_glSeedScanSkips = 0, g_glSeedScans = 0, g_glSeedScanVerifyBad = 0;
}

void GSCpuBackend::glSeedTargetFromVramParams(uint32_t framefbp, uint32_t framefbw, uint32_t framepsm,
                                              uint64_t targetKey, GsGlBatch &b)
{
    const uint32_t fbw = std::max<uint32_t>(framefbw, 1u);
    const uint32_t psm = framepsm;
    // A page is 8 KB, but its SHAPE depends on the format, and not simply on its bit depth --
    // T8H/T4HL/T4HH live in the unused bits of a 32bpp surface yet page like 8/4bpp. gs2PageGeom
    // is the same table gs2RectPages uses to decide which pages a transfer touched, so the
    // forward and inverse mappings here cannot drift apart.
    uint32_t pageW = 64u, pageH = 32u;
    gs2PageGeom(static_cast<uint8_t>(psm), pageW, pageH);
    // The target's height: the per-target high-water mark the batch extent already tracks. Without
    // one there is nothing to seed into yet -- the target has never been drawn.
    uint32_t h = m_glTargetH.count(targetKey) ? m_glTargetH[targetKey] : 0u;
    if (h == 0u) // present-side instance: fall back to the globally published height
        h = g_glTargetHByPage[targetKey & (kGs2Pages - 1u)].load(std::memory_order_relaxed);
    if (h == 0u || !m_vram)
    {
        ++g_glSeedSkippedNoH;
        return;
    }
    const uint32_t pagesAcross = std::max<uint32_t>(fbw * 64u / pageW, 1u);
    const uint32_t pagesDown = (h + pageH - 1u) / pageH;
    const uint32_t nPages = std::min<uint32_t>(pagesAcross * pagesDown, kGs2Pages);
    const uint32_t basePage = static_cast<uint32_t>(targetKey);
    const uint32_t baseBlock = GSInternal::framePageBaseToBlock(framefbp);

    // cont.371: read the range's generation sum BEFORE the scan, so a bump that lands during it
    // makes the next call's sum differ and rescan.
    SeedScanMemo *memo = nullptr;
    uint64_t xferSum = 0;
    if (s_glScanCache && t_seedHonourOwn)
    {
        xferSum = gs2WindowGenSum(g_gs2PageXferGen, g_gs2PageXferGenChunk, basePage, nPages);
        memo = &t_seedScanMemo[targetKey];
        if (memo->basePage == basePage && memo->nPages == nPages &&
            (memo->xferSum == xferSum || (s_glScanMutate && memo->xferSum != ~0ull)))
        {
            ++g_glSeedScanSkips;
            if (s_glScanVerify)
            {
                // Read-only replay of the loop's decisions: a page the full scan would have acted
                // on (seeded, or first-observed) is a skip that changed behaviour.
                for (uint32_t i = 0; i < nPages; ++i)
                {
                    const uint32_t pg = (basePage + i) & (kGs2Pages - 1u);
                    const uint32_t gen = g_gs2PageXferGen[pg].load(std::memory_order_relaxed);
                    if (g_glSeedSeenByPage[pg].load(std::memory_order_relaxed) == gen &&
                        g_glSeedSeenInit[pg].load(std::memory_order_relaxed) != 0u)
                        continue;
                    if (s_glPageOwn && g_glPageDrawAtXfer[pg].load(std::memory_order_relaxed) == gen)
                        continue;
                    ++g_glSeedScanVerifyBad;
                    break;
                }
                if ((g_glSeedScanSkips & ((1ull << 20) - 1ull)) == 0ull)
                    std::fprintf(stderr, "[gs2:scanverify] memo-skips=%llu scans=%llu skip-bad=%llu sums=%llu sum-bad=%llu\n",
                                 g_glSeedScanSkips, g_glSeedScans, g_glSeedScanVerifyBad,
                                 g_glScanSumChecks.load(), g_glScanSumBad.load());
            }
            return;
        }
        ++g_glSeedScans;
    }

    unsigned dirty = 0;
    for (uint32_t i = 0; i < nPages; ++i)
    {
        const uint32_t pg = (basePage + i) & (kGs2Pages - 1u);
        const uint32_t gen = g_gs2PageXferGen[pg].load(std::memory_order_relaxed);
        // ★ cont.331: global, per-page watermark (see g_glSeedSeenByPage) -- the per-instance map
        // is invisible to the present-side backend.
        if (g_glSeedSeenByPage[pg].load(std::memory_order_relaxed) == gen &&
            g_glSeedSeenInit[pg].load(std::memory_order_relaxed) != 0u)
            continue;
        // ★★★★★ cont.331l: GL drew into this page at or after the newest transfer, so GL holds the
        // fresher copy -- seeding it would overwrite this frame's own geometry (the missing UI
        // text). Leave the watermark alone so a genuinely newer transfer is still seeded later.
        if (s_glPageOwn && t_seedHonourOwn &&
            g_glPageDrawAtXfer[pg].load(std::memory_order_relaxed) == gen)
        {
            ++g_glSeedOwnedSkips;
            continue;
        }
        const bool first = g_glSeedSeenInit[pg].exchange(1u, std::memory_order_relaxed) == 0u;
        g_glSeedSeenByPage[pg].store(gen, std::memory_order_relaxed);
        // The FIRST observation of a target only records the baseline. Seeding then would copy
        // whatever happened to be in VRAM before GL ever rendered, which is not an improvement --
        // and on this game's texture memory it is actively wrong.
        if (first)
            continue;
        const uint32_t x0 = (i % pagesAcross) * pageW;
        const uint32_t y0 = (i / pagesAcross) * pageH;
        if (y0 >= h)
            continue;
        const uint32_t rh = std::min<uint32_t>(pageH, h - y0);
        GsGlTargetSeed sd;
        sd.x = x0;
        sd.y = y0;
        sd.w = pageW;
        sd.h = rh;
        sd.rgba.resize(static_cast<size_t>(pageW) * rh * 4u);
        for (uint32_t yy = 0; yy < rh; ++yy)
        {
            for (uint32_t xx = 0; xx < pageW; ++xx)
            {
                const uint32_t c = ReadVramUnlocked(psm, baseBlock, fbw, x0 + xx, y0 + yy);
                uint8_t *o = sd.rgba.data() + (static_cast<size_t>(yy) * pageW + xx) * 4u;
                o[0] = static_cast<uint8_t>(c & 0xFFu);
                o[1] = static_cast<uint8_t>((c >> 8) & 0xFFu);
                o[2] = static_cast<uint8_t>((c >> 16) & 0xFFu);
                // Draws never write alpha into a target (the device masks it), and the clear puts
                // 1.0 there; a promoted CT24 read takes its alpha from TEXA in the shader anyway.
                o[3] = 0xFFu;
                // ★ cont.330b PS2X_GS_GLR_SEEDTEST=1 (diagnostic): replace the decoded VRAM with a
                // constant magenta. The backend proves it decodes real, non-zero picture data and
                // the device proves the blit runs with a COMPLETE framebuffer and no GL error, yet
                // the target reads back pure black. A constant source separates the two halves:
                // magenta in the target means the blit lands and the source data is the problem;
                // still black means the blit or the readback is.
                if (s_glSeedTest)
                {
                    o[0] = 0xFFu;
                    o[1] = 0x00u;
                    o[2] = 0xFFu;
                }
            }
        }
        b.targetSeeds.push_back(std::move(sd));
        ++dirty;
    }
    if (memo)
    {
        memo->basePage = basePage;
        memo->nPages = nPages;
        memo->xferSum = xferSum;
    }
    if (dirty != 0u)
    {
        ++g_glSeedEvents;
        g_glSeedPages += dirty;
        // ★ cont.330: WHICH target, and is the VRAM we copied actually non-zero? "Seeding fired
        // 114 pages/event" was true while target 0x180 stayed black, which is consistent with
        // three different faults (wrong target, zero bytes, or a blit that never lands). Name the
        // target and count the non-zero bytes so those separate.
        if (g_glSeedEvents <= 40ull || (g_glSeedEvents % 512ull) == 0ull)
        {
            size_t nz = 0, tot = 0;
            for (const GsGlTargetSeed &q : b.targetSeeds)
            {
                tot += q.rgba.size();
                for (size_t k = 0; k + 3 < q.rgba.size(); k += 4)
                    if (q.rgba[k] || q.rgba[k + 1] || q.rgba[k + 2])
                        ++nz;
            }
            std::fprintf(stderr,
                         "[gs2:glrseed/t] target=0x%llx h=%u fbw=%u psm=0x%02x dirty=%u rects=%zu "
                         "nonzero-texels=%zu/%zu\n",
                         (unsigned long long)targetKey, h, fbw, psm, dirty, b.targetSeeds.size(),
                         nz, tot / 4);
        }
        if ((g_glSeedEvents % 512ull) == 0ull)
            std::fprintf(stderr, "[gs2:glrseed] events=%llu pages=%llu (%.1f pages/event) "
                                 "skipped-no-height=%llu scans=%llu memo-skips=%llu verify{skip-bad=%llu sums=%llu sum-bad=%llu}\n",
                         g_glSeedEvents, g_glSeedPages,
                         double(g_glSeedPages) / double(g_glSeedEvents), g_glSeedSkippedNoH,
                         g_glSeedScans, g_glSeedScanSkips, g_glSeedScanVerifyBad,
                         g_glScanSumChecks.load(), g_glScanSumBad.load());
    }
}

uint64_t GSCpuBackend::glTexContentHash(uint32_t pageBase, uint32_t pages)
{
    // The palette is part of the content: the same indices through a different CLUT are a
    // different image, and this game re-decodes its CLUT every draw precisely because a draw can
    // stomp the blocks it samples from.
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    if (m_vram && m_vramSize >= 8192u)
    {
        const uint32_t pageCount = m_vramSize / 8192u;
        for (uint32_t p = 0; p < pages && p < 64u; ++p)
        {
            const uint32_t pg = (pageBase + p) % pageCount;
            const uint8_t *src = m_vram + static_cast<size_t>(pg) * 8192u;
            uint64_t w = 0;
            for (uint32_t i = 0; i < 8192u; i += 8u)
            {
                std::memcpy(&w, src + i, sizeof(w)); // unaligned-safe, and the compiler folds it
                mix(w);
            }
        }
    }
    for (uint32_t i = 0; i < 256u; ++i)
        mix(m_draw.clut[i]);
    return h;
}

// ★★★★ cont.331 AUTHORITY MODEL phase A. A GL target is authoritative for its VRAM pages only
// while GL is the one writing them. When something ELSE writes those pages -- a host->local
// upload, a poke, a blit -- the target is stale and must take that content back before it is used.
// The existing seed only ran inside the DRAW path, keyed on m_glTargetSeedSeq[drawTargetKey], so a
// target the guest rarely draws into never got its dirty rects: during an FMV the movie arrives as
// 16x16 macroblock uploads into the DISPLAY buffer while the guest draws only a fade over it, so
// the display target was dirtied every frame and seeded 4 times in a whole run (cont.330h).
// This applies the same dirty rects at the other place a target is USED: the present resolve.
// ★★★★ cont.345: at the flip-snapshot point, bring the pre-flip DISPLAY target(s) up to date with
// any non-draw VRAM writes (the seed the presenter used to apply at present time, now applied here
// on the worker, in FIFO order) and ask the device to resolve them under this snapshot's seq.
std::atomic<uint64_t> g_ps2xGsPreferredDisplaySource{0ull}; // cont.345, see gs_cpu_backend.h
std::atomic<uint64_t> g_ps2xGsLiveDispfb1{0ull}, g_ps2xGsLiveDispfb2{0ull}; // cont.345

void GSCpuBackend::glSnapshotDisplayTargets(uint64_t dispfb1, uint64_t dispfb2, uint64_t seq)
{
#if PS2X_HAS_GS_GPU_DEVICE
    if (!s_gsRendererGl || !s_glSnapResolve || seq == 0ull)
        return;
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev)
        return;
    const GSFrameReg f1 = decodeDisplayFrame(dispfb1);
    const GSFrameReg f2 = decodeDisplayFrame(dispfb2);
    glSeedDisplayTargetForPresent(f1.fbp, f1.fbw, f1.psm);
    dev->SnapshotFrameGl(static_cast<uint64_t>(f1.fbp), seq);
    if (f2.fbp != f1.fbp)
    {
        glSeedDisplayTargetForPresent(f2.fbp, f2.fbw, f2.psm);
        dev->SnapshotFrameGl(static_cast<uint64_t>(f2.fbp), seq);
    }
    // The presenter shows the full-screen copy sprite's SOURCE instead of the display buffer when
    // one is armed for this display (copySource's preferred path); resolve that source here too, or
    // those presents (the whole movie era, 394 of ~2,800 presents on the 1010 replay) fall back to
    // the racing live resolve.
    const uint64_t pref = g_ps2xGsPreferredDisplaySource.load(std::memory_order_acquire);
    if (pref >> 63)
    {
        const uint32_t dest = static_cast<uint32_t>(pref & 0xFFFFu);
        const uint32_t src = static_cast<uint32_t>((pref >> 16) & 0xFFFFu);
        const uint32_t sbw = static_cast<uint32_t>((pref >> 32) & 0xFFu);
        const uint32_t spsm = static_cast<uint32_t>((pref >> 40) & 0xFFu);
        if ((dest == f1.fbp || dest == f2.fbp) && src != f1.fbp && src != f2.fbp)
        {
            glSeedDisplayTargetForPresent(src, sbw, spsm);
            dev->SnapshotFrameGl(static_cast<uint64_t>(src), seq);
        }
    }
#else
    (void)dispfb1; (void)dispfb2; (void)seq;
#endif
}

void GSCpuBackend::glSeedDisplayTargetForPresent(uint32_t fbp, uint32_t fbw, uint32_t psm)
{
    if (!s_glAuthority)
        return;
    ++g_psCalls;
    if (!m_vram)
    { ++g_psNoVram; return; }
    const uint64_t targetKey = static_cast<uint64_t>(fbp);
    g_psLastFbp = fbp; g_psLastFbw = fbw; g_psLastPsm = psm;
    // Only when a non-draw transfer has happened since this target was last brought up to date --
    // the same global transfer sequence the draw-path seed uses, tracked per target.
    const unsigned long long xseq = g_gs2XferSeq.load(std::memory_order_relaxed);
    unsigned long long &seenSeq = m_glPresentSeedSeq[targetKey];
    if (seenSeq == xseq)
    { ++g_psSeqSame; return; }
    seenSeq = xseq;
    GsGlBatch sb;
    sb.targetKey = targetKey;
    glSeedTargetFromVramParams(fbp, fbw, psm, targetKey, sb);
    if (sb.targetSeeds.empty())
    { ++g_psNoRects; return; }
    // The target's extent, as the draw path records it; the device needs it to size the target.
    sb.refW = std::max<uint32_t>(fbw, 1u) * 64u;
    sb.refH = m_glTargetH.count(targetKey) ? m_glTargetH[targetKey] : 0u;
    if (sb.refH == 0u)
        sb.refH = g_glTargetHByPage[targetKey & (kGs2Pages - 1u)].load(std::memory_order_relaxed);
    if (sb.refH == 0u)
    { ++g_psNoH; return; }
    ++g_glPresentSeedEvents;
    g_glPresentSeedRects += sb.targetSeeds.size();
    // Any draws still pending for this target must land BEFORE the seed, or the seed would
    // overwrite them; and the seed must be in before the resolve reads the target back.
    glFlushPending();
    if (GsGpuPresentDevice *dev = gs2GpuDevice())
        dev->DrawBatchGl(std::move(sb));
}

void GSCpuBackend::glSubmitRun(const GSPrimitiveBatch *items, size_t count)
{
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev || count == 0u)
        return;
    // ★★★ cont.329h THE RENDER-TARGET SPLIT THE GL PATH NEVER GOT.
    // `RasterRunFanOut` splits a run on its target before rasterizing -- but that split is BELOW
    // the `s_gsSkipCpuRaster` early return, so only the CPU rasterizer ever sees it. A run that
    // mixes two framebuffers therefore had ALL of its draws assigned to items[0]'s target, which
    // puts a post-process chain's intermediate passes (invert, fill, additive blur) straight onto
    // the VISIBLE frame. The per-draw trace caught exactly that: a 6-vertex full-screen quad with
    // blend (Cs-Cd)*FIX taking the target from luma 19 to 236, then another to 255 pure white.
    // PS2X_GS_GLR_TGTSPLIT=1 gives the GL path the same split, on the ADDRESS alone (the target
    // key is the address by design -- the same buffer is written as CT32 and CT24).
    {
        // ★★★ cont.329k: split on (fbp, FBW), not on fbp alone. `refW = fbw * 64` is the extent the
        // vertex transform divides by, and it is taken from items[0] -- so a run that mixes two
        // FBW values under one address transforms every later draw by the WRONG width. Measured:
        // the FMV/background screens drew their full-screen sprite into a ~1/5-width strip,
        // because it shared a run with draws of a much larger FBW.
        auto sameTarget = [](const GSPrimitiveBatch &a, const GSPrimitiveBatch &b2)
        {
            return a.state.context.frame.fbp == b2.state.context.frame.fbp &&
                   a.state.context.frame.fbw == b2.state.context.frame.fbw;
        };
        bool mixed = false;
        const uint32_t k0 = items[0].state.context.frame.fbp;
        for (size_t i = 1; i < count; ++i)
            if (!sameTarget(items[i], items[0])) { mixed = true; break; }
        (void)k0;
        if (mixed && g_glCensus)
        {
            ++g_glcMixedRuns;
            for (size_t i = 0; i < count; ++i)
                if (!sameTarget(items[i], items[0]))
                    ++g_glcMixedItems;
        }
        if (mixed && s_glTgtSplit)
        {
            ++g_glcSplitRuns;
            size_t start = 0;
            for (size_t i = 1; i <= count; ++i)
                if (i == count || !sameTarget(items[i], items[start]))
                {
                    glSubmitRun(items + start, i - start);
                    start = i;
                }
            return;
        }
    }
    if (g_glCensus) ++g_glcRuns;
    GsGlBatch b;
    b.verts.reserve(count * 6u);
    b.groups.reserve(16u);

    auto zScale = [](uint32_t zpsm) -> float
    {
        switch (zpsm & 0xFu)
        {
        case 0: return 4294967295.0f; // PSMZ32
        case 2: return 65535.0f;      // PSMZ16 (and 16S)
        default: return 16777215.0f;  // PSMZ24 -- 100% of this game's draws (cont.329 census)
        }
    };

    for (size_t i = 0; i < count; ++i)
    {
        const GSPrimitiveBatch &it = items[i];
        if (g_glCensus) ++g_glcPrimIn;
        if (it.vertexCount == 0u)
        {
            if (g_glCensus) ++g_glcPrimZero;
            continue;
        }
        const GSDrawState &st = it.state;
        const bool isSprite = (st.prim.type == GS_PRIM_SPRITE) && it.vertexCount >= 2u;
        const bool isTri = it.vertexCount >= 3u;
        // ★ cont.329 phase 4f: WHAT DOES THE SOFTWARE PATH DRAW THAT WE DO NOT?
        // With the depth test disabled entirely the GL path still reaches only 16115 distinct
        // colours against the software path's 28036, so ~40% of the scene is absent for reasons
        // unrelated to depth. This converter silently drops any primitive shape it does not
        // recognise, which is exactly the kind of gap that never shows up in an error count.
        // Counted per primitive TYPE, translated vs dropped, so the answer is a table.
        if (g_glPrimCensus)
        {
            const unsigned ty = static_cast<unsigned>(st.prim.type) & 7u;
            ++g_glPrimSeen[ty];
            if (!isSprite && !isTri)
                ++g_glPrimDropped[ty];
            if ((++g_glPrimN % 8000000ull) == 0ull)
            {
                static const char *kName[8] = {"POINT", "LINE", "LINESTRIP", "TRI",
                                               "TRISTRIP", "TRIFAN", "SPRITE", "INVALID"};
                std::fprintf(stderr, "[gs2:glprim] after %llu prims:\n", g_glPrimN);
                for (unsigned k = 0; k < 8u; ++k)
                    if (g_glPrimSeen[k])
                        std::fprintf(stderr, "[gs2:glprim]   %-10s seen=%llu dropped=%llu (%.1f%%)\n",
                                     kName[k], g_glPrimSeen[k], g_glPrimDropped[k],
                                     100.0 * double(g_glPrimDropped[k]) / double(g_glPrimSeen[k]));
            }
        }
        if (!isSprite && !isTri)
        {
            if (g_glCensus) ++g_glcPrimDrop;
            continue; // points and lines are not translated this phase
        }
        if (g_glCensus) ++g_glcPrimXlat;

        // TEST bits: ATE 0, ATST 1-3, AREF 4-11, AFAIL 12-13, DATE 14, DATM 15, ZTE 16, ZTST 17-18.
        const uint64_t test = st.context.test;
        const uint8_t zte = static_cast<uint8_t>((test >> 16) & 1u);
        const uint8_t ztst = static_cast<uint8_t>((test >> 17) & 3u);
        const uint8_t zmsk = static_cast<uint8_t>(st.context.zbuf.zmask ? 1u : 0u);
        if (g_glZCensus) { if (zmsk) ++g_glZMsk1; else ++g_glZMsk0; }
        const uint8_t ate = static_cast<uint8_t>(test & 1u);
        const uint8_t atst = static_cast<uint8_t>((test >> 1) & 7u);
        const uint8_t aref = static_cast<uint8_t>((test >> 4) & 0xFFu);
        const uint8_t afail = static_cast<uint8_t>((test >> 12) & 3u);
        const GSScissorReg &sc = st.context.scissor;
        // ★ cont.329 phase 4: which render target this draw writes. The game double-buffers, so
        // the same scene alternates between two framebuffer addresses; merging them into one GL
        // target composited two frames and washed the picture out.
        // ★ KEY ON THE BUFFER ADDRESS ALONE. Including fbw/psm split one buffer into several
        // targets: the game writes the same memory as both CT32 and CT24 (keys ...8000 and ...8001
        // appeared for one buffer), and the DISPLAY register reports a different psm again, so the
        // presenter matched nothing and resolved an empty target -- a black screen. The address is
        // the identity; the format is how it is interpreted.
        const uint64_t drawTargetKey = uint64_t(st.context.frame.fbp);
        m_glTargetKeys.insert(drawTargetKey);
        // ★★★ cont.330: before drawing INTO this target, fold in any page of it the GUEST has
        // written since we last looked (see glSeedTargetFromVram). Gated on a single global
        // transfer counter so the per-draw cost when nothing was transferred is one atomic load.
        if (s_glTgtSeed)
        {
            const unsigned long long xseq = g_gs2XferSeq.load(std::memory_order_relaxed);
            unsigned long long &seenSeq = m_glTargetSeedSeq[drawTargetKey];
            if (seenSeq != xseq)
            {
                // A seed must not overtake draws already queued for this target: flush them first,
                // so the seed lands at the head of a fresh batch and this run's draws follow it.
                const size_t before = b.targetSeeds.size();
                glSeedTargetFromVram(st, drawTargetKey, b);
                // ★★★★★ cont.331o: consume the one-shot ONLY when the target's own pages really
                // were dirty. Stamping `seenSeq = xseq` up front burned it on the FIRST draw into
                // the target after ANY transfer anywhere -- and this game's frame order is
                //     darken pass -> movie upload -> UI glyphs -> flip,
                // so the one-shot was spent on the darken pass, when this target's pages were not
                // dirty yet, and the upload that followed was then never seeded before the glyphs.
                // The darken pass is `Cd*(1-As)` with As = TEXA/128 = 64/128 = 0.5 exactly
                // ([gs2:asprobe]: tbp=12288 CT24 ALPHA{A=2 B=1 C=0 D=1} tcc=1 tfx=1 TEXA{ta0=64}),
                // so with nothing re-supplying the picture the target halves every frame: DUMPTGT
                // read fbp=0x0 back at max=2, i.e. the clear colour (max 36) halved four times,
                // while the frames that DID get a seed read max=238.
                // Not consuming it costs a re-scan of the target's page generations on the next
                // run (atomic loads, no rects built), and buys the seed landing between the upload
                // and the frame's own draws -- which is the order the hardware has.
                if (b.targetSeeds.size() != before)
                {
                    seenSeq = xseq;
                    if (!m_glPendVerts.empty())
                        glFlushPending();
                }
            }
        }
        // This draw makes the GL target the freshest copy of those bytes; remember the VRAM
        // generation it corresponds to (see m_glTargetGen).
        if (s_glTgtFresh)
            m_glTargetGen[drawTargetKey] = gs2TargetGenSum(drawTargetKey);
        // ★ cont.329 phase 3: resolve and decode the texture. resolveDraw/resolveTexMip are the
        // CPU rasterizer's own per-draw resolves, so the CLUT, the mip base and the wrap class come
        // from the same source of truth rather than a second implementation that can drift.
        uint64_t texKey = 0u;
        uint8_t wrapU = 0, wrapV = 0, tfx = 0, tcc = 1, lin = 1, texIsTarget = 0;
        uint8_t srcMode = 0, srcTexa = 0x80, srcAem = 0;
        if (st.prim.tme)
            ++g_glTexDrawsTme;
        else
            ++g_glTexDrawsPlain;
        if (st.prim.tme)
        {
            resolveDraw(st);
            resolveTexMip(st, 0u);
            const GSTex0Reg &t0 = st.context.tex0;
            // ★★★★★ cont.331p: count (and optionally ablate) every draw that samples the watched
            // texture base, at the point the draw's texture is resolved.
            if (s_glDropTbp != 0u && uint32_t(m_draw.texTbp) == s_glDropTbp)
            {
                ++g_glDropTbpSeen;
                if ((g_glDropTbpSeen % 512ull) == 0ull)
                    std::fprintf(stderr,
                                 "[gs2:droptbp] tbp=%u seen=%llu killed=%llu (flip=%llu)\n",
                                 s_glDropTbp, g_glDropTbpSeen, g_glDropTbpKilled,
                                 g_perfFlips.load(std::memory_order_relaxed));
                if (s_glDropTbpKill)
                { ++g_glDropTbpKilled; continue; }
            }
            // ★ cont.329 phase 4b: is this texture a buffer we RENDERED into? TBP0 counts 256-byte
            // blocks and a framebuffer address counts 8 KB pages, so the target address in blocks
            // is `fbp << 5`. Sampling the target we are currently drawing into would be a feedback
            // loop, so that case falls through to the ordinary decode rather than being promoted.
            const uint64_t texAsTargetKey = uint64_t(m_draw.texTbp) >> 5;
            // ★★★ cont.329j: NEVER promote a PALETTED read. The guest's ~200,000 page-0x100 draws
            // read that buffer as PSMT4HL -- a 4-bit index taken from the ALPHA byte, through the
            // CLUT (PCSX2's "channel shuffle"), which a GL colour texture cannot answer. And it
            // does not have to: page 0x100 receives only ~0.9 draws per frame, so its content is
            // UPLOADED, and uploads land in m_vram in both renderers -- the ordinary VRAM decode
            // gets the real thing. Measured (CPU path, PS2X_GS_T4HLPROBE): 16 distinct indices ->
            // 16 distinct colours over a 1024-texel grid, i.e. a real 16-level modulation map,
            // which is why promoting it (or sampling it neutral) loses per-pixel colour on every
            // lit surface. PS2X_GS_GLR_PALTGT=1 restores the old, wrong promotion for an A/B.
            const uint32_t srcPsm = st.context.tex0.psm;
            const bool srcPaletted = (srcPsm == GS_PSM_T4 || srcPsm == GS_PSM_T8 ||
                                      srcPsm == GS_PSM_T4HL || srcPsm == GS_PSM_T4HH ||
                                      srcPsm == GS_PSM_T8H);
            // ⚠ And only while that target is still the freshest copy of those bytes: an upload
            // into the same address after our last draw means VRAM holds the real image and the
            // GL target is stale (or empty).
            bool tgtFresh = true;
            if (s_glTgtFresh)
            {
                auto gi = m_glTargetGen.find(texAsTargetKey);
                tgtFresh = (gi != m_glTargetGen.end()) &&
                           gi->second == gs2TargetGenSum(texAsTargetKey);
            }
            // ★ cont.330d: the diagnostic refusal. Counted, so a null result is distinguishable
            // from a knob that never fired.
            const bool promoteRefused =
                (s_glNoPromote == 0xFFFFFFFEu) || (uint64_t(s_glNoPromote) == texAsTargetKey);
            if (promoteRefused && m_glTargetKeys.count(texAsTargetKey) != 0u)
                ++g_glPromoteRefused;
            if (!promoteRefused && m_glTargetKeys.count(texAsTargetKey) != 0u &&
                texAsTargetKey != drawTargetKey &&
                (uint64_t(m_draw.texTbp) & 31u) == 0u && (!srcPaletted || s_glPalTgt) && tgtFresh)
            {
                // ★★★★★ cont.331n: bring the SOURCE target up to date from VRAM before sampling
                // it. Own batch for the source key, ahead of the batch that reads it; the pending
                // batch is flushed first so nothing is reordered past the seed.
                if (s_glSrcSeed && s_glAuthority && m_vram)
                {
                    ++g_srcSeedCalls;
                    const unsigned long long xseq = g_gs2XferSeq.load(std::memory_order_relaxed);
                    unsigned long long &seenSrc = g_glSrcSeedSeq[texAsTargetKey];
                    if (seenSrc == xseq)
                        ++g_srcSeedSeqSame;
                    else
                    {
                        seenSrc = xseq;
                        GsGlBatch sb;
                        sb.targetKey = texAsTargetKey;
                        t_seedHonourOwn = false; // a source seed asks a different question
                        glSeedTargetFromVramParams(static_cast<uint32_t>(texAsTargetKey),
                                                   st.context.tex0.tbw, st.context.tex0.psm,
                                                   texAsTargetKey, sb);
                        t_seedHonourOwn = true;
                        if (sb.targetSeeds.empty())
                            ++g_srcSeedNoRects;
                        else
                        {
                            sb.refW = std::max<uint32_t>(st.context.tex0.tbw, 1u) * 64u;
                            sb.refH = m_glTargetH.count(texAsTargetKey) ? m_glTargetH[texAsTargetKey] : 0u;
                            if (sb.refH == 0u)
                                sb.refH = g_glTargetHByPage[texAsTargetKey & (kGs2Pages - 1u)]
                                              .load(std::memory_order_relaxed);
                            if (sb.refH == 0u)
                                ++g_srcSeedNoH;
                            else
                            {
                                ++g_srcSeedEvents;
                                g_srcSeedRects += sb.targetSeeds.size();
                                glFlushPending();
                                if (GsGpuPresentDevice *sdev = gs2GpuDevice())
                                    sdev->DrawBatchGl(std::move(sb));
                            }
                        }
                    }
                }
                texKey = texAsTargetKey ? texAsTargetKey : 1u;
                texIsTarget = 1;
                // How the guest asked for this target to be READ (see GsGlGroup::srcMode).
                const uint32_t spsm = st.context.tex0.psm;
                if (spsm == GS_PSM_CT24 && s_glSrcTexa)
                    srcMode = 1;
                else if (s_glSrcNeutral &&
                         (spsm == GS_PSM_T4 || spsm == GS_PSM_T8 || spsm == GS_PSM_T4HL ||
                          spsm == GS_PSM_T4HH || spsm == GS_PSM_T8H))
                    srcMode = 2;
                srcTexa = static_cast<uint8_t>(st.texa.ta0);
                srcAem = static_cast<uint8_t>(st.texa.aem ? 1u : 0u);
                wrapU = static_cast<uint8_t>(m_draw.texWrapU == 0u ? 0u : 1u);
                wrapV = static_cast<uint8_t>(m_draw.texWrapV == 0u ? 0u : 1u);
                tfx = static_cast<uint8_t>(st.context.tex0.tfx & 3u);
                tcc = static_cast<uint8_t>(st.context.tex0.tcc ? 1u : 0u);
                lin = static_cast<uint8_t>(st.linearFilter ? 1u : 0u);
                ++g_glTexFromTarget;
            }
            // ★★★ cont.330 TEXTRACE, the PROMOTION half. The decode trace below only runs when a
            // draw was NOT promoted, so a promoted read printed nothing at all -- exactly the case
            // the FMV bug turns on. Print every input to the decision, so "why did TGTFRESH not
            // stop this" is answered by the numbers rather than by reading the rule.
            if (s_glTexTrace != 0u && m_draw.texTbp == s_glTexTrace)
            {
                const uint64_t tk = uint64_t(m_draw.texTbp) >> 5;
                const auto gi = m_glTargetGen.find(tk);
                std::fprintf(stderr,
                             "[gs2:textrace/src] tbp=%u psm=0x%02x %dx%d -> tgtKey=0x%llx known=%d "
                             "aligned=%d paletted=%d gen{stamped=%lld live=%llu} PROMOTED=%d "
                             "| drawn-into fbp=0x%llx\n",
                             m_draw.texTbp, st.context.tex0.psm, int(m_draw.texW), int(m_draw.texH),
                             (unsigned long long)tk,
                             m_glTargetKeys.count(tk) != 0u ? 1 : 0,
                             (uint64_t(m_draw.texTbp) & 31u) == 0u ? 1 : 0, srcPaletted ? 1 : 0,
                             gi != m_glTargetGen.end() ? (long long)gi->second : -1ll,
                             (unsigned long long)gs2TargetGenSum(tk), texIsTarget ? 1 : 0,
                             (unsigned long long)drawTargetKey);
            }
            const uint32_t tpsmForSpan = st.context.tex0.psm;
            const uint32_t tw = static_cast<uint32_t>(m_draw.texW);
            const uint32_t th = static_cast<uint32_t>(m_draw.texH);
            // ★ cont.329 phase 3d: WHY a textured draw falls back to untextured. The fallback
            // paints flat vertex colour, so a large quad whose texture was rejected reads as a
            // solid white background -- which is exactly the symptom. Counted by reason so the
            // cause is a number rather than a guess.
            if (tw == 0u || th == 0u)
                ++g_glTexRejZero;
            else if (tw > 1024u || th > 1024u)
                ++g_glTexRejBig;
            if (!texIsTarget && tw != 0u && th != 0u && tw <= 1024u && th <= 1024u)
            {
                // The generation makes a CHANGED texture a different key, so a stale decode can
                // never be served. ⚠ g_gs2PageGen covers uploads/transfers/clears but NOT draw
                // writes (gs_cpu_backend.cpp), so render-to-texture is still not invalidated --
                // that is phase 4, and it is why this is not yet a correct texture cache.
                // ⚠ The page span must be computed from the texture's BITS PER TEXEL. Build 804
                // used a fixed >>11 (the 32bpp span) for every format, so a 256x256 T4 texture --
                // 4 pages -- hashed 32 pages of generations. Any upload anywhere near it changed
                // the key, so EVERY draw decoded a fresh texture: 2279 uploads for 2280 binds,
                // zero cache hits, and the run crawled to 101 hero frames instead of ~2600.
                const bool paletted_ = (tpsmForSpan == GS_PSM_T4 || tpsmForSpan == GS_PSM_T8 ||
                                        tpsmForSpan == GS_PSM_T4HL || tpsmForSpan == GS_PSM_T4HH ||
                                        tpsmForSpan == GS_PSM_T8H);
                uint32_t bits = 32u;
                switch (tpsmForSpan)
                {
                case GS_PSM_T4: bits = 4u; break;
                case GS_PSM_T8: bits = 8u; break;
                case GS_PSM_CT16:
                case GS_PSM_CT16S: bits = 16u; break;
                default: bits = 32u; break; // CT32/CT24, and T4HL/T4HH/T8H which live in the
                                            // unused bits of a 32bpp surface, so span it as 32.
                }
                const uint64_t texBytes = (static_cast<uint64_t>(tw) * th * bits) / 8ull;
                const uint32_t pages = static_cast<uint32_t>(texBytes / 8192ull) + 1u;
                uint64_t gen = 0u;
                const uint32_t pageBase = m_draw.texTbp >> 5;
                for (uint32_t pg = 0; pg < pages && pg < 64u; ++pg)
                    gen += g_gs2PageGen[(pageBase + pg) & 511u].load(std::memory_order_relaxed);
                // ★★★ cont.329k THE PALETTE MUST BE PART OF THE GENERATION, NOT JUST THE HASH.
                // The content hash covers the CLUT -- but it is only consulted when the generation
                // MOVES, and the generation summed only the texture's own pages. A CLUT lives at
                // `cbp`, on a different page, so re-uploading a new palette to the same cbp moved
                // nothing: the cache kept serving the first decode, with its FROZEN palette, for a
                // game whose textures are 99% paletted (T4 68.3% + T8 31.4%).
                // PS2X_GS_GLR_CLUTGEN=0 restores the old behaviour for an A/B.
                if (paletted_ && s_glClutGen)
                {
                    const uint32_t clutPage = (t0.cbp >> 5) & 511u;
                    gen += g_gs2PageGen[clutPage].load(std::memory_order_relaxed);
                    gen += g_gs2PageGen[(clutPage + 1u) & 511u].load(std::memory_order_relaxed);
                }

                // ★ cont.329 phase 3b: THE GENERATION IS A HINT, NOT THE KEY.
                // Measured: 13312 decodes produced THREE distinct textures -- 100% of the work was
                // the same images re-decoded because their page generation moved. Folding the
                // generation straight into the key made every bump a cache miss, and a bump means
                // only "these pages were written", not "this texture changed".
                // So: the generation gates a cheap CONTENT HASH over the texture's own bytes plus
                // its palette, and only a changed hash forces a decode. Hashing ~33 KB linearly is
                // far cheaper than 66k swizzled texel reads with a palette lookup each. This is
                // PCSX2's shape too (GSTextureCache keeps per-source hashes alongside its dirty
                // tracking rather than trusting the dirty flag alone).
                // ★★★★★ cont.332l THE KEY MUST BE INJECTIVE, AND THE OLD ONE WAS NOT.
                // It XORed shifted fields, and `cbp << 44` overlapped `tbp << 40` -- so
                // (tbp, cbp) and (tbp ^ (d << 4), cbp ^ d) landed on the SAME entry. That is not
                // theoretical: PS2X_GS_GLR_KEYCENSUS found FOUR distinct 128x128 PSMT8 character
                // textures on one key (tbp 12291/12355/12419/12483 with cbp 12579/12583/12587/
                // 12591), i.e. one GL object shared by four textures with last-upload-wins. It is
                // the pale, hue-drained character on the character-select screen (the draw's own
                // fixed-texel probe did not match its own decode), and it also explains the
                // cont.332 re-decode thrash: colliding entries invalidate each other forever.
                // Disjoint bit ranges instead -- tbp 14 | cbp 14 | tbw 6 | psm 6 | tw 11 | th 11.
                const uint64_t baseKey =
                    g_glTexKeyPacked
                        ? ((uint64_t(m_draw.texTbp) & 0x3FFFull) |
                           ((uint64_t(t0.cbp) & 0x3FFFull) << 14) |
                           ((uint64_t(m_draw.texTbw) & 0x3Full) << 28) |
                           ((uint64_t(t0.psm) & 0x3Full) << 34) |
                           ((uint64_t(tw) & 0x7FFull) << 40) |
                           ((uint64_t(th) & 0x7FFull) << 51))
                        : ((uint64_t(m_draw.texTbp) << 40) ^ (uint64_t(m_draw.texTbw) << 32) ^
                           (uint64_t(t0.psm) << 26) ^ (uint64_t(tw) << 16) ^ uint64_t(th) ^
                           (uint64_t(t0.cbp) << 44));
                if (g_glKeyCensus)
                {
                    std::lock_guard<std::mutex> lk(g_glKeyMutex);
                    auto &v = g_glKeyMap[baseKey];
                    bool seen = false;
                    for (const GlKeyTuple &k : v)
                        if (k.tbp == uint32_t(m_draw.texTbp) && k.tbw == uint32_t(m_draw.texTbw) &&
                            k.psm == uint32_t(t0.psm) && k.tw == uint32_t(tw) &&
                            k.th == uint32_t(th) && k.cbp == uint32_t(t0.cbp))
                        { seen = true; break; }
                    if (!seen)
                    {
                        v.push_back(GlKeyTuple{uint32_t(m_draw.texTbp), uint32_t(m_draw.texTbw),
                                               uint32_t(t0.psm), uint32_t(tw), uint32_t(th),
                                               uint32_t(t0.cbp)});
                        if (v.size() > 1 && ++g_glKeyCollisions <= 32ull)
                            std::fprintf(stderr,
                                         "[gs2:glrkey] COLLISION key=%016llx now #%zu: "
                                         "tbp=%u tbw=%u psm=0x%02x %ux%u cbp=%u | first: "
                                         "tbp=%u tbw=%u psm=0x%02x %ux%u cbp=%u\n",
                                         (unsigned long long)baseKey, v.size(),
                                         unsigned(m_draw.texTbp), unsigned(m_draw.texTbw),
                                         unsigned(t0.psm), unsigned(tw), unsigned(th),
                                         unsigned(t0.cbp),
                                         v[0].tbp, v[0].tbw, v[0].psm, v[0].tw, v[0].th, v[0].cbp);
                    }
                }
                GlTexEntry &ent = m_glTexCache[baseKey];
                bool needDecode = !ent.shipped;
                if (ent.shipped && ent.gen != gen)
                {
                    // The pages moved. Confirm with content before paying for a decode.
                    const uint64_t h = glTexContentHash(pageBase, pages);
                    if (h != ent.hash)
                    {
                        needDecode = true;
                        ent.hash = h;
                    }
                    ent.gen = gen; // either way, this generation is now accounted for
                    if (!needDecode)
                        ++g_glTexHashSaved;
                }
                // ★ cont.330f: DUMPTEX lives inside the `needDecode` branch, so a texture served
                // from the cache never dumps -- and in the gameplay era GL never re-decodes at all
                // (it never writes m_vram, so the content hash is frozen). Force a decode for the
                // watched TBP while the dump is armed, so the probe can SEE what is bound.
                // ★ cont.355d: this FORCES the decode so the dump below has pixels -- a cached
                // texture never re-decodes and the dump site is simply never reached. DUMPTEXANY
                // has to widen THIS gate too, which is why the first attempt at an inventory came
                // back empty while decodes were plainly happening. The de-duplication by content
                // hash stays at the dump site; here we only need the budget and the window.
                if (s_glDumpTexDir && s_glDumpTexDir[0] && g_glDumpTexShown < s_glDumpTexN &&
                    probeFlipOk() &&
                    (s_glDumpTexAny ||
                     (s_glDumpTexTbp != 0u && uint32_t(m_draw.texTbp) == s_glDumpTexTbp &&
                      (s_glDumpTexCbp == 0u || uint32_t(t0.cbp) == s_glDumpTexCbp))))
                    needDecode = true;
                // ★★★ cont.330 TEXTRACE: the three quantities that decide whether this draw
                // re-decodes. `live` is the content hash computed NOW, independent of the cache's
                // own bookkeeping, so "the generation never moved" and "the generation moved but
                // the bytes did not" are distinguishable rather than both showing up as silence.
                if (s_glTexTrace != 0u && m_draw.texTbp == s_glTexTrace)
                {
                    const uint64_t live = glTexContentHash(pageBase, pages);
                    std::fprintf(stderr,
                                 "[gs2:textrace/draw] #%llu tbp=%u tbw=%u psm=0x%02x %ux%u "
                                 "pages=%u..%u | gen{cached=%llu now=%llu moved=%d} "
                                 "hash{cached=%016llx live=%016llx changed=%d} decode=%d shipped=%d\n",
                                 ++g_glTexTraceDraws, m_draw.texTbp, m_draw.texTbw, t0.psm, tw, th,
                                 pageBase, pageBase + pages - 1u,
                                 (unsigned long long)ent.gen, (unsigned long long)gen,
                                 ent.gen != gen ? 1 : 0,
                                 (unsigned long long)ent.hash, (unsigned long long)live,
                                 ent.hash != live ? 1 : 0, needDecode ? 1 : 0,
                                 ent.shipped ? 1 : 0);
                }
                // The device keys one GL object per TEXTURE and replaces its contents when an
                // upload arrives, so the key must NOT carry the change epoch -- doing so created a
                // new object per change, leaked them, and cost a full reallocation each time.
                texKey = baseKey ? baseKey : 1u; // 0 means "untextured" downstream
                wrapU = static_cast<uint8_t>(m_draw.texWrapU == 0u ? 0u : 1u);
                wrapV = static_cast<uint8_t>(m_draw.texWrapV == 0u ? 0u : 1u);
                tfx = static_cast<uint8_t>(t0.tfx & 3u);
                tcc = static_cast<uint8_t>(t0.tcc ? 1u : 0u);
                lin = static_cast<uint8_t>(st.linearFilter ? 1u : 0u);

                // ★ cont.329 phase 3b INSTRUMENT. Two runs now have failed to answer "does the
                // texture cache reuse anything", because the renderer never reaches gameplay. So
                // measure the key's VOLATILITY directly instead of inferring it from a frame rate:
                // split every decode into "a texture we have never seen" vs "one we have seen,
                // whose CONTENT GENERATION changed". Those two point at completely different
                // fixes -- a faster/lazier decode versus an invalidation that is too eager -- and
                // guessing between them is how the last two cycles were wasted.
                // ★★ cont.332: this used to be where `new`/`regen` were counted -- OUTSIDE the
                // `needDecode` branch they describe, i.e. once per textured DRAW. The draw count is
                // still worth having, under its own name; the decode split moved into the branch.
                ++g_glTexDraws;

                if (needDecode)
                {
                    const bool hadPrev = ent.shipped;
                    if (hadPrev)
                        ++g_glTexRegen;   // same texture, re-decoded (content really did change)
                    else
                        ++g_glTexNew;     // genuinely a new texture
                    uint64_t decHash = 0;
                    bool decHashed = false;
                    ++ent.epoch;            // kept as a decode counter only
                    ent.gen = gen;
                    ent.hash = glTexContentHash(pageBase, pages);
                    ent.shipped = true;
                    const uint32_t tpsm = t0.psm;
                    const bool paletted = (tpsm == GS_PSM_T4 || tpsm == GS_PSM_T8 ||
                                           tpsm == GS_PSM_T4HL || tpsm == GS_PSM_T4HH ||
                                           tpsm == GS_PSM_T8H);
                    // ★★★ cont.329j: the whole chain, level by level. resolveTexMip(st, k) is the
                    // rasterizer's own resolve, so each level's address comes from MIPTBP1/2
                    // exactly as the CPU sampler reads it -- and level 0 is restored afterwards,
                    // because the vertex UV scaling below uses m_draw.texW / texFstScale.
                    const uint32_t mxl = s_glMips ? static_cast<uint32_t>((st.context.tex1 >> 2) & 7u) : 0u;
                    uint32_t tpBaseW = 0, tpBaseH = 0;   // cont.355e: level 0's size, for the pack's scale
                    uint64_t tpKey = 0;                  // cont.355h: the DECODED-content key
                    for (uint32_t lvl = 0; lvl <= mxl; ++lvl)
                    {
                        if (lvl != 0u)
                            resolveTexMip(st, lvl);
                        const uint32_t lw = static_cast<uint32_t>(m_draw.texW);
                        const uint32_t lh = static_cast<uint32_t>(m_draw.texH);
                        if (lw == 0u || lh == 0u)
                            break;
                        if (lvl == 0u) { tpBaseW = lw; tpBaseH = lh; }
                        GsGlTexUpload up;
                        up.key = texKey;
                        up.level = static_cast<uint8_t>(lvl);
                        up.w = lw;
                        up.h = lh;
                        up.rgba.resize(static_cast<size_t>(lw) * lh * 4u);
                        for (uint32_t yy = 0; yy < lh; ++yy)
                        {
                            for (uint32_t xx = 0; xx < lw; ++xx)
                            {
                                const uint32_t raw =
                                    ReadVramUnlocked(tpsm, m_draw.texTbp, m_draw.texTbw, xx, yy);
                                uint32_t c = paletted ? m_draw.clut[raw & m_draw.clutMask] : raw;
                                if (!paletted && tpsm == GS_PSM_CT24)
                                {
                                    // ★ cont.330e: TA0 unconditionally (default), or the CPU
                                    // sampler's full AEM rule under PS2X_GS_GLR_TEXAEM.
                                    const bool rgbZero = (c & 0x00FFFFFFu) == 0u;
                                    const uint32_t a = (s_glTexAem && st.texa.aem && rgbZero)
                                                           ? 0u : uint32_t(st.texa.ta0);
                                    c = (c & 0x00FFFFFFu) | (a << 24);
                                }
                                else if (!paletted && s_glTexAem &&
                                         (tpsm == GS_PSM_CT16 || tpsm == GS_PSM_CT16S))
                                {
                                    const bool rgbZero = (c & 0x00FFFFFFu) == 0u;
                                    const uint32_t a = ((c >> 24) & 0x80u)
                                                           ? uint32_t(st.texa.ta1)
                                                           : ((st.texa.aem && rgbZero) ? 0u
                                                                                       : uint32_t(st.texa.ta0));
                                    c = (c & 0x00FFFFFFu) | (a << 24);
                                }
                                if (s_glAsProbe != 0u && uint32_t(m_draw.texTbp) == s_glAsProbe)
                                {
                                    const uint32_t av = (c >> 24) & 0xFFu;
                                    if (av < g_glAsMin) g_glAsMin = av;
                                    if (av > g_glAsMax) g_glAsMax = av;
                                    g_glAsSum += av; ++g_glAsN;
                                }
                                uint8_t *o = up.rgba.data() + (static_cast<size_t>(yy) * lw + xx) * 4u;
                                o[0] = static_cast<uint8_t>(c & 0xFFu);
                                o[1] = static_cast<uint8_t>((c >> 8) & 0xFFu);
                                o[2] = static_cast<uint8_t>((c >> 16) & 0xFFu);
                                o[3] = static_cast<uint8_t>((c >> 24) & 0xFFu);
                            }
                        }
                        // ★★★★★ cont.355h: the pack's key is a hash of the DECODED PIXELS, not of
                        // VRAM. glTexContentHash hashes VRAM PAGES, and a texture's pages carry
                        // more than that texture -- the menu and gameplay draw a byte-identical
                        // glyph sheet under two different VRAM hashes. Keying on what the texture
                        // LOOKS LIKE makes one asset serve every occurrence. Computed at level 0
                        // and reused for the chain, and used for the dump's filename too so a dump
                        // and a pack entry always agree.
                        if (lvl == 0u) tpKey = ps2x::texpack::contentKey(up.rgba.data(), lw, lh);
                        // ★ cont.330e: a picture of exactly what GL is about to bind.
                        const bool dumpThis =
                            s_glDumpTexDir && s_glDumpTexDir[0] && g_glDumpTexShown < s_glDumpTexN &&
                            probeFlipOk() &&
                            (s_glDumpTexAny
                                 // ★ cont.355d: any texture, but each CONTENT once (a frame draws
                                 // the same atlas hundreds of times).
                                 ? (lvl == 0u && glDumpTexFirstSighting(tpKey))
                                 : (s_glDumpTexTbp != 0u && uint32_t(m_draw.texTbp) == s_glDumpTexTbp &&
                                    (s_glDumpTexCbp == 0u || uint32_t(t0.cbp) == s_glDumpTexCbp)));
                        if (dumpThis)
                        {
                            ++g_glDumpTexShown;
                            char path[512];
                            std::snprintf(path, sizeof(path), "%s/tex_f%06llu_%02d_h%016llx_tbp%u_cbp%u_l%u_%ux%u.ppm",
                                          s_glDumpTexDir,
                                          (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                                          g_glDumpTexShown,
                                          // ★ cont.355b: the CONTENT HASH is the replacement key
                                          // (glTexContentHash). Without it the dump cannot be
                                          // joined to a texture pack keyed by hash.
                                          (unsigned long long)tpKey,
                                          unsigned(m_draw.texTbp), unsigned(t0.cbp), unsigned(lvl), lw, lh);
                            if (FILE *f = std::fopen(path, "wb"))
                            {
                                std::fprintf(f, "P6\n%u %u\n255\n", lw, lh);
                                for (size_t i = 0; i < size_t(lw) * lh; ++i)
                                { std::fputc(up.rgba[i*4], f); std::fputc(up.rgba[i*4+1], f);
                                  std::fputc(up.rgba[i*4+2], f); }
                                std::fclose(f);
                            }
                            std::snprintf(path, sizeof(path), "%s/tex_f%06llu_%02d_h%016llx_tbp%u_cbp%u_l%u_%ux%u_ALPHA.ppm",
                                          s_glDumpTexDir,
                                          (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                                          g_glDumpTexShown,
                                          // ★ cont.355b: the CONTENT HASH is the replacement key
                                          // (glTexContentHash). Without it the dump cannot be
                                          // joined to a texture pack keyed by hash.
                                          (unsigned long long)tpKey,
                                          unsigned(m_draw.texTbp), unsigned(t0.cbp), unsigned(lvl), lw, lh);
                            if (FILE *f = std::fopen(path, "wb"))
                            {
                                std::fprintf(f, "P6\n%u %u\n255\n", lw, lh);
                                for (size_t i = 0; i < size_t(lw) * lh; ++i)
                                { const unsigned char a = up.rgba[i*4+3];
                                  std::fputc(a, f); std::fputc(a, f); std::fputc(a, f); }
                                std::fclose(f);
                            }
                            std::fprintf(stderr,
                                         "[gs2:dumptex] #%d tbp=%u lvl=%u %ux%u psm=0x%02x tbw=%u -> %s\n",
                                         g_glDumpTexShown, unsigned(m_draw.texTbp), unsigned(lvl),
                                         lw, lh, tpsm, unsigned(m_draw.texTbw), path);
                        }
                        if (g_glCensus) ++g_glcTexUp;
                        // ★★★★ cont.355e HD TEXTURE PACK. Deliberately AFTER the dump above, so
                        // PS2X_GS_GLR_DUMPTEX always yields the ORIGINAL -- a pack is authored from
                        // the source art, and a dump that returned the replacement would make the
                        // workflow eat its own tail. Keyed by the same content hash the dump names
                        // its files with. No geometry change is needed: the GL shader samples
                        // normalised, so a denser image at the same logical extent lands exactly
                        // where the original did.
                        if (ps2x::texpack::enabled())
                            ps2x::texpack::substitute(tpKey, tpBaseW, tpBaseH, up.w, up.h, up.rgba);
                        // ★★★★ cont.332g: is this decode UNIFORM? (the blank-texture detector)
                        if (s_glFlatTex && lvl == 0u && up.rgba.size() >= 16u)
                        {
                            const uint8_t *p = up.rgba.data();
                            const size_t texels = up.rgba.size() / 4u;
                            uint32_t first = 0;
                            std::memcpy(&first, p, 4);
                            bool flat = true;
                            for (size_t i = 64; i < texels; i += 64u)
                            {
                                uint32_t t = 0;
                                std::memcpy(&t, p + i * 4u, 4);
                                if (t != first) { flat = false; break; }
                            }
                            if (flat)
                            {
                                ++g_glFlatDecodes;
                                if (!ent.flatLast)
                                {
                                    ++g_glFlatKeys;
                                    static unsigned logs = 0;
                                    if (logs < 32u)
                                    {
                                        ++logs;
                                        std::fprintf(stderr,
                                                     "[gs2:flattex] FLAT decode #%llu tbp=%u tbw=%u psm=0x%02x"
                                                     " %ux%u cbp=%u rgba=%02x%02x%02x%02x draws-so-far=%llu\n",
                                                     g_glFlatKeys, unsigned(m_draw.texTbp),
                                                     unsigned(m_draw.texTbw), tpsm, lw, lh,
                                                     unsigned(t0.cbp), p[0], p[1], p[2], p[3],
                                                     g_glTexDraws);
                                    }
                                }
                                ent.flatLast = true;
                            }
                            else if (ent.flatLast)
                            {
                                ++g_glFlatRecovered;   // it filled in later: the cache DID recover
                                ent.flatLast = false;
                            }
                        }
                        // ★★★ cont.332 DECVERIFY: hash the texels we are about to ship, before the
                        // upload is moved into the batch. Level 0 only -- it is the level the
                        // invalidation question is about, and MIPS is default OFF.
                        if (s_glDecVerify && lvl == 0u && !up.rgba.empty())
                        {
                            uint64_t h = 1469598103934665603ull;
                            const uint8_t *p = up.rgba.data();
                            const size_t n = up.rgba.size();
                            size_t i = 0;
                            for (; i + 8u <= n; i += 8u)
                            {
                                uint64_t w = 0;
                                std::memcpy(&w, p + i, sizeof(w));
                                h ^= w;
                                h *= 1099511628211ull;
                            }
                            for (; i < n; ++i)
                            {
                                h ^= p[i];
                                h *= 1099511628211ull;
                            }
                            decHash = h;
                            decHashed = true;
                        }
                        b.texUploads.push_back(std::move(up));
                        g_glTexTexels += static_cast<unsigned long long>(lw) * lh;
                    }
                    if (mxl != 0u)
                        resolveTexMip(st, 0u);
                    // ★★★ cont.332: was this re-decode necessary? Only if the TEXELS differ.
                    if (s_glDecVerify && decHashed)
                    {
                        if (!hadPrev)
                            ++g_glTexDecFirst;
                        else if (decHash == ent.outHash)
                            ++g_glTexDecSame;   // the page moved; the texture did not
                        else
                            ++g_glTexDecDiff;   // a real content change
                        ent.outHash = decHash;
                    }
                    if (((g_glTexNew + g_glTexRegen) % 256ull) == 0ull)
                        std::fprintf(stderr,
                                     "[gs2:glrtex] decodes=%llu new=%llu regen=%llu | "
                                     "textured-draws=%llu hash-saved=%llu texels=%llu"
                                     " | decverify{first=%llu same=%llu diff=%llu}"
                                     " | flat{decodes=%llu keys=%llu recovered=%llu}\n",
                                     g_glTexNew + g_glTexRegen, g_glTexNew, g_glTexRegen,
                                     g_glTexDraws, g_glTexHashSaved, g_glTexTexels,
                                     g_glTexDecFirst, g_glTexDecSame, g_glTexDecDiff,
                                     g_glFlatDecodes, g_glFlatKeys, g_glFlatRecovered);
                    // ★★ cont.332: one stderr write per decode, previously UNGATED -- 58k of them in
                    // a 155 s run, in the worker's hot path. Gated on PS2X_GS_THRUPUT (which every
                    // harness run sets, and cont.331s already used to gate [gsgpu:clock]), so the
                    // line still gives tools/harness/tailcen.py its per-frame decode counter in a
                    // measured run while a plain play run pays nothing.
                    if (s_gsThruput)
                        std::fprintf(stderr,
                                     "[gs2:glrtexrej] tme-draws=%llu untextured-draws=%llu | "
                                     "rejected{zero=%llu too-big=%llu} from-target=%llu\n",
                                     g_glTexDrawsTme, g_glTexDrawsPlain, g_glTexRejZero,
                                     g_glTexRejBig, g_glTexFromTarget);
                }
            }
        }

        // ★★★★★ cont.331l: stamp this draw's destination pages with the transfer sequence in
        // force, so the seed can tell whether VRAM or GL wrote a page last. Same walk as the page
        // map below; done separately because the map is a diagnostic and this is production.
        if (s_glAuthority && s_glPageOwn)
        {
            const GSFrameReg &frO = st.context.frame;
            const uint32_t fbwO = frO.fbw ? frO.fbw : 1u;
            int ox0 = 1 << 30, oy0 = 1 << 30, ox1 = -(1 << 30), oy1 = -(1 << 30);
            for (unsigned vi = 0; vi < it.vertexCount && vi < 3u; ++vi)
            {
                const int vx = int(it.vertices[vi].x) - int(st.context.xyoffset.ofx >> 4);
                const int vy = int(it.vertices[vi].y) - int(st.context.xyoffset.ofy >> 4);
                if (vx < ox0) ox0 = vx; if (vx > ox1) ox1 = vx;
                if (vy < oy0) oy0 = vy; if (vy > oy1) oy1 = vy;
            }
            ox0 = std::max(ox0, int(sc.x0)); oy0 = std::max(oy0, int(sc.y0));
            ox1 = std::min(ox1, int(sc.x1)); oy1 = std::min(oy1, int(sc.y1));
            if (ox1 >= ox0 && oy1 >= oy0)
                for (int py = oy0 >> 5; py <= (oy1 >> 5); ++py)
                    for (int px = ox0 >> 6; px <= (ox1 >> 6); ++px)
                    {
                        const uint32_t pg = (frO.fbp + uint32_t(py) * fbwO + uint32_t(px)) &
                                            (kGs2Pages - 1u);
                        g_glPageDrawAtXfer[pg].store(
                            g_gs2PageXferGen[pg].load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
                    }
        }
        if (g_glPageMap)
        {
            // WRITTEN: the draw's own vertex bbox, clipped to the scissor, walked over the page
            // grid (a CT32 page is 64x32 pixels, and fbw pages per row).
            const GSFrameReg &fr0 = st.context.frame;
            const uint32_t fbw0 = fr0.fbw ? fr0.fbw : 1u;
            int bx0 = 1 << 30, by0 = 1 << 30, bx1 = -(1 << 30), by1 = -(1 << 30);
            for (unsigned vi = 0; vi < it.vertexCount && vi < 3u; ++vi)
            {
                const int vx = int(it.vertices[vi].x) - int(st.context.xyoffset.ofx >> 4);
                const int vy = int(it.vertices[vi].y) - int(st.context.xyoffset.ofy >> 4);
                if (vx < bx0) bx0 = vx; if (vx > bx1) bx1 = vx;
                if (vy < by0) by0 = vy; if (vy > by1) by1 = vy;
            }
            bx0 = std::max(bx0, int(sc.x0)); by0 = std::max(by0, int(sc.y0));
            bx1 = std::min(bx1, int(sc.x1)); by1 = std::min(by1, int(sc.y1));
            if (bx1 >= bx0 && by1 >= by0)
                for (int py = by0 >> 5; py <= (by1 >> 5); ++py)
                    for (int px = bx0 >> 6; px <= (bx1 >> 6); ++px)
                    {
                        const uint32_t pg = fr0.fbp + uint32_t(py) * fbw0 + uint32_t(px);
                        if (pg < 512u) { ++g_pgWrite[pg]; g_pgRendered[pg] = true; }
                    }
            // READ: the texture's page span, from its bits per texel.
            if (st.prim.tme)
            {
                bool stale = false;
                uint32_t bits = 32u;
                switch (st.context.tex0.psm)
                {
                case GS_PSM_T4: bits = 4u; break;
                case GS_PSM_T8: bits = 8u; break;
                case GS_PSM_CT16:
                case GS_PSM_CT16S: bits = 16u; break;
                default: bits = 32u; break;
                }
                const uint64_t bytes = (uint64_t(m_draw.texW) * uint64_t(m_draw.texH) * bits) / 8ull;
                const uint32_t p0 = uint32_t(m_draw.texTbp) >> 5;
                const uint32_t np = uint32_t(bytes / 8192ull) + 1u;
                for (uint32_t k = 0; k < np && p0 + k < 512u; ++k)
                {
                    ++g_pgRead[p0 + k];
                    if (g_pgRendered[p0 + k]) stale = true;
                }
                if (stale)
                {
                    ++g_staleDraws;
                    unsigned r = 0;
                    for (; r < g_staleN; ++r)
                        if (g_stale[r].tbp == uint32_t(m_draw.texTbp) &&
                            g_stale[r].tw == uint32_t(m_draw.texW) &&
                            g_stale[r].th == uint32_t(m_draw.texH)) break;
                    if (r == g_staleN && g_staleN < 24u)
                    {
                        g_stale[g_staleN].tbp = uint32_t(m_draw.texTbp);
                        g_stale[g_staleN].tw = uint32_t(m_draw.texW);
                        g_stale[g_staleN].th = uint32_t(m_draw.texH);
                        g_stale[g_staleN].psm = st.context.tex0.psm;
                        ++g_staleN;
                    }
                    if (r < 24u) ++g_stale[r].draws;
                }
                else
                    ++g_freshDraws;
            }
        }
        if (g_glTgtSrcCensus)
        {
            const GSFrameReg &fr = st.context.frame;
            unsigned t = 0;
            for (; t < g_glcTgtN; ++t)
                if (g_glcTgt[t].fbp == fr.fbp && g_glcTgt[t].fbw == fr.fbw && g_glcTgt[t].psm == fr.psm) break;
            if (t == g_glcTgtN && g_glcTgtN < 16u)
            {
                g_glcTgt[g_glcTgtN].fbp = fr.fbp; g_glcTgt[g_glcTgtN].fbw = fr.fbw;
                g_glcTgt[g_glcTgtN].psm = fr.psm; ++g_glcTgtN;
            }
            if (t < 16u)
            {
                ++g_glcTgt[t].draws;
                g_glcTgt[t].verts += it.vertexCount;
                g_glcTgt[t].zbp = st.context.zbuf.zbp;
                g_glcTgt[t].zpsm = st.context.zbuf.psm;
                // The real extent this target is drawn over -- the scissor can be stale, the
                // vertices cannot.
                for (unsigned vi = 0; vi < it.vertexCount && vi < 3u; ++vi)
                {
                    const int vx = int(it.vertices[vi].x) - int(st.context.xyoffset.ofx >> 4);
                    const int vy = int(it.vertices[vi].y) - int(st.context.xyoffset.ofy >> 4);
                    if (vx < g_glcTgt[t].vminX) g_glcTgt[t].vminX = vx;
                    if (vy < g_glcTgt[t].vminY) g_glcTgt[t].vminY = vy;
                    if (vx > g_glcTgt[t].vmaxX) g_glcTgt[t].vmaxX = vx;
                    if (vy > g_glcTgt[t].vmaxY) g_glcTgt[t].vmaxY = vy;
                }
                if (static_cast<uint32_t>(sc.x1) > g_glcTgt[t].maxX) g_glcTgt[t].maxX = sc.x1;
                if (static_cast<uint32_t>(sc.y1) > g_glcTgt[t].maxY) g_glcTgt[t].maxY = sc.y1;
                if (!st.prim.tme) ++g_glcTgt[t].noTex;
                {
                    const uint32_t fmv = uint32_t(st.context.frame.fbmsk);
                    if (fmv == 0xFFFFFFFFu) ++g_glcTgt[t].mskAll;
                    else if (fmv == 0u) ++g_glcTgt[t].mskNone;
                    else ++g_glcTgt[t].mskOther;
                }
            }
            // A texture whose page lands at or above the first scratch page (0x100 here) is a
            // candidate read of something the GL path rendered rather than of uploaded VRAM.
            // ★ cont.330d: ALSO record, unfiltered by page, every textured draw into the target
            // named by PS2X_GS_GLR_TGTSRC_DST -- the page filter is what hid the sprite that blacks
            // out 0x180, whose source sits below page 0x100.
            if (st.prim.tme && ((uint32_t(m_draw.texTbp) >> 5) >= 0x100u || fr.fbp == s_glTgtSrcDst))
            {
                unsigned k = 0;
                for (; k < g_glcSrcN; ++k)
                    if (g_glcSrc[k].tbp == uint32_t(m_draw.texTbp) && g_glcSrc[k].dstFbp == fr.fbp &&
                        g_glcSrc[k].tw == uint32_t(m_draw.texW) && g_glcSrc[k].th == uint32_t(m_draw.texH)) break;
                if (k == g_glcSrcN && g_glcSrcN >= 96u) ++g_glcSrcDropped;
                if (k == g_glcSrcN && g_glcSrcN < 96u)
                {
                    g_glcSrc[g_glcSrcN].tbp = uint32_t(m_draw.texTbp);
                    g_glcSrc[g_glcSrcN].tbw = uint32_t(m_draw.texTbw);
                    g_glcSrc[g_glcSrcN].psm = st.context.tex0.psm;
                    g_glcSrc[g_glcSrcN].tw = uint32_t(m_draw.texW);
                    g_glcSrc[g_glcSrcN].th = uint32_t(m_draw.texH);
                    g_glcSrc[g_glcSrcN].dstFbp = fr.fbp;
                    g_glcSrc[g_glcSrcN].tfx = st.context.tex0.tfx;
                    g_glcSrc[g_glcSrcN].tcc = st.context.tex0.tcc;
                    g_glcSrc[g_glcSrcN].cbp = st.context.tex0.cbp;
                    g_glcSrc[g_glcSrcN].cpsm = st.context.tex0.cpsm;
                    g_glcSrc[g_glcSrcN].csa = st.context.tex0.csa;
                    g_glcSrc[g_glcSrcN].csm = st.context.tex0.csm;
                    g_glcSrc[g_glcSrcN].fst = st.prim.fst ? 1u : 0u;
                    ++g_glcSrcN;
                }
                if (k < 96u)
                {
                    ++g_glcSrc[k].draws; if (texIsTarget) ++g_glcSrc[k].promoted;
                    // The sampled extent in TEXELS. UV mode carries u/v in 1/16 texel; STQ mode
                    // carries normalised s/t, which the GS scales by the TEX0 size.
                    for (unsigned vi = 0; vi < it.vertexCount && vi < 3u; ++vi)
                    {
                        const GSVertex &vtx = it.vertices[vi];
                        const int tu = st.prim.fst ? int(vtx.u >> 4)
                                                   : int(vtx.s * float(m_draw.texW ? m_draw.texW : 1));
                        const int tv = st.prim.fst ? int(vtx.v >> 4)
                                                   : int(vtx.t * float(m_draw.texH ? m_draw.texH : 1));
                        if (tu < g_glcSrc[k].uMin) g_glcSrc[k].uMin = tu;
                        if (tv < g_glcSrc[k].vMin) g_glcSrc[k].vMin = tv;
                        if (tu > g_glcSrc[k].uMax) g_glcSrc[k].uMax = tu;
                        if (tv > g_glcSrc[k].vMax) g_glcSrc[k].vMax = tv;
                    }
                }
            }
        }
        // ALPHA bits: A 0-1, B 2-3, C 4-5, D 6-7, FIX 32-39. The selectors travel raw; the device
        // maps the combination onto GL blend factors (cont.329 census: four are live here).
        const uint64_t alpha = st.context.alpha;
        const uint8_t abe = static_cast<uint8_t>(st.prim.abe ? 1u : 0u);
        const uint8_t ba = static_cast<uint8_t>(alpha & 3u);
        const uint8_t bb = static_cast<uint8_t>((alpha >> 2) & 3u);
        const uint8_t bc = static_cast<uint8_t>((alpha >> 4) & 3u);
        const uint8_t bd = static_cast<uint8_t>((alpha >> 6) & 3u);
        const uint8_t bfix = static_cast<uint8_t>((alpha >> 32) & 0xFFu);
        // ★★★ cont.330e ASPROBE: the whole colour-contribution decision for one watched TBP, in
        // one line. cont.330d ruled out the texture SOURCE (this draw contributes nothing whatever
        // it resolves to), so what is left is the ALPHA and the blend selectors that consume it.
        // ★★★★ cont.330g FRAMEHIST: every draw in the window, not the first N.
        if (s_frameHist && probeFlipOk())
        {
            const uint32_t hAte = uint32_t(st.context.test & 1u);
            const uint32_t hAtst = uint32_t((st.context.test >> 1) & 7u);
            const uint32_t hAref = uint32_t((st.context.test >> 4) & 0xFFu);
            const uint32_t hAfail = uint32_t((st.context.test >> 12) & 3u);
            const uint32_t hZte = uint32_t((st.context.test >> 16) & 1u);
            const uint32_t hZtst = uint32_t((st.context.test >> 17) & 3u);
            const uint32_t hFbmsk = uint32_t(st.context.frame.fbmsk);
            const uint32_t hTme = st.prim.tme ? 1u : 0u;
            const uint32_t hFst = st.prim.fst ? 1u : 0u;          // cont.355b: the GS 2D signal
            const uint32_t hPrim = static_cast<uint32_t>(st.prim.type);
            unsigned r = 0;
            for (; r < g_frameHistN; ++r)
            {
                const FrameHistRow &q = g_frameHist[r];
                if (q.ate == hAte && q.atst == hAtst && q.aref == hAref && q.afail == hAfail &&
                    q.zte == hZte && q.ztst == hZtst && q.abe == abe && q.fbmsk == hFbmsk &&
                    q.tme == hTme && q.tgt == texIsTarget && q.ba == ba && q.bb == bb &&
                    q.bc == bc && q.bd == bd && q.fst == hFst && q.prim == hPrim)
                    break;
            }
            if (r == g_frameHistN && g_frameHistN < 48u)
            {
                FrameHistRow &q = g_frameHist[g_frameHistN++];
                q.ate = hAte; q.atst = hAtst; q.aref = hAref; q.afail = hAfail;
                q.zte = hZte; q.ztst = hZtst; q.abe = abe; q.fbmsk = hFbmsk;
                q.tme = hTme; q.tgt = texIsTarget; q.tfx = tfx; q.tcc = tcc;
                q.ba = ba; q.bb = bb; q.bc = bc; q.bd = bd;
                q.fst = hFst; q.prim = hPrim; q.tbp = st.context.tex0.tbp0;
            }
            if (r < 48u)
            {
                ++g_frameHist[r].draws;
                // The scissor-clipped geometry area this primitive covers (same quantity the
                // glcensus GEOMETRY AREA uses), so "covers the screen" and "covers nothing" do
                // not look alike in the table.
                const double sw = double(sc.x1 >= sc.x0 ? sc.x1 - sc.x0 + 1 : 0);
                const double sh = double(sc.y1 >= sc.y0 ? sc.y1 - sc.y0 + 1 : 0);
                double a = 0.0;
                if (it.vertexCount >= 2u)
                {
                    const double dx = std::fabs(double(it.vertices[1].x) - double(it.vertices[0].x));
                    const double dy = std::fabs(double(it.vertices[1].y) - double(it.vertices[0].y));
                    a = dx * dy;
                }
                const double cap = sw * sh;
                g_frameHist[r].area += (cap > 0.0 && a > cap) ? cap : a;
                if (it.vertexCount >= 2u)
                {
                    const double w = std::fabs(double(it.vertices[1].x) - double(it.vertices[0].x));
                    if (w < g_frameHist[r].wMin) g_frameHist[r].wMin = w;
                    if (w > g_frameHist[r].wMax) g_frameHist[r].wMax = w;
                }
                // ★★★★★ cont.356: q and the SCREEN bbox, over EVERY vertex -- the width/area above
                // read vertices[0..1] only, which is a sprite's diagonal but NOT a strip's extent.
                {
                    const int hofx = static_cast<int>(st.context.xyoffset.ofx) >> 4;
                    const int hofy = static_cast<int>(st.context.xyoffset.ofy) >> 4;
                    FrameHistRow &h = g_frameHist[r];
                    h.ofx = hofx;
                    h.ofy = hofy;
                    // GSPrimitiveBatch::vertices is a fixed std::array<GSVertex,3>; clamp rather
                    // than trust vertexCount, so a census can never read past it.
                    const unsigned nv = it.vertexCount < 3u ? unsigned(it.vertexCount) : 3u;
                    for (unsigned vi = 0; vi < nv; ++vi)
                    {
                        const double vq = double(it.vertices[vi].q);
                        if (vq < h.qMin) h.qMin = vq;
                        if (vq > h.qMax) h.qMax = vq;
                        const double sx = double(it.vertices[vi].x) - double(hofx);
                        const double sy = double(it.vertices[vi].y) - double(hofy);
                        if (sx < h.xMin) h.xMin = sx;
                        if (sx > h.xMax) h.xMax = sx;
                        if (sy < h.yMin) h.yMin = sy;
                        if (sy > h.yMax) h.yMax = sy;
                    }
                }
            }
            else
                ++g_frameHistDropped;
        }
        // ★ cont.330f: ASPROBE=0xFFFFFFFF matches ANY draw (textured or not), so the probe can
        // ask "what is this frame made of" rather than "what does this one texture do" -- which is
        // the question in the black gameplay era, where every fragment is being lost somewhere.
        const bool asProbeAny = (s_glAsProbe == 0xFFFFFFFFu);
        if (s_glAsProbe != 0u && (asProbeAny || (st.prim.tme && uint32_t(m_draw.texTbp) == s_glAsProbe)) &&
            g_glAsShown < s_glAsProbeN && probeFlipOk())
        {
            ++g_glAsShown;
            std::fprintf(stderr,
                         "[gs2:asprobe] flip=%llu #%d tbp=%u psm=0x%02x %dx%d -> fbp=0x%x | abe=%u "
                         "ALPHA{A=%u B=%u C=%u D=%u FIX=%u} = (%s - %s) * %s + %s | tcc=%u tfx=%u "
                         "TEXA{ta0=%u ta1=%u aem=%u} | promoted=%u srcMode=%u srcTexa=%u srcAem=%u "
                         "| vtx-rgba=(%u,%u,%u,%u) | decoded-alpha{min=%u max=%u mean=%.1f n=%llu} aemfix=%u"
                         " | TEST{ate=%u atst=%u aref=%u afail=%u date=%u zte=%u ztst=%u}"
                         " ZBUF{zbp=0x%x psm=0x%02x zmsk=%u} FBMSK=%08x tme=%u"
                         " z=%.0f..%.0f\n",
                         (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                         g_glAsShown, m_draw.texTbp, st.context.tex0.psm,
                         int(m_draw.texW), int(m_draw.texH), st.context.frame.fbp, abe,
                         ba, bb, bc, bd, bfix,
                         ba == 0 ? "Cs" : ba == 1 ? "Cd" : "0",
                         bb == 0 ? "Cs" : bb == 1 ? "Cd" : "0",
                         bc == 0 ? "As" : bc == 1 ? "Ad" : "FIX",
                         bd == 0 ? "Cs" : bd == 1 ? "Cd" : "0",
                         st.context.tex0.tcc, st.context.tex0.tfx,
                         st.texa.ta0, st.texa.ta1, st.texa.aem ? 1u : 0u,
                         texIsTarget, srcMode, srcTexa, srcAem,
                         unsigned(it.vertices[0].r), unsigned(it.vertices[0].g),
                         unsigned(it.vertices[0].b), unsigned(it.vertices[0].a),
                         g_glAsN ? g_glAsMin : 0u, g_glAsMax,
                         g_glAsN ? double(g_glAsSum) / double(g_glAsN) : 0.0,
                         g_glAsN, s_glTexAem ? 1u : 0u,
                         unsigned(st.context.test & 1u),
                         unsigned((st.context.test >> 1) & 7u),
                         unsigned((st.context.test >> 4) & 0xFFu),
                         unsigned((st.context.test >> 12) & 3u),
                         unsigned((st.context.test >> 14) & 1u),
                         unsigned((st.context.test >> 16) & 1u),
                         unsigned((st.context.test >> 17) & 3u),
                         st.context.zbuf.zbp, st.context.zbuf.psm,
                         st.context.zbuf.zmask ? 1u : 0u,
                         unsigned(st.context.frame.fbmsk), st.prim.tme ? 1u : 0u,
                         double(it.vertices[0].z),
                         double(it.vertices[it.vertexCount > 2u ? 2u : (it.vertexCount - 1u)].z));
        }

        // A new group whenever ANY state the device sets per group changes. Compared field by
        // field rather than packed into a key: the state no longer fits in 64 bits, and a packing
        // bug here is invisible -- two draws with different blending would silently share one draw
        // call and one of them would come out wrong.
        (void)drawTargetKey; // runs are split on the target upstream; kept for the assert below
        GsGlGroup want;
        want.sx0 = sc.x0; want.sy0 = sc.y0; want.sx1 = sc.x1; want.sy1 = sc.y1;
        want.ztst = ztst; want.zmsk = zmsk; want.zte = zte;
        want.abe = abe; want.ba = ba; want.bb = bb; want.bc = bc; want.bd = bd; want.bfix = bfix;
        want.colclamp = static_cast<uint8_t>(st.colclamp & 1ull); // cont.343b
        want.texKey = texKey; want.wrapU = wrapU; want.wrapV = wrapV;
        want.tfx = tfx; want.tcc = tcc; want.lin = lin;
        want.ate = ate; want.atst = atst; want.aref = aref; want.afail = afail;
        want.texIsTarget = texIsTarget;
        if (texIsTarget) // rotk row 257: the read's TW/TH (see GsGlGroup::texW)
        {
            want.texW = static_cast<uint16_t>(m_draw.texW > 0 ? m_draw.texW : 0);
            want.texH = static_cast<uint16_t>(m_draw.texH > 0 ? m_draw.texH : 0);
        }
        want.fbmsk = static_cast<uint32_t>(st.context.frame.fbmsk);
        want.srcMode = srcMode; want.srcTexa = srcTexa; want.srcAem = srcAem;
        want.minFilter = static_cast<uint8_t>((st.context.tex1 >> 6) & 7u);
        want.maxLevel = static_cast<uint8_t>((st.context.tex1 >> 2) & 7u);
        want.lcm = static_cast<uint8_t>(st.context.tex1 & 1u);
        want.lodL = static_cast<uint8_t>((st.context.tex1 >> 19) & 3u);
        // TEX1.K is 12-bit SIGNED at bits 32..43, in 1/16 units.
        {
            const int32_t rawK = static_cast<int32_t>((st.context.tex1 >> 32) & 0xFFFu);
            want.lodK = static_cast<int16_t>(rawK >= 0x800 ? rawK - 0x1000 : rawK);
        }
        want.fge = static_cast<uint8_t>(st.prim.fge ? 1u : 0u);
        want.fogR = st.fogR; want.fogG = st.fogG; want.fogB = st.fogB;
        if (g_glCensus)
        {
            if (st.prim.fge) { ++g_glcFgeDraws; g_glcFgeVerts += it.vertexCount; }
            else ++g_glcNoFgeDraws;
            ++g_glcColclamp[st.colclamp ? 1u : 0u];
            ++g_glcFba[(st.context.fba & 1u) ? 1u : 0u];
            ++g_glcPabe[st.pabe ? 1u : 0u];
            if (st.prim.tme)
            {
                ++g_glcWrap[m_draw.texWrapU & 3u];
                ++g_glcWrap[m_draw.texWrapV & 3u];
                ++g_glcTexPsm[st.context.tex0.psm & 63u];
                ++g_glcTexPsmN;
                if (st.texa.aem) ++g_glcAemTex;
                const uint64_t t1 = st.context.tex1;
                const unsigned mxl = static_cast<unsigned>((t1 >> 2) & 7u);
                const unsigned mmin = static_cast<unsigned>((t1 >> 6) & 7u);
                ++g_glcMip[mmin & 7u][mxl & 7u];
                ++g_glcMipN;
            }
        }
        auto sameState = [](const GsGlGroup &x, const GsGlGroup &y)
        {
            return x.sx0 == y.sx0 && x.sy0 == y.sy0 && x.sx1 == y.sx1 && x.sy1 == y.sy1 &&
                   x.ztst == y.ztst && x.zmsk == y.zmsk && x.zte == y.zte && x.abe == y.abe &&
                   x.ba == y.ba && x.bb == y.bb && x.bc == y.bc && x.bd == y.bd &&
                   (!x.abe || (x.bfix == y.bfix && x.colclamp == y.colclamp)) && x.texKey == y.texKey &&
                   x.texIsTarget == y.texIsTarget &&
                   (!x.texIsTarget || (x.texW == y.texW && x.texH == y.texH)) &&
                   (x.texKey == 0u || (x.wrapU == y.wrapU && x.wrapV == y.wrapV &&
                                       x.tfx == y.tfx && x.tcc == y.tcc && x.lin == y.lin &&
                                       x.minFilter == y.minFilter && x.maxLevel == y.maxLevel &&
                                       x.lcm == y.lcm && x.lodL == y.lodL && x.lodK == y.lodK)) &&
                   x.ate == y.ate && x.fbmsk == y.fbmsk &&
                   x.fge == y.fge &&
                   (!x.fge || (x.fogR == y.fogR && x.fogG == y.fogG && x.fogB == y.fogB)) &&
                   x.srcMode == y.srcMode &&
                   (x.srcMode != 1 || (x.srcTexa == y.srcTexa && x.srcAem == y.srcAem)) &&
                   (!x.ate || (x.atst == y.atst && x.aref == y.aref && x.afail == y.afail));
        };
        if (b.groups.empty() || !sameState(b.groups.back(), want))
        {
            want.first = static_cast<uint32_t>(b.verts.size());
            b.groups.push_back(want);
            if (g_glCensus) ++g_glcGroups;
        }

        // XYOFFSET is 1/16 fixed point; vertex x/y are already divided by 16 by the frontend.
        const float ofx = static_cast<float>(st.context.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(st.context.xyoffset.ofy >> 4);
        const float zdiv = zScale(st.context.zbuf.psm);
        // Texture coordinates are emitted NORMALISED as (s, t, q) so s/q lands in [0,1] and both
        // coordinate modes share one shader path. FST draws carry direct texels in 1/16 fixed
        // point (texFstScale folds the mip shift in) with q = 1; STQ draws pass through, and the
        // shader does the per-pixel divide on screen-linear values, as the GS does.
        const bool fst = st.prim.fst != 0u;
        const float invTexW = (texKey != 0u && m_draw.texW > 0) ? 1.0f / static_cast<float>(m_draw.texW) : 1.0f;
        const float invTexH = (texKey != 0u && m_draw.texH > 0) ? 1.0f / static_cast<float>(m_draw.texH) : 1.0f;
        const float fstScale = m_draw.texFstScale;
        // ★★★★★ cont.356: does the 2D/HUD ASPECT CORRECTION apply to this primitive? Set just
        // below, per primitive, and stamped onto every vertex it emits -- a GL state GROUP can mix
        // HUD and non-HUD draws (measured: one state row carried both a 32 px icon and a 512 px
        // full-screen pass), so this cannot be a per-group flag.
        uint8_t hud2dFlag = 0;
        // ★★★★★ cont.358: and WHERE that correction is anchored. Stays at the centre unless the
        // game's anchor hook both exists and answers for this primitive -- so a title with no hook,
        // and every draw a hook declines, are bit-for-bit as they were before this existed.
        uint16_t hudAnchorCode = kHudAnchorCentre;
        auto emit = [&](const GSVertex &v, const GSVertex &cv)
        {
            GsGlVertex o;
            o.hud2d = hud2dFlag;
            o.hudAnchor = hudAnchorCode;
            o.x = v.x - ofx;
            o.y = v.y - ofy;
            // ★ cont.329 phase 4e. The census found depth BIMODAL -- 89.9% crushed below 0.1 and
            // 10.1% pinned at exactly 1.0, with nothing between. The pinned tenth is hitting the
            // clamp, and clamped fragments become mutually EQUAL at maximum nearness, write that
            // depth, and then reject everything behind them. The software path never has this
            // problem: it interpolates the raw value as a double and compares raw, with no
            // normalisation and no clamp.
            // PS2X_GS_GLR_ZMODE picks the correction so the candidates can be measured rather
            // than argued: 0 = as-was (clamp at the Z24 max), 1 = mask to the format's width, as
            // the hardware stores it, 2 = scale by 2^32 so nothing saturates at all.
            double zr = v.z;
            switch (g_glZMode)
            {
            case 1: zr = double(static_cast<uint64_t>(zr < 0.0 ? 0.0 : zr) & 0xFFFFFFull) / 16777215.0; break;
            case 2: zr = zr / 4294967295.0; break;
            default: zr = zr / static_cast<double>(zdiv); break;
            }
            o.z = static_cast<float>(zr);
            if (o.z < 0.f) o.z = 0.f;
            if (o.z > 1.f) o.z = 1.f;
            if (fst)
            {
                o.s = static_cast<float>(v.u) * fstScale * invTexW;
                o.t = static_cast<float>(v.v) * fstScale * invTexH;
                o.q = 1.0f;
            }
            else
            {
                o.s = v.s;
                o.t = v.t;
                // A zero/denormal q would divide to infinity in the shader; the GS clamps such a
                // primitive away, so give it a harmless coordinate instead of a NaN.
                o.q = (v.q > 1e-9f || v.q < -1e-9f) ? v.q : 1.0f;
            }
            o.r = cv.r; o.g = cv.g; o.b = cv.b; o.a = cv.a;
            o.f = v.fog; // ★★ cont.329j: the per-vertex fog value, interpolated screen-linearly
            if (g_glCensus && st.prim.fge) { ++g_glcFogHist[v.fog >> 5]; ++g_glcFogN; }
            if (g_glCensus && abe && bc == 0u) // C = As
            {
                ++g_glcAsN;
                ++g_glcAsHist[cv.a >> 5];
                if (cv.a > 0x80u) ++g_glcAsOver;
            }
            // ★ cont.329 phase 4e: the Z distribution. Depth is rejecting geometry that the
            // software path draws, and the two candidate causes -- a wrong scale and a wrong
            // direction -- look identical in a screenshot but completely different in a histogram.
            // ★ cont.329 phase 4i: what do the emitted vertices actually LOOK like? Untextured,
            // the wall is still absent, every primitive is translated, and correcting the
            // reference extent changed nothing -- so measure the vertices rather than reason about
            // them. Off-screen coordinates and near-white vertex colours are two very different
            // diagnoses and both are invisible in a screenshot.
            if (g_glVtxCensus)
            {
                ++g_glVtxN;
                if (o.x < g_glVtxMinX) g_glVtxMinX = o.x;
                if (o.x > g_glVtxMaxX) g_glVtxMaxX = o.x;
                if (o.y < g_glVtxMinY) g_glVtxMinY = o.y;
                if (o.y > g_glVtxMaxY) g_glVtxMaxY = o.y;
                if (o.x < 0.f || o.y < 0.f || o.x > 4096.f || o.y > 4096.f) ++g_glVtxOff;
                if (o.x != o.x || o.y != o.y || o.z != o.z) ++g_glVtxNaN;
                const unsigned lum = (unsigned(o.r) + unsigned(o.g) + unsigned(o.b)) / 3u;
                ++g_glVtxLum[lum >> 5];
                if ((g_glVtxN % 20000000ull) == 0ull)
                {
                    std::fprintf(stderr,
                                 "[gs2:glvtx] n=%llu x=[%.0f..%.0f] y=[%.0f..%.0f] offscreen=%llu nan=%llu | vtx-luma",
                                 g_glVtxN, g_glVtxMinX, g_glVtxMaxX, g_glVtxMinY, g_glVtxMaxY,
                                 g_glVtxOff, g_glVtxNaN);
                    for (unsigned k = 0; k < 8u; ++k)
                        std::fprintf(stderr, " %.0f%%", 100.0 * double(g_glVtxLum[k]) / double(g_glVtxN));
                    std::fprintf(stderr, "\n");
                }
            }
            if (g_glZCensus)
            {
                ++g_glZN;
                g_glZSum += o.z;
                if (o.z < g_glZMin) g_glZMin = o.z;
                if (o.z > g_glZMax) g_glZMax = o.z;
                ++g_glZHist[o.z >= 1.0f ? 9u : static_cast<unsigned>(o.z * 10.0f)];
                if ((g_glZN % 4000000ull) == 0ull)
                {
                    std::fprintf(stderr, "[gs2:glz] n=%llu min=%.6f max=%.6f mean=%.6f zmsk0=%llu zmsk1=%llu | hist",
                                 g_glZN, g_glZMin, g_glZMax, g_glZSum / double(g_glZN),
                                 g_glZMsk0, g_glZMsk1);
                    for (unsigned k = 0; k < 10u; ++k)
                        std::fprintf(stderr, " %.1f%%", 100.0 * double(g_glZHist[k]) / double(g_glZN));
                    std::fprintf(stderr, "\n");
                }
            }
            if (g_glCensus) ++g_glcVerts;
            b.verts.push_back(o);
        };

        if (g_glCensus)
        {
            // Guest screen coordinates: the XYOFFSET cancels in a difference, so no need to
            // subtract it here. The scissor is INCLUSIVE at both ends (cont.329 phase 2).
            const double scArea = double(int(sc.x1) - int(sc.x0) + 1) * double(int(sc.y1) - int(sc.y0) + 1);
            double a = 0.0;
            if (isSprite)
            {
                const GSVertex &s0 = it.vertices[0], &s1 = it.vertices[1];
                a = std::fabs(double(s1.x - s0.x) * double(s1.y - s0.y));
                ++g_glcNSprite;
                g_glcAreaSprite += a;
            }
            else
            {
                const GSVertex &t0v = it.vertices[0], &t1v = it.vertices[1], &t2v = it.vertices[2];
                a = 0.5 * std::fabs(double(t1v.x - t0v.x) * double(t2v.y - t0v.y) -
                                    double(t2v.x - t0v.x) * double(t1v.y - t0v.y));
                ++g_glcNTri;
                g_glcAreaTri += a;
            }
            g_glcAreaClipped += (a < scArea ? a : scArea);
            const uint32_t fm = static_cast<uint32_t>(st.context.frame.fbmsk);
            unsigned slot = 0;
            for (; slot < g_glcFbmskUsed; ++slot)
                if (g_glcFbmskVal[slot] == fm) break;
            if (slot == g_glcFbmskUsed && g_glcFbmskUsed < 8u)
                g_glcFbmskVal[g_glcFbmskUsed++] = fm;
            if (slot < 8u) { ++g_glcFbmskN[slot]; g_glcFbmskArea[slot] += (a < scArea ? a : scArea); }
        }
        // ★★★★★ cont.356: THE 2D CLASSIFIER for the anamorphic HUD counter-scale.
        //
        //   q == 1  &&  ZTST == ALWAYS  &&  x-extent < frac * scissor width
        //
        // Measured over two complete census windows on the Helm's Deep fight (115,186 draws / 20
        // states and 1,799,501 / 21) and confirmed by overlaying each row's bbox on the frame: this
        // selects EXACTLY the HUD -- the L1/X button badges, the text, the portrait/sword/ring
        // cluster and the untextured bar fills -- and nothing else.
        //
        // ⚠ Why not PRIM.FST, which is the GS's own 2D flag: it does not work on this game. Only
        // the TEXT is FST=1; every pictorial HUD element is FST=0, so an FST predicate corrects the
        // numbers and leaves the bars and portrait stretched. FST=1 is sufficient evidence of 2D
        // but NOT necessary -- a title can emit screen-space geometry through STQ, and this one does.
        //
        // ⚠ ZTST is load-bearing, not decoration: ~49k untextured SHADOW-VOLUME draws carry q == 1
        // too and are separated only by their ZTST=GREATER. And q can be outright garbage on
        // untextured geometry (one census row read -3.3e38), so this tests exact equality to 1.0 and
        // never a tolerance band.
        //
        // ⚠ The width term excludes full-screen 2D -- a fade, the tgt=1 composite, the letterbox
        // band -- which must NOT be counter-scaled or they grow side bars. It is a heuristic, hence
        // the knob; a game whose HUD legitimately spans the full width would need it raised.
        if (s_perspCount)
        {
            const unsigned nv = it.vertexCount < 3u ? unsigned(it.vertexCount) : 3u;
            bool allQ1 = nv > 0u;
            double xlo = 1e30, xhi = -1e30;
            for (unsigned vi = 0; vi < nv; ++vi)
            {
                if (it.vertices[vi].q != 1.0f) allQ1 = false;
                const double vx = double(it.vertices[vi].x);
                if (vx < xlo) xlo = vx;
                if (vx > xhi) xhi = vx;
            }
            ++g_pcDraws;
            if (!allQ1)
                ++g_pcPersp;
            else
            {
                const uint32_t zte = uint32_t((st.context.test >> 16) & 1u);
                const uint32_t ztst = uint32_t((st.context.test >> 17) & 3u);
                const double scW = double(sc.x1 >= sc.x0 ? sc.x1 - sc.x0 + 1 : 0);
                if (zte == 1u && ztst == 1u && scW > 0.0)
                {
                    if ((xhi - xlo) >= s_hudAspectFrac * scW) ++g_pcFull2d;
                    else                                      ++g_pcNarrow2d;
                }
            }
        }
        if (hudAspectFullOn())
        {
            const unsigned nv = it.vertexCount < 3u ? unsigned(it.vertexCount) : 3u;
            bool allQ1 = nv > 0u;
            double xlo = 1e30, xhi = -1e30;
            for (unsigned vi = 0; vi < nv; ++vi)
            {
                if (it.vertices[vi].q != 1.0f) allQ1 = false;
                const double vx = double(it.vertices[vi].x);
                if (vx < xlo) xlo = vx;
                if (vx > xhi) xhi = vx;
            }
            if (!allQ1)
                ++g_fsPersp;
            else
            {
                const uint32_t zte = uint32_t((st.context.test >> 16) & 1u);
                const uint32_t ztst = uint32_t((st.context.test >> 17) & 3u);
                const double scW = double(sc.x1 >= sc.x0 ? sc.x1 - sc.x0 + 1 : 0);
                if (zte == 1u && ztst == 1u && scW > 0.0 && (xhi - xlo) >= s_hudAspectFrac * scW)
                {
                    ++g_fsFull2d;
                    // ★ Only a TEXTURED full-screen draw is corrected. The untextured full-screen
                    // sprites on these screens are clears and fades, and they must keep covering the
                    // whole raster -- otherwise the pillarbox the correction creates has nothing
                    // behind it and the side bars show stale content instead of the clear colour.
                    // cont.356e: the draw-level apply is RETIRED -- nothing repaints the
                    // artwork per frame, so there was no draw to scale. Kept as a counter
                    // only, because it is what proved the apply point wrong.
                    if (g_fs2dActive.load(std::memory_order_relaxed) && st.prim.tme)
                        ++g_fsApplied;
                }
                else
                {
                    // Screen-space UI at/below the width gate: the EVIDENCE that this
                    // frame is a real 2D screen rather than a near-empty transition.
                    ++g_fsNarrow2d;
                }
            }
        }
        if (s_hudAspect)
        {
            const uint32_t hZte = uint32_t((st.context.test >> 16) & 1u);
            const uint32_t hZtst = uint32_t((st.context.test >> 17) & 3u);
            if (hZte == 1u && hZtst == 1u)   // ZTST_ALWAYS: an overlay, not depth-placed
            {
                const unsigned nv = it.vertexCount < 3u ? unsigned(it.vertexCount) : 3u;
                bool allQ1 = nv > 0u;
                double xlo = 1e30, xhi = -1e30;
                for (unsigned vi = 0; vi < nv; ++vi)
                {
                    if (it.vertices[vi].q != 1.0f) { allQ1 = false; break; }
                    const double vx = double(it.vertices[vi].x);
                    if (vx < xlo) xlo = vx;
                    if (vx > xhi) xhi = vx;
                }
                const double scW = double(sc.x1 >= sc.x0 ? sc.x1 - sc.x0 + 1 : 0);
                if (allQ1)
                {
                    if (scW > 0.0 && (xhi - xlo) < s_hudAspectFrac * scW)
                        hud2dFlag = 1u;
                }
                // ★★★★★ cont.358: ask the game where this one is anchored. Deliberately INSIDE
                // the classified branch -- the hook fires only for draws the engine has already
                // decided to counter-scale (measured at 106 of 109,729 on a Helm's Deep flip, and
                // 0.1% in both cont.356 census windows), so the cost is a predictable-branch load
                // per draw and an indirect call on a thousandth of them. The full bbox is built
                // here rather than above because the classifier itself needs only x, and a game
                // without a hook must not pay for the y pass.
                const Ps2xHudAnchorFn fn = (hud2dFlag && s_hudAnchorOn)
                                               ? g_hudAnchorFn.load(std::memory_order_acquire)
                                               : nullptr;
                const bool wantHudDump = hud2dFlag && s_hudDump && probeFlipOk();
                const bool wantHudClust = hud2dFlag && s_hudCluster && probeFlipOk();
                if (fn || wantHudDump || wantHudClust)
                {
                    {
                        double ylo = 1e30, yhi = -1e30;
                        for (unsigned vi = 0; vi < nv; ++vi)
                        {
                            const double vy = double(it.vertices[vi].y);
                            if (vy < ylo) ylo = vy;
                            if (vy > yhi) yhi = vy;
                        }
                        const uint32_t fbw = st.context.frame.fbw;
                        Ps2xHudDraw d;
                        // The bbox in the vertices' own space: emit() subtracts XYOFFSET, so the
                        // hook must see the same coordinates the shader will, not raw GS ones.
                        d.x0 = float(xlo - double(ofx));
                        d.x1 = float(xhi - double(ofx));
                        d.y0 = float(ylo - double(ofy));
                        d.y1 = float(yhi - double(ofy));
                        d.refW = float(fbw ? fbw * 64u : 640u);   // as buildGlBatch computes uRef
                        d.scissorX = float(sc.x0);
                        d.scissorY = float(sc.y0);
                        d.scissorW = float(scW);
                        d.scissorH = float(sc.y1 >= sc.y0 ? sc.y1 - sc.y0 + 1 : 0);
                        d.tbp0 = (texKey != 0u) ? uint32_t(st.context.tex0.tbp0) : 0u;
                        d.prim = uint32_t(st.prim.type);
                        d.screenKind = g_declaredKind.load(std::memory_order_relaxed);
                        // cont.358f: which block is this draw in? Looked up in the PREVIOUS flip's
                        // map; 0 when nothing contains it, which the contract says to decline.
                        d.clusterId = 0u; d.clusterDraws = 0u;
                        d.clusterX0 = d.clusterY0 = d.clusterX1 = d.clusterY1 = 0.f;
                        d.clusterTbpCount = 0u;
                        for (unsigned t = 0; t < 8u; ++t) d.clusterTbps[t] = 0u;
                        d.frameId = g_perfFlips.load(std::memory_order_relaxed);
                        if (fn)
                        {
                            std::lock_guard<std::mutex> lk(g_hudMapMx);
                            // ★★★★★ cont.358l: SPATIAL grouping, from the previous flip's map.
                            // ⚠ Emission ORDER was tried (cont.358i-k) and is not spatially
                            // coherent on this game -- the two HUD corners are interleaved in the
                            // draw stream, so order-based runs fracture a block and its own text
                            // lands in a different run from its bars. Union-find over positions
                            // does not care how the draws were ordered, and measured ZERO torn
                            // strings at realistic glyph distances in every revision that used it.
                            // The residual lag is harmless HERE because the allow-list means only
                            // the gameplay HUD can move at all, and that layout is static.
                            // ★ cont.358n: the match may be LOOSE now. A cluster used to decide
                            // which edge a draw went to, so a sloppy match could put two parts of
                            // one thing on opposite sides -- that is why this was strict. Since
                            // the side is taken from the draw's own position, a cluster now only
                            // answers "is this draw part of a HUD block", and being slightly off
                            // cannot change an anchor. So a draw that has drifted out of last
                            // frame's rect (a counter gaining a digit, an element easing into
                            // place) attaches to the nearest block instead of declining -- which
                            // is what was left of the flicker.
                            constexpr float kTol = 8.f;
                            const HudClusterRect *hit = nullptr;
                            for (const HudClusterRect &c : g_hudMap)
                                if (d.x0 >= c.x0 - kTol && d.x1 <= c.x1 + kTol &&
                                    d.y0 >= c.y0 - kTol && d.y1 <= c.y1 + kTol)
                                { hit = &c; break; }
                            if (!hit)
                            {
                                float nearest = s_hudClusterGap;
                                for (const HudClusterRect &c : g_hudMap)
                                {
                                    const float dx = std::max(0.f, std::max(c.x0 - d.x1, d.x0 - c.x1));
                                    const float dy = std::max(0.f, std::max(c.y0 - d.y1, d.y0 - c.y1));
                                    const float dist = std::max(dx, dy);
                                    if (dist <= nearest) { nearest = dist; hit = &c; }
                                }
                            }
                            g_hudFrameSeen.fetch_add(1, std::memory_order_relaxed);
                            if (hit)
                            {
                                d.clusterId = hit->id; d.clusterDraws = hit->n;
                                d.clusterX0 = hit->x0; d.clusterY0 = hit->y0;
                                d.clusterX1 = hit->x1; d.clusterY1 = hit->y1;
                                d.clusterTbpCount = hit->tbpN;
                                for (unsigned t = 0; t < 8u; ++t) d.clusterTbps[t] = hit->tbp[t];
                            }
                            else
                            {
                                g_hudFrameMiss.fetch_add(1, std::memory_order_relaxed);
                                if (probeFlipOk()) g_hudNoCluster.fetch_add(1, std::memory_order_relaxed);
                            }
                            if (g_hudPending.size() < 20000u)
                                g_hudPending.push_back(HudClusterDraw{d.x0, d.y0, d.x1, d.y1, d.tbp0});
                        }
                        // ★ cont.358c: the ANSWER is dumped beside the question. Without it the
                        // instrument shows which draws were offered to the policy but not what the
                        // policy did with them -- and "the table matched the right draws" is
                        // exactly the claim a region table needs checked.
                        float anchorRet = -1.0f;
                        if (fn)
                        {
                            anchorRet = fn(d);
                            hudAnchorCode = hudAnchorEncode(anchorRet);
                        }
                        if (wantHudClust)
                        {
                            std::lock_guard<std::mutex> lk(g_hudClusterMx);
                            if (g_hudClusterDraws.size() < 20000u)
                                g_hudClusterDraws.push_back(HudClusterDraw{d.x0, d.y0, d.x1, d.y1, d.tbp0});
                        }
                        if (wantHudDump)
                        {
                            const bool keep = (d.tbp0 >= s_hudDumpTbpMin) || (anchorRet >= 0.f) ||
                                              (s_hudDumpYMin > 0 && d.y0 >= float(s_hudDumpYMin));
                            if (keep) g_hudDumpTotal.fetch_add(1, std::memory_order_relaxed);
                            if (keep && g_hudDumpShown.fetch_add(1, std::memory_order_relaxed) < s_hudDumpMax)
                            {
                                char ans[32];
                                if (!fn)                  std::snprintf(ans, sizeof(ans), "no-hook");
                                else if (!(anchorRet >= 0.f)) std::snprintf(ans, sizeof(ans), "DECLINE");
                                else                      std::snprintf(ans, sizeof(ans), "%.3f", double(anchorRet));
                                // cont.358f: the FLIP, so a multi-screen capture can be split by
                                // scene and re-clustered offline at any threshold without a rebuild.
                                std::fprintf(stderr,
                                    "[gs2:huddump] flip=%llu x=[%.0f..%.0f] y=[%.0f..%.0f] w=%.0f h=%.0f "
                                    "tbp=%u prim=%u refW=%.0f scis=[%.0f,%.0f %.0fx%.0f] screen=%u anchor=%s\n",
                                    (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                                    d.x0, d.x1, d.y0, d.y1, d.x1 - d.x0, d.y1 - d.y0,
                                    d.tbp0, d.prim, d.refW,
                                    d.scissorX, d.scissorY, d.scissorW, d.scissorH, d.screenKind, ans);
                            }
                        }
                    }
                }
            }
        }
        const size_t uvProbeVStart = b.verts.size(); // cont.329k: this primitive's first vertex
        if (isSprite)
        {
            // A GS sprite is an axis-aligned rect from two opposite corners, flat-shaded from the
            // SECOND vertex (IIP does not apply to sprites). Expand to two triangles.
            const GSVertex &v0 = it.vertices[0];
            const GSVertex &v1 = it.vertices[1];
            GSVertex a = v0, bb = v1, c = v1, d = v0;
            a.x = v0.x; a.y = v0.y;
            bb.x = v1.x; bb.y = v0.y;
            c.x = v1.x; c.y = v1.y;
            d.x = v0.x; d.y = v1.y;
            a.z = bb.z = c.z = d.z = v1.z;
            // The rect's texture coordinates come from the SAME two corners as its position.
            a.u = v0.u; a.v = v0.v;   a.s = v0.s; a.t = v0.t; a.q = v1.q;
            bb.u = v1.u; bb.v = v0.v; bb.s = v1.s; bb.t = v0.t; bb.q = v1.q;
            c.u = v1.u; c.v = v1.v;   c.s = v1.s; c.t = v1.t; c.q = v1.q;
            d.u = v0.u; d.v = v1.v;   d.s = v0.s; d.t = v1.t; d.q = v1.q;
            emit(a, v1); emit(bb, v1); emit(c, v1);
            emit(a, v1); emit(c, v1); emit(d, v1);
        }
        else
        {
            // IIP=0 means flat shading from the LAST vertex; IIP=1 is gouraud per vertex.
            const GSVertex &flat = it.vertices[2];
            for (int k = 0; k < 3; ++k)
                emit(it.vertices[k], st.prim.iip ? it.vertices[k] : flat);
        }
        // Gated on the DRAWCMP arm point too, so the probe can be aimed at a chosen era rather
        // than always firing on the first screens the game draws.
        if (s_glUvProbe && g_glUvShown < s_glUvProbe && st.prim.tme &&
            b.verts.size() > uvProbeVStart + 2u &&
            (s_glUvProbeTW == 0 || m_draw.texW == s_glUvProbeTW) &&
            (s_drawCmpPrim == 0ull || g_perfCpuPrims >= s_drawCmpPrim) && probeFlipOk())
        {
            const GsGlVertex &v0 = b.verts[uvProbeVStart];
            const GsGlVertex &v1 = b.verts[uvProbeVStart + 2u];
            // The PRIMITIVE's own guest width -- not the group's span, which covers many draws.
            const float wpx = std::fabs(float(it.vertices[isSprite ? 1 : 2].x) - float(it.vertices[0].x));
            if (wpx >= s_glUvProbeW)
            {
                ++g_glUvShown;
                const GSVertex &g0 = it.vertices[0];
                const GSVertex &g1 = it.vertices[isSprite ? 1 : 2];
                std::fprintf(stderr,
                             "[gs2:uvprobe] flip=%llu %s tbp=%u tbw=%u psm=0x%02x tex=%dx%d fst=%u TEX0.tw/th=%u/%u "
                             "fstScale=%.6f fbw=%u | guest x %.0f..%.0f y %.0f..%.0f | raw u %u..%u v %u..%u "
                             "| stq (%.4f,%.4f,%.4f)..(%.4f,%.4f,%.4f) | EMITTED s %.4f..%.4f t %.4f..%.4f\n",
                             (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                             isSprite ? "SPR" : "TRI", m_draw.texTbp, m_draw.texTbw,
                             st.context.tex0.psm, m_draw.texW, m_draw.texH, st.prim.fst ? 1u : 0u,
                             st.context.tex0.tw, st.context.tex0.th, m_draw.texFstScale,
                             st.context.frame.fbw,
                             double(g0.x), double(g1.x), double(g0.y), double(g1.y),
                             unsigned(g0.u), unsigned(g1.u), unsigned(g0.v), unsigned(g1.v),
                             double(g0.s), double(g0.t), double(g0.q),
                             double(g1.s), double(g1.t), double(g1.q),
                             double(v0.s), double(v1.s), double(v0.t), double(v1.t));
            }
        }
        b.groups.back().count = static_cast<uint32_t>(b.verts.size()) - b.groups.back().first;
    }

    if (b.verts.empty())
    {
        if (g_glCensus) ++g_glcRunsNoVerts;
        return;
    }
    // The reference extent the transform divides by: the draw target's width from FBW, and the
    // tallest scissor seen. Phase 4 replaces this with real render-target tracking.
    const GSPrimitiveBatch &first = items[0];
    const uint64_t batchTargetKey = uint64_t(first.state.context.frame.fbp);
    const uint32_t batchZbp = first.state.context.zbuf.zbp, batchZpsm = first.state.context.zbuf.psm; // cont.344
    uint32_t refW = first.state.context.frame.fbw * 64u;
    if (refW == 0u) refW = 640u;
    // ★ cont.329 phase 4h: THE REFERENCE EXTENT IS THE TARGET, NOT THE SCISSOR.
    // The vertex transform divides game pixel coordinates by this to reach clip space, so it must
    // be the render target's size. Taking it from the tallest SCISSOR in the batch meant any run
    // drawn with a small scissor was divided by a small number -- scaling that geometry up and
    // throwing it off screen. Measured: with texturing disabled entirely the wall was still
    // absent, so the geometry was translated and drawn but landed nowhere visible.
    // The height is tracked as a per-target high-water mark so it is STABLE across batches;
    // coalescing runs with different extents would otherwise shift the mapping mid-frame.
    uint32_t &tgtH = m_glTargetH[batchTargetKey];
    // Publish for the present-side instance (see g_glTargetHByPage).
    const uint64_t tghKey = batchTargetKey & (kGs2Pages - 1u);
    for (const GsGlGroup &g : b.groups)
        if (static_cast<uint32_t>(g.sy1) + 1u > tgtH) tgtH = static_cast<uint32_t>(g.sy1) + 1u;
    if (tgtH > g_glTargetHByPage[tghKey].load(std::memory_order_relaxed))
        g_glTargetHByPage[tghKey].store(tgtH, std::memory_order_relaxed);
    uint32_t refH = tgtH;
    if (refH == 0u) refH = 448u;
    b.refW = refW;
    b.refH = refH;

    // ★ cont.329 phase 3c: COALESCE RUNS BEFORE SUBMITTING.
    // Measured (build 809, device-thread timing): a batch costs ~7.3 ms of driver work and
    // carries a mean of 42 vertices -- about fourteen triangles -- while the draw loop inside it
    // costs 0.03 ms. Across a 234 s run that was 181 s of submission overhead for 0.8 s of actual
    // drawing. The GPU is not the bottleneck; the number of submissions is. So accumulate runs and
    // ship fewer, larger batches.
    // ⚠ The bound is a FRAME: a resolve happens roughly every ten runs, and merging across one
    // would draw a run into the wrong frame. PS2X_GS_GLR_BATCHVERTS keeps the threshold tunable
    // without a rebuild so it can be swept against that limit rather than guessed.
    static const uint32_t kBatchVerts = []
    {
        const char *e = std::getenv("PS2X_GS_GLR_BATCHVERTS");
        const long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 512;
        return static_cast<uint32_t>(v < 1 ? 1 : (v > 1u << 20 ? 1u << 20 : v));
    }();

    // A target change ends the accumulation -- both the extent (the transform divides by it) and
    // the RENDER TARGET (draws for different framebuffers must not share a batch).
    if (!m_glPendVerts.empty() &&
        (m_glPendRefW != refW || m_glPendRefH != refH || m_glPendTargetKey != batchTargetKey ||
         m_glPendZbp != batchZbp || m_glPendZpsm != batchZpsm))
    {
        if (g_glCensus) ++g_glcFlushTargetChange;
        glFlushPending();
    }

    // ★★★★ cont.330e: an upload whose key an ALREADY-PENDING GROUP binds would overwrite the
    // very contents that group is supposed to sample, because the device applies every upload
    // before any draw and keeps one object per key. Flush first. (Target sources are exempt: they
    // are not uploaded, they bind a render target.)
    if (s_glTexSplit && !m_glPendVerts.empty() && !b.texUploads.empty())
    {
        bool overwrite = false;
        for (const GsGlTexUpload &u : b.texUploads)
        {
            for (const GsGlGroup &pg : m_glPendGroups)
                if (!pg.texIsTarget && pg.texKey == u.key) { overwrite = true; break; }
            if (overwrite) break;
        }
        if (overwrite)
        {
            ++g_glcFlushTexOverwrite;
            glFlushPending();
        }
    }

    const uint32_t base = static_cast<uint32_t>(m_glPendVerts.size());
    m_glPendVerts.insert(m_glPendVerts.end(), b.verts.begin(), b.verts.end());
    for (GsGlGroup g : b.groups)
    {
        g.first += base;
        m_glPendGroups.push_back(g);
    }
    for (GsGlTexUpload &u : b.texUploads)
        m_glPendTex.push_back(std::move(u));
    for (GsGlTargetSeed &sd : b.targetSeeds)
        m_glPendSeeds.push_back(std::move(sd));
    m_glPendRefW = refW;
    m_glPendRefH = refH;
    m_glPendTargetKey = batchTargetKey;
    m_glPendZbp = batchZbp; m_glPendZpsm = batchZpsm;

    m_glPendCount.store(static_cast<uint32_t>(m_glPendVerts.size()), std::memory_order_relaxed);
    g_glPendGlobal.store(m_glPendVerts.size(), std::memory_order_relaxed); // cont.331j
    if (m_glPendVerts.size() >= kBatchVerts)
    {
        if (g_glCensus) ++g_glcFlushThreshold;
        glFlushPending();
    }
}

void GSCpuBackend::glFlushPending()
{
    if (m_glPendVerts.empty())
        return;
    GsGpuPresentDevice *dev = gs2GpuDevice();
    if (!dev)
    {
        m_glPendVerts.clear();
        m_glPendGroups.clear();
        m_glPendTex.clear();
        m_glPendSeeds.clear();
        m_glPendCount.store(0u, std::memory_order_relaxed);
        g_glPendGlobal.store(0u, std::memory_order_relaxed); // cont.331j
        return;
    }
    if (g_glCensus)
    {
        ++g_glcBatchesOut;
        g_glcVertsOut += m_glPendVerts.size();
    }
    GsGlBatch out;
    out.verts.swap(m_glPendVerts);
    out.groups.swap(m_glPendGroups);
    out.texUploads.swap(m_glPendTex);
    out.targetSeeds.swap(m_glPendSeeds);
    out.refW = m_glPendRefW;
    out.refH = m_glPendRefH;
    out.targetKey = m_glPendTargetKey;
    out.zbp = m_glPendZbp; out.zpsm = m_glPendZpsm;
    out.trace = s_drawCmpTrace != 0ull && s_drawCmpPrim != 0ull &&
                g_perfCpuPrims >= s_drawCmpPrim && g_perfCpuPrims < s_drawCmpPrim + s_drawCmpTrace;
    m_glPendCount.store(0u, std::memory_order_relaxed);
    g_glPendGlobal.store(0u, std::memory_order_relaxed); // cont.331j
    dev->DrawBatchGl(std::move(out));
}
#endif

// ★★ cont.329h: snapshot the draw target at a run boundary of the armed flip. Counting only in
// every other frame; inside the armed frame it costs a full decode (CPU) or a flush + resolve +
// readback (GL) per run, which is the point -- it is one frame.
void GSCpuBackend::drawCmpCheckpoint(const GSPrimitiveBatch *items, size_t count)
{
    if (s_drawCmpPrim == 0ull || items == nullptr || count == 0u)
        return;
    if (g_perfCpuPrims < s_drawCmpPrim || g_drawCmpN >= s_drawCmpMax)
        return;
    size_t prims = 0;
    for (size_t i = 0; i < count; ++i)
        if (items[i].vertexCount != 0u)
            ++prims;
    g_drawCmpPrims += prims;
    const uint32_t w = g_drawCmpW ? g_drawCmpW : 640u;
    const uint32_t h = g_drawCmpH ? g_drawCmpH : 448u;
    // The run's own render target: the census split runs on it, so items[0] names it for the run.
    const GSFrameReg &fr = items[0].state.context.frame;
    thread_local std::vector<uint8_t> px;
    bool ok = false;
    const char *mode = "cpu";
#if PS2X_HAS_GS_GPU_DEVICE
    if (s_gsRendererGl)
    {
        mode = "gl";
        glFlushPending(); // everything up to this primitive must have been DRAWN, not pending
        if (GsGpuPresentDevice *dev = gs2GpuDevice())
            ok = dev->RenderFrameGl(uint64_t(fr.fbp), w, h, px);
    }
    else
#endif
        ok = CopyFrameToHostRgbaCpu(fr, w, h, px, false, true, true, 0u, 0u);
    if (!ok || px.empty())
        return;
    // Distinct RGB24 values, mean colour and lit-pixel count -- the three numbers that have
    // tracked this defect all arc (distinct colours CPU 28036 vs GL ~10-13k, mean luma 40 vs 190).
    static std::vector<uint64_t> seen(1u << 18); // 2^24 bits
    std::fill(seen.begin(), seen.end(), 0ull);
    unsigned long long sr = 0, sg = 0, sb = 0, lit = 0, distinct = 0, n = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
        {
            const uint8_t *p = px.data() + (static_cast<size_t>(y) * 640u + x) * 4u;
            sr += p[0]; sg += p[1]; sb += p[2];
            if ((unsigned(p[0]) + p[1] + p[2]) > 24u) ++lit;
            const uint32_t c = (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | p[2];
            uint64_t &wd = seen[c >> 6];
            const uint64_t bit = 1ull << (c & 63u);
            if (!(wd & bit)) { wd |= bit; ++distinct; }
            ++n;
        }
    const double fn = n ? double(n) : 1.0;
    std::fprintf(stderr,
                 "[gs2:drawcmp] mode=%s snap=%u at-prim=%llu since-arm=%llu run=%zu flip=%llu "
                 "fbp=%u fbw=%u psm=0x%x mean=(%.1f,%.1f,%.1f) luma=%.1f distinct=%llu lit=%.1f%%\n",
                 mode, g_drawCmpN, g_perfCpuPrims, g_drawCmpPrims, prims,
                 (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                 fr.fbp, fr.fbw, fr.psm,
                 double(sr) / fn, double(sg) / fn, double(sb) / fn,
                 (double(sr) + double(sg) + double(sb)) / (3.0 * fn), distinct,
                 100.0 * double(lit) / fn);
    if (s_drawCmpDir && s_drawCmpDir[0])
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/%s_%03u.ppm", s_drawCmpDir, mode, g_drawCmpN);
        if (FILE *f = std::fopen(path, "wb"))
        {
            std::fprintf(f, "P6\n%u %u\n255\n", w, h);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                {
                    const uint8_t *p = px.data() + (static_cast<size_t>(y) * 640u + x) * 4u;
                    std::fputc(p[0], f); std::fputc(p[1], f); std::fputc(p[2], f);
                }
            std::fclose(f);
        }
    }
    ++g_drawCmpN;
}

void GSCpuBackend::RasterRunFanOut(const GSPrimitiveBatch *items, size_t count)
{
    if (count == 0)
        return;
    // ★★★★ cont.331j ROWCENSUS: the renderer-INDEPENDENT seam. Noted here, ABOVE the
    // s_gsSkipCpuRaster / PS2X_GS_RENDERER guard below, so =cpu and =gl census the same draws.
    RowCensusDepth rcDepth;
    if ((s_rowCensus || s_drawPages) && rcDepth.top())
        for (size_t i = 0; i < count; ++i)
            if (items[i].vertexCount != 0u)
                rowCensusNote(items[i]);
    // ★ cont.231: the NORASTER ablation must skip the RUN path too. The flag was only checked inside
    // DrawPrimitive, which the band threads never enter (bandWorkerLoop -> RasterBand directly), so
    // with the pool on by default (cont.210) "raster ABLATED" runs still drew two of every three bands
    // -- a null ablation that read as "the raster is not the wall". Count the submissions, do no work.
    if (s_gsSkipCpuRaster)
    {
        for (size_t i = 0; i < count; ++i)
            if (items[i].vertexCount != 0u && (++g_perfCpuPrims % 500000ull) == 0ull)
                gs2PrintRasterThroughput();
#if PS2X_HAS_GS_GPU_DEVICE
        // ★ cont.329 phase 2a: translate the run into the GL geometry stream. NORASTER keeps the
        // old behaviour (count and drop) -- only the GL renderer builds anything.
        if (s_gsRendererGl)
            glSubmitRun(items, count);
#endif
        return;
    }
    // ★ Never fan out for the diagnostic paths: the per-pixel FONTTRACE counters and the CENSUS2
    // cyan probes are plain non-atomic statics, and the raster-verify bisect swaps m_vram around
    // the call. RasterFanOut has carried this guard since cont.207; the RUN path needs it too,
    // and it matters more now that the pool is on by default (cont.210). All are default OFF.
    if (m_rawDraw || gs2PerPixelDiag() || s_gsCensus2 || s_gsTexCensus || s_gsTexDiag /* cont.320: the oracle single-threaded by construction */ || s_gsTriCensus || s_gsGeoCensus || s_gsBigPrim > 0)
    {
        for (size_t i = 0; i < count; ++i)
            if (items[i].vertexCount != 0u)
                DrawPrimitive(items[i]);
        return;
    }
    {
        const auto key = [](const GSPrimitiveBatch &b) {
            const auto &c = b.state.context;
            return std::make_tuple(c.frame.fbp, c.frame.fbw, c.frame.psm, c.zbuf.zbp, c.zbuf.psm);
        };
        size_t start = 0;
        for (size_t i = 1; i <= count; ++i)
        {
            if (i == count || key(items[i]) != key(items[start]))
            {
                if (i - start != count)          // only recurse when an actual split happened
                {
                    RasterRunFanOut(items + start, i - start);
                    start = i;
                    continue;
                }
                break;                            // whole run shares one target: fall through
            }
        }
        if (start != 0)
            return;                               // handled by the split above
    }
    if (s_gsUpDep)
    {
        // cont.232 updep v2: the run's texture page set (per distinct TEX0, cached) and the RAW test
        // against the last four upload chunks.
        Gs2PageMask upUnion; for (unsigned k = 0; k < 4u; ++k) upUnion.orWith(t_upMasks[k]);
        Gs2PageMask &rm = t_runMasks[t_runMaskIdx & 3u]; rm.clear();
        struct TexCacheEnt { uint64_t key = ~0ull; Gs2PageMask m; bool raw = false; };
        thread_local TexCacheEnt cache[64];
        unsigned long long raw = 0;
        for (size_t i = 0; i < count; ++i)
        {
            const GSDrawState &st = items[i].state;
            if (items[i].vertexCount == 0u || !st.prim.tme) continue;
            const GSTex0Reg &t = st.context.tex0;
            const uint64_t key = (static_cast<uint64_t>(t.tbp0) << 40) ^ (static_cast<uint64_t>(t.tbw) << 32) ^
                                 (static_cast<uint64_t>(t.psm) << 24) ^ (static_cast<uint64_t>(t.tw) << 20) ^
                                 (static_cast<uint64_t>(t.th) << 16) ^ (static_cast<uint64_t>(t.cbp) & 0xFFFFull);
            TexCacheEnt &e = cache[(key ^ (key >> 17) ^ (key >> 31)) & 63u];
            if (e.key != key) { e.key = key; e.m.clear(); gs2TexPageMask(e.m, t); e.raw = e.m.intersects(upUnion); }
            rm.orWith(e.m);
            if (e.raw) ++raw;
        }
        ++t_runMaskIdx; if (t_runMasksValid < 4u) ++t_runMasksValid;
        g_gs2UpDep_runs.fetch_add(1, std::memory_order_relaxed);
        g_gs2UpDep_rawBatches.fetch_add(raw, std::memory_order_relaxed);
        if (raw) g_gs2UpDep_rawRuns.fetch_add(1, std::memory_order_relaxed);
    }
    // ★ cont.321b PS2X_GS_RAWSPLIT (default 1 = COUNT only; 2 = SPLIT; 0 = off): a draw whose TEXTURE pages
    // intersect the run's frame/Z pages (render-to-texture inside a run) reads what other threads are still
    // writing -- the banded raster has no order between threads inside a run. The hardware serialises it, so
    // the draw gets a run of its own (a barrier before and after). e59 at 7 threads hashed non-deterministically
    // once the writers got faster (build 766); the count says how often this game does it.
    // The run's frame/Z page set (the key split above made the target common to every item; the scissor
    // is the union). Used by the RAW scan below and published for the claim loops' note gate.
    Gs2PageMask tgt;
    {
        const GSContext &c0 = items[0].state.context;
        uint32_t mx = 0u, my = 0u;
        for (size_t i = 0; i < count; ++i)
        {
            const auto &sc = items[i].state.context.scissor;
            if (static_cast<uint32_t>(sc.x1) > mx) mx = static_cast<uint32_t>(sc.x1);
            if (static_cast<uint32_t>(sc.y1) > my) my = static_cast<uint32_t>(sc.y1);
        }
        gs2RectPages(tgt, c0.frame.fbp, c0.frame.fbw, c0.frame.psm, 0u, 0u, mx, my);
        gs2RectPages(tgt, c0.zbuf.zbp, c0.frame.fbw, c0.zbuf.psm, 0u, 0u, mx, my);
    }
    if (s_gsRawSplit != 0 && count > 1)
    {
        struct RawTexEnt { uint64_t key = ~0ull; Gs2PageMask m; };
        thread_local RawTexEnt rawCache[64];
        size_t start = 0;
        for (size_t i = 0; i < count; ++i)
        {
            const GSDrawState &st = items[i].state;
            if (items[i].vertexCount == 0u || !st.prim.tme) continue;
            const GSTex0Reg &t = st.context.tex0;
            const uint64_t key = (static_cast<uint64_t>(t.tbp0) << 40) ^ (static_cast<uint64_t>(t.tbw) << 32) ^
                                 (static_cast<uint64_t>(t.psm) << 24) ^ (static_cast<uint64_t>(t.tw) << 20) ^
                                 (static_cast<uint64_t>(t.th) << 16) ^ (static_cast<uint64_t>(t.cbp) & 0xFFFFull);
            RawTexEnt &e = rawCache[(key ^ (key >> 17) ^ (key >> 31)) & 63u];
            if (e.key != key) { e.key = key; e.m.clear(); gs2TexPageMask(e.m, t); }
            if (!e.m.intersects(tgt)) continue;
            g_gs2RawTexItems.fetch_add(1, std::memory_order_relaxed);
            if (s_gsRawSplitLog)
            {
                static unsigned long s_logged = 0;
                if (++s_logged <= 48ul || (s_logged % 2000ul) == 0ul)
                {
                    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
                    for (uint32_t k = 0; k < items[i].vertexCount; ++k)
                    {
                        const GSVertex &v = items[i].vertices[k];
                        x0 = std::min(x0, v.x); x1 = std::max(x1, v.x); y0 = std::min(y0, v.y); y1 = std::max(y1, v.y);
                    }
                    const GSVertex &va = items[i].vertices[0], &vb = items[i].vertices[items[i].vertexCount > 1u ? 1u : 0u];
                    std::fprintf(stderr, "[gs2:rawtex] #%lu run-item %zu/%zu type=%u n=%u frame{fbp=%u fbw=%u psm=%u} z{zbp=%u psm=%u} tex0{tbp0=%u tbw=%u psm=%u tw=%u th=%u cbp=%u} fst=%u lin=%u abe=%u xy=[%.0f..%.0f,%.0f..%.0f] uv0=(%u,%u) uv1=(%u,%u) st0=(%.3f,%.3f) st1=(%.3f,%.3f)\n",
                                 s_logged, i, count, static_cast<unsigned>(st.prim.type), items[i].vertexCount,
                                 st.context.frame.fbp, st.context.frame.fbw, st.context.frame.psm, st.context.zbuf.zbp, st.context.zbuf.psm,
                                 t.tbp0, t.tbw, t.psm, t.tw, t.th, t.cbp, st.prim.fst ? 1u : 0u, st.linearFilter ? 1u : 0u, st.prim.abe ? 1u : 0u,
                                 x0, x1, y0, y1, static_cast<unsigned>(va.u), static_cast<unsigned>(va.v), static_cast<unsigned>(vb.u), static_cast<unsigned>(vb.v),
                                 va.s, va.t, vb.s, vb.t);
                }
            }
            if (s_gsRawSplit >= 2)
            {
                // the hazardous draw alone: the run so far fans out, then the coordinator draws it WHOLE by
                // itself (a draw that samples its own target cannot be banded -- its rows read each other).
                if (i > start) RasterRunFanOut(items + start, i - start);
                DrawPrimitive(items[i]);
                if (s_gsThruput) { if (--g_perfPrintCountdown == 0u) { g_perfPrintCountdown = 500000u; gs2PrintRasterThroughput(); } ++g_perfCpuPrims; }
                g_gs2RawSplits.fetch_add(1, std::memory_order_relaxed);
                start = i + 1u;
            }
        }
        if (start != 0)
        {
            if (start < count) RasterRunFanOut(items + start, count - start);
            return;
        }
    }
    ++m_bandRuns;
    m_bandRunPrims += count;
    g_gs2RunCount.fetch_add(1, std::memory_order_relaxed);
    g_gs2RunPrims.fetch_add(count, std::memory_order_relaxed);
    g_gs2ItemsDraw.fetch_add(1, std::memory_order_relaxed);
    const auto t_fanout = std::chrono::steady_clock::now();
    g_gs2ItemDequeueNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_fanout - t_itemStart).count()), std::memory_order_relaxed);

    // ★ Band over the run's ACTUAL drawn extent, not the 2048-row VRAM space. The framebuffer is
    // only ~448 rows tall, so splitting 2048 evenly put every primitive in band 0 and left the
    // other threads idle -- measured as a dead-flat result, with 6 threads oddly beating 4 because
    // a smaller band 0 finally spilled into band 1.
    int runY0 = INT_MAX, runY1 = INT_MIN;
    for (size_t i = 0; i < count; ++i)
    {
        if (items[i].vertexCount == 0u)
            continue;
        const auto &sc = items[i].state.context.scissor;
        if (static_cast<int>(sc.y0) < runY0) runY0 = static_cast<int>(sc.y0);
        if (static_cast<int>(sc.y1) > runY1) runY1 = static_cast<int>(sc.y1);
    }
    if (runY0 > runY1)
        return;
    const bool fastCoord = s_gsFastCoord && !m_rawDraw && !s_gsStateCensus && !gs2PerPixelDiag() &&
                           !s_gsTexCensus && !s_gsClutVerify && !s_gsSpanVerify && !s_gsGeoCensus &&
                           !s_gsCensus2 && !s_gsNoRaster && !s_gsGpuBatchPerf
#if PS2X_HAS_GS_GPU_DEVICE
                           && s_gsGpuBatchVerify <= 0 && s_gsGpuRasterVerify <= 0
#endif
                           ;
    // cont.318: dynamic groups need the fast coordinator (the diagnostic DrawPrimitive path draws
    // whole primitives on the coordinator and must keep doing so).
    const bool dyn = s_gsBandDynRows > 0 && fastCoord;
    unsigned dynGroups = 0; int dynBase = 0, dynRows = 0;
    if (dyn)
    {
        dynRows = s_gsBandDynRows;
        dynBase = (runY0 >= 0 ? (runY0 / dynRows) : ((runY0 - dynRows + 1) / dynRows)) * dynRows;
        dynGroups = static_cast<unsigned>((runY1 - dynBase) / dynRows) + 1u;
        m_bandExt.resize(count * 2u);
        m_bandPg.resize(count * 4u);
        for (size_t i = 0; i < count; ++i)
        {
            gs2BandExtent(items[i], m_bandExt[i * 2u], m_bandExt[i * 2u + 1u]);
            const GSContext &c = items[i].state.context;
            const uint32_t fbw = std::max<uint32_t>(c.frame.fbw, 1u);
            const uint32_t span = (static_cast<uint32_t>(c.scissor.y1) >> 5) * fbw + (static_cast<uint32_t>(c.scissor.x1) >> 6);
            const uint32_t f0 = GSInternal::framePageBaseToBlock(c.frame.fbp) / GSMem::BLOCKS_PER_PAGE;
            const uint32_t z0 = GSInternal::framePageBaseToBlock(c.zbuf.zbp) / GSMem::BLOCKS_PER_PAGE;
            m_bandPg[i * 4u] = f0; m_bandPg[i * 4u + 1u] = f0 + span; m_bandPg[i * 4u + 2u] = z0; m_bandPg[i * 4u + 3u] = z0 + span;
        }
    }
    {
        std::lock_guard<std::mutex> lk(m_bandMutex);
        m_bandRunItems = items;
        m_bandRunCount = count;
        m_bandMinY = runY0;
        m_bandMaxY = runY1;
        m_bandBatch = nullptr;   // run mode, not single-primitive mode
        m_bandDynGroups = dynGroups;
        m_bandDynBase = dynBase;
        m_bandDynRows = dynRows;
        for (int k = 0; k < 8; ++k) m_bandTgtW[k] = tgt.w[k];
        m_bandDynNext.store(0u, std::memory_order_relaxed);
        m_bandPending.store(m_bandCount - 1u, std::memory_order_relaxed);
        m_bandSeq.fetch_add(1, std::memory_order_release);
        if (s_gsBandAdapt != 0u && m_bandOwnerSeq == 0u)
        {
            // first run: equal shares, the cont.210 round-robin layout
            for (unsigned i = 0; i < m_bandCount && i < kBandMaxParticipants; ++i)
                m_bandShare[i] = 1.0 / double(m_bandCount);
            bandRebuildOwners();
        }
    }
    m_bandStartCv.notify_all();

    // The coordinator owns band 0 and additionally runs each primitive's prologue (census,
    // markDrawPages, capture, GPU-mirror bookkeeping) exactly once, in submission order.
    const int rows = runY1 - runY0 + 1;
    const int per = (rows + static_cast<int>(m_bandCount) - 1) / static_cast<int>(m_bandCount);
    const bool striped = gs2BandStriped();
    m_bandRunActive = true;
    m_bandRunY0 = runY0;
    // ★ cont.210: striped, every participant clips to the WHOLE run extent and the row table
    // decides ownership; contiguous, band 0 is the first strip as in cont.209.
    m_bandRunY1 = striped ? runY1 : (runY0 + per - 1);
    {
        static int s_once = 0;
        if (s_once < 2 && std::getenv("PS2X_GS_BANDDEBUG"))
        {
            ++s_once;
            if (striped)
            {
                std::fprintf(stderr,
                             "[gs2:banddbg] run y=%d..%d rows=%d N=%u STRIPED shift=%d (%d rows/unit)",
                             runY0, runY1, rows, m_bandCount, s_gsBandStripeShift,
                             1 << s_gsBandStripeShift);
                for (unsigned k = 0; k < m_bandCount; ++k)
                {
                    const uint8_t *t = gs2BandRowTable(k, m_bandCount);
                    int owned = 0;
                    for (int y = runY0; y <= runY1; ++y)
                        owned += t[static_cast<unsigned>(y) & (kBandRowSpace - 1u)];
                    std::fprintf(stderr, " band%u=%drows%s", k, owned, owned ? "" : "(EMPTY)");
                }
                std::fprintf(stderr, "\n");
            }
            else
            {
                std::fprintf(stderr, "[gs2:banddbg] run y=%d..%d rows=%d N=%u per=%d | band0=%d..%d",
                             runY0, runY1, rows, m_bandCount, per, m_bandRunY0, m_bandRunY1);
                for (unsigned k = 1; k < m_bandCount; ++k)
                {
                    const int b0 = runY0 + static_cast<int>(k) * per;
                    const int b1 = (b0 + per - 1) < runY1 ? (b0 + per - 1) : runY1;
                    std::fprintf(stderr, " band%u=%d..%d%s", k, b0, b1, (b0 > b1) ? "(EMPTY)" : "");
                }
                std::fprintf(stderr, "\n");
            }
        }
    }
    {
        const uint8_t *coordRows = striped ? (s_gsBandAdapt != 0u ? gs2BandRowTableFrom(m_bandOwnerUnits, m_bandOwnerSeq, 0u)
                                                                  : gs2BandRowTable(0u, m_bandCount))
                                           : nullptr;
        BandRowScope bandScope(dyn ? nullptr : coordRows);
        if (dyn)
        {
            // Pre-pass, once per primitive in submission order: capture + the point path.
            for (size_t i = 0; i < count; ++i)
            {
                const GSPrimitiveBatch &b = items[i];
                if (b.vertexCount == 0u)
                    continue;
                rasterCapRecord(b);
                if (b.state.prim.type == GS_PRIM_POINT)
                    DrawPrimitive(b);
                if (s_gsThruput)
                {
                    if (--g_perfPrintCountdown == 0u)
                    {
                        g_perfPrintCountdown = 500000u;
                        gs2PrintRasterThroughput();
                    }
                    ++g_perfCpuPrims;
                }
            }
            bandDynClaimLoop(items, count, runY0, runY1, dynGroups, dynBase, dynRows);
        }
        else if (fastCoord)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const GSPrimitiveBatch &b = items[i];
                if (b.vertexCount == 0u)
                    continue;
                rasterCapRecord(b); // early-returns unless a capture is armed
                if (b.state.prim.type == GS_PRIM_POINT)
                {
                    DrawPrimitive(b); // the point path (WritePixel) lives there
                    continue;
                }
                if (s_gsBandSkip && !gs2BandOwnsAnyRow(b, coordRows))
                {
                    noteDrawWrites(b.state.context); // cont.320: its writes still stomp our cached pages
                    continue; // cont.318: no owned row in reach -- skip the whole prologue
                }
                RasterBand(b, m_bandRunY0, m_bandRunY1);
                if (s_gsThruput)
                {
                    if (--g_perfPrintCountdown == 0u)
                    {
                        g_perfPrintCountdown = 500000u;
                        gs2PrintRasterThroughput();
                    }
                    ++g_perfCpuPrims;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
                if (items[i].vertexCount != 0u)
                    DrawPrimitive(items[i]);
        }
    }
    m_bandRunActive = false;

    {
        const auto w0 = std::chrono::steady_clock::now();
        g_gs2CoordRasterNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(w0 - t_fanout).count()), std::memory_order_relaxed);
        if (s_gsBandSpinNs != 0ull)
        {
            // cont.318: poll the helpers' completion before sleeping (PCSX2 WaitForEmptyWithSpin).
            while (m_bandPending.load(std::memory_order_acquire) != 0u)
            {
                for (int k = 0; k < 16; ++k) _mm_pause();
                if (static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count()) >= s_gsBandSpinNs)
                    break;
            }
        }
        std::unique_lock<std::mutex> lk(m_bandMutex);
        m_bandDoneCv.wait(lk, [&] { return m_bandPending.load(std::memory_order_acquire) == 0u; });
        m_bandRunItems = nullptr;
        m_bandRunCount = 0;
        const auto w1 = std::chrono::steady_clock::now();
        {
            // cont.318: attribute the barrier. The coordinator is participant 0; its raster walk ran
            // t_fanout..w0 and it "finished" at w0. Workers wrote their end/busy under this mutex.
            const unsigned n = m_bandCount < kBandMaxParticipants ? m_bandCount : kBandMaxParticipants;
            m_bandEndNs[0] = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(w0.time_since_epoch()).count());
            m_bandRunBusyNs[0] = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(w0 - t_fanout).count());
            {
                static thread_local int lastCpu = -1;
                const int cpu = sched_getcpu();
                if (lastCpu != -1 && cpu != lastCpu) g_gs2BandMig[0].fetch_add(1, std::memory_order_relaxed);
                lastCpu = cpu; g_gs2BandCpu[0].store(cpu, std::memory_order_relaxed);
            }
            unsigned long long maxEnd = 0ull, maxBusy = 0ull; unsigned last = 0u;
            for (unsigned i = 0; i < n; ++i)
            {
                if (m_bandEndNs[i] > maxEnd) { maxEnd = m_bandEndNs[i]; last = i; }
                if (m_bandRunBusyNs[i] > maxBusy) maxBusy = m_bandRunBusyNs[i];
            }
            for (unsigned i = 0; i < n; ++i)
            {
                g_gs2BandBusyNs[i].fetch_add(m_bandRunBusyNs[i], std::memory_order_relaxed);
                g_gs2BandBarrierIdleNs[i].fetch_add(maxEnd - m_bandEndNs[i], std::memory_order_relaxed);
            }
            g_gs2BandLast[last].fetch_add(1, std::memory_order_relaxed);
            g_gs2BandRunsTimed.fetch_add(1, std::memory_order_relaxed);
            g_gs2BandN.store(n, std::memory_order_relaxed);
            if (s_gsBandAdapt != 0u && striped && maxBusy >= 200000ull)
            {
                // share_i <- blend(share_i, rate_i / sum rate), rate_i = share_i / busy_i: the fixed
                // point is equal finish times. Clamp so no participant starves or hogs, renormalise.
                const double alpha = double(s_gsBandAdapt) / 100.0;
                double rate[kBandMaxParticipants] = {}, sum = 0.0;
                for (unsigned i = 0; i < n; ++i)
                {
                    const double busy = double(m_bandRunBusyNs[i] < 1000ull ? 1000ull : m_bandRunBusyNs[i]);
                    rate[i] = m_bandShare[i] / busy;
                    sum += rate[i];
                }
                const double lo = 0.25 / double(n), hi = 4.0 / double(n);
                double tot = 0.0;
                for (unsigned i = 0; i < n; ++i)
                {
                    double sh = m_bandShare[i] * (1.0 - alpha) + (rate[i] / sum) * alpha;
                    if (sh < lo) sh = lo;
                    if (sh > hi) sh = hi;
                    m_bandShare[i] = sh;
                    tot += sh;
                }
                for (unsigned i = 0; i < n; ++i)
                {
                    m_bandShare[i] /= tot;
                    g_gs2BandShareView[i] = m_bandShare[i];
                }
                bandRebuildOwners();
            }
        }
        g_gs2CoordWaitNs.fetch_add(static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count()),
            std::memory_order_relaxed);
        g_gs2ItemTailNs.fetch_add(static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count()), std::memory_order_relaxed);
        t_itemStart = w1; // the next sub-run of this item measures only its own setup
    }
}

void GSCpuBackend::RasterBand(const GSPrimitiveBatch &batch, int bandY0, int bandY1)
{
    resolveDraw(batch.state);
    switch (batch.state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch, bandY0, bandY1);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch, bandY0, bandY1);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch, bandY0, bandY1);
        break;
    default:
        break;
    }
}

void GSCpuBackend::bandWorkerLoop(unsigned index)
{
    ThreadNaming::PinCurrentThreadForRoleIndex(ThreadNaming::CpuRole::GS, index); // cont.230 (auto plan / PS2X_GS_CPUS); cont.318 PS2X_GS_PIN1TO1 -> one CPU
    uint64_t seen = 0;
    for (;;)
    {
        const GSPrimitiveBatch *batch = nullptr;
        const GSPrimitiveBatch *runItems = nullptr;
        size_t runCount = 0;
        int y0 = 0, y1 = 0;
        uint64_t ownerSeq = 0;
        unsigned dynGroups = 0; int dynBase = 0, dynRows = 0;
        {
            const auto w0 = std::chrono::steady_clock::now();
            if (s_gsBandSpinNs != 0ull)
            {
                // cont.318: poll for the next run before sleeping (PCSX2 WaitForWorkWithSpin).
                while (m_bandSeq.load(std::memory_order_acquire) == seen && !m_bandStop.load(std::memory_order_relaxed))
                {
                    for (int k = 0; k < 16; ++k) _mm_pause();
                    if (static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count()) >= s_gsBandSpinNs)
                        break;
                }
            }
            std::unique_lock<std::mutex> lk(m_bandMutex);
            m_bandStartCv.wait(lk, [&] { return m_bandStop.load(std::memory_order_relaxed) || m_bandSeq.load(std::memory_order_acquire) != seen; });
            g_gs2BandIdleNs.fetch_add(static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count()),
                std::memory_order_relaxed);
            if (m_bandStop.load(std::memory_order_relaxed))
                return;
            seen = m_bandSeq.load(std::memory_order_acquire);
            batch = m_bandBatch;
            runItems = m_bandRunItems;
            runCount = m_bandRunCount;
            y0 = m_bandMinY;
            y1 = m_bandMaxY;
            ownerSeq = m_bandOwnerSeq;
            dynGroups = m_bandDynGroups;
            dynBase = m_bandDynBase;
            dynRows = m_bandDynRows;
        }
        const auto tb0 = std::chrono::steady_clock::now();
        // cont.210: striped -> clip to the whole extent and let the row table pick the rows;
        // contiguous (PS2X_GS_BAND_STRIPE<0) -> the cont.209 slice of [y0, y1] for helper `index`.
        const bool striped = gs2BandStriped();
        const int rows = y1 - y0 + 1;
        const int per = (rows + static_cast<int>(m_bandCount) - 1) / static_cast<int>(m_bandCount);
        const int cb0 = y0 + static_cast<int>(index) * per;
        const int b0 = striped ? y0 : cb0;
        const int b1 = striped ? y1 : ((cb0 + per - 1) < y1 ? (cb0 + per - 1) : y1);
        if (runItems != nullptr && dynGroups != 0u)
        {
            // cont.318 dynamic groups: claim until the run is exhausted.
            BandRowScope bandScope(nullptr);
            bandDynClaimLoop(runItems, runCount, y0, y1, dynGroups, dynBase, dynRows);
        }
        else if (runItems != nullptr)
        {
            // cont.209 run mode: this thread's share of the run's drawn extent, every primitive.
            if (b0 <= b1)
            {
                // cont.318: the adaptive owner table is only rewritten by the coordinator AFTER the
                // barrier, so reading it here (before our m_bandPending release) is race-free.
                const uint8_t *myRows = striped ? (s_gsBandAdapt != 0u ? gs2BandRowTableFrom(m_bandOwnerUnits, ownerSeq, index)
                                                                       : gs2BandRowTable(index, m_bandCount))
                                                : nullptr;
                BandRowScope bandScope(myRows);
                for (size_t i = 0; i < runCount; ++i)
                {
                    const GSPrimitiveBatch &b = runItems[i];
                    if (b.vertexCount == 0u)
                        continue;
                    if (s_gsBandSkip && !gs2BandOwnsAnyRow(b, myRows))
                    {
                        noteDrawWrites(b.state.context); // cont.320: its writes still stomp our cached pages
                        continue; // cont.318: no owned row in reach -- skip the whole prologue
                    }
                    RasterBand(b, b0, b1);
                }
            }
        }
        else if (batch != nullptr)
        {
            if (b0 <= b1)
            {
                BandRowScope bandScope(striped ? gs2BandRowTable(index, m_bandCount) : nullptr);
                RasterBand(*batch, b0, b1);
            }
        }
        {
            const auto tb1 = std::chrono::steady_clock::now();
            {
                static thread_local int lastCpu = -1;
                const int cpu = sched_getcpu();
                if (lastCpu != -1 && cpu != lastCpu) g_gs2BandMig[index < kGs2BandMax ? index : 0u].fetch_add(1, std::memory_order_relaxed);
                lastCpu = cpu; g_gs2BandCpu[index < kGs2BandMax ? index : 0u].store(cpu, std::memory_order_relaxed);
            }
            std::lock_guard<std::mutex> lk(m_bandMutex);
            if (index < kBandMaxParticipants)
            {
                m_bandEndNs[index] = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tb1.time_since_epoch()).count());
                m_bandRunBusyNs[index] = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tb1 - tb0).count());
            }
            if (m_bandPending.fetch_sub(1, std::memory_order_acq_rel) == 1u)
                m_bandDoneCv.notify_one();
        }
    }
}

// cont.318: the dynamic claim loop shared by the coordinator and the helpers. Group g covers rows
// [dynBase + g*dynRows, +dynRows-1] clipped to the run's drawn extent; the extents table
// (m_bandExt, published before m_bandSeq and immutable during the run) says which primitives can
// touch it. Everything a group needs is passed by value so the loop never re-reads run state.
void GSCpuBackend::bandDynClaimLoop(const GSPrimitiveBatch *items, size_t count, int runY0, int runY1,
                                    unsigned dynGroups, int dynBase, int dynRows)
{
    const int32_t *ext = m_bandExt.data();
    const uint32_t *pgt = m_bandPg.data();
    // cont.321b: do any of THIS thread's cached pages (levels, the CLUT page) lie in the run's target pages?
    // Only then can a skipped primitive stomp them; the per-primitive test below is gated on it, and it is
    // re-evaluated whenever this thread fills a slot (a level cached mid-run may be render-to-texture).
    const auto tgtHas = [this](uint32_t page) { page &= 511u; return (m_bandTgtW[page >> 6] >> (page & 63u)) & 1ull; };
    unsigned long long fillsSeen = ~0ull; bool noteNeeded = false;
    const auto noteGate = [&]() {
        fillsSeen = t_texFillSeq; noteNeeded = false;
        if (!s_gsDynNote) return;
        if (m_draw.clutKey != ~0ull && !m_draw.clutStomped && tgtHas(m_draw.clutPage)) { noteNeeded = true; return; }
        if (!s_gsTexCache) return;
        for (const TexSlot &sl : t_texSlots)
        {
            if (sl.key == ~0ull || sl.stomped) continue;
            for (uint32_t i = 0; i < sl.nPages; ++i) if (tgtHas(sl.pages[i])) { noteNeeded = true; return; }
        }
    };
    bool claimedAny = false;
    for (;;)
    {
        const unsigned g = m_bandDynNext.fetch_add(1u, std::memory_order_acq_rel);
        if (g >= dynGroups)
        {
            // ★ cont.321b: a thread that claims NO group of this run never walks its primitives, so it never
            // learns that the run overwrote the pages one of its cached levels / its CLUT page came from --
            // the stale copy then serves the next run (e59 at 7 threads: ~5-13% of replays diverged). Walk
            // the run once, notes only, when the gate says a cached page lies in the run's target.
            if (!claimedAny)
            {
                noteGate();
                if (noteNeeded)
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (items[i].vertexCount == 0u) continue;
                        const uint32_t *pg = pgt + i * 4u;
                        bool hit = false;
                        if (m_draw.clutKey != ~0ull && !m_draw.clutStomped)
                        {
                            const uint32_t cp = m_draw.clutPage;
                            hit = (cp >= pg[0] && cp <= pg[1]) || (cp >= pg[2] && cp <= pg[3]);
                        }
                        if (!hit && s_gsTexCache)
                        {
                            if (t_texUnionDirty) texUnionRefresh();
                            hit = t_texUnionLo <= t_texUnionHi &&
                                  ((pg[0] <= t_texUnionHi && pg[1] >= t_texUnionLo) || (pg[2] <= t_texUnionHi && pg[3] >= t_texUnionLo));
                        }
                        if (hit) noteDrawWrites(items[i].state.context);
                    }
            }
            return;
        }
        claimedAny = true;
        noteGate();
        int gy0 = dynBase + static_cast<int>(g) * dynRows;
        int gy1 = gy0 + dynRows - 1;
        if (gy0 < runY0) gy0 = runY0;
        if (gy1 > runY1) gy1 = runY1;
        if (gy0 > gy1)
            continue;
        for (size_t i = 0; i < count; ++i)
        {
            const int32_t e0 = ext[i * 2u], e1 = ext[i * 2u + 1u];
            if (e0 > e1 || e1 < gy0 || e0 > gy1)
            {
                // ★ cont.321b: a primitive this group never rasterizes still overwrites, on other rows, the
                // pages a cached palette / texture level came from. The striped loop notes that for its
                // skipped draws (cont.320); the dynamic loop skipped them SILENTLY, so a thread that never
                // owned a writer's rows kept a stale decoded texture -- a render-to-texture run (e59) then
                // hashed non-deterministically at 7 threads once the writers got faster (build 766).
                if (noteNeeded && items[i].vertexCount != 0u)
                {
                    // Four compares against this thread's cached pages first (build 767 called noteDrawWrites
                    // for every skipped primitive: +5..9% on the captures); the full walk only on a hit.
                    const uint32_t *pg = pgt + i * 4u;
                    bool hit = false;
                    if (m_draw.clutKey != ~0ull && !m_draw.clutStomped)
                    {
                        const uint32_t cp = m_draw.clutPage;
                        hit = (cp >= pg[0] && cp <= pg[1]) || (cp >= pg[2] && cp <= pg[3]);
                    }
                    if (!hit && s_gsTexCache)
                    {
                        if (t_texUnionDirty) texUnionRefresh();
                        hit = t_texUnionLo <= t_texUnionHi &&
                              ((pg[0] <= t_texUnionHi && pg[1] >= t_texUnionLo) || (pg[2] <= t_texUnionHi && pg[3] >= t_texUnionLo));
                    }
                    if (hit)
                        noteDrawWrites(items[i].state.context);
                }
                continue;
            }
            RasterBand(items[i], gy0, gy1);
            if (t_texFillSeq != fillsSeen) noteGate();
        }
    }
}

// cont.318: lay the stripe units out by smooth weighted round-robin over m_bandShare (deficit
// credit: every unit goes to the participant with the largest accumulated credit, which then pays
// one unit). Called by the coordinator under m_bandMutex, only between runs.
void GSCpuBackend::bandRebuildOwners()
{
    const unsigned n = m_bandCount < kBandMaxParticipants ? m_bandCount : kBandMaxParticipants;
    double credit[kBandMaxParticipants] = {};
    for (unsigned u = 0; u < kBandUnitSpace; ++u)
    {
        unsigned best = 0u;
        double bc = -1e300;
        for (unsigned i = 0; i < n; ++i)
        {
            credit[i] += m_bandShare[i];
            if (credit[i] > bc) { bc = credit[i]; best = i; }
        }
        credit[best] -= 1.0;
        m_bandOwnerUnits[u] = static_cast<uint8_t>(best);
    }
    ++m_bandOwnerSeq;
}

void GSCpuBackend::startBandPool()
{
    m_bandCount = s_gsRasterThreads;
    if (m_bandCount <= 1u)
        return;
    m_bandStop.store(false, std::memory_order_relaxed);
    for (unsigned i = 1; i < m_bandCount; ++i)
        m_bandThreads.emplace_back(&GSCpuBackend::bandWorkerLoop, this, i);
}

void GSCpuBackend::stopBandPool()
{
    if (m_bandThreads.empty())
        return;
    {
        std::lock_guard<std::mutex> lk(m_bandMutex);
        m_bandStop.store(true, std::memory_order_relaxed);
    }
    m_bandStartCv.notify_all();
    for (auto &t : m_bandThreads)
        if (t.joinable())
            t.join();
    m_bandThreads.clear();
}

void GSCpuBackend::RasterFanOut(const GSPrimitiveBatch &batch, int minY, int maxY)
{
    const int rows = maxY - minY + 1;
    // ★ Never fan out for the diagnostic paths: the per-pixel FONTTRACE counters and the CENSUS2
    // cyan probes are plain non-atomic statics, and the raster-verify bisect swaps m_vram around
    // the call. All are default OFF, so this costs nothing in a normal run.
    const long long approxPix = static_cast<long long>(rows) * gs2BandWidth(batch);
    if (m_bandThreads.empty() || m_rawDraw || gs2PerPixelDiag() || s_gsCensus2 || s_gsTexCensus || s_gsTexDiag /* cont.320: the oracle single-threaded by construction */ || s_gsTriCensus || s_gsGeoCensus || s_gsBigPrim > 0 ||
        approxPix < s_gsBandMinPix || rows < static_cast<int>(m_bandCount))
    {
        ++m_bandDirect;
        g_gs2BandDirect.fetch_add(1, std::memory_order_relaxed);
        RasterBand(batch, INT_MIN, INT_MAX);
        return;
    }
    ++m_bandFanOuts;
    g_gs2BandFan.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(m_bandMutex);
        m_bandBatch = &batch;
        m_bandMinY = minY;
        m_bandMaxY = maxY;
        m_bandPending = m_bandCount - 1u;
        ++m_bandSeq;
    }
    m_bandStartCv.notify_all();
    // The caller is band 0.
    const int per = (rows + static_cast<int>(m_bandCount) - 1) / static_cast<int>(m_bandCount);
    const bool striped = gs2BandStriped();
    const int b1 = striped ? maxY : ((minY + per - 1) < maxY ? (minY + per - 1) : maxY);
    {
        BandRowScope bandScope(striped ? gs2BandRowTable(0u, m_bandCount) : nullptr);
        RasterBand(batch, minY, b1);
    }
    {
        std::unique_lock<std::mutex> lk(m_bandMutex);
        m_bandDoneCv.wait(lk, [&] { return m_bandPending == 0u; });
        m_bandBatch = nullptr;
    }
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    // ★★★★ cont.331j ROWCENSUS: the other renderer-independent entry. Depth 0 only -- inside a
    // RasterRunFanOut the run has already been noted.
    if ((s_rowCensus || s_drawPages) && t_rowCensusDepth == 0u && batch.vertexCount != 0u)
        rowCensusNote(batch);
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    if (!m_rawDraw)
        rasterCapRecord(batch);
    if (s_gsStateCensus)
        stateCensusRecord(batch);
    // PS2X_GS_FONTTRACE: draw-ORDER stamps -- a glyph draw vs the full-screen cover quad. The
    // menu text is drawn with white texels (proven by [gs2:fontsamp]) yet the completed frame
    // holds none: if an untextured full-width sprite lands on fbp 0/128 AFTER the glyphs (seq
    // stamp = g_gs2Draw.total), the cover is erasing the text -- a path-order divergence.
    if (g_gs2FontTrace && !state.prim.tme && state.prim.type == GS_PRIM_SPRITE &&
        batch.vertexCount >= 2u && ctx.frame.fbp <= 128u &&
        std::fabs(batch.vertices[1].x - batch.vertices[0].x) > 400.0f)
    {
        static unsigned long s_nc = 0;
        if (++s_nc <= 64u)
            std::fprintf(stderr,
                         "[gs2:coverdraw] seq=%lu #%lu fbp=%u w=%.0f rgba=(%u,%u,%u,%u) abe=%u v0{%.1f,%.1f} v1{%.1f,%.1f}\n",
                         g_gs2Draw.total, s_nc, ctx.frame.fbp,
                         std::fabs(batch.vertices[1].x - batch.vertices[0].x),
                         batch.vertices[1].r, batch.vertices[1].g, batch.vertices[1].b, batch.vertices[1].a,
                         state.prim.abe ? 1u : 0u,
                         batch.vertices[0].x, batch.vertices[0].y, batch.vertices[1].x, batch.vertices[1].y);
    }
    // PS2X_GS_FONTTRACE: does a menu-font glyph draw (T4HL/T4HH texture, CLUT at 12288-12290)
    // reach the rasterizer at all, and with what state?
    if (g_gs2FontTrace && state.prim.tme &&
        (ctx.tex0.psm == GS_PSM_T4HL || ctx.tex0.psm == GS_PSM_T4HH) &&
        ctx.tex0.cbp >= 12288u && ctx.tex0.cbp <= 12290u)
    {
        static unsigned long s_n = 0;
        if (++s_n <= 48u)
            std::fprintf(stderr,
                         "[gs2:fontdraw] seq=%lu #%lu prim=%u n=%u abe=%u aa1=%u fst=%u "
                         "texclut{cbw=%u cou=%u cov=%u} "
                         "tex0{tbp=%u bw=%u psm=0x%x cbp=%u csa=%u tfx=%u tcc=%u} test=%llx alpha=%llx "
                         "frame{fbp=%u psm=0x%x msk=%x} scis{%u-%u,%u-%u} off{%u,%u} "
                         "v0{%.1f,%.1f uv=%u,%u rgba=%u,%u,%u,%u} v1{%.1f,%.1f}\n",
                         g_gs2Draw.total, s_n, static_cast<unsigned>(state.prim.type), batch.vertexCount,
                         state.prim.abe ? 1u : 0u, state.prim.aa1 ? 1u : 0u, state.prim.fst ? 1u : 0u,
                         static_cast<unsigned>(state.texclut.cbw), static_cast<unsigned>(state.texclut.cou),
                         static_cast<unsigned>(state.texclut.cov),
                         ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, ctx.tex0.cbp, ctx.tex0.csa,
                         ctx.tex0.tfx, ctx.tex0.tcc,
                         static_cast<unsigned long long>(ctx.test),
                         static_cast<unsigned long long>(ctx.alpha),
                         ctx.frame.fbp, ctx.frame.psm, ctx.frame.fbmsk,
                         ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1,
                         ctx.xyoffset.ofx, ctx.xyoffset.ofy,
                         batch.vertices[0].x, batch.vertices[0].y,
                         batch.vertices[0].u >> 4, batch.vertices[0].v >> 4,
                         batch.vertices[0].r, batch.vertices[0].g, batch.vertices[0].b, batch.vertices[0].a,
                         batch.vertices[1].x, batch.vertices[1].y);
    }
    if (s_gsTexCensus)
    {
        // Per-draw cadence (not per-pixel): lets a real game run report the same census the
        // deterministic bench does, so the capture's texture mix can be checked for drift.
        static unsigned long s_texCensusDraws = 0;
        if ((++s_texCensusDraws % 200000ul) == 0ul)
            gs2TexCensusDump("game");
    }
    if (s_gsClutVerify || s_gsTexVerify) // cont.320: the texture oracle's dump no longer needs CLUTVERIFY
    {
        static unsigned long s_clutVerifyDraws = 0;
        if ((++s_clutVerifyDraws % 200000ul) == 0ul)
        {
            if (s_gsClutVerify) gs2ClutVerifyDump("game");
            gs2TexVerifyDump("game");
        }
    }
    if (s_gsSpanVerify)
    {
        static unsigned long s_spanVerifyDraws = 0;
        if ((++s_spanVerifyDraws % 200000ul) == 0ul)
            gs2SpanVerifyDump("game");
    }
    if (s_gsGeoCensus)
    {
        static unsigned long s_geoCensusDraws = 0;
        if ((++s_geoCensusDraws % 500000ul) == 0ul)
            gs2GeoCensusDump("game");
    }
    if (s_gsCensus2)
    {
        ++g_gs2Draw.total;
        if (state.prim.tme)
        {
            ++g_gs2Draw.tme;
            switch (ctx.tex0.psm)
            {
            case GS_PSM_T4: ++g_gs2Draw.psmT4; break;
            case GS_PSM_T8: ++g_gs2Draw.psmT8; break;
            case GS_PSM_CT32: ++g_gs2Draw.psmCt32; break;
            case GS_PSM_CT16: case GS_PSM_CT16S: ++g_gs2Draw.psmCt16; break;
            default: ++g_gs2Draw.psmOther; break;
            }
            g_gs2Draw.lastTbp0 = ctx.tex0.tbp0;
            g_gs2Draw.lastPsm = ctx.tex0.psm;
            g_gs2Draw.lastTw = ctx.tex0.tw;
            g_gs2Draw.lastTh = ctx.tex0.th;
            g_gs2Draw.lastTfx = ctx.tex0.tfx;
            g_gs2Draw.lastFbp = ctx.frame.fbp;
        }
        // Era gate (PS2X_GS_ERA_FILE=<path>): closed until the sentinel file exists (the run
        // driver touches it at MOVIE-END), so censuses/dumps are era-locked to the level.
        // Without the env the gate is always open (old behavior). Rechecked every 2048 draws.
        {
            static const char *s_eraFile = std::getenv("PS2X_GS_ERA_FILE");
            if (s_eraFile && !g_gs2EraOpen && (g_gs2Draw.total & 2047u) == 0u)
            {
                if (FILE *ef = std::fopen(s_eraFile, "rb"))
                {
                    std::fclose(ef);
                    g_gs2EraOpen = true;
                    g_gs2EraOpenDraw = g_gs2Draw.total;
                    std::fprintf(stderr, "[gs2:era] OPEN at draw %lu -- degen counters reset\n",
                                 g_gs2Draw.total);
                    g_gs2DegenReset = true;
                }
            }
            else if (!s_eraFile)
            {
                g_gs2EraOpen = true;
            }
        }
        // One-shot level-era VRAM dump (PS2X_GS_VRAMDUMP=<path>): the full 4MB local memory,
        // for offline texture decoding. With an era gate: 30000 draws after the gate opens;
        // without: at draw 420000 (legacy).
        if (m_vram && ((g_gs2EraOpenDraw != 0u && g_gs2Draw.total == g_gs2EraOpenDraw + 30000u) ||
                       (g_gs2EraOpenDraw == 0u && g_gs2Draw.total == 420000u)))
        {
            if (const char *vp = std::getenv("PS2X_GS_VRAMDUMP"))
            {
                if (FILE *f = std::fopen(vp, "wb"))
                {
                    std::fwrite(m_vram, 1, m_vramSize, f);
                    std::fclose(f);
                    std::fprintf(stderr, "[gs2:vramdump] wrote %s (%u bytes) at draw %lu\n",
                                 vp, m_vramSize, g_gs2Draw.total);
                }
            }
        }
        // Draw-target trace (part of PS2X_GS_CENSUS2): one line per FRAME/SCISSOR/XYOFFSET change.
        {
            static uint64_t lastKey = ~0ull;
            const uint64_t key = (static_cast<uint64_t>(ctx.frame.fbp) << 40) ^
                                 (static_cast<uint64_t>(ctx.frame.fbw) << 32) ^
                                 (static_cast<uint64_t>(ctx.frame.psm) << 24) ^
                                 (static_cast<uint64_t>(ctx.scissor.x1) << 12) ^
                                 (static_cast<uint64_t>(ctx.scissor.y1)) ^
                                 (static_cast<uint64_t>(ctx.xyoffset.ofx) << 48);
            if (key != lastKey)
            {
                lastKey = key;
                std::fprintf(stderr,
                             "[gs2:target] frame{fbp=%u fbw=%u psm=0x%x} scissor=(%u..%u,%u..%u) ofx=%u ofy=%u\n",
                             ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm,
                             ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1,
                             ctx.xyoffset.ofx >> 4, ctx.xyoffset.ofy >> 4);
            }
        }
        // Statistical draw sampler (1-in-4096, era-tagged by total): the first-N peeks below
        // sample one instant and mislead (rule 8/16); this gives the population.
        if ((g_gs2Draw.total % 4096u) == 7u && batch.vertexCount >= 2u)
        {
            const auto &v0 = batch.vertices[0];
            const auto &v1 = batch.vertices[1];
            const auto &v2 = batch.vertices[batch.vertexCount >= 3u ? 2u : 1u];
            const float xmin = std::min({v0.x, v1.x, v2.x});
            const float xmax = std::max({v0.x, v1.x, v2.x});
            const float ymin = std::min({v0.y, v1.y, v2.y});
            const float ymax = std::max({v0.y, v1.y, v2.y});
            const uint8_t ztst = static_cast<uint8_t>((ctx.test >> 17) & 0x3u);
            const uint8_t zmsk = ctx.zbuf.zmask ? 1u : 0u;
            char uvinfo[96];
            if (state.prim.tme && state.prim.fst)
                std::snprintf(uvinfo, sizeof(uvinfo), "uv=(%u..%u,%u..%u)",
                              std::min({v0.u, v1.u, v2.u}) >> 4, std::max({v0.u, v1.u, v2.u}) >> 4,
                              std::min({v0.v, v1.v, v2.v}) >> 4, std::max({v0.v, v1.v, v2.v}) >> 4);
            else if (state.prim.tme)
                std::snprintf(uvinfo, sizeof(uvinfo), "stq0=(%g,%g,%g)", v0.s, v0.t, v0.q);
            else
                std::snprintf(uvinfo, sizeof(uvinfo), "flat rgba=(%u,%u,%u,%u) alpha=0x%llx pabe=%u",
                              static_cast<unsigned>(v2.r), static_cast<unsigned>(v2.g),
                              static_cast<unsigned>(v2.b), static_cast<unsigned>(v2.a),
                              static_cast<unsigned long long>(ctx.alpha), state.pabe ? 1u : 0u);
            std::fprintf(stderr,
                         "[gs2:samp] n=%lu fbp=%u type=%u tme=%u fst=%u abe=%u iip=%u bbox=(%.0f..%.0f,%.0f..%.0f) "
                         "z0=%.0f ztst=%u zmsk=%u texpsm=0x%x tbp0=%u %s\n",
                         g_gs2Draw.total, ctx.frame.fbp, static_cast<unsigned>(state.prim.type),
                         state.prim.tme ? 1u : 0u, state.prim.fst ? 1u : 0u,
                         state.prim.abe ? 1u : 0u, state.prim.iip ? 1u : 0u,
                         xmin, xmax, ymin, ymax, static_cast<double>(v0.z), ztst, zmsk,
                         state.prim.tme ? ctx.tex0.psm : 0u, state.prim.tme ? ctx.tex0.tbp0 : 0u,
                         uvinfo);
        }
        // Degenerate-triangle split by frame.fbp (all-zero XYZ population — which buffer phase?)
        if (batch.vertexCount >= 3u &&
            batch.vertices[0].x == 0.0f && batch.vertices[1].x == 0.0f && batch.vertices[2].x == 0.0f &&
            batch.vertices[0].y == 0.0f && batch.vertices[1].y == 0.0f && batch.vertices[2].y == 0.0f)
        {
            static unsigned long s_degen[2] = {0, 0};
            static unsigned long s_degenTot = 0;
            if (g_gs2DegenReset)
            {
                g_gs2DegenReset = false;
                s_degen[0] = s_degen[1] = 0;
                s_degenTot = 0;
            }
            ++s_degen[(ctx.frame.fbp != 0u) ? 1 : 0];
            if ((++s_degenTot % 8192u) == 1u)
                std::fprintf(stderr, "[gs2:degen] total=%lu fbp0=%lu fbpN=%lu (lastfbp=%u)\n",
                             s_degenTot, s_degen[0], s_degen[1], ctx.frame.fbp);
        }
        // Level-era vertex peek: dump the first perspective-textured triangles' full vertex
        // data once the draw count is past the boot/movie (are S/T/Q sane or exploded?).
        if (g_gs2Draw.total > 200000u && state.prim.tme && state.prim.fst &&
            batch.vertexCount >= 3u)
        {
            static unsigned long s_fpeek = 0;
            if (++s_fpeek <= 8u)
                std::fprintf(stderr,
                             "[gs2:fvtx] #%lu type=%u tex(psm=0x%x tw=%u th=%u tbw=%u) "
                             "v0{xy=%.1f,%.1f uv=%u,%u} v1{xy=%.1f,%.1f uv=%u,%u} v2{xy=%.1f,%.1f uv=%u,%u}\n",
                             s_fpeek, static_cast<unsigned>(state.prim.type),
                             ctx.tex0.psm, ctx.tex0.tw, ctx.tex0.th, ctx.tex0.tbw,
                             batch.vertices[0].x, batch.vertices[0].y,
                             batch.vertices[0].u >> 4, batch.vertices[0].v >> 4,
                             batch.vertices[1].x, batch.vertices[1].y,
                             batch.vertices[1].u >> 4, batch.vertices[1].v >> 4,
                             batch.vertices[2].x, batch.vertices[2].y,
                             batch.vertices[2].u >> 4, batch.vertices[2].v >> 4);
        }
        if (g_gs2Draw.total > 200000u && state.prim.tme && !state.prim.fst &&
            batch.vertexCount >= 3u)
        {
            static unsigned long s_vpeek = 0;
            if (++s_vpeek <= 10u)
                std::fprintf(stderr,
                             "[gs2:vtx] #%lu type=%u iip=%u abe=%u tex(psm=0x%x tw=%u th=%u) "
                             "v0{xy=%.1f,%.1f z=%.0f stq=%g,%g,%g rgba=%u,%u,%u,%u} "
                             "v1{xy=%.1f,%.1f stq=%g,%g,%g} v2{xy=%.1f,%.1f stq=%g,%g,%g}\n",
                             s_vpeek, static_cast<unsigned>(state.prim.type), state.prim.iip ? 1u : 0u,
                             state.prim.abe ? 1u : 0u, ctx.tex0.psm, ctx.tex0.tw, ctx.tex0.th,
                             batch.vertices[0].x, batch.vertices[0].y, static_cast<double>(batch.vertices[0].z),
                             batch.vertices[0].s, batch.vertices[0].t, batch.vertices[0].q,
                             batch.vertices[0].r, batch.vertices[0].g, batch.vertices[0].b, batch.vertices[0].a,
                             batch.vertices[1].x, batch.vertices[1].y,
                             batch.vertices[1].s, batch.vertices[1].t, batch.vertices[1].q,
                             batch.vertices[2].x, batch.vertices[2].y,
                             batch.vertices[2].s, batch.vertices[2].t, batch.vertices[2].q);
        }
        if ((g_gs2Draw.total % 8192u) == 0u)
        {
            std::fprintf(stderr,
                         "[gs2:draw] n=%lu tme=%lu psm{t4:%lu t8:%lu ct32:%lu ct16:%lu other:%lu} "
                         "lastTex(tbp0=%u psm=0x%x tw=%u th=%u tfx=%u) fbp=%u\n",
                         g_gs2Draw.total, g_gs2Draw.tme, g_gs2Draw.psmT4, g_gs2Draw.psmT8,
                         g_gs2Draw.psmCt32, g_gs2Draw.psmCt16, g_gs2Draw.psmOther,
                         g_gs2Draw.lastTbp0, g_gs2Draw.lastPsm, g_gs2Draw.lastTw, g_gs2Draw.lastTh,
                         g_gs2Draw.lastTfx, g_gs2Draw.lastFbp);
            gs2PrintXfer(); // same cadence, so the transfer state is era-resolvable
        }
    }
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    // Per-batch resolved draw state (cont.158c perf): batch.state is immutable for the
    // batch, so resolve the psm-table function pointers and block bases ONCE here instead
    // of per pixel in WritePixel/SampleTexture. No VRAM = every read 0 / write no-op, so
    // skipping the raster entirely is behavior-identical.
    if (!m_vram)
        return;
    resolveDraw(state);

    if (s_gsBatchCensus && !m_rawDraw)
        gs2BatchCensusRecord(batch, m_draw.fbp, m_draw.fbw, m_draw.zbp);

#if PS2X_HAS_GS_GPU_DEVICE
    // cont.168: record the VRAM pages this draw could write, so the full-mirror verify can
    // report stale pages the RASTERIZER cannot explain. Without it "stale = draw output"
    // is only an inference; with it, a stale page outside this set is proof of a real
    // mirror gap. Scissor-clipped primitive bbox, so it is conservative in area but never
    // misses a written page. Only armed with PS2X_GS_GPU_MIRRORVERIFY.
    if (s_gsGpuMirrorVerifyMs > 0 && batch.vertexCount != 0u && !m_rawDraw)
    {
        const int ofx = ctx.xyoffset.ofx >> 4;
        const int ofy = ctx.xyoffset.ofy >> 4;
        int minX = INT32_MAX, minY = INT32_MAX, maxX = INT32_MIN, maxY = INT32_MIN;
        for (uint8_t i = 0; i < batch.vertexCount && i < batch.vertices.size(); ++i)
        {
            const int vx = static_cast<int>(batch.vertices[i].x) - ofx;
            const int vy = static_cast<int>(batch.vertices[i].y) - ofy;
            minX = std::min(minX, vx);
            minY = std::min(minY, vy);
            maxX = std::max(maxX, vx);
            maxY = std::max(maxY, vy);
        }
        minX = std::max(minX, static_cast<int>(ctx.scissor.x0));
        minY = std::max(minY, static_cast<int>(ctx.scissor.y0));
        maxX = std::min(maxX, static_cast<int>(ctx.scissor.x1));
        maxY = std::min(maxY, static_cast<int>(ctx.scissor.y1));
        if (minX <= maxX && minY <= maxY)
        {
            markDrawPages(ctx.frame.psm, m_draw.fbp, m_draw.fbw,
                          static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                          static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY));
            if (!ctx.zbuf.zmask)
                markDrawPages(ctx.zbuf.psm, m_draw.zbp, m_draw.fbw,
                              static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                              static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY));
        }
    }
#endif

    // PS2X_GS_FONTTRACE: evaluate the sampling chain in-situ for a font glyph draw -- the raw
    // palette index read from the atlas at the glyph's UV midpoint, the final combined texel
    // (CLUT + TEXA applied), and the global CLAMP/texture-size state the packet never sets.
    if (g_gs2FontTrace && state.prim.tme &&
        (ctx.tex0.psm == GS_PSM_T4HL || ctx.tex0.psm == GS_PSM_T4HH) &&
        ctx.tex0.cbp >= 12288u && ctx.tex0.cbp <= 12290u && batch.vertexCount >= 2u)
    {
        static unsigned long s_n2 = 0;
        if (++s_n2 <= 24u)
        {
            const auto &sv0 = batch.vertices[0];
            const auto &sv1 = batch.vertices[1];
            const uint16_t mu = static_cast<uint16_t>((static_cast<uint32_t>(sv0.u) + sv1.u) / 2u);
            const uint16_t mv = static_cast<uint16_t>((static_cast<uint32_t>(sv0.v) + sv1.v) / 2u);
            const uint32_t rawIdx = m_draw.texRead(m_vram, ctx.tex0.tbp0, ctx.tex0.tbw, mu >> 4, mv >> 4);
            const uint32_t texel = SampleTexture(state, 0.0f, 0.0f, 1.0f, mu, mv);
            std::fprintf(stderr,
                         "[gs2:fontsamp] #%lu uv=(%u,%u) rawIdx=%u texel=%08x clamp=%llx texWH=%ux%u linear=%d texa{ta0=%u aem=%d ta1=%u}\n",
                         s_n2, mu >> 4, mv >> 4, rawIdx, texel,
                         static_cast<unsigned long long>(ctx.clamp),
                         state.textureWidth, state.textureHeight, state.linearFilter ? 1 : 0,
                         static_cast<unsigned>(state.texa.ta0), state.texa.aem ? 1 : 0,
                         static_cast<unsigned>(state.texa.ta1));
        }
    }

    // PS2X_GS_FONTTRACE: count framebuffer writes issued by each menu-glyph draw (proves whether
    // the raster loop writes ANY pixels for these sprites, independent of any single watch pixel).
    const bool fontCountThis = g_gs2FontTrace && state.prim.tme &&
                               (ctx.tex0.psm == GS_PSM_T4HL || ctx.tex0.psm == GS_PSM_T4HH) &&
                               ctx.tex0.cbp >= 12288u && ctx.tex0.cbp <= 12290u;
    if (fontCountThis)
    {
        ++g_gs2FontDraws;
        g_gs2LastFontFbp.store(ctx.frame.fbp, std::memory_order_relaxed);
    }
    const unsigned long fontWritesBefore = g_gs2FontPixWrites;

#if PS2X_HAS_GS_GPU_DEVICE
    // Phase 2a: hand a sampled primitive to the GPU rasterizer BEFORE the CPU draws it
    // (the mirror is seeded with the pre-draw bytes here), then compare after.
    RasterVerifyCtx rvCtx;
    if (m_rawDraw)
    {
        // bisect replay: pixels only
    }
    else if (s_gsGpuBatchPerf || s_gsGpuBatchVerify > 0)
    {
        gpuBatchAppend(batch); // batch mode owns the GPU path; the per-prim gate stands down
    }
    else if (s_gsGpuRasterVerify > 0)
    {
        static unsigned long s_rvN = 0;
        if ((++s_rvN % static_cast<unsigned long>(s_gsGpuRasterVerify)) == 0u)
            rasterVerifyBegin(batch, rvCtx);
    }
#endif

    // cont.231 (build 399): the per-primitive wall clock is for the bench / batch-perf / ablation
    // figures only. PS2X_GS_THRUPUT is in every canonical run's env and only needs the COUNT, yet it
    // paid two steady_clock::now() per primitive on the raster coordinator (~3 ms of a 125 ms level
    // frame, on the thread that is the pool's long pole).
    const bool perfCount = s_gsGpuBatchPerf || s_gsSkipCpuRaster || s_gsThruput;
    const bool perfTime = s_gsGpuBatchPerf || s_gsSkipCpuRaster;
    const auto cpuT0 = perfTime ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    if (s_gsSkipCpuRaster)
    {
        // Count the submission, do none of the work. No batch path runs in this mode, so
        // the throughput line is driven off the primitive count itself.
        if ((++g_perfCpuPrims % 500000ull) == 0ull)
            gs2PrintRasterThroughput();
#if PS2X_HAS_GS_GPU_DEVICE
        // ★ cont.329 phase 2a: this is the SINGLE-THREADED path (Submit's inline call and
        // workerLoop's empty-band-pool arm). It must translate too, or PS2X_GS_RASTER_THREADS<=1
        // would silently render nothing while the banded path rendered fine.
        if (s_gsRendererGl)
            glSubmitRun(&batch, 1u);
#endif
        return;
    }
    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        // ★ cont.207: one entry point for every scanline-based primitive, so the band fan-out
        // decision lives in exactly one place. `m_draw` was resolved above, before any helper
        // thread can run, and helpers only read it.
        if (m_bandRunActive)
            RasterBand(batch, m_bandRunY0, m_bandRunY1); // coordinator's band inside a run
        else
            RasterFanOut(batch, gs2BandMinY(batch), gs2BandMaxY(batch));
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx2 = state.context;
        int px = static_cast<int>(v.x) - (ctx2.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx2.xyoffset.ofy >> 4);
        WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }

    if (perfCount)
    {
        if (s_gsThruput && !s_gsGpuBatchPerf && --g_perfPrintCountdown == 0u)
        {
            g_perfPrintCountdown = 500000u;
            gs2PrintRasterThroughput();
        }
        // Wall time in the CPU rasterizer for THIS primitive -- the apples-to-apples
        // counterpart of the GPU's GL_TIME_ELAPSED for the same primitives.
        if (perfTime)
            g_perfCpuNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - cpuT0).count());
        ++g_perfCpuPrims;
    }

    if (fontCountThis)
    {
        static unsigned long s_nfc = 0;
        if (++s_nfc <= 48u)
            std::fprintf(stderr, "[gs2:fontwrites] #%lu seq=%lu wrote %lu px (v0 rgba=%u,%u,%u,%u)\n",
                         s_nfc, g_gs2Draw.total, g_gs2FontPixWrites - fontWritesBefore,
                         batch.vertices[0].r, batch.vertices[0].g, batch.vertices[0].b, batch.vertices[0].a);
    }

#if PS2X_HAS_GS_GPU_DEVICE
    if (rvCtx.active)
        rasterVerifyEnd(rvCtx); // the CPU result is now authoritative for this rect
#endif
}

// ★★★ cont.225 (PS2X_GS_FASTPATH, default ON, "=0" restores the function-pointer path): the
// DEVIRTUALISED rasterizer path for the format pair this game actually draws with -- CT32 frame,
// Z24 depth. cont.224 measured the swizzle/access block at ~24% of raster self time and then proved
// the cost is INDIRECT CALLS plus a 128 KB page table that misses L1, NOT the address arithmetic:
// splitting the accessors behind extra function pointers turned N indirect calls into N+1 and ran
// 1-4% SLOWER. So this removes calls instead. Calling PixelStorageTraits<C32>/<Z32> directly lets
// Address/ReadAt/WriteAt inline, and the address is then computed ONCE per buffer per pixel rather
// than once per access (the `TODO: only one address lookup for rmw` the source has carried).
// Every other format keeps the original path unchanged.
static const bool s_gsFastPath = []
{ const char *e = std::getenv("PS2X_GS_FASTPATH"); return !(e && e[0] == '0'); }();

// ★★ cont.231: the per-primitive constants of the pixel write. WritePixel used to re-derive every one
// of these from the context ON EVERY PIXEL (the alpha-test decode, the z-test method, the blend
// selectors, the fbmsk/fba/psm tests); they are loop-invariant, so the plan is built once per
// primitive by the fast triangle loop (and once per call by the generic WritePixel wrapper, which is
// what it always cost). Every field is the same expression the old code evaluated in place.
struct GSCpuBackend::PixelPlan
{
    uint64_t test = 0;
    uint8_t fpsm = 0;
    uint32_t fbmsk = 0;
    bool fbaSet = false;        // (ctx.fba & 1) && psm != CT24: alpha write forces bit 7
    bool zmask = false;
    bool abe = false;
    bool pabe = false;
    bool colwrap = false;       // cont.343b COLCLAMP.CLAMP=0: blend results wrap mod 256 (see s_gsColClamp)
    uint32_t ztestMethod = 0;
    bool needsDateRead = false; // destination alpha test needs the frame pixel
    bool atEnable = false;
    uint8_t atst = 0;
    uint8_t aref = 0;
    PixelWriteMask failMask{};  // what classifyAlphaTest returns when the test FAILS
    bool is16 = false;
    bool isCt24 = false;
    uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0, fix = 0;
    uint8_t fogR = 0, fogG = 0, fogB = 0;
    u32 fbp = 0, fbw = 1, zbp = 0;
    // cont.322d PS2X_GS_FZB: per-(fbp, zbp, fbw) row and column address tables (PCSX2's fzbr / fzbc); nullptr = not available
    const uint32_t *fzRowF = nullptr, *fzColF = nullptr, *fzRowZ = nullptr, *fzColZ = nullptr;
    bool fzPair = false; // cont.325 PS2X_GS_FZPAIR: the tables' 4-aligned groups are two pixel pairs 16 bytes apart (verified at build)
    const uint32_t *fzColMaxF = nullptr, *fzColMaxZ = nullptr; // cont.325: prefix maxima of the column tables (the per-row wrap guard)
};

// ★ cont.322d PS2X_GS_FZB (default 1): the pixel write's frame/Z addressing as PCSX2 does it. WHY: the
// four-wide loops spent 15% of their instructions in PixelStorageTraits<C32/Z32>::Address() -- two calls
// per quad (PageId's divide/multiply, the three-level page table) plus a page-table row read per lane
// (`tmp/loopsplit.py` on the Helm's bench, build 775; the census proved it is NOT texture taps). The GS
// swizzle is SEPARABLE: every block table (C32/Z32/C16/Z16 and their S variants, P8, P4) and the 32/16-bit
// column tables are row + column sums (checked numerically, cont.322c), and PageId is (y / py) * stride +
// x / px, so Address(x, y) == row[y] + col[x] with row[y] = Address(0, y) and col[x] = Address(x, 0) -
// Address(0, 0). Two tables of 2048 entries per buffer, built once per (fbp, zbp, fbw) configuration and
// kept per thread (4 configurations, LRU), SELF-CHECKED against Address() on a grid at build time -- a
// configuration that fails the check keeps the old path, so the output cannot change (the capture hashes
// are the gate). PCSX2 reference: GSScanlineEnvironment `fzbr` / `fzbc` (GSRendererSW::GetScanlineGlobalData
// `gd.fzbr = context->offset.fzb4->row; gd.fzbc = context->offset.fzb4->col;`, GSLocalMemory's GSOffset),
// applied in GSDrawScanline::DrawScanline as `fza_base = &fzbr[top]; fza_offset = &fzbc[left >> 2]`.
static const bool s_gsFzb = []
{ const char *e = std::getenv("PS2X_GS_FZB"); return !(e && e[0] == '0'); }();
// ★ cont.325 PS2X_GS_FZPAIR (default 1; `=0` = the per-lane gathers): the quad write reads and writes a
// 4-aligned group's frame and Z as TWO QWORDS each (movq at the group's first pixel, movhps 16 bytes on),
// PCSX2's ReadPixel / WritePixel shape (GSDrawScanlineCodeGenerator.all.cpp: `movq(dst, qword[base]);
// movhps(dst, qword[base + 8 * 2])`, and the same pair for the store). WHY it is exact: the CT32 / Z32
// column table's rows run 0,1,4,5 / 8,9,12,13 / ... (ColumnTable32, shared by C32 and Z32), so a group
// that starts on a multiple of 4 in x occupies words +0,+1,+4,+5 of one block -- two contiguous pairs,
// 16 bytes apart -- and the row term (fzRowF[y]) is common to the four lanes. The table builder checks
// the pattern on every 4-aligned column (t.pair) and a draw whose group could cross the 4 MB wrap
// (the lane form's `& (MEMORY_SIZE - 4)`) keeps the lane path per quad (`paired`). The cont.324d
// annotate of the hottest loop put 46 instructions (8.2% of its samples) on the address chain --
// table loads and lane offsets, four extracts + four leas + four pinsrd per buffer, four
// pextrd-to-memory stores, and a 25-instruction scalar Z read-modify-write -- for what is 2 + 2 + 2 + 6.
static const bool s_gsFzPair = []
{ const char *e = std::getenv("PS2X_GS_FZPAIR"); return !(e && e[0] == '0'); }();
// PS2X_GS_FZPAIR_LOG=1 (default off): print each table build and the first rows' guard numbers (diagnostic).
static const bool s_gsFzPairLog = []
{ const char *e = std::getenv("PS2X_GS_FZPAIR_LOG"); return e && e[0] == '1'; }();
namespace
{
    struct Gs2FzbTables
    {
        uint64_t key = ~0ull;
        uint64_t lastUse = 0ull;
        bool ok = false;
        bool pair = false; // cont.325: every 4-aligned column group is words +0,+1,+4,+5 (frame AND z)
        std::vector<uint32_t> fRow, fCol, zRow, zCol; // 2048 each
        std::vector<uint32_t> fColMax, zColMax;       // cont.325: prefix maxima of fCol / zCol (the per-row wrap guard's bound)
    };
    thread_local Gs2FzbTables t_fzb[4];
    thread_local uint64_t t_fzbClock = 0ull;
    const Gs2FzbTables *gs2FzbLookup(uint32_t fbp, uint32_t zbp, uint32_t fbw)
    {
        const uint64_t key = (static_cast<uint64_t>(fbp) << 40) ^ (static_cast<uint64_t>(zbp) << 20) ^ static_cast<uint64_t>(fbw);
        Gs2FzbTables *lru = &t_fzb[0];
        for (Gs2FzbTables &t : t_fzb)
        {
            if (t.key == key) { t.lastUse = ++t_fzbClock; return &t; }
            if (t.lastUse < lru->lastUse) lru = &t;
        }
        Gs2FzbTables &t = *lru;
        t.key = key; t.lastUse = ++t_fzbClock; t.ok = false;
        t.fRow.resize(2048); t.fCol.resize(2048); t.zRow.resize(2048); t.zCol.resize(2048);
        const uint32_t f00 = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, 0u, 0u);
        const uint32_t z00 = GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, 0u, 0u);
        for (uint32_t i = 0; i < 2048u; ++i)
        {
            t.fRow[i] = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, 0u, i);
            t.fCol[i] = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, i, 0u) - f00;
            t.zRow[i] = GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, 0u, i);
            t.zCol[i] = GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, i, 0u) - z00;
        }
        // the separability self-check: a grid that crosses columns, blocks and pages in both axes
        static const uint32_t kX[] = {0, 1, 3, 4, 7, 8, 13, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 511, 512, 640, 1023, 1024, 1500, 2047};
        static const uint32_t kY[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 200, 255, 256, 447, 448, 511, 512, 1000, 1023, 1024, 2047};
        bool ok = true;
        for (uint32_t y : kY)
            for (uint32_t x : kX)
            {
                if (GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, x, y) != t.fRow[y] + t.fCol[x]) ok = false;
                if (GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, x, y) != t.zRow[y] + t.zCol[x]) ok = false;
            }
        t.ok = ok;
        if (!ok) std::fprintf(stderr, "[gs2:fzb] fbp=%u zbp=%u fbw=%u NOT separable -- Address() path kept\n", fbp, zbp, fbw);
        // cont.325: the pair pattern behind PS2X_GS_FZPAIR, checked on every 4-aligned group of both tables.
        bool pair = ok;
        for (uint32_t x = 0; x < 2048u && pair; x += 4u)
            pair = t.fCol[x + 1] == t.fCol[x] + 1u && t.fCol[x + 2] == t.fCol[x] + 4u && t.fCol[x + 3] == t.fCol[x] + 5u &&
                   t.zCol[x + 1] == t.zCol[x] + 1u && t.zCol[x + 2] == t.zCol[x] + 4u && t.zCol[x + 3] == t.zCol[x] + 5u;
        t.pair = pair;
        // ⚠ The column entries are SIGNED offsets from Address(0, 0): the Z32 block table is not monotonic in x
        // (its block 0 is not the lowest), so zCol[x] wraps "negative" for some columns and the u32 sum
        // row + col wraps back to the true address. The prefix maximum is therefore taken as int32 (build 789
        // read a wrapped 4294966285 and rejected every row past x = 16).
        t.fColMax.resize(2048); t.zColMax.resize(2048);
        int32_t fm = INT32_MIN, zm = INT32_MIN;
        for (uint32_t i = 0; i < 2048u; ++i)
        {
            fm = std::max(fm, static_cast<int32_t>(t.fCol[i])); zm = std::max(zm, static_cast<int32_t>(t.zCol[i]));
            t.fColMax[i] = static_cast<uint32_t>(fm); t.zColMax[i] = static_cast<uint32_t>(zm);
        }
        if (s_gsFzPairLog)
            std::fprintf(stderr, "[gs2:fzpair] table fbp=%u zbp=%u fbw=%u ok=%d pair=%d fRow[0]=%u fRow[447]=%u fCol[639]=%u fColMax[2047]=%u zRow[0]=%u zRow[447]=%u zColMax[2047]=%u\n",
                         fbp, zbp, fbw, ok ? 1 : 0, pair ? 1 : 0, t.fRow[0], t.fRow[447], t.fCol[639], t.fColMax[2047], t.zRow[0], t.zRow[447], t.zColMax[2047]);
        if (ok && !pair) std::fprintf(stderr, "[gs2:fzb] fbp=%u zbp=%u fbw=%u NOT pairable -- lane gathers kept\n", fbp, zbp, fbw);
        return &t;
    }
}

void GSCpuBackend::makePixelPlan(const GSDrawState &state, PixelPlan &p) const
{
    const auto &ctx = state.context;
    p.test = ctx.test;
    p.fpsm = static_cast<uint8_t>(ctx.frame.psm);
    p.fbmsk = ctx.frame.fbmsk;
    p.fbaSet = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
    p.zmask = ctx.zbuf.zmask;
    p.abe = state.prim.abe;
    p.pabe = state.pabe;
    p.colwrap = s_gsColClamp && p.abe && (state.colclamp & 1ull) == 0ull;
    p.ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    // ★ cont.319b PS2X_GS_ZTE (default ON, "=0" restores the old reading): TEST.ZTE clear means the
    // depth test is OFF. PCSX2 GSRendererSW::GetScanlineGlobalData: `zm = ZBUF.ZMSK || TEST.ZTE == 0 ?
    // 0xffffffff : 0` (no Z write) and `ztest = TEST.ZTE && TEST.ZTST > ZTST_ALWAYS`, so ZTE=0 draws
    // with ZTST_ALWAYS and never writes Z (the GS manual marks ZTE=0 prohibited). We ignored ZTE and
    // applied ZTST alone -- so this game's full-screen ZTE=0/ZTST=NEVER sprites (two per frame, abe
    // ABCD=0122 FIX=128, cclamp 0 then 1; 166k per 150 s fight = 5.4% of raster ticks) were rejected
    // at the z test and drew NOTHING, where PCSX2 draws them. PS2X_GS_STATECENSUS keys
    // a00000a400114046 / 800000a400114046.
    if (s_gsZteAlways && ((ctx.test >> 16) & 1u) == 0u)
    {
        p.ztestMethod = 1u;
        p.zmask = 1u;
    }
    p.needsDateRead = ((ctx.test >> 14) & 0x1u) != 0u &&
                      (p.fpsm == GS_PSM_CT32 || p.fpsm == GS_PSM_CT16 || p.fpsm == GS_PSM_CT16S);
    p.atEnable = (ctx.test & 0x1u) != 0u;
    p.atst = static_cast<uint8_t>((ctx.test >> 1) & 0x7u);
    p.aref = static_cast<uint8_t>((ctx.test >> 4) & 0xFFu);
    switch (static_cast<uint8_t>((ctx.test >> 12) & 0x3u)) // classifyAlphaTest's FAIL branch
    {
    case 1: p.failMask = {true, true, false}; break;                                       // FB_ONLY
    case 2: p.failMask = {false, false, true}; break;                                      // ZB_ONLY
    case 3: p.failMask = (p.fpsm == GS_PSM_CT32) ? PixelWriteMask{true, false, false}
                                                  : PixelWriteMask{true, true, false}; break; // RGB_ONLY
    case 0:
    default: p.failMask = {false, false, false}; break;                                    // KEEP
    }
    p.is16 = bitsPerPixel(p.fpsm) == 16;
    p.isCt24 = p.fpsm == GS_PSM_CT24;
    const uint64_t alphaReg = ctx.alpha;
    p.asel = static_cast<uint8_t>(alphaReg & 3);
    p.bsel = static_cast<uint8_t>((alphaReg >> 2) & 3);
    p.csel = static_cast<uint8_t>((alphaReg >> 4) & 3);
    p.dsel = static_cast<uint8_t>((alphaReg >> 6) & 3);
    p.fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);
    p.fogR = state.fogR;
    p.fogG = state.fogG;
    p.fogB = state.fogB;
    p.fbp = m_draw.fbp;
    p.fbw = m_draw.fbw;
    p.zbp = m_draw.zbp;
    p.fzRowF = p.fzColF = p.fzRowZ = p.fzColZ = nullptr;
    p.fzPair = false;
    if (s_gsFzb)
    {
        const Gs2FzbTables *t = gs2FzbLookup(p.fbp, p.zbp, std::max<uint32_t>(p.fbw, 1u));
        if (t->ok) { p.fzRowF = t->fRow.data(); p.fzColF = t->fCol.data(); p.fzRowZ = t->zRow.data(); p.fzColZ = t->zCol.data(); p.fzPair = t->pair; p.fzColMaxF = t->fColMax.data(); p.fzColMaxZ = t->zColMax.data(); }
    }
}

// The pixel write, in exactly the old WritePixel's order of decisions (scissor, fog, alpha test ->
// write mask, frame read, destination alpha test, z test, blend, fba, fbmsk, preserved alpha, write,
// z write), with the loop-invariant state read from the plan and the CT32/Z24 addressing and the fog
// resolved at compile time. kFast == (s_gsFastPath && m_draw.fastCt32Z24) exactly as before.
// ★★★ cont.262 PS2X_GS_PIPESPEC (default ON; "=0" restores the pre-cont.262 all-runtime pixel
// write) -- the pipeline-class selector, this backend's answer to PCSX2's GSScanlineSelector.
// cont.261 diffed our rasterizer against PCSX2's: it JITs one scanline per draw state so the
// emitted code branches on none of ~30 state dimensions, while we hoisted the DECODE into
// PixelPlan and still branched on its VALUES per pixel. PS2X_GS_STATECENSUS then showed this game
// uses only 46 distinct draw states with the TOP 3 = 89.7% of 48.8M draws, all sharing
// ABCD=0101 + ZTST=GEQUAL -- so one specialised class captures what a JIT would, without one.
static const bool s_gsPipeSpec = []
{ const char *e = std::getenv("PS2X_GS_PIPESPEC"); return !(e && e[0] == '0'); }();

// ★★★ cont.267 PS2X_GS_SIMD4 (default OFF) -- TIER 2: the row walk four pixels per iteration.
// cont.265 retracted "the rasterizer is memory-bound": `perf` says IPC 3.13 and a 0.38% L1 miss
// rate over 90.76 G instructions, i.e. INSTRUCTION-bound at ~928 instructions per shaded pixel
// (a competent software rasterizer does 20-50). cont.266 disassembled the hot symbol: 6.6% SIMD,
// 93% scalar byte manipulation, and cont.266b proved the byte work has no hot spot to vectorise
// (movzbl 15-56 in every one of 12 equal regions => ~2.6% available at any single site).
// The only lever that DIVIDES an instruction count is width. Default OFF until it is both
// bit-exact (bench hash aa9cd72b600021ce) and measured faster (perf stat -e instructions).
//
// ⚠ HISTORY, because the verdict REVERSED and both halves matter (cont.267 -> cont.270):
// As built in cont.267 -- a wide skeleton feeding a SCALAR sampler and pixel write -- this was
// bit-exact and SLOWER: 61.602 G (4-aligned) / 61.272 G (unaligned) instructions vs 60.756 G
// scalar, and worse still (+3.8%) with the tail ablated, i.e. it lost in the very component it
// widened. PS2X_GS_TRICENSUS said why: meanWalkedW=4.31 px (7.47 over non-empty rows), 42.25% of
// rows empty, only 60.12% of walked pixels covered -- 27 pixels per triangle, so a row is under
// two four-lane groups and the store-wide/reload-scalar glue cost more than the width saved.
// The alignment A/B pinned head-lane waste at only 0.54 of those 1.39 points: THE GLUE WAS THE BULK.
//
// cont.269 then diffed us against PCSX2 and found the missing half: its JITted scanline never
// leaves the vector registers, so it pays no glue. cont.270 added PS2X_GS_QUADWRITE (the pixel
// write four at a time, colour resident in 16-bit lanes) and the verdict flipped --
// 56.573 G instructions, -3.43% against the scalar path, bit-exact at 1, 3 and 8 threads.
// ★★★ THE LESSON: partial widening cannot pay, because the glue at the seam costs more than the
// width saves. Widen the WHOLE chain or none of it. The remaining seam is the sampler
// (24.7% setup + 17.9% bilinear of the hot symbol), which is the next increment.
static const bool s_gsSimd4 = []
{ const char *e = std::getenv("PS2X_GS_SIMD4"); return !(e && e[0] == '0'); }();
// PS2X_GS_SIMD4_ALIGN (default 1): start each row's group loop at `spanX0 & ~3` so a group's four
// pixels share one swizzled Address() (what step 2 needs), at the cost of up to 3 wasted head lanes.
// `=0` starts at spanX0 instead -- no head waste, no alignment. With a mean walked row of only
// 4.31 px (cont.267 tricensus) that trade is the whole question, so it is a runtime knob.
static const bool s_gsSimd4Align = []
{ const char *e = std::getenv("PS2X_GS_SIMD4_ALIGN"); return !(e && e[0] == '0'); }();
// ★★★ cont.317 PS2X_GS_SEAMKEY20 (default ON): key 20 (textured, non-FST, bilinear, FLAT, no fog)
// joins the selector seam. The cont.261 census counted DRAWS (keys 22/23 = 89.7%); the cont.317
// TIME census (PS2X_GS_KEYCENSUS, live Helm's Deep fight, 234 s) put key 20 at 71% of raster
// ticks, 23.5% of it kPipeStdAlpha + PSMT4 -- exactly the shape the four-wide loop already
// serves (the SIMD4 loop carries the flat-colour branch: v2's colour, as the scalar loop does).
// Two instantiations (Simd4 + the Fast StdAlpha fallback), no cross product. `=0` restores the
// generic table for key 20 -- the live A/B. Bench hash is the bit-exactness gate as always.
static const bool s_gsSeamKey20 = []
{ const char *e = std::getenv("PS2X_GS_SEAMKEY20"); return !(e && e[0] == '0'); }();
// ★★★ cont.317 PS2X_GS_SEAMADD (default ON): the SECOND pipe class. After SEAMKEY20 + FASTP8 the
// time census still put 28% of raster ticks on key 20 draws with abe ABCD=0201 (Cs,0,As,Cd), fix
// unused, ztst=GEQUAL, zmask=1, ate=0, fbmsk=0, DATE off, CT32 -- the terrain -- on the generic
// scalar loop. kPipeAddAlpha resolves that blend at compile time in writePixelT and writePixelQuad
// (Cd + (Cs*As>>7): a 16-bit unsigned product, exact with mullo + logical >>7), and key 20 gets
// its Simd4 + Fast instantiations for it. `=0` sends those draws back to the generic table.
static const bool s_gsSeamAdd = []
{ const char *e = std::getenv("PS2X_GS_SEAMADD"); return !(e && e[0] == '0'); }();
// ★ cont.321 PS2X_GS_SEAMKEY2 (default ON): key 2 (UNTEXTURED, gouraud, no fog) on the StdAlpha class joins
// the four-wide loop -- 2.6% of the live fight's raster ticks (build 764 key census) on the generic per-pixel
// path. The Simd4 loop already carries the kTme=false arms (colour lerp in lanes, no sampler); two
// instantiations (Simd4 + the Fast StdAlpha fallback). `=0` restores the generic table for key 2.
static const bool s_gsSeamKey2 = []
{ const char *e = std::getenv("PS2X_GS_SEAMKEY2"); return !(e && e[0] == '0'); }();
// ★ cont.321b PS2X_GS_SEAMADDFIX (default ON): key 0 (untextured, flat, no fog) on the ADD-FIX plan (abe ABCD=0221
// FIX=128 => Cd + Cs, ZTST=GREATER, zmask -- 3.9% of the live fight's raster ticks, 12 M tiny draws) takes the four-wide
// loop with kZGreater; the scalar tail and every fallback stay on the generic arm. `=0` restores the generic table.
static const bool s_gsSeamAddFix = []
{ const char *e = std::getenv("PS2X_GS_SEAMADDFIX"); return !(e && e[0] == '0'); }();

// The selector. Returns the class whose specialised code is BIT-IDENTICAL to the generic path for
// this plan; anything not exactly matched falls back to kPipeGeneric. Keep this predicate and the
// `if constexpr` blocks in writePixelT in lockstep -- a mismatch here is a silent rendering bug,
// which is why the bench hash (aa9cd72b600021ce) is the gate on every change to either.
// cont.319b: the sprite plans, from PS2X_GS_KEYCENSUS on the recorded fight (all ZTST=ALWAYS):
//   spr 24 (FST nearest, CT32 texture) abe ABCD=2101 -> kPipeDarken (11.5% of raster ticks)
//   spr 22 (bilinear, T4) abe ABCD=0101 + alpha test          -> kPipeStdAlpha (7.6%)
//   spr 2 / spr 0 untextured, abe off (the clears, the fills) -> kPipeOpaque (5.0% + 3.8%)
int GSCpuBackend::spritePipeClassFor(const PixelPlan &p)
{
    if (p.ztestMethod != 1u || p.colwrap) // cont.343b: colclamp=0 sprites take the generic arm
        return -1;
    if (!p.abe)
        return kPipeOpaque;
    if (p.pabe)
        return -1;
    if (p.asel == 0u && p.bsel == 1u && p.csel == 0u && p.dsel == 1u)
        return kPipeStdAlpha;
    if (p.asel == 2u && p.bsel == 1u && p.csel == 0u && p.dsel == 1u)
        return kPipeDarken;
    // cont.321: ABCD=0122 (A=Cs, B=Cd, C=FIX, D=0) => ((Cs - Cd) * FIX >> 7): the ZTE=0 full-screen pair.
    if (p.asel == 0u && p.bsel == 1u && p.csel == 2u && p.dsel == 2u)
        return kPipeSubFix;
    return -1;
}

// cont.321b: the tri-0 plan of the live fight -- abe ABCD=0221 (A=Cs, B=0, C=FIX, D=Cd), no PABE, ZTST=GREATER.
// Kept OUT of pipeClassFor: the key census packs that value into two bits, and every other class is GEQUAL.
bool GSCpuBackend::isAddFixPlan(const PixelPlan &p)
{
    return p.abe && !p.pabe && !p.colwrap && p.asel == 0u && p.bsel == 2u && p.csel == 2u && p.dsel == 1u && p.ztestMethod == 3u;
}

int GSCpuBackend::pipeClassFor(const PixelPlan &p)
{
    if (p.colwrap) // cont.343b: only the generic arm wraps
        return GSCpuBackend::kPipeGeneric;
    // kPipeStdAlpha: standard source-alpha blending over a GEQUAL depth test -- the shape the top
    // three draw states share. ABCD=0101 is (A=Cs, B=Cd, C=As, D=Cd) => Cd + (Cs-Cd)*As>>7.
    if (p.abe && !p.pabe &&
        p.asel == 0u && p.bsel == 1u && p.csel == 0u && p.dsel == 1u &&
        p.ztestMethod == 2u)
    {
        return GSCpuBackend::kPipeStdAlpha;
    }
    // kPipeAddAlpha (cont.317): ABCD=0201 is (A=Cs, B=0, C=As, D=Cd) => Cd + (Cs*As>>7) over the same
    // GEQUAL depth test -- the Helm's Deep terrain's plan (key 20, 28% of raster ticks).
    if (p.abe && !p.pabe &&
        p.asel == 0u && p.bsel == 2u && p.csel == 0u && p.dsel == 1u &&
        p.ztestMethod == 2u)
    {
        return GSCpuBackend::kPipeAddAlpha;
    }
    return GSCpuBackend::kPipeGeneric;
}

// cont.322d: the scalar pixel path's frame/Z addresses through the PS2X_GS_FZB row+column tables when the plan
// carries them (see gs2FzbLookup), else Address() -- the same substitution writePixelQuad makes.
template <class Plan> __attribute__((always_inline)) static inline u32 gs2FAddr(const Plan &p, u32 fbp, u32 fbw, int x, int y)
{
    return p.fzRowF != nullptr ? p.fzRowF[y] + p.fzColF[x]
                               : GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, static_cast<u32>(x), static_cast<u32>(y));
}
template <class Plan> __attribute__((always_inline)) static inline u32 gs2ZAddr(const Plan &p, u32 zbp, u32 fbw, int x, int y)
{
    return p.fzRowZ != nullptr ? p.fzRowZ[y] + p.fzColZ[x]
                               : GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, static_cast<u32>(x), static_cast<u32>(y));
}

template <bool kFast, bool kFge, int kPipe>
__attribute__((always_inline)) inline void GSCpuBackend::writePixelT(const GSDrawState &state, const PixelPlan &p, int x, int y, int z,
                               uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    if (s_gsRasterAbl & 2u) // ABLATION: size the frame/Z read-modify-write (garbage output)
        return;
    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;

    if constexpr (kFge)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, p.fogR);
        g = applyFog(g, p.fogG);
        b = applyFog(b, p.fogB);
    }

    const u32 fbp = p.fbp;
    const u32 fbw = p.fbw;
    const u32 fpsm = p.fpsm;
    const u32 zbp = p.zbp;

    PixelWriteMask writeMask{};
    {
        bool pass = true;
        if (p.atEnable)
        {
            switch (p.atst)
            {
            case 0: pass = false; break;
            case 1: pass = true; break;
            case 2: pass = a < p.aref; break;
            case 3: pass = a <= p.aref; break;
            case 4: pass = a == p.aref; break;
            case 5: pass = a >= p.aref; break;
            case 6: pass = a > p.aref; break;
            case 7: pass = a != p.aref; break;
            default: pass = true; break;
            }
        }
        if (!pass)
            writeMask = p.failMask;
    }
    if (!writeMask.writesAnything())
    {
        return;
    }

    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool frmw = p.needsDateRead || (writeMask.writesFramebuffer() && ((p.fbmsk != 0) || p.abe || preserveDestinationAlpha));

    u32 fbPixelAddr = 0;
    bool fbAddrValid = false;
    u32 zPixelAddr = 0;
    bool zAddrValid = false;

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        if constexpr (kFast)
        {
            fbPixelAddr = gs2FAddr(p, fbp, fbw, x, y);
            fbAddrValid = true;
            rawFramebufferPixel = GSMem::PixelStorageTraits<GSMem::C32>::ReadAt(m_vram, fbPixelAddr);
        }
        else
            rawFramebufferPixel = m_draw.frameRead(m_vram, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if constexpr (!kFast)
        {
            if (p.is16)
            {
                fbrgba = Rgba5551ToRgba8888(fbrgba);
            }
            else if (p.isCt24)
            {
                fbrgba |= 0x80000000u;
            }
        }
    }

    if (!passesDestinationAlphaTest(p.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    bool zpass = false;
    uint32_t storedZ = 0u;
    if constexpr (kPipe != kPipeGeneric)
    {
        // ZTST=GEQUAL, guaranteed by pipeClassFor for every specialised class -- identical to
        // `case 2` below, no switch.
        if constexpr (kFast)
        {
            zPixelAddr = gs2ZAddr(p, zbp, fbw, x, y);
            zAddrValid = true;
            storedZ = GSMem::PixelStorageTraits<GSMem::Z24>::ReadAt(m_vram, zPixelAddr);
        }
        else
            storedZ = m_draw.zRead(m_vram, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) >= storedZ;
    }
    else
    switch (p.ztestMethod)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        if constexpr (kFast)
        {
            zPixelAddr = gs2ZAddr(p, zbp, fbw, x, y);
            zAddrValid = true;
            storedZ = GSMem::PixelStorageTraits<GSMem::Z24>::ReadAt(m_vram, zPixelAddr);
        }
        else
            storedZ = m_draw.zRead(m_vram, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) >= storedZ;
        break;
    case 3:
        if constexpr (kFast)
        {
            zPixelAddr = gs2ZAddr(p, zbp, fbw, x, y);
            zAddrValid = true;
            storedZ = GSMem::PixelStorageTraits<GSMem::Z24>::ReadAt(m_vram, zPixelAddr);
        }
        else
            storedZ = m_draw.zRead(m_vram, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) > storedZ;
        break;
    }

    if (!zpass)
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if constexpr (kPipe == kPipeStdAlpha)
        {
            // ABE with ABCD=0101 and no PABE, guaranteed by pipeClassFor. The generic arm below
            // evaluates pickRGB() NINE times per pixel on runtime selectors; this is the same
            // arithmetic with the selectors resolved: Cd + ((Cs - Cd) * As >> 7), same int
            // promotions, same >>7 on a possibly-negative int, same clampU8 -- bit-identical.
            const int dr = static_cast<int>(fbrgba & 0xFF);
            const int dg = static_cast<int>((fbrgba >> 8) & 0xFF);
            const int db = static_cast<int>((fbrgba >> 16) & 0xFF);
            const int cAlpha = static_cast<int>(a);
            r = clampU8((((static_cast<int>(r) - dr) * cAlpha >> 7)) + dr);
            g = clampU8((((static_cast<int>(g) - dg) * cAlpha >> 7)) + dg);
            b = clampU8((((static_cast<int>(b) - db) * cAlpha >> 7)) + db);
        }
        else if constexpr (kPipe == kPipeAddAlpha)
        {
            // cont.317: ABCD=0201, no PABE, guaranteed by pipeClassFor. The generic arm with the
            // selectors resolved: pickRGB(0)=Cs, pickRGB(2)=0, pickRGB(1)=Cd, cAlpha=As --
            // ((Cs - 0) * As >> 7) + Cd, same int promotions, same clampU8 -- bit-identical.
            const int dr = static_cast<int>(fbrgba & 0xFF);
            const int dg = static_cast<int>((fbrgba >> 8) & 0xFF);
            const int db = static_cast<int>((fbrgba >> 16) & 0xFF);
            const int cAlpha = static_cast<int>(a);
            r = clampU8(((static_cast<int>(r) * cAlpha >> 7)) + dr);
            g = clampU8(((static_cast<int>(g) * cAlpha >> 7)) + dg);
            b = clampU8(((static_cast<int>(b) * cAlpha >> 7)) + db);
        }
        else if (p.abe)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            if (!(p.pabe && (a & 0x80u) == 0u))
            {
                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (p.csel == 0) ? a : (p.csel == 1) ? da
                                                               : p.fix;

                // cont.343b: COLCLAMP=0 keeps the low 8 bits of the (signed) result -- the GS wrap.
                auto finishRGB = [&](int v) -> uint8_t
                { return p.colwrap ? static_cast<uint8_t>(v & 0xFF) : clampU8(v); };
                r = finishRGB(((pickRGB(p.asel, r, dr) - pickRGB(p.bsel, r, dr)) * cAlpha >> 7) + pickRGB(p.dsel, r, dr));
                g = finishRGB(((pickRGB(p.asel, g, dg) - pickRGB(p.bsel, g, dg)) * cAlpha >> 7) + pickRGB(p.dsel, g, dg));
                b = finishRGB(((pickRGB(p.asel, b, db) - pickRGB(p.bsel, b, db)) * cAlpha >> 7) + pickRGB(p.dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && p.fbaSet)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (p.fbmsk != 0)
        {
            pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        if constexpr (!kFast)
        {
            if (p.is16)
            {
                pixel = Rgba8888ToRgba5551(pixel);
            }
        }

        if constexpr (kFast)
        {
            if (!fbAddrValid)
                fbPixelAddr = gs2FAddr(p, fbp, fbw, x, y);
            GSMem::PixelStorageTraits<GSMem::C32>::WriteAt(m_vram, fbPixelAddr, pixel);
        }
        else
            m_draw.frameWrite(m_vram, fbp, fbw, x, y, pixel);
        if (g_gs2FontTrace)
            ++g_gs2FontPixWrites;

        // ★★★★★ cont.332h PS2X_GS_PIXWATCH: every draw that actually WRITES the watched pixel,
        // with the whole TEX0 -- the question "which draw paints this" that bbox censuses and
        // snapshot diffs could not answer. Flip-gated by PS2X_GS_PROBEFLIP, capped by _N.
        if (s_pixWatch && int(x) == s_pixWatchX && int(y) == s_pixWatchY &&
            g_pixWatchShown < s_pixWatchN && probeFlipOk())
        {
            ++g_pixWatchShown;
            const GSTex0Reg &t0w = ctx.tex0;
            std::fprintf(stderr,
                         "[gs2:pixwatch] #%lu flip=%llu (%d,%d) fbp=0x%x wrote=%08x dst=%08x "
                         "z=%u storedZ=%u "   // cont.332m
                         "src=(%u,%u,%u,%u) prim=%u tme=%u iip=%u fst=%u | "
                         "TEX0{tbp=%u tbw=%u psm=0x%02x tw=%u th=%u cbp=%u cpsm=0x%x tfx=%u tcc=%u} "
                         "abe=%u ABCD=%u%u%u%u FIX=%u fbmsk=%08x TEST{ate=%u atst=%u aref=%u zte=%u ztst=%u} "
                         "CLAMP{wms=%u wmt=%u minu=%u maxu=%u minv=%u maxv=%u}\n",
                         g_pixWatchShown,
                         (unsigned long long)g_perfFlips.load(std::memory_order_relaxed),
                         int(x), int(y), unsigned(fbp), pixel, fbrgba,
                         unsigned(static_cast<uint32_t>(z)), storedZ,   // cont.332m
                         srcR, srcG, srcB, a,
                         unsigned(state.prim.type), state.prim.tme ? 1u : 0u,
                         state.prim.iip ? 1u : 0u, state.prim.fst ? 1u : 0u,
                         unsigned(t0w.tbp0), unsigned(t0w.tbw), unsigned(t0w.psm),
                         unsigned(t0w.tw), unsigned(t0w.th), unsigned(t0w.cbp), unsigned(t0w.cpsm),
                         unsigned(t0w.tfx), unsigned(t0w.tcc),
                         state.prim.abe ? 1u : 0u,
                         unsigned(ctx.alpha & 3u), unsigned((ctx.alpha >> 2) & 3u),
                         unsigned((ctx.alpha >> 4) & 3u), unsigned((ctx.alpha >> 6) & 3u),
                         unsigned((ctx.alpha >> 32) & 0xFFu),
                         unsigned(ctx.frame.fbmsk),
                         unsigned(ctx.test & 1u), unsigned((ctx.test >> 1) & 7u),
                         unsigned((ctx.test >> 4) & 0xFFu),
                         unsigned((ctx.test >> 16) & 1u), unsigned((ctx.test >> 17) & 3u),
                         unsigned(m_draw.texWrapU), unsigned(m_draw.texWrapV),   // cont.332m
                         unsigned(m_draw.texMinU), unsigned(m_draw.texMaxU),
                         unsigned(m_draw.texMinV), unsigned(m_draw.texMaxV));
        }

        if (g_gs2FontTrace && fbp == 0u && x == 128 && y == 428)
        {
            static unsigned long s_nw = 0;
            if (++s_nw <= 64u)
            {
                const uint32_t rb = m_draw.frameRead(m_vram, fbp, fbw, x, y);
                std::fprintf(stderr,
                             "[gs2:pixwatch] #%lu seq=%lu wrote=%08x readback=%08x tme=%u prim=%u tex0psm=0x%x abe=%u fbmsk=%x srcRGBA=(%u,%u,%u,%u)\n",
                             s_nw, g_gs2Draw.total, pixel, rb, state.prim.tme ? 1u : 0u,
                             static_cast<unsigned>(state.prim.type), ctx.tex0.psm,
                             state.prim.abe ? 1u : 0u, ctx.frame.fbmsk, srcR, srcG, srcB, a);
            }
        }
    }

    if (writeMask.writeDepth && !p.zmask)
    {
        if constexpr (kFast)
        {
            if (!zAddrValid)
                zPixelAddr = gs2ZAddr(p, zbp, fbw, x, y);
            GSMem::PixelStorageTraits<GSMem::Z24>::WriteAt(m_vram, zPixelAddr, z);
        }
        else
            m_draw.zWrite(m_vram, zbp, fbw, x, y, z);
    }
}

// The generic entry (sprites, lines, the generic triangle loop): build the plan and dispatch. This
// costs exactly what the old per-pixel derivations cost; the fast triangle loop builds it once.
void GSCpuBackend::writePixelDispatch(const GSDrawState &state, const PixelPlan &p, bool fast, int x, int y, int z,
                                      uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    if (fast)
    {
        if (state.prim.fge) writePixelT<true, true>(state, p, x, y, z, r, g, b, a, fog);
        else                writePixelT<true, false>(state, p, x, y, z, r, g, b, a, fog);
    }
    else
    {
        if (state.prim.fge) writePixelT<false, true>(state, p, x, y, z, r, g, b, a, fog);
        else                writePixelT<false, false>(state, p, x, y, z, r, g, b, a, fog);
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    PixelPlan p;
    makePixelPlan(state, p);
    writePixelDispatch(state, p, s_gsFastPath && m_draw.fastCt32Z24, x, y, z, r, g, b, a, fog);
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint32_t cbp,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    // NOTE (cont.155 audit): this image-space CLUT read looks unlike PCSX2's (linear-from-block
    // CSM1) but is EQUIVALENT for palettes uploaded as images: upload and read go through the
    // same addrPSM mapping (the physical layout cancels) and resolveClutIndex's bit3/4 swap
    // reproduces the hardware's net permutation. Verified identical to the pre-refactor
    // rasterizer (ee14958 ps2_gs_rasterizer.cpp lookupCLUT), which rendered this game's
    // HUD/menu/terrain palettes correctly — do NOT "fix" this toward PCSX2's representation.
    if (s_gsTexCensus)
    {
        ++g_texCensus.clutCalls;
        ++g_texCensus.clutByCpsm[cpsm & 0x3Fu];
        if (csm != 0u) ++g_texCensus.clutCsm2;
        const uint64_t key = gs2ClutKey(cbp, cpsm, csm, csa, sourcePsm);
        if (key != g_texCensus.lastClutKey)
        {
            ++g_texCensus.clutSwitch;
            g_texCensus.lastClutKey = key;
        }
        ++g_texCensus.clutKeys[key];
    }
    const uint32_t clutIndex = resolveClutIndex(index, cpsm, csm, csa, sourcePsm);
    // TEXCLUT (cbw/cou/cov) applies to CSM2 ONLY -- for CSM1 the CLUT sits in its fixed layout at
    // CBP and TEXCLUT must be ignored (PCSX2 GS/GSLocalMemory.cpp: TEXCLUT is only consulted in
    // the *CSM2* ReadCLUT/WriteCLUT variants; GSTextureCacheSW ignores it for CSM1). Applying a
    // stale TEXCLUT (e.g. left over from earlier CSM2 use) to a CSM1 lookup offsets every entry
    // read off the palette -- LOTR EUR menu-font CLUTs sample as zeros -> ATST kills all glyphs.
    const bool csm2 = (csm != 0u);
    const uint32_t clutWidth = (csm2 && state.texclut.cbw != 0u) ? static_cast<uint32_t>(state.texclut.cbw) : 1u;
    const uint32_t clutX = (csm2 ? static_cast<uint32_t>(state.texclut.cou) : 0u) + (clutIndex & 0x0Fu);
    const uint32_t clutY = (csm2 ? static_cast<uint32_t>(state.texclut.cov) : 0u) + (clutIndex >> 4);

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        uint32_t entry = GSMem::ReadCT32(m_vram, cbp, clutWidth, clutX, clutY);
        // CLUT shadow (cont.165): a CSM1 entry whose RGB lanes were zeroed by a colliding
        // render target (FBMSK=ff000000 scene draws over the CLUT blocks) while its alpha
        // matches the uploaded payload is served from the upload -- the value real hardware
        // would sample had the game's VRAM allocator not placed the CLUT inside the target.
        if (s_gsClutShadow && csm == 0u && (entry & 0x00FFFFFFu) == 0u && !m_clutShadow.empty())
        {
            auto it = m_clutShadow.find(cbp);
            if (it != m_clutShadow.end() && clutX < 8u && clutY < 2u)
            {
                const uint32_t up = it->second[clutY * 8u + clutX];
                if ((up & 0xFF000000u) == (entry & 0xFF000000u) && (up & 0x00FFFFFFu) != 0u)
                {
                    entry = up;
                    if (s_gsTexCensus) ++g_texCensus.clutShadowHit;
                }
            }
        }
        return applyTexa(state.texa, cpsm, entry);
    }
    case GS_PSM_CT24:
        return applyTexa(state.texa, cpsm, GSMem::ReadCT24(m_vram, cbp, clutWidth, clutX, clutY));
    case GS_PSM_CT16:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(GSMem::ReadCT16(m_vram, cbp, clutWidth, clutX, clutY)));
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(GSMem::ReadCT16S(m_vram, cbp, clutWidth, clutX, clutY)));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

// ★★ cont.228: every diagnostic for the PSMT4 fast path, kept OUT of the sampling lambda so the
// production path is one branch and two loads. PS2X_GS_CLUTVERIFY compares the inlined read plus
// decoded-palette hit against the original composition (m_draw.texRead -> LookupCLUT), which is
// what keeps the oracle on the REAL path -- without this the verifier would silently stop covering
// 83% of all texel fetches. PS2X_GS_CLUTMUTATE proves it can fail: 2 rotates the palette index,
// 3 corrupts the READ's coordinate (so the address computation is covered too, not just the hit).
__attribute__((noinline))
uint32_t GSCpuBackend::sampleFastP4Diag(const GSDrawState &state, uint32_t tbp, uint32_t tbw,
                                        int sampleU, int sampleV)
{
    const auto &tex = state.context.tex0;
    const u32 mu = static_cast<u32>(s_gsClutMutate == 3 ? (sampleU ^ 1) : sampleU);
    uint32_t idx = GSMem::PixelStorageTraits<GSMem::P4>::Read(
                       GSMem::PageTableP4, m_vram, tbp, tbw, mu,
                       static_cast<u32>(sampleV)) & 15u;
    if (s_gsClutMutate == 2)
        idx = (idx + 1u) & 15u;
    const uint32_t fast = m_draw.clut[idx];
    if (s_gsClutVerify)
    {
        const u32 ref = m_draw.texRead(m_vram, tbp, tbw, sampleU, sampleV);
        const uint32_t live = LookupCLUT(state, static_cast<u8>(ref), tex.cbp, tex.cpsm,
                                         tex.csm, tex.csa, tex.psm);
        ++g_clutVerify.checks;
        if (fast != live && g_clutVerify.mismatches++ < 8ull)
            std::fprintf(stderr,
                         "[gs2:clutverify] FASTP4 MISMATCH u=%d v=%d idx=%u ref=%u"
                         " fast=%08x live=%08x tbp=%u tbw=%u\n",
                         sampleU, sampleV, idx, ref, fast, live, tbp, tbw);
    }
    return fast;
}

// cont.317: the PSMT8 twin of the diagnostic above -- same oracle (PS2X_GS_CLUTVERIFY) and the
// same mutations (2 rotates the palette index, 3 corrupts the read coordinate).
__attribute__((noinline))
uint32_t GSCpuBackend::sampleFastP8Diag(const GSDrawState &state, uint32_t tbp, uint32_t tbw,
                                        int sampleU, int sampleV)
{
    const auto &tex = state.context.tex0;
    const u32 mu = static_cast<u32>(s_gsClutMutate == 3 ? (sampleU ^ 1) : sampleU);
    uint32_t idx = GSMem::PixelStorageTraits<GSMem::P8>::Read(
                       GSMem::PageTableP8, m_vram, tbp, tbw, mu,
                       static_cast<u32>(sampleV)) & 255u;
    if (s_gsClutMutate == 2)
        idx = (idx + 1u) & 255u;
    const uint32_t fast = m_draw.clut[idx];
    if (s_gsClutVerify)
    {
        const u32 ref = m_draw.texRead(m_vram, tbp, tbw, sampleU, sampleV);
        const uint32_t live = LookupCLUT(state, static_cast<u8>(ref), tex.cbp, tex.cpsm,
                                         tex.csm, tex.csa, tex.psm);
        ++g_clutVerify.checks;
        if (fast != live && g_clutVerify.mismatches++ < 8ull)
            std::fprintf(stderr,
                         "[gs2:clutverify] FASTP8 MISMATCH u=%d v=%d idx=%u ref=%u"
                         " fast=%08x live=%08x tbp=%u tbw=%u\n",
                         sampleU, sampleV, idx, ref, fast, live, tbp, tbw);
    }
    return fast;
}

template <bool kFst, bool kLinear>
__attribute__((always_inline)) inline uint32_t GSCpuBackend::sampleTextureT(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v)
{
    if (s_gsRasterAbl & 1u) // ABLATION: size the whole texel-fetch path (garbage output)
        return 0x80808080u;
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    // ★★ cont.228: the whole prologue below used to run on EVERY texel fetch. It depends only on
    // TEX0/CLAMP and the mip, all constant for a primitive, so it is resolved once per draw --
    // keyed on the mip so the call sites need no change (DrawTriangle's triMip is not known until
    // after its winding setup).
    // ★ With the flag OFF this resolves on EVERY call, which is exactly the pre-cont.228
    // behaviour -- so the kill switch reverts BOTH halves of the change (the hoist and the
    // devirtualised read) without keeping a second copy of the prologue that could drift.
    // It is very slightly pessimistic: the OFF path stores into m_draw where the original
    // computed into locals, so a measured gain is an upper bound by those few stores.
    // cont.231: the caller resolved the mip (SampleTexture below, or the fast loop once per primitive).

    if (s_gsTexCache)
        m_draw.texTaps += state.linearFilter ? 4u : 1u; // cont.320: the predictor's input
    if (s_gsTexCensus)
    {
        ++g_texCensus.calls;
        if (state.linearFilter) ++g_texCensus.linear;
        if (m_draw.texMip > 0u) ++g_texCensus.mipped;
        if (state.prim.fst) ++g_texCensus.fst;
        ++g_texCensus.wrap[m_draw.texWrapU & 3u][m_draw.texWrapV & 3u];
    }

    const uint32_t mipTbp = m_draw.texTbp;
    const uint32_t mipTbw = m_draw.texTbw;
    const int texW = m_draw.texW;
    const int texH = m_draw.texH;
    const uint8_t wrapU = m_draw.texWrapU;
    const uint8_t wrapV = m_draw.texWrapV;
    const uint16_t minU = m_draw.texMinU;
    const uint16_t maxU = m_draw.texMaxU;
    const uint16_t minV = m_draw.texMinV;
    const uint16_t maxV = m_draw.texMaxV;
    const bool fastP4 = m_draw.texFastP4;
    const bool fastP8 = m_draw.texFastP8; // cont.317
    const uint32_t *const clut = m_draw.clut;
    const uint8_t *const texLin = m_draw.texLin; // cont.320: the decoded-index cache, or null
    const uint32_t texLinW = m_draw.texLinW;

    float texUf, texVf;
    if constexpr (kFst)
    {
        // Multiply by the exact reciprocal of 16 << mip -- bit-identical to the old division.
        texUf = static_cast<float>(u) * m_draw.texFstScale;
        texVf = static_cast<float>(v) * m_draw.texFstScale;
    }
    else
    {
        // ★★ cont.264 PS2X_GS_RASTERABL bit 8: ABLATION ONLY -- replace the per-pixel perspective
        // divide with an SSE reciprocal APPROXIMATION (rcpss, ~12 bits, ~4 cycles) instead of a
        // true divide (~14 cycles, poorly pipelined). This is the last un-decomposed piece of
        // cont.255's "skeleton" (48.8% of raster; with texture AND frame/Z both free the raster
        // still costs 62.21 ms). The top three draw states are all fst=0, so EVERY shaded pixel on
        // the live path pays this divide -- 6.53 M of them per bench replay.
        // ⚠ Deliberately an APPROXIMATION, not a constant: a constant would send every texel fetch
        // to the same address and change the memory behaviour, which is the cont.231 ablation-drift
        // trap -- this keeps the sampled coordinates within ~1 ulp*2^-12 so the access pattern,
        // wrap/clamp decisions and cache behaviour stay essentially the live ones, and it isolates
        // the DIVIDE's arithmetic cost. The bench VRAM hash will NOT match; that is expected.
        // It also stands in for what a 4-wide SoA loop would buy here, since divps computes four
        // lanes for about the cost of one divss.
        const float invQ = (s_gsRasterAbl & 8u)
                               ? _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(fabsQ(q))))
                               : 1.0f / fabsQ(q);
        texUf = s * invQ * m_draw.texWf;
        texVf = t * invQ * m_draw.texHf;
    }

    // cont.231: samplePointWrapped takes ALREADY-WRAPPED coordinates (the bilinear path wraps its
    // two u and two v once); samplePoint below is the original wrap-then-fetch entry.
    // ---- PS2X_GS_TEXVERIFY: the cache-fill this oracle validates (see the flag's comment).
    // Thread-local, because m_draw is (cont.250 section 16: a SHARED per-draw structure cost 5.7x
    // on the production rasterizer -- the cache and its invalidation must both be per-thread).
    // Rebuilt whenever the descriptor key changes; NOT yet persistent across draws -- persistence
    // and invalidation are the next increment, and deliberately not conflated with proving the
    // decode correct.
    struct Gs2LinTex
    {
        uint64_t key = ~0ull;
        int w = 0, h = 0;
        std::vector<uint32_t> texels;
        bool usable = false;
    };
    static thread_local Gs2LinTex s_linTex;

    auto texVerifyFill = [&]() __attribute__((noinline)) -> bool
    {
        const uint64_t key = (uint64_t(mipTbp) << 40) ^ (uint64_t(mipTbw) << 32) ^
                             (uint64_t(tex.psm) << 24) ^ (uint64_t(texW) << 12) ^
                             uint64_t(texH) ^ (uint64_t(tex.cbp) << 44) ^
                             (uint64_t(tex.csa) << 52) ^ (uint64_t(tex.cpsm) << 56);
        if (s_linTex.key == key)
            return s_linTex.usable;
        s_linTex.key = key;
        s_linTex.w = texW;
        s_linTex.h = texH;
        s_linTex.usable = false;
        // The census (cont.250 section 14) collapses the matrix to T4 82.6% / T8 11.1% /
        // CT24 6.0% / CT32 ~0 = 99.6%. Anything else keeps the production path and is counted.
        const uint8_t psm = tex.psm;
        const bool supported = (psm == GS_PSM_T4 || psm == GS_PSM_T8 ||
                                psm == GS_PSM_CT24 || psm == GS_PSM_CT32);
        if (!supported || texW <= 0 || texH <= 0 ||
            static_cast<long long>(texW) * texH > (4LL << 20))
        {
            ++g_texVerify.skipped;
            return false;
        }
        s_linTex.texels.assign(static_cast<size_t>(texW) * static_cast<size_t>(texH), 0u);
        for (int y = 0; y < texH; ++y)
        {
            for (int x = 0; x < texW; ++x)
            {
                uint32_t raw = 0u;
                switch (psm)
                {
                case GS_PSM_T4:   raw = GSMem::ReadP4(m_vram, mipTbp, mipTbw, u32(x), u32(y)); break;
                case GS_PSM_T8:   raw = GSMem::ReadP8(m_vram, mipTbp, mipTbw, u32(x), u32(y)); break;
                case GS_PSM_CT24: raw = GSMem::ReadCT24(m_vram, mipTbp, mipTbw, u32(x), u32(y)); break;
                default:          raw = GSMem::ReadCT32(m_vram, mipTbp, mipTbw, u32(x), u32(y)); break;
                }
                uint32_t out;
                if (psm == GS_PSM_T4 || psm == GS_PSM_T8)
                {
                    uint32_t idx = raw;
                    if (s_gsTexMutate == 3)
                        idx += 1u;
                    out = LookupCLUT(state, static_cast<u8>(idx), tex.cbp, tex.cpsm,
                                     tex.csm, tex.csa, psm);
                }
                else
                {
                    out = (s_gsTexMutate == 2) ? raw : applyTexa(state.texa, psm, raw);
                }
                s_linTex.texels[static_cast<size_t>(y) * static_cast<size_t>(texW) +
                                static_cast<size_t>(x)] = out;
            }
        }
        ++g_texVerify.fills;
        g_texVerify.fillTexels += static_cast<unsigned long long>(texW) * static_cast<unsigned long long>(texH);
        s_linTex.usable = true;
        return true;
    };

    auto texVerifyCheck = [&](int su, int sv, uint32_t produced) __attribute__((noinline))
    {
        if (!texVerifyFill())
            return;
        if (su < 0 || sv < 0 || su >= texW || sv >= texH)
            return;   // wrap modes 2/3 can leave the range; those keep the production path
        const size_t li = (s_gsTexMutate == 1)
                              ? (static_cast<size_t>(su) * static_cast<size_t>(texH) + static_cast<size_t>(sv))
                              : (static_cast<size_t>(sv) * static_cast<size_t>(texW) + static_cast<size_t>(su));
        if (li >= s_linTex.texels.size())
            return;
        const uint32_t cached = s_linTex.texels[li];
        ++g_texVerify.checks;
        ++g_texVerify.checkByPsm[tex.psm & 0x3Fu];
        if (cached != produced)
        {
            ++g_texVerify.missByPsm[tex.psm & 0x3Fu];
            if (s_gsTexFresh)
            {
                uint32_t raw = 0u;
                switch (tex.psm)
                {
                case GS_PSM_T4:   raw = GSMem::ReadP4(m_vram, mipTbp, mipTbw, u32(su), u32(sv)); break;
                case GS_PSM_T8:   raw = GSMem::ReadP8(m_vram, mipTbp, mipTbw, u32(su), u32(sv)); break;
                case GS_PSM_CT24: raw = GSMem::ReadCT24(m_vram, mipTbp, mipTbw, u32(su), u32(sv)); break;
                default:          raw = GSMem::ReadCT32(m_vram, mipTbp, mipTbw, u32(su), u32(sv)); break;
                }
                const uint32_t fresh =
                    (tex.psm == GS_PSM_T4 || tex.psm == GS_PSM_T8)
                        ? LookupCLUT(state, static_cast<u8>(raw), tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm)
                        : applyTexa(state.texa, tex.psm, raw);
                if (fresh == produced)
                    ++g_texVerify.staleByPsm[tex.psm & 0x3Fu];
                else
                    ++g_texVerify.decodeBadByPsm[tex.psm & 0x3Fu];
            }
            if (g_texVerify.mismatches++ < 8ull)
                std::fprintf(stderr,
                             "[gs2:texverify] MISMATCH psm=0x%02x (u,v)=(%d,%d) tex=%dx%d"
                             " linear=%08x production=%08x tbp=%u tbw=%u cbp=%u cpsm=0x%x csa=%u\n",
                             tex.psm, su, sv, texW, texH, cached, produced,
                             mipTbp, mipTbw, tex.cbp, tex.cpsm, tex.csa);
        }
    };

    auto samplePointWrappedRaw = [&](int sampleU, int sampleV) __attribute__((always_inline)) -> uint32_t
    {

        // ★★ The devirtualised path: 83.0% of this game's texel fetches. Calling the traits
        // directly inlines the whole swizzled lookup (no m_draw.texRead indirect call, no ReadP4
        // frame), and the palette hit is the cont.227 decoded CLUT -- so a PSMT4 texel is now an
        // inlined address computation plus two loads. `fastP4` is loop-invariant, so the branch
        // hoists out of the four bilinear taps.
        // Census before the fast path, so it still counts every texel fetch (it is what
        // characterises the workload; a census blind to 83% of it would be worthless).
        if (s_gsTexCensus)
        {
            ++g_texCensus.texels;
            ++g_texCensus.texelByPsm[tex.psm & 0x3Fu];
            {
                const uint32_t tw = tex.tw & 0x0Fu, th = tex.th & 0x0Fu;
                const uint64_t k = (uint64_t(tex.tbp0) << 40) ^ (uint64_t(tex.tbw) << 32) ^
                                   (uint64_t(tex.psm) << 24) ^ (uint64_t(tw) << 20) ^
                                   (uint64_t(th) << 16) ^ (uint64_t(tex.cbp) << 4) ^ uint64_t(tex.csa);
                auto &e = g_texCensus.texKeys[k];
                ++e.first;
                e.second = (1u << tw) * (1u << th);
                // cont.320: the same sample attributed to (texture, upload generation).
                auto ins = g_texCensus.texGensIdx.try_emplace(k ^ (g_texCensus.curSigIdx * 0x9E3779B97F4A7C15ull));
                if (ins.second) { ins.first->second.texels = e.second; ++g_texCensus.gensPerTex[k]; }
                ++ins.first->second.samples;
                auto &gf = g_texCensus.texGensFull[k ^ (g_texCensus.curSigFull * 0xC2B2AE3D27D4EB4Full)];
                gf.texels = e.second;
                ++gf.samples;
                auto insB = g_texCensus.texGensBlk.try_emplace(k ^ (g_texCensus.curSigBlk * 0x94D049BB133111EBull));
                if (insB.second)
                {
                    insB.first->second.texels = e.second; insB.first->second.key = k; ++g_texCensus.gensPerTexBlk[k];
                    auto insI = g_texCensus.texInfo.try_emplace(k);
                    if (insI.second)
                        insI.first->second = Gs2TexCensus::TexInfo{tex.tbp0, tex.tbw, tex.cbp, static_cast<uint8_t>(tex.psm & 0x3Fu),
                                                                   static_cast<uint8_t>(tw), static_cast<uint8_t>(th)};
                }
                ++insB.first->second.samples;
            }
        }

        if (texLin != nullptr) // cont.320: the same palette index, read linearly (TEXVERIFY checks it)
            return clut[texLin[static_cast<uint32_t>(sampleV) * texLinW + static_cast<uint32_t>(sampleU)]];
        if (fastP4)
        {
            if (s_gsClutDiag)
                return sampleFastP4Diag(state, mipTbp, mipTbw, sampleU, sampleV);
            return clut[GSMem::PixelStorageTraits<GSMem::P4>::Read(
                            GSMem::PageTableP4, m_vram, mipTbp, mipTbw,
                            static_cast<u32>(sampleU), static_cast<u32>(sampleV)) & 15u];
        }
        if (fastP8) // cont.317: the PSMT8 twin -- one inlined address computation, two loads
        {
            if (s_gsClutDiag)
                return sampleFastP8Diag(state, mipTbp, mipTbw, sampleU, sampleV);
            return clut[GSMem::PixelStorageTraits<GSMem::P8>::Read(
                            GSMem::PageTableP8, m_vram, mipTbp, mipTbw,
                            static_cast<u32>(sampleU), static_cast<u32>(sampleV)) & 255u];
        }

        u32 out = m_draw.texRead(m_vram, mipTbp, mipTbw, sampleU, sampleV);

        switch (tex.psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return applyTexa(state.texa, tex.psm, out);
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
        case GS_PSM_T8:
        case GS_PSM_T8H:
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            // ★★ cont.227: one L1 load instead of resolveClutIndex + a swizzled VRAM read + a
            // std::map probe + applyTexa. clutValid is only ever set for the T4 family, whose
            // texel is a 4-bit field, so `out & 15` is exactly what resolveClutIndex masked.
            if (m_draw.clutValid)
            {
                const uint32_t cached =
                    m_draw.clut[(out + (s_gsClutMutate == 2 ? 1u : 0u)) & m_draw.clutMask];
                if (s_gsClutVerify)
                {
                    const uint32_t live = LookupCLUT(state, static_cast<u8>(out), tex.cbp,
                                                     tex.cpsm, tex.csm, tex.csa, tex.psm);
                    ++g_clutVerify.checks;
                    if (cached != live && g_clutVerify.mismatches++ < 8ull)
                        std::fprintf(stderr,
                                     "[gs2:clutverify] MISMATCH idx=%u cached=%08x live=%08x"
                                     " cbp=%u cpsm=0x%x csm=%u csa=%u psm=0x%x\n",
                                     out & 15u, cached, live, tex.cbp, tex.cpsm, tex.csm,
                                     tex.csa, tex.psm);
                }
                return cached;
            }
            return LookupCLUT(state, static_cast<u8>(out), tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        // Undefined TEX0 psm decodes as PSMCT32 on real GS (PCSX2 GSLocalMemory: undefined
        // psm table entries take the PSMCT32 layout) -- and ReadVramUnlocked above already
        // read the texel through the CT32 fallback. Returning the magenta debug texel here
        // instead paints every draw that uses an undefined psm solid magenta: LOTR ROTK's
        // full-screen loading/movie sprites carry a literal TEX0 psm=0x20. (Same fix as the
        // pre-refactor rasterizer sample dispatch, cont.116q.)
        return applyTexa(state.texa, GS_PSM_CT32, out);
    };
    // PS2X_GS_TEXVERIFY seam: production result first (authoritative and unchanged), then the
    // oracle compares the independent linear decode against it. One predicted branch when off.
    auto samplePointWrapped = [&](int sampleU, int sampleV) __attribute__((always_inline)) -> uint32_t
    {
        const uint32_t produced = samplePointWrappedRaw(sampleU, sampleV);
        if (s_gsTexDiag)
            texVerifyCheck(sampleU, sampleV, produced);
        return produced;
    };

    auto samplePoint = [&](int sampleU, int sampleV) __attribute__((always_inline)) -> uint32_t
    {
        return samplePointWrapped(wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU),
                                  wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV));
    };

    // ★★ cont.268 PS2X_GS_TEXTAP: the tap with every loop-invariant gate already decided. Body is
    // the `fastP4` arm of samplePointWrappedRaw verbatim, reached only when no diagnostic is armed.
    const bool tapFast = m_draw.texTapFast && texLin == nullptr; // resolved per draw, see resolveTexMip; cont.320: a cached level takes the general arm
    // ⚠⚠⚠ cont.332o THE SCALAR FAST TAP READ PSMT8 AS PSMT4. `texTapFast` is set for
    // `texFastP4 || texFastP8` (cont.317 widened it), but this lambda was "the fastP4 arm
    // verbatim" (cont.268) and always read through PixelStorageTraits<P4> / PageTableP4 with a
    // 4-bit mask -- so every PSMT8 draw that reached it took a 4-bit read of an 8-bit texture and
    // indexed only the first 16 CLUT entries. PS2X_GS_QUADTEX_VERIFY found it: 80.9% of PSMT8
    // texels disagreed with the four-wide sampler (PSMT4: 0 of 2.8 BILLION), every failing line
    // reading `tapFast=1 lin=0 fastP8=1`.
    // It is LATENT in the default configuration -- the decoded-index cache (cont.320) usually
    // serves PSMT8, and `tapFast` is false whenever it does -- which is why it survived: the
    // bench capture is overwhelmingly PSMT4, and the live path mostly takes the cache.
    auto tapP4 = [&](int sampleU, int sampleV) __attribute__((always_inline)) -> uint32_t
    {
        if (fastP8)
            return clut[GSMem::PixelStorageTraits<GSMem::P8>::Read(
                            GSMem::PageTableP8, m_vram, mipTbp, mipTbw,
                            static_cast<u32>(sampleU), static_cast<u32>(sampleV)) & 255u];
        return clut[GSMem::PixelStorageTraits<GSMem::P4>::Read(
                        GSMem::PageTableP4, m_vram, mipTbp, mipTbw,
                        static_cast<u32>(sampleU), static_cast<u32>(sampleV)) & 15u];
    };

    // ★★ cont.268 PS2X_GS_TEXWRAP: which wrap pair this primitive uses, decided once per call
    // instead of four times per pixel. 0 = REPEAT/REPEAT (96.15% of this game's texel fetches),
    // 1 = CLAMP/CLAMP (3.85%), 2 = anything else, which keeps the original runtime dispatch.
    // Arms 0 and 1 are `wrapTextureCoordinate`'s case 0 and case 1 copied verbatim.
    const int wrapClass = m_draw.texWrapClass;   // resolved per draw, see resolveTexMip
    const uint32_t wrapMaskU = m_draw.texWrapMaskU;
    const uint32_t wrapMaskV = m_draw.texWrapMaskV;

    if constexpr (!kLinear)
    {
        const int pu = static_cast<int>(texUf), pv = static_cast<int>(texVf);
        int wpu, wpv;
        if (wrapClass == 0)
        {
            wpu = static_cast<int>(static_cast<uint32_t>(pu) & wrapMaskU);
            wpv = static_cast<int>(static_cast<uint32_t>(pv) & wrapMaskV);
        }
        else if (wrapClass == 1)
        {
            wpu = clampInt(pu, 0, texW - 1);
            wpv = clampInt(pv, 0, texH - 1);
        }
        else
        {
            wpu = wrapTextureCoordinate(pu, texW, wrapU, minU, maxU);
            wpv = wrapTextureCoordinate(pv, texH, wrapV, minV, maxV);
        }
        return tapFast ? tapP4(wpu, wpv) : samplePointWrapped(wpu, wpv);
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    // ★ cont.231: the four taps share two u and two v coordinates, so wrap each ONCE (four
    // wrapTextureCoordinate calls per pixel instead of eight); the wrapped values are exactly what
    // samplePoint would have computed for each tap. PS2X_GS_FASTLERP=0 restores the scalar
    // per-channel lerpChannel path (the A/B and the kill switch); both produce the same bytes.
    if (s_gsFastLerp)
    {
        int wu0, wu1, wv0, wv1;
        if (wrapClass == 0) // REPEAT/REPEAT -- wrapTextureCoordinate case 0, verbatim
        {
            wu0 = static_cast<int>(static_cast<uint32_t>(u0) & wrapMaskU);
            wu1 = static_cast<int>(static_cast<uint32_t>(u1) & wrapMaskU);
            wv0 = static_cast<int>(static_cast<uint32_t>(v0) & wrapMaskV);
            wv1 = static_cast<int>(static_cast<uint32_t>(v1) & wrapMaskV);
        }
        else if (wrapClass == 1) // CLAMP/CLAMP -- wrapTextureCoordinate case 1, verbatim
        {
            wu0 = clampInt(u0, 0, texW - 1);
            wu1 = clampInt(u1, 0, texW - 1);
            wv0 = clampInt(v0, 0, texH - 1);
            wv1 = clampInt(v1, 0, texH - 1);
        }
        else
        {
            wu0 = wrapTextureCoordinate(u0, texW, wrapU, minU, maxU);
            wu1 = wrapTextureCoordinate(u1, texW, wrapU, minU, maxU);
            wv0 = wrapTextureCoordinate(v0, texH, wrapV, minV, maxV);
            wv1 = wrapTextureCoordinate(v1, texH, wrapV, minV, maxV);
        }
        if (tapFast)
            return bilinear4(tapP4(wu0, wv0), tapP4(wu1, wv0), tapP4(wu0, wv1), tapP4(wu1, wv1), fx, fy);
        const uint32_t c00 = samplePointWrapped(wu0, wv0);
        const uint32_t c10 = samplePointWrapped(wu1, wv0);
        const uint32_t c01 = samplePointWrapped(wu0, wv1);
        const uint32_t c11 = samplePointWrapped(wu1, wv1);
        return bilinear4(c00, c10, c01, c11, fx, fy);
    }

    const uint32_t c00 = samplePoint(u0, v0);
    const uint32_t c10 = samplePoint(u1, v0);
    const uint32_t c01 = samplePoint(u0, v1);
    const uint32_t c11 = samplePoint(u1, v1);

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v, uint32_t mip)
{
    if (m_draw.texMip != mip || !s_gsFastTex)
        resolveTexMip(state, mip);
    if (state.prim.fst)
        return state.linearFilter ? sampleTextureT<true, true>(state, s, t, q, u, v)
                                  : sampleTextureT<true, false>(state, s, t, q, u, v);
    return state.linearFilter ? sampleTextureT<false, true>(state, s, t, q, u, v)
                              : sampleTextureT<false, false>(state, s, t, q, u, v);
}

struct GSCpuBackend::SpriteSetup
{
    int drawX0, drawX1, bandY0, bandY1, unclippedX0, unclippedY0;
    float spriteW, spriteH, u0f, v0f, u1f, v1f;
    u32 z1;
    uint8_t r, g, b, a, fog;
};

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch, int bandY0, int bandY1)
{
    const GSDrawState &state = batch.state;
    if (s_skipZTex && state.prim.tme &&
        (state.context.tex0.tbp0 >> 5) == (state.context.zbuf.zbp >> 5))
        return; // cont.329j ablation: this draw samples the Z buffer as a texture
    if (s_skipTgtPage != 0u && state.context.frame.fbp == s_skipTgtPage)
        return; // cont.329j ablation: this draw renders into the ablated target
    if (s_skipTexPage != 0u && state.prim.tme)
    {
        const uint32_t tp = state.context.tex0.tbp0 >> 5;
        if (tp >= s_skipTexPage && tp < s_skipTexPage + s_skipTexPages)
            return; // cont.329k ablation: this draw SAMPLES the ablated page range
    }
    if (s_skipTexTbp != 0u && state.prim.tme && state.context.tex0.tbp0 == s_skipTexTbp)
        return; // cont.330e ablation: this draw SAMPLES the ablated exact TBP
    // cont.231: the pixel-write plan once per primitive (WritePixel built it per pixel).
    PixelPlan wpPlan;
    makePixelPlan(state, wpPlan);
    const bool wpFast = s_gsFastPath && m_draw.fastCt32Z24;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;
    // cont.318: the key census covered triangles only; charge sprites too (slot type 1).
    struct SpriteCensusScope
    {
        int idx; unsigned long long t0; uint64_t sig;
        ~SpriteCensusScope()
        {
            if (idx < 0) return;
            const unsigned long long dt = __builtin_ia32_rdtsc() - t0;
            g_gsKeyTicks[idx].fetch_add(dt, std::memory_order_relaxed);
            if (sig != 0ull) // cont.319b: the sprite's generic-pipe plan, in the same [gs:keysig] table (typ = spr)
            {
                std::lock_guard<std::mutex> lock(g_gsKeySigMutex);
                auto &e = g_gsKeySig[sig];
                e.first += dt;
                ++e.second;
            }
        }
    } spriteCensus__{-1, 0ull, 0ull};
    if (s_gsKeyCensus)
    {
        const bool tme = state.prim.tme;
        const unsigned key = (tme ? 16u : 0u) | ((tme && state.prim.fst) ? 8u : 0u) | ((tme && state.linearFilter) ? 4u : 0u) |
                             (state.prim.iip ? 2u : 0u) | (state.prim.fge ? 1u : 0u);
        const uint8_t psm = static_cast<uint8_t>(ctx.tex0.psm & 0x3Fu);
        const int texClass = !tme ? 3 : (psm == GS_PSM_T4) ? 0 : (psm == GS_PSM_T8) ? 1
                             : (psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S) ? 2 : 3;
        spriteCensus__.idx = static_cast<int>(key) | (pipeClassFor(wpPlan) << 5) | (texClass << 7) | (1 << 9);
        if (((spriteCensus__.idx >> 5) & 3) == kPipeGeneric)
        {
            const PixelPlan &pp = wpPlan;
            const unsigned fail = (pp.failMask.writeRgb ? 1u : 0u) | (pp.failMask.writeAlpha ? 2u : 0u) | (pp.failMask.writeDepth ? 4u : 0u);
            spriteCensus__.sig = uint64_t(key) | (uint64_t(pp.abe) << 5) | (uint64_t(pp.pabe) << 6) |
                                 (uint64_t(pp.asel & 3u) << 7) | (uint64_t(pp.bsel & 3u) << 9) | (uint64_t(pp.csel & 3u) << 11) | (uint64_t(pp.dsel & 3u) << 13) |
                                 (uint64_t(pp.fix) << 15) | (uint64_t(pp.ztestMethod & 3u) << 23) | (uint64_t(pp.zmask) << 25) | (uint64_t(pp.atEnable) << 26) |
                                 (uint64_t(pp.atst & 7u) << 27) | (uint64_t(pp.aref) << 30) | (uint64_t(fail) << 38) | (uint64_t(pp.fbmsk != 0u) << 41) |
                                 (uint64_t(pp.needsDateRead) << 42) | (uint64_t(pp.fpsm & 63u) << 43) | (1ull << 61) | (1ull << 62);
        }
        spriteCensus__.t0 = __builtin_ia32_rdtsc();
        g_gsKeyDraws[spriteCensus__.idx].fetch_add(1ull, std::memory_order_relaxed);
        const long long w = std::llabs(static_cast<long long>(v1.x) - static_cast<long long>(v0.x)) + 1;
        const long long h = std::llabs(static_cast<long long>(v1.y) - static_cast<long long>(v0.y)) + 1;
        g_gsKeyArea[spriteCensus__.idx].fetch_add(static_cast<unsigned long long>(w * h), std::memory_order_relaxed);
    }
    // cont.319b ABLATION (PS2X_GS_RASTERABL bit 16, garbage output): sprites draw NOTHING, to size their
    // share of the live frame (the key census charged them 34.6% of raster ticks on the recorded fight).
    if (s_gsRasterAbl & 16u)
        return;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    // ★ Band clip (cont.207): this thread owns only scanlines [bandY0, bandY1], so its pixel
    // writes are disjoint from every other band's -- no lock, and order is preserved per band.
    const int bandDrawY0 = drawY0 > bandY0 ? drawY0 : bandY0;
    const int bandDrawY1 = drawY1 < bandY1 ? drawY1 : bandY1;

    if (s_gsTriCensus && bandDrawY1 >= bandDrawY0 && drawX1 >= drawX0)
    {
        ++g_triCensus.sprites;
        g_triCensus.spritePixels += static_cast<unsigned long long>(drawX1 - drawX0 + 1) *
                                    static_cast<unsigned long long>(bandDrawY1 - bandDrawY0 + 1);
    }
    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        // ★★★ cont.319b: the four-wide sprite loop, decided once per sprite (see s_gsSprite4).
        if (s_gsSprite4 != 0u && wpFast && !wpPlan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u &&
            !state.prim.fge && bandDrawY0 <= bandDrawY1)
        {
            const int pc = spritePipeClassFor(wpPlan);
            const bool fst = state.prim.fst, lin = state.linearFilter;
            if (m_draw.texMip != 0u || !s_gsFastTex) // what SampleTexture(mip 0) does on every call
                resolveTexMip(state, 0u);
            const SpriteSetup sp{drawX0, drawX1, bandDrawY0, bandDrawY1, unclippedX0, unclippedY0,
                                 spriteW, spriteH, u0f, v0f, u1f, v1f, z1, r, g, b, a, v1.fog};
            if (pc == kPipeDarken && fst && !lin && (s_gsSprite4 & 1u))
            {
                drawSpriteRowsSimd4<true, true, false, kPipeDarken>(state, wpPlan, sp);
                return;
            }
            if (pc == kPipeStdAlpha && !fst && lin && (s_gsSprite4 & 2u))
            {
                drawSpriteRowsSimd4<true, false, true, kPipeStdAlpha>(state, wpPlan, sp);
                return;
            }
        }

        for (int y = bandDrawY0; y <= bandDrawY1; ++y)
        {
            // ★ cont.210: striped band -- this row belongs to another thread.
            if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
                continue;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(state, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleTexture(state, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                writePixelDispatch(state, wpPlan, wpFast, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        // ★★★ cont.319b: the four-wide untextured sprite (the clears and the flat fills).
        // cont.321: + the ZTE=0 full-screen pair (kPipeSubFix, bit 8), the same shape with the FIX blend.
        if ((s_gsSprite4 & 12u) && wpFast && !wpPlan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u &&
            !state.prim.fge && bandDrawY0 <= bandDrawY1)
        {
            const int pc = spritePipeClassFor(wpPlan);
            const SpriteSetup sp{drawX0, drawX1, bandDrawY0, bandDrawY1, unclippedX0, unclippedY0,
                                 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, z1, r, g, b, a, v1.fog};
            if (pc == kPipeOpaque && (s_gsSprite4 & 4u))
            {
                drawSpriteRowsSimd4<false, false, false, kPipeOpaque>(state, wpPlan, sp);
                return;
            }
            if (pc == kPipeSubFix && (s_gsSprite4 & 8u))
            {
                drawSpriteRowsSimd4<false, false, false, kPipeSubFix>(state, wpPlan, sp);
                return;
            }
        }
        for (int y = bandDrawY0; y <= bandDrawY1; ++y)
        {
            // ★ cont.210: striped band -- this row belongs to another thread.
            if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                writePixelDispatch(state, wpPlan, wpFast, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

// ★★ cont.231 PS2X_GS_FASTPIX: the specialised triangle row loop. Everything the generic loop
// computes per pixel is computed here with the SAME expressions in the SAME order; the only
// differences are compile-time: texturing (kTme) / uv-vs-stq (kFst) / point-vs-bilinear (kLinear) /
// flat-vs-gouraud (kIip) / fog (kFge) are template constants, the sampler and the pixel write are
// the shared sampleTextureT / writePixelT bodies inlined with kFast = true, and the pixel-write plan
// is built once per primitive instead of once per pixel. The row span solve is the cont.229 code
// verbatim (it is copied from the generic loop at build time of this file -- see the python edit in
// the cont.231 journal; if it ever diverges the bench hash says so). The loop-body diagnostics
// (TRICENSUS, SPANVERIFY, SPANMUTATE, the CENSUS2 cyan probe) live only in the generic loop, so the
// dispatch below refuses to engage while any of them is armed.
struct GSCpuBackend::TriRowSetup
{
    const GSPrimitiveBatch *batch = nullptr;
    int minX = 0, maxX = 0, minY = 0, maxY = 0;
    float fx0 = 0, fy0 = 0, fx1 = 0, fy1 = 0, fx2 = 0, fy2 = 0;
    float winding = 1.0f, invAbsDenom = 1.0f;
    uint32_t triMip = 0;
    double sa0 = 0, sa1 = 0, sa2 = 0, sb0 = 0, sb1 = 0, sox = 0, umax = 0, adk = 0;
    double sA0 = 0, sA1 = 0, sB0 = 0, sB1 = 0, sinv0 = 0, sinv1 = 0, sinv2 = 0;
    PixelPlan plan{};
};

// ★ cont.267: the cont.229 row-span solve, lifted VERBATIM out of drawTriangleRowsFast so the
// scalar fast loop and the SIMD4 loop share one body. (The generic loop keeps its own copy as the
// reference implementation; this keeps the count at two, not three.) On entry spanX0/spanX1 are
// the primitive's clamped bbox; on exit they are the row's covered interval, empty if spanX0 >
// spanX1. Callers must have verified `spanOk`.
__attribute__((always_inline)) inline void GSCpuBackend::solveRowSpan(const TriRowSetup &rs, float py,
                                                                     int &spanX0, int &spanX1)
{
    constexpr float kEdgeEpsilon = 1.0e-4f;
    constexpr double kSpanUlp = 32.0 / 16777216.0;
    constexpr int kSpanMargin = 0;
    const double vy = static_cast<double>(py) - static_cast<double>(rs.fy2);
    const double vya = std::fabs(vy);
    const double sc0 = rs.sb0 * vy;
    const double sc1 = rs.sb1 * vy;
    // Per-constraint float-error bound (see kSpanUlp above). w2 is derived from w0 and
    // w1, so its error is theirs plus a rounding of a quantity of order 1.
    const double e0 = kSpanUlp * ((rs.sA0 * rs.umax + rs.sB0 * vya) * rs.adk + 1.0);
    const double e1 = kSpanUlp * ((rs.sA1 * rs.umax + rs.sB1 * vya) * rs.adk + 1.0);
    const double ee[3] = {e0, e1, e0 + e1 + kSpanUlp};
    const double aa[3] = {rs.sa0, rs.sa1, rs.sa2};
    const double ii[3] = {rs.sinv0, rs.sinv1, rs.sinv2};
    const double cc[3] = {sc0, sc1, 1.0 - sc0 - sc1};
    double lo = static_cast<double>(rs.minX), hi = static_cast<double>(rs.maxX);
    bool empty = false;
    for (int i = 0; i < 3; ++i)
    {
        const double relaxed = static_cast<double>(kEdgeEpsilon) + ee[i];
        const double lim = -relaxed - cc[i];
        if (aa[i] > 0.0)
        {
            const double b = rs.sox + lim * ii[i];
            if (b > lo) lo = b;
        }
        else if (aa[i] < 0.0)
        {
            const double b = rs.sox + lim * ii[i];
            if (b < hi) hi = b;
        }
        else if (cc[i] < -relaxed)
        {
            empty = true;
            break;
        }
    }
    if (empty)
    {
        spanX0 = rs.minX + 1;
        spanX1 = rs.minX; // provably no coverage on this row
    }
    else if (std::isfinite(lo) && std::isfinite(hi))
    {
        const double flo = std::floor(lo) - kSpanMargin + s_gsSpanMutate;
        const double fhi = std::ceil(hi) + kSpanMargin - s_gsSpanMutate;
        if (flo > static_cast<double>(spanX0))
            spanX0 = (flo > static_cast<double>(spanX1)) ? spanX1 + 1
                                                         : static_cast<int>(flo);
        if (fhi < static_cast<double>(spanX1))
            spanX1 = (fhi < static_cast<double>(spanX0)) ? spanX0 - 1
                                                         : static_cast<int>(fhi);
    }
}

template <bool kTme, bool kFst, bool kLinear, bool kIip, bool kFge, int kPipe>
void GSCpuBackend::drawTriangleRowsFast(const TriRowSetup &rs)
{
    const GSPrimitiveBatch &batch = *rs.batch;
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;
    const PixelPlan &plan = rs.plan;
    const float fx0 = rs.fx0, fy0 = rs.fy0, fx1 = rs.fx1, fy1 = rs.fy1, fx2 = rs.fx2, fy2 = rs.fy2;
    const float winding = rs.winding;
    const float invAbsDenom = rs.invAbsDenom;
    const int minX = rs.minX, maxX = rs.maxX;
    constexpr float kEdgeEpsilon = 1.0e-4f;
    constexpr bool spanOk = true; // the dispatch requires it
    (void)ctx;

    if constexpr (kTme)
    {
        // The generic loop resolves the mip lazily on the first texel; per primitive the result is
        // the same (resolveDraw invalidates texMip for every draw), so resolve once up front.
        if (m_draw.texMip != rs.triMip)
            resolveTexMip(state, rs.triMip);
    }

    // ★★ cont.231: the per-pixel attribute interpolation as SSE lanes with the generic loop's
    // exact per-lane expression -- ((v0 * w0) + (v1 * w1)) + (v2 * w2), single precision, no FMA in
    // this build -- so every lane is bit-identical to its scalar counterpart. Colour lanes are
    // [r, g, b, a] (bytes -> float exactly), attribute lanes are [s, t, q, fog] or, with FST,
    // [u, v, fog, 0]; the casts afterwards are the scalar casts (truncation, clampU8 == packus).
    const __m128 c0 = _mm_set_ps(static_cast<float>(v0.a), static_cast<float>(v0.b), static_cast<float>(v0.g), static_cast<float>(v0.r));
    const __m128 c1 = _mm_set_ps(static_cast<float>(v1.a), static_cast<float>(v1.b), static_cast<float>(v1.g), static_cast<float>(v1.r));
    const __m128 c2 = _mm_set_ps(static_cast<float>(v2.a), static_cast<float>(v2.b), static_cast<float>(v2.g), static_cast<float>(v2.r));
    __m128 t0, t1, t2;
    if constexpr (kTme && kFst)
    {
        t0 = _mm_set_ps(0.0f, static_cast<float>(v0.fog), static_cast<float>(v0.v), static_cast<float>(v0.u));
        t1 = _mm_set_ps(0.0f, static_cast<float>(v1.fog), static_cast<float>(v1.v), static_cast<float>(v1.u));
        t2 = _mm_set_ps(0.0f, static_cast<float>(v2.fog), static_cast<float>(v2.v), static_cast<float>(v2.u));
    }
    else
    {
        t0 = _mm_set_ps(static_cast<float>(v0.fog), v0.q, v0.t, v0.s);
        t1 = _mm_set_ps(static_cast<float>(v1.fog), v1.q, v1.t, v1.s);
        t2 = _mm_set_ps(static_cast<float>(v2.fog), v2.q, v2.t, v2.s);
    }
    const auto lerp3 = [](__m128 a0, __m128 a1, __m128 a2, __m128 w0v, __m128 w1v, __m128 w2v) __attribute__((always_inline))
    {
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(a0, w0v), _mm_mul_ps(a1, w1v)), _mm_mul_ps(a2, w2v));
    };

    for (int y = rs.minY; y <= rs.maxY; ++y)
    {
        if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
            continue;
        float py = static_cast<float>(y) + 0.5f;

        int spanX0 = minX, spanX1 = maxX;
        if (spanOk)
            solveRowSpan(rs, py, spanX0, spanX1); // cont.267: one shared body, see above
        for (int x = spanX0; x <= spanX1; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            const __m128 w0v = _mm_set1_ps(w0), w1v = _mm_set1_ps(w1), w2v = _mm_set1_ps(w2);
            const __m128 attr = lerp3(t0, t1, t2, w0v, w1v, w2v);
            alignas(16) float attrL[4];
            _mm_store_ps(attrL, attr);

            uint8_t r, g, b, a;
            if constexpr (kIip)
            {
                __m128i ci = _mm_cvttps_epi32(lerp3(c0, c1, c2, w0v, w1v, w2v));
                ci = _mm_packus_epi32(ci, ci);
                ci = _mm_packus_epi16(ci, ci);
                const uint32_t cp = static_cast<uint32_t>(_mm_cvtsi128_si32(ci));
                r = static_cast<uint8_t>(cp & 0xFFu);
                g = static_cast<uint8_t>((cp >> 8) & 0xFFu);
                b = static_cast<uint8_t>((cp >> 16) & 0xFFu);
                a = static_cast<uint8_t>(cp >> 24);
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if constexpr (kTme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if constexpr (kFst)
                {
                    const __m128i uvI = _mm_cvttps_epi32(attr);
                    iu = static_cast<uint16_t>(_mm_cvtsi128_si32(uvI));
                    iv = static_cast<uint16_t>(_mm_extract_epi32(uvI, 1));
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    is = attrL[0];
                    it = attrL[1];
                    iq = attrL[2];
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = sampleTextureT<kFst, kLinear>(state, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(ctx.tex0, r, g, b, a, tr, tg, tb, ta);
                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            uint8_t fog;
            if constexpr (kTme && kFst)
                fog = clampU8(_mm_extract_epi32(_mm_cvttps_epi32(attr), 2));
            else
                fog = clampU8(static_cast<int>(attrL[3]));
            writePixelT<true, kFge, kPipe>(state, plan, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
}

// ★★★ cont.271 PS2X_GS_QUADTEX (default ON, "=0" restores the per-pixel sampler) -- the bilinear
// sampler's SETUP four pixels at a time. cont.267's attribution put 24.7% of the hot symbol in
// this prologue (perspective divide, texture scale, -0.5, floor, the fractional weights, the
// wrap), all of it pure lane arithmetic on values the SIMD4 loop already holds in SoA.
// ⚠ Scope, deliberately narrow: the 16 texel gathers stay scalar because they are irreducible,
// and `bilinear4` stays per pixel BECAUSE IT IS ALREADY LANE-PARALLEL ACROSS RGBA -- widening it
// to four pixels would move the same 16 float values through the same op count, so it is not a
// win, only a rewrite. Only the devirtualised PSMT4 tap (82.56% of texels) with a REPEAT/REPEAT
// or CLAMP/CLAMP wrap (100% of the pairs this game uses, cont.268) is served here; anything else
// keeps the scalar sampleTextureT.
// Every lane below is the scalar expression with the same operands in the same order, so the
// bench hash (aa9cd72b600021ce) is the gate, as it was for cont.270.
// ⚠⚠⚠ cont.332o: cont.332n FLIPPED THIS OFF AND THAT WAS WRONG -- RESTORED TO ON.
// The evidence that flipped it was a DOT COUNT (isolated dark pixels fell 14.5 -> 4.4/frame with
// it off), and a dot count is not a correctness measure. Compared PIXEL-WISE against the GL
// renderer over the character box at the same flip, it reads the other way round:
//     QUADTEX=1  mad 2.19   |   QUADTEX=0  mad 39.09   (TEXCACHE=0 and FASTTEX=0 also 2.19)
// With it OFF the face reads (64,52,42) where GL (and the decoded texture) say (111,85,68): the
// whole character darkens, which REMOVES isolated dark pixels while making everything worse.
// The real defect the ablation stumbled into is in the SCALAR fallback -- see the tap8 fix below.
// The dots are NOT explained by this flag and remain open.
// (Original cont.271 note follows.) The user reported
// isolated dark dots on the character under the CPU renderer. Ablated one flag at a time on the
// character-select replay (26 matched frames, dots per frame in the character box):
//     CPU default 14.5 | TEXCACHE=0 14.5 | FASTTEX=0 14.5 | **QUADTEX=0 4.4** | GL 4.4
// i.e. turning this off reproduces the GL renderer's count exactly. At one such dot the four-wide
// setup fetches a texel of ~(73,64,43) where the decoded texture's whole 9x8 neighbourhood around
// its own sample point spans 109..221 -- a texel that is not there. The scalar path and the GL
// renderer agree; this path does not. `=1` restores it for debugging the defect.
// ⚠ The bench hash aa9cd72b600021ce was recorded WITH this on, so it moves -- the recorded hash
// was of the defect. cont.271's -6.11% instruction win is forfeit until the lane arithmetic is
// fixed (perspective divide / scale / -0.5 / floor / weights / wrap, gs_cpu_backend.cpp ~11946).
// ★★★★★ cont.332o PS2X_GS_QUADTEX_VERIFY (default OFF, diagnostic): the four-wide sampler's own
// oracle. cont.332n proved PS2X_GS_QUADTEX wrong from the OUTSIDE (dots on screen); this compares
// it against the scalar sampler texel by texel, on the same s/t/q, and prints the first mismatches
// WITH their inputs -- which is what a fix needs. The pattern is PS2X_GS_TEXVERIFY's
// (production result first, oracle second), so the verified path stays authoritative.
// ⚠⚠ cont.332p: two of its three sites had the verify OUTSIDE an unbraced `if (quadTex)`, so a
// draw the four-wide sampler never served was still compared -- against the zero-initialised texq.
// That, and nothing else, was the "psm=0x20 is 100% mismatched (262,144 texels)" row reported in
// cont.332o: quadTex requires texTapFast, i.e. plain PSMT4/PSMT8 with a valid decoded CLUT
// (resolveTexMip, ~line 7101), so an UNDEFINED psm cannot reach this path at all. The PSMT8
// (0x13) and PSMT4 (0x14) rows are unaffected -- those draws really do take it.
static const bool s_gsQuadVerify = []
{ const char *e = std::getenv("PS2X_GS_QUADTEX_VERIFY"); return e && e[0] && e[0] != '0'; }();
// cont.332o: print only this PSM's mismatches (0 = any). The loudest class is not the biggest one.
static const unsigned s_gsQuadVerifyPsm = []
{ const char *e = std::getenv("PS2X_GS_QUADTEX_VERIFY_PSM"); return (e && e[0]) ? unsigned(std::strtoul(e, nullptr, 0)) : 0u; }();
static std::atomic<unsigned long long> g_quadVerChecks{0}, g_quadVerBad{0}, g_quadVerShown{0};
static std::atomic<unsigned long long> g_quadVerPsm[64], g_quadVerPsmBad[64];
static void quadVerifyTally(unsigned psm, bool bad)
{
    g_quadVerPsm[psm & 63u].fetch_add(1, std::memory_order_relaxed);
    if (bad) g_quadVerPsmBad[psm & 63u].fetch_add(1, std::memory_order_relaxed);
    const unsigned long long n = g_quadVerChecks.load(std::memory_order_relaxed);
    if ((n % 4000000ull) != 0ull || n == 0ull) return;
    const unsigned long long bd = g_quadVerBad.load(std::memory_order_relaxed);
    std::fprintf(stderr, "[gs2:quadverify] checks=%llu MISMATCH=%llu (%.4f%%) | by psm:", n, bd,
                 100.0 * double(bd) / double(n));
    for (unsigned i = 0; i < 64u; ++i)
    {
        const unsigned long long c = g_quadVerPsm[i].load(std::memory_order_relaxed);
        if (!c) continue;
        const unsigned long long b = g_quadVerPsmBad[i].load(std::memory_order_relaxed);
        std::fprintf(stderr, " 0x%02x{n=%llu bad=%llu %.3f%%}", i, c, b, 100.0 * double(b) / double(c));
    }
    std::fprintf(stderr, "\n");
}
static const bool s_gsQuadTex = []
{ const char *e = std::getenv("PS2X_GS_QUADTEX"); return !(e && e[0] == '0'); }();
// ★★★ cont.319 PS2X_GS_QUADSHADE (default ON, "=0" restores the per-lane tail) -- the SEAM between
// the four-wide sampler and the four-wide write, widened. The cont.318 precise-instruction profile
// of the Helm's Deep capture (perf record -e instructions:pp) put ~16% of the hot four-wide symbol
// in the per-covered-lane tail: combineTexture x4, pack32 x4, the fog clamp, the z round, the
// ctz loop, and four 4-byte stores into cq/fq/zq that the quad write then re-read as one 16-byte
// load -- a store-forwarding failure (ld_blocks.store_forward = 191 M per bench run). Here the
// MODULATE combine runs on all four pixels in 16-bit lanes (the same expression as
// combineTexture's fast path: t*v <= 65025 exact in u16, >>7, packus = clampU8), the vertex colour
// comes straight from the lerp results (cvttps -> packus_epi32 -> min_epu16(255) is the same
// [0,255] saturation as packChan's packus_epi32 + packus_epi16), fog is cvttps + min/max = the
// scalar clampU8((int)f), z is cvttpd of (z + 0.5) with a scalar fallback whenever a lane converts
// to the 0x80000000 indefinite (out of int32 range or NaN -- the scalar (u32)(double) differs
// there). Uncovered lanes carry garbage, exactly as the quad write already masks them. Only
// TFX=MODULATE with the fast combine; DECAL/HIGHLIGHT keep the scalar tail.
static const bool s_gsQuadShade = []
{ const char *e = std::getenv("PS2X_GS_QUADSHADE"); return !(e && e[0] == '0'); }();
// pshufb masks that drop a texel (bytes 0-3 of the source) into lane i and zero the rest, so the
// sampler can build its four-texel vector in a register instead of four 4-byte stores.
static const __m128i s_laneInsertMask[4] = {
    _mm_setr_epi8(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1),
    _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3)};

// cont.321b: population count of a 4-bit cover mask (the build has no -mpopcnt; __builtin_popcount was a libgcc call).
static constexpr uint8_t kPop4[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

// ★ cont.324 PS2X_GS_TAPIDX (default 1; "=0" restores the per-pixel index maths + u32 palette): the
// four-wide lin tap's ADDRESSING done in lanes. The cont.323 annotate of the hottest loop put 55
// instructions on every covered pixel of this tap loop: `v * lw + u` twice in scalar (two imul, three
// zero-extending moves, four stack reloads of the wrapped coordinates) and, per texel, a widen
// (pmovzxbd) plus a convert (cvtdq2ps) of the palette entry. Here the two row bases and four texel
// indices are lane arithmetic on the vectors the prologue already holds (two pmulld + four paddd per
// QUAD; 32-bit lanes wrap exactly as the scalar u32 did), and the palette is read as floats
// (ResolvedDraw::clutF, filled at decode with bilinear4's own unpack) -- one aligned load per texel
// into the SAME filter (bilinear4Lerp). Same loads, same arithmetic, same order: the bench hash gates it.
static const bool s_gsTapIdx = []
{ const char *e = std::getenv("PS2X_GS_TAPIDX"); return !(e && e[0] == '0'); }();

// ★ cont.324: the per-draw sampler state the four-wide loops read, copied ONCE per primitive out of
// the thread-local m_draw into a caller-local. WHY: the pixel write stores through a `u8 *` (VRAM),
// and a byte store may alias any object with a visible address -- the thread-local m_draw included --
// so the compiler re-read texWf/texHf/texLin/texLinW/the wrap masks/the palette pointer from TLS on
// EVERY quad (eight `fs:` loads per group in the cont.323 annotate). A local whose address never
// escapes (the sampler is always_inline) cannot be aliased by those stores, so it is read once per
// draw. `taps` accumulates here and the row function flushes it to m_draw.texTaps once -- the
// predictor reads it at the next resolve, on this same thread, so the sum is what it was.
struct GSCpuBackend::QuadTexCtx
{
    uint8_t *vram = nullptr;
    const uint8_t *lin = nullptr;         // the decoded-index cache (cont.320), or null = swizzled taps
    uint32_t lw = 0u;
    const uint32_t *clut = nullptr;
    const float (*clutF)[4] = nullptr;
    float wf = 1.0f, hf = 1.0f;
    int wrapClass = 2;                    // 0 REPEAT/REPEAT, 1 CLAMP/CLAMP (2 never reaches the quad path)
    uint32_t maskU = 0u, maskV = 0u;
    int wMax = 0, hMax = 0;
    bool fastP8 = false;
    uint32_t tbp = 0u, tbw = 0u;
    uint32_t taps = 0u;
    // cont.324b: the prologue's broadcasts, built once per draw -- the compiler re-materialised each from .rodata / a scalar
    // (movss + shufps, movd + pshufd) on every quad: ten instructions for one aligned load each here.
    __m128 wfv = _mm_setzero_ps(), hfv = _mm_setzero_ps(), negZero = _mm_setzero_ps(), eps = _mm_setzero_ps(),
           one = _mm_setzero_ps(), half = _mm_setzero_ps();
    __m128i maskUv = _mm_setzero_si128(), maskVv = _mm_setzero_si128(), wMaxv = _mm_setzero_si128(), hMaxv = _mm_setzero_si128(),
            lwv = _mm_setzero_si128(), onei = _mm_setzero_si128();
};

GSCpuBackend::QuadTexCtx GSCpuBackend::makeQuadTexCtx() const
{
    QuadTexCtx tc;
    tc.vram = m_vram;
    tc.lin = m_draw.texLin;
    tc.lw = m_draw.texLinW;
    tc.clut = m_draw.clut;
    tc.clutF = m_draw.clutF;
    tc.wf = m_draw.texWf;
    tc.hf = m_draw.texHf;
    tc.wrapClass = m_draw.texWrapClass;
    tc.maskU = m_draw.texWrapMaskU;
    tc.maskV = m_draw.texWrapMaskV;
    tc.wMax = m_draw.texW - 1;
    tc.hMax = m_draw.texH - 1;
    tc.fastP8 = m_draw.texFastP8;
    tc.tbp = m_draw.texTbp;
    tc.tbw = m_draw.texTbw;
    tc.wfv = _mm_set1_ps(tc.wf);
    tc.hfv = _mm_set1_ps(tc.hf);
    tc.negZero = _mm_set1_ps(-0.0f);
    tc.eps = _mm_set1_ps(1.0e-8f);
    tc.one = _mm_set1_ps(1.0f);
    tc.half = _mm_set1_ps(0.5f);
    tc.maskUv = _mm_set1_epi32(static_cast<int>(tc.maskU));
    tc.maskVv = _mm_set1_epi32(static_cast<int>(tc.maskV));
    tc.wMaxv = _mm_set1_epi32(tc.wMax);
    tc.hMaxv = _mm_set1_epi32(tc.hMax);
    tc.lwv = _mm_set1_epi32(static_cast<int>(tc.lw));
    tc.onei = _mm_set1_epi32(1);
    return tc;
}

__attribute__((always_inline)) inline void GSCpuBackend::sampleQuadBilinearP4(
    QuadTexCtx &tc, const float *sIn, const float *tIn, const float *qIn, int coverMask, uint32_t *out)
{
    const __m128 s = _mm_load_ps(sIn), t = _mm_load_ps(tIn), q = _mm_load_ps(qIn);

    // fabsQ: (std::fabs(q) > 1.0e-8f) ? q : 1.0f   (the broadcasts are per-draw ctx loads, cont.324b)
    const __m128 absQ = _mm_andnot_ps(tc.negZero, q);
    const __m128 qOk = _mm_cmpgt_ps(absQ, tc.eps);
    const __m128 qs = _mm_blendv_ps(tc.one, q, qOk);
    // 1.0f / fabsQ(q) -- divps is correctly rounded per lane, identical to the scalar divide.
    const __m128 invQ = _mm_div_ps(tc.one, qs);
    // texUf = s * invQ * texWf -- left-associative, exactly as the scalar writes it.
    const __m128 texUf = _mm_mul_ps(_mm_mul_ps(s, invQ), tc.wfv);
    const __m128 texVf = _mm_mul_ps(_mm_mul_ps(t, invQ), tc.hfv);

    const __m128 half = tc.half;
    const __m128 su = _mm_sub_ps(texUf, half), sv = _mm_sub_ps(texVf, half);
    const __m128i u0 = _mm_cvttps_epi32(_mm_floor_ps(su)); // (int)std::floor(sampleU)
    const __m128i v0 = _mm_cvttps_epi32(_mm_floor_ps(sv));
    const __m128i one = tc.onei;
    const __m128i u1 = _mm_add_epi32(u0, one), v1 = _mm_add_epi32(v0, one);
    // fx = sampleU - (float)u0 -- convert BACK from the int, not from the floor result, so it is
    // the scalar expression even where the two could differ.
    alignas(16) float fxs[4], fys[4];
    _mm_store_ps(fxs, _mm_sub_ps(su, _mm_cvtepi32_ps(u0)));
    _mm_store_ps(fys, _mm_sub_ps(sv, _mm_cvtepi32_ps(v0)));

    __m128i wu0v, wu1v, wv0v, wv1v;
    if (tc.wrapClass == 0) // REPEAT/REPEAT -- wrapTextureCoordinate case 0, verbatim
    {
        const __m128i mu = tc.maskUv, mv = tc.maskVv;
        wu0v = _mm_and_si128(u0, mu);
        wu1v = _mm_and_si128(u1, mu);
        wv0v = _mm_and_si128(v0, mv);
        wv1v = _mm_and_si128(v1, mv);
    }
    else // CLAMP/CLAMP -- clampInt(c, 0, size - 1), verbatim
    {
        const __m128i zero = _mm_setzero_si128();
        const __m128i hu = tc.wMaxv, hv = tc.hMaxv;
        wu0v = _mm_min_epi32(_mm_max_epi32(u0, zero), hu);
        wu1v = _mm_min_epi32(_mm_max_epi32(u1, zero), hu);
        wv0v = _mm_min_epi32(_mm_max_epi32(v0, zero), hv);
        wv1v = _mm_min_epi32(_mm_max_epi32(v1, zero), hv);
    }

    // cont.319: the four texels are assembled in a register (pshufb lane insert) and stored ONCE
    // as 16 bytes, so a consumer that loads them as one vector never trips store forwarding.
    // `out` must be 16-byte aligned; uncovered lanes are 0, as the zero-initialised array was.
    const uint32_t *const clut = tc.clut;
    __m128i acc = _mm_setzero_si128();
    if (const uint8_t *lin = tc.lin) // cont.320: the decoded-index cache, one byte load per tap
    {
        // cont.321b: __builtin_popcount was a libgcc CALL per quad (no -mpopcnt): 1.4% of raster samples. Four bits: a table.
        tc.taps += 4u * static_cast<uint32_t>(kPop4[coverMask & 15]);
        const uint32_t lw = tc.lw;
        if (s_gsTapIdx)
        {
            // Row bases and texel indices in lanes: (u32)v * lw + (u32)u, mod 2^32 exactly as the scalar.
            const __m128i r0 = _mm_mullo_epi32(wv0v, tc.lwv), r1 = _mm_mullo_epi32(wv1v, tc.lwv);
            alignas(16) uint32_t i00[4], i10[4], i01[4], i11[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(i00), _mm_add_epi32(r0, wu0v));
            _mm_store_si128(reinterpret_cast<__m128i *>(i10), _mm_add_epi32(r0, wu1v));
            _mm_store_si128(reinterpret_cast<__m128i *>(i01), _mm_add_epi32(r1, wu0v));
            _mm_store_si128(reinterpret_cast<__m128i *>(i11), _mm_add_epi32(r1, wu1v));
            const float (*const cf)[4] = tc.clutF;
            for (int m = coverMask; m != 0; m &= m - 1)
            {
                const int i = __builtin_ctz(static_cast<unsigned>(m));
                const uint32_t tv = bilinear4Lerp(_mm_load_ps(cf[lin[i00[i]]]), _mm_load_ps(cf[lin[i10[i]]]),
                                                  _mm_load_ps(cf[lin[i01[i]]]), _mm_load_ps(cf[lin[i11[i]]]),
                                                  fxs[i], fys[i]);
                acc = _mm_or_si128(acc, _mm_shuffle_epi8(_mm_cvtsi32_si128(static_cast<int>(tv)), s_laneInsertMask[i]));
            }
            _mm_store_si128(reinterpret_cast<__m128i *>(out), acc);
            return;
        }
        alignas(16) int32_t wu0[4], wu1[4], wv0[4], wv1[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(wu0), wu0v);
        _mm_store_si128(reinterpret_cast<__m128i *>(wu1), wu1v);
        _mm_store_si128(reinterpret_cast<__m128i *>(wv0), wv0v);
        _mm_store_si128(reinterpret_cast<__m128i *>(wv1), wv1v);
        const auto tapL = [&](int u, int v) __attribute__((always_inline)) -> uint32_t
        { return clut[lin[static_cast<uint32_t>(v) * lw + static_cast<uint32_t>(u)]]; };
        for (int m = coverMask; m != 0; m &= m - 1)
        {
            const int i = __builtin_ctz(static_cast<unsigned>(m));
            const uint32_t t = bilinear4(tapL(wu0[i], wv0[i]), tapL(wu1[i], wv0[i]),
                                         tapL(wu0[i], wv1[i]), tapL(wu1[i], wv1[i]), fxs[i], fys[i]);
            acc = _mm_or_si128(acc, _mm_shuffle_epi8(_mm_cvtsi32_si128(static_cast<int>(t)), s_laneInsertMask[i]));
        }
        _mm_store_si128(reinterpret_cast<__m128i *>(out), acc);
        return;
    }

    alignas(16) int32_t wu0[4], wu1[4], wv0[4], wv1[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(wu0), wu0v);
    _mm_store_si128(reinterpret_cast<__m128i *>(wu1), wu1v);
    _mm_store_si128(reinterpret_cast<__m128i *>(wv0), wv0v);
    _mm_store_si128(reinterpret_cast<__m128i *>(wv1), wv1v);
    // ---- the irreducible part: 16 swizzled gathers, then the unchanged bilinear filter -------
    const uint32_t tbp = tc.tbp, tbw = tc.tbw;
    uint8_t *const vram = tc.vram;
    const auto tap = [&](int u, int v) __attribute__((always_inline)) -> uint32_t
    {
        return clut[GSMem::PixelStorageTraits<GSMem::P4>::Read(
                        GSMem::PageTableP4, vram, tbp, tbw,
                        static_cast<u32>(u), static_cast<u32>(v)) & 15u];
    };
    // cont.317: the PSMT8 tap, selected once per draw (texFastP8), so the gather loop stays
    // branch-free per texel.
    const auto tap8 = [&](int u, int v) __attribute__((always_inline)) -> uint32_t
    {
        return clut[GSMem::PixelStorageTraits<GSMem::P8>::Read(
                        GSMem::PageTableP8, vram, tbp, tbw,
                        static_cast<u32>(u), static_cast<u32>(v)) & 255u];
    };
    if (s_gsTexCache)
        tc.taps += 4u * static_cast<uint32_t>(kPop4[coverMask & 15]);
    if (tc.fastP8)
    {
        for (int m = coverMask; m != 0; m &= m - 1)
        {
            const int i = __builtin_ctz(static_cast<unsigned>(m));
            const uint32_t t = bilinear4(tap8(wu0[i], wv0[i]), tap8(wu1[i], wv0[i]),
                                         tap8(wu0[i], wv1[i]), tap8(wu1[i], wv1[i]), fxs[i], fys[i]);
            acc = _mm_or_si128(acc, _mm_shuffle_epi8(_mm_cvtsi32_si128(static_cast<int>(t)), s_laneInsertMask[i]));
        }
        _mm_store_si128(reinterpret_cast<__m128i *>(out), acc);
        return;
    }
    for (int m = coverMask; m != 0; m &= m - 1)
    {
        const int i = __builtin_ctz(static_cast<unsigned>(m));
        const uint32_t t = bilinear4(tap(wu0[i], wv0[i]), tap(wu1[i], wv0[i]),
                                     tap(wu0[i], wv1[i]), tap(wu1[i], wv1[i]), fxs[i], fys[i]);
        acc = _mm_or_si128(acc, _mm_shuffle_epi8(_mm_cvtsi32_si128(static_cast<int>(t)), s_laneInsertMask[i]));
    }
    _mm_store_si128(reinterpret_cast<__m128i *>(out), acc);
}

// ★★★ cont.270 PS2X_GS_QUADWRITE (default OFF) -- the pixel write for FOUR pixels with colour
// resident in 16-bit SIMD lanes: the STRUCTURE of PCSX2's `GSDrawScanline`
// (GS/Renderers/SW/GSDrawScanline.cpp -- `rb`/`ga` 16-bit lanes carried through fog, blend and
// write, colour never unpacked to scalar bytes) with OUR arithmetic preserved bit-for-bit.
//
// ⚠ Structure, NOT code (cont.269 diff). PCSX2 keeps colour PRE-SCALED in its lanes
// (`gaf.srl16<7>()`) and blends with `modulate16<1>` = `sll16<2>` + `mul16hrs`, i.e.
// (rb*As + 32) >> 6 -- a different scale AND rounded where ours truncates. Porting it verbatim
// would change every blended pixel and cost us the bench-hash oracle on a game that works.
// So each step is our own expression re-expressed in lanes:
//   * blend ((s-d)*a)>>7 needs 17 signed bits, which a 16-bit lane cannot hold, so the product
//     is reconstructed exactly from mullo (bits 0-15) + mulhi (bits 16-31).
//   * fog (f*c)>>8 has both products <= 65025, which DOES fit an unsigned 16-bit lane, so
//     mullo alone is exact.
//   * clampU8 is min/max against 0/255 (packus saturation), as cont.231's FASTCOMBINE argued.
// Inactive lanes are written back with the value just read: observationally identical, because
// band threads own whole rows (cont.210), and it keeps the store branch-free.
// ⚠ The caller must exclude p.needsDateRead (DATE is not implemented here) and any armed
// diagnostic; kPipeStdAlpha already guarantees ABE 0101 / no PABE / ZTST=GEQUAL, and kFast
// guarantees a CT32 frame with a Z24 depth buffer.
static const bool s_gsQuadWrite = []
{ const char *e = std::getenv("PS2X_GS_QUADWRITE"); return !(e && e[0] == '0'); }();
// Bisect handle for a hash break: each bit DISABLES one stage of the quad path (1 = fog,
// 2 = alpha test, 4 = blend, 8 = fbmsk/preserve-alpha). Diagnostic only; default 0.
static const unsigned s_gsQuadOff = []
{ const char *e = std::getenv("PS2X_GS_QUADOFF"); return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 0u; }();
// cont.324b: cover mask -> lane mask (bit i set -> lane i all ones), one aligned load instead of a broadcast + and + compare.
static const __m128i s_laneCoverMask[16] = {
    _mm_setr_epi32(0, 0, 0, 0),   _mm_setr_epi32(-1, 0, 0, 0),   _mm_setr_epi32(0, -1, 0, 0),   _mm_setr_epi32(-1, -1, 0, 0),
    _mm_setr_epi32(0, 0, -1, 0),  _mm_setr_epi32(-1, 0, -1, 0),  _mm_setr_epi32(0, -1, -1, 0),  _mm_setr_epi32(-1, -1, -1, 0),
    _mm_setr_epi32(0, 0, 0, -1),  _mm_setr_epi32(-1, 0, 0, -1),  _mm_setr_epi32(0, -1, 0, -1),  _mm_setr_epi32(-1, -1, 0, -1),
    _mm_setr_epi32(0, 0, -1, -1), _mm_setr_epi32(-1, 0, -1, -1), _mm_setr_epi32(0, -1, -1, -1), _mm_setr_epi32(-1, -1, -1, -1)};

namespace
{
    // Exact ((d * f) >> 7) per 16-bit lane. d in [-255,255], f in [0,255], so the product needs
    // 17 signed bits: mullo holds bits 0-15, mulhi bits 16-31, and the result (in [-508,508]) is
    // bits 7-22 -- shift each half into place and OR. Verified against the scalar shift at the
    // extremes: d=-255,f=255 -> p=-65025, lo=0x01FF, hi=0xFFFF, (0x01FF>>7)|(0xFFFF<<9)=0xFE03
    // = -509 = -65025 >> 7.
    __attribute__((always_inline)) inline __m128i mulShift7_epi16(__m128i d, __m128i f)
    {
        const __m128i lo = _mm_mullo_epi16(d, f);
        const __m128i hi = _mm_mulhi_epi16(d, f);
        return _mm_or_si128(_mm_srli_epi16(lo, 7), _mm_slli_epi16(hi, 9));
    }
}

// ★ cont.324: the per-draw write state, resolved ONCE per primitive (makeQuadWriteCtx) into a
// caller-local. WHY: `plan` was a reference into the caller's TriRowSetup, whose address has escaped,
// so every VRAM byte store forced the compiler to re-read fbaSet / zmask / atEnable / atst / aref /
// the fail mask / fbmsk / the four table pointers -- and the plan pointer itself -- on EVERY quad
// (the cont.323 annotate of the hottest loop: six reloads of the pointer and a dozen byte compares
// per group). The fail-mask and PS2X_GS_QUADOFF booleans fold into lane masks / a resolved ATST here,
// so the quad ORs where it branched. Values only; every expression below is the previous one.
struct GSCpuBackend::QuadWriteCtx
{
    uint8_t *vram = nullptr;
    const uint32_t *fzRowF = nullptr, *fzColF = nullptr, *fzRowZ = nullptr, *fzColZ = nullptr;
    uint32_t fbp = 0u, fbw = 1u, zbp = 0u, fbmsk = 0u;
    int atst = 1;                                     // as the quad applies it: 1 (ALWAYS) when ATE is clear or bisected off
    bool zmask = false, fbaSet = false;
    bool fogOn = true, blendOn = true, maskOn = true; // PS2X_GS_QUADOFF bits 1 / 4 / 8 clear
    bool pairOk = false;                              // cont.325 PS2X_GS_FZPAIR: tables present + pairable + 4-aligned groups
    const uint32_t *colMaxF = nullptr, *colMaxZ = nullptr; // cont.325: prefix maxima of the column tables (makeQuadRow's wrap guard)
    __m128i z24v = _mm_setzero_si128(), hi8v = _mm_setzero_si128(); // 0x00FFFFFF / 0xFF000000 lanes, loaded not re-materialised
    __m128i arefv = _mm_setzero_si128(), failRgb = _mm_setzero_si128(), failAlpha = _mm_setzero_si128(),
            failDepth = _mm_setzero_si128(), fogRB = _mm_setzero_si128(), fogG = _mm_setzero_si128(),
            fixv = _mm_setzero_si128(), fbmskv = _mm_setzero_si128();
};

GSCpuBackend::QuadWriteCtx GSCpuBackend::makeQuadWriteCtx(const PixelPlan &p) const
{
    QuadWriteCtx q;
    q.vram = m_vram;
    q.fzRowF = p.fzRowF;
    q.fzColF = p.fzColF;
    q.fzRowZ = p.fzRowZ;
    q.fzColZ = p.fzColZ;
    q.fbp = p.fbp;
    q.fbw = p.fbw;
    q.zbp = p.zbp;
    q.fbmsk = p.fbmsk;
    q.atst = (p.atEnable && (s_gsQuadOff & 2u) == 0u) ? static_cast<int>(p.atst) : 1;
    q.zmask = p.zmask;
    q.fbaSet = p.fbaSet;
    q.fogOn = (s_gsQuadOff & 1u) == 0u;
    q.blendOn = (s_gsQuadOff & 4u) == 0u;
    q.maskOn = (s_gsQuadOff & 8u) == 0u;
    q.pairOk = s_gsFzPair && p.fzRowF != nullptr && p.fzPair && s_gsSimd4Align; // both four-wide loops start groups at x & ~3
    q.colMaxF = p.fzColMaxF;
    q.colMaxZ = p.fzColMaxZ;
    q.z24v = _mm_set1_epi32(0x00FFFFFF);
    q.hi8v = _mm_set1_epi32(static_cast<int>(0xFF000000u));
    const __m128i all = _mm_set1_epi32(-1), zero = _mm_setzero_si128();
    q.arefv = _mm_set1_epi32(p.aref);
    q.failRgb = p.failMask.writeRgb ? all : zero;
    q.failAlpha = p.failMask.writeAlpha ? all : zero;
    q.failDepth = p.failMask.writeDepth ? all : zero;
    q.fogRB = _mm_set1_epi32((static_cast<int>(p.fogB) << 16) | static_cast<int>(p.fogR));
    q.fogG = _mm_set1_epi32(static_cast<int>(p.fogG));
    q.fixv = _mm_set1_epi16(static_cast<short>(p.fix));
    q.fbmskv = _mm_set1_epi32(static_cast<int>(p.fbmsk));
    return q;
}

// ★ cont.325: per ROW -- the row terms of the two address tables and the 4 MB wrap guard. With the maximum
// column offset over the row's span known (the prefix maximum of the column table at the span's last lane),
// every group of the row stays below MEMORY_SIZE - 24 bytes iff ((row + colMax) << 2) does (64-bit here;
// conservative, never a false positive), so the quad needs no per-group compare and no lane mask: `pair` is
// the row's verdict. ⚠ Build 788 used the maximum over ALL 2048 columns and rejected every row of the
// bench (the profile put 0 samples on the paired moves): the bound must be the row's, not the table's.
struct GSCpuBackend::QuadRowCtx
{
    uint32_t rowF = 0u, rowZ = 0u;
    bool pair = false;
};
__attribute__((always_inline)) inline GSCpuBackend::QuadRowCtx GSCpuBackend::makeQuadRow(const QuadWriteCtx &q, int y, int xLast)
{
    QuadRowCtx r;
    if (q.pairOk)
    {
        const uint32_t xl = std::min<uint32_t>(static_cast<uint32_t>(xLast) | 3u, 2047u); // the last lane of the row's last group
        r.rowF = q.fzRowF[y];
        r.rowZ = q.fzRowZ[y];
        // the true last address of the row = row + (signed) column max; the lane form's u32 add wraps to the same value
        const int64_t fEnd = (static_cast<int64_t>(r.rowF) + static_cast<int32_t>(q.colMaxF[xl])) * 4;
        const int64_t zEnd = (static_cast<int64_t>(r.rowZ) + static_cast<int32_t>(q.colMaxZ[xl])) * 4;
        const int64_t lim = static_cast<int64_t>(GSMem::MEMORY_SIZE) - 24;
        r.pair = fEnd >= 0 && fEnd <= lim && zEnd >= 0 && zEnd <= lim;
        if (__builtin_expect(s_gsFzPairLog, 0))
        {
            static std::atomic<int> shown{0};
            if (shown.fetch_add(1) < 16)
                std::fprintf(stderr, "[gs2:fzpair] row y=%d xLast=%d rowF=%u rowZ=%u colMaxF=%u colMaxZ=%u fEnd=%lld zEnd=%lld pair=%d\n",
                             y, xLast, r.rowF, r.rowZ, q.colMaxF[xl], q.colMaxZ[xl],
                             static_cast<long long>(fEnd), static_cast<long long>(zEnd), r.pair ? 1 : 0);
        }
    }
    return r;
}

template <bool kFge, int kPipe, int kZ>
__attribute__((always_inline)) inline void GSCpuBackend::writePixelQuad(
    const QuadWriteCtx &q, const QuadRowCtx &row, int xg, int y,
    const int32_t *z, const uint32_t *colourIn, const int32_t *fogIn, int coverMask)
{
    const __m128i colour = _mm_load_si128(reinterpret_cast<const __m128i *>(colourIn));
    const __m128i fog = _mm_load_si128(reinterpret_cast<const __m128i *>(fogIn));
    uint8_t *const vram = q.vram;
    // cont.319b kZAlways: no depth test, so the Z buffer is read only when the store below needs
    // the old values for its masked lanes (zmask clear = a Z write); with zmask set, Z is untouched.
    const bool needZ = (kZ != kZAlways) || !q.zmask;

    // ★ cont.325 PS2X_GS_FZPAIR: a 4-aligned group is two contiguous pixel pairs 16 bytes apart (see s_gsFzPair),
    // so its frame / Z are one movq + one movhps each way -- PCSX2's ReadPixel / WritePixel. `paired` is per
    // quad only for the 4 MB wrap the lane form's mask handles: a group whose byte offsets could wrap keeps
    // the lane path (conservative: the OR of the two offsets is compared, never a false positive).
    const __m128i z24mask = q.z24v;
    const auto ld2q = [vram](uint32_t off) __attribute__((always_inline)) -> __m128i
    {
        const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(vram + off));
        return _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(lo), reinterpret_cast<const __m64 *>(vram + off + 16u)));
    };
    const auto st2q = [vram](uint32_t off, __m128i v) __attribute__((always_inline))
    {
        _mm_storel_epi64(reinterpret_cast<__m128i *>(vram + off), v);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(vram + off + 16u), _mm_castsi128_ps(v));
    };
    const auto ld32 = [vram](uint32_t off) __attribute__((always_inline)) -> int
    {
        uint32_t v;
        std::memcpy(&v, vram + off, 4u);
        return static_cast<int>(v);
    };
    const auto st32 = [vram](uint32_t off, uint32_t v) __attribute__((always_inline))
    { std::memcpy(vram + off, &v, 4u); };
    const bool paired = row.pair; // per row (makeQuadRow): tables present, pairable, no wrap on this row
    uint32_t fo0 = 0u, zo0 = 0u;
    if (paired)
    {
        fo0 = (row.rowF + q.fzColF[xg]) << 2;
        if (needZ)
            zo0 = (row.rowZ + q.fzColZ[xg]) << 2;
    }
    alignas(16) uint32_t fo[4], zo[4] = {0u, 0u, 0u, 0u};
    __m128i dst, zbuf = _mm_setzero_si128();
    if (__builtin_expect(paired, 1))
    {
        dst = ld2q(fo0);
        if (needZ) // Z24 ReadAt's `v & 0x00FFFFFF`, applied to the four raw dwords at once
            zbuf = _mm_and_si128(ld2q(zo0), z24mask);
    }
    else
    {
    // ---- addresses: PCSX2's fzbr[y] + fzbc[x] (cont.322d) in lanes, else one Address() per buffer
    __m128i fA, zA = _mm_setzero_si128();
    if (q.fzRowF != nullptr)
    {
        fA = _mm_add_epi32(_mm_set1_epi32(static_cast<int>(q.fzRowF[y])),
                           _mm_loadu_si128(reinterpret_cast<const __m128i *>(q.fzColF + xg)));
        if (needZ)
            zA = _mm_add_epi32(_mm_set1_epi32(static_cast<int>(q.fzRowZ[y])),
                               _mm_loadu_si128(reinterpret_cast<const __m128i *>(q.fzColZ + xg)));
    }
    else
    {
        // A 4-aligned group never crosses a page in x (page extent 64, 64 % 4 == 0), so page and
        // block are shared and only the in-page table entry differs (the pre-cont.322d shape).
        const u32 fbp = q.fbp, fbw = q.fbw, zbp = q.zbp;
        const u32 fA0 = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, fbp, fbw, static_cast<u32>(xg), static_cast<u32>(y));
        const u32 zA0 = GSMem::PixelStorageTraits<GSMem::Z32>::Address(GSMem::PageTableZ32, zbp, fbw, static_cast<u32>(xg), static_cast<u32>(y));
        const uint16_t *const fRow = &GSMem::PageTableC32[fbp % 32u][static_cast<u32>(y) & 31u][static_cast<u32>(xg) & 63u];
        const uint16_t *const zRow = &GSMem::PageTableZ32[zbp % 32u][static_cast<u32>(y) & 31u][static_cast<u32>(xg) & 63u];
        alignas(16) u32 fAddr[4], zAddr[4];
        for (int i = 0; i < 4; ++i)
        {
            fAddr[i] = fA0 + static_cast<u32>(static_cast<int>(fRow[i]) - static_cast<int>(fRow[0]));
            zAddr[i] = zA0 + static_cast<u32>(static_cast<int>(zRow[i]) - static_cast<int>(zRow[0]));
        }
        fA = _mm_load_si128(reinterpret_cast<const __m128i *>(fAddr));
        zA = _mm_load_si128(reinterpret_cast<const __m128i *>(zAddr));
    }
    // ---- byte offsets, exactly ReadAt/WriteAt's (addr * 32 / 8) & (MEMORY_SIZE - 4), in lanes ----
    // (C32 and Z24 both pack to 32 bits at bit offset 0; the mask is the VRAM wrap for large fbp/y.)
    const __m128i offMask = _mm_set1_epi32(static_cast<int>(GSMem::MEMORY_SIZE - 4u));
    _mm_store_si128(reinterpret_cast<__m128i *>(fo), _mm_and_si128(_mm_slli_epi32(fA, 2), offMask));
    if (needZ)
        _mm_store_si128(reinterpret_cast<__m128i *>(zo), _mm_and_si128(_mm_slli_epi32(zA, 2), offMask));

    // cont.319: the four frame / Z reads go straight into lanes (pinsrd) -- four 4-byte stores
    // followed by one 16-byte load was a store-forwarding failure on every quad.
    dst = _mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(
        _mm_cvtsi32_si128(ld32(fo[0])), ld32(fo[1]), 1), ld32(fo[2]), 2), ld32(fo[3]), 3);
    if (needZ) // Z24 ReadAt's `v & 0x00FFFFFF`, applied to the four raw dwords at once
        zbuf = _mm_and_si128(_mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(
                                 _mm_cvtsi32_si128(ld32(zo[0])), ld32(zo[1]), 1), ld32(zo[2]), 2), ld32(zo[3]), 3),
                             z24mask);
    }

    // ---- colour into rb / ga 16-bit lanes (PCSX2's layout) ---------------------------------
    const __m128i mask00FF = _mm_set1_epi32(0x00FF00FF);
    const __m128i zero = _mm_setzero_si128(), c255 = _mm_set1_epi16(255);
    __m128i srb = _mm_and_si128(colour, mask00FF);                    // [r0,b0, r1,b1, ...]
    __m128i sga = _mm_and_si128(_mm_srli_epi32(colour, 8), mask00FF); // [g0,a0, g1,a1, ...]

    if constexpr (kFge)
    {
        if (q.fogOn)
        {
            // Scalar: ((fog * input) >> 8) + ((255 - fog) * fogColor >> 8), on r/g/b only.
            const __m128i f16 = _mm_or_si128(fog, _mm_slli_epi32(fog, 16));   // [f,f] per pixel
            const __m128i inv = _mm_sub_epi16(_mm_set1_epi16(255), f16);
            srb = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(f16, srb), 8),
                                _mm_srli_epi16(_mm_mullo_epi16(inv, q.fogRB), 8));
            const __m128i gFog = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(f16, sga), 8),
                                               _mm_srli_epi16(_mm_mullo_epi16(inv, q.fogG), 8));
            sga = _mm_blend_epi16(gFog, sga, 0xAA); // odd lanes are alpha -- fog never touches it
        }
    }
    else
        (void)fog;

    // ---- alpha test -> per-lane write masks (q.atst is per draw: one branch per group) -----
    const __m128i all = _mm_set1_epi32(-1);
    const __m128i alpha = _mm_and_si128(_mm_srli_epi32(colour, 24), _mm_set1_epi32(0xFF));
    const __m128i arefv = q.arefv;
    __m128i pass = all;
    if (__builtin_expect(q.atst != 1, 0)) // cont.324b: ATST=ALWAYS (the test off) skips the jump table
    switch (q.atst)
    {
    case 0: pass = zero; break;
    case 1: break;
    case 2: pass = _mm_cmplt_epi32(alpha, arefv); break;
    case 3: pass = _mm_or_si128(_mm_cmplt_epi32(alpha, arefv), _mm_cmpeq_epi32(alpha, arefv)); break;
    case 4: pass = _mm_cmpeq_epi32(alpha, arefv); break;
    case 5: pass = _mm_or_si128(_mm_cmpgt_epi32(alpha, arefv), _mm_cmpeq_epi32(alpha, arefv)); break;
    case 6: pass = _mm_cmpgt_epi32(alpha, arefv); break;
    case 7: pass = _mm_xor_si128(_mm_cmpeq_epi32(alpha, arefv), all); break;
    default: break;
    }
    // failMask.writeX ? all : pass  ==  pass | (writeX ? all : 0)
    const __m128i wRgb   = _mm_or_si128(pass, q.failRgb);
    const __m128i wAlpha = _mm_or_si128(pass, q.failAlpha);
    const __m128i wDepth = _mm_or_si128(pass, q.failDepth);

    // ---- z test: ZTST=GEQUAL, guaranteed by pipeClassFor. The scalar compare is UNSIGNED
    // ((uint32_t)z >= storedZ) and SSE4.1 has no unsigned 32-bit compare, so bias both sides by
    // 0x80000000 and use the signed one -- exact across the whole u32 range.
    const __m128i bias = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i zsrc = _mm_load_si128(reinterpret_cast<const __m128i *>(z));
    __m128i zpass = _mm_set1_epi32(-1);
    if constexpr (kZ == kZGequal)
    {
        const __m128i zs = _mm_xor_si128(zsrc, bias), zd = _mm_xor_si128(zbuf, bias);
        zpass = _mm_or_si128(_mm_cmpgt_epi32(zs, zd), _mm_cmpeq_epi32(zs, zd));
    }
    else if constexpr (kZ == kZGreater)
    {
        // cont.321b ZTST=GREATER: the scalar `case 3` is the unsigned (uint32_t)z > storedZ -- the same bias trick.
        const __m128i zs = _mm_xor_si128(zsrc, bias), zd = _mm_xor_si128(zbuf, bias);
        zpass = _mm_cmpgt_epi32(zs, zd);
    }
    else
        (void)bias; // ZTST=ALWAYS: the scalar sets zpass = true

    // cont.319: the cover mask expanded in registers (bit i set -> lane i all ones), not through
    // four scalar stores and a vector reload.
    __m128i live = _mm_load_si128(&s_laneCoverMask[coverMask & 15]); // cont.324b: one table load
    live = _mm_and_si128(live, _mm_or_si128(_mm_or_si128(wRgb, wAlpha), wDepth)); // writesAnything
    live = _mm_and_si128(live, zpass);

    // ---- blend on r,g,b; alpha keeps the source value ---------------------------------------
    //   kPipeStdAlpha: Cd + ((Cs - Cd) * As >> 7)        (17-bit signed product: mulShift7_epi16)
    //   kPipeAddAlpha: Cd + (Cs * As >> 7)   (cont.317)  (Cs*As <= 65025 fits an unsigned 16-bit
    //                  lane, so mullo + a LOGICAL >>7 is the exact scalar `int * int >> 7`)
    static_assert(kPipe == kPipeStdAlpha || kPipe == kPipeAddAlpha || kPipe == kPipeDarken || kPipe == kPipeOpaque ||
                      kPipe == kPipeSubFix || kPipe == kPipeAddFix,
                  "quad write: unknown pipe class");
    __m128i orb = srb, oga = sga;
    // kPipeOpaque (cont.319b): abe off -- the scalar leaves r,g,b,a as the source; nothing to do.
    if (kPipe != kPipeOpaque && q.blendOn)
    {
        const __m128i drb = _mm_and_si128(dst, mask00FF);
        const __m128i dga = _mm_and_si128(_mm_srli_epi32(dst, 8), mask00FF);
        // As broadcast into both 16-bit lanes of each pixel's pair ([g,a] -> [a,a]).
        const __m128i asrc = _mm_shufflelo_epi16(_mm_shufflehi_epi16(sga, 0xF5), 0xF5);
        if constexpr (kPipe == kPipeStdAlpha)
        {
            orb = _mm_add_epi16(mulShift7_epi16(_mm_sub_epi16(srb, drb), asrc), drb);
            oga = _mm_add_epi16(mulShift7_epi16(_mm_sub_epi16(sga, dga), asrc), dga);
        }
        else if constexpr (kPipe == kPipeDarken)
        {
            // cont.319b ABCD=2101: pickRGB(2)=0, so ((0 - Cd) * As >> 7) + Cd -- the same 17-bit signed
            // product as StdAlpha with Cs = 0 (d in [-255,0], exact in mulShift7_epi16).
            orb = _mm_add_epi16(mulShift7_epi16(_mm_sub_epi16(zero, drb), asrc), drb);
            oga = _mm_add_epi16(mulShift7_epi16(_mm_sub_epi16(zero, dga), asrc), dga);
        }
        else if constexpr (kPipe == kPipeSubFix)
        {
            // cont.321 ABCD=0122: pickRGB(0)=Cs, pickRGB(1)=Cd, cAlpha=FIX, pickRGB(2)=0 -- the scalar's
            // clampU8(((Cs - Cd) * fix >> 7) + 0): the same 17-bit signed product as StdAlpha with FIX
            // for As (d in [-255,255], f in [0,255], exact in mulShift7_epi16), no destination term.
            orb = mulShift7_epi16(_mm_sub_epi16(srb, drb), q.fixv);
            oga = mulShift7_epi16(_mm_sub_epi16(sga, dga), q.fixv);
            (void)asrc;
        }
        else if constexpr (kPipe == kPipeAddFix)
        {
            // cont.321b ABCD=0221: pickRGB(0)=Cs, pickRGB(2)=0, cAlpha=FIX, pickRGB(1)=Cd -- the scalar's
            // clampU8(((Cs - 0) * fix >> 7) + Cd): AddAlpha's unsigned 16-bit product with FIX for As.
            orb = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(srb, q.fixv), 7), drb);
            oga = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(sga, q.fixv), 7), dga);
            (void)asrc;
        }
        else
        {
            orb = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(srb, asrc), 7), drb);
            oga = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(sga, asrc), 7), dga);
        }
        oga = _mm_blend_epi16(oga, sga, 0xAA); // alpha is not blended
        orb = _mm_min_epi16(_mm_max_epi16(orb, zero), c255); // clampU8
        oga = _mm_min_epi16(_mm_max_epi16(oga, zero), c255);
    }

    if (q.fbaSet)
        oga = _mm_or_si128(oga, _mm_and_si128(_mm_set1_epi32(0x00800000), wAlpha)); // a |= 0x80

    __m128i pix = _mm_or_si128(orb, _mm_slli_epi32(oga, 8));

    if (q.maskOn)
    {
        if (q.fbmsk != 0)
            pix = _mm_or_si128(_mm_andnot_si128(q.fbmskv, pix), _mm_and_si128(dst, q.fbmskv));
        // preserveDestinationAlpha = writeRgb && !writeAlpha (fpsm is CT32 under kFast)
        const __m128i keepA = _mm_andnot_si128(wAlpha, wRgb);
        const __m128i merged = _mm_or_si128(_mm_and_si128(pix, _mm_set1_epi32(0x00FFFFFF)),
                                            _mm_and_si128(dst, _mm_set1_epi32(static_cast<int>(0xFF000000u))));
        pix = _mm_blendv_epi8(pix, merged, keepA);
    }

    // ---- stores, masked by blending against what was read ----------------------------------
    // cont.324: four explicit stores (the rolled loop reloaded the VRAM base every iteration).
    // cont.325: the paired form stores the same four dwords as two qwords (lanes 0,1 then 2,3; the
    // four addresses are distinct, so the order among them is immaterial).
    const __m128i outPixV = _mm_blendv_epi8(dst, pix, _mm_and_si128(live, _mm_or_si128(wRgb, wAlpha)));
    if (__builtin_expect(paired, 1))
        st2q(fo0, outPixV);
    else
    {
        alignas(16) uint32_t outPix[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(outPix), outPixV);
        st32(fo[0], outPix[0]);
        st32(fo[1], outPix[1]);
        st32(fo[2], outPix[2]);
        st32(fo[3], outPix[3]);
    }
    // cont.319b: no Z store at all when ZMSK is set. Before, the four lanes were written back with
    // the values just read (a no-op); with kZAlways the read is skipped when ZMSK is set, so a
    // write-back would store zeros -- build 754's hash break (f13400adf72c3265).
    if (!q.zmask)
    {
        // Z24 WriteAt's read-modify-write, (old & 0xFF000000) | (value & 0x00FFFFFF), with `old`
        // re-read AFTER the frame stores exactly as the per-lane WriteAt did (frame and Z may alias).
        const __m128i outZV = _mm_and_si128(_mm_blendv_epi8(zbuf, zsrc, _mm_and_si128(live, wDepth)), z24mask);
        if (__builtin_expect(paired, 1))
        {
            // cont.325: the same read-after-the-frame-stores, the same (old & 0xFF000000) | value per lane,
            // as two qwords each way.
            const __m128i old = ld2q(zo0);
            st2q(zo0, _mm_or_si128(_mm_and_si128(old, q.hi8v), outZV));
        }
        else
        {
        alignas(16) uint32_t outZ[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(outZ), outZV);
        st32(zo[0], (static_cast<uint32_t>(ld32(zo[0])) & 0xFF000000u) | outZ[0]);
        st32(zo[1], (static_cast<uint32_t>(ld32(zo[1])) & 0xFF000000u) | outZ[1]);
        st32(zo[2], (static_cast<uint32_t>(ld32(zo[2])) & 0xFF000000u) | outZ[2]);
        st32(zo[3], (static_cast<uint32_t>(ld32(zo[3])) & 0xFF000000u) | outZ[3]);
        }
    }
}

// ★★★ cont.267 TIER 2, step 1 (PS2X_GS_SIMD4=1): the row walk FOUR PIXELS PER ITERATION.
//
// Why width and nothing else: cont.265 put `perf` on the raster bench and got IPC 3.13 with a
// 0.38% L1 miss rate -- the loop is INSTRUCTION-bound at ~928 instructions per shaded pixel, not
// memory-bound (that label, held since cont.234, is RETRACTED and it is why four sessions of
// memory-shaped levers all measured ~0). cont.266 then disassembled the hot symbol: 6.6% SIMD,
// 93% scalar byte manipulation spread evenly over the body, and cont.266b showed no single site
// is worth more than ~2.6%. An instruction-bound loop at near-peak IPC is exactly where SIMD
// width pays linearly, so the fix is to divide the count: PCSX2's `CDrawScanline` shape.
//
// This step widens the SKELETON (the row walk, the edge test, z, and every interpolant --
// 306 of the 928 instructions per pixel) and keeps the scalar sampler and pixel write in a tail
// loop over the covered lanes. Steps 2 and 3 would pull writePixelT (359) and sampleTextureT (262)
// into the same lanes; the tail is the seam they would replace.
// ⚠ Steps 2 and 3 were NOT built: step 1 measured SLOWER (+0.85..1.39% instructions) while staying
// bit-exact, and the tricensus geometry explains why width cannot pay on this workload. The full
// numbers, the mechanism and the instruction-level attribution are on s_gsSimd4 above -- READ THAT
// BEFORE RE-ATTEMPTING THIS. The code is kept, default OFF, because it is a correct four-wide
// skeleton and the measurement that retires the idea; deleting it would invite a fourth attempt.
//
// ⚠ BIT-EXACTNESS IS THE GATE, not an aspiration. Every lane evaluates the scalar expression
// with the same operands in the same order. This build is -msse4.1 -O2 with no FMA available, so
// nothing can be contracted and per-lane SSE arithmetic is IEEE-identical to the scalar code --
// the same argument cont.231 used for FASTLERP. The oracle is the bench VRAM hash
// (aa9cd72b600021ce); it must not move with the flag on.
template <bool kTme, bool kFst, bool kLinear, bool kIip, bool kFge, int kPipe, int kZ>
void GSCpuBackend::drawTriangleRowsSimd4(const TriRowSetup &rs)
{
    const GSPrimitiveBatch &batch = *rs.batch;
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;
    const PixelPlan &plan = rs.plan;
    const float fx0 = rs.fx0, fy0 = rs.fy0, fx1 = rs.fx1, fy1 = rs.fy1, fx2 = rs.fx2, fy2 = rs.fy2;
    constexpr float kEdgeEpsilon = 1.0e-4f;
    (void)ctx;

    if constexpr (kTme)
    {
        // As in the scalar fast loop: resolveDraw invalidates texMip per draw, so once is enough.
        if (m_draw.texMip != rs.triMip)
            resolveTexMip(state, rs.triMip);
    }

    // ★★★ cont.270: can this draw use the four-wide pixel write? Decided ONCE per primitive.
    // DATE is not implemented in the quad path, and an armed diagnostic (font trace, ablation)
    // must keep the scalar path so its side effects still happen.
    bool quadWrite = false;
    if constexpr (kPipe != kPipeGeneric)
        quadWrite = s_gsQuadWrite && !plan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u;
    // ★★★ cont.271: can this draw use the four-wide sampler prologue? Also once per primitive.
    // resolveTexMip has already run above, so m_draw's per-draw sampler decisions are current.
    bool quadTex = false;
    if constexpr (kTme && !kFst && kLinear)
        quadTex = s_gsQuadTex && m_draw.texTapFast && m_draw.texWrapClass != 2 &&
                  !s_gsTexCensus && (s_gsRasterAbl & 15u) == 0u;
    // ★★★ cont.319: can the shade between them run four-wide too? Once per primitive: it needs
    // both four-wide neighbours, and the MODULATE fast combine (TFX=0) -- combineTexture's other
    // modes keep the scalar tail.
    bool quadShade = false;
    if constexpr (kTme && !kFst && kLinear && kPipe != kPipeGeneric)
        quadShade = s_gsQuadShade && quadTex && quadWrite && s_gsFastCombine && ctx.tex0.tfx == 0;
    // ★ cont.324: the per-draw sampler / write state as caller-locals (QuadTexCtx / QuadWriteCtx above).
    [[maybe_unused]] QuadTexCtx tc{};
    if constexpr (kTme && !kFst && kLinear)
        if (quadTex)
            tc = makeQuadTexCtx();
    [[maybe_unused]] QuadWriteCtx qw{};
    if constexpr (kPipe != kPipeGeneric)
        if (quadWrite)
            qw = makeQuadWriteCtx(plan);
    const bool texHasAlpha = ctx.tex0.tcc != 0u;
    // Flat colour as [r,g,b,a] 16-bit lanes for two pixels (broadcast to both halves below).
    const __m128i flat16 = _mm_setr_epi16(static_cast<short>(v2.r), static_cast<short>(v2.g),
                                          static_cast<short>(v2.b), static_cast<short>(v2.a),
                                          static_cast<short>(v2.r), static_cast<short>(v2.g),
                                          static_cast<short>(v2.b), static_cast<short>(v2.a));
    const __m128i alphaByteMask = _mm_set1_epi32(static_cast<int>(0xFF000000u));
    const __m128i c255w = _mm_set1_epi16(255);
    const __m128i c255i = _mm_set1_epi32(255);
    const __m128i zeroi = _mm_setzero_si128();
    const __m128i intMin = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128d halfd = _mm_set1_pd(0.5);

    // ---- per-primitive broadcasts -------------------------------------------------------------
    const __m128 vFx2 = _mm_set1_ps(fx2);
    const __m128 vA0 = _mm_set1_ps(fy1 - fy2); // the scalar's (fy1 - fy2)
    const __m128 vA1 = _mm_set1_ps(fy2 - fy0); // the scalar's (fy2 - fy0)
    const __m128 vWind = _mm_set1_ps(rs.winding);
    const __m128 vInvD = _mm_set1_ps(rs.invAbsDenom);
    const __m128 vOne = _mm_set1_ps(1.0f);
    const __m128 vHalf = _mm_set1_ps(0.5f);
    const __m128 vNegEps = _mm_set1_ps(-kEdgeEpsilon);
    const __m128i vLane = _mm_setr_epi32(0, 1, 2, 3);
    const __m128d vZ0 = _mm_set1_pd(v0.z), vZ1 = _mm_set1_pd(v1.z), vZ2 = _mm_set1_pd(v2.z);

    // Corner values, one broadcast per COMPONENT: the scalar loop's AoS lanes ([r,g,b,a] and
    // [s,t,q,fog], one pixel per vector) become SoA vectors of four pixels. Unused components are
    // dead code the compiler drops per instantiation.
    const __m128 cR0 = _mm_set1_ps(static_cast<float>(v0.r)), cR1 = _mm_set1_ps(static_cast<float>(v1.r)), cR2 = _mm_set1_ps(static_cast<float>(v2.r));
    const __m128 cG0 = _mm_set1_ps(static_cast<float>(v0.g)), cG1 = _mm_set1_ps(static_cast<float>(v1.g)), cG2 = _mm_set1_ps(static_cast<float>(v2.g));
    const __m128 cB0 = _mm_set1_ps(static_cast<float>(v0.b)), cB1 = _mm_set1_ps(static_cast<float>(v1.b)), cB2 = _mm_set1_ps(static_cast<float>(v2.b));
    const __m128 cA0 = _mm_set1_ps(static_cast<float>(v0.a)), cA1 = _mm_set1_ps(static_cast<float>(v1.a)), cA2 = _mm_set1_ps(static_cast<float>(v2.a));
    const __m128 aS0 = _mm_set1_ps(v0.s), aS1 = _mm_set1_ps(v1.s), aS2 = _mm_set1_ps(v2.s);
    const __m128 aT0 = _mm_set1_ps(v0.t), aT1 = _mm_set1_ps(v1.t), aT2 = _mm_set1_ps(v2.t);
    const __m128 aQ0 = _mm_set1_ps(v0.q), aQ1 = _mm_set1_ps(v1.q), aQ2 = _mm_set1_ps(v2.q);
    const __m128 aU0 = _mm_set1_ps(static_cast<float>(v0.u)), aU1 = _mm_set1_ps(static_cast<float>(v1.u)), aU2 = _mm_set1_ps(static_cast<float>(v2.u));
    const __m128 aV0 = _mm_set1_ps(static_cast<float>(v0.v)), aV1 = _mm_set1_ps(static_cast<float>(v1.v)), aV2 = _mm_set1_ps(static_cast<float>(v2.v));
    const __m128 aF0 = _mm_set1_ps(static_cast<float>(v0.fog)), aF1 = _mm_set1_ps(static_cast<float>(v1.fog)), aF2 = _mm_set1_ps(static_cast<float>(v2.fog));

    // The scalar lerp3, per component instead of per lane: ((p0*w0) + (p1*w1)) + (p2*w2).
    const auto lerp = [](__m128 p0, __m128 p1, __m128 p2, __m128 w0, __m128 w1, __m128 w2) __attribute__((always_inline))
    {
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(p0, w0), _mm_mul_ps(p1, w1)), _mm_mul_ps(p2, w2));
    };
    // The scalar colour cast, one CHANNEL of four pixels at a time. Scalar does packus_epi32 then
    // packus_epi16 on [r,g,b,a]; saturation is per lane, so regrouping by channel is identical.
    const auto packChan = [](__m128i v) __attribute__((always_inline))
    {
        __m128i t = _mm_packus_epi32(v, v);
        t = _mm_packus_epi16(t, t);
        return static_cast<uint32_t>(_mm_cvtsi128_si32(t));
    };

    for (int y = rs.minY; y <= rs.maxY; ++y)
    {
        if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
            continue;
        const float py = static_cast<float>(y) + 0.5f;

        int spanX0 = rs.minX, spanX1 = rs.maxX;
        solveRowSpan(rs, py, spanX0, spanX1);
        if (spanX0 > spanX1)
            continue;
        [[maybe_unused]] QuadRowCtx qr{};
        if constexpr (kPipe != kPipeGeneric)
            if (quadWrite)
                qr = makeQuadRow(qw, y, spanX1); // cont.325: the row's address terms + wrap guard, once per row

        // The y-dependent half of each barycentric numerator. The scalar loop recomputes exactly
        // this product on every pixel of the row; float multiplication is deterministic, so
        // hoisting it to once per row yields the identical value.
        const __m128 vK0 = _mm_set1_ps((fx2 - fx1) * (py - fy2));
        const __m128 vK1 = _mm_set1_ps((fx0 - fx2) * (py - fy2));
        const __m128i vLo = _mm_set1_epi32(spanX0 - 1);
        const __m128i vHi = _mm_set1_epi32(spanX1 + 1);

        // Groups are aligned to 4 in x -- head and tail lanes outside the span are masked, and the
        // alignment is what lets step 2 share one swizzled Address() across a group (the CT32
        // 8-aligned trick cont.231 already uses for uploads). PS2X_GS_SIMD4_ALIGN=0 drops the
        // alignment (start at spanX0) to isolate what the wasted head lanes cost.
        for (int xg = s_gsSimd4Align ? (spanX0 & ~3) : spanX0; xg <= spanX1; xg += 4)
        {
            const __m128i xi = _mm_add_epi32(_mm_set1_epi32(xg), vLane);
            const __m128 px = _mm_add_ps(_mm_cvtepi32_ps(xi), vHalf);
            const __m128 dx = _mm_sub_ps(px, vFx2);
            const __m128 w0 = _mm_mul_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vA0, dx), vK0), vWind), vInvD);
            const __m128 w1 = _mm_mul_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vA1, dx), vK1), vWind), vInvD);
            const __m128 w2 = _mm_sub_ps(_mm_sub_ps(vOne, w0), w1);

            // ⚠ The scalar test is `if (w < -eps) continue`, so a NaN weight DRAWS the pixel (every
            // comparison against NaN is false). Negating the same `<` comparisons reproduces that
            // exactly; a `>=` mask would silently drop NaN lanes and diverge.
            const __m128 ltAny = _mm_or_ps(_mm_or_ps(_mm_cmplt_ps(w0, vNegEps), _mm_cmplt_ps(w1, vNegEps)),
                                           _mm_cmplt_ps(w2, vNegEps));
            const __m128i inSpan = _mm_and_si128(_mm_cmpgt_epi32(xi, vLo), _mm_cmplt_epi32(xi, vHi));
            int cover = _mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(ltAny), inSpan)));
            if (cover == 0)
                continue;

            // z: `v0.z * w0 + v1.z * w1 + v2.z * w2`. GSVertex::z is double, so the float weights
            // promote and the whole chain is double -- kept as double here for the same reason.
            const __m128d w0l = _mm_cvtps_pd(w0), w0h = _mm_cvtps_pd(_mm_movehl_ps(w0, w0));
            const __m128d w1l = _mm_cvtps_pd(w1), w1h = _mm_cvtps_pd(_mm_movehl_ps(w1, w1));
            const __m128d w2l = _mm_cvtps_pd(w2), w2h = _mm_cvtps_pd(_mm_movehl_ps(w2, w2));
            alignas(16) double zl[4];
            const __m128d z01 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(vZ0, w0l), _mm_mul_pd(vZ1, w1l)), _mm_mul_pd(vZ2, w2l));
            const __m128d z23 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(vZ0, w0h), _mm_mul_pd(vZ1, w1h)), _mm_mul_pd(vZ2, w2h));
            _mm_store_pd(zl + 0, z01);
            _mm_store_pd(zl + 2, z23);

            // The colour lerps as int32 lanes per channel; the scalar tail packs them per channel
            // (packChan), the four-wide shade transposes them per pixel. Same values either way.
            [[maybe_unused]] __m128i cRi = zeroi, cGi = zeroi, cBi = zeroi, cAi = zeroi;
            if constexpr (kIip)
            {
                cRi = _mm_cvttps_epi32(lerp(cR0, cR1, cR2, w0, w1, w2));
                cGi = _mm_cvttps_epi32(lerp(cG0, cG1, cG2, w0, w1, w2));
                cBi = _mm_cvttps_epi32(lerp(cB0, cB1, cB2, w0, w1, w2));
                cAi = _mm_cvttps_epi32(lerp(cA0, cA1, cA2, w0, w1, w2));
            }

            alignas(16) float sL[4], tL[4], qL[4], fL[4];
            alignas(16) int32_t uL[4], vL[4];
            if constexpr (kTme && kFst)
            {
                _mm_store_si128(reinterpret_cast<__m128i *>(uL), _mm_cvttps_epi32(lerp(aU0, aU1, aU2, w0, w1, w2)));
                _mm_store_si128(reinterpret_cast<__m128i *>(vL), _mm_cvttps_epi32(lerp(aV0, aV1, aV2, w0, w1, w2)));
            }
            else if constexpr (kTme)
            {
                _mm_store_ps(sL, lerp(aS0, aS1, aS2, w0, w1, w2));
                _mm_store_ps(tL, lerp(aT0, aT1, aT2, w0, w1, w2));
                _mm_store_ps(qL, lerp(aQ0, aQ1, aQ2, w0, w1, w2));
            }
            const __m128 fV = lerp(aF0, aF1, aF2, w0, w1, w2);
            _mm_store_ps(fL, fV);

            // ---- per-lane shading into SoA scratch, then ONE dispatch (cont.270) -----------
            // The colour/fog/z of the covered lanes are gathered here so the pixel write can be
            // taken four at a time with colour resident in 16-bit lanes (writePixelQuad),
            // or one at a time through the unchanged scalar writePixelT when the draw is not
            // eligible. One copy of the shading, two ways to retire it.
            // cont.324b: no zero-init here -- the four-wide shade below stores all four lanes of each (four dead stores per
            // quad otherwise); the scalar tail zeroes them itself before writing its covered lanes.
            alignas(16) uint32_t cq[4];
            alignas(16) int32_t fq[4];
            alignas(16) int32_t zq[4];
            const int coverAll = cover;
            alignas(16) uint32_t texq[4];
            // ★★★ cont.319: the four-wide shade -- texels and colour never leave the lanes.
            if constexpr (kTme && !kFst && kLinear && kPipe != kPipeGeneric)
            {
                if (quadShade)
                {
                    sampleQuadBilinearP4(tc, sL, tL, qL, coverAll, texq);
                    if (s_gsQuadVerify)
                    {
                        for (int m = coverAll; m != 0; m &= m - 1)
                        {
                            const int vi = __builtin_ctz(static_cast<unsigned>(m));
                            const uint32_t ref = sampleTextureT<kFst, kLinear>(
                                state, sL[vi], tL[vi], qL[vi], 0, 0);
                            g_quadVerChecks.fetch_add(1, std::memory_order_relaxed);
                            quadVerifyTally(unsigned(ctx.tex0.psm), ref != texq[vi]);
                            if (ref != texq[vi])
                            {
                                g_quadVerBad.fetch_add(1, std::memory_order_relaxed);
                                if ((s_gsQuadVerifyPsm == 0u || unsigned(ctx.tex0.psm) == s_gsQuadVerifyPsm) &&
                                    g_quadVerShown.fetch_add(1, std::memory_order_relaxed) < 24ull)
                                {
                                    const float iq = (std::fabs(qL[vi]) > 1.0e-8f) ? qL[vi] : 1.0f;
                                    const float uF = sL[vi] / iq * m_draw.texWf;
                                    const float vF = tL[vi] / iq * m_draw.texHf;
                                    std::fprintf(stderr,
                                                 "[gs2:quadverify] MISMATCH lane=%d quad=%08x scalar=%08x | "
                                                 "s=%.6f t=%.6f q=%.6f -> u=%.4f v=%.4f | "
                                                 "tbp=%u tbw=%u psm=0x%02x %dx%d wrapClass=%u maskU=%u maskV=%u "
                                                 "tapFast=%d lin=%d fastP8=%d cover=0x%x\n",
                                                 vi, texq[vi], ref, double(sL[vi]), double(tL[vi]),
                                                 double(qL[vi]), double(uF), double(vF),
                                                 unsigned(m_draw.texTbp), unsigned(m_draw.texTbw),
                                                 unsigned(ctx.tex0.psm), m_draw.texW, m_draw.texH,
                                                 unsigned(m_draw.texWrapClass), unsigned(m_draw.texWrapMaskU),
                                                 unsigned(m_draw.texWrapMaskV), m_draw.texTapFast ? 1 : 0,
                                                 m_draw.texLin ? 1 : 0, m_draw.texFastP8 ? 1 : 0,
                                                 unsigned(coverAll));
                                }
                            }
                        }
                    }
                    const __m128i t8 = _mm_load_si128(reinterpret_cast<const __m128i *>(texq));
                    __m128i v16lo, v16hi;
                    if constexpr (kIip)
                    {
                        // [r0..r3 g0..g3] and [b0..b3 a0..a3] saturated to [0,255] (packus_epi32
                        // clamps to [0,65535], min_epu16 the rest -- the scalar's two packus), then
                        // a 4x4 16-bit transpose into [r g b a] per pixel.
                        const __m128i rg = _mm_min_epu16(_mm_packus_epi32(cRi, cGi), c255w);
                        const __m128i ba = _mm_min_epu16(_mm_packus_epi32(cBi, cAi), c255w);
                        const __m128i rb = _mm_unpacklo_epi16(rg, ba); // r0 b0 r1 b1 r2 b2 r3 b3
                        const __m128i ga = _mm_unpackhi_epi16(rg, ba); // g0 a0 g1 a1 g2 a2 g3 a3
                        v16lo = _mm_unpacklo_epi16(rb, ga);           // r0 g0 b0 a0 r1 g1 b1 a1
                        v16hi = _mm_unpackhi_epi16(rb, ga);           // r2 g2 b2 a2 r3 g3 b3 a3
                    }
                    else
                    {
                        v16lo = flat16;
                        v16hi = flat16;
                    }
                    // MODULATE, combineTexture's fast path on eight lanes at a time.
                    const __m128i t16lo = _mm_cvtepu8_epi16(t8);
                    const __m128i t16hi = _mm_cvtepu8_epi16(_mm_srli_si128(t8, 8));
                    const __m128i mlo = _mm_srli_epi16(_mm_mullo_epi16(t16lo, v16lo), 7);
                    const __m128i mhi = _mm_srli_epi16(_mm_mullo_epi16(t16hi, v16hi), 7);
                    __m128i c8 = _mm_packus_epi16(mlo, mhi);
                    if (!texHasAlpha) // out.a = va when TCC is clear
                        c8 = _mm_blendv_epi8(c8, _mm_packus_epi16(v16lo, v16hi), alphaByteMask);
                    // fog: clampU8(static_cast<int>(f)) per lane.
                    const __m128i f8 = _mm_min_epi32(_mm_max_epi32(_mm_cvttps_epi32(fV), zeroi), c255i);
                    // z: static_cast<u32>(z + 0.5). cvttpd matches the scalar truncation for every
                    // lane inside int32 range; a lane that converts to the indefinite 0x80000000
                    // (out of range or NaN) sends the whole quad through the scalar expression.
                    const __m128i zi = _mm_unpacklo_epi64(_mm_cvttpd_epi32(_mm_add_pd(z01, halfd)),
                                                          _mm_cvttpd_epi32(_mm_add_pd(z23, halfd)));
                    if (__builtin_expect(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(zi, intMin))) != 0, 0))
                    {
                        for (int i = 0; i < 4; ++i)
                            zq[i] = static_cast<int32_t>(static_cast<u32>(zl[i] + 0.5));
                    }
                    else
                        _mm_store_si128(reinterpret_cast<__m128i *>(zq), zi);
                    _mm_store_si128(reinterpret_cast<__m128i *>(cq), c8);
                    _mm_store_si128(reinterpret_cast<__m128i *>(fq), f8);
                    writePixelQuad<kFge, kPipe, kZ>(qw, qr, xg, y, zq, cq, fq, coverAll);
                    continue;
                }
            }
            _mm_store_si128(reinterpret_cast<__m128i *>(cq), zeroi);
            _mm_store_si128(reinterpret_cast<__m128i *>(fq), zeroi);
            _mm_store_si128(reinterpret_cast<__m128i *>(zq), zeroi);
            _mm_store_si128(reinterpret_cast<__m128i *>(texq), zeroi);
            uint32_t pr = 0u, pg = 0u, pb = 0u, pa = 0u;
            if constexpr (kIip)
            {
                pr = packChan(cRi);
                pg = packChan(cGi);
                pb = packChan(cBi);
                pa = packChan(cAi);
            }
            if constexpr (kTme && !kFst && kLinear)
            {
                if (quadTex)
                    sampleQuadBilinearP4(tc, sL, tL, qL, coverAll, texq);
                    // ⚠ cont.332p: ALSO guard on quadTex. The `if (quadTex)` above is UNBRACED, so this
                    // block ran even when the four-wide sampler had NOT, comparing the zero-initialised
                    // texq against a real scalar sample and counting every texel as a mismatch.
                    if (s_gsQuadVerify && quadTex)
                    {
                        for (int m = coverAll; m != 0; m &= m - 1)
                        {
                            const int vi = __builtin_ctz(static_cast<unsigned>(m));
                            const uint32_t ref = sampleTextureT<kFst, kLinear>(
                                state, sL[vi], tL[vi], qL[vi], 0, 0);
                            g_quadVerChecks.fetch_add(1, std::memory_order_relaxed);
                            quadVerifyTally(unsigned(ctx.tex0.psm), ref != texq[vi]);
                            if (ref != texq[vi])
                            {
                                g_quadVerBad.fetch_add(1, std::memory_order_relaxed);
                                if ((s_gsQuadVerifyPsm == 0u || unsigned(ctx.tex0.psm) == s_gsQuadVerifyPsm) &&
                                    g_quadVerShown.fetch_add(1, std::memory_order_relaxed) < 24ull)
                                {
                                    const float iq = (std::fabs(qL[vi]) > 1.0e-8f) ? qL[vi] : 1.0f;
                                    const float uF = sL[vi] / iq * m_draw.texWf;
                                    const float vF = tL[vi] / iq * m_draw.texHf;
                                    std::fprintf(stderr,
                                                 "[gs2:quadverify] MISMATCH lane=%d quad=%08x scalar=%08x | "
                                                 "s=%.6f t=%.6f q=%.6f -> u=%.4f v=%.4f | "
                                                 "tbp=%u tbw=%u psm=0x%02x %dx%d wrapClass=%u maskU=%u maskV=%u "
                                                 "tapFast=%d lin=%d fastP8=%d cover=0x%x\n",
                                                 vi, texq[vi], ref, double(sL[vi]), double(tL[vi]),
                                                 double(qL[vi]), double(uF), double(vF),
                                                 unsigned(m_draw.texTbp), unsigned(m_draw.texTbw),
                                                 unsigned(ctx.tex0.psm), m_draw.texW, m_draw.texH,
                                                 unsigned(m_draw.texWrapClass), unsigned(m_draw.texWrapMaskU),
                                                 unsigned(m_draw.texWrapMaskV), m_draw.texTapFast ? 1 : 0,
                                                 m_draw.texLin ? 1 : 0, m_draw.texFastP8 ? 1 : 0,
                                                 unsigned(coverAll));
                                }
                            }
                        }
                    }
            }
            while (cover)
            {
                const int i = __builtin_ctz(static_cast<unsigned>(cover));
                cover &= cover - 1;
                const int x = xg + i;
                (void)x;

                uint8_t r, g, b, a;
                if constexpr (kIip)
                {
                    r = static_cast<uint8_t>(pr >> (8 * i));
                    g = static_cast<uint8_t>(pg >> (8 * i));
                    b = static_cast<uint8_t>(pb >> (8 * i));
                    a = static_cast<uint8_t>(pa >> (8 * i));
                }
                else
                {
                    r = v2.r;
                    g = v2.g;
                    b = v2.b;
                    a = v2.a;
                }

                if constexpr (kTme)
                {
                    float is = 0.0f, it = 0.0f, iq = 1.0f;
                    uint16_t iu = 0, iv = 0;
                    if constexpr (kFst)
                    {
                        iu = static_cast<uint16_t>(uL[i]);
                        iv = static_cast<uint16_t>(vL[i]);
                    }
                    else
                    {
                        is = sL[i];
                        it = tL[i];
                        iq = qL[i];
                    }

                    uint32_t texel;
                    if constexpr (kTme && !kFst && kLinear)
                        texel = quadTex ? texq[i]
                                        : sampleTextureT<kFst, kLinear>(state, is, it, iq, iu, iv);
                    else
                        texel = sampleTextureT<kFst, kLinear>(state, is, it, iq, iu, iv);
                    const TextureCombineResult color =
                        combineTexture(ctx.tex0, r, g, b, a,
                                       static_cast<uint8_t>(texel & 0xFF),
                                       static_cast<uint8_t>((texel >> 8) & 0xFF),
                                       static_cast<uint8_t>((texel >> 16) & 0xFF),
                                       static_cast<uint8_t>((texel >> 24) & 0xFF));
                    r = color.r;
                    g = color.g;
                    b = color.b;
                    a = color.a;
                }

                const uint8_t fog = clampU8(static_cast<int>(fL[i]));
                cq[i] = pack32(r, g, b, a);
                fq[i] = static_cast<int32_t>(fog);
                zq[i] = static_cast<int32_t>(static_cast<u32>(zl[i] + 0.5));
            }

            if constexpr (kPipe != kPipeGeneric)
            {
                if (quadWrite)
                {
                    writePixelQuad<kFge, kPipe, kZ>(qw, qr, xg, y, zq, cq, fq, coverAll);
                    continue;
                }
            }
            for (int m2 = coverAll; m2 != 0; m2 &= m2 - 1)
            {
                const int i = __builtin_ctz(static_cast<unsigned>(m2));
                const uint32_t c = cq[i];
                // cont.321b: writePixelT's specialised arm assumes ZTST=GEQUAL; a kZ != GEQUAL class retires its
                // scalar tail through the generic (runtime-selector) arm instead.
                writePixelT<true, kFge, (kZ == kZGequal ? kPipe : kPipeGeneric)>(state, plan, xg + i, y, static_cast<u32>(zq[i]),
                                               static_cast<uint8_t>(c), static_cast<uint8_t>(c >> 8),
                                               static_cast<uint8_t>(c >> 16), static_cast<uint8_t>(c >> 24),
                                               static_cast<uint8_t>(fq[i]));
            }
        }
    }
    if constexpr (kTme && !kFst && kLinear)
        m_draw.texTaps += tc.taps; // cont.324: the predictor's input, accumulated per draw (see QuadTexCtx)
}

// ★★★ cont.319b: sprite rows four pixels at a time (see s_gsSprite4). Per lane, DrawSprite's scalar
// expressions with the same operands in the same order:
//   tx = ((float)(x - unclippedX0) + 0.5f) / spriteW;  texUf = u0f + (u1f - u0f) * tx
//   FST:  fixedU = (int)((texUf * 16.0f) + 0.5f) clamped to [0, 0xFFFF] -> sampleTextureT<true, lin>
//   else: sampleTextureT<false, lin>(texUf / texW, texVf / texH, 1.0f) -- or the four-wide prologue
//         with s = texUf / texW, t = texVf / texH, q = 1.0f (1/1.0f == 1.0f exactly, so s*invQ == s)
//   then combineTexture (MODULATE in lanes, as the triangle loop's cont.319 shade; other TFX per lane)
//   and writePixelQuad<false, kPipe, kZAlways> with the sprite's z, colour and fog broadcast.
template <bool kTme, bool kFst, bool kLinear, int kPipe>
void GSCpuBackend::drawSpriteRowsSimd4(const GSDrawState &state, const PixelPlan &plan, const SpriteSetup &sp)
{
    const auto &ctx = state.context;
    const bool texHasAlpha = ctx.tex0.tcc != 0u;
    const bool modulate = s_gsFastCombine && ctx.tex0.tfx == 0;
    const int texW = state.textureWidth, texH = state.textureHeight; // DrawSprite's divisors
    bool quadTex = false;
    if constexpr (kTme && !kFst && kLinear)
        quadTex = s_gsQuadTex && m_draw.texTapFast && m_draw.texWrapClass != 2 && !s_gsTexCensus;
    // ★ cont.324: the per-draw sampler / write state as caller-locals (QuadTexCtx / QuadWriteCtx).
    [[maybe_unused]] QuadTexCtx tc{};
    if constexpr (kTme && !kFst && kLinear)
        if (quadTex)
            tc = makeQuadTexCtx();
    const QuadWriteCtx qw = makeQuadWriteCtx(plan);

    const __m128i vLane = _mm_setr_epi32(0, 1, 2, 3);
    const __m128 vHalf = _mm_set1_ps(0.5f), vSixteen = _mm_set1_ps(16.0f);
    const __m128 vSpriteW = _mm_set1_ps(sp.spriteW);
    const __m128 vU0 = _mm_set1_ps(sp.u0f), vDu = _mm_set1_ps(sp.u1f - sp.u0f);
    const __m128 vTexW = _mm_set1_ps(static_cast<float>(texW));
    const __m128i vLo = _mm_set1_epi32(sp.drawX0 - 1), vHi = _mm_set1_epi32(sp.drawX1 + 1);
    const __m128i vUncX0 = _mm_set1_epi32(sp.unclippedX0);
    const __m128i zeroi = _mm_setzero_si128(), c65535 = _mm_set1_epi32(0xFFFF);
    const __m128i flat16 = _mm_setr_epi16(static_cast<short>(sp.r), static_cast<short>(sp.g), static_cast<short>(sp.b), static_cast<short>(sp.a),
                                          static_cast<short>(sp.r), static_cast<short>(sp.g), static_cast<short>(sp.b), static_cast<short>(sp.a));
    const __m128i flat8 = _mm_set1_epi32(static_cast<int>(pack32(sp.r, sp.g, sp.b, sp.a)));
    const __m128i alphaByteMask = _mm_set1_epi32(static_cast<int>(0xFF000000u));
    alignas(16) int32_t zq[4] = {static_cast<int32_t>(sp.z1), static_cast<int32_t>(sp.z1), static_cast<int32_t>(sp.z1), static_cast<int32_t>(sp.z1)};
    alignas(16) int32_t fq[4] = {sp.fog, sp.fog, sp.fog, sp.fog};
    (void)texH; (void)vSixteen; (void)c65535; (void)vTexW; (void)zeroi; (void)vUncX0; (void)vHalf; (void)vU0; (void)vDu; (void)vSpriteW;

    for (int y = sp.bandY0; y <= sp.bandY1; ++y)
    {
        if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
            continue;
        const QuadRowCtx qr = makeQuadRow(qw, y, sp.drawX1); // cont.325
        [[maybe_unused]] uint16_t sampleV = 0;
        [[maybe_unused]] float tRow = 0.0f;
        if constexpr (kTme)
        {
            const float ty = (static_cast<float>(y - sp.unclippedY0) + 0.5f) / sp.spriteH;
            const float texVf = sp.v0f + (sp.v1f - sp.v0f) * ty;
            if constexpr (kFst)
            {
                const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
            }
            else
                tRow = texVf / static_cast<float>(texH);
        }
        for (int xg = sp.drawX0 & ~3; xg <= sp.drawX1; xg += 4)
        {
            const __m128i xi = _mm_add_epi32(_mm_set1_epi32(xg), vLane);
            const int cover = _mm_movemask_ps(_mm_castsi128_ps(_mm_and_si128(_mm_cmpgt_epi32(xi, vLo), _mm_cmplt_epi32(xi, vHi))));
            if (cover == 0)
                continue;
            alignas(16) uint32_t cq[4] = {0u, 0u, 0u, 0u};
            if constexpr (kTme)
            {
                const __m128 tx = _mm_div_ps(_mm_add_ps(_mm_cvtepi32_ps(_mm_sub_epi32(xi, vUncX0)), vHalf), vSpriteW);
                const __m128 texUf = _mm_add_ps(vU0, _mm_mul_ps(vDu, tx));
                alignas(16) uint32_t texq[4] = {0u, 0u, 0u, 0u};
                if constexpr (kFst)
                {
                    const __m128i fixedU = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(texUf, vSixteen), vHalf));
                    alignas(16) int32_t su[4];
                    _mm_store_si128(reinterpret_cast<__m128i *>(su), _mm_min_epi32(_mm_max_epi32(fixedU, zeroi), c65535));
                    for (int m = cover; m != 0; m &= m - 1)
                    {
                        const int i = __builtin_ctz(static_cast<unsigned>(m));
                        texq[i] = sampleTextureT<true, kLinear>(state, 0.0f, 0.0f, 1.0f, static_cast<uint16_t>(su[i]), sampleV);
                    }
                }
                else
                {
                    alignas(16) float sL[4], tL[4], qL[4];
                    _mm_store_ps(sL, _mm_div_ps(texUf, vTexW));
                    _mm_store_ps(tL, _mm_set1_ps(tRow));
                    _mm_store_ps(qL, _mm_set1_ps(1.0f));
                    if (quadTex)
                        sampleQuadBilinearP4(tc, sL, tL, qL, cover, texq);
                    // ⚠ cont.332p: same unbraced-`if (quadTex)` defect as the site above -- see there.
                    if (s_gsQuadVerify && quadTex)
                    {
                        for (int m = cover; m != 0; m &= m - 1)
                        {
                            const int vi = __builtin_ctz(static_cast<unsigned>(m));
                            const uint32_t ref = sampleTextureT<kFst, kLinear>(
                                state, sL[vi], tL[vi], qL[vi], 0, 0);
                            g_quadVerChecks.fetch_add(1, std::memory_order_relaxed);
                            quadVerifyTally(unsigned(ctx.tex0.psm), ref != texq[vi]);
                            if (ref != texq[vi])
                            {
                                g_quadVerBad.fetch_add(1, std::memory_order_relaxed);
                                if ((s_gsQuadVerifyPsm == 0u || unsigned(ctx.tex0.psm) == s_gsQuadVerifyPsm) &&
                                    g_quadVerShown.fetch_add(1, std::memory_order_relaxed) < 24ull)
                                {
                                    const float iq = (std::fabs(qL[vi]) > 1.0e-8f) ? qL[vi] : 1.0f;
                                    const float uF = sL[vi] / iq * m_draw.texWf;
                                    const float vF = tL[vi] / iq * m_draw.texHf;
                                    std::fprintf(stderr,
                                                 "[gs2:quadverify] MISMATCH lane=%d quad=%08x scalar=%08x | "
                                                 "s=%.6f t=%.6f q=%.6f -> u=%.4f v=%.4f | "
                                                 "tbp=%u tbw=%u psm=0x%02x %dx%d wrapClass=%u maskU=%u maskV=%u "
                                                 "tapFast=%d lin=%d fastP8=%d cover=0x%x\n",
                                                 vi, texq[vi], ref, double(sL[vi]), double(tL[vi]),
                                                 double(qL[vi]), double(uF), double(vF),
                                                 unsigned(m_draw.texTbp), unsigned(m_draw.texTbw),
                                                 unsigned(ctx.tex0.psm), m_draw.texW, m_draw.texH,
                                                 unsigned(m_draw.texWrapClass), unsigned(m_draw.texWrapMaskU),
                                                 unsigned(m_draw.texWrapMaskV), m_draw.texTapFast ? 1 : 0,
                                                 m_draw.texLin ? 1 : 0, m_draw.texFastP8 ? 1 : 0,
                                                 unsigned(cover));
                                }
                            }
                        }
                    }
                    else
                        for (int m = cover; m != 0; m &= m - 1)
                        {
                            const int i = __builtin_ctz(static_cast<unsigned>(m));
                            texq[i] = sampleTextureT<false, kLinear>(state, sL[i], tRow, 1.0f, 0u, 0u);
                        }
                }
                if (modulate)
                {
                    const __m128i t8 = _mm_load_si128(reinterpret_cast<const __m128i *>(texq));
                    const __m128i t16lo = _mm_cvtepu8_epi16(t8);
                    const __m128i t16hi = _mm_cvtepu8_epi16(_mm_srli_si128(t8, 8));
                    const __m128i mlo = _mm_srli_epi16(_mm_mullo_epi16(t16lo, flat16), 7);
                    const __m128i mhi = _mm_srli_epi16(_mm_mullo_epi16(t16hi, flat16), 7);
                    __m128i c8 = _mm_packus_epi16(mlo, mhi);
                    if (!texHasAlpha)
                        c8 = _mm_blendv_epi8(c8, flat8, alphaByteMask);
                    _mm_store_si128(reinterpret_cast<__m128i *>(cq), c8);
                }
                else
                {
                    for (int m = cover; m != 0; m &= m - 1)
                    {
                        const int i = __builtin_ctz(static_cast<unsigned>(m));
                        const uint32_t texel = texq[i];
                        const TextureCombineResult c = combineTexture(ctx.tex0, sp.r, sp.g, sp.b, sp.a,
                                                                      static_cast<uint8_t>(texel & 0xFF), static_cast<uint8_t>((texel >> 8) & 0xFF),
                                                                      static_cast<uint8_t>((texel >> 16) & 0xFF), static_cast<uint8_t>((texel >> 24) & 0xFF));
                        cq[i] = pack32(c.r, c.g, c.b, c.a);
                    }
                }
            }
            else
                _mm_store_si128(reinterpret_cast<__m128i *>(cq), flat8);
            writePixelQuad<false, kPipe, kZAlways>(qw, qr, xg, y, zq, cq, fq, cover);
        }
    }
    if constexpr (kTme && !kFst && kLinear)
        m_draw.texTaps += tc.taps; // cont.324: see QuadTexCtx
}


bool GSCpuBackend::drawTriangleFastDispatch(const TriRowSetup &rs)
{
    const GSDrawState &state = rs.batch->state;
    const bool tme = state.prim.tme;
    const bool fst = tme && state.prim.fst;
    const bool lin = tme && state.linearFilter;
    const unsigned key = (tme ? 16u : 0u) | (fst ? 8u : 0u) | (lin ? 4u : 0u) |
                         (state.prim.iip ? 2u : 0u) | (state.prim.fge ? 1u : 0u);

    // cont.317 PS2X_GS_KEYCENSUS: charge this triangle's ticks to its (key, pipe class) slot on
    // every return path. Pipe class is resolved here only when the census is on.
    struct KeyCensusScope
    {
        int idx; unsigned long long t0; uint64_t sig;
        ~KeyCensusScope()
        {
            if (idx < 0) return;
            const unsigned long long dt = __builtin_ia32_rdtsc() - t0;
            g_gsKeyTicks[idx].fetch_add(dt, std::memory_order_relaxed);
            if (sig != 0ull)
            {
                std::lock_guard<std::mutex> lock(g_gsKeySigMutex);
                auto &e = g_gsKeySig[sig];
                e.first += dt;
                ++e.second;
            }
        }
    } keyCensus__{-1, 0ull, 0ull};
    if (s_gsKeyCensus)
    {
        const uint8_t psm = static_cast<uint8_t>(state.context.tex0.psm & 0x3Fu);
        const int texClass = !tme ? 3 : (psm == GS_PSM_T4) ? 0 : (psm == GS_PSM_T8) ? 1
                             : (psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S) ? 2 : 3;
        keyCensus__.idx = static_cast<int>(key) | (pipeClassFor(rs.plan) << 5) | (texClass << 7);
        keyCensus__.t0 = __builtin_ia32_rdtsc();
        g_gsKeyDraws[keyCensus__.idx].fetch_add(1ull, std::memory_order_relaxed);
        if (((keyCensus__.idx >> 5) & 3) == kPipeGeneric)
        {
            const PixelPlan &pp = rs.plan;
            const unsigned fail = (pp.failMask.writeRgb ? 1u : 0u) | (pp.failMask.writeAlpha ? 2u : 0u) | (pp.failMask.writeDepth ? 4u : 0u);
            keyCensus__.sig = uint64_t(key) | (uint64_t(pp.abe) << 5) | (uint64_t(pp.pabe) << 6) |
                              (uint64_t(pp.asel & 3u) << 7) | (uint64_t(pp.bsel & 3u) << 9) | (uint64_t(pp.csel & 3u) << 11) | (uint64_t(pp.dsel & 3u) << 13) |
                              (uint64_t(pp.fix) << 15) | (uint64_t(pp.ztestMethod & 3u) << 23) | (uint64_t(pp.zmask) << 25) | (uint64_t(pp.atEnable) << 26) |
                              (uint64_t(pp.atst & 7u) << 27) | (uint64_t(pp.aref) << 30) | (uint64_t(fail) << 38) | (uint64_t(pp.fbmsk != 0u) << 41) |
                              (uint64_t(pp.needsDateRead) << 42) | (uint64_t(pp.fpsm & 63u) << 43) | (1ull << 62);
        }
        const long long w = static_cast<long long>(rs.maxX) - rs.minX + 1, h = static_cast<long long>(rs.maxY) - rs.minY + 1;
        if (w > 0 && h > 0)
            g_gsKeyArea[keyCensus__.idx].fetch_add(static_cast<unsigned long long>(w * h), std::memory_order_relaxed);
    }

    // ★★★ cont.262 THE SELECTOR SEAM. An EXPLICIT LIST of measured-common (geometry key, pipe
    // class) pairs -- never a cross product, or 5 bool dims x N classes would multiply a binary
    // that is already 900 MB. keys 22/23 = textured, non-FST, bilinear, gouraud, fog off/on:
    // exactly the top three draw states (89.7% of 48.8M draws, cont.261). Everything else falls
    // through to the generic table below, unchanged and bit-exact.
    const int pipeClass = s_gsPipeSpec ? pipeClassFor(rs.plan) : kPipeGeneric;
    if (pipeClass == kPipeStdAlpha)
    {
        // ★★★ cont.267 PS2X_GS_SIMD4 (default OFF): the four-wide row walk, on exactly the keys
        // the selector seam already specialises -- the top three draw states, 89.7% of draws --
        // so the bench hash is a real proof of bit-exactness while the binary grows by two
        // instantiations, not a cross product. Everything else falls through unchanged.
        // ★ cont.270: take the four-wide loop only where the four-wide WRITE can follow it. With
        // PS2X_GS_QUADWRITE on, a draw the quad path cannot serve (DATE, or an armed diagnostic
        // whose side effects must still happen) falls through to the scalar loop rather than to
        // the hybrid, which cont.267 measured at +2.3%. With QUADWRITE=0 the hybrid is taken on
        // purpose -- that is the A/B that documents why the seam has to go.
        const bool quadOk = !s_gsQuadWrite ||
                            (!rs.plan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u);
        if (s_gsSimd4 && quadOk)
        {
            switch (key)
            {
            case 22u: drawTriangleRowsSimd4<true, false, true, true, false, kPipeStdAlpha>(rs); return true;
            case 23u: drawTriangleRowsSimd4<true, false, true, true, true, kPipeStdAlpha>(rs); return true;
            case 20u: if (s_gsSeamKey20) { drawTriangleRowsSimd4<true, false, true, false, false, kPipeStdAlpha>(rs); return true; } break;
            case 2u: if (s_gsSeamKey2) { drawTriangleRowsSimd4<false, false, false, true, false, kPipeStdAlpha>(rs); return true; } break; // cont.321
            default: break;
            }
        }
        switch (key)
        {
        case 22u: drawTriangleRowsFast<true, false, true, true, false, kPipeStdAlpha>(rs); return true;
        case 23u: drawTriangleRowsFast<true, false, true, true, true, kPipeStdAlpha>(rs); return true;
        case 20u: if (s_gsSeamKey20) { drawTriangleRowsFast<true, false, true, false, false, kPipeStdAlpha>(rs); return true; } break;
        case 2u: if (s_gsSeamKey2) { drawTriangleRowsFast<false, false, false, true, false, kPipeStdAlpha>(rs); return true; } break; // cont.321
        default: break;
        }
    }
    // ★★★ cont.317 kPipeAddAlpha: key 20 only (the census saw nothing else on this plan worth a
    // slot). Same shape as above: the four-wide loop where the quad write can follow, else the
    // Fast loop with the blend resolved.
    if (pipeClass == kPipeAddAlpha && s_gsSeamAdd)
    {
        const bool quadOk = !s_gsQuadWrite ||
                            (!rs.plan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u);
        if (s_gsSimd4 && quadOk && key == 20u)
        {
            drawTriangleRowsSimd4<true, false, true, false, false, kPipeAddAlpha>(rs);
            return true;
        }
        if (key == 20u)
        {
            drawTriangleRowsFast<true, false, true, false, false, kPipeAddAlpha>(rs);
            return true;
        }
    }

    // ★ cont.321b kPipeAddFix: key 0 only, the four-wide loop only (no Fast instantiation -- writePixelT's specialised
    // arm assumes GEQUAL); anything the quad write cannot serve falls through to the generic table.
    if (s_gsSeamAddFix && s_gsSimd4 && key == 0u && isAddFixPlan(rs.plan) &&
        (!s_gsQuadWrite || (!rs.plan.needsDateRead && !gs2PerPixelDiag() && (s_gsRasterAbl & 15u) == 0u)))
    {
        drawTriangleRowsSimd4<false, false, false, false, false, kPipeAddFix, kZGreater>(rs);
        return true;
    }

    switch (key)
    {
    case 0u: drawTriangleRowsFast<false, false, false, false, false>(rs); return true;
    case 1u: drawTriangleRowsFast<false, false, false, false, true>(rs); return true;
    case 2u: drawTriangleRowsFast<false, false, false, true, false>(rs); return true;
    case 3u: drawTriangleRowsFast<false, false, false, true, true>(rs); return true;
    case 16u: drawTriangleRowsFast<true, false, false, false, false>(rs); return true;
    case 17u: drawTriangleRowsFast<true, false, false, false, true>(rs); return true;
    case 18u: drawTriangleRowsFast<true, false, false, true, false>(rs); return true;
    case 19u: drawTriangleRowsFast<true, false, false, true, true>(rs); return true;
    case 20u: drawTriangleRowsFast<true, false, true, false, false>(rs); return true;
    case 21u: drawTriangleRowsFast<true, false, true, false, true>(rs); return true;
    case 22u: drawTriangleRowsFast<true, false, true, true, false>(rs); return true;
    case 23u: drawTriangleRowsFast<true, false, true, true, true>(rs); return true;
    case 24u: drawTriangleRowsFast<true, true, false, false, false>(rs); return true;
    case 25u: drawTriangleRowsFast<true, true, false, false, true>(rs); return true;
    case 26u: drawTriangleRowsFast<true, true, false, true, false>(rs); return true;
    case 27u: drawTriangleRowsFast<true, true, false, true, true>(rs); return true;
    case 28u: drawTriangleRowsFast<true, true, true, false, false>(rs); return true;
    case 29u: drawTriangleRowsFast<true, true, true, false, true>(rs); return true;
    case 30u: drawTriangleRowsFast<true, true, true, true, false>(rs); return true;
    case 31u: drawTriangleRowsFast<true, true, true, true, true>(rs); return true;
    default:
        return false;
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch, int bandY0, int bandY1)
{
    const GSDrawState &state = batch.state;
    if (s_skipZTex && state.prim.tme &&
        (state.context.tex0.tbp0 >> 5) == (state.context.zbuf.zbp >> 5))
        return; // cont.329j ablation: this draw samples the Z buffer as a texture
    if (s_skipTgtPage != 0u && state.context.frame.fbp == s_skipTgtPage)
        return; // cont.329j ablation: this draw renders into the ablated target
    if (s_skipTexPage != 0u && state.prim.tme)
    {
        const uint32_t tp = state.context.tex0.tbp0 >> 5;
        if (tp >= s_skipTexPage && tp < s_skipTexPage + s_skipTexPages)
            return; // cont.329k ablation: this draw SAMPLES the ablated page range
    }
    if (s_skipTexTbp != 0u && state.prim.tme && state.context.tex0.tbp0 == s_skipTexTbp)
        return; // cont.330e ablation: this draw SAMPLES the ablated exact TBP
    // cont.231: the pixel-write plan once per primitive (WritePixel built it per pixel).
    PixelPlan wpPlan;
    makePixelPlan(state, wpPlan);
    const bool wpFast = s_gsFastPath && m_draw.fastCt32Z24;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    if (s_gsGeoCensus)
    {
        Gs2GeoCensus &g = g_geoCensus;
        ++g.tris;
        const bool finite = std::isfinite(fx0) && std::isfinite(fy0) && std::isfinite(fx1) &&
                            std::isfinite(fy1) && std::isfinite(fx2) && std::isfinite(fy2);
        if (!finite)
            ++g.nonFinite;
        else
        {
            const double mag = std::max({std::fabs((double)fx0), std::fabs((double)fy0),
                                         std::fabs((double)fx1), std::fabs((double)fy1),
                                         std::fabs((double)fx2), std::fabs((double)fy2)});
            if (mag > g.worstMag) g.worstMag = mag;
            ++g.magBucket[mag < 1e3 ? 0 : mag < 1e4 ? 1 : mag < 1e5 ? 2 : mag < 1e6 ? 3 : 4];
            // Unclipped extent vs the scissor: an unclipped near-plane crossing shows up here as
            // a bbox many times the size of the screen.
            const double bw = (double)maxX - (double)minX;
            const double bh = (double)maxY - (double)minY;
            const double sw = std::max(1.0, (double)ctx.scissor.x1 - (double)ctx.scissor.x0);
            const double sh = std::max(1.0, (double)ctx.scissor.y1 - (double)ctx.scissor.y0);
            const double ratio = std::max(bw / sw, bh / sh);
            if (ratio > 2.0) ++g.overScissor2;
            if (ratio > 8.0) ++g.overScissor8;
            if (ratio > 64.0) ++g.overScissor64;
            const double lo = std::min(bw, bh), hi = std::max(bw, bh);
            if (hi > 256.0 && lo > 0.0 && hi / lo >= 64.0) ++g.thin;
            {
                const double l0 = ((double)v0.r + v0.g + v0.b) / 3.0;
                const double l1 = ((double)v1.r + v1.g + v1.b) / 3.0;
                const double l2 = ((double)v2.r + v2.g + v2.b) / 3.0;
                const double lm = (l0 + l1 + l2) / 3.0;
                g.lumSum += lm;
                ++g.lumN;
                int lb = (int)(lm / 32.0);
                if (lb < 0) lb = 0;
                if (lb > 7) lb = 7;
                ++g.lumBucket[lb];
            }
            if (!state.prim.fst && state.prim.tme)
            {
                const double qm = std::min({std::fabs((double)v0.q), std::fabs((double)v1.q),
                                            std::fabs((double)v2.q)});
                if (qm < 1e-3) ++g.tinyQ;
            }
            if (ratio > 8.0 && g.dumped < 12ull)
            {
                ++g.dumped;
                std::fprintf(stderr,
                             "[gs2:geo] #%llu ratio=%.1f mag=%.4g bbox=[%d,%d]x[%d,%d]"
                             " scis=[%u,%u]x[%u,%u] tme=%u fst=%u iip=%u |"
                             " v0=(%.2f,%.2f,q=%.4g) v1=(%.2f,%.2f,q=%.4g) v2=(%.2f,%.2f,q=%.4g)\n",
                             g.dumped, ratio, mag, minX, maxX, minY, maxY,
                             ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1,
                             state.prim.tme ? 1u : 0u, state.prim.fst ? 1u : 0u,
                             state.prim.iip ? 1u : 0u,
                             fx0, fy0, (double)v0.q, fx1, fy1, (double)v1.q,
                             fx2, fy2, (double)v2.q);
            }
        }
    }

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    if (s_gsBigPrim > 0)
        gs2BigPrimReport("TRI", state, minX, maxX, minY, maxY, v2.r, v2.g, v2.b, v2.a,
                         m_draw.clut, m_draw.clutValid,
                         (double)v0.z, (double)v1.z, (double)v2.z);

    if (s_gsDropBig > 0 && !state.prim.tme && state.prim.abe &&
        v2.r < 16u && v2.g < 16u && v2.b < 16u)
    {
        const double sw = std::max(1.0, (double)ctx.scissor.x1 - (double)ctx.scissor.x0);
        const double sh = std::max(1.0, (double)ctx.scissor.y1 - (double)ctx.scissor.y0);
        const double cov = 100.0 * (((double)maxX - minX) * ((double)maxY - minY)) / (sw * sh);
        if (cov >= (double)s_gsDropBig)
        {
            static unsigned long long dropped = 0;
            if ((++dropped & 4095ull) == 1ull)
                std::fprintf(stderr, "[gs2:dropbig] dropped=%llu (cover=%.1f%%)\n", dropped, cov);
            return;
        }
    }

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    const uint32_t triMip = gs2TriangleMip(batch, denom);

    // ★ Band clip: this thread owns only [bandY0, bandY1], so its pixel writes are disjoint
    // from every other band's and no lock is needed.
    if (bandY0 > minY) minY = bandY0;
    if (bandY1 < maxY) maxY = bandY1;

    if (s_gsTriCensus)
        ++g_triCensus.tris;
        if (g_triCensus.pend) // cont.327: an even row left pending by the previous triangle is a block of its own
        {
            Gs2TriCensus &c = g_triCensus;
            if (c.pendX1 >= c.pendX0) { c.lanes4x2 += 8ull * static_cast<unsigned long long>(((c.pendX1 >> 2) - (c.pendX0 >> 2)) + 1); ++c.pairs; ++c.pairRowsEmptyOne; }
            c.pend = false;
        }

    // ★★★ cont.229: coefficients of the three edge functions as exact affine forms in x.
    // w_i(x) = a_i * (x - ox) + c_i, with c_i depending only on the row. `winding` is exactly
    // +-1.0, so folding it into k is bit-exact -- and note the per-pixel code below still does
    // its own two multiplies; this double copy exists ONLY to bound the span, never to produce a
    // pixel value.
    const double dk = static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double sa0 = static_cast<double>(fy1 - fy2) * dk;
    const double sa1 = static_cast<double>(fy2 - fy0) * dk;
    const double sa2 = -(sa0 + sa1);
    const double sb0 = static_cast<double>(fx2 - fx1) * dk;
    const double sb1 = static_cast<double>(fx0 - fx2) * dk;
    const double sox = static_cast<double>(fx2) - 0.5;
    // ★★ The slack must be DERIVED, not guessed. The float predicate computes
    // E = A*(px-fx2) + B*(py-fy2) and then scales by k; when the triangle is thin, E is a
    // catastrophic cancellation of two large terms and its absolute error is set by the
    // MAGNITUDE OF THOSE TERMS, not by the tiny result. A fixed slack is therefore unsound in
    // exactly the case that matters. Bound it instead by (|A|*umax + |B|*|vy|) * |k| * K, with
    // K = 32 * 2^-24 -- a generous multiple of the unit roundoff for the handful of operations
    // involved. (cont.229: the first cut used a fixed 1e-3 with a 2-pixel margin. It happened to
    // verify clean, but the margin was so wide that a deliberately shrunk span ALSO verified
    // clean -- the mutation test could not fail, which is the tell that a bound is not doing any
    // work. It also cost most of the win: 2 px x 2 sides x 131,631 rows is more waste than the
    // narrowing removed.)
    constexpr double kSpanUlp = 32.0 / 16777216.0;
    constexpr int kSpanMargin = 0; // floor/ceil below already widen to whole pixels
    const double umax = std::max(std::fabs(static_cast<double>(minX) + 0.5 - static_cast<double>(fx2)),
                                 std::fabs(static_cast<double>(maxX) + 0.5 - static_cast<double>(fx2)));
    const double adk = std::fabs(dk);
    const double sA0 = std::fabs(static_cast<double>(fy1 - fy2));
    const double sA1 = std::fabs(static_cast<double>(fy2 - fy0));
    const double sB0 = std::fabs(static_cast<double>(fx2 - fx1));
    const double sB1 = std::fabs(static_cast<double>(fx0 - fx2));
    // The three divisions do NOT depend on y -- hoist them out of the row loop (131,631 rows vs
    // 19,712 triangles in the capture). Multiplying by the reciprocal costs at most an extra ULP
    // on a BOUND, which floor/ceil absorbs whole.
    const double sinv0 = (sa0 != 0.0) ? 1.0 / sa0 : 0.0;
    const double sinv1 = (sa1 != 0.0) ? 1.0 / sa1 : 0.0;
    const double sinv2 = (sa2 != 0.0) ? 1.0 / sa2 : 0.0;
    const bool spanOk = s_gsFastTri && std::isfinite(sa0) && std::isfinite(sa1) &&
                        std::isfinite(sb0) && std::isfinite(sb1) && std::isfinite(sox) &&
                        std::isfinite(umax) && std::isfinite(sinv0) && std::isfinite(sinv1) &&
                        std::isfinite(sinv2);

    // ★★ cont.231 PS2X_GS_FASTPIX: hand the row walk to the specialised loop when it can reproduce
    // the generic one bit-for-bit -- the CT32/Z24 fast write (kFast), the hoisted texture prologue
    // (FASTTEX, which sampleTextureT assumes), the cont.229 span (spanOk), and no loop-body
    // diagnostic armed. Everything else (16-bit / CT24 targets, the census and verify modes, the
    // kill switches) still walks the generic loop below.
    if (s_gsFastPix && spanOk && s_gsFastPath && m_draw.fastCt32Z24 && s_gsFastTex &&
        !s_gsTriCensus && !s_gsSpanVerify && s_gsSpanMutate == 0 && !s_gsCensus2)
    {
        TriRowSetup rs;
        rs.batch = &batch;
        rs.minX = minX; rs.maxX = maxX; rs.minY = minY; rs.maxY = maxY;
        rs.fx0 = fx0; rs.fy0 = fy0; rs.fx1 = fx1; rs.fy1 = fy1; rs.fx2 = fx2; rs.fy2 = fy2;
        rs.winding = winding; rs.invAbsDenom = invAbsDenom; rs.triMip = triMip;
        rs.sa0 = sa0; rs.sa1 = sa1; rs.sa2 = sa2; rs.sb0 = sb0; rs.sb1 = sb1; rs.sox = sox;
        rs.umax = umax; rs.adk = adk; rs.sA0 = sA0; rs.sA1 = sA1; rs.sB0 = sB0; rs.sB1 = sB1;
        rs.sinv0 = sinv0; rs.sinv1 = sinv1; rs.sinv2 = sinv2;
        makePixelPlan(state, rs.plan);
        if (drawTriangleFastDispatch(rs))
            return;
    }

    for (int y = minY; y <= maxY; ++y)
    {
        // ★ cont.210: striped band -- this row belongs to another thread.
        if (s_bandOwnRows && !s_bandOwnRows[static_cast<unsigned>(y) & (kBandRowSpace - 1u)])
            continue;
        float py = static_cast<float>(y) + 0.5f;
        int censusFirst = -1, censusLast = -1, censusCovered = 0;

        // ★★ Solve the row's covered interval. Each constraint a*(x-ox) + c >= -eps' is a
        // half-line; intersect the three, then widen. A row whose intersection is empty is
        // skipped entirely -- the census measured 42.79% of scanlines covering nothing.
        int spanX0 = minX, spanX1 = maxX;
        if (spanOk)
        {
            const double vy = static_cast<double>(py) - static_cast<double>(fy2);
            const double vya = std::fabs(vy);
            const double sc0 = sb0 * vy;
            const double sc1 = sb1 * vy;
            // Per-constraint float-error bound (see kSpanUlp above). w2 is derived from w0 and
            // w1, so its error is theirs plus a rounding of a quantity of order 1.
            const double e0 = kSpanUlp * ((sA0 * umax + sB0 * vya) * adk + 1.0);
            const double e1 = kSpanUlp * ((sA1 * umax + sB1 * vya) * adk + 1.0);
            const double ee[3] = {e0, e1, e0 + e1 + kSpanUlp};
            const double aa[3] = {sa0, sa1, sa2};
            const double ii[3] = {sinv0, sinv1, sinv2};
            const double cc[3] = {sc0, sc1, 1.0 - sc0 - sc1};
            double lo = static_cast<double>(minX), hi = static_cast<double>(maxX);
            bool empty = false;
            for (int i = 0; i < 3; ++i)
            {
                const double relaxed = static_cast<double>(kEdgeEpsilon) + ee[i];
                const double lim = -relaxed - cc[i];
                if (aa[i] > 0.0)
                {
                    const double b = sox + lim * ii[i];
                    if (b > lo) lo = b;
                }
                else if (aa[i] < 0.0)
                {
                    const double b = sox + lim * ii[i];
                    if (b < hi) hi = b;
                }
                else if (cc[i] < -relaxed)
                {
                    empty = true;
                    break;
                }
            }
            if (empty)
            {
                spanX0 = minX + 1;
                spanX1 = minX; // provably no coverage on this row
            }
            else if (std::isfinite(lo) && std::isfinite(hi))
            {
                const double flo = std::floor(lo) - kSpanMargin + s_gsSpanMutate;
                const double fhi = std::ceil(hi) + kSpanMargin - s_gsSpanMutate;
                if (flo > static_cast<double>(spanX0))
                    spanX0 = (flo > static_cast<double>(spanX1)) ? spanX1 + 1
                                                                 : static_cast<int>(flo);
                if (fhi < static_cast<double>(spanX1))
                    spanX1 = (fhi < static_cast<double>(spanX0)) ? spanX0 - 1
                                                                 : static_cast<int>(fhi);
            }
        }
        // The verifier walks the ORIGINAL bbox and flags any covered pixel the span excluded,
        // which is exactly the property that makes the narrowing sound.
        const int walkX0 = s_gsSpanVerify ? minX : spanX0;
        const int walkX1 = s_gsSpanVerify ? maxX : spanX1;
        if (s_gsSpanVerify)
        {
            ++g_spanVerify.rows;
            g_spanVerify.skipped += static_cast<unsigned>((spanX0 - minX) + (maxX - spanX1));
        }

        if (s_gsTriCensus)
        {
            ++g_triCensus.rows;
            g_triCensus.bboxW += static_cast<unsigned>(maxX - minX + 1);
            g_triCensus.walked += static_cast<unsigned>(walkX1 >= walkX0 ? walkX1 - walkX0 + 1 : 0);
            { // cont.327: 4x1 and 4x2 lane counts (see Gs2TriCensus)
                Gs2TriCensus &c = g_triCensus;
                const bool has = walkX1 >= walkX0;
                if (has)
                    c.lanes4x1 += 4ull * static_cast<unsigned long long>(((walkX1 >> 2) - (walkX0 >> 2)) + 1);
                if (c.pend && c.pendY == y - 1 && (y & 1))
                {
                    // the odd row pairs with the pending even row: the block covers the union of the two spans
                    const int u0 = c.pendX1 >= c.pendX0 ? (has ? std::min(c.pendX0, walkX0) : c.pendX0) : walkX0;
                    const int u1 = c.pendX1 >= c.pendX0 ? (has ? std::max(c.pendX1, walkX1) : c.pendX1) : walkX1;
                    if (u1 >= u0)
                        c.lanes4x2 += 8ull * static_cast<unsigned long long>(((u1 >> 2) - (u0 >> 2)) + 1);
                    if ((c.pendX1 < c.pendX0) != !has) ++c.pairRowsEmptyOne;
                    ++c.pairs;
                    c.pend = false;
                }
                else
                {
                    if (c.pend && c.pendX1 >= c.pendX0) // an even row left alone (the triangle ended, or a skipped row)
                    {
                        c.lanes4x2 += 8ull * static_cast<unsigned long long>(((c.pendX1 >> 2) - (c.pendX0 >> 2)) + 1);
                        ++c.pairs; ++c.pairRowsEmptyOne;
                    }
                    if (y & 1)
                    { // an odd row with no even partner: a block of its own
                        if (has) { c.lanes4x2 += 8ull * static_cast<unsigned long long>(((walkX1 >> 2) - (walkX0 >> 2)) + 1); ++c.pairs; ++c.pairRowsEmptyOne; }
                        c.pend = false;
                    }
                    else
                    {
                        c.pend = true; c.pendY = y; c.pendX0 = walkX0; c.pendX1 = walkX1;
                    }
                }
            }
        }
        for (int x = walkX0; x <= walkX1; ++x)
        {
            if (s_gsTriCensus)
                ++g_triCensus.tested;
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            if (s_gsTriCensus)
            {
                ++g_triCensus.covered;
                ++censusCovered;
                if (censusFirst < 0) censusFirst = x;
                censusLast = x;
            }

            if (s_gsSpanVerify && (x < spanX0 || x > spanX1))
            {
                if (g_spanVerify.escaped++ < 8ull)
                    std::fprintf(stderr,
                                 "[gs2:spanverify] ESCAPED x=%d y=%d span=[%d,%d] bbox=[%d,%d]"
                                 " w=%.9g/%.9g/%.9g\n",
                                 x, y, spanX0, spanX1, minX, maxX,
                                 static_cast<double>(w0), static_cast<double>(w1),
                                 static_cast<double>(w2));
            }

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(state, is, it, iq, iu, iv, triMip);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;

                // Cyan-source probe (part of PS2X_GS_CENSUS2): the level scene renders large
                // pure-cyan masses — log which INPUT carries the cyan for the first hits.
                if (s_gsCensus2 && r < 32u && g > 224u && b > 224u)
                {
                    static unsigned long s_cyan = 0;
                    ++s_cyan;
                    if (s_cyan <= 24u || (s_cyan % 2000000u) == 0u)
                        std::fprintf(stderr,
                                     "[gs2:cyan] n=%lu TEXTURED texel=%02x/%02x/%02x/%02x vtx=%02x/%02x/%02x/%02x "
                                     "tex0(psm=0x%x tbp0=%u tfx=%u tcc=%u tw=%u) iip=%u fst=%u\n",
                                     s_cyan, tr, tg, tb, ta, shadeR, shadeG, shadeB, shadeA,
                                     ctx.tex0.psm, ctx.tex0.tbp0, ctx.tex0.tfx, ctx.tex0.tcc,
                                     ctx.tex0.tw, state.prim.iip, state.prim.fst);
                }
            }
            else if (s_gsCensus2 && r < 32u && g > 224u && b > 224u)
            {
                static unsigned long s_cyanFlat = 0;
                ++s_cyanFlat;
                if (s_cyanFlat <= 24u || (s_cyanFlat % 2000000u) == 0u)
                    std::fprintf(stderr,
                                 "[gs2:cyan] n=%lu UNTEXTURED vtx=%02x/%02x/%02x/%02x iip=%u abe=%u v0rgba=%02x/%02x/%02x/%02x\n",
                                 s_cyanFlat, r, g, b, a, state.prim.iip, state.prim.abe,
                                 static_cast<unsigned>(v0.r), static_cast<unsigned>(v0.g),
                                 static_cast<unsigned>(v0.b), static_cast<unsigned>(v0.a));
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            writePixelDispatch(state, wpPlan, wpFast, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
        if (s_gsTriCensus)
        {
            if (censusCovered == 0)
                ++g_triCensus.emptyRows;
            else
            {
                // ★ relative to the WALK, not the bbox: with the span active the two would
                // otherwise double-count the pixels the span already skipped (cont.229 first
                // reading showed lead+trail = 131% of tested, which is the tell).
                g_triCensus.lead += static_cast<unsigned>(censusFirst - walkX0);
                g_triCensus.trail += static_cast<unsigned>(walkX1 - censusLast);
            }
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch, int bandY0, int bandY1)
{
    const GSDrawState &state = batch.state;
    // cont.231: the pixel-write plan once per primitive (WritePixel built it per pixel).
    PixelPlan wpPlan;
    makePixelPlan(state, wpPlan);
    const bool wpFast = s_gsFastPath && m_draw.fastCt32Z24;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    // cont.318: charge lines to the key census too (slot type 2; area = steps + 1).
    struct LineCensusScope
    {
        int idx; unsigned long long t0;
        ~LineCensusScope()
        {
            if (idx >= 0) g_gsKeyTicks[idx].fetch_add(__builtin_ia32_rdtsc() - t0, std::memory_order_relaxed);
        }
    } lineCensus__{-1, 0ull};
    if (s_gsKeyCensus)
    {
        const GSDrawState &st = batch.state;
        const unsigned key = (st.prim.tme ? 16u : 0u) | (st.prim.iip ? 2u : 0u) | (st.prim.fge ? 1u : 0u);
        lineCensus__.idx = static_cast<int>(key) | (3 << 7) | (2 << 9);
        lineCensus__.t0 = __builtin_ia32_rdtsc();
        g_gsKeyDraws[lineCensus__.idx].fetch_add(1ull, std::memory_order_relaxed);
        g_gsKeyArea[lineCensus__.idx].fetch_add(static_cast<unsigned long long>(totalSteps + 1), std::memory_order_relaxed);
    }
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        // ★ band clip (cont.207) + cont.210 striped row ownership.
        if (y0 >= bandY0 && y0 <= bandY1 &&
            (!s_bandOwnRows || s_bandOwnRows[static_cast<unsigned>(y0) & (kBandRowSpace - 1u)]))
            writePixelDispatch(state, wpPlan, wpFast, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    if (s_gsThread)
    {
        WorkItem item;
        item.kind = WorkItem::Kind::Transfer;
        item.transfer = command;
        enqueueWork(std::move(item));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    BeginTransferUnlocked(command);
}

void GSCpuBackend::BeginTransferUnlocked(const GSTransferCommand &command)
{
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    // PS2X_GS_FONTTRACE (env, default OFF): trace every transfer targeting the font-CLUT blocks
    // (LOTR menu text RE, cont.165 -- the CLUT RGB bytes arrive in VRAM zeroed, alpha intact).
    if (g_gs2FontTrace && command.bitbltbuf.dbp >= 12288u && command.bitbltbuf.dbp <= 12290u)
    {
        static unsigned long s_n = 0;
        if (++s_n <= 24u)
            std::fprintf(stderr,
                         "[gs2:fontxfer] #%lu dir=%u blt{sbp=%u sbw=%u spsm=0x%x dbp=%u dbw=%u dpsm=0x%x} "
                         "pos{ssax=%u ssay=%u dsax=%u dsay=%u} reg{%ux%u}\n",
                         s_n, command.direction, command.bitbltbuf.sbp, command.bitbltbuf.sbw,
                         command.bitbltbuf.spsm, command.bitbltbuf.dbp, command.bitbltbuf.dbw,
                         command.bitbltbuf.dpsm, command.trxpos.ssax, command.trxpos.ssay,
                         command.trxpos.dsax, command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh);
    }

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (s_gsThread)
    {
        if (!data || sizeBytes == 0u)
            return;
        WorkItem item;
        item.kind = WorkItem::Kind::Upload;
        item.bytes.assign(data, data + sizeBytes);
        enqueueWork(std::move(item));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    UploadImageUnlocked(data, sizeBytes);
}

void GSCpuBackend::UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes)
{
    struct UploadTimer
    {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        uint32_t before;
        const GSTransferSnapshot &st;
        explicit UploadTimer(const GSTransferSnapshot &s) : before(s.copiedPixels), st(s) {}
        ~UploadTimer()
        {
            // copiedPixels resets to 0 when a transfer completes; count the chunk either way
            const uint32_t after = st.copiedPixels;
            g_gs2UploadPixels.fetch_add(after >= before ? after - before : after, std::memory_order_relaxed);
            g_gs2UploadNs.fetch_add(static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                std::memory_order_relaxed);
        }
    } uploadTimer(m_transferState);
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); // cont.227b: invalidate cached CLUTs
    // cont.320: g_gs2PageGen is bumped over the chunk's EXACT pages below (it was whole-VRAM here and
    // again per pixel in WriteVramUnlocked -- 512 atomics per pixel on the generic path).
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    ++g_vramMutationSeq;
    // A batch is rasterized at flush time, so it has to reach the device BEFORE this
    // mutation's own mirror op does -- otherwise the GPU sees the upload ahead of draws the
    // oracle already applied. Flush eagerly here, not lazily at the next primitive.
    if (s_gsGpuBatchVerify > 0 && !m_gpuBatch.prims.empty())
    { ++g_bvFlushExplicit; gpuBatchFlush(); }
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    // cont.357c PS2X_GS_UPCENSUS (read-only): accumulate this transfer's payload hash. A transfer
    // arrives as one or more chunks and `copiedPixels == 0` is the runtime's own "first chunk of a
    // new transfer" test (the same one the updep census uses).
    if (s_upCensus)
        upCensusChunk(data, sizeBytes, m_transferState.copiedPixels == 0u,
                      (uint32_t)((uint64_t)m_transferState.totalPixels *
                                 std::max<uint32_t>(bitsPerPixel(dpsm), 4u) / 8u),
                      dbp, dbw, dpsm, dsax, m_transfer.trxpos.dsay, rrw, m_transfer.trxreg.rrh,
                      g_perfFlips.load(std::memory_order_relaxed));
    // cont.320: this chunk's footprint (its rows, the transfer's full column span -- the updep
    // geometry, page granularity absorbs a mid-row start), for the production g_gs2PageGen (the
    // texture cache's validity) and, under the census, its own copies.
    const uint32_t chunkBpp = bitsPerPixel(dpsm);
    const uint32_t chunkPix = chunkBpp ? sizeBytes * 8u / chunkBpp : sizeBytes / 4u;
    const uint32_t rowsHere = (chunkPix + rrw - 1u) / rrw + 1u;
    {
        Gs2PageMask um;
        gs2RectPages(um, dbp / 32u, dbw, dpsm, dsax, m_transferState.y, dsax + rrw - 1u, m_transferState.y + rowsHere);
        for (int wi = 0; wi < 8; ++wi) { uint64_t b = um.w[wi]; while (b) { const int i = __builtin_ctzll(b); b &= b - 1; gs2BumpPageRange(static_cast<uint32_t>(wi * 64 + i), static_cast<uint32_t>(wi * 64 + i)); } }
        if (s_gsTexCensus)
            gs2TexCensusBumpMask(um);
        // ★★★ cont.330: the same exact pages on the transfer-only counter (see g_gs2PageXferGen).
        for (int wi = 0; wi < 8; ++wi)
        {
            uint64_t bx = um.w[wi];
            while (bx)
            {
                const uint32_t pg = static_cast<uint32_t>(wi * 64 + __builtin_ctzll(bx));
                bx &= bx - 1;
                gs2BumpXferPage(pg);
            }
        }
        g_gs2XferSeq.fetch_add(1u, std::memory_order_relaxed);
        if (s_glTexTrace != 0u || s_mutCensus || s_pageWatch) // cont.331j: PAGEWATCH needs these sites too
        {
            uint32_t lo = 0xFFFFFFFFu, hi = 0u;
            for (int wi = 0; wi < 8; ++wi)
            {
                uint64_t b = um.w[wi];
                while (b)
                {
                    const uint32_t pg = static_cast<uint32_t>(wi * 64 + __builtin_ctzll(b));
                    b &= b - 1;
                    lo = std::min(lo, pg);
                    hi = std::max(hi, pg);
                }
            }
            if (lo != 0xFFFFFFFFu)
                glTexTraceMutation("upload(host->local)", lo, hi, dpsm, dbw, dsax,
                                   m_transferState.y, dsax + rrw - 1u,
                                   m_transferState.y + rowsHere, true);
        }
    }
    if (s_gsTexCensus)
    {
        ++g_texCensus.upChunks;
        if (m_transferState.copiedPixels == 0u) ++g_texCensus.upXfers;
        Gs2BlkMask bm;
        gs2RectBlocks(bm, dpsm, dbp, dbw, dsax, m_transferState.y, dsax + rrw - 1u, m_transferState.y + rowsHere);
        gs2TexCensusBumpBlk(bm);
    }
    uint32_t offset = 0u;

    // CLUT shadow capture (PS2X_GS_CLUTSHADOW, default ON): remember every 16-entry CT32 CLUT
    // upload so sampling can survive the render-target stomp that zeroes its RGB lanes between
    // upload and use (see the member comment in gs_cpu_backend.h; LOTR menu text, cont.165).
    if (s_gsClutShadow && dpsm == GS_PSM_CT32 && m_transfer.trxreg.rrw == 8u &&
        m_transfer.trxreg.rrh == 2u && sizeBytes >= 64u && m_clutShadow.size() < 64u)
    {
        std::array<uint32_t, 16> pay{};
        std::memcpy(pay.data(), data, 64u);
        m_clutShadow[dbp] = pay;
    }

    // PS2X_GS_FONTTRACE: payload arriving for the font-CLUT blocks -- is the RGB already gone here?
    if (g_gs2FontTrace && dbp >= 12288u && dbp <= 12290u)
    {
        static unsigned long s_n = 0;
        if (++s_n <= 24u)
        {
            uint32_t w[4] = {0, 0, 0, 0};
            std::memcpy(w, data, std::min<uint32_t>(sizeBytes, 16u));
            std::fprintf(stderr,
                         "[gs2:fontup] #%lu dbp=%u dbw=%u dpsm=0x%x %ux%u bytes=%u data[0..3]=%08x %08x %08x %08x\n",
                         s_n, dbp, dbw, dpsm, m_transfer.trxreg.rrw, m_transfer.trxreg.rrh,
                         sizeBytes, w[0], w[1], w[2], w[3]);
        }
    }

    if (s_gsCensus2)
    {
        ++g_gs2Xfer.n;
        g_gs2Xfer.bytes += sizeBytes;
        switch (dpsm)
        {
        case GS_PSM_T4: ++g_gs2Xfer.t4; break;
        case GS_PSM_T8: ++g_gs2Xfer.t8; break;
        case GS_PSM_CT32: ++g_gs2Xfer.ct32; break;
        case GS_PSM_CT16: case GS_PSM_CT16S: ++g_gs2Xfer.ct16; break;
        default: ++g_gs2Xfer.other; break;
        }
        if (m_transfer.trxreg.rrw > 1024u || m_transfer.trxreg.rrh > 1024u)
            ++g_gs2Xfer.badDims; // the cycle-73 malformed-descriptor discriminator
        if (dbp >= 12288u)
        {
            const bool tile = (m_transfer.trxreg.rrw == 16u && m_transfer.trxreg.rrh == 16u);
            if (tile)
                ++g_gs2Xfer.arenaTile;
            else
            {
                ++g_gs2Xfer.arenaNonTile;
                g_gs2Xfer.lastNtDbp = dbp;
                g_gs2Xfer.lastNtDpsm = dpsm;
                g_gs2Xfer.lastNtRrw = m_transfer.trxreg.rrw;
                g_gs2Xfer.lastNtRrh = m_transfer.trxreg.rrh;
            }
        }
        g_gs2Xfer.lastDbp = dbp;
        g_gs2Xfer.lastDbw = dbw;
        g_gs2Xfer.lastDpsm = dpsm;
        g_gs2Xfer.lastRrw = m_transfer.trxreg.rrw;
        g_gs2Xfer.lastRrh = m_transfer.trxreg.rrh;
        if ((g_gs2Xfer.n % 32768u) == 0u)
            gs2PrintXfer();
    }

#if PS2X_HAS_GS_GPU_DEVICE
    // Phase-1 GPU VRAM mirror (cont.167): post this chunk to the device BEFORE the CPU
    // loop mutates the cursor (the chunk bakes in the pre-loop startPixel; the device
    // executes strictly in post order = the GS worker's serial upload order). The CPU
    // loop writes exactly min(floor(bytes/4), remaining) pixels for CT32 -- partial
    // word tails are dropped, and payload past totalPixels is ignored -- so mirrorCount
    // reproduces it.
    const bool gpuMirrorWanted = gs2MirrorWanted();
    uint32_t mirrorStart = 0, mirrorCount = 0;
    bool mirrorOn = false;
    // cont.168: a chunk the swizzle kernel cannot take is no longer dropped -- it is
    // raw-patched from CPU VRAM after the loop, so record where the cursor started.
    uint32_t rawStartPixel = 0;
    bool rawMirror = false;
    if (gpuMirrorWanted)
    {
        mirrorFlushPokes(); // this upload must land behind any pending poke box
        mirrorOn = dpsm == GS_PSM_CT32 &&
                   m_transfer.trxreg.rrw <= 1024u && m_transfer.trxreg.rrh <= 1024u;
        if (!mirrorOn)
        {
            // A format with no kernel (the T4HL font atlas is this game's only one) or a
            // badDims descriptor (malformed rects can wrap two pixels onto one masked
            // address, where GPU write order != CPU serial order). Both mirror correctly
            // through the raw path, which copies whatever the CPU actually wrote.
            rawMirror = true;
            rawStartPixel = m_transferState.copiedPixels;
            if ((++g_gpuXferSkipped % 4096u) == 1u)
                std::fprintf(stderr, "[gsgpu:xfer] no-kernel=%lu -> raw patch (dpsm=0x%x %ux%u)\n",
                             g_gpuXferSkipped, dpsm, m_transfer.trxreg.rrw, m_transfer.trxreg.rrh);
        }
        else
        {
            mirrorStart = m_transferState.copiedPixels;
            mirrorCount = std::min<uint32_t>(sizeBytes / 4u,
                                             m_transferState.totalPixels - mirrorStart);
            if (mirrorCount != 0u)
            {
                if (GsGpuPresentDevice *dev = gs2GpuDevice())
                {
                    GsGpuUploadChunk c;
                    c.dbp = dbp;
                    c.dbw = dbw;
                    c.dsax = dsax;
                    c.dsay = m_transfer.trxpos.dsay;
                    c.rrw = rrw;
                    c.startPixel = mirrorStart;
                    c.payload.resize(mirrorCount);
                    std::memcpy(c.payload.data(), data, static_cast<size_t>(mirrorCount) * 4u);
                    dev->MirrorUpload(std::move(c));
                }
                else
                    mirrorOn = false; // device unavailable; skip the verify below too
            }
        }
    }
    // Raw-patch on EVERY exit path: the write loop returns early on a short payload tail
    // (`sizeBytes - offset < bpp`), and a chunk that wrote 100 pixels then returned still
    // has to reach the mirror. A destructor is the only thing those returns cannot skip.
    struct RawMirrorGuard
    {
        GSCpuBackend *self;
        const GSTransferSnapshot *state;
        bool armed;
        uint32_t psm, bp, bw, dsax, dsay, rrw, startPixel;
        ~RawMirrorGuard()
        {
            if (!armed || rrw == 0u)
                return;
            const uint32_t endPixel = state->copiedPixels;
            if (endPixel <= startPixel)
                return; // nothing was applied
            // Conservative: whole rows [startPixel/rrw .. (endPixel-1)/rrw], full width.
            self->mirrorPatchRect(psm, bp, bw,
                                  dsax, dsay + startPixel / rrw,
                                  dsax + rrw - 1u, dsay + (endPixel - 1u) / rrw,
                                  MirrorPatchSource::Upload);
        }
    } rawGuard{this, &m_transferState, rawMirror, dpsm, dbp, dbw,
               dsax, m_transfer.trxpos.dsay, rrw, rawStartPixel};
#endif

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    // ★★★★★ cont.357d PS2X_GS_UPDEDUP (default OFF; see the flag block for the full reasoning).
    // Skip a chunk whose bytes VRAM already holds. CT32/Z32 only (that is where cont.357c's 2.4 MB
    // per flip lives) and only when the whole chunk lands in this call, so a stored copy always
    // describes a complete write.
    if (s_upDedup && sizeBytes >= s_upDedupMin && m_vram && m_transferState.direction == 0u &&
        (dpsm == GS_PSM_CT32 || dpsm == GS_PSM_Z32) &&
        m_transferState.copiedPixels + chunkPix <= m_transferState.totalPixels)
    {
        // The pages this chunk covers, and the generation sum over them. Same rect the bump above
        // used, so the two are talking about the same pages.
        Gs2PageMask dm;
        gs2RectPages(dm, dbp / 32u, dbw, dpsm, dsax, m_transferState.y, dsax + rrw - 1u,
                     m_transferState.y + rowsHere);
        uint64_t genSum = 0ull;
        uint32_t nPages = 0u;
        for (int wi = 0; wi < 8; ++wi)
        {
            uint64_t b = dm.w[wi];
            while (b)
            {
                const uint32_t pg = static_cast<uint32_t>(wi * 64 + __builtin_ctzll(b));
                b &= b - 1;
                ++nPages;
                genSum += g_gs2PageGen[pg & (kGs2Pages - 1u)].load(std::memory_order_relaxed);
            }
        }
        uint64_t key = 1469598103934665603ull;
        const uint64_t kparts[9] = {dbp, dbw, dpsm, dsax, m_transfer.trxpos.dsay, rrw,
                                    m_transfer.trxreg.rrh, m_transferState.copiedPixels, sizeBytes};
        for (uint64_t v : kparts) { key ^= v; key *= 1099511628211ull; }

        std::lock_guard<std::mutex> lk(g_upDedupMutex);
        ++g_ddChunks;
        UpDedupEntry *ent = nullptr;
        for (UpDedupEntry &c : g_upDedupEntries)
            if (c.valid && c.key == key) { ent = &c; break; }
        bool already = false;
        if (ent)
        {
            // ★ The generation test, stated as "stored + our own bump": bumps only increase, so any
            // other writer since the store makes this too large and the entry goes unused.
            if (ent->genSum + nPages != genSum)
                ++g_ddGenMiss;
            else if (ent->payload.size() != sizeBytes ||
                     std::memcmp(ent->payload.data(), data, sizeBytes) != 0)
                ++g_ddCmpMiss;
            else
                already = true;
        }
        if (already && s_upDedup >= 2)
        {
            // ★ THE ORACLE (=2): prove the claim instead of acting on it -- read the destination
            // back through the psm's own read function and compare it to the bytes we would have
            // skipped writing. Mirrors cont.231's PS2X_GS_UPLOADVERIFY. Behaviour is unchanged:
            // the write below still runs.
            const ReadVramFunc rf = m_readVramFuncs[dpsm & 0x3Fu];
            uint32_t vx = dsax, vy = m_transferState.y, bad = 0u;
            for (uint32_t i = 0; i < chunkPix; ++i)
            {
                uint32_t want;
                std::memcpy(&want, data + i * 4u, 4u);
                const uint32_t got = rf(m_vram, dbp, dbw, vx, vy);
                if (got != want) { ++bad; if (bad > 8u) break; }
                if (++vx == dsax + rrw) { vx = dsax; ++vy; }
            }
            if (bad) ++g_ddVerifyBad; else ++g_ddVerifyOk;
            already = false; // never skip in verify mode
        }
        if (already)
        {
            ++g_ddHits;
            g_ddBytesSaved += sizeBytes;
            ent->genSum = genSum;
            ent->lastUse = ++g_upDedupClock;
            advancePixel(chunkPix);
            return;
        }
        // Remember this chunk. Stored BEFORE the write rather than after it because `data` is not
        // modified by the write and every path below writes the whole chunk -- and because the
        // function has several exits, where a post-write store would need a guard object to be
        // reached from all of them.
        if (!ent)
        {
            if (g_upDedupEntries.size() >= kUpDedupMax || g_upDedupBytes + sizeBytes > kUpDedupBytesMax)
            {
                // Evict the least recently used entry rather than growing.
                size_t victim = 0;
                for (size_t i = 1; i < g_upDedupEntries.size(); ++i)
                    if (g_upDedupEntries[i].lastUse < g_upDedupEntries[victim].lastUse) victim = i;
                if (!g_upDedupEntries.empty())
                {
                    g_upDedupBytes -= g_upDedupEntries[victim].payload.size();
                    g_upDedupEntries.erase(g_upDedupEntries.begin() + static_cast<long>(victim));
                }
            }
            g_upDedupEntries.emplace_back();
            ent = &g_upDedupEntries.back();
        }
        else
        {
            g_upDedupBytes -= ent->payload.size();
        }
        ent->key = key;
        ent->payload.assign(data, data + sizeBytes);
        ent->genSum = genSum;
        ent->lastUse = ++g_upDedupClock;
        ent->valid = true;
        g_upDedupBytes += sizeBytes;
        ++g_ddStores;
    }

    // ★★★ cont.357d PS2X_GS_UPABL (default OFF, ABLATION ONLY -- see the flag): skip the pixel
    // writes for a large chunk while advancing the transfer state EXACTLY as a real upload would, so
    // the GIF/transfer state machine, the page-generation bumps and every census see an unchanged
    // sequence and only the VRAM writes disappear. Ablating a CONSUMER is safe here -- no producer is
    // starved, the draw count is unchanged, and the only difference is stale texels.
    if (s_upAbl && sizeBytes >= s_upAblMin)
    {
        advancePixel(chunkPix);
        return;
    }

    // ★ cont.231 PS2X_GS_FASTUPLOAD (default ON; "=0" restores the per-pixel generic path): the
    // CT32/Z32 and T4 uploads -- the level's texture streaming, 8.5% of the raster worker's time in
    // the live profile, during which the band threads idle -- call the psm's write function directly
    // (WriteVramUnlocked's per-pixel ATOMIC generation bump and psm switch are gone; the generation
    // was already bumped once at the top of this function) and step x/y incrementally instead of
    // dividing per pixel. Same function, same (x, y, value) per pixel: byte-identical VRAM.
    if (s_gsFastUpload && (dpsm == GS_PSM_CT32 || dpsm == GS_PSM_Z32 || dpsm == GS_PSM_T4) &&
        m_transferState.direction == 0u && m_vram)
    {
        const WriteVramFunc wf = m_writeVramFuncs[dpsm & 0x3Fu];
        const uint32_t total = m_transferState.totalPixels;
        const uint32_t dsay = m_transfer.trxpos.dsay;
        uint32_t copied = m_transferState.copiedPixels;
        uint32_t px = dsax + (copied % rrw);
        uint32_t py = dsay + (copied / rrw);
        auto step = [&]() {
            ++copied;
            if (++px == dsax + rrw) { px = dsax; ++py; }
        };
        if (dpsm == GS_PSM_T4)
        {
            while (offset < sizeBytes && copied < total)
            {
                const uint8_t packed = data[offset++];
                wf(m_vram, dbp, dbw, px, py, packed & 0x0Fu);
                step();
                if (copied < total)
                {
                    wf(m_vram, dbp, dbw, px, py, (packed >> 4u) & 0x0Fu);
                    step();
                }
            }
        }
        else
        {
            // ★★ cont.231 (build 397): CT32/Z32 row groups. In the CT32 column layout the eight
            // pixels of an 8-aligned group on one row live in ONE 8x8 block: the same page, the same
            // block, offsets ColumnTable32[y%8][0..7] from the block base. So one Address() per
            // group and eight dword stores at table offsets replace eight Address() calls. The
            // offsets are DERIVED from the traits on first use (Address(x0+i, y) - Address(x0, y)
            // for a block-aligned x0, per y%8), never hand-copied, so they are exact by
            // construction; PS2X_GS_UPLOADVERIFY re-checks every pixel through the generic read.
            // The level streams ~4.9 MB of CT32 per frame through here (census, cont.231 §10).
            static const auto colOff = []() {
                std::array<std::array<uint32_t, 8>, 8> t{};
                for (uint32_t yy = 0; yy < 8u; ++yy)
                    for (uint32_t i = 0; i < 8u; ++i)
                        t[yy][i] = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, 0u, 1u, i, yy) -
                                   GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, 0u, 1u, 0u, yy);
                return t;
            }();
            const bool rowGroups = (dpsm == GS_PSM_CT32) && rrw >= 8u;
            // cont.357e: the row-pair block write (see PS2X_GS_UPBLOCK). Needs 8-aligned columns in
            // both the destination x and the row stride, which every transfer this game makes
            // satisfies (dsax=0, rrw=64/128).
            const bool pairBlocks = s_upBlock && rowGroups && (dsax & 7u) == 0u && (rrw & 7u) == 0u;
            while (offset + 4u <= sizeBytes && copied < total)
            {
                const uint32_t rowEnd = dsax + rrw; // exclusive
                if (pairBlocks && px == dsax && (py & 1u) == 0u && copied + 2u * rrw <= total &&
                    offset + 2u * rrw * 4u <= sizeBytes)
                {
                    const uint8_t *rowA = data + offset;
                    const uint8_t *rowB = rowA + rrw * 4u;
                    bool wrapped = false;
                    for (uint32_t gx = 0; gx < rrw; gx += 8u)
                    {
                        const uint32_t pxAddr = GSMem::PixelStorageTraits<GSMem::C32>::Address(
                            GSMem::PageTableC32, dbp, dbw, dsax + gx, py);
                        const uint32_t byteAddr = pxAddr * 4u;
                        if (byteAddr + 64u > static_cast<uint32_t>(GSMem::MEMORY_SIZE))
                        { wrapped = true; ++g_upBlockWrapFalls; break; }
                        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rowA + gx * 4u));
                        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rowA + gx * 4u + 16u));
                        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rowB + gx * 4u));
                        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rowB + gx * 4u + 16u));
                        uint8_t *d = m_vram + byteAddr;
                        const __m128i w0 = _mm_unpacklo_epi64(a0, b0);
                        const __m128i w1 = _mm_unpackhi_epi64(a0, b0);
                        const __m128i w2 = _mm_unpacklo_epi64(a1, b1);
                        const __m128i w3 = _mm_unpackhi_epi64(a1, b1);
                        if (s_upStream && (reinterpret_cast<uintptr_t>(d) & 63u) == 0u)
                        {
                            // A column is exactly one 64-byte line when the base is aligned, which it
                            // is whenever m_vram is: column bases are blockBase + {0,64,128,192}.
                            ++g_upStreamLines;
                            _mm_stream_si128(reinterpret_cast<__m128i *>(d +  0), w0);
                            _mm_stream_si128(reinterpret_cast<__m128i *>(d + 16), w1);
                            _mm_stream_si128(reinterpret_cast<__m128i *>(d + 32), w2);
                            _mm_stream_si128(reinterpret_cast<__m128i *>(d + 48), w3);
                        }
                        else
                        {
                            if (s_upStream) ++g_upStreamUnaligned;
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(d +  0), w0);
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(d + 16), w1);
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(d + 32), w2);
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(d + 48), w3);
                        }
                    }
                    if (!wrapped)
                    {
                        ++g_upBlockPairs;
                        offset += 2u * rrw * 4u;
                        copied += 2u * rrw;
                        py += 2u;      // px is already dsax and stays there
                        continue;
                    }
                    // A wrapping group falls through to the scalar path below, which reproduces the
                    // hardware's wrap. Any groups already written above carry the same bytes the
                    // scalar path will write, so re-writing them is a no-op.
                }
                if (rowGroups && (px & 7u) == 0u && px + 8u <= rowEnd && copied + 8u <= total &&
                    offset + 32u <= sizeBytes)
                {
                    const uint32_t base = GSMem::PixelStorageTraits<GSMem::C32>::Address(GSMem::PageTableC32, dbp, dbw, px, py);
                    const std::array<uint32_t, 8> &off = colOff[py & 7u];
                    for (uint32_t i = 0; i < 8u; ++i)
                    {
                        uint32_t value = 0u;
                        std::memcpy(&value, data + offset + i * 4u, sizeof(value));
                        GSMem::PixelStorageTraits<GSMem::C32>::WriteAt(m_vram, base + off[i], value);
                    }
                    offset += 32u;
                    copied += 8u;
                    px += 8u;
                    if (px == rowEnd) { px = dsax; ++py; }
                    continue;
                }
                uint32_t value = 0u;
                std::memcpy(&value, data + offset, sizeof(value));
                wf(m_vram, dbp, dbw, px, py, value);
                offset += 4u;
                step();
            }
        }
        // cont.357f: non-temporal stores are weakly ordered -- fence before anyone else (the texture
        // decode, the present path) can observe this chunk.
        if (s_upStream)
            _mm_sfence();
        if (s_gsUploadVerify && dpsm != GS_PSM_T4)
        {
            // Re-walk the chunk: every pixel's stored value must equal the value it carried.
            const uint32_t first = m_transferState.copiedPixels;
            uint32_t vx = dsax + (first % rrw), vy = dsay + (first / rrw);
            uint32_t voff = 0;
            for (uint32_t pI = first; pI < copied && voff + 4u <= sizeBytes; ++pI, voff += 4u)
            {
                uint32_t want = 0u;
                std::memcpy(&want, data + voff, sizeof(want));
                const uint32_t got = GSMem::PixelStorageTraits<GSMem::C32>::Read(GSMem::PageTableC32, m_vram, dbp, dbw, vx, vy);
                ++g_upVerifyPixels;
                if (got != want && g_upVerifyMismatch++ < 16ull)
                    std::fprintf(stderr, "[gs2:upverify] MISMATCH dbp=%u dbw=%u x=%u y=%u got=%08x want=%08x (chunk px %u..%u, rr=%ux%u)\n",
                                 dbp, dbw, vx, vy, got, want, first, copied, rrw, m_transfer.trxreg.rrh);
                if (++vx == dsax + rrw) { vx = dsax; ++vy; }
            }
            if ((++g_upVerifyChunks % 4096ull) == 0ull)
                std::fprintf(stderr, "[gs2:upverify] chunks=%llu pixels=%llu mismatches=%llu\n",
                             g_upVerifyChunks, g_upVerifyPixels, g_upVerifyMismatch);
        }
        // Write the transfer state back exactly as advancePixel would have left it.
        m_transferState.copiedPixels = copied;
        if (copied >= total)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
        }
        else
        {
            m_transferState.x = dsax + (copied % rrw);
            m_transferState.y = dsay + (copied / rrw);
        }
        if (offset < sizeBytes && m_transferState.direction == 0u && dpsm != GS_PSM_T4)
            return; // a trailing partial pixel: the generic path returned here too
    }

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            WriteVramUnlocked(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }

#if PS2X_HAS_GS_GPU_DEVICE
    // Phase-1 sampled verify: the CPU just applied this chunk (worker-serial, nothing
    // interleaves), so reading the destination pixels back IS the ground truth of what
    // the mirror kernel must have written. The device compares on-GPU behind the queued
    // MirrorUpload (strict FIFO).
    if (mirrorOn && mirrorCount != 0u && s_gsGpuXferVerify > 0)
    {
        static unsigned long s_xvN = 0;
        if ((++s_xvN % static_cast<unsigned long>(s_gsGpuXferVerify)) == 0u)
        {
            if (GsGpuPresentDevice *dev = gs2GpuDevice())
            {
                GsGpuUploadChunk c;
                c.dbp = dbp;
                c.dbw = dbw;
                c.dsax = dsax;
                c.dsay = m_transfer.trxpos.dsay;
                c.rrw = rrw;
                c.startPixel = mirrorStart;
                c.payload.resize(mirrorCount);
                for (uint32_t i = 0; i < mirrorCount; ++i)
                {
                    const uint32_t p = mirrorStart + i;
                    c.payload[i] = ReadVramUnlocked(GS_PSM_CT32, dbp, dbw,
                                                    dsax + (p % rrw),
                                                    m_transfer.trxpos.dsay + (p / rrw));
                }
                dev->MirrorVerify(std::move(c));
            }
        }
    }
#endif
    // cont.357c: advancePixel sets direction=3 on the last pixel, so a non-zero direction here means
    // this chunk completed the transfer -- the point at which the payload hash is the whole surface.
    if (s_upCensus && m_transferState.direction != 0u)
        upCensusFinish(g_perfFlips.load(std::memory_order_relaxed));
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.227b: invalidate cached CLUTs
    if (!m_vram)
        return;
    ++g_vramMutationSeq;
    if (s_gsGpuBatchVerify > 0 && !m_gpuBatch.prims.empty())
    { ++g_bvFlushExplicit; gpuBatchFlush(); }

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }
    // cont.357c: the OTHER route into VRAM, counted so a host->local null is interpretable.
    if (s_upCensus) { ++g_upBlits; g_upBlitPixels += total; }
    // ★★★ cont.330 -- ⚠ A LOCAL->LOCAL BLIT MUST NOT DIRTY A TARGET FOR SEEDING, even though it
    // does write the destination pages. Its SOURCE is VRAM, and under the GL renderer VRAM does
    // not contain what GL drew, so the blit moves stale/empty bytes: treating its destination as
    // "the guest wrote this" makes the seed copy that emptiness into the target and ERASE the
    // rendered frame. Measured (build 871, first attempt): the game blits full-screen every frame
    // in its post-process chain, so this dirtied all 128 pages of the display targets and the GL
    // picture went BLACK from flip ~1350 on (mean luma 0.00 against the CPU's 93.5), with
    // 106-114 pages seeded per event. Only HOST->LOCAL uploads are trustworthy here: their bytes
    // come from EE memory, which is the same in both renderers. Servicing a local->local blit
    // properly means a target-to-target copy inside GL, which is a separate piece of work.
    if (s_glTexTrace != 0u || s_mutCensus || s_pageWatch) // cont.331j: PAGEWATCH needs these sites too
    {
        const uint32_t dstPage = m_transfer.bitbltbuf.dbp / 32u;
        const uint32_t dbwPx = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u) * 64u;
        const uint32_t bpp = std::max<uint32_t>(bitsPerPixel(m_transfer.bitbltbuf.dpsm), 1u);
        const uint32_t lastPage =
            dstPage + ((m_transfer.trxpos.dsay + rrh) * dbwPx * bpp) / (8u * 8192u) + 1u;
        glTexTraceMutation("blit(local->local)", dstPage, lastPage, m_transfer.bitbltbuf.dpsm,
                           m_transfer.bitbltbuf.dbw, m_transfer.trxpos.dsax, m_transfer.trxpos.dsay,
                           m_transfer.trxpos.dsax + rrw - 1u, m_transfer.trxpos.dsay + rrh - 1u, true);
    }
    if (s_gsTexCensus)
    {
        // cont.320: the copy's destination rectangle, page-exact.
        Gs2PageMask um;
        gs2RectPages(um, m_transfer.bitbltbuf.dbp / 32u, std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                     m_transfer.bitbltbuf.dpsm, m_transfer.trxpos.dsax, m_transfer.trxpos.dsay,
                     m_transfer.trxpos.dsax + rrw - 1u, m_transfer.trxpos.dsay + rrh - 1u);
        gs2TexCensusBumpMask(um);
        ++g_texCensus.l2lXfers;
        Gs2BlkMask bm;
        gs2RectBlocks(bm, m_transfer.bitbltbuf.dpsm, m_transfer.bitbltbuf.dbp, std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                      m_transfer.trxpos.dsax, m_transfer.trxpos.dsay, m_transfer.trxpos.dsax + rrw - 1u, m_transfer.trxpos.dsay + rrh - 1u);
        gs2TexCensusBumpBlk(bm);
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;

#if PS2X_HAS_GS_GPU_DEVICE
    // cont.168: a local->local blit reads and writes VRAM entirely inside the GS, so the
    // upload kernel never sees it. Mirror the destination rect from the authoritative
    // bytes the loop just produced. (The dir flags only mirror the walk order within the
    // same rect, so the rect bounds are direction-independent.)
    if (gs2MirrorWanted())
        mirrorPatchRect(m_transfer.bitbltbuf.dpsm,
                        m_transfer.bitbltbuf.dbp,
                        std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                        m_transfer.trxpos.dsax,
                        m_transfer.trxpos.dsay,
                        m_transfer.trxpos.dsax + rrw - 1u,
                        m_transfer.trxpos.dsay + rrh - 1u,
                        MirrorPatchSource::LocalToLocal);
#endif
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    // The bool result feeds the caller's fallback decision, so this cannot be deferred:
    // drain (order wrt queued draws) and execute inline. Clears are rare.
    drainQueue();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || context.frame.fbw == 0u)
        return false;
    ++g_vramMutationSeq;
    g_gs2VramGen.fetch_add(1, std::memory_order_relaxed); gs2BumpAllPages(); // cont.320: was per pixel in WriteVramUnlocked
    if (s_gsGpuBatchVerify > 0 && !m_gpuBatch.prims.empty())
    { ++g_bvFlushExplicit; gpuBatchFlush(); }

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (s_glTexTrace != 0u || s_mutCensus || s_pageWatch) // cont.331j: PAGEWATCH needs these sites too
    {
        // FRAME.FBP already counts 8 KB pages; the clear covers the scissor rect.
        const uint32_t bpp = std::max<uint32_t>(bitsPerPixel(context.frame.psm), 1u);
        const uint32_t lastPage =
            context.frame.fbp + ((y1 + 1u) * fbw * 64u * bpp) / (8u * 8192u) + 1u;
        glTexTraceMutation("clear(framebuffer)", context.frame.fbp, lastPage, context.frame.psm,
                           fbw, x0, y0, x1, y1, true);
    }
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
#if PS2X_HAS_GS_GPU_DEVICE
        if (gs2MirrorWanted())
            mirrorPatchRect(context.frame.psm, fbp, fbw, x0, y0, x1, y1, MirrorPatchSource::Clear);
#endif
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
#if PS2X_HAS_GS_GPU_DEVICE
        if (gs2MirrorWanted())
            mirrorPatchRect(context.frame.psm, fbp, fbw, x0, y0, x1, y1, MirrorPatchSource::Clear);
#endif
        return true;
    }
    return false;
}

namespace
{
    // PS2X_GS_GPU_VERIFY bookkeeping: byte-compare the GPU rect against the CPU rect
    // (identical source bytes, identical parameters). Loud on the first few divergences,
    // then a periodic summary. Zero mismatches over a long soak is the promotion gate
    // for defaulting PS2X_GS_GPU_PRESENT on.
    void gs2GpuVerifyCompare(bool cpuOk,
                             const std::vector<uint8_t> &cpuPixels,
                             const std::vector<uint8_t> &gpuPixels,
                             const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height)
    {
        ++g_gpuVerifyCompares;
        if (!cpuOk)
        {
            // GPU decoded a rect the CPU rejected -- a support-matrix divergence, not a
            // pixel bug. Should be impossible (same psm set); scream once.
            static bool s_logged = false;
            if (!s_logged)
            {
                s_logged = true;
                std::fprintf(stderr, "[gsgpu:verify] CPU rejected a GPU-decoded rect (psm=0x%x)\n",
                             frame.psm);
            }
            return;
        }
        if (cpuPixels.size() == gpuPixels.size() &&
            std::memcmp(cpuPixels.data(), gpuPixels.data(), cpuPixels.size()) != 0)
        {
            ++g_gpuVerifyMismatchRects;
            unsigned long long rectPixels = 0;
            uint32_t firstX = 0, firstY = 0, firstCpu = 0, firstGpu = 0;
            for (uint32_t y = 0; y < height; ++y)
                for (uint32_t x = 0; x < width; ++x)
                {
                    const size_t o = (static_cast<size_t>(y) * kHostFrameWidth + x) * 4u;
                    uint32_t c, g;
                    std::memcpy(&c, cpuPixels.data() + o, 4u);
                    std::memcpy(&g, gpuPixels.data() + o, 4u);
                    if (c != g)
                    {
                        if (rectPixels == 0u)
                        {
                            firstX = x;
                            firstY = y;
                            firstCpu = c;
                            firstGpu = g;
                        }
                        ++rectPixels;
                    }
                }
            g_gpuVerifyMismatchPixels += rectPixels;
            if (g_gpuVerifyMismatchRects <= 8u)
                std::fprintf(stderr,
                             "[gsgpu:verify] MISMATCH rect#%lu psm=0x%x fbp=%u fbw=%u %ux%u: "
                             "%llu px differ, first (%u,%u) cpu=%08x gpu=%08x\n",
                             g_gpuVerifyMismatchRects, frame.psm, frame.fbp, frame.fbw,
                             width, height, rectPixels, firstX, firstY, firstCpu, firstGpu);
        }
        if ((g_gpuVerifyCompares % 512u) == 0u)
            std::fprintf(stderr, "[gsgpu:verify] compares=%lu mismatch-rects=%lu mismatch-px=%llu\n",
                         g_gpuVerifyCompares, g_gpuVerifyMismatchRects, g_gpuVerifyMismatchPixels);
    }
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
#if PS2X_HAS_GS_GPU_DEVICE
    // ★ cont.329 phase 1: with PS2X_GS_RENDERER=gl the frame comes from a GL render target, NOT
    // from VRAM -- so the present decode is bypassed entirely rather than decoding a buffer the
    // GL path never wrote. Everything downstream (the 640-stride repack, UpdateTexture, the
    // aspect-fit blit) is unchanged, because RenderFrameGl returns the same display-sized,
    // top-down RGBA8 this function already produces.
    // If the device is unavailable we deliberately FALL THROUGH to the CPU decode rather than
    // fail the present: a black window and a working one are hard to tell apart from a log, and
    // the stale-VRAM picture is at least honest about what happened.
    if (s_gsRendererGl && useLocalMemoryLayout)
    {
        if (GsGpuPresentDevice *dev = gs2GpuDevice())
        {
            // ★ cont.329 phase 4: resolve the target the DISPLAY points at. `frame` here IS the
            // display frame register, so its address selects among the game's buffers -- which is
            // what stops the two double-buffered frames compositing into one picture.
            // ★★★★ cont.331i: resolve the PRE-FLIP buffer when asked, so post-flip glyph draws
            // (the UI text) are included, matching the CPU flip-snapshot semantics.
            uint64_t displayKey = uint64_t(frame.fbp);
            if (s_glFlipResolve)
            {
                const uint32_t pf = g_glPreFlipFbp.load(std::memory_order_relaxed);
                if (pf != 0xFFFFFFFFu)
                { displayKey = uint64_t(pf); g_glFlipResolveUsed.fetch_add(1, std::memory_order_relaxed); }
            }
            // ★★★★ cont.331 authority model phase A: bring the DISPLAY target up to date with any
            // non-draw VRAM writes before resolving it. This is the other place a target is USED.
            // The present path is const, but seeding necessarily mutates GL-side cache state (the
            // per-target watermark and the pending batch); const_cast here rather than making the
            // whole present path non-const or scattering `mutable` across the cache members.
            const_cast<GSCpuBackend *>(this)->glSeedDisplayTargetForPresent(frame.fbp, frame.fbw,
                                                                           frame.psm);
            // ★★ cont.329g census: geometry still in the accumulator AT THIS INSTANT belongs to
            // the frame being resolved, but nothing flushes on a frame boundary -- it will be
            // drawn into the NEXT frame instead (and after that frame's clear). Counted here,
            // where the boundary actually is.
            if (g_glCensus)
            {
                // ★★★★ cont.331j: NOT m_glPendCount -- this is the present-side snapshotBackend,
                // a different instance whose accumulator is always empty. Read the worker's count.
                const unsigned long long pend = g_glPendGlobal.load(std::memory_order_relaxed);
                ++g_glcResolveSamples;
                if (pend)
                {
                    ++g_glcStrandedEvents;
                    g_glcStrandedVerts += pend;
                    if (pend > g_glcStrandedMax) g_glcStrandedMax = pend;
                }
                if ((g_glcResolveSamples % 256ull) == 0ull)
                {
                    const double f = double(g_glcResolveSamples);
                    std::fprintf(stderr,
                                 "[gs2:glcensus] resolves=%llu | runs=%llu (no-verts=%llu) "
                                 "prims{in=%llu zero-vtx=%llu dropped=%llu translated=%llu} "
                                 "verts=%llu groups=%llu tex-uploads=%llu | "
                                 "flush{threshold=%llu target-change=%llu tex-overwrite=%llu} batches-out=%llu "
                                 "verts-out=%llu | STRANDED at resolve: events=%llu/%llu verts=%llu "
                                 "(mean %.1f/resolve, max %llu)\n",
                                 g_glcResolveSamples, g_glcRuns, g_glcRunsNoVerts, g_glcPrimIn,
                                 g_glcPrimZero, g_glcPrimDrop, g_glcPrimXlat, g_glcVerts,
                                 g_glcGroups, g_glcTexUp, g_glcFlushThreshold,
                                 g_glcFlushTargetChange,
                                 g_glcFlushTexOverwrite, g_glcBatchesOut, g_glcVertsOut,
                                 g_glcStrandedEvents, g_glcResolveSamples, g_glcStrandedVerts,
                                 double(g_glcStrandedVerts) / f, g_glcStrandedMax);
                    {
                        std::fprintf(stderr,
                                     "[gs2:glcensus] IGNORED/APPROXIMATED STATE: COLCLAMP{clamp=%llu WRAP=%llu} "
                                     "FBA{0=%llu 1=%llu} PABE{0=%llu 1=%llu} wrap-modes{REPEAT=%llu CLAMP=%llu "
                                     "REGION_CLAMP=%llu REGION_REPEAT=%llu} texa-aem-draws=%llu\n",
                                     g_glcColclamp[1], g_glcColclamp[0], g_glcFba[0], g_glcFba[1],
                                     g_glcPabe[0], g_glcPabe[1], g_glcWrap[0], g_glcWrap[1],
                                     g_glcWrap[2], g_glcWrap[3], g_glcAemTex);
                        std::fprintf(stderr, "[gs2:glcensus] TEXTURE PSM mix:");
                        for (unsigned k = 0; k < 64u; ++k)
                            if (g_glcTexPsm[k])
                                std::fprintf(stderr, " 0x%02x=%.2f%%", k,
                                             100.0 * double(g_glcTexPsm[k]) /
                                                 double(g_glcTexPsmN ? g_glcTexPsmN : 1ull));
                        std::fprintf(stderr, "\n");
                        std::fprintf(stderr, "[gs2:glcensus] TEX1 mip state, (MMIN,MXL) -> %% of textured draws:");
                        for (unsigned mm = 0; mm < 8u; ++mm)
                            for (unsigned mx = 0; mx < 8u; ++mx)
                                if (g_glcMip[mm][mx])
                                    std::fprintf(stderr, " (%u,%u)=%.2f%%", mm, mx,
                                                 100.0 * double(g_glcMip[mm][mx]) /
                                                     double(g_glcMipN ? g_glcMipN : 1ull));
                        std::fprintf(stderr, "\n");
                    }
                    std::fprintf(stderr,
                                 "[gs2:glcensus] BLEND FACTOR As over the vertices of As-blended draws: "
                                 "n=%llu OVER 0x80 (factor > 1.0, unreachable with GL fixed function)=%llu "
                                 "(%.2f%%) | histogram in 32s: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
                                 g_glcAsN, g_glcAsOver,
                                 100.0*double(g_glcAsOver)/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[0])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[1])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[2])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[3])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[4])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[5])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[6])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[7])/double(g_glcAsN?g_glcAsN:1),
                                 100.0*double(g_glcAsHist[8])/double(g_glcAsN?g_glcAsN:1));
                    std::fprintf(stderr,
                                 "[gs2:glcensus] FOG: fge-draws=%llu (%.1f%% of draws) fge-verts=%llu"
                                 " | per-vertex F distribution 0..255 in eighths: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
                                 g_glcFgeDraws,
                                 100.0 * double(g_glcFgeDraws) /
                                     double(g_glcFgeDraws + g_glcNoFgeDraws ? g_glcFgeDraws + g_glcNoFgeDraws : 1ull),
                                 g_glcFgeVerts,
                                 100.0*double(g_glcFogHist[0])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[1])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[2])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[3])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[4])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[5])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[6])/double(g_glcFogN?g_glcFogN:1),
                                 100.0*double(g_glcFogHist[7])/double(g_glcFogN?g_glcFogN:1));
                    std::fprintf(stderr,
                                 "[gs2:glcensus] MIXED-TARGET RUNS: runs=%llu items-on-the-wrong-target=%llu"
                                 " split=%llu (PS2X_GS_GLR_TGTSPLIT=%d)\n",
                                 g_glcMixedRuns, g_glcMixedItems, g_glcSplitRuns, s_glTgtSplit ? 1 : 0);
                    std::fprintf(stderr,
                                 "[gs2:glcensus] PER RESOLVE: prims-in=%.1f translated=%.1f "
                                 "verts=%.0f groups=%.1f batches-out=%.2f verts-out=%.0f\n",
                                 double(g_glcPrimIn) / f, double(g_glcPrimXlat) / f,
                                 double(g_glcVerts) / f, double(g_glcGroups) / f,
                                 double(g_glcBatchesOut) / f, double(g_glcVertsOut) / f);
                    std::fprintf(stderr,
                                 "[gs2:glcensus] GEOMETRY AREA (guest coords, the fragments the "
                                 "primitives SHOULD produce): sprites=%llu area=%.0f (mean %.0f px) "
                                 "| tris=%llu area=%.0f (mean %.1f px) | scissor-clipped total=%.0f "
                                 "= %.0f px/resolve\n",
                                 g_glcNSprite, g_glcAreaSprite,
                                 g_glcNSprite ? g_glcAreaSprite / double(g_glcNSprite) : 0.0,
                                 g_glcNTri, g_glcAreaTri,
                                 g_glcNTri ? g_glcAreaTri / double(g_glcNTri) : 0.0,
                                 g_glcAreaClipped, g_glcAreaClipped / f);
                    if (g_glPageMap)
                    {
                        std::fprintf(stderr, "[gs2:pagemap] pages written by draws AND read as texture "
                                             "(page: writes/reads) -- these are served STALE by GL:\n");
                        unsigned shown = 0;
                        unsigned long long bothW = 0, bothR = 0, wOnly = 0, rOnly = 0;
                        for (unsigned pg = 0; pg < 512u; ++pg)
                        {
                            if (g_pgWrite[pg] && g_pgRead[pg])
                            {
                                bothW += g_pgWrite[pg]; bothR += g_pgRead[pg];
                                if (shown < 48u)
                                {
                                    std::fprintf(stderr, "[gs2:pagemap]   0x%03x: w=%-10llu r=%-10llu\n",
                                                 pg, g_pgWrite[pg], g_pgRead[pg]);
                                    ++shown;
                                }
                            }
                            else if (g_pgWrite[pg]) wOnly += g_pgWrite[pg];
                            else if (g_pgRead[pg]) rOnly += g_pgRead[pg];
                        }
                        std::fprintf(stderr,
                                     "[gs2:pagemap] STALE READS: %llu textured draws sample a page this "
                                     "renderer has DRAWN into (%.2f%% of textured draws); %llu do not. "
                                     "Worst offenders:\n",
                                     g_staleDraws,
                                     100.0 * double(g_staleDraws) /
                                         double(g_staleDraws + g_freshDraws ? g_staleDraws + g_freshDraws : 1ull),
                                     g_freshDraws);
                        for (unsigned r = 0; r < g_staleN; ++r)
                            if (g_stale[r].draws)
                                std::fprintf(stderr,
                                             "[gs2:pagemap]   tbp=%-6u (page 0x%03x) %ux%u psm=0x%02x draws=%llu\n",
                                             g_stale[r].tbp, g_stale[r].tbp >> 5, g_stale[r].tw,
                                             g_stale[r].th, g_stale[r].psm, g_stale[r].draws);
                        std::fprintf(stderr,
                                     "[gs2:pagemap] TOTALS: pages both-written-and-read carry %llu writes and "
                                     "%llu reads; write-only pages %llu writes; read-only pages %llu reads\n",
                                     bothW, bothR, wOnly, rOnly);
                    }
                    if (s_mutCensus)
                    {
                        std::fprintf(stderr,
                                     "[gs2:mutcensus] NON-DRAW VRAM writes inside the probe flip window: "
                                     "%llu events\n", g_mutCensusTotal);
                        for (int si = 0; si < kMutSites; ++si)
                        {
                            if (!g_mutSiteCount[si]) continue;
                            // Print the page set as ranges, so "which pages" is readable.
                            char buf[512]; size_t used = 0; int runStart = -1;
                            for (uint32_t pg = 0; pg <= kGs2Pages; ++pg)
                            {
                                const bool set = pg < kGs2Pages &&
                                    (g_mutSitePages[si][pg >> 6] >> (pg & 63)) & 1ull;
                                if (set && runStart < 0) runStart = int(pg);
                                else if (!set && runStart >= 0)
                                {
                                    if (used < sizeof(buf) - 24)
                                        used += size_t(std::snprintf(buf + used, sizeof(buf) - used,
                                                                     " %d-%d", runStart, int(pg) - 1));
                                    runStart = -1;
                                }
                            }
                            buf[used] = 0;
                            std::fprintf(stderr, "[gs2:mutcensus]   %-20s events=%-8llu pages:%s\n",
                                         kMutSiteName[si], g_mutSiteCount[si], used ? buf : " (none)");
                            for (unsigned fr = 0; fr < g_mutFmtN; ++fr)
                                if (g_mutFmt[fr].site == uint32_t(si))
                                    std::fprintf(stderr,
                                                 "[gs2:mutcensus]       WROTE psm=0x%02x bw=%-3u events=%llu\n",
                                                 g_mutFmt[fr].psm, g_mutFmt[fr].bw, g_mutFmt[fr].n);
                        }
                    }
                    if (s_glAuthority)
                        std::fprintf(stderr,
                                     "[gs2:authority] present-seed calls=%llu events=%llu rects=%llu | "
                                     "declined{no-vram=%llu seq-unchanged=%llu no-dirty-rects=%llu "
                                     "no-target-height=%llu} seed-skipped-no-h=%llu | "
                                     "pages-owned-by-GL-draws (cont.331l, not seeded)=%llu | "
                                     "source-seed{calls=%llu events=%llu rects=%llu "
                                     "seq-unchanged=%llu no-rects=%llu no-height=%llu}\n",
                                     g_psCalls, g_glPresentSeedEvents, g_glPresentSeedRects,
                                     g_psNoVram, g_psSeqSame, g_psNoRects, g_psNoH,
                                     g_glSeedSkippedNoH, g_glSeedOwnedSkips,
                                     g_srcSeedCalls, g_srcSeedEvents, g_srcSeedRects,
                                     g_srcSeedSeqSame, g_srcSeedNoRects, g_srcSeedNoH);
                        std::fprintf(stderr,
                                     "[gs2:authority] present-seed DECODES WITH fbp=0x%x fbw=%u psm=0x%02x"
                                     " (from DISPFB) -- compare against [gs2:mutcensus] WROTE lines\n",
                                     g_psLastFbp, g_psLastFbw, g_psLastPsm);
                    if (s_frameHist && !g_frameHistDumped) { g_frameHistDumped = true; dumpFrameHist(); }
                    if (g_glTgtSrcCensus)
                    {
                        std::fprintf(stderr, "[gs2:tgtsrc] DRAW TARGETS:\n");
                        for (unsigned k = 0; k < g_glcTgtN; ++k)
                            std::fprintf(stderr,
                                         "[gs2:tgtsrc]   fbp=0x%-4x (page %4u, byte %8u) fbw=%u (%4u px) psm=0x%02x "
                                         "draws=%-10llu (untextured=%llu fbmsk{none=%llu all=%llu other=%llu}) "
                                         "verts=%-11llu scissor-max=%u,%u "
                                         "vertex-extent=[%d..%d]x[%d..%d] ZBUF{zbp=0x%x psm=0x%02x}\n",
                                         g_glcTgt[k].fbp, g_glcTgt[k].fbp, g_glcTgt[k].fbp * 8192u,
                                         g_glcTgt[k].fbw, g_glcTgt[k].fbw * 64u, g_glcTgt[k].psm,
                                         g_glcTgt[k].draws, g_glcTgt[k].noTex,
                                         g_glcTgt[k].mskNone, g_glcTgt[k].mskAll, g_glcTgt[k].mskOther,
                                         g_glcTgt[k].verts,
                                         g_glcTgt[k].maxX, g_glcTgt[k].maxY,
                                         g_glcTgt[k].vminX, g_glcTgt[k].vmaxX,
                                         g_glcTgt[k].vminY, g_glcTgt[k].vmaxY,
                                         g_glcTgt[k].zbp, g_glcTgt[k].zpsm);
                        if (s_glTgtSrcDst != 0xFFFFFFFFu)
                            std::fprintf(stderr,
                                         "[gs2:tgtsrc] TEXTURE SOURCES at/above page 0x100, PLUS every source"
                                         " drawn into fbp=0x%x (PS2X_GS_GLR_TGTSRC_DST, no page filter):\n",
                                         s_glTgtSrcDst);
                        else
                            std::fprintf(stderr, "[gs2:tgtsrc] TEXTURE SOURCES at/above page 0x100:\n");
                        for (unsigned k = 0; k < g_glcSrcN; ++k)
                            std::fprintf(stderr,
                                         "[gs2:tgtsrc]   tbp=%-6u (page 0x%-4x +%2u blocks) tbw=%u psm=0x%02x %ux%u "
                                         "tfx=%llu tcc=%llu CLUT{cbp=0x%-5x cpsm=0x%02x csm=%u csa=%u} fst=%u "
                                         "uv=[%d..%d]x[%d..%d] -> drawn into fbp=0x%-4x draws=%-9llu promoted=%llu\n",
                                         g_glcSrc[k].tbp, g_glcSrc[k].tbp >> 5, g_glcSrc[k].tbp & 31u,
                                         g_glcSrc[k].tbw, g_glcSrc[k].psm, g_glcSrc[k].tw, g_glcSrc[k].th,
                                         g_glcSrc[k].tfx, g_glcSrc[k].tcc,
                                         g_glcSrc[k].cbp, g_glcSrc[k].cpsm, g_glcSrc[k].csm, g_glcSrc[k].csa,
                                         g_glcSrc[k].fst,
                                         g_glcSrc[k].uMin, g_glcSrc[k].uMax, g_glcSrc[k].vMin, g_glcSrc[k].vMax,
                                         g_glcSrc[k].dstFbp, g_glcSrc[k].draws, g_glcSrc[k].promoted);
                        if (s_glNoPromote != 0xFFFFFFFFu)
                            std::fprintf(stderr,
                                         "[gs2:tgtsrc] PROMOTION REFUSED (PS2X_GS_GLR_NOPROMOTE): %llu reads fell back"
                                         " to the VRAM decode\n", g_glPromoteRefused);
                        if (g_glcSrcDropped)
                            std::fprintf(stderr,
                                         "[gs2:tgtsrc]   ⚠ %llu distinct sources did NOT fit the table"
                                         " -- the list above is incomplete\n", g_glcSrcDropped);
                    }
                    std::fprintf(stderr, "[gs2:glcensus] FBMSK values (draws | px they cover):");
                    for (unsigned k = 0; k < g_glcFbmskUsed; ++k)
                        std::fprintf(stderr, " %08x=%llu/%.0f", g_glcFbmskVal[k], g_glcFbmskN[k],
                                     g_glcFbmskArea[k]);
                    std::fprintf(stderr, "\n");
                }
            }
            thread_local std::vector<uint8_t> glPixels;
            // cont.345: a flip/Sync capture's seq (m_glSnapSeq, 0 = live VRAM); the device returns
            // the worker's flip-point resolve for it (s_glSnapResolve).
            const uint64_t snapSeq = m_glSnapSeq;
            if (dev->RenderFrameGl(displayKey, width, height, glPixels, snapSeq))
            {
                outPixels.swap(glPixels);
                return true;
            }
        }
    }
    // GPU present decode (m_gpuDecode is only ever set on Present()'s snapshot backend).
    // The linear (non-local-memory) layout path is CPU-only -- it is dead in the present
    // flow (every present call site passes useLocalMemoryLayout=true).
    if (m_gpuDecode && useLocalMemoryLayout)
    {
        thread_local std::vector<uint8_t> gpuPixels;
        if (m_gpuDecode->DecodeFrameRect(m_vram, m_vramSize, m_gpuSourceId, frame, width, height,
                                         frameBaseIsPages, sourceOriginX, sourceOriginY,
                                         preserveAlpha, gpuPixels))
        {
            if (!s_gsGpuVerify)
            {
                outPixels.swap(gpuPixels);
                return true;
            }
            const bool cpuOk = CopyFrameToHostRgbaCpu(frame, width, height, outPixels,
                                                      preserveAlpha, useLocalMemoryLayout,
                                                      frameBaseIsPages, sourceOriginX,
                                                      sourceOriginY);
            gs2GpuVerifyCompare(cpuOk, outPixels, gpuPixels, frame, width, height);
            return cpuOk; // CPU stays authoritative under verify
        }
    }
#endif
    return CopyFrameToHostRgbaCpu(frame, width, height, outPixels, preserveAlpha,
                                  useLocalMemoryLayout, frameBaseIsPages,
                                  sourceOriginX, sourceOriginY);
}

bool GSCpuBackend::CopyFrameToHostRgbaCpu(const GSFrameReg &frame,
                                          uint32_t width,
                                          uint32_t height,
                                          std::vector<uint8_t> &outPixels,
                                          bool preserveAlpha,
                                          bool useLocalMemoryLayout,
                                          bool frameBaseIsPages,
                                          uint32_t sourceOriginX,
                                          uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Frames actually presented -- the only metric that answers "is the game faster?".
    // Primitives/second turned out to be confounded (it depends where in the timeline a run
    // has reached), so the ablation A/B is judged on this instead.
    ++g_perfPresents;
    // Snapshot local memory, then perform the expensive display conversion without
    // holding the producer-side raster lock. Prefer the frame-complete copy taken at the
    // last drained Sync (VRAM caught up with the display registers); fall back to live
    // VRAM (no-drain, possibly mid-frame) when the guest hasn't Synced recently. The
    // main thread must never starve on the worker queue (input pump).
    thread_local std::vector<uint8_t> snapshot;
    GSPresentationRequest effectiveRequest = request;
    const uint64_t copyTick = m_presentCopyTickMs.load(std::memory_order_acquire);
    const uint64_t nowMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    bool usedPresentCopy = false;
    uint64_t copySeq = 0;
    // cont.231 PS2X_GS_PRESENT_CACHE: the decoded frame of the last present, keyed on the snapshot
    // sequence and everything PresentFromLocalMemory reads from the request.
    struct PresentKey
    {
        uint64_t seq, pmode, smode2, dispfb1, display1, dispfb2, display2, bgcolor, field;
        GSFrameReg ctx0, ctx1, pref;
        uint32_t prefDest;
        bool hasPref;
        bool operator==(const PresentKey &o) const
        {
            auto fr = [](const GSFrameReg &a, const GSFrameReg &b)
            { return a.fbp == b.fbp && a.fbw == b.fbw && a.psm == b.psm && a.fbmsk == b.fbmsk; };
            return seq == o.seq && pmode == o.pmode && smode2 == o.smode2 && dispfb1 == o.dispfb1 &&
                   display1 == o.display1 && dispfb2 == o.dispfb2 && display2 == o.display2 &&
                   bgcolor == o.bgcolor && field == o.field && fr(ctx0, o.ctx0) && fr(ctx1, o.ctx1) &&
                   fr(pref, o.pref) && prefDest == o.prefDest && hasPref == o.hasPref;
        }
    };
    // The decode reads the vsync parity only in FIELD mode (interlaced, not frame mode); outside
    // it the parity must not be part of the key (it flips every vblank and made the cache miss
    // ~85% of presents in build 398). In field mode the two fields alternate, so keep one slot each.
    auto fieldOf = [](const GSPresentationRequest &r) -> uint64_t {
        const GSSmode2State sm = decodeSMode2(r.smode2);
        return (sm.interlaced && !sm.frameMode && s_gsFieldBob) ? (r.vsyncTick & 1ull) : 0ull;
    };
    auto makeKey = [&fieldOf](uint64_t seq, const GSPresentationRequest &r) {
        return PresentKey{seq, r.pmode, r.smode2, r.dispfb1, r.display1, r.dispfb2, r.display2, r.bgcolor,
                          fieldOf(r), r.contextFrames[0], r.contextFrames[1], r.preferredSource,
                          r.preferredDestFbp, r.hasPreferredSource};
    };
    thread_local PresentationFrame s_cachedFrames[2];
    thread_local PresentKey s_cachedKeys[2]{};
    thread_local bool s_cacheValids[2] = {false, false};
    // ★★★★ cont.345: in GL mode a STALE copy is still presented (its flip-point snapshot is a
    // complete frame), because the alternative -- resolving the live target -- shows whatever the
    // guest has drawn so far into the buffer it is working on: the one-flip flash of a half-drawn
    // frame at every slow frame (46-52 of ~1,300 real-time presents fell back to live before this).
    // No-flip eras still move: the per-Sync capture bumps the copy (with its own GL snapshot).
    const bool keepStaleCopy = s_gsRendererGl && s_glSnapResolve;
    if (copyTick != 0u && (keepStaleCopy || nowMs - copyTick <= kPresentCopyFreshMs))
    {
        usedPresentCopy = true;
        std::lock_guard<std::mutex> plock(m_presentCopyMutex);
        if (s_gsPresentCache)
        {
            GSPresentationRequest probe = request;
            if (m_presentCopyHasRegs)
            {
                probe.dispfb1 = m_presentCopyDispfb1;
                probe.dispfb2 = m_presentCopyDispfb2;
            }
            const size_t slot = static_cast<size_t>(fieldOf(probe));
            if (s_cacheValids[slot] && makeKey(m_presentCopySeq, probe) == s_cachedKeys[slot])
            {
                ++g_perfPresentsCached;
                return s_cachedFrames[slot];
            }
        }
        snapshot = m_presentCopy;
        copySeq = m_presentCopySeq;
        // A display-flip snapshot pairs the VRAM copy with the DISPLAY REGISTERS from the
        // same instant (the pre-flip front buffer, the one the frame was completed in --
        // its UI text included). Rendering the copy with the LIVE registers would show the
        // freshly flipped buffer, whose post-flip front-buffer text isn't drawn yet.
        if (m_presentCopyHasRegs)
        {
            effectiveRequest.dispfb1 = m_presentCopyDispfb1;
            effectiveRequest.dispfb2 = m_presentCopyDispfb2;
        }
    }
    else
    {
        SnapshotVramNoDrain(snapshot);
    }

    // ★★ cont.230: how often does the presenter fall back to LIVE, possibly mid-draw VRAM?
    // The frame-complete snapshot above is used only while it is <= 500 ms old, but a level-era
    // guest frame can take far longer than that -- and every fallback present can show a
    // partially rasterised buffer, which is the on-screen "artifact" class being chased.
    // Always on: two counters and one line per 256 presents costs nothing at ~40 presents/s.
    {
        static std::atomic<unsigned long long> s_presents{0}, s_liveFallback{0};
        const unsigned long long n = ++s_presents;
        if (!usedPresentCopy)
            ++s_liveFallback;
        if ((n & 255ull) == 0ull)
        {
            const unsigned long long fb = s_liveFallback.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                         "[gs2:presentsrc] presents=%llu liveFallback=%llu (%.1f%%) ageMs=%llu\n",
                         n, fb, 100.0 * static_cast<double>(fb) / static_cast<double>(n),
                         (copyTick != 0u && nowMs >= copyTick) ? (nowMs - copyTick) : 0ull);
        }
    }
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    snapshotBackend.m_gpuDecode = nullptr;
    // cont.345: the flip/Sync capture this present renders, for the GL flip-point snapshot lookup
    // (m_gpuSourceId below is only set when the GPU present decode is on, which GL mode is not).
    snapshotBackend.m_glSnapSeq = usedPresentCopy ? copySeq : 0ull;
#if PS2X_HAS_GS_GPU_DEVICE
    if (s_gsGpuPresent || s_gsGpuVerify)
    {
        // Upload-cache key: flip/Sync captures use the capture SEQUENCE number (unique
        // per capture -- the capture tick is milliseconds and collides, which made the
        // GPU decode a stale upload in run172; both rects of a present, and consecutive
        // presents of the same capture, still hit the cached upload); a live snapshot
        // gets a fresh id per Present, shared by that present's rects. Bit 63 separates
        // the two id spaces.
        static uint64_t s_livePresentSeq = 0;
        snapshotBackend.m_gpuDecode = gs2GpuDevice();
        snapshotBackend.m_gpuSourceId =
            usedPresentCopy ? (copySeq | (1ull << 63)) : ++s_livePresentSeq;
    }
#endif
    PresentationFrame out = snapshotBackend.PresentFromLocalMemory(effectiveRequest);
    if (s_gsPresentCache && usedPresentCopy && static_cast<bool>(out))
    {
        const size_t slot = static_cast<size_t>(fieldOf(effectiveRequest));
        s_cachedFrames[slot] = out;
        s_cachedKeys[slot] = makeKey(copySeq, effectiveRequest);
        s_cacheValids[slot] = true;
    }
    return out;
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode && s_gsFieldBob; // cont.232: full frame by default
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    if (s_drawCmpPrim != 0ull && width1 != 0u && height1 != 0u)
    {
        g_drawCmpW = width1 < 640u ? width1 : 640u; // the snapshot buffer is 640-stride
        g_drawCmpH = height1;
    }
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    // ★★★★ cont.332d THE PRESENTED PIXEL ASPECT. The framebuffer's pixel count says nothing about
    // the shape on screen: the CRTC magnifies DW+1 VCK units across the SAME raster width whatever
    // the source width is, so this game's 512x511 display is stretched to exactly the width a
    // 640-wide one would be -- and presenting it 1:1 (which is what the window blit did) makes
    // every character ~33% too narrow. PCSX2 models this with a per-video-mode table
    // (GSState.cpp VideoModeOffsets / VideoModeDividers: PAL = 640x256 per field over a 2560x288
    // raster, NTSC = 640x224 over 2560x240) and then fits that nominal display into a 4:3 output.
    // Mirrored here from the registers, so a mode change follows automatically:
    //   aspect = 4/3 * ((DW+1)/2560) / ((DH+1)/visibleLines)
    // i.e. 4:3 scaled by the fraction of the raster this display actually occupies. The mode comes
    // from DH itself (PAL's full-height display is 512 lines, NTSC's 448) rather than from the
    // region flag, so the two cannot disagree.
    {
        const uint64_t disp = valid1 ? request.display1 : request.display2;
        const uint32_t dw = static_cast<uint32_t>((disp >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((disp >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((disp >> 23) & 0x0Fu);
        const uint32_t magv = static_cast<uint32_t>((disp >> 27) & 0x03u);
        if ((valid1 || valid2) && dw != 0u && dh != 0u)
        {
            // ★★★★ cont.332d MEASURED, NOT ASSUMED: this game's DISPLAY carries **MAGH=0 and
            // DW+1=512** -- i.e. the source width itself, with no magnification. On hardware the
            // CRTC would stretch 512 source pixels across the full 2560-VCK raster with MAGH=4
            // (libgraph's sceGsSetDefDispEnv writes DW = w*(magh+1)-1), so a register-derived
            // fraction computed from DW alone reads 0.2 of the raster and produces a nonsense
            // aspect of 0.27. Whether the game really writes that or our sceGsPutDispEnv HLE
            // flattens it is an open question (see progress.md cont.332d).
            // So: use the register fraction only when the CRTC is actually magnifying -- which is
            // the case the fraction is meaningful in -- and otherwise fall back to the SD raster's
            // own shape, 4:3, which is what PCSX2 presents an SD mode at (its Aspect Ratio setting
            // defaults to 4:3 for NTSC/PAL; the DISPLAY rect only places the image inside it).
            const uint32_t srcW = (dw + 1u) / (magh + 1u);
            const bool magProgrammed = magh > 0u && (dw + 1u) >= 2u * srcW;
            const double visibleLines = (dh + 1u) > 480u ? 512.0 : 448.0;
            const double aspect = magProgrammed
                                      ? (4.0 / 3.0) * (double(dw + 1u) / 2560.0) /
                                            (double(dh + 1u) / visibleLines)
                                      : (4.0 / 3.0);
            const float prev = g_gsPresentAspect.exchange(static_cast<float>(aspect),
                                                          std::memory_order_relaxed);
            static unsigned s_dispLogs = 0;
            if (s_dispLogs < 32u && (prev == 0.0f || std::fabs(prev - float(aspect)) > 0.001f))
            {
                ++s_dispLogs;
                std::fprintf(stderr,
                             "[gs:disp] crt%d raw=0x%016llx DW+1=%u DH+1=%u MAGH=%u MAGV=%u ->"
                             " source %ux%u | %s | present aspect %.4f (4:3 = %.4f)\n",
                             valid1 ? 1 : 2, (unsigned long long)disp, dw + 1u, dh + 1u, magh, magv,
                             srcW, dh + 1u,
                             magProgrammed ? "magnified: raster fraction used"
                                           : "MAGH=0, DW carries no magnification -> SD raster 4:3",
                             aspect, 4.0 / 3.0);
            }
        }
    }
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;
        // Present-output save (PS2X_GS_PRESENT_SAVE=<dir>): every 512th present, save the
        // EXACT pixels handed to the viewer (rotating 4 slots) — separates "presentation reads
        // wrong" from "viewer blit wrong".
        {
            static unsigned long s_presN = 0;
            ++s_presN;
            const char *pd = std::getenv("PS2X_GS_PRESENT_SAVE");
            // cont.230: PS2X_GS_PRESENT_SAVE_EVERY (default 32) / PS2X_GS_PRESENT_SAVE_SLOTS (default
            // 16) make the rotation cover a whole level-era walk instead of its last ~13 s.
            static const unsigned long s_presEvery = []
            { const char *e = std::getenv("PS2X_GS_PRESENT_SAVE_EVERY"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 32ul; return v ? v : 32ul; }();
            static const unsigned long s_presSlots = []
            { const char *e = std::getenv("PS2X_GS_PRESENT_SAVE_SLOTS"); unsigned long v = e ? std::strtoul(e, nullptr, 10) : 16ul; return v ? v : 16ul; }();
            // ★ cont.329 phase 4c: PS2X_GS_PRESENT_SAVE_BYFLIP=<n> names the file by the guest
            // DISPLAY FLIP count instead of by a rotating presentation slot. Presentation is
            // host-paced, so two renderers running at different speeds save different MOMENTS and
            // a side-by-side compares nothing -- which defeated two attempts at exactly that
            // comparison. The flip counter is guest-driven and, under deterministic replay,
            // identical across runs, so flip K is the same instant in both.
            static const unsigned long s_presByFlip = []
            { const char *e = std::getenv("PS2X_GS_PRESENT_SAVE_BYFLIP"); return e ? std::strtoul(e, nullptr, 10) : 0ul; }();
            static unsigned long s_lastSavedFlip = ~0ul;
            const unsigned long s_flipNow = static_cast<unsigned long>(g_perfFlips.load());
            const bool saveByFlip = pd && s_presByFlip && !pixels.empty() &&
                                    (s_flipNow % s_presByFlip) == 0ul && s_flipNow != s_lastSavedFlip;
            if (saveByFlip)
                s_lastSavedFlip = s_flipNow;
            if (saveByFlip || (pd && !s_presByFlip && g_gs2EraOpen &&
                               (s_presN % s_presEvery) == 0u && !pixels.empty()))
            {
                char path[512];
                if (saveByFlip)
                    std::snprintf(path, sizeof(path), "%s/flip_%06lu.ppm", pd, s_flipNow);
                else
                    std::snprintf(path, sizeof(path), "%s/present_%02lu.ppm", pd, (s_presN / s_presEvery) % s_presSlots);
                if (FILE *f = std::fopen(path, "wb"))
                {
                    std::fprintf(f, "P6\n%u %u\n255\n", width, height);
                    for (uint32_t yy = 0; yy < height; ++yy)
                        for (uint32_t xx = 0; xx < width; ++xx)
                        {
                            const uint8_t *pp = pixels.data() + (yy * kHostFrameWidth + xx) * 4u;
                            std::fputc(pp[0], f);
                            std::fputc(pp[1], f);
                            std::fputc(pp[2], f);
                        }
                    std::fclose(f);
                    std::fprintf(stderr, "[gs2:present-save] %s src fbp=%u fbw=%u psm=0x%x %ux%u\n",
                                 path, selected.fbp, selected.fbw, selected.psm, width, height);
                }
            }
        }
        // Presentation-source trace (part of PS2X_GS_CENSUS2): one line per source change.
        if (s_gsCensus2)
        {
            static uint32_t lastKey = 0xFFFFFFFFu;
            const uint32_t key = (selected.fbp << 12) ^ (selected.fbw << 6) ^ selected.psm ^
                                 (usedPreferred ? 0x80000000u : 0u) ^ (width << 16);
            if (key != lastKey)
            {
                lastKey = key;
                std::fprintf(stderr,
                             "[gs2:present] src{fbp=%u fbw=%u psm=0x%x} disp{fbp=%u fbw=%u psm=0x%x} "
                             "size=%ux%u preferred=%u\n",
                             selected.fbp, selected.fbw, selected.psm,
                             displayFrame.fbp, displayFrame.fbw, displayFrame.psm,
                             width, height, usedPreferred ? 1u : 0u);
            }
        }

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
