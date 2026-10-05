#ifndef PS2_VU1_DETAIL_H
#define PS2_VU1_DETAIL_H

#include <cstdint>
#include <cstring>
#include <smmintrin.h> // SSE4.1 (_mm_blendv_epi8); the runtime is built with -msse4.1

// ---------------------------------------------------------------------------
// cont.181: SSE quad forms of the two VU float clamps.
//
// These became possible only after cont.180 adopted PCSX2's FMAC result model: both clamps are now
// pure functions of a lane's BITS, with no per-lane double arithmetic and no data-dependent
// branches, so all four lanes can be done at once. The scalar versions in ps2_vu1_core.cpp remain
// the reference — these must stay bit-identical to them, and PS2X_VU1_SIMD=0 selects them.
//
//   vuNormOperandBits  == VU1Interpreter::normalizeOperand == PCSX2 VUops.cpp vuDouble
//                         (exp 0 -> signed zero; exp 0xFF -> signed 0x7F7FFFFF)
//   vuNormResultBits   == VU1Interpreter::normalizeResult  == PCSX2 VUflags.cpp VU_MAC_UPDATE
//                         (sign -> flag 0x2; mag 0 -> 0x1; exp 0 -> 0x5 + flush; 0xFF -> 0x8 + clamp)
// ---------------------------------------------------------------------------

// Lane c (0..3) is enabled iff dest & (1 << (3 - c)) — matches laneForComponent().
static inline __m128i vuDestMask(uint8_t dest)
{
    return _mm_set_epi32((dest & 0x1u) ? -1 : 0,  // element 3 == component 3
                         (dest & 0x2u) ? -1 : 0,
                         (dest & 0x4u) ? -1 : 0,
                         (dest & 0x8u) ? -1 : 0); // element 0 == component 0
}

static inline __m128i vuNormOperandBits(__m128i bits)
{
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i signMask = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i maxMag = _mm_set1_epi32(0x7F7FFFFF);
    const __m128i e = _mm_and_si128(bits, expMask);
    const __m128i sign = _mm_and_si128(bits, signMask);
    const __m128i isDenorm = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    const __m128i isInf = _mm_cmpeq_epi32(e, expMask);
    __m128i r = _mm_blendv_epi8(bits, _mm_or_si128(sign, maxMag), isInf);
    return _mm_blendv_epi8(r, sign, isDenorm);
}

static inline void vuNormOperandQuad(const float *in, float *out)
{
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out),
                     vuNormOperandBits(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in))));
}

// Clamp only (no flags) — the shape used when lazy flags proved nothing reads MAC/status.
static inline void vuNormResultQuadValue(float *inout, uint8_t dest)
{
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inout));
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i signMask = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i e = _mm_and_si128(bits, expMask);
    const __m128i sign = _mm_and_si128(bits, signMask);
    const __m128i mag = _mm_and_si128(bits, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i isZero = _mm_cmpeq_epi32(mag, _mm_setzero_si128());
    // denormal == exponent 0 AND not exactly zero (zero keeps its bits)
    const __m128i isDenorm = _mm_andnot_si128(isZero, _mm_cmpeq_epi32(e, _mm_setzero_si128()));
    const __m128i isInf = _mm_cmpeq_epi32(e, expMask);
    __m128i r = _mm_blendv_epi8(bits, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)), isInf);
    r = _mm_blendv_epi8(r, sign, isDenorm);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(inout),
                     _mm_blendv_epi8(bits, r, vuDestMask(dest)));
}

// Clamp + per-lane MAC flags (bit0 Z, bit1 S, bit2 U, bit3 O), masked to `dest`.
static inline void vuNormResultQuad(float *inout, uint8_t dest, uint8_t laneFlags[4])
{
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inout));
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i signMask = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i e = _mm_and_si128(bits, expMask);
    const __m128i sign = _mm_and_si128(bits, signMask);
    const __m128i mag = _mm_and_si128(bits, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i isZero = _mm_cmpeq_epi32(mag, _mm_setzero_si128());
    const __m128i isDenorm = _mm_andnot_si128(isZero, _mm_cmpeq_epi32(e, _mm_setzero_si128()));
    const __m128i isInf = _mm_cmpeq_epi32(e, expMask);
    const __m128i dm = vuDestMask(dest);

    __m128i r = _mm_blendv_epi8(bits, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)), isInf);
    r = _mm_blendv_epi8(r, sign, isDenorm);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(inout), _mm_blendv_epi8(bits, r, dm));

    // sign bit 31 -> flag bit 1 (0x2); the rest are masked constants.
    __m128i f = _mm_srli_epi32(sign, 30);
    f = _mm_or_si128(f, _mm_and_si128(isZero, _mm_set1_epi32(0x1)));
    f = _mm_or_si128(f, _mm_and_si128(isDenorm, _mm_set1_epi32(0x5)));
    f = _mm_or_si128(f, _mm_and_si128(isInf, _mm_set1_epi32(0x8)));
    f = _mm_and_si128(f, dm);

    alignas(16) uint32_t lanes[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(lanes), f);
    laneFlags[0] = static_cast<uint8_t>(lanes[0]);
    laneFlags[1] = static_cast<uint8_t>(lanes[1]);
    laneFlags[2] = static_cast<uint8_t>(lanes[2]);
    laneFlags[3] = static_cast<uint8_t>(lanes[3]);
}

// Instruction field extraction helpers
static inline uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
static inline uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

// Lower instruction field helpers
static inline uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t VIT(uint32_t i) { return (uint8_t)((i >> 16) & 0xF); }
static inline uint8_t VIS(uint32_t i) { return (uint8_t)((i >> 11) & 0xF); }
static inline uint8_t VID(uint32_t i) { return (uint8_t)((i >> 6) & 0xF); }
static inline int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
static inline int16_t IMM15(uint32_t i)
{
    uint32_t lo11 = i & 0x7FF;
    uint32_t hi4 = (i >> 21) & 0xF;
    uint32_t raw = (hi4 << 11) | lo11;
    return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
}

#endif
