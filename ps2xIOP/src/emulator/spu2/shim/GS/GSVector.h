// ps2x SPU2 port shim: the GSVector4i operations ReverbResample.cpp's SSE path uses (PCSX2 GS/GSVector4i.h, same
// intrinsics: load/loadu, _mm_mulhrs_epi16, _mm_adds_epi16, _mm_hadds_epi16). The AVX2 path is not built (_M_SSE 0x401).
#pragma once
#include "common/Pcsx2Types.h"
#include <immintrin.h>

class GSVector4i
{
public:
    union
    {
        __m128i m;
        s16 I16[8];
    };

    GSVector4i() = default;
    explicit GSVector4i(__m128i value) : m(value) {}

    template <bool aligned>
    static GSVector4i load(const void *p)
    {
        return GSVector4i(aligned ? _mm_load_si128(static_cast<const __m128i *>(p)) : _mm_loadu_si128(static_cast<const __m128i *>(p)));
    }
    GSVector4i mul16hrs(const GSVector4i &v) const { return GSVector4i(_mm_mulhrs_epi16(m, v.m)); }
    GSVector4i adds16(const GSVector4i &v) const { return GSVector4i(_mm_adds_epi16(m, v.m)); }
    GSVector4i hadds16(const GSVector4i &v) const { return GSVector4i(_mm_hadds_epi16(m, v.m)); }
};
