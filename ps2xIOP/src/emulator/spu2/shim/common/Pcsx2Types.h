// ps2x SPU2 port shim: the PCSX2 base types the SPU2 core uses (pcsx2 common/Pcsx2Types.h + Pcsx2Defs.h subset).
#pragma once
#include <cstddef>
#include <cstdint>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using uptr = std::uintptr_t;
using sptr = std::intptr_t;
using uint = unsigned int;

#ifndef __forceinline
// PCSX2 common/Pcsx2Defs.h (GCC): the attribute WITHOUT `inline`, so a __forceinline function defined in one file
// (spu2sys.cpp TimeUpdate, CounterUpdate, GetMemPtr) still has an external definition the other files link to.
#define __forceinline __attribute__((always_inline, unused))
#endif
#ifndef __fi
#define __fi __forceinline
#endif
#ifndef __ri
#define __ri __attribute__((noinline))
#endif
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

static constexpr bool IsDevBuild = false;
static constexpr bool IsDebugBuild = false;

// PCSX2 common/Pcsx2Defs.h: a switch default the code declares unreachable. Release PCSX2 makes it an optimiser hint;
// here it is a plain break (identical when the assumption holds, and harmless if it does not).
#define jNO_DEFAULT \
    default:        \
    {               \
        break;      \
    }
