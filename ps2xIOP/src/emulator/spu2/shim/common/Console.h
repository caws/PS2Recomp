// ps2x SPU2 port shim: PCSX2's Console/DevCon, routed to stderr (only reached from diagnostic paths).
#pragma once
#include "common/Pcsx2Types.h"
#include <cstdarg>
#include <cstdio>

struct Ps2xSpu2ConsoleShim
{
    void WriteLn(const char *fmt, ...) const
    {
        va_list args;
        va_start(args, fmt);
        std::fputs("[spu2] ", stderr);
        std::vfprintf(stderr, fmt, args);
        std::fputc('\n', stderr);
        va_end(args);
    }
    void Warning(const char *fmt, ...) const
    {
        va_list args;
        va_start(args, fmt);
        std::fputs("[spu2:warning] ", stderr);
        std::vfprintf(stderr, fmt, args);
        std::fputc('\n', stderr);
        va_end(args);
    }
    void Error(const char *fmt, ...) const
    {
        va_list args;
        va_start(args, fmt);
        std::fputs("[spu2:error] ", stderr);
        std::vfprintf(stderr, fmt, args);
        std::fputc('\n', stderr);
        va_end(args);
    }
};
// static: the microVU port defines its own Console/DevCon (internal linkage keeps the two shims apart).
static constexpr Ps2xSpu2ConsoleShim Console{};
static constexpr Ps2xSpu2ConsoleShim DevCon{};
