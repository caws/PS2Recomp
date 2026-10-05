// ps2xRuntime shim for PCSX2's Console.h: printf-style logging to stderr (DevCon is silent by default).
#pragma once
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
enum ConsoleColors { Color_Default, Color_Black, Color_Green, Color_Red, Color_Blue, Color_Magenta, Color_Orange, Color_Gray, Color_Cyan, Color_Yellow, Color_White, Color_StrongBlack, Color_StrongRed, Color_StrongGreen, Color_StrongBlue, Color_StrongMagenta, Color_StrongOrange, Color_StrongGray, Color_StrongCyan, Color_StrongYellow, Color_StrongWhite };
struct ConsoleShim
{
	const char* tag; bool enabled;
	void vout(const char* fmt, va_list ap) const { if (!enabled) return; std::fprintf(stderr, "[%s] ", tag); std::vfprintf(stderr, fmt, ap); std::fputc('\n', stderr); }
	void WriteLn(const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void WriteLn(ConsoleColors, const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void Warning(const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void Warning(ConsoleColors, const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void Error(const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void Error(ConsoleColors, const char* fmt, ...) const { va_list ap; va_start(ap, fmt); vout(fmt, ap); va_end(ap); }
	void WriteLn() const {}
};
extern ConsoleShim Console;
extern ConsoleShim DevCon;
