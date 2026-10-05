// ps2xRuntime shim for fmt: a minimal fmt::format supporting "{}" placeholders (program logger only).
#pragma once
#include <sstream>
#include <string>
namespace fmt
{
	inline void fmt_next(std::ostringstream&, const char*& p) { while (*p) { if (p[0] == '{' && p[1] == '}') { p += 2; return; } p++; } }
	template <typename T> inline void fmt_arg(std::ostringstream& os, const char*& p, const T& v)
	{
		const char* start = p; while (*p && !(p[0] == '{' && p[1] == '}')) ++p; os.write(start, p - start); if (*p) { p += 2; os << v; }
	}
	template <typename... Args> inline std::string format(const char* f, const Args&... args)
	{
		std::ostringstream os; const char* p = f; (fmt_arg(os, p, args), ...); os << p; return os.str();
	}
}
