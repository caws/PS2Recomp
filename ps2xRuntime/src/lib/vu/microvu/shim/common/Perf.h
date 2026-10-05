// ps2xRuntime shim for PCSX2's Perf.h: symbol registration for perf/VTune is a no-op here.
#pragma once
#include <cstddef>
namespace Perf
{
	struct Group
	{
		void Register(const void*, size_t, const char*) {}
		void RegisterPC(const void*, size_t, unsigned) {}
		void RegisterKey(const void*, size_t, const char*, unsigned long long) {}
	};
	extern Group any, vu0, vu1;
}
