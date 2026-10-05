// ps2xRuntime: minimal assertion failure handler for the copied PCSX2 headers.
#include "common/Assertions.h"
#include <cstdio>
#include <cstdlib>
void pxOnAssertFail(const char* file, int line, const char* func, const char* msg)
{
	std::fprintf(stderr, "[microvu] ASSERT %s:%d %s: %s\n", file, line, func, msg ? msg : "");
	std::abort();
}
