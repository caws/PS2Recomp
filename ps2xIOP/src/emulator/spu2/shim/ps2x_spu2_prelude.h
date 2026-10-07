// ps2x SPU2 port: force-included into every SPU2 core file (-include), standing in for PCSX2's precompiled header
// (common types, assertions, the C library) and its build-wide _M_SSE level (the runtime builds with -msse4.1).
#pragma once
#define _M_SSE 0x401
#include "common/Pcsx2Types.h"
#include "common/Assertions.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
