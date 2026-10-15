// ps2x SIO2 port: force-included into every file of the port, standing in for PCSX2's precompiled header.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include "common/Pcsx2Types.h"
#include "common/Console.h"

// PCSX2's translation markers (UI strings only).
#define TRANSLATE_NOOP(context, str) (str)
#define TRANSLATE_FS(context, str) (str)

// PCSX2 uses {fmt} for one on-screen message in PadDualshock2.cpp; the message itself is dropped (Host shim).
namespace fmt
{
    template <typename... Args>
    inline std::string format(Args &&...) { return {}; }
}
