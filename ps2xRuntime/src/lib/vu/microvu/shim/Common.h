// ps2xRuntime shim for PCSX2's Common.h: only what microVU needs.
#pragma once
#include "common/Pcsx2Defs.h"
#include "common/Pcsx2Types.h"
#include "common/Assertions.h"
#include "common/VectorIntrin.h"
#include "common/AlignedMalloc.h"
#include "common/Console.h"
#include "Config.h"
#include "R5900.h"
#include "VU.h"
#include "VUmicro.h"

// Savestate freeze interface (unused here; microVU.cpp defines vuJITFreeze against it).
class SaveStateBase
{
public:
	bool IsSaving() const { return false; }
	bool IsOkay() const { return true; }
	template <typename T> void Freeze(T &) {}
	bool vuJITFreeze();
};
// The recompiler code cache (allocated RWX by the seam, ../ps2x_microvu.cpp).
namespace SysMemory
{
	u8* GetVU0Rec();
	u8* GetVU0RecEnd();
	u8* GetVU1Rec();
	u8* GetVU1RecEnd();
}
