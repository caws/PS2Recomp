// ps2x SPU2 port shim: PCSX2's IOP counter 6 is the SPU2's own wake-up timer (IopCounters.cpp). CounterUpdate() only
// shortens it for DMA timing; here the seam clocks the SPU2 every IOP slice, so these are bookkeeping only.
#pragma once
#include "common/Pcsx2Types.h"
struct Ps2xSpu2Counter
{
    u32 startCycle = 0;
    s32 deltaCycles = 0;
    u32 rate = 0;
};
extern Ps2xSpu2Counter psxCounters[8];
extern s32 psxNextDeltaCounter;
extern u32 psxNextStartCounter;
