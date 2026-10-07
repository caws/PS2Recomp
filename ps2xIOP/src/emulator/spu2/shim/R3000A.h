// ps2x SPU2 port shim: the IOP CPU state the SPU2 core reads -- only the cycle counter (PCSX2 psxRegs.cycle, IOP cycles
// at 36.864 MHz). The seam keeps it equal to the emulated IOP's cycle count before every SPU2 call.
#pragma once
#include "common/Pcsx2Types.h"
struct Ps2xSpu2IopRegs
{
    u32 cycle = 0;
};
extern Ps2xSpu2IopRegs psxRegs;
