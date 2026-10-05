// ps2xRuntime shim for PCSX2's R5900.h: the EE register block microVU touches (text pointer base, cycle).
#pragma once
#include "common/Pcsx2Defs.h"
#include "common/SingleRegisterTypes.h"
struct GPRregs { u128 r[32]; };
struct alignas(16) cpuRegisters
{
	GPRregs GPR;
	u32 cycle = 0;
	u32 code = 0;
};
extern cpuRegisters cpuRegs;
extern void hwIntcIrq(int n);
