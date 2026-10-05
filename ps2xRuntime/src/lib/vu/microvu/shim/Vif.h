// ps2xRuntime shim for PCSX2's Vif.h: the VIF registers a VU program reads (XTOP/XITOP).
#pragma once
#include "common/Pcsx2Defs.h"
union tVIF_STAT { struct { u32 VPS : 2, VEW : 1, VGW : 1, _r0 : 2, MRK : 1, DBF : 1, VSS : 1, VFS : 1, VIS : 1, INT : 1, ER0 : 1, ER1 : 1, _r1 : 9, FQC : 5, _r2 : 3; }; u32 _u32 = 0; };
struct VIFregisters
{
	tVIF_STAT stat;
	u32 top = 0;  // read as ptr16 by XTOP
	u32 itop = 0; // read as ptr16 by XITOP
};
extern VIFregisters vif0Regs, vif1Regs;
