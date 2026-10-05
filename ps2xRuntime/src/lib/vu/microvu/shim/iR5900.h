// ps2xRuntime shim for PCSX2's iR5900.h: the text-pointer base used by the VU dispatcher and the
// fastmem base register name (only consulted in COP2 mode, never set here).
#pragma once
#include "R5900.h"
#include "iCore.h"
#define R5900_TEXTPTR (&cpuRegs.GPR.r[9])
#define RFASTMEMBASE x86Emitter::rbp
