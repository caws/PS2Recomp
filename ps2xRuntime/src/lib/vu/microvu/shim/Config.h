// ps2xRuntime shim for PCSX2's Config.h: the EmuConfig fields microVU reads, with our defaults.
#pragma once
#include "common/Pcsx2Defs.h"
#include "common/FPControl.h"

struct Pcsx2Config
{
	struct RecompilerOptions
	{
		bool EnableVU0 = false, EnableVU1 = true, EnableEE = false, EnableFastmem = false;
		bool vu0Overflow = true, vu0ExtraOverflow = false, vu0SignOverflow = false, vu0Underflow = false;
		bool vu1Overflow = true, vu1ExtraOverflow = false, vu1SignOverflow = false, vu1Underflow = false;
	};
	struct CpuOptions
	{
		RecompilerOptions Recompiler;
		FPControlRegister FPUFPCR;
		FPControlRegister VU0FPCR;
		FPControlRegister VU1FPCR;
	};
	struct SpeedhackOptions
	{
		bool vuFlagHack = false; // PCSX2 default is ON; set from PS2X_MICROVU_FLAGHACK (default ON) in ps2x_microvu.cpp
		bool vuThread = false;
		bool vu1Instant = false;
		s8 EECycleRate = 0;
		u8 EECycleSkip = 0;
	};
	struct GamefixOptions
	{
		bool IbitHack = false, VUSyncHack = false, FullVU0SyncHack = false, VuAddSubHack = false,
		     XgKickHack = false, VUOverflowHack = false;
	};
	CpuOptions Cpu;
	SpeedhackOptions Speedhacks;
	GamefixOptions Gamefixes;
};
extern Pcsx2Config EmuConfig;

#define REC_VU1 true
#define THREAD_VU1 false
#define INSTANT_VU1 (EmuConfig.Speedhacks.vu1Instant)
#define CHECK_FASTMEM false
#define CHECK_VUADDSUBHACK (EmuConfig.Gamefixes.VuAddSubHack)
#define CHECK_XGKICKHACK (EmuConfig.Gamefixes.XgKickHack)
#define CHECK_VUOVERFLOWHACK (EmuConfig.Gamefixes.VUOverflowHack)
#define CHECK_VU_FLAGHACK (EmuConfig.Speedhacks.vuFlagHack)
#define CHECK_VU_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0Overflow : EmuConfig.Cpu.Recompiler.vu1Overflow)
#define CHECK_VU_EXTRA_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0ExtraOverflow : EmuConfig.Cpu.Recompiler.vu1ExtraOverflow)
#define CHECK_VU_SIGN_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0SignOverflow : EmuConfig.Cpu.Recompiler.vu1SignOverflow)
#define CHECK_VU_UNDERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0Underflow : EmuConfig.Cpu.Recompiler.vu1Underflow)
