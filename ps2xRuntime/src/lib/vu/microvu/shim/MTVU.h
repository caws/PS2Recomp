// ps2xRuntime shim for PCSX2's MTVU.h: no VU thread (THREAD_VU1 is false); the members are stubs.
#pragma once
#include "common/Pcsx2Defs.h"
#include "Vif.h"
#include <atomic>
struct VU_Thread
{
	enum InterruptFlag
	{
		InterruptFlagFinish = 1 << 0,
		InterruptFlagSignal = 1 << 1,
		InterruptFlagLabel = 1 << 2,
		InterruptFlagVUEBit = 1 << 3,
		InterruptFlagVUTBit = 1 << 4,
	};
	VIFregisters vifRegs;
	u32 vuFBRST = 0;
	std::atomic<u32> mtvuInterrupts{0};
	bool IsOpen() const { return false; }
	void Open() {}
	void Close() {}
	void WaitVU() {}
	void Get_MTVUChanges() {}
};
extern VU_Thread vu1Thread;
