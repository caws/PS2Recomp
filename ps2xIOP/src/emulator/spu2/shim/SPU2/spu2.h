// ps2x SPU2 port shim: replaces pcsx2/SPU2/spu2.h (whose spu2.cpp is the PCSX2 host glue: output streams, config,
// capture). The functions the SPU2 core calls across files are declared here and defined by the core or the seam.
#pragma once
#include "common/Pcsx2Types.h"

namespace SPU2
{
    static constexpr u32 SAMPLE_RATE = 48000;
    static constexpr u32 PSX_SAMPLE_RATE = 44100;
    bool IsRunningPSXMode();
    u32 GetConsoleSampleRate();
}

void SPU2write(u32 mem, u16 value);
u16 SPU2read(u32 mem);
void SPU2readDMA4Mem(u16 *pMem, u32 size);
void SPU2writeDMA4Mem(u16 *pMem, u32 size);
void SPU2interruptDMA4();
void SPU2interruptDMA7();
void SPU2readDMA7Mem(u16 *pMem, u32 size);
void SPU2writeDMA7Mem(u16 *pMem, u32 size);

extern u64 lClocks;
extern void CounterUpdate(u32 DMAICounter);
extern void TimeUpdate(u64 cClocks);
extern void SPU2_FastWrite(u32 rmem, u16 value);
