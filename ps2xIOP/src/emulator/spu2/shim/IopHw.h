// ps2x SPU2 port shim: the IOP DMA channel 4/7 address registers the SPU2 core reads and advances (PCSX2 IopHw.h
// HW_DMA4_MADR = psxHu32(0x10c0), HW_DMA4_TADR = psxHu32(0x10cc), DMA7 at 0x1500 / 0x150c). The seam keeps them in
// step with the emulated IOP's registers around every SPU2 call.
#pragma once
#include "common/Pcsx2Types.h"
struct Ps2xSpu2DmaRegs
{
    u32 dma4Madr = 0, dma4Tadr = 0, dma7Madr = 0, dma7Tadr = 0;
};
extern Ps2xSpu2DmaRegs ps2xSpu2Dma;
#define HW_DMA4_MADR (ps2xSpu2Dma.dma4Madr)
#define HW_DMA4_TADR (ps2xSpu2Dma.dma4Tadr)
#define HW_DMA7_MADR (ps2xSpu2Dma.dma7Madr)
#define HW_DMA7_TADR (ps2xSpu2Dma.dma7Tadr)
// IOP RAM as the SPU2's DMA source/destination (PCSX2 iopPhysMem).
void *iopPhysMem(u32 address);
