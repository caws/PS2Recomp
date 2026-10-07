// ps2x SPU2 port shim: the IOP DMA hooks the SPU2 core calls (PCSX2 IopDma.cpp); implemented in the seam.
#pragma once
void spu2DMA4Irq();
void spu2DMA7Irq();
void spu2Irq();
