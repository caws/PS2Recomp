# PCSX2 IPU port (cont.245, 2026-09-05)

`pcsx2/IPU/` holds files copied **verbatim** from [PCSX2](https://github.com/PCSX2/pcsx2)
(`pcsx2/IPU/`, master, fetched 2026-09-05), licensed `GPL-3.0+` / `GPL-2.0+` (SPDX headers kept).
PS2Recomp is GPLv3, so the import is license-compatible.

| copied | what |
|---|---|
| `IPU.cpp`, `IPU.h` | the register block (CMD/CTRL/BP/TOP), the command dispatcher (BCLR/IDEC/BDEC/VDEC/FDEC/SETIQ/SETVQ/CSC/PACK/SETTH), the soft reset |
| `IPU_Fifo.cpp`, `IPU_Fifo.h` | the 8-qword input/output FIFOs and the bit-position reader (`tIPU_BP`) |
| `IPUdma.cpp`, `IPUdma.h` | DMA channel 3 (fromIPU) and 4 (toIPU): source-chain tags, the FIFO handshake, the end-of-transfer interrupts |
| `IPU_MultiISA.cpp`, `IPU_MultiISA.h`, `mpeg2_vlc.h` | the MPEG-2 decoder: macroblock modes/address increment, DC/AC VLC, the IDCT, the resumable IDEC/BDEC slice state machines, CSC/PACK/VQ, the worker |
| `yuv2rgb.cpp`, `yuv2rgb.h`, `IPUdither.cpp` | the BT.601 integer colour conversion (SSE2) and the RGB16 dither |

Nothing in `pcsx2/` is modified.

`shim/` replaces the PCSX2 headers those files include but this runtime does not have (`Common.h`,
`Config.h`, `GS/MultiISA.h`) with the minimum they need; the PCSX2 base-type headers
(`common/Pcsx2Defs.h`, `Pcsx2Types.h`, `SingleRegisterTypes.h`, `Assertions.h`) are the copies the
microVU port already carries (`src/lib/vu/microvu/pcsx2/common`).

`ps2x_ipu.{h,cpp}` is the seam: the EE-side state PCSX2 keeps in its core (`cpuRegs`, the hardware
register page `eeHw`, the DMA channels `ipu0ch`/`ipu1ch`, `dmacRegs`), the source-chain tag helpers
(`hwDmacSrcChain`, `hwDmacSrcTadrInc`, `dmaGetAddr`), the interrupt lines (`hwIntcIrq`, `hwDmacIrq`)
and the event scheduler (`CPU_INT`) -- replaced by a synchronous pump that runs the scheduled IPU/DMA
events to quiescence inside the guest store that caused them -- plus the entry points
`ps2_memory.cpp` calls for the 0x10002000 registers, the 0x10007000 FIFOs and the CHCR kicks of
DMA channels 3 and 4.

Why: LOTR's frontend backgrounds are `ipum` sections -- 512x512 MPEG-2 intra pictures the game
decodes strip by strip with FDEC/IDEC through the IPU (`sub_00143B20`). The runtime's IPU was a
register stub, so the raw bitstreams were uploaded as pixels (noise with a 16-px block grid).
`PS2X_IPU=0` restores the stub for A/B.
