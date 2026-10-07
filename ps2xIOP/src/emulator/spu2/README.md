# PCSX2 SPU2 port (rotk row 273, 2026-10-07)

`pcsx2/SPU2/` holds files copied **verbatim** from [PCSX2](https://github.com/PCSX2/pcsx2) (master `e487304`,
fetched 2026-10-07), licensed `GPL-3.0+` (SPDX headers kept). PS2Recomp is GPLv3, so the import is license-compatible
(same as the microVU port, `ps2xRuntime/src/lib/vu/microvu/`).

| copied | what |
|---|---|
| `spu2sys.cpp`, `RegTable.cpp`, `regs.h`, `defs.h` | the two cores, their register file, voice/key-on logic, `TimeUpdate` |
| `Mixer.cpp`, `ADSR.cpp`, `interpolate_table.h` | ADPCM decode, envelopes, interpolation, the mix |
| `Reverb.cpp`, `ReverbResample.cpp` | reverb (SSE path, via `shim/GS/GSVector.h`) |
| `Dma.cpp`, `Dma.h`, `ReadInput.cpp` | DMA 4/7 into sound RAM, AutoDMA streaming input |
| `Debug.h`, `spdif.h` | as included (debug compiled out: no `PCSX2_DEVBUILD`) |

**Not copied:** `spu2.cpp` (PCSX2's host glue: AudioStream, config, capture, save states), `Debug.cpp`,
`Wavedump_wav.cpp`, `spu2freeze.cpp`. `ps2x_spu2.cpp` is the seam: the `spu2.cpp` wrappers the core needs
(`SPU2read/write`, the DMA entry points, `spu2Output` + its DC filter, kept verbatim) and PCSX2's IOP side
(`IopDma.cpp psxDmaGeneric`, `spu2DMA4Irq/7Irq`, `IopIrq.cpp spu2Irq` = IRQ 9) against ps2xIOP's emulator.

`shim/` replaces the PCSX2 headers the core includes (`common/Pcsx2Types.h`, `Console.h`, `Assertions.h`, `Config.h`,
`GS/MultiISA.h`, `GS/GSVector.h`, `R3000A.h`, `IopCounters.h`, `IopDma.h`, `IopHw.h`, `SPU2/spu2.h`) with the minimum
they need; `shim/ps2x_spu2_prelude.h` is force-included, standing in for PCSX2's precompiled header.

Coupling: `psxRegs.cycle` = the IOP's cycle count (set before every instruction while enabled; the core mixes one
48 kHz sample per 768 IOP cycles); `IopMemory` routes `0x1F900000-0x1F90FFFF` and IOP DMA 4/7 (`MADR`/`TADR` backed
by the core) to the seam; completions and IRQ 9 are queued on the emulator's interrupt map, never run nested.
**Off by default** (`IopSubsystem::setSpu2Enabled`); upstream's register store + fixed-delay SPU DMA otherwise.
