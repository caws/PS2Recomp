# PCSX2 microVU port (cont.230, 2026-09-02)

`pcsx2/` holds files copied **verbatim** from [PCSX2](https://github.com/PCSX2/pcsx2) (master,
fetched 2026-09-02), licensed `GPL-3.0+` (SPDX headers kept). PS2Recomp is GPLv3, so the import is
license-compatible.

| copied | from |
|---|---|
| `pcsx2/x86/microVU*.{cpp,h,inl}` | `pcsx2/x86/` — the VU recompiler |
| `pcsx2/VU.h`, `VUops.h`, `VUmicro.h` | `pcsx2/` — the VU register file and CPU interface |
| `pcsx2/common/emitter/**` | `common/emitter/` — the x86-64 emitter |
| `pcsx2/common/*.h`, `AlignedMalloc.cpp` | `common/` — base types, assertions, intrinsics, FP control |

**Modified copies** (documented in place):
- `pcsx2/x86/microVU_Macro.inl` — replaced by a stub: COP2 macro mode needs the EE recompiler
  (`iR5900`, `vtlb`), which this runtime does not have (the EE is statically recompiled, VU0 keeps
  its interpreter).

`shim/` replaces the PCSX2 headers microVU includes but this runtime does not have (`Common.h`,
`Config.h`, `R5900.h`, `iR5900.h`, `iCore.h`, `Vif.h`, `Gif.h`, `Gif_Unit.h`, `MTVU.h`, `GS.h`,
`GS/MultiISA.h`, `common/Console.h`, `common/Perf.h`, …) with the minimum those includes need.
`ps2x_microvu.{h,cpp}` is the seam: the globals PCSX2 code expects (`EmuConfig`, `cpuRegs`,
`vuRegs`, `vif1Regs`, `vu1Thread`, `gifUnit`), the RWX code cache, XGKICK routed to our GIF arbiter,
and the entry points the VU1 interpreter calls.

Design and rationale: `docs/vu1-program-compiler.md`; the survey of what is being ported:
`docs/microvu-survey.md`.
