# PCSX2 SIO2 + DualShock 2 port (amo_recomp, 2026-10-15)

`pcsx2/SIO/` holds files copied **verbatim** from [PCSX2](https://github.com/PCSX2/pcsx2) (master `685637d`,
fetched 2026-10-15), licensed `GPL-3.0+` — the same arrangement as the SPU2 and USB ports (`../spu2/`, `../usb/`).

| copied | what |
|---|---|
| `Sio2.cpp`, `Sio2.h`, `SioTypes.h` | the SIO2 command engine: SEND3 queue, FIFOs, PAD / MULTITAP / INFRARED / MEMCARD routing, IRQ 17 |
| `Pad/PadBase.*`, `Pad/PadDualshock2.*`, `Pad/PadNotConnected.*`, `Pad/PadTypes.h` | the DualShock 2 protocol (poll, config/analog modes, pressure, vibration map) and the empty port |
| `Multitap/MultitapProtocol.*` | multitap protocol (not attached: `IsMultitapPortEnabled` is false) |
| `Memcard/MemoryCardProtocol.*` | memory-card protocol (no card yet: `_mcd::IsPresent` is false, every card command answers DISCONNECTED) |

**Not copied:** `SIO/Pad/Pad.cpp` (PCSX2's bindings, profiles, settings), `SIO/Sio.cpp`, `SIO/Memcard/MemoryCardFile*`,
the other controllers. `ps2x_sio2.cpp` is the seam, carrying the PCSX2 pieces the copied files expect from elsewhere:
the SIO2 register map from `ps2/Iop/IopHwRead.cpp` / `IopHwWrite.cpp` (32-bit SEND1/2/3, CTRL, RECV1-3, ISTAT; 8-bit TX/RX),
`IopDma.cpp psxDma11/psxDma12`, `Sio.cpp`'s pad/slot mapping, and `Pad.cpp`'s pad ownership (`Pad::GetPad`).

`shim/` stands in for the PCSX2 headers they include (`Common.h`, `Config.h` — the binding/setting table types kept so the
tables compile verbatim, `Host.h`, `StateWrapper.h` — no save states, `SIO/Sio.h` — `_mcd` without the file backend,
`SIO/Sio0.h`, `SIO/Pad/Pad.h`, `Input/InputManager.h`, `IconsPromptFont.h`, `common/*`); `shim/ps2x_sio2_prelude.h` is
force-included in place of PCSX2's precompiled header.

**Input:** ports 1A and 2A get a DualShock 2 while the runtime's pad backend (`PSPadBackend::readState`: keyboard,
gamepad, `PS2X_PAD_SCRIPT`) drives them, and `PadNotConnected` otherwise; the state is sampled once per SIO2 transfer (the
CTRL START write), translated from libpad's raw buffer into `PadDualshock2::Set` / `SetRawAnalogs`. Rumble arrives at
`InputManager::SetPadVibrationIntensity` → `Host::padVibration` (not wired to the host yet).

`IopMemory` routes `0x1F808200-0x1F808283` and DMA 11/12 (`CHCR` start → transfer now, interrupt `0x2C`/`0x2D` after
size/4 cycles, as PCSX2's `PSX_INT(IopEvt_Dma11/12, size >> 2)`); IRQ 17 is queued on the emulator's pending-interrupt
map. **On by default** — every console has it; `PS2X_IOP_SIO2=0` restores the plain register store.

Why: Auto Modellista runs its real `DBCMAN.IRX` / `SIO2D.IRX` / `DS1O.IRX` / `SIO2MAN.IRX` (and `MCMAN`/`MCSERV`) on the
emulated IOP; without SIO2 the pad never answered and the boot sat at LANGUAGE SELECT.
