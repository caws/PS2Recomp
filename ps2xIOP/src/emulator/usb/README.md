# PCSX2 USB (OHCI) port (amo_recomp, 2026-10-15)

`pcsx2/USB/qemu-usb/` holds files copied **verbatim** from [PCSX2](https://github.com/PCSX2/pcsx2) (master `685637d`,
fetched 2026-10-15), licensed `GPL-3.0+` (headers kept) — the same arrangement as the SPU2 port (`../spu2/README.md`).

| copied | what |
|---|---|
| `usb-ohci.cpp`, `USBinternal.h` | the OHCI host controller: registers, soft/hard reset, root hub, frame/list processing |
| `qusb.h`, `queue.h`, `core.cpp`, `bus.cpp` | QEMU's USB core the controller is written against (packets, ports, endpoints) |

**Not copied:** PCSX2's devices (`usb-pad`, `usb-hid`, `usb-mic`, ...) and `pcsx2/USB/USB.cpp` (host glue, config,
save states). `ps2x_usb.cpp` is the seam: the `USB.cpp` parts the controller needs (`USBopen` = `ohci_create(0x1f801600,
2)`, `USBreset` = `ohci_hard_reset`, `USBread32`/`USBwrite32`, `usb_get_clock` = the IOP cycle count,
`usb_get_ticks_per_second` = `PSXCLK`, the `g_usb_*` timing globals) and `IopIrq.cpp usbIrq` (IRQ 22, queued on the
emulator's pending-interrupt map after the requested delay, never run nested).

`shim/` stands in for the PCSX2 headers the files include: `common/Pcsx2Types.h`, `common/Console.h` (to stderr) and
`IopMem.h` (`iopMem->Main` = IOP RAM, a macro onto the seam so it cannot collide with another port's symbols).

**No device is attached** (both root-hub ports empty, a console with nothing plugged in). PCSX2's `USBasync` returns at
once in that case, so — as in PCSX2 — no frames are run and no transfer lists are walked; the controller only answers
register accesses. That is what `USBD.IRX` needs to initialise: its reset (USBD +0x35cc) sets `HcCommandStatus.HCR` and
polls for the controller to clear it, and a plain register store never did, so USBD backed out, every
`sceUsbdRegisterLdd` failed, and modules that need it (Auto Modellista's `LGDEV.IRX`, the Logitech wheel driver) unloaded
before registering their RPC server (sid `0x046D046D`), leaving the EE's `lgDevInit` in its bind loop.

`IopMemory` routes `0x1F801600-0x1F8016FF` here (32-bit; sub-word accesses arrive as read-modify-write of the word,
where PCSX2 returns 0 / drops them). **On by default** — it is hardware every console has; `PS2X_IOP_USB=0` restores
the plain register store for A/B.
