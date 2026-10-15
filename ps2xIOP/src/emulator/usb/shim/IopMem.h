// ps2x USB (OHCI) port shim: PCSX2's IopMem.h for usb-ohci.cpp, which reads the guest's endpoint/transfer
// descriptors straight from IOP RAM as `iopMem->Main + addr`. `iopMem` is a macro onto the seam (not a global),
// so it cannot collide with another port's shim.
#pragma once
#include "common/Pcsx2Types.h"

// PCSX2 pcsx2/Memory.h: ExposedIopRam, the IOP RAM size usb-ohci.cpp bounds its descriptor/buffer accesses with (2 MB).
namespace Ps2MemSize
{
    static constexpr u32 ExposedIopRam = 0x00200000;
}

struct Ps2xUsbIopMem
{
    u8 *Main;
};
Ps2xUsbIopMem *ps2xUsbIopMem();
#define iopMem (ps2xUsbIopMem())
