// ps2x SIO2 port shim: PCSX2 Common.h. Sio2::Interrupt raises IOP IRQ 17 through iopIntcIrq (the seam).
#pragma once
#include "common/Pcsx2Types.h"
#include "Config.h"   // PCSX2's Common.h reaches Config.h (EmuConfig) through its include chain
void iopIntcIrq(uint irq);
