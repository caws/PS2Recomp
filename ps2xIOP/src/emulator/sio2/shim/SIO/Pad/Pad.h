// ps2x SIO2 port shim: PCSX2 SIO/Pad/Pad.h. PCSX2's Pad.cpp (config, bindings, profiles) is not copied; the seam
// (ps2x_sio2.cpp) owns the pads and implements the lookups the copied files call.
#pragma once
#include "Config.h"
#include "Input/InputManager.h"
#include "SIO/Pad/PadTypes.h"

class PadBase;
class StateWrapper;

namespace Pad
{
    constexpr size_t DEFAULT_EJECT_TICKS = 50;
    PadBase *GetPad(u8 port, u8 slot);
    PadBase *GetPad(const u8 unifiedSlot);
}
