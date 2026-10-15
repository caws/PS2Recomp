// ps2x SIO2 port shim: PCSX2 SIO/Sio0.h -- the PS1 serial port. PadDualshock2 pulses its ACK line, which matters
// only to PS1-mode software (SIO0); on the PS2 side SIO2 ignores it.
#pragma once
#include "common/Pcsx2Types.h"
struct Ps2xSio0Shim
{
    void SetAcknowledge(bool) {}
};
inline Ps2xSio0Shim g_Sio0;
