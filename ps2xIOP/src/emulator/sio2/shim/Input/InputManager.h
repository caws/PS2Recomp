// ps2x SIO2 port shim: PCSX2 Input/InputManager.h -- PadDualshock2 reports rumble here (seam: ps2x_sio2.cpp).
#pragma once
#include "common/Pcsx2Types.h"
namespace InputManager
{
    void SetPadVibrationIntensity(u32 pad_index, float large_or_single_motor_intensity, float small_motor_intensity);
}
