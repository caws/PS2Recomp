// ps2x SIO2 port shim: PCSX2 Host.h -- only the on-screen-message call PadDualshock2 makes (analog toggle); dropped.
#pragma once
#include <string>
namespace Host
{
    static constexpr float OSD_INFO_DURATION = 3.0f;
    template <typename... Args>
    inline void AddKeyedOSDMessage(Args &&...) {}
}
