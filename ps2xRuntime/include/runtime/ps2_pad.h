#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);

    // cont.346c: which host device backs each PS2 controller port.
    //   gamepadForPort(n) = the n-th AVAILABLE raylib gamepad (-1 = none).
    //   portHasDevice(n)  = is this port driven by something real? The analogue of PCSX2's
    //                       Pad::HasConnectedPad() / ControllerType::NotConnected
    //                       (pcsx2/SIO/Pad/Pad.cpp). The game-side co-op gate consults this so
    //                       port 1 is only ever reported CONNECTED when a real device drives it.
    static int gamepadForPort(int port);
    static bool portHasDevice(int port);

    // rumble: the DualShock 2 actuator bytes the game last sent for `port` -- `small` is the
    // on/off motor (bit 0), `large` the 0-255 motor. Callable from ANY thread (the EE thread's
    // pad HLE); it only records them. pumpVibration() applies them on the MAIN thread, where
    // raylib/SDL may be called, once per presenter-loop iteration. A port with no gamepad (the
    // keyboard) has no motor and gets nothing.
    static void setVibration(int port, uint8_t small, uint8_t large);
    static void pumpVibration();
};

#endif
