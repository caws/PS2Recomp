// The seam between PCSX2's OHCI model (pcsx2/USB/qemu-usb/, verbatim) and ps2xIOP's emulator: the parts of
// PCSX2 pcsx2/USB/USB.cpp (USBopen/USBreset/USBread32/USBwrite32, usb_get_clock, usb_get_ticks_per_second, the
// g_usb_* globals) and pcsx2/IopIrq.cpp (usbIrq) the controller needs. USBasync is not wired: with no device
// attached PCSX2's returns immediately (`if (!s_usb_device[0] && !s_usb_device[1]) return;`).
#include "ps2x_usb.h"

#include "IopMem.h"
#include "USB/qemu-usb/USBinternal.h"

int64_t g_usb_frame_time = 0;
int64_t g_usb_bit_time = 0;
int64_t g_usb_last_cycle = 0;

namespace
{
    ps2x::usb::Host *s_host = nullptr;
    OHCIState *s_ohci = nullptr;
    Ps2xUsbIopMem s_iopMem{};
    constexpr int kPsxClock = 36864000;   // PCSX2 PSXCLK: the IOP clock, usb_get_ticks_per_second()
}

Ps2xUsbIopMem *ps2xUsbIopMem()
{
    s_iopMem.Main = s_host ? s_host->iopRam() : nullptr;
    return &s_iopMem;
}

int usb_get_ticks_per_second()
{
    return kPsxClock;
}

s64 usb_get_clock()
{
    return s_host ? static_cast<s64>(s_host->iopCycles()) : 0;
}

void usbIrq(int cycles)
{
    if (s_host)
        s_host->usbInterrupt(cycles);
}

namespace ps2x::usb
{
    void attach(Host *host)
    {
        s_host = host;
        if (!s_ohci)
            s_ohci = ohci_create(kRegBase, 2);
    }

    void reset()
    {
        g_usb_last_cycle = 0;
        if (s_ohci)
            ohci_hard_reset(s_ohci);
    }

    uint32_t read32(uint32_t address)
    {
        return s_ohci ? ohci_mem_read(s_ohci, address) : 0u;
    }

    void write32(uint32_t address, uint32_t value)
    {
        if (s_ohci)
            ohci_mem_write(s_ohci, address, value);
    }
}
