#pragma once

#include <cstdint>

// PCSX2's USB OHCI controller (pcsx2/USB/qemu-usb/usb-ohci.cpp, see README.md) as the IOP's USB host at
// 0x1F801600-0x1F8016FF, IRQ 22. No USB devices are attached (both root-hub ports empty, as a console with nothing
// plugged in), so -- exactly as PCSX2's USBasync, which returns early without devices -- the controller never runs
// frames: it answers register reads/writes (resets, root-hub port status, start/stop). USBD.IRX's init needs that:
// it sets HcCommandStatus.HCR and polls for the controller to clear it.
namespace ps2x::usb
{
    struct Host
    {
        virtual ~Host() = default;
        virtual uint8_t *iopRam() = 0;
        virtual void usbInterrupt(int delayCycles) = 0;   // IOP IRQ 22 (PCSX2 IopIrq.cpp usbIrq -> IopEvt_USB)
        virtual uint64_t iopCycles() const = 0;
    };

    constexpr uint32_t kRegBase = 0x1F801600u;
    constexpr uint32_t kRegEnd = 0x1F801700u;

    void attach(Host *host);   // creates the controller on first use (PCSX2 USBopen: ohci_create(0x1f801600, 2))
    void reset();              // PCSX2 USBreset: ohci_hard_reset
    [[nodiscard]] uint32_t read32(uint32_t address);
    void write32(uint32_t address, uint32_t value);
}
