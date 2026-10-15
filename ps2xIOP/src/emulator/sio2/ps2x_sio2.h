#pragma once

#include <cstddef>
#include <cstdint>

// PCSX2's SIO2 (pcsx2/SIO/, verbatim; see README.md): the IOP's controller/memory-card serial interface at
// 0x1F808200-0x1F808283, IRQ 17, DMA 11 (SIO2 in) / DMA 12 (SIO2 out). Ports 1A and 2A carry a DualShock 2 when the
// runtime's pad backend drives them (keyboard, gamepad, PS2X_PAD_SCRIPT), else nothing; no multitap and -- in this
// first cut -- no memory card (every card command answers DISCONNECTED, PCSX2's empty-slot behaviour).
namespace ps2x::sio2
{
    struct Host
    {
        virtual ~Host() = default;
        virtual uint8_t *iopRam() = 0;
        virtual void sio2Interrupt() = 0;   // IOP IRQ 17 (PCSX2 Sio2::Interrupt -> iopIntcIrq(17))
        // The runtime's pad backend: libpad's raw buffer (bytes 2-3 buttons active-low, 4-7 rx ry lx ly); false = no pad.
        virtual bool padState(int port, uint8_t *raw, size_t size) = 0;
        virtual void padVibration(int port, float large, float small) { (void)port; (void)large; (void)small; }
    };

    constexpr uint32_t kRegBase = 0x1F808200u;
    constexpr uint32_t kRegEnd = 0x1F808284u;
    constexpr uint32_t kTx = 0x1F808260u;   // HW_SIO2_TX: 8-bit command bytes in
    constexpr uint32_t kRx = 0x1F808264u;   // HW_SIO2_RX: 8-bit response bytes out

    void attach(Host *host);
    void reset();
    [[nodiscard]] uint32_t read32(uint32_t address);
    void write32(uint32_t address, uint32_t value);
    [[nodiscard]] uint8_t read8(uint32_t address);
    void write8(uint32_t address, uint8_t value);

    // PCSX2 IopDma.cpp psxDma11 / psxDma12. Return the transfer size in words (0 = CHCR mode not handled, no transfer).
    uint32_t dma11(uint32_t &madr, uint32_t bcr, uint32_t chcr);
    uint32_t dma12(uint32_t &madr, uint32_t bcr, uint32_t chcr);
}
