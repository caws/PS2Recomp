// ps2x SPU2 port (rotk row 273): the seam between PCSX2's SPU2 core (pcsx2/SPU2, GPL-3.0+, see README.md) and the
// emulated IOP. The core is a process-wide singleton, as in PCSX2; one IOP drives it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ps2x::spu2
{
    // What the core needs from the IOP: its RAM (DMA), and two interrupt paths. Both are QUEUED by the IOP (never run
    // nested inside the register access or mixer tick that raised them).
    class Host
    {
    public:
        virtual ~Host() = default;
        virtual uint8_t *iopRam() = 0;
        virtual void spu2Interrupt() = 0;            // IOP IRQ 9 (PCSX2 IopIrq.cpp spu2Irq)
        virtual void spu2DmaComplete(int core) = 0;  // DMA 4 (core 0) / DMA 7 (core 1) finished: clear CHCR busy, raise it
    };

    void attach(Host *host);
    // Power-on state (PCSX2 SPU2::Open + InternalReset(false)); `iopCycle` = the IOP cycle count now.
    void reset(uint64_t iopCycle);

    // The IOP's cycle count (36.864 MHz); the core mixes one 48 kHz output sample per 768 cycles of it.
    void setCycle(uint64_t iopCycle);
    void advance(uint64_t iopCycle);   // PCSX2 SPU2async: mix up to now

    uint16_t read16(uint32_t address);
    void write16(uint32_t address, uint16_t value);

    // IOP DMA channel 4 / 7 started (PCSX2 IopDma.cpp psxDmaGeneric). madr/bcr/chcr as written to the channel.
    void dmaStart(int core, uint32_t madr, uint32_t bcr, uint32_t chcr);
    // The channel registers the core reads and advances (PCSX2 HW_DMA4/7_MADR/TADR).
    uint32_t &dmaMadr(int core);
    uint32_t &dmaTadr(int core);

    // Mixed output, 48 kHz interleaved stereo s16, oldest first; returns frames copied (frames = stereo pairs).
    size_t takeSamples(int16_t *destination, size_t maxFrames);
    uint64_t samplesMixed();
}
