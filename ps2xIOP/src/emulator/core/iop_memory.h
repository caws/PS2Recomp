#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory
    {
    public:
        static constexpr uint32_t RamSize = 2u * 1024u * 1024u;
        static constexpr uint32_t ScratchBase = 0x1F800000u;
        static constexpr uint32_t ScratchSize = 0x400u;
        static constexpr uint32_t HardwareBase = 0x1F801000u;
        static constexpr uint32_t HardwareEnd = 0x1F900000u;
        static constexpr uint32_t Spu2Base = 0x1F900000u;
        static constexpr uint32_t Spu2End = 0x1FA00000u;
        static constexpr uint32_t SifBase = 0x1D000000u;
        static constexpr uint32_t SifEnd = 0x1D001000u;
        // One pool for module images AND AllocSysMemory, first-fit, as the IOP's sysmem/loadcore keep it (loadcore takes
        // a module's image from sysmem too). Upstream 75d729c split it at 0x120000 (images below, a bump heap above): a
        // large IRX (rotk AUDIOPF: 768 KB + 32 KB + thread stacks) then exhausted the 896 KB heap while ~800 KB above the
        // loaded images stayed unused, and its CreateThread failed with KE_NO_MEMORY (rotk row 267).
        static constexpr uint32_t HeapBase = 0x00010000u;
        static constexpr uint32_t HeapLimit = 0x001F0000u;

        struct Allocation
        {
            uint32_t address = 0;
            uint32_t size = 0;
        };

        struct DmaStart
        {
            int irq = 0;
            uint64_t delayCycles = 0;
        };

        IopMemory();

        void reset();

        // RAM is the common case of every fetch, load and store: inline it (rotk row 272); everything else (scratchpad,
        // hardware, unaligned, out of range) takes the out-of-line path, unchanged.
        [[nodiscard]] uint8_t read8(uint32_t address) const
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            return phys < RamSize ? m_ram[phys] : read8Slow(address);
        }
        [[nodiscard]] uint16_t read16(uint32_t address) const
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys + 1u < RamSize)
            {
                uint16_t value;
                std::memcpy(&value, m_ram.data() + phys, sizeof(value));
                return value;
            }
            return read16Slow(address);
        }
        [[nodiscard]] uint32_t read32(uint32_t address) const
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if ((phys & 3u) == 0u && phys + 3u < RamSize)
            {
                uint32_t value;
                std::memcpy(&value, m_ram.data() + phys, sizeof(value));
                return value;
            }
            return read32Slow(address);
        }
        void write8(uint32_t address, uint8_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys < RamSize)
            {
                m_ram[phys] = value;
                m_owned[phys] = 1u;
                return;
            }
            write8Slow(address, value);
        }
        void write16(uint32_t address, uint16_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys + 1u < RamSize)
            {
                std::memcpy(m_ram.data() + phys, &value, sizeof(value));
                m_owned[phys] = m_owned[phys + 1u] = 1u;
                return;
            }
            write16Slow(address, value);
        }
        void write32(uint32_t address, uint32_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if ((phys & 3u) == 0u && phys + 3u < RamSize)
            {
                std::memcpy(m_ram.data() + phys, &value, sizeof(value));
                uint32_t owned;
                std::memcpy(&owned, m_owned.data() + phys, sizeof(owned));
                if (owned != 0x01010101u)
                    std::memset(m_owned.data() + phys, 1, sizeof(owned));
                return;
            }
            write32Slow(address, value);
        }

        [[nodiscard]] bool readRam(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeRam(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroRam(uint32_t address, size_t size);
        [[nodiscard]] bool ownsRamRange(uint32_t address, size_t size) const;
        [[nodiscard]] bool isHardwareAddress(uint32_t address) const;
        [[nodiscard]] std::string readString(uint32_t address, size_t limit = 1024u) const;

        [[nodiscard]] uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt);
        [[nodiscard]] bool freeAllocation(uint32_t address);
        [[nodiscard]] uint32_t maxFreeMemory() const;
        [[nodiscard]] std::optional<Allocation> allocationContaining(uint32_t address) const;

        [[nodiscard]] uint32_t interruptStatus() const noexcept { return m_interruptStatus; }
        [[nodiscard]] uint32_t interruptMask() const noexcept { return m_interruptMask; }
        [[nodiscard]] uint32_t interruptControl() const noexcept { return m_interruptControl; }
        void setInterruptStatus(uint32_t value) noexcept { m_interruptStatus = value; }
        void setInterruptMask(uint32_t value) noexcept { m_interruptMask = value; }
        void setInterruptControl(uint32_t value) noexcept { m_interruptControl = value & 1u; }

        [[nodiscard]] std::optional<DmaStart> takeDmaStart() noexcept;
        [[nodiscard]] bool hasDmaStart() const noexcept { return m_dmaStart.has_value(); }
        [[nodiscard]] std::span<const uint8_t> ram() const noexcept { return m_ram; }

        [[nodiscard]] static constexpr uint32_t physicalAddress(uint32_t address) noexcept { return address & 0x1FFFFFFFu; }

    private:
        [[nodiscard]] uint8_t read8Slow(uint32_t address) const;
        [[nodiscard]] uint16_t read16Slow(uint32_t address) const;
        [[nodiscard]] uint32_t read32Slow(uint32_t address) const;
        void write8Slow(uint32_t address, uint8_t value);
        void write16Slow(uint32_t address, uint16_t value);
        void write32Slow(uint32_t address, uint32_t value);
        [[nodiscard]] uint32_t readHardware32(uint32_t address) const;
        void writeHardware32(uint32_t address, uint32_t value);
        void markOwned(uint32_t address, size_t size);

        std::vector<uint8_t> m_ram;
        std::vector<uint8_t> m_owned;
        std::vector<uint8_t> m_scratch;
        std::unordered_map<uint32_t, uint32_t> m_hardware;
        std::vector<Allocation> m_allocations;
        uint32_t m_heapCursor = HeapBase;
        uint32_t m_interruptStatus = 0;
        uint32_t m_interruptMask = 0;
        uint32_t m_interruptControl = 1;
        std::optional<DmaStart> m_dmaStart;
    };
}
