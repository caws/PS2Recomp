#include "iop_sysmem.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <string>

namespace ps2x::iop::detail
{
    IopSysmem::IopSysmem(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    bool IopSysmem::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const uint32_t a2 = cpu.gpr[6];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        switch (ordinal)
        {
        case 4: // AllocSysMemory
        {
            // sysmem manages RAM in 256-byte blocks: sizes round up ((size + 255) >> 8) and every block starts on a
            // 256-byte boundary; ALLOC_ADDRESS (mode 2) takes the block containing the requested address (ps2sdk
            // iop/system/sysmem/src/sysmem.c; PCSX2 runs the real IOP kernel). Callers rely on it: USBD.IRX puts the
            // OHCI HCCA, which must be 256-aligned, at the start of its block and gives up when (block & 0xff) != 0.
            constexpr uint32_t kBlock = 0x100u;
            const uint32_t size = (a1 + (kBlock - 1u)) & ~(kBlock - 1u);
            const uint32_t address = a0 == 2u
                                         ? m_memory.allocate(size, kBlock, a2 & ~(kBlock - 1u))
                                         : m_memory.allocate(size, kBlock);
            setV0(address);
            return true;
        }
        case 5: // FreeSysMemory
            setV0(m_memory.freeAllocation(a0) ? 0u : 0xFFFFFFFFu);
            return true;
        case 6: // QueryMemSize
            setV0(IopMemory::RamSize);
            return true;
        case 7: // QueryMaxFreeMemSize
        case 8: // QueryTotalFreeMemSize
            setV0(m_memory.maxFreeMemory());
            return true;
        case 9: // QueryBlockTopAddress
            if (const auto block = m_memory.allocationContaining(a0))
                setV0(block->address);
            else
                setV0(0u);
            return true;
        case 10: // QueryBlockSize
            if (const auto block = m_memory.allocationContaining(a0))
                setV0(block->size);
            else
                setV0(0xFFFFFFFFu);
            return true;
        case 14: // Kprintf
        {
            const std::string format = m_memory.readString(a0, 512u);
            m_host.log(LogLevel::Info, std::string("[IOP Kprintf] ") + format);
            setV0(static_cast<uint32_t>(format.size()));
            return true;
        }
        case 15:
            setV0(0u);
            return true;
        default:
            return false;
        }
    }
}
