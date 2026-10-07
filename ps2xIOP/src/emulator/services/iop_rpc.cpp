#include "iop_rpc.h"

#include "../core/iop_cpu.h"
#include "../core/iop_kernel.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace ps2x::iop::detail
{
    IopRpcBridge::IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept
        : m_host(host), m_memory(memory), m_kernel(kernel)
    {
    }

    void IopRpcBridge::reset()
    {
        m_servers.clear();
        m_nextDmaId = 1u;
        m_sifInitialized = false;
        m_sregs.fill(0u);
        m_usrHandlerTable = 0u;
        m_usrHandlerCount = 0u;
        m_sysHandlerTable = 0u;
        m_sysHandlerCount = 0u;
        m_sysHandlers.fill({});
        m_handlerGp.clear();
        m_receiveBuffer = 0u;
        m_servicingEeCommands = false;
        std::lock_guard<std::mutex> lock(m_eeCommandMutex);
        m_eeCommands.clear();
        m_eeCommandCount.store(0u, std::memory_order_release);
    }

    namespace
    {
        constexpr uint32_t kSifCmdSystem = 0x80000000u;
        constexpr uint32_t kSifCmdChangeSaddr = 0x80000000u;
        constexpr uint32_t kSifCmdSetSreg = 0x80000001u;
        constexpr uint32_t kSifCmdInitCmd = 0x80000002u;
        constexpr uint32_t kSifCmdHeaderSize = 16u;
        constexpr uint32_t kSifCmdMaxPacket = 112u;
        constexpr uint32_t kSifCmdSysEntries = 32u;

        // An IOP address that names IOP RAM (kuseg / kseg0 / kseg1 view of the 2 MB).
        bool isIopRamRange(uint32_t address, uint32_t size)
        {
            const bool segment = address < IopMemory::RamSize ||
                                 (address >= 0x80000000u && address < 0x80200000u) ||
                                 (address >= 0xA0000000u && address < 0xA0200000u);
            if (!segment)
                return false;
            const uint32_t physical = IopMemory::physicalAddress(address);
            return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
        }
    }

    bool IopRpcBridge::queueEeCommand(uint32_t cid, const void *packet, uint32_t packetSize,
                                      uint32_t eeExtraSource, uint32_t iopExtraDestination, uint32_t extraSize)
    {
        if (!packet || packetSize < kSifCmdHeaderSize || packetSize > kSifCmdMaxPacket)
            return false;

        // The extra data is the head of the SIF1 chain (EE sceSifSendCmd), so it is in IOP RAM before the packet is
        // received. A destination outside IOP RAM is dropped, not wrapped: it would land on unrelated IOP memory.
        if (extraSize != 0u && eeExtraSource != 0u)
        {
            if (!isIopRamRange(iopExtraDestination, extraSize))
            {
                static uint32_t s_badDest = 0u;
                if (s_badDest++ < 4u)
                {
                    char line[160];
                    std::snprintf(line, sizeof(line), "[IOP] SIF command 0x%08x: extra data to 0x%08x (+0x%x) is not IOP RAM; dropped",
                                  cid, iopExtraDestination, extraSize);
                    m_host.log(LogLevel::Warning, line);
                }
            }
            else
            {
                std::vector<uint8_t> extra(extraSize);
                if (m_host.readGuest(eeExtraSource, extra.data(), extra.size()))
                    (void)m_memory.writeRam(iopExtraDestination, extra.data(), extra.size());
            }
        }

        PendingEeCommand command;
        std::memcpy(command.packet.data(), packet, packetSize);
        command.size = packetSize;
        // Header words as the EE's sifcmd writes them: psize | dsize << 8, dest, cid (opt is the caller's).
        const uint32_t sizeWord = (packetSize & 0xFFu) | (extraSize << 8u);
        std::memcpy(command.packet.data() + 0u, &sizeWord, sizeof(sizeWord));
        std::memcpy(command.packet.data() + 4u, &iopExtraDestination, sizeof(iopExtraDestination));
        std::memcpy(command.packet.data() + 8u, &cid, sizeof(cid));
        std::lock_guard<std::mutex> lock(m_eeCommandMutex);
        m_eeCommands.push_back(command);
        m_eeCommandCount.fetch_add(1u, std::memory_order_release);
        return true;
    }

    bool IopRpcBridge::hasPendingEeCommands() const
    {
        return m_eeCommandCount.load(std::memory_order_acquire) != 0u;
    }

    // The handlers sifcmd installs for itself at InitCmd (ps2sdk sifcmd.c _change_addr / _set_sreg /
    // sif_sys_cmd_handler_init_from_ee), used when the system slot holds no guest handler.
    bool IopRpcBridge::dispatchBuiltinSystemCommand(uint32_t cid, const uint8_t *packet, uint32_t size)
    {
        if (cid == kSifCmdSetSreg && size >= kSifCmdHeaderSize + 8u)
        {
            int32_t index = 0;
            uint32_t value = 0u;
            std::memcpy(&index, packet + 16u, sizeof(index));
            std::memcpy(&value, packet + 20u, sizeof(value));
            if (index >= 0 && static_cast<uint32_t>(index) < m_sregs.size())
                m_sregs[static_cast<uint32_t>(index)] = value;
            return true;
        }
        // CHANGE_SADDR / INIT_CMD only move sifcmd's EE send buffer, which the host transport does not use.
        return cid == kSifCmdChangeSaddr || cid == kSifCmdInitCmd;
    }

    void IopRpcBridge::serviceEeCommands(IopGuestExecutor &executor)
    {
        if (m_servicingEeCommands)
            return;
        m_servicingEeCommands = true;
        for (;;)
        {
            PendingEeCommand command;
            {
                std::lock_guard<std::mutex> lock(m_eeCommandMutex);
                if (m_eeCommands.empty())
                    break;
                command = m_eeCommands.front();
                m_eeCommands.pop_front();
                m_eeCommandCount.fetch_sub(1u, std::memory_order_acq_rel);
            }
            uint32_t cid = 0u;
            std::memcpy(&cid, command.packet.data() + 8u, sizeof(cid));
            const bool system = (cid & kSifCmdSystem) != 0u;
            const uint32_t index = cid & 0x7FFFFFFFu;

            // _sceSifCmdIntrHdlr: index < the table's size, and a non-null handler, else the packet is dropped.
            uint32_t handler = 0u;
            uint32_t argument = 0u;
            if (!system)
            {
                if (m_usrHandlerTable != 0u && index < m_usrHandlerCount)
                {
                    handler = m_memory.read32(m_usrHandlerTable + index * 8u);
                    argument = m_memory.read32(m_usrHandlerTable + index * 8u + 4u);
                }
            }
            else if (m_sysHandlerTable != 0u)
            {
                if (index < m_sysHandlerCount)
                {
                    handler = m_memory.read32(m_sysHandlerTable + index * 12u);
                    argument = m_memory.read32(m_sysHandlerTable + index * 12u + 4u);
                }
            }
            else if (index < kSifCmdSysEntries)
            {
                handler = m_sysHandlers[index].handler;
                argument = m_sysHandlers[index].argument;
            }

            if (handler == 0u)
            {
                if (!(system && dispatchBuiltinSystemCommand(cid, command.packet.data(), command.size)))
                {
                    static uint32_t s_dropped = 0u;
                    if (s_dropped++ < 8u)
                    {
                        char line[96];
                        std::snprintf(line, sizeof(line), "[IOP] SIF command 0x%08x from the EE has no handler; dropped", cid);
                        m_host.log(LogLevel::Warning, line);
                    }
                }
                continue;
            }

            // The handler gets a copy of the packet (sifcmd copies it to a stack buffer before the call).
            if (m_receiveBuffer == 0u)
                m_receiveBuffer = m_memory.allocate(kSifCmdMaxPacket, 16u);
            if (m_receiveBuffer == 0u || !m_memory.writeRam(m_receiveBuffer, command.packet.data(), command.size))
                continue;
            const auto gp = m_handlerGp.find(cid);
            (void)executor.executeGuestFunctionWithBudget(handler, m_receiveBuffer, argument, 0u, 0u,
                                                          gp != m_handlerGp.end() ? gp->second : 0u, 100000u);
        }
        m_servicingEeCommands = false;
    }

    bool IopRpcBridge::dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 4: // sceSifDma2Init
        case 5: // sceSifInit
            m_sifInitialized = true;
            setV0(0u);
            return true;
        case 7: // sceSifSetDma
        {
            constexpr uint32_t kDescriptorSize = 16u;
            constexpr uint32_t kMaxDescriptors = 32u;
            const uint32_t descriptorAddress = cpu.gpr[4];
            const uint32_t descriptorCount = cpu.gpr[5];
            if (descriptorAddress == 0u || descriptorCount == 0u || descriptorCount > kMaxDescriptors)
            {
                setV0(0u);
                return true;
            }

            struct PendingTransfer
            {
                uint32_t source = 0u;
                uint32_t destination = 0u;
                uint32_t size = 0u;
            };

            std::array<uint32_t, kMaxDescriptors * 4u> descriptorWords{};
            const size_t descriptorBytes = static_cast<size_t>(descriptorCount) * kDescriptorSize;
            if (!m_memory.readRam(descriptorAddress, descriptorWords.data(), descriptorBytes))
            {
                setV0(0u);
                return true;
            }

            std::array<PendingTransfer, kMaxDescriptors> pending{};
            uint32_t pendingCount = 0u;
            uint32_t largestTransfer = 0u;
            for (uint32_t i = 0u; i < descriptorCount; ++i)
            {
                const uint32_t source = descriptorWords[i * 4u + 0u];
                const uint32_t destination = descriptorWords[i * 4u + 1u];
                const int32_t signedSize = static_cast<int32_t>(descriptorWords[i * 4u + 2u]);
                if (signedSize <= 0)
                    continue;

                const uint32_t size = static_cast<uint32_t>(signedSize);
                if (!m_memory.ownsRamRange(source, size))
                {
                    setV0(0u);
                    return true;
                }
                pending[pendingCount++] = {source, destination, size};
                largestTransfer = std::max(largestTransfer, size);
            }

            // IOP-side sceSifSetDma sends IOP RAM to the EE. Validate all EE
            // destinations before committing any write so a bad chain cannot
            // partially update guest memory, but maybe we could skip this check if we trust the EE-side SIF driver to validate the chain ?!
            // TODO check later
            std::vector<uint8_t> scratch(largestTransfer);
            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_host.readGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_memory.readRam(transfer.source, scratch.data(), transfer.size) || !m_host.writeGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            {
                m_nextDmaId = 1u;
            }
            setV0(dmaId);
            return true;
        }
        case 8: // sceSifDmaStat
            setV0(0xFFFFFFFFu);
            return true;
        case 29: // sceSifCheckInit
            setV0(m_sifInitialized ? 1u : 0u);
            return true;
        default:
            setV0(0u);
            return true;
        }
    }

    bool IopRpcBridge::dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        // Ordinals and behaviour: ps2sdk iop/system/sifcmd (exports.tab, sifcmd.c).
        case 6: // sceSifGetSreg
            setV0(cpu.gpr[4] < m_sregs.size() ? m_sregs[cpu.gpr[4]] : 0u);
            return true;
        case 7: // sceSifSetSreg
            if (cpu.gpr[4] < m_sregs.size())
                m_sregs[cpu.gpr[4]] = cpu.gpr[5];
            setV0(0u);
            return true;
        case 8: // sceSifSetCmdBuffer(db, size)
            m_usrHandlerTable = cpu.gpr[4];
            m_usrHandlerCount = cpu.gpr[5];
            setV0(0u);
            return true;
        case 9: // sceSifSetSysCmdBuffer(db, size)
            m_sysHandlerTable = cpu.gpr[4];
            m_sysHandlerCount = cpu.gpr[5];
            setV0(0u);
            return true;
        case 10: // sceSifAddCmdHandler(cid, handler, harg)
        case 11: // sceSifRemoveCmdHandler(cid) = AddCmdHandler(cid, NULL, NULL)
        {
            const uint32_t cid = cpu.gpr[4];
            const uint32_t handler = ordinal == 10 ? cpu.gpr[5] : 0u;
            const uint32_t argument = ordinal == 10 ? cpu.gpr[6] : 0u;
            const uint32_t index = cid & 0x7FFFFFFFu;
            if ((cid & 0x80000000u) == 0u)
            {
                if (m_usrHandlerTable != 0u)
                {
                    m_memory.write32(m_usrHandlerTable + index * 8u, handler);
                    m_memory.write32(m_usrHandlerTable + index * 8u + 4u, argument);
                }
            }
            else if (m_sysHandlerTable != 0u)
            {
                m_memory.write32(m_sysHandlerTable + index * 12u, handler);
                m_memory.write32(m_sysHandlerTable + index * 12u + 4u, argument);
            }
            else if (index < m_sysHandlers.size())
            {
                m_sysHandlers[index] = {handler, argument};
            }
            m_handlerGp[cid] = cpu.gpr[28];
            setV0(0u);
            return true;
        }
        case 4: // InitCmd
        case 5:
        case 14: // InitRpc
        case 15:
        case 16:
            setV0(0);
            return true;
        case 12: // sceSifSendCmd
        case 13: // isceSifSendCmd
        {
            constexpr uint32_t kHeaderSize = 16u;
            constexpr uint32_t kMaxPacketSize = 112u;
            const uint32_t commandId = cpu.gpr[4];
            const uint32_t packetAddress = cpu.gpr[5];
            const uint32_t packetSize = cpu.gpr[6];
            const uint32_t extraSource = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            const uint32_t extraDestination = m_memory.read32(stackPointer + 16u);
            const int32_t signedExtraSize = static_cast<int32_t>(m_memory.read32(stackPointer + 20u));

            if (packetAddress == 0u || packetSize < kHeaderSize || packetSize > kMaxPacketSize ||
                !m_memory.ownsRamRange(packetAddress, packetSize))
            {
                setV0(0u);
                return true;
            }

            std::array<uint8_t, kMaxPacketSize> packet{};
            if (!m_memory.readRam(packetAddress, packet.data(), packetSize))
            {
                setV0(0u);
                return true;
            }

            uint32_t extraSize = 0u;
            if (signedExtraSize > 0)
            {
                extraSize = static_cast<uint32_t>(signedExtraSize);
                if (extraSource == 0u || extraDestination == 0u ||
                    !m_memory.ownsRamRange(extraSource, extraSize) ||
                    !m_host.writeGuest(extraDestination, m_memory.ram().data() + IopMemory::physicalAddress(extraSource), extraSize))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t sizeWord = packetSize | (extraSize << 8u);
            std::memcpy(packet.data() + 0u, &sizeWord, sizeof(sizeWord));
            std::memcpy(packet.data() + 4u, &extraDestination, sizeof(extraDestination));
            std::memcpy(packet.data() + 8u, &commandId, sizeof(commandId));

            if (!m_host.sendSifCommand(commandId, packet.data(), packetSize))
            {
                // A command without an EE handler is still a completed DMA on  real hardware. Only malformed packets fail above.
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                m_nextDmaId = 1u;
            setV0(dmaId);
            return true;
        }
        case 17: // sceSifRegisterRpc
        {
            RpcServer server;
            server.serverData = cpu.gpr[4];
            server.sid = cpu.gpr[5];
            server.function = cpu.gpr[6];
            server.gp = cpu.gpr[28];
            server.buffer = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            server.callback = m_memory.read32(stackPointer + 16u);
            server.callbackBuffer = m_memory.read32(stackPointer + 20u);
            server.queue = m_memory.read32(stackPointer + 24u);
            m_servers[server.sid] = server;
            if (server.serverData != 0u)
            {
                m_memory.write32(server.serverData + 0x20u, server.sid);
                m_memory.write32(server.serverData + 0x28u, server.function);
                m_memory.write32(server.serverData + 0x2Cu, server.buffer);
            }
            setV0(server.serverData);
            return true;
        }
        case 18:
            setV0(0);
            return true;
        case 19: // SetRpcQueue
            setV0(cpu.gpr[4]);
            return true;
        case 20:
        case 21:
            setV0(0);
            return true;
        case 22: // RpcLoop
            m_kernel.sleepCurrent(cpu);
            setV0(0);
            return true;
        case 23:
            setV0(0);
            return true;
        case 24: // RemoveRpc
        {
            const uint32_t serverData = cpu.gpr[4];
            for (auto server = m_servers.begin(); server != m_servers.end(); ++server)
            {
                if (server->second.serverData == serverData)
                {
                    m_servers.erase(server);
                    break;
                }
            }
            setV0(0);
            return true;
        }
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
            setV0(0);
            return true;
        default:
            return false;
        }
    }

    RpcResult IopRpcBridge::handleRpc(const RpcRequest &request, IopGuestExecutor &executor)
    {
        RpcResult result{};
        const auto serverIt = m_servers.find(request.sid);
        if (serverIt == m_servers.end() || serverIt->second.function == 0u)
            return result;

        RpcServer &server = serverIt->second;
        if (request.send.size != 0u && server.buffer != 0u)
        {
            const uint32_t copySize = std::min<uint32_t>(request.send.size, IopMemory::RamSize - std::min(server.buffer, IopMemory::RamSize));
            if (copySize != 0u)
            {
                std::vector<uint8_t> payload(copySize);
                if (m_host.readGuest(request.send.address, payload.data(), payload.size()))
                    (void)m_memory.writeRam(server.buffer, payload.data(), payload.size());
            }
        }

        uint32_t returnPointer = executor.executeGuestFunction(server.function,
                                                               request.function,
                                                               server.buffer,
                                                               request.send.size,
                                                               0u,
                                                               server.gp);
        if (returnPointer == 0u)
            returnPointer = server.buffer;
        if (request.receive.address != 0u && request.receive.size != 0u && returnPointer != 0u)
        {
            const uint32_t physical = IopMemory::physicalAddress(returnPointer);
            if (physical < IopMemory::RamSize)
            {
                const uint32_t copySize = std::min<uint32_t>(request.receive.size, IopMemory::RamSize - physical);
                (void)m_host.writeGuest(request.receive.address, m_memory.ram().data() + physical, copySize);
                if (copySize < request.receive.size)
                    (void)m_host.zeroGuest(request.receive.address + copySize, request.receive.size - copySize);
            }
        }

        result.handled = true;
        result.resultAddress = request.receive.address;
        result.serverDispatchPolicy = ServerDispatchPolicy::Suppress;
        result.signalNowaitCompletion = true;
        result.signalCompletion = true;
        return result;
    }

    void IopRpcBridge::onSifTransfer(const SifTransfer &transfer)
    {
        // The EE SIF transport owns the actual directional memory movement.
        // Services still receive both phases through IopSubsystem, but mirroring
        // IOP bytes through an equal-numbered EE address would alias two distinct
        // PS2 address spaces and can overwrite live game data.
        (void)transfer;
    }

    void IopRpcBridge::removeServersInRange(uint32_t base, uint32_t size)
    {
        for (auto server = m_servers.begin(); server != m_servers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(server->second.function);
            if (function >= base && function < base + size)
                server = m_servers.erase(server);
            else
                ++server;
        }
    }

    bool IopRpcBridge::hasServer(uint32_t sid) const noexcept
    {
        const auto server = m_servers.find(sid);
        return server != m_servers.end() && server->second.function != 0u;
    }
}
