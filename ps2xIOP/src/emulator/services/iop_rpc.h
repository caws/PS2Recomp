#pragma once

#include "ps2x/iop/iop_types.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopMemory;

    class IopGuestExecutor
    {
    public:
        virtual ~IopGuestExecutor() = default;

        [[nodiscard]] virtual uint32_t executeGuestFunction(uint32_t address,
                                                            uint32_t a0,
                                                            uint32_t a1,
                                                            uint32_t a2,
                                                            uint32_t a3,
                                                            uint32_t gp) = 0;
        [[nodiscard]] virtual uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                                      uint32_t a0,
                                                                      uint32_t a1,
                                                                      uint32_t a2,
                                                                      uint32_t a3,
                                                                      uint32_t gp,
                                                                      uint32_t instructionBudget)
        {
            return executeGuestFunction(address, a0, a1, a2, a3, gp);
        }
    };

    class IopRpcBridge
    {
    public:
        IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept;

        void reset();
        [[nodiscard]] bool dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] bool dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request, IopGuestExecutor &executor);
        void onSifTransfer(const SifTransfer &transfer);
        void removeServersInRange(uint32_t base, uint32_t size);

        // EE -> IOP SIF command (the EE's sceSifSendCmd). The extra data is copied into IOP RAM now (on hardware it
        // rides the same SIF1 DMA chain, ahead of the packet); the packet is queued and handed to its handler by
        // serviceEeCommands(), the IOP-side receive interrupt. Thread-safe: the EE thread queues, the IOP drains.
        bool queueEeCommand(uint32_t cid, const void *packet, uint32_t packetSize,
                            uint32_t eeExtraSource, uint32_t iopExtraDestination, uint32_t extraSize);
        [[nodiscard]] bool hasPendingEeCommands() const;
        [[nodiscard]] uint32_t softwareRegister(uint32_t index) const noexcept
        {
            return index < m_sregs.size() ? m_sregs[index] : 0u;
        }
        void serviceEeCommands(IopGuestExecutor &executor);

        [[nodiscard]] bool hasServer(uint32_t sid) const noexcept;
        [[nodiscard]] size_t serverCount() const noexcept { return m_servers.size(); }

    private:
        struct RpcServer
        {
            uint32_t sid = 0;
            uint32_t serverData = 0;
            uint32_t function = 0;
            uint32_t gp = 0;
            uint32_t buffer = 0;
            uint32_t callback = 0;
            uint32_t callbackBuffer = 0;
            uint32_t queue = 0;
        };

        struct PendingEeCommand
        {
            std::array<uint8_t, 112> packet{};
            uint32_t size = 0u;
        };

        struct SysHandler
        {
            uint32_t handler = 0u;
            uint32_t argument = 0u;
        };

        bool dispatchBuiltinSystemCommand(uint32_t cid, const uint8_t *packet, uint32_t size);

        IopHost &m_host;
        IopMemory &m_memory;
        IopKernel &m_kernel;
        std::unordered_map<uint32_t, RpcServer> m_servers;
        uint32_t m_nextDmaId = 1u;
        bool m_sifInitialized = false;

        // sifcmd state (ps2sdk iop/system/sifcmd/src/sifcmd.c): 32 software registers, the user handler table the
        // module hands over with sceSifSetCmdBuffer (guest RAM, 8-byte {handler, harg} entries), and the system
        // table -- sifcmd's own unless sceSifSetSysCmdBuffer supplies one (12-byte entries).
        std::array<uint32_t, 32> m_sregs{};
        uint32_t m_usrHandlerTable = 0u;
        uint32_t m_usrHandlerCount = 0u;
        uint32_t m_sysHandlerTable = 0u;
        uint32_t m_sysHandlerCount = 0u;
        std::array<SysHandler, 32> m_sysHandlers{};
        std::unordered_map<uint32_t, uint32_t> m_handlerGp;
        uint32_t m_receiveBuffer = 0u;
        bool m_servicingEeCommands = false;
        mutable std::mutex m_eeCommandMutex;
        std::deque<PendingEeCommand> m_eeCommands;
        std::atomic<uint32_t> m_eeCommandCount{0u};   // checked every IOP instruction: no lock on the empty path
    };
}
