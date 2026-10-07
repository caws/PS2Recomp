#pragma once

#include "ps2x/iop/iop_host.h"
#include "ps2x/iop/iop_types.h"
#include "ps2x/iop/iop_native_bridge.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ps2x::iop
{
    class IopSubsystem
    {
    public:
        explicit IopSubsystem(IopHost &host);
        ~IopSubsystem();

        IopSubsystem(const IopSubsystem &) = delete;
        IopSubsystem &operator=(const IopSubsystem &) = delete;
        IopSubsystem(IopSubsystem &&) noexcept;
        IopSubsystem &operator=(IopSubsystem &&) noexcept;

        void reset();

        [[nodiscard]] ModuleLoadResult loadModule(std::string_view path, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] bool stopModule(int32_t moduleId, int32_t *result = nullptr);
        void runEeCycles(uint64_t eeCycles) noexcept;

        [[nodiscard]] RpcAbi selectRpcAbi(const RpcAbiRequest &request) const;
        [[nodiscard]] bool canBindRpc(uint32_t sid) const noexcept;
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request);
        void onSifTransfer(const SifTransfer &transfer);

        // EE -> IOP SIF command (the EE's sceSifSendCmd): the extra data is copied into IOP RAM at once, the packet is
        // delivered to the IOP handler registered for cid on the IOP's next run. False = malformed packet.
        bool sendSifCommand(uint32_t cid, const void *packet, uint32_t packetSize,
                            uint32_t eeExtraSource, uint32_t iopExtraDestination, uint32_t extraSize);
        // The IOP's sifcmd software register `index` (0..31), as sceSifGetSreg on the IOP would read it.
        [[nodiscard]] uint32_t iopSoftwareRegister(uint32_t index) const noexcept;
        // rotk row 273: emulate the SPU2 (PCSX2's core) behind the IOP's sound registers and DMA 4/7. OFF by default.
        void setSpu2Enabled(bool enabled);
        // rotk row 276: run statically recompiled IOP modules through `executor` (nullptr = interpret everything).
        void setNativeExecutor(IopNativeExecutor *executor);
        [[nodiscard]] uint64_t iopCycles() const noexcept;   // the emulated IOP's cycle count
        // The SPU2's mixed output (48 kHz interleaved stereo s16), oldest first; returns frames copied.
        size_t takeSpu2Samples(int16_t *destination, size_t maxFrames);

        // Physical IOP RAM access shared by the emulator, SIF DMA, and HLE services. Addresses are IOP addresses.
        [[nodiscard]] uint32_t allocateMemory(uint32_t size, uint32_t alignment = 16u);
        [[nodiscard]] bool freeMemory(uint32_t address);
        [[nodiscard]] bool readMemory(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeMemory(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroMemory(uint32_t address, size_t size);
        [[nodiscard]] bool isMemoryRange(uint32_t address, size_t size) const;

        [[nodiscard]] DebugSnapshot debugSnapshot() const;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
