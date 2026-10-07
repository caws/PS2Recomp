#pragma once

// rotk row 276: the host statically recompiled IOP modules run against (ps2xRecomp R3000 mode; see
// ps2xIOP/tools/ps2x_irx_image.cpp and ps2xIOP/include/ps2x/iop/iop_native_bridge.h).
//
// Generated module code is the EE generator's output in its own namespace, where `PS2Runtime` names Host (the
// generator emits `using PS2Runtime = ::ps2x::iop::native::Host;`). So it calls the same few runtime functions the EE
// code does -- dispatchGuestBranch, eeCheckpointDue, a syscall/break handler -- plus iopImport for import stubs, and its
// loads/stores go to IOP memory (the macros below replace the EE ones in those files): RAM directly, keeping the
// emulator's per-byte "written" map, anything else through the emulator (hardware, SPU2, scratchpad).

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2x/iop/iop_native_bridge.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ps2x::iop::native
{
    class Host;
    using NativeFunction = void (*)(uint8_t *, R5900Context *, Host *);

    class Registry
    {
    public:
        virtual ~Registry() = default;
        virtual void add(uint32_t address, NativeFunction function) = 0;
    };

    // A recompiled module the build produced: matched to a loaded IRX by its file hash and load base.
    struct CompiledModule
    {
        const char *name = nullptr;
        uint64_t imageHash = 0;   // FNV-1a 64 of the IRX file
        uint32_t base = 0;        // the base it was relocated to (ps2x_irx_image)
        void (*registerFunctions)(Registry &) = nullptr;
    };
    // Called by the build's generated module table (static initialisers).
    void addCompiledModule(const CompiledModule &module);
    // PS2X_IOP_NATIVE=1 (EXPERIMENT, default OFF): the executor the runtime installs into the IOP; nullptr when off.
    [[nodiscard]] IopNativeExecutor *defaultExecutor();
    [[nodiscard]] uint64_t imageHash(const uint8_t *data, size_t size);

    class Host
    {
    public:
        using GuestBranchKind = PS2Runtime::GuestBranchKind;
        static constexpr uint32_t kRamSize = 0x200000u;

        // ---- what generated code calls --------------------------------------------------------------------------
        void iopImport(uint8_t *rdram, R5900Context *ctx, const char *library, uint32_t ordinal);
        bool dispatchGuestBranch(uint8_t *rdram, R5900Context *ctx, uint32_t target, uint32_t sourcePc, uint32_t returnPc,
                                 GuestBranchKind kind, const char *debugName);
        bool eeCheckpointDue(uint32_t cycles = 8u);
        void handleBreak(uint8_t *rdram, R5900Context *ctx);
        void handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t code = 0u);

        // ---- IOP memory ----------------------------------------------------------------------------------------
        uint8_t read8(uint32_t address)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            return phys < kRamSize ? m_ram[phys] : static_cast<uint8_t>(m_services->read(address, 1u));
        }
        uint16_t read16(uint32_t address)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys + 1u < kRamSize)
            {
                uint16_t v;
                std::memcpy(&v, m_ram + phys, 2);
                return v;
            }
            return static_cast<uint16_t>(m_services->read(address, 2u));
        }
        uint32_t read32(uint32_t address)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if ((phys & 3u) == 0u && phys + 3u < kRamSize)
            {
                uint32_t v;
                std::memcpy(&v, m_ram + phys, 4);
                return v;
            }
            return m_services->read(address, 4u);
        }
        void write8(uint32_t address, uint8_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys < kRamSize)
            {
                m_ram[phys] = value;
                m_owned[phys] = 1u;
                return;
            }
            m_services->write(address, value, 1u);
        }
        void write16(uint32_t address, uint16_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if (phys + 1u < kRamSize)
            {
                std::memcpy(m_ram + phys, &value, 2);
                m_owned[phys] = m_owned[phys + 1u] = 1u;
                return;
            }
            m_services->write(address, value, 2u);
        }
        void write32(uint32_t address, uint32_t value)
        {
            const uint32_t phys = address & 0x1FFFFFFFu;
            if ((phys & 3u) == 0u && phys + 3u < kRamSize)
            {
                std::memcpy(m_ram + phys, &value, 4);
                std::memset(m_owned + phys, 1, 4);
                return;
            }
            m_services->write(address, value, 4u);
        }

        // ---- the executor's side --------------------------------------------------------------------------------
        void bind(IopNativeServices *services, IopNativeCpu *cpu, const std::vector<NativeFunction> *table, R5900Context *ctx)
        {
            m_ctx = ctx;
            m_services = services;
            m_cpu = cpu;
            m_table = table;
            m_ram = services->ram();
            m_owned = services->owned();
            m_unwinding = false;
            m_accountedInstructions = 0u;
        }
        [[nodiscard]] NativeFunction lookup(uint32_t pc) const
        {
            const uint32_t phys = pc & 0x1FFFFFFFu;
            return phys < kRamSize ? (*m_table)[phys >> 2] : nullptr;
        }
        [[nodiscard]] bool unwinding() const { return m_unwinding; }
        // The generated code counts every instruction in ctx->insn_count; hand the new ones to the emulator.
        void flushAccounting(const R5900Context &ctx)
        {
            const uint64_t executed = ctx.insn_count - m_accountedInstructions;
            if (executed)
            {
                m_services->account(static_cast<uint32_t>(executed));
                m_accountedInstructions = ctx.insn_count;
            }
        }
        [[nodiscard]] uint64_t unaccounted(const R5900Context &ctx) const { return ctx.insn_count - m_accountedInstructions; }
        void syncToCpu(const R5900Context &ctx);
        void syncFromCpu(R5900Context &ctx) const;

    private:
        IopNativeServices *m_services = nullptr;
        IopNativeCpu *m_cpu = nullptr;
        const std::vector<NativeFunction> *m_table = nullptr;
        uint8_t *m_ram = nullptr;
        uint8_t *m_owned = nullptr;
        R5900Context *m_ctx = nullptr;   // the one context run() executes on
        bool m_unwinding = false;
        uint64_t m_accountedInstructions = 0u;
    };
}

// ---- the generated code's loads and stores, retargeted at IOP memory ---------------------------------------------------
#undef READ8
#undef READ16
#undef READ32
#undef WRITE8
#undef WRITE16
#undef WRITE32
#undef FAST_READ8
#undef FAST_READ16
#undef FAST_READ32
#undef FAST_WRITE8
#undef FAST_WRITE16
#undef FAST_WRITE32
#define READ8(addr) runtime->read8(static_cast<uint32_t>(addr))
#define READ16(addr) runtime->read16(static_cast<uint32_t>(addr))
#define READ32(addr) runtime->read32(static_cast<uint32_t>(addr))
#define FAST_READ8(addr) runtime->read8(static_cast<uint32_t>(addr))
#define FAST_READ16(addr) runtime->read16(static_cast<uint32_t>(addr))
#define FAST_READ32(addr) runtime->read32(static_cast<uint32_t>(addr))
#define WRITE8(addr, val) runtime->write8(static_cast<uint32_t>(addr), static_cast<uint8_t>(val))
#define WRITE16(addr, val) runtime->write16(static_cast<uint32_t>(addr), static_cast<uint16_t>(val))
#define WRITE32(addr, val) runtime->write32(static_cast<uint32_t>(addr), static_cast<uint32_t>(val))
#define FAST_WRITE8(addr, val) runtime->write8(static_cast<uint32_t>(addr), static_cast<uint8_t>(val))
#define FAST_WRITE16(addr, val) runtime->write16(static_cast<uint32_t>(addr), static_cast<uint16_t>(val))
#define FAST_WRITE32(addr, val) runtime->write32(static_cast<uint32_t>(addr), static_cast<uint32_t>(val))
// The EE write tracer has no meaning for IOP memory.
#define ps2TraceGuestWrite(...) ((void)0)
