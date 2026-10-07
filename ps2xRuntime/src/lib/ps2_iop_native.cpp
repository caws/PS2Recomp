// rotk row 276: running statically recompiled IOP modules (ps2x_iop_native.h) inside the IOP emulator.
//
// The executor owns one table of native entries indexed by IOP RAM word (function starts AND their resume entries --
// the generated `switch (ctx->pc)` labels at call-return sites), filled when a loaded IRX matches a module the build
// compiled (file hash + base). run() maps the thread's registers into an R5900Context, runs native functions in a
// dispatch loop while the pc stays on native entries, and hands the thread back when it leaves native code, blocks in an
// import, or the emulator asks for a break at a checkpoint. A guest call runs as a nested native call while it can; when
// it cannot finish (block / break), every native frame returns and the thread resumes later from its saved pc through
// the resume labels -- the EE's own scheme.
#include "ps2x_iop_native.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace ps2x::iop::native
{
    namespace
    {
        std::vector<CompiledModule> &compiledModules()
        {
            static std::vector<CompiledModule> modules;
            return modules;
        }
    }

    void addCompiledModule(const CompiledModule &module)
    {
        compiledModules().push_back(module);
    }

    uint64_t imageHash(const uint8_t *data, size_t size)
    {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i)
            h = (h ^ data[i]) * 1099511628211ull;
        return h;
    }

    // ---- the thread's registers <-> the generated code's context -------------------------------------------------
    void Host::syncToCpu(const R5900Context &ctx)
    {
        for (int i = 1; i < 32; ++i)
            m_cpu->gpr[i] = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx.r[i]));
        *m_cpu->hi = static_cast<uint32_t>(ctx.hi);
        *m_cpu->lo = static_cast<uint32_t>(ctx.lo);
        *m_cpu->pc = ctx.pc;
    }

    void Host::syncFromCpu(R5900Context &ctx) const
    {
        // MIPS I registers are 32-bit; the generated code compares/extends them as the EE's 64-bit ones, so they are
        // held sign-extended, as every 32-bit EE operation leaves them.
        for (int i = 0; i < 32; ++i)
            ctx.r[i] = _mm_set_epi64x(0, static_cast<int64_t>(static_cast<int32_t>(m_cpu->gpr[i])));
        ctx.hi = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(*m_cpu->hi)));
        ctx.lo = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(*m_cpu->lo)));
        ctx.pc = *m_cpu->pc;
    }

    // ---- generated-code entry points ---------------------------------------------------------------------------
    void Host::iopImport(uint8_t *, R5900Context *ctx, const char *library, uint32_t ordinal)
    {
        // ctx->pc is the stub's address (the stub function sets no pc of its own).
        flushAccounting(*ctx);
        const uint32_t stub = ctx->pc;
        syncToCpu(*ctx);
        const bool goOn = m_services->callImport(*m_cpu, stub, library, static_cast<uint16_t>(ordinal));
        syncFromCpu(*ctx);
        if (!goOn)
            m_unwinding = true;   // the thread blocked (WaitSema, SleepThread, ...): hand it back
    }

    bool Host::dispatchGuestBranch(uint8_t *rdram, R5900Context *ctx, uint32_t target, uint32_t, uint32_t returnPc,
                                   GuestBranchKind kind, const char *)
    {
        ctx->pc = target;
        if (kind == GuestBranchKind::Return || m_unwinding)
            return false;
        // Run the callee nested while it stays native and nothing asks for a break; true = it came back to returnPc and
        // the caller goes on inline.
        while (ctx->pc != returnPc)
        {
            NativeFunction function = lookup(ctx->pc);
            if (!function || m_unwinding)
                return false;
            if (unaccounted(*ctx) >= 64u)
            {
                flushAccounting(*ctx);
                if (m_services->breakRequested())
                {
                    m_unwinding = true;
                    return false;
                }
            }
            function(rdram, ctx, this);
            if (m_unwinding)
                return false;
            if (kind != GuestBranchKind::DirectCall && kind != GuestBranchKind::IndirectCall)
                return false;   // a jump: the loop that dispatched us follows it
        }
        return true;
    }

    bool Host::eeCheckpointDue(uint32_t)
    {
        // A loop back-edge: hand the instructions executed so far to the emulator, and stop here if it wants the
        // thread back (an interrupt, an SPU2 tick, the budget).
        if (unaccounted(*m_ctx) < 64u)
            return false;
        flushAccounting(*m_ctx);
        if (m_services->breakRequested())
        {
            m_unwinding = true;
            return true;
        }
        return false;
    }

    void Host::handleBreak(uint8_t *, R5900Context *ctx)
    {
        static int reported = 0;
        if (reported++ < 4)
            std::fprintf(stderr, "[iop:native] break at 0x%x (ignored)\n", ctx->pc);
    }

    void Host::handleSyscall(uint8_t *, R5900Context *ctx, uint32_t)
    {
        static int reported = 0;
        if (reported++ < 4)
            std::fprintf(stderr, "[iop:native] syscall at 0x%x (ignored)\n", ctx->pc);
    }

    // ---- the executor --------------------------------------------------------------------------------------------
    namespace
    {
        class Executor final : public IopNativeExecutor, public Registry
        {
        public:
            Executor() : m_table(Host::kRamSize / 4u, nullptr) {}

            void add(uint32_t address, NativeFunction function) override
            {
                const uint32_t phys = address & 0x1FFFFFFFu;
                if (phys < Host::kRamSize)
                    m_table[phys >> 2] = function;
            }

            void onModuleLoaded(std::string_view name, const uint8_t *image, size_t imageSize, uint32_t base, uint32_t) override
            {
                const uint64_t hash = imageHash(image, imageSize);
                for (const CompiledModule &module : compiledModules())
                {
                    if (module.imageHash != hash)
                        continue;
                    if (module.base != base)
                    {
                        std::fprintf(stderr, "[iop:native] %.*s: compiled for base 0x%x, loaded at 0x%x -- interpreted\n",
                                     static_cast<int>(name.size()), name.data(), module.base, base);
                        return;
                    }
                    module.registerFunctions(*this);
                    std::fprintf(stderr, "[iop:native] %.*s at 0x%x: native\n", static_cast<int>(name.size()), name.data(), base);
                    return;
                }
            }

            void onModuleUnloaded(uint32_t base, uint32_t size) override
            {
                for (uint32_t a = base & ~3u; a < base + size && (a >> 2) < m_table.size(); a += 4u)
                    m_table[a >> 2] = nullptr;
            }

            bool has(uint32_t pc) const override
            {
                const uint32_t phys = pc & 0x1FFFFFFFu;
                return phys < Host::kRamSize && m_table[phys >> 2] != nullptr;
            }

            void run(IopNativeCpu &cpu, IopNativeServices &services) override
            {
                Host host;
                alignas(16) R5900Context ctx{};
                host.bind(&services, &cpu, &m_table, &ctx);
                host.syncFromCpu(ctx);
                uint8_t *ram = services.ram();
                while (NativeFunction function = host.lookup(ctx.pc))
                {
                    function(ram, &ctx, &host);
                    if (host.unwinding())
                        break;
                    host.flushAccounting(ctx);
                    if (services.breakRequested())
                        break;
                }
                host.flushAccounting(ctx);
                host.syncToCpu(ctx);
            }

        private:
            std::vector<NativeFunction> m_table;
        };
    }

    // PS2X_IOP_NATIVE=1 (EXPERIMENT, default OFF): the executor PS2Runtime installs into the IOP.
    IopNativeExecutor *defaultExecutor()
    {
        static const bool on = [] { const char *e = std::getenv("PS2X_IOP_NATIVE"); return e && e[0] == '1'; }();
        if (!on)
            return nullptr;
        static Executor executor;
        return &executor;
    }
}
