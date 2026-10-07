#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"
#include "spu2/ps2x_spu2.h"
#include "ps2x/iop/iop_native_bridge.h"

#include <algorithm>
#include <chrono>
#include <x86intrin.h>
#include <functional>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;
        constexpr int kSpu2Irq = 9;   // PCSX2 IopIrq.cpp spu2Irq: iopIntcIrq(9)
        constexpr uint64_t kSpu2TickCycles = 768u;   // PCSX2 IopCounters: counter 6 (the SPU2) rate, one 48 kHz sample

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor, public ps2x::spu2::Host, public ps2x::iop::IopNativeServices
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports)
        {
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        // ---- rotk row 276: statically recompiled modules (IopNativeServices) --------------------------------------
        ps2x::iop::IopNativeExecutor *nativeExecutor = nullptr;
        CpuState *nativeCpu = nullptr;          // the thread native code is running for
        uint64_t nativeBudgetEnd = 0u;          // runCpu's instruction budget, as an absolute count

        uint8_t *ram() override { return memory.ramData(); }
        uint8_t *owned() override { return memory.ownedData(); }

        uint32_t read(uint32_t address, unsigned bytes) override
        {
            return bytes == 1u ? memory.read8(address) : bytes == 2u ? memory.read16(address) : memory.read32(address);
        }

        void write(uint32_t address, uint32_t value, unsigned bytes) override
        {
            if (bytes == 1u)
                write8(address, static_cast<uint8_t>(value));
            else if (bytes == 2u)
                write16(address, static_cast<uint16_t>(value));
            else
                write32(address, value);
        }

        bool callImport(ps2x::iop::IopNativeCpu &, uint32_t stubAddress, std::string_view library, uint16_t ordinal) override
        {
            // As stepImportStub: the import's HLE (or the jump into the exporting module), then the stub's `jr ra`.
            CpuState &cpu = *nativeCpu;
            const auto decoded = imports.decode(stubAddress);
            const IopImportCall call = decoded ? *decoded : IopImportCall{std::string(library), ordinal, 0u};
            const ImportDisposition disposition = dispatchImport(call, cpu);
            ++totalInstructions;
            ++totalCycles;
            if (disposition != ImportDisposition::JumpToGuest)
            {
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
            }
            return !cpu.stopped && !cpu.yielded;
        }

        void account(uint32_t instructions) override
        {
            totalInstructions += instructions;
            totalCycles += instructions;
        }

        bool breakRequested() override
        {
            const CpuState &cpu = *nativeCpu;
            if (cpu.stopped || cpu.yielded || totalInstructions >= nativeBudgetEnd)
                return true;
            if (!pendingDmaInterrupts.empty() || !pendingGuestCallbacks.empty() || rpc.hasPendingEeCommands())
                return true;
            if (memory.spu2Enabled() && totalCycles - spu2LastAdvance >= kSpu2TickCycles)
                return true;   // the SPU2 (and its DMA completions) advance in runCpu, between native runs
            const uint32_t status = cpu.cop0[12];
            return (status & 1u) != 0u && (status & 2u) == 0u && memory.interruptControl() != 0u &&
                   (memory.interruptStatus() & memory.interruptMask()) != 0u;
        }

        void setNativeExecutor(ps2x::iop::IopNativeExecutor *executor)
        {
            nativeExecutor = executor;
        }

        bool stepNative(CpuState &cpu)
        {
            ps2x::iop::IopNativeCpu view{cpu.gpr.data(), &cpu.hi, &cpu.lo, &cpu.pc, &cpu.stopped, &cpu.yielded};
            CpuState *previous = nativeCpu;
            nativeCpu = &cpu;
            nativeExecutor->run(view, *this);
            nativeCpu = previous;
            schedulePendingDma();
            return !cpu.stopped;
        }

        // ---- rotk row 273: the SPU2 core's view of this IOP (ps2x::spu2::Host) --------------------------------------
        uint8_t *iopRam() override
        {
            return const_cast<uint8_t *>(memory.ram().data());
        }

        void spu2Interrupt() override
        {
            pendingDmaInterrupts[kSpu2Irq] = totalCycles;   // IRQ 9, serviced like the DMA ones (never nested)
        }

        void spu2DmaComplete(int core) override
        {
            memory.spu2DmaComplete(core);
            schedulePendingDma();
        }

        void setSpu2Enabled(bool enabled)
        {
            if (enabled == memory.spu2Enabled())
                return;
            if (enabled)
            {
                ps2x::spu2::attach(this);
                ps2x::spu2::reset(totalCycles);
            }
            memory.setSpu2Enabled(enabled);
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (!memory.hasDmaStart())
                return;
            if (const auto dma = memory.takeDmaStart())
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        // PS2X_IOP_IMPORTLOG=1 (diagnostic, default OFF; rotk row 267): one line per IOP import call -- library:ordinal,
        // a0-a3 on entry, v0 and the disposition on return. Answers "which call made this module fail" without a debugger.
        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            static const bool s_importLog = [] { const char *e = std::getenv("PS2X_IOP_IMPORTLOG"); return e && e[0] == '1'; }();
            if (!s_importLog)
                return dispatchImportImpl(call, cpu);
            const uint32_t a0 = cpu.gpr[4], a1 = cpu.gpr[5], a2 = cpu.gpr[6], a3 = cpu.gpr[7], ra = cpu.gpr[31];
            const ImportDisposition d = dispatchImportImpl(call, cpu);
            std::fprintf(stderr, "[iop:import] %s:%u a0=%08x a1=%08x a2=%08x a3=%08x ra=%08x -> v0=%08x %s\n",
                         call.library.c_str(), static_cast<unsigned>(call.ordinal), a0, a1, a2, a3, ra, cpu.gpr[2],
                         d == ImportDisposition::Handled ? "hle" : d == ImportDisposition::JumpToGuest ? "guest" : "MISSING");
            return d;
        }

        ImportDisposition dispatchImportImpl(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                return rpc.dispatchSifManImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        // The rare half of step(), kept out of line so step() itself stays small (rotk row 272): true = pc was an import stub
        // and was dispatched; `running` = step()'s result.
        [[gnu::noinline]] bool stepImportStub(CpuState &cpu, bool &running)
        {
            const auto import = imports.decode(cpu.pc);
            if (!import)
                return false;
            const ImportDisposition disposition = dispatchImport(*import, cpu);
            ++totalInstructions;
            ++totalCycles;
            if (disposition == ImportDisposition::JumpToGuest)
            {
                running = true;
                return true;
            }
            cpu.pc = cpu.gpr[31];
            cpu.branchPending = false;
            running = !cpu.stopped;
            return true;
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            // rotk row 276: a statically recompiled module's code runs natively (at a function or resume entry, with no
            // branch or load delay in flight).
            if (nativeExecutor && !cpu.branchPending && !cpu.pendingLoad && nativeExecutor->has(cpu.pc))
                return stepNative(cpu);

            // Import stubs are `jr ra; addiu zero,zero,<ordinal>`: only that word can be one (rotk row 272: test it
            // inline instead of an out-of-line decode() per instruction).
            if (memory.read32(cpu.pc) == 0x03E00008u)
            {
                bool running = true;
                if (stepImportStub(cpu, running))
                    return running;
            }

            // PS2X_IOP_PCPROF=<seconds> (diagnostic, default OFF; rotk row 271): a histogram of executed IOP pcs,
            // the top 40 printed every <seconds> of host time -- which guest loop the IOP's time goes to.
            static const int s_pcProf = [] { const char *e = std::getenv("PS2X_IOP_PCPROF"); return e ? std::atoi(e) : 0; }();
            if (s_pcProf > 0)
                profilePc(cpu.pc, s_pcProf);
            if (memory.spu2Enabled())
                ps2x::spu2::setCycle(totalCycles);   // PCSX2 psxRegs.cycle: SPU2 register accesses mix up to "now"
            const bool running = cpuCore.executeInstruction(cpu);
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        void profilePc(uint32_t pc, int periodSeconds)
        {
            static std::vector<uint32_t> s_hist(kRamSize / 4u, 0u);
            static auto s_last = std::chrono::steady_clock::now();
            ++s_hist[(physicalAddress(pc) % kRamSize) / 4u];
            static uint32_t s_tick = 0u;
            if ((++s_tick & 0xFFFFu) != 0u)
                return;
            const auto now = std::chrono::steady_clock::now();
            if (now - s_last < std::chrono::seconds(periodSeconds))
                return;
            s_last = now;
            std::vector<std::pair<uint32_t, uint32_t>> top;
            uint64_t total = 0u;
            for (uint32_t i = 0; i < s_hist.size(); ++i)
                if (s_hist[i]) { top.emplace_back(s_hist[i], i * 4u); total += s_hist[i]; }
            std::sort(top.begin(), top.end(), std::greater<>());
            // PS2X_IOP_PCPROF_FILE=<path>: also write the whole histogram (pc count module offset), for aggregating by symbol.
            if (const char *path = std::getenv("PS2X_IOP_PCPROF_FILE"); path && *path)
            {
                if (FILE *f = std::fopen(path, "w"))
                {
                    for (const auto &[count, pcAddress] : top)
                    {
                        const Module *owner = nullptr;
                        for (const auto &[id, module] : modules)
                            if (pcAddress >= module.base && pcAddress < module.base + module.size)
                                owner = &module;
                        std::fprintf(f, "%08x %u %s %x\n", pcAddress, count, owner ? owner->name.c_str() : "?",
                                     owner ? pcAddress - owner->base : pcAddress);
                    }
                    std::fclose(f);
                }
            }
            std::fprintf(stderr, "[iop:pcprof] %llu instructions; top pcs:\n", static_cast<unsigned long long>(total));
            for (size_t i = 0; i < top.size() && i < 40u; ++i)
            {
                const Module *owner = nullptr;
                for (const auto &[id, module] : modules)
                    if (top[i].second >= module.base && top[i].second < module.base + module.size)
                        owner = &module;
                std::fprintf(stderr, "[iop:pcprof]   %08x %6.2f%% %s+0x%x\n", top[i].second, 100.0 * top[i].first / static_cast<double>(total),
                             owner ? owner->name.c_str() : "?", owner ? top[i].second - owner->base : top[i].second);
            }
            std::fill(s_hist.begin(), s_hist.end(), 0u);
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            const uint64_t previousBudgetEnd = nativeBudgetEnd;
            nativeBudgetEnd = start + instructionBudget;
            struct BudgetRestore
            {
                uint64_t &slot;
                uint64_t value;
                ~BudgetRestore() { slot = value; }
            } budgetRestore{nativeBudgetEnd, previousBudgetEnd};
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
                if (rpc.hasPendingEeCommands())
                    rpc.serviceEeCommands(*this);
                // The SPU2 runs on IOP time inside ANY guest execution, synchronous calls (module start, RPC) included:
                // a driver waiting for an SPU DMA to finish inside its start routine must see it finish.
                if (memory.spu2Enabled() && totalCycles - spu2LastAdvance >= kSpu2TickCycles)
                {
                    spu2LastAdvance = totalCycles;
                    ps2x::spu2::advance(totalCycles);
                    schedulePendingDma();
                }
            }
            activeCpu = previous;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            runCpu(cpu, budget);
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                    (void)intrman.dispatchInterrupt(irq, *this);
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        void runCycles(uint64_t cycles) noexcept
        {
            // PS2X_IOP_PERF=<seconds> (diagnostic, default OFF; rotk row 272): host TSC ticks spent in this function per
            // IOP instruction executed, printed every <seconds> -- the interpreter's cost per guest instruction.
            static const int s_perf = [] { const char *e = std::getenv("PS2X_IOP_PERF"); return e ? std::atoi(e) : 0; }();
            if (s_perf > 0)
            {
                static uint64_t s_ticks = 0u, s_instr = 0u;
                static auto s_last = std::chrono::steady_clock::now();
                const uint64_t t0 = __rdtsc(), i0 = totalInstructions;
                runCyclesImpl(cycles);
                s_ticks += __rdtsc() - t0;
                s_instr += totalInstructions - i0;
                const auto now = std::chrono::steady_clock::now();
                if (now - s_last >= std::chrono::seconds(s_perf))
                {
                    std::fprintf(stderr, "[iop:perf] %llu instructions, %.1f TSC ticks/instruction\n",
                                 static_cast<unsigned long long>(s_instr), s_instr ? double(s_ticks) / double(s_instr) : 0.0);
                    s_last = now;
                    s_ticks = s_instr = 0u;
                }
                return;
            }
            runCyclesImpl(cycles);
        }

        void runCyclesImpl(uint64_t cycles) noexcept
        {
            try
            {
                const uint64_t target = totalCycles + cycles;
                while (totalCycles < target)
                {
                    servicePendingDmaInterrupts();
                    servicePendingGuestCallbacks();
                    if (rpc.hasPendingEeCommands())
                        rpc.serviceEeCommands(*this);
                    if (memory.spu2Enabled())
                        ps2x::spu2::advance(totalCycles);   // PCSX2 SPU2async (counter 6): mix up to now
                    timrman.serviceDue(totalCycles, *this);
                    IopThread *next = kernel.beginNextReady(totalCycles);
                    if (!next)
                    {
                        uint64_t nextWake = kernel.nextWakeCycle(target);
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            nextWake = std::min(nextWake, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                        nextWake = timrman.nextEventCycle(nextWake);
                        totalCycles = std::max(totalCycles + 1u, std::min(target, nextWake));
                        continue;
                    }
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                    if (totalCycles == before)
                        ++totalCycles;
                }
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            if (nativeExecutor)   // rotk row 276: bind recompiled code before the module's own start runs
                nativeExecutor->onModuleLoaded(module.name, image.data(), image.size(), module.base, module.size);
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            if (nativeExecutor)
                nativeExecutor->onModuleUnloaded(it->second.base, it->second.size);
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        uint64_t spu2LastAdvance = 0u;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        const uint64_t iopCycles = total / 8u;
        m_impl->eeCycleCarry = total % 8u;
        if (iopCycles)
            m_impl->runCycles(iopCycles);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    bool IopEmulator::queueSifCommand(uint32_t cid, const void *packet, uint32_t packetSize,
                                      uint32_t eeExtraSource, uint32_t iopExtraDestination, uint32_t extraSize)
    {
        return m_impl->rpc.queueEeCommand(cid, packet, packetSize, eeExtraSource, iopExtraDestination, extraSize);
    }

    void IopEmulator::setNativeExecutor(ps2x::iop::IopNativeExecutor *executor)
    {
        m_impl->setNativeExecutor(executor);
    }

    void IopEmulator::setSpu2Enabled(bool enabled)
    {
        m_impl->setSpu2Enabled(enabled);
    }

    uint32_t IopEmulator::softwareRegister(uint32_t index) const noexcept
    {
        return m_impl->rpc.softwareRegister(index);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}
