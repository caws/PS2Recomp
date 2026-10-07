#pragma once

// rotk row 276: running statically recompiled IOP modules (ps2xRecomp R3000 mode, ps2x_irx_image) inside the IOP
// emulator. The emulator stays the IOP: its kernel, threads, imports, memory and hardware. When a thread's pc lands on
// a function a recompiled module provides, the emulator hands the thread to an IopNativeExecutor instead of
// interpreting; the executor runs native code until it returns to code it does not have, the thread blocks in an
// import, or the emulator asks for a break (interrupt, budget), and hands the thread back with its registers updated.
// The executor lives on the runtime side (ps2xRuntime, ps2_iop_native.cpp), where the generated code's types are.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ps2x::iop
{
    // A thread's registers, as the interpreter keeps them.
    struct IopNativeCpu
    {
        uint32_t *gpr = nullptr;   // [32]
        uint32_t *hi = nullptr;
        uint32_t *lo = nullptr;
        uint32_t *pc = nullptr;
        bool *stopped = nullptr;
        bool *yielded = nullptr;
    };

    // What the emulator offers native code.
    class IopNativeServices
    {
    public:
        virtual ~IopNativeServices() = default;
        [[nodiscard]] virtual uint8_t *ram() = 0;     // IOP RAM (2 MB), physical addressing
        [[nodiscard]] virtual uint8_t *owned() = 0;   // the emulator's per-byte "written" map, kept in step by native stores
        [[nodiscard]] virtual uint32_t read(uint32_t address, unsigned bytes) = 0;    // anything not in RAM
        virtual void write(uint32_t address, uint32_t value, unsigned bytes) = 0;
        // The import stub at `stubAddress` (`library:ordinal`) was called, with the thread's registers in `cpu`; it is
        // decoded and dispatched exactly as the interpreter does (version included), then the stub's `jr ra` is done
        // here (or the jump into the exporting module). True = the thread can go on.
        virtual bool callImport(IopNativeCpu &cpu, uint32_t stubAddress, std::string_view library, uint16_t ordinal) = 0;
        virtual void account(uint32_t instructions) = 0;   // guest instructions executed natively since the last call
        [[nodiscard]] virtual bool breakRequested() = 0;   // stop at the next checkpoint (interrupts, budget, ...)
    };

    class IopNativeExecutor
    {
    public:
        virtual ~IopNativeExecutor() = default;
        // A module image was placed at `base` (`size` bytes; `image` = the IRX file). The executor binds the native
        // functions it has for exactly this module at exactly this base, if any.
        virtual void onModuleLoaded(std::string_view name, const uint8_t *image, size_t imageSize, uint32_t base, uint32_t size) = 0;
        virtual void onModuleUnloaded(uint32_t base, uint32_t size) = 0;
        [[nodiscard]] virtual bool has(uint32_t pc) const = 0;
        // Run native code from *cpu.pc (the emulator calls this only when has(*cpu.pc), no branch or load delay pending).
        virtual void run(IopNativeCpu &cpu, IopNativeServices &services) = 0;
    };
}
