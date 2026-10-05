#ifndef PS2_ADDRESS_H
#define PS2_ADDRESS_H

#include <cstdint>

#include "runtime/ps2_memory.h"

static inline constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_BASE = 0x20000000u;
static inline constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_SIZE = 0x20000000u;
static inline constexpr uint32_t PS2_KSEG0_BASE = 0x80000000u;
static inline constexpr uint32_t PS2_KSEG0_KSEG1_SIZE = 0x40000000u;
static inline constexpr uint32_t PS2_KSEG2_BASE = 0xC0000000u;

static inline constexpr bool Ps2AddressInRange(uint32_t value, uint32_t base, uint32_t size)
{
    return (value - base) < size;
}

static inline constexpr bool Ps2IsUncachedRamMirrorAddress(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_EE_UNCACHED_RAM_MIRROR_BASE, PS2_EE_UNCACHED_RAM_MIRROR_SIZE);
}

static inline constexpr bool Ps2IsKseg01Address(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_KSEG0_BASE, PS2_KSEG0_KSEG1_SIZE);
}

static __attribute__((always_inline)) inline constexpr bool Ps2IsKseg23Address(uint32_t addr)
{
    return addr >= PS2_KSEG2_BASE;
}

static inline constexpr uint32_t Ps2DirectMappedPhysicalAddress(uint32_t addr)
{
    return addr & 0x1FFFFFFFu;
}

static __attribute__((always_inline)) inline constexpr uint32_t Ps2PhysicalAddress(uint32_t addr)
{
    return (addr >= PS2_KSEG0_BASE) ? Ps2DirectMappedPhysicalAddress(addr) : addr;
}

static __attribute__((always_inline)) inline constexpr bool Ps2IsPhysicalSpecialAddress(uint32_t physAddr)
{
    return Ps2AddressInRange(physAddr, PS2_BIOS_BASE, PS2_BIOS_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_SCRATCHPAD_BASE, PS2_SCRATCHPAD_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_IO_BASE, PS2_IO_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE) ||
           // The EE maps ALL FOUR VU memories, 0x11000000..0x1100FFFF (PCSX2 Memory.cpp memMapVUmicro: VU0
           // micro 0x11000000, VU0 data 0x11004000, VU1 micro 0x11008000, VU1 data 0x1100c000, 0x4000 each).
           // This used to start at VU0 DATA and stop at the end of VU1 MICRO, so a store to VU0 micro or
           // VU1 data took the RAM fast path and landed masked in main RAM (rotk: an `sq` to 0x1100C840
           // zeroed 64 bytes of the HUD texture set at 0x0100C840 -- a stray green dash under the health bar).
           (physAddr >= PS2_VU0_CODE_BASE && physAddr < (PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE));
}

// cont.317: force-inlined -- it was an out-of-line call on every guest load/store (see ps2_runtime_macros.h).
static __attribute__((always_inline)) inline constexpr bool Ps2IsSpecialAddress(uint32_t addr)
{
    if (Ps2IsKseg23Address(addr))
        return true;

    return Ps2IsPhysicalSpecialAddress(Ps2PhysicalAddress(addr));
}

#endif // PS2_ADDRESS_H
