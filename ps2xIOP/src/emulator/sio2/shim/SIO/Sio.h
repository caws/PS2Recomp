// ps2x SIO2 port shim: PCSX2 SIO/Sio.h without the memory-card file backend (MemoryCardFile). In this first cut no
// memory card is attached: IsPresent() is false, so SIO2 answers every card command DISCONNECTED (PCSX2 behaviour
// for an empty slot). The port/slot helpers and AutoEject/MemcardBusy hooks are implemented by the seam.
#pragma once
#include "common/Pcsx2Types.h"
#include "SIO/SioTypes.h"   // PCSX2 reaches it through Sio.h's include chain (Terminator, SIO::PORTS)
#include <string>
#include <tuple>
#include <vector>
#include <cstring>

// PCSX2 SIO/Memcard/MemoryCardFile.h
struct McdSizeInfo
{
    u16 SectorSize;
    u16 EraseBlockSizeInSectors;
    u32 McdSizeInSectors;
    u8 Xor;
};

struct _mcd
{
    u8 currentCommand;
    u8 term;
    bool goodSector;
    u8 msb;
    u8 lsb;
    u32 sectorAddr;
    u32 transferAddr;
    std::vector<u8> buf;
    u8 FLAG;
    u8 port;
    u8 slot;
    size_t autoEjectTicks;

    // No card attached: the FileMcd_* backend calls PCSX2 makes here have nothing to act on.
    void GetSizeInfo(McdSizeInfo &info) { info = McdSizeInfo{}; }
    bool IsPSX() { return false; }
    void EraseBlock() {}
    void Read(u8 *dest, int size) { std::memset(dest, 0xff, static_cast<size_t>(size)); }
    void Write(u8 *, int) {}
    bool IsPresent() { return false; }
    u64 GetChecksum() { return 0; }
};

extern _mcd mcds[2][4];
extern _mcd *mcd;

extern std::tuple<u32, u32> sioConvertPadToPortAndSlot(u32 index);
extern u32 sioConvertPortAndSlotToPad(u32 port, u32 slot);

namespace AutoEject
{
    extern void SetAll();
}

namespace MemcardBusy
{
    extern void SetBusy();
}

extern uint32_t sioLastFrameMcdBusy;
