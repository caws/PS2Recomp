// ps2xRuntime shim for PCSX2's Gif_Unit.h: the three calls XGKICK makes, routed to our GIF arbiter
// (implemented in ../ps2x_microvu.cpp).
#pragma once
#include "common/Pcsx2Defs.h"
#include "Gif.h"
struct Gif_PathShim
{
	void CopyGSPacketData(u8* pMem, u32 size, bool aligned = false);
};
struct Gif_UnitShim
{
	Gif_PathShim gifPath[3];
	u32 GetGSPacketSize(GIF_PATH pathIdx, u8* pMem, u32 offset = 0, u32 size = ~0u, bool flush = false);
	u32 TransferGSPacketData(GIF_TRANSFER_TYPE tranType, u8* pMem, u32 size, bool aligned = false);
};
extern Gif_UnitShim gifUnit;
