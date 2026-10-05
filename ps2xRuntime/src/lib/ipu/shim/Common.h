// ps2xRuntime shim for PCSX2's Common.h -- only what the IPU sources (src/lib/ipu/pcsx2/IPU/*) need.
// The PS2-side state PCSX2 keeps in its EE core (cpuRegs, the hardware register page, the two IPU DMA
// channels, the DMAC control block, the event scheduler and the interrupt lines) is provided by the
// seam, ../ps2x_ipu.cpp, in the shapes the verbatim files expect. See ../README.md.
#pragma once
// The microVU port's seam (src/lib/vu/microvu/ps2x_microvu.cpp) defines PCSX2's cpuRegs / Console /
// DevCon / hwIntcIrq for the recompiler; both static libraries link into one runner, so this port's
// copies of the same PCSX2 names are renamed at the preprocessor level (the verbatim sources stay
// untouched and keep spelling them the PCSX2 way).
#define cpuRegs ps2x_ipu_cpuRegs
#define Console ps2x_ipu_Console
#define DevCon ps2x_ipu_DevCon
#define hwIntcIrq ps2x_ipu_hwIntcIrq
#define hwDmacIrq ps2x_ipu_hwDmacIrq
#define CPU_INT ps2x_ipu_CPU_INT
#define dmaGetAddr ps2x_ipu_dmaGetAddr
#define hwDmacSrcChain ps2x_ipu_hwDmacSrcChain
#define hwDmacSrcTadrInc ps2x_ipu_hwDmacSrcTadrInc
#define eeHw ps2x_ipu_eeHw
#define dmacRegs ps2x_ipu_dmacRegs
#define ipu0ch ps2x_ipu_ipu0ch
#define ipu1ch ps2x_ipu_ipu1ch
#include "common/Pcsx2Defs.h"
#include "common/Pcsx2Types.h"
#include "common/Assertions.h"
#include "common/SingleRegisterTypes.h"
#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#ifndef ARCH_X86
#define ARCH_X86 1
#endif
#ifndef _M_X86
#define _M_X86 1
#endif

typedef u128 mem128_t;

// ---- logging: the IPU files log through these; only warnings/errors are kept ---------------------
#define IPU_LOG(...) ((void)0)
#define DMA_LOG(...) ((void)0)
struct Ps2xIpuConsole
{
	void WriteLn() const {}
	void WriteLn(const std::string&) const {}
	void WriteLn(const char*, ...) const {}
	void Warning(const char* fmt, ...) const
	{
		va_list ap; va_start(ap, fmt);
		std::fprintf(stderr, "[ipu:pcsx2] ");
		std::vfprintf(stderr, fmt, ap);
		std::fprintf(stderr, "\n");
		va_end(ap);
	}
	void Error(const char* fmt, ...) const
	{
		va_list ap; va_start(ap, fmt);
		std::fprintf(stderr, "[ipu:pcsx2] ERROR ");
		std::vfprintf(stderr, fmt, ap);
		std::fprintf(stderr, "\n");
		va_end(ap);
	}
};
extern Ps2xIpuConsole Console;
extern Ps2xIpuConsole DevCon;

namespace StringUtil
{
	inline std::string StdStringFromFormat(const char* fmt, ...)
	{
		char buf[512];
		va_list ap; va_start(ap, fmt);
		std::vsnprintf(buf, sizeof(buf), fmt, ap);
		va_end(ap);
		return std::string(buf);
	}
}

// ---- savestates: the freeze members compile against this and are never called ---------------------
class SaveStateBase
{
public:
	bool FreezeTag(const char*) { return true; }
	template <typename T> void Freeze(T&) {}
	bool IsOkay() const { return true; }
	bool ipuFreeze();
	bool ipuDmaFreeze();
};

// ---- the EE core state the IPU touches ------------------------------------------------------------
// PCSX2 event ids (R5900.h) that the IPU files use with CPU_INT / cpuRegs.interrupt.
enum
{
	DMAC_FROM_IPU = 3,
	DMAC_TO_IPU = 4,
	IPU_PROCESS = 17,
};
enum
{
	INTC_IPU = 8, // INTC_STAT bit 8
};
#define BIAS 2

struct Ps2xCpuRegs
{
	u32 interrupt = 0;   // pending-event bitmask (PCSX2 tests bits 4 and IPU_PROCESS)
	u32 eCycle[32] = {}; // per-event cycle counts; 0x9999 marks a DMA parked on the IPU
	u64 cycle = 0;
	u32 pc = 0;
};
extern Ps2xCpuRegs cpuRegs;

// The hardware register page: PCSX2 maps ipuRegs onto eeHw[0x2000] (the 0x10002000 block).
extern u8 eeHw[0x10000]; // defined alignas(16) in the seam
#define psHu32(mem) (*(u32*)&eeHw[(mem) & 0xffff])
#define psHu64(mem) (*(u64*)&eeHw[(mem) & 0xffff])
#define IPU_CMD 0x10002000
#define IPU_CTRL 0x10002010
#define IPU_BP 0x10002020
#define IPU_TOP 0x10002030

// DMA tags and channels (PCSX2 Dmac.h), the subset the IPU DMA code reads.
enum TagId
{
	TAG_REFE = 0,
	TAG_CNT = 1,
	TAG_NEXT = 2,
	TAG_REF = 3,
	TAG_REFS = 4,
	TAG_CALL = 5,
	TAG_RET = 6,
	TAG_END = 7,
};
enum ChcrMode
{
	NORMAL_MODE = 0,
	CHAIN_MODE = 1,
	INTERLEAVE_MODE = 2,
};
enum
{
	STS_fromIPU = 2, // DMAC CTRL.STS = fromIPU stall source
};

// One 32-bit word of a DMA tag (PCSX2 Dmac.h): a tag qword is tDMA_TAG[4]; [0] = control, [1] = ADDR.
union tDMA_TAG
{
	struct
	{
		u32 QWC : 16;
		u32 _reserved2 : 10;
		u32 PCE : 2;
		u32 ID : 3;
		u32 IRQ : 1;
	};
	u32 _u32;
};

union tDMA_CHCR
{
	struct
	{
		u32 DIR : 1;
		u32 _reserved1 : 1;
		u32 MOD : 2;
		u32 ASP : 2;
		u32 TTE : 1;
		u32 TIE : 1;
		u32 STR : 1;
		u32 _reserved2 : 7;
		u32 TAG : 16;
	};
	u32 _u32;
	tDMA_TAG tag() const { tDMA_TAG t; t._u32 = TAG << 16; return t; }
};

struct DMACh
{
	tDMA_CHCR chcr;
	u32 madr;
	u32 qwc;
	u32 tadr;
	u32 asr0;
	u32 asr1;
	u32 sadr;
	// PCSX2 DMACh::transfer: read the chain tag into the channel (qwc + CHCR.TAG). The tag word is
	// validated by the caller through hwDmacSrcChain's ID handling.
	bool transfer(const char* /*s*/, tDMA_TAG* tag)
	{
		chcr.TAG = (tag->_u32 >> 16) & 0xffffu;
		qwc = tag->QWC;
		return true;
	}
};
extern DMACh ipu0ch; // fromIPU (channel 3)
extern DMACh ipu1ch; // toIPU  (channel 4)

struct Ps2xDmacRegs
{
	struct { u32 STS = 0; } ctrl;
	struct { u32 ADDR = 0; } stadr;
};
extern Ps2xDmacRegs dmacRegs;

// Event scheduling (PCSX2 R5900 CPU_INT): the seam records the request; ps2xIpuPump() runs them.
void CPU_INT(int evt, int cycles);
inline void CPU_SET_DMASTALL(int /*evt*/, bool /*stall*/) {}
void hwIntcIrq(int cause);
void hwDmacIrq(int channel);
// Guest memory access for the DMA (PCSX2 Hw.cpp dmaGetAddr): a host pointer into RAM/scratchpad.
tDMA_TAG* dmaGetAddr(u32 addr, bool write);
// Source-chain tag handling (PCSX2 Hw.cpp hwDmacSrcChain / hwDmacSrcTadrInc).
bool hwDmacSrcChain(DMACh& dma, int id);
void hwDmacSrcTadrInc(DMACh& dma);
// The FIFO register accessors (PCSX2 Hw.h), defined in IPU/IPU_Fifo.cpp.
void WriteFIFO_IPUin(const mem128_t* value);
void ReadFIFO_IPUout(mem128_t* out);
