// ps2xRuntime <-> PCSX2 IPU seam. See ps2x_ipu.h and README.md.
#include "Common.h"
#include "IPU/IPU.h"
#include "IPU/IPU_Fifo.h"
#include "IPU/IPUdma.h"
#include "IPU/IPU_MultiISA.h"
#include "ps2x_ipu.h"

#include <cstdlib>
#include <cstring>

// ---- the EE-side globals the verbatim files reference -----------------------------------------------
Ps2xIpuConsole Console;
Ps2xIpuConsole DevCon;
Ps2xCpuRegs cpuRegs;
alignas(16) u8 eeHw[0x10000];
DMACh ipu0ch{};
DMACh ipu1ch{};
Ps2xDmacRegs dmacRegs;

namespace
{
	uint8_t* s_rdram = nullptr;
	uint8_t* s_spr = nullptr;
	PS2Memory* s_memory = nullptr;
	uint32_t s_pending = 0;      // CPU_INT requests not yet run
	bool s_dmaIrq[8] = {};       // hwDmacIrq(channel) raised since the last take
	bool s_intcIpu = false;      // hwIntcIrq(INTC_IPU) raised since the last take
	unsigned long s_kicks = 0, s_cmds = 0, s_pumpMax = 0;
	unsigned long s_stops = 0, s_ignoredKicks = 0, s_cmdHist[16] = {}; // cont.247: PS2X_IPU_STATS
	uint32_t s_fromBase = 0, s_fromQwc = 0; // the last fromIPU kick (PS2X_IPU_DUMP)
	int s_dumps = 0, s_dumpSeen = 0;

	// PS2X_IPU_DUMP=<dir>: after a fromIPU transfer of >= 256 macroblocks completes, write the guest
	// buffer as two PPMs -- the RGB32 macroblocks read row-major (the IPU's order per PCSX2) and
	// column-major (the layout LOTR's old HLE decoder wrote) -- to tell a decoder fault from a layout one.
	void dumpFromIpu(const uint8_t* base, uint32_t qwc)
	{
		const char* dir = std::getenv("PS2X_IPU_DUMP");
		if (!dir || !*dir || s_dumps >= 4) return;
		const uint32_t mbs = qwc / 64u;
		if (mbs < 256u || !base) return;
		// cont.247: PS2X_IPU_DUMP_SKIP=<n> skips the first n qualifying transfers (the natural boot decodes
		// the title/menu ipum pictures before the FMV's CSC batches; the budget stays 4 dumps).
		static const int s_skip = []{ const char* e = std::getenv("PS2X_IPU_DUMP_SKIP"); return e ? std::atoi(e) : 0; }();
		// PS2X_IPU_DUMP_CSC=1 dumps only buffers filled by a CSC command (libmpeg's FMV output), never IDEC's
		// (the menu's ipum pictures, re-decoded every frame -- a skip count cannot separate the two).
		static const bool s_cscOnly = []{ const char* e = std::getenv("PS2X_IPU_DUMP_CSC"); return e && e[0] && e[0] != '0'; }();
		if (s_cscOnly && (ipu_cmd.current >> 28) != 7u) return;
		if (s_dumpSeen++ < s_skip) return;
		const uint32_t mbw = 32u, mbh = (mbs + mbw - 1u) / mbw;
		const uint32_t W = mbw * 16u, H = mbh * 16u;
		for (int mode = 0; mode < 2; ++mode)
		{
			char path[512];
			std::snprintf(path, sizeof(path), "%s/ipu_from_%d_%s.ppm", dir, s_dumps, mode == 0 ? "rowmajor" : "colmajor");
			FILE* f = std::fopen(path, "wb");
			if (!f) return;
			std::fprintf(f, "P6\n%u %u\n255\n", W, H);
			for (uint32_t y = 0; y < H; ++y)
				for (uint32_t x = 0; x < W; ++x)
				{
					const uint32_t mb = (y / 16u) * mbw + (x / 16u);
					const uint32_t ix = x & 15u, iy = y & 15u;
					const uint32_t off = mb * 1024u + (mode == 0 ? (iy * 16u + ix) : (ix * 16u + iy)) * 4u;
					uint8_t px[3] = {0, 0, 0};
					if (mb < mbs) { px[0] = base[off]; px[1] = base[off + 1]; px[2] = base[off + 2]; }
					std::fwrite(px, 1, 3, f);
				}
			std::fclose(f);
			std::fprintf(stderr, "[ipu] dumped %s (%u MBs, %ux%u)\n", path, mbs, W, H);
		}
		++s_dumps;
	}

	// cont.247: PS2X_IPU_STATS=<n> prints a summary line every n IPU commands (the FMV issues several
	// commands per macroblock; PS2X_IPU_LOG is far too verbose for it).
	void stats()
	{
		static const unsigned long every = []{ const char* e = std::getenv("PS2X_IPU_STATS"); return e ? std::strtoul(e, nullptr, 10) : 0ul; }();
		if (!every || (s_cmds % every) != 0) return;
		std::fprintf(stderr, "[ipu:stats] cmds=%lu bclr=%lu idec=%lu bdec=%lu vdec=%lu fdec=%lu setiq=%lu setvq=%lu csc=%lu pack=%lu setth=%lu"
		             " | kicks=%lu stops=%lu ignoredKicks=%lu pumpMax=%lu | busy=%u ifc=%u ofc=%u ipu0{str=%u qwc=%u} ipu1{str=%u qwc=%u}\n",
		             s_cmds, s_cmdHist[0], s_cmdHist[1], s_cmdHist[2], s_cmdHist[3], s_cmdHist[4], s_cmdHist[5], s_cmdHist[6], s_cmdHist[7], s_cmdHist[8], s_cmdHist[9],
		             s_kicks, s_stops, s_ignoredKicks, s_pumpMax, ipuRegs.ctrl.BUSY, g_BP.IFC, ipuRegs.ctrl.OFC,
		             ipu0ch.chcr.STR, ipu0ch.qwc, ipu1ch.chcr.STR, ipu1ch.qwc);
	}

	bool verbose()
	{
		static const bool v = []{ const char* e = std::getenv("PS2X_IPU_LOG"); return e && e[0] && e[0] != '0'; }();
		return v;
	}

	// Run the scheduled IPU/DMA "interrupt events" until nothing is pending. PCSX2 orders these by EE
	// cycle; here the decoder is serviced first (it consumes input and produces output), then the output
	// DMA, then the input DMA -- every order converges because each handler re-schedules what it still
	// needs, and the loop is bounded.
	void pump()
	{
		unsigned long n = 0;
		while (s_pending)
		{
			if (++n > 2000000ul)
			{
				std::fprintf(stderr, "[ipu] pump did not converge (pending=0x%x cmd=0x%x busy=%u ifc=%u ofc=%u ipu0{str=%u qwc=%u} ipu1{str=%u qwc=%u})\n",
				             s_pending, ipu_cmd.current, ipuRegs.ctrl.BUSY, g_BP.IFC, ipuRegs.ctrl.OFC,
				             ipu0ch.chcr.STR, ipu0ch.qwc, ipu1ch.chcr.STR, ipu1ch.qwc);
				s_pending = 0;
				cpuRegs.interrupt = 0;
				break;
			}
			if (s_pending & (1u << IPU_PROCESS))
			{
				s_pending &= ~(1u << IPU_PROCESS);
				cpuRegs.interrupt &= ~(1u << IPU_PROCESS);
				ipuCMDProcess();
				continue;
			}
			if (s_pending & (1u << DMAC_FROM_IPU))
			{
				s_pending &= ~(1u << DMAC_FROM_IPU);
				cpuRegs.interrupt &= ~(1u << DMAC_FROM_IPU);
				ipu0Interrupt();
				continue;
			}
			if (s_pending & (1u << DMAC_TO_IPU))
			{
				s_pending &= ~(1u << DMAC_TO_IPU);
				cpuRegs.interrupt &= ~(1u << DMAC_TO_IPU);
				ipu1Interrupt();
				continue;
			}
			s_pending = 0; // an event id we do not model
		}
		if (n > s_pumpMax) s_pumpMax = n;
	}
}

// ---- PCSX2 core services ----------------------------------------------------------------------------
void CPU_INT(int evt, int cycles)
{
	cpuRegs.interrupt |= 1u << evt;
	cpuRegs.eCycle[evt] = static_cast<u32>(cycles);
	s_pending |= 1u << evt;
}

void hwIntcIrq(int /*cause*/)
{
	s_intcIpu = true;
}

void hwDmacIrq(int channel)
{
	if (channel >= 0 && channel < 8) s_dmaIrq[channel] = true;
}

tDMA_TAG* dmaGetAddr(u32 addr, bool /*write*/)
{
	// PCSX2 Dmac.cpp dmaGetAddr: `if (DMA_TAG(addr).SPR) return &eeMem->Scratch[addr & 0x3ff0]` -- bit 31 of
	// a DMA address selects the scratchpad (libmpeg's receiveDataFromIPU lands the BDEC blocks there:
	// D3_MADR = (spr | 0x80000000)); the 0x70000000 CPU mirror maps there too (PCSX2: addr & 0x1ffffff0
	// < 0x10004000 -> Scratch).
	if ((addr & 0x80000000u) || (addr & 0x70000000u) == 0x70000000u)
	{
		if (!s_spr) return nullptr;
		return reinterpret_cast<tDMA_TAG*>(s_spr + (addr & 0x3FF0u));
	}
	const u32 phys = addr & 0x01FFFFFFu;
	if (!s_rdram || phys >= 0x02000000u) return nullptr;
	return reinterpret_cast<tDMA_TAG*>(s_rdram + phys);
}

// PCSX2 Hw.cpp hwDmacSrcChain: the caller has already loaded MADR from the tag's ADDR word and QWC from
// the tag; this applies the per-ID MADR/TADR rules and says whether the chain ends after this packet.
bool hwDmacSrcChain(DMACh& dma, int id)
{
	switch (id)
	{
		case TAG_REFE: // transfer QWC from ADDR, then end
			dma.tadr += 16;
			return true;
		case TAG_CNT: // transfer QWC following the tag; the next tag follows the data
			dma.madr = dma.tadr + 16;
			dma.tadr = dma.madr;
			return false;
		case TAG_NEXT: // transfer QWC following the tag; next tag at ADDR
		{
			const u32 next = dma.madr;
			dma.madr = dma.tadr + 16;
			dma.tadr = next;
			return false;
		}
		case TAG_REF:
		case TAG_REFS: // transfer QWC from ADDR; next tag follows this tag
			dma.tadr += 16;
			return false;
		case TAG_CALL: // transfer QWC following the tag; push the return tag; next tag at ADDR
		{
			const u32 next = dma.madr;
			dma.madr = dma.tadr + 16;
			const u32 ret = dma.madr + (dma.qwc << 4);
			if (dma.chcr.ASP == 0) { dma.asr0 = ret; dma.chcr.ASP = 1; }
			else if (dma.chcr.ASP == 1) { dma.asr1 = ret; dma.chcr.ASP = 2; }
			dma.tadr = next;
			return false;
		}
		case TAG_RET: // transfer QWC following the tag; next tag from the stack (or end)
			dma.madr = dma.tadr + 16;
			if (dma.chcr.ASP == 2) { dma.tadr = dma.asr1; dma.chcr.ASP = 1; return false; }
			if (dma.chcr.ASP == 1) { dma.tadr = dma.asr0; dma.chcr.ASP = 0; return false; }
			return true;
		case TAG_END: // transfer QWC following the tag, then end
			dma.madr = dma.tadr + 16;
			dma.tadr = dma.madr;
			return true;
		default:
			return true;
	}
}

void hwDmacSrcTadrInc(DMACh& dma)
{
	if (dma.chcr.STR == 0) return;
	if (dma.chcr.MOD != CHAIN_MODE) return;
	const u32 tagid = (dma.chcr.TAG >> 12) & 0x7u;
	if (tagid == TAG_CNT) dma.tadr = dma.madr;
}

// ---- the runtime-facing API ------------------------------------------------------------------------
namespace ps2x_ipu
{
	bool enabled()
	{
		static const bool on = []{ const char* e = std::getenv("PS2X_IPU"); return !(e && e[0] == '0'); }();
		return on;
	}

	void reset()
	{
		std::memset(eeHw, 0, sizeof(eeHw));
		std::memset(&ipu0ch, 0, sizeof(ipu0ch));
		std::memset(&ipu1ch, 0, sizeof(ipu1ch));
		cpuRegs = Ps2xCpuRegs{};
		dmacRegs = Ps2xDmacRegs{};
		s_pending = 0;
		std::memset(s_dmaIrq, 0, sizeof(s_dmaIrq));
		s_intcIpu = false;
		ipuReset();
	}

	void init(PS2Memory* memory, uint8_t* rdram, uint8_t* scratchpad)
	{
		s_memory = memory;
		s_rdram = rdram;
		s_spr = scratchpad;
		reset();
		std::fprintf(stderr, "[ipu] PCSX2 IPU active (PS2X_IPU=0 restores the register stub; PS2X_IPU_LOG=1 traces commands and DMA)\n");
	}

	uint32_t read32(uint32_t address)
	{
		const uint32_t v = ipuRead32(address);
		if (verbose() && (address & 0xff) == 0x00)
			std::fprintf(stderr, "[ipu] read CMD -> 0x%08x (busy=%u)\n", v, ipuRegs.cmd.BUSY ? 1u : 0u);
		return v;
	}

	void write32(uint32_t address, uint32_t value)
	{
		if (verbose())
			std::fprintf(stderr, "[ipu] write 0x%08x = 0x%08x%s\n", address, value,
			             (address & 0xff) == 0 ? " (CMD)" : (address & 0xff) == 0x10 ? " (CTRL)" : "");
		if ((address & 0xff) == 0x00) { ++s_cmds; ++s_cmdHist[value >> 28]; stats(); }
		ipuWrite32(address, value);
		pump();
		if (verbose() && (address & 0xff) == 0x00 && (value >> 28) == 5)
		{
			const u8* q = ((value >> 27) & 1) ? decoder.niq : decoder.iq;
			std::fprintf(stderr, "[ipu] SETIQ %s ->", ((value >> 27) & 1) ? "non-intra" : "intra");
			for (int i = 0; i < 64; ++i) std::fprintf(stderr, "%s%u", (i % 8) ? " " : " | ", q[i]);
			std::fprintf(stderr, "  (busy=%u ifc=%u)\n", ipuRegs.ctrl.BUSY ? 1u : 0u, g_BP.IFC);
		}
		if (verbose() && (address & 0xff) == 0x00 && (value >> 28) == 1)
			std::fprintf(stderr, "[ipu] IDEC qsc=%u dtd=%u sgn=%u dte=%u ofm=%u | ctrl idp=%u as=%u ivf=%u qst=%u mp1=%u | BP=%u IFC=%u FP=%u\n",
			             (value >> 16) & 0x1f, (value >> 24) & 1, (value >> 25) & 1, (value >> 26) & 1, (value >> 27) & 1,
			             ipuRegs.ctrl.IDP, ipuRegs.ctrl.AS, ipuRegs.ctrl.IVF, ipuRegs.ctrl.QST, ipuRegs.ctrl.MP1, g_BP.BP, g_BP.IFC, g_BP.FP);
	}

	uint64_t read64(uint32_t address)
	{
		// PCSX2 HwRead.cpp _hwRead64: page 0x02 -> ipuRead64 (IPU_CMD: cmd.DATA | cmd.BUSY << 32; IPU_TOP:
		// top | topbusy << 32 -- the guest tests bit 63 with bgez/bltz).
		const uint64_t v = ipuRead64(address);
		if (verbose() && (address & 0xff) == 0x00)
			std::fprintf(stderr, "[ipu] read64 CMD -> 0x%016llx\n", static_cast<unsigned long long>(v));
		return v;
	}

	void write64(uint32_t address, uint64_t value)
	{
		// PCSX2 HwWrite.cpp: page 0x02 -> ipuWrite64 (only IPU_CMD takes a 64-bit write; the low word is
		// the command).
		if (verbose())
			std::fprintf(stderr, "[ipu] write64 0x%08x = 0x%016llx\n", address, static_cast<unsigned long long>(value));
		if ((address & 0xff) == 0x00) { ++s_cmds; ++s_cmdHist[(static_cast<uint32_t>(value)) >> 28]; stats(); }
		if (ipuWrite64(address, value))
			psHu64(address) = value; // "TRUE if the caller should do the writeback itself"
		pump();
	}

	void writeFifoIn(const uint32_t* qword)
	{
		mem128_t v;
		std::memcpy(&v, qword, 16);
		WriteFIFO_IPUin(&v);
		pump();
	}

	void readFifoOut(uint32_t* qword)
	{
		mem128_t v;
		std::memset(&v, 0, sizeof(v));
		ReadFIFO_IPUout(&v);
		std::memcpy(qword, &v, 16);
		pump();
	}

	void dmaWriteStopped(int channel, uint32_t* chcr)
	{
		DMACh& ch = (channel == 3) ? ipu0ch : ipu1ch;
		const int evt = (channel == 3) ? DMAC_FROM_IPU : DMAC_TO_IPU;
		if (verbose())
			std::fprintf(stderr, "[ipu] dma ch%d CHCR=0x%08x (STR clear; was 0x%08x madr=0x%08x qwc=%u tadr=0x%08x)\n",
			             channel, *chcr, ch.chcr._u32, ch.madr, ch.qwc, ch.tadr);
		if (ch.chcr.STR)
		{
			// PCSX2 Dmac.cpp DmaExec, `if (reg.chcr.STR) { if (chcr.STR == 0) { reg.chcr.STR = 0; cpuClearInt(channel);
			// QueuedDMA &= ~(1 << channel); } return; }`: the stop keeps every other field and the progress.
			ch.chcr.STR = 0;
			s_pending &= ~(1u << evt);
			cpuRegs.interrupt &= ~(1u << evt);
			++s_stops;
		}
		else
		{
			ch.chcr._u32 = *chcr; // DmaExec: `reg.chcr.set(value)` on a stopped channel (STR stays clear: no start)
		}
		*chcr = ch.chcr._u32;
	}

	void dmaKick(int channel, uint32_t* chcr, uint32_t* madr, uint32_t* qwc, uint32_t* tadr,
	             uint32_t* asr0, uint32_t* asr1, bool* finished)
	{
		DMACh& ch = (channel == 3) ? ipu0ch : ipu1ch;
		if (ch.chcr.STR)
		{
			// PCSX2 Dmac.cpp DmaExec: a CHCR write while the channel runs changes nothing but STR ("Fields other
			// than STR can only be written to when the DMA is stopped"); with STR set again it is ignored.
			++s_ignoredKicks;
			static unsigned long n = 0;
			if (++n <= 20 || verbose())
				std::fprintf(stderr, "[ipu] WARN dma kick ch%d ignored: channel still running (chcr=0x%08x madr=0x%08x qwc=%u tadr=0x%08x; write 0x%08x madr=0x%08x qwc=%u tadr=0x%08x)\n",
				             channel, ch.chcr._u32, ch.madr, ch.qwc, ch.tadr, *chcr, *madr, *qwc & 0xFFFFu, *tadr);
			*chcr = ch.chcr._u32;
			*madr = ch.madr;
			*qwc = ch.qwc;
			*tadr = ch.tadr;
			*asr0 = ch.asr0;
			*asr1 = ch.asr1;
			*finished = false;
			return;
		}
		ch.chcr._u32 = *chcr;
		ch.madr = *madr;
		ch.qwc = *qwc & 0xFFFFu;
		ch.tadr = *tadr;
		ch.asr0 = *asr0;
		ch.asr1 = *asr1;
		// PCSX2 Dmac.cpp DmaExec: "if NORMAL mode is started with 0 QWC it will actually transfer 1 QWC
		// then underflows and transfer another 0xFFFF QWC's" -- modelled as 0x10000 qwords; dmaIPU0 then
		// ends the channel if nothing was there to transfer. LOTR's ipum driver arms fromIPU for a whole
		// 1024-macroblock picture (65536 qwords) first, i.e. a 16-bit QWC of 0.
		if (ch.chcr.STR && ch.chcr.MOD == 0 && ch.qwc == 0)
			ch.qwc = 0x10000u;
		++s_kicks;
		// PS2X_IPU_DUMP bookkeeping uses the guest's QWC (a 0 = the driver's empty whole-picture arm, which
		// the rule above ends at once with nothing transferred -- not a picture to dump).
		if (channel == 3) { s_fromBase = *madr; s_fromQwc = *qwc & 0xFFFFu; }
		if (verbose())
			std::fprintf(stderr, "[ipu] dma kick ch%d chcr=0x%08x madr=0x%08x qwc=%u tadr=0x%08x\n", channel, *chcr, *madr, ch.qwc, *tadr);
		if (channel == 3)
			dmaIPU0();
		else
			dmaIPU1();
		pump();
		*chcr = ch.chcr._u32;
		*madr = ch.madr;
		*qwc = ch.qwc;
		*tadr = ch.tadr;
		*asr0 = ch.asr0;
		*asr1 = ch.asr1;
		*finished = s_dmaIrq[channel];
		s_dmaIrq[channel] = false;
		if (channel == 3 && *finished)
			dumpFromIpu(reinterpret_cast<const uint8_t*>(dmaGetAddr(s_fromBase, false)), s_fromQwc);
	}

	bool dmaRunning(int channel)
	{
		const DMACh& ch = (channel == 3) ? ipu0ch : ipu1ch;
		return ch.chcr.STR != 0;
	}

	void channelState(int channel, uint32_t* chcr, uint32_t* madr, uint32_t* qwc, uint32_t* tadr, bool* finished)
	{
		const DMACh& ch = (channel == 3) ? ipu0ch : ipu1ch;
		*chcr = ch.chcr._u32;
		*madr = ch.madr;
		*qwc = ch.qwc;
		*tadr = ch.tadr;
		*finished = s_dmaIrq[channel];
		s_dmaIrq[channel] = false;
		if (channel == 3 && *finished)
			dumpFromIpu(reinterpret_cast<const uint8_t*>(dmaGetAddr(s_fromBase, false)), s_fromQwc);
	}

	bool takeIntc()
	{
		const bool v = s_intcIpu;
		s_intcIpu = false;
		return v;
	}
}
