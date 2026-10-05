// SPDX-License-Identifier: GPL-3.0+
// ps2xRuntime <-> PCSX2 microVU seam (cont.230 port): the globals PCSX2 code expects, the code
// cache, the GIF unit shim (XGKICK -> our GIF arbiter) and the EE-side stubs.
#include "ps2x_microvu.h"
#include "Common.h"
#include "iR5900.h"
#include "VU.h"
#include "VUmicro.h"
#include "Vif.h"
#include "MTVU.h"
#include "Gif_Unit.h"
#include "GS/MultiISA.h"
#include "common/Perf.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"
#include <sys/mman.h>
#include <cstring>
#include <cstdlib>
#include <vector>

// ---- PCSX2 globals ------------------------------------------------------------------------------
Pcsx2Config EmuConfig;
cpuRegisters cpuRegs;
alignas(16) VURegs vuRegs[2];
VIFregisters vif0Regs, vif1Regs;
VU_Thread vu1Thread;
Gif_UnitShim gifUnit;
ConsoleShim Console{"microvu", true};
ConsoleShim DevCon{"microvu:dev", std::getenv("PS2X_MICROVU_DEVCON") != nullptr};
ProcessorFeatures g_cpu;
namespace Perf { Group any, vu0, vu1; }
_x86regs x86regs[iREGCNT_GPR] = {};
_xmmregs xmmregs[iREGCNT_XMM] = {};
u16 g_x86AllocCounter = 0;
// recMicroVU0/1 CpuMicroVU0/1 and SaveStateBase::vuJITFreeze are defined in microVU.cpp.
BaseVUmicroCPU *CpuVU0 = &CpuMicroVU0;
BaseVUmicroCPU *CpuVU1 = &CpuMicroVU1;

void hwIntcIrq(int) {} // T-bit/D-bit stops are read back from VPU_STAT by the caller
void BaseVUmicroCPU::ExecuteBlock(bool) {}
void BaseVUmicroCPU::ExecuteBlockJIT(BaseVUmicroCPU *, bool) {}

// ---- code cache ---------------------------------------------------------------------------------
namespace
{
	constexpr size_t kVu0RecSize = 8u << 20;
	constexpr size_t kVu1RecSize = 64u << 20;
	u8 *g_rec = nullptr;
	u8 *recBase()
	{
		if (!g_rec)
		{
			void *p = mmap(nullptr, kVu0RecSize + kVu1RecSize, PROT_READ | PROT_WRITE | PROT_EXEC,
			               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
			if (p == MAP_FAILED)
			{
				std::fprintf(stderr, "[microvu] mmap of the code cache failed\n");
				std::abort();
			}
			g_rec = static_cast<u8 *>(p);
		}
		return g_rec;
	}
}
namespace SysMemory
{
	u8 *GetVU0Rec() { return recBase(); }
	u8 *GetVU0RecEnd() { return recBase() + kVu0RecSize; }
	u8 *GetVU1Rec() { return recBase() + kVu0RecSize; }
	u8 *GetVU1RecEnd() { return recBase() + kVu0RecSize + kVu1RecSize; }
}

// ---- GIF unit shim: XGKICK ----------------------------------------------------------------------
namespace
{
	PS2Memory *g_memory = nullptr;
	ps2x_microvu::Stats g_stats;
	// PS2X_MICROVU_KICKLOG=<n> (default 0): log run starts and every XGKICK event of the first n
	// runs, and dump micro memory at the first logged run (tmp/mvu_code_run<N>.bin).
	const unsigned long long g_kickLogRuns = []{ const char *e = std::getenv("PS2X_MICROVU_KICKLOG"); return e ? std::strtoull(e, nullptr, 10) : 0ull; }();
	bool kickLogOn() { return g_stats.runs <= g_kickLogRuns; }
	std::vector<u8> g_kickBuf; // a packet split at the 0x4000 wrap is reassembled here
	bool g_recordKicks = false;
	std::vector<ps2x_microvu::KickRecord> g_kickRecords;

	// Bytes of the GIF packet at pMem[offset..] (tags + payload), stopping at EOP or `size`.
	// Mirrors PCSX2 Gif_Unit::GetGSPacketSize for PATH1 without the wrap (the caller splits).
	u32 packetSize(const u8 *pMem, u32 offset, u32 limit)
	{
		u32 pos = offset;
		for (;;)
		{
			if (pos + 16u > limit)
				return limit - offset;
			const u64 lo = *reinterpret_cast<const u64 *>(pMem + pos);
			const u32 nloop = static_cast<u32>(lo & 0x7FFFu);
			const bool eop = (lo & 0x8000u) != 0u;
			const u32 flg = static_cast<u32>((lo >> 58) & 3u);
			u32 nreg = static_cast<u32>((lo >> 60) & 0xFu);
			if (nreg == 0u) nreg = 16u;
			u32 qw;
			if (flg == 0u) qw = nloop * nreg;                 // PACKED
			else if (flg == 1u) qw = (nloop * nreg + 1u) / 2u; // REGLIST
			else qw = nloop;                                   // IMAGE / disabled
			pos += 16u + qw * 16u;
			if (eop || nloop == 0u && flg != 0u && false)
				return pos - offset;
			if (eop)
				return pos - offset;
			if (pos - offset >= limit - offset)
				return limit - offset;
		}
	}
}

void Gif_PathShim::CopyGSPacketData(u8 *pMem, u32 size, bool)
{
	g_kickBuf.insert(g_kickBuf.end(), pMem, pMem + size);
}

u32 Gif_UnitShim::GetGSPacketSize(GIF_PATH, u8 *pMem, u32 offset, u32 size, bool)
{
	const u32 limit = (size == ~0u) ? 0x4000u : std::min<u32>(offset + size, 0x4000u);
	const u32 r = packetSize(pMem, offset, limit);
	if (kickLogOn())
		std::fprintf(stderr, "[microvu:kick] run=%llu GetGSPacketSize off=0x%x -> %u\n", g_stats.runs, offset, r);
	return r;
}

u32 Gif_UnitShim::TransferGSPacketData(GIF_TRANSFER_TYPE, u8 *pMem, u32 size, bool)
{
	const u8 *data = pMem;
	u32 total = size;
	if (!g_kickBuf.empty())
	{
		g_kickBuf.insert(g_kickBuf.end(), pMem, pMem + size);
		data = g_kickBuf.data();
		total = static_cast<u32>(g_kickBuf.size());
	}
	++g_stats.kicks;
	g_stats.kickBytes += total;
	if (kickLogOn())
	{
		std::fprintf(stderr, "[microvu:kick] run=%llu Transfer size=%u total=%u hash=%016llx\n", g_stats.runs, size, total,
		             (unsigned long long)ps2x_microvu::fnv1a(data, total));
		static bool dumped = false;
		if (!dumped && vuRegs[1].Mem)
		{
			dumped = true;
			if (FILE *f = std::fopen("tmp/mvu_data_firstkick.bin", "wb")) { std::fwrite(vuRegs[1].Mem, 1, 0x4000, f); std::fclose(f); }
			std::fprintf(stderr, "[microvu:kick] dumped VU data memory (kick data offset %ld)\n", (long)(pMem - vuRegs[1].Mem));
		}
	}
	if (g_recordKicks)
		g_kickRecords.push_back({total, ps2x_microvu::fnv1a(data, total)});
	if (g_memory && total != 0u)
		g_memory->submitGifPacket(GifPathId::Path1, data, total);
	g_kickBuf.clear();
	return size;
}

// ---- public seam --------------------------------------------------------------------------------
namespace ps2x_microvu
{
	static bool s_inited = false;

	// Shared one-time setup (FP control words, clamp mode, kick buffer). Both units need it, and
	// whichever runs first performs it.
	static void ensureShared()
	{
		if (s_inited)
			return;
		s_inited = true;
		vuRegs[0].idx = 0;
		vuRegs[1].idx = 1;
		g_kickBuf.reserve(0x8000);
		// MXCSR for the compiled VU code: round toward zero, flush-to-zero, denormals-are-zero,
		// exceptions masked (PCSX2 default VU FPCR).
		EmuConfig.Cpu.VU1FPCR = FPControlRegister::GetDefault().DisableExceptions().SetDenormalsAreZero(true).SetFlushToZero(true).SetRoundMode(FPRoundMode::ChopZero);
		EmuConfig.Cpu.VU0FPCR = EmuConfig.Cpu.VU1FPCR;
		// ★ The dispatcher's EXIT reloads MXCSR from FPUFPCR: that must be the HOST state the rest of
		// this runtime runs under (round-to-nearest, no DAZ/FTZ, exceptions masked), not PCSX2's EE
		// setting -- otherwise every VU run would silently switch the recompiled EE code's float
		// rounding for the rest of the frame.
		EmuConfig.Cpu.FPUFPCR = FPControlRegister::GetDefault().DisableExceptions();
		// ★ VU clamp mode (PS2X_MICROVU_CLAMP: normal | extra | extrasign, default extrasign).
		// PCSX2's default "normal" clamps results only, so 0 x inf reaches a register as NaN
		// (PROGVERIFY: vf4 = -nan where the interpreter has 0); the hardware has no NaN/Inf and
		// this runtime's interpreter normalises every operand and clamps every result, and its GS
		// drops a NaN vertex. "extrasign" = PCSX2's Extra + Preserve Sign: operands clamped
		// sign-preservingly (mVUclamp2) and results clamped (mVUclamp1) -- the closest to the
		// interpreter's semantics.
		{
			const char *cm = std::getenv("PS2X_MICROVU_CLAMP");
			const bool extra = !(cm && std::strcmp(cm, "normal") == 0);
			const bool sign = !(cm && (std::strcmp(cm, "normal") == 0 || std::strcmp(cm, "extra") == 0));
			EmuConfig.Cpu.Recompiler.vu1Overflow = true;
			EmuConfig.Cpu.Recompiler.vu1ExtraOverflow = extra;
			EmuConfig.Cpu.Recompiler.vu1SignOverflow = sign;
			EmuConfig.Cpu.Recompiler.vu0Overflow = true;
			EmuConfig.Cpu.Recompiler.vu0ExtraOverflow = extra;
			EmuConfig.Cpu.Recompiler.vu0SignOverflow = sign;
		}
		// ★★ rotk row 260 PS2X_MICROVU_FLAGHACK (default ON = PCSX2's own default "mVU Flag Hack",
		// EmuConfig.Speedhacks.vuFlagHack; `=0` = exact status flags): microVU computes the status flag only in
		// blocks that read it (microVU_Compile.inl: mVUsFlagHack = CHECK_VU_FLAGHACK, read per block at compile
		// time, so it is set before the first compile). Measured on the 60-fps fight (tools/harness/vucyc.sh,
		// GsPipeline core cycles per VU1 cycle, mirrored order): 15.28/15.21 -> 14.23/14.03 (-7.5%), instructions
		// per VU cycle 34.9 -> 29.8. Verified: PS2X_VU1_PROGVERIFY over 3.2 M programs vf=vi=qp=pc=kick=0 (mem: the
		// same pre-existing startPC=0x140 class as without it, 22 vs 56); only the STICKY status bits diverge more
		// (126k -> 480k); no guest code can read them (the ELF's 27 CFC2 read VU0 VI 6/8/11, FBRST, VPU-STAT, never 16-18). Flip hash 0 diffs / 6213 flips.
		{
			const char *fh = std::getenv("PS2X_MICROVU_FLAGHACK");
			EmuConfig.Speedhacks.vuFlagHack = !(fh && fh[0] == '0');
		}
	}

	void init(uint8_t *vu1Code, uint8_t *vu1Data)
	{
		vuRegs[1].Micro = vu1Code;
		vuRegs[1].Mem = vu1Data;
		vuRegs[1].idx = 1;
		vuRegs[0].idx = 0;
		ensureShared();
		static bool s_vu1Reserved = false;
		if (s_vu1Reserved)
			return;
		s_vu1Reserved = true;
		CpuMicroVU1.Reserve();
		CpuMicroVU1.Reset();
	}

	// ★ cont.250: the VU0 half of the seam. vuRegs[0] already existed and CpuVU0/CpuMicroVU0 were
	// already defined -- only the register-block pointers and the reserve were missing.
	void initVU0(uint8_t *vu0Code, uint8_t *vu0Data)
	{
		vuRegs[0].Micro = vu0Code;
		vuRegs[0].Mem = vu0Data;
		vuRegs[0].idx = 0;
		vuRegs[1].idx = 1;
		ensureShared();
		static bool s_vu0Reserved = false;
		if (s_vu0Reserved)
			return;
		s_vu0Reserved = true;
		CpuMicroVU0.Reserve();
		CpuMicroVU0.Reset();
	}

	void setMemory(PS2Memory *memory) { g_memory = memory; }

	void clearProgram(uint32_t addr, uint32_t size)
	{
		if (s_inited)
			CpuMicroVU1.Clear(addr, size);
	}

	const Stats &stats() { return g_stats; }
	uint64_t fnv1a(const uint8_t *p, size_t n)
	{
		uint64_t h = 1469598103934665603ull;
		for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
		return h;
	}
	void setKickRecording(bool on) { g_recordKicks = on; if (on) g_kickRecords.clear(); }
	const KickRecord *kickRecords(size_t *count) { *count = g_kickRecords.size(); return g_kickRecords.data(); }

	// ---- state conversion -----------------------------------------------------------------------
	// Our VU1State keeps VI as sign-extended 16-bit values in int32; PCSX2 keeps the raw 16 bits in
	// a u32. Flags: our single status/mac/clip words become all four microVU instances (a program
	// starts with a settled flag pipeline, same as after an E-bit). Q/P: the active value and the
	// pending value are both the current one (pipelines are empty at MSCAL; a budget-exit resume
	// relies on microVU's own lpState, which we never disturb between calls).
	static void toMicroVU(int unit, const VU1State &st)
	{
		VURegs &vu = vuRegs[unit];
		for (int i = 0; i < 32; ++i)
			std::memcpy(vu.VF[i].F, st.vf[i], 16);
		vu.VF[0].f = {0.0f, 0.0f, 0.0f, 1.0f};
		for (int i = 0; i < 16; ++i)
			vu.VI[i].UL = static_cast<u32>(st.vi[i]) & 0xFFFFu;
		vu.VI[0].UL = 0;
		std::memcpy(vu.ACC.F, st.acc, 16);
		vu.VI[REG_STATUS_FLAG].UL = st.status;
		vu.VI[REG_MAC_FLAG].UL = st.mac;
		vu.VI[REG_CLIP_FLAG].UL = st.clip;
		vu.VI[REG_R].UL = st.r;
		vu.VI[REG_I].F = st.i;
		vu.VI[REG_Q].F = st.q;
		vu.VI[REG_P].F = st.p;
		std::memcpy(&vu.pending_q, &st.q, 4);
		std::memcpy(&vu.pending_p, &st.p, 4);
		for (int i = 0; i < 4; ++i)
		{
			vu.micro_statusflags[i] = st.status;
			vu.micro_macflags[i] = st.mac;
			vu.micro_clipflags[i] = st.clip;
		}
		if (unit == 1)
		{
			// VIF1's TOP/ITOP are VU1's; VU0 is fed by VIF0 and our VU1State carries no VU0 equivalent.
			vif1Regs.top = st.top;
			vif1Regs.itop = st.itop;
		}
		// FBRST holds BOTH units' D/T enables in one register (VU0 bits 0x4/0x8, VU1 0x400/0x800 --
		// microVU_Compile.inl `isVU1 ? 0x400 : 0x4` / `isVU1 ? 0x800 : 0x8`), so update only this
		// unit's bits instead of assigning the whole word and clobbering the other unit.
		{
			const u32 deBit = (unit == 1) ? 0x400u : 0x4u;
			const u32 teBit = (unit == 1) ? 0x800u : 0x8u;
			u32 fb = vuRegs[0].VI[REG_FBRST].UL & ~(deBit | teBit);
			if (st.dBitEnabled) fb |= deBit;
			if (st.tBitEnabled) fb |= teBit;
			vuRegs[0].VI[REG_FBRST].UL = fb;
		}
	}

	static void fromMicroVU(int unit, VU1State &st)
	{
		const VURegs &vu = vuRegs[unit];
		for (int i = 0; i < 32; ++i)
			std::memcpy(st.vf[i], vu.VF[i].F, 16);
		st.vf[0][0] = 0.0f; st.vf[0][1] = 0.0f; st.vf[0][2] = 0.0f; st.vf[0][3] = 1.0f;
		for (int i = 1; i < 16; ++i)
			st.vi[i] = static_cast<int32_t>(static_cast<int16_t>(vu.VI[i].US[0]));
		st.vi[0] = 0;
		std::memcpy(st.acc, vu.ACC.F, 16);
		st.status = vu.VI[REG_STATUS_FLAG].UL;
		st.mac = vu.VI[REG_MAC_FLAG].UL;
		st.clip = vu.VI[REG_CLIP_FLAG].UL;
		st.r = vu.VI[REG_R].UL;
		st.i = vu.VI[REG_I].F;
		st.q = vu.VI[REG_Q].F;
		st.p = vu.VI[REG_P].F;
	}

	uint64_t runUnit(int unit, VU1State &st, uint32_t startPC, bool resume, uint32_t maxCycles)
	{
		const bool isVU1 = (unit == 1);
		VURegs &vu = vuRegs[unit];
		VURegs &vu0 = vuRegs[0]; // VPU_STAT / FBRST live here for BOTH units -- only the bit differs
		// VU0's control bits are VU1's shifted down 8 (microVU_Branch.inl: `isVU1 ? 0x100 : 0x001` busy,
		// `0x200 : 0x2` D-stop, `0x400 : 0x4` T-stop) and its micro memory is 4 KB against VU1's 16 KB,
		// so the TPC / start_pc masks differ too (VU0_PROGSIZE 0x1000, VUmicro.h).
		const u32 busyBit  = isVU1 ? 0x100u : 0x001u;
		const u32 stopBits = isVU1 ? 0x600u : 0x006u;
		const u32 dStopBit = isVU1 ? 0x200u : 0x002u;
		const u32 tStopBit = isVU1 ? 0x400u : 0x004u;
		const u32 progSize = isVU1 ? 0x4000u : 0x1000u;
		const u32 tpcMask  = isVU1 ? 0x7FFu : 0x1FFu;
		const u32 pcMask   = progSize - 8u;
		// Micro memory rewritten by VIF MPG since the last run: microVU re-validates its programs
		// (mVUclear: nothing is freed, the next search memcmp's the recompiled ranges).
		static uint64_t s_seenGen[2] = {~0ull, ~0ull};
		if (g_memory)
		{
			const uint64_t gen = isVU1 ? g_memory->getVU1CodeGeneration() : g_memory->getVU0CodeGeneration();
			if (gen != s_seenGen[unit])
			{
				s_seenGen[unit] = gen;
				if (isVU1) CpuMicroVU1.Clear(0, progSize); else CpuMicroVU0.Clear(0, progSize);
				++g_stats.clears;
			}
		}
		toMicroVU(unit, st);
		if (!resume)
			vu.VI[REG_TPC].UL = (startPC >> 3) & tpcMask;
		// MSCNT continues at TPC; PCSX2's vu1ExecMicro sets start_pc = TPC << 3 in both cases.
		vu.start_pc = (vu.VI[REG_TPC].UL << 3) & pcMask;
		if (kickLogOn())
		{
			std::fprintf(stderr, "[microvu:run] run=%llu startPC=0x%x resume=%d TPC=0x%x start_pc=0x%x top=%u itop=%u\n",
			             g_stats.runs + 1, startPC, resume ? 1 : 0, vu.VI[REG_TPC].UL, vu.start_pc, vif1Regs.top, vif1Regs.itop);
			static bool dumped = false;
			if (!dumped && vu.Micro)
			{
				dumped = true;
				char path[256];
				std::snprintf(path, sizeof path, "tmp/mvu_code_run%llu.bin", g_stats.runs + 1);
				if (FILE *f = std::fopen(path, "wb")) { std::fwrite(vu.Micro, 1, progSize, f); std::fclose(f); }
			}
		}
		u32 &vpuStat = isVU1 ? g_vu1VpuStat : vu0.VI[REG_VPU_STAT].UL; // cont.317 stage 2: VU1's own word
		vpuStat |= busyBit; // VBS0 / VBS1: this unit running
		vpuStat &= ~stopBits;
		vu.cycle = 0;
		++g_stats.runs;
		// microVU's Execute runs until the E-bit or its cycle budget; a budget exit leaves VBS1 set
		// and the pipeline state in lpState, so looping re-enters exactly where it stopped.
		uint64_t total = 0;
		while (vpuStat & busyBit)
		{
			const u64 before = vu.cycle;
			(isVU1 ? CpuVU1 : CpuVU0)->Execute(maxCycles);
			const u64 used = vu.cycle - before;
			total += used;
			if (vpuStat & stopBits) // D-bit / T-bit stop
				break;
			if (total >= maxCycles)
				break;
			if (used == 0u) // no progress: never spin here
				break;
		}
		st.stoppedByD = (vpuStat & dStopBit) != 0u;
		st.stoppedByT = (vpuStat & tStopBit) != 0u;
		if (st.stoppedByT) ++g_stats.tbits;
		if (st.stoppedByD) ++g_stats.dbits;
		vpuStat &= ~stopBits;
		vu.flags &= ~VUFLAG_INTCINTERRUPT;
		// The interpreter leaves ebit/haltAfterDelaySlot/branchPending CLEAR after a program ends
		// (they are its in-flight control bits, not "the program has ended"); a resumed run that
		// starts with ebit set halts after one pair. Mirror that. Whether the program ended is
		// VBS1 (a budget exit leaves it set and microVU's lpState holds the pipeline state).
		st.ebit = false;
		st.haltAfterDelaySlot = false;
		st.branchPending = false;
		st.pc = (vu.VI[REG_TPC].UL << 3) & pcMask;
		fromMicroVU(unit, st);
		st.cycles += total;
		g_stats.cycles += total;
		if (kickLogOn())
			std::fprintf(stderr, "[microvu:run] run=%llu end TPC=0x%x cycles=%llu VPU_STAT=0x%x\n", g_stats.runs, vu.VI[REG_TPC].UL, (unsigned long long)total, vu0.VI[REG_VPU_STAT].UL);
		return total;
	}

	// VU1 wrapper: the hot path keeps its exact original signature and behaviour.
	uint64_t run(VU1State &st, uint32_t startPC, bool resume, uint32_t maxCycles)
	{
		return runUnit(1, st, startPC, resume, maxCycles);
	}
}
