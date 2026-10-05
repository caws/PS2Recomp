// ps2xRuntime <-> PCSX2 microVU seam (cont.230 port). GPL-3.0+ (PCSX2 code under pcsx2/).
#pragma once
#include <cstdint>
#include <cstddef>

class PS2Memory;
struct VU1State;

namespace ps2x_microvu
{
	// One-time setup: point microVU's VU1 register block at our VU1 code/data memories and build
	// the dispatchers. Safe to call more than once (later calls are no-ops).
	void init(uint8_t *vu1Code, uint8_t *vu1Data);
	// The memory object whose GIF arbiter receives XGKICK packets (set per execute()).
	void setMemory(PS2Memory *memory);
	// Micro memory was written (VIF MPG): microVU re-validates its cached programs.
	void clearProgram(uint32_t addr, uint32_t size);

	// Run one VU1 program (MSCAL at startPC, or MSCNT when resume is set: continue at the state's
	// pc) from `st` until the E-bit, a T/D-bit stop, or `maxCycles`. Updates `st` (registers, flags,
	// pc, cycles, ebit, stoppedByD/T) and submits XGKICK packets through the memory set by
	// setMemory(). Returns the VU cycles consumed.
	uint64_t run(VU1State &st, uint32_t startPC, bool resume, uint32_t maxCycles);

	// ★ cont.250 VU0: the same seam for VU0. PCSX2's microVU is natively dual-unit (recMicroVU0 /
	// CpuMicroVU0 exist alongside the VU1 pair and vuRegs[] holds both register blocks) -- this port
	// only ever wired VU1, which is why VU0 was left running the block JIT at 85 ns/pair against
	// microVU's 4 ns/cycle on VU1. initVU0() points microVU's VU0 register block at our VU0 code/data
	// memories and reserves the VU0 recompiler; runUnit() is run() with the unit selected.
	// VU0's control bits are VU1's shifted down 8 throughout (microVU_Branch.inl / microVU_Compile.inl
	// select them as `isVU1 ? 0x100 : 0x001` busy, `0x200 : 0x2` D-stop, `0x400 : 0x4` T-stop,
	// `0x400 : 0x4` FBRST DE, `0x800 : 0x8` FBRST TE), and VPU_STAT/FBRST live in vuRegs[0] for BOTH
	// units -- only the bit differs. VU0's micro/data memories are 4 KB against VU1's 16 KB, so the
	// TPC and start_pc masks differ too (0x1FF/0x0FF8 vs 0x7FF/0x3FF8).
	void initVU0(uint8_t *vu0Code, uint8_t *vu0Data);
	uint64_t runUnit(int unit, VU1State &st, uint32_t startPC, bool resume, uint32_t maxCycles);

	// PROGVERIFY support: while recording, every submitted XGKICK packet's (size, FNV-1a hash) is
	// appended to the list so the interpreter's dry run can be compared against it.
	void setKickRecording(bool on);
	struct KickRecord { uint32_t size; uint64_t hash; };
	const KickRecord *kickRecords(size_t *count);
	uint64_t fnv1a(const uint8_t *p, size_t n);

	// Kick statistics (for the run log).
	struct Stats
	{
		unsigned long long runs = 0, kicks = 0, kickBytes = 0, cycles = 0, tbits = 0, dbits = 0, clears = 0;
	};
	const Stats &stats();
}
