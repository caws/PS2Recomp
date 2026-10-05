# microVU (PCSX2) — design survey for a VU1 whole-program compiler

Read-only survey of PCSX2's `pcsx2/x86/microVU*` (master, fetched 2026-09-02) made as the DESIGN
REFERENCE for the ps2xRuntime VU1 program compiler (see `vu1-program-compiler.md`). Line numbers refer
to the copies under `LLMPS2Recomp/tmp/pcsx2/microVU/`. PCSX2 is GPLv3: this document describes the
design; it does not copy code.

---

## 1. PROGRAM CACHE

### 1.1 What a "microProgram" is

`microVU.h:46-53`, `struct microProgram`:

| field | type | meaning |
|---|---|---|
| `data[mProgSize]` | `u32[0x1000]` | snapshot copy of VU micro memory (`mProgSize = 0x4000/4`) |
| `block[mProgSize/2]` | `microBlockManager*[0x800]` | one block-manager per 8-byte instruction slot (`PC/8`) |
| `ranges` | `std::deque<microRange>*` | `{s32 start; s32 end;}` byte ranges of micro-mem that have actually been recompiled |
| `startPC` | `u32` | program entry PC in units of 8 bytes |
| `idx` | `int` | serial number |

Owning container `microProgManager` (`microVU.h:63-77`): `prog[mProgSize/2]` — a deque of programs
per start-PC (MRU-ordered); `quick[mProgSize/2]` — `{microBlockManager* block; microProgram* prog;}`
one-entry cache per start-PC; `cur`, `total`, `isSame`, `cleared`, the code-cache cursors, and
`lpState` (a `microRegInfo` = "pipeline state where the program left off").

### 1.2 Identification — no hash in the lookup path

`mVUcmpProg` (`microVU.cpp:216-241`) compares, for each already-recompiled range, `prog.data` against
the live micro memory with `memcmp` (`doWholeProgCompare = false` by default). The production path is
a linear memcmp over a per-startPC list; `mVUrangesHash` exists only for debug output.

### 1.3 The search

`mVUsearchProg<vuIndex>` (`microVU.cpp:244-299`): `quick[start_pc/8]` hit → use it; else scan the
list with `mVUcmpProg`, move-to-front on a hit, else `mVUcreateProg` + compile via `mVUblockFetch`.
Programs are keyed on `regs().start_pc`, blocks on the block's own `startPC`.

### 1.4 Invalidation

`mVUclear(mV, addr, size)` (`microVU.cpp:102-114`) is called on every micro-memory write: it ignores
`addr`/`size`, sets `cleared = 1`, zeroes `lpState` and nulls every `quick[]` entry. Nothing is freed;
the next search re-runs the memcmp and either finds the same program or creates a new one.
`mVUsetupRange` (`microVU_Compile.inl:36-108`) maintains the recompiled ranges (merging, PC
wrap-around) and re-snapshots `prog.data`. Full teardown only on reset or code-cache overflow
(3 MB safe zone).

### 1.5 What a "block" is

`microBlock` (`microVU_IR.h:71-77`): `pState` (entry pipeline state), `pStateEnd` (exit state, used
by JR/JALR), `x86ptrStart`, `jumpCache` (lazily allocated `[mProgSize/2]` for JR/JALR blocks). A block
is a straight-line run ending at a branch's delay slot, an E-bit, an M-bit sync (VU0) or a bad opcode;
there is no instruction-count cap except `blockType != 0` ⇒ exactly one instruction.

`microBlockManager` (`microVU.h:140-273`) per PC slot: `qBlockList` (blocks with
`needExactMatch == 0`), `fBlockList` (exact-match blocks, MRU-reordered), and `quickLookup`
(`{microBlock*, u64 quick}` for ALL blocks). No cap on blocks per PC.

---

## 2. BLOCK ENTRY STATE — `microRegInfo`

`microVU_IR.h:22-61`. Exactly 96 bytes, 16-byte aligned:

| off | field | meaning |
|---|---|---|
| 0 | `u8 needExactMatch` | bit0 = successor reads Status, bit1 = Mac, bit2 = Clip (bit3 = scratch during the flag pass). Non-zero ⇒ full 96-byte compare |
| 1 | `u8 flagInfo` | bits2-3 = `xS` (next Status write instance), bits4-5 = `xM`, bits6-7 = `xC`; bit0 dead |
| 2 | `u8 q` | cycles until the pending Q lands (0 = none) |
| 3 | `u8 p` | cycles until the pending P lands |
| 4 | `u8 xgkick` | cycles until the deferred XGKICK fires |
| 5 | `u8 viBackUp` | VI written in a branch delay slot (old value lives in `mVU.VIbackup`) |
| 6 | `u8 blockType` | 0 normal; 1 E-bit ending (compile 1 instruction); 2 "evil" (branch in delay slot) |
| 7 | `u8 r` | R cycle info, always forced to 0 |
| 8-11 | `u32 xgkickcycles` | accumulated cycles for the XGKICK sync hack |
| 12-15 | pad / `vi15v` / `vi15` | const-prop (off by default) |
| 16-31 | `u8 VI[16]` | per-VI cycles until the pending write lands |
| 32-95 | `regCycleInfo VF[32]` | 2 bytes per VF: `x:4, y:4, z:4, w:4` cycles until each lane lands |

Bytes 0-7 alias `quick64[0]`.

### 2.1 The comparison rule

`microBlockManager::search` (`microVU.h:212-250`): if `needExactMatch`, walk `fBlockList` with the
JIT-generated `compareStateF` (`mVUGenerateCompareState`, `microVU_Execute.inl:253-310`: SSE/AVX2
compare of all 96 bytes); else walk `quickLookup` comparing only `quick64` (plus the const-prop
fields). So in the fast path only `{needExactMatch, flagInfo, q, p, xgkick, viBackUp, blockType, r}`
must match; per-VF/VI pending cycles are not compared. Tolerable because `mVUoptimizePipeState`
collapses 1→0 so most exits have all-zero VF/VI state. `add` searches first, then appends and pushes
into `quickLookup`.

### 2.2 Why there is no runtime entry guard

Everything that could differ at entry is baked into the state and hence the code: `mVUinitFirstPass`
(`microVU_Compile.inl:530-554`) copies the incoming state into the compiler's working state; every
block starts with `q = p = 0` instances and `mVUsetupBranch` renormalizes `xmmPQ` at exit; the flag
instances are permuted at every exit so instance k means "k cycles old"; direct branches emit a bare
`jmp` to the successor's entry (`microVU_Branch.inl:283`) or fall through into it. Only JR/JALR keep a
runtime lookup, memoized per site.

---

## 3. STATIC PIPELINE MODEL

### 3.1 Per-instruction cycle accounting

Pass 1 of `mVUcompile` (`microVU_Compile.inl:704-854`), per pair: `mVUincCycles(1)`; upper analyze
(sets `mVUstall`, temp write records); lower analyze; `mVUsetCycles`.

`mVUincCycles(mV, x)` (`:331-380`): `mVUcycles += x`; every `VF[1..31].{x,y,z,w}` and `VI[1..15]`
counter is saturating-decremented by x (`calcCycles(reg, x) = reg > x ? reg - x : 0`); `q` likewise
(with `doDivFlag` set when it crosses from >4 to ≤4, and the Q instance flipped when it reaches 0);
`p` (flipped when 0 or when a new EFU issues); `xgkick` (sets `doXGKICK` + `XGKICKPC` when it
reaches 0); `r`.

### 3.2 Where stalls come from (`microVU_Analyze.inl`)

Readers: `analyzeReg1` per component `mVUstall = max(mVUstall, VF[reg].lane)`; `analyzeReg3` (BC
ops), `analyzeReg4` (CLIP reads w), `analyzeReg5` (Fsf/Ftf), `analyzeReg6` (MR32 reads lane+1);
`analyzeVIreg1` for VI. Writers: `analyzeReg2` writes 4 into the temp record for the written lanes;
`analyzeVIreg2(reg, write, cycles)` with 1 for IALU/MOVE/MTIR/XTOP/XITOP/flag ops and 4 for
ILW/ILWR; `analyzeQreg(x)` stalls on an outstanding Q; `analyzePreg(x)` stalls by `p - 1` (EFU
result forwardable a cycle early); `analyzeXGkick1/2`.

Latencies: FMAC/VF 4; VI ALU 1; ILW 4; DIV/SQRT 7; RSQRT 13; EFU 11…54 per op (ESADD 11,
ELENG/ESQRT/ESUM/ERCPR 12, ERSADD/ERSQRT 18, ERLENG 24, ESIN 29, EEXP 44, EATAN* 54); XGKICK 1.

### 3.3 Committing — `mVUsetCycles` (`:395-434`)

`mVUincCycles(mVUstall)` (the stall is pure accounting: no NOPs are emitted); upper/lower writing the
same VF ⇒ lower becomes `noWriteVF` or a NOP; lower reads a VF the upper writes ⇒ `swapOps`; both
directions ⇒ `backupVF`; then every written counter is `max`ed with the new latency.

Trace of the 4-cycle VF latency: N writes VF5 ⇒ `VF[5].x = 4`; N+1: `incCycles(1)` ⇒ 3, a read of
VF5 ⇒ `stall = 3`, `setCycles` ⇒ `incCycles(3)` ⇒ 0. Four cycles write→use.

### 3.4 What crosses the block boundary

End of pass 1 (`:856-862`): const-prop fields, `mVUsetFlags` (computes `flagInfo`),
`mVUoptimizePipeState` (`:313-329`: every VF/VI counter `1 → 0`, q/p likewise with an instance flip,
`r = 0`; rationale: the successor's first `incCycles(1)` would do it anyway, and collapsing massively
increases reuse), `mVUtestCycles`. `mVUinitFirstPass` for the successor inherits `VF[]`, `VI[]`, `q`,
`p`, `xgkick`, `xgkickcycles`, `vi15*` and re-derives `needExactMatch` (7 if the predecessor's
`blockType` was set, else 0), `blockType = 0`, `viBackUp = 0`, `flagInfo = 0`.

---

## 4. FLAGS

Status lives in four host GPRs `gprF0..gprF3` = `r12d..r15d` (`microVU_Misc.h:144-147`); Mac and
Clip in `mVU.macFlag[4]` / `mVU.clipFlag[4]` (`microVU.h:84-86`); `statFlag[4]` is only an XGKICK
backup.

### 4.1 The 4-instance ring

Per instruction `microFlagInst` (`microVU_IR.h:145-152`) for s/m/c: `doFlag`, `doNonSticky`, `write`
(instance to write), `lastWrite`, `read` (instance a lower op should read).

`mVUsetFlags` (`microVU_Flags.inl:107-246`):
1. Walk backwards from the block end forcing the last ~4 flag-producing instructions exact (`mFLAG.doFlag`
   if the successor needs Mac, `sFLAG.doNonSticky` if it needs Status).
2. Seed `mFC.xStatus/xMac/xClip[i]`: default `= i`; if the block's own `needExactMatch` bit for that
   flag is clear, all −1 except `[(x-1)&3] = 0` with x from `flagInfo` ("only the last-written instance
   is known-valid at cycle 0").
3. Forward walk: `cycles += stall`; `read = findFlagInst(table, cycles)` (the instance with the
   largest write time ≤ cycles); `write = xS/xM/xC`; `lastWrite = (x-1)&3`; on a flag write
   `table[x] = cycles + 4; x = (x+1)&3`; `cycles++`. `sFlagCond = doFlag || isFSSET || doDivFlag`.
4. Publish `flagInfo |= (__Status ? 0 : xS<<2) | (xM<<4) | (__Clip ? 0 : xC<<6)`.

### 4.2 Where `doFlag` comes from

Arithmetic FMACs set `sFLAG.doFlag = 1` (`mVUanalyzeFMAC1/3`); CLIP sets `cFLAG.doFlag`;
ABS/FTOI/ITOF none; MIN/MAX cleared by `setupPass1`. `mFLAG.doFlag` is only set lazily and
retroactively by `flagSet` (`microVU_Analyze.inl:367-393`) when an FMxxx is seen: walk backwards
accumulating stalls, mark the last 4 calculations (`doNonSticky`, and `mFLAG.doFlag` for Mac readers).
`mVUanalyzeSflag/Mflag/Cflag` set `readFlags` and `swapOps` (read before the same pair's upper writes).
`mVUstatusFlagOp` turns off Status updates preceding an FSSET. The emitter `mVUupdateFlags`
(`microVU_Upper.inl:27-118`) early-returns when neither s nor m is needed; else MOVMSKPS for signs,
CMPEQPS+MOVMSKPS for zeros, masked by the dest lanes, ORed into the write instance.

### 4.3 Block-boundary normalization

`mVUsetupFlags` (`microVU_Flags.inl:255-337`), from `mVUsetupBranch`: `sortFlag` computes which
instance the successor reads at its cycle i, permutes `gprF0..F3` accordingly (1/2/3/4-distinct cases)
and shuffles `macFlag[4]` / `clipFlag[4]` with one `SHUFPS`, each guarded by whether the successor
reads that flag.

### 4.4 The flag pass over successors

`mVUsetFlagInfo` (`:433-475`) at every branch in pass 1: B/BAL → `_mVUflagPass(target)`; conditional →
both target and fall-through, ORed; JR/JALR → `needExactMatch |= 7`. `_mVUflagPass` (`:359-424`)
runs the analyzers' "pass 4" over up to 4 flag-producing instructions of the successor path (recursing
into further branches, stopping at E-bit), where upper FMACs mark bit3 (a flag WRITE) and FCxxx/FMxxx/
FSxxx mark bits 2/1/0 (a READ). Semantics: "within the first 4 flag-producing instructions of any
successor path, does anything read Status/Mac/Clip?" If yes, this block emits exact instances and the
successor full-compares.

---

## 5. Q AND P PIPELINES

`xmmPQ = xmm15`, reserved; lanes `[Q0, Q1, P0, P1]`. Loaded by the dispatcher from `VI[REG_Q]`,
`VI[REG_P]`, `pending_q`, `pending_p` (`microVU_Execute.inl:39-55`). `getQreg(reg, qInst)`,
`writeQreg(reg, qInst)`, `getPreg` (`microVU_Alloc.inl:117-133`).

Per instruction: `readQ = mVU.q; writeQ = !mVU.q; readP = mVU.p; writeP = !mVU.p`
(`microVU_Compile.inl:796-799`). A DIV writes the inactive lane; readers use the active one; when
`incCycles` drives `q` to 0 the instance flips (`mVU.q ^= 1`). P flips as soon as a new EFU issues or
when it lands (`:367-368`).

Stalls: `mVUanalyzeFDIV` → `analyzeQreg`; `WAITQ` = `stall = max(stall, q)` and emits nothing;
`WAITP` stalls by `p-1`; `MFP` never stalls (P is not interlocked). DIV status bits: `mVU.divFlag`
(invalid `0x1040000`, divide-by-zero `0x2080000`) is ORed into the Status write instance by `mVUdivSet`
exactly 4 cycles before Q lands (`doDivFlag`).

Block boundary: the values stay in `xmmPQ` across the `jmp`, renormalized by `PSHUFD` with
`shufflePQ = (p ? 0xb0 : 0xe0) | (q ? 0x01 : 0x04)` so both instances are 0 on entry; timing in
`pState.q/p`. On program exit both are spilled (`mVUendProgram`, `microVU_Branch.inl:186-202`); an
E-bit end first runs `mVUincCycles(100)` to drain them.

---

## 6. BRANCHES AND CONTROL FLOW

`branchAddr = (((iPC + 2) + (imm11 * 2)) & progMemMask) * 4`. Branch kinds 1=B, 2=BAL, 3..8=IBxx,
9=JR, 10=JALR. Pass 2 dispatches at the end of the delay slot (`microVU_Compile.inl:956-992`).

### 6.1 Direct branches

`normBranch` (`microVU_Branch.inl:332-408`) handles the D/T/M/E-bit variants, then
`mVUsetupBranch` (flush all, `mVUsetupFlags`, PQ renormalize) + `normBranchCompile`
(`:277-286`): `search(target, state)` → `jmp` to it, else compile it inline (fall through).

`condBranch` (`:410-556`): the condition was computed AT the branch instruction into `mVU.branch`;
after the delay slot `cmp [branch], 0`; if the fall-through block exists, `jcc(!cc)` to it and compile
the taken side inline; else a rel32 placeholder, compile the fall-through inline (saving `regBackup`
first because compilation clobbers the working state), then `mVUblockFetch(target, regBackup)` and
patch the placeholder.

### 6.2 VI hazards around branches

`analyzeBranchVI` (`microVU_Analyze.inl:494-570`): a branch reads its VI operand as it was up to 4
instructions earlier. Walk back ≤4 instructions/cycles; if the register was written in that window,
mark the instruction two slots before that write to store the pre-write value into `mVU.VIbackup`
and make the branch read the backup; if the window runs off the block start and `pState.viBackUp`
names the register, read the backup written by the previous block. `branchWarning` sets
`viBackUp` in the successor state when the delay slot writes a VI.

### 6.3 JR/JALR

`normJumpCompile` (`:288-330`): record `pStateEnd`, setup branch, backup regs, lazily allocate a
2048-entry `jumpCache`, call `mVUcompileJIT(target, block)` → `jmp rax`. `mVUcompileJIT`
(`microVU_Compile.inl:1039-1074`): the per-site cache entry is valid while `jc.prog ==
prog.quick[target/8].prog` (nulled by any micro-memory write); else `mVUsearchProg(target, pStateEnd)`.

### 6.4 Branch in a branch delay slot ("evil branch")

`mVUbranchCheck` (`microVU_Analyze.inl:573-629`): the second branch gets `evilBranch`, `blockType =
2`, `needExactMatch |= 7`; runtime slots `branch`/`badBranch`/`evilBranch`/`evilevilBranch` and
`condEvilBranch` select the target; the evil block compiles as one instruction ending in an indirect
jump. BAL/JALR in that position cannot produce a link register (error). E-bit in a delay slot ⇒
`blockType = 1`; a branch in an E-bit delay slot is NOPed.

### 6.5 Program end — `mVUendProgram` (`microVU_Branch.inl:140-263`)

Modes 0 (budget exit), 1 (E-bit), 2 (prepare only), 3 (M-bit). Picks the final flag instances
(`getLastFlagInst`), writes back registers, drains P/Q on E-bit (`incCycles(100)`), fires a pending
XGKICK, spills Q/P, normalizes Status/Mac/Clip into the VI file (all four instances backed up on a
budget exit; the single final value broadcast on E-bit), stores TPC, clears VBS1 on E-bit, jumps to
`exitFunct`. `mVUDTendProgram` is the D/T-bit twin (sets VPU_STAT bits, requests the interrupt).

---

## 7. XGKICK

Default model: `mVUanalyzeXGkick` stalls on a pending kick and schedules this one 1 cycle out
(`analyzeXGkick2(1)`); `incCycles` sets `doXGKICK` on the instruction where the countdown reaches 0;
pass 2 emits `mVU_XGKICK_DELAY` there (`microVU_Lower.inl:1808-1820`: flush caller-saved, backup,
`call mVU_XGKICK_(addr)`, restore) with the address stashed by the XGKICK op. A second XGKICK while
one is pending runs the pending one first. `mVU_XGKICK_` (`:1698-1714`) sizes the packet with
`GetGSPacketSize(PATH1)` and transfers it whole, splitting at the 0x4000 wrap.

Hack model (`CHECK_XGKICKHACK`): `xgkickcycles` accumulates `1 + stall` per non-kick instruction and
is flushed into `kickcycles` at memory writes and block ends; pass 2 emits `mVU_XGKICK_SYNC` before
each such instruction, transferring `min(remaining, cycles*8, wrapDiff)` bytes incrementally.

---

## 8. DISPATCH AND CYCLES

`mVUdispatcherAB` (`microVU_Execute.inl:23-88`): a stack frame saving callee-saved regs; call
`mVUexecute(startPC, cycles)` → code pointer; load MXCSR; load `xmmPQ`; load `macFlag`/`clipFlag`
from the VU register file; load `gprF0..F3`; `jmp rax`. Exit: restore MXCSR, `mVUcleanUp`, return.
Persistent host state: `xmm15` = PQ, `r12d..r15d` = Status instances, `eax/ecx` scratch; everything
else under `microRegAlloc` (15 xmm slots for VF/ACC/I, GPRs for VI).

`mVUexecute` sets `cycles`/`totalCycles`, resumes the emit cursor and calls `mVUsearchProg(startPC,
lpState)` — compilation happens inside the run call. `mVUtestCycles` at the top of every block:
`mov eax,[cycles]; sub eax, 1; jns skip; <save entry state to lpState>; endProgram(0)`;
`skip: sub [cycles], blockCycles`. `mVUcleanUp` folds the consumed cycles into `regs().cycle`.

Callers: `vu1RunCycles = 3000000`; `recMicroVU1::Execute(cycles)` runs when VBS1 is set;
`BaseVUmicroCPU::ExecuteBlock` slices by EE delta (min 16 cycles); `vif1VUFinish` spins until the VU
stops. Stops: E-bit, budget at a block top, T/D-bit, M-bit (VU0), code-cache overflow.

---

## 9. MTVU (`MTVU.cpp`, `MTVU.h`)

A 16 MB ring of `u32` with atomic read/write positions. Commands: `VU_EXECUTE` (addr, top, itop,
fbrst → run to completion with `vu1RunCycles`, then `FinishGSPacketMTVU` + `semaXGkick.Post`, and
record `VU1.cycle` into a 4-entry average), `VU_WRITE_MICRO` (clear + copy), `VU_WRITE_DATA`,
`VU_WRITE_VIREGS/VFREGS`, `VIF_WRITE_COL/ROW`, `VIF_UNPACK` (the unpack runs ON the VU thread),
`NULL_PACKET` (wrap). XGKICK on the VU thread copies into path 1's MTVU packet queue
(`Gif_Unit.h:620-631`); the EE side injects a fake path-1 placeholder per program start.

Sync: `WaitVU()` (shutdown/reset/savestate, and when VU0 touches VU1 registers via `mVUaddrFix`);
`Get_vuCycles()` (average of the last 4 runs) drives the EE-side `VPU_STAT |= 0x100` + a timed
`VU_MTVU_BUSY` interrupt; `mtvuInterrupts` (Finish/Signal/Label/VUEBit/VUTBit) are drained on the EE by
`Get_MTVUChanges()`.

---

## 10. Quirks microVU deliberately emulates

1. Branch VI-delay (reads a VI value from up to 4 instructions earlier; `VIbackup`, `viBackUp`).
2. Branch in a branch delay slot, including chains (`badBranch`/`evilBranch`/`evilevilBranch`,
   `blockType == 2`); BAL/JALR there cannot link.
3. E-bit in a delay slot (`blockType = 1`); a branch in an E-bit delay slot is NOPed.
4. Upper/lower same-cycle hazards: `swapOps`, `noWriteVF`, `backupVF` (XOR-swap dance).
5. Four flag instances with `+4` write latency, renormalized at every exit.
6. Lazy flags: Mac only when read within 4 flag producers; Status suppressed under the flag hack
   unless `doNonSticky`; FSSET kills preceding Status updates.
7. Q pipeline: 7 (DIV/SQRT) / 13 (RSQRT) cycles; a second FDIV stalls; WAITQ stalls; I/D bits land
   4 cycles before Q; two instances live in `xmmPQ`.
8. P is not interlocked: MFP never stalls, WAITP stalls `p-1`, the instance flips on a new EFU.
9. XGKICK fires one cycle after the instruction; a second kick forces the first; wrap at 0x4000.
10. Clamp modes (`microVU_Clamp.inl`): result clamp to ±0x7f7fffff, sign-preserving operand clamp,
    extra-overflow variants; skipped for the I register and VF0-sourced values.
11. VF0 = (0,0,0,1) and VI0 = 0 are constants excluded from the cycle loops.
12. The I-bit immediate replaces the lower instruction and is clamped if ≥ 0x7f800000.
13. The BIOS's reversed NOP pair `0x8000033c` is whitelisted.
14. `B` to self+8 (imm11 == 1) is a NOP unless in a delay slot.
15. M-bit on VU0 forces an exact-flag mid-block program end and re-entry.
16. VU0 reading VU1 registers syncs the VU1 thread first.

---

## Minimal design to reach ~10 host cycles per pair

**Essential**
1. Block linking by entry state, with the state baked into the code: no per-block entry guard.
2. Lazy flag computation (the backward `flagSet` walk + the successor `_mVUflagPass`).
3. Status flags in four callee-saved GPRs; Mac/Clip in 4-element arrays shuffled once at exit.
4. The 4-instance flag ring with compile-time instance selection (`findFlagInst` / `sortFlag`).
5. Q/P in one reserved xmm with compile-time instance selection.
6. Cross-instruction register allocation for VF (xmm) and VI (GPR) within a block, flushed only at
   block boundaries; preload at the block top.
7. Upper/lower hazard resolution at compile time.
8. Branch VI-delay handling and a memory slot for the branch condition.
9. A cycle counter charged once per block.
10. A PC→code table for indirect jumps with per-site memoization.

**Optional / droppable** for a runtime that knows its programs: the program cache, memcmp
identification and range bookkeeping (keep a "micro memory was overwritten" bail-out); the quick-vs-full
two-tier search if every PC gets one canonical entry state; the full per-VF/VI cycle model when
cycle-exact `VU1.cycle` is not needed (microVU's own fast path ignores it) — but keep the Q/P/XGKICK
counters and the flag `+4` distance; const propagation; D-bit handling; MTVU until a second thread is
wanted; the XGKICK sync hack; the XGKICK resume dispatcher; VU0/COP2 integration.

**Biggest single risk:** the flag model. Lazy Mac flags need the successor flag pass, which needs the
4-instance ring and the block-entry `needExactMatch`/`flagInfo` key. Budget the three as one unit.
