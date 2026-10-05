# VU1 program compiler — design note (cont.230, 2026-09-02)

**Goal.** The level of the example game does 5.17 M VU1 pairs per guest frame; a 30 fps frame budget
is 33 ms, so VU1 must run at **≤ ~6 ns per pair all-in** (≈ 20 host cycles) including XGKICK and the
GIF hand-off. Today (build 378, EE thread pinned): 41–55 ns/pair. The straight-line block JIT
(`ps2_vu1_jit.h`, cont.195–222) has reached its design floor: 244–375 host cycles per 4.4-pair block
entry (dispatch, guard, ready-table replay) plus as much again in the interpreter loop around it, and
~700 cycles per interpreted pair (7% of pairs: DIV 83%, ERLENG 6%, SQRT, FMxxx, WAITQ, MFP). No
micro-lever closes a 7–10× gap; cont.219/220/230 each measured ≤ 7%.

**Reference.** PCSX2's microVU, surveyed in [`microvu-survey.md`](microvu-survey.md). License:
PCSX2 is `GPL-3.0+` and PS2Recomp (this repository) is GPLv3 as well, so **porting microVU's code is
license-compatible** (keep the SPDX headers and attribution on any copied file); the choice between
porting and reimplementing the design is an engineering one, see §6. The two mechanisms that matter most were
verified in the source directly: the block cache keyed by a compile-time pipeline state
(`microRegInfo`, 96 bytes, 8-byte quick compare) and the purely static stall model (`mVUincCycles` /
`mVUsetCycles`) with 1→0 normalization at exits so blocks link with a bare `jmp`.

## 1. The shape

A **program compiler** that owns a VU1 run from MSCAL to the E-bit. `VU1Interpreter::execute()`
(called synchronously from the MSCAL callback with a 65536-cycle budget, `ps2_runtime.cpp`) enters
the dispatcher at `(startPC, lpState)`; compiled blocks run and link among themselves; control returns
only at the E-bit, a cycle-budget check at a block top, a T/D-bit stop, or an **uncompilable pair**
(bail: the block writes its exit state to `lpState`, the interpreter executes that pair from the same
state, and the dispatcher re-enters at the next PC — the microVU `blockType`/`lpState` mechanism).
The existing interpreter stays as the fallback and the oracle.

### 1.1 Block key = (pc, PipeState)

`PipeState` mirrors `microRegInfo`:

| field | content |
|---|---|
| `needExact` (3 bits) | successor reads Status / Mac / Clip within its first 4 flag producers |
| `flagPos` | next write instance for Status, Mac, Clip (2 bits each) |
| `q`, `p` | cycles until the pending Q / P lands (0 = none) |
| `xgkick` | cycles until the deferred kick fires |
| `viBackUp` | VI written in the predecessor's delay slot (old value saved for a branch read) |
| `blockType` | 0 normal, 1 E-bit ending, 2 branch-in-delay-slot |
| `vi[16]` | cycles until each pending VI write lands |
| `vf[32]` × 4 lanes | cycles until each pending lane lands (4-bit each) |

Search: quick compare of the header word when `needExact == 0`, full compare otherwise. At every
exit the state is normalized (`1 → 0` for every VF/VI lane, Q/P likewise with an instance flip) so
the common exit state is all-zero and blocks reuse. The state is **baked into the emitted code**;
there is no runtime guard, no ready-table replay, no per-entry `applyGuardAdmitted`.

### 1.2 Static cycle model

Per pair, at compile time: `cycles += 1`; stall = max over the read operands' remaining cycles;
`cycles += stall`; the written registers get their latency (FMAC 4, VI ALU 1, ILW 4, DIV/SQRT 7,
RSQRT 13, EFU per op, XGKICK 1). No NOPs are emitted; the stall is bookkeeping that keeps the
pending-write model consistent. The block's total is charged once (`m_cycle += blockCycles` after the
budget test at the block top). Upper/lower same-cycle hazards (`swapOps`, `noWriteVF`, `backupVF`)
are resolved at compile time.

### 1.3 Linking

- B/BAL/IBxx: `mVUsetupBranch`-style exit (flush allocated registers, permute flag instances, shuffle
  Q/P lanes to instance 0), then a direct `jmp` to the successor block compiled for the exit state, or
  fall through into it when it does not exist yet. Conditional branches compute the condition at the
  branch pair into a memory slot and test it after the delay slot; the fall-through side links with
  the inverted condition.
- The branch VI-delay contract (the VU reads a VI as it was up to 4 instructions earlier) is kept as
  in the block JIT's cont.194 rule, generalized: the pre-write value is saved to a backup slot by the
  writing pair, and `viBackUp` carries the case across a block boundary.
- JR/JALR: exit state recorded; a per-site jump cache (target PC → code pointer) validated against
  the program generation; miss → search/compile at `(target, exitState)`.
- Branch in a delay slot, E-bit in a delay slot: the microVU `blockType` rules (§6.4 of the survey).

### 1.4 Flags

Four instances per flag file. Status in four callee-saved GPRs, Mac and Clip in 4-entry arrays
shuffled with one `PSHUFD` at exit. Instance selection is compile-time: a write lands `+4` cycles
later in the ring; a reader takes the newest landed instance. **Lazy:** an FMAC emits flag code only
when a reader exists within the next 4 flag producers, found by the backward walk at every
FSxxx/FMxxx/FCxxx and by the successor flag pass at every branch (both sides, recursive, stops at
E-bit or 4 producers); JR/JALR force exact. FSSET cancels the preceding Status updates. DIV's I/D
bits land in the Status instance 4 cycles before Q. This is the highest-risk unit (survey, last
section) and is built as one piece with its own oracle.

### 1.5 Q and P

One reserved xmm: lanes `[Q0, Q1, P0, P1]`. Compile-time instance flip: DIV/SQRT/RSQRT write the
inactive Q lane; readers use the active one; the instance flips when the countdown reaches 0. WAITQ
is stall bookkeeping only. P is not interlocked (MFP never stalls, WAITP stalls by `p−1`, the instance
flips as soon as a new EFU issues). At exit a `PSHUFD` normalizes both instances to 0. This removes
the entire interpreted population of the current design (DIV/SQRT/RSQRT/WAITQ/EFU/MFP).

### 1.6 XGKICK

The kick executes one cycle after the instruction as a call into the runtime (our `startXgkick` on
the **eager** path, `PS2X_VU1_KICKEAGER`, verified admissible over 1,027,251 kicks with 0 packet
differences): the whole packet is copied and tag-walked at issue and submitted through the arbiter.
The PATH1 timing our runtime models (16 bytes per 2 cycles; a second kick stalls until the previous
transfer's credit is exhausted) is kept by the same call: it receives the block-relative cycle of the
kick and returns the stall cycles to add, so `pipelinesPending()` and the submission cycle are
unchanged. The per-cycle `progressXgkick` streaming loop disappears from the compiled path.

### 1.7 Dispatcher and state

Entry: a stack frame saving callee-saved registers, a fixed GPR holding the VU state block (VF file,
VI file, ACC, I, Q/P spill, flag files, backup slots, cycle counter, budget), load the persistent host
state (Q/P xmm, Status GPRs, Mac/Clip arrays), `jmp` to the block. Exit: store TPC, spill Q/P, write
the final flag instances into the VI file, fold the cycle count, return. Program identity: our micro
memory is written by VIF MPG; a generation counter bumped on every MPG write invalidates the jump
caches and the block search (the block code stays until the program is re-compiled under a new
generation — the same "nothing is freed, the search re-validates" rule as microVU's `mVUclear`).

### 1.8 Register allocation (later milestone)

Within a block: VF in xmm with partial-lane merging, VI in GPRs, flushed only at exits; a preload at
the block top. Until then the existing per-pair load/store emitters are used (they are what the block
JIT emits today; the win of M1–M4 comes from removing the driver, not from allocation).

## 2. What is reused

- The x86 `Emitter` and the upper/lower op emitters with the VU clamp semantics (`FmacOp`,
  `LowerOp`, `OperandSrc`, the SSE quad clamps, `vuNormOperandBits`).
- `classifyBranch`, the branch-delay VI backup rule (cont.194), the I-bit handling (cont.215).
- The verify shapes: `BLOCKVERIFY` (run the compiled block, run the interpreter from the same snapshot,
  compare) becomes **`PS2X_VU1_PROGVERIFY`** at every block exit; `FLAGVERIFY`/`READYVERIFY` shapes for
  the flag ring; `JITCENSUS` opcode histograms at matched pairs; `KICKSCAN`/`flung.py`; the
  present-series visual montage.

## 3. What is dropped

The block driver (`blockByPair` lookup, the entry guard, `applyGuardAdmitted`, the ready-table replay,
the flag/clip replay lists, `kMaxBlockPairs = 8`) and the per-cycle interpreter loop around blocks.
The old path stays selectable (`PS2X_VU1_BLOCK`) until the new one covers everything, then retires.

## 4. Milestones (each measured with `framestats.py`: level fps, VU1 ns/pair, share)

| # | milestone | done when |
|---|---|---|
| M1 | PipeState-keyed blocks + direct linking for B/BAL/IBxx/fall-through; cycle charged per block; PROGVERIFY oracle; Q/P/EFU/flag-readers still bail to the interpreter | PROGVERIFY clean over a full run; entries per pair drop by ≥ 5×; a measured ns/pair |
| M2 | Q/P pipelines in the compiler (DIV/SQRT/RSQRT/WAITQ, EFU/MFP/WAITP) | the interpreted share < 1% of pairs |
| M3 | Flag ring, lazy flags, successor flag pass, Status in GPRs (the one-unit risk) | FLAGVERIFY-style oracle clean; FSxxx/FMxxx/FCxxx compiled |
| M4 | JR/JALR jump cache, deferred XGKICK call, E/T/D-bit exits, budget exits via `lpState`; the interpreter loop removed from the compiled path | a run with 0 interpreter pairs in the level |
| M5 | Block-level register allocation + preload | ns/pair ≤ ~8 |
| M6 | VU1 on its own thread (MTVU shape: ring of micro/data writes, unpacks, execute; XGKICK packets queued to the arbiter) | EE thread share of VU1 ≈ 0 |

Order of value: M1+M2 remove the driver and the interpreted population (the two largest costs
measured); M3 is the correctness cliff and is scheduled third so M1/M2 can be measured with the
existing (correct, slower) flag path; M5/M6 are the last 2×.

## 5. Risks and how they are covered

- **Flag semantics** (M3): one unit, own oracle, mutation-tested (a deliberately wrong instance must
  make the oracle fire — the cont.217 rule).
- **Silent state divergence** at links (the cont.230 dark-wedge class): PROGVERIFY compares the FULL
  state at every exit, including pending Q/P, the flag files and the cycle count, in production —
  never in a separate verify-only branch (the cont.230 verify-branch-divergence lesson).
- **XGKICK timing**: the eager path is already verified equivalent; the stall model is unchanged.
- **Measurement**: host clock drift (`therm_<run>.txt`); only alternating pairs, frame-matched.

## 6. Port microVU or reimplement its design?

Both are allowed (GPLv3 on both sides). The trade:

- **Port** (~10.6k lines of microVU + PCSX2's `x86emitter` it is written against + `microRegAlloc`,
  adapted at four seams: the VU register file layout (`VURegs`/`vuRegs[]` → our `VU1State`), the GIF
  hand-off (`Gif_Unit` → our arbiter's PATH1 submit + PATH1 timing), the configuration (`EmuConfig`
  gamefixes → env flags), and the caller (`recMicroVU1::Execute` → `VU1Interpreter::execute`).
  Gains 15+ years of game-fix history: every quirk in the survey's §10 already handled, and the flag
  model (our highest-risk unit) proven across the PS2 library. Cost: a large integration, PCSX2's
  emitter DSL and register allocator come along, and our verify harnesses must be re-aimed at it.
- **Reimplement** (~5–8k new lines on our own emitter, reusing the upper/lower emitters and the
  verify shapes): smaller and fully understood, but the block JIT's history (cont.194–230, some 35
  cycles to reach 93% coverage with the flag/clip/Q divergences found one at a time) is the honest
  estimate of how long the flag model takes to get right from scratch.

Recommendation: **port**, seam by seam, keeping our interpreter as the oracle (`PROGVERIFY` at every
block exit) exactly as planned for M1. The milestones stay the same; M3's risk moves from "write the
flag model" to "adapt it without breaking it".

