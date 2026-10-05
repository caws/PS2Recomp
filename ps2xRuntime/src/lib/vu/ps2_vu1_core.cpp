#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_jit.h"
#include "microvu/ps2x_microvu.h"

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <map>
#include <cctype>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <limits>
#include <utility>
#include <x86intrin.h>
#include <ps2_log.h>

namespace
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    // cont.175 stage 1a: walk only the SET bits of a lane/register mask instead of scanning
    // every slot. laneForComponent(c) == 1<<(3-c), so bit index b is component 3-b.
    // The visit ORDER changes (high bit first -> lowest component index last), which is
    // immaterial: every consumer is either std::max() (commutative/associative) or a write to
    // an independent per-lane/per-register slot. Same slots visited, same values ==> bit-exact.
    inline uint32_t nextSetBit(uint32_t &mask)
    {
        const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(mask));
        mask &= mask - 1u;
        return bit;
    }
    constexpr uint32_t kViRegMask = 0xFFFEu; // VI regs 1..15 (reg 0 is hardwired zero)
}

// ---- Pipeline-commit tracking (PS2X_VU1_COMMITSKIP, default ON; =0 reverts to the
// unconditional scans). Profiling (game cont.158a, 2026-08-29) measured the per-cycle
// commitReadyPipelines() walk over all seven pipeline arrays (~50 slots) at 28% of total
// runtime — called once per pair-loop iteration AND once per advanceOneCycle(), where the
// top-of-loop call always re-scans at the same m_cycle the previous advance already
// scanned. We track (a) a lower bound on the earliest readyCycle among valid entries and
// (b) per-array valid-entry counts, so commitReadyPipelines() returns in O(1) when
// nothing can be due, skips empty arrays when it does scan, and pipelinesPending()
// answers in O(1). Purely a memoization: commit timing, order, and values are untouched
// (no modeled hardware behavior changes, so no PCSX2 semantics to mirror).
// File-static on purpose (a member would touch include/runtime/ps2_vu1.h, which the
// generated code includes: ~60 min rebuild). The owner pointer keeps a second
// interpreter instance (VU0) safe: the counters describe ONE instance, and any owner
// switch or reset falls back to a full (status-quo) scan that rebuilds exact state.
// Staleness is one-directional by construction — it can only ADD scans, never skip a due
// commit: entries become valid only in the queue* helpers (each lowers the watermark via
// notePipeInsert) and are cleared only inside the scan itself or resetScheduler() (which
// drops ownership).
static constexpr uint64_t kPipeIdle = ~0ull;
static const bool s_vu1CommitSkip = []
{ const char *e = std::getenv("PS2X_VU1_COMMITSKIP"); return !(e && e[0] == '0'); }();
// PS2X_VU1_NOSCHED (default OFF) -- ABLATION, NOT A FIX. Makes the cycle-accurate
// pipeline/hazard model free: no stall scan, no write-ready marking, and every queued
// write applies to architectural state immediately (so the pipeline arrays stay empty and
// commitReadyPipelines() answers O(1) via the cont.158b watermark). Output is deliberately
// unfaithful -- latency/hazard semantics are gone -- and it exists ONLY to measure the
// CEILING on any VU1 work that resolves hazards statically (a JIT), per the
// ablate-before-optimizing rule. Never enable it for a correctness run.
// PS2X_VU1_NOFLAGS (default OFF): ABLATION -- skip ALL FMAC flag work (the per-lane exact
// recompute, the clamp, the MAC/status derivation and the flag-pipeline entry). Deliberately
// unfaithful; it sizes "lazy flags" (compute flags only for blocks that actually read them),
// which is the main remaining interpreter-level lever before a JIT.
static const bool s_vu1NoFlags = []
{ const char *e = std::getenv("PS2X_VU1_NOFLAGS"); return e && e[0] && e[0] != '0'; }();
static const bool s_vu1NoSched = []
{ const char *e = std::getenv("PS2X_VU1_NOSCHED"); return e && e[0] == '1'; }();
// PS2X_VU1_FASTSCAN (default ON, =0 reverts): cont.175 stage 1a -- walk only the SET bits of
// the lane/VI masks in calculatePairReadyCycle/markPairWrites instead of scanning every slot.
// Pure loop transformation: same slots visited, and every consumer is std::max() or a write to
// an independent slot, so the visit order is immaterial and results are bit-identical. The kill
// switch keeps the old scans available for A/B in ONE binary (build-to-build noise removed).
static const bool s_vu1FastScan = []
{ const char *e = std::getenv("PS2X_VU1_FASTSCAN"); return !(e && e[0] == '0'); }();
// PS2X_VU1_PERF (default OFF): interpreter-throughput instrument. Wall ns inside run() and the
// number of instruction pairs issued, reported as ns/pair -- a near-pure measure of interpreter
// cost that is far less scene-confounded than wall-clock frame timing (which is bimodal here).
// One clock read per run() CALL, not per instruction. VU1 only (VU0 runs the same class).
static const bool s_vu1Perf = []
{ const char *e = std::getenv("PS2X_VU1_PERF"); return e && e[0] && e[0] != '0'; }();
// PS2X_VU1_PIPEVERIFY (default OFF): cont.175 stage 1b self-check -- in the tracked path,
// cross-check the valid-slot bitmasks against a full array scan and report any slot the mask
// MISSED (the only direction that could lose a commit). Proof-by-construction is the primary
// argument; this makes it empirically checkable on a real run.
static const bool s_vu1PipeVerify = []
{ const char *e = std::getenv("PS2X_VU1_PIPEVERIFY"); return e && e[0] && e[0] != '0'; }();
// PS2X_VU1_LAZYFLAGS (cont.177; default 1 = ON, "=0" reverts): LAZY FMAC FLAGS. The MAC/status
// flags are derived for every FMAC but are only OBSERVABLE if something reads them, so for a
// microprogram that contains no reader the derivation is dead work and skipping it is BIT-EXACT.
//   0 = off (the kill switch: full flag derivation on every FMAC, the pre-cont.177 behaviour).
//   1 = scan-gated (the honest optimization, DEFAULT): the whole-buffer scan in
//       rebuildDecodedCodeCache must prove the program contains no MAC/status reader.
//   2 = FORCE, an ABLATION: take the fast path regardless of the scan. Deliberately unfaithful
//       (a program that DOES read flags reads stale ones). ⚠ It turned out to be USELESS as a
//       ceiling instrument here: feeding stale flags to the ~1M MAC reads/run CHANGES WHAT THE
//       GUEST DOES, so it measured 208.6 ns/pair — no faster than the 210.7 baseline and slower
//       than the honest mode 1 (175.5). Kept only to reproduce that result. Never a correctness run.
// ★ Scope: this skips ONLY calculateFmacProductSticky + updateFmacFlags. It does NOT skip
// normalizeFmacResult, because that call also CLAMPS the lane values in place (fmacClampExact /
// normalizeResult take the float by reference) — the PS2 has no NaN/Inf, so dropping the clamp
// changes architectural VF/ACC values, not just flags. That is exactly why the fast path here is
// NOT PS2X_VU1_NOFLAGS (which skips the clamp too and is therefore value-unfaithful).
// Restricted to VU1: VU0's flags ARE EE-visible (copyVu0StateToContext copies state.mac/clip/
// status into the R5900 context on every VCALLMS), whereas no vu1_mac/vu1_clip/vu1_status field
// exists anywhere in the runtime, so VU1's flags escape only through the reader opcodes below.
static const int s_vu1LazyFlags = []
{ const char *e = std::getenv("PS2X_VU1_LAZYFLAGS"); return (e && e[0]) ? std::atoi(e) : 1; }();
// PS2X_VU1_LAZYVERIFY (default OFF): self-check for the scan — report if a MAC/status reader ever
// ISSUES while the fast path is armed (i.e. the static scan missed a reader the run executes).
// Same role PS2X_VU1_PIPEVERIFY played for stage 1b: make the by-construction argument empirically
// checkable on a real run.
static const bool s_vu1LazyVerify = []
{ const char *e = std::getenv("PS2X_VU1_LAZYVERIFY"); return e && e[0] && e[0] != '0'; }();

// PS2X_VU1_FASTDISPATCH (cont.178; default ON, "=0" reverts): removes two pieces of pure
// per-pair dispatch overhead the cont.178 profile found (memset 6% + memcpy 5% + memmove 2% +
// getDecodedInstructionPairForPc 4% of the EE thread). Both are "same result, less work" — they
// are the interpreter half of the JIT's third bucket (per-instruction overhead), takeable now:
//   (a) The decode cache is resolved ONCE per run() instead of per pair. The old path called
//       getDecodedInstructionPairForPc every iteration, which re-ran the five-field freshness
//       check and then returned the ~80-byte DecodedInstructionPair BY VALUE; the loop now holds
//       a const reference straight into m_decodedCodeCache. Hoisting the freshness check is safe
//       because the guest cannot upload microcode while its own VU program is running (the same
//       argument that lets lazy flags arm once per run).
//   (b) run()'s six 16-byte vf/acc scratch arrays are no longer zero-initialised. Every one of
//       them is memcpy-filled before its only read, under the identical guard (hasUpperWrite /
//       hasDistinctLowerWrite / accWrite != 0), so the 96 bytes of memset per pair were dead.
// The kill switch keeps the old behaviour available for A/B in ONE binary (build-to-build noise
// removed), exactly as PS2X_VU1_FASTSCAN does for stage 1a.
static const bool s_vu1FastDispatch = []
{ const char *e = std::getenv("PS2X_VU1_FASTDISPATCH"); return !(e && e[0] == '0'); }();
// PS2X_VU1_FLOATCLAMP (cont.180; default ON, "=0" restores the exact model): PCSX2's FMAC result
// model. Adopted on a USER DECISION after the divergence was measured, not assumed — over 801M
// lanes of real play the float model differs from the exact one in **0.001% of lanes by value**
// (1 in ~95,000) and 0.003% by flags, and the dominant case is flags-only with an IDENTICAL value
// (`op=0x1c raw=7f7fffff float=…/f0 exact=…/f8`): the float result saturates to exactly FLT_MAX
// without becoming Inf, so the float model sees a normal number where the exact one flags overflow.
// With lazy flags those MAC flags are unobserved in ~89% of programs and no status reader executes
// at all. Verified: 0 degenerate primitives and 0 kick-drops over 14.3M sampled draws, MOVIE-END
// reached, and correct level-era frames. Today every lane is computed twice — once in float by execUpper, then again in
// DOUBLE by fmacExactLane so fmacClampExact can derive the clamp and the flags from the exact
// result. PCSX2 does not do this: `VUflags.cpp VU_MAC_UPDATE` derives both the clamp and the MAC
// flags purely from the FLOAT result's own bits (sign; f==0 -> Z; exp==0 -> Z|U and flush to signed
// zero; exp==255 -> O and clamp to sign|0x7F7FFFFF; else unchanged). Our `normalizeResult` is
// already a bit-exact re-implementation of that function, so this flag simply routes every lane
// through it and skips the double recompute entirely.
// ★ WHY IT MATTERS FOR THE JIT (this is the point, not the interpreter's few %): the exact path is
// four SCALAR double computations plus per-lane branches per instruction and cannot be vectorised,
// so it would dominate a translated block. The float model is a pure function of the result bits —
// exactly the shape that becomes a handful of SSE ops over all four lanes at once, which is what
// ~60 host cycles/pair requires.
static const bool s_vu1FloatClamp = []
{ const char *e = std::getenv("PS2X_VU1_FLOATCLAMP"); return !(e && e[0] == '0'); }();
// PS2X_VU1_FLOATVERIFY (default OFF): measure the divergence instead of arguing about it. Runs BOTH
// models per lane and tallies how often the clamped VALUE and the FLAGS differ — while keeping the
// exact result, so the run itself stays on the current model.
static const bool s_vu1FloatVerify = []
{ const char *e = std::getenv("PS2X_VU1_FLOATVERIFY"); return e && e[0] && e[0] != '0'; }();
// PS2X_VU1_SIMD (cont.181; default ON, "=0" reverts to the scalar per-lane loops): now that both VU
// clamps are pure functions of a lane's bits (cont.180), do all four lanes at once with SSE4.1.
// The scalar paths remain the reference and stay reachable for A/B in the same binary.
static const bool s_vu1Simd = []
{ const char *e = std::getenv("PS2X_VU1_SIMD"); return !(e && e[0] == '0'); }();
// PS2X_VU1_SIMDVERIFY (default OFF): one-shot exhaustive equivalence test of the SSE quad clamps
// against the scalar reference (see the sweep in run()).
static const bool s_vu1SimdVerify = []
{ const char *e = std::getenv("PS2X_VU1_SIMDVERIFY"); return e && e[0] && e[0] != '0'; }();
// PS2X_VU1_JITSELFTEST (cont.182; default OFF): emit a native operand-clamp function with the JIT
// emitter and verify it against the C++ quad helper over an exhaustive bit sweep. Proves the
// x86-64 encodings and the emitted SSE sequence before any emitted code touches guest state.
static const bool s_vu1JitSelfTest = []
{ const char *e = std::getenv("PS2X_VU1_JITSELFTEST"); return e && e[0] && e[0] != '0'; }();

// PS2X_VU1_BLOCK (cont.195; ★ cont.218: now DEFAULT ON, "=0" is the kill switch): run compiled
// BLOCKS of consecutive pairs in place of the interpreter's inner loop. PS2X_VU1_BLOCKVERIFY=1
// shadow-verifies every block against the interpreter (the block runs on a COPY; the interpreter
// still does the real execution).
// ★★ Why it is safe to ship on: 93.7% of VU1 pairs and 85.7% of VU0 pairs run as compiled blocks,
// and every layer now has an oracle that has been shown capable of FAILING --
// `PS2X_VU1_BLOCKVERIFY` (computation, 12.7G comparisons, 0 mismatches),
// `PS2X_VU1_READYVERIFY` (the ready-table replay, mutation-proven, 0 mismatches) and
// `PS2X_VU1_FLAGVERIFY` (the flag replay, mutation-proven against four injected defects;
// 16,900,001 checks / 80,435,178 events / 0 mismatches). Frame-matched 12.4x on VU1; 1.89-2.55x on
// VU0 at 19 matched cumulative-pair points. The interpreter remains the reference implementation
// and every verifier still runs against it.
static const bool s_vu1Block = []
{ const char *e = std::getenv("PS2X_VU1_BLOCK"); return !(e && e[0] == '0'); }();
// ★★ cont.214 (PS2X_VU0_BLOCK; REQUIRES PS2X_VU1_BLOCK -- it does not imply it, `blockArmed`
// tests s_vu1Block first, so `PS2X_VU0_BLOCK=1` alone silently does nothing):
// let VU0 micro-mode use the same block JIT. It was excluded by a bare `m_unit == Unit::VU1` and
// never measured -- `perfTrack` was VU1-only too -- yet a direct count puts VU0 at 42% of VU time
// (90M pairs at 213 ns/pair against VU1's 62), from just TWO programs at 17 pairs per VCALLMS.
// The historical blocker was flag VISIBILITY: VU0's MAC/status/clip are EE-readable via CFC2, so
// lazy flags (which SKIP the flag math) can never apply. cont.204's flag-emitting mode does not
// skip it -- it reconstructs it -- and needs only "no STATUS reader", which the scan confirms for
// both VU0 programs (clean=0 readsStatus=0). And in this game the EE never reads them at all:
// `ctx->vu0_status`/`vu0_mac_flags`/`vu0_clip_flags` appear ZERO times across every generated file.
// ★ cont.218: DEFAULT ON, "=0" is the kill switch. Note this flag only has effect while
// PS2X_VU1_BLOCK is also on -- `blockArmed` tests `s_vu1Block` first, because both units share the
// block machinery.
static const bool s_vu0Block = []
{ const char *e = std::getenv("PS2X_VU0_BLOCK"); return !(e && e[0] == '0'); }();
// ★ cont.218: now that blocks are the DEFAULT, the run log must say so -- a reader six months from
// now should not have to know which build flipped it. Defined after both flags in the same TU, so
// their dynamic initialisers have already run.
static const bool s_vuBlockBanner = []
{
    std::fprintf(stderr, "[vu:block] VU1=%s VU0=%s (default ON since cont.218; "
                         "PS2X_VU1_BLOCK=0 / PS2X_VU0_BLOCK=0 revert to the interpreter)\n",
                 s_vu1Block ? "on" : "off", (s_vu1Block && s_vu0Block) ? "on" : "off");
    return true;
}();
// PS2X_VU1_BLOCKSTRICT=1 restores the cont.195 guard, which rejected a block whenever ANY pending
// VF write intersected its touched slots (99.7% of all rejections). Kept for same-binary A/B.
// ★★★ cont.230: PS2X_VU1_MICROVU runs VU1 programs on the ported PCSX2 microVU. DEFAULT ON since
// build 389: verified against this interpreter by PS2X_VU1_PROGVERIFY over 1.53 M programs
// (registers, Q/P/I/R, pc, VU memory but 5 boot-era runs, every XGKICK packet identical; sticky
// status bits differ per PCSX2's lazy-flag approximation), present images of a scripted walk intact,
// VU1 4.4-5.2 ns per cycle vs 33 ns per pair here. "=0" restores the interpreter + block JIT.
static const bool s_vu1MicroVu = []
{ const char *e = std::getenv("PS2X_VU1_MICROVU"); return !(e && e[0] == '0'); }();
static unsigned long long g_microVuNs = 0;
// ★ cont.250 PS2X_VU0_MICROVU (default OFF until verified): run VU0 microprograms on the ported
// microVU as well. cont.250 measured VU0 at 85.34 ns/pair on our block JIT against microVU's
// 4.07 ns/cycle on VU1 -- ~21x less efficient per unit of work -- while VU0 costs ~16.7 ms/frame,
// 27% of the EE half. The workload is a JIT best case: progs=2, 51.5 M invocations, 18.3 pairs each.
// Its own ns accumulator so VU0's time never pollutes the [vu1:microvu] figures.
// ★★★ cont.250: DEFAULT ON since build 589, cleared by PS2X_VU0_PROGVERIFY over 46,313,472 programs:
// vf=0 vi=0 qp=0 pc=0 mem=0 -- every value the guest can observe is BIT-EXACT (that also verifies the
// hand-derived VU0 masks: pc=0 proves TPC/start_pc 0x1FF/0x0FF8, mem=0 proves the data path). The only
// divergent class is `flags` (0.51%) plus sticky (5.2%), microVU's documented lazy-flag approximation
// -- and it is PROVABLY UNOBSERVABLE here: the generated code contains ZERO `cfc2` (no instruction can
// read a COP2 control register; only one `ctc2` write exists), never references ctx->vu0_{mac,status,
// clip}_flags, and the VU0 programs themselves report readsStatus=0. cycleDiff is 100%, the same
// accepted situation VU1 has shipped with since build 389. "=0" restores the block JIT for A/B.
static const bool s_vu0MicroVu = []
{ const char *e = std::getenv("PS2X_VU0_MICROVU"); return !(e && e[0] == '0'); }();
static unsigned long long g_microVu0Ns = 0;
static unsigned long long g_microVu0Runs = 0;
// ★ cont.250 PS2X_VU0_PROGVERIFY=1 (default OFF): the VU0 differential oracle, the PROGVERIFY analogue
// that must clear PS2X_VU0_MICROVU before its default can flip. After every microVU VU0 program, re-run
// the SAME program on the block JIT / interpreter from the same input state on a COPY of VU0 data
// memory, and compare registers, flags, Q/P/I/R, pc and VU memory. No kick comparison: XGKICK is gated
// to Unit::VU1, so VU0 has none. Sticky status bits (6-11) are tallied apart, as on VU1 -- microVU's
// lazy flags only guarantee the last four producers before a reader (PCSX2's documented approximation).
static const bool s_vu0ProgVerify = []
{ const char *e = std::getenv("PS2X_VU0_PROGVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_pv0Runs = 0, g_pv0Bad = 0, g_pv0BadVf = 0, g_pv0BadVi = 0, g_pv0BadFlags = 0,
                          g_pv0BadQP = 0, g_pv0BadMem = 0, g_pv0BadPc = 0, g_pv0Sticky = 0, g_pv0CycleDiff = 0;
// ★ cont.230 PS2X_VU1_PROGVERIFY=1 (default OFF): after every microVU program run, re-run the SAME
// program on this interpreter from the same input state on a COPY of VU data memory with XGKICK
// recorded instead of submitted, and compare registers, flags, pc, VU memory and the kick list.
// The differential oracle of the port (docs/vu1-program-compiler.md §5).
static const bool s_vu1ProgVerify = []
{ const char *e = std::getenv("PS2X_VU1_PROGVERIFY"); return e && e[0] && e[0] != '0'; }();
static bool g_vu1DryKicks = false;
static std::vector<ps2x_microvu::KickRecord> g_vu1DryKickList;
static unsigned long long g_pvRuns = 0, g_pvBad = 0, g_pvBadVf = 0, g_pvBadVi = 0, g_pvBadFlags = 0,
                          g_pvBadQP = 0, g_pvBadMem = 0, g_pvBadKick = 0, g_pvBadPc = 0, g_pvCycleDiff = 0, g_pvStickyDiff = 0;
static const bool s_vu1BlockStrict = []
{ const char *e = std::getenv("PS2X_VU1_BLOCKSTRICT"); return e && e[0] && e[0] != '0'; }();
static const bool s_vu1BlockVerify = []
{ const char *e = std::getenv("PS2X_VU1_BLOCKVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_blockVerifyChecked = 0, g_blockVerifyMismatch = 0;

// PS2X_VU1_BLOCKPROF (default OFF): rdtsc-time each SEGMENT of the block fast path, so "where does
// the per-entry cost go" is answered by measurement instead of argument. Measurement only -- it
// changes no behaviour, so unlike an ablation it cannot make the guest diverge (which would make
// the comparison meaningless here, since VU1 output feeds the geometry the guest then processes).
// ★ cont.213: LEVELS, because the probe is a real fraction of what it measures. 1 = per-segment
// timestamps (5 rdtsc reads per entry); 2 = ONE pair around the whole entry, so `whole` is clean.
// Comparing level 2's whole against level 1's whole measures the probe overhead itself, which is
// the only way to know how much of a ~40-cycle segment is the segment and how much is the rdtsc.
static const int s_vu1BlockProfLevel = []
{ const char *e = std::getenv("PS2X_VU1_BLOCKPROF"); return (e && e[0]) ? std::atoi(e) : 0; }();
static const bool s_vu1BlockProf = s_vu1BlockProfLevel > 0;
static const bool s_vu1BlockProfSeg = s_vu1BlockProfLevel == 1;
// cont.219: split the "lookup" segment into DISPATCH (find the Block*) and GUARD (the
// pending-write test). They were measured together, which made block LINKING look like it
// could reclaim ~40 cycles/entry -- but a linked successor still needs its own guard, so
// only the dispatch half is reclaimable. This says how big that half actually is.
static unsigned long long g_bpDispatch = 0;
static unsigned long long g_bpLookup = 0, g_bpSched = 0, g_bpApply = 0, g_bpFn = 0,
                          g_bpRetire = 0, g_bpEntries = 0, g_bpMiss = 0;
// ★★ cont.212: `retire` is the largest driver segment (77 of 260 cycles/entry, cont.211) but it
// averages over TWO populations that need completely different fixes -- blocks that replay
// updateFmacFlags per FMAC (`emitsFlags`, i.e. cont.204 mode: only programs containing a MAC
// reader, ~122/1166 by the cont.177 scan, yet 68.8% of still-interpreted cycles) and blocks that
// do not and only replay the ready tables. Split the SAME timestamps by population and count the
// work volume; deliberately NO new rdtsc points inside retire, because an rdtsc probe on a path
// this hot measures itself and retire is only ~77 cycles wide (the cont.202 trap).
// PS2X_VU1_READYVERIFY (default OFF): DIFFERENTIAL check of the cont.212 quad-coalesced ready
// records. It replays the block's list BOTH ways from one snapshot -- coalesced (the production
// consumer) and expanded back to the pre-cont.212 per-lane records -- into the REAL tables, then
// compares all four of them. It exercises the production code rather than restating it, which is
// what makes it a test and not an argument. PS2X_VU1_BLOCKVERIFY cannot cover this: the whole
// retire path is inside `if (!s_vu1BlockVerify)`, so verify mode never runs it.
static const bool s_vu1ReadyVerify = []
{ const char *e = std::getenv("PS2X_VU1_READYVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_rvChecked = 0, g_rvBad = 0;
// cont.215 diagnosis: the last few blocks each unit executed, so a reserved-instruction report can
// say what put the PC there. BLOCKVERIFY cannot see this -- it never lets a block drive execution.
// PS2X_VU_LASTBLK=1 (default OFF): it writes on EVERY block entry, so it is gated.
static const bool s_vuLastBlk = []
{ const char *e = std::getenv("PS2X_VU_LASTBLK"); return e && e[0] && e[0] != '0'; }();
// cont.216 diagnosis (PS2X_VU_DECPROBE=1, default OFF): at a reserved-instruction report, print the
// pair the loop ACTUALLY decoded beside a FRESH read of the same address out of the code buffer.
// That separates "the decode path returned garbage" from "the PC is not what we think it is" --
// the two hypotheses cont.215 could not tell apart, because the reserved report prints the decoded
// word and the code dump prints memory, and nothing printed the two side by side for ONE address.
static const bool s_vuDecProbe = []
{ const char *e = std::getenv("PS2X_VU_DECPROBE"); return e && e[0] && e[0] != '0'; }();
// ★★ cont.217 (PS2X_VU1_FLAGVERIFY=1, default OFF): the differential oracle the block driver's FLAG
// REPLAY has never had. `BLOCKVERIFY` cannot cover it -- the entire driver path from early-apply
// through retire lives inside `if (!s_vu1BlockVerify)`, so verify mode never executes it (that
// structural blind spot is exactly what hid the cont.216 overflow). cont.212's `READYVERIFY` covers
// the READY-TABLE replay, a different list.
// Same shape as READYVERIFY: from ONE snapshot, run the production path and an INDEPENDENT
// restatement, then compare. Production walks clip events and MAC events in two separate passes and
// picks slots via the valid-slot bitmask + ctz; the reference merges both event streams into one
// ascending-cycle sequence (which is the order the INTERPRETER would have issued them in, one pair
// at a time) and uses plain linear scans for both retire and slot selection. So the check is not a
// restatement of the same code -- it is the interpreter's ordering contract versus the driver's.
// ★★ cont.220 (PS2X_VU1_FASTFLAGRETIRE, default ON, "=0" reverts): cont.216 made the block driver
// call retireDueFlagEntries() once PER REPLAYED FLAG EVENT (~6 per entry), and each call is a
// pipeFor() plus a walk of the flag pipeline -- which cont.216 also doubled from 8 to 16 slots. The
// cont.219 segment split found `retire` is now the DOMINANT driver segment (115 cycles/entry
// VU1-only = 38% of the entry, 145 blended with VU0 = 43%), so those repeated walks are worth
// removing. Track the earliest readyCycle among live flag entries and skip the walk entirely when
// nothing can be due -- the same short-circuit commitReadyPipelines() already uses via
// pipe.nextReady. Correctness is unaffected BY CONSTRUCTION: it only ever skips a walk that would
// have retired nothing.
// cont.220: PROBE-FREE evidence of how much work the guard removes -- ns/pair at matched
// cumulative pairs swings ~20% run to run, far wider than this effect, and cont.213 already
// showed an rdtsc probe on a path this hot measures itself. Counted only while BLOCKPROF is on.
static unsigned long long g_ffrCalls = 0, g_ffrSkipped = 0, g_ffrRetired = 0;
// ★★★ cont.221 (PS2X_VU1_FASTKICK, default ON, "=0" restores the byte loop): XGKICK's per-qword
// copy was 16 iterations of
//     packet[copiedBytes + i] = vuData[(sourceAddress + copiedBytes + i) % m_activeVuDataSize];
// -- a byte at a time, with an integer MODULO by a runtime variable on every byte. The compiler
// cannot prove m_activeVuDataSize is a power of two, so that is 16 hardware `div`s (~20-40 cycles
// each) per 16 bytes streamed. VU data memory IS a power of two (VU1 16 KB, VU0 4 KB -- the block
// entry guard already relies on it), so the whole qword is one masked memcpy, split in two only
// when it wraps the end of VU memory. Bit-identical BY CONSTRUCTION: for base < size and i < 16,
// (base + i) % size is base + i when it fits and base + i - size when it does not, which is exactly
// the two-part copy. PCSX2 parity: `Gif_Path::CopyGSPacketData` / `_vuXGKICK` (Gif_Unit.h,
// VUops.cpp) likewise mask into VU memory and bulk-copy -- behaviour is unchanged, only the cost.
// ★★★ cont.222 (PS2X_VU1_FASTKICKRESET, default ON, "=0" restores `m_xgkick = {}`): resetScheduler
// value-initialised the WHOLE XgkickPipeline on every execute(), and that struct carries a
// std::array<uint8_t, 0x10000> -- a 64 KB memset. For VU0 that runs on EVERY VCALLMS (5.3M per
// run) = ~347 GB of memset, and the cont.222 profile caught it as 8.3% of EE-thread SELF time,
// every sample on the stack executeVU0Microprogram -> execute -> resetScheduler -> memset.
// Only the CONTROL fields need clearing. `packet` bytes are always written before they are read:
// the qword is copied to `packet[copiedBytes]` before the GIFtag is parsed from that same offset,
// and finishXgkick submits only `totalBytes`, which is `currentTagEnd <= copiedBytes`. So no stale
// byte is reachable once copiedBytes is 0.
// ★ cont.158d already did this same surgery to the pipeline arrays here ("the unconditional ~4KB
// of array clears were the top memset"); the 64 KB one hiding inside m_xgkick was missed because
// `m_xgkick = {}` does not look like a bulk clear.
static const bool s_vu1FastKickReset = []
{ const char *e = std::getenv("PS2X_VU1_FASTKICKRESET"); return !(e && e[0] == '0'); }();
// ★★ cont.230 PS2X_VU1_KICKEAGER (default OFF -- MEASURED NEUTRAL: time to reach matched primitive counts was
// identical with it on and off, 35M prims at 307 s vs 306 s; the profile's 19% under progressXgkick was its
// call into the GIF parse + submit, not the copy. Verified equivalent anyway: 1,027,251 kicks, 0 EAGER DIFF.
// "=1" enables): XGKICK copies and
// tag-walks the WHOLE packet at issue, in one pass through the same code, and the per-cycle
// progressXgkick keeps only the PATH1 clock (16 bytes per 2 credits) that decides when the kick
// completes -- so the stall on a second XGKICK, pipelinesPending(), the block guard and the SUBMISSION
// CYCLE are all unchanged. What changes is only WHEN the bytes are read from VU memory: at issue
// instead of progressively. A program that overwrote the kicked region before PATH1 reached it would
// differ -- on hardware that is a data race games avoid by construction (they double-buffer and rely
// on the very stall modelled above), and PS2X_VU1_KICKVERIFY=1 measures it: at completion the eager
// packet is compared with a fresh copy of the region (`[vu1:kickverify] EAGER DIFF`). Why: the
// cont.230 profile put progressXgkick at 19% of the EE thread -- 357M calls per run to move 179M
// quadwords, one per emulated cycle.
static const bool s_vu1KickEager = []
{ const char *e = std::getenv("PS2X_VU1_KICKEAGER"); return e && e[0] == '1'; }();
static bool g_xgkickEagerPass = false;
static unsigned long long g_kvEagerChecked = 0, g_kvEagerDiff = 0;
static const bool s_vu1FastKick = []
{ const char *e = std::getenv("PS2X_VU1_FASTKICK"); return !(e && e[0] == '0'); }();
// Probe-free sizing (cont.220's lesson: count the thing before optimising it).
static unsigned long long g_kickQwords = 0, g_kickCalls = 0, g_kickWraps = 0;
// ★★ cont.221 (PS2X_VU1_KICKVERIFY=1, default OFF): differential cover for the fast copy.
// TWO parts, because the live run alone cannot cover it -- the real workload measured `wraps=0`,
// so the wrap split NEVER executes in game and a bug there would sit latent until some future
// program kicked from the end of VU memory.
//   (a) a one-shot EXHAUSTIVE self-test over every base offset of a synthetic buffer, which hits
//       all 15 wrap alignments by construction;
//   (b) a live per-qword check of the fast result against the original byte loop.
static unsigned long long g_kvChecked = 0, g_kvBad = 0;
static unsigned long long g_kvSubmits = 0, g_kvUnwritten = 0;
static const bool s_vu1KickVerify = []
{ const char *e = std::getenv("PS2X_VU1_KICKVERIFY"); return e && e[0] && e[0] != '0'; }();
// Exhaustive: for a power-of-two buffer, every base 0..size-1 x the two paths must agree.
static bool kickCopySelfTest()
{
    static uint8_t buf[4096];
    static uint8_t fast[16], ref[16];
    for (uint32_t i = 0; i < sizeof(buf); ++i)
        buf[i] = uint8_t(i * 31u + (i >> 5));
    const uint32_t size = uint32_t(sizeof(buf));
    unsigned long cases = 0, bad = 0, wrapCases = 0;
    for (uint32_t src = 0; src < size; ++src)
    {
        const uint32_t base = src & (size - 1u);
        if (base + 16u <= size)
        {
            std::memcpy(fast, buf + base, 16u);
        }
        else
        {
            const uint32_t head = size - base;
            std::memcpy(fast, buf + base, head);
            std::memcpy(fast + head, buf, 16u - head);
            ++wrapCases;
        }
        for (uint32_t i = 0; i < 16u; ++i)
            ref[i] = buf[(src + i) % size];
        ++cases;
        if (std::memcmp(fast, ref, 16u) != 0)
            ++bad;
    }
    std::fprintf(stderr, "[vu1:kickverify] selftest cases=%lu wrapCases=%lu mismatches=%lu -> %s\n",
                 cases, wrapCases, bad, bad == 0 ? "BIT-IDENTICAL" : "MISMATCH");
    return bad == 0;
}

static const bool s_vu1FastFlagRetire = []
{ const char *e = std::getenv("PS2X_VU1_FASTFLAGRETIRE"); return !(e && e[0] == '0'); }();
static const bool s_vu1FlagVerify = []
{ const char *e = std::getenv("PS2X_VU1_FLAGVERIFY"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_fvChecked = 0, g_fvBad = 0, g_fvEvents = 0, g_fvOverflow = 0;
static unsigned long long g_fvDeferred = 0;
// ★★ cont.219 (PS2X_VU1_LINKCENSUS=1, default OFF): size the two candidate "link" levers BEFORE
// building either -- the standing directive is to measure an optimization's ceiling by ablation or
// census first (the GPU rasterizer was 2.1x faster and worth 0 fps).
//   Lever A, POINTER LINKING: cache the successor Block* so a chained entry skips the blockByPair
//   lookup. It can only ever save the LOOKUP segment -- sched/apply/retire are semantic pipeline
//   work a linked successor still needs -- so its ceiling is (lookup share) x (chained & predicted).
//   Lever B, TRACE EXTENSION: a block that ends in an UNCONDITIONAL branch (B/BAL) has a
//   compile-time-constant target, so planBlock could simply keep going there with ZERO speculation,
//   and a block that ends without a branch continues at startPc + 8*pairs. Merging those removes a
//   whole ENTRY (~230 cycles), not just its lookup -- an order of magnitude more valuable per hit.
// This census answers which lever is worth building, execution-weighted rather than per compiled
// block (a block compiled once may run millions of times).
static const bool s_vu1LinkCensus = []
{ const char *e = std::getenv("PS2X_VU1_LINKCENSUS"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_lcEntries[2] = {}, g_lcChained[2] = {}, g_lcHit1[2] = {}, g_lcHit2[2] = {};
static unsigned long long g_lcKind[2][12] = {}, g_lcNoBranch[2] = {}, g_lcToInterp[2] = {};
static unsigned long long g_lcPairs[2] = {}, g_lcChainPairs[2] = {};
// successor memory, indexed by block slot
static uint32_t g_lcSucc[2][vu1jit::BlockStore::kMaxBlocks][2] = {};
static int32_t g_lcPrevIdx[2] = {-1, -1};
static bool g_lcPrevWasBlock[2] = {false, false};
static void lcReport()
{
    static const char *kKindName[12] = {"none", "B",     "BAL",   "JR",    "JALR",  "IBEQ",
                                        "IBNE", "IBLTZ", "IBGTZ", "IBLEZ", "IBGEZ", "?"};
    for (unsigned u = 0; u < 2u; ++u)
    {
        if (g_lcEntries[u] == 0ull)
            continue;
        const double e = double(g_lcEntries[u]);
        std::fprintf(stderr,
                     "[vu%u:linkcensus] entries=%llu pairs=%llu meanPairs=%.2f"
                     " chained=%llu (%.1f%%) hit1=%.1f%% hit2=%.1f%%"
                     " toInterp=%llu (%.1f%%) noBranch=%llu (%.1f%%)\n",
                     u, g_lcEntries[u], g_lcPairs[u], g_lcPairs[u] / e, g_lcChained[u],
                     100.0 * g_lcChained[u] / e,
                     g_lcChained[u] ? 100.0 * g_lcHit1[u] / double(g_lcChained[u]) : 0.0,
                     g_lcChained[u] ? 100.0 * g_lcHit2[u] / double(g_lcChained[u]) : 0.0,
                     g_lcToInterp[u], 100.0 * g_lcToInterp[u] / e, g_lcNoBranch[u],
                     100.0 * g_lcNoBranch[u] / e);
        std::fprintf(stderr, "[vu%u:linkcensus] terminator kinds:", u);
        for (unsigned k = 1; k < 12u; ++k)
            if (g_lcKind[u][k])
                std::fprintf(stderr, " %s=%.1f%%", kKindName[k], 100.0 * g_lcKind[u][k] / e);
        std::fprintf(stderr, "\n");
    }
}
// ★★ cont.217 (PS2X_VU1_FLAGMUTATE=<n>, default 0 = off): DELIBERATE DEFECTS injected into the flag
// replay, so FLAGVERIFY can be shown CAPABLE OF FAILING. A verifier that has only ever passed has
// proven nothing (the cont.194 lesson: 10 injected defects found a real gap in the JIT self-tests).
// Diagnostics only -- never enable during a real run.
//   1 = never retire during the replay (the cont.216 bug itself)
//   2 = drop the last MAC event
//   3 = replay MAC events in REVERSE pair order
//   4 = replay MAC events one cycle late
static const int s_vu1FlagMutate = []
{ const char *e = std::getenv("PS2X_VU1_FLAGMUTATE"); return e ? std::atoi(e) : 0; }();
struct LastBlk { uint32_t entryPc, pairs, endsBranch, outPc, cycles; };
static LastBlk g_lastBlk[2][4] = {};
static unsigned g_lastBlkN[2] = {};
static unsigned long long g_bpWhole = 0, g_bpNoPending = 0, g_bpSupersede = 0;
static unsigned long long g_bpPairsHist[9] = {};
// cont.213: why did the block stop growing -- ALL entries, and the 2-pair entries alone (they are
// ~47% of entries but only ~23% of pairs, so they carry the worst overhead ratio in the system).
static unsigned long long g_bpStop[12] = {}, g_bpStop2[12] = {};
// cont.230: reason-5 ("branchrej") breakdown -- keyed by the lower word's opcode fields so a
// real branch at the cap boundary is told apart from each uncompilable lower op. Counted per
// ENTRY while BLOCKPROF is on (same population as g_bpStop).
static std::unordered_map<uint32_t, unsigned long long> g_bpStop5Ops;
static inline uint32_t vu1LowerOpKey(uint32_t lower)
{
    const uint32_t opHi = (lower >> 25) & 0x7Fu;
    uint32_t key = opHi << 25;
    if (opHi == 0x40u)
    {
        key |= lower & 0x3Fu;
        if ((lower & 0x3Cu) == 0x3Cu)
            key |= lower & 0x7C0u;
    }
    return key;
}
static unsigned long long g_bpRetireF = 0, g_bpRetireC = 0;   // retire cycles: flagged / clean
static unsigned long long g_bpFnF = 0, g_bpFnC = 0;           // emitted-code cycles, same split
static unsigned long long g_bpEntF = 0, g_bpEntC = 0;         // entries, same split
static unsigned long long g_bpPairsF = 0, g_bpPairsC = 0;     // pairs, same split
static unsigned long long g_bpReadyN = 0, g_bpClipN = 0, g_bpFlagN = 0; // replay work volume
// ★ RESIDUAL census: with blocks on, WHICH pairs is the interpreter still running, and what do
// they cost? Once coverage is high the residual is what bounds everything, and it is not the
// average pair -- blocks take the cheap ones.
static unsigned long long g_resPairs = 0, g_resCycles = 0;
static unsigned long long g_resLower[128] = {}, g_resLowerCy[128] = {};
static unsigned long long g_resSpecial[128] = {}, g_resSpecialCy[128] = {};
static unsigned long long g_resUpper[128] = {}, g_resUpperCy[128] = {};
static unsigned long long g_resUpSpec[128] = {}, g_resUpSpecCy[128] = {};
// Split the residual by WHY the pair could not be in a block: upper uncovered, lower uncovered,
// both, or neither (in which case the block simply did not form / was guard-blocked).
static unsigned long long g_resWhy[4] = {}, g_resWhyCy[4] = {};
// Is the pair interpreted because the whole PROGRAM could not be blocked (lazy flags not armed --
// the emitted code derives no MAC/status flags, so it may only run for reader-free programs), or
// for a per-pair reason? These need completely different fixes.
// ★ cont.204 flag verification. In verify mode the block runs on a COPY and the driver's flag
// replay does not run, so comparing committed mac/status would be meaningless. Instead the
// interpreter's queued flag ENTRIES are captured over the window and compared against the entries
// the block's derived flags would have produced -- a direct check of the thing that changed.
// Mirrors ps2_vu1_upper.cpp case 0x1F (CLIP) exactly: RAW-bit ordered comparisons, not float.
static inline uint32_t vuClipFlagsFrom(const float *vs, const float *vt)
{
    uint32_t wBits = 0u;
    std::memcpy(&wBits, &vt[3], sizeof(wBits));
    const int32_t limit = (wBits & 0x7F800000u) != 0u
                              ? static_cast<int32_t>(wBits & 0x7FFFFFFFu)
                              : 0x007FFFFF;
    const auto exceeds = [limit](float value, uint32_t signMask) {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        bits ^= signMask;
        int32_t ordered = 0;
        std::memcpy(&ordered, &bits, sizeof(ordered));
        return ordered > limit;
    };
    uint32_t flags = 0u;
    if (exceeds(vs[0], 0x00000000u)) flags |= 0x01u;
    if (exceeds(vs[0], 0x80000000u)) flags |= 0x02u;
    if (exceeds(vs[1], 0x00000000u)) flags |= 0x04u;
    if (exceeds(vs[1], 0x80000000u)) flags |= 0x08u;
    if (exceeds(vs[2], 0x00000000u)) flags |= 0x10u;
    if (exceeds(vs[2], 0x80000000u)) flags |= 0x20u;
    return flags;
}
static bool g_ccActive = false;
static uint32_t g_ccN = 0;
static uint32_t g_ccClip[32] = {};
static uint64_t g_ccIssue[32] = {};
static unsigned long long g_ccChecked = 0, g_ccBad = 0;
static bool g_fcActive = false;
static uint32_t g_fcN = 0;
static uint32_t g_fcMac[32] = {}, g_fcStatus[32] = {};
static uint64_t g_fcIssue[32] = {};
static unsigned long long g_fcChecked = 0, g_fcBad = 0;
static unsigned long long g_resNoLazy = 0, g_resNoLazyCy = 0;
static unsigned long long g_resNoArm = 0, g_resNoArmCy = 0;
static unsigned long long g_fcLanes = 0, g_fcValueDiff = 0, g_fcFlagDiff = 0, g_fcReported = 0;

// PS2X_VU1_JITCENSUS (default OFF): cont.178 groundwork for the VU1 block JIT. Histograms what the
// interpreter ACTUALLY executes -- upper opcode, lower opcode, branch density, and the distribution
// of run() entry PCs -- so the JIT's opcode coverage and block-cache design are driven by measured
// workload rather than by the ISA manual. Per-instruction increments, so it is gated OFF.
static const bool s_vu1JitCensus = []
{ const char *e = std::getenv("PS2X_VU1_JITCENSUS"); return e && e[0] && e[0] != '0'; }();
static unsigned long long g_jitUpper[64] = {};
static unsigned long long g_jitUpperSpecial[128] = {};
static unsigned long long g_jitLower[128] = {};
static unsigned long long g_jitPairs = 0, g_jitBranches = 0, g_jitIBit = 0, g_jitEBit = 0;
static unsigned long long g_schedCalls = 0, g_schedStalledPairs = 0, g_schedStallCycles = 0;
// cont.184: run-length histogram of block-compilable pairs (index 16 = 16 or longer).
static unsigned long long g_blkRunHist[17] = {}, g_blkPairsInRuns = 0;
static unsigned g_blkRunCur = 0;
// cont.186: cumulative lower-opcode tiers (see the tier comment in run()).
static unsigned long long g_tierPairs[7] = {}, g_tierPairsGe2[7] = {}, g_tierRuns[7] = {};
static unsigned g_tierRunCur[7] = {};

// The ONLY instructions that can observe m_state.mac / m_state.status inside a microprogram
// (ps2_vu1_lower.cpp, switch on opHi = (instr >> 25) & 0x7F):
//   0x14 FSEQ, 0x16 FSAND, 0x17 FSOR  -> read m_state.status
//   0x18 FMEQ, 0x1A FMAND, 0x1B FMOR  -> read m_state.mac
// The CLIP readers (0x10 FCEQ, 0x12 FCAND, 0x13 FCOR, 0x1C FCGET) are deliberately NOT here:
// clip is produced by the CLIP instruction via queueClip(), which this optimization does not
// touch. Including them would disqualify most T&L programs (CLIP+FCAND is the standard cull
// test) for no correctness gain.
// Returns which flag register the lower instruction observes: bit 0 = m_state.mac,
// bit 1 = m_state.status. They are tracked separately because their staleness differs (see the
// dirty guard in run()): mac is OVERWRITTEN wholesale by the next FMAC flag write, while the
// status STICKY bits (6..11) only accumulate and are cleared solely by FSSET.
static constexpr unsigned kFlagReadMac = 1u;
static constexpr unsigned kFlagReadStatus = 2u;
static inline unsigned lowerFlagReads(uint32_t lower)
{
    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    if (opHi == 0x18u || opHi == 0x1Au || opHi == 0x1Bu) // FMEQ / FMAND / FMOR
        return kFlagReadMac;
    if (opHi == 0x14u || opHi == 0x16u || opHi == 0x17u) // FSEQ / FSAND / FSOR
        return kFlagReadStatus;
    return 0u;
}

static unsigned long long g_vu1Ns = 0, g_vu1Pairs = 0, g_vu1Calls = 0, g_vu1NextReport = 20000000ull;
// ★★ cont.214: VU0 has NEVER been measured -- `perfTrack` was `s_vu1Perf && m_unit == Unit::VU1`,
// so every ns/pair, coverage and census number in this arc is VU1-only. The cont.211 profile put
// VU0 micro-mode at 28.6% of VU time on 28 samples; this counts it directly. Same units, so the
// two ns/pair figures are directly comparable.
static unsigned long long g_vu0Ns = 0, g_vu0Pairs = 0, g_vu0Calls = 0,
                          g_vu0NextReport = 2000000ull;
static unsigned long long g_vu0ProgsScanned = 0, g_vu0ProgsClean = 0, g_vu0ProgsStatus = 0;
// cont.177 lazy-flag tallies. progsScanned/progsClean are per code-buffer rebuild (negligible, so
// always on); runsArmed counts run() calls that took the fast path; readerIssues counts every
// MAC/status reader that actually EXECUTES and is only maintained under PS2X_VU1_LAZYVERIFY (it is
// a per-pair test). readerIssues == 0 over a full run is the empirical proof that the flags are
// dead in this workload -- which is also what closes the cross-program staleness question (a
// reader in a LATER program would otherwise observe flags an earlier fast-path program skipped).
static unsigned long long g_lazyProgsScanned = 0, g_lazyProgsClean = 0,
                          g_lazyRunsArmed = 0, g_lazyReaderIssues = 0,
                          g_lazyMacReads = 0, g_lazyStatusReads = 0,
                          g_lazyReadsBeforeWrite = 0, g_lazyDirtyReads = 0;
namespace
{
    // Result of the whole-code-buffer flag-reader scan, recorded by rebuildDecodedCodeCache.
    // File-static on purpose: a member would touch include/runtime/ps2_vu1.h, which the generated
    // code includes (~60 min rebuild). The identity fields mirror EXACTLY the freshness check in
    // getDecodedInstructionPairForPc, so "record matches" == "the decode cache this scan walked is
    // still the live one" (m_decodedCodeCacheValid is only ever set true, never invalidated).
    struct LazyScan
    {
        const void *owner = nullptr;
        const uint8_t *code = nullptr;
        const void *memory = nullptr;
        uint32_t codeSize = 0;
        uint64_t generation = ~0ull;
        bool complete = false;         // the scan covered every pair in the buffer
        bool readsMacOrStatus = true;  // conservative default
        // ★ cont.204: split, so a program that reads MAC but never STATUS can still run
        // compiled blocks (they emit MAC flags but not the product-sticky STATUS bits).
        bool readsStatus = true;
    };
    // ★ cont.214: PER-UNIT. The scan already ran for both units; only the RECORDING was VU1-only
    // because this was a single shared slot, which is why VU0 could never arm blocks. Index 1 = VU1.
    LazyScan s_lazyScan[2];
    // Armed once per run() call (VU1 only) — read per FMAC, so it must be a plain bool test.
    bool s_lazyActive = false;
    // Latched off permanently by the cross-program dirty guard (see run()).
    bool s_lazyDisabled = false;
    // Staleness bookkeeping for that guard. Both are set when an armed run skips flag writes.
    // s_macDirty clears on the next real flag write (mac is overwritten wholesale); s_stickyDirty
    // clears only on FSSET, because the status sticky bits accumulate and a later write PRESERVES
    // bits 4..11 — so a skipped write leaves them permanently under-set until FSSET resets them.
    bool s_macDirty = false;
    bool s_stickyDirty = false;
    // Has any real flag write been queued during the current run()?
    bool s_flagWriteThisRun = false;
    struct PipeTrack
    {
        const void *owner = nullptr;
        uint64_t nextReady = kPipeIdle; // lower bound while owner holds; kPipeIdle = none valid
        // cont.175 stage 1b: these were per-array VALID COUNTS; they are now per-array
        // VALID-SLOT BITMASKS (bit i == m_<array>[i].valid). Every previous use compared them
        // against 0 ("is this array non-empty"), which a mask answers identically -- and the
        // mask additionally lets commitReadyPipelines retire O(valid) instead of O(capacity)
        // and lets the queue* helpers pick a free slot with one ctz instead of a linear probe.
        // fdiv is a single entry, so its "mask" is 0/1.
        uint32_t flag = 0, efu = 0, store = 0, vf = 0, vi = 0, acc = 0, fdiv = 0;
        uint32_t total() const { return flag | efu | store | vf | vi | acc | fdiv; }
    };

    // Retire only the slots the tracking marks valid. `retire(entry)` returns true if the entry
    // committed (slot freed), false if it is still pending. Untracked => the old full scan, so
    // behavior is identical whenever the mask is not trustworthy. Returns the survivor mask.
    template <typename Array, typename Fn>
    inline uint32_t retirePipe(Array &arr, uint32_t validMask, bool tracked, Fn &&retire)
    {
        uint32_t survivors = 0u;
        if (tracked)
        {
            uint32_t m = validMask;
            while (m != 0u)
            {
                const uint32_t i = static_cast<uint32_t>(__builtin_ctz(m));
                m &= m - 1u;
                if (!arr[i].valid)
                    continue; // mask superset of reality: harmless, cannot lose a commit
                if (!retire(arr[i]))
                    survivors |= 1u << i;
            }
            return survivors;
        }
        for (uint32_t i = 0; i < arr.size(); ++i)
        {
            if (!arr[i].valid)
                continue;
            if (!retire(arr[i]))
                survivors |= 1u << i;
        }
        return survivors;
    }

    // Free-slot pick: one ctz off the complement of the valid mask, with the array's own
    // .valid as the arbiter -- if the mask ever disagreed we fall through to the linear probe,
    // so a stale mask can cost a scan but can never overwrite a live entry.
    template <typename Array>
    inline int freeSlot(Array &arr, uint32_t validMask, bool tracked)
    {
        const uint32_t all = (arr.size() >= 32u) ? ~0u : ((1u << arr.size()) - 1u);
        if (tracked)
        {
            const uint32_t free = ~validMask & all;
            if (free != 0u)
            {
                const uint32_t i = static_cast<uint32_t>(__builtin_ctz(free));
                if (!arr[i].valid)
                    return static_cast<int>(i);
            }
        }
        for (uint32_t i = 0; i < arr.size(); ++i)
            if (!arr[i].valid)
                return static_cast<int>(i);
        return -1;
    }
}
// One tracking slot per live interpreter instance (VU0 + VU1), keyed by pointer identity
// (cont.160): with a single shared slot, every VU0<->VU1 alternation evicted the other
// unit's tracking, forcing a full status-quo scan per switch — measured as
// commitReadyPipelines staying the top profile leaf and the resetScheduler skip almost
// never firing for VU0 (a VCALLMS always runs right after VU1 activity). A slot whose
// owner != self reads as "untracked" at every use site, and the next full scan installs
// self as that slot's owner; with more than two instances (never happens today) slot 0
// gets evicted, which degrades to full scans — safe, never wrong.
static PipeTrack s_pipeSlots[2];
static inline PipeTrack &pipeFor(const void *self)
{
    if (s_pipeSlots[0].owner == self)
        return s_pipeSlots[0];
    if (s_pipeSlots[1].owner == self)
        return s_pipeSlots[1];
    if (s_pipeSlots[0].owner == nullptr)
        return s_pipeSlots[0];
    if (s_pipeSlots[1].owner == nullptr)
        return s_pipeSlots[1];
    return s_pipeSlots[0];
}
static inline void notePipeInsert(const void *self, uint32_t PipeTrack::*mask, uint32_t slot, uint64_t readyCycle)
{
    PipeTrack &pipe = pipeFor(self);
    if (pipe.owner == self)
    {
        pipe.*mask |= (1u << slot);
        if (readyCycle < pipe.nextReady)
            pipe.nextReady = readyCycle;
    }
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    // Per-program scheduler prep — called at every execute() start, which for VU0 is
    // EVERY VCALLMS at math-library rates (profiling cont.158d: the unconditional ~4KB
    // of array clears here were the top memset in the run). Two cost cuts, both exact:
    //
    // (1) The in-flight pipeline arrays only need clearing when something is in flight.
    // With the commit tracking (cont.158b) owning this instance, total()==0 PROVES all
    // seven arrays are empty (a normal program end drains them via flushPipelines), so
    // the clears are skipped. Any doubt — untracked, other owner, live entries — means
    // a full clear, exactly the old behavior.
    //
    // (2) The ready/latest-write bookkeeping (m_vfReady/m_viReady/m_accReady/
    // m_*LatestWrite/m_nextWriteSequence) is no longer cleared here at all: m_cycle and
    // the write sequence are monotonic across programs (only reset() zeroes them, and it
    // clears this bookkeeping itself), so stale entries are inert — a stale readyCycle
    // is at most its issue cycle + the instruction latency, costing at worst a few
    // bounded stall cycles on a register whose queued write a budget-exhausted program
    // abandoned (the wholesale clear dropped those writes just the same).
    //
    // PCSX2 parity: vu0ExecMicro (VU0micro.cpp) performs no wholesale reset on VCALLMS —
    // it syncs flags/cycle and starts the program with all other state persisting.
    PipeTrack &pipe = pipeFor(this);
    const bool pipesProvedEmpty =
        s_vu1CommitSkip && pipe.owner == this && pipe.total() == 0u && !m_fdiv.valid;
    if (!pipesProvedEmpty)
    {
        if (pipe.owner == this)
            pipe = {}; // wholesale clear below invalidates the tracked state
        m_flagPipeline = {};
        m_fdiv = {};
        m_efu = {};
        m_storePipeline = {};
        m_vfWritePipeline = {};
        m_viWritePipeline = {};
        m_accWritePipeline = {};
    }
    if (s_vu1FastKickReset)
    {
        // Everything except `packet` (64 KB), which is write-before-read -- see the flag comment.
        m_xgkick.sourceAddress = 0;
        m_xgkick.totalBytes = 0;
        m_xgkick.copiedBytes = 0;
        m_xgkick.currentTagEnd = 0;
        m_xgkick.cycleCredit = 0;
        m_xgkick.issueCycle = 0;
        m_xgkick.active = false;
        m_xgkick.currentTagEop = false;
        m_xgkick.eagerDone = false;
        m_xgkick.streamedBytes = 0;
    }
    else
    {
        m_xgkick = {};
    }
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    // m_cycle just went backward to 0, so the monotonicity that lets resetScheduler()
    // keep the ready/latest-write bookkeeping must be re-established by clearing it here.
    m_vfReady = {};
    m_viReady = {};
    m_accReady = {};
    m_vfLatestWrite = {};
    m_viLatestWrite = {};
    m_accLatestWrite = {};
    m_nextWriteSequence = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ---- FMAC exact-result fast path (cont.159 perf). The old shape re-DECODED the upper
// instruction, re-normalized q/i, and re-ran the whole arithmetic in x87 `long double`
// PER LANE (×4 per FMAC op — measured ~12% of runtime as the normalizeFmac*/
// calculateFmacExact leaf family). Restructured: decode + q/i normalization happen ONCE
// per instruction (normalizeFmacResult), and the per-lane arithmetic is precision-generic
// — `double` by default, `PS2X_VU1_EXACTLD=1` restores the old x87 long-double path.
// Fidelity: the exact value only drives CLAMP/FLAG decisions (in-range lanes keep the
// native float result untouched), float×float products are EXACT in double (24+24 ≤ 53
// mantissa bits), zero-detection is exact in any IEEE format, and PCSX2's reference
// interpreter (VUops.cpp `vuDouble` — returns *float*) computes VU arithmetic in plain
// FLOAT with operand normalization + result clamp, so double remains strictly more
// precise than the proven reference; double-vs-long-double verdicts can differ only for
// sums within one double-ulp of the FLT_MAX/FLT_MIN boundaries.
static const bool s_vu1ExactLongDouble = []
{ const char *e = std::getenv("PS2X_VU1_EXACTLD"); return e && e[0] == '1'; }();

// File-static twin of normalizeOperand (denormal -> signed zero, Inf/NaN -> signed max),
// so the templated helpers below need no member access.
static inline float fmacNormOperand(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
        bits &= 0x80000000u;
    else if (exponent == 0xFFu)
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Precision-generic clamp+flags of an exact result onto the lane's float result —
// identical logic to the old normalizeFmacExactResult (which now delegates here).
template <typename Real>
static inline uint8_t fmacClampExact(float &value, Real exactResult)
{
    const bool negative = std::signbit(exactResult);
    const Real magnitude = std::fabs(exactResult);
    const Real maximum = static_cast<Real>(std::numeric_limits<float>::max());
    const Real minimum = static_cast<Real>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == static_cast<Real>(0))
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

// Per-instruction FMAC decode, computed once (the old code derived all of this per lane).
struct FmacDecode
{
    uint8_t op = 0;
    uint8_t special = 0;
    uint8_t fs = 0;
    uint8_t ft = 0;
    float q = 0.0f; // normalized once
    float i = 0.0f; // normalized once
};

static inline FmacDecode decodeFmac(uint32_t upper, float qRaw, float iRaw)
{
    FmacDecode d;
    d.op = static_cast<uint8_t>(upper & 0x3Fu);
    d.special = d.op >= 0x3Cu
                    ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                    : 0xFFu;
    d.fs = FS(upper);
    d.ft = FT(upper);
    d.q = fmacNormOperand(qRaw);
    d.i = fmacNormOperand(iRaw);
    return d;
}

// The per-lane exact arithmetic — the op/special chains are byte-for-byte the old
// calculateFmacExactResult body, just parameterized on the precision and fed the
// pre-decoded instruction.
template <typename Real>
static bool fmacExactLane(const FmacDecode &d, const float (*vf)[4], const float *accReg,
                          uint32_t component, Real &result)
{
    const auto vs = [&](uint32_t lane)
    { return static_cast<Real>(fmacNormOperand(vf[d.fs][lane])); };
    const auto vt = [&](uint32_t lane)
    { return static_cast<Real>(fmacNormOperand(vf[d.ft][lane])); };
    const auto acc = [&](uint32_t lane)
    { return static_cast<Real>(fmacNormOperand(accReg[lane])); };
    const Real q = static_cast<Real>(d.q);
    const Real i = static_cast<Real>(d.i);
    const uint8_t op = d.op;

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? static_cast<Real>(0)
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    const uint8_t special = d.special;
    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? static_cast<Real>(0)
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

// cont.180: score the PCSX2 float model against the exact model for one lane. The caller passes
// what each model produced; this only tallies the difference and keeps the exact answer.
// File-static (not a member) because adding a method would touch include/runtime/ps2_vu1.h, which
// the generated code includes — a ~60 min rebuild.
static void scoreFloatModel(float floatValue, uint32_t floatFlags, float rawResult,
                            float exactValue, uint8_t exactFlags, uint32_t upperInstr)
{
    uint32_t fv = 0, ev = 0;
    std::memcpy(&fv, &floatValue, 4);
    std::memcpy(&ev, &exactValue, 4);
    ++g_fcLanes;
    const bool valueDiff = (fv != ev);
    const bool flagDiff = (static_cast<uint8_t>(floatFlags) != exactFlags);
    if (valueDiff)
        ++g_fcValueDiff;
    if (flagDiff)
        ++g_fcFlagDiff;
    if ((valueDiff || flagDiff) && g_fcReported < 12u)
    {
        ++g_fcReported;
        uint32_t rb = 0;
        std::memcpy(&rb, &rawResult, 4);
        std::fprintf(stderr,
                     "[vu1:floatdiv] op=0x%02x raw=%08x float=%08x/f%x exact=%08x/f%x\n",
                     static_cast<unsigned>(upperInstr & 0x3Fu),
                     rb, fv, static_cast<unsigned>(floatFlags), ev,
                     static_cast<unsigned>(exactFlags));
    }
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    // cont.180 PCSX2 model (PS2X_VU1_FLOATCLAMP): clamp + flags straight from the float result's
    // bits, exactly as PCSX2's VUflags.cpp VU_MAC_UPDATE does — no double recompute at all.
    if (s_vu1FloatClamp && !s_vu1FloatVerify)
    {
        if (s_vu1Simd)
        {
            // cont.181: all four lanes at once. When lazy flags proved nothing reads MAC/status
            // for this program, the flags are dead too — clamp the values and nothing else.
            if (s_lazyActive)
            {
                vuNormResultQuadValue(result, dest);
                laneFlags[0] = laneFlags[1] = laneFlags[2] = laneFlags[3] = 0u;
            }
            else
            {
                vuNormResultQuad(result, dest, laneFlags);
            }
            return;
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            laneFlags[component] = 0u;
            if ((dest & laneForComponent(component)) == 0u)
                continue;
            uint32_t flags = 0u;
            result[component] = normalizeResult(result[component], flags);
            laneFlags[component] = static_cast<uint8_t>(flags);
        }
        return;
    }

    // Decode + q/i normalization ONCE per instruction; per-lane exact arithmetic in
    // double (default) or long double (PS2X_VU1_EXACTLD=1) — see the fast-path comment
    // above fmacExactLane.
    const FmacDecode d = decodeFmac(m_currentUpperInstruction, m_state.q, m_state.i);

    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        // cont.180 divergence meter: keep the raw pre-clamp float so both models can be scored.
        const float rawResult = result[component];

        if (s_vu1ExactLongDouble)
        {
            long double exactResult = 0.0L;
            if (fmacExactLane(d, m_state.vf, m_state.acc, component, exactResult))
            {
                laneFlags[component] = fmacClampExact(result[component], exactResult);
                if (s_vu1FloatVerify)
                {
                    uint32_t ff = 0u;
                    const float fvv = normalizeResult(rawResult, ff);
                    scoreFloatModel(fvv, ff, rawResult, result[component],
                                    laneFlags[component], m_currentUpperInstruction);
                }
                continue;
            }
        }
        else
        {
            double exactResult = 0.0;
            if (fmacExactLane(d, m_state.vf, m_state.acc, component, exactResult))
            {
                laneFlags[component] = fmacClampExact(result[component], exactResult);
                if (s_vu1FloatVerify)
                {
                    uint32_t ff = 0u;
                    const float fvv = normalizeResult(rawResult, ff);
                    scoreFloatModel(fvv, ff, rawResult, result[component],
                                    laneFlags[component], m_currentUpperInstruction);
                }
                continue;
            }
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    // Thin wrapper kept for the header contract; the hot path (normalizeFmacResult)
    // decodes once and calls fmacExactLane directly.
    const FmacDecode d = decodeFmac(m_currentUpperInstruction, m_state.q, m_state.i);
    return fmacExactLane(d, m_state.vf, m_state.acc, component, result);
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    // Thin wrapper kept for the header contract; the hot paths use fmacClampExact directly.
    return fmacClampExact(value, exactResult);
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        // double is EXACT for a product of two normalized floats (24+24 <= 53 mantissa
        // bits), so this is provably identical to the old long-double computation.
        const double exactProduct = static_cast<double>(left) * static_cast<double>(right);
        const uint8_t productFlags = fmacClampExact(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    // cont.177 staleness bookkeeping (see the dirty guard in run()): a real flag write overwrites
    // mac wholesale, so any earlier skipped write stops being observable in mac. It does NOT clear
    // s_stickyDirty — the commit preserves status bits 4..11, so skipped sticky contributions stay
    // missing until an FSSET resets them.
    s_flagWriteThisRun = true;
    s_macDirty = false;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    if (s_vu1NoSched)
    {
        // Same effect as the commit path's writesMac + writesStatus, applied now.
        m_state.mac = mac;
        const uint32_t current = status & 0xFu;
        m_state.status = (m_state.status & 0xFF0u) | current | ((current | extraSticky) << 6);
        return;
    }

    PipeTrack &flagPipe = pipeFor(this);
    const int flagSlot = freeSlot(m_flagPipeline, flagPipe.flag,
                                  s_vu1CommitSkip && flagPipe.owner == this);
    if (flagSlot < 0)
    {
        reportPipelineFull("flag(mac)");
        return;
    }
    if (g_fcActive && g_fcN < 32u)
    {
        g_fcMac[g_fcN] = mac;
        g_fcStatus[g_fcN] = status;
        g_fcIssue[g_fcN] = m_cycle;
        ++g_fcN;
    }
    FlagPipelineEntry *const entry = &m_flagPipeline[static_cast<size_t>(flagSlot)];

    *entry = {};
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
    notePipeInsert(this, &PipeTrack::flag, static_cast<uint32_t>(flagSlot), entry->readyCycle);
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    if (s_vu1NoFlags)
    {
        applyDest(dst, result, dest);
        return;
    }
    uint8_t laneFlags[4]{};
    // normalizeFmacResult stays on BOTH paths: it clamps the lane values IN PLACE (PS2 floats have
    // no NaN/Inf), so skipping it would change architectural VF/ACC contents, not just flags.
    // That is precisely why the lazy path is NOT s_vu1NoFlags.
    normalizeFmacResult(result, dest, laneFlags);
    if (!s_lazyActive)
        updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    if (s_vu1NoFlags)
    {
        applyDestAcc(result, dest);
        return;
    }
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags); // clamps values — never skipped (see applyFmacDest)
    if (!s_lazyActive)
        updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    // FSSET overwrites the status sticky bits outright, so it is the one point where a skipped
    // lazy-flag write stops being observable in status (cont.177 dirty guard).
    s_stickyDirty = false;
    s_flagWriteThisRun = true;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_flagPipeline, pipe.flag, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
            entry.writesSticky = true;
            notePipeInsert(this, &PipeTrack::flag, static_cast<uint32_t>(slot), entry.readyCycle);
            return;
        }
    }
    reportPipelineFull("flag(fsset)");
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_flagPipeline, pipe.flag, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            if (g_ccActive && g_ccN < 32u)
            {
                g_ccClip[g_ccN] = clip & 0x3Fu;
                g_ccIssue[g_ccN] = m_cycle;
                ++g_ccN;
            }
            FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            entry.clip = m_workingClip;
            entry.writesClip = true;
            notePipeInsert(this, &PipeTrack::flag, static_cast<uint32_t>(slot), entry.readyCycle);
            return;
        }
    }
    reportPipelineFull("flag(clip)");
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_flagPipeline, pipe.flag, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(slot)];
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            entry.clip = m_workingClip;
            entry.writesClip = true;
            notePipeInsert(this, &PipeTrack::flag, static_cast<uint32_t>(slot), entry.readyCycle);
            return;
        }
    }
    reportPipelineFull("flag(fcset)");
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    PipeTrack &pipe = pipeFor(this);
    if (pipe.owner == this)
    {
        if (!m_fdiv.valid)
            pipe.fdiv = 1u; // single-entry pipe: its mask is 0/1
        if (m_cycle + latency < pipe.nextReady)
            pipe.nextReady = m_cycle + latency;
    }
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_efu, pipe.efu, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            ScalarPipelineEntry &entry = m_efu[static_cast<size_t>(slot)];
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            notePipeInsert(this, &PipeTrack::efu, static_cast<uint32_t>(slot), entry.readyCycle);
            return;
        }
    }
    reportPipelineFull("efu");
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (s_vu1NoSched)
    {
        if (m_activeVuData && address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = words[component];
            }
            std::memcpy(m_activeVuData + address, oldWords, sizeof(oldWords));
        }
        return;
    }
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_storePipeline, pipe.store, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            PendingStore &store = m_storePipeline[static_cast<size_t>(slot)];
            store.valid = true;
            store.readyCycle = m_cycle + 1u;
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            notePipeInsert(this, &PipeTrack::store, static_cast<uint32_t>(slot), store.readyCycle);
            return;
        }
    }
    reportPipelineFull("store");
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    if (s_vu1NoSched)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((laneMask & laneForComponent(component)) != 0u)
                m_state.vf[reg][component] = value[component];
        }
        return;
    }
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_vfWritePipeline, pipe.vf, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            PendingVfWrite &write = m_vfWritePipeline[static_cast<size_t>(slot)];
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            uint32_t lanes = laneMask & 0xFu;
            while (lanes != 0u)
                m_vfLatestWrite[reg][3u - nextSetBit(lanes)] = write.sequence;
            notePipeInsert(this, &PipeTrack::vf, static_cast<uint32_t>(slot), write.readyCycle);
            return;
        }
    }
    reportPipelineFull("vf");
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    if (s_vu1NoSched)
    {
        m_state.vi[reg] = static_cast<int16_t>(value);
        return;
    }
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_viWritePipeline, pipe.vi, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            PendingViWrite &write = m_viWritePipeline[static_cast<size_t>(slot)];
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            notePipeInsert(this, &PipeTrack::vi, static_cast<uint32_t>(slot), write.readyCycle);
            return;
        }
    }
    reportPipelineFull("vi");
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    if (s_vu1NoSched)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((laneMask & laneForComponent(component)) != 0u)
                m_state.acc[component] = value[component];
        }
        return;
    }
    {
        PipeTrack &pipe = pipeFor(this);
        const int slot = freeSlot(m_accWritePipeline, pipe.acc, s_vu1CommitSkip && pipe.owner == this);
        if (slot >= 0)
        {
            PendingAccWrite &write = m_accWritePipeline[static_cast<size_t>(slot)];
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            uint32_t lanes = laneMask & 0xFu;
            while (lanes != 0u)
                m_accLatestWrite[3u - nextSetBit(lanes)] = write.sequence;
            notePipeInsert(this, &PipeTrack::acc, static_cast<uint32_t>(slot), write.readyCycle);
            return;
        }
    }
    reportPipelineFull("acc");
}

void VU1Interpreter::commitReadyPipelines()
{
    // O(1) fast path (PS2X_VU1_COMMITSKIP): while we own the tracking, pipe.nextReady
    // is a lower bound on the earliest readyCycle among ALL valid entries (kPipeIdle when
    // none are valid), so nothing can be due before it -- the skip is exact, not a
    // heuristic. Any doubt (other owner, flag off) falls through to the full scan.
    PipeTrack &pipe = pipeFor(this);
    const bool tracked = s_vu1CommitSkip && pipe.owner == this;
    if (tracked && m_cycle < pipe.nextReady)
        return;

    // cont.175 stage 1b: when tracked, retire walks only the VALID SLOTS (per-array bitmask)
    // instead of every slot of every non-empty array (~50 slots / ~1.5KB touched per call).
    // Untracked, retirePipe falls back to the identical full scan. Commit order within an
    // array is ascending slot index in BOTH paths (ctz walks low bit first), and each entry's
    // effect is independent of the others in its array, so results are unchanged.
    uint64_t nextReady = kPipeIdle;
    PipeTrack fresh;
    const auto note = [&nextReady](uint64_t readyCycle)
    {
        if (readyCycle < nextReady)
            nextReady = readyCycle;
    };

    if (s_vu1PipeVerify && tracked)
    {
        // Self-check (PS2X_VU1_PIPEVERIFY=1): the only dangerous direction is a mask that
        // MISSES a valid slot, which would skip a due commit. Report and repair.
        static unsigned long s_misses = 0;
        const auto check = [&](const char *name, auto &arr, uint32_t &mask)
        {
            uint32_t actual = 0u;
            for (uint32_t i = 0; i < arr.size(); ++i)
                if (arr[i].valid)
                    actual |= 1u << i;
            if ((actual & ~mask) != 0u && ++s_misses <= 32ul)
                std::fprintf(stderr, "[vu1:pipeverify] MASK MISS %s mask=0x%x actual=0x%x cycle=%llu\n",
                             name, mask, actual, static_cast<unsigned long long>(m_cycle));
            mask = actual;
        };
        check("flag", m_flagPipeline, pipe.flag);
        check("efu", m_efu, pipe.efu);
        check("store", m_storePipeline, pipe.store);
        check("vf", m_vfWritePipeline, pipe.vf);
        check("vi", m_viWritePipeline, pipe.vi);
        check("acc", m_accWritePipeline, pipe.acc);
    }

    fresh.flag = retirePipe(m_flagPipeline, pipe.flag, tracked, [&](FlagPipelineEntry &entry)
    {
        if (entry.readyCycle > m_cycle)
        {
            note(entry.readyCycle);
            return false;
        }
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry = {};
        return true;
    });

    if (m_fdiv.valid)
    {
        if (m_fdiv.readyCycle <= m_cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else
        {
            fresh.fdiv = 1u;
            note(m_fdiv.readyCycle);
        }
    }

    fresh.efu = retirePipe(m_efu, pipe.efu, tracked, [&](ScalarPipelineEntry &entry)
    {
        if (entry.readyCycle > m_cycle)
        {
            note(entry.readyCycle);
            return false;
        }
        m_state.p = entry.value;
        entry = {};
        return true;
    });

    fresh.store = retirePipe(m_storePipeline, pipe.store, tracked, [&](PendingStore &store)
    {
        if (store.readyCycle > m_cycle)
        {
            note(store.readyCycle);
            return false;
        }
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            uint32_t lanes = store.laneMask & 0xFu;
            while (lanes != 0u)
            {
                const uint32_t component = 3u - nextSetBit(lanes);
                oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store = {};
        return true;
    });

    fresh.vf = retirePipe(m_vfWritePipeline, pipe.vf, tracked, [&](PendingVfWrite &write)
    {
        if (write.readyCycle > m_cycle)
        {
            note(write.readyCycle);
            return false;
        }
        uint32_t lanes = write.laneMask & 0xFu;
        while (lanes != 0u)
        {
            const uint32_t component = 3u - nextSetBit(lanes);
            if (m_vfLatestWrite[write.reg][component] == write.sequence)
                m_state.vf[write.reg][component] = write.value[component];
        }
        write = {};
        return true;
    });

    fresh.vi = retirePipe(m_viWritePipeline, pipe.vi, tracked, [&](PendingViWrite &write)
    {
        if (write.readyCycle > m_cycle)
        {
            note(write.readyCycle);
            return false;
        }
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write = {};
        return true;
    });

    fresh.acc = retirePipe(m_accWritePipeline, pipe.acc, tracked, [&](PendingAccWrite &write)
    {
        if (write.readyCycle > m_cycle)
        {
            note(write.readyCycle);
            return false;
        }
        uint32_t lanes = write.laneMask & 0xFu;
        while (lanes != 0u)
        {
            const uint32_t component = 3u - nextSetBit(lanes);
            if (m_accLatestWrite[component] == write.sequence)
                m_state.acc[component] = write.value[component];
        }
        write = {};
        return true;
    });

    fresh.owner = this;
    fresh.nextReady = nextReady;
    pipe = fresh;
}

// XGKICK walk fixes (cont.155, both default ON with kill switches):
// - PS2X_VU1_KICKFLG3 (=0 reverts): GIF-tag FLG=3 is NOT a reserved format — ps2tek documents
//   FLG 11b as "Disable (same as IMAGE)" and PCSX2 (GIF_FLG_IMAGE2) falls through to the IMAGE
//   path. Upstream aborted the kick AND killed the whole microprogram on every FLG=3 packet.
// - PS2X_VU1_KICKDROP_SOFT (=0 reverts): an oversized/garbage tag walk (the historical DROP
//   population) drops THE KICK (bounded-walk host safety) but must not kill the microprogram —
//   real hardware never stops the VU over GIF data content, and the old interpreter dropped the
//   kick and continued. Upstream's m_stopRequested abort turned every garbage-tag kick into a
//   whole dead geometry batch (~140 killed programs / 10 min in the LOTR level era).
extern "C" void ps2xGsSetGifPath(uint32_t); // gs_frontend.cpp per-path GIF resume selector

static const bool s_vu1KickFlg3Image = []
{ const char *e = std::getenv("PS2X_VU1_KICKFLG3"); return !(e && e[0] == '0'); }();
static const bool s_vu1KickDropSoft = []
{ const char *e = std::getenv("PS2X_VU1_KICKDROP_SOFT"); return !(e && e[0] == '0'); }();

// File-static on purpose: adding a member would touch include/runtime/ps2_vu1.h, which the
// generated code includes (a ~60 min rebuild vs ~3 min for this .cpp alone).
static void logXgkickDrop(uint32_t reason, uint32_t pc, unsigned long long cycle,
                          uint32_t srcAddr, const uint8_t *vuData, uint32_t dataSize)
{
    static unsigned long s_drops = 0;
    ++s_drops;
    if (s_drops <= 24u || (s_drops % 1024u) == 0u)
    {
        std::fprintf(stderr, "[VU1 kick-drop] n=%lu reason=0x%x pc=0x%x cycle=%llu src=0x%x\n",
                     s_drops, reason, pc, cycle, srcAddr);
        // Rule 12: dump the bytes the counter counts — the first two qwords at the kick source
        // identify the corruption class (zeros / file data / shifted stream / floats-as-tag).
        if (vuData && dataSize)
        {
            for (uint32_t q = 0; q < 2u; ++q)
            {
                uint32_t w[4];
                for (uint32_t i = 0; i < 4u; ++i)
                {
                    uint32_t v = 0;
                    for (uint32_t bshift = 0; bshift < 4u; ++bshift)
                        v |= static_cast<uint32_t>(vuData[(srcAddr + q * 16u + i * 4u + bshift) % dataSize]) << (8u * bshift);
                    w[i] = v;
                }
                std::fprintf(stderr, "    q%u: %08x %08x %08x %08x\n", q, w[0], w[1], w[2], w[3]);
            }
        }
    }
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++g_kickCalls;
    if (s_vu1KickEager && !g_xgkickEagerPass)
    {
        // Eager mode: the bytes are already in `packet`; only the PATH1 clock remains, at the same
        // rate as the streaming loop below (one quadword per 2 credits, issue cycle counted).
        ++m_xgkick.cycleCredit;
        while (m_xgkick.cycleCredit >= 2u && m_xgkick.streamedBytes < m_xgkick.totalBytes)
        {
            m_xgkick.cycleCredit -= 2u;
            m_xgkick.streamedBytes += 16u;
        }
        if (m_xgkick.eagerDone && m_xgkick.streamedBytes >= m_xgkick.totalBytes)
            finishXgkick();
        return;
    }
    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            if (s_vu1KickDropSoft)
            {
                m_xgkick.active = false;
                logXgkickDrop(0xFFFFFFFBu, m_state.pc, static_cast<unsigned long long>(m_cycle), m_xgkick.sourceAddress, m_activeVuData, m_activeVuDataSize);
                // One-shot per-kick-pc deep dump (first 6 distinct pcs): VI regs + microcode
                // around the kick + more source qwords, to classify the failing program's kick
                // as "output never written" vs "input garbage" (cont.155 RE material).
                {
                    static uint32_t s_dumpedPcs[6] = {0};
                    static int s_nDumped = 0;
                    bool seen = false;
                    for (int i = 0; i < s_nDumped; ++i)
                        if (s_dumpedPcs[i] == m_state.pc)
                            seen = true;
                    if (!seen && s_nDumped < 6)
                    {
                        s_dumpedPcs[s_nDumped++] = m_state.pc;
                        extern unsigned long g_vifQw10WriteGen;
                        std::fprintf(stderr, "[VU1 kick-dump] pc=0x%x src=0x%x w10gen=%lu vi:", m_state.pc, m_xgkick.sourceAddress, g_vifQw10WriteGen);
                        for (int i = 0; i < 16; ++i)
                            std::fprintf(stderr, " %04x", static_cast<uint16_t>(m_state.vi[i]));
                        std::fprintf(stderr, "\n");
                        if (m_cachedVuCode)
                        {
                            const uint32_t base = (m_state.pc >= 32u) ? (m_state.pc - 32u) & ~7u : 0u;
                            for (uint32_t off = 0; off < 64u && base + off + 8u <= m_cachedCodeSize; off += 8u)
                            {
                                uint32_t lo = 0, hi = 0;
                                std::memcpy(&lo, m_cachedVuCode + base + off, 4u);
                                std::memcpy(&hi, m_cachedVuCode + base + off + 4u, 4u);
                                std::fprintf(stderr, "    code 0x%04x: upper=%08x lower=%08x%s\n",
                                             base + off, hi, lo, (base + off == m_state.pc) ? "  <= kick" : "");
                            }
                        }
                    }
                }
                return;
            }
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        ++g_kickQwords;
        const uint32_t vuSize = m_activeVuDataSize;
        if (s_vu1FastKick && (vuSize & (vuSize - 1u)) == 0u)
        {
            const uint32_t base = (m_xgkick.sourceAddress + qwordOffset) & (vuSize - 1u);
            uint8_t *const dst = m_xgkick.packet.data() + qwordOffset;
            if (base + 16u <= vuSize)
            {
                std::memcpy(dst, m_activeVuData + base, 16u);
            }
            else
            {
                // Wraps the end of VU memory -- the byte loop's `% size` did this per byte.
                const uint32_t head = vuSize - base;
                std::memcpy(dst, m_activeVuData + base, head);
                std::memcpy(dst + head, m_activeVuData, 16u - head);
                ++g_kickWraps;
            }
            if (s_vu1KickVerify)
            {
                static bool once = kickCopySelfTest();
                (void)once;
                uint8_t ref[16];
                for (uint32_t i = 0; i < 16u; ++i)
                    ref[i] = m_activeVuData[(m_xgkick.sourceAddress + qwordOffset + i) %
                                            m_activeVuDataSize];
                ++g_kvChecked;
                if (std::memcmp(dst, ref, 16u) != 0 && ++g_kvBad <= 8ull)
                    std::fprintf(stderr,
                                 "[vu1:kickverify] MISMATCH #%llu src=0x%x off=%u base=%u size=%u\n",
                                 g_kvBad, m_xgkick.sourceAddress, qwordOffset,
                                 (m_xgkick.sourceAddress + qwordOffset) & (vuSize - 1u), vuSize);
            }
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u || (format == 3u && s_vu1KickFlg3Image))
                tagBytes += static_cast<uint64_t>(nloop) * 16u; // FLG=3 "Disable" == IMAGE (ps2tek; PCSX2 GIF_FLG_IMAGE2)
            else
            {
                if (s_vu1KickDropSoft)
                {
                    m_xgkick.active = false;
                    logXgkickDrop(0xFFFFFFF8u, m_state.pc, static_cast<unsigned long long>(m_cycle), m_xgkick.sourceAddress, m_activeVuData, m_activeVuDataSize);
                    return;
                }
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                if (s_vu1KickDropSoft)
                {
                    m_xgkick.active = false;
                    logXgkickDrop(0xFFFFFFFBu, m_state.pc, static_cast<unsigned long long>(m_cycle), m_xgkick.sourceAddress, m_activeVuData, m_activeVuDataSize);
                    return;
                }
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
            {
                if (g_xgkickEagerPass)
                {
                    m_xgkick.eagerDone = true; // complete in `packet`; the PATH1 clock releases it
                    return;
                }
                finishXgkick();
            }
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

// ★★ cont.230: PS2X_VU1_KICKSCAN=1 (default OFF) -- the "flung vertex" detector at XGKICK.
// The GS raster-capture bisect found the on-screen dark wedges are near-plane CLIPPER OUTPUT
// (triangle fans whose q is 1/near) of one character's triangles, i.e. a vertex of that mesh sits
// behind the camera; every other vertex in the frame obeys z = K*q. Whether that vertex is
// PRODUCED by VU1 (emulation) or ARRIVES bad in VU1's input (EE animation / VIF unpack) is decided
// at the kick: a flagged vertex is one whose q exceeds 0.1 (w < 10, nearer than any scene
// geometry) while its position lies more than 768 px outside the 2048-centred GS window. On the
// first hit the microprogram, the whole data memory and the packet are dumped for offline RE
// (PS2X_VU1_KICKSCAN_DIR, default "tmp"). Both GIF formats the VU can kick are parsed: PACKED
// (Q rides the ST qword, bits 64-95; A+D carries a 64-bit register) and REGLIST (64-bit regs, Q in
// RGBAQ bits 32-63). IMAGE tags are skipped by their nloop.
static const bool s_vu1KickScan = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN"); return e && e[0] && e[0] != '0'; }();
static const char *s_vu1KickScanDir = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_DIR"); return (e && e[0]) ? e : "tmp"; }();
// PS2X_VU1_KICKSCAN_MINCYCLE (default 450000000 = the level-era gate MSCALPEEK uses): the boot/menu
// era draws at a different XYOFFSET, where an off-window y is legitimate, and the first-hit dump
// must capture the LEVEL-era program.
static const uint64_t s_vu1KickScanMinCycle = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_MINCYCLE"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 450000000ull; }();
// cont.343 PS2X_VU1_KICKSCAN_AT=<px>:<py> (GS pixel coords, default OFF): dump the microprogram,
// data memory and packet at the first kick that emits a vertex within one pixel of that point --
// the raster capture names a vertex (e.g. the over-bright floor vertex), this fetches its producer.
static const int s_vu1KickScanAtX = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_AT"); return (e && e[0]) ? std::atoi(e) : -1; }();
static const int s_vu1KickScanAtY = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_AT"); const char *c = e ? std::strchr(e, ':') : nullptr; return c ? std::atoi(c + 1) : -1; }();
// cont.343b PS2X_VU1_KICKSCAN_ATR=<px> (default 1): the match radius around the AT point. A moving
// mesh (the character's shadow silhouette) does not put a vertex within one pixel of the same point
// two frames later; the static floor did, which is why the floor scan matched and the shadow scan
// found nothing.
static const int s_vu1KickScanAtR = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_ATR"); const int v = (e && e[0]) ? std::atoi(e) : 1; return v < 0 ? 0 : v; }();
// cont.343 PS2X_VU1_KICKSCAN_FLIP=<n> [+ PS2X_VU1_KICKSCAN_FLIPW=<w>, default 2] (default OFF): only
// scan kicks inside guest display flips [n, n+w), so the dump is aimed at the raster capture's frame
// (under PS2X_VIRTUAL_TIME=1 the CPU arm is byte-identical run to run). Without it the first hit on a
// screen point came from the level's first second, a different draw altogether. Inside the window
// EVERY flagged kick is printed and dumped with an index suffix (up to 64), not only the first.
static const unsigned long long s_vu1KickScanFlip = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_FLIP"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
static const unsigned long long s_vu1KickScanFlipW = []
{ const char *e = std::getenv("PS2X_VU1_KICKSCAN_FLIPW"); unsigned long long v = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 2ull; return v ? v : 2ull; }();
unsigned long long gs2CurrentFlip(); // gs_cpu_backend.cpp (cont.331q)
static inline bool vu1KickScanFlipOk()
{
    if (s_vu1KickScanFlip == 0ull)
        return true;
    const unsigned long long f = gs2CurrentFlip();
    return f >= s_vu1KickScanFlip && f < s_vu1KickScanFlip + s_vu1KickScanFlipW;
}
namespace
{
    const uint8_t *g_ksCode = nullptr;
    uint32_t g_ksCodeSize = 0, g_ksStartPC = 0, g_ksTop = 0, g_ksItop = 0;
    unsigned long long g_ksExecSeq = 0, g_ksFlagged = 0, g_ksKicks = 0, g_ksVertices = 0;

    void vu1KickScanFlag(uint32_t kickAddr, uint32_t tagIdx, uint32_t loop, uint32_t reg, float q,
                         uint32_t px, uint32_t py, uint32_t z, const uint8_t *pkt, uint32_t bytes,
                         const uint8_t *vuData, uint32_t dataSize, uint64_t cycle, uint32_t pc)
    {
        const unsigned long long n = ++g_ksFlagged;
        const bool windowed = s_vu1KickScanFlip != 0ull;
        if (n <= 16ull || windowed || (n % 1000ull) == 0ull)
            std::fprintf(stderr,
                         "[vu1:kickscan] #%llu FLUNG px=(%u,%u) z=%u q=%.4f | tag=%u loop=%u reg=0x%x"
                         " kick=0x%x bytes=%u | startPC=0x%x top=0x%x itop=0x%x exec=%llu pc=0x%x"
                         " cycle=%llu flip=%llu | kicks=%llu verts=%llu\n",
                         n, px, py, z, static_cast<double>(q), tagIdx, loop, reg, kickAddr, bytes,
                         g_ksStartPC, g_ksTop, g_ksItop, g_ksExecSeq, pc,
                         static_cast<unsigned long long>(cycle), gs2CurrentFlip(), g_ksKicks, g_ksVertices);
        if (windowed ? (n > 64ull) : (n != 1ull))
            return;
        char path[600];
        auto dump = [&](const char *name, const void *data, size_t size)
        {
            if (windowed)
                std::snprintf(path, sizeof(path), "%s/kickscan_%02llu_%s.bin", s_vu1KickScanDir, n, name);
            else
                std::snprintf(path, sizeof(path), "%s/kickscan_%s.bin", s_vu1KickScanDir, name);
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fwrite(data, 1, size, f);
                std::fclose(f);
                std::fprintf(stderr, "[vu1:kickscan] wrote %s (%zu bytes)\n", path, size);
            }
            else
                std::fprintf(stderr, "[vu1:kickscan] cannot write %s\n", path);
        };
        if (g_ksCode != nullptr && g_ksCodeSize != 0u)
            dump("code", g_ksCode, g_ksCodeSize);
        if (vuData != nullptr && dataSize != 0u)
            dump("data", vuData, dataSize);
        dump("packet", pkt, bytes);
    }

    void vu1KickScan(const uint8_t *pkt, uint32_t bytes, uint32_t kickAddr,
                     const uint8_t *vuData, uint32_t dataSize, uint64_t cycle, uint32_t pc)
    {
        ++g_ksKicks;
        uint32_t off = 0, tagIdx = 0;
        float q = 1.0f;
        auto flagXyz = [&](uint32_t areg, uint32_t x, uint32_t y, uint32_t z, uint32_t loop)
        {
            ++g_ksVertices;
            const uint32_t px = x >> 4, py = y >> 4;
            if (s_vu1KickScanAtX >= 0)
            {
                if (std::abs((int)px - s_vu1KickScanAtX) <= s_vu1KickScanAtR && std::abs((int)py - s_vu1KickScanAtY) <= s_vu1KickScanAtR)
                    vu1KickScanFlag(kickAddr, tagIdx, loop, areg, q, px, py, z, pkt, bytes, vuData,
                                    dataSize, cycle, pc);
                return;
            }
            if (q > 0.1f && (px < 1024u || px > 3072u || py < 1024u || py > 3072u))
                vu1KickScanFlag(kickAddr, tagIdx, loop, areg, q, px, py, z, pkt, bytes, vuData,
                                dataSize, cycle, pc);
        };
        while (off + 16u <= bytes)
        {
            uint64_t lo, hi;
            std::memcpy(&lo, pkt + off, 8);
            std::memcpy(&hi, pkt + off + 8, 8);
            off += 16u;
            ++tagIdx;
            const uint32_t nloop = static_cast<uint32_t>(lo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((lo >> 58) & 3u);
            uint32_t nreg = static_cast<uint32_t>((lo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg >= 2u)
            {
                off += nloop * 16u; // IMAGE / DISABLED: nloop qwords of raw data
                continue;
            }
            if (flg == 0u)
            {
                for (uint32_t l = 0; l < nloop && off + 16u <= bytes; ++l)
                    for (uint32_t r = 0; r < nreg && off + 16u <= bytes; ++r)
                    {
                        const uint32_t reg = static_cast<uint32_t>((hi >> (4u * r)) & 0xFu);
                        uint64_t dlo, dhi;
                        std::memcpy(&dlo, pkt + off, 8);
                        std::memcpy(&dhi, pkt + off + 8, 8);
                        off += 16u;
                        if (reg == 0x2u)
                        {
                            const uint32_t qw = static_cast<uint32_t>(dhi & 0xFFFFFFFFu);
                            std::memcpy(&q, &qw, 4);
                            continue;
                        }
                        if (reg == 0xEu)
                        {
                            const uint32_t areg = static_cast<uint32_t>(dhi & 0xFFu);
                            if (areg == 0x1u)
                            {
                                const uint32_t qw = static_cast<uint32_t>(dlo >> 32);
                                std::memcpy(&q, &qw, 4);
                            }
                            else if (areg == 4u || areg == 5u || areg == 0xCu || areg == 0xDu)
                            {
                                const uint32_t z = (areg == 4u || areg == 0xCu)
                                    ? static_cast<uint32_t>((dlo >> 32) & 0xFFFFFFu)
                                    : static_cast<uint32_t>(dlo >> 32);
                                flagXyz(areg, static_cast<uint32_t>(dlo & 0xFFFFu),
                                        static_cast<uint32_t>((dlo >> 16) & 0xFFFFu), z, l);
                            }
                            continue;
                        }
                        if (reg == 4u || reg == 5u || reg == 0xCu || reg == 0xDu)
                        {
                            const uint32_t z = (reg == 4u || reg == 0xCu)
                                ? static_cast<uint32_t>((dhi >> 4) & 0xFFFFFFu)
                                : static_cast<uint32_t>(dhi & 0xFFFFFFFFu);
                            flagXyz(reg, static_cast<uint32_t>(dlo & 0xFFFFu),
                                    static_cast<uint32_t>((dlo >> 32) & 0xFFFFu), z, l);
                        }
                    }
            }
            else
            {
                const uint32_t total = nloop * nreg;
                const uint32_t base = off;
                for (uint32_t i = 0; i < total && base + i * 8u + 8u <= bytes; ++i)
                {
                    const uint32_t reg = static_cast<uint32_t>((hi >> (4u * (i % nreg))) & 0xFu);
                    uint64_t v;
                    std::memcpy(&v, pkt + base + i * 8u, 8);
                    if (reg == 0x1u)
                    {
                        const uint32_t qw = static_cast<uint32_t>(v >> 32);
                        std::memcpy(&q, &qw, 4);
                    }
                    else if (reg == 4u || reg == 5u || reg == 0xCu || reg == 0xDu)
                    {
                        const uint32_t z = (reg == 4u || reg == 0xCu)
                            ? static_cast<uint32_t>((v >> 32) & 0xFFFFFFu)
                            : static_cast<uint32_t>(v >> 32);
                        flagXyz(reg, static_cast<uint32_t>(v & 0xFFFFu),
                                static_cast<uint32_t>((v >> 16) & 0xFFFFu), z, i / nreg);
                    }
                }
                off = base + ((total + 1u) / 2u) * 16u;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    // ★ cont.222: leaving `packet` uninitialised at reset is safe ONLY while every submitted byte
    // has been written -- i.e. totalBytes <= copiedBytes. That is the exact invariant the
    // optimisation rests on, so check it rather than argue it (PS2X_VU1_KICKVERIFY=1).
    if (s_vu1KickVerify)
    {
        ++g_kvSubmits;
        if (m_xgkick.totalBytes > m_xgkick.copiedBytes && ++g_kvUnwritten <= 8ull)
            std::fprintf(stderr,
                         "[vu1:kickverify] UNWRITTEN SUBMIT #%llu totalBytes=%u copiedBytes=%u\n",
                         g_kvUnwritten, m_xgkick.totalBytes, m_xgkick.copiedBytes);
    }

    if (s_vu1KickEager && s_vu1KickVerify && m_xgkick.eagerDone && m_activeVuData != nullptr && m_activeVuDataSize != 0u)
    {
        // Differential: would progressive streaming have read different bytes than the eager copy?
        ++g_kvEagerChecked;
        bool diff = false;
        for (uint32_t i = 0; i < m_xgkick.totalBytes && !diff; ++i)
            diff = m_xgkick.packet[i] != m_activeVuData[(m_xgkick.sourceAddress + i) % m_activeVuDataSize];
        if (diff && ++g_kvEagerDiff <= 8ull)
            std::fprintf(stderr, "[vu1:kickverify] EAGER DIFF #%llu src=0x%x bytes=%u pc=0x%x cycle=%llu\n",
                         g_kvEagerDiff, m_xgkick.sourceAddress, m_xgkick.totalBytes, m_state.pc,
                         static_cast<unsigned long long>(m_cycle));
    }
    if (s_vu1KickScan && m_unit == Unit::VU1 && m_cycle >= s_vu1KickScanMinCycle && vu1KickScanFlipOk())
        vu1KickScan(m_xgkick.packet.data(), m_xgkick.totalBytes, m_xgkick.sourceAddress,
                    m_activeVuData, m_activeVuDataSize, m_cycle, m_state.pc);
    if (m_activeMemory)
        {
        static const unsigned long long s_kickLog = []{ const char *e = std::getenv("PS2X_MICROVU_KICKLOG"); return e ? std::strtoull(e, nullptr, 10) : 0ull; }();
        static unsigned long long s_kickN = 0;
        if (s_kickLog && ++s_kickN <= s_kickLog * 2u)
            std::fprintf(stderr, "[interp:kick] #%llu addr=0x%x totalBytes=%u dry=%d hash=%016llx\n", s_kickN, m_xgkick.sourceAddress, m_xgkick.totalBytes, g_vu1DryKicks ? 1 : 0,
                         (unsigned long long)ps2x_microvu::fnv1a(m_xgkick.packet.data(), m_xgkick.totalBytes));
        if (g_vu1DryKicks) // PROGVERIFY shadow run: record, never submit
            g_vu1DryKickList.push_back({m_xgkick.totalBytes, ps2x_microvu::fnv1a(m_xgkick.packet.data(), m_xgkick.totalBytes)});
        else
            m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    }
    else if (m_activeGs)
    {
        // ★ cont.230: this fallback was NOT inside the else-branch -- `else if (m_activeGs)` guarded
        // only ps2xGsSetGifPath(1u), and the processGIFPacket() below ran UNCONDITIONALLY after the
        // arbiter submit, so every XGKICK packet was parsed and drawn TWICE (found by the microVU
        // PROGVERIFY oracle: the interpreter's kick list held each kick twice; per-frame primitive
        // counts were double microVU's for identical kick bytes). Braced.
        ps2xGsSetGifPath(1u);
        if (g_vu1DryKicks)
            g_vu1DryKickList.push_back({m_xgkick.totalBytes, ps2x_microvu::fnv1a(m_xgkick.packet.data(), m_xgkick.totalBytes)});
        else
            m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    }
    m_xgkick.active = false;
}

// PS2X_VU1_KICKSTALL (default ON; =0 restores the clobber): real VU1 hardware STALLS on a
// second XGKICK until PATH1 finishes streaming the previous kick (PCSX2 models this stall).
// Upstream reset m_xgkick unconditionally, DESTROYING any in-flight kick — its packet was
// silently never submitted to the GS. PATH1 streams at ~8 bytes/cycle, so a 1KB packet needs
// ~128 cycles; programs issue batch kicks a few dozen cycles apart, which lost nearly every
// packet except each program's last — the bulk of the level scene (cont.155).
static const bool s_vu1KickStall = []
{ const char *e = std::getenv("PS2X_VU1_KICKSTALL"); return !(e && e[0] == '0'); }();

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    if (s_vu1KickStall)
    {
        // Stall until the previous kick completes (progressXgkick terminates: it either
        // finishes at EOP or soft-drops at the 64KB walk bound).
        while (m_xgkick.active)
            advanceOneCycle();
    }

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    // ★★ cont.230: the second copy of cont.222's hidden 64 KB clear. `m_xgkick = {}` zeroes the
    // XgkickPipeline INCLUDING its 0x10000-byte `packet`, on EVERY XGKICK (581k per run ≈ 38 GB of
    // memset; 3.3% of the EE thread's samples in the cont.230 profile, all under startXgkick). Same
    // argument as resetScheduler: packet bytes are written before they are read (KICKVERIFY checks
    // totalBytes <= copiedBytes at every submit), so only the control fields need resetting.
    if (s_vu1FastKickReset)
    {
        m_xgkick.sourceAddress = 0;
        m_xgkick.totalBytes = 0;
        m_xgkick.copiedBytes = 0;
        m_xgkick.currentTagEnd = 0;
        m_xgkick.cycleCredit = 0;
        m_xgkick.issueCycle = 0;
        m_xgkick.currentTagEop = false;
        m_xgkick.eagerDone = false;
        m_xgkick.streamedBytes = 0;
    }
    else
    {
        m_xgkick = {};
    }
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
    if (s_vu1KickEager)
    {
        // Eager pass: enough credits to stream 64 KB, through the unchanged copy + tag-walk loop; at
        // EOP it marks eagerDone instead of submitting. Then restore the PATH1 clock to "just issued".
        m_xgkick.cycleCredit = 2u * (XgkickPipeline::kBufferSize / 16u) + 2u;
        g_xgkickEagerPass = true;
        progressXgkick();
        g_xgkickEagerPass = false;
        m_xgkick.cycleCredit = 1u;
        m_xgkick.streamedBytes = 0u;
    }
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    // O(1) while the commit tracking (PS2X_VU1_COMMITSKIP) owns this instance: the
    // counters are exact, so total()==0 proves every array is empty.
    const PipeTrack &pipe = pipeFor(this);
    if (s_vu1CommitSkip && pipe.owner == this)
        return pipe.total() != 0u;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    for (const FlagPipelineEntry &entry : m_flagPipeline)
        if (entry.valid)
            return true;
    for (const PendingStore &store : m_storePipeline)
        if (store.valid)
            return true;
    for (const PendingVfWrite &write : m_vfWritePipeline)
        if (write.valid)
            return true;
    for (const PendingViWrite &write : m_viWritePipeline)
        if (write.valid)
            return true;
    for (const PendingAccWrite &write : m_accWritePipeline)
        if (write.valid)
            return true;
    return false;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    if (s_vu1NoSched)
        return m_cycle; // ablation: no instruction ever stalls
    uint64_t ready = m_cycle;
    const InstructionUsage *usages[2] = {
        &decoded.upperUsage,
        &decoded.lowerUsage};
    for (const InstructionUsage *usage : usages)
    {
        if (!usage)
            continue;
        if (s_vu1FastScan)
        {
            for (uint32_t index = 0; index < usage->vfReadCount; ++index)
            {
                const VfAccess &access = usage->vfRead[index];
                const uint64_t *const lane = m_vfReady[access.reg].data();
                uint32_t lanes = access.lanes & 0xFu;
                while (lanes != 0u)
                    ready = std::max(ready, lane[3u - nextSetBit(lanes)]);
            }
            uint32_t viRead = static_cast<uint32_t>(usage->viRead) & kViRegMask;
            while (viRead != 0u)
                ready = std::max(ready, m_viReady[nextSetBit(viRead)]);
            uint32_t accRead = static_cast<uint32_t>(usage->accRead) & 0xFu;
            while (accRead != 0u)
                ready = std::max(ready, m_accReady[3u - nextSetBit(accRead)]);
            continue;
        }
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((access.lanes & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_vfReady[access.reg][component]);
            }
        }
        for (uint32_t reg = 1; reg < m_viReady.size(); ++reg)
        {
            if ((usage->viRead & (1u << reg)) != 0u)
                ready = std::max(ready, m_viReady[reg]);
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((usage->accRead & laneForComponent(component)) != 0u)
                ready = std::max(ready, m_accReady[component]);
        }
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    if (s_vu1NoSched)
        return; // ablation: nothing reads m_vfReady/m_viReady/m_accReady when nothing stalls
    if (!s_vu1FastScan)
    {
        // Legacy full-scan path, kept verbatim for A/B (PS2X_VU1_FASTSCAN=0).
        const VfAccess legacyLower = decoded.lowerUsage.vfWrite;
        if (legacyLower.reg != 0u && decoded.suppressedLowerVf != legacyLower.reg)
        {
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((legacyLower.lanes & laneForComponent(component)) != 0u)
                    m_vfReady[legacyLower.reg][component] = m_cycle + latency;
            }
        }
        const VfAccess legacyUpper = decoded.upperUsage.vfWrite;
        if (legacyUpper.reg != 0u)
        {
            const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                         ? decoded.upperUsage.vfLatency
                                         : decoded.upperUsage.latency;
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((legacyUpper.lanes & laneForComponent(component)) != 0u)
                    m_vfReady[legacyUpper.reg][component] = m_cycle + latency;
            }
        }
        for (uint32_t reg = 1; reg < m_viReady.size(); ++reg)
        {
            if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
                m_viReady[reg] = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
                m_accReady[component] = m_cycle + kAccForwardLatency;
        }
        return;
    }
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        uint64_t *const lane = m_vfReady[lowerWrite.reg].data();
        uint32_t lanes = lowerWrite.lanes & 0xFu;
        while (lanes != 0u)
            lane[3u - nextSetBit(lanes)] = m_cycle + latency;
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        uint64_t *const lane = m_vfReady[upperWrite.reg].data();
        uint32_t lanes = upperWrite.lanes & 0xFu;
        while (lanes != 0u)
            lane[3u - nextSetBit(lanes)] = m_cycle + latency;
    }

    uint32_t viWrite = static_cast<uint32_t>(decoded.lowerUsage.viWrite) & kViRegMask;
    if (viWrite != 0u)
    {
        const uint64_t viReady = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
        while (viWrite != 0u)
            m_viReady[nextSetBit(viWrite)] = viReady;
    }
    uint32_t accWrite = static_cast<uint32_t>(decoded.upperUsage.accWrite) & 0xFu;
    while (accWrite != 0u)
        m_accReady[3u - nextSetBit(accWrite)] = m_cycle + kAccForwardLatency;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    // cont.177 lazy flags: one pass, no extra walk — the cache rebuild already visits every pair,
    // and it is re-run whenever the code generation changes, so this is the natural place for the
    // program-level "does anything read MAC/status" question. Conservative by construction: a
    // reader ANYWHERE in the buffer disables the fast path for the whole program (a branch can
    // reach anywhere, so per-block reasoning is not available to an interpreter).
    bool readsMacOrStatus = false;
    bool readsStatus = false;
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        const DecodedInstructionPair &decoded = (m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u));
        // With the I bit set the lower word is a float immediate, NOT an instruction.
        if (!decoded.iBit)
        {
            const unsigned fr = lowerFlagReads(decoded.lower);
            if (fr != 0u)
                readsMacOrStatus = true;
            if ((fr & kFlagReadStatus) != 0u)
                readsStatus = true;
        }
    }
    if (m_unit == Unit::VU0)
    {
        ++g_vu0ProgsScanned;
        if (!readsMacOrStatus && pairCount == codeSize / 8u)
            ++g_vu0ProgsClean;
        if (readsStatus)
            ++g_vu0ProgsStatus;
    }
    {
        // cont.214: record for BOTH units now that each has its own slot.
        LazyScan &lz = s_lazyScan[m_unit == Unit::VU1 ? 1u : 0u];
        lz.owner = this;
        lz.code = vuCode;
        lz.memory = memory;
        lz.codeSize = codeSize;
        lz.generation = generation;
        lz.complete = (pairCount == codeSize / 8u); // else tail pairs went unscanned
        lz.readsMacOrStatus = readsMacOrStatus;
        lz.readsStatus = readsStatus;
        if (m_unit == Unit::VU1)
        {
            ++g_lazyProgsScanned;
            if (!readsMacOrStatus && lz.complete)
                ++g_lazyProgsClean;
        }
    }

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
        return decodeInstructionPair(vuCode, pc);

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return decodeInstructionPair(vuCode, pc);

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    {
        // cont.215: name the blocks that ran just before, newest first.
        static unsigned s_dumped = 0;
        if (s_vuLastBlk && s_dumped < 8u)
        {
            ++s_dumped;
            const unsigned u = (m_unit == Unit::VU1) ? 1u : 0u;
            std::fprintf(stderr, "[vu%u:lastblk] pc=0x%x <-", u, m_state.pc);
            for (unsigned b = 1; b <= 4u && b <= g_lastBlkN[u]; ++b)
            {
                const LastBlk &lb = g_lastBlk[u][(g_lastBlkN[u] - b) & 3u];
                std::fprintf(stderr, " [in=0x%x pairs=%u br=%u out=0x%x cy=%u]",
                             lb.entryPc, lb.pairs, lb.endsBranch, lb.outPc, lb.cycles);
            }
            std::fprintf(stderr, "\n");
            // cont.215: is the microcode the block was planned from still the microcode that is
            // there now? Dump the first pairs plus every generation counter in play.
            if (m_cachedVuCode != nullptr)
            {
                std::fprintf(stderr, "[vu%u:code] cacheGen=%llu storeGen=n/a memGen=%llu |", u,
                             (unsigned long long)m_cachedCodeGeneration,
                             (unsigned long long)(m_cachedMemory
                                 ? (m_unit == Unit::VU1
                                        ? m_cachedMemory->getVU1CodeGeneration()
                                        : m_cachedMemory->getVU0CodeGeneration())
                                 : 0ull));
                for (uint32_t w = 0; w < 20u; ++w)
                {
                    uint32_t word = 0;
                    std::memcpy(&word, m_cachedVuCode + w * 4u, 4);
                    if ((w & 1u) == 0u)
                        std::fprintf(stderr, " %04x:", w * 4u);
                    std::fprintf(stderr, "%08x%s", word, (w & 1u) ? "" : "/");
                }
                std::fprintf(stderr, "\n");
            }
        }
    }
    m_stopRequested = true;
}

// ★★ cont.216: EVERY "pipeline array is full" site used to funnel through
// reportReservedInstruction with a magic 0xFFFFFFF* value in the `instruction` field. That prints
// as `[VU0 reserved upper] ... instruction=0xffffffff`, which reads exactly like a garbage DECODE --
// and cont.215 duly spent a cycle hunting a stale block store / bad PC for what was an 8-slot flag
// array overflowing. The condition is a capacity limit of our own model, so say so. Same halt
// semantics as before (m_stopRequested), so behaviour is unchanged; only the diagnosis is honest.
void VU1Interpreter::reportPipelineFull(const char *which)
{
    static unsigned long s_reported = 0;
    if (++s_reported <= 16ul)
        RUNTIME_ERROR("[VU" << (m_unit == Unit::VU1 ? "1" : "0") << " pipeline FULL] " << which
                            << " cycle=" << m_cycle << " pc=0x" << std::hex << m_state.pc
                            << std::dec
                            << " -- every slot valid; MODEL CAPACITY overflow, not a decode error");
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    if (s_vu1KickScan && m_unit == Unit::VU1)
    {
        // cont.230 kickscan: remember what this execution runs, so a flagged kick can name it.
        g_ksCode = vuCode; g_ksCodeSize = codeSize; g_ksStartPC = startPC; g_ksTop = top; g_ksItop = itop;
        ++g_ksExecSeq;
    }
    // Level-era MSCAL peek (part of PS2X_VIF_CENSUS via env; cont.155): dump what the program
    // RECEIVES — the first qwords at TOP (the VIF-built input buffer) — for the first few
    // level-era executions per distinct startPC.
    {
        static const bool s_peek = []
        { const char *e = std::getenv("PS2X_VU1_MSCALPEEK"); return e && e[0] && e[0] != '0'; }();
        if (s_peek && m_cycle > 450000000ull && vuData)
        {
            static uint32_t s_pcs[8] = {0};
            static int s_n = 0;
            bool seen = false;
            for (int i = 0; i < s_n; ++i)
                if (s_pcs[i] == startPC)
                    seen = true;
            if (!seen && s_n < 8)
            {
                s_pcs[s_n++] = startPC;
                std::fprintf(stderr, "[VU1 mscal-peek] startPC=0x%x top=0x%x itop=0x%x cycle=%llu\n",
                             startPC, top, itop, static_cast<unsigned long long>(m_cycle));
            }
            // Full snapshot for offline RE (PS2X_VU1_SNAPSHOT=<dir>): rolling — rewritten every
            // 25000 executions, so the last capture before the run ends is the current era's.
            {
                static unsigned long s_exec = 0;
                ++s_exec;
                const char *snapDir = std::getenv("PS2X_VU1_SNAPSHOT");
                if (snapDir && vuCode && (s_exec % 2000u) == 1u)
                {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/vu1_code.bin", snapDir);
                    if (FILE *f = std::fopen(path, "wb"))
                    {
                        std::fwrite(vuCode, 1, codeSize, f);
                        std::fclose(f);
                    }
                    std::snprintf(path, sizeof(path), "%s/vu1_data.bin", snapDir);
                    if (FILE *f = std::fopen(path, "wb"))
                    {
                        std::fwrite(vuData, 1, dataSize, f);
                        std::fclose(f);
                    }
                    std::fprintf(stderr, "[VU1 snapshot] wrote %s/vu1_{code,data}.bin (startPC=0x%x exec=%lu)\n",
                                 snapDir, startPC, s_exec);
                }
                for (uint32_t q = 0; q < 4u; ++q)
                {
                    const uint32_t base = ((top & 0x3FFu) * 16u + q * 16u) % dataSize;
                    uint32_t w[4];
                    std::memcpy(w, vuData + base, 16u);
                    std::fprintf(stderr, "    top+%u: %08x %08x %08x %08x\n", q, w[0], w[1], w[2], w[3]);
                }
            }
        }
    }
    // ★ rotk row 255 PS2X_VU0_SKIPRESET (default ON; `=0` = always reset): the scheduler prep is the
    // INTERPRETER's (pipeline arrays, XGKICK, clip/branch backups, halt requests). When VU0 runs on microVU
    // (below) nothing reads it -- the branch returns without touching it, and its PROGVERIFY shadow run
    // calls resetScheduler() itself before the interpreter runs. microVU never takes pipe-tracking
    // ownership, so the "pipes provably empty" skip never applied and every VCALLMS paid the full clear:
    // 5.5% of the EE thread on the 60-fps fight (perf, GameThread).
    static const bool s_vu0SkipReset = []
    { const char *e = std::getenv("PS2X_VU0_SKIPRESET"); return !(e && e[0] == '0'); }();
    if (!(s_vu0SkipReset && s_vu0MicroVu && m_unit == Unit::VU0 && memory))
        resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    const int microVuResumeFlag = 0;
    // ★ cont.250 PS2X_VU0_MICROVU: the same recompiler for VU0. Simpler than the VU1 branch --
    // VU0 has no XGKICK (gated to Unit::VU1), no VIF1 TOP/ITOP, and reaches here only via
    // executeVU0Microprogram (MSCAL/VCALLMS), never resume(), so there is no MSCNT path.
    if (s_vu0MicroVu && m_unit == Unit::VU0 && memory)
    {
        ps2x_microvu::initVU0(vuCode, vuData);
        ps2x_microvu::setMemory(memory);
        static bool s_announced0 = false;
        if (!s_announced0) { s_announced0 = true; std::fprintf(stderr, "[vu0:microvu] active (PS2X_VU0_MICROVU=1)%s\n", s_vu0ProgVerify ? " + PROGVERIFY" : ""); }
        // PROGVERIFY: snapshot the input state and VU0 data memory before microVU touches them.
        static std::vector<uint8_t> s_pv0Mem;
        VU1State pv0In{};
        if (s_vu0ProgVerify)
        {
            pv0In = m_state;
            s_pv0Mem.assign(vuData, vuData + dataSize);
        }
        const auto t0v0 = std::chrono::steady_clock::now();
        const uint64_t used0 = ps2x_microvu::runUnit(0, m_state, startPC, false, maxCycles);
        m_cycle += used0;
        ++g_microVu0Runs;
        if (s_vu0ProgVerify)
        {
            const VU1State outM = m_state;
            // Shadow run on the block JIT / interpreter: same input, COPIED memory. The shadow's
            // cycles are not the guest's, and microVU's result stays authoritative.
            m_state = pv0In;
            resetScheduler();
            const uint64_t savedCycle = m_cycle;
            run(vuCode, codeSize, s_pv0Mem.data(), dataSize, gs, memory, maxCycles);
            m_cycle = savedCycle;
            const VU1State outI = m_state;
            m_state = outM;
            ++g_pv0Runs;
            // Every class evaluated (no short-circuit) so one run names all the ways it differs.
            bool bad = false; std::string why;
            auto flag = [&](const char *w, unsigned long long &ctr) { bad = true; ++ctr; if (!why.empty()) why += ","; why += w; };
            for (int r = 1; r < 32; ++r)
                if (std::memcmp(outM.vf[r], outI.vf[r], 16) != 0) { flag(("vf" + std::to_string(r)).c_str(), g_pv0BadVf); break; }
            for (int r = 1; r < 16; ++r)
                if (outM.vi[r] != outI.vi[r]) { flag(("vi" + std::to_string(r)).c_str(), g_pv0BadVi); break; }
            if (std::memcmp(outM.acc, outI.acc, 16) != 0) flag("acc", g_pv0BadVf);
            if (((outM.status ^ outI.status) & 0x3Fu) != 0u || outM.mac != outI.mac || outM.clip != outI.clip) flag("flags", g_pv0BadFlags);
            if (((outM.status ^ outI.status) & 0xFC0u) != 0u) ++g_pv0Sticky;
            if (std::memcmp(&outM.q, &outI.q, 4) != 0 || std::memcmp(&outM.p, &outI.p, 4) != 0 ||
                std::memcmp(&outM.i, &outI.i, 4) != 0 || outM.r != outI.r) flag("q/p/i/r", g_pv0BadQP);
            if (outM.pc != outI.pc) flag("pc", g_pv0BadPc);
            if (std::memcmp(vuData, s_pv0Mem.data(), dataSize) != 0) flag("vumem", g_pv0BadMem);
            if (outM.cycles != outI.cycles) ++g_pv0CycleDiff;
            static std::map<std::string, unsigned> s_pv0Printed;
            std::string firstClass = why.substr(0, why.find(','));
            for (char &ch : firstClass) if (std::isdigit(static_cast<unsigned char>(ch))) ch = '#';
            if (bad) ++g_pv0Bad;
            if (bad && ++s_pv0Printed[firstClass] <= 8u)
            {
                std::fprintf(stderr, "[vu0:progverify] MISMATCH #%llu run=%llu startPC=0x%x %s | mvu: pc=0x%x st=%08x mac=%08x clip=%08x q=%g cyc=%llu | blk: pc=0x%x st=%08x mac=%08x clip=%08x q=%g cyc=%llu\n",
                             g_pv0Bad, g_pv0Runs, startPC, why.c_str(),
                             outM.pc, outM.status, outM.mac, outM.clip, outM.q, (unsigned long long)outM.cycles,
                             outI.pc, outI.status, outI.mac, outI.clip, outI.q, (unsigned long long)outI.cycles);
                if (why.rfind("vf", 0) == 0 && std::isdigit(static_cast<unsigned char>(why[2])))
                {
                    const int r = std::atoi(why.c_str() + 2);
                    std::fprintf(stderr, "    mvu vf%d=%g %g %g %g | blk vf%d=%g %g %g %g\n",
                                 r, outM.vf[r][0], outM.vf[r][1], outM.vf[r][2], outM.vf[r][3],
                                 r, outI.vf[r][0], outI.vf[r][1], outI.vf[r][2], outI.vf[r][3]);
                }
            }
            if ((g_pv0Runs & 4095u) == 0u)
                std::fprintf(stderr, "[vu0:progverify] runs=%llu bad=%llu (vf=%llu vi=%llu flags=%llu qp=%llu pc=%llu mem=%llu) sticky=%llu cycleDiff=%llu\n",
                             g_pv0Runs, g_pv0Bad, g_pv0BadVf, g_pv0BadVi, g_pv0BadFlags, g_pv0BadQP, g_pv0BadPc, g_pv0BadMem, g_pv0Sticky, g_pv0CycleDiff);
        }
        if (s_vu1Perf)
        {
            g_microVu0Ns += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0v0).count());
            if ((g_microVu0Runs & 4095u) == 0u)
                std::fprintf(stderr, "[vu0:microvu] runs=%llu ns=%llu (%.2f ns/run)\n",
                             g_microVu0Runs, g_microVu0Ns, double(g_microVu0Ns) / double(g_microVu0Runs));
        }
        return;
    }
    // ★ cont.230 PS2X_VU1_MICROVU=1 (default OFF until verified): run the program on the ported
    // PCSX2 microVU recompiler instead of this interpreter (docs/vu1-program-compiler.md).
    if (s_vu1MicroVu && m_unit == Unit::VU1 && memory)
    {
        const bool mvuResume = (microVuResumeFlag != 0);
        m_activeVuData = vuData;
        m_activeVuDataSize = dataSize;
        m_activeGs = &gs;
        m_activeMemory = memory;
        ps2x_microvu::init(vuCode, vuData);
        ps2x_microvu::setMemory(memory);
        static bool s_announced = false;
        if (!s_announced) { s_announced = true; std::fprintf(stderr, "[vu1:microvu] active (PS2X_VU1_MICROVU=1)%s\n", s_vu1ProgVerify ? " + PROGVERIFY" : ""); }
        // PROGVERIFY: snapshot the input state and the VU data memory before microVU touches them.
        static std::vector<uint8_t> s_pvMem;
        VU1State pvIn{};
        if (s_vu1ProgVerify)
        {
            pvIn = m_state;
            s_pvMem.assign(vuData, vuData + dataSize);
            ps2x_microvu::setKickRecording(true);
        }
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t used = ps2x_microvu::run(m_state, mvuResume ? m_state.pc : startPC, mvuResume, maxCycles);
        m_cycle += used;
        if (s_vu1ProgVerify)
        {
            ps2x_microvu::setKickRecording(false);
            size_t mk = 0;
            const ps2x_microvu::KickRecord *mkr = ps2x_microvu::kickRecords(&mk);
            std::vector<ps2x_microvu::KickRecord> mvuKicks(mkr, mkr + mk);
            const VU1State outM = m_state;
            // Shadow run on the interpreter: same input, copied memory, dry kicks.
            m_state = pvIn;
            // The shadow always starts from drained pipelines (a program end drains them on the
            // hardware and in both implementations) and, after the run, completes any XGKICK still
            // streaming so its kick list covers the whole program, as microVU's does.
            resetScheduler();
            g_vu1DryKicks = true;
            g_vu1DryKickList.clear();
            const uint64_t savedCycle = m_cycle;
            run(vuCode, codeSize, s_pvMem.data(), dataSize, gs, memory, maxCycles);
            for (int guard = 0; m_xgkick.active && guard < 2000000; ++guard) { ++m_cycle; progressXgkick(); }
            g_vu1DryKicks = false;
            m_cycle = savedCycle; // the shadow's cycles are not the guest's
            const VU1State outI = m_state;
            m_state = outM; // microVU's result stays authoritative for the run
            ++g_pvRuns;
            // Every class is evaluated (no short-circuit) so one run names all the ways it differs.
            // Not compared: `ebit` (the interpreter clears it at exit) and the STICKY status bits
            // (bits 6-11: microVU's lazy flags only guarantee the last four producers before a
            // reader, PCSX2's documented approximation) -- sticky differences are tallied apart.
            bool bad = false; std::string why;
            auto flag = [&](const char *w, unsigned long long &ctr) { bad = true; ++ctr; if (!why.empty()) why += ","; why += w; };
            for (int r = 1; r < 32; ++r)
                if (std::memcmp(outM.vf[r], outI.vf[r], 16) != 0) { flag(("vf" + std::to_string(r)).c_str(), g_pvBadVf); break; }
            for (int r = 1; r < 16; ++r)
                if (outM.vi[r] != outI.vi[r]) { flag(("vi" + std::to_string(r)).c_str(), g_pvBadVi); break; }
            if (std::memcmp(outM.acc, outI.acc, 16) != 0) flag("acc", g_pvBadVf);
            if (((outM.status ^ outI.status) & 0x3Fu) != 0u || outM.mac != outI.mac || outM.clip != outI.clip) flag("flags", g_pvBadFlags);
            if (((outM.status ^ outI.status) & 0xFC0u) != 0u) ++g_pvStickyDiff;
            if (std::memcmp(&outM.q, &outI.q, 4) != 0 || std::memcmp(&outM.p, &outI.p, 4) != 0 || std::memcmp(&outM.i, &outI.i, 4) != 0 || outM.r != outI.r) flag("q/p/i/r", g_pvBadQP);
            if (outM.pc != outI.pc) flag("pc", g_pvBadPc);
            if (std::memcmp(vuData, s_pvMem.data(), dataSize) != 0) flag("vumem", g_pvBadMem);
            if (mvuKicks.size() != g_vu1DryKickList.size()) flag(("kickcount" + std::to_string(mvuKicks.size()) + "v" + std::to_string(g_vu1DryKickList.size())).c_str(), g_pvBadKick);
            for (size_t k = 0; k < mvuKicks.size() && k < g_vu1DryKickList.size(); ++k)
                if (mvuKicks[k].size != g_vu1DryKickList[k].size || mvuKicks[k].hash != g_vu1DryKickList[k].hash)
                { flag(("kick#" + std::to_string(k) + (mvuKicks[k].size != g_vu1DryKickList[k].size ? "size" + std::to_string(mvuKicks[k].size) + "v" + std::to_string(g_vu1DryKickList[k].size) : "data")).c_str(), g_pvBadKick); break; }
            if (outM.cycles != outI.cycles) ++g_pvCycleDiff;
            // Print the first 8 of EACH class (a boot-era class must not hide a level-era one).
            static std::map<std::string, unsigned> s_pvPrinted;
            std::string firstClass = why.substr(0, why.find(','));
            for (char &ch : firstClass) if (std::isdigit(static_cast<unsigned char>(ch))) ch = '#';
            if (bad) ++g_pvBad;
            if (bad && ++s_pvPrinted[firstClass] <= 8u)
            {
                std::fprintf(stderr, "[vu1:progverify] MISMATCH #%llu run=%llu startPC=0x%x %s | mvu: pc=0x%x ebit=%d st=%08x mac=%08x clip=%08x q=%g cyc=%llu kicks=%zu | interp: pc=0x%x ebit=%d st=%08x mac=%08x clip=%08x q=%g cyc=%llu kicks=%zu\n",
                             g_pvBad, g_pvRuns, startPC, why.c_str(),
                             outM.pc, outM.ebit ? 1 : 0, outM.status, outM.mac, outM.clip, outM.q, (unsigned long long)outM.cycles, mvuKicks.size(),
                             outI.pc, outI.ebit ? 1 : 0, outI.status, outI.mac, outI.clip, outI.q, (unsigned long long)outI.cycles, g_vu1DryKickList.size());
                if (why.rfind("kick", 0) == 0)
                {
                    std::string a, b; char kb[64];
                    for (size_t k = 0; k < mvuKicks.size() && k < 3u; ++k) { std::snprintf(kb, sizeof kb, " %u/%016llx", mvuKicks[k].size, (unsigned long long)mvuKicks[k].hash); a += kb; }
                    for (size_t k = 0; k < g_vu1DryKickList.size() && k < 3u; ++k) { std::snprintf(kb, sizeof kb, " %u/%016llx", g_vu1DryKickList[k].size, (unsigned long long)g_vu1DryKickList[k].hash); b += kb; }
                    std::fprintf(stderr, "    kicks mvu:%s | interp:%s\n", a.c_str(), b.c_str());
                }
                if (why.rfind("vf", 0) == 0 && std::isdigit(static_cast<unsigned char>(why[2])))
                {
                    const int r = std::atoi(why.c_str() + 2);
                    std::fprintf(stderr, "    mvu vf%d=%g %g %g %g | interp vf%d=%g %g %g %g\n", r, outM.vf[r][0], outM.vf[r][1], outM.vf[r][2], outM.vf[r][3], r, outI.vf[r][0], outI.vf[r][1], outI.vf[r][2], outI.vf[r][3]);
                }
            }
            if ((g_pvRuns & 4095u) == 0u)
                std::fprintf(stderr, "[vu1:progverify] runs=%llu bad=%llu (vf=%llu vi=%llu flags=%llu qp=%llu pc=%llu mem=%llu kick=%llu) sticky=%llu cycleDiff=%llu\n",
                             g_pvRuns, g_pvBad, g_pvBadVf, g_pvBadVi, g_pvBadFlags, g_pvBadQP, g_pvBadPc, g_pvBadMem, g_pvBadKick, g_pvStickyDiff, g_pvCycleDiff);
        }
        if (s_vu1Perf)
        {
            g_microVuNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            const auto &ms = ps2x_microvu::stats();
            if ((ms.runs & 4095u) == 0u)
                std::fprintf(stderr, "[vu1:microvu] runs=%llu cycles=%llu ns=%llu (%.2f ns/cycle) kicks=%llu kickBytes=%llu clears=%llu tbit=%llu dbit=%llu\n",
                             ms.runs, ms.cycles, g_microVuNs, ms.cycles ? double(g_microVuNs) / double(ms.cycles) : 0.0,
                             ms.kicks, ms.kickBytes, ms.clears, ms.tbits, ms.dbits);
        }
        return;
    }
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    {
        const int microVuResumeFlag = 1;
        const uint32_t startPC = m_state.pc;
        // ★ cont.230 PS2X_VU1_MICROVU=1 (default OFF until verified): run the program on the ported
        // PCSX2 microVU recompiler instead of this interpreter (docs/vu1-program-compiler.md).
        if (s_vu1MicroVu && m_unit == Unit::VU1 && memory)
        {
            const bool mvuResume = (microVuResumeFlag != 0);
            m_activeVuData = vuData;
            m_activeVuDataSize = dataSize;
            m_activeGs = &gs;
            m_activeMemory = memory;
            ps2x_microvu::init(vuCode, vuData);
            ps2x_microvu::setMemory(memory);
            static bool s_announced = false;
            if (!s_announced) { s_announced = true; std::fprintf(stderr, "[vu1:microvu] active (PS2X_VU1_MICROVU=1)%s\n", s_vu1ProgVerify ? " + PROGVERIFY" : ""); }
            // PROGVERIFY: snapshot the input state and the VU data memory before microVU touches them.
            static std::vector<uint8_t> s_pvMem;
            VU1State pvIn{};
            if (s_vu1ProgVerify)
            {
                pvIn = m_state;
                s_pvMem.assign(vuData, vuData + dataSize);
                ps2x_microvu::setKickRecording(true);
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t used = ps2x_microvu::run(m_state, mvuResume ? m_state.pc : startPC, mvuResume, maxCycles);
            m_cycle += used;
            if (s_vu1ProgVerify)
            {
                ps2x_microvu::setKickRecording(false);
                size_t mk = 0;
                const ps2x_microvu::KickRecord *mkr = ps2x_microvu::kickRecords(&mk);
                std::vector<ps2x_microvu::KickRecord> mvuKicks(mkr, mkr + mk);
                const VU1State outM = m_state;
                // Shadow run on the interpreter: same input, copied memory, dry kicks.
                m_state = pvIn;
                // The shadow always starts from drained pipelines (a program end drains them on the
                // hardware and in both implementations) and, after the run, completes any XGKICK still
                // streaming so its kick list covers the whole program, as microVU's does.
                resetScheduler();
                g_vu1DryKicks = true;
                g_vu1DryKickList.clear();
                const uint64_t savedCycle = m_cycle;
                run(vuCode, codeSize, s_pvMem.data(), dataSize, gs, memory, maxCycles);
                for (int guard = 0; m_xgkick.active && guard < 2000000; ++guard) { ++m_cycle; progressXgkick(); }
                g_vu1DryKicks = false;
                m_cycle = savedCycle; // the shadow's cycles are not the guest's
                const VU1State outI = m_state;
                m_state = outM; // microVU's result stays authoritative for the run
                ++g_pvRuns;
                // Every class is evaluated (no short-circuit) so one run names all the ways it differs.
                // Not compared: `ebit` (the interpreter clears it at exit) and the STICKY status bits
                // (bits 6-11: microVU's lazy flags only guarantee the last four producers before a
                // reader, PCSX2's documented approximation) -- sticky differences are tallied apart.
                bool bad = false; std::string why;
                auto flag = [&](const char *w, unsigned long long &ctr) { bad = true; ++ctr; if (!why.empty()) why += ","; why += w; };
                for (int r = 1; r < 32; ++r)
                    if (std::memcmp(outM.vf[r], outI.vf[r], 16) != 0) { flag(("vf" + std::to_string(r)).c_str(), g_pvBadVf); break; }
                for (int r = 1; r < 16; ++r)
                    if (outM.vi[r] != outI.vi[r]) { flag(("vi" + std::to_string(r)).c_str(), g_pvBadVi); break; }
                if (std::memcmp(outM.acc, outI.acc, 16) != 0) flag("acc", g_pvBadVf);
                if (((outM.status ^ outI.status) & 0x3Fu) != 0u || outM.mac != outI.mac || outM.clip != outI.clip) flag("flags", g_pvBadFlags);
                if (((outM.status ^ outI.status) & 0xFC0u) != 0u) ++g_pvStickyDiff;
                if (std::memcmp(&outM.q, &outI.q, 4) != 0 || std::memcmp(&outM.p, &outI.p, 4) != 0 || std::memcmp(&outM.i, &outI.i, 4) != 0 || outM.r != outI.r) flag("q/p/i/r", g_pvBadQP);
                if (outM.pc != outI.pc) flag("pc", g_pvBadPc);
                if (std::memcmp(vuData, s_pvMem.data(), dataSize) != 0) flag("vumem", g_pvBadMem);
                if (mvuKicks.size() != g_vu1DryKickList.size()) flag(("kickcount" + std::to_string(mvuKicks.size()) + "v" + std::to_string(g_vu1DryKickList.size())).c_str(), g_pvBadKick);
                for (size_t k = 0; k < mvuKicks.size() && k < g_vu1DryKickList.size(); ++k)
                    if (mvuKicks[k].size != g_vu1DryKickList[k].size || mvuKicks[k].hash != g_vu1DryKickList[k].hash)
                    { flag(("kick#" + std::to_string(k) + (mvuKicks[k].size != g_vu1DryKickList[k].size ? "size" + std::to_string(mvuKicks[k].size) + "v" + std::to_string(g_vu1DryKickList[k].size) : "data")).c_str(), g_pvBadKick); break; }
                if (outM.cycles != outI.cycles) ++g_pvCycleDiff;
                // Print the first 8 of EACH class (a boot-era class must not hide a level-era one).
            static std::map<std::string, unsigned> s_pvPrinted;
            std::string firstClass = why.substr(0, why.find(','));
            for (char &ch : firstClass) if (std::isdigit(static_cast<unsigned char>(ch))) ch = '#';
            if (bad) ++g_pvBad;
            if (bad && ++s_pvPrinted[firstClass] <= 8u)
                {
                    std::fprintf(stderr, "[vu1:progverify] MISMATCH #%llu run=%llu startPC=0x%x %s | mvu: pc=0x%x ebit=%d st=%08x mac=%08x clip=%08x q=%g cyc=%llu kicks=%zu | interp: pc=0x%x ebit=%d st=%08x mac=%08x clip=%08x q=%g cyc=%llu kicks=%zu\n",
                                 g_pvBad, g_pvRuns, startPC, why.c_str(),
                                 outM.pc, outM.ebit ? 1 : 0, outM.status, outM.mac, outM.clip, outM.q, (unsigned long long)outM.cycles, mvuKicks.size(),
                                 outI.pc, outI.ebit ? 1 : 0, outI.status, outI.mac, outI.clip, outI.q, (unsigned long long)outI.cycles, g_vu1DryKickList.size());
                    if (why.rfind("kick", 0) == 0)
                {
                    std::string a, b; char kb[64];
                    for (size_t k = 0; k < mvuKicks.size() && k < 3u; ++k) { std::snprintf(kb, sizeof kb, " %u/%016llx", mvuKicks[k].size, (unsigned long long)mvuKicks[k].hash); a += kb; }
                    for (size_t k = 0; k < g_vu1DryKickList.size() && k < 3u; ++k) { std::snprintf(kb, sizeof kb, " %u/%016llx", g_vu1DryKickList[k].size, (unsigned long long)g_vu1DryKickList[k].hash); b += kb; }
                    std::fprintf(stderr, "    kicks mvu:%s | interp:%s\n", a.c_str(), b.c_str());
                }
                if (why.rfind("vf", 0) == 0 && std::isdigit(static_cast<unsigned char>(why[2])))
                    {
                        const int r = std::atoi(why.c_str() + 2);
                        std::fprintf(stderr, "    mvu vf%d=%g %g %g %g | interp vf%d=%g %g %g %g\n", r, outM.vf[r][0], outM.vf[r][1], outM.vf[r][2], outM.vf[r][3], r, outI.vf[r][0], outI.vf[r][1], outI.vf[r][2], outI.vf[r][3]);
                    }
                }
                if ((g_pvRuns & 4095u) == 0u)
                    std::fprintf(stderr, "[vu1:progverify] runs=%llu bad=%llu (vf=%llu vi=%llu flags=%llu qp=%llu pc=%llu mem=%llu kick=%llu) sticky=%llu cycleDiff=%llu\n",
                                 g_pvRuns, g_pvBad, g_pvBadVf, g_pvBadVi, g_pvBadFlags, g_pvBadQP, g_pvBadPc, g_pvBadMem, g_pvBadKick, g_pvStickyDiff, g_pvCycleDiff);
            }
            if (s_vu1Perf)
            {
                g_microVuNs += static_cast<unsigned long long>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
                const auto &ms = ps2x_microvu::stats();
                if ((ms.runs & 4095u) == 0u)
                    std::fprintf(stderr, "[vu1:microvu] runs=%llu cycles=%llu ns=%llu (%.2f ns/cycle) kicks=%llu kickBytes=%llu clears=%llu tbit=%llu dbit=%llu\n",
                                 ms.runs, ms.cycles, g_microVuNs, ms.cycles ? double(g_microVuNs) / double(ms.cycles) : 0.0,
                                 ms.kicks, ms.kickBytes, ms.clears, ms.tbits, ms.dbits);
            }
            return;
        }
    }
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    // PS2X_VU1_PERF: one clock read per run() call (not per instruction).
    const bool perfTrack = s_vu1Perf && m_unit == Unit::VU1;
    const bool perfTrack0 = s_vu1Perf && m_unit == Unit::VU0; // cont.214: size the VU0 slice
    unsigned long long perfPairs = 0;
    std::chrono::steady_clock::time_point perfT0;
    if (perfTrack || perfTrack0)
        perfT0 = std::chrono::steady_clock::now();
    // cont.177: arm lazy MAC/status flags for this program. Armed ONCE per run() call, not per
    // instruction, because the code generation cannot change mid-program. The default is the slow
    // (full-fidelity) path, so every case we cannot prove — VU0, untracked code, a stale or
    // incomplete scan, the very first run after a code upload (the scan is recorded by the cache
    // rebuild that run itself triggers) — simply keeps deriving flags.
    // cont.182: one-shot self-test of the JIT emitter — emits a native operand-clamp function and
    // checks it against the C++ quad helper over an exhaustive bit sweep. Validates the x86-64
    // encodings AND the emitted SSE sequence before any of it is pointed at guest code.
    if (s_vu1JitSelfTest)
    {
        static const bool jitTested = []() {
            static vu1jit::CodeBuffer buf;
            const bool a = vu1jit::selfTest(buf, &vuNormOperandQuad);
            const bool b = vu1jit::selfTestFmac(buf, &vuNormOperandQuad, &vuNormResultQuadValue);
            const bool c = vu1jit::selfTestLQ(buf, false); // LQ
            const bool d = vu1jit::selfTestLQ(buf, true);  // SQ
            // cont.194: IADDIU/ISUBIU + the four clip readers, driven through
            // classifyLowerPlan -> emitLowerPlan (the same path block assembly will use).
            const bool f = vu1jit::selfTestLowerVi(buf);
            (void)c; (void)d; (void)f;
            vu1jit::benchBlock(buf, &vuNormOperandQuad, &vuNormResultQuadValue);
            return a && b;
        }();
        (void)jitTested;
    }

    // cont.181: one-shot exhaustive self-test that the SSE quad clamps are BIT-IDENTICAL to the
    // scalar reference — every exponent (0..255) x representative mantissas x both signs, and for
    // the result form every one of the 16 dest masks, comparing value bits AND flags. Runs once,
    // costs nothing after that, and turns "should be equivalent" into a checked fact.
    if (s_vu1SimdVerify)
    {
        static const bool tested = [this]() {
            unsigned long mismatches = 0, cases = 0;
            static const uint32_t mantissas[4] = {0u, 1u, 0x400000u, 0x7FFFFFu};
            for (uint32_t exp = 0; exp < 256u; ++exp)
            {
                for (uint32_t mi = 0; mi < 4u; ++mi)
                {
                    for (uint32_t sgn = 0; sgn < 2u; ++sgn)
                    {
                        const uint32_t bits = (sgn << 31) | (exp << 23) | mantissas[mi];
                        float f;
                        std::memcpy(&f, &bits, 4);

                        // --- operand clamp ---
                        alignas(16) float in[4] = {f, f, f, f};
                        alignas(16) float out[4];
                        vuNormOperandQuad(in, out);
                        const float ref = normalizeOperand(f);
                        uint32_t a = 0, b = 0;
                        std::memcpy(&a, &out[0], 4);
                        std::memcpy(&b, &ref, 4);
                        ++cases;
                        if (a != b)
                        {
                            if (++mismatches < 8u)
                                RUNTIME_ERROR("[VU1 simd-verify] operand bits=" << std::hex << bits
                                              << " simd=" << a << " scalar=" << b << std::dec);
                        }

                        // --- result clamp + flags, over every dest mask ---
                        for (uint32_t dest = 0; dest < 16u; ++dest)
                        {
                            alignas(16) float sv[4] = {f, f, f, f};
                            uint8_t sf[4] = {0, 0, 0, 0};
                            for (uint32_t c = 0; c < 4u; ++c)
                            {
                                if ((dest & laneForComponent(c)) == 0u)
                                    continue;
                                uint32_t fl = 0u;
                                sv[c] = normalizeResult(sv[c], fl);
                                sf[c] = static_cast<uint8_t>(fl);
                            }
                            alignas(16) float qv[4] = {f, f, f, f};
                            uint8_t qf[4] = {0, 0, 0, 0};
                            vuNormResultQuad(qv, static_cast<uint8_t>(dest), qf);
                            for (uint32_t c = 0; c < 4u; ++c)
                            {
                                uint32_t x = 0, y = 0;
                                std::memcpy(&x, &qv[c], 4);
                                std::memcpy(&y, &sv[c], 4);
                                ++cases;
                                if (x != y || qf[c] != sf[c])
                                {
                                    if (++mismatches < 8u)
                                        RUNTIME_ERROR("[VU1 simd-verify] result bits=" << std::hex << bits
                                                      << " dest=" << dest << " lane=" << c
                                                      << " simd=" << x << "/f" << unsigned(qf[c])
                                                      << " scalar=" << y << "/f" << unsigned(sf[c]) << std::dec);
                                }
                            }
                            // value-only form must match the clamped values too
                            alignas(16) float vv[4] = {f, f, f, f};
                            vuNormResultQuadValue(vv, static_cast<uint8_t>(dest));
                            for (uint32_t c = 0; c < 4u; ++c)
                            {
                                uint32_t x = 0, y = 0;
                                std::memcpy(&x, &vv[c], 4);
                                std::memcpy(&y, &sv[c], 4);
                                ++cases;
                                if (x != y && ++mismatches < 8u)
                                    RUNTIME_ERROR("[VU1 simd-verify] valueOnly bits=" << std::hex << bits
                                                  << " dest=" << dest << " lane=" << c << std::dec);
                            }
                        }
                    }
                }
            }
            std::fprintf(stderr, "[vu1:simdverify] cases=%lu mismatches=%lu -> %s\n",
                         cases, mismatches, mismatches == 0 ? "BIT-IDENTICAL" : "MISMATCH");
            return true;
        }();
        (void)tested;
    }

    // cont.178 (a): resolve the decode cache ONCE for this program run. Mirrors exactly the
    // freshness check in getDecodedInstructionPairForPc; when it holds, the loop indexes the cache
    // by reference instead of paying a check + an ~80-byte struct copy per pair. Running this
    // BEFORE the lazy-flag arming below also means the scan record is already fresh on the first
    // run after a code upload, so that run can arm too (strictly more coverage, same proof).
    const DecodedInstructionPair *decodeCache = nullptr;
    if (s_vu1FastDispatch && memory != nullptr)
    {
        const bool trackedCode = (m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code());
        if (trackedCode)
        {
            const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration()
                                                            : memory->getVU0CodeGeneration();
            if (!m_decodedCodeCacheValid ||
                m_cachedVuCode != vuCode ||
                m_cachedMemory != memory ||
                m_cachedCodeSize != codeSize ||
                m_cachedCodeGeneration != generation)
            {
                rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
            }
            decodeCache = m_decodedCodeCache.data();
        }
    }

    // ★ cont.214: per-unit lazy-scan slot and the matching code-generation counter. Everything
    // below that was implicitly "VU1" now says which unit it means.
    const unsigned unitIdx = (m_unit == Unit::VU1) ? 1u : 0u;
    const uint64_t unitCodeGeneration =
        memory != nullptr ? (m_unit == Unit::VU1 ? memory->getVU1CodeGeneration()
                                                 : memory->getVU0CodeGeneration())
                          : 0ull;
    s_lazyActive = false;
    s_flagWriteThisRun = false;
    if (s_vu1LazyFlags != 0 && m_unit == Unit::VU1 && !s_lazyDisabled)
    {
        if (s_vu1LazyFlags >= 2)
        {
            s_lazyActive = true; // ABLATION (force), see the flag comment
        }
        else if (memory != nullptr &&
                 s_lazyScan[unitIdx].owner == this &&
                 s_lazyScan[unitIdx].code == vuCode &&
                 s_lazyScan[unitIdx].memory == memory &&
                 s_lazyScan[unitIdx].codeSize == codeSize &&
                 s_lazyScan[unitIdx].generation == unitCodeGeneration &&
                 s_lazyScan[unitIdx].complete &&
                 !s_lazyScan[unitIdx].readsMacOrStatus)
        {
            s_lazyActive = true;
        }
        if (s_lazyActive)
        {
            ++g_lazyRunsArmed;
            s_macDirty = true; // this program will skip flag writes
            s_stickyDirty = true;
        }
    }
    // cont.183: the JIT's emitted code derives no MAC/status flags, so it may only run while lazy
    // flags is armed. Mirror the decision into the shared flag ps2_vu1_upper.cpp reads.
    vu1jit::lazyActive() = s_lazyActive;
    // ★★ cont.204: when lazy flags are NOT armed, blocks used to be disabled for the whole program
    // -- 68.8% of all still-interpreted cycles. They can run instead in FLAG-EMITTING mode: each
    // FMAC stashes its pre-clamp result and the driver replays updateFmacFlags from it. The
    // product-STICKY bits (calculateFmacProductSticky) need the operands, which are gone by then,
    // and they only affect `status` -- so this mode requires the program to contain no STATUS
    // reader. cont.177 measured stsRd=0 across full runs, so that is the common case.
    // cont.214: VU0 reaches this too. It is never lazy-armed (its flags are EE-visible), so for
    // VU0 this IS the only block mode -- which is exactly what its programs need (they read MAC,
    // so `clean=0`, but none reads STATUS).
    const bool blockEmitFlags =
        !s_lazyActive && (m_unit == Unit::VU1 || s_vu0Block) && memory != nullptr &&
        s_lazyScan[unitIdx].owner == this && s_lazyScan[unitIdx].code == vuCode &&
        s_lazyScan[unitIdx].memory == memory && s_lazyScan[unitIdx].codeSize == codeSize &&
        s_lazyScan[unitIdx].generation == unitCodeGeneration && s_lazyScan[unitIdx].complete &&
        !s_lazyScan[unitIdx].readsStatus;
    // Scratch for the uncacheable decode path only; hoisted out of the loop so it is not
    // default-constructed (80 bytes) every iteration.
    DecodedInstructionPair decodedFallback;
    // cont.195 block shadow-verify state (PS2X_VU1_BLOCKVERIFY only). Static because VU1State is
    // large and run() executes on the guest thread's small stack.
    // Block-cache key. The decode cache's own validity key is (vuCode, memory, codeSize,
    // generation); mixing all the varying parts into one 64-bit value means the block store is
    // invalidated by ANY of them changing, not just the code generation counter.
    const uint64_t blockGen =
        m_cachedCodeGeneration ^
        (uint64_t(reinterpret_cast<uintptr_t>(vuCode)) * 0x9E3779B97F4A7C15ull) ^
        (uint64_t(codeSize) << 1) ^ (blockEmitFlags ? 0x9E3779B9ull : 0ull);
    // ★★ cont.197b: the block fast path's ARMING conditions are all loop-invariant, and the
    // rdtsc segment profile showed its per-pair cost dominating everything else (300 of 516 cycles
    // per entry, spread over 248M lookups on pairs that have no block at all). Hoist them, resolve
    // the store and the generation once, and reduce the per-pair test to a single array load.
    vu1jit::BlockStore &bstore = vu1jit::blockStore(unitIdx);
    const bool blockArmed =
        s_vu1Block && (m_unit == Unit::VU1 || s_vu0Block) && (s_lazyActive || blockEmitFlags) &&
        decodeCache != nullptr &&
        vuData != nullptr && dataSize >= 16u && (dataSize & (dataSize - 1u)) == 0u;
    if (blockArmed && bstore.generation != blockGen)
    {
        // Hash the microcode so an identical re-upload keeps the compiled blocks. Safe to hoist:
        // the guest cannot upload microcode while its own program is running (the cont.178
        // argument), so the code cannot change under us mid-run().
        uint64_t h = 0xcbf29ce484222325ull;
        const uint64_t *w = reinterpret_cast<const uint64_t *>(vuCode);
        const uint32_t n64 = codeSize / 8u;
        for (uint32_t i = 0; i < n64; ++i)
        {
            h ^= w[i];
            h *= 0x100000001b3ull;
        }
        bstore.revalidate(blockGen, h);
    }
    const int32_t *const blockByPair = bstore.byPair;
    static VU1State blockVerifyRef;
    static std::vector<uint8_t> blockVerifyMem;
    uint8_t blockEntryPendingVf[32] = {};
    uint32_t blockEntryPendingVi = 0u;
    uint8_t blockEntryPendingAcc = 0u;
    bool blockEntryPendingStore = false;
    uint32_t blockClipN = 0;
    uint32_t blockClipRaw[32] = {};
    uint64_t blockClipIssueArr[32] = {};
    uint32_t blockFlagN = 0;
    uint32_t blockFlagMacArr[32] = {}, blockFlagStatusArr[32] = {};
    uint64_t blockFlagIssueArr[32] = {};
    bool blockEntryFlagPending = false;
    bool blockEntryFdivPending = false;
    // ★★ cont.216: THE VU0 DRIVER BUG. The block driver replays a whole block's clip + MAC flag
    // entries in ONE pass, with no retirement between them, into m_flagPipeline. The interpreter
    // never needs more than ~kFmacLatency slots because it calls commitReadyPipelines() at the top
    // of EVERY pair; the block's live set is instead bounded by the BLOCK LENGTH. The array
    // overflowed, freeSlot() returned -1, and the queue helpers HALTED the program mid-flight.
    // VU1 never hit it in the default config -- it is normally lazy-armed, so `emitsFlags` is ~0%
    // of its entries (cont.212 measured 0.0%) and neither replay loop runs. VU0 is NEVER
    // lazy-armed (its flags are EE-visible), so cont.204's flag-emitting mode is its ONLY block
    // mode and every single block replays -- which is exactly why PS2X_VU0_BLOCK=1 produced a
    // storm of them and PS2X_VU0_BLOCK=0 produced none.
    // Retiring the entries that are DUE at each replay step is precisely what the interpreter does,
    // at precisely the same cycle, so this is strictly MORE faithful than deferring them all to the
    // next commitReadyPipelines() -- it also fixes a latent ordering wrinkle, since a deferred
    // batch retires in SLOT order while the interpreter retires in issue order.
    // Flags only: mac/status/clip are written by no other pipeline, so no other commit order can be
    // affected, and the incoming VF/VI/acc writes the block deliberately early-applies are untouched.
    // cont.220: earliest readyCycle among live flag entries, or 0 = "unknown, must walk". Reset at
    // every block entry, so it can never carry staleness across blocks (the pipeline is also
    // touched outside the replay -- by commitReadyPipelines at the loop top and by interpreted
    // pairs -- and a value that is too LARGE would skip a due retire, which is exactly the cont.216
    // overflow. Too SMALL only costs a walk.)
    uint64_t flagMinReady = 0;
    const auto retireDueFlagEntries = [&]()
    {
        if (s_vu1BlockProf)
            ++g_ffrCalls;
        if (s_vu1FastFlagRetire && m_cycle < flagMinReady)
        {
            if (s_vu1BlockProf)
                ++g_ffrSkipped;
            return; // nothing can be due -- skip the pipeFor() and the whole walk
        }
        uint64_t nextMin = kPipeIdle;
        PipeTrack &pipe = pipeFor(this);
        const bool tracked = s_vu1CommitSkip && pipe.owner == this;
        const uint32_t survivors =
            retirePipe(m_flagPipeline, pipe.flag, tracked, [&](FlagPipelineEntry &entry)
            {
                if (entry.readyCycle > m_cycle)
                {
                    if (entry.readyCycle < nextMin)
                        nextMin = entry.readyCycle;
                    return false;
                }
                if (entry.writesMac)
                    m_state.mac = entry.mac;
                if (entry.writesStatus)
                {
                    const uint32_t current = entry.status & 0xFu;
                    m_state.status =
                        (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
                }
                if (entry.writesSticky)
                    m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
                if (entry.writesClip)
                    m_state.clip = entry.clip;
                entry = {};
                if (s_vu1BlockProf)
                    ++g_ffrRetired;
                return true;
            });
        // Only ever write our OWN tracking slot. `nextReady` is deliberately left alone: it is a
        // LOWER bound used to skip work, so leaving it too small costs one scan and can never skip
        // a due commit.
        if (tracked)
            pipe.flag = survivors;
        // ★ The walk visited every live entry (when tracked the mask is a SUPERSET of reality, so
        // it still covers them all), so `nextMin` is exact for what remains.
        flagMinReady = nextMin;
    };
    // ★★ cont.217 FLAGVERIFY reference. Deliberately an INDEPENDENT restatement, not a refactor of
    // the production path: it merges the clip and MAC event streams into ONE ascending-cycle
    // sequence (the order the interpreter issues them in, one pair at a time) and uses plain linear
    // scans where production uses the valid-slot bitmask and ctz. Statics because run() executes on
    // the guest thread's small stack (the cont.183 SIGBUS trap).
    struct FvEvent { uint64_t cycle; uint32_t a, b; bool isClip; };
    static FvEvent fvEv[64];
    static std::array<FlagPipelineEntry, kMaxFlagEntries> fvSnapPipe, fvProdPipe, fvRefPipe;
    static uint32_t fvSnapMac, fvSnapStatus, fvSnapClip, fvSnapWorking;
    static uint32_t fvProdMac, fvProdStatus, fvProdClip, fvProdWorking;
    static std::array<FlagPipelineEntry, kMaxFlagEntries> fvKeepPipe;
    static uint32_t fvKeepMac, fvKeepStatus, fvKeepClip;
    static uint32_t fvSnapMask;
    // Apply one committed entry exactly as commitReadyPipelines does.
    const auto fvApply = [](const FlagPipelineEntry &e, uint32_t &mac, uint32_t &status,
                            uint32_t &clip)
    {
        if (e.writesMac)
            mac = e.mac;
        if (e.writesStatus)
        {
            const uint32_t cur = e.status & 0xFu;
            status = (status & 0xFF0u) | cur | ((cur | e.extraSticky) << 6);
        }
        if (e.writesSticky)
            status = (status & 0x03Fu) | (e.status & 0xFC0u);
        if (e.writesClip)
            clip = e.clip;
    };
    // ★★ Bring a pipeline to a CHECKPOINT the way the interpreter would: retire every entry due
    // by `cycle`, lowest ISSUE cycle first. Issue order is canonical here because every flag entry
    // has the same kFmacLatency, so ready order == issue order -- and it is slot-index independent,
    // which matters because production's ctz-off-the-mask and the reference's linear scan can pick
    // different (equally free) slots for the same entry.
    // ★ WHY A CHECKPOINT AT ALL: production defers the commits the interpreter would have made
    // DURING the block to the next commitReadyPipelines() at m_cycle = entry + dynCycles (or to
    // flushPipelines() at program end, ps2_vu1_core.cpp). That deferral is unobservable -- nothing
    // reads mac/status/clip between the block returning and that commit -- so comparing before it
    // reports a transient, not a defect. Comparing AFTER it is the real contract.
    const auto fvDrainDue = [&fvApply](std::array<FlagPipelineEntry, kMaxFlagEntries> &pipe,
                                       uint32_t &mac, uint32_t &status, uint32_t &clip,
                                       uint64_t cycle)
    {
        for (;;)
        {
            uint32_t best = kMaxFlagEntries;
            for (uint32_t i = 0; i < kMaxFlagEntries; ++i)
            {
                if (!pipe[i].valid || pipe[i].readyCycle > cycle)
                    continue;
                if (best == kMaxFlagEntries || pipe[i].issueCycle < pipe[best].issueCycle)
                    best = i;
            }
            if (best == kMaxFlagEntries)
                break;
            fvApply(pipe[best], mac, status, clip);
            pipe[best] = {};
        }
    };
    // Canonical view of what is still in flight: the LIVE entries sorted by issue cycle. Comparing
    // by SLOT INDEX would report benign divergence, because a valid-slot mask that is a strict
    // superset of reality makes freeSlot's ctz pick a different (still free) slot than a linear
    // scan -- same contents, different index.
    const auto fvLive = [](const std::array<FlagPipelineEntry, kMaxFlagEntries> &pipe,
                           FlagPipelineEntry *out) -> uint32_t
    {
        uint32_t n = 0;
        for (uint32_t i = 0; i < kMaxFlagEntries; ++i)
            if (pipe[i].valid)
                out[n++] = pipe[i];
        for (uint32_t i = 1; i < n; ++i) // insertion sort, n <= 16
        {
            FlagPipelineEntry key = out[i];
            int32_t j = int32_t(i) - 1;
            while (j >= 0 && (out[j].issueCycle > key.issueCycle ||
                              (out[j].issueCycle == key.issueCycle &&
                               out[j].readyCycle > key.readyCycle)))
            {
                out[j + 1] = out[j];
                --j;
            }
            out[j + 1] = key;
        }
        return n;
    };
    const vu1jit::Block *blockVerifyBlk = nullptr;
    uint32_t blockVerifyLeft = 0u;
    uint32_t blockVerifyCycles = 0u;
    uint64_t blockVerifyEntryCycle = 0u;
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        ++perfPairs;
        commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        // ---- cont.195: BLOCK FAST PATH -------------------------------------------------------
        // Replaces the interpreter's inner loop for a run of consecutive pairs with one call into
        // natively compiled code. See ps2_vu1_jit.h for the correctness argument; the guard below
        // is the other half of it.
        //
        // ★★ ENTRY GUARD (cont.195b — the PRECISE form). The first version demanded globally
        // empty write pipelines. That is sound but nearly useless: FMAC latency is 4 and the
        // microcode issues continuously, so something is almost always in flight, and it measured
        // only 2.4% coverage. The guard that matters is far narrower — an incoming pending write
        // is harmless unless the block touches the very slot it targets:
        //   * block READS that slot  -> the interpreter would have stalled until it committed, so
        //     the block would read a stale value;
        //   * block WRITES that slot -> the pending write commits later and clobbers the block's
        //     newer value (the interpreter avoids this with the m_vfLatestWrite supersession
        //     check, which the block does not participate in).
        // Anything not intersecting the block's touched set simply commits after the block: the
        // next iteration's commitReadyPipelines() runs at m_cycle+pairs and applies everything
        // then due, in the same order and with the same supersession rule.
        // ★ `readyEntry > m_cycle` holds exactly when a write to that slot is still pending (both
        // are set together at issue and cleared at commit), so "no read intersects a pending
        // write" ALSO proves no pair stalls on incoming state — the two conditions are one.
        // Blocks are additionally cut before any internally stalling pair, so `m_cycle += pairs`
        // is exact. XGKICK must be idle: it streams VU memory per cycle, which the block does not
        // model.
        // ★ Bound, do not mask: masking an out-of-range pair index would ALIAS a different PC's
        // block and execute the wrong code. VU1 microcode is <= 16 KB so this never trips, but the
        // failure mode is silent corruption, so it is a compare rather than an assumption.
        if (blockArmed && (m_state.pc & 7u) == 0u &&
            (m_state.pc >> 3) < vu1jit::BlockStore::kPairSlots &&
            blockByPair[m_state.pc >> 3] != 0 &&
            !m_state.branchPending && !m_state.ebit && !m_state.haltAfterDelaySlot &&
            !m_xgkick.active)
        {
            const unsigned long long bpT0 = s_vu1BlockProf ? __rdtsc() : 0ull;
            const uint32_t startPair = m_state.pc / 8u;
            const int32_t blockIdx = blockByPair[startPair];
            const vu1jit::Block *blk =
                blockIdx > 0 ? &bstore.blocks[blockIdx] : nullptr;
            if (blockIdx < 0)
            {
                // Flatten the interpreter's (private) DecodedInstructionPair into the JIT's POD.
                // Only on a cache miss, so this costs nothing in steady state.
                static vu1jit::PairInfo infos[vu1jit::kMaxBlockPairs];
                uint32_t avail = 0;
                const uint32_t limit =
                    std::min<uint32_t>(kMaxDecodedPairs, codeSize / 8u);
                while (avail < vu1jit::kMaxBlockPairs && startPair + avail < limit)
                {
                    const DecodedInstructionPair &d = decodeCache[startPair + avail];
                    vu1jit::PairInfo &pi = infos[avail];
                    pi = vu1jit::PairInfo{};
                    pi.upper = d.upper;
                    pi.lower = d.lower;
                    pi.iBit = d.iBit;
                    pi.eBit = d.eBit;
                    pi.dBit = d.dBit;
                    pi.tBit = d.tBit;
                    pi.upperWriteReg = d.upperUsage.vfWrite.reg;
                    pi.upperWriteLanes = d.upperUsage.vfWrite.lanes;
                    pi.upperLatency = d.upperUsage.vfLatency != 0u ? d.upperUsage.vfLatency
                                                                   : d.upperUsage.latency;
                    pi.lowerWriteReg = d.lowerUsage.vfWrite.reg;
                    pi.lowerWriteLanes = d.lowerUsage.vfWrite.lanes;
                    pi.lowerLatency = d.lowerUsage.vfLatency != 0u ? d.lowerUsage.vfLatency
                                                                   : d.lowerUsage.latency;
                    pi.accWrite = d.upperUsage.accWrite;
                    pi.viWriteMask = d.lowerUsage.viWrite;
                    pi.viLatency = d.lowerUsage.viLatency != 0u ? d.lowerUsage.viLatency
                                                                : d.lowerUsage.latency;
                    pi.suppressedLowerVf = d.suppressedLowerVf;
                    pi.lowerDelaysBranchRead = d.lowerUsage.delaysNextBranchRead;
                    uint8_t rc = 0;
                    for (uint32_t u = 0; u < 2u; ++u)
                    {
                        const InstructionUsage &us = (u == 0u) ? d.upperUsage : d.lowerUsage;
                        for (uint32_t k = 0; k < us.vfReadCount && rc < 4u; ++k)
                        {
                            pi.vfReadReg[rc] = us.vfRead[k].reg;
                            pi.vfReadLanes[rc] = us.vfRead[k].lanes;
                            ++rc;
                        }
                    }
                    pi.vfReadCount = rc;
                    pi.viReadMask = uint16_t(d.upperUsage.viRead | d.lowerUsage.viRead);
                    pi.accRead = uint8_t(d.upperUsage.accRead | d.lowerUsage.accRead);
                    ++avail;
                }
                blk = vu1jit::buildBlock(vu1jit::blockBuffer(unitIdx), bstore, blockGen, m_state.pc,
                                         microAddressMask(), infos, avail, blockEmitFlags);
            }
            // ★ Valid-slot bitmasks (cont.175 PipeTrack) so every check below walks only live
            // entries instead of whole arrays. They are only trustworthy while this instance owns
            // the tracking; otherwise fall back to "assume everything is live".
            // ★ Computed ONLY when a block actually exists for this PC: ~88% of pairs have none,
            // and paying pipeFor() + the mask selects on every one of them is pure loss.
            if (blk == nullptr)
            {
                if (s_vu1BlockProf) { g_bpLookup += __rdtsc() - bpT0; ++g_bpMiss; }
                goto blockNoEntry;
            }
            {
            if (s_vu1BlockProfSeg) g_bpDispatch += __rdtsc() - bpT0;
            if (s_vu1LinkCensus)
            {
                ++g_lcEntries[unitIdx];
                g_lcPairs[unitIdx] += blk->pairs;
                if ((g_lcEntries[unitIdx] % 2000000ull) == 0ull)
                    lcReport();
                if (g_lcPrevWasBlock[unitIdx] && g_lcPrevIdx[unitIdx] > 0)
                {
                    ++g_lcChained[unitIdx];
                    g_lcChainPairs[unitIdx] += blk->pairs;
                    const uint32_t *sc = g_lcSucc[unitIdx][g_lcPrevIdx[unitIdx]];
                    if (sc[0] == startPair)
                        ++g_lcHit1[unitIdx];
                    if (sc[0] == startPair || sc[1] == startPair)
                        ++g_lcHit2[unitIdx];
                }
                // What ENDS this block? A block that does not end with a branch continues at
                // startPc + 8*pairs; one that ends with an unconditional branch continues at a
                // compile-time constant. Both are mergeable with no speculation at all. Re-decode
                // the branch pair rather than widening Block (kept to one file for the census).
                if (!blk->endsWithBranch)
                {
                    ++g_lcNoBranch[unitIdx];
                }
                else if (blk->pairs >= 2u)
                {
                    const uint32_t brPc = blk->startPc + 8u * (blk->pairs - 2u);
                    const uint32_t brPair = brPc / 8u;
                    if (brPair < kMaxDecodedPairs)
                    {
                        vu1jit::BranchPlan bp;
                        if (vu1jit::classifyBranch(decodeCache[brPair].lower, brPc,
                                                   microAddressMask(), bp))
                            ++g_lcKind[unitIdx][uint32_t(bp.kind) % 12u];
                    }
                }
            }
            const PipeTrack &blockPipe = pipeFor(this);
            const bool blockTracked = s_vu1CommitSkip && blockPipe.owner == this;
            const uint32_t blockVfMask = blockTracked ? blockPipe.vf : 0xFFFFu;
            const uint32_t blockViMask = blockTracked ? blockPipe.vi : 0xFFu;
            const uint32_t blockAccMask = blockTracked ? blockPipe.acc : 0xFFu;
            const uint32_t blockStoreMask = blockTracked ? blockPipe.store : 0xFFu;
            const uint32_t blockFlagMask = blockTracked ? blockPipe.flag : 0xFFu;
            bool blockApplyClip = false;
            bool blockApplyQ = false;
            const bool blockNoPending =
                blockTracked && (blockPipe.vf | blockPipe.vi | blockPipe.acc) == 0u;

            // The precise pending-write guard (see above). Small loops over small arrays, and
            // only reached when a compiled block exists for this PC.
            bool guardOk = blk != nullptr;
            // ★★ cont.196: a pending VF write no longer rejects the block. It is handled instead,
            // and soundly, by the SAME argument that made immediate writes safe in the first place:
            //   * APPLY IT EARLY. Any pair that would read that slot before the write's readyCycle
            //     stalls until it commits, so the old value is unobservable -- applying it now
            //     cannot change any value the block reads.
            //   * RE-DERIVE THE SCHEDULE. What early application does cost is cycle fidelity: the
            //     reader really would have stalled. Each pair's reads are static, the incoming ready
            //     cycles are the only dynamic input, so the block's issue cycles are shifted below.
            //   * BUMP m_vfLatestWrite afterwards for every slot the block WRITES, so a write still
            //     sitting in the pipeline cannot commit later and clobber the block's newer value
            //     (this is the interpreter's own supersession rule, applied on the block's behalf).
            if (guardOk && s_vu1BlockStrict)
            {
                for (const PendingVfWrite &w : m_vfWritePipeline)
                    if (w.valid && (blk->vfTouched[w.reg & 31u] & w.laneMask & 0xFu) != 0u)
                    { guardOk = false; ++bstore.gbVf; break; }
            }
            if (guardOk)
            {
                uint32_t m = blockViMask;
                while (m != 0u)
                {
                    const uint32_t i = uint32_t(__builtin_ctz(m));
                    m &= m - 1u;
                    const PendingViWrite &w = m_viWritePipeline[i];
                    if (w.valid && (blk->viTouched & (1u << (w.reg & 15u))) != 0u)
                    { guardOk = false; ++bstore.gbVi; break; }
                }
            }
            if (guardOk)
            {
                uint32_t m = blockAccMask;
                while (m != 0u)
                {
                    const uint32_t i = uint32_t(__builtin_ctz(m));
                    m &= m - 1u;
                    const PendingAccWrite &w = m_accWritePipeline[i];
                    if (w.valid && (blk->accTouched & w.laneMask & 0xFu) != 0u)
                    { guardOk = false; ++bstore.gbAcc; break; }
                }
            }
            if (guardOk && blk->touchesVuMem)
            {
                // A pending store lands in VU memory mid-block, which an LQ would read and an SQ
                // would race with.
                if (blockStoreMask != 0u)
                { guardOk = false; ++bstore.gbStore; }
            }
            // ★★ cont.201: clip and Q are the two values whose visibility is NOT protected by a
            // stall -- `calculatePairReadyCycle` ignores `readsClip` entirely and only stalls on Q
            // for FDIV-pipeline/waitQ ops -- so a pending write to either cannot simply be applied
            // early the way VF writes can. But it CAN be applied when it would have committed
            // before the block's first reader of that value anyway, which is a static issue cycle.
            // Rejecting instead was costing 39.6M of 41.3M guard blocks (clip readers follow CLIP
            // closely in cull code, so a clip write is almost always in flight).
            if (guardOk && blk->readsQ && m_fdiv.valid)
            {
                const uint64_t due = m_cycle + uint64_t(blk->firstQIssue);
                if (m_fdiv.readyCycle > due)
                { guardOk = false; ++bstore.gbQ; }
                else
                    blockApplyQ = true;
            }
            if (guardOk && blk->readsClip)
            {
                const uint64_t due = m_cycle + uint64_t(blk->firstClipIssue);
                uint32_t m = blockFlagMask;
                while (m != 0u)
                {
                    const uint32_t i = uint32_t(__builtin_ctz(m));
                    m &= m - 1u;
                    const FlagPipelineEntry &fe = m_flagPipeline[i];
                    if (fe.valid && fe.writesClip && fe.readyCycle > due)
                    { guardOk = false; ++bstore.gbClip; break; }
                }
                if (guardOk)
                    blockApplyClip = true;
            }
            if (blk != nullptr && !guardOk)
                ++bstore.guardBlocked;
            if (guardOk && m_cycle + blk->cycles + 64u > budgetEnd)
                ++bstore.gbBudget;
            if (guardOk && m_cycle + blk->cycles + 64u <= budgetEnd)
            {
                // Re-derive the issue schedule against the incoming ready tables. `shift` is
                // monotonic and the static schedule already encodes the block's internal
                // dependencies, so shifting preserves them and one pass suffices.
                const unsigned long long bpT1 = s_vu1BlockProfSeg ? __rdtsc() : 0ull;
                if (s_vu1BlockProfSeg) g_bpLookup += bpT1 - bpT0;
                uint32_t dynIssue[vu1jit::kMaxBlockPairs];
                if (blockNoPending)
                {
                    // Nothing in flight: the static schedule is already exact.
                    for (uint32_t j = 0; j < blk->pairs; ++j)
                        dynIssue[j] = blk->sched[j].issue;
                }
                else
                {
                    uint32_t shift = 0;
                    for (uint32_t j = 0; j < blk->pairs; ++j)
                    {
                        const vu1jit::PairSched &ps = blk->sched[j];
                        uint32_t want = uint32_t(ps.issue) + shift;
                        const auto bump = [&](uint64_t ready) {
                            if (ready > m_cycle)
                            {
                                const uint32_t rel = uint32_t(ready - m_cycle);
                                if (rel > want)
                                    want = rel;
                            }
                        };
                        for (uint32_t k = 0; k < ps.vfReadCount; ++k)
                        {
                            uint32_t lanes = ps.vfReadLanes[k];
                            while (lanes != 0u)
                            {
                                const uint32_t bit = uint32_t(__builtin_ctz(lanes));
                                lanes &= lanes - 1u;
                                bump(m_vfReady[ps.vfReadReg[k]][3u - bit]);
                            }
                        }
                        uint32_t vr = ps.viRead;
                        while (vr != 0u)
                        {
                            const uint32_t reg = uint32_t(__builtin_ctz(vr));
                            vr &= vr - 1u;
                            bump(m_viReady[reg]);
                        }
                        uint32_t ar = ps.accRead;
                        while (ar != 0u)
                        {
                            const uint32_t bit = uint32_t(__builtin_ctz(ar));
                            ar &= ar - 1u;
                            bump(m_accReady[3u - bit]);
                        }
                        shift = want - uint32_t(ps.issue);
                        dynIssue[j] = want;
                    }
                }
                const unsigned long long bpT2 = s_vu1BlockProfSeg ? __rdtsc() : 0ull;
                if (s_vu1BlockProfSeg) g_bpSched += bpT2 - bpT1;
                const uint32_t dynCycles = dynIssue[blk->pairs - 1u] + 1u;
                if (dynCycles > blk->cycles)
                    ++bstore.rescheduled;
                // ★★★ cont.230: the ONE preamble that honours the entry guard's promises, run by
                // BOTH the production block (on m_state) and the shadow-verify copy
                // (blockVerifyRef). It used to be two copies, and only the second applied the clip
                // and Q promises -- which is how the level-era dark-wedge artifact hid behind
                // "0 mismatches" for ten cycles. Anything the guard admits early MUST be applied
                // here and nowhere else; the counters are the census of how often each fires.
                const auto applyGuardAdmitted = [&](VU1State &dst)
                {
                    // Pending VF writes the block touches (supersession rule: only the latest write
                    // to a lane is live). Walks only the VALID slots via the PipeTrack bitmask --
                    // scanning all 16 PendingVfWrite entries on every entry was itself the cost that
                    // made cont.196's first cut slower than the block it enabled.
                    if (blockVfMask != 0u)
                    {
                        bool applied = false;
                        uint32_t m = blockVfMask;
                        while (m != 0u)
                        {
                            const uint32_t i = uint32_t(__builtin_ctz(m));
                            m &= m - 1u;
                            const PendingVfWrite &w = m_vfWritePipeline[i];
                            if (!w.valid)
                                continue;
                            uint32_t lanes =
                                uint32_t(blk->vfTouched[w.reg & 31u] & w.laneMask & 0xFu);
                            while (lanes != 0u)
                            {
                                const uint32_t bit = uint32_t(__builtin_ctz(lanes));
                                lanes &= lanes - 1u;
                                const uint32_t c = 3u - bit;
                                if (m_vfLatestWrite[w.reg][c] == w.sequence)
                                {
                                    dst.vf[w.reg][c] = w.value[c];
                                    applied = true;
                                }
                            }
                        }
                        if (applied)
                            ++bstore.earlyVf;
                    }
                    // Pending CLIP-flag write(s) the guard proved committed before the block's first
                    // clip reader. The latest-ISSUED entry carries the fully shifted 24-bit register
                    // (queueClip chains m_workingClip); slot order is NOT issue order.
                    if (blockApplyClip)
                    {
                        uint64_t latestIssue = 0;
                        bool any = false;
                        uint32_t latestClip = dst.clip;
                        uint32_t m = blockFlagMask;
                        while (m != 0u)
                        {
                            const uint32_t i = uint32_t(__builtin_ctz(m));
                            m &= m - 1u;
                            const FlagPipelineEntry &fe = m_flagPipeline[i];
                            if (fe.valid && fe.writesClip && (!any || fe.issueCycle >= latestIssue))
                            {
                                latestIssue = fe.issueCycle;
                                latestClip = fe.clip;
                                any = true;
                            }
                        }
                        if (any)
                        {
                            dst.clip = latestClip;
                            ++bstore.earlyClip;
                        }
                    }
                    // Pending FDIV result the guard proved committed before the block's first Q reader.
                    if (blockApplyQ && m_fdiv.valid)
                    {
                        dst.q = m_fdiv.value;
                        ++bstore.earlyQ;
                    }
                };
                if (!s_vu1BlockVerify)
                {
                    const uint8_t tailReg = blk->tailViBackupReg;
                    const int32_t tailOld = tailReg != 0u ? m_state.vi[tailReg] : 0;

                    applyGuardAdmitted(m_state);

                    const unsigned long long bpT3 = s_vu1BlockProfSeg ? __rdtsc() : 0ull;
                    if (s_vu1BlockProfSeg) g_bpApply += bpT3 - bpT2;

                    blk->fn(&m_state, vuData, dataSize - 1u);

                    const unsigned long long bpT4 = s_vu1BlockProfSeg ? __rdtsc() : 0ull;
                    // cont.212: `flagged` = this block replays MAC flags per FMAC in retire.
                    const bool bpFlagged =
                        s_vu1BlockProf && blk->emitsFlags && blk->flagCount != 0u;
                    if (s_vu1BlockProfSeg)
                    {
                        const unsigned long long fnCy = bpT4 - bpT3;
                        g_bpFn += fnCy;
                        if (bpFlagged) g_bpFnF += fnCy; else g_bpFnC += fnCy;
                    }
                    if (s_vu1BlockProf)
                    {
                        ++g_bpEntries;
                        // Probe-free counters: which driver paths actually run, and how long the
                        // blocks are. These cost an increment, so they cannot measure themselves.
                        if (blockNoPending) ++g_bpNoPending;
                        if (blockVfMask != 0u) ++g_bpSupersede;
                        g_bpPairsHist[blk->pairs < 9u ? blk->pairs : 8u]++;
                        const unsigned sr = blk->stopReason < 12u ? blk->stopReason : 11u;
                        ++g_bpStop[sr];
                        if (blk->pairs <= 2u)
                            ++g_bpStop2[sr];
                        if (sr == 5u)
                            ++g_bpStop5Ops[vu1LowerOpKey(blk->stopLower)];
                    }

                    // cont.212 differential check: snapshot every table the replay touches.
                    static thread_local std::array<std::array<uint64_t, 4>, 32> rvVf0{}, rvVfC{},
                                                                                rvLw0{}, rvLwC{};
                    static thread_local std::array<uint64_t, 16> rvVi0{}, rvViC{};
                    static thread_local std::array<uint64_t, 4> rvAcc0{}, rvAccC{};
                    static thread_local uint64_t rvSeq0 = 0, rvSeqC = 0;
                    if (s_vu1ReadyVerify)
                    {
                        rvVf0 = m_vfReady;
                        rvVi0 = m_viReady;
                        rvAcc0 = m_accReady;
                        rvLw0 = m_vfLatestWrite;
                        rvSeq0 = m_nextWriteSequence;
                    }

                    // Supersede any pending write to a slot the block just wrote, so it cannot
                    // commit later and overwrite the newer value. Only needed while something is
                    // actually in flight, and driven off the block's own write list rather than a
                    // 32-register scan.
                    if (blockVfMask != 0u)
                    {
                        for (uint32_t k = 0; k < blk->readyCount; ++k)
                        {
                            const vu1jit::ReadyWrite &rw = blk->ready[k];
                            if (rw.kind != 0u)
                                continue;
                            // cont.212 masked record. The per-lane records it replaces were emitted
                            // by a ctz walk (bit b -> lane 3-b), i.e. lanes 3,2,1,0 for a full
                            // dest, so stamping in that order keeps the sequence numbers identical.
                            if (rw.lanes == 0xFu)
                            {
                                m_vfLatestWrite[rw.reg][3] = ++m_nextWriteSequence;
                                m_vfLatestWrite[rw.reg][2] = ++m_nextWriteSequence;
                                m_vfLatestWrite[rw.reg][1] = ++m_nextWriteSequence;
                                m_vfLatestWrite[rw.reg][0] = ++m_nextWriteSequence;
                            }
                            else
                            {
                                uint32_t m = rw.lanes;
                                while (m != 0u)
                                {
                                    const uint32_t bit = uint32_t(__builtin_ctz(m));
                                    m &= m - 1u;
                                    m_vfLatestWrite[rw.reg][3u - bit] = ++m_nextWriteSequence;
                                }
                            }
                        }
                    }

                    // Replay the ready-table updates markPairWrites would have made, in pair
                    // order (it SETS rather than maxes, so order reproduces it exactly).
                    const uint64_t entry = m_cycle;
                    for (uint32_t k = 0; k < blk->readyCount; ++k)
                    {
                        const vu1jit::ReadyWrite &rw = blk->ready[k];
                        const uint64_t rc = entry + dynIssue[rw.pair] + rw.latency;
                        // cont.212: one record per write-group. The per-lane records it replaces
                        // were ADJACENT in the list, so relative order with everything else is
                        // unchanged, and every lane in a group takes the same `rc`.
                        if (rw.kind == 1u)
                        {
                            m_viReady[rw.reg] = rc;
                            continue;
                        }
                        uint64_t *row = (rw.kind == 0u) ? m_vfReady[rw.reg].data()
                                                        : m_accReady.data();
                        if (rw.lanes == 0xFu)
                        {
                            row[0] = rc;
                            row[1] = rc;
                            row[2] = rc;
                            row[3] = rc;
                        }
                        else
                        {
                            uint32_t m = rw.lanes;
                            while (m != 0u)
                            {
                                const uint32_t bit = uint32_t(__builtin_ctz(m));
                                m &= m - 1u;
                                row[3u - bit] = rc;
                            }
                        }
                    }
                    if (s_vu1ReadyVerify)
                    {
                        // Keep what the coalesced replay produced, rewind, and replay the list in
                        // its pre-cont.212 shape: every kind-3 record expanded back to the four
                        // per-lane records recordPairWrites used to emit, in ITS order (the ctz
                        // walk maps bit b to lane 3-b, so lanes 3,2,1,0).
                        rvVfC = m_vfReady;
                        rvViC = m_viReady;
                        rvAccC = m_accReady;
                        rvLwC = m_vfLatestWrite;
                        rvSeqC = m_nextWriteSequence;
                        m_vfReady = rvVf0;
                        m_viReady = rvVi0;
                        m_accReady = rvAcc0;
                        m_vfLatestWrite = rvLw0;
                        m_nextWriteSequence = rvSeq0;
                        if (blockVfMask != 0u)
                        {
                            for (uint32_t k = 0; k < blk->readyCount; ++k)
                            {
                                const vu1jit::ReadyWrite &rw = blk->ready[k];
                                if (rw.kind != 0u)
                                    continue;
                                // Reference expansion, derived DIFFERENTLY from production on
                                // purpose: walk lanes 3..0 and bit-test, where production walks the
                                // mask with ctz. If the bit<->lane mapping were wrong in one, the
                                // two would disagree.
                                for (int lane = 3; lane >= 0; --lane)
                                    if (rw.lanes & (1u << (3 - lane)))
                                        m_vfLatestWrite[rw.reg][lane] = ++m_nextWriteSequence;
                            }
                        }
                        for (uint32_t k = 0; k < blk->readyCount; ++k)
                        {
                            const vu1jit::ReadyWrite &rw = blk->ready[k];
                            const uint64_t rc2 = entry + dynIssue[rw.pair] + rw.latency;
                            if (rw.kind == 1u)
                                m_viReady[rw.reg] = rc2;
                            else
                                for (int lane = 3; lane >= 0; --lane)
                                    if (rw.lanes & (1u << (3 - lane)))
                                    {
                                        if (rw.kind == 0u)
                                            m_vfReady[rw.reg][lane] = rc2;
                                        else
                                            m_accReady[lane] = rc2;
                                    }
                        }
                        ++g_rvChecked;
                        if (m_vfReady != rvVfC || m_viReady != rvViC || m_accReady != rvAccC ||
                            m_vfLatestWrite != rvLwC || m_nextWriteSequence != rvSeqC)
                        {
                            if (++g_rvBad <= 8u)
                                std::fprintf(stderr,
                                             "[vu1:readyverify] MISMATCH #%llu pc=0x%x pairs=%u"
                                             " ready=%u vfReady=%d viReady=%d accReady=%d"
                                             " latest=%d seq=%d\n",
                                             g_rvBad, m_state.pc, blk->pairs, blk->readyCount,
                                             m_vfReady != rvVfC, m_viReady != rvViC,
                                             m_accReady != rvAccC, m_vfLatestWrite != rvLwC,
                                             m_nextWriteSequence != rvSeqC);
                        }
                        else
                        {
                            // Identical: either state is correct, keep the production one.
                            m_vfReady = rvVfC;
                            m_viReady = rvViC;
                            m_accReady = rvAccC;
                            m_vfLatestWrite = rvLwC;
                            m_nextWriteSequence = rvSeqC;
                        }
                    }
                    // cont.220: the flag pipeline is also touched between blocks, so the
                    // minimum is only trusted within one entry's replay.
                    flagMinReady = 0;
                    // ---- cont.217 FLAGVERIFY: snapshot everything the replay can touch ------
                    const bool fvOn = s_vu1FlagVerify && !s_vu1BlockVerify &&
                                      (blk->clipCount != 0u ||
                                       (blk->emitsFlags && blk->flagCount != 0u));
                    if (fvOn)
                    {
                        fvSnapPipe = m_flagPipeline;
                        fvSnapMac = m_state.mac;
                        fvSnapStatus = m_state.status;
                        fvSnapClip = m_state.clip;
                        fvSnapWorking = m_workingClip;
                        PipeTrack &fvPipe = pipeFor(this);
                        fvSnapMask = fvPipe.flag;
                        // Reuse the existing per-insertion recorders to capture the event stream
                        // production actually queues (values are already covered by the BLOCKVERIFY
                        // flag check; what is under test here is their SCHEDULING).
                        g_ccActive = true;
                        g_fcActive = true;
                        g_ccN = 0;
                        g_fcN = 0;
                    }
                    // ★ Replay each CLIP's queueClip at its own issue cycle, in pair order --
                    // m_workingClip shifts, so order is load-bearing.
                    if (blk->clipCount != 0u)
                    {
                        const uint64_t saveCycle = m_cycle;
                        for (uint32_t k = 0; k < blk->clipCount; ++k)
                        {
                            const auto &cw = blk->clipWrites[k];
                            float vs[4], vt[4];
                            std::memcpy(vs, vu1jit::blockBuffer(unitIdx).resultSlot(cw.slot), 16);
                            std::memcpy(vt, vu1jit::blockBuffer(unitIdx).resultSlot(cw.slot + 1u), 16);
                            m_cycle = entry + dynIssue[cw.pair];
                            if (s_vu1FlagMutate != 1)
                                retireDueFlagEntries(); // cont.216: free the slots due by now
                            queueClip(vuClipFlagsFrom(vs, vt));
                            // cont.220: queueClip just inserted at m_cycle + kFmacLatency.
                            if (m_cycle + kFmacLatency < flagMinReady)
                                flagMinReady = m_cycle + kFmacLatency;
                        }
                        m_cycle = saveCycle;
                    }
                    // ★ Replay each FMAC's MAC flags at ITS OWN issue cycle. updateFmacFlags reads
                    // m_cycle only for the entry's issue/ready cycles, so setting it per FMAC
                    // reproduces exactly the entries the interpreter would have queued, in order.
                    if (blk->emitsFlags && blk->flagCount != 0u)
                    {
                        const uint64_t saveCycle = m_cycle;
                        // PS2X_VU1_FLAGMUTATE (default 0) injects defects here; see the flag.
                        const uint32_t fmN = (s_vu1FlagMutate == 2 && blk->flagCount != 0u)
                                                 ? blk->flagCount - 1u
                                                 : blk->flagCount;
                        for (uint32_t i = 0; i < fmN; ++i)
                        {
                            const uint32_t k = (s_vu1FlagMutate == 3) ? (fmN - 1u - i) : i;
                            const auto &fw = blk->flagWrites[k];
                            float res[4];
                            std::memcpy(res, vu1jit::blockBuffer(unitIdx).resultSlot(fw.slot), 16);
                            uint8_t laneFlags[4]{};
                            vuNormResultQuad(res, fw.dest, laneFlags);
                            m_cycle = entry + dynIssue[fw.pair] + (s_vu1FlagMutate == 4 ? 1u : 0u);
                            if (s_vu1FlagMutate != 1)
                                retireDueFlagEntries(); // cont.216: free the slots due by now
                            updateFmacFlags(laneFlags, fw.dest, 0u);
                            // cont.220: updateFmacFlags just inserted at m_cycle + kFmacLatency.
                            if (m_cycle + kFmacLatency < flagMinReady)
                                flagMinReady = m_cycle + kFmacLatency;
                        }
                        m_cycle = saveCycle;
                    }
                    // ---- cont.217 FLAGVERIFY: replay the same events the interpreter's way -----
                    if (fvOn)
                    {
                        g_ccActive = false;
                        g_fcActive = false;
                        ++g_fvChecked;
                        // Capture what production produced, then rewind to the snapshot.
                        // Two copies: one to drain to the checkpoint for the comparison, one
                        // pristine, because verification must never alter execution.
                        fvProdPipe = m_flagPipeline;
                        fvKeepPipe = m_flagPipeline;
                        fvProdMac = m_state.mac;
                        fvProdStatus = m_state.status;
                        fvProdClip = m_state.clip;
                        fvProdWorking = m_workingClip;
                        fvKeepMac = m_state.mac;
                        fvKeepStatus = m_state.status;
                        fvKeepClip = m_state.clip;

                        // ★★ cont.217 mutation-test finding: building the reference's event list
                        // from the RECORDERS made it inherit production's own count and cycles, so
                        // "drop an event" and "issue it a cycle late" both SURVIVED. The count and
                        // the cycles must come from the BLOCK's own description instead; only the
                        // VALUES are taken from the recorders (those are already covered by the
                        // BLOCKVERIFY flag/clip checks). Same gap cont.194 found in the JIT
                        // self-tests: calling the producer directly proves nothing about which
                        // field feeds which argument.
                        const uint32_t fvExpClip = blk->clipCount;
                        const uint32_t fvExpMac = blk->emitsFlags ? blk->flagCount : 0u;
                        const bool countOk = (g_ccN == fvExpClip) && (g_fcN == fvExpMac);
                        bool cycleOk = true;
                        uint32_t n = 0;
                        for (uint32_t k = 0; k < fvExpClip && n < 64u; ++k)
                        {
                            const uint64_t derived = entry + dynIssue[blk->clipWrites[k].pair];
                            if (k < g_ccN && g_ccIssue[k] != derived)
                                cycleOk = false;
                            fvEv[n++] = FvEvent{derived, k < g_ccN ? g_ccClip[k] : 0u, 0u, true};
                        }
                        for (uint32_t k = 0; k < fvExpMac && n < 64u; ++k)
                        {
                            const uint64_t derived = entry + dynIssue[blk->flagWrites[k].pair];
                            if (k < g_fcN && g_fcIssue[k] != derived)
                                cycleOk = false;
                            fvEv[n++] = FvEvent{derived, k < g_fcN ? g_fcMac[k] : 0u,
                                                k < g_fcN ? g_fcStatus[k] : 0u, false};
                        }
                        g_fvEvents += n;
                        // Ascending issue cycle == the order the interpreter issues pairs in. A
                        // pair has ONE upper, so a clip and a MAC event can never share a cycle and
                        // the order is total. Insertion sort; n <= 64.
                        for (uint32_t i = 1; i < n; ++i)
                        {
                            const FvEvent key = fvEv[i];
                            int32_t j = int32_t(i) - 1;
                            while (j >= 0 && fvEv[j].cycle > key.cycle)
                            {
                                fvEv[j + 1] = fvEv[j];
                                --j;
                            }
                            fvEv[j + 1] = key;
                        }

                        fvRefPipe = fvSnapPipe;
                        uint32_t rMac = fvSnapMac, rStatus = fvSnapStatus, rClip = fvSnapClip;
                        uint32_t rWorking = fvSnapWorking;
                        bool overflow = false;
                        for (uint32_t k = 0; k < n; ++k)
                        {
                            const FvEvent &ev = fvEv[k];
                            // Retire by FULL linear scan, ascending slot -- no mask, no ctz.
                            for (uint32_t i = 0; i < kMaxFlagEntries; ++i)
                            {
                                if (!fvRefPipe[i].valid || fvRefPipe[i].readyCycle > ev.cycle)
                                    continue;
                                fvApply(fvRefPipe[i], rMac, rStatus, rClip);
                                fvRefPipe[i] = {};
                            }
                            uint32_t slot = kMaxFlagEntries;
                            for (uint32_t i = 0; i < kMaxFlagEntries; ++i)
                                if (!fvRefPipe[i].valid) { slot = i; break; }
                            if (slot == kMaxFlagEntries) { overflow = true; break; }
                            FlagPipelineEntry &e = fvRefPipe[slot];
                            e = {};
                            e.valid = true;
                            e.issueCycle = ev.cycle;
                            e.readyCycle = ev.cycle + kFmacLatency;
                            if (ev.isClip)
                            {
                                // Independently recompute the shift chain from the RAW clip bits.
                                rWorking = ((rWorking << 6) | (ev.a & 0x3Fu)) & 0xFFFFFFu;
                                e.clip = rWorking;
                                e.writesClip = true;
                            }
                            else
                            {
                                e.mac = ev.a;
                                e.status = ev.b;
                                e.writesMac = true;
                                e.writesStatus = true;
                            }
                        }

                        // Count the pre-checkpoint difference separately: it is production's
                        // legitimate deferral, and worth knowing the rate of, but it is not a fault.
                        static FlagPipelineEntry fvLiveP[kMaxFlagEntries], fvLiveR[kMaxFlagEntries];
                        if (fvLive(fvProdPipe, fvLiveP) != fvLive(fvRefPipe, fvLiveR) ||
                            rMac != fvProdMac || rStatus != fvProdStatus || rClip != fvProdClip)
                            ++g_fvDeferred;
                        // Both sides to the same checkpoint: the commit the next loop iteration
                        // performs at m_cycle = entry + dynCycles.
                        const uint64_t fvCheckpoint = entry + dynCycles;
                        fvDrainDue(fvProdPipe, fvProdMac, fvProdStatus, fvProdClip, fvCheckpoint);
                        fvDrainDue(fvRefPipe, rMac, rStatus, rClip, fvCheckpoint);
                        const uint32_t np = fvLive(fvProdPipe, fvLiveP);
                        const uint32_t nr = fvLive(fvRefPipe, fvLiveR);
                        bool same = !overflow && countOk && cycleOk && np == nr &&
                                    rMac == fvProdMac && rStatus == fvProdStatus &&
                                    rClip == fvProdClip && rWorking == fvProdWorking;
                        for (uint32_t i = 0; same && i < np; ++i)
                            same = fvLiveP[i].issueCycle == fvLiveR[i].issueCycle &&
                                   fvLiveP[i].readyCycle == fvLiveR[i].readyCycle &&
                                   fvLiveP[i].mac == fvLiveR[i].mac &&
                                   fvLiveP[i].status == fvLiveR[i].status &&
                                   fvLiveP[i].clip == fvLiveR[i].clip &&
                                   fvLiveP[i].writesMac == fvLiveR[i].writesMac &&
                                   fvLiveP[i].writesStatus == fvLiveR[i].writesStatus &&
                                   fvLiveP[i].writesClip == fvLiveR[i].writesClip;
                        if (overflow)
                            ++g_fvOverflow;
                        // Periodic, because a timeout-killed run never reaches the exit summary --
                        // and "0 mismatches" is worthless without knowing how much was checked.
                        if ((g_fvChecked % 100000ull) == 1ull)
                            std::fprintf(stderr,
                                         "[vu1:flagverify] progress checked=%llu events=%llu"
                                         " mismatches=%llu overflow=%llu deferred=%llu\n",
                                         g_fvChecked, g_fvEvents, g_fvBad, g_fvOverflow,
                                         g_fvDeferred);
                        if (!same && ++g_fvBad <= 8ull)
                            std::fprintf(stderr,
                                         "[vu%u:flagverify] MISMATCH #%llu pc=0x%x pairs=%u ev=%u"
                                         " (clip=%u/%u mac=%u/%u) overflow=%d count=%d cycle=%d"
                                         " live=%u/%u mac=%d status=%d clip=%d working=%d\n",
                                         unitIdx, g_fvBad, m_state.pc, blk->pairs, n,
                                         g_ccN, fvExpClip, g_fcN, fvExpMac,
                                         overflow ? 1 : 0, !countOk, !cycleOk, np, nr,
                                         rMac != fvProdMac, rStatus != fvProdStatus,
                                         rClip != fvProdClip, rWorking != fvProdWorking);
                        // Verification must never alter execution: the PRISTINE production
                        // result stands (fvProdPipe was drained for the comparison).
                        m_flagPipeline = fvKeepPipe;
                        m_state.mac = fvKeepMac;
                        m_state.status = fvKeepStatus;
                        m_state.clip = fvKeepClip;
                        m_workingClip = fvProdWorking;
                    }
                    m_cycle += dynCycles;
                    m_state.cycles = m_cycle;
                    // ★★ the cont.194 branch-delay contract: a branch in the pair immediately
                    // after the block must still read the OLD value of a VI register the block's
                    // final pair wrote (planBlock guarantees no earlier pair wrote it).
                    m_viBranchBackupValid = false;
                    if (tailReg != 0u)
                        recordViWriteForBranch(tailReg, tailOld);

                    const uint32_t blkEntryPc = m_state.pc;
                    if (!blk->endsWithBranch)
                    {
                        uint32_t nextPc = m_state.pc + 8u * blk->pairs;
                        if (nextPc >= codeSize)
                            nextPc = 0u;
                        m_state.pc = nextPc;
                    }
                    if (s_vuLastBlk)
                    {
                        LastBlk *ring = g_lastBlk[unitIdx];
                        const unsigned k = g_lastBlkN[unitIdx] & 3u;
                        ring[k] = LastBlk{blkEntryPc, blk->pairs, blk->endsWithBranch ? 1u : 0u,
                                          m_state.pc, uint32_t(dynCycles)};
                        ++g_lastBlkN[unitIdx];
                    }
                    // else: the block executed the branch and its delay slot and wrote pc itself.

                    if (s_vu1BlockProf)
                        g_bpWhole += __rdtsc() - bpT0;
                    if (s_vu1BlockProfSeg)
                    {
                        const unsigned long long rtCy = __rdtsc() - bpT4;
                        g_bpRetire += rtCy;
                        if (bpFlagged) { g_bpRetireF += rtCy; ++g_bpEntF; g_bpPairsF += blk->pairs; }
                        else           { g_bpRetireC += rtCy; ++g_bpEntC; g_bpPairsC += blk->pairs; }
                        g_bpReadyN += blk->readyCount;
                        g_bpClipN += blk->clipCount;
                        g_bpFlagN += blk->emitsFlags ? blk->flagCount : 0u;
                    }
                    ++bstore.entered;
                    bstore.pairsRun += blk->pairs;
                    perfPairs += blk->pairs - 1u; // one was already counted at the loop top
                    if (s_vu1LinkCensus)
                    {
                        const uint32_t nextPair = m_state.pc / 8u;
                        uint32_t *sc = g_lcSucc[unitIdx][blockIdx > 0 ? blockIdx : 0];
                        if (sc[0] != nextPair && sc[1] != nextPair)
                        {
                            sc[1] = sc[0];
                            sc[0] = nextPair;
                        }
                        g_lcPrevIdx[unitIdx] = blockIdx;
                        g_lcPrevWasBlock[unitIdx] = true;
                    }
                    continue;
                }
                // ---- shadow-verify (PS2X_VU1_BLOCKVERIFY=1) --------------------------------
                // ★ Never start a nested verification: the pair after a block entry usually has a
                // compiled block of its own, and overwriting the reference mid-window compares the
                // wrong two states. (That alone produced 2.67M bogus mismatches.)
                if (blockVerifyLeft != 0u)
                    goto blockVerifyBusy;
                // The block runs on a COPY; the interpreter still does the real execution, so
                // verify mode can never corrupt the run. The comparison happens `pairs` pairs
                // later, skipping every slot the interpreter still has a write in flight for —
                // those legitimately lag, because the block applies its writes immediately.
                blockVerifyRef = m_state;
                applyGuardAdmitted(blockVerifyRef);
                blockVerifyCycles = dynCycles;
                blockVerifyEntryCycle = m_cycle;
                // ★ Slots with a write already in flight at block ENTRY are not comparable: the
                // block cannot see them (the guard guarantees it never touches them), while the
                // interpreter commits them during the window. Record them so the comparison skips
                // them -- otherwise every such commit reads as a mismatch.
                std::memset(blockEntryPendingVf, 0, sizeof(blockEntryPendingVf));
                blockEntryPendingVi = 0u;
                blockEntryPendingAcc = 0u;
                blockEntryPendingStore = false;
                for (const PendingVfWrite &w : m_vfWritePipeline)
                    if (w.valid)
                        blockEntryPendingVf[w.reg & 31u] |= uint8_t(w.laneMask & 0xFu);
                for (const PendingViWrite &w : m_viWritePipeline)
                    if (w.valid)
                        blockEntryPendingVi |= uint32_t(1u << (w.reg & 15u));
                for (const PendingAccWrite &w : m_accWritePipeline)
                    if (w.valid)
                        blockEntryPendingAcc |= uint8_t(w.laneMask & 0xFu);
                for (const PendingStore &st : m_storePipeline)
                    if (st.valid)
                        blockEntryPendingStore = true;
                blockEntryFlagPending = false;
                for (const FlagPipelineEntry &fe : m_flagPipeline)
                    if (fe.valid)
                        blockEntryFlagPending = true;
                blockEntryFdivPending = m_fdiv.valid;
                if (blk->touchesVuMem)
                {
                    if (blockVerifyMem.size() < dataSize)
                        blockVerifyMem.resize(dataSize);
                    std::memcpy(blockVerifyMem.data(), vuData, dataSize);
                    blk->fn(&blockVerifyRef, blockVerifyMem.data(), dataSize - 1u);
                }
                else
                {
                    blk->fn(&blockVerifyRef, vuData, dataSize - 1u);
                }
                // Derive what the driver's replay WOULD queue, then capture what the interpreter
                // actually queues over the same pairs, and compare at the end of the window.
                blockFlagN = 0;
                if (blk->emitsFlags)
                {
                    for (uint32_t k = 0; k < blk->flagCount && blockFlagN < 32u; ++k)
                    {
                        const auto &fw = blk->flagWrites[k];
                        float res[4];
                        std::memcpy(res, vu1jit::blockBuffer(unitIdx).resultSlot(fw.slot), 16);
                        uint8_t laneFlags[4]{};
                        vuNormResultQuad(res, fw.dest, laneFlags);
                        if (fw.dest == 0u)
                            continue;
                        uint32_t mac = 0u, status = 0u;
                        for (uint32_t c = 0; c < 4u; ++c)
                        {
                            const uint8_t lane = laneForComponent(c);
                            if ((fw.dest & lane) == 0u)
                                continue;
                            const uint32_t fl = laneFlags[c];
                            if (fl & 0x1u) mac |= lane;
                            if (fl & 0x2u) mac |= uint32_t(lane) << 4;
                            if (fl & 0x4u) mac |= uint32_t(lane) << 8;
                            if (fl & 0x8u) mac |= uint32_t(lane) << 12;
                            status |= fl;
                        }
                        blockFlagMacArr[blockFlagN] = mac;
                        blockFlagStatusArr[blockFlagN] = status;
                        blockFlagIssueArr[blockFlagN] = m_cycle + dynIssue[fw.pair];
                        ++blockFlagN;
                    }
                }
                blockClipN = 0;
                for (uint32_t k = 0; k < blk->clipCount && blockClipN < 32u; ++k)
                {
                    const auto &cw = blk->clipWrites[k];
                    float vs[4], vt[4];
                    std::memcpy(vs, vu1jit::blockBuffer(unitIdx).resultSlot(cw.slot), 16);
                    std::memcpy(vt, vu1jit::blockBuffer(unitIdx).resultSlot(cw.slot + 1u), 16);
                    blockClipRaw[blockClipN] = vuClipFlagsFrom(vs, vt);
                    blockClipIssueArr[blockClipN] = m_cycle + dynIssue[cw.pair];
                    ++blockClipN;
                }
                g_fcActive = true;
                g_fcN = 0;
                g_ccActive = true;
                g_ccN = 0;
                blockVerifyBlk = blk;
                blockVerifyLeft = blk->pairs;
                ++bstore.entered;
                bstore.pairsRun += blk->pairs;
            }
            }
        }
        blockNoEntry:;
        blockVerifyBusy:;

        if (s_vu1LinkCensus && g_lcPrevWasBlock[unitIdx])
        {
            // Reaching the interpreted path means the previous block's exit landed on a pair with
            // no compiled block -- the chain is broken and no link could have been taken.
            ++g_lcToInterp[unitIdx];
            g_lcPrevWasBlock[unitIdx] = false;
        }
        const uint32_t pairIndex = m_state.pc / 8u;
        const DecodedInstructionPair *decodedPtr;
        if (decodeCache != nullptr && (m_state.pc & 7u) == 0u && pairIndex < kMaxDecodedPairs)
        {
            decodedPtr = decodeCache + pairIndex;
        }
        else
        {
            decodedFallback = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
            decodedPtr = &decodedFallback;
        }
        const DecodedInstructionPair &decoded = *decodedPtr;
        // ★ CROSS-PROGRAM SAFETY GUARD (always on when lazy flags are enabled). The scan proves
        // the flags are unobservable WITHIN a program, but m_state.mac/status persist ACROSS
        // programs, so a reader in a LATER program could observe flags an earlier fast-path
        // program skipped writing. Reader-containing programs are real in this workload (343 of
        // 2078), so this is not hypothetical. The exact dangerous condition is checked here, at
        // the only place it can matter: a MAC/status reader ISSUING while (a) the flags are stale
        // from a skipped write and (b) nothing in THIS run has written them yet. It latches the
        // optimization off for the rest of the session and reports once, so exposure is bounded to
        // the single instruction that detects it and the condition can never be silent.
        if (s_vu1JitCensus && m_unit == Unit::VU1)
        {
            ++g_jitPairs;
            const uint8_t uop = static_cast<uint8_t>(decoded.upper & 0x3Fu);
            ++g_jitUpper[uop];
            if (uop >= 0x3Cu)
                ++g_jitUpperSpecial[(decoded.upper & 3u) | ((decoded.upper >> 4) & 0x7Cu)];
            if (decoded.iBit)
            {
                ++g_jitIBit;
            }
            else
            {
                const uint8_t lop = static_cast<uint8_t>((decoded.lower >> 25) & 0x7Fu);
                ++g_jitLower[lop];
                // 0x20..0x2F is the branch/jump block in the lower opcode map: these are the
                // instructions that TERMINATE a JIT basic block.
                if (lop >= 0x20u && lop <= 0x2Fu)
                    ++g_jitBranches;
            }
            if (decoded.eBit)
                ++g_jitEBit;

            // cont.184: how long are the RUNS of consecutive pairs a block JIT could compile?
            // A pair is block-compilable when its lower is NOP (so there is no upper/lower
            // interleaving to model) and its upper is a covered FMAC op. Blocks only pay if these
            // occur in runs — one call amortised over N pairs — so this histogram is the ceiling
            // measurement for the next stage, taken BEFORE building it.
            {
                const bool lowerNop = decoded.iBit ? false
                                                   : (decoded.lower == 0u || decoded.lower == 0x8000033Cu);
                vu1jit::FmacOp cop;
                int cbc = 0;
                bool cacc = false;
                const bool compilable =
                    lowerNop && !decoded.eBit && vu1jit::classifyUpper(decoded.upper, cop, cbc, cacc);
                if (compilable)
                {
                    ++g_blkRunCur;
                }
                else if (g_blkRunCur != 0u)
                {
                    ++g_blkRunHist[g_blkRunCur < 16u ? g_blkRunCur : 16u];
                    g_blkPairsInRuns += g_blkRunCur;
                    g_blkRunCur = 0u;
                }

                // cont.186: WHICH lower opcodes are worth emitting? Tier the lower slot by how much
                // codegen each step needs, then measure the block coverage each tier would unlock —
                // before writing any of it. Tiers are cumulative:
                //   0 NOP · 1 +SQ/LQ (VU memory) · 2 +IADDIU/ISUBIU (VI ALU)
                //   3 +clip readers (FCEQ/FCAND/FCOR/FCGET) · 4 +ILW/ISW
                {
                    // Tiers 0-5 are CUMULATIVE (each assumes the ones below it are emitted).
                    // Tier 6 deliberately is NOT: see below.
                    bool tierOk[7] = {};
                    if (!decoded.iBit)
                    {
                        const bool isNop =
                            (decoded.lower == 0u || decoded.lower == 0x8000033Cu);
                        const uint8_t lop = static_cast<uint8_t>((decoded.lower >> 25) & 0x7Fu);
                        const bool isMem = !isNop && (lop == 0x00u || lop == 0x01u);
                        const bool isIalu = !isNop && (lop == 0x08u || lop == 0x09u);
                        const bool isClip = !isNop && (lop == 0x10u || lop == 0x12u ||
                                                       lop == 0x13u || lop == 0x1Cu);
                        const bool isIlwIsw = !isNop && (lop == 0x04u || lop == 0x05u);
                        // ★ A BRANCH is not an opcode gap — it is the block TERMINATOR. Counting
                        // it as uncovered understates what a real block compiler reaches, since a
                        // block simply ends there.
                        const bool isBranch = !isNop && (lop >= 0x20u && lop <= 0x2Fu);
                        tierOk[0] = isNop;
                        tierOk[1] = tierOk[0] || isMem;
                        tierOk[2] = tierOk[1] || isIalu;
                        tierOk[3] = tierOk[2] || isClip;
                        tierOk[4] = tierOk[3] || isIlwIsw;
                        tierOk[5] = tierOk[4] || isBranch;
                        // ★★ cont.194 — TIER 6 IS THE ONE THAT SIZES THE NEXT STEP: what block
                        // assembly can reach with the lower slot AS EMITTED TODAY, plus branch
                        // terminators. It is tier 3 + branch, NOT tier 5, because tier 5 also
                        // assumes ILW/ISW and **ILW cannot use the immediate-write equivalence**:
                        // its usage is PipelineLsu with latency 4 (decodeLowerUsage case 0x04), so
                        // unlike every other emitted VI write (all latency 1) its value is not
                        // visible until four cycles later. Modelling that needs a real VI write
                        // pipeline, so ILW stays a gap and tier 5 is not yet reachable.
                        tierOk[6] = tierOk[3] || isBranch;
                    }
                    const bool upperOk =
                        !decoded.eBit && vu1jit::classifyUpper(decoded.upper, cop, cbc, cacc);
                    for (unsigned t = 0; t < 7u; ++t)
                    {
                        if (upperOk && tierOk[t])
                        {
                            ++g_tierRunCur[t];
                        }
                        else if (g_tierRunCur[t] != 0u)
                        {
                            ++g_tierRuns[t];
                            g_tierPairs[t] += g_tierRunCur[t];
                            if (g_tierRunCur[t] >= 2u)
                                g_tierPairsGe2[t] += g_tierRunCur[t];
                            g_tierRunCur[t] = 0u;
                        }
                    }
                }
            }
        }
        const unsigned long long resT0 =
            (s_vu1BlockProf && s_vu1Block) ? __rdtsc() : 0ull;
        const unsigned flagReads =
            (s_vu1LazyFlags != 0 && !decoded.iBit) ? lowerFlagReads(decoded.lower) : 0u;
        if (flagReads != 0u)
        {
            if (s_vu1LazyVerify)
            {
                if (flagReads & kFlagReadMac)
                    ++g_lazyMacReads;
                if (flagReads & kFlagReadStatus)
                    ++g_lazyStatusReads;
                // Every executed reader, armed or not, plus the subset that could see another
                // program's flags: readsBeforeWrite == 0 over a full run is the empirical proof
                // that no reader ever depends on flags produced outside its own program.
                ++g_lazyReaderIssues;
                if (!s_flagWriteThisRun)
                    ++g_lazyReadsBeforeWrite;
                if (s_lazyActive)
                {
                    // The scan claimed this program never reads MAC/status, yet a reader is
                    // issuing. Under LAZYFLAGS=1 that is a scan bug; under the =2 ablation it is
                    // expected. Rate-limited (this site is per-instruction).
                    static uint64_t lazyMisses = 0;
                    if (lazyMisses < 32u)
                    {
                        ++lazyMisses;
                        RUNTIME_ERROR("[VU1 lazy-flag MISS] pc=0x" << std::hex << m_state.pc
                                                                   << " lower=0x" << decoded.lower << std::dec
                                                                   << " (mode=" << s_vu1LazyFlags << ")");
                    }
                }
            }
            const bool couldSeeStale =
                ((flagReads & kFlagReadMac) != 0u && s_macDirty && !s_flagWriteThisRun) ||
                ((flagReads & kFlagReadStatus) != 0u && s_stickyDirty);
            if (couldSeeStale && !s_lazyDisabled)
            {
                ++g_lazyDirtyReads;
                s_lazyDisabled = true;
                s_lazyActive = false;
                vu1jit::lazyActive() = false;
                RUNTIME_ERROR("[VU1 lazy-flag DIRTY] a "
                              << (((flagReads & kFlagReadMac) != 0u) ? "MAC" : "status")
                              << " reader issued at pc=0x" << std::hex << m_state.pc << std::dec
                              << " while lazy flags had skipped writes -- lazy flags are now "
                                 "disabled for the rest of this session");
            }
        }
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            if (s_vuDecProbe)
            {
                // Everything the choice at the top of this iteration depended on, plus a FRESH
                // decode of the same PC straight out of the code buffer.
                const DecodedInstructionPair fresh = decodeInstructionPair(vuCode, m_state.pc);
                uint32_t memLo = 0u, memHi = 0u;
                std::memcpy(&memLo, vuCode + m_state.pc, 4);
                std::memcpy(&memHi, vuCode + m_state.pc + 4u, 4);
                std::fprintf(stderr,
                             "[vu%u:decprobe] pc=0x%x pairIdx=%u cache=%d valid=%d "
                             "used(lo/hi)=%08x/%08x fresh(lo/hi)=%08x/%08x mem(lo/hi)=%08x/%08x "
                             "freshResU=%d cacheCode=%p vuCode=%p cacheSize=%u codeSize=%u "
                             "cacheGen=%llu memGen=%llu byPair=%d armed=%d emitFlags=%d\n",
                             (m_unit == Unit::VU1) ? 1u : 0u, m_state.pc, pairIndex,
                             decodeCache != nullptr ? 1 : 0, m_decodedCodeCacheValid ? 1 : 0,
                             decoded.lower, decoded.upper, fresh.lower, fresh.upper, memLo, memHi,
                             fresh.upperUsage.reserved ? 1 : 0,
                             (const void *)m_cachedVuCode, (const void *)vuCode,
                             m_cachedCodeSize, codeSize,
                             (unsigned long long)m_cachedCodeGeneration,
                             (unsigned long long)(memory ? (m_unit == Unit::VU1
                                                                ? memory->getVU1CodeGeneration()
                                                                : memory->getVU0CodeGeneration())
                                                         : 0ull),
                             (pairIndex < vu1jit::BlockStore::kPairSlots) ? blockByPair[pairIndex] : -999,
                             blockArmed ? 1 : 0, blockEmitFlags ? 1 : 0);
                // The cache around this index: is it THIS program, or a different one?
                if (decodeCache != nullptr)
                {
                    std::fprintf(stderr, "[vu%u:decprobe] cache[%u-3..+3]:",
                                 (m_unit == Unit::VU1) ? 1u : 0u, pairIndex);
                    for (int32_t d = -3; d <= 3; ++d)
                    {
                        const int64_t i = int64_t(pairIndex) + d;
                        if (i < 0 || i >= int64_t(kMaxDecodedPairs))
                            continue;
                        std::fprintf(stderr, " %04x:%08x/%08x", uint32_t(i) * 8u,
                                     decodeCache[i].lower, decodeCache[i].upper);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = calculatePairReadyCycle(decoded);
        if (s_vu1JitCensus && m_unit == Unit::VU1)
        {
            // cont.179 scheduler census: how much of the stall machinery actually fires? The
            // register reduction inside calculatePairReadyCycle is provably INVARIANT across
            // advanceTo (m_vfReady/m_viReady/m_accReady are written only by markPairWrites and
            // reset(), never by commitReadyPipelines), so every recompute after the first is
            // redundant except for the m_cycle floor and the fdiv/efu/xgkick tail terms. This
            // counts how often that redundancy is paid before building anything on it.
            ++g_schedCalls;
            if (readyCycle > m_cycle)
            {
                ++g_schedStalledPairs;
                g_schedStallCycles += (readyCycle - m_cycle);
            }
        }
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
            if (s_vu1JitCensus && m_unit == Unit::VU1)
                ++g_schedCalls;
        }
        if (m_cycle >= budgetEnd)
            break;

        // Lowest set VI-write bit == what the old ascending 1..15 scan found first.
        uint8_t writtenVi = 0u;
        int32_t oldVi = 0;
        if (const uint32_t viWriteMask = static_cast<uint32_t>(decoded.lowerUsage.viWrite) & kViRegMask)
        {
            writtenVi = static_cast<uint8_t>(__builtin_ctz(viWriteMask));
            oldVi = m_state.vi[writtenVi];
        }

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = upperWrite.reg != 0u;
        const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        // cont.178 (b): deliberately UNINITIALISED. Each array is memcpy-filled before its only
        // read under the identical guard, so zeroing them was 96 bytes of dead memset per pair.
        float oldUpperVf[4];
        float newUpperVf[4];
        float oldLowerVf[4];
        float newLowerVf[4];
        float oldAcc[4];
        float newAcc[4];
        if (!s_vu1FastDispatch)
        {
            // Kill switch: reproduce the old zero-initialisation exactly for same-binary A/B.
            std::memset(oldUpperVf, 0, sizeof(oldUpperVf));
            std::memset(newUpperVf, 0, sizeof(newUpperVf));
            std::memset(oldLowerVf, 0, sizeof(oldLowerVf));
            std::memset(newLowerVf, 0, sizeof(newLowerVf));
            std::memset(oldAcc, 0, sizeof(oldAcc));
            std::memset(newAcc, 0, sizeof(newAcc));
        }
        if (hasUpperWrite)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (hasDistinctLowerWrite)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

        if (decoded.iBit)
        {
            execUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
            execUpper(decoded.upper);
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        m_viBranchBackupValid = false;

        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (decoded.upperUsage.accWrite != 0u)
        {
            std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
            std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
            // ACC is forwarded to the next upper instruction. Its arithmetic
            // flags still use the normal four-cycle FMAC timeline.
            queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                          kAccForwardLatency);
        }
        if (writtenVi != 0u)
        {
            const int32_t newVi = m_state.vi[writtenVi];
            m_state.vi[writtenVi] = oldVi;
            const uint32_t latency =
                decoded.lowerUsage.viLatency != 0u
                    ? decoded.lowerUsage.viLatency
                    : decoded.lowerUsage.latency;
            queueViWrite(writtenVi, newVi, latency);
        }

        markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        if (s_vu1BlockProf && s_vu1Block)
        {
            const unsigned long long d = __rdtsc() - resT0;
            ++g_resPairs;
            g_resCycles += d;
            {
                const uint8_t uop = uint8_t(decoded.upper & 0x3Fu);
                ++g_resUpper[uop];
                g_resUpperCy[uop] += d;
                if (uop >= 0x3Cu)
                {
                    const uint8_t usp = uint8_t((decoded.upper & 3u) | ((decoded.upper >> 4) & 0x7Cu));
                    ++g_resUpSpec[usp & 0x7Fu];
                    g_resUpSpecCy[usp & 0x7Fu] += d;
                }
                vu1jit::UpperPlan up;
                vu1jit::LowerPlan lo;
                const bool upOk = !decoded.eBit && vu1jit::classifyUpperPlan(decoded.upper, up);
                const bool loOk = !decoded.iBit && vu1jit::classifyLowerPlan(decoded.lower, lo);
                const unsigned idx = (upOk ? 0u : 1u) | (loOk ? 0u : 2u);
                ++g_resWhy[idx];
                g_resWhyCy[idx] += d;
                if (!s_lazyActive)
                { ++g_resNoLazy; g_resNoLazyCy += d; }
                else if (!blockArmed)
                { ++g_resNoArm; g_resNoArmCy += d; }
            }
            if (!decoded.iBit)
            {
                const uint8_t lop = uint8_t((decoded.lower >> 25) & 0x7Fu);
                ++g_resLower[lop];
                g_resLowerCy[lop] += d;
                if (lop == 0x40u)
                {
                    const uint8_t f = uint8_t(decoded.lower & 0x3Fu);
                    const uint8_t sp = (f >= 0x3Cu)
                        ? uint8_t((decoded.lower & 3u) | ((decoded.lower >> 4) & 0x7Cu))
                        : f;
                    ++g_resSpecial[sp & 0x7Fu];
                    g_resSpecialCy[sp & 0x7Fu] += d;
                }
            }
        }
        advanceOneCycle();
        // ---- cont.195 block shadow-verify: compare once the block's pair window has elapsed ----
        if (blockVerifyLeft != 0u && --blockVerifyLeft == 0u)
        {
            const vu1jit::Block *vb = blockVerifyBlk;
            blockVerifyBlk = nullptr;
            // Slots the interpreter still has a write in flight for legitimately differ (the
            // block applies its writes immediately), so they are skipped — and the count of slots
            // actually compared is reported, so a vacuous "0 mismatches" is visible as such.
            uint8_t pendingVf[32] = {};
            uint32_t pendingVi = 0u;
            uint8_t pendingAcc = 0u;
            bool pendingStore = false;
            for (uint32_t r = 0; r < 32u; ++r)
                pendingVf[r] = blockEntryPendingVf[r];
            pendingVi = blockEntryPendingVi;
            pendingAcc = blockEntryPendingAcc;
            pendingStore = blockEntryPendingStore;
            for (const PendingVfWrite &w : m_vfWritePipeline)
                if (w.valid)
                    pendingVf[w.reg & 31u] |= (w.laneMask & 0xFu);
            for (const PendingViWrite &w : m_viWritePipeline)
                if (w.valid)
                    pendingVi |= (1u << (w.reg & 15u));
            for (const PendingAccWrite &w : m_accWritePipeline)
                if (w.valid)
                    pendingAcc |= (w.laneMask & 0xFu);
            for (const PendingStore &st : m_storePipeline)
                if (st.valid)
                    pendingStore = true;

            unsigned long checked = 0, bad = 0;
            // ★ CYCLE FIDELITY. The register comparison cannot catch a scheduling error, and the
            // schedule is the risky part of cont.196: the block must consume exactly as many
            // cycles as the interpreter did over the same pairs, or guest timing drifts silently.
            {
                const uint64_t actual = m_cycle - blockVerifyEntryCycle;
                ++checked;
                if (actual != uint64_t(blockVerifyCycles) && ++bad <= 4ul)
                    RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                  << std::hex << (vb != nullptr ? vb->startPc : 0u) << std::dec
                                  << " CYCLES interp=" << actual
                                  << " block=" << blockVerifyCycles);
            }
            // ★ For a branch-terminated block the PC is the whole point: the block computed the
            // condition, the target and the delay slot itself, so a wrong branch shows up here.
            if (vb != nullptr && vb->endsWithBranch)
            {
                ++checked;
                if (m_state.pc != blockVerifyRef.pc && ++bad <= 4ul)
                    RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                  << std::hex << vb->startPc << " BRANCH interp=0x" << m_state.pc
                                  << " block=0x" << blockVerifyRef.pc << std::dec);
            }
            for (uint32_t r = 1; r < 32u; ++r)
            {
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if ((pendingVf[r] & laneForComponent(c)) != 0u)
                        continue;
                    uint32_t a = 0, b = 0;
                    std::memcpy(&a, &m_state.vf[r][c], 4);
                    std::memcpy(&b, &blockVerifyRef.vf[r][c], 4);
                    ++checked;
                    if (a != b && ++bad <= 4ul)
                        RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                      << std::hex << (vb != nullptr ? vb->startPc : 0u)
                                      << " vf[" << std::dec << r << "][" << c << "] interp="
                                      << std::hex << a << " block=" << b << std::dec);
                }
            }
            for (uint32_t r = 1; r < 16u; ++r)
            {
                if ((pendingVi & (1u << r)) != 0u)
                    continue;
                ++checked;
                if (m_state.vi[r] != blockVerifyRef.vi[r] && ++bad <= 4ul)
                    RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                  << std::hex << (vb != nullptr ? vb->startPc : 0u)
                                  << " vi[" << std::dec << r << "] interp=" << m_state.vi[r]
                                  << " block=" << blockVerifyRef.vi[r]);
            }
            // ★ The scalar/flag state the verifier previously ignored. mac/status matter now that
            // flag emission is on the table, and clip/q are early-applied by the entry guard --
            // an error in either would have been invisible.
            {
                const auto cmpScalar = [&](const char *name, uint32_t a, uint32_t b) {
                    ++checked;
                    if (a != b && ++bad <= 4ul)
                        RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                      << std::hex << (vb != nullptr ? vb->startPc : 0u) << " "
                                      << name << " interp=" << a << " block=" << b << std::dec);
                };
                bool flagPending = false;
                for (const FlagPipelineEntry &fe : m_flagPipeline)
                    if (fe.valid)
                        flagPending = true;
                const bool emitsFlags = vb != nullptr && vb->emitsFlags;
                if (!flagPending && !blockEntryFlagPending && !emitsFlags)
                {
                    // Only meaningful when the block does NOT produce flags itself: in that mode
                    // the driver's replay is skipped under verify (the block runs on a copy), and
                    // committed status legitimately differs because the block passes extraSticky=0.
                    cmpScalar("mac", m_state.mac, blockVerifyRef.mac);
                    cmpScalar("status", m_state.status, blockVerifyRef.status);
                }
                if (!flagPending && !blockEntryFlagPending &&
                    (vb == nullptr || vb->clipCount == 0u))
                    cmpScalar("clip", m_state.clip, blockVerifyRef.clip);
                {
                    g_ccActive = false;
                    ++checked;
                    if (g_ccN != blockClipN && ++bad <= 4ul)
                        RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x" << std::hex
                                      << (vb ? vb->startPc : 0u) << std::dec
                                      << " CLIPCOUNT interp=" << g_ccN << " block=" << blockClipN);
                    const uint32_t nc = g_ccN < blockClipN ? g_ccN : blockClipN;
                    for (uint32_t k = 0; k < nc; ++k)
                    {
                        ++g_ccChecked;
                        if (g_ccIssue[k] != blockClipIssueArr[k] ||
                            g_ccClip[k] != (blockClipRaw[k] & 0x3Fu))
                        {
                            ++g_ccBad;
                            if (++bad <= 4ul)
                                RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x" << std::hex
                                              << (vb ? vb->startPc : 0u) << " CLIP[" << std::dec
                                              << k << "] interp raw=" << std::hex << g_ccClip[k]
                                              << " cy=" << std::dec << g_ccIssue[k]
                                              << " | block raw=" << std::hex
                                              << (blockClipRaw[k] & 0x3Fu) << " cy=" << std::dec
                                              << blockClipIssueArr[k]);
                        }
                    }
                }
                if (emitsFlags)
                {
                    g_fcActive = false;
                    ++checked;
                    if (g_fcN != blockFlagN && ++bad <= 4ul)
                        RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x" << std::hex << vb->startPc
                                      << std::dec << " FLAGCOUNT interp=" << g_fcN
                                      << " block=" << blockFlagN);
                    const uint32_t nf = g_fcN < blockFlagN ? g_fcN : blockFlagN;
                    for (uint32_t k = 0; k < nf; ++k)
                    {
                        ++g_fcChecked;
                        if (g_fcMac[k] != blockFlagMacArr[k] ||
                            g_fcStatus[k] != blockFlagStatusArr[k] ||
                            g_fcIssue[k] != blockFlagIssueArr[k])
                        {
                            ++g_fcBad;
                            if (++bad <= 4ul)
                                RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x" << std::hex
                                              << vb->startPc << " FLAG[" << std::dec << k
                                              << "] interp mac=" << std::hex << g_fcMac[k]
                                              << " st=" << g_fcStatus[k] << " cy=" << std::dec
                                              << g_fcIssue[k] << " | block mac=" << std::hex
                                              << blockFlagMacArr[k] << " st="
                                              << blockFlagStatusArr[k] << " cy=" << std::dec
                                              << blockFlagIssueArr[k]);
                        }
                    }
                }
                if (!m_fdiv.valid && !blockEntryFdivPending)
                {
                    uint32_t qa = 0, qb = 0;
                    std::memcpy(&qa, &m_state.q, 4);
                    std::memcpy(&qb, &blockVerifyRef.q, 4);
                    cmpScalar("q", qa, qb);
                }
            }
            for (uint32_t c = 0; c < 4u; ++c)
            {
                if ((pendingAcc & laneForComponent(c)) != 0u)
                    continue;
                uint32_t a = 0, b = 0;
                std::memcpy(&a, &m_state.acc[c], 4);
                std::memcpy(&b, &blockVerifyRef.acc[c], 4);
                ++checked;
                if (a != b && ++bad <= 4ul)
                    RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                  << std::hex << (vb != nullptr ? vb->startPc : 0u)
                                  << " acc[" << std::dec << c << "] interp=" << std::hex << a
                                  << " block=" << b << std::dec);
            }
            if (vb != nullptr && vb->touchesVuMem && !pendingStore &&
                blockVerifyMem.size() >= dataSize && vuData != nullptr)
            {
                for (uint32_t o = 0; o < dataSize; o += 4u)
                {
                    uint32_t a = 0, b = 0;
                    std::memcpy(&a, vuData + o, 4);
                    std::memcpy(&b, blockVerifyMem.data() + o, 4);
                    ++checked;
                    if (a != b && ++bad <= 4ul)
                        RUNTIME_ERROR("[VU1 block MISMATCH] pc=0x"
                                      << std::hex << (vb != nullptr ? vb->startPc : 0u)
                                      << " vumem+0x" << o << " interp=" << a << " block=" << b
                                      << std::dec);
                }
            }
            g_fcActive = false;
            g_ccActive = false;
            g_blockVerifyChecked += checked;
            g_blockVerifyMismatch += bad;
        }
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
    if (perfTrack0)
    {
        g_vu0Ns += static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - perfT0).count());
        g_vu0Pairs += perfPairs;
        ++g_vu0Calls;
        if (g_vu0Pairs >= g_vu0NextReport)
        {
            g_vu0NextReport = g_vu0Pairs + 2000000ull;
            std::fprintf(stderr,
                         "[vu0:perf] pairs=%llu calls=%llu ns=%llu | %.2f ns/pair %.1f pairs/call"
                         " | progs=%llu clean=%llu readsStatus=%llu"
                         " | BLOCKS entered=%lu pairsRun=%lu (%.1f%%) meanPairs=%.2f compiled=%lu"
                         " rejected=%lu guardBlocked=%lu"
                         " | VU1 pairs=%llu ns=%llu (%.2f ns/pair) => VU0 is %.1f%% of VU time\n",
                         g_vu0Pairs, g_vu0Calls, g_vu0Ns,
                         g_vu0Pairs ? double(g_vu0Ns) / double(g_vu0Pairs) : 0.0,
                         g_vu0Calls ? double(g_vu0Pairs) / double(g_vu0Calls) : 0.0,
                         g_vu0ProgsScanned, g_vu0ProgsClean, g_vu0ProgsStatus,
                         vu1jit::blockStore(0u).entered, vu1jit::blockStore(0u).pairsRun,
                         g_vu0Pairs ? 100.0 * double(vu1jit::blockStore(0u).pairsRun) /
                                          double(g_vu0Pairs)
                                    : 0.0,
                         vu1jit::blockStore(0u).entered
                             ? double(vu1jit::blockStore(0u).pairsRun) /
                                   double(vu1jit::blockStore(0u).entered)
                             : 0.0,
                         vu1jit::blockStore(0u).compiled, vu1jit::blockStore(0u).rejected,
                         vu1jit::blockStore(0u).guardBlocked,
                         g_vu1Pairs, g_vu1Ns,
                         g_vu1Pairs ? double(g_vu1Ns) / double(g_vu1Pairs) : 0.0,
                         (g_vu0Ns + g_vu1Ns) ? 100.0 * double(g_vu0Ns) / double(g_vu0Ns + g_vu1Ns)
                                             : 0.0);
        }
    }
    if (perfTrack)
    {
        g_vu1Ns += static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - perfT0).count());
        g_vu1Pairs += perfPairs;
        ++g_vu1Calls;
        if (g_vu1Pairs >= g_vu1NextReport)
        {
            g_vu1NextReport = g_vu1Pairs + 20000000ull;
            std::fprintf(stderr,
                         "[vu1:perf] pairs=%llu calls=%llu ns=%llu | %.2f ns/pair "
                         "%.1f pairs/call (fastscan=%d nosched=%d lazy=%d off=%d) "
                         "| lazy: progs=%llu clean=%llu armed=%llu macRd=%llu stsRd=%llu "
                         "preWrite=%llu dirtyReads=%llu\n",
                         g_vu1Pairs, g_vu1Calls, g_vu1Ns,
                         g_vu1Pairs ? double(g_vu1Ns) / double(g_vu1Pairs) : 0.0,
                         g_vu1Calls ? double(g_vu1Pairs) / double(g_vu1Calls) : 0.0,
                         s_vu1FastScan ? 1 : 0, s_vu1NoSched ? 1 : 0, s_vu1LazyFlags,
                         s_lazyDisabled ? 1 : 0,
                         g_lazyProgsScanned, g_lazyProgsClean, g_lazyRunsArmed,
                         g_lazyMacReads, g_lazyStatusReads,
                         g_lazyReadsBeforeWrite, g_lazyDirtyReads);
            if (vu1jit::enabled())
                vu1jit::report();
            if (s_vu1Block)
            {
                const vu1jit::BlockStore &bs = vu1jit::blockStore();
                std::fprintf(stderr,
                             "[vu1:block] entered=%lu pairsRun=%lu (%.1f%% of pairs) "
                             "meanPairs=%.2f compiled=%lu rejected=%lu resched=%lu inval=%lu hashHit=%lu guardBlocked=%lu"
                             " (vf=%lu vi=%lu acc=%lu store=%lu q=%lu clip=%lu budget=%lu)"
                             " early-apply(vf=%lu clip=%lu q=%lu)"
                             " | verify checked=%llu mismatches=%llu flagEntries=%llu flagBad=%llu clipEntries=%llu clipBad=%llu\n",
                             bs.entered, bs.pairsRun,
                             g_vu1Pairs ? 100.0 * double(bs.pairsRun) / double(g_vu1Pairs) : 0.0,
                             bs.entered ? double(bs.pairsRun) / double(bs.entered) : 0.0,
                             bs.compiled, bs.rejected, bs.rescheduled, bs.invalidations, bs.hashHits,
                             bs.guardBlocked,
                             bs.gbVf, bs.gbVi, bs.gbAcc, bs.gbStore, bs.gbQ, bs.gbClip,
                             bs.gbBudget, bs.earlyVf, bs.earlyClip, bs.earlyQ,
                             g_blockVerifyChecked, g_blockVerifyMismatch,
                             g_fcChecked, g_fcBad, g_ccChecked, g_ccBad);
                if (s_vu1ReadyVerify)
                    std::fprintf(stderr, "[vu1:readyverify] checked=%llu mismatches=%llu\n",
                                 g_rvChecked, g_rvBad);
                std::fprintf(stderr,
                             "[vu1:xgkick] progress calls=%llu qwords=%llu (%.2f/call)"
                             " bytes=%llu wraps=%llu fast=%d verify=%llu/%llu eager=%d eagerdiff=%llu/%llu\n",
                             g_kickCalls, g_kickQwords,
                             g_kickCalls ? double(g_kickQwords) / double(g_kickCalls) : 0.0,
                             g_kickQwords * 16ull, g_kickWraps, s_vu1FastKick ? 1 : 0,
                             g_kvChecked, g_kvBad, s_vu1KickEager ? 1 : 0, g_kvEagerDiff, g_kvEagerChecked);
                std::fprintf(stderr, "[vu1:kickverify] submits=%llu unwritten=%llu\n",
                             g_kvSubmits, g_kvUnwritten);
                if (s_vu1LinkCensus)
                    lcReport();
                if (s_vu1FlagVerify)
                    std::fprintf(stderr,
                                 "[vu1:flagverify] checked=%llu events=%llu mismatches=%llu"
                                 " overflow=%llu deferred=%llu\n",
                                 g_fvChecked, g_fvEvents, g_fvBad, g_fvOverflow, g_fvDeferred);
                if (s_vu1BlockProf && g_bpEntries)
                {
                    const double e = double(g_bpEntries);
                    std::fprintf(stderr,
                                 "[vu1:blockprof0] level=%d entries=%llu WHOLE=%.0f cycles/entry"
                                 " dispatch=%.1f"
                                 " | noPending=%.1f%% supersede=%.1f%% | pairs:"
                                 " 1=%.1f%% 2=%.1f%% 3=%.1f%% 4=%.1f%% 5=%.1f%% 6=%.1f%% 7=%.1f%%"
                                 " 8=%.1f%%\n",
                                 s_vu1BlockProfLevel, g_bpEntries, double(g_bpWhole) / e,
                                 double(g_bpDispatch) / e,
                                 100.0 * double(g_bpNoPending) / e,
                                 100.0 * double(g_bpSupersede) / e,
                                 100.0 * double(g_bpPairsHist[1]) / e,
                                 100.0 * double(g_bpPairsHist[2]) / e,
                                 100.0 * double(g_bpPairsHist[3]) / e,
                                 100.0 * double(g_bpPairsHist[4]) / e,
                                 100.0 * double(g_bpPairsHist[5]) / e,
                                 100.0 * double(g_bpPairsHist[6]) / e,
                                 100.0 * double(g_bpPairsHist[7]) / e,
                                 100.0 * double(g_bpPairsHist[8]) / e);
                    {
                        static const char *kStop[12] = {
                            "cap8", "ibits", "upper", "issue", "clipread", "branchrej", "ds-bits",
                            "ds-uncomp", "vihazard", "readyfull", "branch", "tailtrim"};
                        std::string a, b2;
                        const double e2 = double(g_bpStop2[0] + g_bpStop2[1] + g_bpStop2[2] +
                                                 g_bpStop2[3] + g_bpStop2[4] + g_bpStop2[5] +
                                                 g_bpStop2[6] + g_bpStop2[7] + g_bpStop2[8] +
                                                 g_bpStop2[9] + g_bpStop2[10] + g_bpStop2[11]);
                        for (unsigned i = 0; i < 12u; ++i)
                        {
                            char buf[64];
                            if (g_bpStop[i])
                            {
                                std::snprintf(buf, sizeof buf, " %s=%.1f%%", kStop[i],
                                              100.0 * double(g_bpStop[i]) / e);
                                a += buf;
                            }
                            if (g_bpStop2[i] && e2 > 0.0)
                            {
                                std::snprintf(buf, sizeof buf, " %s=%.1f%%", kStop[i],
                                              100.0 * double(g_bpStop2[i]) / e2);
                                b2 += buf;
                            }
                        }
                        std::fprintf(stderr, "[vu1:blockstop] ALL:%s | 2-PAIR ONLY (%.1f%% of"
                                             " entries):%s\n",
                                     a.c_str(), 100.0 * e2 / e, b2.c_str());
                        if (!g_bpStop5Ops.empty())
                        {
                            std::vector<std::pair<unsigned long long, uint32_t>> v;
                            for (const auto &kv : g_bpStop5Ops) v.emplace_back(kv.second, kv.first);
                            std::sort(v.begin(), v.end(), std::greater<>());
                            std::string o;
                            for (size_t i = 0; i < v.size() && i < 10u; ++i)
                            {
                                char buf[48];
                                std::snprintf(buf, sizeof buf, " %08x=%.1f%%", v[i].second,
                                              100.0 * double(v[i].first) / double(g_bpStop[5]));
                                o += buf;
                            }
                            std::fprintf(stderr, "[vu1:blockstop5] rejected-lower opcode keys"
                                                 " (%% of reason 5, opHi<<25 | special fn bits):%s\n",
                                         o.c_str());
                        }
                    }
                    if (s_vu1BlockProfSeg)
                    std::fprintf(stderr,
                                 "[vu1:blockprof] entries=%llu lookupMiss=%llu | per ENTRY cycles:"
                                 " lookup+guard=%.0f sched=%.0f apply=%.0f fn=%.0f retire=%.0f"
                                 " | fn per PAIR=%.1f | flagRetire calls=%.2f/entry"
                                 " skipped=%.1f%% retired=%.2f/entry\n",
                                 g_bpEntries, g_bpMiss,
                                 double(g_bpLookup) / e, double(g_bpSched) / e,
                                 double(g_bpApply) / e, double(g_bpFn) / e,
                                 double(g_bpRetire) / e,
                                 double(g_bpFn) / double(bs.pairsRun ? bs.pairsRun : 1),
                                 double(g_ffrCalls) / e,
                                 g_ffrCalls ? 100.0 * double(g_ffrSkipped) / double(g_ffrCalls) : 0.0,
                                 double(g_ffrRetired) / e);
                    // cont.212: retire split by population -- flag-replaying blocks vs the rest.
                    if (s_vu1BlockProfSeg)
                    {
                        const double ef = double(g_bpEntF ? g_bpEntF : 1);
                        const double ec = double(g_bpEntC ? g_bpEntC : 1);
                        std::fprintf(stderr,
                                     "[vu1:blockprof2] FLAGGED entries=%llu (%.1f%%) pairs/e=%.2f"
                                     " retire=%.0f fn=%.0f flags/e=%.2f"
                                     " | CLEAN entries=%llu pairs/e=%.2f retire=%.0f fn=%.0f"
                                     " | ready/e=%.2f clip/e=%.3f"
                                     " | retire share: flagged=%.1f%% clean=%.1f%%\n",
                                     g_bpEntF, 100.0 * double(g_bpEntF) / e,
                                     double(g_bpPairsF) / ef, double(g_bpRetireF) / ef,
                                     double(g_bpFnF) / ef,
                                     double(g_bpFlagN) / ef,
                                     g_bpEntC, double(g_bpPairsC) / ec,
                                     double(g_bpRetireC) / ec, double(g_bpFnC) / ec,
                                     double(g_bpReadyN) / e, double(g_bpClipN) / e,
                                     g_bpRetire ? 100.0 * double(g_bpRetireF) / double(g_bpRetire) : 0.0,
                                     g_bpRetire ? 100.0 * double(g_bpRetireC) / double(g_bpRetire) : 0.0);
                    }
                    // Residual: the pairs the interpreter still runs, ranked by TOTAL cycles.
                    std::fprintf(stderr,
                                 "[vu1:residual] pairs=%llu (%.1f%%) cycles=%llu meanCy=%.0f\n",
                                 g_resPairs,
                                 g_vu1Pairs ? 100.0 * double(g_resPairs) / double(g_vu1Pairs) : 0.0,
                                 g_resCycles,
                                 g_resPairs ? double(g_resCycles) / double(g_resPairs) : 0.0);
                    const auto top = [](const char *what, const unsigned long long *n,
                                        const unsigned long long *cy, unsigned long long tot)
                    {
                        std::pair<unsigned long long, size_t> t[8]{};
                        for (size_t i = 0; i < 128; ++i)
                            for (size_t k = 0; k < 8; ++k)
                                if (cy[i] > t[k].first)
                                { for (size_t m = 7; m > k; --m) t[m] = t[m - 1]; t[k] = {cy[i], i}; break; }
                        std::fprintf(stderr, "[vu1:residual] %s:", what);
                        for (size_t k = 0; k < 8 && t[k].first; ++k)
                            std::fprintf(stderr, " %02zx=%.1f%%(n=%llu,cy=%.0f)", t[k].second,
                                         tot ? 100.0 * double(t[k].first) / double(tot) : 0.0,
                                         n[t[k].second],
                                         n[t[k].second] ? double(cy[t[k].second]) / double(n[t[k].second]) : 0.0);
                        std::fprintf(stderr, "\n");
                    };
                    top("lower", g_resLower, g_resLowerCy, g_resCycles);
                    top("special", g_resSpecial, g_resSpecialCy, g_resCycles);
                    top("upper", g_resUpper, g_resUpperCy, g_resCycles);
                    top("upSpec", g_resUpSpec, g_resUpSpecCy, g_resCycles);
                    std::fprintf(stderr,
                                 "[vu1:residual] why: bothOk=%.1f%%(n=%llu) upperBad=%.1f%%(n=%llu)"
                                 " lowerBad=%.1f%%(n=%llu) bothBad=%.1f%%(n=%llu)\n",
                                 g_resCycles ? 100.0 * double(g_resWhyCy[0]) / double(g_resCycles) : 0.0, g_resWhy[0],
                                 g_resCycles ? 100.0 * double(g_resWhyCy[1]) / double(g_resCycles) : 0.0, g_resWhy[1],
                                 g_resCycles ? 100.0 * double(g_resWhyCy[2]) / double(g_resCycles) : 0.0, g_resWhy[2],
                                 g_resCycles ? 100.0 * double(g_resWhyCy[3]) / double(g_resCycles) : 0.0, g_resWhy[3]);
                    std::fprintf(stderr,
                                 "[vu1:residual] blocked-by-program: noLazy=%.1f%%(n=%llu)"
                                 " noArm=%.1f%%(n=%llu)\n",
                                 g_resCycles ? 100.0 * double(g_resNoLazyCy) / double(g_resCycles) : 0.0,
                                 g_resNoLazy,
                                 g_resCycles ? 100.0 * double(g_resNoArmCy) / double(g_resCycles) : 0.0,
                                 g_resNoArm);
                }
            }
            if (s_vu1JitCensus && g_jitPairs)
            {
                // Sorted top-N per category, plus the block-shape numbers the JIT needs.
                auto dumpTop = [](const char *what, const unsigned long long *tab, size_t n)
                {
                    std::pair<unsigned long long, size_t> top[12]{};
                    for (size_t i = 0; i < n; ++i)
                    {
                        if (!tab[i]) continue;
                        for (size_t s = 0; s < 12; ++s)
                        {
                            if (tab[i] > top[s].first)
                            {
                                for (size_t k = 11; k > s; --k) top[k] = top[k - 1];
                                top[s] = {tab[i], i};
                                break;
                            }
                        }
                    }
                    std::fprintf(stderr, "[vu1:jitcensus] %s:", what);
                    for (const auto &e : top)
                        if (e.first) std::fprintf(stderr, " %02zx=%.1f%%", e.second,
                                                  100.0 * double(e.first) / double(g_jitPairs));
                    std::fprintf(stderr, "\n");
                };
                std::fprintf(stderr,
                             "[vu1:jitcensus] pairs=%llu branches=%llu (1 per %.1f pairs = mean "
                             "block length) ibit=%.1f%% ebit=%llu\n",
                             g_jitPairs, g_jitBranches,
                             g_jitBranches ? double(g_jitPairs) / double(g_jitBranches) : 0.0,
                             100.0 * double(g_jitIBit) / double(g_jitPairs), g_jitEBit);
                if (g_fcLanes)
                    std::fprintf(stderr,
                                 "[vu1:floatmodel] lanes=%llu valueDiff=%llu (%.6f%%) "
                                 "flagDiff=%llu (%.6f%%)\n",
                                 g_fcLanes, g_fcValueDiff,
                                 100.0 * double(g_fcValueDiff) / double(g_fcLanes),
                                 g_fcFlagDiff,
                                 100.0 * double(g_fcFlagDiff) / double(g_fcLanes));
                {
                    unsigned long long runs = 0, pairsGe2 = 0;
                    for (unsigned k = 1; k <= 16u; ++k)
                    {
                        runs += g_blkRunHist[k];
                        if (k >= 2u)
                            pairsGe2 += g_blkRunHist[k] * (k < 16u ? k : 16u);
                    }
                    {
                        static const char *tierName[7] = {
                            "NOP only", "+SQ/LQ", "+IADDIU", "+clip rd", "+ILW/ISW", "+branch",
                            "EMITTED+br"};
                        for (unsigned t = 0; t < 7u; ++t)
                            std::fprintf(stderr,
                                         "[vu1:tier%u] %-9s compilable=%llu (%.1f%%) runs=%llu "
                                         "meanRun=%.2f inRunsGe2=%llu (%.1f%%)\n",
                                         t, tierName[t], g_tierPairs[t],
                                         100.0 * double(g_tierPairs[t]) / double(g_jitPairs),
                                         g_tierRuns[t],
                                         g_tierRuns[t] ? double(g_tierPairs[t]) / double(g_tierRuns[t]) : 0.0,
                                         g_tierPairsGe2[t],
                                         100.0 * double(g_tierPairsGe2[t]) / double(g_jitPairs));
                    }
                    std::fprintf(stderr,
                                 "[vu1:blockcensus] compilablePairs=%llu (%.1f%%) runs=%llu "
                                 "meanRun=%.2f pairsInRunsGe2=%llu (%.1f%%) hist=",
                                 g_blkPairsInRuns,
                                 100.0 * double(g_blkPairsInRuns) / double(g_jitPairs),
                                 runs, runs ? double(g_blkPairsInRuns) / double(runs) : 0.0,
                                 pairsGe2, 100.0 * double(pairsGe2) / double(g_jitPairs));
                    for (unsigned k = 1; k <= 16u; ++k)
                        if (g_blkRunHist[k])
                            std::fprintf(stderr, " %u:%llu", k, g_blkRunHist[k]);
                    std::fprintf(stderr, "\n");
                }
                std::fprintf(stderr,
                             "[vu1:schedcensus] readyCalls=%llu (%.3f per pair) stalledPairs=%llu "
                             "(%.1f%%) stallCycles=%llu (%.2f per stalled pair)\n",
                             g_schedCalls, double(g_schedCalls) / double(g_jitPairs),
                             g_schedStalledPairs,
                             100.0 * double(g_schedStalledPairs) / double(g_jitPairs),
                             g_schedStallCycles,
                             g_schedStalledPairs ? double(g_schedStallCycles) / double(g_schedStalledPairs) : 0.0);
                dumpTop("upper", g_jitUpper, 64);
                dumpTop("upperSpecial", g_jitUpperSpecial, 128);
                dumpTop("lower", g_jitLower, 128);
            }
        }
    }
}
