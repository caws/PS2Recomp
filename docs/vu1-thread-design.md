# VU1 off the EE thread — design (cont.317, 2026-09-11)

Status: **design + stage 0 (the oracle)**. Stages 1-3 are the build. Nothing here is enabled by
default until its stage's oracle passes.

## Why

perf + `/proc/<tid>/schedstat` on the recorded Helm's Deep fight at 7 raster workers (build 735,
230 s, 15 fps): the EE thread is 17.5% vsync sleep, **26% VIF1+VU1 (VU1 alone 25% of wall =
~17 ms/frame)**, 19% GIF packet parse, 37% guest code + HLE; the raster workers 60-70%. Each side
is ~45 ms of a 66 ms frame that snaps to whole 20 ms vblanks. The next step (3 -> 2 vblanks,
15 -> 25 fps) needs BOTH sides under 40 ms. On the EE, VU1 is the only item big enough to do it
alone; with the VIF1 stream and the GIF parse it is ~35% of the EE's wall time. (progress.md
cont.317, the "where the frame goes" section.)

## Ground truth (what the runtime does today — survey 2026-09-11, file:line in the fork)

- **Everything below runs on the EE thread**, synchronously, inside the guest's CHCR store:
  `writeIORegister` (CHCR with STR, `ps2_memory.cpp:1320`) -> `processPendingTransfers()` (`:2066`)
  -> `processVIF1Data(ptr,size)` (`ps2_vif1_interpreter.cpp:697`) -> per VIFcode: UNPACK writes
  `m_vu1Data`, MPG writes `m_vu1Code` (+ `markVU1CodeModified`), MSCAL/MSCNT call the runtime's
  callbacks (`ps2_runtime.cpp:659-694`) which run the microprogram to completion
  (`VU1Interpreter::execute/resume` -> `ps2x_microvu::runUnit`, `ps2x_microvu.cpp:325`), DIRECT
  submits path-2 packets, MSKPATH3 toggles `m_path3Masked` and flushes the masked path-3 FIFO.
- **XGKICK** (microVU `_vuXGKICKTransfermVU` -> shim `ps2x_microvu.cpp:157`) calls
  `PS2Memory::submitGifPacket(Path1, ...)` (`ps2_memory.cpp:2377`) -> `GifArbiter::submit` (copies)
  -> **`drain()` immediately** -> `GS::processGIFPacket` (`gs_frontend.cpp:782`, under
  `m_stateMutex`) -> batches -> `GSCpuBackend::enqueueWork` (the worker queue). So the GS parse
  runs on whatever thread runs VU1.
- **Ordering between GIF paths is call order.** Path 1 (XGKICK) and path 2 (DIRECT) always drain
  immediately; only path-3 DMA batches queue (`drainImmediately=false`, `ps2_memory.cpp:2091/2120/2139`)
  and are drained at the end of the pass (`:2265`) or by the next immediate drain — where the
  arbiter's stable sort puts them AFTER the path-1/2 packets of that drain (`ps2_gif_arbiter.cpp:39-73`).
  Path-3 packets submitted while MSKPATH3 is set wait in `m_path3MaskedFifo` until the unmask, in
  stream order.
- The GIF parser keeps **per-path resume state across packets** (`g_gifParseState[4]`,
  `g_gifActivePath`, `gs_frontend.cpp:758-770`), selected by `ps2xGsSetGifPath` from the arbiter.
- **The guest observes no VU1 timing at all.** `readIORegister` has no VIF1 STAT case (VBS/VPS
  read back as whatever the guest last wrote: `ps2_memory.cpp:2780/2884`), FLUSH/FLUSHE/FLUSHA are
  no-ops (`ps2_vif1_interpreter.cpp:974-977`), the generated code contains no `cfc2`
  (`ps2_vu1_core.cpp:248`), and `R5900Context::vu0_vpu_stat` never carries the VU1 busy bit. The
  VIF1 DMA "completes" (STR cleared, D_STAT raised) synchronously inside the CHCR store
  (`:2301-2310`). `sceGsSyncPath` therefore never waits (`Kernel/Stubs/GS.cpp:1251`).
- **EE access to VU1 memory** goes through `Ps2IsSpecialAddress` -> `Load*/Store*` ->
  `mapVuMemory` (`ps2_memory.cpp:549`): 6 `lui 0x1100` sites in the generated code. ⚠ Constant-folded
  addresses bypass the check and land in rdram (`sub_00147940`: four `FAST_WRITE128(0x1100C8x0)`)
  — a pre-existing hole, unrelated to threading, to be fixed on its own.
- **VIF1 DMA data lifetime**: chain-mode transfers are copied (`chainBuf`, `:1473`), normal-mode
  transfers hand `processVIF1Data` a raw `m_rdram`/`m_scratchpad` pointer (`:2238/2255`); the guest
  may overwrite that memory the instant the CHCR store returns.
- **VU0 shares microVU's globals with VU1**: `vuRegs[0].VI[REG_VPU_STAT]` is RMW'd by VU1's JIT'd
  code for bits 0x100/0x200/0x400 (`microVU_Branch.inl:126/247/348/369/425`,
  `microVU_Compile.inl:569/587`) and by VU0's for bits 1/2/4 — a non-atomic shared word; the code
  cache is one mmap with disjoint per-unit regions (`ps2x_microvu.cpp:49-70`), the emitter pointer
  is `thread_local`, `g_kickBuf` is VU1-only. The MTVU machinery from PCSX2 is a shim
  (`shim/Config.h: THREAD_VU1 false`).

## PCSX2's model (MTVU + MTGS), as the reference

- The EE-side VIF1 interpreter forwards every VU1-affecting operation into a 16 MB ring
  (`MTVU.h`): `VU_EXECUTE(addr, top, itop, fbrst)`, `WRITE_MICRO`, `WRITE_DATA`, `VIF_UNPACK`
  (the RAW data plus a copy of the VIF registers; the unpack itself runs on the VU thread),
  `WRITE_ROW/COL`, `WRITE_VI/VFREGS`. The VU thread runs the programs.
- **Path-1 ordering is reserved by the EE at MSCAL time**: `ExecuteVU` calls
  `gifUnit.TransferGSPacketData(GIF_TRANS_MTVU)`, which increments `path1.mtvu.fakePackets` — a
  placeholder in the GIF unit's path order (`Gif_Unit.h:623-636`). The VU thread's XGKICK copies the
  packet into the path-1 buffer and parses only the A+D SIGNAL/FINISH/LABEL it needs to hand back
  (`ExecuteGSPacketMTVU`, `Gif_HandlerAD_MTVU`); after the program, `FinishGSPacketMTVU` pushes the
  packet onto `gsPackQueue`, which the MTGS (render) thread pops when its ring reaches the
  placeholder. Path 3 submitted later by the EE queues behind the placeholder.
- **EE waits on the VU thread only where it reads VU1 state**: VIF1 ROW/COL register reads
  (`Vif.cpp:300-335`), VU1 memory/register access (`VUmicroMem.cpp:57`, `VU1micro.cpp:25-29`),
  save states. MSCAL under MTVU does not wait for VU1 idle (`Vif_Codes.cpp:120`); VPU_STAT's busy
  bit is an *estimate* (`ExecuteVU`: busy for the last run's cycle count), and INSTANT_VU1 never
  sets it. GS interrupts raised by the VU thread's packets (SIGNAL/FINISH/LABEL) are latched in
  `mtvuInterrupts` and delivered by the EE in `Get_MTVUChanges`.
- Under MTVU the VU1-side writes to `VPU_STAT` are routed through the thread's own flags
  (`InterruptFlagVUEBit/VUTBit`), not the shared word.

## The design for this runtime: one **GS pipeline thread**

Because the guest already sees the VIF1+VU1+GIF pipeline as instantaneous, the cheapest correct
shape is not PCSX2's two threads but ONE consumer thread that runs, **in program order, exactly the
code that runs today**: the VIF1 interpreter (unpacks, MPG, MSCAL/MSCNT -> microVU), the path-3
packet handling including MSKPATH3 masking, the arbiter, and the GS parse into the worker queue.
The EE thread only *enqueues*: it copies each DMA transfer's bytes (or each HLE GIF packet) into a
ring and returns. Ordering is program order by construction (a single FIFO of "what the EE handed
the pipeline, in the order it handed it"), so no placeholder scheme is needed.

Ring commands (all produced on the EE thread, consumed by the pipeline thread):
- `VIF1_DATA(bytes)` — one `processVIF1Data` call's input (chain buffer or a copy of the rdram/
  scratchpad range).
- `GIF3(bytes, drainImmediately)` — one path-3 `submitGifPacket`.
- `DRAIN` — the explicit end-of-pass drain (`ps2_memory.cpp:2265`).
- `VIF1_REG(addr, value)` — a guest write to the VIF1 register block (FBRST reset, STAT, CYCLE...).
- `VU1_MEM_WRITE(addr, bytes)` — a guest store into VU1 memory (rare).
- `FENCE(token)` — the EE waits for the consumer to reach it (the sync points below).

EE-side sync points (`FENCE` + wait):
1. Guest LOADS from VU1 code/data memory (`mapVuMemory` reads) — drain first, then read.
2. VIF1 ROW/COL/... register READS that reflect stream state (today they read `m_ioRegisters`, i.e.
   the last guest write — keep that; no fence needed until STAT gains VBS).
3. GS readbacks (LocalToHost, DebugReadback) and `Sync(Reset)` — drain the pipeline, then the GS
   queue, as today.
4. Display flip (`OnDisplayFlip`): the flip snapshot must be ordered after the packets before it:
   send it through the ring as a command (`FLIP(dispfb1, dispfb2)`) so the pipeline thread
   enqueues the FlipSnapshot work item at the right point.
5. GS priv-register writes by the EE that change parse state (`gs_regs`: CSR, IMR, DISPFB, ...) —
   the frontend handles them separately from packets; CSR FINISH/SIGNAL set by the parse become
   asynchronous. This game never uses FINISH (the `[gsgpu:drain]` counters); SIGNAL/LABEL usage:
   to be censused before stage 2.
6. Shutdown / save-state-like paths: drain.

VU0/VU1 concurrency: VU0 keeps running on the EE thread. The shared `VPU_STAT` word must be split:
VU1's emitted RMWs (busy 0x100, D 0x200, T 0x400) go to a VU1-owned word (the shim already has the
`isVU1` selects at the emission sites); the runtime's MSCAL callback reads the D/T stop bits from
that word after the run. Everything else is already per-unit or `thread_local`.

Data lifetime: the ring COPIES every payload (chain buffers can be moved). A frame's VIF1+path-3
traffic is ~10-20 MB (`[ee:prof] bytes`); a 64 MB ring holds a few frames; the EE blocks on a full
ring (that is the pipeline's back-pressure, replacing the today's synchronous cost).

Diagnostics that assume the EE thread (`VifProfScope`, `g_vif*` counters, `g_gifParseState`, the
fast-unpack verify buffers) all move with the code onto the pipeline thread; they stay
single-threaded because there is exactly one consumer.

## Stages, each with its oracle

- **Stage 0 — the oracle (this session): `PS2X_PIPE_CAP` / `PS2X_PIPE_BENCH`.** Capture, from a
  quiescent pass boundary, the VU1 code+data image, the VIF1 registers, the mask state, and then
  every ring command above as the EE produces it (N passes). Replay it at startup through the real
  pipeline (VIF1 interpreter + microVU + arbiter + GS frontend), hashing every packet the arbiter
  hands to `GS::processGIFPacket` (path id + bytes, in order) and the final VU1 data image. The
  hash is the gate for every later stage (the threaded pipeline must produce the same packet
  sequence), and the replay's wall time is the pipeline's own bench, free of the game's real-time
  noise — the same role `tmp/cap_hel_fight.bin` plays for the rasterizer.
- **Stage 1 — the ring, consumed synchronously.** Every EE-side entry (`processPendingTransfers`'s
  VIF1 and GIF sections, `processGIFPacket`, the VIF1 register block, VU1 memory stores) writes ring
  commands; a `pumpPipeline()` executes them on the EE thread right away. Bit-exact by the oracle;
  a pure refactor that isolates the seam.
- **Stage 2 — the consumer thread.** `pumpPipeline` becomes a thread; the fences are added at the
  sync points; the VPU_STAT split lands; the flip goes through the ring. Oracle: the capture hash
  (the consumer runs the same code) + live soaks (600 s, audio on) + the raster capture hash.
  Measure: EE thread on-CPU per present, the pipeline thread's on-CPU, presents in the fight window.
- **Stage 3 — tuning.** Ring size / back-pressure; whether the GS parse stays on the pipeline
  thread or moves to the workers' side; the T8/other raster work runs in parallel.

## Stage 1 (landed as fork row 77, build 738): the ring, consumed synchronously

`include/runtime/ps2_gs_pipeline.h` / `src/lib/ps2_gs_pipeline.cpp`, `PS2X_GS_PIPELINE` (default ON,
`=0` = the direct path). Header-free: each producer entry (`processVIF1Data(ptr,size)`,
`submitGifPacket(Path3, ...)`, the end-of-pass drain in `processPendingTransfers`, the VIF1
register block in `writeIORegister`, the five VU1-memory store sites) tests `ps2gs::producerSide()`
(pipeline enabled AND not inside `pump()`), copies the command into the FIFO and calls `pump()`;
`pump()` sets a thread-local consumer flag and re-enters the same public entry points, so the
existing bodies run unchanged in program order. The capture taps moved to the enqueue side. VU1
memory stores are performed by the consumer only (stage 2 needs a single writer). Normal-mode
VIF1 DMA data is now COPIED into the ring (the raw-`m_rdram`-pointer lifetime hazard is gone).

## Stage 2 notes (from the survey, to build against)

- **VPU_STAT split.** The microVU port already carries PCSX2's `if (!mVU.index || !THREAD_VU1)`
  guards around every VU1-side write of `VU0.VI[REG_VPU_STAT]` (`microVU_Branch.inl:126/247/348/
  369/425`, `microVU_Compile.inl:569/587`). Rather than PCSX2's interrupt hand-back, point the VU1
  emission at a VU1-private status word (`g_vu1Stat`) and make `ps2x_microvu::runUnit` read the
  busy / D / T bits for VU1 from it (`ps2x_microvu.cpp:371-395`). VU0's word stays where it is.
- **Fences (EE waits for the consumer to drain):** the five VU1-memory READ sites
  (`ps2_memory.cpp:710/746/797/840/886`, via `mapVuMemory`), the GS syncs that need quiescent VRAM
  (`Sync(LocalToHost)` through `ConsumeLocalToHostBytes`, `Sync(DebugReadback)`, `Sync(Reset)`:
  `gs_frontend.cpp:167/209/1955/1968`), shutdown. VIF1 register READS keep returning
  `m_ioRegisters` (the last guest write) as today -- no fence until STAT gains VBS.
- **The flip** (`OnDisplayFlip` from the EE's DISPFB write) becomes a ring command so the
  FlipSnapshot work item is enqueued after the packets before it.
- **Consumer-side state that is EE-visible today** and must be handled: the MSCAL/MSCNT callbacks
  write `cpuContext->vu0_vpu_stat` D/T bits (`ps2_runtime.cpp:673/691`) -- route to an atomic the EE
  reads (nothing reads it today); `m_vif1PendingPath2ImageQwc` / the DIRECT-span statics are
  consumer-only; `m_ioRegisters` is written by the VIF1 register apply on the consumer (the map is
  unguarded and also read by the debug panel: give the VIF1 block its own array).
- **Back-pressure:** cap the FIFO's pending bytes (64 MB) and block the producer; a full ring is
  the pipeline's own budget signal.
- **Gates:** `tmp/pipebench.sh` hash (the consumer runs the same code, so it must be identical),
  `tmp/bench.sh` on the raster capture, 600 s soaks with audio, `eeprof.sh` (EE on-CPU per present,
  the pipeline thread's share), presents in the fight window.

## Stage 2 (fork row 78, builds 739-742): the consumer thread -- CORRECT, and NOT FASTER while the raster is the pole

`PS2X_GS_PIPELINE=2` (opt-in; the default stays 1). What landed: the consumer thread with a mutex/condvar
FIFO; fences at the five VU1-memory read sites, the three GS syncs that need quiescent VRAM and the
GS->EE readback; the display flip as a ring command; the VPU_STAT split (`g_vu1VpuStat`, VU.h) with the
ten JIT emission sites and `runUnit` on VU1's own word; a recursive mutex around `mVUblockFetch` (VU0
compiles on the EE, VU1 on the pipeline thread); the MSCAL/MSCNT callbacks off the EE's live context;
shutdown joins after the EE thread; back-pressure at 16 MB with bytes released at dequeue;
`[gs:pipeline]` statistics every 10 s.

Gates: `tmp/pipebench.sh` `pktHash=061e22452b040a80` in threaded mode (the consumer runs the same code),
raster `f754386ce41560b1`, 235 s fight and 150 s audio-on fight clean.

**Measured, four alternating 150 s fights (build 742):** threaded 1804 / 1835 presents vs synchronous
2073 / 2084 (-12%), EE frames 832 / 864 vs 1110 / 1099 (-23%); the EE blocked 23-29 s in ring
back-pressure; **no fence ever fired** (the game reads no VU1 memory from the EE). The GS coordinator
thread is the pole at ~60 ms of rasterization per frame in BOTH modes (`coordRaster` / presents), with
the workers ~30% idle behind its per-run barrier (`tail`), so the pipeline thread cannot buy frames:
the EE just runs ahead until the ring fills, then blocks at vblank-misaligned moments and loses whole
vblanks. With the 64 MB ring (build 741) it was worse (1723 vs 2104; 42 s blocked; ~9 frames of lag).

**Conclusion:** the EE side is no longer where the frame is. Stage 2 stays opt-in until the raster's
per-frame consumption drops below the EE's ~28 ms; the next lever is the GS coordinator (the per-run
fan-out barrier and band balance: `[gsgpu:items] tail`, `[gsgpu:handoff] bandIdle`), then stage 2
should be re-measured -- the machinery is in place and gated.

### cont.318 re-measure and the defect (fork rows 79-81)
Four alternating 150 s fights on build 746: threaded 1793 / 1744 presents vs synchronous 2273 / 2173 (-20%, as before). Not
oversubscription (6-thread pool), not placement (6-CPU pool set), not the ring cap (`PS2X_PIPE_CAP_MB`, row 80). Sliced per
matched `[loadkick:frame]` window the raster was busy 20-60% MORE ms per frame for equal primitives, and `perf record` on the
raster threads said why: ~60% of the raster on the SCALAR fallback (`SampleTexture` 14.7% vs 2.1%, four-wide loops 34% vs 60%,
`WriteCT24`/`ReadZ24` only in mode 2) -- the draws were not seeing the CT32+Z24 target.
**Cause:** `Kernel/Stubs/GS.cpp` `sceGsSwapDBuff` / `sceGsSwapDBuffDc` / `sceGsPutDrawEnv` applied the draw environment
(FRAME/ZBUF/XYOFFSET/SCISSOR/TEST...), the HLE clear and the clear packet with direct `gs().writeRegister` calls from the game
thread -- ordered when that thread drains the stream, a frame EARLY when the consumer does. The capture oracle never carried
those writes. **Fix (row 81):** ring commands `GsReg` / `GsClear` (+ capture events 7/8); `gsWriteRegisterOrdered` /
`gsClearFramebufferOrdered` in Support.h ride the ring when `threaded() && producerSide()`. Build 750: threaded 2272 / 2265
vs synchronous 2184 / 2287 presents -- parity; threaded soak + audio clean. Default stays 1 until the raster is cheaper: at
parity the raster is the pole in both modes, and mode 2 (EE ~35%, coordinator never starving) is the one that benefits first.
★ Rule for any further stage: grep every writer of GS/VU1 state outside the captured stream (HLE stubs, overrides) -- an oracle
only checks the traffic it captures.

## Expected outcome

EE compute ~46 -> ~28 ms/frame (VIF1-own ~2, VU1 ~17, GIF parse ~5, path-3 copies ~1 leave; a
ring copy of ~15 MB/frame arrives). The pipeline thread carries ~30 ms/frame. The raster's ~43 ms
becomes the pole until its own -10%; both together reach the 2-vblank step.
