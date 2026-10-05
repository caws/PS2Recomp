#pragma once

#include "runtime/gs/gs_backend.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class GsGpuPresentDevice;
struct GsGpuState; // gs_gpu_device.h -- only referenced here, never laid out
struct GsGpuGeom;
struct GsGlVertex;    // gs_gpu_device.h -- only referenced here, never laid out
struct GsGlGroup;
struct GsGlTexUpload;
struct GsGlTargetSeed; // ★ cont.330 -- VRAM seed rects for a GL render target
struct GsGlBatch;

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void EndPacket() override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    // Display-flip snapshot (cont.165): at the instant the guest flips DISPFB1/2, the OTHER
    // (pre-flip) buffer holds the last fully completed frame -- including the UI text the game
    // drew onto it right after ITS flip. Capture VRAM plus the PRE-flip display registers as one
    // unit; Present() renders the copy with those registers. No drain (the pre-flip buffer's
    // work was executed long ago; concurrent worker writes only touch the new back buffer).
    // Called from the GS privileged-register write paths via ps2xGsNotifyDisplayFlip().
    void OnDisplayFlip(uint64_t preFlipDispfb1, uint64_t preFlipDispfb2);
    PresentationFrame Present(const GSPresentationRequest &request) override;
    // cont.232: the presenter's cadence source (PS2X_GS_PRESENT_ONFLIP). Returns whether Present()
    // would render the frame-complete copy right now (a flip snapshot or per-Sync capture at most
    // kPresentCopyFreshMs old) and that copy's sequence number; false = Present() would fall back to
    // live VRAM. Lock-free; any thread.
    bool PresentSourceFresh(uint64_t &outSeq) const;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);
    void BeginTransferUnlocked(const GSTransferCommand &command);
    void UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes);
    // cont.168 (GPU-backend arc phase 1b, mirror completeness): after a VRAM-mutating op
    // the swizzle upload kernel does not cover (non-CT32 upload, local->local blit, clear,
    // direct WriteVram), copy the rect's authoritative bytes straight from CPU VRAM into
    // the device's mirror. Page-granular and deliberately conservative -- over-copying
    // authoritative bytes cannot be wrong. Call with m_vram stable (locked/drained).
    // Declared unconditionally (one class definition for every TU); only DEFINED when the
    // desktop GPU device is compiled in (PS2X_HAS_GS_GPU_DEVICE, a ps2_runtime-private
    // definition), and only called from inside that same guard.
    enum class MirrorPatchSource : uint8_t
    {
        Upload = 0,      // a host->local chunk the CT32 swizzle kernel could not take
        LocalToLocal = 1, // a GS-internal blit
        Clear = 2,       // ClearFramebuffer
        Poke = 3,        // direct WriteVram (host-side diagnostics)
    };
    void mirrorPatchRect(uint32_t psm, uint32_t bp, uint32_t bw,
                         uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                         MirrorPatchSource src);
    // Direct WriteVram pokes arrive ONE PIXEL AT A TIME and in bulk -- LOTR's movie HLE
    // blits each decoded FMV frame as ~229k single-pixel pokes (run176a measured 23M in
    // 100s). Mirroring each one individually is hopeless, so pokes accumulate into dirty
    // bounding boxes and mirror as whole rects.
    //
    // SEVERAL boxes, not one: that blit walks four column strips per row
    // (bp = 0x3003 + (x/128)*0x400), so the destination key changes every 128 pixels and a
    // single box would flush 1792x per frame (measured, run177a). One slot per live key
    // lets all four strips accumulate across the whole frame and flush once each.
    // Flushed when a slot is evicted, when a box gets large, before any other mirror op
    // (device FIFO order must equal VRAM write order), and at every drained boundary.
    // Call with m_mutex held.
    struct PokeBox
    {
        uint32_t psm = 0, bp = 0, bw = 0;
        uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        bool used = false;
    };
    static constexpr size_t kPokeBoxes = 8;
    void mirrorFlushPokes();            // all slots
    void mirrorFlushPokeSlot(size_t i); // one slot
    PokeBox m_pokeBox[kPokeBoxes]{};
    // Pages the CPU rasterizer could have written since the last full-mirror verify (one
    // byte per 8KB GS page). The verify subtracts these, so what it reports as stale is
    // divergence the rasterizer does NOT explain -- i.e. a real mirror gap. Armed only
    // under PS2X_GS_GPU_MIRRORVERIFY; touched under m_mutex (DrawPrimitive runs there).
    void markDrawPages(uint32_t psm, uint32_t bp, uint32_t bw,
                       uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1);
    std::array<uint8_t, 512> m_drawPages{};

    // cont.169 (GPU-backend arc phase 2a): per-PRIMITIVE shadow verify of the GPU
    // rasterizer against this CPU oracle. Sampling one primitive at a time deliberately
    // separates the two hard problems: this proves the PIXEL PIPELINE (perspective
    // interpolation, texture wrap/bilinear/CLUT, TFX, alpha test, Z test, blend, FBMSK)
    // with no tiling or submission-order machinery involved at all; ordering correctness
    // is then a separate question the whole-mirror MIRRORVERIFY gate already answers.
    //
    // Begin (before the CPU draws): seed the mirror's frame/Z pages with the CPU's
    // PRE-draw bytes and post the GPU raster. End (after): post the CPU's POST-draw bytes
    // for the same ranges. The device FIFO keeps seed -> raster -> compare in order.
    struct RasterVerifyCtx
    {
        bool active = false;
        // The Z range must be SEEDED whenever the depth test READS it (ZTST 2/3), not only
        // when the draw writes it -- a ZMSK=1 draw still depth-tests against a Z buffer the
        // mirror would otherwise hold a stale copy of, and the GPU then keeps or discards
        // different pixels than the oracle. It is VERIFIED only when actually written.
        bool zSeed = false;
        bool zWritten = false;
        uint32_t frameOff = 0, frameLen = 0;
        uint32_t zOff = 0, zLen = 0;
        uint64_t tag = 0;
    };
    // With bs/bg non-null the primitive is only BUILT (state + geometry filled) and
    // nothing is seeded or posted -- that is how the batch accumulator reuses exactly the
    // same, already-verified setup code as the single-primitive gate.
    bool rasterVerifyBegin(const GSPrimitiveBatch &batch, RasterVerifyCtx &out,
                           GsGpuState *bs = nullptr, GsGpuGeom *bg = nullptr);
    // cont.170: DrawSprite is not a degenerate triangle (flat colour/Z/fog from v1, an
    // axis-aligned rect, screen-space UV lerp), so it gets its own setup; everything that
    // does not depend on the primitive's SHAPE lives in gs2FillCommonPrimState, and the
    // seed-then-raster sequence in rasterSeedAndPost, shared by both.
    bool rasterVerifyBeginSprite(const GSPrimitiveBatch &batch, RasterVerifyCtx &out,
                                 GsGpuState *bs = nullptr, GsGpuGeom *bg = nullptr);

    // cont.171 (phase 2b step 2): the BATCH accumulator. A batch is a run of consecutive
    // primitives sharing one render target with no non-draw VRAM mutation between them --
    // both boundaries are load-bearing. Tiles are only VRAM-disjoint within one target
    // (change fbp/fbw and two screen tiles can alias the same word), and a batch is
    // rasterized at FLUSH time, so an upload landing mid-batch would be visible to every
    // primitive in it while the oracle shows it only to those that follow.
    struct GpuBatchAccum
    {
        std::vector<GsGpuState> states;
        std::vector<GsGpuGeom> prims;
        std::vector<uint64_t> stateHashes; // parallel to states; linear scan (<=128)
        uint64_t targetKey = ~0ull;
        unsigned long mutSeq = 0;
        bool verifying = false;
        // Primitives the GPU path declined since this batch began. A batch that
        // skipped ANY primitive cannot be verified: the oracle drew it and the GPU
        // did not, so a mismatch would say nothing about ordering.
        unsigned long skipped = 0;
        // Pages this batch has already WRITTEN (frame + Z of every primitive so far), as a
        // 512-bit map over the 8KB GS pages. A primitive that SAMPLES one of these is a
        // read-after-write hazard: within a tile the walk is ordered, but two tiles are
        // separate workgroups with no ordering between them, so the reader could run before
        // the writer. Such a primitive has to start a new batch.
        std::array<uint64_t, 8> writtenPages{};
        // Verified batches keep the PRE-batch VRAM so a failure can be replayed one
        // primitive at a time -- the decisive A/B between "tiling/ordering is wrong" and
        // "one primitive in here shades wrong".
        std::vector<uint8_t> preVram;
        // The originating draw commands, kept only for verified batches so a failure can be
        // BISECTED: re-run the oracle into a scratch VRAM for prims [0..k) and compare
        // against a GPU replay of the same prefix. ~12 iterations name the exact primitive.
        std::vector<GSPrimitiveBatch> srcPrims;
        uint32_t maxX = 0, maxY = 0; // batch bbox, sizes the tile grid
        uint32_t lastStateIdx = ~0u; // cont.233 census: consecutive same-state runs
    };
    GpuBatchAccum m_gpuBatch;
    unsigned long m_lastDegen = 0, m_lastOffscr = 0; // draws-nothing rejections
    // While set, DrawPrimitive draws PIXELS ONLY -- no census, no batch accumulation, no
    // verify gate. The bisect replay re-enters the oracle and must not recurse into any of
    // that. Touched only under m_mutex, on the thread already inside gpuBatchFlush.
    bool m_rawDraw = false;
    std::vector<uint8_t> m_bisectScratch;
    void gpuBisectFailingBatch(const std::vector<GsGpuState> &states,
                               const std::vector<GsGpuGeom> &prims);
    void gpuBatchAppend(const GSPrimitiveBatch &batch);
    void gpuBatchFlush();
    void gs2FillCommonPrimState(const GSPrimitiveBatch &batch, GsGpuState &p, uint32_t mip);
    void rasterSeedAndPost(const GsGpuState &st, GsGpuGeom &&g, const RasterVerifyCtx &out);
    void rasterVerifyEnd(const RasterVerifyCtx &ctx);

    // Off-thread rasterization (PS2X_GS_THREAD, default OFF): draw/transfer/upload
    // commands enqueue to a worker; read-side entry points drain first. All queue items
    // are self-contained copies.
    struct WorkItem
    {
        enum class Kind : uint8_t { Draw, Transfer, Upload, DrawRun, FlipSnapshot } kind = Kind::Draw;
        GSPrimitiveBatch batch{};
        std::vector<GSPrimitiveBatch> run; // DrawRun: consecutive draws, one queue item (cont.230)
        uint64_t flipFb1 = 0, flipFb2 = 0;  // FlipSnapshot: the pre-flip display registers (cont.231)
        GSTransferCommand transfer{};
        std::vector<uint8_t> bytes;
    };
    void workerLoop();
    void enqueueWork(WorkItem &&item);
    void drainQueue() const;
    // cont.230 PS2X_GS_DRAWRUN: draws accumulate here (EE thread only) and go to the worker as ONE
    // DrawRun item per GIF packet / drain / non-draw operation, instead of one lock+notify per primitive.
    void flushPendingDraws() const;
    mutable std::vector<GSPrimitiveBatch> m_pendingRun;
    static size_t queuePrimsOf(const WorkItem &item);
    void recycleRunVectorLocked(std::vector<GSPrimitiveBatch> &&v);
    static constexpr size_t kRunPoolMax = 16;
    mutable std::vector<std::vector<GSPrimitiveBatch>> m_runPool; // guarded by m_queueMutex
    size_t m_queuePrims = 0;                                     // guarded by m_queueMutex
    void SnapshotVramNoDrain(std::vector<uint8_t> &out) const;

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    // ★ cont.207 BANDED RASTERIZATION. Each rasterizer thread owns a disjoint band of SCANLINES
    // (`bandY0..bandY1`, inclusive), so no two threads ever write the same pixel and primitive
    // order is preserved within a band -- no VRAM lock, no ordering change. The coordinator
    // resolves `m_draw` ONCE before fanning out, so helpers only ever READ it.
    // A band of [INT_MIN, INT_MAX] is the whole screen, i.e. the single-threaded path.
    void DrawSprite(const GSPrimitiveBatch &batch, int bandY0, int bandY1);
    void DrawTriangle(const GSPrimitiveBatch &batch, int bandY0, int bandY1);
    void DrawLine(const GSPrimitiveBatch &batch, int bandY0, int bandY1);
    // Rasterize `batch` restricted to one band; the shared entry the pool's threads call.
    void RasterBand(const GSPrimitiveBatch &batch, int bandY0, int bandY1);
    // Fan `batch` across the band pool and join. Falls back to a direct full-screen call when
    // the pool is off or the primitive is too short to be worth the handoff.
    void RasterFanOut(const GSPrimitiveBatch &batch, int minY, int maxY);
    // cont.208: deterministic rasterizer capture + replay bench (see gs_cpu_backend.cpp).
    // Resolve the per-batch draw state into this thread's m_draw. Every band thread calls it for
    // every primitive it rasterizes.
    // ★ cont.329 phase 2/3 (the GL renderer arc): translate a run of draws into the GL geometry
    // stream, decoding any texture the run introduces. A MEMBER because it needs the per-draw
    // resolves (resolveDraw / resolveTexMip), the decoded CLUT and the VRAM read helpers -- using
    // the rasterizer's own source of truth rather than a second implementation that can drift.
    void glSubmitRun(const GSPrimitiveBatch *items, size_t count);
    // ★ cont.329 phase 3b. One entry per DISTINCT texture (tbp/tbw/psm/size/cbp), not per
    // generation. Measured: folding the page generation into the key made 13312 decodes produce
    // three distinct textures -- 100% waste -- because a generation bump means "these pages were
    // written", not "this texture changed". The generation is now only a cheap hint that gates a
    // content hash; `epoch` advances only on a REAL content change, and the device-side key is
    // derived from it, so a stale decode still cannot be served.
    struct GlTexEntry
    {
        uint64_t gen = ~0ull;  // the page-generation sum last seen
        uint64_t hash = 0;     // content hash at the last decode
        // ★★★ cont.332: the hash of the DECODED RGBA this key last produced. `hash` above is over
        // the texture's whole 8 KB VRAM pages, so a neighbouring write in the same page forces a
        // re-decode of an unchanged texture; comparing the decoded OUTPUT separates "the texture
        // really changed" from "the page moved under it". Filled only under PS2X_GS_GLR_DECVERIFY.
        uint64_t outHash = 0;
        bool flatLast = false; // cont.332g: the last decode came out UNIFORM (a blank texture)
        uint32_t epoch = 0;    // bumped only when the content actually changed
        bool shipped = false;
    };
    std::unordered_map<uint64_t, GlTexEntry> m_glTexCache;
    // ★ cont.329 phase 3c: runs accumulated before submission. A batch costs ~7.3 ms of driver
    // work regardless of size and carried only ~42 vertices, so submitting per run spent 181 s of
    // a 234 s run on overhead for 0.8 s of drawing. Flushed on a vertex threshold
    // (PS2X_GS_GLR_BATCHVERTS) or a render-target change. Held as the three vectors rather than a
    // GsGlBatch because this header only forward-declares the device's types, as it already does
    // for GsGpuState/GsGpuGeom.
    std::vector<GsGlVertex> m_glPendVerts;
    std::vector<GsGlGroup> m_glPendGroups;
    std::vector<GsGlTexUpload> m_glPendTex;
    // ★★★ cont.330: VRAM seed rects waiting to go out with the pending batch (see GsGlTargetSeed).
    std::vector<GsGlTargetSeed> m_glPendSeeds;
    uint32_t m_glPendRefW = 0, m_glPendRefH = 0;
    uint64_t m_glPendTargetKey = 0;
    uint32_t m_glPendZbp = 0, m_glPendZpsm = 0; // cont.344: the pending batch's ZBUF (pages, psm)
    // ★★ cont.329g census: how many vertices are sitting in the accumulator RIGHT NOW. The
    // presenter thread samples this when it asks for a frame, so "geometry still unflushed at the
    // moment the frame was resolved" becomes a number instead of an argument. Atomic because the
    // GS worker writes it and the presenter reads it; relaxed, since it is a diagnostic.
    std::atomic<uint32_t> m_glPendCount{0};
    // Framebuffer addresses we have rendered into, so a texture naming one can be promoted to
    // sample that GL target instead of decoding memory the GL path never writes (phase 4b).
    std::unordered_set<uint64_t> m_glTargetKeys;
    // Per-target height high-water mark. The vertex transform divides by this, so it must be the
    // TARGET's extent and must not change between batches (see glSubmitRun).
    std::unordered_map<uint64_t, uint32_t> m_glTargetH;
    // ★★★ cont.329k: the page-generation fingerprint of each GL target's memory AT THE MOMENT WE
    // LAST DREW INTO IT. A draw that samples a target's address must only be promoted to sampling
    // the GL target while that target is still the FRESHEST copy: this game uploads a movie/still
    // image into a buffer it also renders into, and promoting then sampled an EMPTY GL target, so
    // the composite contributed nothing and a 4-strip backdrop stayed on screen.
    std::unordered_map<uint64_t, uint64_t> m_glTargetGen;
    // ★★★ cont.330 TARGET-FROM-VRAM SEEDING. Per target, the per-PAGE transfer generation as of
    // the last time we folded VRAM into it, and the global transfer sequence as of the last check.
    // The sequence is the cheap per-draw gate; the per-page vector is the dirty set itself, which
    // is what keeps the seed from overwriting GL-rendered pixels on pages the guest never wrote.
    std::unordered_map<uint64_t, std::vector<uint32_t>> m_glTargetSeedGen;
    std::unordered_map<uint64_t, unsigned long long> m_glTargetSeedSeq;
    // ★ cont.331 authority model: the same per-target transfer-sequence watermark, for the
    // PRESENT-time seed (separate from the draw-path one so neither suppresses the other).
    std::unordered_map<uint64_t, unsigned long long> m_glPresentSeedSeq;
    void glSeedTargetFromVram(const GSDrawState &st, uint64_t targetKey, GsGlBatch &b);
    // ★★★★ cont.331 authority model phase A: the same seed, addressed by explicit surface
    // parameters so it can run OUTSIDE the draw path -- specifically at the present resolve, where
    // there is no GSDrawState. glSeedTargetFromVram() forwards to this.
    void glSeedTargetFromVramParams(uint32_t fbp, uint32_t fbw, uint32_t psm, uint64_t targetKey,
                                    GsGlBatch &b);
    // Apply any pending dirty VRAM rects to the DISPLAY target before it is resolved for present.
    void glSeedDisplayTargetForPresent(uint32_t fbp, uint32_t fbw, uint32_t psm);
    void glSnapshotDisplayTargets(uint64_t dispfb1, uint64_t dispfb2, uint64_t seq); // cont.345
    void glFlushPending();
    // ★★ cont.329h PER-DRAW-CALL COMPARISON (`PS2X_GS_DRAWCMP=<flip>`): snapshot the DRAW TARGET
    // at every run boundary of one guest frame, in whichever renderer is active. The hook is the
    // worker loop's Draw/DrawRun arm, which both renderers pass through in the same order, so the
    // snapshots of a CPU run and a GL run line up run for run and can be differenced offline.
    void drawCmpCheckpoint(const GSPrimitiveBatch *items, size_t count);
    // FNV-1a over the texture's page range plus the decoded palette. Linear reads with no swizzle
    // math and no per-texel palette lookup, so it costs a small fraction of the decode it avoids.
    uint64_t glTexContentHash(uint32_t pageBase, uint32_t pages);

    void resolveDraw(const GSDrawState &state);
    // cont.228: fill the mip-dependent half of the resolved texture state. Split out of
    // resolveDraw because the mip is not known until the primitive's own setup (DrawTriangle
    // computes triMip after the winding/denominator), so SampleTexture calls it on its first
    // fetch of each draw.
    void resolveTexMip(const GSDrawState &state, uint32_t mip);
    // ★ cont.320: the decoded-index cache decision for the resolved level (hit / fill / swizzled),
    // and the per-draw write notice that stomps cached palettes/textures the draw's targets cover --
    // run for EVERY draw of a run on every thread, including the ones band-skip never rasterizes.
    void texCacheResolve(const GSDrawState &state);
    void noteDrawWrites(const GSContext &ctx);
    // cont.228: the PSMT4 fast path's diagnostics (PS2X_GS_CLUTVERIFY / _CLUTMUTATE), deliberately
    // out-of-line so they cost the production path one predicted branch and nothing else.
    uint32_t sampleFastP4Diag(const GSDrawState &state, uint32_t tbp, uint32_t tbw,
                              int sampleU, int sampleV);
    uint32_t sampleFastP8Diag(const GSDrawState &state, uint32_t tbp, uint32_t tbw,
                              int sampleU, int sampleV);
    // Rasterize a whole RUN of primitives, one band per thread. One fan-out amortised over the
    // entire run, so even single-pixel primitives are parallel -- which per-primitive fan-out
    // could not do (only ~5% of prims cleared its threshold).
    void RasterRunFanOut(const GSPrimitiveBatch *items, size_t count);
    void rasterCapRecord(const GSPrimitiveBatch &batch);
    void rasterBenchRun();
    void startBandPool();
    void stopBandPool();
    void bandWorkerLoop(unsigned index);
    void bandRebuildOwners(); // cont.318 adaptive stripe ownership
    void bandDynClaimLoop(const GSPrimitiveBatch *items, size_t count, int runY0, int runY1, unsigned dynGroups, int dynBase, int dynRows); // cont.318 dynamic groups
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v, uint32_t mip = 0);
    // ★★ cont.231 PS2X_GS_FASTPIX: the per-state specialised pixel path. sampleTextureT and
    // writePixelT are the ONE body each of SampleTexture / WritePixel, templated on the
    // loop-invariant draw state so the compiler drops the per-pixel branches; the public
    // functions dispatch to them, so the generic row loop and the fast triangle loop share the
    // same arithmetic (no second copy to drift). PixelPlan holds the per-primitive constants the
    // pixel write used to re-derive from the context on every pixel.
    struct PixelPlan;
    struct TriRowSetup;
    void makePixelPlan(const GSDrawState &state, PixelPlan &plan) const;
    void writePixelDispatch(const GSDrawState &state, const PixelPlan &plan, bool fast, int x, int y, int z,
                            uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    // ★★★ cont.262 THE SELECTOR SEAM (PCSX2 `GSScanlineSelector` / `SetupDraw`, adapted).
    // PCSX2 JITs one scanline per draw state, keying a u64 over ~30 dimensions so the emitted code
    // branches on NONE of them (GS/Renderers/SW/GSDrawScanline.cpp). We hoisted the state DECODE
    // into PixelPlan but still branched on its VALUES per pixel -- that difference is the renderer
    // gap (progress.md cont.261). We do not need PCSX2's JIT: `PS2X_GS_STATECENSUS` measured this
    // game at 48.8M draws over only **46 distinct draw states, top 3 = 89.7%**, so an explicit LIST
    // of specialised classes covers what a JIT would. `kPipe` is that class; `pipeClassFor()` is the
    // selector. A JIT, if a future game's census ever shows a long tail, plugs in at exactly this
    // seam by supplying another kPipe value -- no caller changes.
    // ⚠ Specialise an explicit list of MEASURED-COMMON keys, never a cross product of dimensions:
    // the row loop already has 5 boolean dims (32 instantiations) and this binary is 900 MB.
    enum PipeClass : int
    {
        kPipeGeneric = 0,  // every branch stays runtime -- the pre-cont.262 path, bit-exact
        kPipeStdAlpha = 1, // ate runtime, ztst=GEQUAL, abe with ABCD=0101 (Cs,Cd,As,Cd), no PABE
        // cont.317: the Helm's Deep terrain -- 28% of raster ticks after FASTP8, all of it key 20 on the
        // generic pipe. [gs:keysig]: abe, ABCD=0201 (Cs,0,As,Cd) => Cd + (Cs*As>>7), no PABE, ztst=GEQUAL.
        kPipeAddAlpha = 2,
        // cont.319b: the sprite plans (PS2X_GS_KEYCENSUS on the recorded fight, sprites = 34% of raster ticks).
        kPipeDarken = 3,   // abe with ABCD=2101 (0,Cd,As,Cd) => Cd + ((0 - Cd)*As>>7): the full-screen darken passes
        kPipeOpaque = 4,   // abe off: the colour is the source (the clears and flat fills)
        // cont.321: the ZTE=0 full-screen sprite PAIR cont.319b un-dropped -- abe with ABCD=0122 (Cs,Cd,FIX,0)
        // => ((Cs - Cd) * FIX >> 7), FIX=128, ZTST=ALWAYS, zmask. 13.4% of the live fight's raster ticks on the
        // per-pixel path (two 512x448 draws per frame), the largest single non-four-wide item on 764.
        kPipeSubFix = 5,
        // cont.321b: tri 0 (untextured, flat, no fog) abe ABCD=0221 FIX=128 (Cs,0,FIX,Cd) => Cd + (Cs*FIX>>7), ZTST=GREATER,
        // zmask: 3.9% of the live fight's raster ticks (12 M tiny draws) on the generic loop. Selected by isAddFixPlan()
        // -- NOT by pipeClassFor(), whose value the key census packs into two bits.
        kPipeAddFix = 6,
    };
    static int pipeClassFor(const PixelPlan &plan);
    // cont.319b: the depth-test shape of a four-wide write. The triangle classes are GEQUAL; the
    // sprite plans are ZTST=ALWAYS (no Z read unless the write-back needs it).
    enum ZClass : int
    {
        kZGequal = 0,
        kZAlways = 1,
        kZGreater = 2, // cont.321b: ZTST=GREATER (unsigned z > stored), the kPipeAddFix triangles
    };
    static bool isAddFixPlan(const PixelPlan &plan);
    static int spritePipeClassFor(const PixelPlan &plan);
    struct SpriteSetup;
    template <bool kTme, bool kFst, bool kLinear, int kPipe>
    void drawSpriteRowsSimd4(const GSDrawState &state, const PixelPlan &plan, const SpriteSetup &sp);

    template <bool kFast, bool kFge, int kPipe = kPipeGeneric>
    void writePixelT(const GSDrawState &state, const PixelPlan &plan, int x, int y, int z,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    template <bool kFst, bool kLinear>
    uint32_t sampleTextureT(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    template <bool kTme, bool kFst, bool kLinear, bool kIip, bool kFge, int kPipe = kPipeGeneric>
    void drawTriangleRowsFast(const TriRowSetup &rs);
    // ★★★ cont.267 TIER 2 (PS2X_GS_SIMD4, default OFF): the same row walk FOUR PIXELS AT A TIME.
    // cont.265/266 measured the rasterizer at ~928 instructions per shaded pixel with IPC 3.13 and
    // a 0.38% L1 miss rate -- instruction-bound, not memory-bound -- and the hot pixel body at 6.6%
    // SIMD / 93% scalar byte manipulation. Nothing is separable (cont.266b: the byte work is
    // uniformly diffuse, ~2.6% available at any one site), so the only lever that divides the count
    // is WIDTH: four pixels per iteration, PCSX2's `CDrawScanline` shape.
    // ⚠ Every lane must evaluate the SCALAR expression with the same operands in the same order --
    // this build is -msse4.1 with no FMA, so per-lane SSE is IEEE-identical to scalar and the bench
    // hash (aa9cd72b600021ce) is the gate proving it.
    template <bool kTme, bool kFst, bool kLinear, bool kIip, bool kFge, int kPipe = kPipeGeneric, int kZ = kZGequal>
    void drawTriangleRowsSimd4(const TriRowSetup &rs);
    // The row-span solve (cont.229), shared by the scalar fast loop and the SIMD4 loop so the two
    // cannot drift -- the generic loop keeps its own copy as the reference implementation.
    static void solveRowSpan(const TriRowSetup &rs, float py, int &spanX0, int &spanX1);
    // ★★★ cont.270 (PS2X_GS_QUADWRITE): the pixel write for FOUR pixels with colour resident in
    // 16-bit SIMD lanes -- the structure of PCSX2's `GSDrawScanline` (rb/ga 16-bit lanes carried
    // through the whole chain), with OUR arithmetic preserved exactly so the bench hash still
    // gates it. Only the kPipeStdAlpha + kFast shape; the caller falls back per draw otherwise.
    // Returns nothing: inactive lanes are written back with their original value, which is
    // observationally identical because raster rows are partitioned across band threads.
    // Arrays (16-byte aligned at the call site) rather than __m128i, so this header stays free of
    // the target-specific SIMD include (x86 immintrin.h vs sse2neon.h on ARM).
    // ★ cont.324: the per-draw plan values arrive PRE-RESOLVED in a QuadWriteCtx (a caller-local
    // built once per primitive by makeQuadWriteCtx) so the quad never re-reads the plan through an
    // aliased reference on every group -- the .cpp explains what the annotate showed.
    struct QuadWriteCtx;
    QuadWriteCtx makeQuadWriteCtx(const PixelPlan &p) const;
    // cont.325 PS2X_GS_FZPAIR: the row's frame/Z address terms and the wrap guard, resolved once per ROW
    // (makeQuadRow) -- a VRAM byte store may alias the u32 tables, so per quad the compiler re-read them.
    struct QuadRowCtx;
    static QuadRowCtx makeQuadRow(const QuadWriteCtx &q, int y, int xLast);
    template <bool kFge, int kPipe, int kZ = kZGequal>
    void writePixelQuad(const QuadWriteCtx &q, const QuadRowCtx &row, int xg, int y,
                                const int32_t *z, const uint32_t *colour, const int32_t *fog,
                                int coverMask);
    // ★★★ cont.271 (PS2X_GS_QUADTEX): the bilinear sampler's per-pixel SETUP for four pixels at
    // once -- perspective divide, texture scale, floor, the fractional weights and the wrap --
    // which cont.267's attribution put at 24.7% of the hot symbol. The 16 texel gathers stay
    // scalar (they are irreducible), and `bilinear4` stays exactly as it is, per pixel: it is
    // already lane-parallel across RGBA, so widening it would not reduce its op count.
    // Only the devirtualised PSMT4 tap with a REPEAT/REPEAT or CLAMP/CLAMP wrap (82.56% of
    // texels, 100% of wrap pairs -- cont.268's census); anything else keeps the scalar sampler.
    // ★ cont.324: the per-draw sampler state (m_draw's texture fields) arrives in a QuadTexCtx, a
    // caller-local built once per primitive by makeQuadTexCtx -- same reason as QuadWriteCtx.
    struct QuadTexCtx;
    QuadTexCtx makeQuadTexCtx() const;
    void sampleQuadBilinearP4(QuadTexCtx &tc, const float *sIn, const float *tIn, const float *qIn,
                              int coverMask, uint32_t *out);
    bool drawTriangleFastDispatch(const TriRowSetup &rs);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    // Dispatcher: tries the GPU present decode (m_gpuDecode, set only on the Present()
    // snapshot backend when PS2X_GS_GPU_PRESENT/_VERIFY is on) and falls back to -- or in
    // verify mode compares against -- the CPU decode in CopyFrameToHostRgbaCpu.
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;
    bool CopyFrameToHostRgbaCpu(const GSFrameReg &frame,
                                uint32_t width,
                                uint32_t height,
                                std::vector<uint8_t> &outPixels,
                                bool preserveAlpha,
                                bool useLocalMemoryLayout,
                                bool frameBaseIsPages,
                                uint32_t sourceOriginX,
                                uint32_t sourceOriginY) const;

    // Raw function pointers, not std::function: every table entry is a plain GSMem free
    // function, and the tables are hit per PIXEL (profiling cont.158b measured the
    // type-erasure trampoline in the hot draw path) — raw pointers drop that layer.
    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    // Per-batch resolved draw state (set once at DrawPrimitive entry; batch.state is
    // immutable for the batch, so these cannot go stale): the psm-table lookups and
    // page-base->block conversions hoisted out of the per-pixel WritePixel/SampleTexture.
    struct ResolvedDraw
    {
        ReadVramFunc frameRead = nullptr;
        WriteVramFunc frameWrite = nullptr;
        ReadVramFunc zRead = nullptr;
        WriteVramFunc zWrite = nullptr;
        ReadVramFunc texRead = nullptr;
        // ★★ cont.225: true when this draw targets the overwhelmingly common pair -- CT32 frame +
        // Z24 depth. WritePixel then calls PixelStorageTraits<> DIRECTLY, so Address/ReadAt/WriteAt
        // inline and each pixel's address is computed once per buffer instead of once per access.
        bool fastCt32Z24 = false;
        uint32_t fbp = 0;
        uint32_t fbw = 1;
        uint32_t zbp = 0;
        // ★★ cont.227: the DECODED CLUT -- this file's own "// TODO: clut cache", and PCSX2's
        // actual model (GSClut::Read32 expands the palette into m_buff32 once per dirty
        // TEX0/TEXA; the texture unit samples THAT, never VRAM per texel). Filled in
        // resolveDraw, so it is at most ONE PRIMITIVE stale -- strictly fresher than hardware,
        // which reloads only when TEX0/CLD changes. clutMask is entries-1 (T4 family = 16).
        // cont.317: 256 entries so PSMT8 (half the Helm's Deep raster ticks) decodes too.
        uint32_t clut[256] = {};
        // ★ cont.324: the same palette as [r,g,b,a] floats -- bilinear4's unpack (cvtepu8 -> cvtepi32_ps)
        // evaluated once per decoded entry, so the four-wide lin tap loads its operand directly.
        alignas(16) float clutF[256][4] = {};
        bool clutValid = false;
        uint32_t clutMask = 0u;
        // ★★ cont.227b: REUSE the decode across draws. Under the band pool every thread
        // resolves EVERY primitive (see resolveDraw's contract above), so a decode-per-draw is
        // paid N times while its saving is only this thread's share of the texels -- measured as
        // a 12% REGRESSION at 8 threads against a 7.7% win at 1. PCSX2 re-reads the palette only
        // on a dirty TEX0/TEXA (GSClut::Read32) and invalidates it explicitly
        // (GSClut::InvalidateRange(start, end, is_draw)); this mirrors both. Two invalidations:
        // a process-wide VRAM generation (uploads/transfers/reset) and clutStomped, the
        // draw-writes-the-page-it-samples case. Thread-local is sound because every thread walks
        // the same run in the same order and applies the same tests.
        uint64_t clutKey = ~0ull;
        uint64_t clutGen = ~0ull;
        uint32_t clutPage = ~0u;
        bool clutStomped = false;
        // ★★ cont.228: the SampleTexture PROLOGUE, resolved once per primitive. Every field here
        // is invariant for a draw -- the mip is constant per primitive (`triMip` is computed once
        // per triangle, and sprites always pass 0) -- yet all of it was recomputed on EVERY texel
        // fetch: the MIPTBP1/2 decode, texW/texH, and the six CLAMP fields. `texMip` is the mip
        // they hold; resolveDraw sets it to ~0u so the first SampleTexture call of each primitive
        // refills them, which needs no change at the call sites.
        uint32_t texMip = ~0u;
        uint32_t texTbp = 0;
        uint32_t texTbw = 0;
        int texW = 1;
        int texH = 1;
        float texWf = 1.0f;
        float texHf = 1.0f;
        // 1 / (16 << mip). Exact: the divisor is a power of two, so multiplying by the reciprocal
        // is bit-identical to the division it replaces.
        float texFstScale = 1.0f / 16.0f;
        uint8_t texWrapU = 0;
        uint8_t texWrapV = 0;
        uint16_t texMinU = 0, texMaxU = 0, texMinV = 0, texMaxV = 0;
        // PSMT4 with a decoded CLUT: the devirtualised sampler, which calls
        // PixelStorageTraits<P4> directly instead of going through m_draw.texRead.
        bool texFastP4 = false;
        // cont.317: the same devirtualised tap for PSMT8 (PixelStorageTraits<P8> + the 256-entry decode).
        bool texFastP8 = false;
        // ★★ cont.268: the sampler's loop-invariant decisions, resolved here rather than per call.
        // texWrapClass: 0 = REPEAT/REPEAT (96.15% of this game's texel fetches), 1 = CLAMP/CLAMP
        // (3.85%), 2 = keep the runtime `switch (mode & 3)`. texTapFast: the devirtualised PSMT4
        // tap with every diagnostic gate already known off.
        int texWrapClass = 2;
        uint32_t texWrapMaskU = 0u, texWrapMaskV = 0u;
        bool texTapFast = false;
        // ★ cont.320 PS2X_GS_TEXCACHE: the decoded-INDEX texture cache, resolved once per draw in
        // texCacheResolve (from resolveTexMip). texLin = the resolved level's palette indices in a
        // linear [v * texLinW + u] byte array (T4 nibble or T8 byte), or null = the swizzled tap.
        const uint8_t *texLin = nullptr;
        uint32_t texLinW = 0u;
        uint32_t texTaps = 0u;   // taps served under the current key (the predictor's input)
        void *texPred = nullptr; // the current key's predictor entry
    };

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    // ★ cont.209: PER-THREAD. Under batch-level banding every rasterizer thread walks the whole
    // run at its own pace, so at any instant two threads are on DIFFERENT primitives and therefore
    // different resolved state. A single shared copy would be a data race and would draw with the
    // wrong psm handlers. Static (not per-instance) is fine: each thread resolves it before use,
    // and only one backend rasterizes.
    static thread_local ResolvedDraw m_draw;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    mutable std::mutex m_queueMutex;
    mutable std::condition_variable m_queueCv;     // wakes the worker
    mutable std::condition_variable m_queueIdleCv; // wakes drainers/backpressured producers
    mutable std::atomic<bool> m_presentPriority{false}; // presenter needs m_mutex next
    // CLUT shadow (PS2X_GS_CLUTSHADOW, default ON; cont.165): the last-uploaded 16-entry CT32
    // CLUT payload per destination block. The LOTR menu scene renders into a 512x512 target that
    // covers the font-CLUT blocks with FBMSK=ff000000 (RGB written, alpha kept), zeroing the CLUT
    // colors between their upload and the text draws (the text DMA stalls on FLUSHA behind the
    // very VU1 batch that stomps). When a CSM1 CT32 CLUT entry reads back RGB-zeroed with the
    // uploaded alpha intact, LookupCLUT serves the uploaded entry -- what real hardware would
    // sample had the buffers not collided. Touched only under m_mutex.
    std::map<uint32_t, std::array<uint32_t, 16>> m_clutShadow;
    mutable std::mutex m_presentCopyMutex;
    std::vector<uint8_t> m_presentCopy;                  // VRAM at the last frame-complete moment
    mutable std::atomic<uint64_t> m_presentCopyTickMs{0};
    // Display registers paired with m_presentCopy (a flip snapshot renders with the PRE-flip
    // registers -- the buffer the completed frame lives in). Guarded by m_presentCopyMutex.
    bool m_presentCopyHasRegs = false;
    uint64_t m_presentCopyDispfb1 = 0;
    uint64_t m_presentCopyDispfb2 = 0;
    // Bumped on EVERY m_presentCopy capture (flip + Sync sites, under m_presentCopyMutex):
    // the GPU decode's upload-cache key. The capture TICK (milliseconds) is not unique --
    // two Sync captures in the same ms made the GPU decode a stale upload (run172's CT24
    // adjacent-video-frame LSB mismatches).
    std::atomic<uint64_t> m_presentCopySeq{0}; // atomic: the presenter polls it (PresentSourceFresh)
    std::deque<WorkItem> m_queue;
    size_t m_queueUploadBytes = 0;
    bool m_workerBusy = false;
    bool m_workerStop = false;
    std::thread m_worker;

    // ---- cont.207: banded rasterizer pool -------------------------------------------------
    // PS2X_GS_RASTER_THREADS=N (default 1 = the previous single-threaded behaviour, so this is
    // opt-in and A/B-able in one binary). The pool holds N-1 helpers; the calling worker thread
    // rasterizes band 0 itself, so N=4 means 3 helpers + the caller.
    std::vector<std::thread> m_bandThreads;
    std::mutex m_bandMutex;
    std::condition_variable m_bandStartCv;
    std::condition_variable m_bandDoneCv;
    const GSPrimitiveBatch *m_bandBatch = nullptr;
    int m_bandMinY = 0;
    int m_bandMaxY = 0;
    unsigned m_bandCount = 1;          // total participants (helpers + caller)
    // cont.318: atomics so the PS2X_GS_BAND_SPIN pre-wait can poll them lock-free; every WRITE still
    // happens under m_bandMutex so the condition-variable protocol is unchanged.
    std::atomic<uint64_t> m_bandSeq{0};      // bumped per fan-out; helpers wait for a new value
    std::atomic<unsigned> m_bandPending{0};  // helpers still working on the current fan-out
    std::atomic<bool> m_bandStop{false};
    unsigned long long m_bandFanOuts = 0, m_bandDirect = 0;
    // cont.209 run fan-out: the run is published once and every thread walks all of it.
    const GSPrimitiveBatch *m_bandRunItems = nullptr;
    size_t m_bandRunCount = 0;
    bool m_bandRunActive = false;   // coordinator is inside a run fan-out -> band 0 only, no nesting
    int m_bandRunY0 = 0, m_bandRunY1 = 0;
    unsigned long long m_bandRuns = 0, m_bandRunPrims = 0;
    // ★ cont.318: per-participant run accounting (index 0 = the coordinator). Workers write their
    // end time / busy ns under m_bandMutex before releasing m_bandPending; the coordinator reads them
    // after the barrier to attribute the tail (who finished last, by how much) and, with
    // PS2X_GS_BAND_ADAPT, to re-weight the stripe ownership so every participant finishes together.
    static constexpr unsigned kBandMaxParticipants = 8;
    static constexpr unsigned kBandUnitSpace = 2048; // >= kBandRowSpace >> stripe shift (shift >= 0)
    unsigned long long m_bandEndNs[kBandMaxParticipants] = {};
    unsigned long long m_bandRunBusyNs[kBandMaxParticipants] = {};
    double m_bandShare[kBandMaxParticipants] = {};      // adaptive share of stripe units per participant
    uint8_t m_bandOwnerUnits[kBandUnitSpace] = {};      // owner per stripe unit for the published run
    uint64_t m_bandOwnerSeq = 0;                        // bumped when m_bandOwnerUnits changes
    // ★ cont.318 PS2X_GS_BAND_DYN: dynamic row-group claiming inside a run. The coordinator publishes
    // the run's per-primitive conservative row extents (m_bandExt, pairs y0,y1; y0>y1 = nothing to
    // draw by helpers, e.g. points) and the group geometry; every participant claims the next group
    // from m_bandDynNext and rasterizes every primitive whose extent meets it, clipped to the group.
    std::vector<int32_t> m_bandExt;
    // cont.321b: per-primitive frame/Z page intervals [f0,f1,z0,z1] (gs2DrawCoversPage's superset), built with
    // the extents, so the dynamic claim loop can test a SKIPPED primitive's writes against this thread's cached
    // CLUT page / texture-level pages in four compares before calling noteDrawWrites.
    std::vector<uint32_t> m_bandPg;
    // cont.321b: the run's frame/Z page set (512 bits), published with the extents: a thread notes skipped
    // primitives' writes only while one of its cached levels / its CLUT page lies inside it.
    uint64_t m_bandTgtW[8] = {};
    std::atomic<unsigned> m_bandDynNext{0};
    unsigned m_bandDynGroups = 0;     // groups in the published run (0 = static striped mode)
    int m_bandDynBase = 0;            // first row of group 0 (runY0 aligned down to the group size)
    int m_bandDynRows = 0;            // rows per group for the published run
    // GPU present decode (cont.166 phase 0, PS2X_GS_GPU_PRESENT / PS2X_GS_GPU_VERIFY):
    // non-owning; Present() points its thread_local snapshot backend at the process-wide
    // device before PresentFromLocalMemory, with m_gpuSourceId keying the device's VRAM
    // upload cache (the flip/Sync capture tick; 0 = live snapshot, always re-upload).
    GsGpuPresentDevice *m_gpuDecode = nullptr;
    uint64_t m_gpuSourceId = 0;
    uint64_t m_glSnapSeq = 0; // cont.345: the flip/Sync capture seq this (present-side) instance renders; 0 = live
};

// Route a guest DISPFB1/2 flip to the active CPU backend (see GSCpuBackend::OnDisplayFlip).
// Pass the PRE-write dispfb1/2 values (the completed frame's registers). Defined in
// gs_cpu_backend.cpp; safe to call from any thread; no-op when no backend is live.
void ps2xGsNotifyDisplayFlip(uint64_t preFlipDispfb1, uint64_t preFlipDispfb2);
// ★ cont.345: the frontend's "preferred display source" (the full-screen copy sprite's SOURCE, which
// Present() shows instead of the display buffer -- gs_frontend.cpp displayCopy), published for the GL
// flip-point snapshot so that source is resolved at the flip too. Packed: bit 63 = valid,
// bits 0-15 dest fbp, 16-31 source fbp, 32-39 source fbw, 40-47 source psm. 0 = none.
extern std::atomic<uint64_t> g_ps2xGsPreferredDisplaySource;
// ★ cont.345: the LIVE DISPFB1/2 registers (every privileged write, gs_frontend.cpp), for the
// per-Sync capture's GL snapshot (that capture pairs with the live registers, not pre-flip ones).
extern std::atomic<uint64_t> g_ps2xGsLiveDispfb1, g_ps2xGsLiveDispfb2;
// cont.232: GSCpuBackend::PresentSourceFresh on the active backend (false + seq 0 when none is live).
bool ps2xGsPresentSourceFresh(uint64_t &outSeq);
// ★★★★ cont.332c PS2X_GS_PRESENT_HIRES: take the GL renderer's SCENE-RESOLUTION frame (display
// size x PS2X_GS_SCALE, tightly packed, rows top-down) so the presenter can show it instead of the
// display-sized resolve. False = the feature is off, the GL device is not up, or no new frame has
// landed since the last call. The 640-stride present path is untouched either way.
bool ps2xGsTakeHiresPresentFrame(std::vector<uint8_t> &out, uint32_t &w, uint32_t &h);
// ★★★★ cont.332d: the aspect ratio the presented frame should be DRAWN at, computed from the
// DISPLAY registers (see PresentFromLocalMemory). The framebuffer's own width:height is NOT it --
// the CRTC magnifies the source across a fixed raster, so a 512x511 display belongs in the same
// 4:3 rect a 640x512 one would. 0 = not known yet (draw it pixel-square).
float ps2xGsPresentAspect();
// cont.356: the ratio the frame is actually PRESENTED at -- PS2X_WINDOW_ASPECT parsed once
// (> 1.1 = that explicit ratio, e.g. 1.7778; unset/1 = derive from DISPLAY; 0 = pixel-square).
// The 2D/HUD counter-scale is naturalAspect / presentedAspect, so it must read the same value
// the window fit does rather than re-deriving one.
double ps2xWindowAspect();
// cont.356e: true when the frame being presented is a 2D screen whose art comes straight from
// the framebuffer (PS2X_GS_HUD_ASPECT_FULL). Such a frame is PILLARBOXED at the natural aspect
// rather than stretched, and its per-draw HUD counter-scale is suppressed.
bool ps2xGsIs2dScreen();
// ★★★★★ cont.356g: the GAME declares what kind of screen it is showing; the engine holds no
// per-game policy. Called from a game override (for LOTR: mods/widescreen, from the screen-request
// hook). PS2X_GS_HUD_ASPECT_FULL selects who decides:
//   unset / 0  -> OFF: nothing is corrected, whatever the game declares (the default on every title)
//   1          -> DECLARED: only what the game declares, and a frame that draws perspective
//                 geometry cancels it immediately -- a mis-declared screen self-corrects in a frame
//   auto       -> the frame-level HEURISTIC, a BRING-UP TOOL for a game with no table yet: it
//                 reports what it would classify, and disagrees loudly with any declaration
// ⚠ The heuristic's thresholds were measured on ONE game and do not transfer; that is exactly why
// it is not the default. See docs/llmps2recomp-patches.md row 198.
// kPs2xScreen2D is gated on evidence: a frame that draws perspective geometry cancels it, so a
// MIXED screen can be declared 2D safely and its 3D phases simply do not engage.
// ★ kPs2xScreen2DStrict skips that gate -- 'I have looked at this screen, it is flat, do not
// second-guess me'. It exists because a screen can render decorative elements through a
// perspective projection while still being an entirely flat 4:3 layout, which the gate cannot
// tell from a real 3D scene. Use it only on a screen someone has actually looked at: a strict
// entry on a real 3D screen WILL pillarbox that scene, with nothing left to catch it.
enum Ps2xScreenKind { kPs2xScreenUnknown = 0, kPs2xScreen3D = 1, kPs2xScreen2D = 2,
                      kPs2xScreen2DStrict = 3 };
void ps2xGsDeclareScreen(unsigned kind);

// ★★★★★ cont.358: the ANCHOR hook -- the companion to ps2xGsDeclareScreen, and its mirror image.
// That one is a game->engine PUSH ("the screen I am showing is flat"); this is an engine->game PULL
// ("I am about to counter-scale this HUD draw -- where do you want it anchored?").
//
// WHY IT EXISTS. Anamorphic widescreen renders a wider frustum into the game's own raster and
// stretches at presentation, so the 3D world is correct and 2D is not; PS2X_GS_HUD_ASPECT
// counter-scales the 2D draws to restore their authored proportions. That scale needs a FIXED
// POINT, and the engine's default is the screen CENTRE -- which keeps every element's shape right
// and needs no per-game knowledge, but pulls corner elements inward, so the HUD ends up
// pillarboxed inside a natural-aspect region while the world fills the wider frame. Anchoring an
// element to the edge it was authored against instead is a decision only the GAME can make, and
// only per element: the engine can see that a draw is 2D, never that it is "the bottom-left status
// cluster". So the engine keeps the classification and the arithmetic, and asks.
//
// ★ DECLINING IS THE POINT. A hook that returns negative for a draw gets the engine's default, so
// a game covers the elements it has actually measured and leaves every other draw EXACTLY as it
// was. A policy can therefore be curated one element at a time with no regression in between --
// unlike a table that has to be right before any of it is usable.
struct Ps2xHudDraw
{
    float x0, y0, x1, y1;      // primitive bbox, GS pixel space, XYOFFSET already subtracted --
                               // the same space the vertices are emitted in
    float refW;                // the vertex path's HORIZONTAL normalisation: it computes
                               // p.x = pos.x / refW, and the anchor you return is in that [0,1]
                               // space. ⓘ There is deliberately no refH: the anchor is horizontal
                               // (an anamorphic stretch is), and the vertical normalisation is the
                               // render target's height, which is not resolved until the batch is
                               // assembled -- supplying a guess here would be worse than omitting
                               // it. Use scissorH to reason about vertical placement.
    // The scissor rect this draw is clipped to -- ORIGIN AND EXTENT, deliberately both.
    // ⚠ The extents alone are not enough to tell two viewports apart: a left/right split would give
    // both halves the same scissorW and differ only in the origin, so a policy keyed on extent
    // would silently apply one viewport's anchors to the other. Carrying the origin makes the
    // struct sufficient whether a game is shared-screen or split-screen, which is why it is here
    // before any game needs it -- adding it later would be an ABI break for every consumer.
    float scissorX, scissorY;
    float scissorW, scissorH;
    uint32_t tbp0;             // TEX0 base, 0 when the draw is untextured (the bar fills are)
    uint32_t prim;             // GS primitive type
    unsigned screenKind;       // whatever the game last declared (Ps2xScreenKind)

    // ★★★★★ cont.358f: THE BLOCK THIS DRAW BELONGS TO, and the reason a policy can be correct.
    // A draw cannot know where its string ends, so ANY rule keyed on one draw's geometry cuts
    // strings at its own boundary -- measured, twice (cont.358b, cont.358d). The engine therefore
    // groups each flip's classified draws into connected blocks and hands every draw of a block
    // the SAME cluster rect, so a policy keyed on the cluster gives one answer per block and a
    // string cannot be split by construction rather than by argument.
    //
    // ⚠ clusterId == 0 means NO CLUSTER IS KNOWN for this draw (the first frame, or an element
    // that just appeared). The cluster fields are then meaningless and a policy MUST decline --
    // falling back to the draw's own rect would make a single glyph look like a tiny edge-hugging
    // element and anchor it alone, which is the exact defect this field exists to prevent.
    //
    // ⓘ The grouping comes from the PREVIOUS flip: clustering needs the whole frame, and a draw is
    // classified while the frame is still being built. That one-frame lag is the same trade
    // ps2xGsIs2dScreen()'s debounce already makes deliberately; a HUD moves slowly relative to it.
    unsigned clusterId;        // 0 = unknown (DECLINE); otherwise a per-flip block id
    unsigned clusterDraws;     // how many classified draws are in that block
    float clusterX0, clusterY0, clusterX1, clusterY1;   // the block's rect, same space as x0..y1
    // ★★★★★ cont.358j: the DISTINCT TEXTURE BASES in this block (up to 8; `clusterTbpCount` says
    // how many are valid, and 8 means "8 or more"). This is what lets a game IDENTIFY a block
    // rather than infer its role from geometry.
    //
    // WHY IT IS NEEDED. A block's own text carries the shared font atlas, so a draw's `tbp0` says
    // nothing about which UI element it belongs to -- the same atlas appears in the gameplay HUD
    // and in every menu. But a block that contains, say, the health-bar texture is unambiguously
    // the health cluster, and its glyphs are in that same block. So a game can name the blocks it
    // wants moved by the textures they contain, and leave every other block alone.
    //
    // ⚠ This exists because a purely geometric rule DOES NOT GENERALISE. An "is it off-centre?"
    // test separated cleanly on three measured scenes and then misfired on the main menu -- a
    // left-aligned list sits just far enough off-centre to look like an edge element. Geometry can
    // say WHICH EDGE a block belongs to; only the game can say whether it should move at all.
    unsigned clusterTbpCount;
    uint32_t clusterTbps[8];
    // ★ cont.358p: the display-flip counter this draw belongs to. A policy that needs to reason
    // about a WHOLE FRAME -- "where were the elements I recognise, this frame?" -- has no other way
    // to know where one frame ends, since it is called per draw. Comparing it against the last
    // value seen is a frame boundary.
    unsigned long long frameId;
};

// Return the horizontal anchor in [0,1] of refW: 0 = the left edge, 0.5 = the centre (the default),
// 1 = the right edge. ★ Return ANY NEGATIVE value to DECLINE and take the engine's default.
//
// ⚠⚠ CALLED FROM THE BATCH-BUILDING PATH, which is reached from the WORKER thread as well as the
// present thread -- cont.331n had to make a flag thread_local for exactly this reason. The hook
// MUST be pure and reentrant: no guest memory reads, no allocation, no locks, no logging. Latch
// anything you need from guest state on the EE thread beforehand (mods/widescreen already does
// this for the screen id) and only READ that latch here.
typedef float (*Ps2xHudAnchorFn)(const Ps2xHudDraw &);

// Install once, at startup. A raw pointer held in an atomic rather than a std::function: this is
// read from the draw path, and a std::function reassigned while another thread reads it is a race
// (gs_cpu_backend.h's WriteVramFunc carries the same reasoning for the per-pixel tables).
// Passing nullptr uninstalls. PS2X_GS_HUD_ANCHOR=0 ignores an installed hook = the A/B control.
void ps2xGsSetHudAnchorFn(Ps2xHudAnchorFn fn);
