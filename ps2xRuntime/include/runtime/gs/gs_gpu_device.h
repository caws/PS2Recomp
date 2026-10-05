#pragma once

#include "runtime/gs/gs_types.h"

#include <cstddef>   // offsetof -- the GsGlVertex layout assertions
#include <cstdint>
#include <memory>
#include <vector>

// GPU presentation-decode device (cont.166, phase 0 of the GPU-backend arc).
//
// Owns an EGL GL 4.3 core context (independent of raylib's GL 3.3 window context --
// no sharing needed: results come back via buffer readback) and a compute kernel that
// replicates GSCpuBackend::CopyFrameToHostRgbaCpu's local-memory display decode
// bit-exactly: the same GSMem page-lookup tables (built from the identical
// PixelStorageTraits templates), the same address masking, the same CT32/CT24/CT16/
// CT16S unpack rules. Guest VRAM (the 4MB present snapshot) is uploaded once per
// sourceId (the flip/Sync capture tick) and each display rect decodes on the GPU.
//
// Threading: ALL EGL/GL work runs on a dedicated device thread that exclusively owns
// the context (a thread has ONE current GL context regardless of binding API, and the
// present thread already holds raylib's GLX context -- eglMakeCurrent there fails with
// EGL_BAD_ACCESS). Jobs flow through a bounded FIFO: DecodeFrameRect is synchronous
// (the caller waits for its job), the phase-1 VRAM-mirror uploads/verifies are
// fire-and-forget. Callable from any thread. Any init failure disables the device for
// the run (logged).
//
// Phase 1 (cont.167, VRAM residency): a persistent 4MB guest-VRAM MIRROR SSBO fed by
// host->local upload chunks (CT32 -- empirically 99.9997% of this game's 1.76M
// transfers; other formats are counted + skipped, leaving the mirror stale there until
// their kernels land). MirrorVerify compares the mirror against the CPU-authoritative
// post-upload bytes ON THE GPU (atomic mismatch counter, 16-byte readback) for sampled
// chunks -- the same shadow-verify gate the phase-0 present decode passed.
//
// Phase 1b (cont.168, mirror COMPLETENESS): MirrorPatchRaw copies an authoritative CPU
// VRAM byte range straight into the mirror at the same offset -- bit-exact by
// construction, with no swizzle kernel to get wrong -- which is how every remaining
// VRAM-mutating op the swizzle kernel does not cover gets mirrored (non-CT32 uploads
// such as the T4HL font atlas, local->local blits, framebuffer clears, direct
// WriteVram). MirrorVerifyFull then compares the WHOLE 4MB mirror against CPU VRAM and
// reports which 8KB pages diverge -- the completeness gate, and the instrument that
// names any still-unmirrored writer. Once phase 2's rasterizer writes into the mirror,
// the same full verify becomes its per-flush shadow gate.

// One host->local upload chunk, self-contained (the CPU transfer cursor state is baked
// into startPixel; geometry from BITBLTBUF/TRXPOS/TRXREG). For MirrorUpload, payload =
// the chunk's CT32 words; for MirrorVerify, payload = the CPU-read post-upload values.
struct GsGpuUploadChunk
{
    uint32_t dbp = 0;  // destination base, BLOCKS
    uint32_t dbw = 0;  // destination buffer width, 64-pixel units (>=1)
    uint32_t dsax = 0; // destination rect origin
    uint32_t dsay = 0;
    uint32_t rrw = 0;  // rect width (pixel cursor wraps at it)
    uint32_t startPixel = 0;
    std::vector<uint32_t> payload; // one word per pixel
};
// The rasterizer's inputs, split the way the hardware batch wants them (cont.171).
//
// A primitive record carries ONLY geometry plus an index into a per-batch STATE TABLE.
// That split is measured, not assumed: PS2X_GS_BATCHCENSUS found a mean of 3.9 distinct
// draw states per render-target run (max 61), so a 128-entry table covers every batch this
// game produces -- while GSPrimitiveBatch hands us ~200 bytes of GSDrawState per PRIMITIVE
// and there are 7.2M of them per run.
//
// All members are 4-byte scalars in a fixed order, mirrored field-for-field by the shader's
// std430 blocks. The HOST resolves everything that would otherwise need a GPU division or
// transcendental -- the per-triangle winding and 1/|denom|, and the mip level's tbp/tbw/
// texture size/region bounds -- because GLSL only guarantees ~2.5 ULP for division while
// the CPU oracle divides with correctly-rounded IEEE.
struct GsGpuState
{
    // Frame / Z / scissor.
    uint32_t fbp = 0, fbw = 1, fpsm = 0, fbmsk = 0;
    uint32_t zbp = 0, zpsm = 0, zmsk = 0;
    uint32_t scissorX0 = 0, scissorY0 = 0, scissorX1 = 0, scissorY1 = 0;

    // PRIM flags.
    uint32_t primType = 0, iip = 0, tme = 0, fge = 0, fst = 0, abe = 0;

    // TEST / ALPHA / FBA / fog.
    uint32_t ate = 0, atst = 0, aref = 0, afail = 0, zte = 0, ztst = 0;
    uint32_t blendA = 0, blendB = 0, blendC = 0, blendD = 0, blendFix = 0;
    uint32_t fba = 0;
    uint32_t fogR = 0, fogG = 0, fogB = 0;

    // TEX0 / TEXA / CLAMP (mip already folded into tbp/tbw/texW/texH/min*/max*).
    uint32_t tbp = 0, tbw = 1, tpsm = 0, tcc = 0, tfx = 0;
    uint32_t cbp = 0, cpsm = 0, csm = 0, csa = 0;
    uint32_t texW = 1, texH = 1, linear = 0;
    uint32_t wms = 0, wmt = 0, minU = 0, maxU = 0, minV = 0, maxV = 0;
    uint32_t ta0 = 0, ta1 = 0, aem = 0;
    float fstDiv = 16.0f; // 16 << mip, the FST texel divisor

    // CLUT shadow (PS2X_GS_CLUTSHADOW): the uploaded 16-entry payload for this cbp, so the
    // shader can reproduce LookupCLUT's RGB-zeroed-entry substitution exactly.
    uint32_t clutShadowHave = 0;
    // ★ cont.250 PS2X_GS_LINTEX: byte-offset (in texels) of this state's host-decoded texture inside
    // the linTexels SSBO, or kLinTexNone when it was not decoded and the shader must walk the swizzle
    // + CLUT path as before. Repurposes what was pad0, so the std430 layout is unchanged.
    uint32_t linTexOff = 0xFFFFFFFFu;
    uint32_t pad1 = 0, pad2 = 0;
    uint32_t clutShadow[16] = {};
};

// One primitive: geometry, its scissor-clipped bbox (also the binning key), and its state.
struct GsGpuGeom
{
    // Vertices in screen space, XYOFFSET already subtracted (as DrawTriangle does).
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    // Vertex Z as the two halves of the CPU's double (packDouble2x32), so the triangle
    // path's double interpolation is exact.
    uint32_t z0lo = 0, z0hi = 0, z1lo = 0, z1hi = 0, z2lo = 0, z2hi = 0;
    uint32_t rgba0 = 0, rgba1 = 0, rgba2 = 0; // r | g<<8 | b<<16 | a<<24
    float q0 = 0, q1 = 0, q2 = 0;
    float s0 = 0, s1 = 0, s2 = 0;
    float t0 = 0, t1 = 0, t2 = 0;
    uint32_t uv0 = 0, uv1 = 0, uv2 = 0; // u | v<<16
    uint32_t fog0 = 0, fog1 = 0, fog2 = 0;

    float winding = 1.0f;
    float invAbsDenom = 0.0f;
    uint32_t minX = 0, minY = 0, maxX = 0, maxY = 0;

    // SPRITE path (DrawSprite is not a degenerate triangle): colour, Z and fog come FLAT
    // from v1 (it ignores IIP), Z is a plain truncation of v1.z with none of the triangle
    // path's +0.5, and UV is a screen-space lerp between two corners already in TEXEL units.
    uint32_t spriteZ = 0;
    uint32_t unclipX0 = 0, unclipY0 = 0;
    float spriteW = 1.0f, spriteH = 1.0f;
    float su0 = 0, sv0 = 0, su1 = 0, sv1 = 0;

    uint32_t stateIndex = 0;
    uint32_t pad0 = 0, pad1 = 0, pad2 = 0;
};

// A batch of primitives sharing one render target, with no non-draw VRAM mutation between
// them (see g_vramMutationSeq in gs_cpu_backend.cpp -- an upload landing mid-batch would be
// visible to every primitive in it, while the oracle shows it only to those that follow).
//
// Binning is done on the HOST for the tile kernel: it costs almost nothing (mean 491 prims x
// 4.21 tiles per batch) and it is trivially in SUBMISSION ORDER, which a GPU atomic append
// would not be -- and submission order per pixel is the whole correctness requirement.
// cont.233: when PS2X_GS_HWRASTER=1 asks for the hardware-raster path the host leaves the
// bins EMPTY (the GPU's own rasterizer does the walking, one triangle per primitive, and an
// ordered fragment-shader interlock supplies the per-pixel order); the device bins the batch
// itself if it has to fall back to the tile kernel.
// ★ cont.329 phase 2a (the GL renderer arc): the geometry stream handed to the GL renderer.
// Deliberately NOT GSPrimitiveBatch (312 bytes/prim, with a full context snapshot each): the GPU
// wants a flat vertex array plus a small number of state groups, which is also the shape one
// glDrawArrays per group needs.
struct GsGlVertex
{
    float x = 0.f, y = 0.f;  // GS pixel space, XYOFFSET already subtracted
    float z = 0.f;           // raw Z scaled to [0,1] by the Z format's max
    // ★ cont.329 phase 3: texture coordinates as (s, t, q), NORMALISED so s/q is in [0,1].
    // The GS interpolates s, t and q LINEARLY IN SCREEN SPACE and divides per pixel -- which is
    // `noperspective` interpolation plus an explicit divide in the shader, NOT GL's usual
    // perspective-correct path. FST (direct UV) draws arrive here as s = u/texW, t = v/texH, q = 1,
    // so both texture-coordinate modes take the same shader path.
    float s = 0.f, t = 0.f, q = 1.f;
    uint8_t r = 0, g = 0, b = 0, a = 0; // PS2 units -- a is 0..0x80 = 0..1.0; the x255/128
                                        // rescale for GL blending happens in the shader
    // ★★ cont.329j: the per-vertex FOG value (0..255). The GS interpolates it screen-linearly and
    // the pixel write does C = (F*C + (255-F)*FOGCOL) >> 8 -- the CPU rasterizer has always done
    // this (writePixelT's kFge arm); the GL renderer did not implement fog at all.
    uint8_t f = 0;
    // ★★★★★ cont.356: 1 = this vertex belongs to a draw the 2D/HUD ASPECT CORRECTION applies to
    // (anamorphic widescreen counter-scale). Classified PER PRIMITIVE in the backend by
    // `q == 1 && ZTST == ALWAYS && x-extent < frac * scissor` -- see the comment at that site for
    // why PRIM.FST is not the test. It rides in the existing padding, so the vertex does not grow.
    uint8_t hud2d = 0;
    // ★★★★★ cont.358: WHERE that counter-scale is anchored, as an unsigned 16-bit fixed point of
    // the normalised screen x: anchor = (hudAnchor - 1) / 65534, so 0 = the left edge, 65535 = the
    // right edge, and **32768 is EXACTLY 0.5** (65534 = 2 x 32767, so the quotient is exact in
    // IEEE). 32768 is therefore the engine's own default and reproduces the pre-cont.358 behaviour
    // bit-for-bit -- the value a game's anchor hook never having been installed, or having
    // declined this draw, leaves in place. It rides in the padding hud2d already sat beside, so
    // the vertex STILL does not grow (32 bytes).
    uint16_t hudAnchor = 32768;
};
// ★ The "it still does not grow" claim above, asserted rather than trusted: this is the per-vertex
// buffer uploaded on every batch, so a silent 4-byte growth would be a real cost nobody noticed.
static_assert(sizeof(GsGlVertex) == 32, "GsGlVertex must stay 32 bytes -- it is the per-batch upload");
static_assert(offsetof(GsGlVertex, hudAnchor) == 30, "hudAnchor must sit in the old padding");

// One decoded texture handed to the device. Shipped only when the key is new or its contents
// changed; the device caches by key.
struct GsGlTexUpload
{
    uint64_t key = 0;
    uint32_t w = 0, h = 0;
    // ★★★ cont.329j: which MIP LEVEL these pixels are. 74.4% of this game's textured draws carry
    // MMIN=5 (trilinear) with MXL=3 -- a four-level chain the GL renderer was ignoring entirely,
    // sampling level 0 for every draw while the CPU rasterizer sampled the level the GS selects.
    uint8_t level = 0;
    std::vector<uint8_t> rgba; // w*h*4, already CLUT- and TEXA-resolved
};

// A run of consecutive vertices that share every piece of GL state, so it is one draw call.
struct GsGlGroup
{
    uint32_t first = 0, count = 0; // range into GsGlBatch::verts
    // ⚠ GS scissor bounds are INCLUSIVE on both ends; glScissor takes width/height.
    uint16_t sx0 = 0, sy0 = 0, sx1 = 0, sy1 = 0;
    uint8_t ztst = 1; // 0 NEVER, 1 ALWAYS, 2 GEQUAL, 3 GREATER (PS2: bigger Z is nearer)
    uint8_t zmsk = 0; // 1 = depth writes disabled
    uint8_t zte = 1;  // 0 = depth test off AND Z not written (PCSX2's rule)
    // ★ cont.329 phase 2b: blending. The PS2 computes ((A - B) * C) >> 7 + D, where A/B/D each
    // select Cs / Cd / 0 and C selects As / Ad / FIX. Only four combinations are live in this game
    // (cont.329 census: 0101 = 94.3%, 0221 = 3.7%, 0121 = 1.0%, 0201 = 1.0%) and all map to GL
    // fixed-function factors. The device does the mapping so the selectors travel raw.
    uint8_t abe = 0;                        // PRIM.ABE -- 0 disables blending entirely
    uint8_t ba = 0, bb = 0, bc = 0, bd = 0; // ALPHA.A/B/C/D selectors (0 = Cs, 1 = Cd, 2 = 0/FIX)
    uint8_t bfix = 0x80;                    // ALPHA.FIX, in PS2 units where 0x80 == 1.0
    // ★★★★ cont.343b: COLCLAMP.CLAMP. 0 = the blend result WRAPS mod 256 (the GS), which GL
    // fixed-function blending cannot do: the device gives such an additive draw's target an HDR
    // (RGBA16F) colour buffer instead, so the +1/-1 shadow-volume count survives (see gs_gpu_device.cpp).
    uint8_t colclamp = 1;
    // ★ cont.329 phase 3: texturing. key 0 = untextured.
    uint64_t texKey = 0;
    // rotk row 257: the guest texture size (2^TW x 2^TH) this draw's normalised s,t are relative to -- needed to
    // address a PROMOTED target, whose GL texture is sized by the target's own extent, not by TW/TH.
    uint16_t texW = 0, texH = 0;
    uint8_t wrapU = 0, wrapV = 0; // 0 REPEAT, 1 CLAMP (95.4% / 0.8% here; REGION modes are rare)
    uint8_t tfx = 0;              // 0 MODULATE (96.1%), 1 DECAL, 2 HIGHLIGHT, 3 HIGHLIGHT2
    uint8_t tcc = 1;              // 1 = use the texture's alpha, 0 = vertex alpha only
    uint8_t lin = 1;              // bilinear (96.1%) vs nearest
    // ★★★ cont.329j: TEX1.MMIN verbatim (0 NEAREST, 1 LINEAR, 2-5 the mipmap modes) and the
    // chain's top level (TEX1.MXL), so the device can ask GL for the same filter the GS uses.
    uint8_t minFilter = 1;
    uint8_t maxLevel = 0;
    // ★★★ cont.329j: the GS computes its OWN level of detail -- LOD = (log2(1/|Q|) << L) + K,
    // or a FIXED K when LCM=1 -- which is not GL's max-derivative rule. Letting GL choose blurred
    // a grazing ground plane three times smoother than the reference (adjacent-pixel roughness
    // 2.10 vs 6.63), so the level is computed in the shader and asked for with textureLod.
    uint8_t lcm = 0;   // TEX1.LCM: 1 = fixed LOD (= K)
    uint8_t lodL = 0;  // TEX1.L
    int16_t lodK = 0;  // TEX1.K, 12-bit signed in 1/16 units
    // ★★ cont.329h FRAME.FBMSK -- which framebuffer BITS this draw must not write (1 = keep).
    // It was not carried at all: the device wrote glColorMask(1,1,1,0) for every draw on the
    // strength of a census that said FBMSK is 'A' (0xff000000) on 100% of draws. The per-draw
    // trace found full-screen post-process quads that invert and then whiten the whole target,
    // which is what an ignored RGB mask looks like.
    uint32_t fbmsk = 0;
    // ★★★ cont.329i: HOW A PROMOTED RENDER TARGET IS READ. The promotion binds the target's GL
    // colour texture, but the guest asked for a specific PIXEL FORMAT, and the two composites in
    // this game do NOT ask for RGBA: one reads the target as PSMCT24 (24-bit colour -- its alpha
    // comes from TEXA, not from the buffer) and one as PSMT4HL (a CHANNEL SHUFFLE: the texel is a
    // 4-bit index taken from the alpha byte, looked up through the CLUT).
    //   0 = sample as-is   1 = CT24: alpha := TEXA   2 = paletted read, neutral (diagnostic)
    uint8_t srcMode = 0;
    uint8_t srcTexa = 0x80; // TEXA.TA0
    uint8_t srcAem = 0;    // TEXA.AEM: RGB==0 expands alpha to 0 (the CPU sampler's rule)
    // ★★ cont.329j PRIM.FGE and the FOGCOL register, per draw.
    uint8_t fge = 0;
    uint8_t fogR = 0, fogG = 0, fogB = 0;
    // ★ cont.329 phase 3d: the alpha test. 56.4% of this game's draws are
    // ATE=1 ATST=GEQUAL AFAIL=FB_ONLY ZMSK=0 -- alpha-test failures still write COLOUR but not
    // DEPTH. Without it, failing texels draw solid AND occlude what is behind them, which removes
    // objects from the scene rather than merely miscolouring them.
    uint8_t ate = 0;   // 0 = no alpha test
    uint8_t atst = 1;  // 0 NEVER 1 ALWAYS 2 LESS 3 LEQUAL 4 EQUAL 5 GEQUAL 6 GREATER 7 NOTEQUAL
    uint8_t aref = 0;  // compared against the fragment's alpha in PS2 units (0..255)
    uint8_t afail = 0; // 0 KEEP, 1 FB_ONLY, 2 ZB_ONLY, 3 RGB_ONLY
    // ★ cont.329 phase 4b: this draw samples a buffer the game RENDERED INTO, so `texKey` names a
    // render target rather than a decoded texture. Our GL path no longer writes video memory, so
    // decoding that address from memory yields stale nothing -- which is why the composited
    // background read as a blank field while ordinary textured objects drew correctly.
    uint8_t texIsTarget = 0;
};

// ★★★ cont.330 TARGET-FROM-VRAM SEEDING. A GL render target held only what GL had DRAWN into it:
// nothing ever copied guest VRAM back in, so when the game UPLOADED an image into pages a target
// occupies, the GL copy never saw it. That is the FMV / pre-rendered background 4x-tiling bug --
// the movie picture is uploaded into pages 0x180+, the composite samples that address, GL promotes
// it to the (empty) target and the read contributes nothing. PCSX2 solves the same case with
// GSTextureCache::InvalidateVideoMem plus per-Target DIRTY RECTS flushed into the target before it
// is used (pcsx2/GS/Renderers/HW/GSTextureCache.cpp); this is that flush. One entry per dirty PAGE
// rect, in the target's own guest pixel space (y measured from the TOP, as the GS does).
struct GsGlTargetSeed
{
    uint32_t x = 0, y = 0, w = 0, h = 0;
    std::vector<uint8_t> rgba; // w*h*4, decoded from VRAM in the TARGET's own psm/fbw
};

struct GsGlBatch
{
    // ★★ cont.329h: trace this batch draw call by draw call -- after each group the device reads
    // the target back and reports the group's full state next to the picture it produced. Set by
    // the backend for a primitive window (PS2X_GS_DRAWCMP + _TRACE), so the cost is one window.
    bool trace = false;
    std::vector<GsGlVertex> verts;
    std::vector<GsGlGroup> groups;
    std::vector<GsGlTexUpload> texUploads; // textures this batch introduced or refreshed
    // ★★★ cont.330: VRAM rects to write INTO the target before any of this batch's draws (see
    // GsGlTargetSeed). The backend flushes the pending batch before it emits a seed, so a seed can
    // never overtake draws that were already queued for the same target.
    std::vector<GsGlTargetSeed> targetSeeds;
    // The GS pixel-space extent the vertices are expressed in -- the transform divides by this.
    uint32_t refW = 0, refH = 0;
    // ★ cont.329 phase 4: WHICH render target these draws belong to. The game double-buffers, so
    // the same scene is drawn alternately to two framebuffer addresses. Rendering both into ONE GL
    // target composited two frames together, and translucent layers then compound -- a 16%-strength
    // fog quad applied repeatedly approaches white, which is the washed-out picture. Measured:
    // blending was correct on 100% of vertices, so the fault was never the blend, only how often
    // it was applied. One GL target per framebuffer address fixes it at the source.
    uint64_t targetKey = 0;
    // ★★★★ cont.344: the draw context's Z buffer (ZBUF.ZBP pages, ZBUF.PSM). The target key is the
    // frame address alone (one GL target per framebuffer), so the depth texture is pooled by THIS.
    uint32_t zbp = 0, zpsm = 0;
};

struct GsGpuBatch
{
    // ★ cont.250: host guest-VRAM for PS2X_GS_LINTEX's texture decode. The device otherwise only has
    // the GPU-side mirror, which would have to be read back to decode from -- far too slow.
    const uint8_t *vram = nullptr;
    uint32_t vramSize = 0;
    std::vector<GsGpuState> states;   // <=128, dedup'd
    std::vector<GsGpuGeom> prims;     // in submission order
    // CSR bins over a tilesX x tilesY grid of 16x16 screen tiles: tile t owns
    // binEntries[binOffsets[t] .. binOffsets[t+1]), each entry a prims[] index, ascending.
    std::vector<uint32_t> binOffsets;
    std::vector<uint32_t> binEntries;
    std::vector<uint32_t> tileList;   // non-empty tiles, one workgroup each
    uint32_t tilesX = 0, tilesY = 0;
};

class GsGpuPresentDevice
{
public:
    GsGpuPresentDevice();
    ~GsGpuPresentDevice();

    GsGpuPresentDevice(const GsGpuPresentDevice &) = delete;
    GsGpuPresentDevice &operator=(const GsGpuPresentDevice &) = delete;

    // True once the EGL context + kernel are live (attempts init on first call).
    bool EnsureInitialized();

    // GPU replica of the useLocalMemoryLayout=true path of CopyFrameToHostRgba:
    // decode a width x height rect of the frame (origin applied in source space)
    // into kHostFrameWidth-stride RGBA bytes. sourceId keys the VRAM upload cache
    // (0 = always re-upload, e.g. a live no-flip snapshot). Returns false -- with
    // outPixels untouched -- when the device is unavailable, the psm is unsupported
    // (only CT32/CT24/CT16/CT16S display formats), or vram is not the full 4MB;
    // the caller then runs the CPU decode.
    bool DecodeFrameRect(const uint8_t *vram,
                         uint32_t vramSize,
                         uint64_t sourceId,
                         const GSFrameReg &frame,
                         uint32_t width,
                         uint32_t height,
                         bool frameBaseIsPages,
                         uint32_t originX,
                         uint32_t originY,
                         bool preserveAlpha,
                         std::vector<uint8_t> &outPixels);

    // Phase-1 VRAM mirror (both async fire-and-forget; bounded queue backpressures the
    // producer). MirrorUpload swizzle-writes the chunk into the mirror SSBO; MirrorVerify
    // compares the mirror at the chunk's addresses against payload (the CPU-read
    // post-upload values), tallying into [gsgpu:xverify] counters printed periodically.
    // ★ cont.329 phase 1 (the GL renderer arc, PS2X_GS_RENDERER=gl): render one frame into a real
    // GL colour+depth target at `PS2X_GS_SCALE` x the display size, resolve it DOWN to display
    // size on the GPU, and read that back as top-down RGBA8. Resolving before the readback is what
    // keeps readback cost flat in the upscale factor, and returning the presenter's existing pixel
    // shape is what lets the whole present path stay unchanged.
    // ★ `outPixels` is the presenter's FIXED 640x512 RGBA8 buffer with a 640-pixel row stride and
    // the live image in the top-left sub-rect -- the same contract CopyFrameToHostRgbaCpu obeys.
    // It is NOT a tightly-packed display-sized image; returning one makes every consumer read
    // 640-stride rows out of a smaller allocation.
    // PHASE 1 only CLEARS the target -- nothing is drawn into it yet.
    // Synchronous (the presenter needs the pixels now). False = unavailable; caller falls back.
    // `targetKey` selects WHICH target to resolve -- the one the DISPLAY registers point at, now
    // that there is more than one.
    // ★★★★ cont.332c PS2X_GS_PRESENT_HIRES: take the latched SCENE-RESOLUTION frame (dispW x
    // dispH times PS2X_GS_SCALE), rows already top-down, tightly packed. False = nothing new.
    bool TakeHiresPresentFrame(std::vector<uint8_t> &out, uint32_t &w, uint32_t &h);
    // ★★★★ cont.345: `snapshotSeq` (0 = none) names the flip snapshot this present renders; when a
    // SnapshotFrameGl() resolve exists for that (seq, target), its pixels are returned instead of
    // resolving live -- the live resolve races the guest's next frame (see SnapshotFrameGl).
    bool RenderFrameGl(uint64_t targetKey, uint32_t displayWidth, uint32_t displayHeight,
                       std::vector<uint8_t> &outPixels, uint64_t snapshotSeq = 0ull);
    // ★★★★ cont.345: resolve `targetKey` NOW, in FIFO order behind every draw queued so far, and
    // keep the pixels under `seq` for the presenter. Asynchronous. Called by the GS worker at the
    // display-flip snapshot point -- the same instant the CPU arm copies VRAM -- so the frame the
    // presenter shows is what the buffer held AT THE FLIP, not whatever the guest had drawn into
    // it by the time the presenter thread got round to asking.
    void SnapshotFrameGl(uint64_t targetKey, uint64_t seq);

    // ★ cont.329 phase 2a: queue geometry into the GL target. Asynchronous, like RasterBatch --
    // the draws accumulate and RenderFrameGl() resolves what has landed. Phase 2a is UNTEXTURED
    // and UNBLENDED: it proves the vertex path, the GS-pixel-space transform and the depth test
    // before any blend/alpha-test state is translated.
    void DrawBatchGl(GsGlBatch &&batch);

    void MirrorUpload(GsGpuUploadChunk &&chunk);
    void MirrorVerify(GsGpuUploadChunk &&chunk);

    // Phase-1b raw range patch (async, same FIFO + backpressure): overwrite [byteOffset,
    // byteOffset + bytes.size()) of the mirror with authoritative CPU VRAM bytes. The
    // caller snapshots those bytes AFTER its op completed, while the GS worker is the
    // only writer, so FIFO order preserves op order. Out-of-range/empty patches drop.
    // Correct for ANY write pattern -- a conservative superset range is fine, it just
    // copies more authoritative bytes.
    void MirrorPatchRaw(uint32_t byteOffset, std::vector<uint8_t> &&bytes);

    // Phase-1b full-mirror shadow verify (SYNCHRONOUS: queued behind every pending mirror
    // job and the caller waits, so it compares the mirror with all prior posts applied).
    // vram must be the full 4MB and stable across the call -- invoke it from a drained,
    // locked backend state. Reports total divergent words, the first divergent offset,
    // and a per-8KB-page histogram ([gsgpu:mverify]) naming WHICH pages are stale.
    // drawPageMask (optional, 512 bytes -- one per 8KB GS page, nonzero = the CPU
    // rasterizer may have written there since the last call): pages it marks are reported
    // separately, because until phase 2 mirrors draws they are EXPECTED to diverge. What
    // remains -- stale pages the rasterizer cannot explain -- is the actual gate.
    // Returns true when the mirror matches vram exactly.
    // `quiet` suppresses the report line -- the bisect runs ~12 of these per failure and
    // only cares about the boolean. `outFirstWord`/`outStaleWords`, when given, still return
    // WHERE it failed: the bisect must not assume the divergence lies inside the culprit's
    // rect (cont.173 -- that assumption is exactly what produced a contradiction).
    bool MirrorVerifyFull(const uint8_t *vram, uint32_t vramSize,
                          const uint8_t *drawPageMask = nullptr, bool quiet = false,
                          uint32_t *outFirstWord = nullptr, uint32_t *outStaleWords = nullptr);

    // Phase 2a (cont.169): rasterize ONE primitive into the mirror. Async, FIFO-ordered
    // behind every prior mirror op, so the caller's seed patches are already applied.
    // Single-primitive dispatch is race-free by construction: one invocation per pixel of
    // the scissor-clipped bbox, and a pixel owns its frame word and its Z word.
    // A batch is rasterized one workgroup per non-empty 16x16 tile, one thread per pixel,
    // each thread walking its tile's primitive list IN SUBMISSION ORDER. That needs no
    // barriers at all: pixels are independent of one another, and the only ordering the GS
    // requires is per pixel. The phase-2a single-primitive path is just a batch of one.
    void RasterBatch(GsGpuBatch &&batch);

    // Phase 2a shadow verify: compare mirror[byteOffset .. +expected.size()) against the
    // CPU-authoritative post-draw bytes. `tag` is echoed on divergence so a mismatch names
    // the draw-state combo that produced it. Async; ordered behind the raster.
    void MirrorVerifyRange(uint32_t byteOffset, std::vector<uint8_t> &&expected, uint64_t tag);

    // Read the mirror back (synchronous). For the bisect's final step: with the culprit
    // primitive isolated, the only way to say WHY it differs is to compare the actual
    // pixels, which needs the GPU's bytes on the host.
    void MirrorReadback(uint32_t byteOffset, uint32_t byteLen, std::vector<uint8_t> &out);

    // Perf mode (cont.174): time each raster dispatch with a GL timer query so the GPU's
    // own execution time can be compared against the CPU rasterizer's, WITHOUT flipping
    // authority. EnablePerfTiming must be called before any raster job.
    void EnablePerfTiming();
    void PerfStats(unsigned long long &gpuNs, unsigned long long &prims,
                   unsigned long &dispatches) const;

    // cont.233: true when PS2X_GS_HWRASTER asked for the hardware-raster path and it has not
    // failed to initialise. The batch accumulator skips host binning then (the device bins
    // itself if it ever has to fall back to the tile kernel).
    bool HardwareRasterRequested() const;

    // Phase-2a tallies, printed by the caller alongside the other mirror counters.
    void RasterStats(unsigned long &prims, unsigned long &verifies,
                     unsigned long &mismatchRanges, unsigned long long &mismatchWords) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// ★ cont.331q: the GUEST DISPLAY FLIP counter, readable from the device thread. The present
// captures are flip-indexed (PS2X_GS_PRESENT_SAVE_BYFLIP) because presentation is host-paced and
// only the flip counter is a guest-driven instant, so an era named from a capture ("the regression
// at flip 900") can only be aimed at on the device side through the same counter -- glrFrames is a
// device-side resolve count and its ratio to flips is not fixed. Defined in gs_cpu_backend.cpp,
// where the counter lives.
unsigned long long gs2CurrentFlip();
