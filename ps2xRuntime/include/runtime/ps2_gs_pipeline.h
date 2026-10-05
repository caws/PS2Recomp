// cont.317 — the GS pipeline ring (docs/vu1-thread-design.md, stages 1 and 2).
// Internal header (runtime .cpp files only). Every EE-side input to the VIF1 / VU1 / GIF pipeline
// -- a VIF1 data chunk, an external path-3 packet, the end-of-pass drain, a VIF1 register-block
// write, a guest store into VU1 memory, the display flip -- becomes a copied command in a FIFO
// executed by re-entering the runtime's own public entry points with the consumer flag set, so
// the existing bodies run unchanged and in program order.
//   PS2X_GS_PIPELINE=0  the direct path (no ring)
//   PS2X_GS_PIPELINE=1  the ring, consumed synchronously on the EE thread (stage 1)
//   PS2X_GS_PIPELINE=2  the ring, consumed by the pipeline thread (stage 2); the EE waits only at
//                       fence() -- guest reads of VU1 memory, the GS syncs that need quiescent
//                       VRAM, shutdown -- and on a full ring (back-pressure). DEFAULT since cont.319
//                       (build 752): it leads mode 1 at matched prims/frame once the raster got cheaper.
//   PS2X_PIPE_CPUS      cont.322: the consumer thread's CPU list -- `gs` (default: the raster pool's
//                       list; the thread used to INHERIT the EE's one-CPU mask and time-slice with it),
//                       `ee` (the old behaviour, A/B), `all` (no pin), or an explicit list.
//   PS2X_PIPE_POOL      cont.322: payload buffers recycled between commands (default 1; `=0` = a fresh
//                       std::vector per command -- a 5.5 MB malloc + page-faulted copy per VIF1 transfer);
//                       PS2X_PIPE_POOL_MB (default 64) bounds the pool.
#pragma once
#include <cstdint>
#include <vector>

class PS2Memory;
class GSCpuBackend;
class GS;

namespace ps2gs
{
    int mode();               // 0 / 1 / 2 as above
    bool threaded();          // mode() == 2
    bool producerSide();      // pipeline enabled AND the caller is not inside the consumer
    bool consumerSide();      // the caller is the consumer (inside pump() or on the pipeline thread)
    void enqueueVif1(PS2Memory *mem, const uint8_t *data, uint32_t size);
    // ★ rotk row 254: the VIF1 DMA chain buffer handed over WITHOUT a copy (it is already a private copy of
    // guest memory), and a recycled buffer to build the next chain in (capacity kept: no reallocation).
    void enqueueVif1Owned(PS2Memory *mem, std::vector<uint8_t> &&data);
    std::vector<uint8_t> takeBuffer();
    void enqueueGif3(PS2Memory *mem, const uint8_t *data, uint32_t size, bool drainImmediately);
    void enqueueDrain(PS2Memory *mem);
    void enqueueVifReg(PS2Memory *mem, uint32_t addr, uint32_t value);
    void enqueueVu1Mem(PS2Memory *mem, bool code, uint32_t offset, const void *data, uint32_t size);
    void enqueueFlip(GSCpuBackend *backend, uint64_t preFlipDispfb1, uint64_t preFlipDispfb2);
    // ★ cont.318: GS drawing-context writes that originate OUTSIDE the packet stream (the kernel HLE's
    // sceGsSwapDBuff / sceGsPutDrawEnv draw environments and clear packets, and its HLE framebuffer
    // clear). In mode 2 they must ride the ring, or they land in the GS context while the consumer is
    // still drawing the previous frame (measured: ~60% of the raster on the scalar fallback, the frame
    // target/format seen by the draws being the NEXT buffer's).
    void enqueueGsReg(GS *gs, uint8_t addr, uint64_t value);
    void enqueueGsClear(GS *gs, uint32_t contextIndex, uint32_t rgba);
    // ★ rotk 2026-10-03 (row 253) PS2X_GS_SPLIT=1 (default OFF): stage 3, the GS parse on its own thread
    // ("GsParse"). The pipeline thread keeps VIF1 + VU1 + the GIF arbiter; what the arbiter hands the GS
    // (each packet with its path, the end-of-drain) and the ring's GS commands (GsReg / GsClear / Flip)
    // go, in program order, to a second FIFO the GsParse thread executes.
    void setParseGs(GS *gs);  // the GS the parse thread feeds (PS2Runtime init)
    bool splitSink();         // true on the pipeline thread when the split is on: the arbiter hands over
    void splitPacket(uint32_t path, std::vector<uint8_t> &&data);
    void splitEnd();          // the arbiter's end-of-drain (GS::endPacket)
    void pump();              // stage 1: execute everything queued now; stage 2: wake the consumer
    void fence(int site = 0); // wait until everything queued has been executed (site: 0 other, 1 VU1 read, 2 GS reset, 3 GS readback, 4 local-to-host)
    void shutdown();          // fence, then stop and join the consumer thread
}
