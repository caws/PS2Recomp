// cont.317 — the GS-pipeline capture/replay oracle (docs/vu1-thread-design.md, stage 0).
// Internal header: included by runtime .cpp files only (never by ps2_runtime.h, so no generated
// code rebuild). Records, from a quiescent DMA-pass boundary, everything the EE thread hands the
// VIF1/VU1/GIF pipeline (PS2X_PIPE_CAP), and replays it at startup through the real pipeline
// while hashing the packet sequence the arbiter hands the GS (PS2X_PIPE_BENCH).
#pragma once
#include <cstdint>

class PS2Memory;
class PS2Runtime;

namespace ps2pipe
{
    bool capEnabled();
    // Called at the start of every processPendingTransfers pass with the pass's quiescence.
    void capPassBegin(PS2Memory &mem, bool quiescent);
    void capPassEnd();
    void capVif1(const uint8_t *data, uint32_t size);
    void capGif3(const uint8_t *data, uint32_t size, bool drainImmediately);
    void capDrain();
    void capVifReg(uint32_t addr, uint32_t value);
    void capVu1Mem(bool code, uint32_t offset, const void *data, uint32_t size);
    void capGsReg(uint8_t addr, uint64_t value);          // cont.318: out-of-stream GS context write
    void capGsClear(uint32_t contextIndex, uint32_t rgba); // cont.318: the HLE framebuffer clear
    // PS2X_PIPE_BENCH=<file>: replay and exit. Returns false when not configured.
    bool benchRun(PS2Runtime &rt);
}
