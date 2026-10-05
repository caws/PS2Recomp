#pragma once

#include <cstdint>

class PS2Runtime;

// DBCMAN (DualShock controller manager) pad HLE — runtime side.
//
// Re-expressed for upstream ee14958, which deleted the old ps2_iop_dbcman.cpp and removed
// runtime->iop().handleRPC. The example game drives DBCMAN over the RAW SIF transport
// (guest 0x11bfa8/0x11c178) that native ee14958 (SID-routed off sceSifCallRpc) never sees, so the
// GAME OVERRIDE intercepts that transport and calls answerDbcManRpc() directly (a plain free
// function). deliverPadData() MUST be called at the vblank-interrupt-worker frequency
// (Interrupt.cpp) — a guest-side override hook fires far too rarely under present starvation, so
// the per-vblank work-table re-assertion is lost and the controller reads "removed" (self-exit).
namespace ps2_dbcman_hle
{
    // Answer a DBCMAN-family / event RPC (sid 0x80001300 family, or the 0x8000131b/c event
    // channels). Writes the reply into guest memory, sets resultPtr = recv, returns true (handled).
    bool answerDbcManRpc(uint8_t *rdram, PS2Runtime *runtime, uint32_t sid, uint32_t rpcNum,
                         uint32_t sendBuf, uint32_t sendSize, uint32_t recvBuf, uint32_t recvSize,
                         uint32_t &resultPtr);

    // Per-vblank: re-assert the per-port connection table + refresh the pad double-buffers. Call
    // from the runtime vblank interrupt worker (see Kernel/Syscalls/Interrupt.cpp).
    void deliverPadData(uint8_t *rdram, PS2Runtime *runtime);

    // Reset all pad state (sockets, work addr, seq). Call on IOP reset.
    void reset();
}
