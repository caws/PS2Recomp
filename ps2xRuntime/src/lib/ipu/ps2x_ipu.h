// ps2xRuntime <-> PCSX2 IPU seam (cont.245, 2026-09-05). See README.md.
//
// The PS2's Image Processing Unit (MPEG-2 macroblock decoder) as PCSX2 implements it, driven by the
// game's own register writes and DMA kicks. This runtime executes the guest on one EE thread and has no
// cycle-accurate event scheduler, so the PCSX2 "interrupt events" the IPU/DMA code schedules are run
// to quiescence synchronously (ps2xIpuPump) inside the register/DMA write that caused them. The game
// therefore sees every IPU command and every IPU DMA complete before the store instruction returns,
// which is a legal (fast) ordering of the real hardware's behaviour.
#pragma once
#include <cstdint>

class PS2Memory;

namespace ps2x_ipu
{
	// PS2X_IPU (default 1): the PCSX2 IPU. "0" keeps the old register stub (no decoding).
	bool enabled();
	// Called once memory exists: binds the DMA to RAM/scratchpad and resets the IPU.
	void init(PS2Memory* memory, uint8_t* rdram, uint8_t* scratchpad);
	void reset();

	// The 0x10002000 register block (CMD/CTRL/BP/TOP), 32-bit accesses.
	uint32_t read32(uint32_t address);
	void write32(uint32_t address, uint32_t value);
	// cont.247: the same block, 64-bit accesses (PCSX2 HwRead.cpp _hwRead64 page 0x02 -> ipuRead64,
	// HwWrite.cpp -> ipuWrite64). libmpeg polls IPU_CMD and IPU_TOP with `ld`: bit 63 is the BUSY flag
	// (ipuRegs.cmd.BUSY / ipuRegs.topbusy), bits 0-31 the VDEC/FDEC result.
	uint64_t read64(uint32_t address);
	void write64(uint32_t address, uint64_t value);
	// The IPU FIFOs: IPU_in at 0x10007010 (128-bit writes), IPU_out at 0x10007000 (128-bit reads).
	void writeFifoIn(const uint32_t* qword);
	void readFifoOut(uint32_t* qword);

	// A CHCR write with STR set on channel 3 (fromIPU, 0x1000B000) or 4 (toIPU, 0x1000B400).
	// regs: the channel's register block as the guest programmed it (offsets 0x00 CHCR, 0x10 MADR,
	// 0x20 QWC, 0x30 TADR, 0x40 ASR0, 0x50 ASR1, 0x80 SADR). On return the same words hold the
	// channel's state after the transfer ran (STR cleared when it finished); *finished says whether the
	// DMAC end-of-transfer interrupt for that channel should be raised.
	void dmaKick(int channel, uint32_t* chcr, uint32_t* madr, uint32_t* qwc, uint32_t* tadr,
	             uint32_t* asr0, uint32_t* asr1, bool* finished);
	// cont.247: a CHCR write with STR CLEAR on channel 3 or 4 (PCSX2 Dmac.cpp DmaExec): a running
	// channel is stopped in place -- only STR drops, MADR/QWC/TADR keep the transfer's progress and the
	// channel's pending DMA event is cancelled; a stopped channel takes the whole CHCR value. libipu's
	// sceIpuStopDMA relies on this to park the bitstream chain (D4) while the CSC borrows the channel,
	// then reads MADR/QWC/TADR back to rewind by the FIFO's IFC+FP qwords on sceIpuRestartDMA.
	void dmaWriteStopped(int channel, uint32_t* chcr);
	// Whether a channel is still in progress (STR set) after the last pump.
	bool dmaRunning(int channel);
	// The channel's current state (a decode kicked by a later IPU command finishes a DMA armed earlier:
	// call after every IPU register/FIFO write to mirror the channel back into the guest's registers).
	void channelState(int channel, uint32_t* chcr, uint32_t* madr, uint32_t* qwc, uint32_t* tadr, bool* finished);
	// An INTC IPU interrupt was raised since the last call.
	bool takeIntc();
}
