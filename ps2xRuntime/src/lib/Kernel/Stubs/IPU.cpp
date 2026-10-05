#include "Common.h"
#include "IPU.h"

namespace
{
    constexpr uint32_t REG_IPU_CTRL = 0x10002010u;
    constexpr uint32_t REG_IPU_CMD = 0x10002000u;
    constexpr uint32_t REG_IPU_IN_FIFO = 0x10007010u;
    constexpr uint32_t REG_D4_CHCR = 0x1000B400u;

    // cont.245: the tables SCE's libipu sceIpuInit loads into the IPU, as constants. The previous
    // version read them from two hard-coded GUEST addresses (0x1721e0 / 0x172230) -- one game's copy of
    // libipu's data -- which on any other ELF is whatever happens to live there (SLES_520.17: code; a
    // quantiser matrix of instruction bytes gives striped, over-saturated IDEC pictures). These bytes
    // are libipu's own (read from SLES_520.17's sceIpuInit at 0x1189B0: `lq` x8 from 0x26FA90, x2 from
    // 0x26FAE0): the MPEG-2 default intra quantiser matrix (ISO 13818-2 6.3.11) in ZIGZAG order -- the
    // order SETIQ takes, and the order PCSX2 indexes it (IPU_MultiISA.cpp get_intra_block:
    // quant_matrix[i], i = scan position) -- then one 16-byte all-16 row that libipu sends four times
    // for the non-intra matrix, and the 16-entry RGB16 CLUT for SETVQ.
    alignas(16) constexpr uint8_t kLibIpuIQ[80] = {
        8, 16, 16, 19, 16, 19, 22, 22, 22, 22, 22, 22, 26, 24, 26, 27,
        27, 27, 26, 26, 26, 26, 27, 27, 27, 29, 29, 29, 34, 34, 34, 29,
        29, 29, 27, 27, 29, 29, 32, 32, 34, 34, 37, 38, 37, 35, 35, 34,
        35, 38, 38, 40, 40, 40, 48, 48, 46, 46, 56, 56, 58, 69, 69, 83,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    };
    alignas(16) constexpr uint8_t kLibIpuVQ[32] = {
        0, 0, 33, 4, 66, 8, 224, 3, 132, 16, 165, 20, 198, 24, 231, 28,
        31, 0, 41, 37, 74, 41, 0, 124, 140, 49, 173, 53, 255, 127, 206, 57,
    };

    __m128i loadTable(const uint8_t *p)
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    }

    // libipu sceIpuInit, register for register (SLES_520.17 0x1189B0): its helper first points DMA
    // channel 4 at the IPU (D4_CHCR = DIR, STR clear), then reset, BCLR, the 8-qword IQ FIFO fill
    // (intra table, then the all-16 row four times), SETIQ intra + non-intra, the 2-qword VQ fill,
    // SETVQ, SETTH(0,0), reset, BCLR. The real function polls IPU_CTRL.BUSY between commands; this
    // runtime's IPU completes each command inside the store, so no polling is needed here.
    void completeIpuInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PS2Memory &mem = runtime->memory();
        mem.write32(REG_D4_CHCR, 0x1u);
        mem.write32(REG_IPU_CTRL, 0x40000000u);
        mem.write32(REG_IPU_CMD, 0u);

        for (int i = 0; i < 4; ++i)
            mem.write128(REG_IPU_IN_FIFO, loadTable(kLibIpuIQ + 16 * i));
        for (int i = 0; i < 4; ++i)
            mem.write128(REG_IPU_IN_FIFO, loadTable(kLibIpuIQ + 64));

        mem.write32(REG_IPU_CMD, 0x50000000u);
        mem.write32(REG_IPU_CMD, 0x58000000u);

        mem.write128(REG_IPU_IN_FIFO, loadTable(kLibIpuVQ + 0));
        mem.write128(REG_IPU_IN_FIFO, loadTable(kLibIpuVQ + 16));

        mem.write32(REG_IPU_CMD, 0x60000000u);
        mem.write32(REG_IPU_CMD, 0x90000000u);
        mem.write32(REG_IPU_CTRL, 0x40000000u);
        mem.write32(REG_IPU_CMD, 0u);
        setReturnS32(ctx, 0);
    }
}

namespace ps2_stubs
{
    void sceIpuInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!runtime)
            return;

        if (!runtime->memory().getRDRAM())
        {
            if (!runtime->memory().initialize())
            {
                setReturnS32(ctx, -1);
                return;
            }
        }

        if (!runtime->syncCoreSubsystems())
        {
            setReturnS32(ctx, -1);
            return;
        }

        completeIpuInit(rdram, ctx, runtime);
    }

    void sceIpuRestartDMA(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceIpuStopDMA(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceIpuSync(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }
}
