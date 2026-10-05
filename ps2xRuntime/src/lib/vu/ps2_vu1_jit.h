#ifndef PS2_VU1_JIT_H
#define PS2_VU1_JIT_H

// ---------------------------------------------------------------------------
// cont.182: VU1 block-JIT infrastructure — executable code buffer + a minimal x86-64/SSE4.1
// emitter, with an exhaustive self-test.
//
// WHY A JIT AT ALL (all measured, see docs/vu1-jit.md): the interpreter is at ~120 ns/pair, which
// is ~420 host cycles to execute two VU instructions. 30 fps needs 17 ns/pair (~60 cycles). The
// profile is flat — no leaf above 12% — because the cost is per-instruction plumbing (decode fetch,
// switch dispatch, pipeline queue/commit, memcpy shadow dance), not arithmetic. That shape is what
// only a code generator fixes.
//
// WHY IT IS BUILDABLE NOW: cont.180 adopted PCSX2's FMAC result model, so both VU clamps became
// pure functions of a lane's bits — branchless quad SSE sequences (cont.181 proved them
// bit-identical to the scalar reference over 264,192 cases). A translated pair is therefore
// load / clamp / arithmetic / clamp / blend / store with no branches, ~15-20 SSE ops.
//
// This header is INCLUDED ONLY BY ps2_vu1_core.cpp, so everything is static: no new translation
// unit and no CMakeLists change.
// ---------------------------------------------------------------------------

#include "ps2_vu1_detail.h" // cont.215: vuNormOperandBits, to FOLD the I immediate at plan time
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <sys/mman.h>

namespace vu1jit
{
    // ⚠ These self-tests run from inside VU1Interpreter::run, i.e. on the GUEST thread's stack,
    // which is small. Their large state objects are therefore `static`, not locals: a few hundred
    // bytes each of VU1State/FakeState exhausted that stack and corrupted the frame, surfacing as
    // a SIGBUS at the top of run() with unreadable locals — nothing to do with the emitted code.

    // ---- executable code buffer ------------------------------------------------------------
    // One RWX region. Kept simple deliberately: blocks are append-only and the whole buffer is
    // reset when the guest uploads new microcode (the decode cache already detects that).
    struct CodeBuffer
    {
        uint8_t *base = nullptr;   // start of the mapping; the constant pool lives here
        uint8_t *code = nullptr;   // first byte available for code (after the pool)
        size_t capacity = 0;
        size_t used = 0;           // bytes of CODE emitted (relative to `code`)
        bool failed = false;

        // ★ The constant pool MUST live inside the mapping. x86-64 RIP-relative displacements are
        // signed 32-bit, and mmap places the code buffer wherever it likes — typically far more
        // than ±2GB from the binary's data segment, which silently truncates the displacement and
        // jumps into nowhere. (That is exactly how the first version of this crashed.)
        static constexpr size_t kPoolBytes = 1536;
        // ★ cont.204: 512..767 is a RESULT SCRATCH -- one quadword per FMAC in a block. When the
        // program is not lazy-flag clean the block stores each FMAC's PRE-CLAMP result here and the
        // C++ driver derives the MAC flags from it with the existing bit-exact helper, instead of
        // the block emitting the flag math itself. One store per FMAC buys blocks in programs that
        // were previously excluded wholesale -- 68.8% of all still-interpreted cycles.
        static constexpr size_t kResultScratch = 512;

        bool ensure(size_t bytes)
        {
            if (base != nullptr || failed)
                return base != nullptr;
            void *p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE | PROT_EXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED)
            {
                // W^X-hardened kernels can refuse RWX; the JIT simply stays off.
                failed = true;
                return false;
            }
            base = static_cast<uint8_t *>(p);
            code = base + kPoolBytes;
            capacity = bytes;
            used = 0;
            initPool();
            return true;
        }

        // Pool layout, each entry 16-byte aligned:
        //   [0] expMask  [16] signMask  [32] maxMag  [48] magMask
        //   [64 + 16*d]  dest-lane mask for dest nibble d (0..15)
        void initPool()
        {
            auto put = [&](size_t off, uint32_t v) {
                uint32_t quad[4] = {v, v, v, v};
                std::memcpy(base + off, quad, sizeof(quad));
            };
            put(0, 0x7F800000u);
            put(16, 0x80000000u);
            put(32, 0x7F7FFFFFu);
            put(48, 0x7FFFFFFFu);
            // ITOF scale factors 1/2^n for n = 0, 4, 12, 15, at 320/336/352/368.
            // Exact powers of two, so multiplying by 1/2^n is identical to the interpreter's
            // division by 2^n for every finite input.
            {
                const float sc[4] = {1.0f, 1.0f / 16.0f, 1.0f / 4096.0f, 1.0f / 32768.0f};
                for (uint32_t k = 0; k < 4u; ++k)
                {
                    float quad[4] = {sc[k], sc[k], sc[k], sc[k]};
                    std::memcpy(base + 320 + 16 * k, quad, sizeof(quad));
                }
            }
            // FTOI scale-up factors 2^n (n = 0,4,12,15) at 384/400/416/432, then the positive
            // saturation threshold and INT32_MAX at 448/464.
            {
                const float sc[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
                for (uint32_t k = 0; k < 4u; ++k)
                {
                    float quad[4] = {sc[k], sc[k], sc[k], sc[k]};
                    std::memcpy(base + 384 + 16 * k, quad, sizeof(quad));
                }
                // (float)INT32_MAX rounds to 2147483648.0f, which is the smallest float that
                // satisfies the interpreter's `scaled >= (double)INT32_MAX` test.
                const float thr = 2147483648.0f;
                float tq[4] = {thr, thr, thr, thr};
                std::memcpy(base + 448, tq, sizeof(tq));
                put(464, 0x7FFFFFFFu);
            }
            for (uint32_t d = 0; d < 16u; ++d)
            {
                // lane c (0..3) enabled iff d & (1 << (3 - c)) — matches laneForComponent()
                uint32_t quad[4];
                for (uint32_t c = 0; c < 4u; ++c)
                    quad[c] = (d & (1u << (3u - c))) ? 0xFFFFFFFFu : 0u;
                std::memcpy(base + 64 + 16 * d, quad, sizeof(quad));
            }
        }
        const void *expMask() const { return base + 0; }
        const void *signMask() const { return base + 16; }
        const void *maxMag() const { return base + 32; }
        const void *magMask() const { return base + 48; }
        const void *destMask(uint8_t d) const { return base + 64 + 16 * (d & 15u); }
        const void *itofScale(uint8_t k) const { return base + 320 + 16 * (k & 3u); }
        const void *ftoiScale(uint8_t k) const { return base + 384 + 16 * (k & 3u); }
        const void *ftoiThreshold() const { return base + 448; }
        const void *intMax() const { return base + 464; }
        void *resultSlot(uint32_t k) const { return base + kResultScratch + 16u * (k & 31u); }

        void reset() { used = 0; }
        size_t room() const { return capacity - kPoolBytes - used; }
    };

    // ---- minimal x86-64 SSE emitter --------------------------------------------------------
    // Restricted on purpose to xmm0..xmm7 and base registers rdi/rsi/rdx/rcx, so **no REX prefix
    // is ever required** — that removes the single most bug-prone part of x86 encoding. Every
    // instruction here is exercised by selfTest() below.
    struct Emitter
    {
        uint8_t *p = nullptr;
        uint8_t *end = nullptr;
        bool overflow = false;

        void u8(uint8_t v)
        {
            if (p >= end) { overflow = true; return; }
            *p++ = v;
        }
        void u32(uint32_t v)
        {
            u8(uint8_t(v)); u8(uint8_t(v >> 8)); u8(uint8_t(v >> 16)); u8(uint8_t(v >> 24));
        }

        // ModRM with a register-direct operand (mod = 11).
        void modrmReg(uint8_t reg, uint8_t rm) { u8(uint8_t(0xC0 | ((reg & 7) << 3) | (rm & 7))); }
        // ModRM with [base] and no displacement (base must not be rbp/rsp/r12/r13).
        void modrmMem(uint8_t reg, uint8_t base) { u8(uint8_t(0x00 | ((reg & 7) << 3) | (base & 7))); }
        // ModRM RIP-relative: mod=00, rm=101, then disp32 from the END of the instruction.
        void modrmRip(uint8_t reg, const void *target)
        {
            u8(uint8_t(0x00 | ((reg & 7) << 3) | 5));
            // disp32 is relative to the next instruction, i.e. after these 4 bytes.
            const int64_t delta = reinterpret_cast<const uint8_t *>(target) - (p + 4);
            u32(uint32_t(int32_t(delta)));
        }

        // movups xmm(reg), [base]
        void movups_load(uint8_t reg, uint8_t base) { u8(0x0F); u8(0x10); modrmMem(reg, base); }
        // movups [base], xmm(reg)
        void movups_store(uint8_t base, uint8_t reg) { u8(0x0F); u8(0x11); modrmMem(reg, base); }
        // movaps xmm(dst), xmm(src)
        void movaps_rr(uint8_t dst, uint8_t src) { u8(0x0F); u8(0x28); modrmReg(dst, src); }

        void pand_rr(uint8_t d, uint8_t s) { u8(0x66); u8(0x0F); u8(0xDB); modrmReg(d, s); }
        void pandn_rr(uint8_t d, uint8_t s) { u8(0x66); u8(0x0F); u8(0xDF); modrmReg(d, s); }
        void por_rr(uint8_t d, uint8_t s) { u8(0x66); u8(0x0F); u8(0xEB); modrmReg(d, s); }
        void pxor_rr(uint8_t d, uint8_t s) { u8(0x66); u8(0x0F); u8(0xEF); modrmReg(d, s); }
        void pcmpeqd_rr(uint8_t d, uint8_t s) { u8(0x66); u8(0x0F); u8(0x76); modrmReg(d, s); }

        // 128-bit constant loads, RIP-relative (constants live outside the code buffer).
        void movaps_rip(uint8_t reg, const void *c) { u8(0x0F); u8(0x28); modrmRip(reg, c); }
        void pand_rip(uint8_t reg, const void *c) { u8(0x66); u8(0x0F); u8(0xDB); modrmRip(reg, c); }
        void por_rip(uint8_t reg, const void *c) { u8(0x66); u8(0x0F); u8(0xEB); modrmRip(reg, c); }
        void pcmpeqd_rip(uint8_t reg, const void *c) { u8(0x66); u8(0x0F); u8(0x76); modrmRip(reg, c); }

        // [base + disp32] forms (mod = 10)
        void modrmDisp32(uint8_t reg, uint8_t base, int32_t disp)
        {
            u8(uint8_t(0x80 | ((reg & 7) << 3) | (base & 7)));
            u32(uint32_t(disp));
        }
        void movups_load_d(uint8_t reg, uint8_t base, int32_t disp)
        { u8(0x0F); u8(0x10); modrmDisp32(reg, base, disp); }
        void movups_store_d(uint8_t base, int32_t disp, uint8_t reg)
        { u8(0x0F); u8(0x11); modrmDisp32(reg, base, disp); }

        void movss_load_d(uint8_t reg, uint8_t base, int32_t disp)
        { u8(0xF3); u8(0x0F); u8(0x10); modrmDisp32(reg, base, disp); }
        // mov dword ptr [base + disp32], imm32   (C7 /0) -- cont.215, the I-register store
        void mov_mem_imm32(uint8_t base, int32_t disp, uint32_t imm)
        { u8(0xC7); modrmDisp32(0, base, disp); u32(imm); }
        void cvtdq2ps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x5B); modrmReg(d, s); }
        void cvttps2dq_rr(uint8_t d, uint8_t s) { u8(0xF3); u8(0x0F); u8(0x5B); modrmReg(d, s); }
        void cmpps_rri(uint8_t d, uint8_t s, uint8_t imm) { u8(0x0F); u8(0xC2); modrmReg(d, s); u8(imm); }
        void mulps_rip(uint8_t reg, const void *c) { u8(0x0F); u8(0x59); modrmRip(reg, c); }
        void maxps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x5F); modrmReg(d, s); }
        void minps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x5D); modrmReg(d, s); }
        void andps_rip(uint8_t reg, const void *c) { u8(0x0F); u8(0x54); modrmRip(reg, c); }
        void mulps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x59); modrmReg(d, s); }
        void addps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x58); modrmReg(d, s); }
        void subps_rr(uint8_t d, uint8_t s) { u8(0x0F); u8(0x5C); modrmReg(d, s); }
        // shufps xmm(d), xmm(s), imm8
        void shufps_rri(uint8_t d, uint8_t s, uint8_t imm)
        { u8(0x0F); u8(0xC6); modrmReg(d, s); u8(imm); }
        void pandn_rip(uint8_t reg, const void *c) { u8(0x66); u8(0x0F); u8(0xDF); modrmRip(reg, c); }
        // movups [rip + c], xmm(reg)
        void movups_store_rip(const void *c, uint8_t reg) { u8(0x0F); u8(0x11); modrmRip(reg, c); }

        // ---- 32-bit GPR forms (eax/ecx/edx/esi/edi only -> still no REX) ----
        // mov r32, [base + disp32]
        void mov_r32_mem(uint8_t reg, uint8_t base, int32_t disp)
        { u8(0x8B); modrmDisp32(reg, base, disp); }
        // add eax, imm32  (short form)
        void add_eax_imm32(int32_t imm) { u8(0x05); u32(uint32_t(imm)); }
        // shl r32, imm8
        void shl_r32_imm8(uint8_t reg, uint8_t imm) { u8(0xC1); modrmReg(4, reg); u8(imm); }
        // and r32(dst), r32(src)
        void and_r32_r32(uint8_t dst, uint8_t src) { u8(0x23); modrmReg(dst, src); }

        // mov [base + disp32], r32
        void mov_mem_r32(uint8_t base, int32_t disp, uint8_t reg)
        { u8(0x89); modrmDisp32(reg, base, disp); }
        // sub eax, imm32  (short form)
        void sub_eax_imm32(int32_t imm) { u8(0x2D); u32(uint32_t(imm)); }
        // and eax, imm32  (short form; sets ZF, which FCAND consumes)
        void and_eax_imm32(uint32_t imm) { u8(0x25); u32(imm); }
        // or eax, imm32   (short form)
        void or_eax_imm32(uint32_t imm) { u8(0x0D); u32(imm); }
        // cmp eax, imm32  (short form)
        void cmp_eax_imm32(uint32_t imm) { u8(0x3D); u32(imm); }
        // movsx r32, r16  -- the (int16_t) truncation IADDIU/ISUBIU apply before the VI store
        void movsx_r32_r16(uint8_t d, uint8_t s) { u8(0x0F); u8(0xBF); modrmReg(d, s); }
        // setcc r/m8 (reg field is /0). cc 0x4 = E/Z, 0x5 = NE/NZ. al/cl/dl/bl need no REX.
        void setcc_r8(uint8_t cc, uint8_t reg)
        { u8(0x0F); u8(uint8_t(0x90 | (cc & 0xFu))); modrmReg(0, reg); }
        // movzx r32, r8
        void movzx_r32_r8(uint8_t d, uint8_t s) { u8(0x0F); u8(0xB6); modrmReg(d, s); }
        // movsx r32, WORD [base + disp32]  -- the (int16_t) a branch compare applies to vi[]
        void movsx_r32_mem16(uint8_t reg, uint8_t base, int32_t disp)
        { u8(0x0F); u8(0xBF); modrmDisp32(reg, base, disp); }
        // movzx r32, WORD [base + disp32]  -- JR/JALR zero-extend vi[is] before *8
        void movzx_r32_mem16(uint8_t reg, uint8_t base, int32_t disp)
        { u8(0x0F); u8(0xB7); modrmDisp32(reg, base, disp); }
        // mov r32, imm32
        void mov_r32_imm32(uint8_t reg, uint32_t imm) { u8(uint8_t(0xB8 + (reg & 7))); u32(imm); }
        // cmp r32(a), r32(b)
        void cmp_r32_r32(uint8_t a, uint8_t b) { u8(0x3B); modrmReg(a, b); }
        // cmp r32, imm32 (long form, any low reg)
        void cmp_r32_imm32(uint8_t reg, int32_t imm)
        { u8(0x81); modrmReg(7, reg); u32(uint32_t(imm)); }
        // cmovcc r32(d), r32(s).  cc: 4=E 5=NE C=L D=GE E=LE F=G
        void cmovcc_r32_r32(uint8_t cc, uint8_t d, uint8_t src)
        { u8(0x0F); u8(uint8_t(0x40 | (cc & 0xFu))); modrmReg(d, src); }
        // mov r32(d), r32(s)
        void mov_r32_r32(uint8_t d, uint8_t src) { u8(0x8B); modrmReg(d, src); }
        // movd xmm(d), r32(s)
        void movd_xmm_r32(uint8_t d, uint8_t src) { u8(0x66); u8(0x0F); u8(0x6E); modrmReg(d, src); }
        // ALU r32, [base + disp32] -- second operand straight from memory, so no scratch GPR is
        // needed. ★ This matters: ECX is RESERVED to carry a branch target across the delay slot
        // (see emitBranch), so a lower-slot emitter must never touch it.
        //   add 03 /r   sub 2B /r   and 23 /r   or 0B /r
        void alu_r32_mem(uint8_t opcode, uint8_t reg, uint8_t base, int32_t disp)
        { u8(opcode); modrmDisp32(reg, base, disp); }
        // pshufd xmm(d), xmm(s), imm8
        void pshufd_rri(uint8_t d, uint8_t src, uint8_t imm)
        { u8(0x66); u8(0x0F); u8(0x70); modrmReg(d, src); u8(imm); }

        // [base + index*1] via SIB (mod=00, rm=100). Used for VU-memory access: base = rsi
        // (vuData), index = rax (the computed byte address).
        void modrmSib(uint8_t reg, uint8_t base, uint8_t index)
        {
            u8(uint8_t(0x00 | ((reg & 7) << 3) | 4));
            u8(uint8_t(((index & 7) << 3) | (base & 7)));
        }
        // [base + index*1 + disp8] : mod=01, rm=100 (SIB), then disp8
        void modrmSibDisp8(uint8_t reg, uint8_t base, uint8_t index, int8_t disp)
        {
            u8(uint8_t(0x40 | ((reg & 7) << 3) | 4));
            u8(uint8_t(((index & 7) << 3) | (base & 7)));
            u8(uint8_t(disp));
        }
        // movsx r32, WORD [base + index + disp8]
        void movsx_r32_mem16_sib(uint8_t reg, uint8_t base, uint8_t index, int8_t disp)
        { u8(0x0F); u8(0xBF); modrmSibDisp8(reg, base, index, disp); }
        void movups_load_sib(uint8_t reg, uint8_t base, uint8_t index)
        { u8(0x0F); u8(0x10); modrmSib(reg, base, index); }
        void movups_store_sib(uint8_t base, uint8_t index, uint8_t reg)
        { u8(0x0F); u8(0x11); modrmSib(reg, base, index); }

        void ret() { u8(0xC3); }
    };

    // select(mask, a, b) == (a & mask) | (b & ~mask), using pand/pandn/por so that no register is
    // implicitly constrained (blendvps would force the mask into xmm0).
    // Clobbers `tmp`; result lands in `a`.
    inline void emitSelect(Emitter &e, uint8_t a, uint8_t b, uint8_t mask, uint8_t tmp)
    {
        e.pand_rr(a, mask);      // a &= mask
        e.movaps_rr(tmp, mask);  // tmp = mask
        e.pandn_rr(tmp, b);      // tmp = ~mask & b
        e.por_rr(a, tmp);        // a |= tmp
    }

    // Emit the operand clamp (PCSX2 vuDouble / VU1Interpreter::normalizeOperand) on xmm(v).
    // Uses t0..t3 as scratch. Mirrors vuNormOperandBits in ps2_vu1_detail.h exactly.
    inline void emitNormOperand(Emitter &e, const CodeBuffer &buf, uint8_t v,
                                uint8_t t0, uint8_t t1, uint8_t t2, uint8_t t3)
    {
        e.movaps_rr(t0, v);
        e.pand_rip(t0, buf.expMask());      // t0 = exponent field
        e.movaps_rr(t1, v);
        e.pand_rip(t1, buf.signMask());     // t1 = sign
        e.pxor_rr(t2, t2);
        e.pcmpeqd_rr(t2, t0);               // t2 = isDenorm (exp == 0)
        e.movaps_rr(t3, t0);
        e.pcmpeqd_rip(t3, buf.expMask());   // t3 = isInf (exp == 0xFF)
        // maxv = sign | 0x7F7FFFFF, reuse t0 (exponent no longer needed)
        e.movaps_rr(t0, t1);
        e.por_rip(t0, buf.maxMag());
        // v = select(isInf, maxv, v)  -> compute into t0, then move back
        emitSelect(e, t0, v, t3, /*tmp*/ t3);
        e.movaps_rr(v, t0);
        // v = select(isDenorm, sign, v)
        emitSelect(e, t1, v, t2, /*tmp*/ t2);
        e.movaps_rr(v, t1);
    }

    // Emit the RESULT clamp, value only (== vuNormResultQuadValue == PCSX2 VU_MAC_UPDATE's value
    // half). Differs from the operand clamp in the zero case: exact zero keeps its bits, only a
    // nonzero denormal flushes to signed zero. Flags are not produced — lazy flags (cont.177)
    // proved they are unobservable for the programs this will compile.
    inline void emitNormResultValue(Emitter &e, const CodeBuffer &buf, uint8_t v,
                                    uint8_t t0, uint8_t t1, uint8_t t2, uint8_t t3)
    {
        e.movaps_rr(t0, v);
        e.pand_rip(t0, buf.expMask());      // t0 = exponent
        e.movaps_rr(t1, v);
        e.pand_rip(t1, buf.signMask());     // t1 = sign
        e.movaps_rr(t2, v);
        e.pand_rip(t2, buf.magMask());      // t2 = magnitude
        e.pxor_rr(t3, t3);
        e.pcmpeqd_rr(t3, t2);               // t3 = isZero
        // isDenorm = ~isZero & (exp == 0)  -> reuse t2
        e.pxor_rr(t2, t2);
        e.pcmpeqd_rr(t2, t0);               // t2 = (exp == 0)
        e.pandn_rr(t3, t2);                 // t3 = ~isZero & (exp == 0) = isDenorm
        e.pcmpeqd_rip(t0, buf.expMask());   // t0 = isInf
        // v = select(isInf, sign|maxMag, v)
        e.movaps_rr(t2, t1);
        e.por_rip(t2, buf.maxMag());        // t2 = sign|maxMag
        emitSelect(e, t2, v, t0, /*tmp*/ t0);
        e.movaps_rr(v, t2);
        // v = select(isDenorm, sign, v)
        emitSelect(e, t1, v, t3, /*tmp*/ t3);
        e.movaps_rr(v, t1);
    }

    // The FMAC shapes covered. Max/Min/Abs use the interpreter's `applyDest` (no result clamp and
    // no flags) rather than `applyFmacDest` — a real semantic difference, so it is explicit here.
    enum class FmacOp { Add, Sub, Mul, Madd, Msub, Max, Min, Abs, Nop, Itof, Ftoi, Clip };

    // Where the second operand comes from. cont.186 showed the q/i-operand family is the single
    // biggest gap in upper coverage (MULq alone is 6.8% of executed pairs).
    enum class OperandSrc { VtLane, VtQuad, ScalarQ, ScalarI, None };

    struct UpperPlan
    {
        FmacOp op = FmacOp::Add;
        OperandSrc src = OperandSrc::VtQuad;
        int bc = 0;              // lane index when src == VtLane
        bool clampResult = true; // false for MAX/MINI/ABS (they use applyDest)
        bool writesAcc = false;  // the `special` A-forms accumulate into ACC
        bool writesFt = false;   // ABS writes vf[ft], not vf[fd]
        bool readsAcc = false;
        uint8_t itofShift = 0;   // index into itofScale for ITOF0/4/12/15
    };

    // Emit one complete upper instruction against a VU state pointer in rdi.
    //   vd[dest] = maybeClamp( f(clamp(vf[fs]), operand, clamp(acc)) )
    // Offsets are byte offsets from rdi. Uses xmm0..xmm7; no REX, no branches.
    inline void emitUpper(Emitter &e, const CodeBuffer &buf, const UpperPlan &plan,
                          int32_t offFs, int32_t offFt, int32_t offAcc, int32_t offQ,
                          int32_t offI, int32_t offDst, uint8_t dest,
                          const void *resultSink = nullptr)
    {
        if (plan.op == FmacOp::Nop)
            return; // upper NOP: no operands read, no destination written

        // ★ CLIP writes neither vf nor acc -- only the clip register, through the flag pipeline.
        // Its comparisons are on RAW bits (not normalized operands), so rather than emit them the
        // block stashes vf[fs] and vf[ft] and the driver replays the interpreter's own comparison
        // and queueClip at the right cycle. Bit-exact by construction, and one CLIP no longer ends
        // a block (it was 9.5% of residual cycles, and it fragments the runs around it).
        if (plan.op == FmacOp::Clip)
        {
            if (resultSink != nullptr)
            {
                e.movups_load_d(0, 7, offFs);
                e.movups_store_rip(resultSink, 0);
                e.movups_load_d(0, 7, offFt);
                e.movups_store_rip(static_cast<const uint8_t *>(resultSink) + 16, 0);
            }
            return;
        }

        if (plan.op == FmacOp::Ftoi)
        {
            // FTOI uses the CLAMPED operand (unlike ITOF) and writes vf[ft] via applyDest.
            // ★ Saturation differs from the hardware instruction: the interpreter's vuFloatToInt
            // clamps to INT32_MIN/INT32_MAX, but cvttps2dq yields 0x80000000 for out-of-range in
            // BOTH directions. Negative overflow therefore already matches (0x80000000 == INT32_MIN),
            // and only POSITIVE overflow needs fixing up to 0x7FFFFFFF.
            e.movups_load_d(0, 7, offFs);
            emitNormOperand(e, buf, 0, 1, 2, 3, 4);
            if (plan.itofShift != 0u)
                e.mulps_rip(0, buf.ftoiScale(plan.itofShift));
            e.movaps_rr(5, 0);          // keep the scaled float for the compare
            e.cvttps2dq_rr(0, 0);       // truncate toward zero
            e.movaps_rip(1, buf.ftoiThreshold());
            e.cmpps_rri(5, 1, 5);       // predicate 5 = NLT, i.e. t >= 2147483648.0f
            e.movaps_rip(2, buf.intMax());
            e.pand_rr(2, 5);            // INT32_MAX where saturating
            e.pandn_rr(5, 0);           // truncated value elsewhere
            e.por_rr(2, 5);
            e.movaps_rr(0, 2);

            e.movaps_rip(2, buf.destMask(dest));
            e.pand_rr(0, 2);
            e.movups_load_d(1, 7, offDst);
            e.pandn_rr(2, 1);
            e.por_rr(0, 2);
            e.movups_store_d(7, offDst, 0);
            return;
        }

        if (plan.op == FmacOp::Itof)
        {
            // ★ ITOF reads vf[fs] as RAW INT32 BITS, not as a clamped float operand — the
            // interpreter memcpy's the bits into an int32. Clamping here would corrupt every
            // value whose bit pattern happens to look like a denormal or an infinity.
            // No result clamp either: ITOF uses applyDest, not applyFmacDest.
            e.movups_load_d(0, 7, offFs);
            e.cvtdq2ps_rr(0, 0);
            if (plan.itofShift != 0u)
                e.mulps_rip(0, buf.itofScale(plan.itofShift));
            e.movaps_rip(2, buf.destMask(dest));
            e.pand_rr(0, 2);
            e.movups_load_d(1, 7, offDst);
            e.pandn_rr(2, 1);
            e.por_rr(0, 2);
            e.movups_store_d(7, offDst, 0);
            return;
        }

        // vs
        e.movups_load_d(0, 7, offFs);
        emitNormOperand(e, buf, 0, 1, 2, 3, 4);
        e.movaps_rr(5, 0); // xmm5 = vs

        // second operand -> xmm6
        switch (plan.src)
        {
        case OperandSrc::VtLane:
            e.movups_load_d(0, 7, offFt);
            emitNormOperand(e, buf, 0, 1, 2, 3, 4);
            e.shufps_rri(0, 0, uint8_t(plan.bc * 0x55));
            e.movaps_rr(6, 0);
            break;
        case OperandSrc::VtQuad:
            e.movups_load_d(0, 7, offFt);
            emitNormOperand(e, buf, 0, 1, 2, 3, 4);
            e.movaps_rr(6, 0);
            break;
        case OperandSrc::ScalarQ:
        case OperandSrc::ScalarI:
            // The interpreter normalises q and i exactly like any other operand, then broadcasts.
            e.movss_load_d(0, 7, plan.src == OperandSrc::ScalarQ ? offQ : offI);
            e.shufps_rri(0, 0, 0x00);
            emitNormOperand(e, buf, 0, 1, 2, 3, 4);
            e.movaps_rr(6, 0);
            break;
        case OperandSrc::None:
            break;
        }

        if (plan.readsAcc)
        {
            e.movups_load_d(0, 7, offAcc);
            emitNormOperand(e, buf, 0, 1, 2, 3, 4);
            e.movaps_rr(7, 0); // xmm7 = acc
        }

        e.movaps_rr(0, 5);
        switch (plan.op)
        {
        case FmacOp::Add: e.addps_rr(0, 6); break;
        case FmacOp::Sub: e.subps_rr(0, 6); break;
        case FmacOp::Mul: e.mulps_rr(0, 6); break;
        case FmacOp::Madd: e.mulps_rr(0, 6); e.addps_rr(0, 7); break;
        case FmacOp::Msub: e.mulps_rr(0, 6); e.movaps_rr(1, 7); e.subps_rr(1, 0); e.movaps_rr(0, 1); break;
        // maxps/minps are `dst = (dst OP src) ? dst : src`, matching the interpreter's
        // `(vs > bc) ? vs : bc` exactly. Operands are already clamped, so no NaN can reach here
        // and maxps's NaN behaviour never applies.
        case FmacOp::Max: e.maxps_rr(0, 6); break;
        case FmacOp::Min: e.minps_rr(0, 6); break;
        case FmacOp::Abs: e.andps_rip(0, buf.magMask()); break;
        }

        // ★ The PRE-clamp result is what the MAC flags derive from (cont.180's float model), so
        // it is stashed before the clamp -- the driver then runs the same vuNormResultQuad the
        // interpreter uses, making the flags bit-exact by construction rather than by re-derivation.
        if (resultSink != nullptr && plan.clampResult)
            e.movups_store_rip(resultSink, 0);
        if (plan.clampResult)
            emitNormResultValue(e, buf, 0, 1, 2, 3, 4);

        // dst = (result & destMask) | (dst & ~destMask).
        // `pandn dst, src` is `dst = ~dst & src`, so the MASK must be the destination operand —
        // putting the old value there would compute ~old & mask instead.
        e.movaps_rip(2, buf.destMask(dest));
        e.pand_rr(0, 2);
        e.movups_load_d(1, 7, offDst);
        e.pandn_rr(2, 1);
        e.por_rr(0, 2);
        e.movups_store_d(7, offDst, 0);
    }

    // Offsets into VU1State, taken from the struct itself rather than hardcoded — a layout change
    // would otherwise silently point the emitted loads at the wrong fields.
    static constexpr int32_t kOffVf = int32_t(offsetof(VU1State, vf));
    static constexpr int32_t kOffAcc = int32_t(offsetof(VU1State, acc));
    static constexpr int32_t kOffQ = int32_t(offsetof(VU1State, q));
    static constexpr int32_t kOffI = int32_t(offsetof(VU1State, i));
    static constexpr int32_t kOffVi = int32_t(offsetof(VU1State, vi));
    static constexpr int32_t kOffClip = int32_t(offsetof(VU1State, clip));
    static constexpr int32_t kOffPc = int32_t(offsetof(VU1State, pc));

    // ---- lower slot: VU-memory access ------------------------------------------------------
    // Emitted lower-slot code needs a wider calling convention than the upper slot:
    //     void f(VU1State *rdi, uint8_t *vuData rsi, uint32_t dataSizeMask edx)
    // rdi/rsi/rdx are all low registers, so this still needs no REX prefix.
    //
    // Address arithmetic, identical to ps2_vu1_lower.cpp:
    //     addr = ((uint32_t)(int32_t)(vi[base] + imm)) * 16;  addr &= (dataSize - 1);
    //     if (addr + 16 <= dataSize) { ... }
    //
    // ★ That guard is ALWAYS TRUE once the mask is applied, so the emitted code needs no branch:
    // `dataSize` is a power of two >= 16, `x * 16` has its low four bits clear, and masking with
    // (dataSize - 1) preserves that — so addr <= dataSize - 16, hence addr + 16 <= dataSize.
    typedef void (*LowerFn)(void *state, void *vuData, uint32_t dataSizeMask);

    // Compute the VU-memory byte address for vi[base] + imm into EAX.
    inline void emitVuAddress(Emitter &e, uint8_t viBase, int32_t imm)
    {
        e.mov_r32_mem(0 /*eax*/, 7 /*rdi*/, kOffVi + int32_t(viBase) * 4);
        if (imm != 0)
            e.add_eax_imm32(imm);
        e.shl_r32_imm8(0, 4);      // * 16
        e.and_r32_r32(0, 2 /*edx*/); // & (dataSize - 1)
    }

    // LQ: load a quadword from VU memory and applyDest it into vf[ft]. Applied IMMEDIATELY by the
    // interpreter (no pipeline), so the emitted form is a plain masked load plus a dest blend.
    // No operand clamp: the bytes go into the register exactly as stored.
    inline void emitLQ(Emitter &e, const CodeBuffer &buf, uint8_t viBase, uint8_t vfDst,
                       int32_t imm, uint8_t dest)
    {
        emitVuAddress(e, viBase, imm);
        e.movups_load_sib(0, 6 /*rsi*/, 0 /*rax*/);
        e.movaps_rip(2, buf.destMask(dest));
        e.pand_rr(0, 2);
        e.movups_load_d(1, 7, kOffVf + int32_t(vfDst) * 16);
        e.pandn_rr(2, 1);
        e.por_rr(0, 2);
        e.movups_store_d(7, kOffVf + int32_t(vfDst) * 16, 0);
    }

    // SQ: store a quadword from vf[fs] into VU memory, dest-lane masked.
    //
    // ★ The interpreter queues this through `queueStore` with **readyCycle = m_cycle + 1**, but an
    // IMMEDIATE store is equivalent: `commitReadyPipelines()` runs at the TOP of the next pair's
    // iteration, so the queued store lands before pair i+1 executes either way, and nothing else
    // runs in pair i's own cycle that could observe the difference (a pair has exactly one lower
    // instruction, and upper ops never touch VU memory).
    // Residual, documented: XGKICK streams VU memory per cycle, so a store it is concurrently
    // reading could be seen one cycle earlier. That is a race on real hardware too, and the
    // block-level shadow-verify would surface it.
    inline void emitSQ(Emitter &e, const CodeBuffer &buf, uint8_t viBase, uint8_t vfSrc,
                       int32_t imm, uint8_t dest)
    {
        emitVuAddress(e, viBase, imm);
        e.movups_load_d(0, 7, kOffVf + int32_t(vfSrc) * 16); // value to store
        e.movaps_rip(2, buf.destMask(dest));
        e.pand_rr(0, 2);
        e.movups_load_sib(1, 6 /*rsi*/, 0 /*rax*/);          // existing memory
        e.pandn_rr(2, 1);
        e.por_rr(0, 2);
        e.movups_store_sib(6 /*rsi*/, 0 /*rax*/, 0);
    }

    // ---- lower slot: VI-writing ops (IADDIU/ISUBIU and the clip readers) --------------------
    //
    // ★★ THE CONSTRAINT THAT GOVERNS ALL OF THESE: VI WRITES AND THE BRANCH DELAY.
    //
    // Unlike LQ (immediate) and SQ (a latency-1 store that commit makes indistinguishable from an
    // immediate one), every VI write goes through `queueViWrite`. Two questions had to be settled
    // before emitting any of them, and both are settled HERE rather than discovered from a
    // wrong-path bug later:
    //
    // 1. **Is an immediate VI write equivalent to the queued one?** YES for these ops, by exactly
    //    the SQ argument. `queueViWrite` is called with `latency = 1` for IADDIU/ISUBIU and for
    //    all four clip readers (`decodeLowerUsage` case 0x08/0x09, 0x10/0x12/0x13, 0x1C: all
    //    `PipelineIalu`, `latency = 1u`), so `readyCycle = m_cycle + 1` and
    //    `commitReadyPipelines()` runs at the TOP of the next pair's iteration — the write lands
    //    before pair i+1 executes either way. Nothing in pair i's own cycle can observe the
    //    difference: a pair has exactly ONE lower instruction, and upper ops never read VI.
    //
    // 2. **The branch-read backup is NOT equivalent, and must be honoured by the caller.** The
    //    interpreter keeps `m_viBranchBackupValue/Reg/Valid`: when an instruction whose usage sets
    //    `delaysNextBranchRead` writes a VI register, a branch in the IMMEDIATELY FOLLOWING pair
    //    reads the OLD value (`readBranchVi`). That is the VU's architectural branch delay.
    //    Since branches TERMINATE blocks, a block's final pair is exactly the pair a branch reads
    //    from — so a JIT'd VI write there that bypasses the backup hands the branch the NEW value
    //    and takes the WRONG PATH.
    //
    //    ⚠ It is not uniform across these ops, so it is reported per-instruction in `LowerPlan`:
    //      * IADDIU / ISUBIU  -> delaysNextBranchRead = TRUE   (decodeLowerUsage case 0x08/0x09)
    //      * FCEQ/FCAND/FCOR/FCGET -> FALSE                    (cases 0x10/0x12/0x13, 0x1C)
    //    Block assembly MUST either exclude a `delaysNextBranchRead` plan from a block's final
    //    pair, or replicate the backup at the block exit. The emitters below deliberately do NOT
    //    write the backup: they take only `VU1State*`, and the backup lives in the interpreter
    //    object, so making it the caller's contract keeps the emitted code free of it.

    // IADDIU / ISUBIU: vi[it] = (int16_t)(vi[is] +/- imm15).
    //   `imm` is the ZERO-extended 15-bit field, so 0..0x7FFF and never negative
    //   (ps2_vu1_lower.cpp: `(int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800)`).
    //   The (int16_t) truncation is architectural (VI registers are 16-bit) and is emitted as
    //   `movsx eax, ax`, matching the int16 -> int32 sign extension of the C++ store.
    //   it == 0 writes nothing: VI0 is hardwired zero and the interpreter guards with `if (it != 0)`.
    // Reading vi[is] straight from the field is exactly what the interpreter does; vi[0] can never
    // hold anything but 0 because every VI write to reg 0 is suppressed at the source.
    inline void emitViAluImm(Emitter &e, uint8_t viDst, uint8_t viSrc, int32_t imm, bool isSub)
    {
        if (viDst == 0u)
            return;
        e.mov_r32_mem(0 /*eax*/, 7 /*rdi*/, kOffVi + int32_t(viSrc) * 4);
        if (imm != 0)
        {
            if (isSub)
                e.sub_eax_imm32(imm);
            else
                e.add_eax_imm32(imm);
        }
        e.movsx_r32_r16(0, 0);
        e.mov_mem_r32(7, kOffVi + int32_t(viDst) * 4, 0);
    }

    enum class ClipOp { Ceq, Cand, Cor, Cget };

    // The clip readers. All four read `m_state.clip` directly and write one VI register; none of
    // them touches the FMAC flag pipeline, and cont.177's lazy-flag scan deliberately EXCLUDES
    // clip readers (only MAC/status readers arm it), so `clip` is always live and these are safe
    // to emit unconditionally.
    //
    // ★ Mirrored bit-for-bit from ps2_vu1_lower.cpp, INCLUDING where a mask is and is not applied:
    //   FCEQ  : vi[1] = ((clip & 0xFFFFFF) == imm24)       -> operand masked
    //   FCAND : vi[1] = ((clip & imm24) != 0)              -> imm24 is already <= 0xFFFFFF
    //   FCOR  : vi[1] = ((clip | imm24) == 0xFFFFFF)       -> NO mask on clip; bits above 24 make
    //                                                         the comparison fail, and replicating
    //                                                         that is the difference between
    //                                                         bit-exact and nearly-right
    //   FCGET : vi[it] = clip & 0xFFF                      -> note 0xFFF, not 0xFFFFFF
    inline void emitClipReader(Emitter &e, ClipOp op, uint32_t imm24, uint8_t viDst)
    {
        if (viDst == 0u)
            return; // only reachable for FCGET with it == 0; FCEQ/FCAND/FCOR target VI1
        e.mov_r32_mem(0 /*eax*/, 7 /*rdi*/, kOffClip);
        switch (op)
        {
        case ClipOp::Ceq:
            e.and_eax_imm32(0xFFFFFFu);
            e.cmp_eax_imm32(imm24);
            e.setcc_r8(0x4, 0); // sete al
            e.movzx_r32_r8(0, 0);
            break;
        case ClipOp::Cand:
            e.and_eax_imm32(imm24); // sets ZF from the result
            e.setcc_r8(0x5, 0);     // setne al
            e.movzx_r32_r8(0, 0);
            break;
        case ClipOp::Cor:
            e.or_eax_imm32(imm24);
            e.cmp_eax_imm32(0xFFFFFFu);
            e.setcc_r8(0x4, 0); // sete al
            e.movzx_r32_r8(0, 0);
            break;
        case ClipOp::Cget:
            e.and_eax_imm32(0xFFFu);
            break;
        }
        e.mov_mem_r32(7, kOffVi + int32_t(viDst) * 4, 0);
    }

    // PS2X_VU1_BLOCKNOMEM=1 disables the cont.200 memory ops inside blocks, so a batch can be
    // A/B'd in ONE binary. Comparing two builds across measurement sessions is not trustworthy --
    // the machine drifts, and the game is real-time driven so a faster run reaches different
    // content at the same cumulative pair count.
    inline bool blockNoMem()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_BLOCKNOMEM"); return e && e[0] && e[0] != '0'; }();
        return v;
    }

    enum class LowerOp { Nop, Lq, Sq, IaddIu, IsubIu, Fceq, Fcand, Fcor, Fcget,
                         // cont.199: the `lower special` block (opHi 0x40)
                         Iadd, Isub, Iaddi, Iand, Ior, Move, Mr32, Mtir, Mfir,
                         // cont.200: integer + auto-index memory access
                         Ilw, Isw, Ilwr, Iswr, Lqi, Sqi, Lqd, Sqd };

    // Blend xmm(src) into vf at `offDst` under the dest lane mask -- the interpreter's applyDest.
    // Clobbers xmm2 and xmm1.
    inline void emitApplyDestXmm(Emitter &e, const CodeBuffer &buf, uint8_t src,
                                 int32_t offDst, uint8_t dest)
    {
        if (dest == 0u)
            return;
        if (dest == 0xFu)
        {
            e.movups_store_d(7, offDst, src);
            return;
        }
        e.movaps_rip(2, buf.destMask(dest));
        e.pand_rr(src, 2);
        e.movups_load_d(1, 7, offDst);
        e.pandn_rr(2, 1);
        e.por_rr(src, 2);
        e.movups_store_d(7, offDst, src);
    }

    // Address from a VI register with NO immediate, ZERO-extended:
    //     addr = ((uint32_t)(uint16_t)vi[reg]) * 16;  addr &= (dataSize - 1)
    // Used by LQI/SQI/LQD/SQD and ILWR/ISWR. (LQ/SQ/ILW/ISW instead sign-extend vi[base] + imm --
    // a real difference, mirrored exactly.) The `addr + 16 <= dataSize` guard is a tautology after
    // the mask, exactly as for LQ (§5h), so no branch is emitted.
    inline void emitVuAddressViOnly(Emitter &e, uint8_t viReg)
    {
        if (viReg == 0u)
            e.mov_r32_imm32(0, 0u);
        else
            e.movzx_r32_mem16(0 /*eax*/, 7 /*rdi*/, kOffVi + int32_t(viReg) * 4);
        e.shl_r32_imm8(0, 4);
        e.and_r32_r32(0, 2 /*edx*/);
    }

    // vi[reg] += delta, truncated to int16 (the post-increment / pre-decrement of LQI/SQI/LQD/SQD).
    inline void emitViStep(Emitter &e, uint8_t viReg, int32_t delta)
    {
        if (viReg == 0u)
            return;
        e.mov_r32_mem(0 /*eax*/, 7, kOffVi + int32_t(viReg) * 4);
        e.add_eax_imm32(delta);
        e.movsx_r32_r16(0, 0);
        e.mov_mem_r32(7, kOffVi + int32_t(viReg) * 4, 0);
    }

    // The single component ILW/ILWR selects from the addressed quadword: the FIRST set dest lane.
    inline uint8_t ilwComponent(uint8_t dest)
    {
        if (dest & 0x8u) return 0u;
        if (dest & 0x4u) return 1u;
        if (dest & 0x2u) return 2u;
        return 3u;
    }

    // ILW / ILWR: vi[it] = (int16_t)(the 16 low bits of the word at addr + comp*4).
    // `eax` must already hold the masked byte address; vuData is in rsi.
    inline void emitIlwTail(Emitter &e, uint8_t viDst, uint8_t comp)
    {
        if (viDst == 0u)
            return;
        e.movsx_r32_mem16_sib(0 /*eax*/, 6 /*rsi*/, 0 /*rax*/, int8_t((comp & 3u) * 4u));
        e.mov_mem_r32(7, kOffVi + int32_t(viDst) * 4, 0);
    }

    // ISW / ISWR: broadcast (uint16_t)vi[it] into four words and store under the dest mask.
    // ★ The VALUE is computed into xmm0 FIRST, because computing the address needs eax.
    inline void emitIswValue(Emitter &e, uint8_t viSrc)
    {
        if (viSrc == 0u)
            e.mov_r32_imm32(0, 0u);
        else
            e.movzx_r32_mem16(0 /*eax*/, 7 /*rdi*/, kOffVi + int32_t(viSrc) * 4);
        e.movd_xmm_r32(0, 0);
        e.pshufd_rri(0, 0, 0x00);
    }

    // Store xmm0 to VU memory at [rsi + eax] under the dest mask (address already in eax).
    inline void emitVuStoreMasked(Emitter &e, const CodeBuffer &buf, uint8_t dest)
    {
        if (dest == 0u)
            return;
        if (dest == 0xFu)
        {
            e.movups_store_sib(6, 0, 0);
            return;
        }
        e.movaps_rip(2, buf.destMask(dest));
        e.pand_rr(0, 2);
        e.movups_load_sib(1, 6, 0);
        e.pandn_rr(2, 1);
        e.por_rr(0, 2);
        e.movups_store_sib(6, 0, 0);
    }

    // Load a quadword from [rsi + eax] and applyDest it into vf[dst] (address already in eax).
    inline void emitVuLoadBlend(Emitter &e, const CodeBuffer &buf, uint8_t vfDst, uint8_t dest)
    {
        if (vfDst == 0u || dest == 0u)
            return;
        e.movups_load_sib(0, 6 /*rsi*/, 0 /*rax*/);
        emitApplyDestXmm(e, buf, 0, kOffVf + int32_t(vfDst) * 16, dest);
    }

    // IADD / ISUB / IAND / IOR: vi[id] = vi[is] OP vi[it].
    // ★ IADD/ISUB truncate to int16; IAND/IOR do NOT (mirrored from ps2_vu1_lower.cpp verbatim).
    inline void emitViAluReg(Emitter &e, uint8_t viDst, uint8_t viA, uint8_t viB,
                             LowerOp op)
    {
        if (viDst == 0u)
            return;
        // ★ ECX IS RESERVED (it carries a branch target across the delay slot), so the second
        // operand is addressed straight out of memory rather than loaded into a scratch register.
        // Using ECX here was a real bug: an IADD in a delay slot clobbered the pending branch
        // target and the block jumped to a VI value instead of an address.
        e.mov_r32_mem(0 /*eax*/, 7, kOffVi + int32_t(viA) * 4);
        const int32_t offB = kOffVi + int32_t(viB) * 4;
        switch (op)
        {
        case LowerOp::Iadd: e.alu_r32_mem(0x03, 0, 7, offB); e.movsx_r32_r16(0, 0); break;
        case LowerOp::Isub: e.alu_r32_mem(0x2B, 0, 7, offB); e.movsx_r32_r16(0, 0); break;
        case LowerOp::Iand: e.alu_r32_mem(0x23, 0, 7, offB); break; // no int16 truncation
        case LowerOp::Ior:  e.alu_r32_mem(0x0B, 0, 7, offB); break; // no int16 truncation
        default: break;
        }
        e.mov_mem_r32(7, kOffVi + int32_t(viDst) * 4, 0);
    }

    // MOVE / MR32: a masked copy of vf[fs] into vf[ft], with MR32 rotating xyzw -> yzwx.
    // Both use applyDest -- no operand clamp, no result clamp, no flags.
    inline void emitMove(Emitter &e, const CodeBuffer &buf, uint8_t vfDst, uint8_t vfSrc,
                         uint8_t dest, bool rotate)
    {
        if (vfDst == 0u || dest == 0u)
            return;
        e.movups_load_d(0, 7, kOffVf + int32_t(vfSrc) * 16);
        if (rotate)
            e.pshufd_rri(0, 0, 0x39); // lanes 1,2,3,0
        emitApplyDestXmm(e, buf, 0, kOffVf + int32_t(vfDst) * 16, dest);
    }

    // MTIR: vi[it] = (int16_t)(bits of vf[fs][comp] & 0xFFFF). Raw bits, not a float convert.
    inline void emitMtir(Emitter &e, uint8_t viDst, uint8_t vfSrc, uint8_t comp)
    {
        if (viDst == 0u)
            return;
        e.movsx_r32_mem16(0, 7, kOffVf + int32_t(vfSrc) * 16 + int32_t(comp & 3u) * 4);
        e.mov_mem_r32(7, kOffVi + int32_t(viDst) * 4, 0);
    }

    // MFIR: broadcast (int16_t)vi[is] as RAW int32 bits into every lane, then applyDest.
    inline void emitMfir(Emitter &e, const CodeBuffer &buf, uint8_t vfDst, uint8_t viSrc,
                         uint8_t dest)
    {
        if (vfDst == 0u || dest == 0u)
            return;
        if (viSrc == 0u)
            e.mov_r32_imm32(0, 0u);
        else
            e.movsx_r32_mem16(0, 7, kOffVi + int32_t(viSrc) * 4);
        e.movd_xmm_r32(0, 0);
        e.pshufd_rri(0, 0, 0x00); // broadcast lane 0
        emitApplyDestXmm(e, buf, 0, kOffVf + int32_t(vfDst) * 16, dest);
    }

    // ---- lower-slot plan: the contract block assembly consumes -------------------------------
    // `classifyLowerPlan` is the lower-slot twin of `classifyUpperPlan`: it decodes one lower
    // instruction word into an emittable plan, or returns false if the block must stop. Beyond the
    // opcode it reports the three facts a block compiler needs and cannot recover from the emitted
    // bytes: which VI register is written, whether the write carries the branch-delay hazard, and
    // whether the instruction touches VU memory (which the XGKICK residual noted for SQ concerns).

    // ---- block terminators: branches (cont.198) --------------------------------------------
    // A branch ENDS a block, but the block still executes it and its DELAY SLOT, then writes
    // m_state.pc. That is what turns a "straight-line run" into a real basic block: without it the
    // interpreter has to run the branch AND its delay slot, which at a branch every ~6.1 pairs is
    // most of what a block could otherwise cover.
    //
    // Every target except JR/JALR is a COMPILE-TIME CONSTANT: `(pc + 8 + imm*8) & pcMask` with pc
    // known at compile time. The condition compares are on `(int16_t)readBranchVi(reg)`.
    enum class BranchKind { None, B, Bal, Jr, Jalr, Ibeq, Ibne, Ibltz, Ibgtz, Iblez, Ibgez };

    struct BranchPlan
    {
        BranchKind kind = BranchKind::None;
        uint8_t viS = 0, viT = 0;   // condition / jump-target registers
        uint8_t linkReg = 0;        // BAL / JALR write (pc+16)/8 here
        uint32_t target = 0;        // taken target (static kinds)
        uint32_t fallthrough = 0;   // not-taken pc == branch pc + 16
        uint32_t linkValue = 0;
    };

    // `pc` is the branch pair's guest PC; `pcMask` is microAddressMask() (0x3FFF for VU1).
    inline bool classifyBranch(uint32_t lower, uint32_t pc, uint32_t pcMask, BranchPlan &b)
    {
        b = BranchPlan{};
        const uint8_t opHi = uint8_t((lower >> 25) & 0x7Fu);
        const int32_t imm = int32_t(IMM11(lower));
        b.target = (pc + 8u + uint32_t(imm * 8)) & pcMask;
        b.fallthrough = pc + 16u;
        b.linkValue = (pc + 16u) / 8u;
        b.viS = VIS(lower);
        b.viT = VIT(lower);
        switch (opHi)
        {
        case 0x20: b.kind = BranchKind::B; return true;
        case 0x21: b.kind = BranchKind::Bal; b.linkReg = VIT(lower); return true;
        case 0x24: b.kind = BranchKind::Jr; return true;
        case 0x25: b.kind = BranchKind::Jalr; b.linkReg = VIT(lower); return true;
        case 0x28: b.kind = BranchKind::Ibeq; return true;
        case 0x29: b.kind = BranchKind::Ibne; return true;
        case 0x2C: b.kind = BranchKind::Ibltz; return true;
        case 0x2D: b.kind = BranchKind::Ibgtz; return true;
        case 0x2E: b.kind = BranchKind::Iblez; return true;
        case 0x2F: b.kind = BranchKind::Ibgez; return true;
        default: return false;
        }
    }

    // Emit the branch's TARGET SELECTION into ecx. The delay slot then runs (no emitter touches
    // ecx), and the block epilogue stores ecx to m_state.pc -- matching the interpreter, which
    // computes the target at the branch pair and applies it after the delay slot.
    //
    // `backupReg`/`backupValid`: when the immediately preceding pair performed a
    // `delaysNextBranchRead` VI write to a register this branch reads, the architectural value is
    // already the NEW one, so the OLD one was stashed in ECX before that write (see planBlock) --
    // and is consumed here. That is the cont.194 branch-delay hazard, honoured inside the block.
    inline void emitBranch(Emitter &e, const BranchPlan &b, uint32_t pcMask,
                           uint8_t backupReg, bool backupValid)
    {
        // Load a condition operand, preferring the stashed pre-write value when it applies.
        const auto loadVi = [&](uint8_t dst, uint8_t reg) {
            if (backupValid && reg == backupReg && reg != 0u)
                e.movsx_r32_r16(dst, 1 /*ecx*/); // stashed by planBlock's backup emission
            else if (reg == 0u)
                e.mov_r32_imm32(dst, 0u);
            else
                e.movsx_r32_mem16(dst, 7 /*rdi*/, kOffVi + int32_t(reg) * 4);
        };

        if (b.linkReg != 0u)
        {
            e.mov_r32_imm32(0 /*eax*/, b.linkValue);
            e.mov_mem_r32(7, kOffVi + int32_t(b.linkReg) * 4, 0);
        }

        switch (b.kind)
        {
        case BranchKind::B:
        case BranchKind::Bal:
            e.mov_r32_imm32(1 /*ecx*/, b.target);
            return;
        case BranchKind::Jr:
        case BranchKind::Jalr:
            // ((uint32_t)(uint16_t)readBranchVi(is) * 8) & pcMask
            if (b.viS == 0u)
                e.mov_r32_imm32(0, 0u);
            else if (backupValid && b.viS == backupReg)
                e.mov_r32_r32(0, 1); // ecx holds the pre-write value; mask to 16 bits next
            else
                e.movzx_r32_mem16(0, 7, kOffVi + int32_t(b.viS) * 4);
            e.and_eax_imm32(0xFFFFu);
            e.shl_r32_imm8(0, 3);
            e.and_eax_imm32(pcMask);
            e.mov_r32_r32(1, 0); // ecx = target
            return;
        default:
            break;
        }

        // Conditional forms: eax = first operand, edx is NOT free (dataSizeMask), so the second
        // operand goes in ecx -- which is safe because the stashed backup has been consumed by
        // then (it is only ever read into eax/ecx here, before ecx becomes the target).
        uint8_t cc = 4;
        switch (b.kind)
        {
        case BranchKind::Ibeq:  cc = 0x4; break;
        case BranchKind::Ibne:  cc = 0x5; break;
        case BranchKind::Ibltz: cc = 0xC; break;
        case BranchKind::Ibgtz: cc = 0xF; break;
        case BranchKind::Iblez: cc = 0xE; break;
        case BranchKind::Ibgez: cc = 0xD; break;
        default: break;
        }
        if (b.kind == BranchKind::Ibeq || b.kind == BranchKind::Ibne)
        {
            loadVi(0 /*eax*/, b.viS);
            // second operand into esi? rsi is vuData. Use a spill-free trick: compare against a
            // freshly loaded value in ecx, then rebuild ecx as the target afterwards.
            loadVi(1 /*ecx*/, b.viT);
            e.cmp_r32_r32(0, 1);
        }
        else
        {
            loadVi(0 /*eax*/, b.viS);
            e.cmp_r32_imm32(0, 0);
        }
        e.mov_r32_imm32(1 /*ecx*/, b.fallthrough);
        e.mov_r32_imm32(0 /*eax*/, b.target);
        e.cmovcc_r32_r32(cc, 1, 0); // ecx = taken ? target : fallthrough
    }

    // Stash vi[reg] (pre-write) into ECX, for the branch-delay hazard.
    inline void emitBranchBackupStash(Emitter &e, uint8_t reg)
    {
        e.movsx_r32_mem16(1 /*ecx*/, 7 /*rdi*/, kOffVi + int32_t(reg) * 4);
    }

    // Block epilogue for a branch-terminated block: m_state.pc = ecx.
    inline void emitBranchCommit(Emitter &e)
    {
        e.mov_mem_r32(7 /*rdi*/, kOffPc, 1 /*ecx*/);
    }

    struct LowerPlan
    {
        LowerOp op = LowerOp::Nop;
        uint8_t viBase = 0; // LQ: vi[is]; SQ: vi[it]
        uint8_t viSrc = 0;  // IADDIU/ISUBIU source
        uint8_t viDst = 0;  // VI register written (0 == none written)
        uint8_t vfReg = 0;  // LQ destination / SQ source / MOVE-MR32-MFIR destination (vf[ft])
        uint8_t vfSrc = 0;  // MOVE / MR32 / MTIR source (vf[fs])
        uint8_t comp = 0;   // MTIR component selector
        uint8_t dest = 0;   // lane mask
        int32_t imm = 0;    // LQ/SQ: IMM11 (signed); IADDIU/ISUBIU: imm15; clip: imm24
        bool writesVi = false;
        bool writesVf = false;
        bool touchesVuMem = false;
        // ★ See the constraint block above. TRUE means a branch in the IMMEDIATELY FOLLOWING pair
        // must still read the OLD value of vi[viDst]; block assembly must exclude such a plan from
        // a block's final pair or replicate `recordViWriteForBranch`.
        bool delaysNextBranchRead = false;
        // ★★ cont.215: the I BIT. When the upper instruction's bit 31 is set the lower word is NOT
        // an instruction at all -- it is a 32-bit float immediate destined for the I register. The
        // interpreter does `execUpper(upper); state.i = normalizeOperand(bit_cast<float>(lower))`.
        // Since the immediate is a CONSTANT, the clamp folds at plan time and the block only needs
        // one store. `planBlock` used to reject these pairs in the same test as the E/D/T HALT bits,
        // which cost 19 of VU0's 39 uncompilable PCs -- but the I bit is not a halt condition.
        bool setsI = false;
        uint32_t iBits = 0; // already through vuNormOperandBits
    };

    inline bool classifyLowerPlan(uint32_t lower, LowerPlan &plan)
    {
        plan = LowerPlan{};
        if (lower == 0x00000000u || lower == 0x8000033Cu) // both NOP encodings
        {
            plan.op = LowerOp::Nop;
            return true;
        }
        const uint8_t opHi = uint8_t((lower >> 25) & 0x7Fu);
        switch (opHi)
        {
        case 0x00: // LQ
            plan.op = LowerOp::Lq;
            plan.viBase = VIS(lower);
            plan.vfReg = FT(lower);
            plan.dest = DEST(lower);
            plan.imm = IMM11(lower);
            plan.writesVf = true;
            plan.touchesVuMem = true;
            return true;
        case 0x01: // SQ
            plan.op = LowerOp::Sq;
            plan.viBase = VIT(lower);
            plan.vfReg = FS(lower);
            plan.dest = DEST(lower);
            plan.imm = IMM11(lower);
            plan.touchesVuMem = true;
            return true;
        case 0x04: // ILW
            if (blockNoMem()) return false;
            plan.op = LowerOp::Ilw;
            plan.viBase = VIS(lower);
            plan.viDst = VIT(lower);
            plan.dest = DEST(lower);
            plan.imm = IMM11(lower);
            plan.writesVi = plan.viDst != 0u;
            plan.touchesVuMem = true;
            return true;
        case 0x05: // ISW
            if (blockNoMem()) return false;
            plan.op = LowerOp::Isw;
            plan.viBase = VIS(lower);
            plan.viSrc = VIT(lower);
            plan.dest = DEST(lower);
            plan.imm = IMM11(lower);
            plan.touchesVuMem = true;
            return true;
        case 0x08: // IADDIU
        case 0x09: // ISUBIU
            plan.op = (opHi == 0x08u) ? LowerOp::IaddIu : LowerOp::IsubIu;
            plan.viSrc = VIS(lower);
            plan.viDst = VIT(lower);
            // Exactly ps2_vu1_lower.cpp's expression, evaluated at compile time. The result is
            // zero-extended (bit 15 is never set), so it is always in 0..0x7FFF.
            plan.imm = int32_t(uint32_t(int32_t(int16_t(uint16_t(
                ((lower & 0x7FFu) | ((lower >> 10) & 0x7800u)) & 0xFFFFu)))));
            plan.writesVi = plan.viDst != 0u;
            plan.delaysNextBranchRead = true; // decodeLowerUsage case 0x08/0x09
            return true;
        case 0x10: // FCEQ
        case 0x12: // FCAND
        case 0x13: // FCOR
            plan.op = (opHi == 0x10u) ? LowerOp::Fceq
                                      : (opHi == 0x12u ? LowerOp::Fcand : LowerOp::Fcor);
            plan.imm = int32_t(lower & 0xFFFFFFu);
            plan.viDst = 1u; // these three always target VI1
            plan.writesVi = true;
            return true;
        case 0x1C: // FCGET
            plan.op = LowerOp::Fcget;
            plan.viDst = VIT(lower);
            plan.writesVi = plan.viDst != 0u;
            return true;
        case 0x40:
        {
            // The `lower special` block. Direct forms first (funct = low 6 bits), then the
            // Dobie-style special decode for 0x3C..0x3F.
            const uint8_t funct = uint8_t(lower & 0x3Fu);
            const uint8_t viD = VID(lower);
            const uint8_t viS2 = VIS(lower);
            const uint8_t viT2 = VIT(lower);
            switch (funct)
            {
            case 0x30: case 0x31: case 0x34: case 0x35:
                plan.op = (funct == 0x30u) ? LowerOp::Iadd
                        : (funct == 0x31u) ? LowerOp::Isub
                        : (funct == 0x34u) ? LowerOp::Iand : LowerOp::Ior;
                plan.viSrc = viS2;
                plan.viBase = viT2; // second operand
                plan.viDst = viD;
                plan.writesVi = viD != 0u;
                plan.delaysNextBranchRead = true;
                return true;
            case 0x32: // IADDI -- sign-extended 5-bit immediate
                plan.op = LowerOp::Iaddi;
                plan.viSrc = viS2;
                plan.viDst = viT2;
                plan.imm = int32_t(int16_t(int32_t((lower >> 6) & 0x1Fu) << 27 >> 27));
                plan.writesVi = viT2 != 0u;
                plan.delaysNextBranchRead = true;
                return true;
            default: break;
            }
            if (funct < 0x3Cu)
                return false;
            const uint8_t special = uint8_t((lower & 3u) | ((lower >> 4) & 0x7Cu));
            switch (special)
            {
            case 0x30: // MOVE
            case 0x31: // MR32
                plan.op = (special == 0x30u) ? LowerOp::Move : LowerOp::Mr32;
                plan.vfSrc = FS(lower);
                plan.vfReg = FT(lower);
                plan.dest = DEST(lower);
                plan.writesVf = plan.vfReg != 0u;
                return true;
            case 0x34: // LQI (post-increment)
            case 0x36: // LQD (pre-decrement)
                if (blockNoMem()) return false;
                plan.op = (special == 0x34u) ? LowerOp::Lqi : LowerOp::Lqd;
                plan.viBase = viS2;   // also the register stepped
                plan.vfReg = FT(lower);
                plan.dest = DEST(lower);
                plan.writesVf = plan.vfReg != 0u;
                plan.writesVi = viS2 != 0u;
                plan.touchesVuMem = true;
                plan.delaysNextBranchRead = true;
                return true;
            case 0x35: // SQI (post-increment)
            case 0x37: // SQD (pre-decrement)
                if (blockNoMem()) return false;
                plan.op = (special == 0x35u) ? LowerOp::Sqi : LowerOp::Sqd;
                plan.viBase = viT2;   // also the register stepped
                plan.vfSrc = FS(lower);
                plan.dest = DEST(lower);
                plan.writesVi = viT2 != 0u;
                plan.touchesVuMem = true;
                plan.delaysNextBranchRead = true;
                return true;
            case 0x3E: // ILWR
                if (blockNoMem()) return false;
                plan.op = LowerOp::Ilwr;
                plan.viBase = viS2;
                plan.viDst = viT2;
                plan.dest = DEST(lower);
                plan.writesVi = viT2 != 0u;
                plan.touchesVuMem = true;
                return true;
            case 0x3F: // ISWR
                if (blockNoMem()) return false;
                plan.op = LowerOp::Iswr;
                plan.viBase = viS2;
                plan.viSrc = viT2;
                plan.dest = DEST(lower);
                plan.touchesVuMem = true;
                return true;
            case 0x3C: // MTIR
                plan.op = LowerOp::Mtir;
                plan.vfSrc = FS(lower);
                plan.comp = uint8_t((lower >> 21) & 3u);
                plan.viDst = viT2;
                plan.writesVi = viT2 != 0u;
                plan.delaysNextBranchRead = true;
                return true;
            case 0x3D: // MFIR
                plan.op = LowerOp::Mfir;
                plan.viSrc = viS2;
                plan.vfReg = FT(lower);
                plan.dest = DEST(lower);
                plan.writesVf = plan.vfReg != 0u;
                return true;
            default:
                return false;
            }
        }
        default:
            return false; // branch/jump (block terminator), ILW/ISW, anything else: not covered
        }
    }

    // Emit a classified lower plan. Calling convention as for LQ/SQ:
    //     void f(VU1State *rdi, uint8_t *vuData rsi, uint32_t dataSizeMask edx)
    // (the VI-only ops ignore rsi/edx, but sharing one signature keeps block assembly uniform).
    inline void emitLowerPlan(Emitter &e, const CodeBuffer &buf, const LowerPlan &plan)
    {
        // cont.215: I-bit pair. The upper has already been emitted, so it read the OLD I -- which
        // is the required order -- and this store makes the new value visible to later pairs.
        if (plan.setsI)
        {
            e.mov_mem_imm32(7 /*rdi*/, kOffI, plan.iBits);
            return;
        }
        switch (plan.op)
        {
        case LowerOp::Nop:
            return;
        case LowerOp::Lq:
            emitLQ(e, buf, plan.viBase, plan.vfReg, plan.imm, plan.dest);
            return;
        case LowerOp::Sq:
            emitSQ(e, buf, plan.viBase, plan.vfReg, plan.imm, plan.dest);
            return;
        case LowerOp::IaddIu:
            emitViAluImm(e, plan.viDst, plan.viSrc, plan.imm, false);
            return;
        case LowerOp::IsubIu:
            emitViAluImm(e, plan.viDst, plan.viSrc, plan.imm, true);
            return;
        case LowerOp::Fceq:
            emitClipReader(e, ClipOp::Ceq, uint32_t(plan.imm), plan.viDst);
            return;
        case LowerOp::Fcand:
            emitClipReader(e, ClipOp::Cand, uint32_t(plan.imm), plan.viDst);
            return;
        case LowerOp::Fcor:
            emitClipReader(e, ClipOp::Cor, uint32_t(plan.imm), plan.viDst);
            return;
        case LowerOp::Fcget:
            emitClipReader(e, ClipOp::Cget, 0u, plan.viDst);
            return;
        case LowerOp::Iadd:
        case LowerOp::Isub:
        case LowerOp::Iand:
        case LowerOp::Ior:
            emitViAluReg(e, plan.viDst, plan.viSrc, plan.viBase, plan.op);
            return;
        case LowerOp::Iaddi:
            emitViAluImm(e, plan.viDst, plan.viSrc, plan.imm, false);
            return;
        case LowerOp::Move:
            emitMove(e, buf, plan.vfReg, plan.vfSrc, plan.dest, false);
            return;
        case LowerOp::Mr32:
            emitMove(e, buf, plan.vfReg, plan.vfSrc, plan.dest, true);
            return;
        case LowerOp::Mtir:
            emitMtir(e, plan.viDst, plan.vfSrc, plan.comp);
            return;
        case LowerOp::Mfir:
            emitMfir(e, buf, plan.vfReg, plan.viSrc, plan.dest);
            return;
        case LowerOp::Ilw:
            emitVuAddress(e, plan.viBase, plan.imm);
            emitIlwTail(e, plan.viDst, ilwComponent(plan.dest));
            return;
        case LowerOp::Ilwr:
            emitVuAddressViOnly(e, plan.viBase);
            emitIlwTail(e, plan.viDst, ilwComponent(plan.dest));
            return;
        case LowerOp::Isw:
            emitIswValue(e, plan.viSrc);          // value first: the address needs eax
            emitVuAddress(e, plan.viBase, plan.imm);
            emitVuStoreMasked(e, buf, plan.dest);
            return;
        case LowerOp::Iswr:
            emitIswValue(e, plan.viSrc);
            emitVuAddressViOnly(e, plan.viBase);
            emitVuStoreMasked(e, buf, plan.dest);
            return;
        case LowerOp::Lqi:
            emitVuAddressViOnly(e, plan.viBase);
            emitVuLoadBlend(e, buf, plan.vfReg, plan.dest);
            emitViStep(e, plan.viBase, 1);        // post-increment
            return;
        case LowerOp::Lqd:
            emitViStep(e, plan.viBase, -1);       // pre-decrement
            emitVuAddressViOnly(e, plan.viBase);
            emitVuLoadBlend(e, buf, plan.vfReg, plan.dest);
            return;
        case LowerOp::Sqi:
            e.movups_load_d(0, 7, kOffVf + int32_t(plan.vfSrc) * 16);
            emitVuAddressViOnly(e, plan.viBase);
            emitVuStoreMasked(e, buf, plan.dest);
            emitViStep(e, plan.viBase, 1);
            return;
        case LowerOp::Sqd:
            e.movups_load_d(0, 7, kOffVf + int32_t(plan.vfSrc) * 16);
            emitViStep(e, plan.viBase, -1);
            emitVuAddressViOnly(e, plan.viBase);
            emitVuStoreMasked(e, buf, plan.dest);
            return;
        }
    }

    // ---- instruction compiler + cache ------------------------------------------------------
    // First integration slice: compile the UPPER instruction of a pair into a native function
    // `void f(VU1State*)` and call it in place of execUpper's body. Deliberately conservative —
    // it replicates exactly what execUpper + applyFmacDest(Acc) do for the covered ops (operand
    // clamp, broadcast, arithmetic, result clamp, dest-lane blend, store), so every other piece of
    // the interpreter — the write pipeline, the shadow dance in run(), cycle accounting, the lower
    // instruction — is untouched and keeps working. Anything not covered returns null and falls
    // back to the interpreter.
    //
    // The compiled code depends ONLY on the 32-bit instruction word, so the cache is valid across
    // programs and code uploads and never needs invalidating.
    //
    // ★ Only valid when lazy flags is active for the program: the emitted code deliberately does
    // not derive MAC/status flags, which cont.177 proved unobservable for those programs.


    struct Cache
    {
        static constexpr uint32_t kSlots = 8192; // power of two, open addressing
        uint32_t key[kSlots] = {};
        void *fn[kSlots] = {};
        bool used[kSlots] = {};
        unsigned long hits = 0, misses = 0, compiled = 0, rejected = 0;

        static uint32_t hash(uint32_t k)
        {
            k ^= k >> 16; k *= 0x7feb352du; k ^= k >> 15; k *= 0x846ca68bu; k ^= k >> 16;
            return k;
        }
        // Returns the slot for `k`; `found` says whether it already holds an entry for it.
        uint32_t slotFor(uint32_t k, bool &found) const
        {
            uint32_t i = hash(k) & (kSlots - 1);
            for (uint32_t probe = 0; probe < 64u; ++probe)
            {
                if (!used[i]) { found = false; return i; }
                if (key[i] == k) { found = true; return i; }
                i = (i + 1u) & (kSlots - 1);
            }
            found = false;
            return kSlots; // table too full at this chain
        }
    };

    inline bool ftoiTrapArmed()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_FTOITRAP"); return e && e[0] && e[0] != '0'; }();
        return v;
    }

    // Decode an upper instruction into a plan. Returns false if not covered.
    // ★ The main and `special` switches DIVERGE in the 0x10-0x1F range — main 0x10-0x17 are
    // MAXbc/MINIbc while special 0x10-0x17 are ITOF/FTOI — so the two are classified separately
    // rather than through one shared table.
    inline bool classifyUpperPlan(uint32_t instr, UpperPlan &plan)
    {
        const uint8_t o = uint8_t(instr & 0x3Fu);
        const bool special = (o >= 0x3Cu);
        const uint8_t sel = special ? uint8_t((instr & 3u) | ((instr >> 4) & 0x7Cu)) : o;
        plan = UpperPlan{};
        plan.writesAcc = special;

        auto arith = [&](FmacOp op, OperandSrc src, int bc, bool clamp, bool readsAcc) {
            plan.op = op; plan.src = src; plan.bc = bc;
            plan.clampResult = clamp; plan.readsAcc = readsAcc;
            return true;
        };

        // Shared between both switches: the bc arithmetic block and the *q/i and quad forms.
        if (sel <= 0x0Fu)
        {
            const int bc = int(sel & 3u);
            switch (sel >> 2)
            {
            case 0: return arith(FmacOp::Add, OperandSrc::VtLane, bc, true, false);  // ADD(A)bc
            case 1: return arith(FmacOp::Sub, OperandSrc::VtLane, bc, true, false);  // SUB(A)bc
            case 2: return arith(FmacOp::Madd, OperandSrc::VtLane, bc, true, true);  // MADD(A)bc
            case 3: return arith(FmacOp::Msub, OperandSrc::VtLane, bc, true, true);  // MSUB(A)bc
            }
            return false;
        }
        if (sel >= 0x18u && sel <= 0x1Bu)
            return arith(FmacOp::Mul, OperandSrc::VtLane, int(sel & 3u), true, false); // MUL(A)bc

        if (!special)
        {
            switch (sel)
            {
            case 0x10: case 0x11: case 0x12: case 0x13: // MAXbc — applyDest, no clamp
                return arith(FmacOp::Max, OperandSrc::VtLane, int(sel & 3u), false, false);
            case 0x14: case 0x15: case 0x16: case 0x17: // MINIbc — applyDest, no clamp
                return arith(FmacOp::Min, OperandSrc::VtLane, int(sel & 3u), false, false);
            case 0x1C: return arith(FmacOp::Mul, OperandSrc::ScalarQ, 0, true, false);  // MULq
            case 0x1D: return arith(FmacOp::Max, OperandSrc::ScalarI, 0, false, false); // MAXi
            case 0x1E: return arith(FmacOp::Mul, OperandSrc::ScalarI, 0, true, false);  // MULi
            case 0x1F: return arith(FmacOp::Min, OperandSrc::ScalarI, 0, false, false); // MINIi
            case 0x20: return arith(FmacOp::Add, OperandSrc::ScalarQ, 0, true, false);  // ADDq
            case 0x21: return arith(FmacOp::Madd, OperandSrc::ScalarQ, 0, true, true);  // MADDq
            case 0x22: return arith(FmacOp::Add, OperandSrc::ScalarI, 0, true, false);  // ADDi
            case 0x23: return arith(FmacOp::Madd, OperandSrc::ScalarI, 0, true, true);  // MADDi
            case 0x24: return arith(FmacOp::Sub, OperandSrc::ScalarQ, 0, true, false);  // SUBq
            case 0x25: return arith(FmacOp::Msub, OperandSrc::ScalarQ, 0, true, true);  // MSUBq
            case 0x26: return arith(FmacOp::Sub, OperandSrc::ScalarI, 0, true, false);  // SUBi
            case 0x27: return arith(FmacOp::Msub, OperandSrc::ScalarI, 0, true, true);  // MSUBi
            case 0x28: return arith(FmacOp::Add, OperandSrc::VtQuad, 0, true, false);   // ADD
            case 0x29: return arith(FmacOp::Madd, OperandSrc::VtQuad, 0, true, true);   // MADD
            case 0x2A: return arith(FmacOp::Mul, OperandSrc::VtQuad, 0, true, false);   // MUL
            case 0x2B: return arith(FmacOp::Max, OperandSrc::VtQuad, 0, false, false);  // MAX
            case 0x2C: return arith(FmacOp::Sub, OperandSrc::VtQuad, 0, true, false);   // SUB
            case 0x2D: return arith(FmacOp::Msub, OperandSrc::VtQuad, 0, true, true);   // MSUB
            case 0x2F: return arith(FmacOp::Min, OperandSrc::VtQuad, 0, false, false);  // MINI
            default: return false; // 0x2E OPMSUB and everything else: not covered
            }
        }

        switch (sel) // special (A) forms — 0x10-0x17 are ITOF/FTOI here, NOT MAX/MINI
        {
        case 0x1C: return arith(FmacOp::Mul, OperandSrc::ScalarQ, 0, true, false);  // MULAq
        case 0x1D:                                                                   // ABS
            plan.writesFt = true;
            plan.writesAcc = false; // ABS writes vf[ft], not ACC
            return arith(FmacOp::Abs, OperandSrc::None, 0, false, false);
        case 0x1E: return arith(FmacOp::Mul, OperandSrc::ScalarI, 0, true, false);  // MULAi
        case 0x20: return arith(FmacOp::Add, OperandSrc::ScalarQ, 0, true, false);  // ADDAq
        case 0x21: return arith(FmacOp::Madd, OperandSrc::ScalarQ, 0, true, true);  // MADDAq
        case 0x22: return arith(FmacOp::Add, OperandSrc::ScalarI, 0, true, false);  // ADDAi
        case 0x23: return arith(FmacOp::Madd, OperandSrc::ScalarI, 0, true, true);  // MADDAi
        case 0x24: return arith(FmacOp::Sub, OperandSrc::ScalarQ, 0, true, false);  // SUBAq
        case 0x25: return arith(FmacOp::Msub, OperandSrc::ScalarQ, 0, true, true);  // MSUBAq
        case 0x26: return arith(FmacOp::Sub, OperandSrc::ScalarI, 0, true, false);  // SUBAi
        case 0x27: return arith(FmacOp::Msub, OperandSrc::ScalarI, 0, true, true);  // MSUBAi
        case 0x28: return arith(FmacOp::Add, OperandSrc::VtQuad, 0, true, false);   // ADDA
        case 0x29: return arith(FmacOp::Madd, OperandSrc::VtQuad, 0, true, true);   // MADDA
        case 0x2A: return arith(FmacOp::Mul, OperandSrc::VtQuad, 0, true, false);   // MULA
        case 0x2C: return arith(FmacOp::Sub, OperandSrc::VtQuad, 0, true, false);   // SUBA
        case 0x2D: return arith(FmacOp::Msub, OperandSrc::VtQuad, 0, true, true);   // MSUBA
        case 0x14: case 0x15: case 0x16: case 0x17: // FTOI0 / FTOI4 / FTOI12 / FTOI15
            // PS2X_VU1_FTOITRAP prints a diagnostic from the interpreter on every saturating FTOI;
            // the emitted code cannot, so leave FTOI to the interpreter whenever the trap is armed.
            if (ftoiTrapArmed())
                return false;
            plan.writesFt = true;
            plan.writesAcc = false;
            plan.itofShift = uint8_t(sel & 3u);
            return arith(FmacOp::Ftoi, OperandSrc::None, 0, false, false);
        case 0x10: case 0x11: case 0x12: case 0x13: // ITOF0 / ITOF4 / ITOF12 / ITOF15
            plan.writesFt = true;
            plan.writesAcc = false;
            plan.itofShift = uint8_t(sel & 3u);
            return arith(FmacOp::Itof, OperandSrc::None, 0, false, false);
        case 0x1F: // CLIP -- writes only the clip register (via the flag pipeline)
            plan.writesAcc = false;
            return arith(FmacOp::Clip, OperandSrc::None, 0, false, false);
        case 0x2F:
        case 0x30:
            // ★ Upper NOP. It emits NOTHING, but it must count as COVERED: it is 26.4% of executed
            // pairs (cont.178 census), so treating it as an opcode gap would break almost every
            // run a block could otherwise cover.
            plan.writesAcc = false;
            return arith(FmacOp::Nop, OperandSrc::None, 0, false, false);
        default: return false; // ITOF/FTOI/CLIP/OPMULA: not covered yet
        }
    }

    // Back-compat shim for the census, which only asks "is this upper covered?".
    inline bool classifyUpper(uint32_t instr, FmacOp &op, int &bc, bool &writesAcc)
    {
        UpperPlan plan;
        if (!classifyUpperPlan(instr, plan))
            return false;
        op = plan.op; bc = plan.bc; writesAcc = plan.writesAcc;
        return true;
    }

    typedef void (*UpperFn)(void *state);

    inline UpperFn compileUpper(CodeBuffer &buf, Cache &cache, uint32_t instr)
    {
        bool found = false;
        const uint32_t slot = cache.slotFor(instr, found);
        if (slot >= Cache::kSlots)
            return nullptr;
        if (found)
        {
            ++cache.hits;
            return reinterpret_cast<UpperFn>(cache.fn[slot]);
        }
        ++cache.misses;

        UpperPlan plan;
        UpperFn produced = nullptr;
        if (classifyUpperPlan(instr, plan) && plan.op != FmacOp::Nop && buf.room() > 512)
        {
            const uint8_t dest = uint8_t((instr >> 21) & 0xFu);
            const uint8_t ft = uint8_t((instr >> 16) & 0x1Fu);
            const uint8_t fs = uint8_t((instr >> 11) & 0x1Fu);
            const uint8_t fd = uint8_t((instr >> 6) & 0x1Fu);
            const int32_t offDst = plan.writesAcc ? kOffAcc
                                                  : (kOffVf + int32_t(plan.writesFt ? ft : fd) * 16);
            Emitter e;
            e.p = buf.code + buf.used;
            e.end = buf.base + buf.capacity;
            uint8_t *const start = e.p;
            emitUpper(e, buf, plan,
                      kOffVf + int32_t(fs) * 16,
                      kOffVf + int32_t(ft) * 16,
                      kOffAcc, kOffQ, kOffI, offDst, dest);
            e.ret();
            if (!e.overflow)
            {
                buf.used = size_t(e.p - buf.code);
                produced = reinterpret_cast<UpperFn>(start);
                ++cache.compiled;
            }
        }
        else
        {
            ++cache.rejected;
        }
        cache.used[slot] = true;
        cache.key[slot] = instr;
        cache.fn[slot] = reinterpret_cast<void *>(produced); // null = "known uncovered", cached too
        return produced;
    }


    // ---- BLOCK ASSEMBLY (cont.195) ----------------------------------------------------------
    //
    // A block is a run of consecutive pairs emitted as ONE native function, replacing the
    // interpreter's inner loop for that run. This is where the measured 15.6x (§5d) is cashed in:
    // one call instead of N, operands straight out of VU1State, and — the big one — **no write
    // pipeline and no shadow dance inside the block**.
    //
    // ★★ WHY IMMEDIATE WRITES ARE VALUE-EQUIVALENT (the load-bearing argument).
    // The interpreter defers every write through queueVfWrite/queueAccWrite/queueViWrite and only
    // makes it architecturally visible at `readyCycle`. It would therefore seem a block must model
    // the pipeline. It does not, because **a reader always stalls until its operand is ready**:
    // `calculatePairReadyCycle` takes `max` over `m_vfReady[reg][lane]` for EVERY read lane (and
    // `m_viReady` / `m_accReady`), and `run()` then does `while (readyCycle > m_cycle) advanceTo(...)`,
    // which advances m_cycle one cycle at a time calling `commitReadyPipelines()`. So by the time a
    // pair issues, every value it reads has been committed. **A stale read cannot happen**, and
    // applying the write immediately therefore yields the same VALUE the interpreter would read.
    // The pipeline governs *timing*, not visibility — so a block reproduces the timing separately
    // (cycle accounting + ready tables, done in C++ after the call) and skips the machinery.
    //   - WAW is consistent too: with immediate writes the last writer wins, and the interpreter
    //     reaches the same place via the `m_vfLatestWrite[reg][lane] == write.sequence`
    //     supersession check, which drops the older pending write.
    //   - Per-lane granularity is preserved: `m_vfReady` and the supersession check are both
    //     per lane, and so are the emitted dest-mask blends.
    //
    // ★★ INTRA-PAIR ORDERING. Within a pair the interpreter runs execUpper THEN execLower, with a
    // shadow (`upperVfShadowReg`) that hides the upper's VF write from the lower. Rather than
    // replicate the shadow, the emitter EXPLOITS the orthogonality of the covered opcodes:
    //   * SQ / IADDIU / ISUBIU / clip readers are emitted BEFORE the upper. SQ reads VF (so it must
    //     see the pre-pair value — emitting it first gives exactly that); the VI/clip ops touch
    //     neither VF nor ACC, and the upper touches neither VI nor clip, so they are order-free.
    //   * LQ is emitted AFTER the upper, because the upper must see the pre-pair value of LQ's
    //     destination (the interpreter runs execUpper first).
    //   * When LQ's destination IS the upper's destination the interpreter SUPPRESSES the lower
    //     write entirely (`suppressedLowerVf`: execLower writes it, then the shadow restore
    //     overwrites it with the upper's result and it is never queued). LQ has no other effect —
    //     its bounds guard is a tautology (§5h) — so the emitter skips it outright.
    // With that ordering no scratch register and no cut is needed: every classified pair compiles.
    //
    // ★ Writes to vf[0] / vi[0] are skipped. The interpreter performs them and then restores
    // vf[0] = (0,0,0,1) and vi[0] = 0 at the end of every pair, and nothing reads them in between
    // (the lower has already run), so not writing them at all is equivalent.
    //
    // ENTRY GUARD (evaluated in C++; see ps2_vu1_core.cpp). The block is only entered when the
    // write pipelines are EMPTY and XGKICK is idle. That single condition buys three things:
    //   1. no incoming pending write can commit mid-block (the block queues nothing, so the
    //      pipelines stay empty for its whole duration);
    //   2. no incoming stall is possible — a ready-table entry equals its write's readyCycle, and
    //      an empty pipeline means every such write has committed, so every entry is <= m_cycle;
    //   3. no supersession hazard — there is no older pending write for a block write to displace,
    //      so `m_vfLatestWrite` needs no maintenance.
    // Blocks are additionally CUT before any pair that would stall on an earlier pair of the same
    // block, so a block never stalls internally and `m_cycle += pairs` is exact.

    // Per-pair facts the block compiler needs. The interpreter's DecodedInstructionPair is private,
    // so the caller flattens it into this POD — which also keeps the JIT decoupled from it.
    struct PairInfo
    {
        uint32_t upper = 0, lower = 0;
        uint8_t upperWriteReg = 0, upperWriteLanes = 0, upperLatency = 0;
        uint8_t lowerWriteReg = 0, lowerWriteLanes = 0, lowerLatency = 0;
        uint8_t accWrite = 0;      // lane mask
        uint16_t viWriteMask = 0;
        uint8_t viLatency = 0;
        uint8_t suppressedLowerVf = 0;
        // reads, used only for the compile-time stall cut
        uint8_t vfReadReg[4] = {}, vfReadLanes[4] = {};
        uint8_t vfReadCount = 0;
        uint16_t viReadMask = 0;
        uint8_t accRead = 0;
        bool iBit = false, eBit = false, dBit = false, tBit = false;
        bool lowerDelaysBranchRead = false;
    };

    // One ready-table update the C++ side replays after the call, in pair order (the interpreter's
    // markPairWrites SETS rather than maxes, so replaying in order reproduces it exactly).
    struct ReadyWrite
    {
        uint8_t kind;    // 0 = vf, 1 = vi, 2 = acc
        uint8_t reg;     // vf/vi register (unused for acc)
        uint8_t lanes;   // ★ cont.212: LANE MASK (bit b = lane 3-b) for vf/acc; one record per
                         // write-group instead of one per lane. Unused for vi.
        uint8_t latency;
        uint8_t pair;    // index of the writing pair within the block; its issue cycle comes from
                         // the schedule (static in planBlock, re-derived at entry when incoming
                         // pending writes push it -- see cont.196)
    };

    // Per-pair reads + static issue cycle, kept so the driver can RE-DERIVE the schedule at entry
    // when an incoming pending write delays one of the block's reads (cont.196).
    struct PairSched
    {
        uint8_t vfReadReg[4] = {}, vfReadLanes[4] = {};
        uint8_t vfReadCount = 0;
        uint8_t accRead = 0;
        uint16_t viRead = 0;
        uint16_t issue = 0; // static issue cycle relative to block entry
    };

    // ★ cont.197: 8, not 32. Mean block is 3.5 pairs, and `Block` is indexed per entry --
    // sizing its arrays for 32 made it ~2.2 KB, so every block entry touched many cold
    // cache lines. Longer runs are simply truncated, not rejected. (§5d also measured long
    // blocks as slightly WORSE: 256 pairs overflows L1i.)
    static constexpr uint32_t kMaxBlockPairs = 8;
    static constexpr uint32_t kMaxReadyWrites = kMaxBlockPairs * 9;

    typedef void (*BlockFn)(void *state, void *vuData, uint32_t dataSizeMask);

    struct Block
    {
        BlockFn fn = nullptr;
        uint32_t startPc = 0;
        uint32_t pairs = 0;
        uint32_t cycles = 0;  // ★ cycles the block consumes = pairs + absorbed stall cycles
        uint32_t readyCount = 0;
        bool touchesVuMem = false;   // any SQ or LQ -> pending stores matter
        uint8_t tailViBackupReg = 0; // see the branch-delay contract in planBlock()
        bool endsWithBranch = false; // the block writes m_state.pc itself
        uint8_t stopReason = 0;      // cont.213: why the block stopped growing (see planBlock)
        uint32_t stopLower = 0;      // cont.230: the lower word that stopped it (reason 5/7/2 diagnosis)
        // ★ cont.195b: what the block TOUCHES, so the entry guard can be precise instead of
        // demanding globally-empty pipelines. A pending incoming write only matters if the block
        // reads or writes the very slot it targets (read -> the interpreter would have stalled
        // for it; write -> it would clobber the block's newer value on a later commit).
        uint8_t vfTouched[32] = {}; // per-register lane mask, reads and writes together
        PairSched sched[kMaxBlockPairs];
        uint16_t viTouched = 0;
        uint8_t accTouched = 0;
        bool readsQ = false;
        bool readsClip = false;
        // ★ Issue cycle of the EARLIEST pair that reads clip / Q, relative to block entry
        // (0xFFFF = none). A clip or Q read does NOT stall -- `calculatePairReadyCycle` ignores
        // both -- so their visibility is pure timing, and a pending write may be applied early
        // only if it would have committed before this pair anyway.
        uint16_t firstClipIssue = 0xFFFFu;
        uint16_t firstQIssue = 0xFFFFu;
        // FMACs whose MAC flags the driver must replay (only when the block was compiled in
        // flag-emitting mode). `slot` indexes the code buffer's result scratch.
        struct FlagWrite { uint8_t pair; uint8_t dest; uint8_t slot; };
        FlagWrite flagWrites[kMaxBlockPairs];
        uint8_t flagCount = 0;
        bool emitsFlags = false;
        // CLIP pairs whose queueClip the driver must replay (slot indexes a PAIR of scratch
        // quads: vf[fs] then vf[ft]).
        struct ClipWrite { uint8_t pair; uint8_t slot; };
        ClipWrite clipWrites[kMaxBlockPairs];
        uint8_t clipCount = 0;
        bool writesVuMem = false;   // SQ only
        ReadyWrite ready[kMaxReadyWrites];
    };

    // Emit one pair. Returns false only on emitter overflow.
    inline void emitPairInto(Emitter &e, const CodeBuffer &buf, const PairInfo &pi,
                             const UpperPlan &up, const LowerPlan &lo,
                             const void *resultSink = nullptr)
    {
        const uint8_t dest = uint8_t((pi.upper >> 21) & 0xFu);
        const uint8_t ft = uint8_t((pi.upper >> 16) & 0x1Fu);
        const uint8_t fs = uint8_t((pi.upper >> 11) & 0x1Fu);
        const uint8_t fd = uint8_t((pi.upper >> 6) & 0x1Fu);
        const uint8_t upperDstReg = up.writesFt ? ft : fd;

        // (a) everything that does not write VF goes BEFORE the upper (see the ordering note).
        // LQ / MOVE / MR32 / MFIR write vf and must therefore come AFTER it, so the upper still
        // sees the pre-pair value of their destination -- exactly the interpreter's order.
        const bool writesVfLower = (lo.op == LowerOp::Lq || lo.op == LowerOp::Move ||
                                    lo.op == LowerOp::Mr32 || lo.op == LowerOp::Mfir ||
                                    lo.op == LowerOp::Lqi || lo.op == LowerOp::Lqd);
        if (lo.op != LowerOp::Nop && !writesVfLower && !lo.setsI)
            emitLowerPlan(e, buf, lo);

        // (b) the upper. Skipped when it would only write vf[0], which the interpreter restores
        // at the end of the pair anyway (and under lazy flags it has no other effect).
        if (up.op != FmacOp::Nop && (up.writesAcc || upperDstReg != 0u))
        {
            const int32_t offDst =
                up.writesAcc ? kOffAcc : (kOffVf + int32_t(upperDstReg) * 16);
            emitUpper(e, buf, up, kOffVf + int32_t(fs) * 16, kOffVf + int32_t(ft) * 16,
                      kOffAcc, kOffQ, kOffI, offDst, dest, resultSink);
        }

        // ★ cont.215: the I-bit store goes AFTER the upper. The interpreter's order is
        // `execUpper(upper); state.i = normalizeOperand(imm)`, so an upper that READS I in this
        // same pair must still see the OLD value. (Emitting it in group (a) meant it was never
        // emitted at all -- `lo.op` is Nop for an I-bit pair -- which is what the 1.7M-mismatch
        // shadow-verify caught: later pairs read a stale I.)
        if (lo.setsI)
            emitLowerPlan(e, buf, lo);

        // (c) the VF-writing lowers last, and only when the interpreter would not have
        // suppressed them (destination == the upper's destination -> the write is discarded).
        if (writesVfLower)
        {
            const bool suppressed =
                (pi.suppressedLowerVf != 0u && pi.suppressedLowerVf == lo.vfReg);
            if (!suppressed && lo.vfReg != 0u)
            {
                emitLowerPlan(e, buf, lo);
            }
            else if (lo.op == LowerOp::Lqi || lo.op == LowerOp::Lqd)
            {
                // ★ The suppressed VF write is discarded, but the VI step is NOT -- the
                // interpreter performs it regardless. LQ/MOVE/MR32/MFIR have no such side effect.
                emitViStep(e, lo.viBase, lo.op == LowerOp::Lqi ? 1 : -1);
            }
        }
    }

    // Per-pair helpers, shared by the straight-line path and the branch terminator below.

    // Raise `issue` to satisfy every read of this pair against the block-local ready tables.
    inline void applyReadStalls(const PairInfo &pi, int32_t &issue,
                                const int32_t vfReady[32][4], const int32_t viReady[16],
                                const int32_t accReady[4])
    {
        for (uint32_t k = 0; k < pi.vfReadCount && k < 4u; ++k)
        {
            uint32_t lanes = pi.vfReadLanes[k] & 0xFu;
            const uint8_t reg = pi.vfReadReg[k] & 31u;
            while (lanes != 0u)
            {
                const uint32_t bit = uint32_t(__builtin_ctz(lanes));
                lanes &= lanes - 1u;
                const int32_t r = vfReady[reg][3u - bit];
                if (r > issue) issue = r;
            }
        }
        uint32_t vr = pi.viReadMask & 0xFFFEu; // VI0 is hardwired, never pending
        while (vr != 0u)
        {
            const uint32_t reg = uint32_t(__builtin_ctz(vr));
            vr &= vr - 1u;
            if (viReady[reg] > issue) issue = viReady[reg];
        }
        uint32_t ar = pi.accRead & 0xFu;
        while (ar != 0u)
        {
            const uint32_t bit = uint32_t(__builtin_ctz(ar));
            ar &= ar - 1u;
            if (accReady[3u - bit] > issue) issue = accReady[3u - bit];
        }
    }

    // PS2X_VU1_QUADREADY (default ON; =0 emits the pre-cont.212 per-lane records) -- the kill
    // switch for the quad coalescing below, and the A/B baseline for its measurement.
    inline bool quadReadyEnabled()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_QUADREADY"); return !(e && e[0] == '0'); }();
        return v;
    }

    // One vf write-group -> the block-local ready table + either ONE masked record (coalesced) or
    // the pre-cont.212 per-lane records. Bit b of the mask is lane 3-b, matching markPairWrites.
    inline void emitVfWrites(uint8_t reg, uint32_t lanes, uint8_t lat, int32_t issue, bool coalesce,
                             ReadyWrite *ready, uint32_t &readyCount, int32_t vfReady[32][4],
                             uint32_t pairIndex)
    {
        if (lanes == 0u)
            return;
        const int32_t rc = issue + int32_t(lat);
        uint32_t m = lanes;
        while (m != 0u)
        {
            const uint32_t bit = uint32_t(__builtin_ctz(m));
            m &= m - 1u;
            vfReady[reg][3u - bit] = rc;
            if (!coalesce)
                ready[readyCount++] = ReadyWrite{0u, reg, uint8_t(1u << bit), lat,
                                                 uint8_t(pairIndex)};
        }
        if (coalesce)
            ready[readyCount++] = ReadyWrite{0u, reg, uint8_t(lanes), lat, uint8_t(pairIndex)};
    }

    // Record this pair's writes into the block-local ready tables AND the replay list, exactly as
    // markPairWrites would (it SETS rather than maxes, so replaying in order reproduces it).
    inline void recordPairWrites(const PairInfo &pi, int32_t issue,
                                 ReadyWrite *ready, uint32_t &readyCount,
                                 int32_t vfReady[32][4], int32_t viReady[16], int32_t accReady[4],
                                 uint32_t pairIndex)
    {
        // ★★ cont.212: ONE record per WRITE-GROUP, carrying a lane mask -- not one per lane.
        // cont.211's segment split showed `retire` (the largest driver segment) is essentially all
        // ready-table replay, walking 16.49 records per entry over a mean 4.04 pairs. Those records
        // differ only in LANE and write the SAME value into CONTIGUOUS uint64 slots (`m_vfReady` is
        // std::array<std::array<uint64_t,4>,32>, `m_accReady` is std::array<uint64_t,4>), so
        // collapsing a group is exact: same values, same order, one loop trip instead of four.
        // PS2X_VU1_QUADREADY=0 emits the pre-cont.212 per-lane records (A/B + kill switch).
        const bool coalesce = quadReadyEnabled();
        if (pi.lowerWriteReg != 0u && pi.suppressedLowerVf != pi.lowerWriteReg)
            emitVfWrites(pi.lowerWriteReg, pi.lowerWriteLanes & 0xFu, pi.lowerLatency, issue,
                         coalesce, ready, readyCount, vfReady, pairIndex);
        if (pi.upperWriteReg != 0u)
            emitVfWrites(pi.upperWriteReg, pi.upperWriteLanes & 0xFu, pi.upperLatency, issue,
                         coalesce, ready, readyCount, vfReady, pairIndex);
        uint32_t vw = pi.viWriteMask & 0xFFFEu;
        while (vw != 0u)
        {
            const uint32_t reg = uint32_t(__builtin_ctz(vw));
            vw &= vw - 1u;
            viReady[reg] = issue + int32_t(pi.viLatency);
            ready[readyCount++] =
                ReadyWrite{1u, uint8_t(reg), 0u, pi.viLatency, uint8_t(pairIndex)};
        }
        // ★ ACC writes were per-lane too, and MADD/MSUB dominate this workload, so a full-quad ACC
        // write was another four records per pair.
        const uint32_t aw = pi.accWrite & 0xFu;
        if (aw != 0u)
        {
            uint32_t m = aw;
            while (m != 0u)
            {
                const uint32_t bit = uint32_t(__builtin_ctz(m));
                m &= m - 1u;
                accReady[3u - bit] = issue + 1; // kAccForwardLatency
                if (!coalesce)
                    ready[readyCount++] =
                        ReadyWrite{2u, 0u, uint8_t(1u << bit), 1u, uint8_t(pairIndex)};
            }
            if (coalesce)
                ready[readyCount++] = ReadyWrite{2u, 0u, uint8_t(aw), 1u, uint8_t(pairIndex)};
        }
    }

    // Fill the per-pair schedule + touched-slot summary the entry guard and re-derivation use.
    inline void recordPairTouch(Block *touch, const PairInfo &pi, const UpperPlan &up,
                                const LowerPlan &lo, int32_t issue, uint32_t pairIndex)
    {
        if (touch == nullptr)
            return;
        PairSched &ps = touch->sched[pairIndex];
        ps = PairSched{};
        ps.vfReadCount = pi.vfReadCount < 4u ? pi.vfReadCount : 4u;
        for (uint32_t k = 0; k < ps.vfReadCount; ++k)
        {
            ps.vfReadReg[k] = pi.vfReadReg[k] & 31u;
            ps.vfReadLanes[k] = pi.vfReadLanes[k] & 0xFu;
            touch->vfTouched[ps.vfReadReg[k]] |= ps.vfReadLanes[k];
        }
        ps.viRead = uint16_t(pi.viReadMask & 0xFFFEu);
        ps.accRead = uint8_t(pi.accRead & 0xFu);
        ps.issue = uint16_t(issue);
        if (pi.upperWriteReg != 0u)
            touch->vfTouched[pi.upperWriteReg & 31u] |= (pi.upperWriteLanes & 0xFu);
        if (pi.lowerWriteReg != 0u && pi.suppressedLowerVf != pi.lowerWriteReg)
            touch->vfTouched[pi.lowerWriteReg & 31u] |= (pi.lowerWriteLanes & 0xFu);
        touch->viTouched |= uint16_t(pi.viReadMask | pi.viWriteMask);
        touch->accTouched |= uint8_t(pi.accRead | pi.accWrite);
        if (up.src == OperandSrc::ScalarQ)
        {
            touch->readsQ = true;
            if (uint16_t(issue) < touch->firstQIssue)
                touch->firstQIssue = uint16_t(issue);
        }
        if (lo.op == LowerOp::Fceq || lo.op == LowerOp::Fcand ||
            lo.op == LowerOp::Fcor || lo.op == LowerOp::Fcget)
        {
            touch->readsClip = true;
            if (uint16_t(issue) < touch->firstClipIssue)
                touch->firstClipIssue = uint16_t(issue);
        }
        if (lo.op == LowerOp::Sq)
            touch->writesVuMem = true;
    }

    // Plan a block starting at pairs[0] (guest PC `startPc`). Classifies pairs until a terminator,
    // absorbing internal stalls into the schedule. A BRANCH is not a terminator gap: the block
    // takes the branch pair AND its delay slot and then writes m_state.pc (cont.198).
    // Returns the number of pairs accepted (0 = not compilable).
    inline uint32_t planBlock(const PairInfo *pairs, uint32_t avail, uint32_t startPc,
                              uint32_t pcMask, UpperPlan *ups, LowerPlan *los,
                              ReadyWrite *ready, uint32_t &readyCount,
                              bool &touchesVuMem, uint8_t &tailViBackupReg,
                              Block *touch, int32_t *issueOf, uint32_t &cycles,
                              BranchPlan *branchOut, uint8_t &branchBackupReg,
                              bool &endsWithBranch, uint8_t &stopReasonOut)
    {
        // Pair index at which each value becomes readable, in ISSUE cycles relative to entry.
        int32_t vfReady[32][4] = {};
        int32_t viReady[16] = {};
        int32_t accReady[4] = {};
        int32_t prevIssue = -1;
        bool sawClip = false;

        readyCount = 0;
        touchesVuMem = false;
        tailViBackupReg = 0;
        branchBackupReg = 0;
        endsWithBranch = false;
        stopReasonOut = 0u;
        cycles = 0;

        // ★ cont.213: WHY did the block stop growing? cont.213's pair histogram found 47% of all
        // block ENTRIES are 2-pair blocks paying the full ~269-cycle driver cost for two pairs of
        // work, so the reason a block is short is now the most valuable thing to know.
        // 0 = ran to the 8-pair cap, 1 = i/e/d/t bit, 2 = upper uncompilable, 3 = issue overflow,
        // 4 = clip reader after CLIP, 5 = branch rejected, 6 = delay-slot bits,
        // 7 = delay-slot uncompilable, 8 = cont.194 VI hazard, 9 = ready list full,
        // 10 = normal branch terminator, 11 = trimmed by the cont.194 tail contract.
        uint8_t stopReason = 0u;
        uint32_t n = 0;
        for (; n < avail && n < kMaxBlockPairs; ++n)
        {
            const PairInfo &pi = pairs[n];
            // ★ D/T halt bits end the program when FBRST enables them, and a block does not model
            // that -- so such a pair never enters one. (FBRST really can enable them; this was a
            // latent gap before cont.198.)
            // cont.215: `iBit` is NOT a halt bit -- E/D/T end the program, the I bit only says the
            // lower word is a float immediate. It is handled below instead of rejected here.
            if (pi.eBit || pi.dBit || pi.tBit)
                { stopReason = 1u; break; }
            UpperPlan up;
            LowerPlan lo;
            if (!classifyUpperPlan(pi.upper, up))
                { stopReason = 2u; break; }

            int32_t issue = prevIssue + 1;
            applyReadStalls(pi, issue, vfReady, viReady, accReady);
            if (issue > int32_t(0xFFFF - 64))
                { stopReason = 3u; break; } // pathological; keep `issue` inside its uint16_t field

            // cont.215: with the I bit set the lower word is DATA, so it must not be classified as
            // an instruction (that is exactly why these pairs bailed out: classifyLowerPlan saw a
            // float bit pattern). The plan is "no lower op, plus a store of the folded immediate".
            bool lowerOk;
            if (pi.iBit)
            {
                lo = LowerPlan{};
                lo.setsI = true;
                lo.iBits = uint32_t(_mm_cvtsi128_si32(
                    vuNormOperandBits(_mm_cvtsi32_si128(int32_t(pi.lower)))));
                lowerOk = true;
            }
            else
            {
                lowerOk = classifyLowerPlan(pi.lower, lo);
            }
            if (lowerOk && sawClip &&
                (lo.op == LowerOp::Fceq || lo.op == LowerOp::Fcand ||
                 lo.op == LowerOp::Fcor || lo.op == LowerOp::Fcget))
                { stopReason = 4u; break; } // clip reader after a CLIP in the same block -- see the note above
            if (up.op == FmacOp::Clip)
                sawClip = true;
            if (!lowerOk)
            {
                // ---- BRANCH TERMINATOR: execute the branch AND its delay slot ----------------
                BranchPlan bp;
                if (branchOut == nullptr || n + 1u >= avail || n + 1u >= kMaxBlockPairs ||
                    readyCount + 18u > kMaxReadyWrites ||
                    !classifyBranch(pi.lower, startPc + 8u * n, pcMask, bp))
                {
                    // cont.230: reason 5 is really two populations -- a real branch that cannot be
                    // taken into the block (cap / ready-list room) and an UNCOMPILABLE lower op
                    // (classifyBranch fails). Record the word so the driver can histogram them.
                    if (touch) touch->stopLower = pi.lower;
                    stopReason = 5u; break;
                }
                const PairInfo &ds = pairs[n + 1u];
                UpperPlan dsUp;
                LowerPlan dsLo;
                if (ds.eBit || ds.dBit || ds.tBit)
                    { stopReason = 6u; break; }
                if (!classifyUpperPlan(ds.upper, dsUp))
                    { stopReason = 7u; break; }
                // cont.215: a delay slot may carry the I bit too -- same treatment, data not code.
                if (ds.iBit)
                {
                    dsLo = LowerPlan{};
                    dsLo.setsI = true;
                    dsLo.iBits = uint32_t(_mm_cvtsi128_si32(
                        vuNormOperandBits(_mm_cvtsi32_si128(int32_t(ds.lower)))));
                }
                else if (!classifyLowerPlan(ds.lower, dsLo))
                    { stopReason = 7u; break; } // the delay slot must be compilable, never a branch

                // ★ cont.194 hazard: if the pair immediately before the branch made a
                // `delaysNextBranchRead` VI write to a register this branch READS, the branch must
                // still see the OLD value. planBlock records it; emitBlock stashes it in ECX
                // before that write and emitBranch consumes it.
                if (n > 0u)
                {
                    const PairInfo &prev = pairs[n - 1u];
                    const uint32_t vw = prev.viWriteMask & 0xFFFEu;
                    if (prev.lowerDelaysBranchRead && vw != 0u)
                    {
                        const uint8_t r = uint8_t(__builtin_ctz(vw));
                        const bool readsIt =
                            (bp.kind == BranchKind::Ibeq || bp.kind == BranchKind::Ibne);
                        if ((r == bp.viS || (readsIt && r == bp.viT)) && r != 0u)
                            branchBackupReg = r;
                    }
                }

                *branchOut = bp;
                ups[n] = up;
                los[n] = LowerPlan{}; // the branch's lower is emitted by emitBranch, not emitLowerPlan
                issueOf[n] = issue;
                recordPairTouch(touch, pi, up, los[n], issue, n);
                recordPairWrites(pi, issue, ready, readyCount, vfReady, viReady, accReady, n);
                prevIssue = issue;
                ++n;

                int32_t dsIssue = prevIssue + 1;
                applyReadStalls(ds, dsIssue, vfReady, viReady, accReady);
                ups[n] = dsUp;
                los[n] = dsLo;
                issueOf[n] = dsIssue;
                if (dsLo.touchesVuMem)
                    touchesVuMem = true;
                recordPairTouch(touch, ds, dsUp, dsLo, dsIssue, n);
                recordPairWrites(ds, dsIssue, ready, readyCount, vfReady, viReady, accReady, n);
                prevIssue = dsIssue;
                ++n;
                endsWithBranch = true;
                { stopReason = 10u; break; }
            }

            if (readyCount + 9u > kMaxReadyWrites)
                { stopReason = 9u; break; }

            ups[n] = up;
            los[n] = lo;
            issueOf[n] = issue;
            prevIssue = issue;
            if (lo.touchesVuMem)
                touchesVuMem = true;
            recordPairTouch(touch, pi, up, lo, issue, n);
            recordPairWrites(pi, issue, ready, readyCount, vfReady, viReady, accReady, n);
        }

        // ★★ The cont.194 branch-delay contract for the pair AFTER the block. Only applies when the
        // block does NOT end with a branch (if it does, the block already consumed the delay slot
        // and the next pair is a fresh basic block).
        while (!endsWithBranch && n > 0u)
        {
            const PairInfo &last = pairs[n - 1u];
            const uint32_t vw = last.viWriteMask & 0xFFFEu;
            if (!last.lowerDelaysBranchRead || vw == 0u)
                break;
            const uint8_t reg = uint8_t(__builtin_ctz(vw));
            bool earlier = false;
            for (uint32_t k = 0; k + 1u < n; ++k)
                if ((pairs[k].viWriteMask & (1u << reg)) != 0u)
                    earlier = true;
            if (!earlier)
            {
                tailViBackupReg = reg;
                break;
            }
            --n;
            stopReason = 11u; // trimmed after the fact by the cont.194 tail contract
            uint32_t rc = 0;
            for (uint32_t k = 0; k < readyCount; ++k)
                if (ready[k].pair < n)
                    ready[rc++] = ready[k];
            readyCount = rc;
        }
        if (endsWithBranch && n > 0u)
        {
            const PairInfo &last = pairs[n - 1u]; // the delay slot
            const uint32_t vw = last.viWriteMask & 0xFFFEu;
            if (last.lowerDelaysBranchRead && vw != 0u)
            {
                const uint8_t reg = uint8_t(__builtin_ctz(vw));
                bool earlier = false;
                for (uint32_t k = 0; k + 1u < n; ++k)
                    if ((pairs[k].viWriteMask & (1u << reg)) != 0u)
                        earlier = true;
                if (!earlier)
                    tailViBackupReg = reg;
                else
                    return 0u; // cannot preserve the old value; leave the whole block alone
            }
        }
        if (n == 0u)
        {
            readyCount = 0;
            cycles = 0;
            return 0u;
        }
        cycles = uint32_t(issueOf[n - 1u] + 1);
        stopReasonOut = stopReason;
        return n;
    }

    // ---- block store -------------------------------------------------------------------------
    // Direct-mapped by pair index (pc/8), which is exact: VU1 microcode is at most 16 KB, so there
    // are at most 2048 pair slots. Wiped whenever the guest uploads new microcode — the same
    // `generation` the decode cache already tracks.
    inline void blockBufferReset(unsigned unit);

    struct BlockStore
    {
        unsigned unit = 1u; // cont.214: which unit's code buffer this store owns
        static constexpr uint32_t kPairSlots = 0x4000u / 8u;
        static constexpr uint32_t kMaxBlocks = kPairSlots + 1u; // one per entry PC, plus slot 0

        Block blocks[kMaxBlocks];
        int32_t byPair[kPairSlots];
        uint64_t generation = ~0ull;
        // ★ cont.202: the guest re-uploads VU1 microcode constantly, which bumps the generation
        // even when the BYTES are identical. Wiping the store on every bump meant recompiling the
        // same blocks over and over -- measured as blocks being 0.5-0.9x (i.e. a LOSS) through the
        // boot/movie era while the level era ran 3.4x. Keying on a hash of the code instead means
        // a re-upload of identical microcode costs one hash, not a full recompile.
        uint64_t codeHash = 0;
        unsigned long invalidations = 0, hashHits = 0;
        uint32_t blockCount = 1; // ★ index 0 is a permanent fn==nullptr "rejected" sentinel, so a
                                 // full store still caches its rejections instead of retrying every
                                 // pair forever (which is exactly how the first version livelocked)
        unsigned long hits = 0, misses = 0, compiled = 0, rejected = 0;
        unsigned long entered = 0, pairsRun = 0, guardBlocked = 0;
        // which guard condition rejected an entry (aims the next relaxation)
        unsigned long rescheduled = 0;
        unsigned long gbVf = 0, gbVi = 0, gbAcc = 0, gbStore = 0, gbQ = 0, gbClip = 0,
                      gbXgkick = 0, gbBudget = 0;
        // cont.230 census: how often each guard-admitted EARLY APPLY fires (the values a block
        // reads before their pipeline commit -- the exact class the dark-wedge bug lived in).
        unsigned long earlyVf = 0, earlyClip = 0, earlyQ = 0;

        // Returns true when the store was actually wiped.
        bool revalidate(uint64_t gen, uint64_t hash)
        {
            if (generation == gen)
                return false;
            if (codeHash == hash && generation != ~0ull)
            {
                generation = gen; // same microcode, different generation counter -- keep everything
                ++hashHits;
                return false;
            }
            codeHash = hash;
            invalidate(gen);
            return true;
        }

        void invalidate(uint64_t gen)
        {
            ++invalidations;
            generation = gen;
            blockBufferReset(unit);
            blocks[0] = Block{};
            blockCount = 1;
            for (uint32_t i = 0; i < kPairSlots; ++i)
                byPair[i] = -1;
        }
    };

    // ★★ cont.214: PER-UNIT. VU0 and VU1 interleave constantly (5.3M VCALLMS per run against
    // VU1's own work), and both the store's `blockByPair` map and the generation key are keyed by
    // PC -- one shared store would invalidate itself on every switch and thrash. One store and one
    // code buffer per unit; index 1 = VU1 (the default, so every existing call site is unchanged).
    inline BlockStore &blockStore(unsigned unit = 1u)
    {
        static BlockStore s[2];
        static const bool once = [] { s[0].unit = 0u; s[1].unit = 1u; return true; }();
        (void)once;
        return s[unit & 1u];
    }
    // ★ Blocks get their OWN code buffer. They are invalidated on every microcode upload, and
    // emitted block code would otherwise accumulate forever in the shared buffer -- which is
    // exactly what happened: after ~7,400 blocks the 8 MB buffer was exhausted and every later
    // block was silently rejected. A separate buffer can be reset with the store, while the
    // per-instruction upper cache (whose function pointers are keyed by opcode and live forever)
    // keeps the original one.
    inline CodeBuffer &blockBuffer(unsigned unit = 1u) { static CodeBuffer b[2]; return b[unit & 1u]; }
    inline void blockBufferReset(unsigned unit) { blockBuffer(unit).reset(); }

    inline bool blockEnabled()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_BLOCK"); return e && e[0] && e[0] != '0'; }();
        return v;
    }
    inline bool blockVerifyEnabled()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_BLOCKVERIFY"); return e && e[0] && e[0] != '0'; }();
        return v;
    }

    // Compile (or fetch) the block starting at `startPc`. `pairs` must point at the decoded pairs
    // from startPc onward, `avail` being how many are contiguously available.
    // Returns null when nothing at this PC is compilable (that answer is cached too).
    // Split in two so the caller only pays to flatten PairInfo on a cache MISS.
    // `known` comes back true when this PC has already been decided (compiled or rejected).
    inline const Block *findBlock(BlockStore &store, uint64_t generation, uint32_t startPc,
                                  bool &known)
    {
        known = false;
        if (store.generation != generation)
            store.invalidate(generation);
        const uint32_t slot = startPc / 8u;
        if (slot >= BlockStore::kPairSlots)
        {
            known = true; // out of range: never compilable, and nothing to cache
            return nullptr;
        }
        const int32_t existing = store.byPair[slot];
        if (existing < 0)
            return nullptr;
        known = true;
        ++store.hits;
        const Block &b = store.blocks[existing];
        return b.fn != nullptr ? &b : nullptr;
    }

    inline const Block *buildBlock(CodeBuffer &buf, BlockStore &store, uint64_t generation,
                                   uint32_t startPc, uint32_t pcMask,
                                   const PairInfo *pairs, uint32_t avail, bool emitFlags)
    {
        if (store.generation != generation)
            store.invalidate(generation);
        const uint32_t slot = startPc / 8u;
        if (slot >= BlockStore::kPairSlots)
            return nullptr;
        ++store.misses;
        if (store.blockCount >= BlockStore::kMaxBlocks || !buf.ensure(8u << 20))
        {
            store.byPair[slot] = 0; // 0 == known-uncompilable; never retried
            ++store.rejected;
            return nullptr;
        }

        Block &b = store.blocks[store.blockCount];
        b = Block{};
        b.startPc = startPc;

        static UpperPlan ups[kMaxBlockPairs];
        static LowerPlan los[kMaxBlockPairs];
        uint32_t readyCount = 0;
        bool touchesVuMem = false;
        uint8_t tailViBackupReg = 0;
        static int32_t issueOf[kMaxBlockPairs];
        uint32_t cycles = 0;
        BranchPlan branch;
        uint8_t branchBackupReg = 0;
        bool endsWithBranch = false;
        const uint32_t n = planBlock(pairs, avail, startPc, pcMask, ups, los, b.ready, readyCount,
                                     touchesVuMem, tailViBackupReg, &b, issueOf, cycles,
                                     &branch, branchBackupReg, endsWithBranch, b.stopReason);

        // ★ cont.214: when a VU0 PC cannot make a block, say WHY. VU0 has only two programs and a
        // few dozen candidate PCs, so the whole rejection set fits in the log -- which is what
        // decides whether VU0 coverage can reach the ~85% where blocks start to pay.
        if (n < 2u && store.unit == 0u && std::getenv("PS2X_VU0_REJECTLOG") != nullptr)
        {
            static uint32_t seen[256];
            static uint32_t seenN = 0;
            bool known = false;
            for (uint32_t i = 0; i < seenN; ++i)
                if (seen[i] == startPc) { known = true; break; }
            if (!known && seenN < 256u)
            {
                seen[seenN++] = startPc;
                const uint32_t up = avail ? pairs[0].upper : 0u;
                const uint32_t lo = avail ? pairs[0].lower : 0u;
                std::fprintf(stderr,
                             "[vu0:reject] pc=0x%04x n=%u stop=%u upper=0x%08x(op=0x%02x)"
                             " lower=0x%08x(op=0x%02x/sp=0x%02x) iBit=%u\n",
                             startPc, n, b.stopReason, up, (up >> 2) & 0x3Fu, lo,
                             (lo >> 25) & 0x7Fu, lo & 0x3Fu, avail ? pairs[0].iBit : 0u);
            }
        }
        // A 1-pair block cannot pay for the call, and the interpreter already handles it well.
        if (n >= 2u && buf.room() > 1024u * (n + 1u))
        {
            Emitter e;
            e.p = buf.code + buf.used;
            e.end = buf.base + buf.capacity;
            uint8_t *const start = e.p;
            b.emitsFlags = emitFlags;
            b.flagCount = 0;
            b.clipCount = 0;
            uint32_t scratchNext = 0;
            for (uint32_t j = 0; j < n; ++j)
            {
                // A flag-writing FMAC is exactly one with clampResult (MAX/MINI/ABS/ITOF/FTOI use
                // applyDest, which sets no flags; NOP writes nothing).
                const void *sink = nullptr;
                if (ups[j].op == FmacOp::Clip && b.clipCount < kMaxBlockPairs &&
                    scratchNext + 2u <= 16u)
                {
                    sink = buf.resultSlot(scratchNext);
                    b.clipWrites[b.clipCount++] = Block::ClipWrite{uint8_t(j), uint8_t(scratchNext)};
                    scratchNext += 2u;
                }
                else if (emitFlags && ups[j].op != FmacOp::Nop && ups[j].clampResult &&
                    b.flagCount < kMaxBlockPairs && scratchNext < 16u)
                {
                    const uint8_t slot = uint8_t(scratchNext++);
                    sink = buf.resultSlot(slot);
                    b.flagWrites[b.flagCount++] =
                        Block::FlagWrite{uint8_t(j), uint8_t((pairs[j].upper >> 21) & 0xFu), slot};
                }
                // Stash the pre-write VI value for the branch-delay hazard, before the pair that
                // overwrites it (see planBlock).
                // The pair that OVERWRITES the branch's operand sits two before the end
                // (branch is n-2, delay slot n-1), so the stash goes before pair n-3.
                if (endsWithBranch && branchBackupReg != 0u && j + 3u == n)
                    emitBranchBackupStash(e, branchBackupReg);
                if (endsWithBranch && j + 1u == n - 1u)
                {
                    // the branch pair: upper only, then the target selection into ECX
                    emitPairInto(e, buf, pairs[j], ups[j], los[j], sink);
                    emitBranch(e, branch, pcMask, branchBackupReg, branchBackupReg != 0u);
                    continue;
                }
                emitPairInto(e, buf, pairs[j], ups[j], los[j], sink);
            }
            if (endsWithBranch)
                emitBranchCommit(e); // m_state.pc = ECX, after the delay slot has run
            e.ret();
            if (!e.overflow)
            {
                buf.used = size_t(e.p - buf.code);
                b.fn = reinterpret_cast<BlockFn>(start);
                b.pairs = n;
                b.cycles = cycles;
                b.readyCount = readyCount;
                b.touchesVuMem = touchesVuMem;
                b.tailViBackupReg = tailViBackupReg;
                b.endsWithBranch = endsWithBranch;
                ++store.compiled;
            }
        }
        if (b.fn == nullptr)
        {
            // Don't consume a block slot for a rejection -- just remember it as 0.
            ++store.rejected;
            store.byPair[slot] = 0;
            return nullptr;
        }
        store.byPair[slot] = int32_t(store.blockCount);
        ++store.blockCount;
        return &b;
    }

    // ---- process-wide singletons + the interface the interpreter uses -----------------------
    // Function-local statics inside inline functions are shared across translation units, so both
    // ps2_vu1_core.cpp and ps2_vu1_upper.cpp see the same buffer, cache and flags.

    inline CodeBuffer &buffer() { static CodeBuffer b; return b; }
    inline Cache &cache() { static Cache c; return c; }
    // Mirrors the interpreter's per-program lazy-flag arming (set by run()); the emitted code
    // derives no MAC/status flags, so it is only valid while that is true.
    inline bool &lazyActive() { static bool v = false; return v; }
    inline bool &inVerify() { static bool v = false; return v; }

    // PS2X_VU1_JIT (default OFF until measured): use compiled upper instructions.
    inline bool enabled()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_JIT"); return e && e[0] && e[0] != '0'; }();
        return v;
    }
    // PS2X_VU1_JITVERIFY (default OFF): run BOTH and compare on live guest data.
    inline bool verifyMode()
    {
        static const bool v = []
        { const char *e = std::getenv("PS2X_VU1_JITVERIFY"); return e && e[0] && e[0] != '0'; }();
        return v;
    }

    inline UpperFn lookup(uint32_t instr)
    {
        CodeBuffer &b = buffer();
        if (!b.ensure(8u << 20))
            return nullptr;
        return compileUpper(b, cache(), instr);
    }

    struct VerifyStats { unsigned long checked = 0, mismatches = 0, reported = 0; };
    inline VerifyStats &verifyStats() { static VerifyStats s; return s; }

    inline void noteVerify(uint32_t instr, const float *jitVf, const float *jitAcc,
                           const float *refVf, const float *refAcc)
    {
        VerifyStats &s = verifyStats();
        ++s.checked;
        bool bad = false;
        for (int c = 0; c < 4; ++c)
        {
            uint32_t a = 0, b = 0, x = 0, y = 0;
            std::memcpy(&a, &jitVf[c], 4);
            std::memcpy(&b, &refVf[c], 4);
            std::memcpy(&x, &jitAcc[c], 4);
            std::memcpy(&y, &refAcc[c], 4);
            if (a != b || x != y)
                bad = true;
        }
        if (bad)
        {
            ++s.mismatches;
            if (s.reported < 10ul)
            {
                ++s.reported;
                uint32_t a = 0, b = 0;
                std::memcpy(&a, &jitVf[0], 4);
                std::memcpy(&b, &refVf[0], 4);
                std::fprintf(stderr, "[vu1:jitverify] MISMATCH instr=%08x jitVf0=%08x refVf0=%08x\n",
                             instr, a, b);
            }
        }
    }

    inline void report()
    {
        Cache &c = cache();
        VerifyStats &s = verifyStats();
        std::fprintf(stderr,
                     "[vu1:jit] hits=%lu misses=%lu compiled=%lu rejected=%lu codeBytes=%zu"
                     " | verify checked=%lu mismatches=%lu\n",
                     c.hits, c.misses, c.compiled, c.rejected, buffer().used,
                     s.checked, s.mismatches);
    }

    struct FakeState // mirrors just the fields the emitted code touches, at known offsets
    {
        alignas(16) float vf[32][4];
        alignas(16) float acc[4];
        float q;
        float i;
    };

    // ---- block assembly + the cost microbenchmark ------------------------------------------
    // cont.185: the whole remaining JIT investment turns on ONE unknown — what does a COMPILED
    // pair cost, versus the ~420 host cycles (120.5 ns) an interpreted one costs? Chaining pairs
    // into a single function removes exactly what the per-instruction JIT could not: the per-pair
    // call, the decode fetch, the scheduler, the pipeline queue/commit and the shadow dance. This
    // measures that directly instead of estimating it.
    //
    // The block mimics a real T&L inner loop — the MULAbc + MADDAbc chain a 4x4 matrix transform
    // compiles to, which is what the cont.178 census shows dominates (special 0x1b MULAbc 6.9%,
    // 0x09/0x0a MADDAbc 11.4%).

    // Emit `count` FMAC pairs back to back with no call between them.
    inline void emitBlockChain(Emitter &e, const CodeBuffer &buf, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            const int phase = i & 3;
            // phase 0 starts the accumulation (MULA), phases 1..3 accumulate (MADDA).
            const FmacOp op = (phase == 0) ? FmacOp::Mul : FmacOp::Madd;
            UpperPlan plan;
            plan.op = op;
            plan.src = OperandSrc::VtLane;
            plan.bc = phase;
            plan.clampResult = true;
            plan.readsAcc = (op == FmacOp::Madd);
            emitUpper(e, buf, plan,
                      kOffVf + int32_t(1 + phase) * 16, // vs = one row of the source vector
                      kOffVf + int32_t(8) * 16,         // vt = the matrix row, broadcast
                      kOffAcc, kOffQ, kOffI, kOffAcc, 0xF);
        }
    }

    inline void benchBlock(CodeBuffer &buf,
                           void (*normOperandQuad)(const float *, float *),
                           void (*normResultValue)(float *, uint8_t))
    {
        if (!buf.ensure(8u << 20))
            return;
        // ★ Sweep the block LENGTH. The cont.178 census says the mean basic block is only ~6 pairs,
        // so the per-block entry/exit cost is amortised over very few instructions — measuring only
        // a long block would flatter the JIT badly. This shows the amortisation curve directly.
        static const int kLens[] = {1, 2, 4, 6, 8, 16, 64, 256};
        for (int li = 0; li < int(sizeof(kLens) / sizeof(kLens[0])); ++li)
        {
            const int n = kLens[li];
            const int iters = 4000000 / n;
            buf.reset();
            Emitter be;
            be.p = buf.code;
            be.end = buf.base + buf.capacity;
            uint8_t *const bstart = be.p;
            emitBlockChain(be, buf, n);
            be.ret();
            if (be.overflow)
                continue;
            UpperFn bfn = reinterpret_cast<UpperFn>(bstart);
            static FakeState bst; // static: guest-thread stack is small
            bst = FakeState{};
            for (int r = 0; r < 12; ++r)
                for (int c = 0; c < 4; ++c)
                    bst.vf[r][c] = 1.0f + 0.125f * float(r * 4 + c);
            for (int i = 0; i < 2000; ++i)
                bfn(&bst);
            const auto b0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i)
                bfn(&bst);
            const auto b1 = std::chrono::steady_clock::now();
            const double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(b1 - b0).count());
            const double perPair = ns / (double(n) * double(iters));
            std::fprintf(stderr, "[vu1:jitbench] block=%3d pairs -> %6.2f ns/pair  (%.1fx vs 120.5)\n",
                         n, perPair, 120.5 / perPair);
        }

        static constexpr int kPairs = 256;
        static constexpr int kIters = 20000;

        buf.reset();
        Emitter e;
        e.p = buf.code;
        e.end = buf.base + buf.capacity;
        uint8_t *const start = e.p;
        emitBlockChain(e, buf, kPairs);
        e.ret();
        if (e.overflow)
        {
            std::fprintf(stderr, "[vu1:jitbench] overflow\n");
            return;
        }
        buf.used = size_t(e.p - buf.code);
        UpperFn fn = reinterpret_cast<UpperFn>(start);

        static FakeState st;  // static: guest-thread stack is small
        st = FakeState{};
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 4; ++c)
                st.vf[r][c] = 1.0f + 0.125f * float(r * 4 + c);

        // warm up (page in the code, settle the frequency)
        for (int i = 0; i < 200; ++i)
            fn(&st);

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i)
            fn(&st);
        const auto t1 = std::chrono::steady_clock::now();

        // Reference: the same chain done with the C++ SSE helpers the interpreter uses, i.e. the
        // arithmetic WITHOUT any interpreter loop around it. The gap between this and the
        // interpreter's 120.5 ns/pair is pure per-pair plumbing.
        alignas(16) float acc[4] = {0, 0, 0, 0};
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i)
        {
            for (int p = 0; p < kPairs; ++p)
            {
                const int phase = p & 3;
                alignas(16) float vs[4], vt[4], accn[4], res[4];
                normOperandQuad(st.vf[1 + phase], vs);
                normOperandQuad(st.vf[8], vt);
                const float b = vt[phase];
                vt[0] = vt[1] = vt[2] = vt[3] = b;
                if (phase == 0)
                {
                    for (int c = 0; c < 4; ++c)
                        res[c] = vs[c] * vt[c];
                }
                else
                {
                    normOperandQuad(acc, accn);
                    for (int c = 0; c < 4; ++c)
                        res[c] = accn[c] + vs[c] * vt[c];
                }
                normResultValue(res, 0xF);
                for (int c = 0; c < 4; ++c)
                    acc[c] = res[c];
            }
        }
        const auto t3 = std::chrono::steady_clock::now();

        const double jitNs = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        const double refNs = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
        const double pairs = double(kPairs) * double(kIters);
        std::fprintf(stderr,
                     "[vu1:jitbench] compiled block: %.2f ns/pair | C++ SSE (no loop): %.2f ns/pair"
                     " | interpreter is ~120.5 ns/pair -> block is %.1fx\n",
                     jitNs / pairs, refNs / pairs, 120.5 / (jitNs / pairs));
        buf.reset();
    }

    // ---- self-test -------------------------------------------------------------------------
    // Emits `void f(const float* rdi, float* rsi)` applying the operand clamp, then compares it
    // against the C++ quad helper over an exhaustive bit sweep. This validates BOTH the
    // instruction encodings and the emitted clamp sequence, with zero risk to the guest: it runs
    // only under PS2X_VU1_JITSELFTEST and touches no interpreter state.
    typedef void (*ClampFn)(const float *, float *);

    inline bool selfTest(CodeBuffer &buf, void (*refQuad)(const float *, float *))
    {
        if (!buf.ensure(1u << 20))
        {
            std::fprintf(stderr, "[vu1:jit] selftest SKIPPED - could not map RWX memory\n");
            return false;
        }
        buf.reset();
        Emitter e;
        e.p = buf.code;
        e.end = buf.base + buf.capacity;

        ClampFn fn = reinterpret_cast<ClampFn>(buf.code);
        e.movups_load(0, 7);                       // movups xmm0, [rdi]
        emitNormOperand(e, buf, 0, 1, 2, 3, 4);    // clamp xmm0, scratch xmm1..xmm4
        e.movups_store(6, 0);                      // movups [rsi], xmm0
        e.ret();
        buf.used = size_t(e.p - buf.code);

        if (e.overflow)
        {
            std::fprintf(stderr, "[vu1:jit] selftest FAILED - emitter overflow\n");
            return false;
        }
        if (const char *dump = std::getenv("PS2X_VU1_JITDUMP"))
        {
            if (dump[0] && dump[0] != '0')
            {
                std::fprintf(stderr, "[vu1:jit] emitted %zu bytes at %p:", buf.used,
                             static_cast<void *>(buf.code));
                for (size_t i = 0; i < buf.used; ++i)
                    std::fprintf(stderr, " %02x", buf.code[i]);
                std::fprintf(stderr, "\n");
            }
        }

        unsigned long cases = 0, mismatches = 0;
        static const uint32_t mantissas[5] = {0u, 1u, 0x400000u, 0x7FFFFFu, 0x123456u};
        for (uint32_t exp = 0; exp < 256u; ++exp)
        {
            for (uint32_t mi = 0; mi < 5u; ++mi)
            {
                for (uint32_t sgn = 0; sgn < 2u; ++sgn)
                {
                    const uint32_t bits = (sgn << 31) | (exp << 23) | mantissas[mi];
                    alignas(16) float in[4];
                    alignas(16) float gotJit[4];
                    alignas(16) float gotRef[4];
                    for (int i = 0; i < 4; ++i)
                        std::memcpy(&in[i], &bits, 4);
                    fn(in, gotJit);
                    refQuad(in, gotRef);
                    for (int i = 0; i < 4; ++i)
                    {
                        uint32_t a = 0, b = 0;
                        std::memcpy(&a, &gotJit[i], 4);
                        std::memcpy(&b, &gotRef[i], 4);
                        ++cases;
                        if (a != b && ++mismatches <= 6ul)
                            std::fprintf(stderr,
                                         "[vu1:jit] selftest MISMATCH bits=%08x jit=%08x ref=%08x\n",
                                         bits, a, b);
                    }
                }
            }
        }
        std::fprintf(stderr, "[vu1:jit] selftest bytes=%zu cases=%lu mismatches=%lu -> %s\n",
                     buf.used, cases, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        buf.reset();
        return mismatches == 0;
    }

    // ---- LQ self-test ----------------------------------------------------------------------
    // Emits LQ and checks it against a C++ reference that mirrors ps2_vu1_lower.cpp exactly,
    // over negative/positive/wrapping VI bases and every dest mask. This validates the new GPR
    // and SIB encodings, the sign-extended immediate, the *16 / mask arithmetic, and the blend.
    // ★ Uses the REAL VU1State, not a mirror struct. An earlier version declared its own
    // look-alike whose field ORDER differed (vf, acc, q, i, vi instead of vf, vi, acc, q, p, i),
    // so kOffVi pointed into the wrong field and the test failed with the codegen actually correct.
    // Testing against the real struct makes offsets agree by construction.
    inline bool selfTestLQ(CodeBuffer &buf, bool doSq)
    {
        if (!buf.ensure(8u << 20))
            return false;
        static constexpr uint32_t kDataSize = 16u * 1024u;
        static uint8_t mem[kDataSize];
        for (uint32_t k = 0; k < kDataSize; ++k)
            mem[k] = uint8_t(k * 7u + (k >> 5));

        unsigned long cases = 0, mismatches = 0;
        static const int32_t bases[6] = {0, 1, 7, -3, 1023, -1024};
        static const int32_t imms[5] = {0, 1, -1, 15, -16};

        // fresh memory each pass so SQ mutations do not leak between cases
        for (uint32_t k = 0; k < kDataSize; ++k)
            mem[k] = uint8_t(k * 7u + (k >> 5));

        for (int bi = 0; bi < 6; ++bi)
        {
            for (int ii = 0; ii < 5; ++ii)
            {
                for (uint32_t dest = 0; dest < 16u; ++dest)
                {
                    buf.reset();
                    Emitter e;
                    e.p = buf.code;
                    e.end = buf.base + buf.capacity;
                    uint8_t *const start = e.p;
                    // vi[3] is the base register, vf[5] the destination
                    if (doSq)
                        emitSQ(e, buf, 3, 5, imms[ii], uint8_t(dest));
                    else
                        emitLQ(e, buf, 3, 5, imms[ii], uint8_t(dest));
                    e.ret();
                    if (e.overflow)
                        return false;
                    LowerFn fn = reinterpret_cast<LowerFn>(start);

                    static VU1State a;    // static: guest-thread stack is small
                    static VU1State b;
                    a = VU1State{};
                    for (int c = 0; c < 4; ++c)
                    {
                        const uint32_t pat = 0xA5000000u | uint32_t(c * 0x111111);
                        std::memcpy(&a.vf[5][c], &pat, 4);
                    }
                    a.vi[3] = int16_t(bases[bi]);
                    b = a;

                    const uint32_t addr =
                        (uint32_t(int32_t(b.vi[3] + imms[ii])) * 16u) & (kDataSize - 1u);

                    // snapshot the memory the reference will need, then run the emitted code
                    uint32_t refMem[4];
                    std::memcpy(refMem, mem + addr, 16);
                    fn(&a, mem, kDataSize - 1u);

                    if (!doSq)
                    {
                        // reference == ps2_vu1_lower.cpp case 0x00 (LQ)
                        float tmp[4];
                        std::memcpy(tmp, refMem, 16);
                        for (int c = 0; c < 4; ++c)
                            if (dest & (1u << (3 - c)))
                                b.vf[5][c] = tmp[c];
                        for (int c = 0; c < 4; ++c)
                        {
                            uint32_t x = 0, y = 0;
                            std::memcpy(&x, &a.vf[5][c], 4);
                            std::memcpy(&y, &b.vf[5][c], 4);
                            ++cases;
                            if (x != y && ++mismatches <= 8ul)
                                std::fprintf(stderr,
                                             "[vu1:jit] LQ MISMATCH base=%d imm=%d dest=%x lane=%d "
                                             "jit=%08x ref=%08x (addr=%u)\n",
                                             bases[bi], imms[ii], dest, c, x, y, addr);
                        }
                    }
                    else
                    {
                        // reference == ps2_vu1_lower.cpp case 0x01 (SQ), post-commit
                        uint32_t want[4];
                        std::memcpy(want, refMem, 16);
                        for (int c = 0; c < 4; ++c)
                            if (dest & (1u << (3 - c)))
                                std::memcpy(&want[c], &b.vf[5][c], 4);
                        uint32_t got[4];
                        std::memcpy(got, mem + addr, 16);
                        for (int c = 0; c < 4; ++c)
                        {
                            ++cases;
                            if (got[c] != want[c] && ++mismatches <= 8ul)
                                std::fprintf(stderr,
                                             "[vu1:jit] SQ MISMATCH base=%d imm=%d dest=%x lane=%d "
                                             "jit=%08x ref=%08x (addr=%u)\n",
                                             bases[bi], imms[ii], dest, c, got[c], want[c], addr);
                        }
                        // restore so later cases see pristine memory
                        std::memcpy(mem + addr, refMem, 16);
                    }
                }
            }
        }
        std::fprintf(stderr, "[vu1:jit] %s selftest cases=%lu mismatches=%lu -> %s\n",
                     doSq ? "SQ" : "LQ", cases, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        buf.reset();
        return mismatches == 0;
    }

    // ---- VI-writing lower self-test (IADDIU/ISUBIU + the clip readers) ----------------------
    // Drives the FULL path a block compiler will use — encode a real instruction word,
    // `classifyLowerPlan` it, `emitLowerPlan` it, run it — and compares against a reference
    // COPIED from ps2_vu1_lower.cpp rather than re-derived, so a divergence in the C++ expression
    // (the FCOR mask, the (int16_t) truncation, the zero-extended imm15) shows up as a mismatch.
    //
    // ★ All large state is `static`: self-tests run on the GUEST thread's small stack (cont.192),
    // and two extra VU1State locals were enough to smash the frame into a SIGBUS that looked like
    // emitted-code corruption.
    inline bool selfTestLowerVi(CodeBuffer &buf)
    {
        if (!buf.ensure(8u << 20))
            return false;

        unsigned long cases = 0, mismatches = 0;

        auto encIaddiu = [](bool sub, uint8_t it, uint8_t is, uint32_t imm15) {
            uint32_t instr = uint32_t(sub ? 0x09u : 0x08u) << 25;
            instr |= (uint32_t(it) & 0xFu) << 16;
            instr |= (uint32_t(is) & 0xFu) << 11;
            instr |= (imm15 & 0x7FFu);
            instr |= ((imm15 >> 11) & 0xFu) << 21;
            return instr;
        };
        auto encClip = [](uint8_t opHi, uint32_t imm24, uint8_t it) {
            uint32_t instr = uint32_t(opHi) << 25;
            if (opHi == 0x1Cu)
                instr |= (uint32_t(it) & 0xFu) << 16;
            else
                instr |= (imm24 & 0xFFFFFFu);
            return instr;
        };

        // Run one encoded lower instruction through classify -> emit -> execute, and compare the
        // whole VI file against `ref` (which the caller has already mutated the reference way).
        auto check = [&](const char *what, uint32_t instr, const VU1State &in, const VU1State &ref) {
            LowerPlan plan;
            if (!classifyLowerPlan(instr, plan))
            {
                ++cases;
                if (++mismatches <= 8ul)
                    std::fprintf(stderr, "[vu1:jit] %s REJECTED instr=%08x\n", what, instr);
                return;
            }
            buf.reset();
            Emitter e;
            e.p = buf.code;
            e.end = buf.base + buf.capacity;
            uint8_t *const start = e.p;
            emitLowerPlan(e, buf, plan);
            e.ret();
            if (e.overflow)
            {
                ++mismatches;
                return;
            }
            static VU1State got;
            got = in;
            reinterpret_cast<LowerFn>(start)(&got, nullptr, 0u);
            for (int r = 0; r < 16; ++r)
            {
                ++cases;
                if (got.vi[r] != ref.vi[r] && ++mismatches <= 8ul)
                    std::fprintf(stderr,
                                 "[vu1:jit] %s MISMATCH instr=%08x vi[%d] jit=%d ref=%d\n",
                                 what, instr, r, got.vi[r], ref.vi[r]);
            }
        };

        static const int32_t bases[7] = {0, 1, -1, 32767, -32768, 12345, -30000};
        static const uint32_t imms[7] = {0u, 1u, 0x7FFu, 0x800u, 0x1234u, 0x7FFFu, 0x4000u};
        static const uint8_t regs[4] = {0u, 1u, 5u, 15u};

        // --- IADDIU / ISUBIU ---
        for (int sub = 0; sub < 2; ++sub)
            for (int bi = 0; bi < 7; ++bi)
                for (int ii = 0; ii < 7; ++ii)
                    for (int di = 0; di < 4; ++di)
                        for (int si = 0; si < 4; ++si)
                        {
                            const uint8_t it = regs[di], is = regs[si];
                            const uint32_t instr = encIaddiu(sub != 0, it, is, imms[ii]);
                            static VU1State in, ref;
                            in = VU1State{};
                            for (int r = 1; r < 16; ++r)
                                in.vi[r] = int32_t(int16_t(0x1000 + r * 0x137));
                            in.vi[is] = (is == 0u) ? 0 : bases[bi];
                            ref = in;
                            {
                                // == ps2_vu1_lower.cpp case 0x08 / 0x09, verbatim ==
                                int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
                                if (it != 0)
                                    ref.vi[it] = sub ? (int16_t)(ref.vi[is] - imm)
                                                     : (int16_t)(ref.vi[is] + imm);
                            }
                            check(sub ? "ISUBIU" : "IADDIU", instr, in, ref);
                        }

        // --- clip readers: FCEQ / FCAND / FCOR / FCGET ---
        // `clips` deliberately includes values with bits ABOVE bit 23 set, which is the only way
        // to catch a stray mask on FCOR (the interpreter compares the UNMASKED clip | imm24).
        static const uint32_t clips[7] = {0u,          0xFFFFFFu, 0x000001u, 0x123456u,
                                          0x00000FFFu, 0xFF000000u, 0xFFFFFFFFu};
        static const uint32_t imm24s[6] = {0u, 1u, 0xFFFFFFu, 0x123456u, 0xFFF000u, 0x000FFFu};
        static const uint8_t clipOps[3] = {0x10u, 0x12u, 0x13u};
        static const char *clipNames[3] = {"FCEQ", "FCAND", "FCOR"};

        for (int oi = 0; oi < 3; ++oi)
            for (int ci = 0; ci < 7; ++ci)
                for (int mi = 0; mi < 6; ++mi)
                {
                    const uint32_t instr = encClip(clipOps[oi], imm24s[mi], 0u);
                    static VU1State in, ref;
                    in = VU1State{};
                    for (int r = 1; r < 16; ++r)
                        in.vi[r] = int32_t(int16_t(0x2000 + r * 0x91));
                    in.clip = clips[ci];
                    ref = in;
                    {
                        // == ps2_vu1_lower.cpp cases 0x10 / 0x12 / 0x13, verbatim ==
                        uint32_t imm24 = instr & 0xFFFFFF;
                        if (clipOps[oi] == 0x10u)
                            ref.vi[1] = ((ref.clip & 0xFFFFFF) == imm24) ? 1 : 0;
                        else if (clipOps[oi] == 0x12u)
                            ref.vi[1] = ((ref.clip & imm24) != 0) ? 1 : 0;
                        else
                            ref.vi[1] = ((ref.clip | imm24) == 0xFFFFFF) ? 1 : 0;
                    }
                    check(clipNames[oi], instr, in, ref);
                }

        for (int ci = 0; ci < 7; ++ci)
            for (int di = 0; di < 4; ++di)
            {
                const uint8_t it = regs[di];
                const uint32_t instr = encClip(0x1Cu, 0u, it);
                static VU1State in, ref;
                in = VU1State{};
                for (int r = 1; r < 16; ++r)
                    in.vi[r] = int32_t(int16_t(0x3000 + r * 0x55));
                in.clip = clips[ci];
                ref = in;
                {
                    // == ps2_vu1_lower.cpp case 0x1C, verbatim ==
                    if (it != 0)
                        ref.vi[it] = static_cast<int32_t>(ref.clip & 0x0FFFu);
                }
                check("FCGET", instr, in, ref);
            }

        // --- decode contract ---------------------------------------------------------------
        // ★ `emitLQ`/`emitSQ` are exhaustively verified by selfTestLQ, but that test calls them
        // DIRECTLY, so it cannot see a mistake in which instruction FIELD feeds which argument.
        // Block assembly reaches them only through classifyLowerPlan, so the decode itself is
        // asserted here against the interpreter's own field accessors (ps2_vu1_detail.h) — the
        // same helpers ps2_vu1_lower.cpp uses. (Found by mutation testing: swapping SQ's base from
        // VIT to VIS survived every other check in this file.)
        {
            auto encLqSq = [](bool sq, uint8_t vf, uint8_t vi, uint8_t dest, uint32_t imm11) {
                uint32_t instr = uint32_t(sq ? 0x01u : 0x00u) << 25;
                instr |= (uint32_t(dest) & 0xFu) << 21;
                instr |= (uint32_t(vf) & 0x1Fu) << 16; // LQ: ft (dest vf) / SQ: it (base vi)
                instr |= (uint32_t(vi) & 0x1Fu) << 11; // LQ: is (base vi) / SQ: fs (source vf)
                instr |= (imm11 & 0x7FFu);
                return instr;
            };
            auto expect = [&](const char *what, bool ok, uint32_t instr) {
                ++cases;
                if (!ok && ++mismatches <= 8ul)
                    std::fprintf(stderr, "[vu1:jit] decode contract FAILED: %s (instr=%08x)\n",
                                 what, instr);
            };
            for (uint32_t a = 1; a < 0x1Fu; a += 3u)
                for (uint32_t b = 1; b < 0x1Fu; b += 5u)
                    for (uint32_t d = 0; d < 16u; d += 5u)
                        for (uint32_t m = 0; m < 0x800u; m += 331u)
                        {
                            LowerPlan p;
                            uint32_t instr = encLqSq(false, uint8_t(a), uint8_t(b), uint8_t(d), m);
                            expect("LQ classify", classifyLowerPlan(instr, p), instr);
                            expect("LQ op", p.op == LowerOp::Lq, instr);
                            expect("LQ vf==FT", p.vfReg == FT(instr), instr);
                            expect("LQ base==VIS", p.viBase == VIS(instr), instr);
                            expect("LQ dest", p.dest == DEST(instr), instr);
                            expect("LQ imm==IMM11", p.imm == int32_t(IMM11(instr)), instr);
                            expect("LQ writes vf, not vi", p.writesVf && !p.writesVi, instr);
                            expect("LQ no branch delay", !p.delaysNextBranchRead, instr);

                            instr = encLqSq(true, uint8_t(a), uint8_t(b), uint8_t(d), m);
                            expect("SQ classify", classifyLowerPlan(instr, p), instr);
                            expect("SQ op", p.op == LowerOp::Sq, instr);
                            expect("SQ vf==FS", p.vfReg == FS(instr), instr);
                            expect("SQ base==VIT", p.viBase == VIT(instr), instr);
                            expect("SQ dest", p.dest == DEST(instr), instr);
                            expect("SQ imm==IMM11", p.imm == int32_t(IMM11(instr)), instr);
                            expect("SQ writes neither vf nor vi", !p.writesVf && !p.writesVi, instr);
                            expect("SQ touches VU memory", p.touchesVuMem, instr);
                        }

            // IADDIU/ISUBIU and the clip readers: the same field contract.
            for (uint32_t it = 0; it < 16u; ++it)
                for (uint32_t is = 0; is < 16u; is += 3u)
                    for (uint32_t v = 0; v < 0x8000u; v += 1237u)
                    {
                        const uint32_t instr = encIaddiu(false, uint8_t(it), uint8_t(is), v);
                        LowerPlan p;
                        expect("IADDIU classify", classifyLowerPlan(instr, p), instr);
                        expect("IADDIU dst==VIT", p.viDst == VIT(instr), instr);
                        expect("IADDIU src==VIS", p.viSrc == VIS(instr), instr);
                        // the imm15 is ZERO-extended: never negative, always < 0x8000
                        expect("IADDIU imm zero-extended", p.imm >= 0 && p.imm < 0x8000, instr);
                        expect("IADDIU writesVi iff it!=0", p.writesVi == (it != 0u), instr);
                        expect("IADDIU no VU memory", !p.touchesVuMem, instr);
                    }
            for (uint32_t m = 0; m < 0x1000000u; m += 0x111111u)
            {
                LowerPlan p;
                for (uint8_t op : {uint8_t(0x10u), uint8_t(0x12u), uint8_t(0x13u)})
                {
                    const uint32_t instr = encClip(op, m, 0u);
                    expect("clip classify", classifyLowerPlan(instr, p), instr);
                    expect("clip imm24", uint32_t(p.imm) == (instr & 0xFFFFFFu), instr);
                    expect("clip writes VI1", p.viDst == 1u && p.writesVi, instr);
                }
            }
            // Everything a block must STOP at still classifies as uncovered.
            for (uint32_t opHi = 0x20u; opHi <= 0x2Fu; ++opHi)
            {
                LowerPlan p;
                const uint32_t instr = (opHi << 25) | 0x123u;
                expect("branch rejected", !classifyLowerPlan(instr, p), instr);
            }
            for (uint8_t opHi : {uint8_t(0x04u), uint8_t(0x05u), uint8_t(0x11u), uint8_t(0x14u),
                                 uint8_t(0x40u)})
            {
                LowerPlan p;
                const uint32_t instr = (uint32_t(opHi) << 25) | 0x123u;
                expect("uncovered rejected", !classifyLowerPlan(instr, p), instr);
            }
            // ...but both NOP encodings are covered, and write nothing.
            for (uint32_t nop : {0x00000000u, 0x8000033Cu})
            {
                LowerPlan p;
                expect("NOP classify", classifyLowerPlan(nop, p), nop);
                expect("NOP is Nop", p.op == LowerOp::Nop, nop);
                expect("NOP writes nothing",
                       !p.writesVi && !p.writesVf && !p.touchesVuMem && !p.delaysNextBranchRead, nop);
            }
        }

        // The branch-delay contract is a CLASSIFICATION fact, so assert it here too: getting it
        // backwards is silent at run time until a block takes the wrong path.
        {
            LowerPlan p;
            struct { uint32_t instr; bool delays; const char *name; } expect[] = {
                {encIaddiu(false, 3, 4, 7), true, "IADDIU"},
                {encIaddiu(true, 3, 4, 7), true, "ISUBIU"},
                {encClip(0x10u, 5u, 0u), false, "FCEQ"},
                {encClip(0x12u, 5u, 0u), false, "FCAND"},
                {encClip(0x13u, 5u, 0u), false, "FCOR"},
                {encClip(0x1Cu, 0u, 3u), false, "FCGET"},
            };
            for (const auto &x : expect)
            {
                ++cases;
                if (!classifyLowerPlan(x.instr, p) || p.delaysNextBranchRead != x.delays)
                {
                    if (++mismatches <= 8ul)
                        std::fprintf(stderr,
                                     "[vu1:jit] %s branch-delay flag WRONG (got %d want %d)\n",
                                     x.name, int(p.delaysNextBranchRead), int(x.delays));
                }
            }
        }

        std::fprintf(stderr, "[vu1:jit] lowerVI selftest cases=%lu mismatches=%lu -> %s\n",
                     cases, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        buf.reset();
        return mismatches == 0;
    }

    // ---- FMAC pair self-test ---------------------------------------------------------------
    // Emits a complete FMAC pair and compares it against a C++ reference built from the SAME
    // helpers the interpreter uses (vuNormOperandQuad / vuNormResultQuadValue + float arithmetic),
    // over randomized register state including the interesting bit patterns. Proves the emitted
    // arithmetic, the broadcast, the result clamp and the dest blend before any of it is wired
    // into the run loop.
    typedef void (*PairFn)(void *state);

    inline bool selfTestFmac(CodeBuffer &buf,
                             void (*normOperandQuad)(const float *, float *),
                             void (*normResultValue)(float *, uint8_t))
    {
        if (!buf.ensure(1u << 20))
            return false;

        static const FmacOp ops[5] = {FmacOp::Add, FmacOp::Sub, FmacOp::Mul, FmacOp::Madd, FmacOp::Msub};
        static const char *opNames[5] = {"ADD", "SUB", "MUL", "MADD", "MSUB"};
        static const uint32_t pool[8] = {0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,
                                         0x7F7FFFFFu, 0x00400000u, 0x7F800000u, 0x4B1ED4A8u};

        unsigned long cases = 0, mismatches = 0;
        uint32_t rng = 0x12345678u;
        auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };

        for (int oi = 0; oi < 5; ++oi)
        {
            for (int bc = -1; bc < 4; ++bc)
            {
                for (uint32_t dest = 0; dest < 16u; ++dest)
                {
                    buf.reset();
                    Emitter e;
                    e.p = buf.code;
                    e.end = buf.base + buf.capacity;
                    const int32_t offFs = int32_t(offsetof(FakeState, vf) + 1 * 16);
                    const int32_t offFt = int32_t(offsetof(FakeState, vf) + 2 * 16);
                    const int32_t offFd = int32_t(offsetof(FakeState, vf) + 3 * 16);
                    const int32_t offAcc = int32_t(offsetof(FakeState, acc));
                    UpperPlan tplan;
                    tplan.op = ops[oi];
                    tplan.src = (bc >= 0) ? OperandSrc::VtLane : OperandSrc::VtQuad;
                    tplan.bc = (bc >= 0) ? bc : 0;
                    tplan.clampResult = true;
                    tplan.readsAcc = (ops[oi] == FmacOp::Madd || ops[oi] == FmacOp::Msub);
                    emitUpper(e, buf, tplan, offFs, offFt, offAcc,
                              int32_t(offsetof(FakeState, q)), int32_t(offsetof(FakeState, i)),
                              offFd, uint8_t(dest));
                    e.ret();
                    if (e.overflow)
                    {
                        std::fprintf(stderr, "[vu1:jit] fmac selftest FAILED - overflow\n");
                        return false;
                    }
                    PairFn fn = reinterpret_cast<PairFn>(buf.code);

                    for (int iter = 0; iter < 24; ++iter)
                    {
                        static FakeState a;   // static: guest-thread stack is small (see header note)
                        static FakeState b;
                        a = FakeState{};
                        for (int c = 0; c < 4; ++c)
                        {
                            const uint32_t bs = pool[next() & 7u];
                            const uint32_t bt = pool[next() & 7u];
                            const uint32_t ba = pool[next() & 7u];
                            const uint32_t bd = pool[next() & 7u];
                            std::memcpy(&a.vf[1][c], &bs, 4);
                            std::memcpy(&a.vf[2][c], &bt, 4);
                            std::memcpy(&a.acc[c], &ba, 4);
                            std::memcpy(&a.vf[3][c], &bd, 4);
                        }
                        b = a;

                        fn(&a); // emitted code

                        // --- C++ reference, mirroring the interpreter exactly ---
                        alignas(16) float vs[4], vt[4], ac[4], res[4];
                        normOperandQuad(b.vf[1], vs);
                        normOperandQuad(b.vf[2], vt);
                        normOperandQuad(b.acc, ac);
                        if (bc >= 0)
                        {
                            const float s = vt[bc];
                            vt[0] = vt[1] = vt[2] = vt[3] = s;
                        }
                        for (int c = 0; c < 4; ++c)
                        {
                            switch (ops[oi])
                            {
                            case FmacOp::Add: res[c] = vs[c] + vt[c]; break;
                            case FmacOp::Sub: res[c] = vs[c] - vt[c]; break;
                            case FmacOp::Mul: res[c] = vs[c] * vt[c]; break;
                            case FmacOp::Madd: res[c] = ac[c] + vs[c] * vt[c]; break;
                            case FmacOp::Msub: res[c] = ac[c] - vs[c] * vt[c]; break;
                            }
                        }
                        normResultValue(res, uint8_t(dest));
                        for (int c = 0; c < 4; ++c)
                            if (dest & (1u << (3 - c)))
                                b.vf[3][c] = res[c];

                        for (int c = 0; c < 4; ++c)
                        {
                            uint32_t x = 0, y = 0;
                            std::memcpy(&x, &a.vf[3][c], 4);
                            std::memcpy(&y, &b.vf[3][c], 4);
                            ++cases;
                            if (x != y && ++mismatches <= 8ul)
                                std::fprintf(stderr,
                                             "[vu1:jit] fmac MISMATCH op=%s bc=%d dest=%x lane=%d "
                                             "jit=%08x ref=%08x\n",
                                             opNames[oi], bc, dest, c, x, y);
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "[vu1:jit] fmac selftest cases=%lu mismatches=%lu -> %s\n",
                     cases, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        buf.reset();
        return mismatches == 0;
    }

} // namespace vu1jit

#endif
