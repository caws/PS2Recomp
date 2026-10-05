#include "runtime/ps2_vu1.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "ps2_vu1_detail.h"
#include "ps2_vu1_jit.h"

// FTOI saturation trap (PS2X_VU1_FTOITRAP, default OFF): the level-era garbage cascade carries
// saturated-int fingerprints; name the FIRST saturating FTOIs (program pc + input bits).
static void logFtoiSaturation(uint32_t pc, unsigned long long cycle, float input, int component, float scale,
                              const uint8_t *code, uint32_t codeSize)
{
    static const bool s_on = []
    { const char *e = std::getenv("PS2X_VU1_FTOITRAP"); return e && e[0] && e[0] != '0'; }();
    if (!s_on)
        return;
    static unsigned long s_n = 0;
    ++s_n;
    if (s_n <= 24u || (s_n % 65536u) == 0u)
    {
        uint32_t bits;
        std::memcpy(&bits, &input, 4);
        std::fprintf(stderr, "[VU1 ftoi-sat] n=%lu pc=0x%x cycle=%llu in=0x%08x(%g) c=%d scale=%g\n",
                     s_n, pc, cycle, bits, input, component, scale);
        if (s_n == 1u && code)
        {
            const uint32_t base = (pc >= 24u) ? (pc - 24u) & ~7u : 0u;
            for (uint32_t off = 0; off < 48u && base + off + 8u <= codeSize; off += 8u)
            {
                uint32_t lo = 0, hi = 0;
                std::memcpy(&lo, code + base + off, 4u);
                std::memcpy(&hi, code + base + off + 4u, 4u);
                std::fprintf(stderr, "    code 0x%04x: upper=%08x lower=%08x%s\n",
                             base + off, hi, lo, (base + off == pc) ? "  <= ftoi" : "");
            }
        }
    }
}


#include <cmath>
#include <cstring>
#include <limits>

namespace
{
    int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;

    // cont.183: JIT fast path. The compiled function reproduces exactly what this routine plus
    // applyFmacDest(Acc) do for the covered FMAC ops — operand clamp, broadcast, arithmetic,
    // result clamp, dest-lane blend, store — so everything else (the write pipeline, run()'s
    // shadow dance, cycle accounting, the lower instruction) is untouched. Only valid while lazy
    // flags is active, because the emitted code deliberately derives no MAC/status flags.
    if (vu1jit::enabled() && vu1jit::lazyActive() && !vu1jit::inVerify())
    {
        if (vu1jit::UpperFn fn = vu1jit::lookup(instr))
        {
            if (!vu1jit::verifyMode())
            {
                fn(&m_state);
                return;
            }
            // Shadow verify against live guest data: run the JIT, capture what it wrote, restore,
            // then re-enter this function with the JIT suppressed so the interpreter produces the
            // authoritative result, and compare. The recursion guard is what makes "run both" work
            // without duplicating the whole opcode switch.
            const uint8_t jd = FD(instr);
            alignas(16) float savedVf[4], savedAcc[4], jitVf[4], jitAcc[4];
            std::memcpy(savedVf, m_state.vf[jd], sizeof(savedVf));
            std::memcpy(savedAcc, m_state.acc, sizeof(savedAcc));
            fn(&m_state);
            std::memcpy(jitVf, m_state.vf[jd], sizeof(jitVf));
            std::memcpy(jitAcc, m_state.acc, sizeof(jitAcc));
            std::memcpy(m_state.vf[jd], savedVf, sizeof(savedVf));
            std::memcpy(m_state.acc, savedAcc, sizeof(savedAcc));

            vu1jit::inVerify() = true;
            execUpper(instr); // authoritative interpreter result
            vu1jit::inVerify() = false;

            vu1jit::noteVerify(instr, jitVf, jitAcc, m_state.vf[jd], m_state.acc);
            return;
        }
    }

    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    // PS2X_VU1_FASTUPPER (cont.178; default ON, "=0" reverts): this prologue used to normalise
    // FOURTEEN floats — vs[4], vt[4], acc[4], q and i — unconditionally, before even looking at
    // the opcode. The cont.178 profile put normalizeOperand at 10% of the EE thread (plus
    // fmacNormOperand 4%), and the cont.178 opcode census showed **26.4% of all executed pairs
    // have NO upper operation at all** (special 0x2F/0x30 = NOP), so a quarter of that work was
    // being done for instructions that return immediately below. Two skips, both bit-exact
    // because the skipped values are provably never read:
    //   (1) upper NOP returns before the prologue entirely;
    //   (2) acc is normalised only for the ops that READ it — exactly the MADD/MSUB product-sum
    //       family, the same predicate calculateFmacProductSticky already uses (the only `acc[c]`
    //       readers in this file are those cases, verified by inspection of every use).
    // Levels (default 2 = both skips on): 0 = off (old behaviour, for A/B), 1 = NOP skip only,
    // 2 = NOP + acc skip. The split levels exist because a single dark screenshot briefly looked
    // like a geometry regression here; it was NOT — a faster build simply sits at a DIFFERENT game
    // moment at the same wall-clock offset, and a longer burst rendered a fully detailed scene.
    // Keep the levels: they make that kind of question answerable inside one binary.
    // Both skips are sound by construction, verified mechanically against every case label:
    //   - the only `acc[...]` readers in this file are exactly {0x08-0x0F, 0x21, 0x23, 0x25, 0x27,
    //     0x29, 0x2D, 0x2E} in the main switch and {0x08-0x0F, 0x21, 0x23, 0x25, 0x27, 0x29, 0x2D}
    //     in the special switch — an exact match for `readsAcc` below;
    //   - the special 0x2F/0x30 (NOP) case body is a bare `return`, and the prologue it now skips
    //     writes nothing but locals (m_currentUpperInstruction is assigned above it).
    static const int s_fastUpper = []
    { const char *e = std::getenv("PS2X_VU1_FASTUPPER"); return (e && e[0]) ? std::atoi(e) : 2; }();

    const uint8_t specialOp = op >= 0x3Cu
                                  ? static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu))
                                  : 0xFFu;
    if (s_fastUpper >= 1 && (specialOp == 0x2Fu || specialOp == 0x30u))
        return; // NOP: nothing below reads vs/vt/acc/q/i

    const bool readsAcc =
        s_fastUpper < 2 ||
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (specialOp >= 0x08u && specialOp <= 0x0Fu) ||
        specialOp == 0x21u || specialOp == 0x23u || specialOp == 0x25u ||
        specialOp == 0x27u || specialOp == 0x29u || specialOp == 0x2Du;

    // PS2X_VU1_SIMD (cont.181, default ON): the operand clamp is a pure function of each lane's
    // bits (cont.180), so the whole quad is one SSE sequence instead of four scalar calls with
    // their memcpy round-trips. Bit-identical to normalizeOperand by construction — same masks,
    // same order of tests — and `=0` keeps the scalar loop reachable for A/B in one binary.
    static const bool s_simd = []
    { const char *e = std::getenv("PS2X_VU1_SIMD"); return !(e && e[0] == '0'); }();

    float *vd = m_state.vf[fd];
    alignas(16) float normalizedVs[4];
    alignas(16) float normalizedVt[4];
    alignas(16) float normalizedAcc[4];
    if (s_simd)
    {
        vuNormOperandQuad(m_state.vf[fs], normalizedVs);
        vuNormOperandQuad(m_state.vf[ft], normalizedVt);
        if (readsAcc)
            vuNormOperandQuad(m_state.acc, normalizedAcc);
    }
    else
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            normalizedVs[component] = normalizeOperand(m_state.vf[fs][component]);
            normalizedVt[component] = normalizeOperand(m_state.vf[ft][component]);
            if (readsAcc)
                normalizedAcc[component] = normalizeOperand(m_state.acc[component]);
        }
    }
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                if (iv == INT32_MAX || iv == INT32_MIN)
                    logFtoiSaturation(m_state.pc, static_cast<unsigned long long>(m_cycle), vs[c], c, 1.0f, m_cachedVuCode, m_cachedCodeSize);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                if (iv == INT32_MAX || iv == INT32_MIN)
                    logFtoiSaturation(m_state.pc, static_cast<unsigned long long>(m_cycle), vs[c], c, 16.0f, m_cachedVuCode, m_cachedCodeSize);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                if (iv == INT32_MAX || iv == INT32_MIN)
                    logFtoiSaturation(m_state.pc, static_cast<unsigned long long>(m_cycle), vs[c], c, 4096.0f, m_cachedVuCode, m_cachedCodeSize);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                if (iv == INT32_MAX || iv == INT32_MIN)
                    logFtoiSaturation(m_state.pc, static_cast<unsigned long long>(m_cycle), vs[c], c, 32768.0f, m_cachedVuCode, m_cachedCodeSize);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc(result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}
