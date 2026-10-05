#include <cstdio>   // cont.346j: [padsock]
#include "runtime/ps2_dbcman_hle.h"
#include "ps2_runtime.h"
#include "runtime/ps2_pad.h"   // cont.346k: PSPadBackend::portHasDevice

#include <cstdint>
#include <cstring>
#include <cstdlib>

// DBCMAN pad HLE (runtime side). See ps2_dbcman_hle.h for why this lives in the runtime rather than
// the game override: the per-vblank deliverPadData() must run at the vblank-interrupt-worker
// frequency (a guest override hook fires too rarely under present starvation, so the work-table
// re-assertion is lost and the controller reads "removed" -> self-exit). Logic ported from
// patch-05; default-on env gates PS2X_PAD2_HLE/_WORKTBL/_DELIVER preserved.

namespace
{
    constexpr uint32_t kRpcCheckVersion = 0x80001363u;
    constexpr uint32_t kEventSid1 = 0x8000131bu, kEventSid2 = 0x8000131cu;
    constexpr uint32_t kRpcInit = 0x80001300u, kRpcCreateSocket = 0x80001301u, kRpcDeleteSocket = 0x80001302u,
                       kRpcGetDepNumber = 0x80001303u, kRpcSetWorkAddr = 0x80001304u, kRpcInitSocket = 0x80001315u,
                       kRpcGetDeviceStatus = 0x80001317u, kRpcSRData = 0x80001318u, kRpcReceiveData = 0x8000131au,
                       kRpcSearchPortSpecific = 0x8000131du;
    constexpr int kMaxSockets = 8;
    constexpr uint8_t kProfile[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0x00};
    constexpr uint32_t kPadDataLen = 18u, kProfileLen = 5u, kHalfSeqOffset = 124u;
    constexpr uint8_t kPadTypeAnalog = 0x73; // DS2 analog-mode id (desc+0x1A gate)

    struct PadSocket { bool active = false; uint32_t buf0 = 0, buf1 = 0; };
    uint32_t g_padWorkAddr = 0u;
    PadSocket g_padSockets[kMaxSockets];
    int g_padSocketCount = 0;
    unsigned long g_padSeq = 0u;

    inline bool padEnvFlag(const char *n) { const char *v = std::getenv(n); return !(v && v[0] == '0'); }
    inline bool padHleEnabled() { static const bool e = padEnvFlag("PS2X_PAD2_HLE"); return e; }
    inline bool padWorkTblEnabled() { static const bool e = padEnvFlag("PS2X_PAD2_WORKTBL"); return e; }
    inline bool padDeliverEnabled() { static const bool e = padEnvFlag("PS2X_PAD2_DELIVER"); return e; }

    inline uint32_t padRd32(uint8_t *rd, uint32_t a) { if (!a) return 0u; uint32_t v; std::memcpy(&v, rd + (a & 0x01FFFFFFu), 4); return v; }
    inline void padWr32(uint8_t *rd, uint32_t a, uint32_t v) { if (a) std::memcpy(rd + (a & 0x01FFFFFFu), &v, 4); }

    void writeVersionResponse(uint8_t *rd, uint32_t recv, uint32_t recvSize)
    {
        const uint32_t ver = 0x0320u; uint32_t n = recvSize / 4u; if (n > 4u) n = 4u; if (n == 0u) n = 1u;
        for (uint32_t i = 0; i < n; ++i) padWr32(rd, recv + i * 4u, ver);
    }

    void buildPadPayload(PS2Runtime *runtime, int socketIndex, uint8_t out[32])
    {
        uint8_t raw[32]; bool haveState = false;
        // ★★★ cont.346j: socket N takes PAD PORT N. This read was hardcoded to `socketIndex == 0`
        // with port (0,0), so socket 1 -- PLAYER 2 -- always fell through to the neutral payload
        // below and could never receive a single button press, whatever the keyboard mapping said.
        // ⚠ This is the ONLY path this game's pad data travels: it never calls the EE libpad API
        // (cont.346f measured ZERO scePad* calls) and Pad.cpp's scePadRead backend read is a
        // different call site the game never reaches. So the cont.346h/i player-2 key layouts were
        // wired to a function nobody called -- the keys were dead on arrival, not mis-chosen.
        // 2026-09-25: every socket reads its port (was `< 2`). The original game only ever opens
        // sockets 0/1, so it is unaffected; a mod that opens a third/fourth socket (rotk
        // mods/fourplayer) gets pad port 2/3 = the 3rd/4th gamepad, and a port with no device still
        // reports "no pad" (PSPadBackend::readState -> portHasDevice), exactly like before.
        if (runtime && socketIndex >= 0 && socketIndex < kMaxSockets)
            haveState = runtime->padBackend().readState(socketIndex, 0, raw, sizeof(raw));
        {   // one line per socket, so a run says which sockets exist and which carry LIVE input
            static bool announced[kMaxSockets] = {};
            if (socketIndex >= 0 && socketIndex < kMaxSockets && !announced[socketIndex])
            {
                announced[socketIndex] = true;
                std::fprintf(stderr, "[padsock] socket %d -> pad port %d, input=%s\n",
                             socketIndex, socketIndex, haveState ? "LIVE" : "neutral (no device)");
            }
        }
        if (!haveState) { std::memset(raw, 0, sizeof(raw)); raw[2] = 0xFF; raw[3] = 0xFF; raw[4] = raw[5] = raw[6] = raw[7] = 0x80; }
        out[0] = raw[2]; out[1] = raw[3]; out[2] = raw[4]; out[3] = raw[5]; out[4] = raw[6]; out[5] = raw[7];
        const uint16_t btns = (uint16_t)(raw[2] | (raw[3] << 8));
        const uint16_t pm[12] = {0x0020, 0x0080, 0x0010, 0x0040, 0x1000, 0x2000, 0x4000, 0x8000, 0x0400, 0x0800, 0x0100, 0x0200};
        for (int p = 0; p < 12; ++p) out[6 + p] = (btns & pm[p]) ? 0x00 : 0xFF;
        std::memcpy(out + kPadDataLen, kProfile, kProfileLen);
    }

    void writeHalfBuffer(uint8_t *rd, uint32_t half, const uint8_t *payload, uint32_t seq)
    {
        if (!half) return; uint8_t *p = rd + (half & 0x01FFFFFFu);
        p[0] = 1; p[1] = 0; p[2] = (uint8_t)kPadDataLen; p[3] = (uint8_t)kProfileLen;
        const uint32_t total = kPadDataLen + kProfileLen; std::memcpy(p + 4, &total, 4);
        std::memcpy(p + 28, payload, total); std::memcpy(p + kHalfSeqOffset, &seq, 4);
    }

    // cont.346n PS2X_PAD2_WORKDUMP=<n> (default 0 = off): print the 16 work-table words every n
    // deliveries. If the GAME writes into that table, the pattern reveals its real layout -- which is
    // better evidence than any of our guesses about what index 1 means.
    void dumpWorkTable(uint8_t *rd)
    {
        static const long every = []
        { const char *e = std::getenv("PS2X_PAD2_WORKDUMP"); return e ? std::strtol(e, nullptr, 0) : 0; }();
        if (every <= 0 || !g_padWorkAddr) return;
        static unsigned long n = 0; static uint32_t prev[16]; static bool have = false;
        if ((++n % (unsigned long)every) != 0u) return;
        uint32_t cur[16]; bool changed = !have;
        for (int i = 0; i < 16; ++i) { cur[i] = padRd32(rd, g_padWorkAddr + i * 4u); if (have && cur[i] != prev[i]) changed = true; }
        if (!changed) return;
        std::fprintf(stderr, "[padworkdump] vb~%lu:", n);
        for (int i = 0; i < 16; ++i) std::fprintf(stderr, " [%d]=0x%x", i, cur[i]);
        std::fprintf(stderr, "\n");
        for (int i = 0; i < 16; ++i) prev[i] = cur[i];
        have = true;
    }

    void deliverToSocket(uint8_t *rd, PS2Runtime *runtime, int idx)
    {
        dumpWorkTable(rd);
        const PadSocket &s = g_padSockets[idx];
        if (!s.active || !s.buf0 || !s.buf1 || !padDeliverEnabled()) return;
        uint8_t payload[32]; buildPadPayload(runtime, idx, payload);
        const uint32_t seq0 = padRd32(rd, s.buf0 + kHalfSeqOffset), seq1 = padRd32(rd, s.buf1 + kHalfSeqOffset);
        const uint32_t target = (seq0 <= seq1) ? s.buf0 : s.buf1;
        writeHalfBuffer(rd, target, payload, (uint32_t)(++g_padSeq));
        // cont.346k: prove DELIVERY per socket, and show the button word actually written -- a socket
        // can be LIVE at the source and still never reach the game. Logged on CHANGE (plus the first
        // few), so a held key shows up as one line rather than a flood.
        {
            static uint16_t lastBtn[kMaxSockets]; static bool seen[kMaxSockets] = {}; static int n[kMaxSockets] = {};
            static unsigned long vb[kMaxSockets] = {}; static int tail[kMaxSockets] = {}; static int emitted[kMaxSockets] = {};
            const uint16_t btn = (uint16_t)(payload[0] | (payload[1] << 8));
            if (idx >= 0 && idx < kMaxSockets)
            {
                ++vb[idx];
                const bool changed = !seen[idx] || btn != lastBtn[idx];
                // cont.346l: "log on change" cannot show a TEMPORAL pattern -- a clean held press and
                // press/release chatter look identical in it. On every change, also dump the next 8
                // deliveries with their vblank index and the half-buffer written, so the sequence the
                // game actually receives is visible. Bounded (600 lines/socket). PS2X_PAD2_TRACE=0 mutes.
                static const bool trace = []
                { const char *e = std::getenv("PS2X_PAD2_TRACE"); return !(e && e[0] == '0'); }();
                if (changed) tail[idx] = 8;
                const bool want = changed || (trace && tail[idx] > 0);
                if (changed || tail[idx] > 0) { if (!changed) --tail[idx]; }
                if (want && emitted[idx] < 600)
                {
                    ++emitted[idx];
                    std::fprintf(stderr, "[paddeliv] socket %d vb=%lu -> 0x%x btn=0x%04x lx=%02x ly=%02x%s\n",
                                 idx, vb[idx], target, btn, payload[4], payload[5], changed ? "  <-- CHANGE" : "");
                }
                else if (!seen[idx] || ++n[idx] <= 2)
                {
                    std::fprintf(stderr, "[paddeliv] socket %d vb=%lu -> 0x%x btn=0x%04x lx=%02x ly=%02x\n",
                                 idx, vb[idx], target, btn, payload[4], payload[5]);
                }
                seen[idx] = true; lastBtn[idx] = btn;
            }
        }
    }
}

namespace ps2_dbcman_hle
{
    bool answerDbcManRpc(uint8_t *rd, PS2Runtime *runtime, uint32_t sid, uint32_t rpcNum,
                         uint32_t sendBuf, uint32_t /*sendSize*/, uint32_t recvBuf, uint32_t recvSize,
                         uint32_t &resultPtr)
    {
        resultPtr = recvBuf;
        if (sid == kEventSid1 || sid == kEventSid2) return true; // event channels: nothing to deliver
        if (!padHleEnabled()) { if (rpcNum == kRpcCheckVersion) writeVersionResponse(rd, recvBuf, recvSize); return true; }
        switch (rpcNum)
        {
        case kRpcInit: return true;
        case kRpcCheckVersion: writeVersionResponse(rd, recvBuf, recvSize); return true;
        case kRpcSetWorkAddr:
        {
            g_padWorkAddr = padRd32(rd, sendBuf + 4);
            // ★★★ cont.346k: the WORK TABLE is the game's "which ports have a pad" map, and it was
            // hardcoded to port 0 present / ports 1-15 ABSENT. So even with socket 1 created for port 1
            // (the CreateSocket request says so: sendBuf[8] = 0 for socket 0, 1 for socket 1), fed LIVE
            // keyboard state and delivered every vblank, the game still read "no controller in port 1"
            // -- player 2 reached its select screen and could not be controlled.
            // Port 1 is now marked present exactly when something real drives it, so single player is
            // unchanged (PSPadBackend::portHasDevice(1) is false with one pad and PS2X_PAD2=auto).
            if (g_padWorkAddr && padWorkTblEnabled())
            {
                // ⚠ cont.346m: the work table's SEMANTICS are not established. The legacy code wrote
                // [0]=1 and zeroed the rest, which we read as "port 0 present"; it could equally be a
                // COUNT at [0] with per-port info after it. Marking port 1 present (cont.346k) is what
                // finally gave player 2 input, and the menu oscillation appeared in the SAME build --
                // so these are now A/B-able without a rebuild instead of guessed at:
                //   PS2X_PAD2_WORK0=<v>  word [0]              (default 1; try 2 if [0] is a COUNT)
                //   PS2X_PAD2_WORK1=<v>  word [1] = port 1     (default 1 when a device backs port 1;
                //                                               =0 restores the pre-346k table)
                static const long w0 = []
                { const char *e = std::getenv("PS2X_PAD2_WORK0"); return e ? std::strtol(e, nullptr, 0) : 1; }();
                static const bool w1set = std::getenv("PS2X_PAD2_WORK1") != nullptr;
                static const long w1env = []
                { const char *e = std::getenv("PS2X_PAD2_WORK1"); return e ? std::strtol(e, nullptr, 0) : 1; }();
                const uint32_t w1 = (uint32_t)(w1set ? w1env : (PSPadBackend::portHasDevice(1) ? 1 : 0));
                // ⚠ cont.346n: WHICH index means "port 1" is unknown. The table is 16 words, which is
                // exactly 2 ports x 8 multitap slots -- if it is indexed port*8 + slot then port 1 is
                // index 8, and marking index 1 declares "port 0, SLOT 1", a phantom multitap pad on
                // player 1's port. The user's A/B pinned the co-op-menu oscillation to this write
                // ([1]=1 oscillates, [1]=0 is steady but player 2 is ignored), so the index is the
                // next thing to vary. PS2X_PAD2_WORKIDX=<i> (default 1; try 8).
                static const long widx = []
                { const char *e = std::getenv("PS2X_PAD2_WORKIDX"); return e ? std::strtol(e, nullptr, 0) : 1; }();
                padWr32(rd, g_padWorkAddr, (uint32_t)w0);
                for (uint32_t i = 1; i < 16; ++i)
                {
                    // 2026-09-25: ports 2/3 (mods only -- the original game never opens them) are
                    // marked present the same way port 1 is, when a device backs them.
                    const bool extra = (i == 2u || i == 3u) && PSPadBackend::portHasDevice((int)i);
                    padWr32(rd, g_padWorkAddr + i * 4u, (long)i == widx ? w1 : (extra ? 1u : 0u));
                }
                std::fprintf(stderr, "[padwork] table 0x%x: [0]=%ld [%ld]=%u (rest 0)  "
                                     "(PS2X_PAD2_WORK0 / _WORK1 / _WORKIDX override)\n",
                             g_padWorkAddr, w0, widx, w1);
            }
            return true;
        }
        case kRpcCreateSocket:
        {
            int idx = g_padSocketCount < kMaxSockets ? g_padSocketCount++ : kMaxSockets - 1;
            g_padSockets[idx].active = true; g_padSockets[idx].buf0 = padRd32(rd, sendBuf + 40); g_padSockets[idx].buf1 = padRd32(rd, sendBuf + 44);
            // cont.346k: socket -> PORT is currently ASSUMED to be allocation order (socket N = port N).
            // Player 2 reached its select screen with socket 1 LIVE and still did not respond, so that
            // assumption is the next thing to check: dump the whole CreateSocket request and let the
            // GAME say which port/slot it is opening, instead of inferring it from the order of calls.
            {
                std::fprintf(stderr, "[padsock:create] idx=%d buf0=0x%x buf1=0x%x | sendBuf words:",
                             idx, g_padSockets[idx].buf0, g_padSockets[idx].buf1);
                for (uint32_t w = 0; w < 16; ++w) std::fprintf(stderr, " [%u]=0x%x", w * 4u, padRd32(rd, sendBuf + w * 4u));
                std::fprintf(stderr, "\n");
            }
            padWr32(rd, recvBuf + 36, (uint32_t)idx); deliverToSocket(rd, runtime, idx); return true;
        }
        case kRpcDeleteSocket: { const uint32_t idx = padRd32(rd, sendBuf); if (idx < (uint32_t)kMaxSockets) g_padSockets[idx] = PadSocket{}; return true; }
        case kRpcGetDepNumber: padWr32(rd, recvBuf + 4, 0u); return true;
        case kRpcInitSocket: case kRpcGetDeviceStatus: case kRpcSearchPortSpecific: padWr32(rd, recvBuf + 4, 0u); return true;
        case kRpcReceiveData:
        {
            const uint32_t mode = padRd32(rd, sendBuf + 4); uint32_t len = 0; uint8_t payload[16] = {0};
            switch (mode & 0xFFu)
            {
            case 0x02u: len = 1; payload[0] = kPadTypeAnalog; break; // controller-type -> DS2 analog
            case 0x0Cu: len = 1; payload[0] = 1; break;             // pad state -> connected
            default: len = 0; break;
            }
            padWr32(rd, recvBuf + 8, len); if (len > 0) std::memcpy(rd + ((recvBuf + 12) & 0x01FFFFFFu), payload, len);
            padWr32(rd, recvBuf + 0x20C, 0u); return true;
        }
        case kRpcSRData: padWr32(rd, recvBuf + 12, 0u); padWr32(rd, recvBuf + 0x410, 0u); return true;
        default: return true; // generic ack (resultPtr = recv already set)
        }
    }

    void deliverPadData(uint8_t *rdram, PS2Runtime *runtime)
    {
        if (!rdram || !padHleEnabled()) return;
        // Re-assert the per-port connection table every tick (the game wipes this BSS during load, so
        // a one-shot SetWorkAddr write is lost -> GetDepNumber fails -> controller "removed").
        if (g_padWorkAddr && padWorkTblEnabled()) padWr32(rdram, g_padWorkAddr, 1u);
        for (int i = 0; i < g_padSocketCount; ++i) deliverToSocket(rdram, runtime, i);
    }

    void reset()
    {
        g_padWorkAddr = 0u;
        for (auto &s : g_padSockets) s = PadSocket{};
        g_padSocketCount = 0;
        g_padSeq = 0u;
    }
}
