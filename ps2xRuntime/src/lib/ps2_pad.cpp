#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <atomic>
#include <chrono>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    // ===== cont.346c: PER-PORT INPUT =========================================================
    // This backend ignored `port` entirely and always read raylib gamepad 0, so BOTH PS2 ports
    // saw the same device -- player 2 mirrored player 1, and local co-op could never work even
    // once the game's own gate was satisfied. PCSX2 models each port independently
    // (pcsx2/SIO/Pad/Pad.cpp: s_controllers[NUM_CONTROLLER_PORTS], GetPad(port, slot),
    // per-port bindings via GetConfigSection(i) -> "Pad{i+1}", and an absent pad is the explicit
    // ControllerType::NotConnected checked by HasConnectedPad()). We mirror that shape.
    //
    // Device policy. Port N takes the N-th available gamepad. For port 1:
    //   auto (default) -- a real second gamepad; else, when a gamepad has player 1, the KEYBOARD
    //                     (whole layout -- rumble branch row 225, user 2026-09-23: "the keyboard
    //                     should default to player 2"). No gamepad at all = keyboard is player 1
    //                     and port 1 is absent, exactly as before.
    //   pad            -- the pre-row-225 auto: present ONLY when a real second gamepad is plugged in.
    //   keys           -- present, driven by the numpad split (co-op on one keyboard).
    //   off            -- never present.
    // ⚠ Under the new default, plugging in ONE gamepad makes port 1 CONNECTED (the keyboard), so the
    // game's own "second pad?" question (0x142C80(6), cont.346d) answers yes and the New/Load menu
    // gains its co-op entry. Accepted by the user; `PS2X_PAD2=pad` restores the solo menu.
    // ⚠ cont.346d: `auto` must NOT fall back to the keyboard. "Is a second pad connected?" is a
    // question the GAME ASKS (0x142C80(6)), and answering yes changes the menu: with slot 1
    // resolved, the New/Load screen gains its co-op entry and a solo player's menu shifts under
    // them. So the keyboard split is OPT-IN -- a player who wants two-on-one-keyboard asks for it.
    int nthGamepad(int n)
    {
        if (n < 0) n = 0;
        int seen = 0;
        for (int i = 0; i < 8; ++i)
            if (IsGamepadAvailable(i))
            {
                if (seen == n) return i;
                ++seen;
            }
        return -1;
    }

    enum class Pad2Mode { Auto, Pad, Keys, Off };
    Pad2Mode pad2Mode()
    {
        static const Pad2Mode m = []
        {
            const char *e = std::getenv("PS2X_PAD2");
            if (!e || !e[0]) return Pad2Mode::Auto;
            if (e[0] == '0' || std::strcmp(e, "off") == 0) return Pad2Mode::Off;
            if (std::strcmp(e, "keys") == 0) return Pad2Mode::Keys;
            if (std::strcmp(e, "pad") == 0) return Pad2Mode::Pad;
            return Pad2Mode::Auto;
        }();
        return m;
    }

    // Is player 2 currently driven by the KEYBOARD (rather than its own gamepad)? When it is,
    // player 1's keypad ALTERNATES (KP_0 = square, KP_1 = triangle) are released to player 2.
    bool pad2UsesKeys()
    {
        // Only in `keys` mode, and only when port 1 has no gamepad of its own to prefer.
        return pad2Mode() == Pad2Mode::Keys && nthGamepad(1) < 0;
    }

    // Is the keyboard player 2 ON ITS OWN -- a gamepad has player 1, there is no second gamepad,
    // and the default policy applies? (The one-keyboard split is pad2UsesKeys, a different case.)
    bool keyboardIsSoloPlayer2()
    {
        return pad2Mode() == Pad2Mode::Auto && nthGamepad(0) >= 0 && nthGamepad(1) < 0;
    }
}

int PSPadBackend::gamepadForPort(int port) { return nthGamepad(port); }

// ===== rumble =================================================================================
// PCSX2 (pcsx2/SIO/Pad/PadDualshock2.cpp, Poll() command bytes 3-4) turns the two actuator bytes
// into InputManager::SetPadVibrationIntensity(port, large, small) with
//     large = largeMotor * (1/255)           -- a 0-255 motor
//     small = (smallMotor & 1) ? 1.0 : 0.0   -- an on/off motor at full power
// and its SDL source drives the large motor as SDL's LOW-frequency rumble, the small as HIGH --
// raylib's SetGamepadVibration(left, right) is exactly (low, high).
// A PS2 motor stays at its last value until the game changes it; SDL rumble needs a duration, so
// a nonzero request is re-issued before it lapses. If the main loop ever stalls, the motor stops
// on its own within kVibHold rather than running forever.
namespace
{
    std::atomic<uint32_t> g_vibWant[2]{};   // (large << 8) | small, written by any thread
    uint32_t g_vibSent[2] = {0u, 0u};       // main thread only
    std::chrono::steady_clock::time_point g_vibRenew[2]{};
    constexpr float kVibHold = 0.25f;                           // seconds per SDL request
    constexpr auto kVibRenewEvery = std::chrono::milliseconds(150);
}

void PSPadBackend::setVibration(int port, uint8_t small, uint8_t large)
{
    if (port < 0 || port > 1) return;
    g_vibWant[port].store((static_cast<uint32_t>(large) << 8) | (small & 1u), std::memory_order_relaxed);
}

void PSPadBackend::pumpVibration()
{
    // Device census (main thread, once per loop): name every gamepad connect/disconnect, so a report
    // of "the pad does nothing" says at once whether it was ever seen, in which slot, and when.
    static uint32_t s_seen = 0u;
    uint32_t mask = 0u;
    for (int i = 0; i < 8; ++i)
        if (IsGamepadAvailable(i)) mask |= 1u << i;
    if (mask != s_seen)
    {
        for (int i = 0; i < 8; ++i)
        {
            const uint32_t bit = 1u << i;
            if ((mask & bit) && !(s_seen & bit))
                std::fprintf(stderr, "[pad:device] gamepad slot %d CONNECTED: %s\n", i, GetGamepadName(i));
            else if (!(mask & bit) && (s_seen & bit))
                std::fprintf(stderr, "[pad:device] gamepad slot %d DISCONNECTED\n", i);
        }
        s_seen = mask;
    }

    const auto now = std::chrono::steady_clock::now();
    for (int port = 0; port < 2; ++port)
    {
        const uint32_t want = g_vibWant[port].load(std::memory_order_relaxed);
        const int gp = nthGamepad(port);
        if (gp < 0)
        {
            g_vibSent[port] = 0u;             // keyboard: no motor -- never faked
            continue;
        }
        if (want == 0u)
        {
            if (g_vibSent[port] != 0u)
            {
                SetGamepadVibration(gp, 0.0f, 0.0f, 0.05f);   // zero intensities = stop
                g_vibSent[port] = 0u;
            }
            continue;
        }
        if (want != g_vibSent[port] || now >= g_vibRenew[port])
        {
            const float large = static_cast<float>(want >> 8) * (1.0f / 255.0f);
            const float small = (want & 1u) ? 1.0f : 0.0f;
            SetGamepadVibration(gp, large, small, kVibHold);
            g_vibSent[port] = want;
            g_vibRenew[port] = now + kVibRenewEvery;
        }
    }
}

// PCSX2's Pad::HasConnectedPad() analogue. Port 0 always exists (keyboard at minimum). Port 1
// exists when a second gamepad is plugged in, or when the keyboard split stands in for one --
// i.e. only when something REAL drives it, never as a bare fake.
static bool padKeys4();   // PS2X_PAD_KEYS4 (defined below)
bool PSPadBackend::portHasDevice(int port)
{
    if (port <= 0) return true;              // port 0 always exists (keyboard at minimum)
    if (padKeys4() && port <= 3) return true; // PS2X_PAD_KEYS4: the keyboard drives ports 0-3
    // 2026-09-25 (rotk 4-player mod): ports 2/3 exist when the 3rd/4th gamepad does, or when a
    // PS2X_PAD_SCRIPT2/3 drives them (headless tests). The original game never opens them.
    if (port >= 2)
    {
        if (port <= 3)
        {
            static const bool s_script[2] = {std::getenv("PS2X_PAD_SCRIPT2") != nullptr,
                                             std::getenv("PS2X_PAD_SCRIPT3") != nullptr};
            if (s_script[port - 2]) return true;
        }
        return nthGamepad(port) >= 0;
    }
    switch (pad2Mode())
    {
        case Pad2Mode::Off:  return false;
        case Pad2Mode::Keys: return true;    // the numpad stands in for a second pad, on request
        case Pad2Mode::Pad:  return nthGamepad(1) >= 0;   // a REAL second gamepad only
        default:             return nthGamepad(1) >= 0 || keyboardIsSoloPlayer2();
    }
}

// The single-player keyboard layout (arrows = d-pad, WASD = left stick, IJKL = right stick,
// ZXCV = the face buttons, ...). Player 1 uses it when no gamepad drives port 0; since the
// rumble branch (row 225) it is also PLAYER 2's layout when a gamepad has player 1 and the
// keyboard is the only thing left. `p2Keys` = the one-keyboard co-op split is live.
static void keyboardFullLayout(uint8_t *data, uint16_t &btns, bool p2Keys)
{
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };
    // cont.232: the ARROW keys are the d-pad and WASD is the left stick (below) -- one physical key,
    // one PS2 input. Before, every arrow/WASD press cleared the d-pad bit AND pushed the stick, so
    // a menu that accepts either input moved twice per press (user: "up/down presses twice").
    // PS2X_PAD_KEYSTICK=0 restores digital-only keys (arrows and WASD both on the d-pad).
    static const bool s_keyStickMap = []
    { const char *e = std::getenv("PS2X_PAD_KEYSTICK"); return !(e && e[0] == '0'); }();
    // cont.346h: with the co-op keyboard split live the ARROWS belong to PLAYER 2 (user request),
    // so player 1's d-pad moves onto WASD -- which also drives its stick, i.e. one cluster per
    // player. That re-binds what cont.232 deliberately separated (arrows = d-pad, WASD = stick,
    // because a menu accepting either moved twice per press), but with two players on one
    // keyboard there is no second cluster to spare. It applies ONLY while the split is active;
    // single player keeps the cont.232 layout exactly.
    // ★ The d-pad/analog SEPARATION is preserved for BOTH players (user, 2026-09-19). It is the
    // cont.232 fix: one cluster driving both makes every menu that accepts either input move TWICE
    // per press. So each player gets two clusters, never one doing double duty. Player 1's d-pad
    // moves to the T/F/G/H inverted-T while WASD stays its analog stick.
    if ((!p2Keys && IsKeyDown(KEY_UP))    || (p2Keys && IsKeyDown(KEY_T)) || (!s_keyStickMap && IsKeyDown(KEY_W)))
        clearBit(PAD_UP);
    if ((!p2Keys && IsKeyDown(KEY_DOWN))  || (p2Keys && IsKeyDown(KEY_G)) || (!s_keyStickMap && IsKeyDown(KEY_S)))
        clearBit(PAD_DOWN);
    if ((!p2Keys && IsKeyDown(KEY_LEFT))  || (p2Keys && IsKeyDown(KEY_F)) || (!s_keyStickMap && IsKeyDown(KEY_A)))
        clearBit(PAD_LEFT);
    if ((!p2Keys && IsKeyDown(KEY_RIGHT)) || (p2Keys && IsKeyDown(KEY_H)) || (!s_keyStickMap && IsKeyDown(KEY_D)))
        clearBit(PAD_RIGHT);
    if (IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE))
        clearBit(PAD_CROSS);
    if (IsKeyDown(KEY_C) || IsKeyDown(KEY_ESCAPE))
        clearBit(PAD_CIRCLE);
    // cont.346c: KP_0 / KP_1 are player 1 ALTERNATES; when the co-op keyboard split is live
    // the numpad belongs to player 2, so they are released.
    if (IsKeyDown(KEY_Z) || (!p2Keys && IsKeyDown(KEY_KP_0)))
        clearBit(PAD_SQUARE);
    if (IsKeyDown(KEY_V) || (!p2Keys && IsKeyDown(KEY_KP_1)))
        clearBit(PAD_TRIANGLE);
    if (IsKeyDown(KEY_Q))
        clearBit(PAD_L1);
    if (IsKeyDown(KEY_E))
        clearBit(PAD_R1);
    if (IsKeyDown(KEY_LEFT_SHIFT))
        clearBit(PAD_L2);
    // cont.346i: RightShift is player 2's START while the split is live, so player 1's R2 also
    // answers to `2`, a left-side key it can always reach.
    if ((!p2Keys && IsKeyDown(KEY_RIGHT_SHIFT)) || IsKeyDown(KEY_TWO))
        clearBit(PAD_R2);
    if (IsKeyDown(KEY_ENTER))
        clearBit(PAD_START);
    if (IsKeyDown(KEY_TAB))
        clearBit(PAD_SELECT);
    // cont.230 PS2X_PAD_KEYSTICK (default ON, "=0" = digital only): the keyboard also drives the
    // ANALOG sticks -- LOTR moves the character with the left stick and never the d-pad, so with
    // digital-only keys Gandalf could not be walked from the keyboard. cont.232: WASD = left stick
    // ONLY and the arrows = d-pad ONLY (0x00 = up/left, 0xFF = down/right);
    // I/J/K/L = right stick (camera). Without a gamepad only.
    static const bool s_keyStick = []
    { const char *e = std::getenv("PS2X_PAD_KEYSTICK"); return !(e && e[0] == '0'); }();
    if (s_keyStick)
    {
        uint8_t lx = kPadStickCenter, ly = kPadStickCenter, rx = kPadStickCenter, ry = kPadStickCenter;
        if (IsKeyDown(KEY_A))  lx = 0x00;
        if (IsKeyDown(KEY_D)) lx = 0xFF;
        if (IsKeyDown(KEY_W))    ly = 0x00;
        if (IsKeyDown(KEY_S))  ly = 0xFF;
        // cont.346i: I/J/K/L is player 2's LEFT stick while the co-op split is live (a laptop has
        // no numpad to give it), so player 1's camera stick steps aside there. Walking is unaffected.
        if (!p2Keys && IsKeyDown(KEY_J)) rx = 0x00;
        if (!p2Keys && IsKeyDown(KEY_L)) rx = 0xFF;
        if (!p2Keys && IsKeyDown(KEY_I)) ry = 0x00;
        if (!p2Keys && IsKeyDown(KEY_K)) ry = 0xFF;
        data[6] = lx;
        data[7] = ly;
        data[4] = rx;
        data[5] = ry;
    }
}

// ★ 2026-09-26 PS2X_PAD_KEYS4=1 (default OFF): FOUR players on ONE keyboard, for testing/recording 4-player
// sessions (rotk mods/fourplayer). Keys are PHYSICAL positions (raylib/GLFW names = US layout); the user's
// ABNT2 labels in brackets. Movement drives the left stick (not the d-pad: a press counted twice in menus);
// buttons: square, triangle, cross,
// circle, start.
//   P1  up 2  left Q  down W  right E | A S Z X | start 1
//   P2  up 5  left R  down T  right Y | F G V B | start 4
//   P3  up 8  left U  down I  right O | J K M , | start 7
//   P4  up -  left P  down [ (ABNT2 ´)  right ] (ABNT2 [) | ; (ç)  ' (~)  \ (])  / (;) | start 0
static bool padKeys4()
{
    static const bool on = [] { const char *e = std::getenv("PS2X_PAD_KEYS4"); return e && e[0] == '1'; }();
    return on;
}
static void keyboardKeys4Layout(int port, uint8_t *data, uint16_t &btns)
{
    struct Map { int up, left, down, right, square, triangle, cross, circle, start; };
    static const Map kMap[4] = {
        {KEY_TWO, KEY_Q, KEY_W, KEY_E, KEY_A, KEY_S, KEY_Z, KEY_X, KEY_ONE},
        {KEY_FIVE, KEY_R, KEY_T, KEY_Y, KEY_F, KEY_G, KEY_V, KEY_B, KEY_FOUR},
        {KEY_EIGHT, KEY_U, KEY_I, KEY_O, KEY_J, KEY_K, KEY_M, KEY_COMMA, KEY_SEVEN},
        {KEY_MINUS, KEY_P, KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET, KEY_SEMICOLON, KEY_APOSTROPHE, KEY_BACKSLASH, KEY_SLASH, KEY_ZERO},
    };
    const Map &m = kMap[port];
    auto clearBit = [&btns](uint16_t mask) { btns &= ~mask; };
    uint8_t lx = kPadStickCenter, ly = kPadStickCenter;
    // stick only: menus act on the d-pad AND the stick, so driving both made one press count twice
    // (user, 2026-09-26) -- the same reason the 1-player layout keeps WASD stick-only (cont.232)
    if (IsKeyDown(m.up))    ly = 0x00;
    if (IsKeyDown(m.down))  ly = 0xFF;
    if (IsKeyDown(m.left))  lx = 0x00;
    if (IsKeyDown(m.right)) lx = 0xFF;
    if (IsKeyDown(m.square))   clearBit(PAD_SQUARE);
    if (IsKeyDown(m.triangle)) clearBit(PAD_TRIANGLE);
    if (IsKeyDown(m.cross))    clearBit(PAD_CROSS);
    if (IsKeyDown(m.circle))   clearBit(PAD_CIRCLE);
    if (IsKeyDown(m.start))    clearBit(PAD_START);
    data[6] = lx;
    data[7] = ly;
    static bool s_announced = false;
    if (!s_announced)
    {
        s_announced = true;
        std::fprintf(stderr, "[pad:keys4] four players on the keyboard: P1 2/QWE + ASZX start 1 | P2 5/RTY + FGVB start 4 | "
                             "P3 8/UIO + JKM, start 7 | P4 -/P[] + ;'\\/ start 0 (physical US positions)\n");
    }
}

bool PSPadBackend::readState(int port, int slot, uint8_t *data, size_t size)
{
    (void)slot;
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;
    // ★ cont.346l: NOTHING may be sampled before raylib's window exists. IsKeyDown/IsGamepad* return
    // garbage then, and the very first delivery happens inside the DBCMAN CreateSocket RPC -- the
    // trace caught socket 1's first payload carrying `lx=ff` (stick hard right) on a run with no key
    // held. A spurious sample at a state change is exactly the shape of the user's phantom presses.
    // Report the neutral state that the caller already memset until the window is up.
    if (!IsWindowReady())
    {
        data[2] = 0xFF; data[3] = 0xFF;
        return true;
    }
    // cont.346c: the N-th gamepad drives port N (was: gamepad 0 drove every port).
    const int kGamepad = nthGamepad(port);
    const bool useGamepad = (kGamepad >= 0);
    const bool p2Keys = pad2UsesKeys();
    if (port >= 1 && !PSPadBackend::portHasDevice(port))
        return false;   // nothing drives this port -- report it as having no pad
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (padKeys4() && port >= 0 && port <= 3)
    {
        keyboardKeys4Layout(port, data, btns);
    }
    else if (useGamepad)
    {
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    else if (port <= 0)
    {
        keyboardFullLayout(data, btns, p2Keys);
    }
    else if (port == 1 && keyboardIsSoloPlayer2())
    {
        // rumble branch (row 225, user 2026-09-23): a gamepad has player 1, so the keyboard is
        // player 2 -- with the whole single-player layout, since nobody else shares it.
        keyboardFullLayout(data, btns, false);
        static bool s_announced = false;
        if (!s_announced)
        {
            s_announced = true;
            std::fprintf(stderr, "[pad2] player 2 = KEYBOARD (player 1 = gamepad): arrows = D-PAD, WASD = stick, X/Space = cross, C/Esc = circle, Z = square, V = triangle, Enter = start. (PS2X_PAD2=pad: keyboard stays off; =off: no player 2)\n");
        }
    }
    else if (port == 1 && p2Keys)
    {
        // ===== cont.346c: PLAYER 2 ON THE NUMPAD ============================================
        // Reached only when port 1 has no gamepad of its own. Deliberately disjoint from player
        // 1's keys (arrows / WASD / IJKL / ZXCV / Q,E / shifts / Enter / Tab / Space / Esc).
        // cont.346h (user request): player 2 drives the ARROW KEYS, with its action buttons on the
        // keys physically around them, so two players share one keyboard without reaching across it.
        //   d-pad (menus)        : the ARROW KEYS
        //   left stick (walking)  : I / J / K / L   (an inverted-T for the right hand)
        //   cross M | circle , | square N | triangle .   (mirrors player 1's Z X C V ordering)
        //   L1 U | R1 O | L2 Y | R2 P | start RightShift | select /
        // ⚠ cont.346i: NO NUMPAD -- the user is on a LAPTOP that has none (2026-09-19). Every
        // player-2 binding is a letter/arrow key present on a laptop; the numpad set below is kept
        // only as a harmless DESKTOP alternate.
        //   the numpad set is kept as an ALTERNATE: KP_8/2/4/6 move, KP_0 cross, KP_. circle,
        //   KP_1 square, KP_3 triangle, KP_7/KP_9 shoulders. PS2X_PAD2_ARROWS=0 drops the arrow half
        //   (numpad only) and hands the arrows back to player 1.
        // Right stick (camera) is intentionally unmapped -- there is no free pair next to the arrows.
        static const bool p2Arrows = []
        { const char *e = std::getenv("PS2X_PAD2_ARROWS"); return !(e && e[0] == '0'); }();
        if (p2Arrows && IsKeyDown(KEY_UP))    clearBit(PAD_UP);
        if (p2Arrows && IsKeyDown(KEY_DOWN))  clearBit(PAD_DOWN);
        if (p2Arrows && IsKeyDown(KEY_LEFT))  clearBit(PAD_LEFT);
        if (p2Arrows && IsKeyDown(KEY_RIGHT)) clearBit(PAD_RIGHT);

        if (IsKeyDown(KEY_M)      || IsKeyDown(KEY_KP_0))          clearBit(PAD_CROSS);
        if (IsKeyDown(KEY_COMMA)  || IsKeyDown(KEY_KP_DECIMAL))    clearBit(PAD_CIRCLE);
        if (IsKeyDown(KEY_N)      || IsKeyDown(KEY_KP_1))          clearBit(PAD_SQUARE);
        if (IsKeyDown(KEY_PERIOD) || IsKeyDown(KEY_KP_3))          clearBit(PAD_TRIANGLE);
        if (IsKeyDown(KEY_U)      || IsKeyDown(KEY_KP_7))          clearBit(PAD_L1);
        if (IsKeyDown(KEY_O)      || IsKeyDown(KEY_KP_9))          clearBit(PAD_R1);
        if (IsKeyDown(KEY_Y)      || IsKeyDown(KEY_KP_DIVIDE))     clearBit(PAD_L2);
        if (IsKeyDown(KEY_P)      || IsKeyDown(KEY_KP_MULTIPLY))   clearBit(PAD_R2);
        if (IsKeyDown(KEY_RIGHT_SHIFT) || IsKeyDown(KEY_KP_ENTER)) clearBit(PAD_START);
        if (IsKeyDown(KEY_SLASH)       || IsKeyDown(KEY_KP_ADD))   clearBit(PAD_SELECT);

        uint8_t lx = kPadStickCenter, ly = kPadStickCenter;
        if (IsKeyDown(KEY_J) || IsKeyDown(KEY_KP_4)) lx = 0x00;
        if (IsKeyDown(KEY_L) || IsKeyDown(KEY_KP_6)) lx = 0xFF;
        if (IsKeyDown(KEY_I) || IsKeyDown(KEY_KP_8)) ly = 0x00;
        if (IsKeyDown(KEY_K) || IsKeyDown(KEY_KP_2)) ly = 0xFF;
        data[6] = lx;
        data[7] = ly;

        static bool s_announced = false;
        if (!s_announced)
        {
            s_announced = true;
            std::fprintf(stderr, "[pad2] player 2 = KEYBOARD: arrows = D-PAD, IJKL = ANALOG stick, M , N . = X O [] ^, U/O = L1/R1, Y/P = L2/R2, RShift = start, / = select. Player 1: WASD = stick, TFGH = d-pad, ZXCV = X O [] ^, Q/E = L1/R1, Enter = start, 2 = R2. No numpad needed. (PS2X_PAD2=off disables)\n");
        }
    }

    // PS2X_PAD_CROSS_AT=<n> (default 0 = off): headless CROSS driver. XTEST key injection is
    // hopelessly racy on a live desktop (focus is stolen by the real user / other windows, so
    // the key silently never reaches raylib -- it has produced repeated false "input doesn't
    // work" results). This forces the CROSS bit pressed for PS2X_PAD_CROSS_LEN readState() calls
    // (default 100, ~3s) starting at call #N, then releases -- a clean press->release edge the
    // menu can act on -- with NO dependency on window focus. Default-OFF so normal runs are
    // untouched. (Diagnostic driving lever; keep it out of shipped behavior.)
    // cont.346c: PORT 0 ONLY. Every driver below keeps a GLOBAL static call counter and pad
    // scripts are indexed by that count, so letting port 1 advance it too would desynchronise
    // every recorded replay (tmp/*.pad).
    if (port <= 0)
    {
        static const int s_crossAt  = [](){ const char *e = std::getenv("PS2X_PAD_CROSS_AT");  return e ? std::atoi(e) : 0; }();
        static const int s_crossLen = [](){ const char *e = std::getenv("PS2X_PAD_CROSS_LEN"); return e ? std::atoi(e) : 100; }();
        if (s_crossAt > 0)
        {
            static unsigned s_padCalls = 0;
            const unsigned c = ++s_padCalls;
            if ((int)c >= s_crossAt && (int)c < s_crossAt + s_crossLen)
            {
                btns &= ~PAD_CROSS;   // active-low: clear = pressed
                static bool s_announced = false;
                if (!s_announced) { s_announced = true;
                    std::fprintf(stderr, "[pad:force] CROSS held from call #%u for %d calls\n", c, s_crossLen); }
            }
        }
    }

    // cont.230: PS2X_PAD_LY_AT=<call> (default 0 = off) / PS2X_PAD_LY_LEN=<calls> (default 100) /
    // PS2X_PAD_LY=<0..255> (default 0x00 = stick fully UP = walk forward) / PS2X_PAD_LX=<0..255>
    // (default 0x80 = centred) / PS2X_PAD_UP=1 (also hold the d-pad UP bit): headless LEFT-STICK
    // driver, the same shape as the CROSS driver above and for the same reason (XTEST is racy on a
    // live desktop). From readState() call #AT for LEN calls the stick bytes are forced, so a run
    // can walk the player without window focus. Diagnostic driving lever; default OFF.
    // cont.346c: PORT 0 ONLY. Every driver below keeps a GLOBAL static call counter and pad
    // scripts are indexed by that count, so letting port 1 advance it too would desynchronise
    // every recorded replay (tmp/*.pad).
    if (port <= 0)
    {
        static const int s_lyAt  = [](){ const char *e = std::getenv("PS2X_PAD_LY_AT");  return e ? std::atoi(e) : 0; }();
        static const int s_lyLen = [](){ const char *e = std::getenv("PS2X_PAD_LY_LEN"); return e ? std::atoi(e) : 100; }();
        static const int s_lyVal = [](){ const char *e = std::getenv("PS2X_PAD_LY");     return e ? std::atoi(e) : 0x00; }();
        static const int s_lxVal = [](){ const char *e = std::getenv("PS2X_PAD_LX");     return e ? std::atoi(e) : 0x80; }();
        static const bool s_upToo = [](){ const char *e = std::getenv("PS2X_PAD_UP");    return e && e[0] == '1'; }();
        if (s_lyAt > 0)
        {
            static unsigned s_stickCalls = 0;
            const unsigned c = ++s_stickCalls;
            if ((int)c >= s_lyAt && (int)c < s_lyAt + s_lyLen)
            {
                data[6] = static_cast<uint8_t>(s_lxVal & 0xFF);
                data[7] = static_cast<uint8_t>(s_lyVal & 0xFF);
                if (s_upToo)
                    btns &= ~PAD_UP;
                static bool s_stickAnnounced = false;
                if (!s_stickAnnounced) { s_stickAnnounced = true;
                    std::fprintf(stderr, "[pad:force] left stick LX=0x%02x LY=0x%02x%s from call #%u for %d calls\n",
                                 s_lxVal & 0xFF, s_lyVal & 0xFF, s_upToo ? " +UP" : "", c, s_lyLen); }
            }
        }
    }

    // ★ cont.232 PS2X_PAD_SCRIPT=<file>: headless input SCRIPT, the general form of the two drivers
    // above (same reason: XTEST is focus-racy). One line per step, applied from readState() call
    // #from until the next step's call:  "<from> <lx> <ly> <rx> <ry> <press>"  -- stick bytes 0..255
    // (0x80 centred, 0x00 up/left), <press> = hex mask of buttons HELD in PS2 bit order (SELECT=1
    // L3=2 R3=4 START=8 UP=10 RIGHT=20 DOWN=40 LEFT=80 L2=100 R2=200 L1=400 R1=800 TRI=1000
    // CIRCLE=2000 CROSS=4000 SQUARE=8000). Lines starting with '#' are comments; steps must be in
    // ascending call order. Announces each step on stderr as [pad:script]. Diagnostic; default OFF.
    // ★★★★★ cont.358t: PORT 0 AND PORT 1. This was port 0 only, because the call counter the script
    // is indexed by was GLOBAL -- letting port 1 advance it would have desynchronised every existing
    // recording. The counter is now PER PORT, which fixes that at the root: port 0's index counts
    // port 0's reads exactly as before (it already only incremented on port 0), so every `tmp/*.pad`
    // replays unchanged, and port 1 gets its own independent index.
    // ⚠ WHY IT MATTERS: co-op CANNOT BE REACHED without it. Character select requires player 2 to
    // pick a character, so a port-0-only recording stalls there forever -- a replay of one spent its
    // whole capture in the menus and never drew a single HUD element (measured, 334k draws, all menu
    // text). Any co-op regression test needs both ports.
    //   port 0:  PS2X_PAD_SCRIPT   / PS2X_PAD_RECORD
    //   port 1:  PS2X_PAD_SCRIPT1  / PS2X_PAD_RECORD1
    //   port 2/3: PS2X_PAD_SCRIPT2 / PS2X_PAD_SCRIPT3 (2026-09-25, rotk 4-player mod tests)
    if (port >= 0 && port <= 3)
    {
        const int pi = port;
        struct Step { unsigned from; uint8_t lx, ly, rx, ry; unsigned press; };
        auto loadSteps = [](const char *var)
        {
            std::vector<Step> v;
            const char *path = std::getenv(var);
            if (!path || !path[0])
                return v;
            if (std::FILE *f = std::fopen(path, "r"))
            {
                char line[256];
                while (std::fgets(line, sizeof(line), f))
                {
                    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
                        continue;
                    unsigned from = 0, lx = 0x80, ly = 0x80, rx = 0x80, ry = 0x80, press = 0;
                    if (std::sscanf(line, "%u %i %i %i %i %x", &from, &lx, &ly, &rx, &ry, &press) >= 1)
                        v.push_back(Step{from, (uint8_t)lx, (uint8_t)ly, (uint8_t)rx, (uint8_t)ry, press});
                }
                std::fclose(f);
                std::fprintf(stderr, "[pad:script] %zu steps from %s\n", v.size(), path);
            }
            else
                std::fprintf(stderr, "[pad:script] cannot open %s\n", path);
            return v;
        };
        static const std::vector<Step> s_stepsPort[4] = { loadSteps("PS2X_PAD_SCRIPT"),
                                                          loadSteps("PS2X_PAD_SCRIPT1"),
                                                          loadSteps("PS2X_PAD_SCRIPT2"),
                                                          loadSteps("PS2X_PAD_SCRIPT3") };
        const std::vector<Step> &s_steps = s_stepsPort[pi];
        if (!s_steps.empty())
        {
            static unsigned s_scriptCalls[4] = {0, 0, 0, 0}, s_cur[4] = {0, 0, 0, 0};
            static bool s_announced[4] = {false, false, false, false};
            const unsigned c = ++s_scriptCalls[pi];
            while (s_cur[pi] + 1 < s_steps.size() && c >= s_steps[s_cur[pi] + 1].from)
            {
                ++s_cur[pi];
                s_announced[pi] = false;
            }
            if (c >= s_steps[s_cur[pi]].from)
            {
                const Step &st = s_steps[s_cur[pi]];
                data[6] = st.lx; data[7] = st.ly; data[4] = st.rx; data[5] = st.ry;
                btns = static_cast<uint16_t>(0xFFFFu & ~st.press); // the script OWNS the buttons (releases the CROSS driver's hold)
                if (!s_announced[pi])
                {
                    s_announced[pi] = true;
                    std::fprintf(stderr, "[pad:script] port %d step %u at call #%u: LX=%02x LY=%02x RX=%02x RY=%02x press=%04x\n",
                                 pi, s_cur[pi], c, st.lx, st.ly, st.rx, st.ry, st.press);
                }
            }
        }
    }

    // ★ cont.237 PS2X_PAD_RECORD=<file>: the INVERSE of PS2X_PAD_SCRIPT -- record what the player
    // actually does, in exactly the format the script replays. Writing pad scripts by hand does not
    // work for anything that needs real play (cont.232 gave up on "scripting past the orcs"; cont.237
    // got a climb only after several calibration failures, and the timing of when the ladder becomes
    // available varies run to run). Recording removes the guesswork: the user plays the sequence once,
    // and the trace replays headlessly forever.
    //
    // Placed AFTER the CROSS / left-stick / script drivers and before the state is published, so it
    // captures the EFFECTIVE pad state whatever produced it. One line per CHANGE (the state is held
    // until the next line, which is precisely the replay semantics), flushed immediately so a run that
    // is killed still leaves a usable file. Buttons are active-low in `btns`, and the script's `press`
    // field is the held-mask, hence the inversion. Diagnostic; default OFF.
    // cont.358t: PORT 0 AND PORT 1, per-port counters (see the script block above). Recording BOTH
    // is what makes a co-op replay possible at all.
    //   port 2/3: PS2X_PAD_RECORD2 / PS2X_PAD_RECORD3 (2026-09-26, 4-player sessions)
    if (port >= 0 && port <= 3)
    {
        const int pi = port;
        auto openRec = [](const char *var, int p) -> std::FILE *
        {
            const char *path = std::getenv(var);
            if (!path || !path[0]) return nullptr;
            std::FILE *f = std::fopen(path, "w");
            if (!f) { std::fprintf(stderr, "[pad:record] cannot open %s\n", path); return nullptr; }
            std::fprintf(f, "# recorded by %s (port %d) -- replay with %s=<this file>\n",
                         var, p, p == 0 ? "PS2X_PAD_SCRIPT" : p == 1 ? "PS2X_PAD_SCRIPT1" : p == 2 ? "PS2X_PAD_SCRIPT2" : "PS2X_PAD_SCRIPT3");
            std::fprintf(f, "# columns: <from-call> <lx> <ly> <rx> <ry> <press hex>  (one line per change)\n");
            std::fflush(f);
            std::fprintf(stderr, "[pad:record] port %d -> %s\n", p, path);
            return f;
        };
        static std::FILE *s_recPort[4] = { openRec("PS2X_PAD_RECORD", 0), openRec("PS2X_PAD_RECORD1", 1),
                                           openRec("PS2X_PAD_RECORD2", 2), openRec("PS2X_PAD_RECORD3", 3) };
        static unsigned s_recCalls[4] = {0, 0, 0, 0};
        const unsigned c = ++s_recCalls[pi];
        if (s_recPort[pi])
        {
            const unsigned press = static_cast<unsigned>((~btns) & 0xFFFFu);
            static int s_lx[4] = {-1,-1,-1,-1}, s_ly[4] = {-1,-1,-1,-1}, s_rx[4] = {-1,-1,-1,-1}, s_ry[4] = {-1,-1,-1,-1};
            static unsigned s_press[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
            if ((int)data[6] != s_lx[pi] || (int)data[7] != s_ly[pi] ||
                (int)data[4] != s_rx[pi] || (int)data[5] != s_ry[pi] || press != s_press[pi])
            {
                s_lx[pi] = data[6]; s_ly[pi] = data[7]; s_rx[pi] = data[4]; s_ry[pi] = data[5];
                s_press[pi] = press;
                std::fprintf(s_recPort[pi], "%u %d %d %d %d %x\n", c, s_lx[pi], s_ly[pi], s_rx[pi], s_ry[pi], press);
                std::fflush(s_recPort[pi]);
            }
        }
    }

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);

    // PS2X_PAD2_LOG=1: trace what the host sampler actually sees (diagnosing
    // "injected keys never reach the pad buffers"). Heartbeat every ~120 calls
    // plus an immediate line on any non-idle button mask.
    {
        static const bool s_log = [](){ const char *e = std::getenv("PS2X_PAD2_LOG"); return e && e[0] == '1'; }();
        if (s_log)
        {
            static unsigned s_calls = 0;
            ++s_calls;
            if (btns != 0xFFFFu)
                std::fprintf(stderr, "[pad:sample] #%u btns=0x%04x gamepad=%d keyDown(down)=%d\n",
                             s_calls, btns, (int)useGamepad, (int)IsKeyDown(KEY_DOWN));
            else if ((s_calls % 120u) == 0u)
                std::fprintf(stderr, "[pad:sample] #%u idle gamepad=%d keyDown(down)=%d keyDown(x)=%d\n",
                             s_calls, (int)useGamepad, (int)IsKeyDown(KEY_DOWN), (int)IsKeyDown(KEY_X));
        }
    }
    return true;
}
