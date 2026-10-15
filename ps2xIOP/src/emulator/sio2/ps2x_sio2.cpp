// The seam between PCSX2's SIO2 + DualShock 2 code (pcsx2/SIO/, verbatim) and ps2xIOP's emulator. It carries the
// pieces of PCSX2 the copied files expect from elsewhere:
//   - ps2/Iop/IopHwRead.cpp + IopHwWrite.cpp: the SIO2 register map (iopHwRead/Write32_Page8, the 8-bit TX/RX);
//   - IopDma.cpp psxDma11 / psxDma12;
//   - SIO/Sio.cpp: mcds/mcd, sioConvertPadToPortAndSlot / sioConvertPortAndSlotToPad, AutoEject/MemcardBusy hooks;
//   - SIO/Pad/Pad.cpp: the pad objects (Pad::GetPad) -- fed from the runtime's pad backend instead of PCSX2's
//     input bindings, once per SIO2 transfer (CTRL write).
#include "ps2x_sio2.h"

#include "SIO/Pad/Pad.h"
#include "SIO/Pad/PadBase.h"
#include "SIO/Pad/PadDualshock2.h"
#include "SIO/Pad/PadNotConnected.h"
#include "SIO/Sio.h"
#include "SIO/Sio2.h"
#include "SIO/SioTypes.h"
#include "SIO/Multitap/MultitapProtocol.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

_mcd mcds[2][4];
_mcd *mcd = &mcds[0][0];
uint32_t sioLastFrameMcdBusy = 0;

namespace
{
    ps2x::sio2::Host *s_host = nullptr;
    constexpr u8 kUnifiedSlots = 8;
    std::unique_ptr<PadBase> s_pads[kUnifiedSlots];
    bool s_connected[2] = {false, false};

    // PS2X_SIO2_LOG=1 (diagnostic, default OFF): the first transfers and every change of port 1A's buttons.
    bool logOn()
    {
        static const bool on = [] { const char *e = std::getenv("PS2X_SIO2_LOG"); return e && e[0] == '1'; }();
        return on;
    }
    uint64_t s_transfers = 0;
    std::string s_txLog, s_rxLog;   // the current transfer's PIO bytes (logged at the next CTRL write)
    uint16_t s_lastButtons[2] = {0xffff, 0xffff};

    constexpr uint32_t HW_SIO2_TX = 0x1F808260u, HW_SIO2_RX = 0x1f808264u, HW_SIO2_CTRL = 0x1f808268u,
                       HW_SIO2_CMD_STAT = 0x1f80826cu, HW_SIO2_PORT_STAT = 0x1f808270u, HW_SIO2_FIFO_STAT = 0x1f808274u,
                       HW_SIO2_FIFO_TX = 0x1F808278u, HW_SIO2_FIFO_RX = 0x1F80827Cu, HW_SIO2_INTR = 0x1f808280u;

    void setPad(u8 unifiedSlot, bool dualshock)
    {
        if (dualshock)
            s_pads[unifiedSlot] = std::make_unique<PadDualshock2>(unifiedSlot, 0);
        else
            s_pads[unifiedSlot] = std::make_unique<PadNotConnected>(unifiedSlot, 0);
    }

    // Ports 1A/2A follow the pad backend (a pad appears/disappears with its driver); slots B-D never exist (no multitap).
    void refreshPads()
    {
        if (!s_host)
            return;
        for (int port = 0; port < 2; ++port)
        {
            uint8_t raw[32] = {};
            const bool present = s_host->padState(port, raw, sizeof(raw));
            if (present != s_connected[port])
            {
                s_connected[port] = present;
                setPad(static_cast<u8>(port), present);
            }
            if (logOn())
            {
                const uint16_t buttons = static_cast<uint16_t>(raw[2] | (raw[3] << 8));
                if (s_transfers < 8 || (present && buttons != s_lastButtons[port]) || (port == 0 && s_transfers % 600 == 0))
                    std::fprintf(stderr, "[sio2:log] transfer=%llu port=%d present=%d buttons=%04x sticks=%02x%02x%02x%02x\n",
                                 static_cast<unsigned long long>(s_transfers), port, present ? 1 : 0, buttons, raw[4], raw[5],
                                 raw[6], raw[7]);
                s_lastButtons[port] = buttons;
            }
            if (!present)
                continue;
            PadBase *pad = s_pads[port].get();
            // libpad bit order, active-low: byte 2 = SELECT L3 R3 START UP RIGHT DOWN LEFT, byte 3 = L2 R2 L1 R1 TRI CIR X SQ.
            static constexpr u32 kByte2[8] = {PadDualshock2::PAD_SELECT, PadDualshock2::PAD_L3, PadDualshock2::PAD_R3,
                                              PadDualshock2::PAD_START, PadDualshock2::PAD_UP, PadDualshock2::PAD_RIGHT,
                                              PadDualshock2::PAD_DOWN, PadDualshock2::PAD_LEFT};
            static constexpr u32 kByte3[8] = {PadDualshock2::PAD_L2, PadDualshock2::PAD_R2, PadDualshock2::PAD_L1,
                                              PadDualshock2::PAD_R1, PadDualshock2::PAD_TRIANGLE, PadDualshock2::PAD_CIRCLE,
                                              PadDualshock2::PAD_CROSS, PadDualshock2::PAD_SQUARE};
            for (int bit = 0; bit < 8; ++bit)
            {
                pad->Set(kByte2[bit], (raw[2] & (1u << bit)) ? 0.0f : 1.0f);
                pad->Set(kByte3[bit], (raw[3] & (1u << bit)) ? 0.0f : 1.0f);
            }
            pad->SetRawAnalogs(std::make_tuple(raw[6], raw[7]), std::make_tuple(raw[4], raw[5]));
        }
    }
}

// ---- PCSX2 SIO/Sio.cpp ------------------------------------------------------------------------------------------
std::tuple<u32, u32> sioConvertPadToPortAndSlot(u32 index)
{
    if (index > 4) // [5,6,7]
        return std::make_tuple(1, index - 4); // 2B,2C,2D
    else if (index > 1) // [2,3,4]
        return std::make_tuple(0, index - 1); // 1B,1C,1D
    else // [0,1]
        return std::make_tuple(index, 0); // 1A,2A
}

u32 sioConvertPortAndSlotToPad(u32 port, u32 slot)
{
    if (slot == 0)
        return port;
    else if (port == 0) // slot=[0,1]
        return slot + 1; // 2,3,4
    else
        return slot + 4; // 5,6,7
}

namespace AutoEject
{
    void SetAll() {}   // savestate-only (ps2x has none)
}

namespace MemcardBusy
{
    void SetBusy() {}   // PCSX2 UI/savestate bookkeeping
}

// ---- PCSX2 SIO/Pad/Pad.cpp ----------------------------------------------------------------------------------------
PadBase *Pad::GetPad(const u8 unifiedSlot)
{
    return unifiedSlot < kUnifiedSlots ? s_pads[unifiedSlot].get() : nullptr;
}

PadBase *Pad::GetPad(u8 port, u8 slot)
{
    return GetPad(static_cast<u8>(sioConvertPortAndSlotToPad(port, slot)));
}

void InputManager::SetPadVibrationIntensity(u32 pad_index, float large_or_single_motor_intensity, float small_motor_intensity)
{
    if (s_host)
        s_host->padVibration(static_cast<int>(pad_index), large_or_single_motor_intensity, small_motor_intensity);
}

// ---- PCSX2 Common.h iopIntcIrq (only Sio2::Interrupt calls it here: IRQ 17) ---------------------------------------
void iopIntcIrq(uint irq)
{
    (void)irq;
    if (s_host)
        s_host->sio2Interrupt();
}

namespace ps2x::sio2
{
    void attach(Host *host)
    {
        s_host = host;
    }

    // PCSX2 SIO2/Pad init (sioInit/Sio2::Initialize, Pad::Initialize): every slot starts without a device.
    void reset()
    {
        for (u8 i = 0; i < kUnifiedSlots; ++i)
            setPad(i, false);
        s_connected[0] = s_connected[1] = false;
        for (auto &mtap : g_MultitapArr)
            mtap.FullReset();
        g_Sio2.Initialize();
    }

    // PCSX2 IopHwRead.cpp iopHwRead32_Page8 (SIO2 part).
    uint32_t read32(uint32_t addr)
    {
        if (addr >= 0x1f808200 && addr <= 0x1f80823f)
            return g_Sio2.CmdQueue[(addr & 0x3f) >> 2];
        if (addr >= 0x1f808240 && addr <= 0x1f80825f)
        {
            const int parm = (addr & 0x18) >> 3;   // SEND1/SEND2 alternate: first reg Send1, second Send2
            return (addr & 4) ? g_Sio2.PortCtrl1[parm] : g_Sio2.PortCtrl0[parm];
        }
        switch (addr)
        {
        case HW_SIO2_TX: return 0;   // PCSX2: "Unexpected 32 bit read of HW_SIO2_DATAIN"
        case HW_SIO2_RX: return 0;   // PCSX2: "Unexpected 32 bit read of HW_SIO2_FIFO"
        case HW_SIO2_CTRL: return g_Sio2.ctrl;
        case HW_SIO2_CMD_STAT: return g_Sio2.CmdStat;
        case HW_SIO2_PORT_STAT: return g_Sio2.PortStat;
        case HW_SIO2_FIFO_STAT: return g_Sio2.FifoStat;
        case HW_SIO2_FIFO_TX: return g_Sio2.FifoTxPos;
        case HW_SIO2_FIFO_RX: return g_Sio2.FifoRxPos;
        case HW_SIO2_INTR: return g_Sio2.iStat;
        default: return 0;
        }
    }

    // PCSX2 IopHwWrite.cpp iopHwWrite32_Page8 (SIO2 part).
    void write32(uint32_t addr, uint32_t val)
    {
        if (addr >= 0x1f808200 && addr <= 0x1f80823f)
        {
            g_Sio2.SetCmd((addr & 0x3f) >> 2, val);   // SEND3
            return;
        }
        if (addr >= 0x1f808240 && addr <= 0x1f80825f)
        {
            const int parm = (addr & 0x18) >> 3;
            if (addr & 4)
                g_Sio2.PortCtrl1[parm] = val;   // SEND2
            else
                g_Sio2.PortCtrl0[parm] = val;   // SEND1
            return;
        }
        switch (addr)
        {
        case HW_SIO2_CTRL:
            if (val & Sio2Ctrl::START_TRANSFER)
            {
                refreshPads();   // ps2x: sample the pad backend once per SIO2 transfer, before the packet is answered
                ++s_transfers;
            }
            if (logOn() && s_transfers <= 40)
            {
                std::fprintf(stderr, "[sio2:log] CTRL=%08x cmd0=%08x cmd1=%08x cmdstat=%08x | prev tx:%s | rx:%s\n", val,
                             g_Sio2.CmdQueue[0], g_Sio2.CmdQueue[1], g_Sio2.CmdStat, s_txLog.c_str(), s_rxLog.c_str());
                s_txLog.clear();
                s_rxLog.clear();
            }
            g_Sio2.SetCtrl(val);
            break;
        case HW_SIO2_CMD_STAT: g_Sio2.CmdStat = val; break;
        case HW_SIO2_PORT_STAT: g_Sio2.PortStat = val; break;
        case HW_SIO2_FIFO_STAT: g_Sio2.FifoStat = val; break;
        case HW_SIO2_FIFO_TX: g_Sio2.FifoTxPos = val; break;
        case HW_SIO2_FIFO_RX: g_Sio2.FifoRxPos = val; break;
        case HW_SIO2_INTR: g_Sio2.iStat &= ~val; break;
        default: break;   // TX/RX 32-bit writes and the rest: no-ops in PCSX2
        }
    }

    // PCSX2 iopHwRead8_Page8 / iopHwWrite8_Page8: RX pops a response byte, TX pushes a command byte.
    uint8_t read8(uint32_t addr)
    {
        if (addr == HW_SIO2_RX)
        {
            const uint8_t byte = g_Sio2.Read();
            if (logOn() && s_transfers <= 40)
            {
                char b[4];
                std::snprintf(b, sizeof(b), " %02x", byte);
                s_rxLog += b;
            }
            return byte;
        }
        return static_cast<uint8_t>(read32(addr & ~3u) >> ((addr & 3u) * 8u));
    }

    void write8(uint32_t addr, uint8_t value)
    {
        if (addr == HW_SIO2_TX)
        {
            if (logOn() && s_transfers <= 40)
            {
                char b[4];
                std::snprintf(b, sizeof(b), " %02x", value);
                s_txLog += b;
            }
            g_Sio2.Write(value);
        }
    }

    uint32_t dma11(uint32_t &madr, uint32_t bcr, uint32_t chcr)
    {
        const uint32_t size = (bcr >> 16) * (bcr & 0xffff);
        // Set dmaBlockSize, so SIO2 knows to count based on the DMA block rather than SEND3 length.
        g_Sio2.dmaBlockSize = (bcr & 0xffff) * 4;
        if (chcr != 0x01000201 || !s_host)
            return 0;
        const uint8_t *ram = s_host->iopRam();
        const bool log = logOn() && s_transfers <= 40;
        if (log)
            std::fprintf(stderr, "[sio2:log] dma11 bcr=%08x in:", bcr);
        for (uint32_t i = 0; i < (bcr >> 16); i++)
            for (uint32_t j = 0; j < ((bcr & 0xFFFF) * 4); j++)
            {
                const uint8_t byte = ram[madr++ & 0x1FFFFFu];
                if (log)
                    std::fprintf(stderr, " %02x", byte);
                g_Sio2.Write(byte);
            }
        if (log)
            std::fprintf(stderr, "\n");
        return size;
    }

    uint32_t dma12(uint32_t &madr, uint32_t bcr, uint32_t chcr)
    {
        const uint32_t size = ((bcr >> 16) * (bcr & 0xFFFF)) * 4;
        if (chcr != 0x41000200 || !s_host)
            return 0;
        uint8_t *ram = s_host->iopRam();
        const bool log = logOn() && s_transfers <= 40;
        if (log)
            std::fprintf(stderr, "[sio2:log] dma12 bcr=%08x out:", bcr);
        for (uint32_t n = size; n > 0; n--)
        {
            const uint8_t byte = g_Sio2.Read();
            if (log)
                std::fprintf(stderr, " %02x", byte);
            ram[madr++ & 0x1FFFFFu] = byte;
        }
        if (log)
            std::fprintf(stderr, " | cmdstat=%08x\n", g_Sio2.CmdStat);
        return size >> 2;
    }
}
