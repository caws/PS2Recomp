#include "ps2_runtime.h"
#include "ps2_shutdown.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_cpu_backend.h" // cont.232: ps2xGsPresentSourceFresh (presenter cadence)
#include "ipu/ps2x_ipu.h"              // cont.245: the PCSX2 IPU port
#include "ps2_user_dir.h"                // Stage 1 item 6.4: the user-state directory
#include "ps2_debug_panel.h"             // cont.232: g_ps2xDebugUiVisible (the lazy presenter loop)
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_pipe_capture.h"
#include "runtime/ps2_gs_pipeline.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "runtime/ps2_pad.h"       // rumble: PSPadBackend::pumpVibration (main thread)
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
// ★★★ cont.332c PS2X_WINDOW_SCALE (default 1 = today's 640x448 window): open the window at N times
// the guest's display size. The draw already aspect-fits to GetScreenWidth/Height and the window is
// resizable, so this only chooses the starting size -- but without it nothing can SHOW a
// higher-resolution present, because the window is exactly one guest pixel per host pixel.
static int ps2xWindowScale()
{
    static const int s = []
    {
        const char *e = std::getenv("PS2X_WINDOW_SCALE");
        const int v = (e && e[0]) ? std::atoi(e) : 1;
        return v < 1 ? 1 : (v > 4 ? 4 : v);
    }();
    return s;
}
// ★★ cont.332c PS2X_WINDOW_FILTER (default OFF = raylib's default POINT filter, i.e. what every
// screenshot in this project so far shows): bilinear on the window blit. Only matters when the
// window is larger than the presented image -- with PS2X_GS_PRESENT_HIRES at a matching scale the
// blit is 1:1 and this changes nothing.
// ★★★★ cont.332d PS2X_WINDOW_ASPECT (default ON = the CORRECT shape; "=0" restores the old
// pixel-square fit): draw the presented frame at the aspect the DISPLAY registers imply instead of
// at the framebuffer's own width:height. The CRTC magnifies the source across a fixed raster, so
// this game's 512x511 display belongs in the same 4:3 rect a 640x512 one would -- presenting it 1:1
// made every character ~33% too narrow, which is a bigger visual error than the resolution it was
// being rendered at. ps2xGsPresentAspect() does the register maths (PCSX2 GSState.cpp
// VideoModeOffsets / VideoModeDividers).
// cont.332f: it also takes an EXPLICIT ratio now -- "0" off, unset/"1" = derive it from the DISPLAY
// registers (the default), and anything > 1.1 = that aspect, e.g. 1.7778 for the widescreen hack
// (the game's own horizontal FOV has to be widened to match, or the picture is merely stretched:
// game-side LOTR_WIDESCREEN).
// cont.356: no longer static -- the GL 2D/HUD counter-scale needs the PRESENTED ratio, and it has
// to be THE SAME value the window fit uses. A second env read in the device would be a second
// place stating the ratio, which is exactly how these drift apart.
double ps2xWindowAspect()
{
    static const double a = []
    {
        const char *e = std::getenv("PS2X_WINDOW_ASPECT");
        if (!e || !e[0])
            return 1.0;
        const double v = std::atof(e);
        if (v > 1.1)
            return v;
        return (e[0] == '0') ? 0.0 : 1.0;
    }();
    return a;
}
static bool ps2xWindowFilter()
{
    static const bool f = []
    { const char *e = std::getenv("PS2X_WINDOW_FILTER"); return e && e[0] && e[0] != '0'; }();
    return f;
}
// ★★★★ cont.332c: the hi-res present texture (PS2X_GS_PRESENT_HIRES). Owned by the presenter
// thread, created lazily at the scene resolution the GL device reports and recreated if it changes.
// Kept beside the 640x448 path rather than replacing it: the display-sized buffer still feeds the
// present cache, field presentation and PS2X_GS_PRESENT_SAVE.
static Texture2D s_hiresTex{};
static bool s_hiresTexValid = false;
static uint32_t s_hiresTexW = 0, s_hiresTexH = 0;
static bool s_hiresLive = false; // the last UploadFrame had a hi-res frame to show
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    std::filesystem::path defaultMcRoot(const std::filesystem::path &elfDirectory); // defined below

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            // The same resolution as after loadELF, not `<cwd>/mc0`: initialize() resets the IOP, whose
            // mcserv reset calls sceMcInit -> ensureMcRootExists BEFORE configureIoPathsFromElf runs, so a
            // cwd default left empty mc0/ mc1/ in whatever folder the game was launched from.
            defaults.mcRoot = defaultMcRoot(defaults.elfDirectory);
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

// Returns true when a NEW frame was latched and uploaded this call (cont.232: the lazy loop's draw cue).
static bool UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick;
    // ★ cont.232 PS2X_GS_PRESENT_ONFLIP (default ON; "=0" restores the per-vsync-tick latch): the
    // presenter decoded + uploaded on EVERY guest vsync tick (50 Hz, i.e. at whatever rate this loop
    // ran, ~30-40 Hz) although the frame-complete copy it renders changes once per guest FRAME
    // (~8 fps in the level): ~3.5 presents and two field decodes per guest frame, each with a 4 MB
    // VRAM snapshot copy, on a fifth thread over four busy cores -- with the presenter dead the level
    // ran ~15% faster (cont.231 §12). Now a fresh frame-complete copy (the flip snapshot, or the
    // per-Sync capture) is presented ONCE, when it arrives; the per-tick cadence remains only while
    // the copy is stale (Present() then reads live VRAM, which does change between ticks). The
    // window loop still redraws the last texture every iteration (input pump, resize, debug UI).
    static const bool s_presentOnFlip = []
    { const char *e = std::getenv("PS2X_GS_PRESENT_ONFLIP"); return !(e && e[0] == '0'); }();
    static uint64_t s_lastSourceSeq = 0u;
    if (s_presentOnFlip && s_hasLatchedInitialFrame)
    {
        uint64_t seq = 0u;
        if (ps2xGsPresentSourceFresh(seq))
        {
            needsLatch = seq != s_lastSourceSeq;
            if (needsLatch)
                s_lastSourceSeq = seq;
        }
    }
    if (needsLatch)
    {
        rt->gs().latchHostPresentationFrame();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    else if (s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return false;
    }

    // ★★★★ cont.332c: if the GL renderer latched a scene-resolution frame, show THAT. The
    // display-sized path below still runs (it owns `width`/`height`, the present cache and every
    // other consumer); this only redirects what the window blit samples.
    s_hiresLive = false;
    {
        static std::vector<uint8_t> s_hiresPixels;
        uint32_t hw = 0u, hh = 0u;
        if (ps2xGsTakeHiresPresentFrame(s_hiresPixels, hw, hh) && hw != 0u && hh != 0u)
        {
            if (!s_hiresTexValid || s_hiresTexW != hw || s_hiresTexH != hh)
            {
                if (s_hiresTexValid)
                    UnloadTexture(s_hiresTex);
                Image im = GenImageColor(static_cast<int>(hw), static_cast<int>(hh), BLANK);
                s_hiresTex = LoadTextureFromImage(im);
                UnloadImage(im);
                s_hiresTexValid = true;
                s_hiresTexW = hw;
                s_hiresTexH = hh;
                if (ps2xWindowFilter())
                    SetTextureFilter(s_hiresTex, TEXTURE_FILTER_BILINEAR);
                std::fprintf(stderr, "[present:hires] texture %ux%u\n", hw, hh);
            }
            UpdateTexture(s_hiresTex, s_hiresPixels.data());
            s_hiresLive = true;
        }
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return true;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
    return true;
}

// ★ cont.326: the rasterizer unit is compiled with AVX2 (PS2X_RASTER_AVX2, CMake); refuse a host without it with a
// message instead of a SIGILL. A static initializer: it runs before main(), so before any backend object exists.
#if defined(PS2X_RASTER_AVX2) && (defined(__x86_64__) || defined(__i386__))
static const bool s_rasterAvx2Checked = []
{
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2"))
    {
        std::fprintf(stderr, "[runtime] this build's CPU rasterizer needs AVX2 (PS2X_RASTER_AVX2=ON); rebuild with -DPS2X_RASTER_AVX2=OFF\n");
        std::abort();
    }
    return true;
}();
#endif

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);

    m_eeScheduler = std::make_unique<EeScheduler>(*this);

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_asyncCallbackStackTop = PS2_RAM_SIZE;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            // Retire streams BEFORE the device: CloseAudioDevice() uninits AUDIO.System.lock
            // (raudio.c), and UnloadAudioBuffer -> UntrackAudioBuffer locks it, so unloading
            // afterwards would take a destroyed mutex.
            m_audioBackend.streamStopAll();
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModule(std::string_view path, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModule(path, arguments, argumentSize);
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModuleBuffer(guestAddress, arguments, argumentSize);
}

bool PS2Runtime::stopIopModule(int32_t moduleId, int32_t *result)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->stopModule(moduleId, result);
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

bool PS2Runtime::canBindIopRpc(uint32_t sid) const noexcept
{
    return m_iopSubsystem->canBindRpc(sid);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

bool PS2Runtime::sendIopSifCommand(uint32_t cid, const void *packet, uint32_t packetSize,
                                   uint32_t eeExtraSource, uint32_t iopExtraDestination, uint32_t extraSize)
{
    if (!m_iopSubsystem || !m_iopSifCommandForwarding)
        return false;
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->sendSifCommand(cid, packet, packetSize, eeExtraSource, iopExtraDestination, extraSize);
}

void PS2Runtime::advanceIopEeCycles(uint64_t eeCycles) noexcept
{
    m_iopSubsystem->runEeCycles(eeCycles);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

uint32_t PS2Runtime::allocateIopMemory(uint32_t size, uint32_t alignment)
{
    return m_iopSubsystem ? m_iopSubsystem->allocateMemory(size, alignment) : 0u;
}

bool PS2Runtime::freeIopMemory(uint32_t address)
{
    return m_iopSubsystem && m_iopSubsystem->freeMemory(address);
}

bool PS2Runtime::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->readMemory(address, destination, size);
}

bool PS2Runtime::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->writeMemory(address, source, size);
}

bool PS2Runtime::zeroIopMemory(uint32_t address, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->zeroMemory(address, size);
}

bool PS2Runtime::isIopMemoryRange(uint32_t address, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->isMemoryRange(address, size);
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size); });
    m_gifArbiter.setDrainDoneFn([this] { m_gs.endPacket(); }); // cont.230 PS2X_GS_DRAWRUN
    ps2gs::setParseGs(&m_gs); // rotk row 253: the GS the PS2X_GS_SPLIT parse thread feeds
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     // cont.317 stage 2: on the pipeline thread the scheduler's current context is the
                                     // EE's live state -- read FBRST from the main context (one ctc2 in this game) and
                                     // do not write VPU_STAT back (nothing reads it: no cfc2 in the generated code).
                                     R5900Context *cpuContext = (!ps2gs::threaded() && m_eeScheduler) ? m_eeScheduler->currentContext() : nullptr;
                                     if (!cpuContext)
                                     {
                                         cpuContext = &m_cpuContext;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                   m_gs, &m_memory, startPC, top, itop, 65536);
                                     if (!ps2gs::threaded())
                                     cpuContext->vu0_vpu_stat =
                                         (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                         (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                         (m_vu1.state().stoppedByT ? 0x0400u : 0u); });
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     // cont.317 stage 2: on the pipeline thread the scheduler's current context is the
                                     // EE's live state -- read FBRST from the main context (one ctc2 in this game) and
                                     // do not write VPU_STAT back (nothing reads it: no cfc2 in the generated code).
                                     R5900Context *cpuContext = (!ps2gs::threaded() && m_eeScheduler) ? m_eeScheduler->currentContext() : nullptr;
                                     if (!cpuContext)
                                     {
                                         cpuContext = &m_cpuContext;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                  m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                  m_gs, &m_memory, top, itop, 65536);
                                     if (!ps2gs::threaded())
                                     cpuContext->vu0_vpu_stat =
                                         (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                         (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                         (m_vu1.state().stoppedByT ? 0x0400u : 0u); });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }
        // cont.245: the PCSX2 IPU (PS2X_IPU, default on) binds its DMA to guest memory here.
        if (ps2x_ipu::enabled())
            ps2x_ipu::init(&m_memory, m_memory.getRDRAM(), m_memory.getScratchpad());

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        // cont.231 PS2X_WINDOW_ALWAYS_RUN (default ON; "=0" restores raylib's pause): raylib parks the
        // main loop in glfwWaitEvents() while the window is MINIMIZED (no input pump, no presents) unless
        // FLAG_WINDOW_ALWAYS_RUN is set. The guest threads keep running regardless, so a minimized run
        // silently loses its presenter (three cont.231 timing runs did, and only a gdb backtrace of the
        // main thread -- poll < glfwWaitEventsX11 < PollInputEvents < EndDrawing -- said why). An
        // emulator should keep presenting when minimized; it also keeps measurements comparable.
        static const bool s_alwaysRun = []
        { const char *e = std::getenv("PS2X_WINDOW_ALWAYS_RUN"); return !(e && e[0] == '0'); }();
        // rumble branch (row 224): on raylib's SDL platform, SDL DROPS gamepad events while the window
        // is unfocused (SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS defaults to 0); GLFW polled the pad
        // regardless of focus. The user's first SDL run had a dead pad at the language screen. SDL reads
        // hints from the environment at SDL_Init, so set it here, before InitWindow, and never override a
        // value the user exported. Harmless on GLFW (nothing reads it).
        setenv("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1", 0);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | (s_alwaysRun ? FLAG_WINDOW_ALWAYS_RUN : 0u));
        InitWindow(HOST_WINDOW_WIDTH * ps2xWindowScale(), HOST_WINDOW_HEIGHT * ps2xWindowScale(),
                   title);
        // PS2X_AUDIO (default ON since cont.332q; "=0" skips the host audio device). It was default OFF
        // from cont.245 on the premise below -- "the game has no audible output yet" -- which stopped being
        // true at cont.282: the game drives a real 44.1 kHz PCM stream sink (ps2_audio.cpp streamOpen /
        // streamPushAdpcm), and a fight replay opens 5 streams. PCSX2 likewise brings its audio backend up
        // by default (SPU2 + cubeb), so an emulator that boots silent is the odd one out.
        // ⚠ The original reason to default it off is REAL and unfixed, just rarer: raylib's InitAudioDevice
        // runs on THIS thread right after the window exists. When the desktop's audio server is degraded (2026-09-05: the laptop's SOF DSP failed to
        // boot after a resume, PipeWire's sink sat in an error state, `pactl list sinks` segfaulted) the
        // PulseAudio connect wait inside miniaudio (ma_wait_for_pa_stream_to_connect__pulse: loop until
        // READY or FAILED) hangs for ~30 s, the window answers no ping and GNOME shows "not responding".
        // `PS2X_AUDIO=0` is the escape hatch for exactly that: it skips the device and the audio backend
        // simply reports not-ready. If a launch ever hangs ~30 s with the window unresponsive, try it.
        static const bool s_audio = []
        { const char *e = std::getenv("PS2X_AUDIO"); return !(e && e[0] == '0'); }();
        if (s_audio)
        {
            InitAudioDevice();
            m_audioBackend.setAudioReady(IsAudioDeviceReady());
        }
        else
        {
            std::fprintf(stderr, "[audio] host audio device NOT opened (PS2X_AUDIO=0; the game is silent)\n");
            m_audioBackend.setAudioReady(false);
        }
#endif
        SetTargetFPS(60);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

// cont.232: defined in Kernel/EeScheduler.cpp -- the console's boot-time video region.
void ps2xSetDefaultVideoRegion(bool pal) noexcept;

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);
    {
        // Region from the disc ID in the ELF name (SLES/SCES/SLED = Europe = PAL; SLUS/SCUS/SLPS/SLPM/SCPS/
        // SLKA = NTSC), the same inference an OSD makes from its BIOS region. PS2X_REGION overrides.
        std::string base = elfPath;
        const size_t slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        for (char &c : base) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const bool pal = base.rfind("SLES", 0) == 0 || base.rfind("SCES", 0) == 0 || base.rfind("SLED", 0) == 0 ||
                         base.rfind("SCED", 0) == 0 || base.rfind("TCES", 0) == 0;
        ps2xSetDefaultVideoRegion(pal);
        std::fprintf(stderr, "[vblank] ELF %s -> console region default %s (PS2X_REGION=PAL|NTSC overrides)\n",
                     base.c_str(), pal ? "PAL 50 Hz" : "NTSC 59.94 Hz");
    }

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
        m_asyncCallbackStackFloor = std::min(std::max(hardLimit, suggestedHeapBase), PS2_RAM_SIZE);
        m_asyncCallbackStackTop = PS2_RAM_SIZE;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string romError;
    if (!m_romDevice.configure(identity, &romError))
    {
        std::cerr << "[ROM0] failed to configure profile: " << romError << std::endl;
        return false;
    }

    m_iopSubsystem->reset();

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

namespace
{
    // ★ cont.346p PS2X_MC_ROOT=<dir>: the directory that CONTAINS `mc0/` and `mc1/`. Unset = the ELF's
    // own directory, which is exactly the pre-346p behaviour, so nothing changes for anyone who does
    // not set it.
    // Why it exists: the ELF now lives INSIDE `gamefiles/` so a player can drop their whole disc into
    // one folder (user, 2026-09-19). Memory cards must not be FORCED to follow the ELF, because the
    // two-card test harness gives each mirror its own `mc0` while both share one `gamefiles/` -- and
    // if the card root followed the ELF, every mirror would resolve to the same card and the
    // save-present vs fresh-card distinction (a standing validation rule) would silently collapse.
    std::filesystem::path mcRootFromEnv()
    {
        const char *e = std::getenv("PS2X_MC_ROOT");
        if (!e || !e[0])
        {
            return {};
        }
        return std::filesystem::path(e) / "mc0";
    }

    // ★ cont.346q (and the default again since 2026-10-01, user; 2026-09-25..10-01 it was the platform
    // config folder -- ps2_user_dir.h): the directory holding the RUNNING EXECUTABLE. User state -- memory cards now, the
    // settings layer later -- lives in `<exe dir>/config/`, which is the "portable" convention both
    // comparable recompilation projects support (Zelda 64: Recompiled and Unleashed Recompiled both
    // ship a `portable.txt` that does exactly this).
    // Why not beside the ELF: `gamefiles/` is the player's DISC DUMP -- replaceable and re-droppable.
    // People re-dump when a rip was bad, and the natural way is to wipe the folder and re-copy, which
    // would take their saves with it. Saves have a different lifetime, so they get a different home.
    // Resolution order: PS2X_MC_ROOT (explicit) -> the user directory (<exe dir>/config, or
    // PS2X_USER_DIR; ps2_user_dir.h) -> the ELF's directory
    // (the pre-346q behaviour, kept as a last resort when no user directory can be determined).
    std::filesystem::path defaultMcRoot(const std::filesystem::path &elfDirectory)
    {
        const std::filesystem::path fromEnv = mcRootFromEnv();
        if (!fromEnv.empty())
        {
            return fromEnv;
        }
        const std::filesystem::path userDir = ps2x::userdir::directory();
        if (!userDir.empty())
        {
            return userDir / "mc0";
        }
        return elfDirectory / "mc0";
    }
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalizeAbsolutePath(defaultMcRoot(normalized.elfDirectory));
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        // host: and cdrom: SHOULD follow the ELF -- with the ELF inside gamefiles/ they now point at
        // the disc content, which is more faithful than pointing at the game folder.
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = defaultMcRoot(paths.elfDirectory);
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

// ---- PS2X_DISPATCH_UNWIND_FIX (cont.249, default ON; `=0` restores the old behaviour for A/B):
// propagate a checkpoint unwind through dispatchGuestBranch instead of mistaking it for a completed
// call whose callee left ctx->pc untouched. See dispatchGuestBranch(). Validated 2026-09-06: the same
// 600 s play run that died at `source=0x15bbd8 target=0x15a320` with the fix off survived the identical
// event with it on, and reached 1.8x as many hero updates.
static const bool s_dispatchUnwindFix = []
{
    const char *e = std::getenv("PS2X_DISPATCH_UNWIND_FIX");
    return !(e && e[0] == '0');
}();

// ---- PS2X_WRITE_WATCH (cont.248): see ps2TraceGuestWrite in ps2_runtime.h.
uint32_t g_ps2WriteWatchAddr = []
{
    const char *e = std::getenv("PS2X_WRITE_WATCH");
    if (!e || !e[0])
    {
        return 0u;
    }
    return static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) & 0x01FFFFFFu;
}();

uint32_t g_ps2WriteWatchLen = []
{
    const char *e = std::getenv("PS2X_WRITE_WATCH_LEN");
    const uint32_t v = (e && e[0]) ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 4u;
    return v == 0u ? 4u : v;
}();

uint32_t g_ps2WriteWatchPc = []
{
    const char *e = std::getenv("PS2X_WRITE_WATCH_PC");
    return (e && e[0]) ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 0u;
}();
unsigned long g_ps2DispatchSeq = 0;

namespace
{
struct WriteWatchHit
{
    unsigned long seq = 0;
    unsigned long dispatch = 0;
    uint32_t pc = 0, sp = 0, ra = 0, addr = 0, size = 0;
    uint64_t lo = 0;
    const char *op = "";
};
WriteWatchHit g_writeWatchRing[256];
unsigned long g_writeWatchSeq = 0;

// cont.249: the ring rolls over long before a crash (29 M hits in a 10-minute run), so it cannot answer
// "who last wrote THIS word". Keep a per-word last-writer slot for the whole watched range instead: one
// entry per guest word, overwritten in place, so the crash dump can name the writer of the exact saved-ra
// slot the failing `jr ra` read -- regardless of how long ago it was written.
std::vector<WriteWatchHit> g_writeWatchWords;

// Sized once from the flags above (same TU, so those initializers have already run).
const bool g_writeWatchWordsInit = []
{
    if (g_ps2WriteWatchAddr != 0u && g_ps2WriteWatchLen <= 0x100000u)
    {
        g_writeWatchWords.assign((g_ps2WriteWatchLen + 3u) / 4u, WriteWatchHit{});
    }
    return true;
}();

void ps2WriteWatchRecordWords(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi,
                              const char *op, const R5900Context *ctx, unsigned long seq);
} // namespace

void ps2WriteWatchHit(uint8_t *rdram, uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi,
                      const char *op, const R5900Context *ctx)
{
    (void)rdram;
    (void)valueHi;
    WriteWatchHit &h = g_writeWatchRing[g_writeWatchSeq % 256];
    h.seq = ++g_writeWatchSeq;
    ps2WriteWatchRecordWords(guestAddr, size, valueLo, valueHi, op, ctx, h.seq);
    h.dispatch = g_ps2DispatchSeq;
    h.pc = ctx ? ctx->pc : 0u;
    h.sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;
    h.ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
    h.addr = guestAddr;
    h.size = size;
    h.lo = valueLo;
    h.op = op;
    if (g_writeWatchSeq <= 8 || (g_ps2WriteWatchPc != 0u && h.pc == g_ps2WriteWatchPc && g_writeWatchSeq % 64 == 0))
    {
        std::fprintf(stderr, "[writewatch] #%lu d%lu %s 0x%x size=%u value=0x%llx at guest pc=0x%x sp=0x%x ra=0x%x\n",
                     h.seq, h.dispatch, op, guestAddr, size, (unsigned long long)valueLo, h.pc, h.sp, h.ra);
    }
}

namespace
{
void ps2WriteWatchRecordWords(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi,
                              const char *op, const R5900Context *ctx, unsigned long seq)
{
    if (g_writeWatchWords.empty())
    {
        return;
    }
    const uint32_t base = g_ps2WriteWatchAddr;
    const uint32_t a = guestAddr & 0x01FFFFFFu;
    for (uint32_t off = 0; off < size; off += 4u)
    {
        const uint32_t w = a + off;
        if (w < base || w >= base + g_ps2WriteWatchLen)
        {
            continue;
        }
        WriteWatchHit &e = g_writeWatchWords[(w - base) / 4u];
        e.seq = seq;
        e.dispatch = g_ps2DispatchSeq;
        e.pc = ctx ? ctx->pc : 0u;
        e.sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;
        e.ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
        e.addr = w;
        e.size = size;
        e.lo = (off < 8u) ? (valueLo >> (off * 8u)) : (valueHi >> ((off - 8u) * 8u));
        e.op = op;
    }
}
} // namespace

// Name the last writer of every word in [from, to) -- called with the crashing frame's range so the log
// says who put the bogus value in the saved-ra slot.
void ps2WriteWatchDumpRange(uint32_t from, uint32_t to)
{
    if (g_writeWatchWords.empty())
    {
        return;
    }
    const uint32_t base = g_ps2WriteWatchAddr;
    std::fprintf(stderr, "[writewatch] last writer of each word in [0x%x, 0x%x):\n", from, to);
    for (uint32_t w = from & ~3u; w < to; w += 4u)
    {
        if (w < base || w >= base + g_ps2WriteWatchLen)
        {
            continue;
        }
        const WriteWatchHit &e = g_writeWatchWords[(w - base) / 4u];
        if (e.seq == 0u)
        {
            std::fprintf(stderr, "[writewatch]   0x%x: never written while watched\n", w);
            continue;
        }
        std::fprintf(stderr, "[writewatch]   0x%x = 0x%08x  <- #%lu d%lu %s size=%u at guest pc=0x%x sp=0x%x ra=0x%x\n",
                     w, static_cast<uint32_t>(e.lo), e.seq, e.dispatch, e.op, e.size, e.pc, e.sp, e.ra);
    }
}

void ps2WriteWatchDump()
{
    if (g_ps2WriteWatchAddr == 0u)
    {
        return;
    }
    std::fprintf(stderr, "[writewatch] watched [0x%x, +0x%x): %lu writes; the last ones (oldest first):\n",
                 g_ps2WriteWatchAddr, g_ps2WriteWatchLen, g_writeWatchSeq);
    const unsigned long from = g_writeWatchSeq > 256 ? g_writeWatchSeq - 256 + 1 : 1;
    for (unsigned long q = from; q <= g_writeWatchSeq; ++q)
    {
        const WriteWatchHit &h = g_writeWatchRing[(q - 1) % 256];
        std::fprintf(stderr, "[writewatch]   #%lu d%lu %s 0x%x size=%u value=0x%llx at guest pc=0x%x sp=0x%x ra=0x%x\n",
                     h.seq, h.dispatch, h.op, h.addr, h.size, (unsigned long long)h.lo, h.pc, h.sp, h.ra);
    }
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    return generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            return fn;
        }
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    // cont.248: a `jr $ra` with ra == 0 is the runtime's own completion sentinel (top-level invocations and
    // host-pumped chains return to 0), not a missing target -- under PS2X_STRICT_RETURN_DIAGNOSTICS it used to
    // spend the one-shot report at boot, hiding the first REAL bad return. Nothing to report for it.
    if (kind == GuestBranchKind::Return && targetPc == 0u)
    {
        ctx->pc = targetPc;
        return;
    }
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        // 2026-10-02: in the runner (ps2_shutdown.h) a missing target ENDS the process with exit 3: the
        // guest state is already corrupt, and continuing ran away in memory until the machine locked up.
        if (ps2x::shutdown::fatalMissingTarget() && firstReport)
        {
            std::fprintf(stderr, "[runtime:fatal] unrecoverable missing branch target 0x%x (from 0x%x, %s) -- "
                                 "stopping, exit %d\n",
                         targetPc, sourcePc, debugName ? debugName : "?", ps2x::shutdown::kExitMissingTarget);
            ps2x::shutdown::setExitCode(ps2x::shutdown::kExitMissingTarget);
            ps2x::shutdown::armWatchdog("missing branch target");
        }
        requestStop();
    }
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);

    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        return false;
    }

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        return false;
    }

    // cont.317 (cont.): ONE table walk per call. hasFunction() and lookupFunction() each recomputed
    // the slot; the lookup's dispatch-history push is kept, after the found check, as before.
    RecompiledFunction targetFn = nullptr;
    {
        uint32_t slot = 0u;
        if (generatedFunctionTableSlot(targetPc, slot))
            targetFn = g_ps2RecompiledFunctionTable[slot];
    }
    if (targetFn == nullptr)
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            // if you need the app to keep open to open debug pannel change this to false
            return false;
        }

        return false;
    }
    pushDispatchPc(targetPc);
    const uint32_t entryPc = ctx->pc;
    // cont.249: a checkpoint inside the nested call ABANDONS the chain and unwinds to the run loop
    // (see the block comment atop EeScheduler.cpp). Sample the checkpoint epoch across the call so
    // the "ctx->pc == entryPc" test below can tell the two cases apart.
    const uint64_t epochBefore = m_eeScheduler ? m_eeScheduler->checkpointEpoch() : 0ull;
    targetFn(rdram, ctx, this);

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        // Two very different situations land here:
        //  (a) the target is a game OVERRIDE hook that never touches ctx->pc -- it ran, and the
        //      caller must continue at its fallthrough. This is the case the line was written for.
        //  (b) a checkpoint fired at/below this call and unwound the host stack with ctx->pc left
        //      on an in-flight target that HAPPENS to equal this call's target -- which only occurs
        //      when the same address is in flight at two recursion depths (the LOTR scene-graph walk
        //      0x15a320 <-> 0x159a30 does exactly this). Treating (b) as a fallthrough resumes the
        //      OUTER caller inline while sp still belongs to the INNER frame; the caller then runs
        //      its epilogue on a foreign frame and `jr ra` loads a saved s-register as its return
        //      address. That is the "EE scheduler missing-target" failure.
        const bool unwound = m_eeScheduler && m_eeScheduler->checkpointEpoch() != epochBefore;
        if (unwound)
        {
            static std::atomic<unsigned long> s_swallowed{0};
            const unsigned long n = ++s_swallowed;
            if (n <= 16u || (n & 0xFFFu) == 0u)
            {
                std::fprintf(stderr,
                             "[dispatch:swallowed-unwind] #%lu %s target=0x%x source=0x%x fallthrough=0x%x "
                             "sp=0x%x ra=0x%x%s\n",
                             n, debugName ? debugName : "?", targetPc, sourcePc, fallthroughPc,
                             static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[29])),
                             static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[31])),
                             s_dispatchUnwindFix ? " [propagated]" : " [treated as fallthrough]");
            }
            if (s_dispatchUnwindFix)
            {
                // The callee did not complete: leave ctx->pc where the checkpoint put it and let the
                // run loop re-dispatch it, exactly as it does for every other unwound chain.
                return false;
            }
        }
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    // ★★ PS2X_VU0_SLOW2=1 (default OFF) -- MEASUREMENT knob, and BIT-EXACT unlike an ablation.
    // cont.250 section 8: VU0 cannot be sized by ablating it, because the guest CONSUMES its output
    // (VCALLMS computes matrices/collision the game reads back), so falsifying the result changes
    // control flow instead of removing work -- the PS2X_VU0_ABLATE attempt produced runs that were
    // not work-matched (hero updates 1245/2636 vs a reproducible 3763/3797) and was withdrawn.
    // This doubles VU0's WORK while leaving every architectural effect identical: the production
    // pass below runs FIRST and is authoritative, then a throwaway pass re-runs the same program
    // from the same pre-call input state on a SCRATCH COPY of VU0 data memory, and its results are
    // discarded. Safe because (a) XGKICK is gated to Unit::VU1 (ps2_vu1_core.cpp), so VU0 cannot
    // queue GS packets, and (b) nothing outside this function reads m_vu0.state(), which the next
    // call rebuilds via copyVu0ContextToState()'s memset anyway.
    // Read the SLOPE: if doubling VU0 costs X ms/frame, VU0 costs ~X ms/frame. The 4 KB scratch
    // memcpy is charged to the doubled side, so the slope is an UPPER BOUND on VU0's true cost.
    static const bool s_vu0Slow2 = []
    { const char *e = std::getenv("PS2X_VU0_SLOW2"); return e && e[0] && e[0] != '0'; }();
    if (s_vu0Slow2)
    {
        const R5900Context savedCtx = *ctx;   // pre-call input state for the throwaway pass

        copyVu0ContextToState(ctx, m_vu0.state());
        m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                      vu0Data, PS2_VU0_DATA_SIZE,
                      m_gs, &m_memory,
                      startPC, 0u, ctx->vu0_itop, 4096);
        copyVu0StateToContext(m_vu0.state(), ctx);   // authoritative result, already committed

        static thread_local uint8_t scratchVu0Data[PS2_VU0_DATA_SIZE];
        std::memcpy(scratchVu0Data, vu0Data, PS2_VU0_DATA_SIZE);
        copyVu0ContextToState(&savedCtx, m_vu0.state());
        m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                      scratchVu0Data, PS2_VU0_DATA_SIZE,
                      m_gs, &m_memory,
                      startPC, 0u, savedCtx.vu0_itop, 4096);
        // results deliberately NOT copied back -- this pass exists only to burn the same work again
        return;
    }

    // No wholesale reset() per VCALLMS (cont.158d perf): copyVu0ContextToState() below
    // rebuilds the full architectural state from the EE context (it memsets the state
    // struct itself), and execute() runs resetScheduler() — reset() here was 100%
    // redundant work at math-library call rates, and its m_cycle=0 broke the cycle
    // monotonicity the scheduler bookkeeping now relies on. PCSX2 parity: vu0ExecMicro
    // (VU0micro.cpp) does not reset VU0 on VCALLMS either — flag/cycle sync + start.
    copyVu0ContextToState(ctx, m_vu0.state());
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 4096);
    copyVu0StateToContext(m_vu0.state(), ctx);
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    return m_eeScheduler->checkpointDue(cycles);
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    if (ps2pipe::benchRun(*this)) // cont.317 pipeline oracle: replay a capture and exit
    {
        std::fflush(nullptr);
        _exit(0);
    }
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    if (ps2xWindowFilter())
        SetTextureFilter(frameTex, TEXTURE_FILTER_BILINEAR); // cont.332c

    std::atomic<bool> gameThreadFinished{false};

    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        ThreadNaming::PinCurrentThreadForRole(ThreadNaming::CpuRole::EE); // cont.230 (auto plan / PS2X_EE_CPUS)
        {
            const ThreadNaming::CpuPlan &plan = ThreadNaming::AutoCpuPlan();
            const char *ee = std::getenv("PS2X_EE_CPUS"), *gs = std::getenv("PS2X_GS_CPUS");
            std::fprintf(stderr, "[affinity] plan: physicalCores=%u auto=%s EE=%s GS=%s%s\n",
                         plan.physicalCores,
                         ThreadNaming::AutoCpuPinEnabled() ? (plan.valid ? "on" : "off(<3 cores)") : "off(PS2X_CPU_PIN=0)",
                         ee && ee[0] ? ee : (ThreadNaming::AutoCpuPinEnabled() && plan.valid ? plan.ee.c_str() : "-"),
                         gs && gs[0] ? gs : (ThreadNaming::AutoCpuPinEnabled() && plan.valid ? plan.gs.c_str() : "-"),
                         (ee && ee[0]) || (gs && gs[0]) ? " (explicit)" : "");
        }
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PSPadBackend::pumpVibration(); // rumble: raylib/SDL only on this thread; a no-op until the game sends one
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);

            }
        });
        // cont.232 PS2X_GS_NOPRESENT (default OFF): ABLATION, never a mode -- the main loop only polls
        // window events and sleeps: no latch/decode, no GL draw or swap, no ImGui overlay, no target-FPS
        // wait. It is the state a MINIMIZED window used to put the presenter in (cont.231 §12, raylib
        // parked in glfwWaitEvents) without minimizing, so an alternating pair on one binary measures
        // what the whole presenter costs the guest. Nothing is shown while it is on.
        static const bool s_noPresent = []
        { const char *e = std::getenv("PS2X_GS_NOPRESENT"); return e && e[0] && e[0] != '0'; }();
        // ★ cont.232 PS2X_GS_PRESENT_LAZY (default ON; "=0" = draw + swap on every iteration, the 404
        // loop): with presents tied to the flip the loop still drew, swapped and ran the ImGui overlay at
        // 60 Hz between guest frames, and the §5 ablation (PS2X_GS_NOPRESENT) measured that idle loop at
        // ~13% of the level's frame rate (five threads over four cores: the main thread's GL driver work,
        // the overlay, raylib's partial busy-wait). Now an iteration WITHOUT a new frame only polls window
        // events and sleeps 4 ms; the frame is drawn when a new one was latched, while the overlay is open
        // (it stays interactive at 60 Hz; it starts hidden, PS2X_DEBUG_UI=1 / F1 opens it), on F1 or a
        // resize, and at least every 250 ms (expose). The panel itself handles F1, so on an F1 press the
        // loop draws once to let it toggle.
        static const bool s_presentLazy = []
        { const char *e = std::getenv("PS2X_GS_PRESENT_LAZY"); return !(e && e[0] == '0'); }();
        static std::chrono::steady_clock::time_point s_lastDraw{};
        if (s_noPresent)
        {
            PollInputEvents(); // keep answering the WM (ping/close), as the parked loop did
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            if (WindowShouldClose())
            {
                RUNTIME_LOG("[run] window close requested (nopresent), breaking out of loop");
                requestStop();
                break;
            }
            continue;
        }
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        const bool newFrame = UploadFrame(frameTex, this, presentWidth, presentHeight);
        if (s_presentLazy && !newFrame && !g_ps2xDebugUiVisible.load(std::memory_order_relaxed) &&
            !IsKeyPressed(KEY_F1) && !IsWindowResized() &&
            std::chrono::steady_clock::now() - s_lastDraw < std::chrono::milliseconds(250))
        {
            PollInputEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            if (WindowShouldClose())
            {
                RUNTIME_LOG("[run] window close requested, breaking out of loop");
                requestStop();
                break;
            }
            continue;
        }
        s_lastDraw = std::chrono::steady_clock::now();

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        // ★★★★ cont.332d: fit by the DISPLAYED aspect, not the framebuffer's pixel count.
        const double want = ps2xWindowAspect();
        const float regAspect = ps2xGsPresentAspect();
        // ★★★★★ cont.356e: a 2D SCREEN is PILLARBOXED, not stretched. The title/menu artwork is
        // uploaded into the framebuffer during boot and presented from there -- nothing redraws it
        // per frame, so there is no draw to counter-scale (cont.356d proved a draw-level fix cannot
        // reach it). Presenting such a frame at the DISPLAY-derived natural aspect keeps the art in
        // its authored proportions and puts the widescreen bars at the sides, which is the correct
        // treatment for 4:3 art on a wider screen. Only the explicit-ratio case is overridden; with
        // no override the two are already equal and this changes nothing.
        const bool twoDScreen = ps2xGsIs2dScreen() && regAspect > 0.1f && regAspect < 4.0f;
        const float aspect = twoDScreen
                                 ? regAspect                         // pillarbox: natural aspect
                                 : ((want > 1.1)
                                 ? static_cast<float>(want)          // an explicit ratio (16:9, ...)
                                 : ((want > 0.0 && regAspect > 0.1f && regAspect < 4.0f)
                                        ? regAspect                  // derived from DISPLAY
                                        : (srcWidth / srcHeight)));  // off: pixel-square, as before
        float dstWidth = screenWidth;
        float dstHeight = screenWidth / aspect;
        if (dstHeight > screenHeight)
        {
            dstHeight = screenHeight;
            dstWidth = screenHeight * aspect;
        }
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        // ★★★★ cont.332c: the hi-res texture carries the same picture at scene resolution, so the
        // aspect fit above (computed from the GUEST display size) is still the right rectangle --
        // only the source rect changes.
        if (s_hiresLive && s_hiresTexValid)
        {
            const Rectangle hiSrc{0.0f, 0.0f, static_cast<float>(s_hiresTexW),
                                  static_cast<float>(s_hiresTexH)};
            DrawTexturePro(s_hiresTex, hiSrc, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        }
        else
        {
            DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        }
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    // 2026-10-02: the join below waited for ever when the EE thread never reached a stop check, so
    // neither the window's close button nor SIGTERM could end the process. Bound it.
    ps2x::shutdown::armWatchdog("run loop ended");
    if (gameThread.joinable())
    {
        gameThread.join();
        ps2gs::shutdown(); // cont.317 stage 2: drain and join the pipeline thread after the EE stops
    }

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
    UnloadTexture(frameTex);
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}
