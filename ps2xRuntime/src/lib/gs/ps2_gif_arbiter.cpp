#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_pipeline.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>

// gs_frontend.cpp: selects which path's suspended GIF tag the parser resumes (no header on
// purpose -- headers reach the generated code).
extern "C" void ps2xGsSetGifPath(uint32_t path);
// cont.317 pipeline oracle (ps2_pipe_capture.cpp): a tap on every packet handed to the GS, and a
// header-free handle on the live arbiter so the replay can issue the explicit end-of-pass drain.
extern "C" void (*ps2xGifArbiterTap)(uint32_t pathId, const uint8_t *data, uint32_t size) = nullptr;
static GifArbiter *g_currentArbiter = nullptr;
extern "C" void ps2xGifArbiterDrainCurrent() { if (g_currentArbiter) g_currentArbiter->drain(); }

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
    g_currentArbiter = this;
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u || flg == 3u; // flg=3 "Disable" == IMAGE2
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

// cont.344 PS2X_GIF_FIFO (default OFF; =1 drains in EE kick order instead of the path-priority sort
// below). Built to test whether the sort -- which moves every PATH3 packet of a pass behind the PATH1/2
// packets of the same pass -- was why the Paths of the Dead floor sampled a CLUT whose RGB the shadow
// pass had already cleared. MEASURED NULL on the CPU arm (build 998, idle spawn flip 6900: floor
// pixel-identical, the [gs2:pixwatch] trace identical) and ~35% slower wall (draw runs fragment), so
// it stays an A/B knob. PCSX2 Gif_Unit (Gif_Unit.h Execute/checkPaths) prefers PATH1 over a PATH3
// transfer pending at the same time, at packet boundaries; kick order is what FIFO approximates.
static const bool s_gifFifo = []
{ const char *e = std::getenv("PS2X_GIF_FIFO"); return e && e[0] && e[0] != '0'; }();

void GifArbiter::drain()
{
    if (!m_processFn)
        return;

    if (!s_gifFifo)
    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    // ★ rotk row 253 (PS2X_GS_SPLIT): on the pipeline thread the sorted packets and the end-of-drain go to the
    // GsParse thread, in this order, instead of being parsed here (ps2_gs_pipeline.cpp, stage 3). The tap stays
    // here, so the capture oracle sees the same sequence either way.
    if (ps2gs::splitSink())
    {
        for (auto &pkt : m_queue)
        {
            if (pkt.data.empty())
                continue;
            if (ps2xGifArbiterTap)
                ps2xGifArbiterTap(static_cast<uint32_t>(pkt.pathId), pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
            ps2gs::splitPacket(static_cast<uint32_t>(pkt.pathId), std::move(pkt.data));
        }
        const bool anySplit = !m_queue.empty();
        m_queue.clear();
        if (anySplit && m_drainDoneFn)
            ps2gs::splitEnd();
        return;
    }

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
        {
            ps2xGsSetGifPath(static_cast<uint32_t>(pkt.pathId));
            if (ps2xGifArbiterTap)
                ps2xGifArbiterTap(static_cast<uint32_t>(pkt.pathId), pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
        }
    }
    const bool any = !m_queue.empty();
    m_queue.clear();
    if (any && m_drainDoneFn)
        m_drainDoneFn(); // cont.230: the draws of everything just parsed can go to the raster worker
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
