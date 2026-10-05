#include "runtime/ps2_audio.h"
#include "runtime/ps2_audio_vag.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

struct PS2AudioBackend::Impl
{
#if !defined(PLATFORM_VITA)
    // ★ rotk row 256 PS2X_AUDIO_SOUNDCACHE (default ON; `=0` = convert per note): each distinct sample is
    // converted to the device format ONCE (a cached source Sound) and every note plays a raylib ALIAS of it
    // (LoadSoundAlias: shared data, its own pitch / volume / pan / cursor). WHY: play() rebuilt a WAV, parsed it
    // back and ran LoadSoundFromWave -- a full ma_convert_frames resample of the whole sample -- for EVERY note:
    // 5-7% of the EE thread on the 60-fps fight (perf LBR: seqPlayTone -> play -> playDecodedSample ->
    // LoadSoundFromWave -> ma_linear_resampler). Same converter, same input, so the converted data is the same
    // bytes; with matching formats an alias does no conversion at all. A source stays alive while any alias of it
    // plays (shared_ptr); the cache entry is replaced when the sample's PCM changes (keys are RAM addresses a bank
    // reload can reuse: content hash).
    struct CachedSource
    {
        Sound snd{};
        ~CachedSource() { UnloadSound(snd); }
    };
    struct CacheEntry
    {
        std::shared_ptr<CachedSource> src;
        uint64_t hash = 0;
        size_t frames = 0;
        uint32_t rate = 0;
    };
    std::unordered_map<uint32_t, CacheEntry> cache;
#endif
    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
#if !defined(PLATFORM_VITA)
        std::shared_ptr<CachedSource> src; // set = `snd` is an alias of src->snd
#endif
    };
    std::vector<TrackedSound> activeSounds;
#if !defined(PLATFORM_VITA)
    static void unloadTracked(TrackedSound &t)
    {
        if (t.src)
            UnloadSoundAlias(t.snd);
        else
            UnloadSound(t.snd);
        t.src.reset();
    }
#endif
};

#if !defined(PLATFORM_VITA)
namespace
{
    uint64_t pcmHash(const std::vector<int16_t> &pcm)
    {
        uint64_t h = 1469598103934665603ull ^ pcm.size();
        const uint8_t *b = reinterpret_cast<const uint8_t *>(pcm.data());
        const size_t n = pcm.size() * sizeof(int16_t);
        size_t i = 0;
        for (; i + 8u <= n; i += 8u)
        {
            uint64_t w;
            std::memcpy(&w, b + i, 8u);
            h = (h ^ w) * 1099511628211ull;
        }
        for (; i < n; ++i)
            h = (h ^ b[i]) * 1099511628211ull;
        return h;
    }
}
#endif

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    streamStopAll();
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex,
                           float pan)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm, pan);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            Impl::unloadTracked(*it);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm, float pan)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    (void)pan;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    // ★★★★★ cont.332s THE RETRIGGER DROP -- the user-reported "the language page plays the move
    // sound on DOWN but not on UP" (2026-09-17). This loop used to `return` whenever the SAME sample
    // was still sounding, so a second key-on inside the first blip's length was silently discarded.
    // It is not direction: it is TIMING. The user's own log fired the trigger SEVEN times for three
    // presses (`[aud:onupd] EDGE` at flushes 71/86/134/149/162/176/190, all tpl=0x0201) while they
    // heard about two -- press, press-too-soon (dropped), wait, press (heard).
    // The rule is our invention, not the hardware's: SPU2 is 2 cores x 24 voices (PCSX2
    // `SPU2/defs.h` V_Core::Voices[24]) and every key-on takes its own voice, so one waveform can
    // sound many times at once; `SPU2/Mixer.cpp` just mixes them. PS2X_AUDIO_RETRIGGER=0 restores
    // the drop for an A/B.
    auto &sounds = m_impl->activeSounds;
    {
        static const bool s_retrigger = []
        { const char *e = std::getenv("PS2X_AUDIO_RETRIGGER"); return !(e && e[0] == '0'); }();
        // Bounded so a per-frame trigger cannot stack without limit: at the cap, the OLDEST instance
        // of this same key is stolen (hardware steals voices too) -- so a retrigger is always audible.
        static const int s_sameKeyCap = []
        { const char *e = std::getenv("PS2X_AUDIO_SAMEKEY"); return (e && e[0]) ? std::atoi(e) : 3; }();
        static unsigned long long s_retrig = 0, s_stolen = 0;
        int same = 0;
        auto oldestSame = sounds.end();
        for (auto it = sounds.begin(); it != sounds.end(); ++it)
        {
            if (it->sampleKey == sampleKey && IsSoundPlaying(it->snd))
            {
                ++same;
                if (oldestSame == sounds.end()) oldestSame = it;   // push_back order == age order
            }
        }
        if (same > 0)
        {
            if (!s_retrigger)
                return;                                  // the pre-332s behaviour
            if (same >= s_sameKeyCap && oldestSame != sounds.end())
            {
                StopSound(oldestSame->snd);
                Impl::unloadTracked(*oldestSame);
                sounds.erase(oldestSame);
                ++s_stolen;
            }
            if (++s_retrig <= 8 || (s_retrig % 500) == 0)
                std::fprintf(stderr, "[audio] retrigger #%llu key=0x%08x (%d already sounding,"
                                     " cap %d, stolen %llu) -- pre-332s this was DROPPED\n",
                             s_retrig, sampleKey, same, s_sameKeyCap, s_stolen);
        }
    }

    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                Impl::unloadTracked(*it);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    // cont.332s: was a hard 4. SPU2 has 48 voices; 4 meant a busy moment evicted sounds that were
    // still playing (the eviction below stops the OLDEST outright). PS2X_AUDIO_VOICES tunes it.
    static const int kMaxConcurrentSounds = []
    { const char *e = std::getenv("PS2X_AUDIO_VOICES"); return (e && e[0]) ? std::atoi(e) : 16; }();
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        Impl::unloadTracked(sounds.front());
        sounds.erase(sounds.begin());
    }

    static const bool s_soundCache = []
    { const char *e = std::getenv("PS2X_AUDIO_SOUNDCACHE"); const char *d = std::getenv("PS2X_AUDIO_DUMP");
      return !(e && e[0] == '0') && !(d && d[0]); }(); // PS2X_AUDIO_DUMP lives on the per-note path below
    std::shared_ptr<Impl::CachedSource> cachedSrc;
    if (s_soundCache)
    {
        const uint64_t h = pcmHash(sample.pcm);
        auto &entry = m_impl->cache[sampleKey];
        if (entry.src && entry.hash == h && entry.frames == sample.pcm.size() && entry.rate == sample.sampleRate)
            cachedSrc = entry.src;
        else
        {
            // bound the cache: drop sources no note is using (an entry in use stays)
            if (m_impl->cache.size() > 512u)
                for (auto it = m_impl->cache.begin(); it != m_impl->cache.end();)
                {
                    if (it->first != sampleKey && it->second.src.use_count() == 1)
                        it = m_impl->cache.erase(it);
                    else
                        ++it;
                }
            std::vector<uint8_t> wav0 = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
            Wave wave0 = LoadWaveFromMemory(".wav", wav0.data(), static_cast<int>(wav0.size()));
            if (wave0.frameCount <= 0)
                return;
            auto src = std::make_shared<Impl::CachedSource>();
            src->snd = LoadSoundFromWave(wave0);
            UnloadWave(wave0);
            auto &e2 = m_impl->cache[sampleKey]; // (the bound above may have rehashed)
            e2.src = src;
            e2.hash = h;
            e2.frames = sample.pcm.size();
            e2.rate = sample.sampleRate;
            cachedSrc = std::move(src);
        }
        Sound snd = LoadSoundAlias(cachedSrc->snd);
        SetSoundPitch(snd, pitch);
        SetSoundVolume(snd, volume);
        if (pan != 0.5f) SetSoundPan(snd, pan);
        m_impl->activeSounds.push_back({snd, sampleKey, cachedSrc});
        PlaySound(snd);
        return;
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    // ★★★ cont.332w PS2X_AUDIO_DUMP=<dir> (default OFF): write each DISTINCT sample the game plays
    // to <dir>/snd_<key>_<rate>Hz_<frames>.wav. "The sounds are weird" is not a diagnosable report
    // and a log line cannot be listened to; a folder of the actual waveforms can be, by the person
    // who knows what the game should sound like. Deduped by sample key, capped at 256 files.
    {
        static const char *s_dumpDir = std::getenv("PS2X_AUDIO_DUMP");
        if (s_dumpDir && s_dumpDir[0])
        {
            static std::set<uint32_t> seen;
            static bool made = false;
            if (!made) { made = true; ::mkdir(s_dumpDir, 0755); }   // PRESENT_SAVE's lesson: mkdir, or it is a silent sink
            if (seen.size() < 256u && seen.insert(sampleKey).second)
            {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/snd_%08x_%uHz_%zu.wav", s_dumpDir, sampleKey,
                              sample.sampleRate, sample.pcm.size());
                if (FILE *df = std::fopen(path, "wb"))
                {
                    std::fwrite(wav.data(), 1, wav.size(), df);
                    std::fclose(df);
                    std::fprintf(stderr, "[audio] dumped %s (%zu frames)\n", path, sample.pcm.size());
                }
            }
        }
    }
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    if (pan != 0.5f) SetSoundPan(snd, pan);
    m_impl->activeSounds.push_back({snd, sampleKey, nullptr});
    PlaySound(snd);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
#if defined(PLATFORM_VITA)
    (void)voiceId;
#else
    // Stop the sound playing under this key. The game's sequencer issues an explicit key-off
    // (AUDIOPF `OnKeyOff` @0x064c0 -> `Trk_RndrStopTone`), which previously had nowhere to go: this
    // was an empty stub, so a tone always ran to the end of its sample regardless of the score.
    // play() admits one sound per key, so the mapping is one-to-one.
    std::lock_guard<std::mutex> lock(m_mutex);
    const uint32_t key = voiceId & PS2_RAM_MASK;
    auto &sounds = m_impl->activeSounds;
    for (auto it = sounds.begin(); it != sounds.end();)
    {
        if (it->sampleKey == key)
        {
            StopSound(it->snd);
            Impl::unloadTracked(*it);
            it = sounds.erase(it);
        }
        else
            ++it;
    }
#endif
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        Impl::unloadTracked(t);
    }
    m_impl->activeSounds.clear();
    m_impl->cache.clear(); // no alias is left: every source unloads here, while the device is still open
#endif
}

// ================================================================================================
// Continuous streaming
// ------------------------------------------------------------------------------------------------
// `play()` above is a one-shot: the whole sample has to be decoded and handed over before a note
// sounds, so a streamed music track is bounded by whatever the caller was willing to buffer. The
// sink below feeds one instead -- the caller pushes raw PS-ADPCM as it reads it, and the audio
// thread drains decoded PCM out of a ring.
//
// Two details are load-bearing:
//   * the ADPCM filter history (`AdpcmState`) is per stream and survives a chunk boundary, exactly
//     as PCSX2 keeps `V_Voice::Prev1/Prev2` across blocks (pcsx2/SPU2/Mixer.cpp XA_decode_block);
//     restarting it per chunk clicks audibly at every seam.
//   * raylib's `AudioCallback` carries no user pointer, so each slot needs its own trampoline.
//
// The slots are file-static because those trampolines have to reach them without a `this`. There is
// exactly one PS2AudioBackend (a member of PS2Runtime), so that is a naming choice, not a limit.
// ================================================================================================
namespace
{
    // ★★★★ cont.297: was 4, and HEL01 alone asks for EIGHT concurrent streams
    // (0x0201/0x0202/0x0207/0x020d/0x020e/0x020f/0x0210/0x0211) -- four opened and four were REFUSED.
    // Voice-over is streamed, so the refusals were silent VO; a refused music stream also leaves the
    // level partly silent. Idle slots are free (the 512 KB ring is allocated in streamOpen and
    // released in streamStop), so the headroom costs nothing. PS2 hardware is not the constraint
    // here: SPU2 has 48 voices (PCSX2 SPU2/defs.h V_Core::Voices[24] x2), so a handful of
    // simultaneous streamed sources is ordinary for this game, not an abuse.
    constexpr uint32_t kMaxStreams = 12;
    constexpr uint32_t kRingFrames = 1u << 18;      // 262144 frames = 5.9 s @ 44.1 kHz (512 KB)
    constexpr uint32_t kRingMask = kRingFrames - 1; // power of two: index = counter & mask
    constexpr uint32_t kAdpcmBlockBytes = 16;
    constexpr uint32_t kAdpcmBlockFrames = 28;

    struct StreamSlot
    {
        std::atomic<bool> active{false};
        uint32_t key = 0;
        uint32_t sampleRate = 44100;
        AudioStream stream{};
        std::vector<int16_t> ring;
        // Single producer (the pump) / single consumer (the audio thread). The counters are
        // monotonic frame totals, never wrapped, so `wr - rd` is the fill level directly.
        std::atomic<uint64_t> wr{0};
        std::atomic<uint64_t> rd{0};
        std::atomic<uint64_t> underrun{0};
        uint32_t channels = 1;                 // 1 = mono, 2 = block-interleaved stereo
        ps2_vag::AdpcmState adpcm;             // producer-only, channel 0
        ps2_vag::AdpcmState adpcm1;            // producer-only, channel 1
        // ★ cont.334 PS2X_AUDIO_TAP=<dir> (default OFF): capture EXACTLY the frames handed to the
        // device, so a reported artefact can be measured instead of guessed at. Appended on the
        // audio thread (no allocation: reserved once at open, and capped), written out at stop.
        // This is the only place that sees what the user actually hears -- above it the ring may be
        // correct while the output is not, and below it we have no visibility at all.
        std::vector<int16_t> tap;
        bool tapOn = false;
    };

    // The capture directory, or empty when the tap is off. Read once.
    const char *audioTapDir()
    {
        static const char *d = []() -> const char * {
            const char *e = std::getenv("PS2X_AUDIO_TAP");
            return (e && e[0]) ? e : nullptr;
        }();
        return d;
    }
    constexpr size_t kTapMaxFrames = 120u * 48000u; // 120 s per stream, then it stops growing

    StreamSlot g_streams[kMaxStreams];

    void fillSlot(uint32_t i, void *bufferData, unsigned int frames)
    {
        StreamSlot &s = g_streams[i];
        int16_t *out = static_cast<int16_t *>(bufferData);
        if (!out || frames == 0)
            return;
        if (!s.active.load(std::memory_order_acquire))
        {
            std::memset(out, 0, frames * sizeof(int16_t));
            return;
        }

        // The ring stores INTERLEAVED samples, so a raylib "frame" is `channels` of them.
        const uint32_t ch = s.channels ? s.channels : 1u;
        const uint32_t want = frames * ch;
        const uint64_t wr = s.wr.load(std::memory_order_acquire);
        uint64_t rd = s.rd.load(std::memory_order_relaxed);
        uint32_t got = 0;
        while (got < want && rd < wr)
        {
            const uint32_t idx = static_cast<uint32_t>(rd & kRingMask);
            uint32_t run = kRingFrames - idx; // to the end of the ring
            run = static_cast<uint32_t>(std::min<uint64_t>(run, wr - rd));
            run = std::min(run, want - got);
            std::memcpy(out + got, s.ring.data() + idx, run * sizeof(int16_t));
            got += run;
            rd += run;
        }
        s.rd.store(rd, std::memory_order_release);

        if (got < want)
        {
            // The pump fell behind. Silence is the honest answer -- repeating the tail would
            // sound like a stutter and hide the underrun from the counter.
            std::memset(out + got, 0, (want - got) * sizeof(int16_t));
            s.underrun.fetch_add((want - got) / ch, std::memory_order_relaxed);
        }

        // ★ cont.334: the tap sees the finished buffer -- real frames AND the zero-fill of an
        // underrun -- which is precisely what reaches the speakers.
        if (s.tapOn && s.tap.size() + want <= kTapMaxFrames)
            s.tap.insert(s.tap.end(), out, out + want);
    }

    void streamCb0(void *b, unsigned int f) { fillSlot(0, b, f); }
    void streamCb1(void *b, unsigned int f) { fillSlot(1, b, f); }
    void streamCb2(void *b, unsigned int f) { fillSlot(2, b, f); }
    void streamCb3(void *b, unsigned int f) { fillSlot(3, b, f); }
    void streamCb4(void *b, unsigned int f) { fillSlot(4, b, f); }
    void streamCb5(void *b, unsigned int f) { fillSlot(5, b, f); }
    void streamCb6(void *b, unsigned int f) { fillSlot(6, b, f); }
    void streamCb7(void *b, unsigned int f) { fillSlot(7, b, f); }
    void streamCb8(void *b, unsigned int f) { fillSlot(8, b, f); }
    void streamCb9(void *b, unsigned int f) { fillSlot(9, b, f); }
    void streamCb10(void *b, unsigned int f) { fillSlot(10, b, f); }
    void streamCb11(void *b, unsigned int f) { fillSlot(11, b, f); }
    // raylib callbacks carry no user pointer, so each slot needs its own thunk.
    AudioCallback g_streamCallbacks[kMaxStreams] = {streamCb0, streamCb1, streamCb2, streamCb3, streamCb4, streamCb5, streamCb6, streamCb7, streamCb8, streamCb9, streamCb10, streamCb11};

    StreamSlot *findStream(uint32_t key)
    {
        for (uint32_t i = 0; i < kMaxStreams; ++i)
        {
            StreamSlot &s = g_streams[i];
            if (s.active.load(std::memory_order_acquire) && s.key == key)
                return &s;
        }
        return nullptr;
    }
}

bool PS2AudioBackend::streamOpen(uint32_t key, uint32_t sampleRate, float volume, uint32_t channels)
{
#if defined(PLATFORM_VITA)
    (void)key;
    (void)sampleRate;
    (void)volume;
    (void)channels;
    return false;
#else
    if (!m_audioReady)
        return false; // PS2X_AUDIO unset: no host device, so stay silent rather than buffer
    if (sampleRate < 1000u || sampleRate > 96000u)
        return false;
    if (channels < 1u || channels > 2u)
        channels = 1u; // the tone byte is data; clamp rather than trust it
    if (findStream(key))
        return true; // already streaming: a repeated trigger must not restart it

    std::lock_guard<std::mutex> lock(m_mutex);
    for (uint32_t i = 0; i < kMaxStreams; ++i)
    {
        StreamSlot &s = g_streams[i];
        if (s.active.load(std::memory_order_acquire))
            continue;

        s.ring.assign(kRingFrames, 0);
        s.wr.store(0, std::memory_order_relaxed);
        s.rd.store(0, std::memory_order_relaxed);
        s.underrun.store(0, std::memory_order_relaxed);
        s.tapOn = (audioTapDir() != nullptr);
        s.tap.clear();
        if (s.tapOn)
            s.tap.reserve(kTapMaxFrames / 8); // grow rarely; the audio thread must not stall
        s.adpcm = ps2_vag::AdpcmState{};
        s.adpcm1 = ps2_vag::AdpcmState{};
        s.channels = channels;
        s.key = key;
        s.sampleRate = sampleRate;
        s.stream = LoadAudioStream(sampleRate, 16, channels);
        if (!IsAudioStreamValid(s.stream))
            return false;

        // Publish the slot before the callback can run, then start it.
        s.active.store(true, std::memory_order_release);
        SetAudioStreamVolume(s.stream, volume);
        SetAudioStreamCallback(s.stream, g_streamCallbacks[i]);
        PlayAudioStream(s.stream);
        std::fprintf(stderr, "[audio] stream open key=0x%08x %u Hz (ring %u frames = %.1f s)\n",
                     key, sampleRate, kRingFrames, (double)kRingFrames / (double)sampleRate);
        return true;
    }
    return false; // all slots busy
#endif
}

bool PS2AudioBackend::streamIsOpen(uint32_t key) const
{
    return findStream(key) != nullptr;
}

uint32_t PS2AudioBackend::streamFreeAdpcmBytes(uint32_t key) const
{
    const StreamSlot *s = findStream(key);
    if (!s)
        return 0;
    const uint64_t wr = s->wr.load(std::memory_order_relaxed);
    const uint64_t rd = s->rd.load(std::memory_order_acquire);
    // A push group is `channels` ADPCM blocks and yields kAdpcmBlockFrames INTERLEAVED frames,
    // i.e. kAdpcmBlockFrames * channels ring slots.
    const uint32_t ch = s->channels ? s->channels : 1u;
    const uint32_t freeSlots = kRingFrames - static_cast<uint32_t>(wr - rd);
    const uint32_t groups = freeSlots / (kAdpcmBlockFrames * ch);
    return groups * (kAdpcmBlockBytes * ch);
}

uint32_t PS2AudioBackend::streamQueuedFrames(uint32_t key) const
{
    const StreamSlot *s = findStream(key);
    if (!s)
        return 0;
    const uint64_t wr = s->wr.load(std::memory_order_relaxed);
    const uint64_t rd = s->rd.load(std::memory_order_acquire);
    const uint32_t ch = s->channels ? s->channels : 1u;
    return static_cast<uint32_t>(wr - rd) / ch;
}

uint64_t PS2AudioBackend::streamUnderrunFrames(uint32_t key) const
{
    const StreamSlot *s = findStream(key);
    return s ? s->underrun.load(std::memory_order_relaxed) : 0u;
}

uint32_t PS2AudioBackend::streamPushAdpcm(uint32_t key, const uint8_t *adpcm, uint32_t sizeBytes)
{
    StreamSlot *s = findStream(key);
    if (!s || !adpcm)
        return 0;

    // ★★★★★ cont.334: a group is `channels` consecutive 16-byte blocks -- block 0 is channel 0,
    // block 1 channel 1 -- each decoded with its OWN predictor state, then interleaved into the
    // ring. Consuming a group atomically is what keeps the channels in phase; decoding the same
    // bytes linearly with one state is the "stutter" this replaces.
    const uint32_t ch = s->channels ? s->channels : 1u;
    const uint32_t groupBytes = kAdpcmBlockBytes * ch;
    const uint32_t groupSlots = kAdpcmBlockFrames * ch;

    uint64_t wr = s->wr.load(std::memory_order_relaxed);
    uint32_t used = 0;
    while (sizeBytes - used >= groupBytes)
    {
        const uint64_t rd = s->rd.load(std::memory_order_acquire);
        if (kRingFrames - static_cast<uint32_t>(wr - rd) < groupSlots)
            break; // ring full: the caller re-offers the remainder next pump

        int16_t pcm[2][kAdpcmBlockFrames];
        ps2_vag::decodeBlock(adpcm + used, s->adpcm, pcm[0]);
        if (ch > 1)
            ps2_vag::decodeBlock(adpcm + used + kAdpcmBlockBytes, s->adpcm1, pcm[1]);
        used += groupBytes;

        for (uint32_t f = 0; f < kAdpcmBlockFrames; ++f)
            for (uint32_t c = 0; c < ch; ++c)
                s->ring[static_cast<uint32_t>((wr + f * ch + c) & kRingMask)] = pcm[c][f];
        wr += groupSlots;
    }
    // One release publishes every block written above.
    s->wr.store(wr, std::memory_order_release);
    return used;
}

void PS2AudioBackend::streamSetVolume(uint32_t key, float volume)
{
#if defined(PLATFORM_VITA)
    (void)key;
    (void)volume;
#else
    if (!m_audioReady)
        return;
    if (volume < 0.0f)
        volume = 0.0f;
    if (volume > 1.0f)
        volume = 1.0f;
    std::lock_guard<std::mutex> lock(m_mutex);
    StreamSlot *s = findStream(key);
    if (s)
        SetAudioStreamVolume(s->stream, volume);
#endif
}

void PS2AudioBackend::streamStop(uint32_t key)
{
#if defined(PLATFORM_VITA)
    (void)key;
#else
    std::lock_guard<std::mutex> lock(m_mutex);
    StreamSlot *s = findStream(key);
    if (!s)
        return;
    // Retire the slot first: the callback zero-fills from here on, and UnloadAudioStream takes
    // the same raylib lock the mixer holds while calling us, so the ring cannot be freed under it.
    s->active.store(false, std::memory_order_release);
    // ★ cont.334: flush the tap now that the callback can no longer touch it. A .wav, so it can be
    // listened to as well as measured.
    if (s->tapOn && !s->tap.empty())
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/tap_%04x_%uHz.wav", audioTapDir(), s->key, s->sampleRate);
        if (FILE *f = std::fopen(path, "wb"))
        {
            const uint32_t n = static_cast<uint32_t>(s->tap.size());   // interleaved samples
            const uint32_t tch = s->channels ? s->channels : 1u;
            const uint32_t dataBytes = n * 2u, rate = s->sampleRate;
            const uint32_t byteRate = rate * 2u * tch;
            auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
            auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
            std::fwrite("RIFF", 1, 4, f); w32(36u + dataBytes); std::fwrite("WAVE", 1, 4, f);
            std::fwrite("fmt ", 1, 4, f); w32(16u); w16(1); w16((uint16_t)tch);
            w32(rate); w32(byteRate); w16((uint16_t)(2u * tch)); w16(16);
            std::fwrite("data", 1, 4, f); w32(dataBytes);
            std::fwrite(s->tap.data(), 2, n, f);
            std::fclose(f);
            std::fprintf(stderr, "[audio:tap] key=0x%04x wrote %u frames (%.2f s) -> %s\n",
                         s->key, n / tch, (double)(n / tch) / (double)(rate ? rate : 1), path);
        }
        else
        {
            std::fprintf(stderr, "[audio:tap] key=0x%04x CANNOT WRITE %s (does the dir exist?)\n",
                         s->key, path);
        }
        s->tap.clear();
        s->tap.shrink_to_fit();
    }
    StopAudioStream(s->stream);
    SetAudioStreamCallback(s->stream, nullptr);
    UnloadAudioStream(s->stream);
    s->stream = AudioStream{};
    s->ring.clear();
    s->ring.shrink_to_fit();
#endif
}

void PS2AudioBackend::streamStopAll()
{
    for (uint32_t i = 0; i < kMaxStreams; ++i)
    {
        if (g_streams[i].active.load(std::memory_order_acquire))
            streamStop(g_streams[i].key);
    }
}
