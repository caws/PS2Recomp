#ifndef PS2_AUDIO_H
#define PS2_AUDIO_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class PS2AudioBackend
{
public:
    PS2AudioBackend();
    ~PS2AudioBackend();

    void onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes);
    void onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr);
    void onSoundCommand(uint32_t sid, uint32_t rpcNum,
                        const uint8_t *sendBuf, uint32_t sendSize,
                        uint8_t *recvBuf, uint32_t recvSize);

    // `pan` follows raylib's convention: 0.0 = hard LEFT-channel gain 0 ... it is the LEFT gain,
    // so 0.0 is full right, 1.0 is full left and 0.5 is centred (raudio.c MixAudioFrames:
    // `left = buffer->pan; right = 1 - left`, then a sine pan law).
    void play(uint32_t sampleAddr, float pitch = 1.0f, float volume = 1.0f,
              uint32_t voiceIndex = 0xFFFFFFFFu, float pan = 0.5f);
    void stop(uint32_t voiceId);
    void stopAll();
    void setAudioReady(bool ready) { m_audioReady = ready; }
    // True only while a host audio device is actually open (PS2X_AUDIO=1). Callers that
    // allocate for playback should check this rather than assume.
    bool audioReady() const { return m_audioReady; }

    // --- Continuous streaming --------------------------------------------------------------
    // `play()` above is a ONE-SHOT: it wants the whole sample decoded up front, which caps a
    // streamed music/ambience track at whatever the caller was willing to buffer. These feed a
    // long track in pieces instead: the caller pushes raw PS-ADPCM as it reads it, the backend
    // decodes it carrying the filter history across chunk boundaries (as PCSX2 carries
    // V_Voice::Prev1/Prev2 into XA_decode_block, pcsx2/SPU2/Mixer.cpp) into a ring buffer that
    // the audio thread drains. `key` is the caller's own id for the stream (any unique value).
    //
    // Streams are independent of the one-shot sample bank and of each other; at most
    // kMaxStreams (4) can be open. Every call is a no-op returning 0/false for an unknown key.
    // ★★★★★ cont.334 `channels`: a PS2 streamed instrument may be MULTI-CHANNEL, with the
    // channels interleaved one 16-byte ADPCM block at a time (block 0 = ch 0, block 1 = ch 1, ...).
    // The tone record carries the count at `tone[+0x16]` and AUDIOPF branches on it in
    // `StmPrimeVoices` @0x072f8 (`tone[0x16] < 2`). Decoding such a stream as mono plays
    // L-block, R-block, L-block ... which chops the line into alternating fragments -- the
    // user-reported dialogue "stutter" of 2026-09-17. Each channel keeps its own ADPCM predictor
    // state, exactly as PCSX2 gives every SPU2 voice its own `V_Voice::Prev1/Prev2`.
    bool     streamOpen(uint32_t key, uint32_t sampleRate, float volume = 1.0f, uint32_t channels = 1);
    bool     streamIsOpen(uint32_t key) const;
    // Room available RIGHT NOW, in whole 16-byte ADPCM blocks' worth of bytes.
    uint32_t streamFreeAdpcmBytes(uint32_t key) const;
    // Push raw ADPCM (no VAGp header). Returns the bytes actually consumed -- a partial push
    // means the ring is full, and the remainder must be offered again on the next pump.
    uint32_t streamPushAdpcm(uint32_t key, const uint8_t *adpcm, uint32_t sizeBytes);
    uint32_t streamQueuedFrames(uint32_t key) const;   // still to play: the pump's safety margin
    uint64_t streamUnderrunFrames(uint32_t key) const; // silence the ring could not cover
    // A live stream's volume must follow its submix channel. The game drives OnSubmixChan
    // continuously (LOTR: chan 15 ducks to 0.400 then ramps back to 0.800 across a cutscene), so a
    // volume fixed at streamOpen() plays a ducked bed at full level for its whole life.
    void     streamSetVolume(uint32_t key, float volume);
    void     streamStop(uint32_t key);
    void     streamStopAll();

private:
    struct DecodedSample
    {
        std::vector<int16_t> pcm;
        uint32_t sampleRate = 44100;
    };

    struct Impl;
    std::unique_ptr<Impl> m_impl;
    bool m_audioReady = false;
    uint32_t m_mostRecentSampleKey = 0;
    std::vector<DecodedSample> m_loadOrderSamples;
    std::vector<uint32_t> m_loadOrderSampleKeys;
    std::unordered_map<uint32_t, DecodedSample> m_sampleBank;
    std::mutex m_mutex;

    void playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                          bool isBgm = false, float pan = 0.5f);
    void pruneFinishedSounds();
};

#endif
