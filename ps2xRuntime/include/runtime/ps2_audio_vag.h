#ifndef PS2_AUDIO_VAG_H
#define PS2_AUDIO_VAG_H

#include <cstdint>
#include <vector>

namespace ps2_vag
{
    // PS-ADPCM decoder state. The 4-bit ADPCM filters are IIR over the two previously decoded
    // samples, so the history has to survive a block -- and, for a stream decoded in pieces, a
    // chunk boundary. PCSX2 carries exactly this pair per voice (`V_Voice::Prev1/Prev2`, passed
    // by reference into `XA_decode_block`, pcsx2/SPU2/Mixer.cpp); a decoder that restarts them at
    // zero on every chunk clicks once per chunk.
    struct AdpcmState
    {
        int16_t s1 = 0;
        int16_t s2 = 0;
    };

    // Decode ONE 16-byte PS-ADPCM block into 28 samples, advancing `st`.
    void decodeBlock(const uint8_t *block, AdpcmState &st, int16_t *out28);

    // Whole-buffer decode of a VAGp container (48-byte header + blocks). Sample rate comes from
    // the header. Unchanged behaviour -- it just shares decodeBlock() now.
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

#endif
