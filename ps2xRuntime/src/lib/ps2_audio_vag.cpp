#include <cstdlib>
#include "runtime/ps2_audio_vag.h"
#include "runtime/ps2_memory.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace
{
    inline int16_t clamp16(int32_t v)
    {
        if (v < -32768)
            return -32768;
        if (v > 32767)
            return 32767;
        return static_cast<int16_t>(v);
    }

    inline int8_t signExtend4(uint8_t nibble)
    {
        uint8_t s = nibble & 0x0F;
        return static_cast<int8_t>((s & 8) ? static_cast<int8_t>(s | 0xF0) : static_cast<int8_t>(s));
    }
}

namespace ps2_vag
{
    // One 16-byte block -> 28 samples. Header byte 0 = shift (low nibble) + filter (bits 4-6);
    // byte 1 carries the SPU loop flags (XAFLAG_LOOP_END/LOOP/LOOP_START, pcsx2/SPU2/Mixer.cpp),
    // which the SPU acts on but a straight PCM decode does not need.
    void decodeBlock(const uint8_t *block, AdpcmState &st, int16_t *out28)
    {
        uint8_t shift = block[0] & 0x0F;
        if (shift > 12)
            shift = 9;
        uint8_t filter = (block[0] >> 4) & 0x07;
        if (filter > 4)
            filter = 0;

        for (int sampleIdx = 0; sampleIdx < 28; ++sampleIdx)
        {
            const uint8_t byte = block[2 + sampleIdx / 2];
            const uint8_t nibble = (sampleIdx & 1) ? (byte >> 4) : (byte & 0x0F);
            const int8_t rawSample = signExtend4(nibble);
            const int32_t shiftedSample = rawSample << (12 - shift);

            int32_t filteredSample;
            const int32_t old = st.s1;
            const int32_t older = st.s2;
            switch (filter)
            {
            case 0:
                filteredSample = shiftedSample;
                break;
            case 1:
                filteredSample = shiftedSample + (60 * old + 32) / 64;
                break;
            case 2:
                filteredSample = shiftedSample + (115 * old - 52 * older + 32) / 64;
                break;
            case 3:
                filteredSample = shiftedSample + (98 * old - 55 * older + 32) / 64;
                break;
            case 4:
                filteredSample = shiftedSample + (122 * old - 60 * older + 32) / 64;
                break;
            default:
                filteredSample = shiftedSample;
                break;
            }

            const int16_t clamped = clamp16(filteredSample);
            st.s2 = st.s1;
            st.s1 = clamped;
            out28[sampleIdx] = clamped;
        }
    }

    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate)
    {
        if (!data || sizeBytes < 48)
            return false;

        const uint32_t magic = (static_cast<uint32_t>(data[0]) << 24) |
                               (static_cast<uint32_t>(data[1]) << 16) |
                               (static_cast<uint32_t>(data[2]) << 8) |
                               static_cast<uint32_t>(data[3]);
        if (magic != 0x56414770u)
        {
            const uint32_t magicLE = (static_cast<uint32_t>(data[3]) << 24) |
                                     (static_cast<uint32_t>(data[2]) << 16) |
                                     (static_cast<uint32_t>(data[1]) << 8) |
                                     static_cast<uint32_t>(data[0]);
            if (magicLE != 0x56414770u)
                return false;
        }

        uint32_t dataSize = (static_cast<uint32_t>(data[0x0c]) << 24) |
                            (static_cast<uint32_t>(data[0x0d]) << 16) |
                            (static_cast<uint32_t>(data[0x0e]) << 8) |
                            static_cast<uint32_t>(data[0x0f]);
        outSampleRate = (static_cast<uint32_t>(data[0x10]) << 24) |
                        (static_cast<uint32_t>(data[0x11]) << 16) |
                        (static_cast<uint32_t>(data[0x12]) << 8) |
                        static_cast<uint32_t>(data[0x13]);
        if (outSampleRate == 0)
            outSampleRate = 44100;

        const uint32_t numBlocks = (dataSize + 15) / 16;
        outPcm.clear();
        outPcm.reserve(numBlocks * 28);

        AdpcmState st;
        const uint8_t *block = data + 48;

        // PS2X_AUDIO_VAGEND (default ON; =0 decodes every block as before): stop at the block that
        // carries XAFLAG_LOOP_END (byte 1 bit 0), as the SPU2 does -- PCSX2 SPU2/Mixer.cpp
        // DecodeSamples: on LOOP_END it sets ENDX, rewinds NextA to LoopStartA and, with
        // XAFLAG_LOOP clear, calls vc.Stop(); the voice never reaches the bytes after it. Every sample in this game's banks
        // is followed by the SDK terminator block (00 07 77 77 ...: shift 0, flag 7, nibbles 7 =
        // +28672 for 28 samples) and up to five zero blocks before the next sample starts, so a
        // straight decode to the directory's next offset appended a full-scale DC step -- the
        // audible CLICK after every one-shot sound. A LOOP_END block whose LOOP bit is set would
        // loop on hardware; this one-shot decode stops there too (no bank sample carries flag 3).
        static const bool stopAtEnd = []() {
            const char *e = std::getenv("PS2X_AUDIO_VAGEND");
            return !(e && e[0] == '0');
        }();

        for (uint32_t b = 0; b < numBlocks && (block + 16) <= data + sizeBytes; ++b, block += 16)
        {
            int16_t pcm[28];
            decodeBlock(block, st, pcm);
            outPcm.insert(outPcm.end(), pcm, pcm + 28);
            if (stopAtEnd && (block[1] & 0x01u))
                break;
        }

        return true;
    }
}
