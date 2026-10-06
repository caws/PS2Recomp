#pragma once

// The relay's cryptography, on Monocypher (src/netplay/monocypher/): the Noise handshake to the relay server,
// and the primitives under it. Generic (no guest addresses, no region): tools/netplay_server/tests/client_cpp/
// builds it on its own to test it against the real server.
//
// Noise_NK_25519_ChaChaPoly_BLAKE2b (https://noiseprotocol.org/noise.html), the initiator side only: the game
// knows the server's public key and the handshake proves the server holds the private key. Monocypher has the
// primitives (X25519, ChaCha20, Poly1305, BLAKE2b) but not Noise's two compounds, so they are built here:
// ChaCha20-Poly1305 as RFC 8439 defines it (12-byte nonce = 4 zero bytes + the 64-bit counter, little-endian),
// and HMAC/HKDF over BLAKE2b (block 128, output 64). Checked against RFC 8439's test vector and, end to end,
// against the server (Go, github.com/flynn/noise).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lotr::netplay::relay
{
    using Bytes = std::vector<uint8_t>;

    bool randomBytes(uint8_t* out, size_t n);   // the OS's CSPRNG (getrandom)

    // RFC 8439 AEAD. seal appends the 16-byte tag; open returns false if the tag does not match.
    Bytes chachaPolySeal(const uint8_t key[32], uint64_t nonce, const uint8_t* ad, size_t adLen, const uint8_t* plain, size_t len);
    bool chachaPolyOpen(const uint8_t key[32], uint64_t nonce, const uint8_t* ad, size_t adLen, const uint8_t* sealed, size_t len, Bytes& plain);

    // One direction of a Noise transport: a key and its message counter.
    struct CipherState
    {
        uint8_t key[32] = {};
        uint64_t nonce = 0;
        Bytes encrypt(const Bytes& plain) { return chachaPolySeal(key, nonce++, nullptr, 0, plain.data(), plain.size()); }
        bool decrypt(const uint8_t* sealed, size_t len, Bytes& plain) { return chachaPolyOpen(key, nonce++, nullptr, 0, sealed, len, plain); }
    };

    // Noise NK, initiator. writeMessage1 -> send it; readMessage2 with the reply -> send/recv are ready.
    class NoiseInitiator
    {
    public:
        NoiseInitiator(const uint8_t serverPublic[32], const std::string& prologue);
        ~NoiseInitiator();
        Bytes writeMessage1(const Bytes& payload);                 // -> e, es
        bool readMessage2(const Bytes& message, Bytes& payload);   // <- e, ee; false = not the server we know
        CipherState send, recv;

    private:
        void mixHash(const uint8_t* data, size_t n);
        void mixKey(const uint8_t* ikm, size_t n);
        uint8_t h_[64], ck_[64], k_[32], rs_[32], e_[32], ePub_[32];
        bool hasKey_ = false;
        uint64_t n_ = 0;
    };

    // Base64 (standard alphabet, padded) for keys in settings and the session hand-over.
    std::string base64Encode(const uint8_t* p, size_t n);
    bool base64Decode(const std::string& s, Bytes& out);
    std::string hexEncode(const uint8_t* p, size_t n);
    bool hexDecode(const std::string& s, Bytes& out);
}
