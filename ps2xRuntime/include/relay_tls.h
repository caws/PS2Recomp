#pragma once

// TLS for wss:// relay addresses, through the system's OpenSSL, loaded at run time (dlopen of libssl.so.3, then
// libssl.so.1.1): nothing is linked, so the game builds and runs without it, and only a player who connects to a
// wss:// relay needs it (every mainstream Linux desktop has it). Non-blocking, over a socket the caller connected.
//
// The certificate is NOT checked, on purpose: what proves the relay is ours is the Noise handshake against the key
// the player was given (relay_crypto.h). TLS is here to get through proxies and networks that require it (a
// Cloudflare tunnel's hostname, for one), and a forged certificate would only see Noise ciphertext.
// Generic (no guest addresses, no region).

#include <cstddef>
#include <string>
#include <sys/types.h>

namespace lotr::netplay::relay
{
    class TlsConnection
    {
    public:
        ~TlsConnection();
        static bool available(std::string& why);   // OpenSSL could be loaded

        // Begin TLS on the connected, non-blocking socket fd; `host` goes in SNI (proxies route on it).
        bool start(int fd, const std::string& host, std::string& why);
        // Advance the handshake: 1 = done, 0 = call again later, -1 = failed (why says why).
        int handshake(std::string& why);
        // Like recv / send on a non-blocking socket: > 0 bytes, 0 = the peer closed (read only),
        // -1 with errno EAGAIN = try later, -1 otherwise = failed.
        ssize_t read(void* buf, size_t n);
        ssize_t write(const void* buf, size_t n);
        void close();

    private:
        void* ssl_ = nullptr;
    };
}
