#pragma once

// A client of the netplay relay (tools/netplay_server/, its README.md "Protocol" is the contract): one
// WebSocket to the server, the Noise handshake over it, and the relay's messages. Non-blocking and
// single-threaded: the owner calls pump() often (every frame) and takes events with next().
//
// wss:// (TLS, through the system's OpenSSL: relay_tls.h) or plain ws://. Either way the Noise handshake is what
// authenticates the server and encrypts the traffic; TLS gets it through proxies and networks that require it.
// Generic (no guest addresses, no region), like relay_crypto.

#include "relay_crypto.h"

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace lotr::netplay::relay
{
    constexpr uint16_t kProtocolVersion = 1;
    constexpr int kTicketLen = 16;
    constexpr int kSlotAll = 0xFF;

    struct Member { int slot = -1; bool away = false; Bytes info; };
    struct ListedGame { std::string code, name; int players = 0, maxPlayers = 0; Bytes hostInfo; };

    struct Event
    {
        enum class Type
        {
            Ready,          // connected and authenticated: requests may be sent
            Hosted,         // code, ticket, slot, maxPlayers
            Joined,         // ticket, slot, maxPlayers, members
            Resumed,        // slot, maxPlayers, members
            GameList,       // games
            Received,       // slot (from), payload
            MemberJoined,   // slot, info
            MemberLeft,     // slot, reason
            MemberAway,     // slot
            MemberBack,     // slot
            SessionEnded,   // reason
            Error,          // code, text
            Closed,         // the connection is gone (text says why); the client is idle again
        };
        explicit Event(Type t = Type::Ready) : type(t) {}
        Type type;
        int slot = -1, maxPlayers = 0, code = 0, reason = 0;
        std::string text;   // the join code (Hosted), an error's text, why it closed
        Bytes ticket, payload, info;
        std::vector<Member> members;
        std::vector<ListedGame> games;
    };

    // Relay error codes and reasons (README.md).
    enum : int { kErrBadMessage = 1, kErrNotInSession, kErrAlreadyInSession, kErrNoSuchGame, kErrGameFull, kErrRateLimited,
                 kErrServerFull, kErrBadTicket, kErrNotHost, kErrBadSlot };
    enum : int { kReasonLeft = 1, kReasonTimedOut = 2, kReasonHostLeft = 3 };

    class Client
    {
    public:
        Client();
        ~Client();

        // Starts connecting to `url` (wss:// or ws://host[:port]/path) and authenticating with `serverKey` (base64).
        // false + why when the settings themselves are wrong; network failures arrive as a Closed event.
        bool open(const std::string& url, const std::string& serverKey, std::string& why);
        void close();
        bool connecting() const;   // between open() and Ready (or Closed)
        bool ready() const;        // authenticated: requests may be sent
        bool flushed() const;      // nothing is waiting to be written

        void pump();               // send and receive what can be, without blocking
        bool next(Event& ev);      // the oldest event, if any

        void hostGame(int maxPlayers, bool listed, const std::string& name, const Bytes& info);
        void joinGame(const std::string& code, const Bytes& info);
        void listGames();
        void resume(const Bytes& ticket);
        void leave();
        void setListing(bool listed);
        void sendTo(int slot, const Bytes& payload);

    private:
        struct Impl;
        std::unique_ptr<Impl> d_;
    };

    // "K7QXM2P9" -> "K7QX-M2P9" for display.
    std::string formatCode(const std::string& code);
}
