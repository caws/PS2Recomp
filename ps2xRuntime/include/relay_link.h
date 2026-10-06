#pragma once

// The online co-op link through a relay (tools/netplay_server/): a two-player session on the relay, and on top
// of it an end-to-end encrypted, reliable, in-order channel to the other player. The co-op page uses it for its
// pre-game exchange (host / join by code / public list); the game's transport (netplay.cpp) uses it for the
// game link after the restart.
//
//   - Settings: LOTR_RELAY_SERVER (wss:// or ws://host[:port]/relay) and LOTR_RELAY_KEY (the server's public key,
//     base64). Both set = the page offers "Internet (relay)". The page sets them from config/online.ini
//     (coop_page.cpp); exported by the player, they win over the file.
//   - End to end: each player makes an X25519 key per session and puts its public half in its relay `info`; the
//     two derive one key per direction (and per phase: the page and the game use different keys, so a counter
//     restarting in the new process never reuses a nonce). The relay passes sealed frames it cannot read.
//   - Reliable: frames carry a sequence number and the other side's acknowledgement; what is not acknowledged is
//     kept and sent again after a reconnect (the relay drops what it cannot deliver). A dropped connection to the
//     relay is reopened and the session resumed with its ticket, so a short outage is a stall, not a disconnect.
//   - Across the restart: handover() is a string for LOTR_RELAY_SESSION; the game's process restores() it and
//     resumes the same relay session (the server keeps a player's place for RELAY_RESUME_GRACE).
// Generic (no guest addresses, no region).

#include "relay_client.h"

#include <chrono>
#include <deque>
#include <string>
#include <vector>

namespace lotr::netplay::relay
{
    class Link
    {
    public:
        enum class State
        {
            Idle,
            Opening,     // connecting to the relay, then hosting / joining / resuming
            Waiting,     // host: hosted, nobody has joined yet (code() is set)
            Connected,   // the other player is in the session and the encrypted channel is up
            Failed,      // over: error() says why
        };

        // Where the link reports its connection events (dropped, reconnecting, resumed, failed); none by default.
        void setLogger(void (*log)(const std::string& line)) { log_ = log; }

        static bool configured();               // LOTR_RELAY_SERVER and LOTR_RELAY_KEY are set
        static std::string serverAddress();     // LOTR_RELAY_SERVER

        // The page: host (2 players), or join a code. `character` goes in this player's info for the game list.
        bool host(bool listed, const std::string& name, int character);
        bool join(const std::string& code, int character);
        // The public games, refreshed every few seconds while browse() is on (a connection of its own).
        void browse(bool on);
        const std::vector<ListedGame>& games() const { return games_; }
        static int listedCharacter(const ListedGame& g);   // the host's character from its info, or -1

        // The game: continue the session handover() described, in the "game" phase.
        bool restore(const std::string& handover);
        std::string handover() const;

        void pump();                          // call every frame
        bool send(const Bytes& message);      // queued even while the peer is away; false once Failed
        bool receive(Bytes& message);         // the next message from the other player, in order
        bool flush(int timeoutMs);            // try to get queued frames onto the wire (before a restart)
        void leave();                         // end it on purpose: the other player is told
        void drop();                          // lose the relay connection, as a network failure would (tests)

        State state() const { return state_; }
        bool peerAway() const { return peerAway_; }
        const std::string& code() const { return code_; }   // the join code (host), "K7QXM2P9"
        const std::string& error() const { return error_; }
        int slot() const { return slot_; }
        size_t unacked() const { return unacked_.size(); }   // sent, not yet acknowledged (diagnostics)

    private:
        void reset();
        void newKeys();
        Bytes info() const;
        void openClient();
        void fail(const std::string& why);
        void onEvent(Event& e);
        bool peerInfo(const Bytes& info);   // takes the other player's public key; false if unusable
        void peerGone();
        void deriveKeys();
        void sendFrame(uint8_t kind, uint64_t nonce, const uint8_t* plain, size_t n, bool keep);
        void onFrame(const Bytes& frame);
        void onPacket(const Bytes& packet);
        void flushOut();
        void resendUnacked();
        void sendAck();

        Client client_, lister_;
        State state_ = State::Idle;
        enum class Want { None, Host, Join, Resume } want_ = Want::None;
        std::string phase_ = "page", code_, error_, name_;
        bool listed_ = false, browsing_ = false, left_ = false, peerAway_ = false, keysReady_ = false;
        int character_ = 0, slot_ = -1, peerSlot_ = -1;
        Bytes ticket_;
        uint8_t secret_[32] = {}, public_[32] = {}, peerPublic_[32] = {}, txKey_[32] = {}, rxKey_[32] = {};
        uint64_t sendSeq_ = 0, recvNext_ = 0, peerAcked_ = 0, ackNonce_ = 0;
        std::deque<std::pair<uint64_t, Bytes>> unacked_;   // (seq, the sealed frame as sent)
        std::deque<Bytes> outbox_, inbox_;   // frames to pack and send; messages received
        double budget_ = 48000;              // bytes we may send now (refilled over time)
        bool ackDue_ = false;
        std::chrono::steady_clock::time_point ackDueSince_{}, retryAt_{}, droppedAt_{}, nextList_{}, budgetAt_{};   // droppedAt_: {} = connected
        std::vector<ListedGame> games_;
        void (*log_)(const std::string&) = nullptr;
        void note(const std::string& line) { if (log_) log_(line); }
    };
}
