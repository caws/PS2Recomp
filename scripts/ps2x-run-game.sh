#!/usr/bin/env bash
set -euo pipefail

# Run a game built with this PS2Recomp toolchain.
#
# Usage: ps2x-run-game.sh <game_dir> [run_log]
#
# Lives in the fork (scripts/) since 2026-10-02 (was LLMPS2Recomp's ps2x-run-game.sh).
#
# The ELF path is read from <game_dir>/recomp/config.toml `input`. The runner
# binary is whatever was last built by ps2x-build-game.sh (one active game at a
# time — static recompilation).
#
# run_log: optional log-file target (2nd arg, or PS2X_RUN_LOG env). With NO log
# target the runner's output goes to the CONSOLE (stdout/stderr passthrough) —
# for interactive terminal use. Automated/agent invocations MUST pass a log file
# (canonically tmp/run.txt; a spinning runner emits hundreds of MB/s), and runs
# that may OVERLAP (background retry loop + manual run) should use DISTINCT
# paths so logs don't interleave. Relative paths resolve against <game_dir>/tmp.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GAME_DIR="${1:-}"
[[ -n "$GAME_DIR" ]] || { echo "ERROR: no <game_dir> given. Usage: $0 <game_dir>"; exit 1; }
[[ -d "$GAME_DIR" ]] || { echo "ERROR: game dir not found: $GAME_DIR"; exit 1; }
GAME_DIR="$(cd "$GAME_DIR" && pwd)"

# The game's runnable binary is named after the GAME DIR, not after the CMake target: each game
# gets a distinctly-named executable, `pgrep -x <name>` identifies it, and a player sees the game's
# name rather than "ps2EntryRunner". Nothing about a game is hardcoded here -- the name is derived.
# The CMake target keeps its upstream name on purpose: renaming it in our fork would conflict with
# every `git merge upstream/main`. PS2X_RUNNER_NAME overrides the derivation.
RUNNER_NAME="${PS2X_RUNNER_NAME:-$(basename "$GAME_DIR")}"
RUNNER="$GAME_DIR/$RUNNER_NAME"   # per-game binary, placed here by ps2x-build-game.sh
# Log target: 2nd arg > PS2X_RUN_LOG env > default = console passthrough (no redirect).
RUN_LOG="${2:-${PS2X_RUN_LOG:-}}"
if [[ -n "$RUN_LOG" && "$RUN_LOG" != /* ]]; then RUN_LOG="$GAME_DIR/tmp/$RUN_LOG"; fi
CONFIG="$GAME_DIR/recomp/config.toml"

[[ -f "$CONFIG" ]] || { echo "ERROR: config.toml not found: $CONFIG"; exit 1; }

ELF_REL="$(grep -E "^[[:space:]]*input[[:space:]]*=" "$CONFIG" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]+)".*/\1/')"
[[ -n "$ELF_REL" ]] || { echo "ERROR: could not read 'input' from $CONFIG"; exit 1; }
ELF="$GAME_DIR/${ELF_REL#./}"

echo "========================================="
echo " PS2Recomp Run"
echo " game: $GAME_DIR"
echo "========================================="
echo

if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: runner not found: $RUNNER"
    # The binary was called ps2EntryRunner until cont.346t. If one is still lying around, say so
    # -- otherwise "not found" next to a perfectly good 1 GB binary reads as a broken build.
    if [[ -f "$GAME_DIR/ps2EntryRunner" ]]; then
        echo "       Found the old ps2EntryRunner here -- the runner is now named '$RUNNER_NAME'."
        echo "       Rename it:  mv '$GAME_DIR/ps2EntryRunner' '$RUNNER'"
        echo "       or rebuild:"
    fi
    echo "Run:   scripts/ps2x-build-game.sh $GAME_DIR"
    exit 1
fi
[[ -f "$ELF" ]] || { echo "ERROR: ELF not found (from config 'input'): $ELF"; exit 1; }

# --------------------------------------------------
# Disc preflight
# --------------------------------------------------
# A static recompilation is tied to ONE exact executable, so the wrong disc does not fail
# cleanly -- it fails a long way from the cause. Check before launching. Costs milliseconds
# (the ELF hash plus a stat sweep); ps2x-verify-disc.sh --deep checksums everything.
if [[ -x "$SCRIPT_DIR/ps2x-verify-disc.sh" ]]; then
    # `|| DISC_RC=$?` not a bare call: these scripts run under `set -e`, which would abort
    # here on any non-zero code -- killing the run with no explanation, before the message
    # below could say which of the failures it was.
    DISC_RC=0
    "$SCRIPT_DIR/ps2x-verify-disc.sh" "$GAME_DIR" --quiet || DISC_RC=$?
    # 5 = no manifest for this game yet: nothing to check against, and ps2x-verify-disc.sh has
    # already said so. Everything else is a real finding and its message is the explanation.
    if [[ $DISC_RC -ne 0 && $DISC_RC -ne 5 ]]; then
        echo "" >&2
        echo "Stopping: the disc copy in $GAME_DIR/gamefiles did not verify (see above)." >&2
        exit $DISC_RC
    fi
fi

# Wrap the invocation in `timeout` to cap a spin:
#   timeout 20 scripts/ps2x-run-game.sh <game_dir> run.txt
mkdir -p "$GAME_DIR/tmp"
# The pre-game launcher waits for a person. With NO TERMINAL on stdin nobody is there (an automated
# run), so skip it; a player's terminal run -- logged or not -- shows it. An explicit PS2X_LAUNCHER wins.
# (Deciding by "is there a log file" was wrong: play.sh logs, and is interactive.)
if [[ -z "${PS2X_LAUNCHER:-}" && ! -t 0 ]]; then
    export PS2X_LAUNCHER=0
fi
# Game-side overrides (e.g. DBCMAN HLE serving) read gamefiles/ from this env var, so the
# path is never hardcoded and survives a game-folder rename.
export PS2_GAMEFILES="$GAME_DIR/gamefiles"
# cont.346p: the ELF lives INSIDE gamefiles/, and memory cards default to the ELF's own directory
# -- so mc0/ and mc1/ live in gamefiles/ too. EVERYTHING for the game is one folder the player can
# drop their disc into (user, 2026-09-19). No pin here: the runtime default already does this.
# PS2X_MC_ROOT names a directory CONTAINING mc0/ and mc1/; set it only to switch cards (the
# two-card test harness does).
if [[ -n "$RUN_LOG" ]]; then
    mkdir -p "$(dirname "$RUN_LOG")"
    # ALWAYS remove the old log first, so the log that exists afterward is GUARANTEED to be
    # from THIS run. If the runner fails to launch/write, it will be absent/empty rather than
    # a stale leftover that looks like a fresh result (this bit us: a stale run.txt was mistaken
    # for the current run, hiding a fresh binary's output).
    rm -f "$RUN_LOG"
    echo "log: $RUN_LOG"
    exec "$RUNNER" "$ELF" > "$RUN_LOG" 2>&1
else
    # Interactive/console mode: output streams to the terminal. NB a spinning runner can emit
    # hundreds of MB/s — automated invocations should always pass a log file instead.
    echo "log: (console)"
    exec "$RUNNER" "$ELF"
fi
