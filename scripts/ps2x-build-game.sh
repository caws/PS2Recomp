#!/usr/bin/env bash
set -euo pipefail

# Build a game with this PS2Recomp toolchain (the clone this script lives in).
#
# Usage: ps2x-build-game.sh <game_dir> [--skip-regen] [--changed-recomp] [--skip-build] [--release] [--no-debug-info]
#
# Lives in the fork (scripts/) since 2026-10-02 -- moved from LLMPS2Recomp's 03_build_game.sh -- so
# every game built with this toolchain shares one copy. Prerequisite: scripts/ps2x-setup.sh (builds
# ps2_recomp). A game repo usually calls this through its own scripts/build.sh.
#
# <game_dir> is a self-contained per-game repo with this layout:
#   <game_dir>/
#     <ELF>            # path declared in recomp/config.toml `input`
#     recomp/          # config.toml + functions.csv  (provided inputs)
#     src/             # register_overrides.cpp (+ .h) (our overrides: the base game)
#     mods/            # OPTIONAL, one subfolder per mod (enhancement hook bodies)
#     tmp/generated/   # ps2_recomp output            (per-game scratch)
#
# All per-game paths come from recomp/config.toml; nothing about the game is
# hardcoded here. This repo only contributes the toolchain + runner.
#
# Environment: PS2X_RUNNER_NAME (the binary's name; default = basename of <game_dir>),
# PS2X_GAME_SRC_ROOTS (':'-separated override-source folders; default = <game_dir>/src + <game_dir>/mods).

SKIP_BUILD=false
CHANGED_RECOMP=false
SKIP_REGEN=false
BUILD_TYPE=RelWithDebInfo   # default: -O2 + debug info (gdb works). --release = CMake Release (-O3, no LTO).
DEBUG_INFO=true             # --no-debug-info: the same -O2 build without -g -- what a player wants
                            # (a ~860 MB binary shrinks to a fraction, and the compile is lighter).
GAME_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)     SKIP_BUILD=true;     shift ;;
        --changed-recomp) CHANGED_RECOMP=true; shift ;;
        --skip-regen)     SKIP_REGEN=true;     shift ;;
        --release)        BUILD_TYPE=Release;  shift ;;
        --no-debug-info)  DEBUG_INFO=false;    shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  [[ -z "$GAME_DIR" ]] || { echo "Unexpected extra argument: $1"; exit 1; }
            GAME_DIR="$1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS2RECOMP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # the toolchain clone = this repo

[[ -n "$GAME_DIR" ]] || { echo "ERROR: no <game_dir> given. Usage: $0 <game_dir> [flags]"; exit 1; }
[[ -d "$GAME_DIR" ]] || { echo "ERROR: game dir not found: $GAME_DIR"; exit 1; }
GAME_DIR="$(cd "$GAME_DIR" && pwd)"   # absolute

# Completion marker: written on EVERY exit path (success, build failure, validation error) via
# the trap below, containing the exit code. `pgrep -f <this script>` is NOT a reliable
# "is it still running" check — if the poll command itself is a wrapper shell whose command-line
# contains the script's name (e.g. `until ! pgrep -f ps2x-build-game.sh; do ...`),
# pgrep matches ITS OWN wrapper and the loop never sees the real process exit. Poll for this file
# instead: `until [[ -f <game_dir>/tmp/.build_status ]]; do sleep 5; done; cat <game_dir>/tmp/.build_status`
# (0 = success). Removed up front so a stale marker from a previous run can never be mistaken for
# this run's result.
mkdir -p "$GAME_DIR/tmp"
rm -f "$GAME_DIR/tmp/.build_status"
trap 'echo "$?" > "$GAME_DIR/tmp/.build_status"' EXIT

# The game's runnable binary is named after the GAME DIR, not after the CMake target: each game
# gets a distinctly-named executable, `pgrep -x <name>` identifies it, and a player sees the game's
# name rather than "ps2EntryRunner". Nothing about a game is hardcoded here -- the name is derived.
# The CMake target keeps its upstream name on purpose: renaming it in our fork would conflict with
# every `git merge upstream/main`. PS2X_RUNNER_NAME overrides the derivation.
RUNNER_NAME="${PS2X_RUNNER_NAME:-$(basename "$GAME_DIR")}"

PS2_RECOMP_BIN="$PS2RECOMP_ROOT/out/build/ps2xRecomp/ps2_recomp"
RUNTIME_SRC="$PS2RECOMP_ROOT/ps2xRuntime/src/runner"
RUNTIME_INCLUDE="$PS2RECOMP_ROOT/ps2xRuntime/include"

CONFIG="$GAME_DIR/recomp/config.toml"

# Read a quoted string value from the [general] section of the config.
read_cfg() { grep -E "^[[:space:]]*$1[[:space:]]*=" "$CONFIG" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]+)".*/\1/'; }

echo "========================================="
echo " PS2Recomp Build Pipeline"
echo " game: $GAME_DIR"
echo "========================================="

# --------------------------------------------------
# Validation
# --------------------------------------------------

[[ -x "$PS2_RECOMP_BIN" ]] || { echo "ERROR: ps2_recomp not built (run $SCRIPT_DIR/ps2x-setup.sh): $PS2_RECOMP_BIN"; exit 1; }
[[ -f "$CONFIG" ]]         || { echo "ERROR: config.toml not found: $CONFIG"; exit 1; }

ELF_REL="$(read_cfg input)"
OUT_REL="$(read_cfg output)"
[[ -n "$ELF_REL" ]] || { echo "ERROR: could not read 'input' from $CONFIG"; exit 1; }
[[ -n "$OUT_REL" ]] || { echo "ERROR: could not read 'output' from $CONFIG"; exit 1; }
ELF="$GAME_DIR/${ELF_REL#./}"
GENERATED="$GAME_DIR/${OUT_REL#./}"

[[ -f "$ELF" ]] || { echo "ERROR: ELF not found (from config 'input'): $ELF"; exit 1; }

# --------------------------------------------------
# Disc preflight
# --------------------------------------------------
# A static recompilation is tied to ONE exact executable, so the wrong disc does not fail
# cleanly -- it fails a long way from the cause. Check before spending minutes recompiling against it. Costs milliseconds
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

# --------------------------------------------------
# Generate code
# --------------------------------------------------

if [[ "$SKIP_REGEN" == true ]]; then
    echo
    echo "[1/3] Skipping recompilation (--skip-regen)"
    if [[ ! -s "$GENERATED/ps2_recompiled_functions.h" ]]; then
        echo "ERROR: --skip-regen requires a prior successful recompilation ($GENERATED is empty or missing)"
        echo "  Run without --skip-regen first to regenerate."
        exit 1
    fi
else
    echo
    echo "[1/3] Generating recompilation output..."

    rm -rf "$GENERATED"
    mkdir -p "$GENERATED"

    # ps2_recomp resolves the config's relative paths (input, ghidra_output,
    # output) against CWD, so run it from the game dir.
    pushd "$GAME_DIR" >/dev/null
    "$PS2_RECOMP_BIN" "recomp/config.toml"
    popd >/dev/null

    # Guard against the silent-crash case: ps2_recomp swallows exceptions and
    # returns 0, leaving generated files empty.
    if [[ ! -s "$GENERATED/ps2_recompiled_functions.h" ]]; then
        echo "ERROR: ps2_recomp generated an empty ps2_recompiled_functions.h"
        echo "  Recompilation silently failed. Runner files were NOT modified."
        echo "  Check stderr above for the actual error."
        exit 1
    fi
fi

# --------------------------------------------------
# Install generated files + overrides into the runner
# --------------------------------------------------

echo
echo "[2/3] Installing generated files + overrides..."

# Copy a file only if new or changed, so cmake can skip unchanged TUs.
install_if_changed() {
    local src="$1" dst_dir="$2"
    local dst="$dst_dir/$(basename "$src")"
    if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
        cp "$src" "$dst"
    fi
}

# The game's overrides may be organized into SUBFOLDERS (src/overrides/..., src/hle/..., etc.),
# but the runner's CMake glob is FLAT and non-recursive ("src/runner/*.cpp", and headers are
# included as "foo.h" from a single include dir), so we collect the tree recursively and install
# by BASENAME. Two files sharing a basename would flatten onto each other and one would silently
# vanish from the build — so that is a hard error, not a warning. Same for a basename that
# collides with a generated file (ours would clobber the recompiled function, or vice versa).
#
# TWO trees are collected, and they install identically — the split is editorial, not a build
# difference: src/ is the faithful base game (the hooks that make the original run) and mods/ holds
# enhancements, one subfolder per mod. mods/ is OPTIONAL; a game repo without one builds unchanged.
# The basename namespace is SHARED across both, so a mod cannot shadow a base module by accident.
#
# PS2X_GAME_SRC_ROOTS overrides the two defaults: a ':'-separated list of directories, collected the
# same way (same flat install, same duplicate-basename check). For a game repo whose sources do not sit
# in <game_dir> -- e.g. one repo for several regional discs, where each disc has its own <game_dir>
# (recomp/ + gamefiles/) but all share one src/ + mods/, plus a per-disc folder of generated headers.
if [[ -n "${PS2X_GAME_SRC_ROOTS:-}" ]]; then
    GAME_SRC_ROOTS=()
    IFS=':' read -r -a _roots <<< "$PS2X_GAME_SRC_ROOTS"
    for _r in "${_roots[@]}"; do
        [[ -n "$_r" ]] || continue
        [[ -d "$_r" ]] || { echo "ERROR: PS2X_GAME_SRC_ROOTS names a missing directory: $_r"; exit 1; }
        GAME_SRC_ROOTS+=("$(cd "$_r" && pwd)")
    done
    [[ ${#GAME_SRC_ROOTS[@]} -gt 0 ]] || { echo "ERROR: PS2X_GAME_SRC_ROOTS is set but names no directory"; exit 1; }
    echo "  override sources from PS2X_GAME_SRC_ROOTS: ${GAME_SRC_ROOTS[*]}"
else
    GAME_SRC_ROOTS=("$GAME_DIR/src")
    [[ -d "$GAME_DIR/mods" ]] && GAME_SRC_ROOTS+=("$GAME_DIR/mods")
fi
GAME_SRC_FILES=()
while IFS= read -r -d '' f; do GAME_SRC_FILES+=("$f"); done \
    < <(find "${GAME_SRC_ROOTS[@]}" -type f \( -name '*.cpp' -o -name '*.h' \) -print0 | sort -z)

declare -A GAME_SRC_BY_BASE=()
for f in "${GAME_SRC_FILES[@]}"; do
    base="$(basename "$f")"
    if [[ -n "${GAME_SRC_BY_BASE[$base]:-}" ]]; then
        echo "ERROR: duplicate override basename '$base' — the flat install would clobber one:"
        echo "    ${GAME_SRC_BY_BASE[$base]}"
        echo "    $f"
        echo "  Override sources (src/ AND mods/) are installed into the runner by basename;"
        echo "  keep them unique across BOTH trees."
        exit 1
    fi
    if [[ -f "$GENERATED/$base" ]]; then
        echo "ERROR: override '$f' collides with generated file '$GENERATED/$base'."
        echo "  Rename the override; it would overwrite a recompiled function."
        exit 1
    fi
    GAME_SRC_BY_BASE[$base]="$f"
done

if [[ "$CHANGED_RECOMP" == true ]]; then
    # Smart install: only touch changed files. Remove stale runner files that are
    # no longer in the game's generated/ or src/ + mods/ overrides. NOTE: the override check is
    # against the recursive basename map above — testing "$GAME_DIR/src/$base" would treat
    # every file living in an src/ or mods/ SUBFOLDER as stale and delete it on each fast build.
    for f in "$RUNTIME_SRC"/*.cpp; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        if [[ ! -f "$GENERATED/$base" ]] && [[ -z "${GAME_SRC_BY_BASE[$base]:-}" ]]; then
            echo "  removing stale: $base"
            rm "$f"
        fi
    done

    for f in "$GENERATED"/*.cpp "$GENERATED"/*.h; do
        [[ -f "$f" ]] || continue
        if [[ "${f##*.}" == "h" ]]; then install_if_changed "$f" "$RUNTIME_INCLUDE"
        else                              install_if_changed "$f" "$RUNTIME_SRC"; fi
    done
else
    # Full install: wipe runner and copy everything (clean slate).
    find "$RUNTIME_SRC" -maxdepth 1 -name "*.cpp" -delete
    cp "$GENERATED"/*.cpp "$RUNTIME_SRC"/
    cp "$GENERATED"/*.h "$RUNTIME_INCLUDE"/
fi

# Always install the game's overrides (they change independently of the recomp). Flattened by
# basename from the whole src/ + mods/ tree (see GAME_SRC_FILES above): .cpp -> runner/, .h -> include/.
OVERRIDE_CPP_BASENAMES=()
for f in "${GAME_SRC_FILES[@]}"; do
    if [[ "${f##*.}" == "h" ]]; then install_if_changed "$f" "$RUNTIME_INCLUDE"
    else                             install_if_changed "$f" "$RUNTIME_SRC"
                                     OVERRIDE_CPP_BASENAMES+=("$(basename "$f")"); fi
done

# Tell the runtime's CMake which runner sources are hand-written game overrides, so it can keep them
# OUT of the unity build (engine patch 08). Without this, adding/removing an override module
# re-shuffles the unity batches and forces a near-full rebuild, and file-scope statics from separate
# modules can collide inside a shared unity TU. Written via install_if_changed semantics (only
# rewritten when the set changes) so an unchanged manifest never triggers a cmake reconfigure.
MANIFEST_TMP="$(mktemp)"
printf '%s\n' "${OVERRIDE_CPP_BASENAMES[@]}" > "$MANIFEST_TMP"
# ★★★★★ cont.346r LEGAL GUARD: the generated function table must never become committable.
# `ps2xRuntime/src/runner/` is gitignored and regenerated every build, but ONE file inside it is
# TRACKED upstream -- register_functions.cpp, a 438-byte stub in ran-j's history that our build
# overwrites with ~6.6 MB / 81k lines of the GAME's function table (every guest address, derived
# from the player's ELF). Distributing that is precisely the risk we must not take, and the only
# thing preventing it was a `skip-worktree` bit someone set by hand -- a LOCAL index flag that does
# not survive a fresh clone and does not stop `git add -f`.
# Re-assert it on every build, so any clone is protected from its first build onward, and say so
# when it had to be (re)applied rather than doing it silently.
if [[ -d "$PS2RECOMP_ROOT/.git" ]]; then
    _rf="ps2xRuntime/src/runner/register_functions.cpp"
    if git -C "$PS2RECOMP_ROOT" ls-files --error-unmatch "$_rf" >/dev/null 2>&1; then
        if ! git -C "$PS2RECOMP_ROOT" ls-files -v "$_rf" | grep -q '^S'; then
            git -C "$PS2RECOMP_ROOT" update-index --skip-worktree "$_rf" 2>/dev/null \
                && echo "  [guard] skip-worktree (re)applied to $_rf -- the generated table stays uncommittable"
        fi
    fi
fi

MANIFEST="$RUNTIME_SRC/game_overrides.manifest"
if [[ ! -f "$MANIFEST" ]] || ! cmp -s "$MANIFEST_TMP" "$MANIFEST"; then
    mv "$MANIFEST_TMP" "$MANIFEST"
    echo "  override manifest updated (${#OVERRIDE_CPP_BASENAMES[@]} module(s), excluded from unity build)"
else
    rm -f "$MANIFEST_TMP"
fi

# --------------------------------------------------
# Build runner
# --------------------------------------------------

if [ "$SKIP_BUILD" = false ]; then
    echo
    echo "[3/3] Building $RUNNER_NAME..."
    echo "    build type: $BUILD_TYPE  debug info: $DEBUG_INFO  (jobs: ${BUILD_JOBS:-6})"
    # Remove the game's runnable binary BEFORE building, so a failed/incomplete build can't
    # leave a STALE one that ps2x-run-game.sh would silently run as if it were fresh. The new
    # binary is mv'd into place only after a successful cmake build below; if the build fails
    # it is simply absent (04 errors out) rather than running old code.
    rm -f "$GAME_DIR/$RUNNER_NAME"
    # A binary under the pre-rename name would still be runnable by an out-of-date script or an
    # old shell line, and would be stale from this build onward. Drop it too, and say so.
    if [[ "$RUNNER_NAME" != "ps2EntryRunner" && -f "$GAME_DIR/ps2EntryRunner" ]]; then
        echo "    removing the legacy ps2EntryRunner (the runner is now '$RUNNER_NAME')"
        rm -f "$GAME_DIR/ps2EntryRunner"
    fi
    # Link with lld — the ~900MB runner relinks in seconds vs minutes on GNU ld.
    # Fallback if lld misbehaves: -fuse-ld=gold, or drop the flag entirely.
    # NOTE: debug info (-g, from RelWithDebInfo) is kept on purpose — gdb on the runner
    # is part of the workflow. Jobs capped (default 6) to avoid swap/OOM thrash on this
    # 8-core/15GB box; override with BUILD_JOBS=N.
    # The RelWithDebInfo flags are passed EVERY time, not left to the cache: CMake caches them, so a
    # one-off --no-debug-info would otherwise silently strip every later build of the same tree.
    if [[ "$DEBUG_INFO" == true ]]; then RWDI_FLAGS="-O2 -g -DNDEBUG"; else RWDI_FLAGS="-O2 -DNDEBUG"; fi
    # The C++ runtime (libstdc++, libgcc) is linked in, so the runner starts on a machine without
    # gcc-13's runtime library -- most distributions ship an older one, and a binary built in a
    # container must run on the host. PS2X_STATIC_LIBSTDCXX=0 links them dynamically (as before).
    LINK_FLAGS="-pthread -fuse-ld=lld"
    [[ "${PS2X_STATIC_LIBSTDCXX:-1}" == 0 ]] || LINK_FLAGS+=" -static-libstdc++ -static-libgcc"
    cmake -S "$PS2RECOMP_ROOT" -B "$PS2RECOMP_ROOT/out/build" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="$RWDI_FLAGS" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$RWDI_FLAGS" \
        -DCMAKE_EXE_LINKER_FLAGS="$LINK_FLAGS"
    cmake --build "$PS2RECOMP_ROOT/out/build" \
        --target ps2EntryRunner \
        --config "$BUILD_TYPE" \
        -j "${BUILD_JOBS:-6}"

    # Move (not copy) the runner into the game's tmp/ so each game keeps its own
    # binary — there is no stale shared runner to clash with another game being
    # decompiled. The engine build dir is left with no binary, so the next build
    # always relinks fresh (static recomp = one game per binary anyway).
    BUILT_RUNNER="$PS2RECOMP_ROOT/out/build/ps2xRuntime/ps2EntryRunner"   # the CMake target's own name
    GAME_RUNNER="$GAME_DIR/$RUNNER_NAME"                                  # what the game ships as
    mkdir -p "$GAME_DIR/tmp"
    mv -f "$BUILT_RUNNER" "$GAME_RUNNER"

    echo
    echo "Build complete."
    echo
    echo "Runner:  $GAME_RUNNER"
    echo "Run:     $SCRIPT_DIR/ps2x-run-game.sh $GAME_DIR   (or the game's scripts/run.sh)"
else
    echo
    echo "[3/3] Skipping build (--skip-build)"
fi
