#!/usr/bin/env bash
set -euo pipefail
# Build this toolchain's recompiler (ps2_recomp + ps2_analyzer) -- the step before
# ps2x-build-game.sh. Run from anywhere; it works on the clone it lives in.
#
# Usage: ps2x-setup.sh
#
# Lives in the fork (scripts/) since 2026-10-02: the configure + build half of LLMPS2Recomp's
# 01_setup.sh, which still does the CLONING for the engine's layout. A game repo's scripts/build.sh
# clones this fork at the commit its recomp/runtime.lock names, then calls this.
#
# Needs: git, cmake >= 3.21, gcc-13/g++-13 (the recompiler and runtime are C++20 built with
# SSE4.1). Installing those is the game's scripts/build.sh's job; this only checks.

PS2RECOMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== PS2Recomp toolchain setup ==="
echo "    clone: $PS2RECOMP_DIR"

echo "[1/3] Checking prerequisites..."
for cmd in git cmake gcc-13 g++-13; do
    command -v "$cmd" >/dev/null || { echo "ERROR: missing required command: $cmd"; exit 1; }
done
CMAKE_VER="$(cmake --version | sed -n 's/^cmake version \([0-9.]*\).*/\1/p')"
if [[ "$(printf '%s\n3.21\n' "$CMAKE_VER" | sort -V | head -n1)" != "3.21" ]]; then
    echo "ERROR: cmake $CMAKE_VER is too old; 3.21 or newer is required."; exit 1
fi

echo "[2/3] Configuring..."
cmake -B "$PS2RECOMP_DIR/out/build" -S "$PS2RECOMP_DIR" \
  -DCMAKE_C_COMPILER="$(command -v gcc-13)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-13)" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread" \
  -DCMAKE_CXX_FLAGS="-msse4.1"

echo "[3/3] Building ps2_recomp + ps2_analyzer..."
cmake --build "$PS2RECOMP_DIR/out/build" --target ps2_recomp -j"$(nproc)"
cmake --build "$PS2RECOMP_DIR/out/build" --target ps2_analyzer -j"$(nproc)"

echo
echo "Setup complete."
echo "  ps2_recomp: $PS2RECOMP_DIR/out/build/ps2xRecomp/ps2_recomp"
echo "  Next:       $PS2RECOMP_DIR/scripts/ps2x-build-game.sh <game_dir>"
