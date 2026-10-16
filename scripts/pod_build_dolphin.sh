#!/usr/bin/env bash
# Build Dolphin 2603a for Linux (x86_64 or arm64) from source: DolphinQt (`dolphin-emu`) and
# `dolphin-emu-nogui`. Runs ON the pod (scripts/runpod.sh --build-dolphin) and inside
# docker/Dockerfile, so both paths produce the same binaries.
#
# Why 2603a exactly: the user's savestates (NACE01.s0*) carry state version 177, and State.cpp only
# loads a state whose version matches STATE_VERSION. The revision string does not have to match.
# 2603a has STATE_VERSION = 177 (Source/Core/Core/State.cpp); this script refuses to build anything else.
#
# Why DolphinQt and not only nogui: checked against the 2603a source, `dolphin-emu-nogui` has no
# hotkey handling at all. HotkeyManagerEmu is polled by DolphinQt/HotkeyScheduler.cpp; the nogui
# headless platform's MainLoop only dispatches host jobs, and the X11 platform only maps raw
# F1-F8 key presses. So the `Pipes/hk` savestate-load hotkey that env resets rely on only works
# in DolphinQt. nogui also rejects `-b` (a GUI-only option in CommandLineParse.cpp). DolphinQt runs
# fine without a display using QT_QPA_PLATFORM=offscreen (Main.cpp only overrides QT_QPA_PLATFORM
# when it is unset or wayland) or under xvfb-run. That exact combination is still UNVERIFIED on
# a pod; see docs/RUNPOD.md.
#
# Usage:
#   scripts/pod_build_dolphin.sh                 # install build deps, clone, build, install
#   scripts/pod_build_dolphin.sh --deps-only     # just apt build+runtime deps (re-run after a pod restart)
#   scripts/pod_build_dolphin.sh --check         # verify an existing install (ldd + version)
# Env: DOLPHIN_TAG (2603a), DOLPHIN_ROOT (/workspace/dolphin), JOBS (nproc), SKIP_APT=1
set -euo pipefail

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

DOLPHIN_TAG="${DOLPHIN_TAG:-2603a}"
EXPECTED_STATE_VERSION="${EXPECTED_STATE_VERSION:-177}"
DOLPHIN_ROOT="${DOLPHIN_ROOT:-/workspace/dolphin}"
SRC="$DOLPHIN_ROOT/src"
BUILD="$DOLPHIN_ROOT/build"
PREFIX="$DOLPHIN_ROOT/install"
JOBS="${JOBS:-$(nproc)}"

MODE=build
for arg in "$@"; do
  case "$arg" in
    --deps-only) MODE=deps ;;
    --check) MODE=check ;;
    *) log "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

SUDO=""
[[ "$(id -u)" != "0" ]] && SUDO="sudo"

install_deps() {
  [[ "${SKIP_APT:-0}" == "1" ]] && return 0
  export DEBIAN_FRONTEND=noninteractive
  log "apt: build + runtime deps (non-vendored only; everything vendored is built from Externals/)"
  $SUDO apt-get update
  # Build tooling + the deps Dolphin does NOT vendor on Linux (wiki "Building for Linux", 2026-09):
  # Qt6 (DolphinQt), OpenGL/EGL + X11 headers, udev/evdev, systemd, ALSA/Pulse. Runtime-only extras:
  # qt6-qpa-plugins (offscreen platform plugin), xvfb/xauth (fallback display), procps/psmisc.
  # One missing package used to abort the WHOLE install, and with it the rest of the bootstrap: the GPU
  # image is Ubuntu 22.04, where the Qt SVG dev package is libqt6svg6-dev, not qt6-svg-dev (24.04's name).
  # The H100 pod ended up with no Dolphin and no venv because of that single name (2026-10-16). So: try the
  # full set, and if apt refuses, install package-by-package and report what is missing instead of dying.
  local PKGS="build-essential git cmake ninja-build pkg-config ca-certificates gettext \
    qt6-base-dev qt6-base-private-dev qt6-qpa-plugins libqt6svg6 \
    libgl1-mesa-dev libegl1-mesa-dev libx11-dev libxrandr-dev libxi-dev \
    libudev-dev libevdev-dev libsystemd-dev libasound2-dev libpulse-dev \
    xvfb xauth procps psmisc"
  # the SVG dev package, whichever name this release uses
  local SVG=""
  for cand in qt6-svg-dev libqt6svg6-dev; do
    if apt-cache show "$cand" >/dev/null 2>&1; then SVG="$cand"; break; fi
  done
  [[ -n "$SVG" ]] && log "Qt SVG dev package on this release: $SVG" || log "no Qt SVG dev package found"
  if ! $SUDO apt-get install -y --no-install-recommends $PKGS $SVG; then
    log "bulk install failed; falling back to one-by-one so a single missing name cannot abort the bootstrap"
    local missing=""
    for p in $PKGS $SVG; do
      $SUDO apt-get install -y --no-install-recommends "$p" >/dev/null 2>&1 || missing="$missing $p"
    done
    [[ -n "$missing" ]] && log "NOT INSTALLED (may be fine for a prebuilt Dolphin):$missing"
  fi
  # 2603a's CMakeLists.txt requires GCC >= 12 and C++23. Ubuntu 24.04 ships 13; 22.04 ships 11
  # but has g++-12 in its archive.
  local gccv
  gccv="$(gcc -dumpversion | cut -d. -f1)"
  if [[ "$gccv" -lt 12 ]]; then
    log "gcc $gccv < 12, installing g++-12"
    $SUDO apt-get install -y --no-install-recommends gcc-12 g++-12
  fi
}

pick_compiler() {
  if [[ "$(gcc -dumpversion | cut -d. -f1)" -lt 12 ]]; then
    export CC=gcc-12 CXX=g++-12
  fi
}

check_install() {
  local ok=1
  for bin in dolphin-emu dolphin-emu-nogui; do
    local p="$PREFIX/bin/$bin"
    if [[ ! -x "$p" ]]; then
      log "MISSING $p"; ok=0; continue
    fi
    if ldd "$p" | grep -q "not found"; then
      log "$p has unresolved libraries (run --deps-only after a pod restart):"
      ldd "$p" | grep "not found" || true
      ok=0
    fi
    log "$bin --version: $("$p" --version 2>/dev/null | head -1 || echo '?')"
  done
  [[ "$ok" == "1" ]] || { log "Dolphin install check FAILED"; return 1; }
  log "Dolphin install OK: DOLPHIN_BIN=$PREFIX/bin/dolphin-emu"
}

case "$MODE" in
  deps) install_deps; exit 0 ;;
  check) check_install; exit $? ;;
esac

install_deps
pick_compiler
mkdir -p "$DOLPHIN_ROOT"

if [[ ! -d "$SRC/.git" ]]; then
  log "cloning dolphin $DOLPHIN_TAG"
  git clone --depth 1 --branch "$DOLPHIN_TAG" https://github.com/dolphin-emu/dolphin.git "$SRC"
fi
cd "$SRC"
HAVE_TAG="$(git describe --tags --exact-match 2>/dev/null || true)"
[[ "$HAVE_TAG" == "$DOLPHIN_TAG" ]] || { log "source at '$HAVE_TAG', expected $DOLPHIN_TAG (rm -rf $SRC to reclone)"; exit 1; }

# Hard gate: savestate compatibility is decided by this one constant.
SV="$(grep -oE 'constexpr u32 STATE_VERSION = [0-9]+' Source/Core/Core/State.cpp | grep -oE '[0-9]+$')"
[[ "$SV" == "$EXPECTED_STATE_VERSION" ]] || { log "STATE_VERSION=$SV != $EXPECTED_STATE_VERSION; savestates would not load"; exit 1; }
log "STATE_VERSION=$SV matches the user's savestates"

# Externals/Qt and Externals/FFmpeg-bin hold prebuilt Windows/macOS binaries and are unused on Linux.
log "fetching submodules (shallow, skipping Windows-only binaries)"
git -c submodule."Externals/Qt".update=none -c submodule."Externals/FFmpeg-bin".update=none \
  submodule update --init --recursive --depth 1 --jobs 8 \
  || git -c submodule."Externals/Qt".update=none -c submodule."Externals/FFmpeg-bin".update=none \
       submodule update --init --recursive --jobs 8

# USE_SYSTEM_LIBS=OFF: always use the vendored copies, so the build does not depend on which
# -dev packages the base image already has (e.g. an old zlib or fmt). Tests, LLVM disassembly,
# Discord, updater, analytics and FFmpeg frame dumps are off; none of them touches emulation
# or savestate content.
log "configuring (jobs=$JOBS, CC=${CC:-gcc})"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DUSE_SYSTEM_LIBS=OFF \
  -DENABLE_QT=ON -DENABLE_NOGUI=ON \
  -DENABLE_TESTS=OFF -DENABLE_LLVM=OFF -DUSE_DISCORD_PRESENCE=OFF \
  -DENABLE_AUTOUPDATE=OFF -DENABLE_ANALYTICS=OFF -DENCODE_FRAMEDUMPS=OFF

log "building (this is the slow part: ~10 min on 32 vCPU, much longer on small pods)"
cmake --build "$BUILD" --parallel "$JOBS"
cmake --install "$BUILD"
check_install
