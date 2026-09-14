#!/usr/bin/env bash
# Record a checkpoint playing the Deku Baba start slots: fullscreen Dolphin + game audio -> .mp4.
# Recording starts when Dolphin's window appears and stops by itself when the last slot ends.
#
#   scripts/record_play.sh                                   # all 10 slots, final policy
#   scripts/record_play.sh recordings/run.mp4 --deterministic
#   SLOTS="4 7" CKPT=checkpoints/kept/other.pt scripts/record_play.sh
#
# Extra arguments after the output path go to scripts/play.py. Ctrl+C stops early and still saves the video.
# First run: macOS asks for Screen Recording permission for your terminal app; allow it, restart the terminal, rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt}
SLOTS=${SLOTS:-"1 2 3 4 5 6 7 8 9 10"}
OUT=${1:-recordings/deku-baba_$(date +%Y%m%d-%H%M%S).mp4}
shift $(( $# > 0 ? 1 : 0 ))
REC="$HOME/.cache/zelda-rl/screenrec"
export DOLPHIN_BIN=${DOLPHIN_BIN:-$HOME/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin}

mkdir -p "$(dirname "$OUT")"
if [[ ! -x "$REC" || tools/screenrec.swift -nt "$REC" ]]; then
  echo "building screen recorder..."
  swiftc -O tools/screenrec.swift -o "$REC"
fi
slot_args=()
for s in $SLOTS; do slot_args+=(--slot "$s"); done

before=" $( (pgrep -f 'MacOS/Dolphin -u' || true) | tr '\n' ' ') "
.venv/bin/python-dbg -u scripts/play.py "$CKPT" "${slot_args[@]}" -n 1 --fullscreen --end-hold 3 --hud-hz 0.5 "$@" \
  > >(tee "${OUT%.*}.log") 2>&1 &
PLAY=$!

pid=""
for _ in $(seq 1 240); do                     # the Dolphin process play.py launches (not one already running)
  for p in $(pgrep -f 'MacOS/Dolphin -u' || true); do
    [[ "$before" == *" $p "* ]] || pid=$p
  done
  [[ -n "$pid" ]] && break
  kill -0 "$PLAY" 2>/dev/null || { echo "play.py exited before Dolphin started"; exit 1; }
  sleep 0.5
done
[[ -n "$pid" ]] || { echo "Dolphin did not start"; kill -INT "$PLAY"; exit 1; }

"$REC" --pid "$pid" --out "$OUT" &
RECORDER=$!
trap 'kill -INT "$PLAY" "$RECORDER" 2>/dev/null || true' INT TERM

wait "$PLAY" || true
wait "$RECORDER" || true
if [[ -s "$OUT" ]]; then
  echo "saved $OUT ($(du -h "$OUT" | cut -f1)) and ${OUT%.*}.log"
else
  echo "no video written: check Screen Recording permission for this terminal"; exit 1
fi
