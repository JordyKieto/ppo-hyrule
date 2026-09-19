#!/bin/bash
# Re-collect every dataset as state v3, which adds the CAMERA.
#
# Necessary because a stick direction moves Link relative to the camera, so without camera yaw the world
# direction of a walk is unknowable -- and camera yaw was in NO recorded state and was never an encoder
# input. It existed only as a decode target sourced from the stored 483-d obs, which the three scripted
# collections did not save at all (122k of 230k frames had none). The audit traced the dream's inability to
# approach a Baba to exactly this: displacement direction top-1 stalled at 0.532 over 16 sectors.
#
# camera_yaw_sin_cos was in the original PPO's observation, so this restores parity rather than adding to it.
set -eu
export DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin
export PYTHONUNBUFFERED=1
T=$(date +%Y%m%d-%H%M%S)
P=.venv/bin/python-dbg

echo "=== 1/4 main mix: 7 checkpoints x 10 slots ==="
$P -u scripts/collect_transitions.py --transitions 100000 --episodes 400 --workers 10 \
   --out "datasets/transitions/v3-main/$T"

echo "=== 2/4 scenarios: walkout / suicide / wall / flail ==="
$P -u scripts/collect_scenarios.py --scenarios walkout,suicide,wall,flail --episodes 60,60,40,30 \
   --workers 10 --out "datasets/transitions/v3-scenarios/$T"

echo "=== 3/4 moves: roll / backflip / sidehop / jump-attack / shield / swing ==="
$P -u scripts/collect_scenarios.py --scenarios moves --episodes 60 --workers 10 \
   --out "datasets/transitions/v3-moves/$T"

echo "=== 4/4 approach: sustained closing with the shield up ==="
$P -u scripts/collect_scenarios.py --scenarios approach --episodes 70 --workers 10 \
   --out "datasets/transitions/v3-approach/$T"

echo "=== all four collected as v3 (state 363) ==="
