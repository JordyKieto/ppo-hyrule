#!/bin/bash
# gohma-beh, 2026-11-01. The five WATCHED behaviours. Two arms, ONE difference: --reward.
#
# Both arms are v15's own training configuration, continued from v15's weights:
#   init   checkpoints/kept/gohma-kl-v15-r4_latest.pt   md5 397e27b2beb688b9b14d6a260ec07a22  (update 40)
#   anchor checkpoints/inbox/v13_r12_latest.pt          md5 49cf19963f4f...                   (v15's own anchor)
#   obs v7 | 300 s cap | slots 3,4,5,6,7,8 | weights 4=1.125,6=1.125,7=2.25 | kl-coef 0.3 | death-scale 0
#
# --fleet-sync none: EVERY POD IS AN INDEPENDENT SEED. Under FedAvg an arm of N pods is one policy averaged
# over N ranks, i.e. n=1, which cannot address a seed floor at all. The design's whole premise is replicates
# (five identical controls spanned 2.00-2.97, sd 0.380), and replicates require independence. It also matches
# the historical one-pod-per-arm controls, so slopes per update are comparable.
#
# CONTROL ranks 1,3,22 -- chosen on THREE DISTINCT public IPs, each the only pod on its host, all $0.56/16
# vCPU. With n=3 a single contended host could dominate the control, and the 7-pod host 157.157.221.30 is the
# obvious hazard. Giving the control the cleanest pods biases AGAINST the treatment, which is the safe
# direction.
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMON="--profile gohma --obs-version 7 --episode-seconds 300 \
--start-slots 3,4,5,6,7,8 --slot-weights 4=1.125,6=1.125,7=2.25 \
--init-from checkpoints/kept/gohma-kl-v15-r4_latest.pt \
--kl-anchor checkpoints/inbox/v13_r12_latest.pt --kl-coef 0.3 \
--death-scale 0 --fleet-sync none --total-updates 60"

CONTROL_RANKS=1,3,22
TREAT_RANKS=0,2,4,5,6,8,9,10,11,12,13,14,15,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35

# the init and the anchor, by presigned S3 GET, verified by stat-ing the path the trainer opens (AGENTS 16/20)
.venv/bin/python -u -m fleet ckpt --experiment gohma-beh \
  --path checkpoints/kept/gohma-kl-v15-r4_latest.pt,checkpoints/inbox/v13_r12_latest.pt

# CONTROL: v15's reward, unmodified
.venv/bin/python -u -m fleet launch --experiment gohma-beh --ranks "$CONTROL_RANKS" \
  --script train_combat.py --run-tag ctl --num-envs 8 \
  --wandb-project zelda-rl-gohma-ppo --train-args "$COMMON --reward simple"

# TREATMENT: the five watched behaviours
.venv/bin/python -u -m fleet launch --experiment gohma-beh --ranks "$TREAT_RANKS" \
  --script train_combat.py --run-tag beh --num-envs 8 \
  --wandb-project zelda-rl-gohma-ppo --train-args "$COMMON --reward behaviour"
