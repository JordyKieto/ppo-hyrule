#!/bin/bash
# Why did action recoverability fall from 0.139 (joint_v3, v1 state) to 0.069 (joint_v6, v2 state)?
#
# Hypothesis: a fixed 128-d latent now has to encode 122 input features including the whole Link
# body/collision/action block, so it spends capacity on decode at the expense of the transition. Evidence
# that prompted it: every decode head improved over the same change while the latent spread widened
# (top PC 47.2% vs the BC encoder's 61.5%).
#
# Three levers, one at a time:
#   --latent   more capacity
#   --w-pred   weight the predictive term harder
#   --w-v2     weight the v2 decode heads LESS (tests the hypothesis most directly)
#
# The inverse-dynamics number train_joint prints is POOLED over data sources, so it is not comparable to
# joint_v3's 0.139 -- but it IS comparable BETWEEN these runs, since they share data and the slot split.
# Per-source numbers for the winner come from scripts/probe_action_effect.py afterwards.
set -u
D1=$(ls -d datasets/transitions/deku-baba-v2/*/ | tail -1)
D2=$(ls -d datasets/transitions/deku-baba-scenarios/*/ | tail -1)
mkdir -p runs/logs/sweep /tmp/sweep

run() {  # name latent w_pred w_v2
  local n=$1 lat=$2 wp=$3 wv=$4
  .venv/bin/python -u scripts/train_joint.py --data "$D1" --extra-data "$D2" \
      --epochs 25 --lr 3e-4 --quiet --latent "$lat" --w-pred "$wp" --w-v2 "$wv" \
      --out "/tmp/sweep/$n.pt" > "runs/logs/sweep/$n.log" 2>&1
  local skill inv act pos
  skill=$(grep -o 'best tune skill [-0-9.]*' "runs/logs/sweep/$n.log" | awk '{print $4}')
  inv=$(grep -o 'top-1 [0-9.]*' "runs/logs/sweep/$n.log" | awk '{print $2}')
  act=$(grep -o 'link_action [0-9.]*' "runs/logs/sweep/$n.log" | awk '{print $2}')
  pos=$(grep -o 'link_pos MAE [0-9.]*' "runs/logs/sweep/$n.log" | awk '{print $3}')
  printf '%-22s lat %-4s w_pred %-4s w_v2 %-4s | skill %-8s inv_dyn %-8s link_action %-7s pos_mae %s\n' \
      "$n" "$lat" "$wp" "$wv" "${skill:-FAIL}" "${inv:-?}" "${act:-?}" "${pos:-?}" | tee -a runs/logs/sweep/SUMMARY.txt
}

: > runs/logs/sweep/SUMMARY.txt
run base_128          128 1.0 1.0
run lat192            192 1.0 1.0
run lat256            256 1.0 1.0
run lat384            384 1.0 1.0
run wpred3_128        128 3.0 1.0
run wv2_0.3_128       128 1.0 0.3
run lat256_wpred3     256 3.0 1.0
echo "=== done ==="; cat runs/logs/sweep/SUMMARY.txt
