#!/bin/bash
# link_pos MAE rose 9.43 -> 12.46 when the displacement heads were added, so the absolute-position head is
# now competing for capacity with them. Sweep the two weights.
#
# THREE SEEDS per config, reporting median and range. Single-seed differences on this project have been as
# large as shape_yaw 0.7 -> 5.3 deg between runs differing only by a class merge, so a one-run gap is not
# evidence. The earlier capacity sweep was single-seed and its conclusions should be re-read with that in mind.
set -u
D1=$(ls -d datasets/transitions/deku-baba-v2/*/ | tail -1)
D2=$(ls -d datasets/transitions/deku-baba-scenarios/*/ | tail -1)
D3=$(ls -d datasets/transitions/deku-baba-moves/*/ | tail -1)
mkdir -p runs/logs/sweep_pos /tmp/sweep_pos
: > runs/logs/sweep_pos/SUMMARY.txt

run() {  # name w_pos w_dpos
  local n=$1 wp=$2 wd=$3
  local pos=() dr=() ap=() act=() bc=()
  for seed in 0 1 2; do
    .venv/bin/python -u scripts/train_joint.py --data "$D1" --extra-data "$D2" "$D3" \
      --epochs 25 --lr 3e-4 --quiet --seed "$seed" --w-pos "$wp" --w-dpos "$wd" \
      --out "/tmp/sweep_pos/${n}_s${seed}.pt" > "runs/logs/sweep_pos/${n}_s${seed}.log" 2>&1
    local L="runs/logs/sweep_pos/${n}_s${seed}.log"
    pos+=("$(grep -o 'link_pos MAE [0-9.]*' "$L" | awk '{print $3}')")
    dr+=("$(grep -o 'ratio [0-9.]*x' "$L" | head -1 | tr -d 'x' | awk '{print $2}')")
    ap+=("$(grep -o 'MOVED: AP [0-9.]*' "$L" | awk '{print $3}')")
    act+=("$(grep -o 'link_action [0-9.]*' "$L" | awk '{print $2}')")
    bc+=("$(grep -o 'baba_count [0-9.]*' "$L" | awk '{print $2}')")
  done
  med() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{printf "%.4f", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
  rng() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{printf "%.3f-%.3f", a[1], a[NR]}'; }
  printf '%-16s w_pos %-4s w_dpos %-4s | pos_mae %s [%s]  disp_ratio %s [%s]  moved_AP %s  action %s  baba %s\n' \
    "$n" "$wp" "$wd" "$(med "${pos[@]}")" "$(rng "${pos[@]}")" "$(med "${dr[@]}")" "$(rng "${dr[@]}")" \
    "$(med "${ap[@]}")" "$(med "${act[@]}")" "$(med "${bc[@]}")" | tee -a runs/logs/sweep_pos/SUMMARY.txt
}

run base       3.0 2.0
run pos6       6.0 2.0
run pos10      10.0 2.0
run pos6_dpos1 6.0 1.0
echo "=== done ==="; cat runs/logs/sweep_pos/SUMMARY.txt
