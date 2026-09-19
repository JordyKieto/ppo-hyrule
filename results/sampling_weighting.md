# Sampling and weighting for the transition objective: a null result

40 training runs, `scripts/train_joint.py` on the v3 four-dir dataset, each scored with
`scripts/eval_event_windows.py --horizon 40 --per-kind 20 --seeds 1`. Fixed throughout:
`--epochs 4 --lr 3e-4 --w-pos 6 --unroll 8 --unroll-detach 8 --unroll-batch 96`.
Baseline is `--oversample 50 --w-dpos 2 --w-trans 2`.

**Headline: none of the three levers moves any barometer metric beyond the baseline's own
seed-to-seed spread.** The baseline re-run at six seeds spans 0.25-0.55 damage detection and
1.38x-3.43x walk-trajectory ratio. Every alternative setting's mean sits inside that band.

## Results

Median [min-max] over training seeds. Eval windows are identical for every model (the eval's
window sampler is seeded independently of the training seed), so these differences are pure
training variance, not window-sampling noise. Trajectory ratio = `traj_err / traj_err_identity`;
below 1.00 would beat "Link frozen".

| config | n seeds | detect_baba (hit) | HP removed (real 1.25) | traj ratio, hit | traj ratio, walk |
|---|---|---|---|---|---|
| **baseline** OS50 WD2 WT2 | 6 | **0.50 [0.25-0.55]** | **0.94 [0.50-1.15]** | **1.51 [1.02-3.79]** | **2.17 [1.38-3.43]** |
| oversample 10 | 3 | 0.50 [0.25-0.75] | 0.95 [0.75-1.50] | 1.48 [1.22-1.71] | 2.31 [1.80-2.55] |
| oversample 150 | 6 | 0.29 [0.12-0.75] | 0.59 [0.25-1.62] | 1.18 [1.00-2.09] | 1.48 [1.23-2.40] |
| w_dpos 0.5 | 3 | 0.00 [0.00-0.75] | 0.00 [0.00-1.50] | 1.41 [1.02-1.99] | 1.95 [1.94-3.29] |
| w_dpos 4 | 5 | 0.38 [0.12-0.75] | 0.75 [0.25-2.62] | 2.00 [1.30-3.73] | 1.76 [1.30-4.15] |
| w_dpos 8 | 5 | 0.50 [0.12-0.80] | 1.12 [0.12-2.70] | 1.77 [1.21-2.78] | 2.11 [1.38-3.77] |
| w_trans 0.5 | 3 | 0.50 [0.12-0.62] | 1.00 [0.25-1.38] | 1.17 [1.06-2.48] | 1.94 [1.64-3.66] |
| w_trans 8 | 3 | 0.25 [0.12-0.45] | 0.50 [0.25-1.30] | 1.25 [1.22-1.53] | 1.53 [1.35-2.19] |
| OS150 + WD8 (combo) | 6 | 0.38 [0.00-0.62] | 0.50 [0.00-2.12] | 1.58 [1.15-3.20] | 1.57 [1.24-4.23] |

Exact two-sided rank test of each config's seeds against the baseline's six, on the four headline
metrics (32 tests): the smallest p is 0.036 (w_trans 8, detection, n=3) — exactly the one
marginal result you expect from 32 tests by chance, and it does not survive any correction.
Everything else is p > 0.09.

## Baseline seed spread — the number that kills the sweep

Baseline, six seeds, one config:

| metric | mean | sd | min | max | spread |
|---|---|---|---|---|---|
| detect_baba (hit) | 0.467 | 0.108 | 0.25 | 0.55 | 0.30 |
| HP removed (hit) | 0.900 | 0.221 | 0.50 | 1.15 | 0.65 |
| traj ratio (walk) | 2.200 | 0.691 | 1.38 | 3.43 | 2.05 |

The previously reported baseline (55% detection, 1.15 HP, walk 1.38x) is **seed 0, and it is the
best of the six on all three metrics simultaneously**. The typical `--w-pos 6` model is 50%
detection, 0.9 HP, walk ~2.2x, not 1.38x. Anything built on the 1.38x figure should be re-based.

## What a single seed would have told you

Every lever "worked" at seed 0 and none of it replicated:

| seed-0 conclusion | at n=5-6 |
|---|---|
| w_dpos 8 is a big win: detect 0.80 vs 0.55 | 0.46 mean vs 0.47, p=0.97 |
| oversample 10 wrecks trajectory: walk 2.55 vs 1.38 | 2.22 mean vs 2.20, p=0.55 |
| w_dpos 0.5 kills events dead: detect 0.00 | 0.00, 0.00, 0.75 across three seeds |
| oversample 150 kills detection: 0.20 vs 0.55 | 0.35 mean vs 0.47, p=0.22 |
| OS150+WD8 keeps detection and fixes walking (n=3: 0.46 detect, walk 1.47) | at n=6: 0.35 detect, walk 1.96, p=0.24 |

The combo row is the sharpest warning: it looked like the one genuine improvement at three seeds
and was gone at six.

## The one thing worth acting on

Three of the 40 runs produced *exactly* 0.00 detection — a dead event head, not a weak one — and
two of the three were `--w-dpos 0.5` (the third was the OS150+WD8 combo). The displacement heads
appear to be what keeps the event heads alive, so **do not turn `--w-dpos` down**. This is a
warning about a failure mode, not a measured effect: 2 of 3 seeds is not an established rate.

## Conclusion

Keep the baseline: `--oversample 50 --w-dpos 2 --w-trans 2`. None of oversample 10/150, w_dpos
0.5/4/8, or w_trans 0.5/8 beats it on damage detection, HP magnitude, or trajectory by more than
the 0.30-detection / 2.05x-walk spread that the baseline shows against itself across seeds, and
the one combination worth trying (OS150+WD8) lost its apparent edge when taken from three seeds to
six. The cost of keeping the baseline is therefore zero — there is nothing here to trade away.
The real finding is methodological: **on this harness a single-seed A/B cannot resolve anything
smaller than a 0.30 swing in detection or a 2x swing in walk ratio, so single-seed comparisons of
these weights should not be run at all** — they will keep generating winners that evaporate. If
sampling and weighting are to be pushed further, the prerequisite is variance reduction (more
epochs, seed averaging inside the objective, or an eval with more than 20 windows per kind), not
more grid cells. The levers that have actually moved this barometer — turning oversampling on at
all (0 -> 50) and `--w-pos` — both produced effects far larger than this spread, which is why they
were visible at one seed and these are not.
