# Human demonstrations as a PPO regulariser — the arm table

**Read `PROVENANCE.md` §3.3 for the corpus and the measurements.** This file is the live register: one row per
arm, its exact flags, the question it answers, and a result slot filled **when the number lands**, not when
the arm is launched. An empty result column is a running experiment; a `U` is a promise (AGENTS 27).

## The baselines every arm is read against

| reference | boss damage | how it was measured | class |
|---|---|---|---|
| **v15** (`gohma-kl-v15-r4_latest.pt`) | **3.275** | best Gohma policy to date, 2026-10-05 | M |
| v10 from scratch, early | 0.38 | first updates of a random init | M |
| **pure BC clone** | **0.14** | 80 held-out episodes, 8 per start on all 10 starts, 0 defeats, 83.8% deaths | M |
| v20 arms (observation sweep) | ~2.3-2.5 | the tie this fleet was running before | M |

Boss damage out of 10 is the headline: it is what the reward pays for and the only signal with the resolution
to show a trend on a few hundred episodes (AGENTS 34). Success rate is a 1-10% binary and is reported beside
it, never instead of it.

## What the corpus is

13 episodes, 13 wins, 10,532 steps, 8 distinct start states, obs v7 (1033), 108 actions. 93/108 actions used.
In S3 at `s3://zelda-rl/zelda/human/gohma/`, md5s in `datasets/human/gohma/MANIFEST.json`, gated by
`scripts/verify_human_dataset.py`.

Two properties of it that shape every arm below:

- **The clone learns the actions and cannot fight.** 53.4% held-out action accuracy against an 11.1% majority
  floor, and 0 wins in 80 episodes taking 0.14 of her 10 HP. Frame-level imitation accuracy does not predict
  closed-loop competence (claim R46).
- **v15 and the human share almost nothing.** 3% action agreement at update 1, and a GRU-state MSE of **0.94**
  with mean |h| 0.56 -- near-maximal. Two policies that both beat Gohma encode the fight in unrelated ways.
- **80-94% of the human's frames do not lead to damage.** The outcome filter keeps 6.0% of frames at a 1 s
  lookahead, 12.0% at 3 s, 21.3% at 6 s.

## The arms

Common to all: `--init-from checkpoints/kept/gohma-kl-v15-r4_latest.pt`, `--profile gohma --obs-version 7
--reward simple --episode-seconds 120 --start-slots 3,4,5,6,7,8 --slot-weights 4=1.125,6=1.125,7=2.25
--gamma 0.995 --gae-lambda 0.95 --lr 3e-4 --vf-coef 0.5 --hidden 256 --gru 256 --death-scale 0`, 8 envs per
rank, `--bc-data datasets/human/gohma`. Only the BC flags differ, so a difference between arms is the BC
mechanism and nothing else.

| # | arm | BC flags | ranks | experiment / launch | question | result | class |
|---|---|---|---|---|---|---|---|
| 1 | **BC init** | `--bc-coef 0.1`, init from the clone | 12 | `gohma-v20` / `20261015173419` | is the clone a better start than v15? | **NO.** 0.21 -> 0.35 over 6 updates; ~100 updates to reach where v15 already is. Stopped. | **M** |
| 2 | **constant 0.1** | `--bc-coef 0.1` | 12 + 4 | `gohma-v20` / `20261015175011` | does a fixed pull help v15? | **NO.** damage 2.63 -> 2.03, slope **-0.0857 +/- 0.0172, t=-4.98**; replicate -0.134 +/- 0.089 agrees in direction. `bcA` 3% -> 23%, `bcL` 4.43 -> 3.00, so the term worked and cost damage | **M** |
| 3 | **decay + tail** | `--bc-coef 0.1 --bc-coef-final 0.02 --bc-coef-updates 50` | 12 + 4 | `gohma-v20` / `20261015180929`, `gohma-v15bc` / `20261015181013` | does anchoring early and releasing avoid the cost? | **NO.** 26 updates, **0 of 26 above v15's 3.275** (best 2.74 at u1, i.e. closest to v15's own weights). Matched u1-8 vs arm 2: difference +0.0256 +/- 0.0396, **t=+0.65, not distinguishable**. Its own second half, where the coef is lowest, DECLINES: -0.0239 +/- 0.0117, t=-2.05; the 4-rank replicate declines over all 29 updates, -0.0235 +/- 0.0070, **t=-3.37** | **M** |
| 4 | **weak constant** | `--bc-coef 0.02` | 4 | `gohma-bcw` | is arm 3's floor itself safe? | | U |
| 5 | **adaptive** | `--bc-adaptive-frac 1.0` | 4 | `gohma-bc3` ranks 0-3 | was DOMINATION the problem rather than imitation? the term is held at 1x \|policy loss\| instead of ~5-40x | | U |
| 6 | **Q-filter** | `--bc-coef 0.1 --bc-qfilter 30` | 4 | `gohma-bc3` ranks 4-7 | is it better to imitate only the 12% of frames that damage her within 3 s? | | U |
| 7 | **latent alignment** | `--bc-coef 0 --bc-latent-coef 0.05 --bc-teacher checkpoints/kept/bc_human.pt` | 4 | `gohma-bc3` ranks 8-11 | can the clone's REPRESENTATION transfer without its actions? | | U |

## RETRACTION: the decay-vs-constant comparison was first read on an unmatched window

`python -m fleet ppo` prints only the **last 8** fleet-mean rows unless `--last N` is passed. The first
comparison of the decay arm against the constant one therefore took the decay arm's updates **18-25** and the
constant arm's updates **1-8** and reported the difference as `+0.051 +/- 0.027, t=+1.92`, described to the
author as "just under the bar". Recomputed on the matched window 1-8 it is `+0.0256 +/- 0.0396`, **t=+0.65**.

The claim was wrong in the direction of the hypothesis being tested, which is the dangerous direction, and the
instrument hid it: both windows print identically and nothing in the output says which updates are shown.
Every series here is now pulled with `--last 60` and the update range is printed beside the slope (AGENTS 12).

## How to fill a result row

1. `python -m fleet ppo --experiment <E>` for the fleet mean per update. Read `dmgB` first.
2. Slope **with its standard error** over all updates, never a last-k mean (AGENTS 35). `|t| > 2` or do not
   write "improving".
3. Compare arms **at matched update counts** (AGENTS 12), and against v15's 3.275.
4. The eval pod (`gohma-v20` rank 0) runs `--watch-published gohma-v20` and scores every published checkpoint
   over the same 80-episode suite that produced the 0.14 BC baseline -- that is the held-out number, W&B group
   `gohma-v20-v15bcd-eval`, rows appended to `runs/eval_watch.jsonl` on the pod.
5. Copy the finished row into `PROVENANCE.md` §3.3 and set its class; open or answer the matching claim in
   `paper/CLAIMS.md`.

## The mechanisms, and the control that proves each one is not a no-op

`tests/test_bc_regulariser.py` — every arm's mechanism has a positive and a negative case, because a term that
is plumbed through, logged and multiplying nothing looks exactly like a term that works.

| mechanism | positive control | negative control |
|---|---|---|
| constant coefficient | coef 1.0 drives agreement with the demonstrated action to **96.9%** from an 11.1% chance floor | coef 0.0 reports no statistics and reaches **0%** |
| decay floor | exact at the boundary, still held at update 10,000, monotone | asserted **never 0** -- a coefficient reaching 0 leaves `--init-from`-only behaviour |
| Q-filter | kept frames demand action A: learned **100%** | dropped frames demand a decoy: learned **0%** |
| adaptive weight | at a strong fraction the term trains to **94%** | **caught a real bug**: the PPO gate omitted `bc_adaptive_frac`, so with `--bc-coef 0` the whole term was a silent no-op |
| latent alignment | hidden-state gap **0.028** with alignment | **0.172** without; coef 0 reports nothing |

## Deviations from the papers, stated here rather than buried in a commit (AGENTS 13)

- **Q-filter.** Nair et al. 2018 keep a demonstration frame when `Q(s, a_demo) > Q(s, a_pi)`. This project's
  critic is a **V** head, so there is no Q to compare. `--bc-qfilter STEPS` substitutes the outcome the human
  actually achieved in the units the reward pays for: a frame is kept when the boss loses HP within STEPS. The
  boss-HP column is derived from each recording's own `obs_fields` (offset 100, verified monotone with 10
  levels), never hand-counted (AGENTS 3).
- **Latent alignment** is not from a specific paper; it is representation distillation from the frozen clone
  onto the GRU state, chosen because the action-level disagreement (3%) and the latent-level disagreement
  (0.94 MSE) say the two policies differ in representation, not only in output.
- **Adaptive weight** is `coef = frac * |policy loss| / bc_loss`, clipped to [1e-5, 1.0], recomputed per
  minibatch. With `|pg|` 0.01-0.10 and `bc_loss` ~3, `frac 1.0` means a coefficient of 0.003-0.03 -- an order
  of magnitude below the constant 0.1 that caused arm 2's decline.
