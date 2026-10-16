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

## The inverse: the clone as the student, v15 as the teacher

Arms 1-7 all ask "do the human's demonstrations help v15". These ask the opposite question: **is v15's
REPRESENTATION the transferable part, and can it be poured into the clone?** The student starts at
`bc_human.pt` and is pulled toward v15's GRU state on the human's own frames.

Calibration, so "strong" means something: the measured GRU-state gap between v15 and the clone on the corpus
is **MSE 0.94 with mean |h| 0.56** -- near-maximal. A latent coefficient of 1.0 therefore adds ~0.94 to a loss
whose policy term runs 0.01-0.10, i.e. it dominates by 10-90x; 5.0 adds ~4.7.

ONE arm runs this inverse direction, on the author's instruction; every other arm initialises from the PPO
policy (v15), because arm 1 measured the clone to be a worse starting point than v15 by a factor of 15.

| # | arm | flags beyond the common set | ranks | question | result | class |
|---|---|---|---|---|---|---|
| 8 | **inverse: clone student, v15 teacher** | `--init-from checkpoints/kept/bc_human.pt --bc-teacher checkpoints/kept/gohma-kl-v15-r4_latest.pt --bc-latent-coef 1.0 --bc-coef 0` | 4 (`gohma-bc3` r0-3) | does v15's REPRESENTATION lift the clone? | | U |
| 6 | **Q-filter** (v15 init) | `--bc-coef 0.1 --bc-qfilter 30` | 4 (`gohma-bc3` r4-7) | is it better to imitate only the 12% of frames that damage her within 3 s? | | U |
| 7 | **latent into v15, strong** | `--bc-teacher checkpoints/kept/bc_human.pt --bc-latent-coef 1.0 --bc-coef 0` | 4 (`gohma-bc3` r8-11) | can the clone's representation be shared with v15 WITHOUT its actions? | | U |
| 4 | **weak constant** (v15 init) | `--bc-coef 0.02` | 4 (`gohma-bcw`) | is the decay's floor itself safe? | | U |

Coefficient 1.0 rather than 5.0 for the latent arms: at the measured 0.94 gap it already dominates the policy
term by 10-90x, so it IS the strong regime, and a larger value makes a pathological outcome harder to read.
The adaptive arm (5) is parked with the spare capacity.

**What would make arm 8 interesting, stated before it runs (AGENTS 1, 9).** It starts from a policy that takes
0.14 of her HP, so beating v15's 3.275 is not the near-term test. The test is the CLIMB: the clone under plain
PPO rose +0.028/update (0.21 -> 0.35 over 6 updates) and from-scratch v10 started at 0.38. Arm 8 is
interesting if its slope is several times +0.028 with |t| > 2 -- that would say v15's representation is what
transfers. Tracking +0.028 means representation transfer buys nothing either.

**For the v15-init arms (4, 6, 7) the bar is unchanged: 3.275, and beating the constant-0.1 decline.**

## Arms 11-15: the paper-driven designs, and the fleet-wide slope table

All five replace the naive loss-mixing arms. Common: v15 init, the usual base flags, 4 ranks each unless
stated. Slopes are over ALL updates with the window printed, never a last-k mean (AGENTS 35).

| # | arm | mechanism | n | latest | mean | max | slope +/- SE | t | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11 | **FitNets projection** | learned linear map of the student's latent before the MSE, rollout states | 14 | 3.03 | **2.87** | 3.37 | -0.0064 +/- 0.0192 | -0.33 | flat, the only arms holding at v15's level |
| 12 | **latent transition** | MSE on `h_t - h_(t-1)`: match how beliefs UPDATE | 14 | 3.02 | **2.79** | 3.21 | +0.0100 +/- 0.0159 | +0.62 | flat, holding |
| 13 | **mutual alignment** | both nets trainable, contrastive shared space | 22 | 2.23 | 2.66 | 3.28 | +0.0017 +/- 0.0132 | +0.13 | flat |
| 14 | **on-policy distillation** | KL to the clone at the states the POLICY visits (Learning by Cheating) | 7 | 1.77 | 2.43 | 3.67 | **-0.2779 +/- 0.0930** | **-2.99** | **DECLINING** |
| 15 | **IBRL** | clone proposes, a learned Q head picks, no BC loss | 9 | **0.39** | **0.38** | 0.51 | -0.0050 +/- 0.0105 | -0.48 | **BROKEN -- see below** |

**Nothing is rising.** Five arms flat, one declining, one broken. 3 of 15 arms have produced a single update
above v15's 3.275 (fitnet, mutual, onpol) and none has produced a trend.

### IBRL is broken by a COLD Q head, and that is an implementation error, not a result

`bcPick` -- the fraction of steps where the clone's action was executed -- reads **54-78%** while `qL` is
still 2.5-10. The head starts untrained, so for the first many updates its "selection" is noise, and the
policy spends most of its steps executing a policy that takes 0.14 of her 10 HP. v15 went from 2.3 to 0.38
in nine updates.

The paper does not have this problem: its Q is trained by off-policy RL on a replay buffer that includes the
demonstrations, so it is informative before it gates anything. On-policy PPO has no such buffer, and the
proposal was enabled from update 1.

**The fix, not yet run:** gate the proposal on the Q head being useful -- no proposal until `q_loss` falls
below a threshold or for the first N updates, so the head is fitted on the policy's own actions first. Until
then any IBRL number from this repo is a measurement of a random gate, and claim R49 says so.

## THE FLEET WAS LOST AT ~20:14 AND NO ARM'S CHECKPOINTS SURVIVE

All 34 pods disappeared at once -- `fleet list` $0.00/hr, RunPod's API 0 pods. `fleet sync` had never been run
on any of the day's arms, so **no weights from any of the fifteen arms exist**, along with the eval pod's
per-episode rows and every per-rank log. W&B histories were exported afterwards and are the only record; that
export needed a fix first, because `zelda-rl-goal-ppo` -- where `fleet launch` sends combat runs -- was missing
from `scripts/export_wandb.py`'s project list.

**Run `fleet sync` whenever you read a number worth keeping.** Not at the end of a session.

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
