# The tokenized attention world model

This is a **different model from the one in `LATENT_WORLD_MODEL.md`**, and confusing the two is how a result got
filed under the wrong mechanism. That document is 30 sections about predicting the next *latent* of a learned
encoder: six architectures, all of which lose to the identity because the latent moves ~0.04/step so copying it
is already near-optimal. This document is about `scripts/train_wm_tokens.py`, which does not have a latent
rollout at all:

    S_t (entity tokens) + a_t  ->  Delta S_{t+1}, per entity, per field, typed heads

State in, state out. Attention over entity tokens, a delta per field, and the rollout re-encodes the *predicted
state* each step. There is no bottleneck to collapse and no encoder to co-adapt with the predictor. The
`LATENT_WORLD_MODEL.md` negatives constrain the latent approach; they say nothing about this one.

## What is in the model, and why each piece is there

The seven anti-pathology constraints are listed in the script's own docstring and enforced in code, not prose.
In brief: a never-masked no-op key so no softmax row is all-masked; no raw coordinates anywhere
(`zelda_rl/tokens.py` -- actors egocentric to Link, Link relative to his episode start, positions entering only
as grid cell + within-cell offset via `zelda_rl/navgrid.py`); distance as a per-head learnable bias
`-softplus(w_h)*D_ij` rather than a hardcoded radius; separate Link and actor encoders; **zero-initialised delta
heads plus a `LOGIT_K=6` logit residual so the model predicts "nothing changes" exactly at step 0**; twelve typed
heads (L1 for positions, cross-entropy for HP/state/scene, BCE for flags); and the action entering only on the
query side, verified disjoint from `bc.CONTROL_ECHO_FEATS`.

Point 5 is what makes the numbers here readable. **The model starts exactly at persistence**, so
`skill = 1 - dpos/persistence` is 0.000 on the first batch, and any deviation is something training did.

## The fleet run, and the confound that invalidates half of it

20,545,657 transitions on 39 RunPod pods with the `badplay` mix (61% success, 5% death, 5% room-exit, 28%
timeout), then data-parallel FedAvg on the 32 pods that had a shard, `lr=1e-5`, batch 512, averaged every epoch.
wandb project `zelda-wm` (groups `v4-collect-wm`, `v4-collect-wm100`); local copies in `runs/wm5_synced/` and
`runs/wm100_synced/`.

| epoch | train | val | 1-step skill | dcell | rollout H10 skill | |
|---|---|---|---|---|---|---|
| 1 | 2.1344 | 1.0644 | −0.0012 | 0.8816 | −0.0003 | |
| **5** | 1.6313 | 0.9750 | −0.0141 | 0.8813 | **+0.0026** | last epoch before the resume |
| 10 | — | 1.0592 | −1.9876 | — | −0.3134 | **first epoch after the resume** |
| 15 | — | 0.8778 | −1.9807 | — | −0.8469 | |
| 100 | **0.8723** | **0.5998** | −0.9973 | 0.8271 | −0.4098 | |

The 100-epoch run was **resumed from the epoch-5 checkpoint, and `--resume` did not restore optimizer state.**
A fresh Adam on a trained model runs its first steps with zero second moments, so the effective step is ~`lr`
on every parameter regardless of curvature. `dpos` went 1.048 -> 10.572 across that boundary. The collapse is
complete by epoch 10, which is the first epoch after the resume, and nothing between epoch 10 and 100 recovers
it.

So the honest reading:

- **Epochs 1-5 are valid.** Trained from scratch, and they show the model at the persistence baseline with a
  rollout skill of +0.0026 -- under 1%, one seed. A sign of life, not a result.
- **Epochs 10-100 measure a wrecked optimizer, not the effect of training or of scale.** Every conclusion that
  compared them to epoch 5 is void.
- In particular, **"scale was not the constraint" is NOT established.** That was the headline conclusion and it
  rests entirely on the post-resume epochs. The 20M-transition question is still open and has never been
  answered by a clean run.
- The one thing that survives and generalises: **never select or stop on the training loss here**
  (`AGENTS.md` §8). Val loss fell 1.8x while the rollout got worse. Selecting on val loss picks epoch 99. Best
  epoch selection on held-out *position* error is in the script.

What is genuinely unexplained, and is not a resume artefact because it is visible inside the clean epochs:
`a_dcell` (the Baba movement head) is the only head that degrades over training.

## The fix, so this cannot recur

`--resume` now **hard-errors** on a checkpoint with no optimizer state rather than silently starting a fresh
Adam; checkpoints carry `opt`, `hist` and the running `best`, so a resumed run is continuous in the optimizer,
in the metric history and in best-epoch selection. `--resume-fresh-opt` opts into the old behaviour and prints
that the run is discontinuous. A warning was not enough: in a 32-rank fleet launch a warning scrolls past
unread, which is exactly how this one survived. (`AGENTS.md` §21.)

Checkpoints written before this fix (`runs/wm5_synced/*`, `runs/wm100_synced/*`) have no `opt` key and will be
refused.

## The clean experiment this is waiting on

The 66.1M-transition v5 dataset is local (`datasets/transitions/v5/rank*/`, 79,694 episodes; see `MAP.md` for
the pull). The run to do is **100 epochs from scratch, no resume**, on v5, with cosine LR + warmup at
`lr=1e-5`, checkpointing every 5, judged on the H=1/2/5/10 rollout rather than the loss. That is the run that
answers whether scale helps, which the confounded run did not.

The lever to test after it, and only against the rollout: per-head loss weighting or per-head gradient
normalisation, so the 25-way `dcell` and class-weighted `hp` cross-entropies cannot buy their own loss down by
trading away position. That trade is real and was recorded at small scale in `LATENT_WORLD_MODEL.md` §24; what
is no longer established is the claim that 100x the data does not change it.
