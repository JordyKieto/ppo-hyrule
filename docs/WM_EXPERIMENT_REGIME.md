# The current world-model experiment regime

Two models, two GPU pods, one 91.5M-transition dataset. This document is the state of the experiment: what is
running, why each choice was made, what has been measured, and which conclusions are NOT safe yet.

## The two models

| | `scripts/train_wm_tokens.py` (token WM) | `scripts/train_jepa_latent.py` (latent JEPA) |
|---|---|---|
| rolls forward in | OBSERVATION space: predicts `dS`, re-encodes its own predicted state as tokens | LATENT space: `z_{t+1} = T(z_t, a_t)`, decoders read reward fields OUT of `z`, nothing decoded re-enters |
| heads | FULL CLOSURE -- every consumed field is predicted (22 typed heads) | reward fields only, which is the JEPA thesis |
| params | ~2M | enc 5.13M + trans 1.50M + dec 3.08M |

They share the tokenizer (`zelda_rl/tokens.py`), the context builders (`zelda_rl/wm_context.py`), the packing
format (`zelda_rl/wm_data.py`) and the attention block (`Attn`, which lives in the token WM's file and is
IMPORTED by the JEPA -- so a change to it must be pushed to both pods regardless of which model runs there).

## Shared architecture, as of this regime

* **Three entity classes**: Link, actors, and static ENVIRONMENT. `type_emb` is sized 3.
* **Environment tokens are keys/values ONLY.** A 7x7 wall patch is 49 tokens against at most 10 entity tokens,
  so 83% of the pool; as queries they would take attention from 10x11 to 59x60 pairs and produce 49 query rows
  no head reads. A wall has no dynamics to predict. `Attn` takes them through `ctx`.
* **8-frame history, summarised LOCALLY** per entity into one token before the global attention (Wayformer
  polyline factorisation), clamped at the episode start rather than zero-padded.
* **Dual local frames**: each actor's offset in Link's FACING frame and in the CAMERA frame. Movement is
  camera-relative (the stick), combat is facing-relative (the sword hits in front). Both derived from packed
  fields, no repacking.
* **No raw coordinates anywhere.** Position enters as a nav-grid cell EMBEDDING. Measured as the strongest
  single input (+156% on the position head), so this is not a compromise.
* **Typed heads**, HP classified not regressed (damage F1 0.286 -> 0.906), HP binned on the INPUT side too.
* **Zero-init heads + logit residuals**, so skill starts at exactly 0.000 and any deviation is learned.
* **No action leak**: the action conditions the QUERY side only; token fields are verified disjoint from
  `bc.CONTROL_ECHO_FEATS`.

## The training regime: teacher-forced pretrain, then autoregressive fine-tune

This two-stage shape is the finding, not an implementation detail.

**Stage 1 (teacher-forced, one step).** Input is 8 frames of REAL data, target is t+1. Fast and parallel: the
token WM hit 624k rows/s at 99% GPU. Produces a good one-step map and nothing more -- the JEPA decoded almost
perfectly at H1 (HP 1.000, Baba state 0.999, grid IoU 0.871) and fell off a cliff between H2 and H5, with HP
accuracy 0.987 -> 0.023, i.e. confidently predicting damage that does not happen. That is a latent leaving the
manifold its decoder was trained on, which is exactly what one-step training buys.

**Stage 2 (autoregressive, mixed depth).** Resume the stage-1 checkpoint -- architecture unchanged, so the
optimizer state carries -- and roll the model forward on its OWN output.

* **Depth mixture**, sampled per batch, fixed (NOT a curriculum): 60% one-step, 15% H=2, then geometric buckets
  down to 0.15% at H=1800, the full 180-second match. Mean depth 17.4. Note 60% of BATCHES are one-step but
  only ~3% of the COMPUTE, while the 0.15% at H=1800 is ~15% of it.
* **Step decay** `0.5^((h-1)/600)`, renormalised. Parameterised as a HALF-LIFE, not a raw gamma: at gamma=0.95,
  `gamma^1800` is 1e-40 and every deep step you paid for would get exactly zero weight.
* **Drift-correcting targets.** In a rollout step the target is `(true state at j+1) - (ROLLED state at j)`, not
  the true delta from the true state. Targeting the latter would have the model faithfully reproduce the right
  increment from the wrong place, i.e. perpetuate its own error.
* **Truncated BPTT**, backward per window. Detaching the state alone does NOT bound memory: every step's loss
  tensor keeps its own subgraph, so summing 1800 of them OOMs an 80 GB card.
* **Stochastic rollout**: discrete heads are SAMPLED, not argmaxed. A Baba's inner state is unobservable, so its
  next behaviour is genuinely random from our side; the head's distribution is the marginal over that state.

## What stage 2 bought, measured

Latent JEPA, H = position skill vs a persistence control on DECODED fields:

| H | stage 1 (teacher-forced) | stage 2, 12 epochs |
|---|---|---|
| 1 | +0.370 | +0.379 |
| 2 | **-2.63** | **+0.311** |
| 5 | **-2.65** | **+0.174** |
| 10 | -1.29 | +0.078 |
| 20 | -0.83 | +0.016 |

Every horizon positive, with H1 simultaneously at its best. I had predicted the switch would COST H1 accuracy;
it did at first (0.370 -> 0.232 after one epoch) and then recovered. Neither stage alone reaches this.

**Not safe to conclude yet.** Epoch-to-epoch variance is large because the depth mixture is stochastic -- the
epoch after that best draw fell to H5 +0.010, H10 -0.070. The honest summary is "H1 ~ +0.33, H2 ~ +0.23,
H5 ~ +0.08, H10 ~ 0, trending up", not the best row.

## The open problem: HP under rollout

`l_hp` is the ONLY head that gets worse with depth, and it does so in both models. In the token WM's per-step
breakdown it rises 0.0086 at step 2 to 0.0854 at step 16 -- a 10x increase -- while every other head decays.
In the JEPA it is the HP accuracy collapse above. Since it appears in two different architectures it is a
property of the TASK: HP changes are rare discrete events (99.78% of steps have none), so once the rolled state
drifts the model starts inventing damage. This is the thing to fix next, and it matters because damage is what
the reward pays for.

## Utilisation, and a correction

An autoregressive rollout is sequential, so it cannot be batched over steps. I told the user this capped GPU
utilisation at ~27% and that teacher forcing was the only way to saturate the card. That was true of the THIN
model (1 attention block, no history): with 3 blocks, an 8-frame history and 49 environment keys, each step is
enough work to hide the launch overhead and the token WM runs autoregressively at 93%. The JEPA is at 35% and
that gap is unexplained -- roughly 2.5x throughput is sitting idle there.

## Honest limits of the setup

* **The wall map is incomplete geometry**: 209 of 1024 cells, accumulated from frames where Link actually
  touched something. Walls he never bumped into do not exist in it.
* **Five architecture changes went in at once** (env tokens, history, dual frames, HP bins, depth 3). They were
  approved as a set and deployed as a set, so if the numbers moved we do not know which change did it.
  `scripts/ablate_classes.py` exists to untangle that and has not been run on these checkpoints.
* **H=1800 is measured on a thin, biased subset**: only the opening frames of full-length timeout episodes have
  1800 steps of room, ~31k of 52M starts, and timeouts are by definition episodes where the policy never killed
  anything. H100/H400 are the trustworthy deep numbers.
* **Loss is never the criterion** (`AGENTS.md` 8). A run once halved its validation loss while its rollout skill
  fell from +0.0026 to -0.41, and the LR sweep showed loss decreasing monotonically across the entire range
  while rollout skill peaked in the middle.

## Reproducing / watching

```sh
RUN=wm_v3   scripts/token_watch.sh -w     # token WM: entity x horizon, depth buckets, step bins
RUN=jepa_v3 scripts/wm_table.sh -w        # JEPA: entity columns, per-horizon skill, depth buckets
```
Launchers live on the pods as `/root/trun.sh` (token WM) and `/root/jrun.sh` (JEPA); both run
`gpu_clear.sh` first, which kills leftover trainers and refuses to start until the card reports empty.
