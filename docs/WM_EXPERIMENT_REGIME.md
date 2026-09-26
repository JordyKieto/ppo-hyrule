# The teacher-forced -> autoregressive world-model regime (CONCLUDED 2026-09-23)

Two models, two GPU pods, one 91.5M-transition dataset. Both runs are finished and both pods are terminated;
the checkpoints and full `metrics.jsonl` are saved (see **What was kept** at the bottom). This document is the
regime as it was actually run, what it bought, and where it stopped paying.

**The one-line result.** Autoregressive fine-tuning on top of a teacher-forced pretrain is what makes the
latent JEPA beat persistence at all beyond one step -- H2 went -2.63 to a 28-epoch mean of +0.251 -- and it
buys nothing past H5. H10 is a coin flip, H20 is negative in 25 of 28 epochs, and the one deep evaluation
that ran is **-0.289 at H100, -1.75 at H400, -2.44 at H1800**. For the token WM it was a straight regression:
one-step skill fell from +0.188 teacher-forced to a negative mean over 7 epochs and did not recover.

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
* **No raw coordinates anywhere.** Position enters as a nav-grid cell EMBEDDING. RETRACTED: an earlier draft
  of this line claimed "+156% on the position head" for the cell codec. That number has no source in this
  repo and I withdrew it -- position is not even an ablatable class in `scripts/ablate_classes.py`, so it has
  never been measured. What the ablation DOES say about position inputs: `l_bg_bits` +39% (third strongest)
  and `l_cam` +21%.
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

Latent JEPA, skill = 1 - err/persistence_err on DECODED position, 28 epochs of stage 2 (ep 46-73):

| H | stage 1 (teacher-forced) | stage 2 mean | stage 2 best | stage 2 last | epochs negative |
|---|---|---|---|---|---|
| 1 | +0.370 | **+0.333** | +0.379 @ep56 | +0.330 | 0 / 28 |
| 2 | **-2.63** | **+0.251** | +0.311 @ep56 | +0.244 | 0 / 28 |
| 5 | **-2.65** | **+0.091** | +0.174 @ep56 | +0.041 | 0 / 28 |
| 10 | -1.29 | -0.002 | +0.093 @ep52 | -0.064 | 16 / 28 |
| 20 | -0.83 | -0.055 | +0.030 @ep60 | -0.097 | 25 / 28 |
| 100 | -- | -0.289 | (one eval, ep 60) | | 1 / 1 |
| 400 | -- | -1.753 | (one eval, ep 60) | | 1 / 1 |
| 1800 | -- | -2.442 | (one eval, ep 60) | | 1 / 1 |

Read the MEAN column, not the best column. I reported epoch 56 mid-run as "the session best, all positive
through H20"; with all 28 epochs in hand that row is the top of a noisy band -- H20 was positive in 3 epochs
out of 28. The gain is real and it is bounded: **positive through H5, nothing beyond.**

Training loss fell from 2.061 to 1.738 (-16%) over those 28 epochs while H5 went +0.174 to +0.041 and H10
went positive to negative. That is the project's recurring loss/task split (`AGENTS.md` 8) showing up again,
in the regime that was supposed to fix it.

**The token WM went the other way.** Its one-step skill on a fixed 8192-index held-out subset, by epoch:

| ep | 81 | 82 | 83 | 84 | 85 | 86 | 87 |
|---|---|---|---|---|---|---|---|
| skill | +0.013 | -0.617 | +0.040 | -0.030 | -0.406 | -0.358 | -0.130 |

Teacher-forced was +0.188 at its best and +0.187 at its last epoch (`runs/wm_v2/metrics.jsonl`; an
earlier draft of this document said +0.091, which was the JEPA's H5 mean copied into the wrong row -- the
regression is therefore LARGER than first written). The eval subset is a fixed prefix and the rollout uses `sample=False`
(`train_wm_tokens.py` eval path), so this is genuine model movement, not measurement noise. Some one-step
loss was the expected price of changing the objective; ending at a negative MEAN with no recovery trend in 7
epochs is not. The likely mechanism is the depth mixture: 60% of batches are one-step but only ~3% of the
compute, so the head that the one-step metric reads is the one being starved.

## Why the two models diverged under the same regime

Same depth mixture, same decay, opposite outcome. The difference is what rolls forward. The JEPA rolls a
LATENT and its decoders never feed back, so a rollout step is a clean function of `z`; the token WM re-encodes
its own predicted observation, so every head's error enters the next step's input through the tokenizer. Full
closure is what makes the token WM a real world model and it is also what makes its rollout unforgiving.

## The open problem: HP under rollout

`l_hp` is the ONLY head that gets worse with depth, and it does so in both models. In the token WM's final-epoch
per-step breakdown it rises from 0.0116 at step 2 to 0.0481 at step 32, a 4x increase, while every other head
decays monotonically across the same bins (`l_dcell` 0.088 -> 0.087 -> 0.032, `l_vel` 0.010 -> 0.0025).
The effect held every epoch of the run and was 10x at an earlier epoch, so the ratio moves but the sign does not.
In the JEPA it is the HP accuracy collapse above. Since it appears in two different architectures it is a
property of the TASK: HP changes are rare discrete events (99.78% of steps have none), so once the rolled state
drifts the model starts inventing damage. This is the thing to fix next, and it matters because damage is what
the reward pays for.

## Utilisation, and a correction

An autoregressive rollout is sequential, so it cannot be batched over steps. I told the user this capped GPU
utilisation at ~27% and that teacher forcing was the only way to saturate the card. That was true of the THIN
model (1 attention block, no history): with 3 blocks, an 8-frame history and 49 environment keys, each step is
enough work to hide the launch overhead and the token WM held **96%** for the whole run. The JEPA sat at
**33%** and that gap was never explained -- roughly 2.5x throughput was idle on pod 1 for 28 epochs.

## The untested suspect: clipping

The JEPA's gradient norm ran **105-120 against `--clip 20`**, so every single step of all 28 epochs was
scaled down 5-6x. Under that regime the update is a fixed-SIZE step in the gradient's direction and the
optimizer loses all magnitude information -- which is a plausible mechanism for a 20-epoch plateau at
H1 ~ +0.33 with the loss still falling. `docs/NEXT_SESSION_RUNPOD.md` flagged the identical pattern
(`--clip 1.0` against grad 11-15) for an earlier architecture and it was never tested either. The token WM
ran grad ~27 against the same clip, i.e. 1.35x, so it is NOT the explanation there.

This is the cheapest open experiment: rerun the JEPA stage-2 fine-tune from `runs/jepa_v2/ck_0045.pt` with
`--clip 100`, everything else identical, and compare H5/H10.

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

## What was kept

Both pods were terminated after the run. Saved locally under `runs/` (gitignored -- weights do not belong in
git; push to `s3://zelda-rl/zelda/` if they need to outlive this machine):

| path | what |
|---|---|
| `runs/jepa_v2/ck_0045.pt` + `metrics.jsonl` | stage-1 teacher-forced JEPA, the resume point for any stage-2 rerun |
| `runs/jepa_v3/ck_0055.pt`, `ck_0060.pt`, `ck_0070.pt` | stage 2; ck_0055/0060 bracket the ep-56 peak, which was never itself written (saves are every 5 epochs) |
| `runs/jepa_v3/metrics.jsonl` | all 28 stage-2 epochs |
| `runs/wm_v2/ck_0080.pt` + `metrics.jsonl` | stage-1 teacher-forced token WM, best werr 1.806 -- the checkpoint to prefer over anything in v3 |
| `runs/wm_v3/ck_0085.pt` + `metrics.jsonl` | stage 2, kept only as the record of the regression |

## What to run next, in order

1. **`--clip 100` on the JEPA stage-2 rerun.** One variable, a known baseline to compare against, and it
   addresses the plateau rather than the horizon.
2. **Ablate the five changes.** Environment tokens, 8-frame history, dual frames, HP bins and depth 3 all went
   in as one approved set, so none of the v2/v3 gains are attributed. `scripts/ablate_classes.py` exists for
   this and has never been run on these checkpoints.
3. **Fix `l_hp` under rollout** before chasing horizon again. It is the only head that worsens with depth and
   it does so in BOTH models, so it is a property of the task, not the architecture -- and damage is what the
   reward pays for.
4. **Do not extend the horizon mixture further** until 1-3 land. H100 at -0.289 says the deep buckets are
   already being trained on rollouts the model cannot do; paying more compute for H1800 is paying for noise.

## Watching a run of this kind

```sh
RUN=wm_v3   scripts/token_watch.sh -w     # token WM: entity x horizon, depth buckets, step bins
RUN=jepa_v3 scripts/wm_table.sh -w        # JEPA: entity columns, per-horizon skill, depth buckets
```
Both take `POD_IP`/`POD_PORT` from the environment; the defaults in those files point at pods that no longer
exist. Killing a trainer over ssh needs a bracketed pattern (`AGENTS.md` 23) or pkill takes the ssh session
instead and returns 255, which looks exactly like a network failure.
