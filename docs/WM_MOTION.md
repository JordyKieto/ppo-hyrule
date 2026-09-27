# Link's motion is not action-conditioned, and the MOBA world model says why

Motion is the first thing a world model has to get right and ours does not have it. This is the diagnosis,
what the MOBA token world model in `../puffertank` did differently, and what to change here — in the order the
evidence supports, not the order that sounds most impressive.

## The measurement that says "0% solved"

Not the error. The **action ablation**. A world model that has learned motion must get worse when you feed it
the wrong actions; if it does not, it is extrapolating momentum and the action is decoration.

`scripts/smoke_mlp.py`, 1.5M transitions, L1, 5 epochs — position skill against persistence:

| | with the real action | action zeroed | gain from the action |
|---|---|---|---|
| Link position (world distance) | +0.4143 | +0.4127 | **+0.002** |
| Z-target timer | +0.662 | +0.058 | +0.604 |
| every enemy field | — | — | 0.000 |

**The action buys nothing for position.** The only thing it measurably drives is the Z-target press. And this
is not an artifact of the MLP being small: the same ablation class is what the MOBA study used to *prove* its
model had learned movement.

MOBA, held-out, one step (model / persistence / **shuffled actions**):

| field | model | persistence | shuffled | reading |
|---|---|---|---|---|
| hero x / y | **0.32 / 0.31** | 0.56 / 0.54 | **0.67 / 0.66** | conditioned on the move heads |
| hero health | 5.63 | 6.44 | 5.54 | combat NOT conditioned |

Shuffling the actions doubles their position error. That is what a conditioned model looks like. Ours has no
gap at all, so we have never demonstrated action conditioning on position — and the token WM's autoregressive
rollout being worse than doing nothing at every horizon (`scripts/wm_replay.py`: skill −1.57 at step 10,
−0.12 overall) is what an unconditioned motion model produces once error compounds.

## What MOBA does differently

Five differences. The first two are ours to fix cheaply; the third is the one the MOBA notes are most emphatic
about; the last two are where our task is genuinely harder and must be handled rather than copied.

### 1. The action is part of the entity's TOKEN, not just its query

`scripts/train_wm_tokens.py:249` — `ac = self.act_emb(a_onehot)[:, None].expand(...)`, with the comment
"action conditions the QUERY". The same action vector is added to every query row, so the action can change
what Link *attends to* but is not part of what Link *is*, and the position head decodes a token the action
never entered.

MOBA: "Each hero's 6 heads are embedded and summed into one action vector, **added to that hero's token**."

### 2. The action is a bare index over 108 combinations, with the geometry thrown away

Our action space is 9 world directions × 12 button combinations, one-hot. The model must learn from scratch
that index 0..8 correspond to compass directions — the geometry is recoverable only by memorising 108 rows.

And the direction really is world-relative already: `zelda_rl/env.py::_stick` converts a world direction to a
stick deflection using the live camera yaw. So the action already carries a clean world vector, and we hand the
model an integer instead.

**Feed the action's direction as `(sin θ, cos θ)` alongside the index.** This is the cheapest change on this
page and it gives the model the one thing it needs to turn "action 37" into a displacement.

### 3. Regression targets scaled by the std of NON-ZERO deltas, with per-field loss averaging

MOBA: "Regression targets are divided by a per-(class, field) scale: **the std of non-zero deltas** in training
data. Huber loss. Each (class, field) loss is averaged on its own and then summed, so rare fields weigh as much
as common ones. **Supervision masks:** a (class, field) is supervised only where it can change."

Ours (`zelda_rl/wm_data.py:113`): `"doff": (lo_n - T["link_off"])` — a raw delta, no per-field scale, every
frame supervised.

This is the same failure my own smoke test measured from the other direction: with MSE on all frames the
aggregate one-step skill was **−0.379**, and with L1 (the conditional median, which is zero on a no-change
frame) it was **+0.392**. Same network, same data, same epochs. Most frames do not move, so an unscaled,
unmasked loss is dominated by frames where the correct answer is "nothing", and the model learns to say nothing
slightly wrong rather than to say the right thing when something happens.

MOBA hit this too and quantified it: gating a field and averaging its value loss only over ticks where it
changed took walking from **−4 to +6**; rescaling that loss by the field's change rate took it to **+15**.

### 4. Position input: Fourier features on an absolute coordinate, geometry in the attention bias

MOBA: "Position through Fourier features... It is one fixed map: position encodes walls, lanes and waypoints.
It is normalized and Fourier-encoded, **never raw**. Relative geometry goes in the attention bias." Plus a 9×9
local wall patch per entity.

Ours: a 32×32 nav-grid **cell embedding** plus a within-cell offset, actors relative to Link, a 7×7 wall patch,
distance as a per-head learnable bias. Structurally this is the same idea and in one respect better (the
attention bias is already there).

**But we have never measured our codec.** I previously claimed it was worth +156% on the position head; that
number has no source in this repo and I withdrew it. Position is not even an ablatable class in
`scripts/ablate_classes.py`. A 32×32 grid over ~920 × 1830 world units means a cell is ~29 × 57 units, and
Link's walking step is ~13 units — **most steps do not leave the cell**, so the 25-way `dcell` head is being
asked to predict "same cell" almost always while all the real signal sits in the unscaled `doff` regression.
That interacts badly with point 3 and is a plausible primary cause.

### 5. Where our task is harder, and must not be copied

MOBA states plainly: "movement is the action vector applied to the position" — deterministic, which is why
their criterion is "hero position error **well under 0.1 cell**". Zelda's is not:

* displacement depends on the **animation state** (`action_fn_rel`) — a roll, a swing and a walk consume the
  stick differently, and mid-animation the stick is ignored entirely;
* it depends on **momentum** (`speed`, `vel_x/z`), and on collision with geometry we only partly know;
* the ablation already told us the action's one-step effect is nearly nil *because* the animation state
  encodes the previous action's consequence — at 10 Hz with 6 VI per step, where Link goes in the next 100 ms
  is mostly determined by what he is already doing.

So the target is not "action → displacement". It is **"(action, animation state, momentum) → displacement"**,
and any test of conditioning has to hold the animation state fixed.

## Proposal, in order

Each step is one change with one measurement, against the existing baseline. Nothing here needs a fleet.

1. **Build the counterfactual test first, before changing the model.** MOBA's own criterion is a counterfactual:
   restore one state, apply two different action sets, check the model predicts both. We can do this exactly —
   load a savestate, apply action A, reload, apply action B, record both real outcomes. A few hundred pairs on
   one local Dolphin. That gives a *ground-truth* action-effect dataset and turns "is motion conditioned" from
   an argument into a number. **Without this, every change below is unfalsifiable.**
2. **Feed the action's world direction as sin/cos**, and add the action embedding to Link's token rather than
   only to the query (differences 1 and 2 — they are one edit in the same place). Measure on the
   counterfactual set.
3. **Scale `doff` by the std of its non-zero deltas and mask supervision to frames that moved** (difference 3).
   Measure the same way. My smoke test predicts this is worth more than the architecture change.
4. **Then, and only then, revisit the grid.** Measure the cell codec for the first time by adding position as an
   ablatable class in `ablate_classes.py`, and test a finer grid (64×64, so a cell is ~14 × 29 units against a
   13-unit step) against the current 32×32. Fourier features on the within-cell offset are the MOBA-style
   alternative to test in the same run.

Two things NOT to do: do not raise the control rate (the action's weak one-step effect is a property of
animation state, not of timestep), and do not extend the rollout horizon until one-step motion is conditioned —
`docs/WM_EXPERIMENT_REGIME.md` already records that buying depth without a working one-step map yields
−2.44 skill at H=1800.

## Source

`../puffertank/docs/moba/TOKEN_WM_DESIGN.md` (tokens, heads, target scaling),
`../puffertank/docs/moba/EXPERIMENTS.md` (the shuffled-action control, the walking-under-the-gate loss-balance
result), `../puffertank/mobawm/wm/model.py` (action embedding added to the entity token; per-head distance and
team attention bias).

---

# Results (2026-09-26). Two of my four proposals are wrong, and the fourth is the real problem.

All measured locally in minutes, in `scripts/smoke_mlp.py` (1.5M transitions, L1, 5 epochs, ~70 s an arm) and
in direct data analysis. Baseline to beat: **position skill +0.3893**.

## Step 2 — action geometry: DISPROVED

| arm | position skill |
|---|---|
| one-hot action (baseline) | +0.3893 |
| + sin/cos of the action's world direction | +0.3937 |
| + geometry, **action zeroed** (the control) | **+0.3971** |

Feeding the geometry changed nothing, and the arm with *no action at all* scored highest. The encoding was
never the problem: the model does not use the action, so how the action is written down is irrelevant.

### Why — and it is not a model failure

The action's direction barely predicts displacement **in the data**. Cosine between the commanded world
direction and the actual displacement direction, by lag:

| filter | lag 1 | lag 2 | lag 3 | lag 5 |
|---|---|---|---|---|
| all actions | **0.006** | 0.097 | 0.124 | 0.135 |
| move-only actions (no button) | 0.044 | 0.111 | 0.205 | 0.234 |
| move-only, from a standstill | 0.032 | 0.126 | 0.233 | 0.263 |
| move-only, no sword swing | 0.060 | 0.142 | 0.241 | **0.260** |

**At one step the action carries almost no directional information (0.006), and it never gets strong** — the
best conditional alignment is 0.26, from a clean move command with no swing. Momentum, turn-in-place animation,
animation lock and collision dominate.

So a one-step objective cannot be action-conditioned on position: the relationship is not there to learn.
`AGENTS.md` §1 says a number is not a result until its controls are run; this is the control that should have
been run before proposing step 2 at all.

**A correction to my own first attempt at this test.** It reported cosine ≈ 0.000 at *every* lag, which I nearly
wrote up as "no relationship exists". That was my bug: I built the direction vector as `(sin θ, cos θ)`, which
is Link's YAW convention, where the action angles need `(cos θ, sin θ)`. A 90° rotation reads as exactly zero
correlation. Six conventions were then tested against the data and only `(cos, sin)` shows signal.

## Step 3 — moved-only supervision and non-zero-delta scaling: WORSE

| arm | position skill | aggregate over 191 live dims |
|---|---|---|
| baseline | +0.3893 | +0.3597 |
| masked to moved frames + non-zero-delta scale | **+0.2125** | **−181.8** |

Nearly halves position skill and destroys everything else. The reason is that **MOBA pairs masked supervision
with a change GATE** — a "does this field change this tick" classifier, with the value head used only where the
gate says yes. I implemented the mask without the gate, so the model was never trained to emit zero on a
no-change frame, and the metric (error over all frames, 44% of which do not move) punishes that enormously.

MOBA's design is not disproved; my implementation of it was half a design. Their own notes say the gate's
yes/no losses then crowd out movement and need down-weighting ×0.2, which is a second knob this experiment
never reached.

## Step 4 — the grid: this is the real problem

Measured on 400,813 transitions, no training. Per-step displacement: mean **5.50** world units, median 2.28,
zero on 44.1% of steps; among moving steps mean 9.82.

| grid | cell size (x, z) | same-cell, all steps | **same-cell, MOVING steps** | quantisation error |
|---|---|---|---|---|
| **32 (current)** | **142.3 × 65.9** | 92.6% | **86.9%** | **44.4** |
| 48 | 94.8 × 43.9 | 89.3% | 80.8% | 30.0 |
| 64 | 71.1 × 32.9 | 86.0% | 74.9% | 22.6 |
| 96 | 47.4 × 22.0 | 80.3% | 64.8% | 15.0 |
| 128 | 35.6 × 16.5 | 75.6% | 56.4% | 10.7 |

**The 32×32 cell is 142 units wide against a 5.5-unit mean step.** The `dcell` head — a 25-way class whose
whole purpose was to carry the multi-modal "where in the room" decision — answers "same cell" on **87% of
moving steps**, so it is nearly a constant, and every bit of real motion signal sits in `doff`, the unscaled
offset regression that step 3 was supposed to fix. Worse, quantisation error is **44 units**: even a perfect
cell prediction is 8× the mean step away from the truth.

**This also corrects the estimate in this document.** I wrote that a 32×32 grid gives a ~29 × 57 unit cell. The
grid is fitted to the data's robust percentiles (`NavGrid.fit`), and the real extent is much wider on x, so the
cell is 142 × 65.9. The conclusion is stronger than the guess was, but the guess was wrong and was arithmetic I
should have run rather than estimated (§3).

## Where this leaves motion

The ordering in my proposal was wrong. Corrected, by measured effect:

1. **The grid is too coarse by an order of magnitude.** Test 96 or 128 against 32 on the token WM. This is the
   one change with a measured mechanism behind it.
2. **Any action-conditioning test must be at lag ≥ 2 and conditioned on animation state.** At lag 1 there is
   nothing to condition on, so the shuffled-action control will always read flat and no model change can pass
   it. This reframes the autoregressive work: multi-step training is not an optional refinement for motion, it
   is the only regime where the action is learnable at all.
3. **Masked supervision needs the change gate, or not at all.** Half of it is worse than none.
4. Action geometry is not worth pursuing. Closed.

## Not finished

`scripts/collect_counterfactual.py` (proposal step 1, the ground-truth "same state, two actions" set) is
written but has not produced data yet — two bugs found so far, an env built with `task="goal"` (27 actions
instead of the combat 108) and a state reader returning an object rather than the flat 363-vector. It is still
the right test to own, because it would measure action effect directly instead of inferring it from
correlations, and the lag table above says it should branch the actions at least 3 steps apart.
