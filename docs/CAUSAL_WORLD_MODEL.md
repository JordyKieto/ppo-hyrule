# Causal object-oriented world model: background, schema, and the Stage 0 field census

**Plain-English companion: [`CAUSAL_PLAIN_ENGLISH.md`](CAUSAL_PLAIN_ENGLISH.md)** -- same facts, no jargon.
This file is the record of authority; if the two ever disagree, this one is right.

A second methodology trail, started where [`LATENT_WORLD_MODEL.md`](LATENT_WORLD_MODEL.md) stops. That
document records thirty sections of work on `z_{t+1} = M(z_t, a_t)` and ends with six consecutive negatives.
This one records what replaces it and why, and it follows the same rule: what was built, what was measured,
what each change was a response to. Not an argument for a conclusion.

**Method:** Object-Oriented Causal Dynamics Model (OOCDM), Yu, Ruan & Xing, *Learning Causal Dynamics Models
in Object-Oriented Environments*, ICML 2024. Paper PDF and the authors' reference implementation are local
(`~/Downloads/yu24j.pdf`, `~/Documents/dev/oocdm`).

**Status:** Stage 0 COMPLETE. Stage 1 trained at `G1` and section 16 shows WHY that cannot work. The plan of
record is **section 17 (P0-P3)** -- porting the paper's discovery loop. Section 14 is the field-by-field
result table, section 16 is the diagnosis, section 17 is what happens next and nothing outside it is in
scope.

**The code was restored to the paper in commit `<this one>`.** A change gate, per-field `pos_weight`
oversampling, binned direction/magnitude heads and threshold calibration were added in one session, none of
them are in Yu et al., and all have been removed. See AGENTS.md section 13.

---

## 1. Why this exists: the invariant across six failures

Sections 16-30 of the latent trail each changed the *network* and held the *interface* fixed. The interface
was: one 128-d monolithic `z`, next-latent regression, the 363-d state flattened into a single vector, one
predictor. Every result has the same signature.

| § | change | outcome |
|---|---|---|
| 22 | — (first ground-truth rollout) | dream position error 103 vs **50** for "Link never moves" |
| 24 | `w_pos` sweep | position and events provably compete: `w_pos 200` gives trajectory 1.06x and damage **0%**; `--detach-pos` recovers damage and wrecks walking (2.36x). Falsified as avoidable. |
| 25 | 40 runs, 3 levers | **null**: no lever beats baseline; 6-seed baseline spans 1.38x-3.43x |
| 28 | competent policy as encoder | **null**: +0.742 was a control echo + a clock; untrained same-arch scored higher |
| 29 | shallow+tanh / FiLM / transformer | **three negatives**; latent moves 0.023-0.044 per step, action explains 0.04-0.07 of a transition |
| 30 | categorical / VQ latent | predictor learns to **copy `z_t`**; skill exactly +0.000 |
| b88868f | latent vs state-space rollout | latent recursion **3x worse than do-nothing at H=10**; `\|z\|` drifts 13.3 -> 41.0 |

When six changes fail with one signature, the hypothesis is whatever was held constant. AGENTS.md §9 covers
"ablate before you scale"; it did not cover "vary the thing you never varied".

Note what *did* work in those same runs: the action-conditioned heads on real latents (moved AP 0.435, 6.8x
base; enemy-hit AP 0.628; nearest-Baba distance MAE 26.8 against a 64.0 mean-predictor). Those read
`(z_t, a_t) -> what happens`, **never `z_t -> z_{t+1}`**. The component of ours that works already has
OOCDM's shape, and b88868f's STATE-space arm — position integrated from the transition head, `z` re-encoded
every step rather than accumulated — beat the latent arm 2.3x at H=100 and pinned `|z|` at 12.8.

**OOCDM is the principled version of that experiment: a head for every field, no latent recursion at all.**

---

## 2. What an OOCDM is

An **OOMDP** (paper §3.2) is a set of objects, each an instance of a class. A class declares **fields**;
a field is either `state` (predicted at t+1) or `action` (an input, never a target). All instances of a class
share one transition function, so parameters *and the causal graph* are per-class, not per-object, and
neither grows when an actor spawns.

Three things about it are easy to get wrong:

**(a) The graph is over `(class, field)` pairs, not over entities.** There are exactly two kinds of edge
(`core/causal_graph.py`):

- `local_matrices[C]`, shape `(n_state_fields(C), n_fields(C))` — field `U` of an object affects field `V'`
  of **that same object**. e.g. `Link.speed -> Link.pos'`.
- `global_matrix`, one matrix of shape `(n_state_fields_total, n_fields_total)` indexed by `(class, field)`
  across all classes — field `U` of **any other** object of class `Cl` affects `V'` of any object of `Ck`.
  e.g. `En_Dekubaba.pos -> Link.hp'`.

That is the whole compression. Measured on our schema (§5): **8,286 class-level edges against 132,132 for a
flat CDL over the same 363-d state, and the class-level count does not change when an actor spawns or dies.**

**(b) Acyclicity is free, not a constraint to enforce.** Theorem 3.2: if state variables transit
independently given `(S, A)`, the true graph is **bipartite** — every edge runs from a `t` variable to a
`t+1` variable, none within `S'`. The "DAG" is an adjacency matrix from (state ∪ action) at `t` to state at
`t+1`. Nothing to project, nothing to regularise.

That assumption is unusually safe here: **OoT is a deterministic emulator.** Given `(s_t, a_t)`, `s_{t+1}` is
a delta function, so factorisation across fields is exact rather than approximate. The paper's models pay for
stochasticity we largely do not owe. (What we *do* owe is partial observability — see §9.)

**(c) `p` and the supervision are two separate loops.**

- `p̂` is the factored transition distribution `p(S'|S,A) = Π_j p(S'_j | Pa(S'_j))`. Each state field gets a
  shared-per-class head emitting **distribution parameters** — `Normal(mean, scale)`, `Categorical(logits)`
  (`core/ptype.py`). Supervision is per-field NLL averaged **per instance** (their AILL, eq. 8), so the loss
  is invariant to how many actors are alive.
- The **graph** is supervised with no second model. During training the graph is randomly masked (each edge
  kept w.p. λ), so the model learns to answer "what would you predict *without* this parent?". Discovery is
  then a conditional mutual information estimate, `E[log f(V' | G_full) − log f(V' | G_minus_edge)]`,
  thresholded at ε.

**The ablation is the training objective** — AGENTS.md §1 compiled into a loss function.

---

## 3. How it is implemented: four modules and a mask multiply

From `alg/model/object_oriented_model.py`. There is no "causal layer"; the causality is a boolean broadcast.

| piece | what it is |
|---|---|
| schema | plain Python. `c = EnvObjClass('Baba'); c.declear_field('hp', vtype, role='state')`. Not a tensor. |
| `ClassAttributeEncoder` | `HeterogenousLinear` — one `Linear` per field, since fields have different widths → `(B, n_obj, n_field, d)` |
| **the causal graph** | `masked_retain(x, attrmask)` — zero the embeddings of non-parent fields, flatten `n_field × d`, `Linear`. That is the entire mechanism. |
| class sharing | `MultiLinear` with a leading axis over *fields*; the object axis is batched. Instances share weights because they are a batch dim. |
| causation symmetry | key-value attention with self masked out (`m[:, i, i] = False`). Query = object `i`'s local encoding; keys/values = every other object's global encoding. Attention is permutation-equivariant, so Assumption 4.1 is **architectural**, not learned — and it is also how varying object counts work. |
| output | per-field `DistributionDecoder` → `PType` → `log_prob` |

One thing the code answers differently than the paper's framing suggests: `EnvObjClass(name, base=...)`
**copies field declarations, it does not share weights**. Cross-class invariants are expressible in the
*graph* (the global matrix spans classes), but every class gets its own encoder and decoder parameters.

---

## 4. What this dissolves, failure by failure

- **§24 (capacity contention) is reduced, NOT dissolved.** `Link.pos'` and `En_Dekubaba.hp'` are separate
  heads with separate parents and separate loss terms in different units. **This was originally written as
  "dissolves structurally" and that was too strong** -- section 13 measured the same contention reappearing
  in the shared TRUNK: a gate loss weighted 200x collapsed every value head that reads it. Separate heads do
  not help when what they read is shared and one loss is weighted far harder than the others. The contention
  moved; it did not disappear.
- **§30 (identity is the global optimum) changes character.** There is no `z` to copy. Per-field "predict no
  change" is still a strong baseline — §6 quantifies exactly how strong — but it moves from *unavoidable
  capacity contention* to *a per-field loss weight*, which is a tunable.
- **b88868f (latent drift) is bounded.** Error accumulates in world units, not in an unbounded `|z|`.
- **§26/§28 (control echo) becomes impossible by construction.** The action is a `role='action'` field: an
  input to other fields, never a target. There is no `prev_action` column to leak. Had one existed, causal
  discovery would have *printed the edge* rather than producing a mysterious +0.742.
- **§25 (seed variance) does not go away**, but the graph adds a second, more stable measurement axis:
  edge-set stability across seeds is a sharper "did training learn anything" test than a skill number.

---

## 5. The OoT OOMDP schema (Stage 0, deliverable 1)

[`zelda_rl/oomdp.py`](../zelda_rl/oomdp.py). **6 classes, 85 fields (77 state, 8 action)** over the v3 state.

```sh
.venv/bin/python -m zelda_rl.oomdp          # partition assertion + the full field listing
```

| class | instances | source |
|---|---|---|
| `Link` | 1 | 30 state fields + **8 action fields** |
| `Camera` | 1 | `cam_yaw`, `cam_pitch`, `cam_eye` |
| `Room` | 1 | `scene`, `room` — leaving the room *is* `Room.scene` changing |
| `En_Dekubaba` | 0-3 | actor id `0x0055` |
| `En_St` | 0-3 | actor id `0x0037` (Skulltula) |
| `En_Sw` | 0-3 | actor id `0x0095` (Skullwalltula) |

Actor ids are read from `include/tables/actor_table.h` of the OoT decompilation, from each entry's own
`/* 0xNNNN */` comment — **not** by counting `DEFINE_ACTOR` lines, which is off by three because of the
header comment block. Cross-check: the table gives `0x0055 = En_Dekubaba`, matching `combat.DEKU_BABA`,
which was verified live and independently (AGENTS.md §3).

### Three things the schema changes about our state

1. **`targeted_row` and `arrow_row` are gone.** A row index is not permutation-invariant, so it violates the
   paper's causation symmetry outright. They are promoted to per-actor booleans `is_targeted` / `has_arrow`.
2. **The action is a role, not a feature** (AGENTS.md §5). `role='action'` fields are inputs only. Note this
   is about the ACTION, not the ACTION-STATE: `action_fn_rel`, `melee_state` and friends are *outcomes* and
   stay `role='state'` (AGENTS.md §6). They are exactly the fields removed from the encoder *input* as
   control echo (`bc.CONTROL_ECHO_FEATS`); as prediction *targets* they are wanted.
3. **Value types are declared, not assumed.** `hp` is `Categorical` (§3 of the latent trail measured
   HP-as-classification taking damage F1 0.286 → 0.458); angles are wrap-aware; bitfields are `Bits`;
   positions are `Real` predicted as deltas with zero-init heads, so an untrained model is *exactly*
   do-nothing and skill 0 is the floor. Categorical cardinalities are **measured** by `fit_cardinalities()`,
   never written down — a guessed enum size is a hand-computed number.

### Two assertions, not one

- `assert_partition()` — all 363 columns are claimed exactly once, derived from `zelda_rl.state_layout`.
- `assert_roundtrip()` — `decompose` → `recompose` returns the original state **exactly** (verified on 25
  episodes, and run on the first episode of every census). A column partition only proves the bookkeeping;
  this proves the decomposition is lossless.

---

## 6. Stage 0 census: the per-field do-nothing table

[`scripts/eval_field_identity.py`](../scripts/eval_field_identity.py). No training, minutes to run.

```sh
.venv/bin/python scripts/eval_field_identity.py --data datasets/transitions/v3-*/20260919-084052
```

Pooled over **all four v3 collections: 398 episodes, 203,424 transitions.** Pooling was necessary, not
tidiness — `Room.scene` is constant in v3-main alone and changes on 0.035% of pooled steps.

Five columns, and the point is that none means anything alone (AGENTS.md §2): `chg%`, `identity` (the
do-nothing floor), `id|chg` (the do-nothing error *restricted to steps where the field changed*), `marginal`
(best constant predictor), `id/pair` (identity error over the error between two random samples — §2's
move/rand generalised to every value type).

### 6.1 Only 49 of 77 state fields are live

**28 never change at all**; 5 of those vary between episodes but never within one — the free-signal class
AGENTS.md §1 names. Any metric that averages over fields, or over actor rows, is **36% composition**.

### 6.2 Two of the three actor classes are nearly inert

| class | constant fields | behaviour |
|---|---|---|
| `En_Dekubaba` | **1 / 14** | moves on 6.2% of steps, despawns, 13 action functions, 9 behaviour states |
| `En_St` | 9 / 14 | sways only: position changes on 26.9% of steps, by **0.32 world units** |
| `En_Sw` | **11 / 14** | never moves, never dies, one state |

Six of the nine actor rows are almost pure padding. **Any per-row averaged metric is mostly composition** —
a §12-shaped denominator problem sitting underneath every row metric reported in the latent trail.

### 6.3 `identity` and `id|chg` disagree by 4-10x

| field | chg% | identity | **id\|chg** | id/pair |
|---|---|---|---|---|
| `Link.pos` | 21.6 | 0.99 | **4.58** | 0.010 |
| `Link.yaw` | 10.1 | 3.17° | **31.5°** | 0.040 |
| `Camera.eye` | 26.4 | 1.68 | **6.35** | 0.015 |
| `En_Dekubaba.pos` | 6.2 | 0.20 | **3.21** | 0.002 |
| `En_Dekubaba.hp` | 0.057 | 0.0006 | **1.00** | 0.002 |

This is the content of "the identity is a very strong baseline", decomposed. Do-nothing wins on aggregate
only because 78-99% of steps are static. The jump a model must actually produce is 4.6 world units for Link
and 3.2 for a Baba. **Every Stage 1 number is reported as both columns.**

---

## 7. Three defects the census exposed

1. **`Link.y_dist_to_ledge` has `id|chg` = 399.96** — a sentinel, not a continuous quantity. It teleports,
   so `Real` is the wrong value type. It needs a Boolean `has_ledge` companion, the way `bc.py` already pairs
   `has_interact_wall` with `dist_interact_wall`.
2. **`En_Dekubaba.vel` is essentially always exactly 0** (marginal 0.0) while Baba position moves on 6.2% of
   steps. The actor moves without writing `vel` — matching the note in `zelda_rl/obs.py` that `+0x5C` is used
   by projectiles and that many actors move without it. Baba motion must come from `fn_rel` / `state_idx` /
   `timer`; any head fed Baba velocity is reading a dead field.
3. **`En_Dekubaba.hp` changes on 0.057% of object-steps.** Per-field heads remove the *capacity contention*,
   not the *class imbalance*. The §23 event oversampling is still required — but it can now be applied
   per-field instead of to a shared batch.

---

## 8. Controls, before Stage 1 earns compute (AGENTS.md §1)

| control | status | what it answers |
|---|---|---|
| per-field do-nothing | **done, §6** | the floor for each field, in its own units |
| untrained, same architecture | to run | with zero-init residual heads this should be *exactly* do-nothing — an assertion, not an experiment |
| free-signal ablation | **structural** | no `prev_action` column exists; episode constants are flagged in §6.1 |
| permutation invariance | to run | shuffle row order, predictions must be bit-identical. Attention guarantees it, so this is a unit test |
| shuffled-class control | to run | assign actors to classes at random. **If that does as well, class sharing contributed nothing.** The paper does not run this one |
| full `G₁` vs discovered `Ĝ` vs empty graph | Stage 2 | if the empty graph scores the same, parents do not matter |
| graph stability across ≥3 seeds | Stage 2 | edge-set Jaccard. If unstable, ε is deciding, not the data |

---

## 9. Staging, so four things do not change at once

Adopting OOCDM wholesale changes the latent, the recursion, the input structure *and* adds causal discovery
simultaneously; a win would not say which mattered.

- **Stage 0 — done.** Schema declaration, both assertions, per-field census.
- **Stage 1 — in progress.** Per-field, class-shared, attention-based predictor at the **full graph `G₁`**,
  no causal discovery. Isolates "object-oriented + per-field heads + state-space recursion" from "causality".
  Scored against §6 and against `scripts/eval_rollout_space.py`'s existing do-nothing control.
- **Stage 2.** Add masking + CMI discovery; test whether `Ĝ` beats `G₁`.

Expectation to be falsified either way: most of the win is in Stage 1, and Stage 2 buys generalisation to
varying actor counts plus an explainability artifact — which is the paper's own emphasis.

---

## 10. Risks, stated before they can be retracted

1. **It is a Markov model over declared fields.** If a Baba's lunge is triggered by an animation counter we
   do not read, the class transition function is forced to guess. `timer` / `fn_rel` / `state_idx` are the
   right instinct; more may need adding from the decompilation (AGENTS.md §3).
2. **10 Hz decisions = ~2 game-logic ticks per transition.** Causal parents at two-tick separation are wider
   than at one tick. It blurs the graph; it does not break it.
3. **Nothing in the paper promises h=1800.** Its evaluation is log-likelihood and short-horizon planning
   return. Our barometer is real-action replay (AGENTS.md §8). The Stage 1 target is beating do-nothing
   per-field at h=10-100 in world units — and §22's mistake, measuring the scaffolding, is not to be repeated.
4. **The paper's environments are tiny** (position as an int in [-99, 99], health as an int). 43 messy Link
   fields is well outside what it tested.
5. **This census is one task** (Deku Baba combat), one collection date. Which fields are live is
   task-specific; the method is not.


---

## 11. Stage 1 result: 6 of 49 fields beat do-nothing, and the model invents motion

`scripts/train_oocdm.py`, full graph G1, 6 epochs, all four v3 collections (398 episodes, 203,424
transitions), slot split train {1,2,3,5,6,10} / tune {4,8} / test {7,9}. 4.66M params, 77 field heads.
wandb `zelda-rl-oocdm/xq6yb719`, checkpoint `checkpoints/oocdm/stage1.pt`, per-field table
`runs/oocdm-stage1-g1/fields.json`.

### The controls, which are the reason the rest is readable

| control | result |
|---|---|
| identity init | untrained model is **exactly** do-nothing on train, tune and test |
| permutation | swapping two Deku Babas swaps their predictions and moves nothing else: **0.00e+00** |
| untrained test skill | worst \|skill\| **2.7e-03**, on `En_Dekubaba.yaw` |

The identity-init control paid for itself immediately: it caught an untrained model scoring **skill -33 on
`Link.hp`**, because codecs are fitted on TRAIN ONLY and a categorical value first seen at test time matched
no identity slot, so the head fell back to class 0. Invisible on the train split. The UNKNOWN slot now
carries the identity.

The third control is also a worked example of AGENTS.md section 12. `En_Dekubaba.yaw` has the smallest
identity error in the table (0.0034 deg), so a float round-trip of ~1e-5 deg shows up as a relative skill of
2.7e-03. The number moved because the denominator shrank.

### Headline: 6 / 49

| outcome | count |
|---|---|
| beats do-nothing on **both** columns | **6** — `Link.grass`, `Link.melee_anim`, `Link.sf3`, `Room.scene`, `En_Dekubaba.fn_rel`, `En_Dekubaba.state_idx` |
| beats it on changed steps only, pays on static steps | 30 |
| beats it on neither | 13 |
| **learned nothing at all** (skill\|chg exactly 0.000) | **7** |

Best on changed steps: `En_St.pos` +0.718, `Link.melee_state` +0.558, `Link.fall_distance` +0.554,
`Link.sf1` +0.473, `Link.sf3` +0.459, `Link.bg_flags` +0.454, `Link.speed` +0.421,
`Link.ztarget_timer` +0.404, **`Link.pos` +0.268** (5.41 -> 3.96 world units).

`En_St.pos` topping the table is not a win. Section 6.2 measured that class as swaying 0.32 world units; a
near-periodic 0.3-unit sway is the easiest thing in the dataset.

### The model invents motion, and that is not a denominator artifact

`Link.pos` is +0.268 on changed steps and **-2.617 in aggregate**. During the run this was called a
small-denominator artifact of the kind AGENTS.md section 12 warns about. **That reading was wrong for this
field** and the decomposition shows why -- implied error on the steps where the field did NOT change, where
do-nothing scores exactly 0 by construction:

| field | chg% | error on UNCHANGED steps | gain on changed steps | net |
|---|---|---|---|---|
| `Link.ztarget_timer` | 2.3 | **1.50** | +1.58 | -1.43 |
| **`Link.pos`** | 8.3 | **1.41 world units** | +1.45 | **-1.18** |
| `Camera.eye` | 10.8 | 0.92 | -0.12 | -0.83 |
| `Link.speed` | 3.2 | 0.23 | +1.33 | -0.18 |

The model emits ~1.4 world units of motion per step while Link is standing still. At 10 Hz that is metres of
drift in seconds, and it is section 22's failure in new clothes. The artifact reading IS correct for the
genuinely rare fields (`En_Dekubaba.hp` at 0.02% change), where the aggregate ratio is unstable -- but it
cannot be used to wave away `Link.pos` at 8.3%.

### The seven dead heads are the reward function

skill\|chg **exactly 0.000**: `Link.hp`, `Link.held_item`, `En_Dekubaba.hp`, `En_Dekubaba.present`,
`En_Dekubaba.has_arrow`, `En_St.present`, `En_Sw.present`.

Every one is a rare discrete event -- damage, death, despawn, pickup -- and between them they are most of
what `CombatTracker.step()` reads to compute reward. Section 6 predicted this in as many words: per-field
heads remove the capacity contention, not the class imbalance.

### Verdict

**Stage 1 does not beat do-nothing as a world model, and no rollout has been run.** Two caveats in both
directions: the run is undertrained (loss 25.3 -> 9.2, still falling steeply at epoch 6) and single-seed
(AGENTS.md section 7), and this is a SINGLE-STEP metric while the barometer is a ground-truth rollout
(AGENTS.md section 8). These numbers are also not comparable to the latent trail's skill figures, which were
computed in latent space rather than per field in world units.

What the object-oriented factorisation did buy, and it is not nothing: every failure above is **localised to
a named field with a named cause**, instead of one aggregate number that three hypotheses can explain.

### Diagnosis: the heads have no change gate

They regress a delta directly, so a small delta is smeared across every frame. Section 29 of the latent trail
recorded that `TransitionHead`'s `moved` logit worked well (AP 0.435, 6.8x base) -- that gate is exactly what
is missing, and the census `chg%` column is what says so. A per-field "did this change at all" head with the
value conditioned on it addresses the spurious motion AND the seven dead heads at once, because a gate turns
a regression problem into the imbalanced-classification problem this project already knows how to fix
(section 3 of the latent trail: damage F1 0.286 -> 0.458 -> 0.746 -> 0.906).

---

## 12. Open work

Ordered by (measured defect) / (cost), not by appeal. Every item names the evidence that motivates it, so
none can be argued for or against from taste.

### 12.1 Per-field change gate -- RESOLVED, see section 13

**Evidence:** section 11. 1.41 world units of spurious `Link.pos` motion on static steps; 7 heads at exactly
0.000; 30 of 49 fields win on changed steps but pay on static ones.
**Change:** one `changed` logit per field, value loss restricted to changed steps, `pos_weight` per field
from the measured change rate, threshold calibrated on tune. Gate bias initialised negative so the identity
init invariant survives.
**Control:** `--no-gate` keeps the current arm; the 6-epoch run above is the matched baseline, same seed and
config (AGENTS.md section 11 corollary -- train both arms, compare at matched epochs).

### 12.2 Angle loss design -- binning FALSIFIED, see 13.6

**Evidence:** nine of twelve angle heads sat between -0.005 and +0.001 for four straight epochs.
`Link.yaw` was -0.0000 through epoch 4 and reached only +0.012 on test; `En_Dekubaba.shape_yaw` -0.000;
`Link.wall_yaw`, `Link.parallel_yaw`, `Link.floor_pitch` all ~0. The two angles that DID move,
`En_St.yaw_to_player` and `En_Sw.yaw_to_player`, are the two whose delta is nonzero on ~18% of steps.
**Cause:** the loss is `1 - cos(pred - true)` on a delta that is exactly 0 on ~90% of steps. At Delta=0 the
loss is already at its minimum, so there is no gradient. This is a loss-design defect, NOT a capacity limit,
and it is the per-field reappearance of the section 30 pathology.
**Candidates:** the change gate (12.1) may fix it outright by separating "did it turn" from "by how much";
otherwise a von Mises NLL with learned concentration, or a binned sector head (the transition head already
uses 16 sectors and reached 0.532 direction accuracy).

### 12.3 Rare-event supervision, per field

**Evidence:** the 7 dead heads. `En_Dekubaba.hp` changes on 0.057% of object-steps and scored exactly 0.000
at every epoch.
**Change:** section 23's event oversampling, applied PER FIELD rather than to a shared batch -- which is only
possible now that each field has its own loss term. Combine with the gate's `pos_weight`.
**Why it matters more than its size suggests:** these fields are what the reward reads. A dream that never
produces damage cannot train a combat policy, which is section 22's finding restated.

### 12.4 Static geometry as a class (Wall / floor / ceiling)

**Evidence:** `Link.dist_interact_wall` is a derived scalar with a sentinel (`id|chg` 399.96 in the Stage 0
census), scored -1.783 on tune by epoch 5, and section 29 found its continuous head unlearned. More
importantly, section 30 blamed a bimodality -- "Link keeps walking vs Link hits the wall and stops" -- for
the failure of continuous latents and treated it as irreducible. **It is not irreducible, it is unobserved:**
the branch is decided by geometry that is nowhere in the state.
**The paper's precedent:** static entities are declared as ordinary classes with no action fields -- Trap and
Food in Mouse, Mineral in StarCraft CMS -- so the same class-level attention handles them and the model can
learn the global causality `Wall.pos -> Link.pos'`.
**Verified feasible, `scripts/dump_collision.py`:** scene 0 has 1399 vertices and 2321 polygons, split by the
ENGINE's own rule (`z_bgcheck.c:557`, normal.y > 0.5 floor, < -0.8 ceiling, else wall) into 443 floor /
1584 wall / 294 ceiling. Only **16 walls lie within 300 units of Link**, so nearest-K with K ~ 8-16 is an
object set, not a mesh.
**And it needs no re-collection:** all 398 episodes are scene 0, room 0, so static collision is constant.
Read the geometry once from a savestate and compute nearest-K offline from each recorded Link position.
**Design:** `Wall` class, no action fields, fields `pos` (centroid) and `normal`; its own transition is the
identity, exactly like `Mineral.Position`, so its entire value is as attention keys for `Link.pos'`.
**Two traps to avoid:** every `Wall` field registers as CONSTANT and must be excluded from aggregates
explicitly (section 6.1 already measures the table as 36% composition); and nearest-K re-sorts every step, so
"wall 3" is not the same polygon across time -- attention does not care, but nothing downstream may treat the
index as an identity. That is the same trap `targeted_row` was.

### 12.5 Multi-step: the barometer has not been run

**Evidence:** AGENTS.md section 8. Everything in section 11 is single-step. Section 22 is the record of what
happens when single-step metrics are trusted: every number improved while the dream produced zero damage and
zero kills on real action sequences.
**Change:** replay real action sequences from real start states through the OOCDM, re-encoding the state each
step, and score per field at h = 10 / 40 / 100 against the do-nothing control in world units --
`scripts/eval_rollout_space.py` already has the harness and the control.
**Until this runs, nothing here is a claim about a world model.**

### 12.6 Stage 2: causal discovery

Masking plus CMI, and the three controls that decide whether it did anything: discovered `G-hat` vs full `G1`
vs the EMPTY graph; a **shuffled-class control** (assign actors to classes at random -- if that scores the
same, class sharing contributed nothing, and the paper does not run this one); and edge-set Jaccard across
>= 3 seeds, because if the graph is unstable then epsilon is deciding rather than the data. Stage 2 also
needs the paper's per-target-field `f_global` restored (see the `OOCDM` docstring).

### 12.7 Smaller, recorded so they are not rediscovered

- **`Link.y_dist_to_ledge` is a sentinel, not a continuous quantity** (`id|chg` 399.96). Needs a Boolean
  `has_ledge` companion, as `bc.py` already pairs `has_interact_wall` with `dist_interact_wall` -- or
  deleting, if 12.4 makes it redundant.
- **`En_Dekubaba.vel` is a dead field.** Always exactly 0 while the Baba moves on 6.2% of steps; the actor
  moves without writing `vel`. Any head fed it is reading nothing.
- **The collector filters to actor category 5.** Savestate `s03` contains 9 actor ids
  (`0xa, 0xf, 0x15, 0x23, 0x2e, 0x55, 0x95, 0x11b, 0x125`) but the recorded 363-d state has only the three
  enemies. `0x15` is `En_Item00` (collectibles). More classes are available than are being recorded.
- **Training is CPU-bound in batch construction**, not on the GPU: `make_batch` re-runs the numpy codecs over
  every field on every batch (~70% CPU with MPS as the device, 170s/epoch). Pre-encode once.
- **Single seed.** AGENTS.md section 7. Section 25 measured seed spread larger than most effects in the
  latent trail; nothing in section 11 has been repeated.


---

## 13. The change gate: three reversals, and what the 3-minute loop was worth

Section 12.1's gate took four experiments to get right, and the first three conclusions were all wrong. The
method failure is worth more than the result: the first attempt shipped the gate, its `pos_weight` and a
changed-steps-only value loss in ONE commit and then spent 30 minutes per iteration unable to attribute the
outcome to any of them. That is AGENTS.md section 9 exactly.

Two of the three never needed a training run at all. **The firing threshold is applied at evaluation time**,
so one trained model yields every threshold; and **gate quality is a property of a classifier**, measured with
AP. `scripts/probe_gate.py` trains once on a random 120-episode subset and reports the three questions
separately in **186 seconds**.

### 13.1 Two things the probe had to fix about itself first

- **The subset was not representative.** Taking the first 60 episodes of `v3-main` gave `Link.pos` a 0.629%
  change rate against 8.3% on the real test split: `v3-main` is combat, where Link often stands still, while
  `v3-approach` and `v3-moves` are where he walks. A probe whose target event is 13x rarer is measuring a
  different problem. It now samples at random across all four collections and PRINTS its realised change
  rates against the census before anything else in the table can be believed.
- **A constant cannot stand in for a missing gate.** The ungated arm filled gate logits with a constant, and
  AP over constant scores ranks by array order, which is temporal, and changes cluster in time -- producing a
  bogus "40/47 gate informative". Now filled with NaN, which reads as NaN.

### 13.2 The measurements, all at matched epochs on one subset

| arm | value heads beating do-nothing given a change | `Link.pos` val\|chg | mag ratio | gate AP |
|---|---|---|---|---|
| no gate | **28/47** | **+0.066** | 0.418 | -- |
| gate, value on changed steps only | 16/47 | -0.106 | 0.238 | 0.834 |
| gate, value on all steps | **14/47** | -0.051 | **0.119** | 0.870 |
| **gate detached from the trunk** | **28/47** | **+0.064** | **0.410** | 0.648 |

**Reversal 1.** The gate was blamed for finding a degenerate never-fire optimum. Wrong: the gate is an
excellent classifier -- 47/47 fields above 2x base rate, `Link.pos` AP 0.839 against section 29's `moved` AP
of 0.435 for the same question -- and the calibrator was correctly declining to fire value heads that made
things worse. `bestThr = never` tracked `val|chg < 0` almost exactly.

**Reversal 2.** Restricting the value loss to changed steps was then blamed, and it IS harmful (28 -> 16).
But holding that fixed and merely ADDING the gate loss gave 14/47, worse than either. So the restriction was
not the mechanism.

**Reversal 3, and the actual cause.** The gate's BCE carries `pos_weight` up to 200x and backpropagates
through the SHARED trunk. It repurposes the representation the value heads read: predicted magnitude collapses
to 0.119 of the truth. Detaching -- the gate READS the trunk, never shapes it -- recovers the value head
completely (28/47, `Link.pos` +0.064 vs the ungated +0.066) for a gate AP cost of 0.870 -> 0.648. **And for
the first time the calibrator fires on `Link.pos` and reduces error, 0.4773 -> 0.4448.**

### 13.3 Settled design

Gate head on a `detach()`ed trunk; value head trained on ALL steps; threshold calibrated per field on tune and
applied unchanged to test. `--gate-attached` and `--value-on-changed` keep both measured-harmful variants
runnable so they can be re-tested rather than argued about.

### 13.4 Two bugs this found in committed code

- `field_losses` and `predict_raw` decided whether to unwrap `forward`'s output by testing for the `"gate"`
  key rather than `"value"`, so with the gate off they passed the whole wrapper through as the values dict.
  **`--no-gate` -- the matched baseline arm named in its own commit message -- was never runnable.** A flag
  that exists to be the control must be executed when it is written.
- The delta scale `dstd` was measured over all steps while the value head trained on changed steps only,
  where deltas are 3.4x-50x larger (`Link.pos` 2.70 -> 9.11, `y_dist_to_ledge` 7.9 -> 400). Targets reached
  |152| under a Gaussian NLL. The first fix fell back per episode, which silently reinstated the bug for
  exactly the rare fields that needed it, since most episodes contain no changed step at all.

### 13.5 Full run: the gate is a floor-enforcer, not a fix

Settled config (detached gate, value head on all steps, thresholds calibrated on tune), 6 epochs, all four
v3 collections, same seed and split as the ungated baseline. wandb `62m7ob2n`, checkpoint
`checkpoints/oocdm/stage1_gate_detach.pt`.

| TEST slots {7,9}, 6 epochs | ungated baseline | gate detached |
|---|---|---|
| `Link.pos` error on steps where Link did NOT move | **1.4129** | **0.0000** |
| `Link.ztarget_timer` | 1.5031 | 0.0000 |
| `Camera.eye` | 0.9160 | 0.0000 |
| fields WORSE than do-nothing in aggregate | 36 | **2** |
| **fields beating do-nothing in aggregate** | **6** | **6** |
| fields beating do-nothing on changed steps | 36 | 8 |
| heads that learned nothing (skill\|chg == 0) | 7 | **41** |
| `Link.pos` (aggregate / changed) | -2.617 / +0.268 | +0.000 / +0.000 |

**The spurious motion is completely gone** -- 1.41 world units per static step down to exactly zero, which
was the concrete failure section 11 named -- and fields actively worse than do-nothing fell from 36 to 2.

**And it bought no capability whatsoever.** Six fields beat do-nothing in aggregate in BOTH arms. The gate
converted "worse than do-nothing on 36 fields" into "identical to do-nothing on 41 fields": it removed losses
without adding a single win, and paid `Link.pos`'s +0.268 on changed steps to do it. One field got clearly
worse, `En_Dekubaba.state_idx` at +0.150 -> **-1.879** aggregate; it fires and pays.

So the gate is an honest reporter of a bad value head, not a fix for one. It is still worth keeping for 12.5,
the multi-step barometer -- a dream that does nothing beats one drifting 1.41 units per step, which is
precisely how section 22 failed -- but it is not progress on prediction.

**12.3 is a null result as well.** All seven dead heads are still exactly 0.000; `pos_weight` at 200x did not
open `En_Dekubaba.hp`. The dead count went 7 -> 41.

One methodological note, recorded because it was got wrong in-flight: at epoch 3 the trend (value loss
falling while gates-off climbed 12 -> 17 -> 24) was called a monotonic convergence to do-nothing. Epoch 4
broke it in both columns (11 beating, 20 off) and the counts then oscillated 10, 10, 6, 11, 8, 8. That was a
3-point trend read as a law. The mean-regression argument stands on the probe's DIRECT magnitude
measurements (0.41 and 0.09 of truth), not on this curve.

### 13.6 Binned direction + magnitude: falsified, and informatively

Section 12.2's premise was section 29's: a regressed delta is minimised by the conditional mean and shrinks,
so bin it, because "a mode does not shrink". The oracle ceiling was measured FIRST (bin the TRUE delta and
reconstruct): `Link.pos` **0.692**, `En_Dekubaba.pos` 0.739, `Camera.eye` 0.666 -- quantisation is not the
limit, so the experiment was worth running.

| `Link.pos`, matched subset and epochs | continuous | binned |
|---|---|---|
| magnitude ratio | 0.410 | **1.158** |
| direction error | 105.2 deg | **78.8 deg** |
| **val\|chg** | **+0.064** | **-0.193** |

**Binning did exactly what it was designed to do and made the field worse.** Shrinkage is fixed (0.41 ->
1.16) and direction improved below 90 deg for the first time -- and full magnitude in a direction still 79
deg off is worse than a small magnitude in the wrong direction. Shrinkage was partly protecting us.

**Magnitude was never the binding constraint; direction is.** The section 29 analogy does not hold, because
there the direction head was separately trained and working.

### 13.7 The control that should have run first: pos + k*vel beats the model

AGENTS.md section 1 demands a free-signal control and section 28 names this exact one -- "with velocity in z,
next position is linear extrapolation". Stage 1's control table has identity, untrained, permutation and
shuffled-class, but **no trivial-baseline control for the value head**, so four architecture experiments were
scored against a floor that was too low.

Measured on all four v3 collections, changed steps, XZ plane:

| predictor | MAE | val\|chg |
|---|---|---|
| do-nothing | 6.533 | +0.000 |
| `vel[t] * 1.0` | 5.341 | +0.183 |
| **`vel[t] * 1.67` (least squares)** | **4.855** | **+0.257** |
| off-wall | 5.160 | +0.294 |
| touching a wall | 3.839 | +0.028 |

**A zero-parameter linear extrapolation scores +0.257. The trained 4.7M-parameter value head scores +0.064**,
with 79-105 deg direction error against velocity's **13.5 deg**, using an input it already has.
`zelda_rl.oocdm.velocity_baseline` now reports this on every probe run.

The first version of this control was scored over all three components and read **-0.057**, i.e. it said the
baseline LOSES. `vel_y` has a median of -4.0 on 100% of steps -- gravity pulling against the ground -- while
`pos_y` barely moves, so the naive predictor injected a spurious vertical component. The control was flawed,
not the metric. A broken control that exonerates the model is the most expensive kind.

**This demotes both candidate fixes.** Binning is falsified above. Static geometry (12.4) is real --
velocity extrapolation drops to +0.028 when touching a wall, against +0.294 off-wall -- but off-wall
performance is where the model is losing, so collisions are not the blocker either.

**The question for the next experiment is narrower and stranger than any of them: why can a 4.7M-parameter
model with `vel` as an input not match a linear function of `vel`?** Until that is answered, no further
architecture is worth training.

### 13.8 Still open

Direction remains the weak axis: `Link.pos` sits at ~105 deg median error with 24.8% of predictions within
45 deg, and magnitude at 0.41 of the truth. Section 29 hit the same wall and its fix was binning -- "direction
and magnitude are binned, and a mode does not shrink" -- after regression to the mean made displacement 9.3x
too small. That is 12.2's next experiment, and the median is the wrong summary for it: a median direction
error above 90 deg coexists with a POSITIVE `val|chg`, because the model gets direction right on the large
displacements that dominate the error. It needs magnitude-weighting.

Ground truth for that work, measured on the raw data with no model involved: `Link.vel[t]` sits **11.3 deg**
from the next displacement in the XZ plane (62% within 30 deg), so the directional signal is present and
already an input. The commanded stick direction sits at **66.2 deg** with only 31% within 30 deg -- a single
10 Hz step is dominated by momentum and collision, not by what was pressed, which is consistent with section
16's direction accuracy stalling at 0.532 over 16 sectors.

---

## 14. What actually works, field by field

The only arm that is PAPER-SHAPED -- per-field heads, class sharing, attention, full graph G1, no
gate, no binning, no oversampling -- is the 6-epoch run `runs/oocdm-stage1-g1` (wandb `xq6yb719`).
**That is the number of record.** The gated runs are not OOCDM and are excluded here.

TEST slots {7,9}, 46,926 held-out transitions. `AGG` = `1 - model_MAE/identity_MAE` over all scored
(object, step) pairs, which is what a rollout accumulates. `CHG` = the same restricted to steps where
the field changed, which is what copying cannot win. PASS requires BOTH.

| entity.field | chg% | AGG | CHG | verdict |
|---|---:|---:|---:|---|
| `Link.melee_state` | 1.00 | -4.205 | +0.558 | PARTIAL (CHG only) |
| `Link.fall_distance` | 1.55 | -1.220 | +0.554 | PARTIAL (CHG only) |
| `Link.sf1` | 2.04 | -21.612 | +0.473 | PARTIAL (CHG only) |
| `Link.sf3` | 0.97 | +0.413 | +0.459 | **PASS** |
| `Link.bg_flags` | 0.84 | -0.764 | +0.454 | PARTIAL (CHG only) |
| `Link.speed` | 3.15 | -1.850 | +0.421 | PARTIAL (CHG only) |
| `Link.grass` | 0.03 | +0.324 | +0.407 | **PASS** |
| `Link.ztarget_timer` | 2.32 | -15.713 | +0.404 | PARTIAL (CHG only) |
| `Link.floor_height` | 0.74 | -1.085 | +0.384 | PARTIAL (CHG only) |
| `Link.pos` | 8.30 | -2.617 | +0.268 | PARTIAL (CHG only) |
| `Link.melee_anim` | 0.43 | +0.069 | +0.261 | **PASS** |
| `Link.sf2` | 3.72 | -0.361 | +0.214 | PARTIAL (CHG only) |
| `Link.y_dist_to_ledge` | 0.08 | -0.586 | +0.189 | PARTIAL (CHG only) |
| `Link.shape_yaw` | 3.62 | -0.365 | +0.155 | PARTIAL (CHG only) |
| `Link.vel` | 5.98 | -3.194 | +0.115 | PARTIAL (CHG only) |
| `Link.action_fn_rel` | 2.61 | -22.236 | +0.088 | PARTIAL (CHG only) |
| `Link.yaw` | 4.03 | -0.216 | +0.012 | PARTIAL (CHG only) |
| `Link.parallel_yaw` | 0.25 | -0.403 | +0.003 | PARTIAL (CHG only) |
| `Link.wall_yaw` | 0.12 | -0.228 | +0.000 | PARTIAL (CHG only) |
| `Link.nuts` | 0.00 | -0.039 | +0.000 | PARTIAL (CHG only) |
| `Link.hp` | 0.18 | +0.000 | +0.000 | FAIL |
| `Link.held_item` | 0.04 | +0.000 | +0.000 | FAIL |
| `Link.floor_pitch` | 0.42 | -0.965 | -0.001 | FAIL |
| `Link.dist_interact_wall` | 0.56 | -0.528 | -0.007 | FAIL |
| `Camera.yaw` | 6.73 | -0.053 | +0.013 | PARTIAL (CHG only) |
| `Camera.pitch` | 7.82 | -0.171 | -0.003 | FAIL |
| `Camera.eye` | 10.78 | -1.302 | -0.020 | FAIL |
| `Room.scene` | 0.03 | +0.333 | +0.333 | **PASS** |
| `En_Dekubaba.vel` | 0.02 | -1.061 | +0.342 | PARTIAL (CHG only) |
| `En_Dekubaba.state_idx` | 0.43 | +0.150 | +0.150 | **PASS** |
| `En_Dekubaba.timer` | 2.72 | -4.806 | +0.115 | PARTIAL (CHG only) |
| `En_Dekubaba.pos` | 2.50 | -0.976 | +0.109 | PARTIAL (CHG only) |
| `En_Dekubaba.focus` | 2.50 | -5.733 | +0.100 | PARTIAL (CHG only) |
| `En_Dekubaba.fn_rel` | 0.45 | +0.090 | +0.090 | **PASS** |
| `En_Dekubaba.is_targeted` | 0.11 | -0.179 | +0.055 | PARTIAL (CHG only) |
| `En_Dekubaba.yaw_to_player` | 5.76 | -0.179 | +0.001 | PARTIAL (CHG only) |
| `En_Dekubaba.yaw` | 0.00 | -0.570 | +0.000 | PARTIAL (CHG only) |
| `En_Dekubaba.present` | 0.02 | +0.000 | +0.000 | FAIL |
| `En_Dekubaba.hp` | 0.02 | +0.000 | +0.000 | FAIL |
| `En_Dekubaba.has_arrow` | 0.08 | +0.000 | +0.000 | FAIL |
| `En_Dekubaba.shape_yaw` | 1.13 | -0.124 | -0.000 | FAIL |
| `En_St.pos` | 10.43 | -4.380 | +0.718 | PARTIAL (CHG only) |
| `En_St.focus` | 10.43 | -4.129 | +0.709 | PARTIAL (CHG only) |
| `En_St.yaw_to_player` | 7.26 | -1.270 | +0.156 | PARTIAL (CHG only) |
| `En_St.vel` | 10.43 | -3.855 | +0.070 | PARTIAL (CHG only) |
| `En_St.present` | 0.03 | +0.000 | +0.000 | FAIL |
| `En_Sw.yaw_to_player` | 7.47 | -0.510 | +0.128 | PARTIAL (CHG only) |
| `En_Sw.present` | 0.03 | +0.000 | +0.000 | FAIL |
| `En_Sw.focus` | 0.01 | -1.095 | -0.000 | FAIL |

**6 PASS · 30 PARTIAL · 13 FAIL, of 49 live fields.** 7 heads learned nothing at all (CHG exactly 0.000). 28 more fields never change in this data and are not scored.

### 14.1 By entity

| class | instances | live fields | PASS | what it contributes |
|---|---:|---:|---:|---|
| `Link` | 1 | 24 | 3 | the agent. Nothing in locomotion or combat passes. |
| `Camera` | 1 | 3 | 0 | driven by game logic we do not observe; all three fields fail or are CHG-only. |
| `Room` | 1 | 1 | 1 | one rare binary event (the exit). Passes, but it is a terminal, not dynamics. |
| `En_Dekubaba` | 3 | 13 | 2 | the combat target. `hp` -- what damage is made of -- learned nothing. |
| `En_St` | 3 | 5 | 0 | Skulltula. Sways vertically 0.32 units; `pos`/`focus` are CHG-only at high skill because the motion is near-periodic. |
| `En_Sw` | 3 | 3 | 0 | Skullwalltula. Never moves. Nothing to learn. |

### 14.2 Schema: what the OOMDP itself got right and wrong

| aspect | verdict |
|---|---|
| 363 columns partition exactly into (class, field) | **holds**, asserted every run |
| `decompose` -> `recompose` is lossless | **holds**, asserted every run |
| row indices removed (`targeted_row` -> per-actor `is_targeted`) | **holds**; permutation control exact over 77 heads |
| varying instance count (1-3 Babas, 5 distinct slot maps) | **holds** |
| actor ids resolved from the decompilation | **holds** (`0x0055` cross-checks `combat.DEKU_BABA`) |
| class granularity (one class per actor id) | **untested** -- the shuffled-class control was never run |
| `En_Dekubaba.vel` declared a state field | **wrong**: always exactly 0 while the Baba moves. Dead input |
| `Link.y_dist_to_ledge` typed `Real` | **wrong**: a sentinel (399.96), not a continuous quantity |
| `Link.dist_interact_wall` typed `Real` | **wrong**: FAIL in every arm; a derived proxy for geometry we do not model |
| 28 of 77 fields never change | correct but means aggregate metrics are 36% composition |

### 14.3 Causal relationship gaps -- the largest gap in the project

**Zero causal edges have been tested.** Everything trained at the full graph `G1`, where every field is a
parent of every field. The paper's contribution is not in the model, it is in the discovery, and none of it
has been run:

| paper component | file in `../oocdm` | our status |
|---|---|---|
| random-mask training term (`p_edge`) | `mask_generator.py`, `train_ooc.py` | **missing** |
| current-graph training term | `GraphMaskGenerator` | **missing** |
| full-graph training term | `FullMaskGenerator` | we train on this ALONE |
| 3-term objective `-(ll_graph + ll_full + ll_random)/3` | `train_ooc.py:90` | **missing** |
| CMI per class-level edge | `get_causal_graph()` | **missing** |
| threshold `causal_threshold=0.1`, `p_edge=0.9` | `train_ooc.py` defaults | **never used** |
| graph-stability stopping (`n_iter_stable`) | `train_ooc.py` | **missing** |

So of **8,286 candidate class-level edges** (1,741 local + 6,545 global) we have tested **0**, and we cannot
answer a single question the paper exists to answer: which fields actually cause which, whether the
discovered graph beats the full graph, or whether class sharing contributes anything at all.

### 14.4 Controls still owed on Stage 1

| control | status |
|---|---|
| identity init is exactly do-nothing | done, asserted every run |
| permutation invariance | done (was silently vacuous for one commit; now asserts head count) |
| untrained skill ~ 0 | done |
| **trivial baseline `pos + k*vel`** | run LAST instead of first. **+0.257 vs the model's +0.064** |
| **shuffled-class** (does class sharing do anything?) | **NEVER RUN** -- tests the paper's premise |
| **seed variance** (AGENTS.md section 7) | **NEVER RUN** -- everything is single-seed |
| **multi-step rollout** (AGENTS.md section 8, the barometer) | **NEVER RUN** |


---

## 15. The barometer, run at last: the dream travels 4% of the real distance

`scripts/eval_oocdm_rollout.py`, full-state autoregressive rollout -- all 77 fields predicted each step and
fed back through `decompose -> model -> recompose` -- on 26 held-out episodes from test slots, real action
sequences from real start states. Checkpoint `runs/paper-fast-s0/model.pt` (paper-shape, 120-episode random
subset, 2 epochs, seed 0).

| h | do-nothing | OOCDM | ratio | travelled dream / real |
|---|---|---|---|---|
| 1 | 23.52 | 23.62 | 1.00 | **0.2 / 25.3** |
| 5 | 42.16 | 42.33 | 1.00 | 1.7 / 52.1 |
| 10 | 73.04 | 73.08 | 1.00 | 3.5 / 93.3 |
| 25 | 75.49 | 76.48 | 1.01 | **7.3 / 173.6** |

**The model is inert.** It travels 0.8% of the true distance at h=1 and 4% at h=25, so its error equals
do-nothing because it IS do-nothing. This is AGENTS.md section 2 earning its place: the error column alone
reads "1.00x, tied", and only the travel column shows there is no model there. Section 22 found the same
shape in the latent world model ("13x too slow early"); the object-oriented factorisation has not changed it.

Two notes on the controls.

**The velocity control is vacuous in a rollout and was removed from the table.** Episodes start from
savestates where Link is stationary, so `vel_0` is zero and `pos + k*vel_0` is identical to do-nothing at
every horizon. It remains a strong control for SINGLE-STEP prediction (+0.257, section 13.7), where the
velocity is read from the real state each step. It cannot be carried into a rollout unchanged.

**This is a 2-epoch model on a 120-episode subset**, chosen for a 4-minute loop rather than a 25-minute one.
The 6-epoch full-data arm (section 14) scored `Link.pos` aggregate -2.617, i.e. it moved too MUCH rather
than too little, so the inertness here may be undertraining. That is a testable difference and it is not yet
tested -- but a model that must be trained to the point of over-moving before it moves at all is not
obviously on a path to a usable dream either.


---

## 16. Why nothing works: we trained at G1, and G1 is the problem

The question that produced this was "did you frame it so the Link object can affect itself?". The wiring was
checked in the trained weights and it is correct -- Link's 38 own input fields (including `vel`, `stick` and
the buttons) reach Link's 30 own targets through `f_local`, which is the paper's LOCAL causality
`C.U -> V'`, and attention masks self out, which is the paper's GLOBAL causality `j != i`. So the
object-oriented framing is right.

The experiment that followed is the most important measurement in this document. RIDGE REGRESSION -- closed
form, no capacity limit, no optimisation difficulty -- predicting Link's XZ displacement on changed steps,
from nested subsets of exactly what the model receives:

| ridge feature set | dims | val\|chg |
|---|---|---|
| `vel` only | 3 | **+0.219** |
| `vel + speed` | 4 | +0.214 |
| `vel + speed + stick` | 6 | +0.210 |
| `+ yaw + bg_flags` | 18 | +0.191 |
| **ALL 38 Link fields -- what the model actually gets** | 178 | **-0.066** |
| `pos + 1.70*vel`, closed form | 3 | +0.247 |

**Monotonic degradation as irrelevant fields are added.** A linear model loses the entire signal purely by
being handed 175 extra input dimensions. This is not architecture, not capacity, not epochs, and not data
volume -- it happens to a closed-form fit on the same data.

**So the paper's masking is not an optimisation, it is the mechanism.** `Link.pos'` needs roughly
`{pos, vel, speed, stick, bg_flags}`; causal discovery exists to find that set and mask the other ~8,280
candidate edges away. Every result in sections 11-15 was produced by a model reading 8,286 potential parents
for every head:

- the inert dream (section 15, 4% of real travel),
- the 41 heads that learned nothing (section 13.5),
- the value head losing to `pos + k*vel` (section 13.7),
- `Link.pos` at +0.064 against a zero-parameter baseline's +0.257.

All are consistent with one cause, and it is the one the paper removes.

**This also explains why the head-tuning of section 13 was doomed.** A gate, oversampling, binning and
calibration all operate on the OUTPUT of a trunk that is reading 175 irrelevant dimensions. None of them can
recover a signal that has already been drowned on the input side.

**Consequence for the plan.** Stage 2 is not the next phase after Stage 1 is polished; Stage 1 cannot be
made to work without it. The missing pieces are listed in 14.3 and are all in `../oocdm/scripts_/train_ooc.py`:
the random-mask term (`p_edge=0.9`), the current-graph term, the full-graph term, the 3-term objective
`-(ll_graph + ll_full + ll_random)/3`, and the CMI edge test at `causal_threshold=0.1`.

**Cheap prediction to falsify this first.** Before implementing discovery, hand-specify the parent set for
`Link.pos'` from the ridge table (`{pos, vel, speed, stick, bg_flags}`), train the existing model with every
other edge masked, and check whether `Link.pos` reaches ~+0.22. If it does not, masking is not sufficient
and discovery will not save us either. That is one 4-minute run and it gates the whole of Stage 2.


---

## 17. Plan of record: port the paper's discovery loop (P0 - P3)

This section supersedes section 12 as the plan. Section 12 was a backlog of ideas, several of which were
deviations from the paper and are now removed (AGENTS.md section 13). **Nothing outside P0-P3 is in scope.**
Anything else is proposed in a message first.

### 17.1 Components, and where each one lives in the reference implementation

Everything is a port of `~/Documents/dev/oocdm`. Nothing is invented.

| # | component | paper source | status |
|---|---|---|---|
| 1 | `OOCGraph` -- `local_matrices[C]` + `global_matrix` | `core/causal_graph.py` | **P0 BUILT** |
| 2 | `FullMaskGenerator` | `alg/model/mask_generator.py` | implicit (it is what `G1` means) |
| 3 | `RandomMaskGenerator(p_edge=0.9)` | same | P1 |
| 4 | `GraphMaskGenerator` | same | **P0 BUILT** |
| 5 | `DroppedMaskGenerator` (drop one field) | same | P2 |
| 6 | `masked_retain`, **per target field** | `object_oriented_model.py` | **P0 BUILT** |
| 7 | 3-term loss `-(ll_graph + ll_full + ll_random)/3` | `train_ooc.py:90` | P1 |
| 8 | CMI edge test, `causal_threshold = 0.1` | `train_ooc.py:get_causal_graph` | P2 |
| 9 | graph update + `n_iter_stable` stopping | `train_ooc.py:update` | P2 |

**One Stage 1 deviation had to be reverted in P0, and was.** `f_global` had been collapsed to one encoding per
object instead of one per `(object, target state field)`, on the argument that at `G1` there is no masking so
it cannot matter. Masking is per target field, so that simplification is incompatible with having a graph at
all. The model went 4.66M -> 8.14M parameters restoring it.

### 17.2 P0 -- masking, and the falsification gate

Build (1), (4), (6); revert the `f_global` deviation. Then hand-set `Link.pos'`'s parents to
`{pos, vel, speed, stick, bg_flags}` -- the set section 16's ridge probe says it needs -- leave every other
field at `G1`, and train. One variable against `runs/paper-fast-s0`.

**Gate: does `Link.pos` reach ~+0.22?** -> **PASSED at +0.207. See 17.7.**

Reference points for reading the answer: trained model at `G1` **+0.064**; ridge given only the right parents
**+0.219**; zero-parameter `pos + 1.67*vel` **+0.257**.

Asserted before the gate was run, so the gate means something: perturbing a masked-out field moves the output
by **exactly 0.000e+00**, perturbing a kept parent moves it, and identity-init still holds under both the full
and the masked graph.

### 17.3 P1 -- the paper's objective

Add (2), (3), (7). Train with `-(ll_graph + ll_full + ll_random)/3`, graph empty at init, `p_edge = 0.9`.
This is what teaches the model to answer "what would you predict WITHOUT this parent?", which is the question
discovery asks it. Without it, CMI is measuring a model that has never seen a masked input.

Note for the implementer: `MaskedFieldLinear` folds the mask into the weight, which is only valid because a
graph mask is identical for every batch element. `RandomMaskGenerator` produces a **per-batch-element** mask
and needs the expanded path.

### 17.3b P1 result: the objective is plumbed and verified

`runs/p1-paper-obj`, wandb `vxgnxfjj`, group `oocdm-p1`. Same 120-episode subset, seed 0, batch 128.

    epoch 1/2  loss 9.223  [graph 10.07  full 8.71  random 8.89  spread 1.364]  540s

**`ll_spread` = 1.364, not zero** -- the three terms genuinely diverge, so the random-mask term is doing
work and P2's CMI will interrogate a model that has seen masked inputs. And the ORDERING is the real
verification, because it is monotone in the number of available parents:

| term | loss | parents available |
|---|---|---|
| `ll_graph` (empty graph) | **10.07** | none -- the head can only predict do-nothing |
| `ll_random` (`p_edge` 0.9) | 8.89 | 90% |
| `ll_full` (`G1`) | **8.71** | all |

That ordering cannot arise unless the masks reach the computation and genuinely remove information, which is
the property P2 depends on.

**P1 is plumbing, not improvement, and its numbers are not comparable to P0's.** Tune performance dropped
26/44 -> 11/44 because a third of the objective is now a term that can only predict do-nothing. This is
expected: the graph is empty until P2 adds edges.

**COME BACK TO THIS -- speed.** 540s/epoch against 109s at `G1`, a 5x cost from three forwards per batch plus
the expanded `masked_retain` path that a per-batch-element random mask requires. Two things follow. P2's CMI
adds roughly 170 more forwards per graph update (85 global input fields + 85 local across classes) -- those
use BATCH-INDEPENDENT dropped masks so they take the cheap folded path, but the count still has to be sized
before it is launched. And if P1's cost becomes the bottleneck, the fix is to fold the random mask per batch
rather than per batch element; that is a deviation from `RandomMaskGenerator` and must be proposed, not
assumed.

### 17.4 P2 -- discovery

Add (5), (8), (9). For each class-level edge, compare log-likelihood under the full graph against the graph
with that one edge dropped; keep the edge if the difference exceeds `causal_threshold = 0.1`. Recompute the
graph each iteration; stop when it has been stable for `n_iter_stable`.

Cost: one graph update needs roughly `n_field` forward passes for the global edges plus `n_field` per class
for the local ones -- about 180 passes over a batch set, once per update rather than per batch.

### 17.4b P2 result: discovery recovers the physics

`discover_graph` ported from `train_ooc.py:get_causal_graph`. CMI per class-level edge, threshold 0.1, run
against the P0 checkpoint on 256 held-out transitions.

**`Link.pos'`'s parents, discovered rather than specified:**

| CMI | edge |
|---|---|
| **+0.5228** | `Link.vel -> pos'` |
| **+0.2526** | `Link.bg_flags -> pos'` |
| +0.0172 | `Link.stick -> pos'` |
| +0.0050 | `Link.y_dist_to_ledge -> pos'` |
| ~0 | everything else |

It found **velocity and collision** on its own -- the actual physics, and two of the five fields hand-picked
for P0 from the ridge probe. `stick` is an order of magnitude lower, consistent with section 13.7's finding
that a commanded direction sits 66 deg from the real displacement at 10 Hz: what you pressed matters far less
over one step than where you were already going and what you are touching.

**Other top edges, all mechanically sensible:**

| CMI | edge | reading |
|---|---|---|
| +0.7770 | `Link.vel -> fall_distance'` | falling is integrated velocity |
| +0.4495 | `En_Dekubaba.fn_rel -> timer'` | a Baba's action function drives its own timer |
| +0.4380 | `Link.melee_state -> speed'` | swinging a sword stops you |
| +0.3740 | **`En_Dekubaba.has_arrow -> Link.sf1'`** | a GLOBAL edge -- the paper's `Cl.U -> Ck.V'` firing on a real cross-object interaction |

**The graph is extremely sparse: 31 of 8,286 candidate edges at threshold 0.1 -- 0.37%.** CMI median is
exactly 0.0000 and p90 is exactly 0.0000, so the overwhelming majority of candidate parents carry no
information whatsoever. That is section 16's diagnosis confirmed from the other direction: the model was
being handed thousands of inputs of which a few dozen matter.

**Cost, measured not estimated.** 170 dropped-mask forwards per graph update (85 input fields x global +
local), 299 ms each at batch 128 on MPS: **51 s at 1 CMI batch, 66 s at 2, 204 s at 4.** Workable once per
epoch; the paper recomputes every iteration, which at this cost we cannot.

**Known approximation, recorded not hidden.** The paper's heads all emit distributions so its `log f` is a
log-probability. Ours is too for real/count (Gaussian), categorical/boolean (softmax) and bits (Bernoulli),
but the ANGLE head emits a direction and its `1 - cos` is a monotone surrogate. Differences stay correctly
signed per field, but the nats scale is wrong for angles, so `causal_threshold` does not mean the same thing
for them. The clean fix is a von Mises head, which is a change to the model and therefore a proposal.

### 17.4c P2 stability: the graph is not yet reproducible

Run against the **P1** checkpoint -- the one the method specifies, trained with the random-mask term that
makes CMI well-posed -- with two independent gameplay samples of 256 transitions each:

    seed 0:  37 edges
    seed 1:  37 edges
    shared:  12          Jaccard 0.194

**Two discovery passes agree on a third of their edges.** At 256 transitions against a 0.1-nat threshold,
roughly two thirds of the discovered graph is sampling noise: `epsilon` is deciding rather than the data,
which is the failure AGENTS.md section 7 names.

It shows in the edges. Alongside real ones (`Link.vel -> pos'`, `Link.btn_A -> vel'`,
`En_Dekubaba.has_arrow -> Link.sf1'`) sit `Link.floor_height -> En_Dekubaba.pos'` (+0.2034) and
`En_St.vel -> Link.vel'` (+0.3199). Link's floor height does not move a Deku Baba and a Skulltula's sway does
not drive Link. Those are coincidences of a small sample.

**A real and expected effect that complicates the threshold.** CMI values are COMPRESSED on the P1 checkpoint
relative to P0: `Link.vel -> pos'` +0.5228 -> +0.2513, `bg_flags -> pos'` +0.2526 -> +0.1149. That is correct
behaviour -- P1 trained under random masks, so the model is more robust to losing any single parent, which
shrinks every CMI. But it means a FIXED `causal_threshold = 0.1` sits at a different point on the scale than
it does in the paper, and is doing more of the deciding than it should.

**The update loop is NOT being wired on this.** Feeding a 19%-reproducible graph back into training would
bake the noise in and look like progress, which is worse than failing loudly.

**The paper-faithful fix is more samples, not a new criterion**, so `scripts/eval_cmi_stability.py` sweeps the
CMI sample size at 2x, 4x and 8x and reports Jaccard at each. If agreement climbs toward ~0.8 the estimator is
sample-starved and we size there. If it plateaus low, these effects are not identifiable at this threshold on
the data we have -- a finding about the method here, and one that would mean recording more gameplay rather
than spending more compute.

**Two open questions before the update loop (component 9).** Whether 31 edges is too aggressive -- sparsity
is the goal but a graph this thin could starve fields that genuinely need parents -- and whether CMI is
stable at 256 transitions against a 0.1-nat threshold, which is the seed/Jaccard control from P3 pulled
forward. Both are being measured against the P1 checkpoint, which is the one the method actually specifies,
since it was trained with the random-mask term that makes CMI well-posed.

### 17.4d P2 COMPLETE: the loop converges, and the discovered graph beats the hand-picked one

`runs/p2-loop`, wandb `ovre14fq`, group `oocdm-p2`. Three iterations, 34m19s, all four v3 collections,
eps 0.05, CMI 10,240 transitions, `p_edge` 0.9, one epoch per iteration.

| iter | edges | +/- | jaccard prev | `ll_graph` | `ll_full` | fields won | `Link.pos` | `En_Dekubaba.pos` |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | +48/-0 | -- | 37.26 | 35.85 | 17/47 | +0.055 | +0.113 |
| 2 | 39 | +11/-20 | 0.475 | 36.09 | 34.02 | 23/47 | +0.107 | +0.134 |
| 3 | 48 | +11/**-2** | **0.740** | **34.97** | **32.92** | **25/47** | **+0.173** | **+0.154** |

**Every column moves monotonically in the right direction.** The graph converges -- churn collapses from 20
edges removed to 2, agreement rises 0.475 -> 0.740 -- while both loss terms fall and six more fields start
beating do-nothing. `Link.pos` triples.

**The discovered graph beats the hand-specified one.** Read against each arm's own velocity control, which
section 18.1 makes mandatory:

| | `Link.pos` | its velocity control | % of baseline |
|---|---|---|---|
| P0, hand-picked `{pos, vel, speed, stick, bg_flags}` | +0.207 | +0.275 | 75% |
| **P2, discovered** | +0.173 | +0.205 | **84%** |

That closes the loop on section 16: masking is the mechanism, and the paper's CMI finds a better mask than a
human reading a ridge table did.

**What is NOT fixed, so it does not get lost in the good news.** `En_Dekubaba.hp` is exactly **+0.000** --
damage has been unmodelled since section 11 and nothing in P0-P2 touched it. `Camera.eye` is -0.011. And
three iterations is thin for a convergence claim: the paper runs to `n_iter=100` with an `n_iter_stable`
stopping rule, so the honest statement is "trending to stability over three iterations", not "converged".
The loop supports `--resume`, so extending it is cheap.

### 17.5 P3 -- the paper's own controls

These are the results the paper reports, and the only ones that say whether any of this works.

| control | question it answers |
|---|---|
| `G-hat` vs `G1` vs the EMPTY graph | does the discovered graph beat both "everything is a parent" and "nothing is"? |
| **shuffled-class** | assign instances to classes at random so each class holds a MIX. If that scores the same, class sharing contributed nothing. The paper does not run this one; we must |
| >= 3 seeds, edge-set Jaccard | is the graph stable, or is `epsilon` deciding rather than the data? (AGENTS.md section 7) |
| multi-step rollout | the barometer (AGENTS.md section 8). Section 15 is the `G1` reading: 4% of real travel |

### 17.6 Out of scope

Recorded so it cannot drift back in. All four were built once, none is in Yu et al., all are removed, and
none may return without an explicit decision first: the per-field **change gate**, per-field **`pos_weight`
oversampling**, **binned direction/magnitude** heads, and firing **threshold calibration**.

The schema defects of 14.2 (`En_Dekubaba.vel` a dead input, `Link.y_dist_to_ledge` a sentinel typed `Real`,
`Link.dist_interact_wall`) are real but are **not** deviations and **not** P0-P3. They are fixed after P2, or
earlier only if discovery names them as parents of something that matters.


### 17.7 P0 result: the gate passes

`runs/paper-fast-s0` (G1) vs `runs/p0-masked` (hand graph). Same 120-episode random subset, same seed 0,
2 epochs, test slots {7,9}. The ONLY difference is `Link.pos'`'s parent set: 8,168 of 8,286 edges active
instead of all of them.

| arm | `Link.pos` AGG | `Link.pos` CHG |
|---|---|---|
| `G1` -- every field a parent | -0.310 | +0.134 |
| **P0 masked -- `{pos, vel, speed, stick, bg_flags}`** | -0.365 | **+0.207** |
| ridge given only those parents (section 16) | -- | +0.219 |
| `pos + 1.67*vel`, zero parameters | -- | +0.257 |

**+0.134 -> +0.207, a 54% improvement from masking one field's parents, landing essentially at the ridge
ceiling of +0.219.** Section 16's claim holds: masking is the mechanism, not an optimisation. The model can
use those five fields once it is not also handed the other 33.

Three caveats, recorded rather than smoothed:

1. **Aggregate skill got slightly WORSE** (-0.310 -> -0.365). Masking helped the model predict motion when
   motion happens; it did not stop it moving when nothing moved. That failure is unchanged.
2. **Fields left at `G1` moved anyway.** `Link.dist_interact_wall` -27.9 -> -57.5,
   `En_Dekubaba.is_targeted` +0.019 -> +0.336. They share the trunk, so training dynamics couple them: the
   change is clean in the GRAPH but not in the OPTIMISATION, and run-to-run variation is evidently large for
   some fields.
3. **Single seed** (AGENTS.md section 7). Given (2), a +0.073 effect from one run is not a measurement. The
   seed control in P3 is what settles it; the gate is accepted on the strength of the effect size landing on
   an independently predicted value, not on the single run alone.

**Auditability gap, and the fix.** Both P0 arms were run with `--no-wandb`, so they exist only as
`runs/*/train.log` + `fields.json`. That was an omission. The trainer now logs the graph itself --
`graph/edges_active`, `graph/edges_candidate`, `graph/edges_global`, `graph/edges_local`,
`graph/frac_active`, and a `graph/kind` + `graph/hand_parents` summary -- because two runs differing ONLY in
their causal graph are otherwise indistinguishable in wandb. The velocity baseline is logged as
`control/velocity_baseline_val_chg` for the same reason. Every run from here is wandb-audited by default.

---

## 18. Datasets: what we train on, and where it lives

| collection | episodes | `Link.pos` changes | note |
|---|---|---|---|
| `v3-main` | 121 | 14.1% of steps | combat; Link often stationary |
| `v3-approach` | 70 | -- | approach scenarios |
| `v3-moves` | 56 | -- | movement scenarios |
| `v3-scenarios` | 151 | -- | mixed |
| **all four** | **398** | **24.9%** | 203,424 transitions, 363-d v3 state |

The data itself is small -- **16 MB across 403 files** (`episodes/*.npz` + `manifest.jsonl`, and `dataset.json`
in `v3-main` only, which is all `load_many` needs). The 5.3 GB per directory is `instances/env*/StateSaves`,
Dolphin emulator savestates, correctly gitignored by `*.s[0-9][0-9]`.

**S3.** The v1 dataset was already backed up; the v3 collections we have used all session were NOT. They are
now, under the existing convention and verified episode-for-episode:

    s3://zelda-rl/zelda/combat/deku-baba/transitions/v3-main/20260919-084052/        121 episodes
    s3://zelda-rl/zelda/combat/deku-baba/transitions/v3-approach/20260919-084052/     70
    s3://zelda-rl/zelda/combat/deku-baba/transitions/v3-moves/20260919-084052/        56
    s3://zelda-rl/zelda/combat/deku-baba/transitions/v3-scenarios/20260919-084052/   151

To restore (AWS profile `race-wm`, us-west-1, same profile `build_savestate_manifest.py` uses):

    aws s3 sync s3://zelda-rl/zelda/combat/deku-baba/transitions/ datasets/transitions/ --profile race-wm

### 18.1 A silent bug: every run before P2 trained on ONE collection

`scripts/train_oocdm.py`'s `DEFAULT_DATA` was `v3-main` alone, and no run passed `--data`. So the
random-sampling-across-collections fix made earlier was real but **unreachable** -- Stage 1, P0 and P1 all
trained on one of four collections while the code and docs implied otherwise. `--episodes 120` from a
121-episode collection also meant those runs had essentially no episode sampling at all.

It matters for reading their numbers, because the two sets are not the same problem:

| | `Link.pos` changes | trivial `pos + k*vel` control |
|---|---|---|
| `v3-main` only (Stage 1, P0, P1) | 14.1% | **+0.275** |
| all four (P2 onward) | 24.9% | **+0.205** |

So P0's gate result (+0.134 -> +0.207, section 17.7) was measured on the narrower, easier collection, and
**P2's numbers are not directly comparable to it.** Each must be read against its own velocity control.
`DEFAULT_DATA` is now all four collections, with the reason recorded at the definition.
