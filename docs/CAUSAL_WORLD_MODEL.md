# Causal object-oriented world model: background, schema, and the Stage 0 field census

A second methodology trail, started where [`LATENT_WORLD_MODEL.md`](LATENT_WORLD_MODEL.md) stops. That
document records thirty sections of work on `z_{t+1} = M(z_t, a_t)` and ends with six consecutive negatives.
This one records what replaces it and why, and it follows the same rule: what was built, what was measured,
what each change was a response to. Not an argument for a conclusion.

**Method:** Object-Oriented Causal Dynamics Model (OOCDM), Yu, Ruan & Xing, *Learning Causal Dynamics Models
in Object-Oriented Environments*, ICML 2024. Paper PDF and the authors' reference implementation are local
(`~/Downloads/yu24j.pdf`, `~/Documents/dev/oocdm`).

**Status:** Stage 0 complete. Stage 1 first run complete and **it does not beat do-nothing** (section 11).
Open work is section 12; nothing there is a nice-to-have, each item has a measured defect behind it.

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
