# Causal object-oriented world model: background, schema, and the Stage 0 field census

A second methodology trail, started where [`LATENT_WORLD_MODEL.md`](LATENT_WORLD_MODEL.md) stops. That
document records thirty sections of work on `z_{t+1} = M(z_t, a_t)` and ends with six consecutive negatives.
This one records what replaces it and why, and it follows the same rule: what was built, what was measured,
what each change was a response to. Not an argument for a conclusion.

**Method:** Object-Oriented Causal Dynamics Model (OOCDM), Yu, Ruan & Xing, *Learning Causal Dynamics Models
in Object-Oriented Environments*, ICML 2024. Paper PDF and the authors' reference implementation are local
(`~/Downloads/yu24j.pdf`, `~/Documents/dev/oocdm`).

**Status:** Stage 0 complete (schema + field census, no training). Stage 1 in progress.

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

- **§24 (capacity contention) dissolves structurally.** `Link.pos'` and `En_Dekubaba.hp'` are separate heads
  with separate parents and separate loss terms in different units. There is nothing for them to compete
  over. §24's own numbers say both objectives are individually achievable on this data.
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
