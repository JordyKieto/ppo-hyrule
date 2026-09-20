# Latent world model: methodology trail

How the current experiment was arrived at. This is a record of what was built, what was measured, and what
each change was a response to — not an argument for a conclusion. Claims with evidence classes live in
[`../paper/CLAIMS.md`](../paper/CLAIMS.md); the paper draft is [`../paper/paper.md`](../paper/paper.md).

**Current run:** [wandb `nqild5b2`](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/nqild5b2)
(project `zelda-rl-combat-ppo`, group `dream-deku-baba`) — PPO inside the learned model, targeting
40,995,840 dream transitions to match the model-free run's 40.7 M authoritative env steps.

**Reference:** the model-free agent reached **return 55.5, 76% success at global update 150 / 40.7 M env
steps** ([COMBAT_TASK.md](COMBAT_TASK.md), [wandb `vxejacj7`](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/vxejacj7)).

---

## 1. Dataset

`scripts/collect_transitions.py` — 109,817 transitions over 168 episodes, replayed from six checkpoints
along the model-free training curve (G5 through G150, 27% → 81% success) so coverage spans competence
rather than expert play only. Card: [`../datasets/transitions/deku-baba/README.md`](../datasets/transitions/deku-baba/README.md).

State is 335 numbers: 15 Link fields plus 16 actor rows × 20 fields. Rows are permutation-stable within an
episode, and no raw pointers appear — lock-on is stored as a row index.

## 2. Encoder and decoder

A behaviour-cloning policy over the actor-set structure (shared per-row encoder, masked mean+max pooling,
concat with the Link block, MLP to a 128-d latent), with a decoder reading back the raw fields
`CombatTracker.step()` consumes. Splits are three-way **by start slot**: train {1,2,3,5,6,10},
threshold-tuning {4,8}, test {7,9}.

Per-row identity survives the pooling. On test slots: Baba behaviour state 0.998, lock-on row 1.000,
live-Baba count 0.9999, Link HP 0.987, row HP 0.90, Link position MAE ~3–7 world units.

## 3. Event decoding, four refinements

Reward events are rare — damage occurs once in 253 transitions — so recovering them by differencing a
decoded state is unforgiving. Measured progression of damage-event F1:

| step | damage F1 |
|---|---|
| HP regressed, then rounded and differenced | 0.286 |
| HP as **classification** (discrete: Link 7 half-heart values, Baba 0/1/2) | 0.458 |
| **+ monotonicity** — HP never rises (0 of 891,459 real row-steps; Link healed once in 109,649) | 0.746 |
| **+ calibrated score and a threshold chosen on the tuning slots** | 0.906 |

Cross-head consistency (a Baba at 0 HP is dying or despawned) and light oversampling of HP-transition
frames (5%; 15% is harmful) were also kept. Final test-slot average precision: damage 0.89, kill 0.90,
Link damage 0.83.

## 4. Dynamics: what was tried

`scripts/train_dynamics_simple.py`, `scripts/train_dynamics.py`, `scripts/train_jepa.py`,
`scripts/train_joint.py`. Skill is `1 - model_error / identity_error`, where identity means predicting no
change; negative is worse than predicting no change.

| configuration | skill |
|---|---|
| frozen BC latent, MSE, no residual, stride 5 | −7.24 |
| frozen BC latent, MSE, residual, stride 1 | −1.73 |
| unfrozen, EMA target + stop-gradient, L1 on normalised targets, 2-step rollout | −1.78 |
| joint decode+predict, no BC, lr 1e-4 | −1.37 |
| joint, with exit data (`joint_v3`) | −1.14 |

Latent size was swept 32→512: decode quality saturates at 128 and predictability degrades monotonically
with width. BC training length was swept 2→300 epochs: predictability improves to about 100 epochs
(−0.805) and reverses after, while decode quality falls throughout.

Design choices were checked against the source papers rather than recalled — L1 rather than MSE,
normalisation on the target rather than the prediction, a rollout term, and a stride above one frame
(I-JEPA 2301.08243 and its reference implementation, V-JEPA 2404.08471 Eq. 1, V-JEPA 2-AC 2506.09985
Eq. 2–3, DINO-WM 2411.04983).

## 5. Reward parity

An audit of `CombatTracker.step` found the dream implemented **5 of its 18 terms**. The four missing
*outcome* terms were room exit (−20), the timeout penalty (−3 per live target), the kill-speed bonus
(up to +10) and the success-time bonus (up to +15) — the terms that make play urgent.

The fix was to stop re-implementing. `zelda_rl/dream_state.py` rebuilds a real `CombatState` from decoder
outputs and the dream calls **the same `CombatTracker` the emulator uses**, so parity is structural.
Shaping terms (lock-on, approach, shield, block, dodge) come along with it, since they need only geometry
the decoder already produces.

## 6. Room exit: no examples existed

`left_room` is a −20 terminal. The released dataset contains **0 of them in 168 episodes** — every
demonstrator was trained with that penalty and learned never to leave. A model built on it cannot represent
the outcome, so the penalty cannot fire.

`scripts/collect_exits.py` drives the emulator out of the room: a few random steps, then a held direction
away from the nearest Baba. Slot 1 exits reliably (7/7, 7/7 in two runs) because it starts at the entrance;
other slots are deep in the room and time out instead. 27 exit episodes were collected. The exit mechanism
in the data is a **scene transition 0→85**, not a room-number change or a radius trip.

Three bugs each silently nulled the previous fix:

1. **The loader dropped every exit state.** `load()` kept `state[:T]` for `T` actions, and `scene` flips at
   exactly index `T` — so all 27 exits, 100% of the signal, sat one index past what training could see.
   The shared loader now keeps `state[:T+1]`.
2. **The training index dropped them again.** The predictor needs pairs, so its index required each frame
   to have a successor; the exit frame is the last of its episode. The decoder now trains on every frame
   and the predictor on the subset with a successor.
3. **The class weight collapsed.** A raw 3633:1 inverse-frequency weight gave "in room" a weight of 0.0005.
   Square-rooted.

Measured on 7 held-out exit episodes: **recall 1.000 on exit frames, false-positive rate 0.0055 on the 363
in-room frames.**

Decoder heads added for this: `scene` (binary in-room / left, which is what `_left_reason` tests),
`room`, and `baba_count` as one 4-way head rather than a count derived from 16 per-row predictions.

## 7. Effect of the exit data on the latent

Adding 27 scripted walkout episodes moved the action-effect measurement more than any architectural change:

| encoder | action recoverable from a latent pair (top-1) |
|---|---|
| BC encoder | 0.059 |
| joint_v2 (no exits) | 0.052 |
| **joint_v3 (with exits)** | **0.139** |
| single latent, no transition (floor) | 0.056 |
| Bayes ceiling (the demonstrator is stochastic) | 0.220 |

The exit episodes are ~30 steps of consistent directional walking with no combat. The released dataset is
expert combat, where Link is stationary 74% of steps.

## 8. Policy and PPO parity

The dream policy is `DreamPolicy` (`zelda_rl/dream_state.py`), a subclass of the model-free run's
`RecurrentEntityActorCritic`: proj → GRUCell → actor/critic heads, their widths, initialisations,
`act()`, `value()`, `init_hidden()` and `ppo_update_recurrent` are that class's, unchanged. Only
`_features` is bypassed, because the latent is already the state encoding.

Its observation is the 128-d latent plus the **8 episode-memory features the real policy saw**, computed
with the formulas in `combat_features`: kills, damage dealt, hearts lost, time since last hit / damage /
kill, lock-ons, time remaining. These matter because the reward terms are stateful — lock-on is capped at
6 per episode, blocks at 8, and the kill bonus decays with time since the last kill.

Hyperparameters were checked against `scripts/train_combat.py` and aligned:

| | real | dream (before) | dream (now) |
|---|---|---|---|
| gamma | 0.995 | 0.99 | **0.995** |
| gae lambda | 0.95 | 0.95 | 0.95 |
| clip / epochs / ent / vf / grad-norm | 0.2 / 4 / 0.01 / 0.5 / 0.5 | same | same |
| target_kl | 0.03 | none | **0.03** |
| chunk | 60 | 16 | **60** |
| chunks_per_mb | `n_chunks // 16` | 8 fixed | **`n_chunks // 16`** |
| hidden / gru | 256 / 256 | 256 / 256 | 256 / 256 |
| rollout T × N | 1800 × 3 | 32 × 128 | 120 × 128 |

## 9. Current run

`joint_v3` encoder/decoder/predictor, real `CombatTracker` reward, recurrent policy with memory,
aligned PPO. Logged through the repo's `RunLogger` into the same wandb project as the model-free run, with
matching metric names (`episode/return`, `combat/success_rate`, `combat/defeated`, …) so the two overlay.

First 60 updates:

| upd | transitions | return | len | clear% | death% | left% | kills | dmg | hpLost | lock | blk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15,360 | −0.12 | 94 | 13.2% | 0.0% | 0.0% | 0.13 | 1.21 | 0.14 | 0.81 | 0.34 |
| 10 | 153,600 | 2.72 | 88 | 18.6% | 0.0% | 0.0% | 0.19 | 1.07 | 0.10 | 0.73 | 0.34 |
| 20 | 307,200 | 6.04 | 82 | 23.0% | 0.0% | 0.0% | 0.23 | 1.20 | 0.06 | 1.16 | 0.38 |
| 30 | 460,800 | 7.86 | 80 | 22.6% | 0.0% | 0.0% | 0.23 | 1.34 | 0.00 | 1.49 | 0.33 |
| 40 | 614,400 | 3.94 | 86 | 21.3% | 0.0% | 0.0% | 0.21 | 1.79 | 0.00 | 1.80 | 0.34 |
| 50 | 768,000 | 8.88 | 81 | 23.4% | 0.0% | 0.0% | 0.23 | 2.35 | 0.00 | 2.18 | 0.27 |
| 60 | 921,600 | 6.67 | 84 | 21.9% | 0.0% | 0.0% | 0.22 | 2.89 | 0.00 | 2.26 | 0.32 |

Damage dealt rises 1.07 → 2.89 and lock-ons 0.73 → 2.26 over the same span, while `death%` and `left%`
remain at 0.0%.

An open observation, recorded without interpretation: the latent's one-step prediction skill is −1.14 —
worse than predicting no change — yet the decoded fields the reward reads are accurate enough that PPO
produces movement in the shaping terms. Whether that holds at 41 M dream steps, and whether any of it
transfers to the emulator, is what the current run and its evaluation are for.

## Reproduce

```sh
.venv/bin/python scripts/train_joint.py --epochs 25 --lr 1e-4 \
    --extra-data datasets/transitions/deku-baba-exits/*/ --out checkpoints/dynamics/joint_v3.pt
.venv/bin/python scripts/train_dream_ppo.py --model checkpoints/dynamics/joint_v3.pt \
    --transitions 41000000 --out checkpoints/dynamics/dream_v3_41M.pt
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/play_dream_policy.py \
    --policy checkpoints/dynamics/dream_v3_41M.pt --model checkpoints/dynamics/joint_v3.pt --slot 1
```

## 9. State v2: Link's body, collision and action state

The dream was blind to what an action *does*. The v1 state carried Link's position, yaw and three
`stateFlags1` bits, so a jump, a roll, a sword swing and a Z-press with no target were nearly
indistinguishable in it.

Offsets were taken from the **OoT decompilation** rather than recall — `include/actor.h` and
`include/player.h` in [zeldaret/oot](https://github.com/zeldaret/oot). Two corrections were needed. A
recalled `Actor` layout had dropped `Vec3f scale` at `0x50`, putting the whole collision block 0xC too low;
and the decomp prints offsets for a `DEBUG_FEATURES` build whose `Actor` ends with `char dbgPad[0x10]`
(size 0x14C), while retail NTSC 1.2 has no pad, so `Actor` is 0x13C and every `Player` field sits at
**decomp offset − 0x10**. Two offsets already in `zelda_rl/combat.py` confirm the shift independently
(`stateFlags1` 0x67C−0x10 = 0x66C, `heldItemAction` 0x151−0x10 = 0x141). Procedure: `docs/ADDING_OBSERVATIONS.md` §0.

Earlier guesses had been *rejected for the wrong reason*: `+0x84` and `+0x88` were called "constants across
savestates" when `+0x84` is `depthInWater` (−32000, above water) and `+0x88` **is** `bgCheckFlags`
(`0x81` = `GROUND|GROUND_STRICT`). They were correct; they looked constant because Link is grounded and dry
in every start savestate.

23 fields added, state 335 → 358 (`zelda_rl/state_layout.py`, now the single definition shared by the
collector and the model). Verified live by `scripts/verify_link_fields.py`, 2,510 frames, 16/16 checks; the
decisive one is a redundant pair: `BGCHECKFLAG_WALL` == (`wallPoly` != 0) at **1.0000 agreement, 0
disagreements**.

### Data

| run | episodes | transitions | note |
|---|---|---|---|
| standard (7 checkpoints × 10 slots) | 120 | 107,296 | 0 room exits, 2 deaths |
| scripted scenarios | 190 | 79,505 | **33 room exits, 47 deaths** |

`scripts/collect_scenarios.py` adds four scripted scenarios, overlapping on purpose: `walkout` (away from
the nearest Baba), `suicide` (into it, never shielding — dies in exactly 6 hits), `wall` (each compass
direction into the geometry), `flail` (uniform over all 63 actions).

### Decode, test slots (`joint_v6_state2`)

| head | result | baseline |
|---|---|---|
| `link_action` (21-way action state machine) | **0.9308** | majority 0.5822 |
| ↳ recall: jump / jump_attack / attack | 1.00 / 1.00 / 0.97 | n = 1354 / 273 / 985 |
| ↳ recall: knocked_air / wall_push / hurt | 1.00 / 1.00 / 0.85 | n = 40 / 16 / 34 |
| collision WALL (acc / recall) | 1.000 / 1.00 | |
| airborne (acc / recall) | 1.0000 / 1.000 | base rate 0.047 |
| z_target incl. no-target | 1.0000 / 1.000 | 27.8% of frames Z-held with no lock-on |
| wall yaw / shape yaw / parallel yaw | 1.4° / 1.9° / 3.5° | |
| baba_count / link_hp / row_state | 0.9999 / 0.9960 / 0.9875 | |
| link_pos MAE | 12.04 | |
| left-room recall (9 held-out exit episodes) | 1.000, FP 0.00074 | |

The `actionFunc` values were labelled from what co-occurs with each, not from guessed names: `3159404` is
airborne 100% (jump), `3161932` airborne + Z 90% + melee 9% (the timed Z+A kill move), `3157388` damaged
100% (hurt), `3166492` wall 44% at speed 4.8 (pushing a wall). `3156912` is airborne + damaged — knockback,
which matters because `knockback_type` itself reads 0.00%: it is a one-frame flag, invisible at 10 Hz.

### One-step prediction

Best tune skill **−0.711**, against −1.14 for `joint_v3` — the best recorded, still worse than predicting
no change.

### Action recoverability went DOWN, and the measurement had to be fixed twice

`scripts/probe_action_effect.py`. Two measurement errors were corrected before the number could be read:

1. **A pooled number across sources is meaningless.** `flail` draws actions uniformly, so its ceiling is
   near 1/63, while a scripted leg is near-deterministic. Pooling mixes ceilings.
2. **A random frame split leaks.** Neighbouring frames of one episode landed in both halves; since a
   scripted leg holds one direction for ~45 steps that is near-total leakage. It scored `suicide` 0.96 →
   0.96 only after splitting **by episode** did the rest of the table become readable.

| source | n | probe top-1 | most-common-action baseline |
|---|---|---|---|
| `policy` (comparable to v1's 0.139) | 22,240 | **0.0690** | 0.0612 |
| `flail` (uniform: chance 0.0159) | 7,220 | **0.0554 = 3.5× chance** | 0.0184 |
| `suicide` | 2,643 | 0.9576 | 0.6871 |
| `walkout` | 2,043 | 0.6632 | 0.2956 |
| `wall` | 2,896 | 0.6544 | 0.2227 |

On the comparable `policy` population this is **0.069, down from `joint_v3`'s 0.139** — recorded as a
regression, not explained away. Decode quality rose across every head over the same change, and the latent
spread widened (top PC 47.2% / 49 PCs for 99%, against the BC encoder's 61.5% / 63), which is consistent
with a 128-d latent now spending capacity on the richer state. That is a hypothesis, not a finding.

The `flail` row is the first clean measurement of action-effect visibility in this project: its actions are
independent of state, so no state prior can contribute and anything above 1/63 is the transition itself
carrying the action. 3.5× chance is what the latent actually reveals.

### Bug found: the saved normaliser was the wrong one

`train_joint.py` wrote `ref["norm"]` — the *reference* checkpoint's normaliser — instead of the one used for
training. Harmless while the widths matched; with the v2 feature block it would have silently
mis-normalised every consumer of the checkpoint (`train_dream_ppo.py`, `play_dream_policy.py`, the probes).

## 10. The "regression" did not exist, and the capacity hypothesis was wrong

§9 recorded action recoverability falling 0.139 → 0.069 and offered a hypothesis: a fixed 128-d latent
spending capacity on the richer v2 state. Both parts were wrong.

**The sweep falsified the hypothesis** (`scripts/sweep_capacity.sh`, 7 configs, 25 epochs each):

| config | tune skill | inv-dyn (pooled) | link_action | link_pos MAE |
|---|---|---|---|---|
| latent 128 (base) | **−0.711** | 0.0679 | 0.9308 | 12.04 |
| latent 192 | −1.137 | **0.0814** | **0.9810** | **6.32** |
| latent 256 | −1.095 | 0.0769 | 0.9330 | 7.94 |
| latent 384 | −1.289 | 0.0802 | 0.9408 | 9.52 |
| w_pred 3.0 | −0.814 | 0.0710 | 0.9275 | 9.15 |
| **w_v2 0.3** | −0.761 | **0.0726** | 0.9269 | 8.86 |
| latent 256 + w_pred 3 | −1.076 | 0.0685 | 0.9334 | 8.30 |

Down-weighting the v2 decode heads to 0.3 moved action recoverability 0.0679 → 0.0726, nowhere near 0.139.
So the v2 heads are not consuming the capacity, and the stated hypothesis is rejected.

**The comparison itself was invalid.** `joint_v3`'s 0.139 came from a probe trained on a different frame
population, on different data. A second explanation offered — that the v2 policy episodes have a more spread
action distribution — was also checked and **also wrong**: the two collections are nearly identical
(entropy 4.042 vs 4.032 nats, perplexity 56.9 vs 56.4, most-common action 0.078 vs 0.067).

The decisive test is available because v2 only *appends* columns, so a pre-v2 encoder can read a v2 state
through the legacy 45-wide feature path (`featurise(..., link_v2=False)`, selected by `wants_v2(link_dim)`).
Running all three encoders on **identical frames, identical episode split, identical probe protocol**:

| model | state | policy top-1 | most-common baseline |
|---|---|---|---|
| `joint_v3` | v1 | 0.0678 | 0.0448 |
| **`joint_v6_state2`** | v2 | **0.0767** | 0.0448 |
| `lat192` | v2 | 0.0772 | 0.0448 |

**There is no regression.** On the same frames the v2 model is 13% better than `joint_v3`. The apparent drop
was an artifact of comparing numbers measured on different data with different probe-training populations —
the same class of error as comparing raw MSE across latent spaces.

The independent measurement that does not depend on any of this is the `flail` row of §9: uniform-random
actions, so no state prior can contribute, and **0.0554 = 3.5× chance (1/63)** is what the latent transition
reveals about which action was taken.

## 11. Rollout stability picks the model, and single-step decode would have picked wrong

`scripts/eval_rollout.py`, 24 held-out episodes of ≥ 400 steps, autoregressive on the model's own output.

| h = 300 | behaviour | baba_count | link_hp | link_pos MAE |
|---|---|---|---|---|
| `joint_v6_state2` (latent 128) | 0.562 | 0.667 | 0.292 | **91.1** |
| `lat192` | 0.406 | 0.125 | 0.542 | **104,310** |
| *identity (assume no change)* | *0.891* | *0.833* | *0.792* | *30.4* |

`lat192` has the better single-step decode of the two by a wide margin (link_pos MAE 6.32 vs 12.04,
link_action 0.981 vs 0.931) and **diverges** under autoregression — position error reaches 10^5 world units.
Choosing on single-step decode would have selected it. Latent 128 is used for the dream.

Recorded without softening: **both models remain worse than identity** on behaviour, live-Baba count and
Link HP at every horizon ≥ 10. The predictor still has not learned dynamics that beat assuming nothing
changes. What state v2 changed is what the model can *represent* (§9), not yet what it can *predict*.

## 12. Action states: named by driving the input, not by correlation

The first `LINK_ACTION_VALUES` table took the 20 most frequent `actionFunc` values and named each from what
co-occurred with it. **Several names were wrong**, found by driving the inputs
(`scripts/probe_move_states.py`):

| move driven | actual state | first (wrong) name |
|---|---|---|
| roll (run, then A) | `3166492` throughout | "wall_push" — it correlated with walls because `flail` rolls into them |
| shield (R held) | `3155380` throughout | "slow_walk" |
| sidehop (Z + side + A) | `3146184` | **absent from the table** → decoded as `other` |

`3146184` was in the data all along (58 of 187,111 frames); the top-20 cut truncated it. The table now
covers **all 34 observed values** across the three collections.

**Decomp function names could not be recovered.** zeldaret/oot does support `ntsc-1.2`, but `baseroms/ntsc-1.2/`
ships only checksums and a segment map — per-function symbol addresses come from building it. Rank-aligning
our 31 observed addresses against the 89 `Player_Action_*` definitions in source order fails, because the
address gaps (e.g. `0x8039D6CC → 0x803A277C`, 20 KB) span functions we never observe; the alignment put the
most frequent state, which is melee-heavy, on `WaitForPutAway`. A constant retail−debug offset does not exist
either (the builds have different function sizes). So names come from driven input or a definitive flag, and
everything else keeps the neutral name `fn_<value>` — 22 of 34.

### Rolls, backflips and sidehops are not separate action functions

Driving them and reading the state back: backflip, sidehop-left and sidehop-right **all** produce `3159404`
at **100% airborne**. In OoT they are jumps. What separates them is velocity direction relative to facing,
which state v2 carries (`vel`, `speed`, `shape_yaw`, and the shape-vs-world turn offset).

### Two documented techniques applied, both worked

1. **Masking a stale field.** `parallelYaw` is the aim yaw only while Z is held; unmasked it holds a stale
   value. Masking it by Z-active, in features, targets and the probe, took the head from **18.4° → 1.0°** —
   the same fix that took Baba-facing from 103° to 5°.
2. **Merging two routines that are one semantic state.** `3146184` is a 1–2 frame transition inside a
   sidehop. As its own class it scored recall **0.12** even after the `moves` scenario raised its density 30×.
   Merged into `airborne` (`LINK_ACTION_MERGE`), which then scores 0.98 on n=1,483.

### Generating the missing examples

`scripts/collect_scenarios.py` gained a `moves` scenario that cycles roll / backflip / sidehop / jump-attack /
shield / swing deliberately, phase known from `t`. 48 episodes, 22,364 transitions. Sidehop density went from
0.03% to **0.97%**; roll 1.87%. This is the same move that fixed the room-exit head, which had 0 examples
until scripted walkouts were collected.

### Result (`joint_v8_acts`, 209,523 frames, test slots)

| head | recall | n |
|---|---|---|
| roll | **1.00** | 29 |
| shield | **0.99** | 309 |
| airborne | **0.98** | 1,483 |
| airborne_z | **1.00** | 301 |
| hurt_airborne | **1.00** | 40 |
| hurt_still | **0.95** | 86 |
| locomotion | 0.92 | 2,897 |
| attack | 0.85 | 1,013 |
| **hurt** | **0.71** | 34 |

`link_action` 0.8787 over 25 present classes (majority 0.5744). `link_pos` MAE 9.43, `wall_yaw` 1.8°,
`row_state` 0.9961, `baba_count` 0.9998. Inverse dynamics **0.0878**, the best recorded.

### Still failing — flagged, not fixed

- **`bgCheckFlags` GROUND_TOUCH, recall 0.02.** Structural: the decomp comments it "only active for 1 frame",
  and the env samples at 10 Hz, so it is mostly invisible. Same class as `knockback_type`, which reads 0.00%
  across 282 damage events. Knockback is still represented, via the `hurt_airborne` state.
- **`hurt`, recall 0.71 on n=34.** Rare, and the `moves` scenario produces no damage (0 hits). The fix is a
  scenario that takes hits deliberately while recording — `suicide` does this but ends the episode quickly.
- **`attack` 0.85 / `locomotion` 0.92** are the two largest classes and both fell slightly as class count grew
  from 21 to 35.
- **Run-to-run variance is not negligible**: `shape_yaw` moved 0.7° → 5.3° and `speed` MAE 0.058 → 0.067
  between two runs differing only by the merge. Single-run differences of that size should not be read as
  effects.

### What this does not fix

The prediction problem is untouched. Decode answers "what is Link doing"; the dream needs "what happens
AFTER the action", and at h=1 `link_pos` MAE is 10.4 against identity's 12.3. Measured consequence: rolling
the trained policy inside the dream moves Link **250 world units in 100 steps**, while the same policy in the
emulator covers enough ground to leave the room in **21–91 steps**. So the dream never puts Link near the
door, `left%` stays 0.0% through training, and the −20 exit terminal is unreachable — while in the emulator
the policy leaves 3/3 times with the value function still reading V≈3.2 one step before the −20.

## 13. The predictor over-moved by 20x, and fixing it made room exit reachable

The dream's failure was diagnosed as Link moving too *little* (250 world units in 100 steps, against a real
policy that exits in 21–91 steps). Measuring the per-step distribution on held-out slots inverted that:

| per-step XZ displacement (100 ms) | mean | std | p50 | p90 |
|---|---|---|---|---|
| TRUE (real state) | **0.58** | 3.09 | **0.00** | **0.00** |
| decode round-trip, no prediction | 1.79 | 7.32 | 0.00 | 0.00 |
| PREDICTED | **12.11** | 6.79 | 12.52 | 20.88 |

Link is stationary on **95% of real steps**. The predictor moved him ~12 units on *every* step — 20.9× too
far — in incoherent directions, so net drift stayed near zero. Both observations are the same bug.

And the reason no predictor tuning could fix it: **absolute `link_pos` decodes to ~12 units of error while a
real step is 0.58 units**. The signal is ~20× smaller than the noise it was being measured through.

### Three changes, each an already-proven technique

1. **Residual predictor** `z + f(z, a)`, in the new `zelda_rl/dynamics.py` (five scripts had each
   constructed the predictor by hand). The old one mapped `(z, a) → z_{t+1}` directly, rebuilding all 128
   dims every step, so any reconstruction error became motion. The output layer is zero-initialised, so an
   untrained predictor is exactly the identity. The dynamics sweep had measured this on the frozen-BC latent
   as skill −7.24 → −1.73 and it had never been carried into the joint model.
2. **Zero-inflated displacement heads.** `link_moved` (classifier) + `link_dpos` (magnitude, trained *only*
   where Link moved). This is the HP treatment — regression → classification → tuned threshold, which took
   damage F1 0.286 → 0.906 — applied to a quantity that is 95% exact zeros. A regression head cannot emit
   zero; a gated one can.
3. **The dream integrates position** (`pos_t + dpos`, gated by `link_moved`) instead of reading the absolute
   head, so motion accumulates coherently.

### Two bugs the changes exposed

- **The selection metric became degenerate.** `skill = 1 − pred_err/identity_err`, and a zero-initialised
  residual predictor *is* the identity, so skill = 0 at epoch 0 — better than every learned value, which are
  all negative. Early stopping locked in the untrained predictor and froze the decoder at epoch 3 with it.
  Selection now uses the combined objective (decode + `w_pred`·pred); skill is reported, not optimised.
  Worth stating plainly: **the predictor ties identity, it does not beat it.**
- **`link_dpos` was unstandardised.** In raw world units (steps up to 44), its MSE swamped every other head —
  decode loss ran at ~450 instead of ~0.2 and absolute position regressed to MAE 29.5. Added to `CONT`.

### Result (`joint_v9_resid`)

| | before | after |
|---|---|---|
| per-step displacement vs true | **20.85×** | **2.52×** |
| `left%` reachable in the dream, any constant action | **0%** | **100%** (holding E / N / SE) |
| `hurt` recall | 0.71 (n=34) | **1.00** (n=160) |
| `link_pos` MAE | 9.43 | 12.46 |
| baba_count / row_state / link_hp | 0.9998 / 0.9961 / 0.9910 | 0.9992 / 0.9874 / 0.9857 |
| left-room recall (held-out exits) | 1.000 | 1.000, FP 0.00099 |
| inverse dynamics | 0.0878 | 0.0846 |

Room exit becoming reachable is the point: the −20 terminal could not fire during training, so the policy had
no reason to avoid the thing it then did 3/3 times in the emulator with V ≈ 3.2 one step before the penalty.

### Fixes 2 and 3

- **GROUND_TOUCH (recall 0.02) is no longer decoded.** The decomp comments it "only active for 1 frame" and
  the env samples at 10 Hz, so ~5 of 6 occurrences are not in the recorded state. Landing is derived instead
  from the airborne transition (`zelda_rl.combat.landed`), a field that decodes at 1.000/1.000 — the
  documented pattern that recovered blocks from lunge→pull_back cycles. GROUND_LEAVE and the redundant
  GROUND_STRICT dropped with it; collision decode is now ground/wall/ceiling/water/crushed/player_wall, all
  at ≥ 0.99 recall.
- **`hurt` fixed by merging, not by more data.** `hurt_still` and `hurt_airborne` are the same semantic state,
  and merged the class went 0.71 (n=34) → **1.00** (n=160). The planned extra `suicide` collection was not
  needed. Nothing in the reward depends on it anyway: the DAMAGED stateFlags1 bit decodes at 1.000/1.00.

### Still open

- **`MOVED` precision 0.295** at recall 0.797 — it over-predicts movement ~3.4×. The threshold was tuned on
  the tuning slots, where the moved base rate is **0.182**, but on the test slots it is **0.085**: the
  threshold does not transfer because the slots differ in how much Link moves. Tuning per-slot, or on a
  base-rate-matched subset, is the next step.
- Residual displacement over-move of **2.52×**, down from 20.85× but not 1×.
- `link_pos` MAE rose 9.43 → 12.46. The absolute head matters less now that the dream integrates, but it is
  still what grounds enemy distances in the reward.

## 14. Calibrating the movement gate

Three things were left open in §13. All three responded to techniques already used here.

### The threshold did not transfer; the gate was uncalibrated

The dream applies `E[dpos] = P(moved) · E[dpos | moved]`, which equals the truth only if **P is calibrated**.
It was not: trained with `pos_weight 2.0` (needed for the class imbalance), raw `sigmoid(logit)` averaged
**0.564** on the tuning slots where the true movement rate is **0.182** — 3× too high. That is why a soft gate
over-moved by 2.85×, slightly *worse* than a hard threshold, and why a threshold tuned at base rate 0.182 ran
at precision 0.295 where the base rate is 0.085.

Two fixes, both the documented "calibrated score, threshold chosen on the tuning slots" step that took
damage-event F1 from 0.746 to 0.906:

1. **Platt scaling on the tuning slots.** `P = sigmoid(0.450·logit − 2.829)`, fitted unweighted (calibration,
   not balance). Mean predicted P then matches the true rate exactly: **0.182 vs 0.182**.
2. **Report average precision, which is threshold-free.** Any single operating point is the wrong thing to
   quote when the base rate differs across slots. AP is what this repo already uses for rare events.

### Results

| | §13 | §14 |
|---|---|---|
| displacement ratio vs true | 2.85× | **1.93×** |
| `link_pos` MAE | 12.46 | **9.76** (`--w-pos` 3 → 6) |
| `MOVED` AP | 0.3464 | **0.4375** (5.1× base rate) |
| tune skill | −1.379 | −0.998 |
| `row_state` / `baba_count` / `link_hp` | 0.9874 / 0.9992 / 0.9857 | 0.9927 / 0.9996 / 0.9901 |
| action recalls (roll/shield/airborne/hurt) | 1.00 / 0.98 / 0.98 / 1.00 | unchanged |

### The residual 1.93× is slot shift, not a modelling bug

Calibration fitted on the tuning slots is *exact there* (0.182 vs 0.182) but the test slots move at 0.085, so
a calibration fitted elsewhere still over-predicts by about 2×. Slots differ in how much Link moves: the
tuning slots carry more scripted walking, the test slots are combat-only where he is mostly stationary. The
lever that would remove this is a better classifier — AP 0.4375 is weak — not more calibration.

### Methodology note, recorded because it was got wrong

`link_pos` MAE was reported as "rose 9.43 → 12.46". Three seeds of the same config give **12.46–17.23**
(median 13.85), so 12.46 was the bottom of a wide distribution and the single-run comparison overstated its
precision. Seed spread on this project has been as large as `shape_yaw` 0.7° → 5.3° between runs differing
only by a class merge. **A one-run gap of this size is not evidence.** The earlier capacity sweep (§10) was
single-seed and its between-config differences should be read with that caveat; its one load-bearing
conclusion — that down-weighting the v2 heads does not recover action recoverability — rests on a gap far
larger than the spread.

## 15. The gap for approaching and orienting, which never got the room-exit treatment

Room exit got scripted data, a dedicated head, exit-persistence and a constant-action verification sweep.
**Approach, orienting and lock-on got none of that** — only static decode accuracy, never a test of whether
the dream can represent *closing distance* or *acquiring* a lock-on as a consequence of an action. Those are
the shaping terms the reward pays: `approach` pays `w·(d0−d1)/100` while not locked on, `lockon` once per
acquisition (cap 6), `lock_hold` per step held.

### Constant-action sweep, 120 steps from all 10 start states

| held action | Δ dist to nearest Baba | `r_approach` | lock-ons (cap 6) | left% |
|---|---|---|---|---|
| best closer, `S+Z` | **−43.2** | **−0.04** | 0.00 | 0% |
| `E+Shield` | −34.7 | −0.02 | 0.00 | 0% |
| `N` (no Z) | **+7455** | −2.58 | 0.10 | 70% |
| best lock-on, `NE+Z+A` | — | — | **0.90** | 70% |

`r_approach` is ≈0 or negative for **every** action, and the best Z-press acquires 0.90 lock-ons in 120 steps
against a cap of 6. Both terms are effectively unearnable. The dream has two regimes: Z-held, which is nearly
frozen, and non-Z, which diverges by thousands of units and exits the room. Neither is "walk over and engage".

### Displacement direction, which had never been measured

| on truly-moving steps (held-out slots) | |
|---|---|
| median angular error | **26.7°** |
| mean angular error | 47.0° |
| within 30° / 45° / 90° | 53.3% / 63.0% / 80.2% (chance 16.7 / 25.0 / 50.0%) |
| cosine similarity | **+0.548** |

There is real directional signal — cosine +0.548, well above the 0 of noise — but 47° mean error accumulates:
path length over 120 steps is ~143 units while net displacement is ~43, a **tortuosity of 3.3×**. The dream
wanders instead of walking.

### The aggregate metric was hiding the real error

§14 reported displacement at "1.93× of true" and called that nearly fixed. Disaggregated:

| | dream | real |
|---|---|---|
| mean step over ALL frames | 1.76 | 0.91 → **1.93× too much** |
| mean step on TRULY-MOVING frames | **1.19** | **11.03** → **9.3× too little** |

Those are *opposite* errors that partially cancel in the aggregate: the gate fires on frames where Link should
be still (too often), while the magnitude on real moving frames is ~9× too small. A real held direction covers
~1324 units in 120 steps; the dream covers 143 of path and 43 of net. That ~30× shortfall in net approach is
the gap, and the 1.93× aggregate concealed it.

Same class of mistake as the pooled inverse-dynamics number in §10: **report the conditional metric, not the
aggregate.** Added to the probe as a standing check.

### Proposed fix, grounded in patterns that already worked here

1. **Classify displacement instead of regressing it.** Magnitude 9× too small on moving frames is textbook
   regression-to-the-conditional-mean under uncertainty — the same failure that made HP unusable until it was
   classified (0.286 → 0.906). Discretise into direction sector × magnitude bin and classify; a mode does not
   shrink. Direction sectors align naturally with the 9 stick directions of the action space.
2. **Predict displacement from `(z_t, a_t)` directly, not by decoding `z_{t+1}`.** Displacement direction is
   close to a function of the stick direction and the camera, but the action barely survives into the latent
   (inverse dynamics 0.089 against a 0.220 ceiling), so routing it through `z_{t+1}` passes the one quantity
   that needs the action through the weakest channel. This is exactly the event-head pattern already in the
   repo: *predict reward events from a latent transition instead of differencing a decoded state.*
3. **Scripted sustained-approach data.** `suicide` walks at the nearest Baba but dies in ~70 steps, so there
   are few long approach trajectories. A shielded-approach scenario (walk in with the shield up, survive) would
   give the sustained closing that room exit got from scripted walkouts.

## 16. Fixing approach and combat: an action-conditioned transition head, and the camera

### The combat audit, alongside the approach audit of §15

Same constant-action sweep, all 63 actions, 120 steps, all 10 start states:

| | max across all actions | available |
|---|---|---|
| kills | **0.00** | 3 |
| damage dealt | **0.70** | 6 (3 Babas × 2 HP) |
| hearts lost | 1.25 | 3 |

And the damage that appeared came from **Shield** actions (`W+Shield` 0.70), which cannot deal damage in the
real game. `r_edmg` was paying up to **+0.70 for decode noise**. `r_kill` was 0.00 everywhere. Damage taken
peaked on actions that also diverged +5819 world units, so it was coupled to garbage rather than to lunges.

Root cause, shared with approach: the reward reads HP and position out of a decoded state whose predictor ties
the identity. Nothing changes, so the only changes are decode noise — and noise gets billed as combat.

### An aggregate metric was hiding the magnitude error

§14 reported displacement at "1.93× of true". Disaggregated, that mixes two *opposite* errors:

| | dream | real | |
|---|---|---|---|
| all frames | 1.76 | 0.93 | 1.93× too **much** (the gate fires when Link should be still) |
| truly-moving frames | **1.19** | **11.03** | **9.3× too little** (the magnitude is shrunk) |

Same class of mistake as the pooled inverse-dynamics number in §10. Conditional metrics are now standing
checks in the probe.

### Fix: `TransitionHead(z_t, a_t)`

Two changes, both patterns that already worked here:

- **Condition on the action directly** rather than decoding motion and damage out of `z_{t+1}`. Displacement
  direction is close to a function of stick and camera; a sword hit is a function of swinging near and facing a
  Baba. But the action barely survives into the latent (inverse dynamics 0.089 against a 0.220 ceiling), so
  reading these from `z_{t+1}` routes the quantities that *need* the action through the weakest channel. This
  is the event-head pattern already in the repo.
- **Classify, do not regress.** 9.3×-too-small is regression to the conditional mean under uncertainty — the
  failure that made HP unusable until it was classified (damage F1 0.286 → 0.906). 16 direction sectors
  (aligned with the 9 stick directions), 8 magnitude bins, a 17-way "which row took damage", and a Link-hit bit.

| | regressed | transition head |
|---|---|---|
| step magnitude on moving frames | **0.11×** | **1.02×** (10.92 vs 10.68 units) |
| direction top-1 | — | 0.532 (chance 0.062) |
| magnitude top-1 | — | 0.538 (chance 0.125) |
| **enemy-hit AP** | n/a | **0.4411** vs base 0.0004 = **1103× base** |
| **Link-hit AP** | n/a | **0.7488** vs base 0.0014 = **535× base** |
| which row, on hit frames | n/a | 0.882 |

**Reward parity is preserved.** `apply_transition` edits the *state*; `CombatTracker` still computes all 18
terms from states exactly as the emulator run does. Nothing in the head computes reward.

### `approach` scenario

Close on the nearest Baba with the shield up, then back off and re-close, so trajectories are long. 70
episodes, 20,627 transitions, 55 deaths. `suicide` also walks in but dies in ~70 steps, which is why the
dataset had few sustained closing trajectories.

### State v3: the camera, and why approach still failed without it

With magnitude fixed at 1.02× and damage events predictable, the re-audit still showed approach failing —
best distance closed −18.4 units. The reason is structural, and checking it found a real hole:

- **Camera yaw was in no recorded state**, v1 or v2.
- **It was never an encoder input** — it existed only as a *decode target*, read from the stored 483-d obs.
- **The three scripted collections never stored obs at all**, so 122k of 230k frames had no camera. That is
  the "obs available on 46.6% of frames" line in the training log.

A stick direction in OoT moves Link **relative to the camera**. Without camera yaw the encoder cannot know
which world direction a held stick produces, which caps direction accuracy at roughly the camera's own
uncertainty — and camera yaw was itself decoding at 20–36°, right where direction top-1 stalled (0.532 over
22.5° sectors).

State v3 (363-d) adds `cam_yaw`, `cam_pitch`, `cam_eye` and, in the encoder, **heading relative to the
camera** — the quantity a held stick direction actually controls. `camera_yaw_sin_cos` was in the original
PPO's observation, so this **restores parity rather than adding to it**. All four datasets are being
re-collected as v3; every version only appends, so v1/v2 indices stay valid and the encoder width is a
constant 132 with zero-filled blocks for older states.

### The camera is the right reference frame, measured on the data alone

Before relying on state v3, the premise was checked directly on 12,260 moving steps of the v3 main
collection — no model involved, just which reference frame makes the realised walk direction predictable
(circular std, lower is better):

| reference frame | all steps | Z held | no Z |
|---|---|---|---|
| raw world direction | 135.9° | 142.4° | 129.9° |
| **minus CAMERA yaw** | **76.3°** | **75.2°** | **77.6°** |
| minus Link world yaw | 147.0° | 151.8° | 139.5° |
| minus Link shape yaw (body) | 158.6° | 187.9° | 142.3° |

Camera-relative nearly halves the spread, and does so equally with and without lock-on (Z is held on 53% of
moving steps). Link's own facing is **worse than raw**, so the camera is the correct frame and Link's facing
is not — the opposite of the intuitive guess. `cam_yaw` also varies strongly in the collected data (std
110.6°, 1,718 distinct values, heading-relative-to-camera std 90.9°), so it is not redundant with anything
already recorded.

The residual 76° is not explained by facing either; the likely remainder is turn/acceleration dynamics,
since Link rotates toward a new stick direction over several frames rather than snapping to it. Recorded as
the next thing to measure, not as something the camera was expected to fix.

### How much of the direction error is the model's, and how much is the task's

Before treating direction top-1 0.532 as a model failure, the achievable ceiling was estimated the same way
the 0.220 action-recoverability ceiling was: bucket the data by what the model is given and take the
most-frequent outcome per bucket. 12,862 moving steps, 16 sectors (chance 0.062):

| Bayes-optimal top-1 for the displacement sector, given | ceiling |
|---|---|
| nothing (predict the global mode) | 0.092 |
| stick direction alone | **0.113** |
| stick + camera sector | 0.352 |
| velocity sector alone | **0.466** |
| stick + velocity | 0.468 |
| **stick + camera + velocity — what the head sees** | **0.584** |
| *transition head, measured on held-out slots* | **0.532** |

The head is at **91% of the ceiling**, and since that ceiling is estimated over 1,602 buckets from 12,862
samples it is optimistic, so the head is closer still. **Per-step direction prediction is not the remaining
problem.** Two side findings: the stick *alone* is worth almost nothing (0.113 against a 0.092 global mode),
and velocity is the single strongest signal (0.466), stronger than the camera.

Why one step is intrinsically noisy: at 10 Hz a policy step spans 6 game frames, and Link rotates toward a new
stick direction over several of them, so a single step mixes turning with translating. Camera-relative spread
falls monotonically with how long the direction has been held — 79.3° (just changed) → 69.5° → 59.0° → 60.0°
(6+ steps) — and only **2.3%** of moving steps in the demonstrator data hold a direction 6+ steps, because the
policies re-sample almost every step. Displacement aligns best of all with Link's **current velocity** (58.2°).

### The remaining blocker for approach is momentum, not direction

Real walking is autocorrelated: Link keeps going the same way, and velocity is what carries that across steps.
In the dream, velocity comes from the decoded state, whose predictor ties the identity — so it goes stale,
each step's direction is drawn from a degraded `z_t`, and the path wanders (tortuosity 3.3×) even though every
individual prediction is near-optimal.

Note the architectural catch, recorded because it bounds the obvious fix: edits made by `apply_transition` go
into the **state**, while the head reads **`z`**, and `z` evolves only through the predictor. So writing a
corrected velocity into the state does not reach the next step's head. Two options, neither yet applied:

1. **Re-encode each step**: `z_{t+1} = E(edited state)` instead of `z_{t+1} = M(z_t, a_t)`. Edits then feed
   back, and the rollout stays on the data manifold because every `z` comes from a real encoder call on a
   valid state. The policy still consumes `z`, so the latent-RL premise survives; what changes is that the
   learned latent *transition* is no longer in the loop.
2. Give the head state-derived inputs (velocity) alongside `z`, so momentum reaches it without re-encoding.

Option 1 is the larger change and would make the dream a decoded-state model with a latent observation, which
is a different experiment from the one stated in §1. Not taken unilaterally.

## 17. Decoding the prediction fixes every single-step metric and destroys the rollout

§16 left the predictor supervised only by L2 to the true next latent. Reading the training loop showed why that
was never going to give the dream usable velocity: **`dec()` was only ever called on `E(real state)`**. The
predicted latent was never decoded, so nothing asked whether it decodes to the right velocity — and L2 in a
128-d latent is dominated by the high-variance directions, while velocity and position are tiny components.
The decoder had also never seen a predicted latent as input, which is all it is ever fed in a dream.

An earlier proposal to re-encode each step (`z_{t+1} = E(edited state)`) was **withdrawn**: it routes around the
problem and quietly removes the learned latent transition, which is the subject of the experiment.

### The fix, and what it bought

`L += w_decpred · decode_loss(dec(pred(z_t, a_t)), s_{t+1})`, plus training the transition head on predicted
latents for the same exposure-bias reason. Every single-step number improved:

| | §16 | + decode-the-prediction |
|---|---|---|
| enemy-hit AP (base 0.0002) | 0.3637 | **0.6270** |
| Link-hit AP (base 0.0014) | 0.6141 | **0.6652** |
| moved AP | 0.3976 | **0.4918** |
| `baba_count` from the PREDICTED latent | — | **0.9895** (vs 0.9524 from the real latent) |
| `dpos` MAE from the predicted latent | — | 0.6600 (vs 0.6815 from the real one) |

Decode of the predicted latent beating decode of the real latent is the loss doing exactly its job.

### And it made the rollout diverge

Latent norm under a 120-step autoregressive rollout, fixed action:

| step | without decode-the-prediction | with it |
|---|---|---|
| 1 | 12.8 | 16.4 |
| 10 | 10.2 | 88.2 |
| 50 | 10.8 | 23,817 |
| 120 | **13.8** | **1.7 × 10¹⁴** |

The audit went from "approach now earnable" to every one of the 63 actions giving `left% 100%`, distances of
10¹⁰–10¹² world units and returns near −1900.

The mechanism is exact: **to change a decoded field you must move the latent**, so the decode loss pushes
`f(z, a)` to be large — and applied one step ahead, nothing ever penalises applying `f` repeatedly. The
previous predictor was stable precisely because it was nearly the identity: bounded *and* useless. This one is
useful for one step and compounds geometrically.

So the two failure modes are the same coin. A predictor that ties identity is stable and cannot represent
anything; a predictor with enough authority to change the decoded state diverges unless its own compounding is
supervised. That is what `--unroll > 1` is for — it trains the predictor on its own output, and the
decode-the-prediction term is applied at **every** unroll step. It has been off by default all along.

Recorded as the clearest negative result of this line of work: **single-step decode quality and rollout
stability are in direct tension, and optimising either alone produces a useless dream.**

## 18. Unrolled training resolves the tension, and kills become possible

`--unroll 30` (detach every 8, decay 0.97, 97,893 usable starts), with the decode-the-prediction term applied
at every unroll step.

### Latent stability restored without giving up decode authority

| step | no decode-pred | decode-pred, unroll 1 | **decode-pred, unroll 30** |
|---|---|---|---|
| 1 | 12.8 | 16.4 | 15.3 |
| 25 | 9.9 | 295 | 18.2 |
| 50 | 10.8 | 23,817 | 22.4 |
| 120 | 13.8 | **1.7 × 10¹⁴** | **44.5** |
| `max‖pos‖` at 120 | 471 | 5 × 10¹⁵ | **669** |

Positions stay in plausible room coordinates. And the head metrics did **not** trade away — they improved:

| | unroll 1 | unroll 30 |
|---|---|---|
| enemy-hit AP (base 0.0002) | 0.6270 | **0.8091** |
| moved AP | 0.4918 | **0.7243** |
| `baba_count`, predicted / real latent | 0.9895 / 0.9524 | **0.9906 / 0.9906** |
| `speed` MAE, predicted / real | 0.0806 / 0.0375 | **0.0597 / 0.0705** |
| tune skill | −1.904 | **−1.203** |

So the tension of §17 was not fundamental — it was the absence of multi-step supervision. Decode authority and
rollout stability are compatible once the predictor is trained on its own output.

### Audit progression across the whole line of work

| | v10 | v11 | v12 (+camera) | v14 (+decode-pred, unroll) |
|---|---|---|---|---|
| **max kills** (3 available) | 0.00 | 0.00 | 0.00 | **0.30** |
| max damage dealt (6 available) | 0.70 | 0.90 | 0.90 | 0.50 |
| damage attributed to | **Shield** (impossible) | Z+A | Z+A | Z+A / Z+B |
| best distance closed | −43 | −18 | −107 | **−136** |
| best `r_approach` | **−0.04** | +0.13 | +0.18 | **+0.21** |
| step magnitude, moving frames | 0.11× | 1.02× | 0.99× | 1.05× |

**Kills are non-zero for the first time**: `-+Z+A` (a standing jump attack) reaches 0.30 kills and pays
`r_kill` +1.50. The despawn head plus calibrated sampling is what made two hits on one row possible.

### What is still broken, stated plainly

- **Kills 0.30 of 3.** Non-zero, not competent. Clearing a room is still far away.
- **Lock-on: max 0.70 against a cap of 6.** A Z-press still rarely acquires, so `lockon` and `lock_hold`
  remain nearly unearnable. This got an audit but never a fix.
- **`r_blk` is 0.00 for every action** — blocks and dodges never fire. Never audited either.
- **Returns are all negative** (−3.8 to −697), and `hpLost` reaches 1.9 hearts on the movement actions, so
  approaching is still punished more than it pays.
- **`left%` is 10–90% on the approach-heavy actions.** Walking still frequently exits the room.
- **The predictor still does not beat identity** (skill −1.203). Every gain here is in what the model can
  represent, not in its one-step accuracy.

## 19. PPO in the v14 dream: 80% of episodes end in a penalty the policy never caused

Run [wandb `930i7qsx`](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/930i7qsx), killed at update 260
of 2669 (4.0 M of 41 M dream transitions).

| upd | return | len | clear% | death% | left% | kills | hpLost |
|---|---|---|---|---|---|---|---|
| 1 | 17.54 | 40 | 54.8% | 45.2% | 0.0% | 0.55 | 0.40 |
| 50 | −9.50 | 277 | 17.9% | 38.1% | 43.9% | 0.30 | 0.95 |
| 150 | −5.04 | 326 | 22.8% | 36.8% | 40.5% | 0.34 | 0.68 |
| 250 | −7.55 | 400 | 19.4% | 37.4% | 43.2% | 0.31 | 0.68 |
| 260 | −7.33 | 427 | 18.2% | 41.0% | 40.8% | 0.27 | 0.77 |

Return did not improve across updates 100–260, and **~80% of episodes ended in a −20 terminal**.

### The arithmetic that gave it away

`death% 37–41%` alongside `hpLost 0.68–0.77` hearts is impossible: death requires 3 hearts, so 37% dying implies
at least 1.1 hearts lost on average. Something was killing Link without damaging him.

**First hypothesis, wrong and retracted.** `CombatState.dead` is `hp <= 0 OR sf1 bit 7`, and
`decoded_to_states` sets that bit from a raw logit thresholded at 0 — the same shape as the `link_hit_thresh`
bug. Measured over a 400-step autoregressive rollout from all 10 start states: **the DEAD bit never fires
once**. The hypothesis was wrong.

### What is actually happening

The decoded state **decays into a "dead and left the room" attractor**:

| step | `scene != 0` | decoded HP mean | `hp == 0` |
|---|---|---|---|
| 25 | 0/10 | 41.6 | 0/10 |
| 50 | 0/10 | 41.6 | 0/10 |
| 100 | 1/10 | 28.8 | 1/10 |
| 200 | 5/10 | 25.6 | 4/10 |
| 400 | **10/10** | 24.0 | **5/10** |

Link is not dying of damage and not walking out. The **HP head drifts to 0** and the **scene head drifts to
"left"** as the latent leaves the data manifold. That is precisely the 41% / 41% in the log.

And the timing is the finding: **it is stable through step 25–50 and breaks immediately after.** Training used
`--unroll 30`. The rollout is stable exactly as far as it was supervised and no further. §18's bounded `|z|`
(44.5 at step 120) was necessary but **not sufficient** — a bounded latent can still drift into a region that
decodes as death and exit.

This is §11's rollout collapse again, now localised to the two heads that drive terminals, and it caps the
whole approach: dream episodes average 400+ steps against a supervised horizon of 30.

### Options recorded

1. **Unroll to the horizon actually used** (`--unroll 120`+, decode-the-prediction already applies at every
   step). Cost is linear: unroll 30 took ~25 min, so ~100 min at 120. Targets the mechanism directly.
2. **Cap dream episodes at the supervised horizon** (~100 steps). Cheap and honest, but abandons the
   1800-step parity with the real environment, and a 100-step episode cannot contain a full fight.
3. An anchor penalty keeping predicted latents near the encoder's distribution — plausible, but not a
   technique validated on this project, so not led with.

## 20. The fix: HP and the enemy roster are counters, not per-frame observations

§19's drift was not a training problem. `decoded_to_states` re-read **Link HP, enemy HP, the enemy roster and
`scene` from the decode heads on every step**, so whatever the event heads did was immediately overwritten.
The events were cosmetic. Position was the only quantity being carried forward.

HP is a counter: it changes when something hits, not continuously. Same for the roster — actors do not respawn
mid-fight. So `apply_transition` now carries them and lets only events change them:

- `cu.hp = pv.hp`, minus half a heart on a sampled `link_hit`
- enemy HP carried per pointer, minus 1 on a sampled `hit_row`; a row leaves only via `gone_row` or the
  HP-0 despawn backstop; anything the decoder invents is dropped
- `scene` is absorbing, and gated on proximity to a real door

Door positions are data, not a guess: the XZ of the last in-room frame of all **71** real `left_room` episodes,
in `datasets/transitions/exit_points.json`. They cluster at z ≈ 800 (x −66…66) and z ≈ 0.

### Measured, 50 episodes of random play at 400 steps

| | before | after |
|---|---|---|
| deaths | 2/10 | **0/50** |
| mean hearts lost | 1.10 | **0.00** |
| earliest spurious exit | step ~60 | **step 138** |
| spurious exits within a 100-step cap | — | **0/50** |
| within 150 | — | 2/50 |
| within 300 | — | 25/50 |

**Spurious deaths are gone.** Room exit is the one artifact carry-forward cannot remove, because `scene` has no
event head and the door gate cannot help the slots that *start* beside a door — but it is now bounded: nothing
spurious before step 138, against a supervised unroll horizon of 30.

So the trustworthy horizon is ~130 steps, not 30. Two honest consequences:

1. A dream episode capped near 120 steps is clean by measurement (0/50 artifacts), and that is a defensible
   operating point rather than a fudge.
2. It is **not** the 1800-step parity with the real environment, and a 120-step episode cannot contain a full
   three-Baba fight. Choosing it trades task completeness for model validity, and that is a decision about the
   experiment, not a bug fix.

Recorded correction to §19: I proposed ~100 minutes of unroll-120 training for this. It was not needed — the
defect was four lines of state handling, and the diagnosis that "the rollout is stable exactly as far as it was
supervised" was wrong: with correct state handling it is stable to ~4x the supervised horizon.

## 21. The spurious terminals were `fell`, not the scene head

§20 fixed HP and the roster but left room exit "bounded, not eliminated". Chasing it properly took three wrong
turns worth recording, because each was an assumption made instead of a measurement.

**Wrong turn 1: a door gate.** Gating the exit on proximity to a real door (71 door positions extracted from
every real `left_room` episode) changed nothing — false fires per 400 steps were 1.504 with the gate against
1.511 without. The head's false fires *already* happen beside doors. Dead code, removed.

**Wrong turn 2: an exit event head, sampled.** Adding `tr_exit` to the transition head and sampling it made
things *worse in timing*: exits/50 fell 31 → 20 but the earliest moved 138 → 112 and the median 226 → 160. A
Bernoulli draw fires from step 1, whereas drift needs time to build.

The rate was wrong for a specific reason. Calibrated to the dataset marginal, the head fired at 0.00094/step,
against a **real** random-play hazard of **0.000052/step** measured from the `flail` episodes — 18× too high.
The collections are deliberately enriched with scripted walkouts, so their marginal is not the world's.

| | per-step exit hazard | over 400 steps |
|---|---|---|
| real emulator, random actions | **0.000052** | **2.1%** |
| training-data marginal (walkout-enriched) | 0.00024 | 9.2% |
| the calibrated head | 0.00094 | 31.4% |

The head itself is fine — on real exit trajectories it scores **AP 0.6514 at 103× base rate, with 59 of 71 exit
frames in the top 1% of scores.** It knows where the door is. Only the score-to-event conversion was wrong, so
it became a threshold rather than a sample: exit is **absorbing and once per episode**, where over-firing ends
episodes that should continue, while damage is repeatable, where under-firing is what kept kills at 0.00.

**Wrong turn 3, and the actual cause.** With the threshold in, the proof was byte-identical: 20/50 exits at the
same steps. The exit logit in a dream rollout never approaches its threshold — it sits at −13 and falls to −103.
The exit head was firing *never*. `_left_reason` has **four** conditions and only `scene` had been examined:

```python
if cur.scene != self.start_scene: ...        # fixed
if cur.room  != self.start_room:  ...        # room still re-decoded every step
if cur.pos[1] < start_pos[1] - fall_drop     # y integrates an unbounded dy
if hypot(dx, dz) > room_radius    ...
```

Instrumenting it: **all 20 were `fell`**, at a mean y drift of −150.9 against a threshold of −150.0. `dy` is an
unbounded regression added every step, and a bias of ~−1.3 units/step walked Link through the floor.

Falling is not a real outcome here: of **70** real room exits, **70 are scene transitions and 0 are falls**, and
the real per-episode y range is mean 15.1 with a 95th percentile of 38.1. Clamping y to ±60 of the episode's
start preserves every real jump and removes the drift.

### Result: 50 random 400-step episodes, zero spurious terminals

| | v14 | v15 + exit head | **+ y clamp** |
|---|---|---|---|
| spurious deaths | 0/50 | 0/50 | **0/50** |
| spurious exits | 31/50 | 20/50 | **0/50** |
| outcomes | — | — | **50/50 timeout** |

And the dream is not inert — the audit is physically coherent for the first time:

| held action | dmg | kills | `r_kill` | Δdist | `r_approach` | left% | return |
|---|---|---|---|---|---|---|---|
| `S+Z+A` (attack in place) | **1.50** | **0.80** | **+4.00** | +0.6 | 0.08 | **0%** | **+7.15** |
| `NW+Z+B` | 1.30 | 0.70 | +3.50 | −37 | 0.03 | 10% | −105 |
| `NE` (hold a direction) | 0.00 | 0.00 | 0.00 | **−147** | +0.25 | 60% | −776 |
| `E+Z+A` | 0.00 | 0.00 | 0.00 | −96 | **+0.33** | 80% | −1023 |

Holding a direction walks out of the room, which is what the scripted walkout scenario does and therefore
correct rather than spurious. Attacking in place kills and **pays a positive return** — the first held action in
this line of work to do so. Progression of max kills across the audits: 0.00 → 0.00 → 0.00 → 0.30 → **0.80**.

## 22. The metric that was missing: does the dream reproduce a real trajectory?

Every measurement before this was either single-step (decode accuracy on real latents) or self-referential
(does the dream avoid spurious terminals, does a held action close distance). None asked the only question
that matters for planning in a model: **replay a real episode's actions from its real start state and see
whether the dream goes where the game went.**

`scripts/eval_dream_rollout.py`, 12 held-out episodes of ≥ 600 steps, same start, same action sequence, always
against the identity control.

### Position: worse than identity at every horizon past 25

| h | dream err | identity err | travelled dream / real |
|---|---|---|---|
| 5 | 36.3 | 35.4 | **4 / 53** |
| 25 | 46.2 | **41.6** | 20 / 63 |
| 100 | 59.5 | **49.7** | 72 / 88 |
| 200 | 85.0 | **49.7** | 109 / 88 |
| 1800 | **102.7** | **49.7** | **130 / 88** |

The motion profile is wrong three separate ways: **13× too slow early** (4 units travelled against 53 by h=5),
then over-travelling (109 against 88 by h=200), then **frozen** — 130 units at h=200 and still 130 at h=1800.
Final position error 103 units against 50 for assuming Link never moved at all.

### Key events: nothing happens

| h | kills dream/real | damage dream/real | exited dream/real |
|---|---|---|---|
| 100 | 0.00 / 0.08 | **0.00 / 0.17** | 0.00 / 0.00 |
| 400 | 0.00 / 0.08 | **0.00 / 0.17** | **0.25 / 0.00** |
| 1800 | 0.00 / 0.08 | **0.00 / 0.17** | 0.00 / 0.00 |

On real action sequences the dream produces **zero damage and zero kills**, and still exits spuriously in 25%
of episodes that never exit.

### This reframes §18–21 as measuring scaffolding, not the model

The raw latent rollout (`scripts/eval_rollout.py`, absolute decode, no transition head) is worse still: at
h=1800 `link_pos` MAE is **10,030** world units against an identity baseline of **26.3**, and `link_hp` decodes
at 0.077 against 0.923. The dream only looks better because `apply_transition` patches it — carried HP, carried
roster, integrated displacement, event heads. Those are scaffolding around a latent transition that does not
model trajectories past ~25 steps.

The constant-action audit that showed `S+Z+A` reaching 0.80 kills and a +7.15 return was therefore measuring
**which action triggers the event head**, not whether combat is reproduced. Fed real action sequences, the same
machinery produces nothing. Progression of "max kills" across §16–21 (0.00 → 0.30 → 0.80) describes the event
head's responsiveness to held inputs, not the model's fidelity.

**Stated plainly: the world model does not yet model walking to a location or interacting with an enemy.**
Room exit, which §19–21 spent the most effort on, was the least important of the three — it is a terminal, not
the dynamics.

## 23. Event oversampling for the transition objective

### The defect

`transition_frames()` — the oversampler that took damage-event F1 from 0.458 to 0.746 for the decode heads —
was wired only to the HP heads decoded from **real** latents (`train_joint.py` lines 203–204). Batches for the
**transition** objective were drawn uniformly (`rng.permutation(tr_all)`), over data where an actor is hit on
**0.13%** of steps, a row despawns on 0.05%, Link is hit on 0.35% and the room is left on 0.035%. So the
predictor's loss was computed almost entirely on frames whose correct answer is "nothing changed", and three
separate gradients all point there:

| term | weight | minimised by |
|---|---|---|
| `w_pos` position | 6.0 | predict no movement (real per-step median is 0.0) |
| latent MSE to `E(s_next)` | 1.0 | predict `z_{t+1} = z_t` |
| event heads inside the decode loss | 1–3 | predict the 99.9% majority class |

### The fix and its measured effect

`--oversample K` weights rare-event frames by K and motion frames by sqrt(K), for both the batch sampler and
the choice of unroll-window starts. At K=50 the composition of a batch changes from 0.571% rare events and
17.1% motion to **14.9%** and **73.3%**.

Validated on **event-rich windows** (`scripts/eval_event_windows.py`), which is the only place this shows,
because whole held-out episodes are event-poor — identity scores 0.92–1.00 there, so a frozen model looks fine.
Hit windows are selected so the real trajectory contains a Baba HP drop within the horizon, hence real
detection is 100% by construction. Horizon 40, 4 epochs, unroll 8:

| hit windows | v15 baseline | oversample off | **oversample ×50** | real |
|---|---|---|---|---|
| Baba HP removed | 0.04 | 0.40 | **1.15** | 1.25 |
| event detected | 2% | 20% | **55%** | 100% |

Detection 2% → 55% and near-correct magnitude, from four epochs. **Kept.**

It did **not** help walking — travel over walk windows went 95.3 → 63.0 against a real 300.8 — despite motion
rising to 73% of each batch. So walking is not a sampling problem, and the "poor sampling" diagnosis explained
only the event half.

### Objective table to fill in

One row per reward-relevant quantity, each validated the same way: event-rich windows, dream vs real vs
identity, before and after. Blank means not yet measured this way.

| objective | window kind | metric | before | after oversample | bar | status |
|---|---|---|---|---|---|---|
| enemy damage | `hit` | Baba HP removed / detection | 0.40 / 20% | **1.15 / 55%** | real 1.25 / 100% | improved |
| trajectory, hit windows | `hit` | mean position error / identity | 1.21x | **1.11x** | identity 1.00x | improved, still worse than identity |
| trajectory, Link-hit windows | `link_hit` | mean / END position error | 1.41x / — | **1.11x / 0.91x** | identity 1.00x | **end position beats identity** |
| **walking** | `walk` | mean position error / identity | 1.37x | **1.38x** | identity 1.00x | **broken, and NOT a sampling problem** |
| Link damage magnitude | `link_hit` | hearts lost | — | — | — | metric buggy (see below) |
| kills | `kill` | despawn + HP 0 | — | — | — | window kind not implemented |
| blocking | `block` | blocks + dodges | — | — | — | never audited; `r_blk` has been 0.00 in every run |
| lock-on | `lockon` | acquisitions | — | — | — | never audited; tops out at 0.70 of a cap of 6 |

Two known metric defects to fix before trusting their rows:

1. **`link_hit` is inconsistent** — it reports real detection 0% on windows selected *for* containing a real
   Link HP drop, so the window/endpoint comparison is wrong.
2. ~~`walk` measures the wrong thing~~ **FIXED.** It now reports position error at every step against the
   real position, with identity being Link frozen at the window's start. On that metric the model is **1.38x
   worse than not moving Link at all** over walk windows, and oversampling does not shift it (1.37 -> 1.38).
   Cumulative path length, the old metric, was misleading in both directions: it can match while Link is
   nowhere near the right place.

### Where this leaves walking

Confirmed on the correct metric: the model does not reproduce trajectories. Over windows where Link genuinely
walks 120+ world units, the dream's position is 38% further from the truth than simply freezing him. Event
oversampling raised event fidelity and trajectory-near-events, and did nothing here -- so the next lever is
not sampling. The displacement head itself predicts magnitude at 1.05x of true and direction at 91% of its
achievable ceiling (section 15), which means the per-step pieces are close to right and the error is in how
they compose over time -- direction errors of ~27 deg median accumulate into a 3.3x tortuosity.

## 24. Position and events compete for the latent, not for head capacity

Weighting Link's position heavily in the decode-the-prediction loss (`--w-pos`) was A/B'd at 6 / 60 / 200,
everything else fixed (oversample 50, unroll 8, 4 epochs), judged on event-rich windows at horizon 40.

| `w_pos` | walk trajectory | hit trajectory | link_hit trajectory | damage detected | Baba HP removed |
|---|---|---|---|---|---|
| 6 | 1.38x | 1.11x | 1.11x | **55%** | **1.15** / 1.25 |
| 60 | 1.20x | 1.07x | 1.11x | 25% | 0.35 |
| **200** | **1.06x** | **1.00x / end 0.94x** | **0.93x / end 0.89x** | **0%** | **0.00** |

(ratios are against identity = Link frozen at the window start; below 1.00 beats it)

**Heavy position weighting fixes trajectory.** Walking goes 1.38x -> **1.06x**, and on both event window kinds
it beats identity for the first time in this project (hit end 0.94x, link_hit 0.93x mean / 0.89x end).

**And it destroys damage**: detection 55% -> 25% -> 0%, HP removed 1.15 -> 0.35 -> 0.00.

### The contention is in `z`, not in the heads

An earlier note in this log proposed "separating the position pathway from the event heads". That was wrong:
they are **already** separate heads, and `TransitionHead` even has its own 2x512 trunk independent of
`StateDecoder`. Nothing is shared at the output.

What is shared is the **128-d latent and the single predictor**. `w_pos` applies to `link_pos` / `row_pos`
inside `decode_loss`, which is computed on `dec(pred(z, a))` -- so that gradient flows back through the
predictor *and the encoder* and changes what `z` encodes. At `w_pos 200` the latent becomes a position-accurate
representation and the event structure that `TransitionHead(z_t, a_t)` reads out is no longer present. Damage
detection collapses without a single event weight being touched.

Both objectives are individually achievable on this data -- 1.15/1.25 HP removed at low `w_pos`, 0.93x
trajectory at high `w_pos` -- just not simultaneously in one 128-d latent. So the loss weights are deciding
what the latent is *about*, and separate heads cannot help when the representation they read has been
repurposed.

Two candidate fixes, neither yet tested:

1. **Widen `z`.** The obvious one, but the evidence against it is stale: the width sweep in section 10 found
   decode saturating at 128 and predictability degrading with width, and that was single-seed, predates the
   decode-the-prediction objective, and was never evaluated on event-rich windows.
2. **Detach the encoder for the position term**, so position is decoded *from* whatever `z` is rather than
   dictating what `z` becomes. This isolates the mechanism: it tests whether the events need the capacity or
   merely need it not taken away.

### Detach test: the competition is for capacity, and it is not avoidable this way

`--detach-pos` computes the position term on `dec(z.detach())`, so a large `--w-pos` cannot reshape the latent.
If the conflict in section 24 were merely about gradient direction, events would survive and trajectory would
hold. Same setup, 4 epochs, event-rich windows at horizon 40:

| config | Baba HP removed | detected | walk trajectory | hit trajectory |
|---|---|---|---|---|
| `w_pos 6` | 1.15 / 1.25 | 55% | 1.38x | 1.11x |
| `w_pos 60` | 0.35 | 25% | 1.20x | 1.07x |
| `w_pos 200` | **0.00** | **0%** | **1.06x** | **1.00x** |
| **`w_pos 200` + detach** | **0.85** | **45%** | **2.36x** | 1.09x |

Detaching **recovers damage almost fully** (0.00 -> 0.85, 0% -> 45%) and **wrecks walking** (1.06x -> 2.36x,
worse than any other configuration). So the hypothesis is **falsified**: `z` has to genuinely encode position
for trajectories to work, and detaching removes precisely the mechanism that made trajectory work.

The competition is for latent capacity, and it cannot be dodged by controlling where the gradient flows. The
remaining lever is width, which needs re-testing because the section 10 evidence against it is stale --
single-seed, predating the decode-the-prediction objective, and never evaluated on event-rich windows.

Current best-known operating points, all at latent 128:

| want | config | cost |
|---|---|---|
| events | `w_pos 6`, oversample 50 | walking 1.38x of identity |
| trajectory | `w_pos 200`, oversample 50 | damage dead (0%) |
| a compromise | `w_pos 60` | walking 1.20x, damage 25% |
| events with position read-only | `w_pos 200` + `--detach-pos` | walking 2.36x, worst trajectory |

## 25. Seed variance is larger than most of the effects reported above

40 training runs varying `--oversample` {10,50,150}, `--w-dpos` {0.5,2,8} and `--w-trans` {0.5,2,8} one lever at
a time from the baseline (oversample 50, w_dpos 2, w_trans 2), each evaluated on event-rich windows.

**Null result: none of the three levers beats the baseline.** Across 32 exact rank tests the smallest p is
0.036 at n=3, which is what 32 tests produce by chance.

### The number that matters is the baseline against itself

Re-running the **identical** baseline config at six seeds:

| metric | range across 6 seeds of the SAME config |
|---|---|
| damage detection | **0.25 – 0.55** |
| Baba HP removed | **0.50 – 1.15** |
| walk trajectory ratio | **1.38x – 3.43x** |

Every alternative setting's mean lands inside that band.

### Consequences for results reported earlier in this document

1. **The section 23 oversampling figures are the lucky seed.** "detection 55%, HP removed 1.15, walk 1.38x" is
   seed 0, and seed 0 is the best of six seeds on all three metrics simultaneously. The typical `--w-pos 6`
   model is detection ~0.50, HP ~0.9, **walk ~2.2x**. The oversampling effect itself (0 -> 50) is probably real,
   because it was far larger than this spread, but the specific numbers should not be quoted.
2. **The section 24 Pareto frontier needs re-basing.** The headline was `w_pos 200` giving walk 1.06x against
   `w_pos 6` giving 1.38x. If the typical `w_pos 6` is 2.2x, then either the gap is larger than reported or the
   `w_pos 200` figure is itself a lucky draw -- both single-seed. The *direction* of the trade-off was
   reproduced at three w_pos levels and by the detach test, so the trade-off stands; its magnitude does not.
3. **Single-seed results here actively mislead.** At seed 0, every lever looked like it worked: `w_dpos 8` gave
   detection 0.80 vs 0.55, `oversample 10` looked like it wrecked trajectory, `oversample 150` looked like it
   killed detection. All vanished at n=5-6. The sharpest case: `OS150 + WD8` looked at three seeds like the one
   real win (baseline detection, walk 1.47x, best HP calibration) and at six seeds was detection 0.35, walk
   1.96x, p=0.24.

### One actionable failure mode

3 of 40 runs produced **exactly** 0.00 detection -- a dead event head, not a weak one -- and two of those three
were `--w-dpos 0.5`. Do not lower `w_dpos`. Recorded as a failure mode rather than a rate, since 2 of 3 is not
an established frequency.

### What this means for method

The harness cannot resolve anything smaller than a ~0.30 detection swing at 4 epochs / 20 windows / 1 seed. The
two levers that did move it (oversampling 0 -> 50, and `w_pos`) had effects far bigger than the spread, which is
exactly why they were visible at one seed. Further sampling or weighting work is not worth running until
variance is reduced -- more epochs, seed averaging, or more eval windows per kind -- because no untested lever
appears to have an effect that large.

## 26. Control echo in the encoder input, and the frozen-BC premise re-tested without it

### The leak

The encoder was fed 11 features that are near-copies of the controller: `action_fn_rel` (Link's roll / jump /
attack / shield state), `melee_active`, `melee_anim`, `ztarget_active`, `ztarget_timer`, `parallel_yaw`, and the
`carrying` / `z_targeting` / `jumping` / `freefall` stateFlags1 bits. The state at step t is read BEFORE action t
is applied, so these reflect action t-1 -- and the demonstrator holds buttons for many consecutive steps, so
knowing t-1 largely predicts t.

Three things this contaminated:

1. **The original behaviour-cloning objective.** It was trained to predict the action while being fed fields
   that echo the previous one, so it could score by reading "what I was just doing".
2. **The inverse-dynamics analysis.** The "single latent" floor of 0.056, against which the pair value of
   0.059-0.139 was compared to argue that the action leaves no trace in a latent transition, was itself
   partly action echo sitting in `z_t`. It was never a neutral baseline.
3. **The decode numbers quoted as achievements.** `link_action` 0.918 and `melee_state` 0.998 are largely
   reading the control back out. They were listed alongside `baba_count` 0.9986 as if equivalent.

`featurise(..., control_echo=False)` zeroes those 11 columns (indices 24, 28, 31, 32, 66-72), leaving every
other column bit-identical, and `train_bc_decode.py --no-control-echo` uses it.

### Behaviour cloning without the echo

12 epochs on the v3 policy dataset, BC + the same decoder pressure:

| epoch | action CE | decode |
|---|---|---|
| 6 | 3.205 | 0.136 |
| 9 | 3.046 | 0.085 |
| 12 | **2.966** | **0.068** |

Uniform guessing over 63 actions is ln(63) = 4.14, so 2.966 (perplexity ~19.4 effective actions) is real
learning from the world rather than from the echo. Both losses were still falling at epoch 12.

### Then frozen as a predictive surface -- and it is far worse

`train_joint.py --enc-from` loads that encoder, freezes it, keeps its normaliser and its control-echo setting,
and trains only the predictor / decoder / transition head under the world-model objective: current latent +
action -> next latent, with pressure that the next latent decodes to the correct NEXT raw values.

| | frozen BC encoder (echo-free) | learned encoder (joint) |
|---|---|---|
| tune skill | **-5.345** | ~-1.4 |
| walk trajectory vs identity | **6.49x** | ~1.2-2.2x |
| hit trajectory | 4.96x | ~1.1x |
| lockon trajectory | 3.59x | ~1.09x |
| outcome detection (lockon / roll / collision) | 0% / 0% / 0% | ~15% / 75% / 15% |

`jump` detection reads 100%, which is the action echoing through the conditioning -- the model is *given* the
action -- so it is a check that conditioning works, not evidence of prediction.

**Conclusion: the frozen-BC premise does not hold, and the leak does not explain why.** A latent shaped by
"predict the action" plus "decode the present" is not organised for "what happens next"; letting the prediction
objective shape the encoder is strictly better. This reproduces the original section 4 result (frozen BC latent
skill -7.24 / -1.73 against joint -1.14) with the contamination removed, so that earlier finding stands for a
cleaner reason than was known at the time.

Caveat: 12 epochs, single seed. Section 25 established this harness cannot resolve small differences, but
6.49x against ~1.4x is far outside that band.

## 27. Factored control input and one unified decoder: the best world model so far

Two changes, both answering "is the world model actually shaped like a world model".

**1. The control was a flat one-hot.** The predictor was fed `one_hot(a)` over the 63 combined actions, which
are the cross product of 9 stick directions and 7 button combos. Nothing in that encoding says action 5 and
action 12 share a stick direction, so the factorisation had to be rediscovered for every pair.
`--factored-action` feeds the stick direction as a **unit vector** plus **one flag per button**
(`zelda_rl.dynamics.FactoredPredictor`, 9 inputs), so a direction seen with one button generalises to the same
direction with another. Camera is deliberately not an action input -- this action space has no camera control,
so camera reaches the model through the v3 state.

**2. ~18 separate decode heads became one.** `--unified-decoder` (`zelda_rl.bc.UnifiedDecoder`) uses one trunk
and one output layer sliced by a spec, with the **activation** differentiating target kinds: `logits` raw for
cross-entropy, `bits` raw for BCE, **tanh** for sin/cos pairs so they are bounded to [-1, 1], raw for
continuous. Output keys are identical to `StateDecoder`'s so every consumer works unchanged.

### Result, 12 epochs, frozen nothing, oversample 50, unroll 8

| | frozen BC encoder | joint v15 (one-hot, multi-head) | **factored + unified** |
|---|---|---|---|
| tune skill | -5.345 | -1.428 | **-0.621** |
| walk trajectory vs identity | 6.49x | 1.38x - 3.43x (6-seed range) | **1.14x** (end **1.07x**) |
| hit trajectory | 4.96x | 1.11x | 1.20x (end 1.14x) |
| **collision trajectory** | — | 0.92x | **0.96x / end 0.92x** |
| jump trajectory (end) | — | — | **0.99x** |
| block trajectory (end) | — | 1.09x | **1.03x** |
| lockon / roll (end) | — | — | 1.07x / 1.06x |

Skill **-0.621** is roughly twice as good as any previous model, and every trajectory ratio is now <= 1.24x with
several at or below 1.0. Crucially, section 25 measured the old config's walk ratio across six seeds as
1.38x-3.43x, and 1.14x sits **below that entire range** -- so this is not a lucky draw.

Caveats: single seed, 12 epochs with the loss still falling, and the `hit` / `kill` rows still rest on 8 and 7
windows so they are not evidence. `roll` detection fell 75% -> 42% and `jump` 50% -> 30%, but section 26
established those were largely the action echoing back through the conditioning, so a drop there is not clearly
a regression.

All rollout and dream consumers now honour the checkpoint's decoder kind (`unified_decoder`) and predictor kind
(`pred_factored`), so old checkpoints still load.

## 28. Null result: the competent policy's latent is not a better predictive surface

The premise: `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` reached return 55.5 / 76% success on Deku
Baba combat, so its internal representation is demonstrably sufficient to ACT. Take it frozen as an encoder
and measure predictability. The dataset already stores the 483-d `obs` that policy consumes, so it replays
straight through `enc._features -> proj` with no retraining.

Measured on 120k frames of v3-main, 80 episodes, split by episode, the same skill definition train_joint.py
reports (`1 - pred_mse/identity_mse`, identity = the zero-init residual predictor):

| latent | skill | move/rand | pos R2 (ridge) | pos MAE |
|---|---|---|---|---|
| policy `proj`, as-trained          | +0.742 | 0.287 | 0.993 |  4.4 |
| policy `proj`, echo-zeroed         | +0.240 | 0.059 | 0.994 |  4.1 |
| policy `proj`, echo+vel-zeroed     | +0.053 | 0.040 | 0.993 |  4.1 |
| policy `proj`, echo+clock+vel      | -0.012 | 0.040 | 0.991 |  4.9 |
| UNTRAINED same arch, echo+clock    | +0.147 | 0.059 | 0.981 |  6.5 |
| policy GRU hidden, as-trained      | +0.369 | 0.046 | 0.674 | 25.6 |
| OUR features, shallow-tanh, untrained | +0.025 | 0.098 | 0.968 | 7.7 |
| OUR features, deep+LN, untrained   | -0.728 | 0.098 | -17.2 | 150.8 |
| ours (frozen trained encoder)      | -0.482 | 0.070 | 0.948 | 10.7 |

**The +0.742 is entirely artifact.** Decomposed by cumulative ablation: ~0.50 of it is the control echo
(`GLOBAL_FIELDS_V2` dims 22:26 are `prev_action_world_dir_sin_cos` + `prev_action_button`, i.e. `a_t` copied
into `obs_{t+1}`); ~0.19 is `link_vel`, which makes next position a linear extrapolation; ~0.06 is `time_left`
(a deterministic countdown) plus the `goal_*` block (constant within an episode). Remove all three: -0.012.

Three conclusions:

1. **The policy's competence is irrelevant.** The same architecture with random untrained weights scores
   +0.617 (echo-zeroed, buggy slice) / +0.147 (echo+clock), at or above the trained policy. Whatever makes
   that surface look predictable is the shape of the map and the content of the input, not what PPO learned.
2. **The GRU hidden state is a bad surface**, not a good one: pos MAE 25.6 against `proj`'s 4.4. Compressing
   history discards the state the reward reads.
3. **The finding that survives is about our own encoder.** An UNTRAINED random shallow-tanh projection of our
   own features scores +0.025; our trained encoder scores -0.482; untrained deep+LayerNorm scores -0.728.
   Our encoder is *worse than random* as a predictive surface, and the deep trunk + latent LayerNorm is the
   mechanism -- LayerNorm renormalises every frame, so a small state change can rotate the whole vector, and
   latent scale (how far Link moved) is not smoothly representable. That is a ~0.5 skill deficit of our own
   construction, and it is the next thing to test.

Note also that our state features are *richer* than `obs` on physics, not poorer: relative positions, distance,
bearing and per-actor velocity are already in the row block, and Link's `speed`/`vel_xyz` in the v2 link block.
`obs` uniquely has only goal-conditioning, the clock, rupee targets and category one-hots.

Method cost of this section: the +0.742 was reported before any control ran, and a WM + dream PPO was nearly
launched on it. See `AGENTS.md` for the rules added as a result.

## 29. Three architecture changes, three negatives: the wall is the latent, not the network

All four arms share one config (v3 data, latent 128, unroll 30, oversample 50, factored action, unified
decoder, seed 0) and differ only where named, so they are comparable at matched epochs.

| arm | skill | move ratio | inverse dyn | notes |
|---|---|---|---|---|
| deep + LayerNorm (`wm_fu_dist`, 12 ep) | -0.776 | 0.044 | 0.059 | the baseline |
| shallow + tanh (`wm_shallow_tanh`, 12 ep) | **-2.023** | 0.039 | 0.069 | baba 0.961, pos MAE 12.78 |
| shallow + tanh + FiLM (`wm_film`, killed at 4 ep) | **-9.87** @ep4 | - | - | vs -2.55 @ep4 concat |
| transformer + tanh (`wm_tf`, 4 ep) | -2.412 @ep4 | **0.023** | **0.043** | vs -2.547 @ep4 MLP |

**1. Encoder geometry (shallow + tanh).** Chosen from an untrained-init sweep over our own features at matched
latent movement (move/rand ~0.098 for all six variants), where LayerNorm cost 0.73 skill and depth 0.36:

| untrained, our features | skill | pos MAE (ridge) |
|---|---|---|
| shallow + tanh | +0.014 | 7.7 |
| shallow + none | -0.213 | 10.3 |
| deep + tanh | -0.350 | 15.9 |
| deep + none | -0.323 | 27.7 |
| shallow + LN | -0.712 | 26.2 |
| deep + LN (= ours) | -0.673 | 150.8 |

Trained, it scored -2.023 -- 2.6x WORSE than the deep+LN baseline it was meant to fix. Decode moved both ways
(baba_count 0.915 -> 0.961, pos MAE 14.62 -> 12.78, but link_hp 0.968 -> 0.922). The init measurement predicted
nothing; see `AGENTS.md` §11.

**2. FiLM action conditioning.** Motivated by a real diagnosis: `cat([z, factored(a)])` gives the action 9 of
137 input columns, and inverse dynamics is 0.059-0.069 against a 0.220 ceiling, so the action IS being ignored.
`FiLMPredictor` makes it multiplicative (`(1+gamma(a)) * h + beta(a)` on both hidden layers, gamma zero-init so
init is exactly identity). It lost at every matched epoch -- -30.1/-17.0/-13.0/-9.9 against -16.2/-4.3/-2.7/-2.5
-- and was killed. Unresolved whether this is capacity or optimisation: gamma starting pinned at 1 means all
modulation is learned from a dead start, while concat has action gradients from step 1.

**3. Transformer encoder.** The state is already 17 tokens (1 Link + 16 actor rows) and the MLP path mean/max
pools the rows away before the trunk sees them, so self-attention should strictly preserve more. At epoch 4 skill
was a wash (-2.412 vs -2.547, well inside seed variance) for 2.6x the wall clock, decode was worse (476 vs 439),
and every structural indicator moved the wrong way: move ratio 0.023 vs 0.039, inverse dynamics 0.043 vs 0.069,
PCs for 99% variance 24 vs 44 (the representation concentrates rather than spreads).

**What the three share.** Whatever we do to the network, the latent moves 0.023-0.044 of a random pair per step
and the action explains 0.04-0.07 of a latent transition against a 0.220 ceiling. The identity is therefore a
very strong baseline and every trained model of ours is worse than it. Meanwhile the ACTION-CONDITIONED heads on
real latents work in the same runs: moved AP 0.435 (6.8x base), step magnitude ratio 1.04x on truly-moving
frames (the regressed version was 0.11x), enemy-hit AP 0.628, Link-hit AP 0.589, hit-row 0.818, nearest-Baba
distance MAE 26.8 against a 64.0 mean-predictor. Those read `(z_t, a_t) -> what happens`, never `z_t -> z_{t+1}`.

Still untried from the same design list: categorical / VQ latents (the DreamerV3 answer to L2 smoothing into
physically impossible interpolations), and signed-distance encoding of collision in place of the 9 binary
`bgCheckFlags` bits -- where `dist_interact_wall` MAE 2.8-2.9 against a 2.9 mean-predictor shows the current
continuous head is not learned at all.

## 30. Categorical / VQ latents: the identity is the global optimum of per-group cross-entropy

Motivation (sound, and still sound): MSE to a continuous z_{t+1} is minimised by the conditional MEAN, so where
the next state is bimodal -- Link keeps walking vs Link hits the wall and stops -- the optimum is a midpoint the
game cannot occupy. Discrete latents (DreamerV3) cannot interpolate: the forward latent is always a corner of the
simplex. Implemented as `bc.st_onehot` (straight-through hard one-hot over G groups of C classes) plus
`dynamics.CategoricalPredictor` (per-group class logits, cross-entropy, LOGIT RESIDUAL `scale * z + f(z,a)` with
f zero-init so an untrained predictor is EXACTLY the identity).

Four shapes, all 4 epochs, everything else identical to the MLP control:

| shape | width | capacity | best skill | decode | dist MAE | enemy-hit AP |
|---|---|---|---|---|---|---|
| 32x4   |  128 |  64 bits | -0.139 | 737 | 76.6 | 0.0003 |
| 32x32  | 1024 | 160 bits | -0.023 | 677 | 51.5 | 0.187 |
| 256x4  | 1024 | 512 bits | **+0.000** | 607 | 44.0 | 0.139 |
| MLP shallow+tanh (control) | 128 | continuous | -2.547 | **439** | **41.4** | **0.384** |

**`skill +0.000` with tune pred 0.00686 exactly equal to identity 0.00686 is the whole result: the predictor
learned to copy z_t.** Only ~4% of groups flip per step, so per-group cross-entropy is minimised by predicting
"no change", and there is no gradient pressure toward the rare flips that ARE the dynamics. Every shape won skill
by this route, and monotonically more so with size (-0.139 -> -0.023 -> +0.000). Against the matched-epoch MLP
control, everything that depends on the predicted latent is worse: link_pos MAE 0.596 vs 0.367, enemy-hit AP 0.139
vs 0.384, Link-hit 0.168 vs 0.466, displacement ratio 0.60x vs 1.21x, decode 607 vs 439.

MOVED AP (0.378 vs 0.362) and distance (44.0 vs 41.4) tie only because both come from the TRANSITION HEAD, which
reads (z_t, a_t) directly and never consumes the predictor's output.

Note on shape, since "just make it bigger" is the obvious next thought: latent width is G*C but capacity is
G*log2(C), so classes cost width linearly and buy capacity only logarithmically. 256x256 would be a
**65,536-dimensional** latent for 2048 bits (0.03 bits/dim) at ~50x the compute; 256x4 gives 512 bits at width
1024 (0.50 bits/dim). Groups are the efficient knob. Capacity did behave as predicted -- 32x4 -> 256x4 moved
distance MAE 76.6 -> 44.0 and decode 737 -> 607 -- it simply cannot fix an objective whose optimum is the identity.

This is NOT a refutation of DreamerV3's discrete latents. Dreamer pairs them with a KL to a *learned prior*,
stochastic sampling, and training against reward on *imagined* rollouts. A bare next-step cross-entropy on a
latent that moves 0.04 per step has copying as its global optimum, whatever the latent is made of.
