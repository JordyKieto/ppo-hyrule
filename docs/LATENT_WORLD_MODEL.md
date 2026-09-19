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
