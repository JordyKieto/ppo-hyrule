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
