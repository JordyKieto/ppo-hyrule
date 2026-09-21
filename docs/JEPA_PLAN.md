# JEPA from scratch: reward-field reconstruction, frozen encoder, dream PPO

Three stages, each with its own controls and its own falsification. Written before any code, per `AGENTS.md`
§13 (the paper's method is the baseline; deviations are proposed, not shipped) and §9 (ablate before scale).

Supersedes the earlier JEPA attempt recorded in `LATENT_WORLD_MODEL.md` §4 (skill −1.78). That run is not
evidence about this plan: `scripts/train_jepa.py` no longer imports (`LatentDynamics` and `norm_target` were
deleted from `zelda_rl/dynamics.py`), and it used the single-collection `load()`, so it predates the v3 state
(+23 body/collision/action, +5 camera), the 203k four-collection dataset, the transition head, event
oversampling, the factored action and the unified decoder.

## Why this shape

Every dynamics attempt so far has been at or below the identity: −0.776 (deep+LN), −2.023 (shallow+tanh),
−2.412 (transformer), +0.000 (VQ, where the predictor provably learned to copy `z_t`). The recorded diagnosis
across all of them is that the latent carries too little of what an action *does* — inverse dynamics 0.059–0.069
against a 0.220 ceiling — while the decoder carries far more than the reward needs.

So: stop asking the latent to reconstruct 363 state dimensions. Ask it to reconstruct **only the ~12 quantities
the reward function actually reads**, weight that far above the behaviour-cloning term, and let everything else
go. A smaller thing to preserve is a smaller thing to predict in stage 2.

## The reward fields, derived from `CombatTracker.step`

Read from the source, not from recall (`AGENTS.md` §3, §4). These are every state access in
`CombatTracker.step`, `_left_reason`, `live_targets`, `nearest_target` and `_facing`.

### Link

| field | reward terms that read it |
|---|---|
| `hp` | `link_damage`, `link_heal`, `hp_end`, `dead` (hp ≤ 0), lunge-cycle damage attribution |
| `sf1` bit 7 `DEAD` | `death` terminal, via `CombatState.dead` |
| `sf1` bit 22 `SHIELD` | `shield_facing`, `block`, `dodge` |
| `pos` x, y, z | `left_room` (y fall, xz radius), `approach`, `shield_radius`, `block_range`, lunge `dmin` |
| `yaw` | `_facing` → `shield_facing`, `block`, `dodge` |
| `targeted` | `lockon`, `lock_hold` |
| `scene` | `left_room` |
| `room` | `left_room` |

### Per actor row

| field | reward terms that read it |
|---|---|
| present (`ptr in cur.enemies`) | `target_lost` (despawn is not a kill), `live_targets` |
| `hp` | `enemy_damage`, `kill`, `kill_speed_bonus`, `success`, `timeout` |
| `pos` x, y, z | `nearest_target`, `approach`, `shield_radius`, `block_range`, lunge `dmin` |
| `state` (from `fn`) | `stun_hit` (`stunned`/`hit`), lunge cycles (`lunge`/`pull_back`) |
| `id` | target eligibility (`e.id in cfg.target_ids`); `state` is only defined for `DEKU_BABA` |

### Explicitly NOT reconstructed

`sf1` bit 26 `DAMAGED` — **`bc.py`'s existing comment claims the reward reads this; it does not.** `step()`
derives damage from the HP difference. Also out: `shape_yaw`, `yaw_to_player`, `timer`, `params`, `upd`,
`focus`, `cat`, velocity, speed, camera (all 10 v3 features), the 9 `bgCheckFlags`, `actionFunc`, melee state,
`ztarget_timer`, `held_item`, `nuts`, `arrow_row`, rupees, `dist_interact_wall`, `y_dist_to_ledge`, door
fields, `knockback`, `floor_height`, `fall_distance`, and every other `sf1`/`sf2`/`sf3` bit.

Tracker memory (`self.tracks`, `last_kill_t`, `targets_start`, caps) is not a state field — `DreamTracker`
already carries it and is replayed from the real tracker at seed time.

## Stage 1 — encoder: heavy reward-field reconstruction + light BC

    loss = w_recon * recon(D(E(s)), reward_fields(s))  +  w_bc * CE(pi(E(s)), a)      w_recon >> w_bc

- Reconstruction uses the head kinds already validated here: HP **classified** not regressed (the §3 lesson,
  damage F1 0.286 → 0.906), `scene`/`room`/present/`state` as classification, position and yaw as regression
  (yaw as sin/cos).
- BC is the representation pressure, deliberately the smaller term.
- The encoder is the `BCNet` trunk, unchanged, so stage 2 and 3 plumbing and all 17 rebuild sites keep working.

### No control leakage — the hard constraint

BC means: Link sees the **current state** and predicts the **controls**. It must not see the controls.
`featurise(control_echo=False)` zeroes `CONTROL_ECHO_FEATS` — `action_fn_rel`, `melee_active`, `melee_anim`,
`ztarget_active`, `ztarget_timer`, `parallel_yaw_sin/cos`, `sf1` bits 11/15/18/19.

**Two reward fields are themselves action-states, and this needs a decision (`AGENTS.md` §6):**

- `sf1` bit 22 `SHIELD` — "R is held"
- `targeted` / `has_lockon` — "Z was pressed"

Neither is in `CONTROL_ECHO_FEATS`, so both currently survive into the encoder input. They are also reward
fields, so they cannot simply be dropped from the *targets*. Plan: keep them as reconstruction targets, and
**measure the leak** — BC top-1 with and without them in the input. If BC accuracy depends on them, that
accuracy is fake and they come out of the input while staying in the targets.

### Stage 1 controls (all required before stage 2)

1. **Decode floor**: per-field accuracy against predict-the-marginal. A field that does not beat its marginal
   is not reconstructed, whatever the loss says.
2. **Leak probe**: BC top-1 with / without the two action-state inputs above.
3. **Collapse**: latent std, effective rank, `move/rand` ratio. A constant encoder is the degenerate optimum of
   stage 2, so it has to be ruled out at stage 1.
4. **Seed variance**: ≥ 3 seeds before any number is reported as a result (§7).

**Falsified if**: the reward fields do not reconstruct at least as well as the current full-state decoder does
on those same fields (baba_count 0.915–0.999, link_hp 0.92–0.98, row_state 0.94, link_pos MAE 12.8–14.6).
Restricting the target set is supposed to make these *easier*, not harder.

## Stage 2 — transition model on the frozen encoder

JEPA as the papers write it, with the encoder frozen from stage 1 (so no EMA target and no collapse risk —
a frozen target *is* the stop-gradient):

    z_{t+1} = M(z_t, a_t)      L1 on layer-normalised targets, rollout loss, stride > 1

Faithful points from `train_jepa.py`'s existing docstring: L1 on layer-normalised targets (I-JEPA reference
impl; V-JEPA Eq. 1), a rollout term alongside teacher forcing (V-JEPA 2-AC Eq. 2–3), stride above one frame
(74% of steps unchanged at 10 Hz), **no decoder gradient into the encoder** (DINO-WM 2411.04983) — free here,
since the encoder is frozen.

**Deviation to flag, not to ship**: our own §4 table has decoder-in-the-loss (−1.37) beating JEPA (−1.78), i.e.
DINO-WM's finding did not replicate here. The paper's version runs first regardless (§13).

**Falsified if**: skill ≤ 0 against the identity, *or* skill > 0 while the reward fields decode worse than at
stage 1. Both columns get reported together (§2) — this is exactly how VQ produced +0.000.

Real barometer, not the training loss (§8): `eval_rollout_space.py` and `eval_dream_rollout.py` at
H = 10/40/100/400, in world units, against the do-nothing control.

## Stage 3 — PPO in the dream

Only once stage 2 beats the identity *and* the do-nothing control on ground-truth rollouts. Reward parity is
already structural — the dream calls the same `CombatTracker` (§5).

Known pre-existing issue to fix before this stage, recorded in the docs and confirmed by measurement: edits
from `apply_transition` go into the **state** while the transition head reads **`z`**, and `z` evolves only
through the predictor. Velocity therefore goes stale, which the docs name as the real blocker for approach
("the remaining blocker for approach is momentum, not direction" — the direction head is at 91% of its
estimated ceiling, 0.532 vs 0.584).

## Ordering rule

One stage at a time, controls finished before the next begins (§13.4). Stage 1 has four controls; none of them
is optional, and stage 2 does not start while any is unrun.

## Stage 1 result (3 seeds + a leak arm), and one open risk carried forward

All four controls ran. Every reward field beats its floor: `targeted` 0.9996–1.0 (floor 0.235), `row_state`
0.994 (0.640), `baba_count` 0.954–0.991 (0.759), `link_hp` 0.896–0.947 (0.811), `row_hp` 0.899–0.903 (0.009),
`scene` recall 0.75–1.0 on 12 event frames at FPR 0.0001, `link_pos` ~9–11 world units (floor ~70),
`near_dist` ~17–35, `row_pos` ~12–20. `room` is **untestable**: zero off-baseline frames in train or test, so
the reward's room-change branch has no examples in 203k frames (the §6 exit problem again, for a second field).

Leak probe: BC top-1 0.036–0.048 with `sf1_bit22`/`has_lockon` present, 0.044 with them zeroed — inside the
seed spread, so no measurable leak and they stay as inputs.

### OPEN RISK, noted and deliberately not fixed before stage 2

**Effective rank 8.5–8.7 of 128 dimensions**, stable across all three seeds, with `move/rand` 0.048–0.050. The
latent uses ~7% of its dimensions. Some of that is honest — the reward fields really are low-dimensional — but
8.5 effective dimensions for 16 actor rows plus Link's eight quantities is low, and **a near-constant latent is
the degenerate optimum of the stage-2 objective**: VQ scored skill +0.000 by exactly that route, its predictor
provably copying `z_t`. So stage 2 must report skill and informativeness together (§2) and never skill alone.

Second observation on the same axis: at `w_bc 0.05` the reconstruction term outweighed BC **33×** in actual
gradient contribution, and BC top-1 is only 2.3–3.0× chance. This is closer to a pure reward-field autoencoder
than to "BC + reconstruction". **If stage 2 shows this is a poor predictive surface, behaviour-cloning pressure
is the lever to raise** — that is the documented reason BC was in the loss at all (§7: the exit episodes, which
are consistent directional walking, moved action-recoverability 0.059 → 0.139, more than any architecture change).

## Stage 2 result: FALSIFIED as written. The lever is now behaviour-cloning pressure.

Four arms, each adding one thing, then 3 seeds of the best. Ground-truth rollout, world units, 50 held-out
start frames, real action sequences (`AGENTS.md` §8):

| arm | H=10 | H=40 | H=100 | ‖z‖@100 |
|---|---|---|---|---|
| do-nothing control | 75.1 | 106.5 | 161.6 | — |
| pure JEPA (the paper's loss) | 232.4 | 38,601 | 6.8×10⁹ | 2.2×10¹¹ |
| + un-normalised L1 term | 95.0 | 166.0 | 309.3 | 13.9 |
| + decode pressure on the predicted latent | 80.0 | 147.3 | 261.3 | 16.4 |
| + unfrozen encoder | 77.4 | 126.5 | 238.6 | 12.3 |
| + real-latent reconstruction, seed 0 | 70.2 | 136.6 | 219.1 | 13.6 |
| the same, seeds 1 / 2 | 75.3 / 76.8 | 127.2 / 146.6 | 230.8 / 233.0 | — |

**Two findings that hold.**

1. **The paper's loss cannot be used autoregressively here, for a mechanical reason.** `L1(layer_norm(z_pred),
   layer_norm(z_target))` is *exactly* scale-invariant — verified by scaling a prediction 1×/2×/10×/1000× and
   getting a bit-identical loss. Nothing bounds ‖z‖, so a residual predictor inflates 1.03× per step on
   in-distribution inputs and accelerates once the rollout leaves the training range: ‖z‖ reached 2.2×10¹¹ and
   position error 6.8×10⁹ world units by H=100. One un-normalised L1 term fixes it (7 orders of magnitude), and
   this is not seed-sensitive.
2. **Unfreezing the encoder requires keeping stage 1's reconstruction on the REAL latent.** With only the
   predicted latent pressured to decode, the encoder drifts off stage 1: `near_dist` went 17.5 → 35.2 world
   units. `--w-recon` restores it (18.2–25.0 across seeds).

**What is falsified.** The plan's bar was "skill ≤ 0 against the identity ⇒ falsified". Skill is negative in
every arm (best raw-MSE −0.267, and it got *worse* as the rollout got better, which is one more reason §2 exists).
On the barometer the best configuration merely **ties** do-nothing at H=10 across seeds and is clearly worse at
H=40 and H=100. Stage 3 does not start.

**A claim retracted in the same session it was made**: "first rollout to beat the do-nothing control" was seed 0
(70.2 vs 75.1). Seeds 1 and 2 give 75.3 and 76.8. It does not replicate — the §25 / rule-7 pattern again.

**Collapse got worse, consistently.** Effective rank 7.0–7.4 of 128 across the three seeds against stage 1's
8.5, and `move/rand` 0.037–0.039 against 0.049. Unfreezing costs latent rank even with stop-gradient targets.

**So the lever is behaviour-cloning pressure, as called before stage 2 ran.** Stage 2's objective is no longer
the bottleneck — the scale bug is fixed and decode pressure is in — and the surface is still not good enough.
Stage 1 ran at `w_bc 0.05`, where reconstruction outweighed BC **33×** and BC top-1 reached only 2.3–3.0×
chance. Inverse dynamics is 0.071–0.078 against a 0.220 ceiling, i.e. the transition still carries little of
what the action did, which is precisely what BC pressure is supposed to supply. It should also raise the latent
rank, since distinguishing states well enough to choose an action needs more than 7 dimensions.

Next: sweep `w_bc` upward in stage 1 (0.05 → 0.5 → 2 → 10), and for each, report BC top-1, effective rank, the
reward-field reconstruction (which must not regress) and stage 2's rollout. Reconstruction quality and action
legibility are the two axes, and the sweep is looking for the knee.

## Pure JEPA (reconstruction pinned): it does NOT collapse, and it does NOT keep the state

Encoder trained from scratch by the predictive objective ALONE -- no reconstruction, no BC. EMA target encoder
with a stop-gradient (I-JEPA App. A), L1 on layer-normalised targets, teacher forcing plus a rollout term,
stride 1. Deliberately no un-normalised term: that fix was needed for a FROZEN encoder, and with a trainable
one `L1(z_hat, z_target)` is minimised by driving both to zero, i.e. it pays for collapse.

**It did not collapse, and it is the LEAST collapsed latent measured in this project.**

| epoch | train | tune | skill | std | min std | eff. rank | ‖z‖ | move/rand |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1996 | 0.0441 | −0.446 | 0.4504 | 0.2012 | 10.7 | 11.32 | 0.062 |
| 6 | 0.1541 | 0.0853 | −2.170 | 0.4466 | 0.1989 | 10.2 | 11.32 | 0.055 |

Effective rank 10.2–10.7 against stage 1's 8.5 and stage 2's 7.4; no dead dimensions (min std 0.199); ‖z‖ dead
stable at 11.32. So EMA + stop-gradient is sufficient here and the "encoder predicts zero" failure did not
occur. Note the inverse relationship this exposes: **reconstruction pressure is what LOWERS the rank.**

**But the state is not readable out of it.** Freeze the encoder, fit a reward-field decoder on top (post-hoc,
no gradient to the encoder, as DINO-WM asks), against a control that does exactly the same on an UNTRAINED
encoder's latent:

| field | JEPA-trained | untrained control | stage 1 (reconstruction) |
|---|---|---|---|
| link_hp | 0.786 | **0.862** | 0.896–0.947 |
| baba_count | 0.936 | **0.986** | 0.954–0.991 |
| row_state | 0.966 | **0.970** | 0.994 |
| targeted | 0.9986 | **0.9989** | 0.9996–1.0 |
| link_pos | 35.1 | **29.0** | **9–11** |
| row_pos | 18.3 | **17.8** | 12–20 |
| row_hp | **0.902** | 0.899 | 0.899–0.903 |
| near_dist | **48.0** | 53.6 | 17–35 |
| row_dist | **34.9** | 38.0 | — |

**The predictive objective is worse than random initialisation on 7 of 10 reward fields**, and 3–4× worse than
reconstruction on Link's position (35.1 vs 9–11 world units). It does not merely fail to add state information;
relative to a random projection it actively destroys it.

So the answer to "can we encode the latent into the right state" under pure JEPA is **no**. Prediction alone has
no reason to keep HP, position or counts -- they are exactly the quantities that are hard to predict, so an
objective that only rewards predictability is better off discarding them. That is the same overfitting curve as
every other arm (best tune at epoch 1, train falling while tune doubles), and the same identity-error decline
(0.0305 → 0.0269) that says the encoder is smoothing itself.

**Consequence for the plan.** Reconstruction is not a nice-to-have that can be pinned; it is the only thing
keeping the reward's state in the latent. Stage 1's design is vindicated -- and the real tension is now explicit
and measurable: reconstruction buys state fidelity and costs latent rank (8.5 vs 10.2), pure prediction buys rank
and costs the state. Caveats: 6 epochs, one seed, 3 probe epochs (identical budget for both probe arms, so the
comparison is fair even if the absolute numbers would improve).
