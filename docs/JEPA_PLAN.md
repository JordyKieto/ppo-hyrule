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
