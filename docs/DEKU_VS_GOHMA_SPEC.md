# Deku Baba (solved, 78%) vs Queen Gohma: observation and action space, side by side

Both numbers below are read from the **exported run configs**, not from the code as it stands today:
`results/vxejacj7/config.json` (`deku-combat-gru-v2b`, the 78% policy) and `results/kj9ypr7w/config.json`
(`gohma-band-v13`). Regenerate with `scripts/export_wandb.py`.

## 1. The headline

| | Deku Baba (SOLVED) | Queen Gohma | change |
|---|---|---|---|
| `obs_dim` | **483** | **1033** | **+114%** |
| ├ global block | 99 | **297** | **+200%** |
| └ actor table | 16 × 24 = 384 | 16 × **46** = 736 | +92% per row |
| `n_actions` | **63** | **108** | **+71%** |
| `hidden` / `gru` | 256 / 256 | 256 / 256 | **unchanged** |
| parameters | 616,064 | 679,725 | **+10%** |
| **params per obs dim** | **1,275** | **658** | **halved** |

**The observation more than doubled, the action space grew 71%, and the network grew 10%.** Nothing else about
the architecture or any PPO hyperparameter was touched (`lr 3e-4`, `clip 0.2`, `gamma 0.995`, `gae_lambda 0.95`,
`chunk 60`, `epochs 4`, `minibatches 16`, `ent_coef 0.01`, `num_envs 8` — all identical). See `AGENTS.md` §34
for what the unchanged discount settings cost.

## 2. Observation: the global block

**Nothing was removed.** All 47 of Deku's global fields (99 dims) are present in Gohma's 123 fields (297 dims),
field-for-field. Gohma is strictly additive: **+74 fields, +198 dims**.

### Shared by both (46 field names, 99 dims)

Link state (`link_pos`, `link_vel`, `link_yaw_sin_cos`, `link_hp_frac`, `link_hp_cap`, `health_frac`,
`shielding`, `damaged`, `owns_sword_shield`), the goal block (`goal_pos`, `goal_delta`, `goal_dist`,
`goal_rupees`, `goal_flags_missing_extra`, `same_scene_as_goal`, `rupees`, `rupee_gap`), targeting
(`locked_on_target`, `arrow_on_target`, `nearest_target_rel_pos/dist/hp/facing_cos_sin`, `second_target_*`,
`targets_alive_frac`, `targets_start`), decoded enemy behaviour (`t1_state_onehot`, `t1_timer`,
`t1_facing_link_cos_sin`, and `t2_*`), episode memory (`mem_kills_frac`, `mem_damage_dealt`,
`mem_hearts_lost`, `mem_since_last_hit/damage/kill`, `mem_lockons`), the camera (`camera_yaw_sin_cos`),
`time_left`, `state_flags1_bits`, and the control echo (`prev_action_button`,
`prev_action_world_dir_sin_cos` — see AGENTS §5).

### Gohma-only, grouped by what they are for (+198 dims)

| group | dims | fields |
|---|---|---|
| **the boss herself** | 32 | `gohma_present`, `gohma_hp_frac`, `gohma_eye_rel_pos`, `gohma_eye_dist`, `gohma_on_ceiling`, `gohma_state_onehot` (**18**), `gohma_vulnerable`, `gohma_stunned`, `gohma_time_in_state`, `gohma_facing_link_cos_sin`, `locked_on_gohma` |
| **projectile aiming** | 12 | `link_aim_to_eye_cos_sin_elev`, `first_person_aim`, `ready_to_fire`, `aim_forward_world`, `aim_pitch_sin_cos`, `aim_yaw_sin_cos` |
| **ammunition** | 6 | `held_slingshot`, `held_sword`, `seeds`, `since_last_seed`, `nuts`, `since_last_nut` |
| **larvae (3 slots)** | 83 | per larva: `present`, `hp`, `state_onehot` (**11**), `cam`, `body`, `vel_cam`, `facing_vs_link_cos_sin`; plus `larvae_alive`, `larva1/2_rel_pos`, `larva1_dist`, `larva1_hp`, `larva1_state_onehot`, `larva1_facing_link_cos_sin`, `link_facing_larva1_cos_sin` |
| **camera / body reference frames** | 25 | `cam_eye_rel_link`, `link_facing_vs_camera_sin_cos`, `link_velocity_cam`, `gohma_eye_cam`, `gohma_eye_body`, `gohma_vel_cam`, `gohma_facing_cam_fwd_right`, `gohma_facing_body_fwd_right` |
| **seeds in flight (2 slots)** | 20 | per seed: `present`, `cam`, `vel_cam`, `vel_body` |
| **last seed impact** | 9 | `last_impact_present`, `since_last_impact`, `last_impact_cam`, `last_impact_minus_gohma_eye_cam`, `last_impact_to_gohma_eye_dist` |

**Where the growth really is:** two 18-wide and three 11-wide one-hot state vectors alone are **51 dims of
one-hot**, and the larva block is **83 dims for three enemies that are not the objective**. The boss's own
state is only 32 dims of the 198 added.

## 3. Observation: the per-actor row

16 actor slots in both. **Nothing removed, 10 fields added: 24 → 46 dims per row, so +352 dims of the +550.**

| added field | dims | what |
|---|---|---|
| `ego_cam_rel_pos`, `ego_body_rel_pos` | 3 + 3 | position in camera and body frames |
| `ego_cam_vel`, `ego_body_vel` | 3 + 3 | velocity in those frames |
| `facing_cam_fwd_right`, `facing_body_fwd_right` | 2 + 2 | facing in those frames |
| `facing_vs_link_cos_sin` | 2 | facing relative to Link |
| `scale`, `cyl_radius_height`, `mass` | 1 + 2 + 1 | physical extent |

**These are paid 16 times** whether or not a slot is occupied. The Gohma fight has at most 1 boss + 3 larvae
occupied, so ~12 of 16 rows are `present = 0` and still cost 46 dims each.

## 4. Action space

Both are **9 stick directions (8 compass + neutral) × N button combinations**.

| | combos | total |
|---|---|---|
| Deku | 7 | 63 |
| Gohma | **12** | **108** |

| | button combos |
|---|---|
| **shared (7)** | `(none)`, `A`, `B`, `L`, `R`, `L+B`, `L+A` |
| **Gohma adds (5)** | `CD`, `L+CD`, `CL`, `L+R`, `L+CL` |

`CD` = slingshot, `CL` = Deku Nut, `L` = Z-target, `R` = shield. So the five additions are: fire the
slingshot, fire it while Z-targeted, throw a nut, throw a nut while Z-targeted, and Z-target with the shield
up. The last two (`L+R` Z+Shield and `L+CL` Z+Nut) were **absent until 2026-09-26** — two actions the task
requires did not exist for the first ten versions (claim **R7**).

Indices 0–89 keep their meaning across the 90 → 108 widening, so every earlier checkpoint still transfers
(`zelda_rl/transfer.py`).

## 5. What is actually harder, separated from what we made harder

**Intrinsically harder about the task:**
- A Deku Baba dies in ~2 sword hits with no precondition. Gohma has **10 HP** and is only damageable during a
  stun, so a win is **ten stun→hit cycles**.
- Each stun costs a **seed or a nut from finite ammunition** — a resource constraint Deku never had.
- The stun window is short and state-dependent (red eye only), so the policy needs the boss's phase, not just
  her position.
- Observed kills take **45–55 s** against Deku's ~15 s per enemy, which is what makes the unchanged
  `gamma`/`gae_lambda` fatal (AGENTS §34).

**Harder because of choices we made:**
- `obs_dim` 483 → 1033 against an **unchanged 256-wide network**: params per input dim **halved**. `hidden 384`
  restores Deku's ratio at 1,360.
- `n_actions` 63 → 108 from the same 256-wide trunk.
- **~12 of 16 actor rows are empty** and still cost 46 dims each — roughly 550 dims of the 1033 describe
  nothing.
- **51 dims of one-hot** state encoding for the boss and three larvae.
- **83 dims on larvae**, which are not the objective and whose reward terms are now all zeroed in
  `GOHMA_DAMAGE`.

**The untested implication:** of the 1033 inputs, the fraction that is live on any given step is small, and the
network got no extra capacity to find it. A narrower observation — boss block, aiming, ammo, Link, drop the
larva rows and the unoccupied actor slots — would be closer to Deku's 483 at the same capacity, and has never
been tried. That is an observation-side experiment, not another reward term.
