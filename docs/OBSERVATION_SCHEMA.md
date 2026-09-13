# Observation Schema: OoT Game RAM → Policy Input

How the RL agent observes Ocarina of Time: what is read from the game's RAM, how fast, how it becomes the fixed policy vector, and what the actions are. Code: `zelda_rl/obs.py` (schema + featurizer), `zelda_rl/goal.py` (goal distance + reward), `zelda_rl/env.py` (actions + stepping). RAM map background: `docs/FINDINGS.md` §6.

---

## 1. Speed: RAM observations vs. game frames

The game produces new state far more slowly than we can read it. **The step rate is limited by emulation speed, not by observation.**

| Clock / cost | Rate | Source |
|---|---|---|
| Emulated video interrupts (VI) | 60 / s of game time | `0x8011DBB0` counter, verified |
| OoT game logic | ~20 updates / s of game time (every 3 VIs, native N64 rate) | known game behavior |
| **Env decision rate (default)** | **10 Hz of game time** (`vi_per_step = 6`) | `EnvConfig` |
| Read rupees + Link position (zero-copy) | **8 µs** → ~120,000 / s | measured |
| `read_state()`: save ctx, scene, Link, 12 actor lists | **~25 µs** → ~40,000 / s | measured |
| `observe()` = read_state + reward + 44-d featurize (Python) | **~200 µs** → ~5,000 / s | measured |
| Savestate load (reset), verified via VI clock jump | 38–104 ms | measured |

Emulation throughput on the M1 (Metal backend, uncapped speed, measured with the env loop):

| Instances | Speed per instance | Env steps / s (total, 10 Hz decisions) |
|---|---|---|
| 1 | ~4.3× real time (22 ms per step) | ~45 |
| 2 | ~3.5× | ~70 |
| 3 | ~1.7–2.9× with Metal windows; **3.1–3.3× headless (Null)** | ~52–88 Metal; **92–100 headless** |
| RunPod 16 vCPU pod, 8 instances (idle) | 3.4–4.1× each, 28.9× total | ≈290 equivalent |
| RunPod 16 vCPU pod, 12 envs training (10 s episodes) | 1.8–1.9× (CPU saturated) | 214–224 |

At 1× (real time), a 10 Hz step waits 100 ms of wall time and a 60 s episode takes 60 s. Uncapped, a 60 s episode takes ~15–18 s per instance.

**Consequence:** reading RAM faster than 20 Hz of game time only returns duplicate state. The env blocks on the VI counter (`env.ready()`) so each step is exactly `vi_per_step` VIs of game time, whatever the wall speed (measured: 6.00 VIs/step at 1× and uncapped).

---

## 2. Raw game state (`GameState`, read per step)

N64 RAM lives at Wii MEM1 offset `0xF64120`. Addresses are N64 virtual (`addr & 0x3FFFFF` = offset into the 4 MB block), big-endian, NTSC 1.2.

| Field | Type | Address | Meaning |
|---|---|---|---|
| `vi` | u32 | `0x8011DBB0` | 60 Hz emulated clock; restored by savestates (used to verify resets) |
| `scene` | s16 | `gPlayState 0x801C8D60 + 0xA4` | 85 = Kokiri Forest, 52 = Link's House, 81 = Hyrule Field |
| `rupees` | s16 | `gSaveContext 0x8011AC80 + 0x34` | wallet |
| `health` / `health_cap` | s16 | save ctx `+0x30` / `+0x2E` | 16 per heart |
| `pos` | f32 × 3 | `*(gPlayState + 0x1C44) + 0x24` | Link world position (x, y = up, z) |
| `yaw` | u16 | player `+ 0x32` | facing, 0..65535 |
| `rupee_items` | list | actor lists `gPlayState + 0x1C24 + 0x0C`, 12 × {s32 count; Actor* head}, next at `+0x124` | `EnItem00` (id 0x0015) actors whose `params & 0xFF` ∈ {0: 1, 1: 5, 2: 20, 0x13: 50, 0x14: 200} rupees, with world position |
| `player_ok` | bool | player pointer in N64 RAM range | false during some transitions |

**Verified:** rupees, health, scene, Link position (live walking), VI rate, rupee actors. Your slots show the green (+1) and blue (+5) rupees disappearing from the list exactly when the count goes 0 → 1 → 6.

Game-sees-input check used by the trainer: the controller struct at `gPlayState + 0x14` (u16 buttons, s8 stick x, s8 stick y). Frozen states (slot 4) read zeros.

---

## 3. Goal (target state) and "JEPA-style" distance

The target is **a game state, not a hand-written objective**: `read_state_file(NACE01.s05)` gives the goal `GameState` (Kokiri Forest, pos (798, 0, 231), 6 rupees). Each state is embedded in a small, interpretable space where **position and rupees are the only things that matter**:

```
e(s)       = [x, y, z] / 1000  ++  rupees / goal_rupees
pos_dist   = ||pos(s) - pos(goal)||                      (world units)
rupee_gap  = max(goal.rupees - s.rupees, 0)
goal_dist  = pos_dist / 1000 + rupee_weight * rupee_gap / goal.rupees
success    = same scene AND pos_dist < 75 AND rupee_gap == 0
```

Reward per step (potential-style progress, `zelda_rl/goal.py`):

| Term | Value |
|---|---|
| Position progress | `w_pos × (pos_dist_prev − pos_dist_now) / 100` |
| Rupee progress | `w_rupee × Δ min(rupees, goal.rupees)` (rupees beyond the goal count earn nothing) |
| Time | `−0.001` per step |
| Success | `+10`, episode terminates |
| Left the scene (house, shop, Lost Woods…) | `−5`, episode terminates |

The embedding is fixed and hand-designed for now. A learned JEPA encoder (predicting the latent of future states from RAM or pixels) could replace `e(s)` later without changing the env or logging.

---

## 4. Policy input vector (`OBS_DIM = 44`, float32)

Layout = `OBS_FIELDS` in `zelda_rl/obs.py` (also saved in every checkpoint's config):

| # | Field | Dims | Scaling |
|---|---|---|---|
| 0–2 | `link_pos` | 3 | / 1000 |
| 3–5 | `goal_pos` | 3 | / 1000 |
| 6–8 | `goal_delta` (goal − Link) | 3 | / 1000 |
| 9 | `goal_dist` | 1 | / 1000 |
| 10–11 | `link_yaw_sin_cos` | 2 | – |
| 12–14 | `link_vel` (Δpos per step) | 3 | / 10 |
| 15 | `rupees` | 1 | / 20 |
| 16 | `goal_rupees` | 1 | / 20 |
| 17 | `rupee_gap` | 1 | / 20 |
| 18 | `health_frac` | 1 | health / cap |
| 19 | `time_left` | 1 | 1 → 0 over the episode |
| 20–21 | `prev_action_dir_sin_cos` | 2 | 0, 0 if neutral |
| 22–23 | `prev_action_button` (A, B one-hot) | 2 | – |
| 24–43 | 4 nearest rupees × (dx, dy, dz, value, present) | 20 | / 1000, / 20, 0/1 |

Notes:
- The stick is **camera-relative** in OoT. There is no camera yaw in the vector yet; the agent can infer the mapping from `prev_action` + `link_yaw` + `link_vel`. Adding camera yaw is the first candidate improvement.
- NaNs (e.g. no player pointer during a transition) become 0.

---

## 5. Actions (`N_ACTIONS = 27`)

`9 stick directions × 3 buttons`, sent over the pipe controller at each 10 Hz step and held until the next step.

| Stick (index // 3) | 0 neutral, 1 E, 2 NE, 3 N (away from camera), 4 NW, 5 W, 6 SW, 7 S, 8 SE (full tilt) |
|---|---|
| Button (index % 3) | 0 none, 1 A (roll / talk / advance text), 2 B (sword: cuts grass → rupees) |

---

## 6. Episode protocol

- **Start:** slots 1–3, sampled uniformly (slot 4 is frozen and excluded; the trainer's input check drops frozen slots automatically).
- **Reset:** press the slot's load hotkey (slots 1–5 on pipe buttons A/B/X/Y/Z), accepted only when RAM shows the VI clock jump to the slot's saved value.
- **Length:** 60 s game time = 3,600 VIs = 600 steps. Rollout = one episode per env; all envs reset at the start of every PPO update.
- **Logged per episode** (`runs/<run>/episodes.jsonl`): slot, return, length, start/final/min `pos_dist`, start/final/max rupees, `final_rupee_gap`, start/final/min `goal_dist`, success, left_scene, `pos_progress`, action histogram, VI overshoot.

---

# v2 (default since run `kokiri-goal-v2`)

`--obs-version 2`. Checkpoints store `obs_version`, field layouts and model dims in `config`, and `zelda_rl/policy_io.py` rebuilds the right env/model for either version.

## v2 raw state (`GameStateV2` = v1 fields +)

| Field | Source | Status |
|---|---|---|
| `cam_yaw` | `atan2(at.z − eye.z, at.x − eye.x)`, main camera `gPlayState + 0x1E0` (`at` +0x50, `eye` +0x5C) | ✅ calibrated: stick up = camera forward, right = +90°, left = −90°, down = 180° (`scripts/calibrate_stick.py`) |
| `sf1` | Player `+0x66C` stateFlags1 | ✅ slot 1 = Z-targeting + friendly focus; live talking to Mido showed `talking, friendly_focus, in_cutscene` |
| `equipment` | SaveContext `+0x9C` | ✅ 0x1100 |
| `flags` (158 B) | items, equipment, upgrades, quest, eventChkInf, itemGetInf, infTable, current scene's saved flags | ✅ readable; identical across slots 1–5 |
| `actors` | 16 nearest of all actor lists, excluding Link and Navi (`0x0018`) | ✅ |

## v2 policy vector (`OBS_DIM_V2 = 429` = 45 global + 16 × 24 actor table)

Global (45): `link_pos 3, goal_pos 3, goal_delta 3, goal_dist 1, link_yaw_sin_cos 2, camera_yaw_sin_cos 2, link_vel 3, rupees 1, goal_rupees 1, rupee_gap 1, health_frac 1, time_left 1, prev_action_world_dir_sin_cos 2, prev_action_button 2, state_flags1_bits 14, owns_sword_shield 2, goal_flags_missing_extra 2, same_scene_as_goal 1`.

stateFlags1 bits exposed: talking(6), dead(7), getting_item(10), carrying(11), ledge_hang(13), ledge_climb(14), z_targeting(15), friendly_focus(16), jumping(18), freefall(19), climbing_ladder(21), damaged(26), in_water(27), in_cutscene(29).

Actor row (24), sorted by distance, zero-padded with `present = 0`:

| Dims | Field |
|---|---|
| 1 | `id` (raw actor id; the model embeds it, vocab 512) |
| 12 | category one-hot |
| 3 | relative position (dx, dy, dz) / 1000 |
| 1 | distance / 1000 |
| 3 | actor flags: targetable (bit 0), hostile (bit 2), friendly (bit 3) |
| 1 | health / 20 |
| 1 | rupee value / 20 (EnItem00 only) |
| 1 | params low byte / 255 |
| 1 | present |

## v2 model (`EntityActorCritic`, 243k params)

A shared per-actor MLP runs on `[id embedding(16) ++ 23 actor features]`, followed by masked mean + max pooling. The pooled result is concatenated with the global features, then fed to separate actor and critic MLPs (2 × 256 tanh).

## v2 actions: world-relative

Same 27 = 9 × 3 (neutral + 8 directions × none/A/B). Directions are **world** angles in the x/z plane. Each step the env converts them to a stick angle: `stick = 90° − (world − cam_yaw)` (`STICK_HANDEDNESS = −1`). The agent no longer has to learn the camera.

## v2 goal / reward additions

- `flags_missing` = goal flag bits the episode's start state lacked and the current state still lacks.
- Reward: `+w_flags` per bit newly satisfied.
- Success also requires `flags_missing == 0`.
- For slots 1→5 no flag differs, so this term is 0 today. It is the hook for "talked to Mido" or "opened chest" targets.

New per-episode metrics: `final_flags_missing`, `talking_frac`, `ztarget_frac` (wandb `goal/final_flags_missing`, `episode/talking_frac`, `episode/ztarget_frac`).

---

# Playing a checkpoint locally

```sh
export DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin
.venv/bin/python-dbg scripts/play.py                         # newest .pt in checkpoints/inbox/ (else runs/*/checkpoints)
.venv/bin/python-dbg scripts/play.py kokiri-goal-v2 --slot 2 -n 3
.venv/bin/python-dbg scripts/play.py path/to/update_00100.pt --deterministic --fast
.venv/bin/python-dbg scripts/play.py --watch                 # keeps playing; hot-swaps newer checkpoints between episodes
```

Drop a RunPod checkpoint in `checkpoints/inbox/` and run `play.py` with no arguments. It opens a Dolphin window (real time by default) and prints a HUD line every 0.5 s: time, action, value, distance to goal, rupees, return, top-3 action probabilities, Link state flags, nearest actor. Results go to `runs/play/<timestamp>/episodes.jsonl`.
