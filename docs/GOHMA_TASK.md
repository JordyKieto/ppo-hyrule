# Boss Task: Queen Gohma (Inside the Deku Tree, boss room)

The agent must **defeat Queen Gohma without dying**, as fast as possible. Larvae pay per kill, **uncapped**: Gohma doesn't make herself vulnerable again until her larvae are dead, so killing them is progress. Only the episode time limit bounds it.

- **Code:** `zelda_rl/gohma.py` (states, `GohmaTracker` reward, `boss_features` observation block), `zelda_rl/combat.py` (shared RAM reader and config, `profile="gohma"`), `zelda_rl/env.py` (slingshot actions), `zelda_rl/transfer.py` (warm start from the Deku Baba policy), `scripts/train_combat.py --profile gohma`.
- **Built on the Deku Baba task** (`docs/COMBAT_TASK.md`): same RAM reader, the same `pay()` reward bookkeeping, and the same core observation block. The Deku Baba policy transfers.
- **How the fields were found:** the pattern in `docs/ADDING_OBSERVATIONS.md`, applied to two recorded human fights.

---

## Recordings and start states

| Recording | What happened |
|---|---|
| `runs/manual/boss_20260914-160725.jsonl` (20 Hz, 214 s) | **Win:** 6 seeds, 2 stuns (one floor, one ceiling), 10 sword hits, defeat at 167 s, heart container, blue warp |
| `runs/manual/boss_20260914-161434.jsonl` (20 Hz, 320 s) | Mid-fight state: 2 larvae killed, **death** by Gohma's attacks, reload, 2 egg batches (6 larvae) killed with seeds and sword |

**10 savestate slots** (all scene 17, Link with Kokiri Sword, Deku Shield, slingshot on C-down, Deku Sticks on C-right):

| Slot | Link | Seeds | Gohma | Notes |
|---|---|---|---|---|
| 1 | 3.0 ♥ | 30 | 10 HP, on the ceiling, 1,286 away | fight start (entrance) |
| 2 | 3.0 ♥ | 30 | 10 HP, on the ceiling, 528 away | fight start (in the room) |
| 3 | 3.0 ♥ | 25 | 7 HP, floor, 47 away | close range |
| 4 | 3.0 ♥ | 25 | 7 HP, on the ceiling, 450 away | |
| 5 | 3.0 ♥ | 24 | 7 HP, ceiling | **3 larvae** on the floor |
| 6 | 3.0 ♥ | 14 | 7 HP, floor | **excluded: ignores stick input on 3/3 checks** (Link's state flags are all 0 in this save) |
| 7 | 3.0 ♥ | 14 | 7 HP, floor, 214 away | Z-targeting at load |
| 8 | 3.0 ♥ | 13 | 7 HP, ceiling | |
| 9 | 2.5 ♥ | 24 | **4 HP**, floor, 39 away | late fight |
| 10 | 2.5 ♥ | 13 | 7 HP, ceiling | low seeds |

## RAM signals (✅ verified on the recordings, savestates and live probes)

| Signal | Source | Evidence |
|---|---|---|
| Gohma | actor id `0x0028`, **category 9**, params `0xFFFF` | only boss-list actor in all 10 slots; HP 10 |
| Larva / egg | actor id `0x0028` (same id!), **category 5**, params 0–2 (egg index) | 3 per batch, 1.6 s apart; 2 HP |
| Hatch / death debris | id `0x0028`, params ≥ 10 | 15 pieces per hatch, 20 on Gohma's death, HP 8 placeholder: ignored |
| Gohma HP | +0xAF | 10 → 0, exactly one per sword hit while stunned |
| **Gohma action function** | **+0x180** | 18 values; state table below |
| **Larva action function** | **+0x2A0** | 11 states |
| Actor update function | **+0x130** | used to decode action functions relative to the overlay (see trap 1) |
| Aim point | actor focus +0x38 | lock-on / eye position |
| Link held item | Player **+0x141** | 15 slingshot, 4 sword, 0 none (live) |
| Aiming | stateFlags1 bit 20 first-person, bit 9 ready to fire (flickers while aiming), bit 3 item out | live |
| Cutscene / text | stateFlags1 bit 29 / bit 6 | intro, Navi hints, death cutscene |
| Seeds / nuts | save ctx ammo slot 6 / slot 1 | −1 per shot |
| Button items | save ctx +0x68..+0x6B | `[59 sword, 255, 6 slingshot, 0 Deku Stick]` |
| Heart container | actor `0x005F` (misc) | spawns 14 s after defeat; heart containers 3 → 4 on pickup |
| Blue warp | actor `0x005D` | spawns 18 s after defeat |

### Gohma states (+0x180, relative to her update function +0x130)

| Offset | State | Evidence |
|---|---|---|
| −0x36FC | `ceiling_wait` | before the fight (height −300) |
| −0x0FA8, −0x0954 | `ceiling_idle`, `ceiling_move` | height −319; **seeds here never stunned** (11 seeds in fight 2) |
| −0x0D88 | `ceiling_spawn` | lays the egg batch |
| **−0x0E4C** | **`ceiling_eye_red`** | **vulnerable**: the seed at 144.1 s knocked her off the ceiling |
| −0x1068, −0x111C | `fall` | knocked off (or dropping after all larvae die) |
| −0x1348 | `land` | |
| −0x0A20 | `climb` | back up to the ceiling after a floor phase |
| −0x0C90 | `floor_walk` | |
| −0x1794 | `floor_prepare` | eye closed; seed at 103.3 s did nothing; hit Link at 81.6 s |
| −0x168C | `floor_eye_opening` | 1–3 frame transition |
| **−0x1630** | **`floor_eye_red`** | **vulnerable**: the seed at 106.6 s stunned her. Also her attack (hit Link at 74 and 75 s). A seed in this state at 89.5 s **missed on angle** |
| −0x0DF0 | `floor_eye_close` | after the red-eye window; seed at 92.7 s did nothing |
| −0x1248 | `stunned` | sword window |
| −0x13E0 | `stunned_hit` | **every HP drop happens here** (3 hits per stun) |
| −0x12A4 | `stunned_recover` | |
| −0x2604 | `defeated` | HP 0, death cutscene |

**The eye is only vulnerable in `ceiling_eye_red` and `floor_eye_red`.** Both stuns in the recordings came from seeds fired in one of those two states; seeds fired in any other state never stunned. A seed in a red-eye state can still miss if the angle is wrong. Stun credit therefore requires the knock-down that follows, not just firing at the right time.

### Larva states (+0x2A0, relative to +0x130)

`egg` (falls from the ceiling, ~5 s) → `hatching` → `idle` → `prepare_jump` (1.5 s) → `jump` (0.6 s, the attack) → `land` → back to `idle`. Hit: `hit` → `recover` (1 s). Death: `dying` → `dead`. A larva takes two hits to kill (seed or sword).

### Traps
1. **The En_Goma overlay loads at different addresses.** After a death and reload, every larva action function moved by exactly 0x280. Action functions are therefore decoded as `fn − update function (+0x130)`, which is stable. Gohma's overlay didn't move in the recordings, but gets the same treatment.
2. **Same actor id for boss, larvae and debris.** Tell them apart by category and params (see the table).
3. **Gohma drops from the ceiling on her own once all larvae are dead.** A `fall` is not a stun credit unless a red-eye seed preceded it.
4. **Spawn placeholder HP 8** (as with the Deku Babas) on eggs, debris and item drops: covered by the 2-step spawn grace.

## Controls and actions

| Input | Effect (live probe `scripts/probe_boss_controls.py`) |
|---|---|
| C-down tap | draws the slingshot (held item 15) |
| C-down press/release alternating at 10 Hz | fires: first seed about 0.9 s after drawing, then one every 0.4 s |
| C-down held | never fires |
| B | puts the slingshot away (sword) |
| L held + C-down | Z-targeted slingshot: auto-aims at the lock target |

**Actions v5 (81)** = the 63 Deku Baba actions (indices unchanged) + 9 directions × {slingshot, Z + slingshot}.

## Reward (`GohmaTracker`; every term is logged as `reward/<name>`)

| Component | Value | Kind |
|---|---|---|
| `boss_damage` | +4 per Gohma HP (40 total) | outcome |
| **`boss_defeat`** | **+60** | outcome |
| `boss_defeat_speed_bonus` | +60 × 2 × max(0, 1 − minutes/3) | outcome |
| `success` + `success_time_bonus` | +20, and +30 × fraction of the time limit left | outcome |
| `larva_damage` / `larva_kill` | +0.5 per HP, +2 per kill (**uncapped**; larvae only exist when she lays eggs, and the 300 s limit bounds it) | outcome |
| `link_damage` / `link_heal` | −0.25 / +0.1 per HP unit (−2 / +0.8 per half heart) | outcome |
| `death`, `left_room` | −30 | outcome |
| `timeout` | −10 if Gohma is alive at the cap | outcome |
| `lockon`, `lock_hold` | +0.3 per lock on Gohma (cap 8), +0.005/step (cap 1) | shaping |
| `lock_on_red_eye` | +0.02/step locked on (or facing the eye) while the eye is red (cap 2) | shaping |
| `slingshot_on_red_eye` | +0.03/step with the slingshot out, locked on the red eye (cap 2) | shaping |
| **`seed_stun`** | **+6** when she is knocked down ≤ 1 s after a seed fired in a red-eye state (cap 30) | shaping |
| `stun_hit` | +2 per HP dealt while she is stunned (on top of `boss_damage`) | shaping |
| `approach_stunned`, `sword_ready_stunned` | close in while she's stunned; sword out within 90 units (cap 1) | shaping |
| `block` | +1 per attack cycle (`floor_prepare` / `floor_eye_red` / `floor_eye_close`) survived with the shield up, facing her, within 250 (cap 8) | shaping |
| `seed_cost` | −0.03 per seed | shaping |
| `time_penalty` | −0.001/step | shaping |

**Validation: recorded fights replayed at 10 Hz** (`scripts/replay_gohma_trace.py`; the reward parts sum exactly to the return):

| Recording | Detected | Return |
|---|---|---|
| Win | 2/2 real stuns (2 of 6 seeds), 10 HP damage, defeat, 0.5 ♥ lost. Parts: defeat 60, speed bonus 47, damage 40, stun hits 20, success 20 + 19, seed stuns 12, shaping ≈ 7 | **+225** |
| Death segment | 1 larva kill, 3 ♥ lost, death | −38.6 |
| Larva segment | 6 larva kills (+12 kills, +6 damage, uncapped), 0 stun credits from 12 non-red-eye seeds | +18.5 |

## Observation v5 (542 dims = v4 483 + 59)

| Block | Dims | Contents |
|---|---|---|
| v2 global + combat core + episode memory + Baba behaviour | 99 | unchanged from v4. The core "nearest target" fields now describe Gohma, which is what lets the Deku Baba policy transfer |
| **Gohma** | 33 | present, HP fraction, **eye** relative position and distance, on ceiling, 18-way state one-hot, **red eye**, stunned, time in state, Gohma facing Link cos/sin, **Link aim to the eye (horizontal cos, elevation)**, locked on |
| **Larvae** | 18 | alive count, nearest larva relative position, distance, HP, 11-way state one-hot, second-nearest distance |
| **Items and memory** | 8 | slingshot in hand, sword in hand, first-person aim, ready to fire, seeds, time since last seed, stuns so far, larva kills so far |
| Actor table | 16 × 24 | unchanged |

The Gohma and larva blocks are all zeros in rooms without them, so one v5 policy can train on Deku Baba and Gohma slots together.

## Warm start from the Deku Baba policy

`--init-from checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` (global update 150) is widened function-preservingly:
- the 59 new inputs get zero-initialised weights
- the 18 new action rows get zero weights and a bias 2 below the mean logit

Checked: identical logits on the old actions (max difference 4e-7) and identical value. The new actions start with about 3% of the probability.

## Commands

```sh
export DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin
.venv/bin/python-dbg scripts/monitor_boss.py                          # record a human fight (20 Hz trace + live events)
.venv/bin/python scripts/replay_gohma_trace.py runs/manual/boss_<ts>.jsonl 49   # replay a trace through the reward
.venv/bin/python-dbg scripts/train_combat.py --profile gohma --episode-seconds 300 --num-envs 3 \
  --init-from checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt --run-name gohma-local
.venv/bin/python-dbg scripts/play.py runs/gohma-local/checkpoints/latest.pt --slot 2        # watch (Gohma HUD)
python -m fleet launch --experiment gohma --script train_combat.py --wandb-project zelda-rl-gohma-ppo --run-tag v1 \
  --num-envs 8 --train-args "--profile gohma --episode-seconds 300 --start-slots 1,2,3,4,5,7,8,9,10 \
  --init-from checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt --fleet-aggregator 4 --total-updates 400"
```

## wandb metrics
- `gohma/`: `defeat_rate`, `boss_damage`, `boss_hp_end`, `seed_stuns`, `seeds_fired`, `seed_hit_rate`, `stun_hits`, `larva_kills`, `larvae_seen`, `boss_attacks`, `boss_blocks`, `vulnerable_steps`, `vulnerable_lock_steps`, `sling_aim_steps`, `time_to_defeat_s`
- `reward/<component>`: every component above
- `by_slot/sN/boss_damage`, `by_slot/sN/seed_stuns`
- Plus the shared `combat/`, `episode/`, `ppo/`, `perf/` and `fleet/` metrics.
