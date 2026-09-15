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
| `runs/manual/boss_20260914-165641.jsonl` (20 Hz, ~10 min) | **Deku Nut fight:** 7 nut stuns in floor states (walk, prepare, eye closed, eye red), sword hits 8 → 3 HP, 4 larva batches, heals from drops |

**10 savestate slots** (2026-09-14, second set; all scene 17; Kokiri Sword on B, **Deku Nut on C-left**, slingshot on C-down, Deku Sticks on C-right):

| Slot | Link | Seeds / nuts | Gohma | Notes |
|---|---|---|---|---|
| 1 | 3.0 ♥ | 30 / 20 | 10 HP, `ceiling_wait`, 1,286 away | fight start at the entrance |
| 2 | 3.0 ♥ | 30 / 19 | 10 HP, `ceiling_wait`, 376 away | slingshot drawn |
| 3 | 3.0 ♥ | 29 / 18 | 10 HP, **stunned**, 188 away | locked on, slingshot out: sword window |
| 4 | 2.5 ♥ | 29 / 15 | 8 HP, `floor_walk`, 190 away | locked on |
| 5 | 3.0 ♥ | 27 / 18 | 10 HP, **`ceiling_eye_red`**, 438 away | locked on: right before the ceiling eye shot |
| 6 | 3.0 ♥ | 26 / 8 | 6 HP, `floor_prepare`, 135 away | a Deku Nut here stuns her (live probe) |
| 7 | 2.5 ♥ | 25 / 6 | 5 HP, **stunned**, 38 away | locked on, in sword range |
| 8 | 2.5 ♥ | 26 / 16 | 10 HP, **stunned**, 448 away | locked on, slingshot out (knocked down by a seed) |
| 9 | 3.0 ♥ | 26 / 7 | 5 HP, `ceiling_move` | **3 larvae** alive |
| 10 | 3.0 ♥ | 30 / 20 | 10 HP, `ceiling_wait`, 1,143 away | **replaced**: the user's slot 10 was Gohma already defeated. Made with `scripts/make_walk_state.py` (loaded slot 1, walked 165 units into the room, saved; input check passed) |

The first set (used for the recordings above) is replaced. Stick input check: slots 1, 6 and 10 pass 3/3 (their Link state flags read 0, which had been suspicious in the first set).

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
| Seeds / nuts | save ctx ammo slot 6 / slot 1 | −1 per shot / throw |
| Deku Nut throw | held item 6 for a moment; actor `0x0056` spawns 85 units ahead | every throw in the nut fight |
| Button items | save ctx +0x68..+0x6B | `[59 sword, 1 Deku Nut, 6 slingshot, 0 Deku Stick]` |
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

**Deku Nuts** (the nut fight plus a live probe) stun her **in any floor state**: walking, winding up, eye closed, or eye red. The stun starts about 0.22 s after the throw but lasts only **about 2 s** from a fresh stun, against a long knock-down from a seed. A nut at 190 units while she walked did nothing, and nuts never stunned her on the ceiling. Nuts are the weaker tool; the red-eye seed is the main one.

**The eye is only vulnerable to seeds in `ceiling_eye_red` and `floor_eye_red`.** Both stuns in the recordings came from seeds fired in one of those two states; seeds fired in any other state never stunned. A seed in a red-eye state can still miss if the angle is wrong. Stun credit therefore requires the knock-down that follows, not just firing at the right time.

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
| C-left tap | throws a Deku Nut (flash); stuns Gohma about 0.22 s later for about 2 s (live probe, slot 6) |

**Actions v5 (90)** = the 63 Deku Baba actions (indices unchanged) + 9 directions × {slingshot, Z + slingshot} + 9 directions × Deku Nut.

**Item buttons are tapped within the step:** C is pressed for 3 VIs, then released. A held C never fires the slingshot, and fleet run `gohma-v1` fired only 0.3–0.4 seeds per episode because of it. With the tap (verified live), repeating Z + slingshot draws and then fires every 0.4 s. The human technique also works: draw once, hold it cocked with Z, tap to fire the moment the eye turns red.

## Reward (`GohmaTracker`; every term is logged as `reward/<name>`)

| Component | Value | Kind |
|---|---|---|
| `boss_damage` | +4 per Gohma HP (40 total) | outcome |
| **`boss_hp_progress_bonus`** | +4 × 2 × (fraction of her HP gone after the hit): +0.8 for the first hit … +8 for the last (44 total) | outcome |
| **`boss_defeat`** | **+60** | outcome |
| `boss_defeat_speed_bonus` | +60 × 2 × max(0, 1 − minutes/3) | outcome |
| `success` + `success_time_bonus` | +20, and +30 × fraction of the time limit left | outcome |
| `larva_damage` / `larva_kill` | sword: +0.5 per HP, +2 per kill. **Slingshot** (`_sling`: a seed resolved as the hit, larva beyond sword range): +0.25 / +1. Uncapped; the 300 s limit bounds it | outcome |
| `link_damage` / `link_heal` | −0.25 / +0.1 per HP unit (−2 per half heart lost, +0.8 per heart picked up) | outcome |
| `grass_cut` | +0.2 per grass tuft cut (`En_Kusa` 0x0125, 8 in the room; count of standing tufts drops) (cap 2). 🔶 Not in the recordings: check `gohma/grass_cut` on the fleet | shaping |
| `resource_pickup` | seeds +0.05 × 35 / (seeds before + 5) each, nuts +0.15 × 24 / (nuts before + 4) each: worth more when running low (cap 6) | shaping |
| `death`, `left_room` | −30 | outcome |
| `timeout` | −10 if Gohma is alive at the cap | outcome |
| `lockon`, `lock_hold` | +0.3 per lock on Gohma (cap 8), +0.005/step (cap 1) | shaping |
| `lock_on_red_eye` | +0.05/step locked on (or facing the eye) while the eye is red (cap 5) | shaping |
| **`sling_ready`** | **+1.0 once per red-eye window** (contiguous `ceiling_spawn` / `floor_prepare` / `floor_eye_opening` / red-eye span) when the slingshot is **cocked** and Z-targeted or free-aimed at her (cap 6). v5 farmed a per-step version from the entrance slots; v6 farmed it by waiting cocked in slot 6 | shaping |
| **`slingshot_on_red_eye`** | **+0.3/step** with the slingshot out and **aimed at the red eye (Z-lock or free aim), wherever she is** (cap 8) | shaping |
| **`red_eye_seed`** | **+4.0 per seed fired while the eye is red**, hit or miss (cap 12); **such seeds never pay `seed_miss`** | shaping |
| **`unfired_red_eye`** | **−1.5** when a red-eye window closes with the slingshot cocked during the red eye but no seed fired | shaping |
| `elevated_seed` | +0.5 per seed fired at her (aimed / locked) right before or during the red eye (cap 5) | shaping |
| **`dropped_red_eye_shot`** | **−1.0** when the slingshot is put away without firing while her eye is red (targets the Deku Baba habit of switching to the sword) | shaping |
| **`seed_stun`** | **+15** when she is knocked down ≤ 1 s after a seed fired in a red-eye state (cap 60): **the key skill** | shaping |
| `nut_stun` | +2 when she is stunned ≤ 0.6 s after a Deku Nut (cap 10): weaker, short stun | shaping |
| `stun_hit` | +2 per HP dealt while she is stunned (on top of `boss_damage`) | shaping |
| `approach_stunned`, `sword_ready_stunned` | close in while she's stunned; sword out within 90 units (cap 1) | shaping |
| `block` | +1 per attack cycle (`floor_prepare` / `floor_eye_red` / `floor_eye_close`) survived with the shield up, facing her, within 250 (cap 8) | shaping |
| **`seed_miss`**, **`nut_miss`** | per shot that hits nothing (seed: no stun and no larva damage within 1.2 s; nut: no stun within 0.6 s), **scaled by scarcity**: seed −0.15 × 35 / (seeds left + 5), which is −0.15 at 30 seeds and −1.05 at 0; nut −0.3 × 24 / (nuts left + 4), which is −0.3 at 20 nuts and −1.8 at 0. Hits cost nothing | shaping |
| `time_penalty` | −0.001/step | shaping |

**Validation: recorded fights replayed at 10 Hz** (`scripts/replay_gohma_trace.py`; the reward parts sum exactly to the return):

| Recording | Detected | Return |
|---|---|---|
| Win | 2/2 real stuns (2 of 6 seeds), 10 HP damage, defeat, 0.5 ♥ lost. Parts: defeat 60, speed bonus 47, HP progress bonus 44, damage 40, **seed stuns 30**, stun hits 20, success 20 + 19, **slingshot shaping (ready, red-eye aim, red-eye and elevated seeds) 17.5**, other shaping ≈ 7 | **+307** (4 missed seeds −0.65) |
| Death segment | 1 larva kill, 3 ♥ lost, death | −38.6 (before the miss penalties) |
| Larva segment | 6 larva kills (1 slingshot, 5 sword), 0 stun credits; 12 seeds with misses −2.1 | +32.0 |
| Nut fight (0–258 s) | **6/6 nut stuns** detected (cap reached), 5 HP damage (+20 damage, +20 progress bonus), 6 larva kills, 2.5 ♥ lost; **19 nuts with misses −5.1** (grows as nuts ran low) | +84.0 |

## Observation v5 (552 dims = v4 483 + 69)

| Block | Dims | Contents |
|---|---|---|
| v2 global + combat core + episode memory + Baba behaviour | 99 | unchanged from v4. The core "nearest target" fields now describe Gohma, which is what lets the Deku Baba policy transfer |
| **Gohma** | 34 | present, HP fraction, **eye** relative position (x, y, z) and distance, on ceiling, 18-way state one-hot, **red eye**, stunned, time in state, **Gohma facing Link** cos/sin, **Link facing the eye** cos + signed sin (left/right) + **elevation** (aiming up at the ceiling), locked on |
| **Larvae** | 25 | alive count; nearest larva relative position, distance, HP, 11-way state one-hot, **larva facing Link** cos/sin, **Link facing the larva** cos/sin; second-nearest larva relative position and distance |
| **Items and memory** | 10 | slingshot in hand, sword in hand, first-person aim, ready to fire, seeds, time since last seed, stuns so far, larva kills so far, **nuts, time since last nut** |
| Actor table | 16 × 24 | unchanged |

The Gohma and larva blocks are all zeros in rooms without them, so one v5 policy can train on Deku Baba and Gohma slots together.

## Warm start from the Deku Baba policy

`--init-from checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` (global update 150) is widened function-preservingly:
- the 27 new action rows get zero weights; the 69 new inputs get zero weights and a bias 2 below the mean logit

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
  --num-envs 8 --train-args "--profile gohma --episode-seconds 300 \
  --init-from checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt --fleet-aggregator 4 --total-updates 400"
```

## wandb metrics
- `gohma/`: `defeat_rate`, `boss_damage`, `boss_hp_end`, `seed_stuns`, `nut_stuns`, `nuts_thrown`, `seeds_fired`, `red_eye_seeds`, `seed_hit_rate`, `stun_hits`, `larva_kills`, `larvae_seen`, `boss_attacks`, `boss_blocks`, `vulnerable_steps`, `vulnerable_lock_steps`, `sling_aim_steps`, `time_to_defeat_s`
- `reward/<component>`: every component above
- `by_slot/sN/boss_damage`, `by_slot/sN/seed_stuns`
- Plus the shared `combat/`, `episode/`, `ppo/`, `perf/` and `fleet/` metrics.

## Fleet runs

| Run | Change | Result |
|---|---|---|
| `gohma-v1` | first launch (held C-buttons) | 0.3 seeds/episode: held C never fires |
| `gohma-v2`, `v3` | C taps inside the step; stronger slingshot reward; miss penalties; grass and pickups | still ~0.4 seeds: new actions started at only 2–3% probability |
| `gohma-v4`, `v5` | new actions start at 10% (`--transfer-action-margin 0.3`); v5 adds favorable red-eye savestates (slots 4–6) and copies slot 1 into slot 10 | seeds 3–5/episode, first seed stuns. Over v5 updates 1–9 (2,937 episodes): seed stuns only in slots 3–7 (2–3% of episodes); slots 4 and 5 die 98% within ~50 s; **entrance slots farm `sling_ready` (~150 steps, 10 seeds) without starting the fight** |
| `gohma-v6` | `sling_ready` / `elevated_seed` only pre-red or red; `dropped_red_eye_shot` penalty; curriculum: half of all episodes start in slots 4–6 (`--slot-weights 4=2.33,5=2.33,6=2.33`); init from v5 update 10 | 6 updates: entrance farming stopped, nut stuns up (3.4/episode in slot 3), but red-eye seeds still ~0.2/episode in slots 4–5, ~2 dropped shots per episode, **slot 6 waits cocked ~1,400 steps and never fires** |
| `gohma-v7` | red-eye seeds +4 and never penalised as misses; `sling_ready` once per window; `unfired_red_eye` −1.5; init from v6 update 10 | ~26 updates (9,124 episodes): return −18 → −14.5, deaths 52%; **learned Deku Nut stuns (up to 5.9/episode in slot 3), not the slingshot**: red-eye seeds flat ~0.2/episode, 2–3 unfired red-eye windows and ~2.7 dropped shots per episode, seed stuns falling; 3 defeats. Locally (update 20, slot 5): missed the opening red eye, 4 nut stuns, no sword hits, died |
| `gohma-v8` | same config as v7, **trained from scratch** (no Deku Baba transfer) | (running) |

