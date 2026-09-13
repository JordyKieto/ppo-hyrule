# Combat Task: Deku Tree First Room (Deku Babas)

The agent must **defeat every Deku Baba alive at the start without dying or leaving the room, as fast as possible.**

Code: `zelda_rl/combat.py` (RAM reader, reward tracker, observation block), `zelda_rl/env.py` (`task="combat"`), `zelda_rl/ppo.py` (`RecurrentEntityActorCritic`, recurrent PPO), `scripts/train_combat.py`, `scripts/play.py` (combat HUD). How new fields were found and added: `docs/ADDING_OBSERVATIONS.md`.

**Versions:**
- `deku-combat-gru-v1` (2026-09-13 09:37, 20 pods): observation v3 (457 dims, no Baba behaviour state, heuristic block reward). Superseded.
- **`deku-combat-gru-v2` → `gru-v2b` (current, finished):** observation **v4** (483 dims). gru-v2 trained from scratch for 30 updates; gru-v2b continued from its update-30 weights with straggler-tolerant sync for 124 more updates. Fleet stopped 2026-09-13 15:02.

## ✅ Results: the task is solved (2026-09-13)

**Final policy:** `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` (**global update 150** = gru-v2b update 120, 40.7 M fleet env steps, obs v4, GRU). Play it with:
```sh
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/play.py checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt --slot 1 -n 3
```

### Artifacts (S3 + wandb)

Everything lives in the private bucket **`s3://zelda-rl/`** (us-west-1, SSE-AES256, versioned), accessed with aws profile `race-wm` (IAM user `datasets`, inline policy `zelda-rl-s3-scoped`).

| Artifact | S3 | Details |
|---|---|---|
| 10 start savestates | [`s3://zelda-rl/zelda/combat/deku-baba/states/`](https://us-west-1.console.aws.amazon.com/s3/buckets/zelda-rl?region=us-west-1&prefix=zelda/combat/deku-baba/states/) | per-slot description, RAM start conditions, sha256: [`datasets/zelda/combat/deku-baba/`](../datasets/zelda/combat/deku-baba/README.md) |
| **Final checkpoint G150** | [`…/checkpoints/G150_deku-combat-gru-v2b_u00120.pt`](https://us-west-1.console.aws.amazon.com/s3/object/zelda-rl?region=us-west-1&prefix=zelda/combat/deku-baba/checkpoints/G150_deku-combat-gru-v2b_u00120.pt) | 10/10 slots locally; [wandb run](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/vxejacj7) |
| Checkpoints G30, G90, G110, G130 | [`…/checkpoints/`](https://us-west-1.console.aws.amazon.com/s3/buckets/zelda-rl?region=us-west-1&prefix=zelda/combat/deku-baba/checkpoints/) | listed with weight hashes in `manifest.json` |
| Manifest | `s3://zelda-rl/zelda/combat/deku-baba/manifest.json` | regenerate + re-upload: `scripts/build_savestate_manifest.py --upload` |

```sh
aws s3 cp s3://zelda-rl/zelda/combat/deku-baba/checkpoints/G150_deku-combat-gru-v2b_u00120.pt checkpoints/kept/ --profile race-wm
```

### Final policy on every start slot (G150, local Mac, 1 episode per slot, sampled actions)

**10/10 successes, 0 deaths.** Six slots were cleared with no damage, including slots 4 and 7, the weakest in training (32% / 41%).

| Slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Clear time (s) | 24.3 | 14.7 | 10.9 | 2.7 | 2.7 | 17.5 | 18.6 | 8.8 | 0.5 | 12.3 |
| Hearts lost | 0.5 | 0.5 | 0.5 | 0 | 0 | 0 | 1.5 | 0 | 0 | 0 |
| Return | 83.0 | 82.9 | 83.5 | 52.5 | 53.6 | 87.1 | 76.4 | 70.5 | 51.3 | 87.0 |

Logs: `runs/play_slots1-10_gru_v2b_u120.log` (slots 1–6; that run then stalled on slot 7 when the window lost focus), `runs/play_slots7-10_gru_v2b_u120.log`.

### Training runs and wandb

Training is split across **two wandb runs**. Every rank is its own wandb run (20 per group), and all ranks hold identical averaged weights.

| Run (wandb group) | Local updates | = **Global update** | Fleet env steps | wandb |
|---|---|---|---|---|
| `deku-combat-gru-v2`: obs v4, random init | 1–30 (31–38 discarded: ranks diverged at 38) | **1–30** | 0 → 8.6 M | [group](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/groups/deku-combat-gru-v2) · [rank 4 run](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/nhj5bcde) |
| `deku-combat-gru-v2b`: init from gru-v2 u30, straggler-tolerant sync, aggregator rank 4 | 1–124 | **31–154** | 8.6 → 41.8 M | [group](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/groups/deku-combat-gru-v2b) · [rank 4 run (aggregator)](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo/runs/vxejacj7) |

- Project: [drift-ai-models/zelda-rl-combat-ppo](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo)
- **Global update G** = gru-v2 update for G ≤ 30, gru-v2b update + 30 for G > 30. One update = 8 envs × 1,800 steps = 14,400 env steps **per rank**.
- **wandb x-axis:** `global_step` is per rank and **restarts at 0 in gru-v2b**, so wandb step *S* in gru-v2b ≈ global update 30 + S / 14,400.
- **Fleet env steps** = sum over ranks of updates completed. In gru-v2b the slow ranks lag, so it counts what each rank had actually finished when the aggregator (rank 4) reached that update.
- **Checkpoints:**
  - G30 = `checkpoints/inbox/deku-combat-gru-v2_u00030.pt`
  - G150 = `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` (final)
  - also kept in inbox: G90 (`…gru-v2b_u00060.pt`), G110 (`u00080`), G130 (`u00100`)
- Raw export: `runs/wandb_export/gru_v2_v2b_history.json` (all ranks); per-G fleet means: `runs/wandb_export/fleet_progress_by_global_update.json`.

### Watched locally on the Mac: global update 150, slot 1 (3 Babas, Link 3 hearts, starting at the entrance), 3 episodes, stochastic policy

| Episode | Result | Kills at | Hearts lost | Return |
|---|---|---|---|---|
| 1 | **SUCCESS** 3/3 | 5.5, 13.9, 19.5 s | 1.0 | 83.2 |
| 2 | **SUCCESS** 3/3 | 18.1, 25.9, 32.5 s | 1.0 | 77.3 |
| 3 | **SUCCESS** 3/3 | 7.3, 14.5, **17.7 s** | **0.0** | **87.0** |

For comparison, the **global update 30** checkpoint (gru-v2 u30) on the same slot got 1 kill in 67 s and was down to 0.5 hearts (lunge hits 5, blocks 2).

**Learned fighting style** (from the play HUD / events, `runs/play_slot1_gru_v2b_u120.log`):
- **Z-target + jump attack (`Z+A`) is the kill move.** It does 2 damage, so one hit kills a full-HP Baba. Nearly every kill was one jump attack (kill speed multiplier 2.4–2.8).
- **Shield blocks** lunges (1–2 per episode), and it follows up on a **stunned** Baba (`stun_hit`) when the block knocks it back.
- Walks straight to the next Baba (330–400 units) after each kill, then locks on again.
- **Remaining weakness:** every half-heart lost came from a lunge while it was *moving* with the shield up (strafing sideways, 3 of 4), or walking in without shield or lock-on (1 of 4).

### Policy progress by global update (wandb, fleet mean over all ranks, all 10 start slots)

Each row is the mean over a ±2-update window (≈ 20 ranks × 20 episodes per update). First kill: mean over episodes with a kill. Clear time: mean over successful episodes (game time).

| Global update | Run · local update | Fleet env steps | Success | Babas killed | Death | Timeout | Hearts lost | First kill | Clear time | Block rate | Return |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | gru-v2 u1 (random init) | 0.3 M | 24% | 45% | 6.8% | 56% | 0.68 | 34.4 s | 67.8 s | 12% | 11.6 |
| **10** | gru-v2 u10 | 2.9 M | 29% | 49% | 7.0% | 57% | 0.74 | 30.3 s | 70.8 s | 12% | 15.5 |
| **20** | gru-v2 u20 | 5.8 M | 38% | 57% | 8.0% | 51% | 0.83 | 25.3 s | 70.7 s | 12% | 21.4 |
| **30** | gru-v2 u30 → gru-v2b init; **watched locally** | 8.6 M | 44% | 63% | 9.2% | 45% | 0.93 | 22.3 s | 65.1 s | 13% | 25.9 |
| **40** | gru-v2b u10 | 11.3 M | 51% | 66% | 10.5% | 38% | 0.92 | 18.3 s | 54.8 s | 13% | 29.6 |
| **50** | gru-v2b u20 | 14.0 M | 61% | 72% | 8.1% | 31% | 0.87 | 13.3 s | 46.6 s | 14% | 37.2 |
| **60** | gru-v2b u30 | 16.6 M | 65% | 74% | 8.3% | 27% | 0.86 | 11.0 s | 37.0 s | 14% | 40.7 |
| **70** | gru-v2b u40 | 19.2 M | 70% | 78% | 7.1% | 23% | 0.80 | 9.1 s | 29.8 s | 15% | 45.5 |
| **80** | gru-v2b u50 | 21.9 M | 73% | 80% | 6.1% | 21% | 0.77 | 7.9 s | 24.7 s | 17% | 48.8 |
| **90** | gru-v2b u60 | 24.5 M | 74% | 81% | 5.3% | 20% | 0.71 | 6.7 s | 21.3 s | 19% | 51.6 |
| **100** | gru-v2b u70 | 27.2 M | 77% | 83% | 4.1% | 18% | 0.66 | 6.0 s | 18.4 s | 21% | 54.7 |
| **110** | gru-v2b u80 | 29.9 M | 77% | 83% | 3.9% | 19% | 0.65 | 5.4 s | 16.6 s | 21% | 54.9 |
| **120** | gru-v2b u90 | 32.6 M | 78% | 83% | 2.8% | 19% | 0.62 | 5.1 s | 15.2 s | 24% | 56.0 |
| **130** | gru-v2b u100 | 35.3 M | 77% | 83% | 4.9% | 18% | 0.69 | 4.9 s | 14.0 s | 22% | 55.3 |
| **140** | gru-v2b u110 | 38.0 M | 76% | 82% | 4.3% | 19% | 0.67 | 4.8 s | 13.9 s | 23% | 55.7 |
| **150** | gru-v2b u120: **final checkpoint, watched locally** | 40.7 M | 76% | 82% | 4.8% | 19% | 0.67 | 4.8 s | 13.2 s | 24% | 55.5 |
| **154** | gru-v2b u124 (fleet stopped) | 41.8 M | 76% | 82% | 4.5% | 19% | 0.67 | 4.8 s | 13.2 s | 24% | 55.8 |

**Phases:**
1. **G1–30 (gru-v2, 0–8.6 M steps): learn to kill.** Success 24% → 44%, first kill 34 → 22 s. Deaths rise slightly (7% → 9%) as it starts engaging instead of wandering/timing out.
2. **G30–80 (8.6–22 M): get fast.** The kill-speed reward takes over: first kill 22 → 8 s, clear time 65 → 25 s, success 44% → 73%, timeouts 45% → 21%.
3. **G80–120 (22–33 M): get safe.** Deaths 6% → 3%, hearts lost 0.77 → 0.62, block rate 17% → 24%, clear time 25 → 15 s.
4. **G120–154 (33–42 M): plateau.** Success ~77%. The remaining ~19% are timeouts on slots 4 and 7 (hidden Babas it doesn't wake; see below). Only clear time keeps improving (15 → 13 s).

Leaving the room was ~0.3% throughout gru-v2b.

### Final by start slot (rank-4 run log, mean of last 10 updates)

| Slot | Start | Success | Babas killed | Death | Hearts lost | Return |
|---|---|---|---|---|---|---|
| 1 | 3 Babas far, entrance | **90%** | 97% | 4% | 0.96 | 76.0 |
| 2 | 3 Babas close, Z-targeting | 77% | 92% | 0% | 0.46 | 71.5 |
| 3 | 3 Babas, one preparing a lunge | 67% | 88% | 4% | 0.55 | 66.0 |
| 4 | 1 dying + 1 hidden | **32%** | 32% | 0% | 0.03 | 15.4 |
| 5 | 1 hidden | 93% | 93% | 0% | 0.80 | 45.2 |
| 6 | 3 Babas, 211–276 units | 77% | 89% | 9% | 1.07 | 65.6 |
| 7 | 1 mid-lunge + 2 hidden | **41%** | 48% | 0% | 0.42 | 35.2 |
| 8 | Link 2.5 hearts, 2 Babas | 75% | 86% | **21%** | 1.10 | 48.1 |
| 9 | Link 1.5 hearts, 1 stunned Baba | 90% | 90% | 5% | 0.13 | 44.5 |
| 10 | 3 Babas, 1 idle | 65% | 88% | 10% | 1.08 | 60.9 |

- **Slots 4 and 7 are the failures, and they fail by timeout, not death.** Both start with a hidden Baba the policy doesn't go and wake. Slot 4 also still ignores input intermittently on pods (see RUNPOD.md quirk 8), so part of its 32% is the slot, not the policy.
- **Slot 8** (low health, 2.5 hearts) has the only notable death rate (21%).

**Episode stats (fleet, last 10 updates):**
- first kill at 4.7 s; room cleared in 13.2 s (successful episodes); kill speed multiplier 2.6
- 3.1 lunges faced, 1.0 lunge hits taken, 0.8 blocks (block rate 26%), 0.7 dodges, 0.5 stun hits, 4.3 lock-ons

**Where the return comes from** (mean reward per episode): kill speed bonus +15.5, success +14.2, success time bonus +9.9, kill +9.6, enemy damage +3.8, approach +2.2, lock-on +1.3, block +0.7, link damage −2.7, death −1.1, time −0.4. **Outcome terms dominate; shaping is small (~5)**, so the policy optimizes the real objective rather than farming shaping terms.

**PPO health at the end:** explained variance 0.97, approx KL 0.010, clip fraction 0.11, entropy 3.5 (of ln 63 = 4.14), no skipped minibatches.

### Compute and cost
- **41.8 M fleet env steps** in total: 8.6 M in gru-v2 (G1–30) + 33.2 M in gru-v2b (G31–154). That's ≈1,160 game-hours of Deku Baba fighting at 10 Hz.
- 20 CPU pods (16 vCPU), 3,300–3,800 env steps/s fleet-wide, $10.38/hr. Each pod plays ≈17 game-hours per wall-clock hour (≈170 env steps/s).
- gru-v2b ran 11:24 → 15:02 (≈3 h 40 min, **≈$38**); gru-v2 G1–30 took 09:44 → 10:57 (≈1 h 15 min, ≈$13), plus ≈25 min of discarded updates 31–38.
- Pods ran at very different speeds: ranks 1, 2 and 13 fell 25–45 updates behind, but sync kept them on current weights (RUNPOD.md).

### Next stage: Gohma (boss)
Same pattern (`docs/ADDING_OBSERVATIONS.md`): record a human Gohma fight at 20 Hz, decode Gohma's action routine (eye open / stunned / climbing / spawning larvae), boss HP, and Gohma larvae (eggs + Gohma Larva actors). Then: new savestate slots at the boss door and mid-fight, a `boss` reward variant (boss damage, stun via slingshot/Deku nut to the eye, larva kills), and init from this combat policy where the observation layout allows.

---

## Start states (10 savestate slots, scene 0, room 0, Kokiri Sword + Deku Shield)

| Slot | Link HP | Deku Babas at start (HP, distance, decoded state) |
|---|---|---|
| 1 | 3.0 | 3 × 2 HP, 530–830 units, hidden. Starts at the entrance, so walking out is easy |
| 2 | 3.0 | 3 × 2 HP, 144–303 units: idle, hidden, retracting (already Z-targeting) |
| 3 | 3.0 | 3 × 2 HP, 130–355 units: hidden, retracting, **preparing a lunge** (Z-targeting) |
| 4 | 3.0 | 1 **dying** (0 HP) + 1 × 2 HP hidden |
| 5 | 3.0 | 1 × 2 HP, hidden |
| 6 | 3.0 | 3 × 2 HP, 211–276 units, hidden |
| 7 | 3.0 | 1 × 1 HP **mid-lunge** + 2 × 2 HP hidden (Z-targeting) |
| 8 | 2.5 | 2 × 2 HP, hidden |
| 9 | 1.5 | 1 × 1 HP, 91 units, **stunned** |
| 10 | 3.0 | 3 × 2 HP, 179–271 units: 2 hidden, 1 idle |

- **Targets** = Deku Babas (`0x0055`) with HP > 0 at reset.
- **Slot input check:** all 10 slots pass locally. Slot 4 has failed it intermittently on pods; the trainer drops any slot that fails at startup (it was used in gru-v2b, see the per-slot results).

## RAM signals

| Signal | Source | Evidence |
|---|---|---|
| Enemy HP | actor +0xAF | live: 2 → 1 → 0 on two sword hits |
| Enemy defeated | HP → 0, actor removed ~0.8 s later | live |
| **Deku Baba behaviour** | **+0x1B0** action-routine pointer (table below) | live fight timeline; resolves in all 10 slots |
| Baba head direction | shape rotation +0xB6 vs yaw-to-player +0x8A | spans the full circle as the head tracks Link |
| Baba action timer | +0x1B6 | counts down within actions |
| Animation mode | +0x19C = `0x8008D3C4` / `0x8008D50C` (loop vs one-shot updater) | switches with lunges and recoils (not used as a feature) |
| Lock-on target | gPlayState +0x1CCC | equals the Baba's pointer exactly while Z-locked |
| Navi arrow target | gPlayState +0x1CC8 | lockable enemy in range |
| Link HP / death | save ctx +0x30; stateFlags1 bit 7 | both flip in the same frame at death |
| Damaged / shield / Z-target | stateFlags1 bits 26 / 22 / 15 | live |
| Room / scene | gPlayState +0x11CBC / +0xA4 | walking out of slot 1 = scene 0 → 85 |
| Controls | GC L held = Z-target, GC R held = shield, B = sword, A = action / jump attack | live pipe probe |

### Decoded Deku Baba behaviour (+0x1B0)

| Pointer | State | What it looks like |
|---|---|---|
| `0x801EDF30` | `hidden` | underground / waiting (long stretches before engaging) |
| `0x801EDFDC`, `0x801EE444` | `emerge_retract` | coming up (0.7 s) / going back down when Link leaves |
| `0x801EEE94`, `0x801EE910` | `idle` | up, swaying, tracking Link between attacks |
| `0x801EEDC8` | `prepare_lunge` | ~0.35 s wind-up before every lunge |
| `0x801EEB40` | `lunge` | the attack |
| `0x801EF220` | `pull_back` | ~0.5 s after each lunge (Link damage and blocks register around lunge → pull back) |
| `0x801EF318` | `hit` | exactly when its HP drops |
| `0x801EF428` | `stunned` | knocked down for 1.2–1.9 s after a hit: the kill window |
| `0x801EF5EC`, `0x801EF950` | `dying` | HP 0 until removed |

Anything else maps to `other`. Pointers are addresses of the loaded actor code; they are identical in all 10 savestates. A savestate made after a different load order must be re-checked with `docs/ADDING_OBSERVATIONS.md` step 3.

### Traps handled (all seen live)
1. A death or reload despawns every enemy at once. That is not a kill: HP must reach 0 first, with Link alive and in the room.
2. Newly spawned actors report HP 8 for one frame. HP changes are ignored during a 2-step spawn grace.
3. The shield flag flickers off for 1–2 frames on a block. That's too short for 10 Hz sampling, so blocks are detected from Baba lunge cycles instead.

## Episode
- **Timing:** 10 Hz decisions, **max 180 s** of game time.
- **Terminal:** `success` (all targets dead), `death`, or `left_room` (scene / room change, fell ≥150 below start, or >1,400 from start; `left_reason` logged).
- **Truncation:** at 180 s, with a timeout penalty per surviving Baba.

## Reward (per-step RAM deltas; every term is logged as `reward/<name>` and they sum to the return)

| Component | Value | Kind |
|---|---|---|
| `enemy_damage` | +1 per HP point removed from a target | outcome |
| `kill` | +5 per target defeated | outcome |
| `kill_speed_bonus` | +5 × 2 × max(0, 1 − Δt/30 s), Δt = time since the previous kill / episode start | outcome |
| `success` | +20 when all targets are dead | outcome |
| `success_time_bonus` | +15 × fraction of the 180 s remaining | outcome |
| `link_damage` | −0.25 per HP unit (−2 per half heart) | outcome |
| `link_heal` | +0.05 per HP unit | outcome |
| `death` | −20, terminal | outcome |
| `left_room` | −20, terminal (same as death: no escape hatch) | outcome |
| `timeout` | −3 per target alive at 180 s | outcome |
| `lockon` | +0.3 per lock acquired on a live target (cap 6) | shaping |
| `lock_hold` | +0.005 per step locked (cap 1.0) | shaping |
| `approach` | +0.5 per 100 units closer to the nearest live target while not locked (±0.5 per step) | shaping |
| `shield_facing` | +0.01 per step shielding while facing a live target within 150 (cap 0.5) | shaping |
| **`block`** (v4) | +1.0 per lunge cycle (`lunge` → `pull_back` → other) with Link shielding and facing the Baba, within 220, **no HP lost** (cap 8) | shaping |
| **`dodge`** (v4) | +0.2 per lunge cycle survived within 165 without shielding (cap 5) | shaping |
| **`stun_hit`** (v4) | +0.5 per damage dealt while the Baba is `stunned` / `hit` | shaping |
| `time_penalty` | −0.001 per step | shaping |

**Validation on the live fight recording** (`runs/manual/fight_20260913-093042.jsonl`, subsampled to 10 Hz):
- **Detected:** 18 lunges, **5 lunge hits** (= the 5 damage events recorded), **8 blocks** (7 visible shield flickers + 1 shielded lunge without contact), 2 dodges, 3/3 kills (1 on a stunned Baba), success.
- **Reward parts:** block +8, kill +15, kill speed bonus +11.5, success +20, success time +10.4, enemy damage +6, Link damage −10, approach +5.4, lock-on +1.2, lock hold +1.0, shield facing +0.5, stun hit +0.5, dodge +0.4, time −0.56.

On the earlier death recording (v3 reward): one kill, 6 half-hearts, death, no fake kills or hits.

## Observation v4 (483 dims)

| Block | Dims | Contents |
|---|---|---|
| **v2 global** | 45 | Link position / facing / velocity, camera yaw, rupees, health fraction, time left, previous action, 14 stateFlags1 bits, sword/shield owned, flag progress, same scene |
| **Combat core** | 21 | Link HP fraction and cap; targets alive fraction and count at start; locked on a target; Navi arrow on a target; shielding; damaged; **nearest target** relative position, distance, HP, Link-facing-it cos/sin; **second target** relative position, distance, HP; time left |
| **Episode memory** | 7 | kills so far / at start, damage dealt, hearts lost, time since last hit / last damage / last kill (÷30 s), lock-ons so far |
| **Baba behaviour** (v4) | 26 | for the 2 nearest live targets: 10-way state one-hot (`hidden`, `emerge_retract`, `idle`, `prepare_lunge`, `lunge`, `pull_back`, `hit`, `stunned`, `dying`, `other`), Baba-facing-Link cos/sin (+0xB6 vs +0x8A), action timer ÷100 |
| **Actor table** | 16 × 24 | nearest 16 actors of any type: ID (embedded), category one-hot, relative position, distance, targetable / hostile / friendly, health, rupee value, params, present |

Field names and sizes are in `COMBAT_FIELDS` (`zelda_rl/combat.py`) and every checkpoint's `config["obs_fields"]`.

## Actions (63)
9 world-relative directions (neutral, E, NE, N, NW, W, SW, S, SE) × {none, A, B, Z-target, Shield, Z+B, Z+A}. Buttons are held for the whole 10 Hz step. World directions are converted to stick angles with the live camera yaw.

## Policy memory
- **GRU recurrent policy** (`RecurrentEntityActorCritic`): entity encoder → 256-d GRU → actor / critic. Hidden state is zeroed at episode start. Recurrent PPO on 60-step chunks, each re-run from its stored initial hidden state. Disable with `--no-recurrent`.
- **Explicit episode-memory features:** listed above.

## wandb metrics (project `zelda-rl-combat-ppo`, x-axis `global_step`)

| Prefix | Metrics |
|---|---|
| `combat/` | `success_rate`, `kills`, `kill_frac`, `babas_at_start`, `death_rate`, `left_room_rate`, `timeout_rate`, `end_{success,death,left_room,timeout}`, `damage_dealt`, `hp_lost_hearts`, `link_hp_start_hearts`, `link_hp_end_hearts`, `first_kill_s`, `kill_speed_mult`, `time_to_clear_s`, `lockons`, `lock_steps`, `shield_steps`, `blocks`, **`dodges`, `lunges_faced`, `lunge_hits_taken`, `block_rate`, `stun_hits`** (v4) |
| `reward/` | one per component above (mean per episode) |
| `episode/` | `return`, `r_outcome`, `r_shaping`, `length_s`, `count` |
| `by_slot/sN/` | `success_rate`, `kill_frac`, `kills`, `death_rate`, `hp_lost_hearts`, `return` |
| `ppo/`, `perf/`, `fleet/` | losses, KL, entropy, explained variance; env steps/s; sync ranks and wait |

## Commands

```sh
export DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin
.venv/bin/python-dbg scripts/train_combat.py --num-envs 3 --run-name deku-local           # local
.venv/bin/python-dbg scripts/play.py runs/deku-local/checkpoints/latest.pt --slot 7      # watch (Metal window, GRU memory)
python -m fleet launch --experiment deku-combat --script train_combat.py --wandb-project zelda-rl-combat-ppo \
  --run-tag gru-v2 --num-envs 8 --train-args "--total-updates 300"                        # fleet (pushes current code)
```
