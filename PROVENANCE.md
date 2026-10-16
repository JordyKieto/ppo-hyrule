# Provenance

Every number, checkpoint and dataset in this repository traced to its origin. This is a private research
repository; the paper will live elsewhere and must be writable from this file alone, without re-running
anything and without W&B staying up.

Evidence classes, following `../neural-simulators/PROVENANCE.md`:

- **M** measured, with a recoverable artefact in this repo
- **Q** qualitative (a watched episode, a screenshot, an author observation)
- **I** inferred from M or Q
- **A** author-confirmed
- **U** unverified — do not put in a paper without closing it

**Two tracks, tracked separately.** They share a game, a RAM reader and a state layout, and nothing else. A
claim about one is not evidence about the other.

| Track | Question | Status | Doc |
|---|---|---|---|
| **RL** | can PPO learn OoT combat from RAM state? | Deku Baba **solved**; Queen Gohma **in progress** | `docs/COMBAT_TASK.md`, `docs/GOHMA_TASK.md`, `docs/GOHMA_FOCUS.md` |
| **WM** | can a world model of OoT predict usefully far enough to plan in? | **no, so far** | `docs/TOKEN_WORLD_MODEL.md`, `docs/LATENT_WORLD_MODEL.md`, `docs/WM_EXPERIMENT_REGIME.md`, `docs/WM_MOTION.md` |

W&B is exported, not linked. `results/<run_id>/` holds `run.json` (identity, state, the full list of the 20
rank run-ids in that group), `config.json` (every flag), `summary.json` and `history.csv`. `results/INDEX.json`
is the table of contents. A paper cites the directory.

---

## 1. Environment and code, common to both tracks

| Item | Value | Class |
|---|---|---|
| Game | The Legend of Zelda: Ocarina of Time (GC, `NACE01`) | A |
| Emulator | Dolphin **2603a**, built from source; Null video backend, no rendering | M |
| Observation source | RAM only, read zero-copy from Dolphin's shared-memory arena (`/dev/shm`), ~0.8 s attach on Linux | M |
| Control rate | 10 Hz (6 VI per step) | M |
| RAM offsets | from the [OoT decompilation](https://github.com/zeldaret/oot); retail Player fields = decomp offset − 0x10 | M, `docs/RAM_MAP.md` |
| State layout | one definition, `zelda_rl/state_layout.py`, imported by collector and models so they cannot drift | M |
| Compute | RunPod: CPU pods (16 vCPU, `cpu5c`/`cpu3c`) for RL; A100/H100 80 GB for WM | M |

**Reproducibility caveat (U → close before publishing):** macOS live-RAM attach needs a code-signed
interpreter (`.venv/bin/python-dbg`); Linux pods do not. Anyone reproducing on macOS must sign their own. See
`docs/MAP.md`.

---

## 2. Datasets

| Dataset | Contents | Location | Size | Class |
|---|---|---|---|---|
| `transitions/v5` | **91,522,473 transitions / 103,716 episodes** of Deku Baba play, `badplay` policy mix | `s3://zelda-rl/zelda/transitions/v5/` | 103,877 objects, 28.0 GB | M |
| `packed/v5` | the same, packed per-field for training: 579 B/frame, 91,626,189 frames, loads in ~2.5 min instead of ~25 | `s3://zelda-rl/zelda/packed/v5/` | 30 objects, 53.1 GB | M |
| Deku Baba start states | 10 savestates, scene 0 room 0 | `s3://zelda-rl/zelda/combat/deku-baba/states/` | 10 objects, 386.6 MB | M |
| Gohma start states, CURRENT | 10 savestates, scene 17, no `ceiling_wait` | `s3://zelda-rl/zelda/boss/gohma/states/` | 10 objects, 383.8 MB | M |
| Gohma start states, PRE-REBUILD | the same set before slots 1/2/10 were replaced 2026-09-26 | `s3://zelda-rl/zelda/combat/gohma/states/` | 10 objects | M |
| Human Gohma recordings | 3 fights at 20 Hz: a win (234 s), a death (387 s), a Deku-Nut fight (707 s) | `runs/manual/boss_2026091*.jsonl` (local only) | 169 MB | M |
| Counterfactual action pairs | 150 pairs: identical state, two move actions held 3 steps, both real outcomes | `datasets/cf/pairs.npz` | — | M |

`samples/` holds small committed excerpts so the repo is self-describing; the full sets stay in S3.

**The packed set is grid-specific.** Every cell, offset and delta-cell target in `packed/v5` was computed at
nav-grid n=32 (`meta.json`). It cannot be reinterpreted at another resolution, and `train_wm_tokens.py` now
refuses a `--grid` that disagrees rather than running a confounded arm.

### Dataset gaps found and closed (evidence for the paper's data section)

| Gap | Effect | Fix | Class |
|---|---|---|---|
| Collector opened `manifest.jsonl` with mode `"w"` | 28,681 episodes invisible to every loader | append mode + resume numbering; `scripts/rebuild_manifests.py` recovered them | M |
| Per-episode analytics lost in the truncated manifests | outcome mix unrecoverable | `scripts/recover_analytics.py`, 100.00% on 10 of 11 fields with best-constant controls (`d407072`) | M |
| `free` reports the HOST, not the cgroup | a 100-epoch run SIGKILLed 20 min into its load | `wm_data.check_load_fits()` | M |

---

## 3. RL track

### 3.1 Deku Baba — SOLVED, 2026-09-13

**Result:** 10 of 10 start slots cleared locally at global update 150, 0 deaths, 6 slots with no damage taken.
Fleet mean 78% success / 83% Babas killed at G120–150.

| | |
|---|---|
| Policy | `RecurrentEntityActorCritic`: 16×24 actor table → shared per-actor MLP with id embedding → masked mean+max pool → **GRU 256** → actor/critic. `zelda_rl/ppo.py` |
| Observation | **v4, 483 dims** — 45 global + 21 combat core + 7 episode memory + 26 Baba behaviour + 16×24 actor table. `docs/COMBAT_TASK.md` |
| Actions | **63** = 9 world-relative directions × {none, A, B, Z, Shield, Z+B, Z+A} |
| Algorithm | recurrent PPO on 60-step chunks, each re-run from its stored hidden state; FedAvg across ranks via FleetLink (not DDP) |
| Reward | 18 terms, `zelda_rl/combat.py::CombatTracker.step`; the dream calls the same tracker so parity is structural |

**Run chain — resumed, not one run.** This is the thing that must not be guessed at:

| Group | Local updates | = global | Fleet env steps | Exported | Class |
|---|---|---|---|---|---|
| `deku-combat-gru-v1` | superseded (obs v3) | — | — | `results/` (20 ranks listed) | M |
| `deku-combat-gru-v2` | 1–30 | 1–30 | 8.6 M | **`results/nhj5bcde/`** (39 rows) | M |
| `deku-combat-gru-v2b` | 1–124, **init from v2 u30** | 31–154 | 8.6 → 41.8 M | **`results/vxejacj7/`** (124 rows, the aggregator, rank 4) | M |
| `dream-deku-baba` | PPO inside the learned dream | — | — | `results/andojkgi/` (267 rows) | M |

One update = 8 envs × 1,800 steps = **14,400 env steps per rank**. `global_step` is per-rank and **restarts at
0 in v2b**, so wandb step *S* in v2b ≈ global update 30 + S/14,400. Updates 31–38 of v2 were discarded after a
divergence and the run relaunched.

**Cost and wall clock** (from wandb creation/last-logged plus the fleet kill log), 20 CPU pods at $10.24–10.38/hr:

| Phase | Wall clock (PDT 2026-09-13) | Duration | Cost |
|---|---|---|---|
| creation, bootstrap, discarded obs-v3 run | 09:02 → 09:43 | 41 m | ≈ $5 |
| `gru-v2` G1–30 | 09:43 → 10:57 | 1 h 14 m | ≈ $13 |
| discarded v2 tail + relaunch | 10:57 → 11:24 | 27 m | ≈ $5 |
| `gru-v2b` G31–154 | 11:24 → 15:02 | 3 h 38 m | ≈ $38 |
| **total session spend** | 09:02 → 15:02 | **6 h** | **$61.50** (balance $74.42 → $12.92) |

41.8 M env steps ≈ **1,160 game-hours** at 10 Hz, ≈ **$1.3 per million env steps**, 3,300–3,800 env steps/s
fleet-wide. **A**: only 4 h 52 m / ≈ $51 produced the final policy.

**Checkpoint evolution.** Kept checkpoints, md5 (first 12) and size:

| File | md5 | Bytes | What | Class |
|---|---|---|---|---|
| `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` | `51cd69f624b8` | 7,419,801 | **G150, the solved policy** | M |
| `s3://…/deku-baba/checkpoints/G150_deku-combat-gru-v2b_u00120.pt` | — | — | same, archived | I (byte-identity not re-verified) |
| also on S3 | — | — | G90 (`u00060`), G110 (`u00080`), G130 (`u00100`) | M |

**Per-slot final result** (G150, local, 1 episode per slot, sampled actions) is in `docs/COMBAT_TASK.md`; it is
**Q** (one episode per slot, watched) and must not be reported as a rate.

### 3.2 Queen Gohma — IN PROGRESS, 13 numbered attempts over 3 sessions

Gohma has **10 HP**, one per sword hit while stunned. She is only damageable after a stun; a stun comes from a
slingshot seed fired while her eye is red, or a Deku Nut in any floor state. `docs/GOHMA_TASK.md` has the RAM
signals and the 18-value state table, all verified against the recordings and live probes.

**Why there are 13 versions: each is a deliberate stop, review and reward or curriculum change.** Every one is
exported. Session boundaries are where the reward or the start set changed.

| Group | Session | The change | Result | Exported |
|---|---|---|---|---|
| `gohma-v1` | 1 (09-15) | first launch | 0.3 seeds/episode — **held C never fires** | `results/qsx0rwzx/` |
| `gohma-v2`,`v3` | 1 | C taps inside the step, stronger slingshot reward, miss penalties | still ~0.4 seeds: new actions started at 2–3% probability | `results/7new3ao1/`, `results/x9r5ls6e/` |
| `gohma-v4`,`v5` | 1 | new actions start at 10% (`--transfer-action-margin 0.3`); v5 adds red-eye slots | 3–5 seeds/episode, first stuns; **entrance slots farm `sling_ready` without starting the fight** | `results/y4cwofz8/`, `results/km3uezzg/` |
| `gohma-v6` | 1 | reward gated to pre-red/red; `dropped_red_eye_shot`; curriculum 50% slots 4–6 | farming stopped; **slot 6 waits cocked ~1,400 steps and never fires** | `results/pvpx3rar/` |
| `gohma-v7` | 1 | red-eye seeds +4 and never penalised; `unfired_red_eye` −1.5 | 9,124 episodes: learned **Deku Nut** stuns, not the slingshot; 3 defeats | `results/obr4g344/` |
| `gohma-v8` | 1 | same reward, **from scratch** (no Baba transfer) | ~126 updates, return −22 → **−5.6**, 0 defeats | `results/xu1y3d9i/` |
| `gohma-v9` | 1 | **observation v6** (raw aim, camera/body frames) | stopped at 7 updates | `results/w55zpn0s/` |
| `gohma-v10` | 2 (09-15/16) | **observation v7** (all larvae, velocities, facing, shape, seed impacts) | **the baseline that mattered**: 3.29% defeat on engaged slots, 15,560 episodes | `results/1qk7isl2/` (40 ranks) |
| `gohma-v10-ext`, `-ext-lr` | 2 (09-16) | extensions of v10 | — | `results/hynl7kga/`, `results/e1gu0ho4/` |
| `gohma-band-v11` | **3 (09-26)** | `--reward simple`: 19 shaping terms zeroed, **stun credit paid only on conversion**; pre-fight slots replaced; **Z+Shield and Z+Nut added to the action space**; spacing term | defeat **3.5% → 7.0%**, death 70.8% → 62.2%, return −1.2 → +21.1, trend **+0.700/update** | **`results/p20rbwq9/`** (38 rows) |
| `gohma-band-v12` | 3 | curriculum **inverted**: 60% full-HP starts | stalled, trend **+0.098/update**; full-HP wins 4/2,008 | **`results/1czvz74p/`** (31 rows) |
| `gohma-band-v13` | 3 | **graded ladder** 30% at 5 HP / 30% at 6–8 HP / 40% full | peak 7.26% defeat; **full-HP rate 0.64%, 21× v11**; plateaued ~update 50 | **`results/kj9ypr7w/`** (74 rows) |
| `gohma-kl-v14` | **4 (10-05)** | **KL anchor to v13** (`--kl-anchor v13_r12_latest.pt --kl-coef 0.3`, init from the same file) + **death penalty removed** (`--death-scale 0`); v13's ladder unchanged; a 2-pod CPU **held-out evaluator** alongside | running; updates 15–24 fleet mean **succ 8.4 ± 1.5%**, death ~52%, `klA` 0.012 — **flat against its own v13 init (7.4%)** | pending |
| `gohma-kl-v15` | 4 (10-05) | **`g_approach`**: 0.5 per 100 units of NET horizontal progress toward Gohma, cap 3.0, silent during her attack frames. Everything else identical to v14 (same anchor, `--kl-coef 0.3`, `--death-scale 0`, same ladder), resumed per-rank from v14 update ~33 | **stopped at update 40** (960k steps/rank, 834 eps/rank). Train succ 7.8-12.4%, `klA` 0.0146->0.0167. Eval: **9/160 = 5.6%** vs v14's 18/559 = 3.2%, **p 0.122, not significant** | `checkpoints/kept/gohma-kl-v15-r4_latest.pt` |
| `gohma-kl-v16` | **5 (10-13)** | approach coefficient **0.5 -> 1.5**, cap 3.0 -> 7.0 (one variable vs v15); resumed from v15 update 40 | **stopped at update 20.** 0/5 bands moved vs v15 at matched updates; started LEVEL with v15 despite a 40-update head start; worst late slope of any version, **-0.210/update** | `checkpoints/kept/gohma-kl-v16-r4_latest.pt` md5 `0e11fcccb6f5` |
| `gohma-kl-v17` | 5 (10-13) | coefficient **reverted to v15's 0.5** / cap 3.0, and **adaptive start-slot sampling** (`--slot-sampling deficit --slot-deficit-ratio 4.0 --slot-deficit-floor 0.06 --slot-warmup 3`); resumed from **v15** update 40, so one variable vs v15 | running | pending |
| `gohma-kl-v18` | 5 (10-13) | **v14's reward** (`g_approach` off) on **v15's weights**; no deficit sampling; anchor/ladder/`--death-scale 0` unchanged | **stopped at update 13.** Damage 3.40 -> 2.94; slope -0.0370 (t=-3.49) on all 13 points but **-0.0194 (t=-0.95) dropping the first 5**, so the steepness was a start transient. Settled BELOW v15's 3.10-3.36 range and showed no sign of v14's +0.012 | `checkpoints/kept/gohma-kl-v18-r4_latest.pt` md5 `7bd6b58ed9cb` |

**v14 is two changes at once** (anchor + no death penalty), so a positive result will not say which did
it. That was accepted deliberately to get one run instead of two; it is recorded here so no paper claims the
KL anchor alone from this run. **U**.

**Instrument error in the v14 evaluator, found and fixed 2026-10-05 — two reported holdout numbers are
withdrawn.** `scripts/eval_fleet.py` round-robined the start slot **per round** rather than per episode, and a
round ran `--num-envs` episodes. So:

- the first v13 hold-out figure was 48 episodes that all landed on **slot 1**, reported as a four-slot
  hold-out rate. Withdrawn.
- the first v14 full-suite rows (`results/eval/pods/rank{0,1}/v14_full.jsonl`, published update 15) ran 12
  episodes each on 4 slots and **zero on slots 9 and 10**, while listing all five in the row's `slots` field.
  The per-slot sub-rows in those files are real measurements of the slots they name; the aggregate and the
  claim of ten-slot coverage are not. The full-HP figure from them (**0/36 defeats**) stands, since both
  10 HP slots evaluated were among those actually run.

Fixed by replacing the round structure with a per-env work queue, which also removed the round barrier that
idled eleven emulators behind one 300 s episode: 40 episodes now complete in ~2 min instead of ~7. Rows from
the fixed evaluator are written to a **separate file** (`v14_full_q.jsonl`) so the two are never pooled.

**The v14 hold-out rate is confounded, and the confound is structural.** The first 10-slot suite from the
fixed evaluator (published update 20, 80 episodes over the 2 eval pods, `results/eval/v14_full_q_rank{0,1}.jsonl`):

| | n | defeat | mean boss HP taken |
|---|---|---|---|
| held out `[1,2,9,10]` | 32 | **0** | 1.44 |
| trained `[3,4,5,6,7,8]` | 48 | 10.42% | 2.23 |
| full HP (10) | 24 | 4.17% | 2.08 |
| **near** < 200u | 16 | **18.75%** | — |
| **mid** 200–400u | 24 | 4.17% | — |
| **far** > 400u | 40 | 2.50% | — |

The held-out set is defined as the slots the run did not train on, and v13's ladder trains `[3,4,5,6,7,8]` —
so held-out is *necessarily* `[1,2,9,10]`, which is **all 5 HP and all 368–490 units out**. "Held out" is
collinear with "far", and trained slots at the same distance score the same (slot 5 at 472u: 0/8; slot 8 at
448u: 0/8). No trained slot is both far and 5 HP, so **no matched comparison exists on this curriculum** and
the 0/32 is not evidence of a generalisation gap. `scripts/eval_fleet.py` now prints a `CONFOUNDED` line and
sets `holdout_confounded` on every such row.

The useful number from the same 80 episodes is the **distance gradient: 18.75% → 4.17% → 2.50%**, a 7.5× fall
from <200u to >400u. This moves claim **R9** (distance, not boss HP, is the residual wall) from **I** to **M**:
it was inferred from slot 8 alone and is now measured across three bands of a ten-slot suite. Full HP is
1/24 here, which at n=24 is *not* distinguishable from v13's 0.64% — the interval covers both.

**Why v15 exists: Gohma had never had an approach reward at all.** `GohmaTracker.step` fully replaces
`CombatTracker.step` and never calls `super()`, so the generic `approach` term (`w_approach`,
`zelda_rl/combat.py`) **never fired in any of v1–v14**. The only distance term that ran was
`approach_stunned`, gated on `GOHMA_STUNNED` — it pays for closing the last few units *after* she is already
knocked down, and pays nothing for crossing the room to reach her. The 18.75% → 2.50% distance gradient above
is a reward gap, not a capacity limit: nothing ever asked the policy to approach. **A**, confirmed by reading
both `step` implementations.

`g_approach` is paid on **net** progress (closer than the closest Link has been this episode), not the
per-step delta: a signed delta is farmable by oscillating, and `_capped` ignores negative values, so the
retreat half would have been free — the same shape as the `g_sling_ready` and `g_spacing` farming already
recorded above. It is silent during `GOHMA_ATTACK`, where `g_spacing` pays for holding the 150–600 band.

Two bugs were caught by `tests/test_gohma_approach.py` **before** any compute went into them, and both are
the kind that would have produced a confident wrong conclusion:

1. `_d_best` seeded to infinity made the **first step bank the entire 3.0 cap with Link standing still**.
   Seeded from the start distance instead.
2. `math.dist` is 3D and Gohma spends the fight on the **ceiling** before descending, so the gap shrank by
   roughly her ceiling height when *she* came down and Link had not moved. On the recorded human win a third
   of the payment was her descent. `hdist` is horizontal (x/z) — the floor distance Link must cross.

Validated against the three recorded human fights (`runs/manual/boss_*.jsonl`) through
`scripts/replay_gohma_trace.py`'s loader: the human **win** pays `approach` **+3.00 of a +247.17 return
(1.2%)**, so it is a positioning nudge and cannot dominate an episode. It pays the same +3.00 on the human
**death**, which is expected — it is shaping, and the outcome terms are what separate win from loss. **M**.

**v15's result, in full, as of the fleet being stopped at update 40.** 160 evaluation episodes over 4
full-suite runs (its updates 25–35), against v14's 559-episode baseline. One-sided Fisher exact, Wilson
intervals:

| band | v15 | 95% CI | v14 | 95% CI | p |
|---|---|---|---|---|---|
| near < 200u | 5/48 | 4.5–22.2% | 11/128 | 4.9–14.7% | 0.453 |
| mid 200–400u | 2/40 | 1.4–16.5% | 4/159 | 1.0–6.3% | 0.347 |
| far > 400u | 2/72 | 0.8–9.6% | 3/272 | 0.4–3.2% | 0.281 |
| full HP (10) | 2/56 | 1.0–12.1% | 3/176 | 0.6–4.9% | 0.349 |
| **overall** | **9/160** | 3.0–10.3% | 18/559 | 2.0–5.0% | **0.122** |

**The verdict is UNPROVEN, class U, and must not be cited as an improvement.** Every interval overlaps v14's
and nothing approaches p<0.05. What is true and worth keeping is the *shape*: all four bands point the same
way, and the **near** band — the one that was never broken — moved least (5/48 vs 11/128), which is the
asymmetry an approach reward predicts and that noise has no reason to produce. The power calculation says
~400 far-band episodes would be needed for p≈0.04 at the observed 3.57%; 72 were collected.

`reward/approach` was confirmed firing in live episodes at **mean +0.609/episode across 18 ranks** (range
0.257–0.982), i.e. ~120 units of net horizontal progress per episode against start distances of 188–541. The
cap is 3.0 and the human win banks the full 3.0, so **the term was active and the behaviour it pays for was
only ~20% realised** — if the trend is real but weak, raising the coefficient is the untested next step.

**Three instrument bugs were found and fixed during this run, each of which had been corrupting the
measurement silently:**

1. **The publish pointer dropped every checkpoint from the resumed run.** `FleetLink._publish_policy`
   guarded on `update > cur["update"]`, and v15's counter restarted at 0 against v14's stale pointer at 40 —
   so v15's publishes at 5/10/15/20 were all discarded and the eval fleet re-scored a 57-minute-old v14
   checkpoint while logging that it was watching the live experiment. It would have unblocked itself once
   v15 passed update 40. Guard is now per-launch.
2. **A Dolphin failing to boot orphaned the other eleven.** Env construction sat outside the `try/finally`,
   so one `N64 RAM not found` left 12 processes alive owned by nobody, and the next checkpoint would have
   started twelve more. Observed live. Partial failures now close what came up, plus one retry.
3. **The eval shard balanced by slot COUNT across slots that differ 6x in cost.** Slots 6 (210 s) and 8
   (251 s) both landed on one pod (7.4 min/suite vs 3.0), and slots 4/8/10 — three of the four starts beyond
   400 units — were all on the slow pod, throttling the exact band the result depended on. Now greedy
   longest-processing-time: 479 s vs 458 s.

**Step-axis discontinuity:** v15 resumes by `--init-from`**Step-axis discontinuity:** v15 resumes by `--init-from`, not `--resume`, so its `update` counter restarts
at 0 while the weights continue from v14 update ~33. **Reproducing v11–v14 under the `simple` preset now
requires `--combat-json '{"g_approach": 0}'`**, because the preset turns the term on.

### THE CENTRAL RETRACTION: six versions reported as plateaued were improving, and the fix broke them

Every "is it improving" answer in this project came from `combat/success_rate` — boss defeated, a binary event
at 1–10%. On that signal v10 through v15 read flat and two experiments were launched to break the plateau. On
**boss damage**, the HP taken off her out of 10, the same runs read (**M**, W&B history, slope ± SE, `*` =
|t|>2):

| version | `g_approach` | boss damage | slope | return |
|---|---|---|---|---|
| `gohma-v10` | — | 0.38 → 1.21 | **+0.003\*** | −12.58 → 15.46 **+0.105\*** |
| `gohma-band-v11` | — | 1.72 → 2.10 | **+0.015\*** | 1.06 → 18.75 **+0.700\*** |
| `gohma-band-v12` | — | 2.06 → 2.22 | **+0.008\*** | 4.87 → 6.63 +0.081 |
| `gohma-band-v13` | — | 2.41 → 2.75 | **+0.007\*** | 18.93 → 26.36 **+0.168\*** |
| **`gohma-kl-v14`** | **0.0 (absent)** | **2.86 → 3.22** | **+0.012\*** | 45.54 → 52.62 **+0.226\*** |
| `gohma-kl-v15` | 0.5 | 3.17 → 3.32 | +0.004\* | 54.25 → 56.79 +0.078 |
| `gohma-kl-v16` | 1.5 | 3.12 → 2.85 | **−0.017\*** | 56.08 → 46.88 **−0.531\*** |
| `gohma-kl-v17` | 0.5 + deficit sampling | 3.19 → 2.86 | **−0.022\*** | 52.35 → 37.55 **−1.008\*** |

**The project improved 0.38 → 3.32 HP, 8.7×, with every step statistically solid, and it was reported as six
consecutive failures.** A defeat needs ten successful hits; damage counts each one, so one signal can show a
trend on a few hundred episodes and the other cannot. Every claim in this file of the form "version N
plateaued" or "no version was trending" is **withdrawn**.

**And the wrong metric did real harm.** `g_approach` was added in v15 from a defeat-rate band analysis. The
dose-response is monotone in its coefficient — v14 without the term was the last version learning, 0.5 halved
the slope, 1.5 reversed it. **The term introduced to fix the plateau is what created it.** Confirmed on the
mix-invariant per-slot form, so neither regression is a curriculum artefact:

| version | per-slot boss damage, first ⅓ → last ⅓ | slope | t |
|---|---|---|---|
| v15 | 3.250 → 3.275 | +0.0007 | +0.49 (flat) |
| v16 | 3.141 → 2.722 | −0.0284 | **−4.75** |
| v17 | 3.335 → 2.980 | −0.0236 | **−3.78** |

Both v16 and v17 are **my** changes: the 3× coefficient and the deficit sampling. Both are regressions against
the policy they resumed from. Recorded in `paper/CLAIMS.md` as R23/R24 superseded, and as AGENTS 34.

### The dose-response is CONFOUNDED by the starting level -- a retraction of this session's own finding

`g_approach` was removed on the strength of a monotone dose-response (v14 +0.012, v15 +0.004, v16 -0.017).
v18 then ran v14's reward on v15's weights and did **not** recover v14's slope. Reading the damage column as
a curve rather than a league table explains why:

| version | starts at | ends at | slope |
|---|---|---|---|
| v10 | 0.38 | 1.21 | +0.003 |
| v11 | 1.72 | 2.10 | +0.015 |
| v12 | 2.06 | 2.22 | +0.008 |
| v13 | 2.41 | 2.75 | +0.007 |
| v14 | 2.86 | 3.22 | +0.012 |
| v15 | 3.17 | 3.32 | +0.004 |
| v16 | 3.12 | 2.85 | -0.017 |
| v18 | 3.40 | 2.94 | -0.021 (n.s.) |

**Each version starts further along and climbs more slowly.** The slope correlates with the STARTING LEVEL,
not only with the reward term, so the `g_approach` dose-response cannot be separated from saturation: runs
beginning near 2.6 climb and runs beginning near 3.3 do not. **The claim that `g_approach` caused the
plateau is downgraded to I, confounded.** It may still be harmful -- nothing here exonerates it -- but this
data cannot show it.

What the curve does support, at **M**: **boss damage saturates at 3.2-3.4 of 10 across eight versions and
five reward configurations.** That is a statement about the task, and it means the next question is why the
last six HP are unreachable, not which reward term to try ninth. `docs/GOHMA_FOCUS.md` **WHERE WE LEFT OFF**
lists the four candidate investigations in order, the first of which is to watch ten episodes at the ceiling
and find out what stops hit four -- never once done at this level.

### v18: v14's reward, v15's weights, boss damage as the headline

`g_approach` is **off** — which is v14's reward exactly, since the term did not exist before v15. Deficit
sampling is **not used** (the code and its 25 tests stay, recorded as a negative result). Weights resume from
`gohma-kl-v15-r4_latest.pt`, which is the peak level on the mix-invariant damage metric (3.275 against v14's
3.22), so v18 is **v14's reward applied to the best weights**. Anchor, ladder, `--death-scale 0` unchanged.

`combat/boss_damage_balanced` and `combat/boss_damage_hard` now LEAD the trainer's log line, before any
success rate, and are asserted mix-invariant in `tests/test_band_metrics.py`. The reference to beat is
**v14's +0.012 HP/update**, and the judgement is a slope with a standard error, not a level.

### CORRECTION: the start-slot distances were 3D, and six of ten were overstated

Asked whether any start slot is still the PRE-FIGHT state (`ceiling_wait`, which needs the cutscene triggered
by looking up), `scripts/audit_gohma_slots.py` was written to read Gohma's actor state out of the emulator on
a pod rather than trust the audit table in `docs/GOHMA_FOCUS.md`. **M**, `runs/gohma_slot_audit.json`.

**All ten slots are engaged combat states; no `ceiling_wait` remains.** The rebuild of slots 1/2/10 held and
S3 carries the fixed states. But the measured horizontal distances disagree with the table this project has
been banding on:

| slot | state | table (3D) | measured horizontal |
|---|---|---|---|
| 3 | `stunned` | 188 | 188.5 ✓ |
| 6 | `floor_eye_red` | 357 | 357.0 ✓ |
| 7 | `stunned` | 38 | 37.9 ✓ |
| 8 | `stunned` | 448 | 448.0 ✓ |
| 2 | `ceiling_move` | 368 | **140.1** |
| 9 | `ceiling_spawn` | 394 | **210.6** |
| 5 | `ceiling_eye_red` | 472 | **346.7** |
| 4 | `ceiling_spawn` | 541 | **436.0** |
| 1 | `ceiling_move` | 473 | **424.2** |
| 10 | `ceiling_move` | 490 | **424.9** |

The four that agree are exactly the four **floor** states; every slot that disagreed is a **ceiling** state,
and each discrepancy reconciles to ~320–340 units of ceiling height — slot 2 `sqrt(368² − 140²) = 340`, slot 9
333, slot 5 320, slot 4 320. The old table was 3D distance, which includes a vertical component Link never
crosses. `g_approach` had already been switched to horizontal (`hdist`) a week earlier for exactly this
reason and these constants were never reconciled with it: **AGENTS 3**.

**Band reassignment:** slot 2 mid → **near**, slot 5 far → **mid**. Rebanding v14's pooled 559 episodes:

| band | as reported (3D) | corrected (horizontal) |
|---|---|---|
| near < 200u | 8.59% (n=128, slots 7,3) | **6.82%** (n=176, slots 7,2,3) |
| mid 200–400u | 2.52% (n=159) | **2.86%** (n=175, slots 9,5,6) |
| far > 400u | 1.10% (n=272) | **0.48%** (n=208, slots 1,10,4,8) |

**Claim R9 survives and strengthens**: the fall from near to far is **14×**, not the 7.5× reported, with the
far band at **one win in 208 episodes**. The numbers in the earlier R9 row are superseded by these.

The constant now lives once, in `zelda_rl/combat.GOHMA_SLOT_DIST`, imported by `eval_fleet.py`,
`eval_compare.py` and `train_compare.py`; four separate copies existed and the ones in the scripts were the
3D values. `tests/test_band_metrics.py` derives which slot is in which band FROM the constant — it previously
hardcoded "far band = slot 5 (472u)" and failed the moment the correction landed, which is the test working
but should not have been expressible.

**Scope of the error.** v17 is running with the old constant compiled into its `by_band/*` metrics, so its
`mid`/`far` splits put slot 5 in `far`. Its headline `combat/success_hard` is **unaffected**: `hard` excludes
the near band, which for the trained set (3–8) is slots 7 and 3 under both tables. The run was not restarted
for this.

### The version trend table, and the metric that misread three decisions

Fleet-mean `combat/success_rate` by third of each run, from the exported W&B history. **M**.

| version | updates | first 1/3 | mid 1/3 | last 1/3 | slope/upd | late slope |
|---|---|---|---|---|---|---|
| `gohma-v8` | 129 | 0.03% | 0.15% | 0.08% | +0.001 | +0.004 |
| `gohma-v10` | **399** | 0.13% | 0.65% | 1.21% | +0.004 | +0.005 |
| `gohma-band-v11` | 37 | 3.27% | 5.38% | **7.66%** | **+0.175** | **+0.270** |
| `gohma-band-v12` | 30 | 1.88% | 2.14% | 1.65% | −0.013 | +0.098 |
| `gohma-band-v13` | 73 | 5.56% | 6.81% | 6.86% | +0.033 | +0.048 |
| `gohma-kl-v14` | 40 | 7.48% | 8.85% | 8.79% | +0.044 | −0.092 |
| `gohma-kl-v15` | 47 | 9.52% | 10.69% | 9.82% | +0.011 | −0.195 |
| `gohma-kl-v16` | 20 | 9.63% | 9.33% | 7.99% | −0.166 | −0.210 |

**RETRACTION: v11's trend, and therefore the reasoning that produced v12, was an artefact of the curriculum
mix.** v11 was stopped at update 37 because it looked like the fastest-improving version. Its per-slot
history (**M**, W&B `by_slot/s*/success_rate`):

| slot | distance | first 1/3 | last 1/3 | slope |
|---|---|---|---|---|
| **7** | **38 u** | 7.32% | **14.00%** | **+0.298** |
| 3 | 188 u | 0.00% | 0.00% | 0 |
| 6 | 357 u | 0.00% | 0.00% | 0 |
| 2 | 368 u | 0.00% | 0.00% | 0 |
| 8 | 448 u | 0.00% | 0.00% | 0 |
| 5 | 472 u | 0.00% | 0.00% | 0 |
| 1 | 473 u | 0.00% | 0.00% | 0 |
| 4 | 541 u | 0.00% | 0.00% | 0 |

v11 was learning the one start where Link begins on top of Gohma. Nine slots were at **exactly zero for 37
updates**, and the aggregate climbed because that run weighted slot 7 at 50%. **No claim that v11 was
improving on the task survives**, and claim R8's "v11 +0.700/update" is the same artefact.

The instrument then misread in the other direction. v15's per-slot profile (**M**):

| slot | 38 u | 188 u | 357 u | 448 u | 472 u | 541 u |
|---|---|---|---|---|---|---|
| v15 | 28.46% | 3.83% | 4.07% | 0.51% | 6.04% | 1.98% |
| v11 | 14.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

**v15 is the best policy this project has produced**, non-zero on all six trained slots where v11 was non-zero
on one, and ~3.3% against ~0.13% off the easy slot — about **25× broader**. It reads as the flattest run in
the table. Fixed in code by `zelda_rl/combat.band_metrics()`: see AGENTS 29.

Also settled by the table: **more time was not the missing ingredient.** `gohma-v10` ran **399 updates** and
ended at 1.21%; `gohma-v8` ran 129 from scratch and never left 0.1%.

### RETRACTION: the held-out checkpoint selector had no resolution, and nothing read it anyway

The eval fleet was built to answer "which policy do we keep" by held-out defeat rate. It ranked on
**16-episode** held-out subsets (2 held-out slots × 8 episodes per eval rank). Pooled held-out rate over every
v14 evaluation: **7/416 = 1.68%**. At that rate a 16-episode subset reads:

| it shows | probability |
|---|---|
| 0.00% | 76.2% |
| **6.25%** | **20.9%** |
| 12.50% | 2.7% |

So the 6.25% that selected `gohma-kl_r9_u00040.pt` is **one win in sixteen**, which any checkpoint produces
one time in five. Tested directly: the selected checkpoint (2/32) against every other v14 checkpoint (5/384)
gives **one-sided Fisher p = 0.094 — not distinguishable**, CIs 1.7–20.1% against 0.6–3.0%. Separating a
genuinely 2× better checkpoint needs **~256 held-out episodes each**, 16× what it had. **M**.

And the loop was never closed: `.best.json` is written by the evaluator and **read by nothing**. Both resumes
(v16 from v15, and the v14 checkpoint kept) used `latest.pt` chosen by recency while a "best" file sat beside
it — v14's selected `r9_u00040` was never kept at all, and v15's `.best.json` was destroyed with its pods.
That cost nothing *because* the selection was meaningless, which is luck rather than judgement. See AGENTS 30.

### Corrected v15-vs-v14 hold-out comparison

The figure reported during the session (p=0.122) was against a **559-episode subset** of an 880-episode v14
baseline, and the pool also contained retracted rows and duplicates. With `scripts/eval_compare.py` (content
rejection of round-based rows, dedup on `(checkpoint, rank, ts)`, `.withdrawn` / `.run_label` markers):

| band | v15 | 95% CI | v14 | 95% CI | p |
|---|---|---|---|---|---|
| near < 200u | 5/48 | 4.5–22.2% | 16/176 | 5.7–14.3% | 0.482 |
| mid 200–400u | 2/40 | 1.4–16.5% | 8/264 | 1.5–5.9% | 0.388 |
| far > 400u | 2/72 | 0.8–9.6% | 3/440 | 0.2–2.0% | 0.147 |
| full HP | 2/56 | 1.0–12.1% | 3/264 | 0.4–3.3% | 0.212 |
| **overall** | **9/160** | 3.0–10.3% | 27/880 | 2.1–4.4% | **0.087** |

Still **not significant**; 5/5 bands point the right way. **U**. The three pooling errors this corrected are
recorded in AGENTS 32 because the correction is the reusable part.

### v16: the approach coefficient tripled, and bought nothing

`g_approach` 0.5 → 1.5, cap 3.0 → 7.0, one variable against v15, resumed from v15 update 40. The term was
confirmed firing at `reward/approach` mean **+1.932** over 19 ranks (v15's identical behaviour would pay
+1.83, i.e. ~129 units of net progress against v15's ~120). Banded training comparison at matched updates
1–10, same instrument both arms:

| band | v16 | v15 | p |
|---|---|---|---|
| near < 200u | 13.64% | 14.25% | 0.665 |
| mid 200–400u | 2.71% | 4.39% | 0.907 |
| far > 400u | 2.46% | 2.58% | 0.569 |
| full HP | 2.61% | 2.78% | 0.706 |
| overall | 6.23% | 6.77% | 0.741 |

**0/5 bands point the right way.** And because v16 *resumed from* v15's update 40, its update 1 should already
have been above v15's update 1 if the approach reward accumulated anything over v15's 40 updates — it starts
level or marginally below. That is evidence against the approach reward having delivered anything, independent
of the coefficient. Stopped at update 20. **M**.

### v17: the sampling distribution as the experiment

PPO optimises expected return **under the sampling distribution**, so the curriculum weights are part of the
objective. v16 spent 30% of its episodes on the slot it already won 28% of the time and 13% on the one it won
0.51% of. `zelda_rl/combat.deficit_weights()` puts `w_i ∝ 1/(p_i + c)` with `c` solved so the least-solved slot
gets *exactly* `--slot-deficit-ratio` times the most-solved, from an EMA of the policy's own per-slot success.
On v15's measured rates at ratio 4:

| slot | dist | v15 rate | old mix | v17 mix |
|---|---|---|---|---|
| 7 | 38 u | 28.46% | 30.0% | **6.0%** (floor) |
| 3 | 188 u | 3.83% | 13.3% | 17.5% |
| 6 | 357 u | 4.07% | 15.0% | 17.2% |
| 8 | 448 u | 0.51% | 13.3% | **23.8%** |
| 5 | 472 u | 6.04% | 13.3% | 14.9% |
| 4 | 541 u | 1.98% | 15.0% | 20.6% |

Episodes on the five unsolved slots: **69.9% → 94.0%**. The 6% floor is load-bearing — v12 is the warning, a
hand-inverted curriculum that starved what it had learned and stalled at +0.098/update; at ratio 4 slot 7
would get 5.95% unaided and the floor lifts it. This is measurable **only** because `success_balanced` is
mix-invariant. 25 assertions in `tests/test_deficit_weights.py` and `tests/test_band_metrics.py`. **U**.

**Observation evolution** (each is a widening, function-preserving at load via `zelda_rl/transfer.py`):
v4 483 → **v5 552** (+Gohma/larva/item blocks, actions 63 → 90) → **v6 675** (+raw aim pitch/yaw, camera and
body frames) → **v7 1033** (+3 larva slots, velocities, facing, shape, seed impacts). Actions 90 → **108** on
2026-09-26 when `("L","R")` Z+Shield and `("L","CL")` Z+Nut were added; indices 0–89 keep their meaning, so
every earlier checkpoint still transfers.

**Checkpoint evolution:**

| File | md5 | Bytes | What | Class |
|---|---|---|---|---|
| `checkpoints/kept/gohma-init-from-deku-G150.pt` | `df4a8175f22e` | 2,575,724 | Baba G150 widened to the Gohma spec | M |
| `checkpoints/kept/gohma-v5-init-deku-G150-m0.3.pt` | `1ea6b142d37c` | 2,576,375 | same, new actions at 10% | M |
| `checkpoints/kept/gohma-v5-u00010.pt` | `c72ab1070e98` | 7,717,337 | v5 u10 | M |
| `checkpoints/kept/gohma-v6-u00010.pt` | `5e1b0ab08ff1` | 7,717,401 | v6 u10 | M |
| `checkpoints/inbox/gohma-r13_u00050.pt` | `da49fba4a63c` | 8,128,749 | **v10 rank 13 u50** — best of 18 ranks by late defeat rate (8.43%); the v11 starting point | M |
| `runs/gohma_v13/v13_r12_latest.pt` | `49cf19963f4f` | 8,184,493 | **v13 rank 12 update 70 — the best Gohma policy to date**, 7.4% defeat, optimizer state intact | M |

**Recreating the Gohma chain** — the exact resume lineage, which nothing else records:

```
Baba G150 (checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt)
  └─ widened → gohma-v5 … v7        (obs v5, actions 90, --transfer-action-margin 0.3)
     v8, v9, v10 trained FROM SCRATCH (v10 = obs v7, actions 90)
       └─ gohma-r13_u00050.pt  ── v11 ─→ v12 ─→ v13
          (obs v7; actions widened 90→108 at v11; each of v12/v13
           resumed per-rank from `runs/gohma-band-v{N-1}-r{rank}/checkpoints/latest.pt`)
```

**v11/v12/v13 launch commands are in `docs/GOHMA_FOCUS.md`** verbatim, including slot weights. Fleet: 20 CPU
pods, 8 envs each, $10.32/hr, aggregator rank 4, one update = 8 × 3,000 = 24,000 env steps per rank.

---

### 3.3 Human demonstrations and BC-regularised PPO — 2026-10-15

**The corpus (M).** The author played Queen Gohma to death 9 times under `scripts/record_human.py`,
which records at 10 Hz: the v7 observation the policy sees, the action index the human's input decodes to, the
**raw pad** (button mask, stick x/y read from `gPlayState+0x14`), and the camera yaw used for that decode.
Every `.npz` is self-describing -- `obs_fields`, `action_space`, `pad_bits`, `recorded_utc`, `git_commit` --
so a remote trainer never reconstructs the layout from this repo's current source (AGENTS 25, 27).

9 episodes, 6,738 steps, **9 wins out of 9**, 8 distinct start states, obs_dim 1033,
n_actions 108, 93/108 actions used (86%). Mirrored to `s3://zelda-rl/zelda/human/gohma/` and pushed to every
pod by presigned GET, verified by size at the path the trainer opens (AGENTS 16, 20).

| File | md5 | Steps | Damage | Slot | Start state |
|---|---|---|---|---|---|
| `ep04_slot5_WIN.npz` | `adf411fe6829` | 759 | 10/10 | 5 | ceiling_eye_red |
| `ep05_slot6_WIN.npz` | `7314e100c9fa` | 281 | 6/10 | 6 | floor_eye_red |
| `ep06_slot7_WIN.npz` | `ad21a604335d` | 593 | 5/10 | 7 | stunned |
| `ep07_slot8_WIN.npz` | `9edc30c1d70b` | 2379 | 10/10 | 8 | stunned |
| `ep08_slot9_WIN.npz` | `199397154802` | 475 | 5/10 | 9 | ceiling_move |
| `ep09_slot10_WIN.npz` | `784587c2af82` | 547 | 5/10 | 10 | ceiling_move |
| `ep10_slot5_WIN.npz` | `5899fdfe1bff` | 1235 | 10/10 | 5 | ceiling_eye_red |
| `ep11_slot6_WIN.npz` | `6521195a8dd7` | 307 | 6/10 | 6 | floor_eye_red |
| `ep12_slot7_WIN.npz` | `5c54a7ce4d8c` | 162 | 5/10 | 7 | stunned |

Gate before upload: `scripts/verify_human_dataset.py` rejects a corpus with constant pad input, fewer than
three distinct actions, non-finite observations, out-of-range action indices, length mismatches or md5 drift.
It printed the sync command only after all 9 rows passed.

**Input decoding, verified rather than assumed (M).** The pad offset is proven two ways: `--selftest` drives
the pad through the env's own pipe and reads it back (11 cases, all pass, no controller needed), and
`tests/test_record_human.py` round-trips **every** action the policy can emit through the stick the env would
send and requires the same index back at five camera yaws. The first version of that mapping treated the
direction as an octant *index* rather than a world angle and ignored the camera -- it produced plausible
recordings that were wrong for every non-neutral direction. `L = 0x2000` (Z-target) and `R = 0x0010` (shield)
were confirmed **by effect** on the real controller, and `0x0020` is unmapped in OoT.

**Two environment facts that each cost time, now checks (AGENTS 21).** Live RAM attach needs the signed
interpreter `.venv/bin/python-dbg` (`docs/MAP.md` line 66) **and** `DOLPHIN_BIN` exported to
`~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin` (line 73). With the `/Applications` build the game
boots the **Wii System Menu**, `task_for_pid` is denied, and `attach_n64_ram` hides both behind
`TimeoutError: N64 RAM not found in Dolphin process` for 60 s -- a message naming the emulator rather than
either cause.

**The clone (M, single run).** `scripts/train_bc_human.py` trains the same `RecurrentEntityActorCritic` the
PPO trainer builds, split **by episode** across start slots, GRU state carried inside an episode.

| | value |
|---|---|
| held-out action accuracy | **53.4%** (epoch 16) |
| majority-action floor | **11.1%** (action 0, neutral) |
| train accuracy at epoch 56 | 88.6% — overfits after ~16, so the best-val checkpoint is the kept one |
| checkpoint | `checkpoints/kept/bc_human.pt`, md5 `531f2a012817`, 2,726,211 bytes |

**n = 1 run and 2 held-out episodes** (AGENTS 7): this is a single measurement, not a range. The earlier split
held out *both* slot-1 episodes, which measured transfer to an unseen start rather than imitation quality;
fixed to one episode per slot from slots with a spare.

**What 53.4% is NOT (Q).** It is frame-matching, not competence. The first closed-loop rollouts of the BC
weights under PPO died at 20-50 s with 0 kills on every env -- compounding error into states the
demonstrations never visited. The pure-BC policy's held-out boss damage is being measured on the eval pod and
is the **M** number for this; until it lands, any claim that the clone can fight is **U**.

**BC as a regulariser, not only an init (M for the mechanism, U for the outcome).** `--init-from` decays: PPO's
gradient walks away from the demonstrations within a few updates. `zelda_rl/bc_demos.py` pre-chunks the corpus into
168 windows of 60 and `zelda_rl/ppo.py` adds `bc_coef * cross_entropy` on an equal number of human windows per
minibatch -- regularising against the **data**, where `--kl-anchor` regularises against a **policy**.
`tests/test_bc_regulariser.py` is the control: coef 1.0 drives agreement with the demonstrated action to
**96.9%** from an 11.1% chance floor, coef 0.0 reports no statistics and reaches **0%**, so the term provably
trains and is provably off when disabled.

**One self-inflicted incident worth recording.** The demonstration loader was first written to
`zelda_rl/bc.py` with a shell redirect, which **overwrote the WM track's 858-line behaviour-cloned encoder**.
Caught by reading `git diff --stat` before committing (855 deletions on a file that should have been new),
restored with `git checkout`, and the loader moved to `zelda_rl/bc_demos.py`. The fleet was unaffected -- it
had rsynced the code at launch -- and the new module was pushed to all 13 pods so a pod restart cannot import
a file that is not there.

**The run (U until numbers land).** `gohma-v20-bc`, launch id `20261015173419`, repo `b47cbf4`:

| | v20 arms (concluded tie) | `gohma-v20-bc` |
|---|---|---|
| init | `gohma-kl-v15-r4_latest.pt` | `checkpoints/kept/bc_human.pt` |
| BC term | none | `--bc-data datasets/human/gohma --bc-coef 0.1` |
| observation | `--n-actors 4` (481) | 16 actors (1033), to match the recordings |
| ranks | 13 training | 12 training (1,2,3,4,5,7,8,11,12,13,14,19) + **r0 pure eval** |
| aggregator | rank 0 | rank 1 |
| unchanged | `--reward simple`, 120 s episodes, slots 3-8 at v17 weights, gamma 0.995, gae-lambda 0.95, lr 3e-4, hidden/gru 256, death-scale 0 | same |

`bc_coef 0.1` is the one free number: BC cross-entropy sits near 2.2 against a PPO policy loss of 0.01-0.1, so
1.0 would swamp the objective and freeze the policy at the clone. `bcL`/`bcA` lead the trainer's log line so
whether it bites is visible rather than assumed (AGENTS 34). The credit horizon was deliberately **not**
changed in the same run, so the BC effect stays readable.

The eval pod runs the pure-BC policy over all 10 starts (8 episodes each) and then `--watch-published
gohma-v20`, logging held-out metrics to W&B group `gohma-v20-bc-eval` beside the training curve.

**RESULT, arm 1 -- BC init is a worse starting point than v15, and this arm was stopped (M).** 12 ranks,
6 updates, 57,600 env steps per rank, fleet means:

| update | boss damage | death | success |
|---|---|---|---|
| 1 | 0.21 | 61.2% | 0.0% |
| 2 | 0.26 | 51.5% | 0.0% |
| 3 | 0.26 | 58.8% | 0.4% |
| 4 | 0.28 | 55.2% | 0.5% |
| 5 | 0.33 | 54.7% | 1.0% |
| 6 | **0.35** | 56.5% | 1.8% |

It **is** learning -- damage rising, deaths falling, first wins appearing -- but from 0.21 against v15's
**3.275**, and even from-scratch v10 *started* at 0.38. At ~+0.028/update it needs ~100 updates to reach where
v15 already is, so the compute was reallocated after 15 minutes rather than after a session. The clone's 53.4%
action accuracy therefore buys **nothing** as an initialisation for this task: claim R42 is answered NO on the
training-side evidence, pending the eval pod's held-out number.

**The pure-BC policy measured head-on: 0 defeats in 80 held-out episodes (M).** The eval pod played
`bc_human.pt` 8 episodes on each of the 10 starts, Null backend, 120 s cap:

| band | n | defeat | death | boss HP taken |
|---|---|---|---|---|
| ALL | 80 | **0.00%** | 83.8% | **0.14** / 10 |
| full_hp_10 (the task) | 24 | 0.00% | 70.8% | 0.12 |
| mid_hp_6_8 | 16 | 0.00% | 62.5% | 0.00 |
| half_hp_5 | 40 | 0.00% | 100.0% | 0.20 |
| near_lt200 | 24 | 0.00% | 100.0% | 0.46 |
| mid_200_400 | 24 | 0.00% | 75.0% | 0.00 |
| far_gt400 | 32 | 0.00% | 78.1% | 0.00 |

Per slot: 0.0% defeat on every one of the ten, death 100% on seven of them, mean episode length 41 s.

**This is the number that makes the point of the whole corpus exercise.** The clone predicts the human's next
action on 53.4% of held-out frames -- nearly 5x the majority floor -- and takes **1.4% of Gohma's HP**, against
v15's 3.275 (23x more) and from-scratch v10's 0.38 start. Imitation accuracy on single frames is not
competence in closed loop, and the gap is not small or arguable: it is 0 wins in 80 attempts. Claim R42 is
answered **NO** with an artefact (`runs/eval_bc_baseline.jsonl` on rank 0, W&B group `gohma-v20-bc-eval`).

Mechanically the reason is visible in the training logs of arm 2: v15 agrees with the human on **3%** of
frames at update 1. Two policies that both beat Gohma (one human, one v15) share almost no frame-level
behaviour, so action-matching is a weak proxy for the thing being learned.

**Arm 2 -- v15's weights with the BC loss kept on (U).** `gohma-v20-v15bc`, launch id `20261015175011`:
identical to arm 1 except `--init-from checkpoints/kept/gohma-kl-v15-r4_latest.pt`. v15 is an exact tensor
match for the corpus (obs 1033, 108 actions, 256/256), so the mismatch is behavioural, not structural -- the
BC term pulls the best learned policy toward a human strategy it did not discover. 12 ranks, plus 4 pods
created separately (`gohma-v15bc`, ranks 0-3) running the same config as a **second FedAvg group**: their
ranks would collide with the big fleet's inside one namespace, and two independent groups give two replicates
of the arm, which is what AGENTS 7 asks for. Fleet cost $8.40/hr.

The question this arm answers, and the signature to read: v15 starts at 3.275. If `dmgB` rises, demonstrations
help a competent policy; if it falls while `bcA` climbs, the term is overfitting 13 episodes and dragging the
policy off a better strategy, and `--bc-coef` comes down.

**RESULT, arm 2 at a CONSTANT coef 0.1 -- the regulariser works and costs damage (M).** 12 ranks over 8
updates, with a 4-rank replicate group:

| | update 1 | latest | slope +/- SE | t |
|---|---|---|---|---|
| 12-rank group, boss damage | 2.63 | 2.03 (u8) | **-0.0857 +/- 0.0172** | **-4.98** |
| 4-rank replicate, boss damage | 2.92 | 2.33 (u6) | -0.1340 +/- 0.0886 | -1.51 |
| rank 1 `bcA` (agreement with the human) | 3% | **23%** | rising | |
| rank 1 `bcL` (BC cross-entropy) | 4.43 | 3.00 | falling | |

Both groups decline and **every update of both sits below v15's 3.275**. The replicate is not individually
significant at n=6 but agrees in direction, which is what a replicate is for (AGENTS 7). The term is doing
exactly what it was built to do -- agreement up 8x, its own loss down -- and the headline it costs is damage.
Success rate did NOT fall with damage (12-rank 5.9% -> 5.6% with a 9.2% peak; replicate 7.8% -> 11.0%), but
AGENTS 34 says trust the graded signal over a 1-10% binary, so that is read as noise rather than a result.

**RESULT, arm 3 -- the decay does not buy the damage back, and the first comparison of it was wrong (M).**
26 updates on 12 ranks plus a 29-update 4-rank replicate:

| window | slope +/- SE | t | reading |
|---|---|---|---|
| matched u1-8 vs arm 2's constant 0.1 | **+0.0256 +/- 0.0396** | **+0.65** | NOT distinguishable from the constant coefficient |
| decay, all 26 updates | -0.0086 +/- 0.0049 | -1.74 | flat |
| decay, u14-26 (coefficient at its lowest) | -0.0239 +/- 0.0117 | **-2.05** | DECLINING |
| 4-rank replicate, all 29 updates | -0.0235 +/- 0.0070 | **-3.37** | DECLINING |

**0 of 26 updates reached v15's 3.275**, and the best single update was 2.74 at update 1 -- the point closest
to v15's own weights. Lowering the coefficient does not recover the damage.

**Retraction.** This comparison was first reported as `t=+1.92`, "just under the bar", and that number came
from an unmatched window: `fleet ppo` prints only the last 8 fleet-mean rows without `--last N`, so the decay
arm's updates 18-25 were compared against the constant arm's 1-8. The error moved the result toward the
hypothesis under test. Series are now pulled with `--last 60` and the update range is printed beside every
slope (AGENTS 12). The instrument gave no sign: both windows print identically.

### Event-stratified behaviour cloning: oversampling the decisions beats oversampling the frames (M)

Uniform BC spends gradient in proportion to how often something happens, which in this corpus means walking
(first damage at step 190 of 546; 2.99 seeds per win). `scripts/train_bc_human.py --event-weight` weights each
step by the DECISION it belongs to -- seven classes defined and justified in `docs/BC_EVENT_CLASSES.md`, each
marked from the recorded observations. Every arm below trained 60 epochs on the same 100-game corpus (v2,
digest `e57da6027a30`) and was then scored on the SAME 80-episode suite (8 per start, all 10 starts), with the
training and the eval **co-located on one pod** so a checkpoint is scored seconds after it is written.

| arm | scale | classes | val acc | online damage | defeats | death | timeout |
|---|---|---|---|---|---|---|---|
| `s1.5` | 1.5 | all | 72.5% | **0.16** | 0 | 90.0% | 10.0% |
| `s6` | 6.0 | all | 70.9% | **0.33** | 0 | 90.0% | 10.0% |
| `s12` | 12.0 | all | 71.8% | **0.53** | 0 | 85.0% | 15.0% |
| `nolarva` | 3.0 | no larva | 72.6% | **0.49** | 0 | 87.5% | 12.5% |
| `chainonly` | 3.0 | chain only | 73.1% | **0.34** | 1 | 86.2% | 12.5% |
| *13-game clone* | — | — | 53.4% | 0.14 | 0/80 | 83.8% | 16.2% |
| *100-game clone, unstratified* | — | — | 74.5% | 0.33 | 1/80 | 81.2% | 17.5% |
| *v15 (no human data at all)* | — | — | — | **3.275** | — | — | — |

**The scale axis is monotone at matched n: 1.5 -> 6 -> 12 gives 0.16 -> 0.33 -> 0.53.** The best stratified
clone takes **0.53** of her 10 HP, 1.6x the unstratified 100-game clone and 3.8x the original 13-game one.
This is the first thing in the project that has moved BC's online damage at all.

**Removing one class is worth quadrupling the scale.** `nolarva` at scale 3 reaches **0.49**, where the same
scale with larva included sits near 0.26-0.33. The larva class correlates **-0.598** with damage dealt across
the corpus, and the trained policy shows that correlation is causal in the training mix: capacity spent
learning to fight larvae is capacity not spent on the boss. Engaging the larvae is what a human does when the
fight is going badly, and imitating it uniformly teaches the failure.

**But the chain alone is too narrow.** `chainonly` (seed/stun/HP only, no sword, no larva) scores **0.34** --
clearly worse than `nolarva` at the same scale. The sword class, which is the largest at 3,975 events and is
defined as B pressed while she is stunned AND within 120 u, carries real signal. So the lesson is not "fewer
classes" but "drop the class the data says is counterproductive".

**Validation accuracy does not predict online damage, measured rather than asserted.** Across these five arms
the correlation between best validation accuracy and online damage is **r = -0.11**: `chainonly` has the
HIGHEST accuracy (73.1%) and middling damage, `s12` has nearly the lowest (71.8%) and the best damage. A
stratified arm stops optimising the uniform frame distribution by construction, so accuracy measures something
the arm is deliberately not trying to do. Any future BC comparison that cites accuracy is citing the wrong
number.

**What has NOT changed: 0 defeats in 400 episodes** across these arms, with the single exception of
`chainonly`'s one win (1.25%). Deaths are 85-90% everywhere. Damage is scaling; winning is not. At 0.53 the
best clone is **16% of v15's 3.275** and 5.3% of a kill.

**The extrapolation, and why its tail is not credible.** A log-log fit on the scale points gives
`damage ~ 2^(-3.20 + 0.55 log2 scale)`, which needs scale ~59 for 1.0 damage, ~519 to match v15 and **~4,022
for a kill**. The weight is `1 + scale x factor`, so by scale ~60 a positioning step already contributes
nothing relative to a decision step and every larger scale is the same experiment. The curve must saturate far
below those numbers; the expanded sweep (scales 12/24/48/1000 with larva removed, plus a wider span) exists to
find the knee, not to chase 4,022.

**Process failures tonight, each cheap and each repeated from a rule already written down:**

- the pods had the trainer from BEFORE `--event-classes` existed, so exactly the two arms needing that flag
  died on `unrecognized arguments` while the five that did not were unaffected. Pushing code is not the same
  as pushing the code you just changed (AGENTS 16).
- the first BC watch loop used the `aws` CLI, which the pods do not have (boto3 only), so it polled silently
  and did nothing. It was replaced by co-locating the eval with the trainer, which removes the loop entirely.
- the arm script piped training through `grep | tail -20`, which emits nothing until the process exits, so the
  `uniform` control looked hung for twelve minutes while it was training normally. **That is AGENTS 24, in a
  script written the same night it was cited.**
- `set -- $VAR` inside a zsh loop does not word-split, which silently passed empty arguments three separate
  times and launched two shards with no slots. Explicit commands, no reuse of that pattern.

### What is IN the 100-game corpus: the event chain, the slingshot, and the reward it earns (M)

Mined from the recorded observations by field name (`scripts/mine_corpus.py`, `scripts/mine_slingshot.py`),
so it is re-derivable from the dataset alone. Per-episode output in `results/corpus_events.json`.

**The damage chain, from 678 real hits across 100 wins:**

| condition within 2 s before her HP falls | share |
|---|---|
| she was **STUNNED** | **678/678 = 100.0%** |
| a seed impact had landed | 195/678 = 28.8% |
| the `gohma_vulnerable` flag was set | **9/678 = 1.3%** |

**A stun is the gate, absolutely, and `gohma_vulnerable` is the wrong signal to key a reward on** -- it is set
for 1.3% of real damage events. (`gohma_stunned` covers `fall`/`land`/`stunned`/`stunned_hit`, which is what
actually gates the sword.)

**What a human win looks like:** 6.78 hits, **3.13 stuns** (~2 hits per stun), 2.99 seeds fired, 1.71 nuts,
3.72 larva hits, 0.79 times hurt, first damage at step 190 (19 s -- a third of the episode is setup),
Z-targeted 63% of the time, slingshot held 49%, **shield 0%** (the human never blocks), mean range to her eye
400 u but **50 u at the moment damage lands**.

**Correlations with damage dealt** (per episode, n=100): slingshot held **+0.502**, Z-target +0.332,
vulnerable onsets +0.288, seeds fired +0.202, nuts +0.064, times hurt -0.235, **larva hits -0.598**, **time she
spends on the ceiling -0.559**. Two of these contradict the current reward: killing larvae is strongly
NEGATIVE for damage (a distraction, which vindicates zeroing those terms), and so is time she spends on the
ceiling (unhittable up there -- reward should pay for bringing her down, not for waiting).

**The slingshot, 299 fires / 260 landed = 87% hit rate.** By her state when fired: `ceiling_eye_red` **155
fires (52% of all shots), 92.9% land**; `ceiling_idle` 37 (89.2%); `ceiling_move` 34 (85.3%); `fall` 25
(100%); `ceiling_spawn` 19 (84.2%); **`ceiling_wait` 14 (7.1%)** -- the one state where shots are wasted.

Hit vs miss: aim alignment to her eye **0.607 vs 0.270** is the discriminator; **first-person aim 0.115 vs
0.359**, i.e. first-person aiming is associated with MISSING, so a reward paying for it would pay for the
failure mode; range 491 u vs 442 u, so LONGER is better (450 u+ lands 91.8%, 300-450 u 81.2%) because those
are ceiling shots; `ready_to_fire` is irrelevant (0.008 vs 0.026).

**What an impact buys, 371 impacts:** a stun follows 67.7% of the time and **0.70 of her 10 HP** comes off in
the next 3.5 s -- but entirely conditional on her state at impact: `fall`/`land`/`stunned` 100%,
`floor_prepare` 95.2%, `floor_eye_close` 88.9%, **`ceiling_eye_red` only 40.6%**, `floor_walk` 42.3%,
`ceiling_spawn` 26.7%, and **`ceiling_idle`/`ceiling_move` 0% across 71 impacts**. So 71 of 371 impacts were
wasted seeds, and even a red-eye hit converts only ~40% of the time: hitting the red eye is necessary, not
sufficient.

**Reward implications this supports:** pay for an impact CONDITIONED on her state (eye-red, prepare,
eye-close -- never idle/move), pay for the KNOCKDOWN rather than the impact, and pay for neither
first-person aim nor larva kills.

**The return the corpus earns, and why it is not a target (M).** Summing the recorded per-step reward:

| | human corpus (100 eps) | best PPO, decay r1 | best PPO, constant r1 |
|---|---|---|---|
| return | **+316.9** (246 to 372) | +35.7 | +39.4 |
| boss damage | **7.80** | 2.09 | 2.60 |
| return per HP | +40.6 | +17.0 | +15.1 |
| corr(return, damage) | **+0.719** | — | — |

The recorder used the FULL gohma profile and every PPO run uses `--reward simple` (39 terms zeroed), so the
absolute figures are on different functions. The structural finding survives that caveat: fitting the human's
return on damage gives **return ~= +253.1 + 8.19 x damage**, so **~80% of it is damage-independent**. Damage
5-7 earns +295 and a full 10-HP kill earns +335 -- a complete kill is worth 13% more return than a half-kill.
A reward that moves 13% between half-killing and killing her is not a reward for killing her (AGENTS 34). On
`GOHMA_SIMPLE` the same episodes would be worth about `4 x 7.8 + 60 = +91` plus the time bonus, i.e. ~2.5x the
best policy's +36 rather than 8x.

### Does MORE demonstration data win more? 100 games vs 13, measured online (M)

The corpus grew from 13 to **100 human wins** (54,626 steps, 23 pre-fight starts, 9 slots) and a clone was
trained on all of it -- 91 train / 9 validation episodes, one per slot, best-validation checkpoint at epoch 36
of 80. Both corpus versions are pinned in `datasets/human/gohma/MANIFEST.json` with per-episode md5, slot,
start state, start distance, outcome, `recorded_utc` and the recorder's git commit: **v1 = 13 episodes, digest
`5663643e97f889748c1941b56a6b6eb9`; v2 = 100 episodes, digest `e57da6027a3043d9cd7cdb80ab688c15`**, v1 a
prefix of v2 so v1's published numbers stay reproducible. Built by `scripts/dataset_manifest.py`.

Both clones were scored on the SAME suite: 80 episodes, 8 per start, all 10 starts, 120 s cap.

| | 13-game clone (v1) | 100-game clone (v2) |
|---|---|---|
| held-out action accuracy | 53.4% (floor 11.1%) | **74.5%** (floor 20.1%) |
| boss damage, all starts | 0.14 | **0.33** |
| defeats | **0** / 80 | **1** / 80 (1.25%) |
| deaths | 83.8% | 81.2% |
| **full-HP starts (the task)** | **0/24, dmg 0.12** | **0/24, dmg 0.12** |
| near starts (<200u) | 0.46 | **1.08** |
| mid 200-400u / far >400u | 0.00 / 0.00 | **0.00 / 0.00** |

**7.7x the data bought 2.4x the damage and the clone's first kill ever -- and changed nothing on the task.**
On full-HP starts the two clones are identical at 0.12 damage and 0 wins in 24. The single defeat came from
**slot 7: 37.9 units away, 5 HP left** -- finishing a nearly-dead boss it was already standing on. All of the
gain is in the near band (0.46 -> 1.08); mid and far range are **0.00 in both**.

**So dataset size is not the binding constraint, and this is the measurement that says so.** Imitation
accuracy scales with data (53.4% -> 74.5%) and barely converts into damage, because the clone is trained
**teacher-forced**: every step conditions on the human's real observation sequence, so it never learns to
recover from the states its own mistakes create. 81% of its episodes still end in death. The ceiling is the
method, not the corpus -- which is the argument for using the clone as a REPRESENTATION source under PPO
(the latent arms) rather than as a policy.

Artefact: `results/eval/bc_human_100_online.jsonl` (full per-slot and per-band breakdown), pulled off the pod
before it was terminated. Clone: `checkpoints/kept/bc_human_100.pt`, also in S3 via the automatic backup.

**An instrument fix found here:** the evaluator's progress line printed only a defeat counter, which reads 0
for fifteen minutes for any policy below ~10% win rate, so boss damage -- the one signal with resolution
(AGENTS 34) -- was visible only in the final summary. It now prints damage every minute, episode-weighted and
mix-invariant across slots.

### ARTEFACTS LOST: the whole fleet vanished at ~20:14 on 2026-10-15 (A)

`python -m fleet list` returned **$0.00/hr and zero pods**, and RunPod's REST API confirmed **0 pods** on the
account. All 34 pods -- 12 IBRL, 4 IBRL replicate, 12 `gohma-bc3` (FitNets / transition / mutual), 2
on-policy, 4 `gohma-v15bc`, and the eval pod -- were gone at once. The cause is not recoverable from here:
there were no pods left to query for a termination reason, and nothing in this session terminated them (the
only `fleet kill` was `gohma-bcw`, 4 pods, at the author's instruction at 19:14).

**What was lost, precisely:**

- **every arm's checkpoints.** `fleet sync` was never run on any of the 2026-10-15 BC/latent arms, so no
  weights from fifteen arms exist anywhere. The best policy of the day cannot be re-evaluated or resumed.
- **the eval pod's `runs/eval_watch.jsonl`** -- the per-episode rows behind six held-out evaluations.
- **every per-rank training log**, including the `bcPick` / `bcL` / `bcA` / `bcZ` columns at per-update
  resolution.

**What survived, and only because W&B was exported after the loss:** every arm's damage history. The export
itself nearly failed -- `scripts/export_wandb.py` scanned three projects and **`zelda-rl-goal-ppo` was not one
of them**, which is where `fleet launch` sends combat runs. Fixed in the same commit; had the loss happened
before that was noticed, the repo's own tooling could not have exported the day's work.

**THE FIX, because a rule that needs remembering is not a rule (AGENTS 21).** "Sync when you read a number"
would have been another sentence in a document. Instead the write itself now backs itself up:

- **`zelda_rl/ckpt_s3.py`** uploads every checkpoint to `s3://zelda-rl/fleet/ckpt_auto/<run>/` **at the moment
  `torch.save` writes it**, from inside `train_combat.py`. No flag, no operator step, nothing to forget. Pods
  already carry AWS credentials for this (`fleet/secrets.py` ships them so a pod can stream its dataset as it
  collects), so there was nothing to plumb. Failure is logged as `CHECKPOINT BACKUP FAILED ... this run's
  weights exist ONLY on this pod` and never raises -- a lost backup must not kill a run, and must not be
  silent. Verified end to end: a probe checkpoint appeared in S3 and was then removed.
- **`scripts/record_human.py`** writes the manifest row **and pushes the episode to S3 as each episode ends**
  (`record_row`), not at session end. Verified by simulating a killed session: three episodes recorded with no
  session end leave a complete manifest, every row with an md5. `--no-s3` opts out deliberately.

The same bug produced both losses: state that exists only until something is interrupted. Fifteen arms' weights
and 29 of the author's recorded wins were in that state at the same time.

| arm | W&B group | run id | ranks | history rows |
|---|---|---|---|---|
| FitNets projection | `gohma-bc3-fitnet` | `0elfmc4m` | 4 | 19 |
| latent transition | `gohma-bc3-latroll` -> delta | `w3wsholh` | 4 | 17 |
| one-way latent (GRU, demo) | `gohma-bc3-lat` | `vc48mqx8` | 4 | 23 |
| mutual alignment | `gohma-bc3-mutual` | `k3kooogu` | 4 | 28 |
| inverse (clone student) | `gohma-bc3-inv` | `72uww7w7` | 4 | 16 |
| Q-filter | `gohma-bc3-qf` | `vj96cuw5` | 4 | 16 |
| IBRL | `gohma-v20-ibrl` | `xb2e5oli` | 12 | 14 |
| on-policy distillation | `gohma-lbc-onpol` | `e6fssffk` | 2 | 13 |
| decay (12 ranks) | `gohma-v20-v15bcd` | `81mre82x` | 12 | 64 |
| decay replicate | `gohma-v15bc-v15bcd` | `un4fkisv` | 4 | 73 |
| **held-out eval** | `gohma-v20-v15bcd-eval` | `ebzcrdaz` | 1 | 6 |

### THE DAY'S RESULT, fifteen arms: nothing beats v15, and nothing rises (M)

| arm | mechanism | n | mean | max | slope +/- SE | t |
|---|---|---|---|---|---|---|
| FitNets projection | learned linear map before the MSE, rollout states | 14 | **2.87** | 3.37 | -0.0064 +/- 0.0192 | -0.33 |
| latent transition | MSE on `h_t - h_(t-1)` | 14 | **2.79** | 3.21 | +0.0100 +/- 0.0159 | +0.62 |
| mutual alignment | both nets trainable, contrastive | 22 | 2.66 | 3.28 | +0.0017 +/- 0.0132 | +0.13 |
| one-way latent (GRU, demo) | MSE to the frozen clone | 17 | 2.47 | 3.19 | -0.0220 +/- 0.0188 | -1.17 |
| decay 0.1->0.02 | action-level BC, decaying | 26 | 2.32 | 2.74 | -0.0086 +/- 0.0049 | -1.74 |
| on-policy distillation | KL to the clone on rollout states | 7 | 2.43 | 3.67 | **-0.2779 +/- 0.0930** | **-2.99** |
| decay replicate | as above | 29 | 2.08 | — | **-0.0235 +/- 0.0070** | **-3.37** |
| constant 0.1 | action-level BC, fixed | 8 | 2.31 | 2.64 | **-0.0857 +/- 0.0172** | **-4.98** |
| Q-filter | BC on the 12% of frames that damage her | 6 | ~2.0 | 2.42 | — | too few |
| IBRL | clone proposes, Q head picks | 9 | **0.38** | 0.51 | -0.0050 +/- 0.0105 | -0.48 |
| inverse (clone student) | v15's representation into the clone | 9 | 0.48 | 0.76 | — | flat at the floor |
| BC init | clone as the starting policy | 6 | 0.28 | 0.35 | +0.028/upd | rising from nothing |
| **v15 alone** | **no human data at all** | — | **3.275** | — | the bar | |

**The table above is the episode-weighted fleet mean from `fleet ppo`. Regenerated from the exported
histories on the MIX-INVARIANT column (`combat/boss_damage_balanced`, AGENTS 29) by `scripts/arm_trends.py`,
the ordering changes and one arm's series sits above 3.275 -- but every exported history is ONE RANK (the W&B
group's representative run), and per-rank damage on this task spans 1.36-3.37, so no per-arm mean from it is
a result (AGENTS 7). It also compares against a 3.275 that was never re-derived on the same column and
aggregation, which is the AGENTS 12 error. Both tables are kept because neither is a matched comparison on
its own, and saying which is which is the point:**

| arm | one rank, balanced mean | max | slope +/- SE | t |
|---|---|---|---|---|
| latent transition (`latroll-r4`) | **3.30** | 4.40 | -0.0120 +/- 0.0331 | -0.36 |
| FitNets (`fitnet-r0`) | 2.61 | 3.85 | +0.0149 +/- 0.0348 | +0.43 |
| mutual (`mutual-r8`) | 2.60 | 4.00 | +0.0028 +/- 0.0150 | +0.19 |
| constant 0.1 (`v15bc-r1`) | 2.60 | 3.52 | +0.0484 +/- 0.0590 | +0.82 |
| one-way latent (`lat-r8`) | 2.54 | 3.33 | -0.0011 +/- 0.0147 | -0.07 |
| Q-filter (`qf-r4`) | 2.18 | 2.95 | -0.0052 +/- 0.0242 | -0.22 |
| decay replicate (`v15bcd-r0`) | 2.13 | 3.83 | +0.0016 +/- 0.0030 | +0.53 |
| decay (`v15bcd-r1`) | 2.09 | 3.31 | -0.0086 +/- 0.0031 | **-2.75** |
| on-policy distil (`onpol-r1`) | 1.75 | 3.25 | -0.1617 +/- 0.0291 | **-5.56** |
| IBRL, broken (`ibrl-r1`) | 0.74 | 1.61 | +0.0543 +/- 0.0295 | +1.84 |
| inverse (`inv-r0`) | 0.49 | 0.96 | +0.0337 +/- 0.0102 | **+3.32 rising from the floor** |

**What the next session must do before any claim about the latent arms:** re-derive v15's number on
`combat/boss_damage_balanced`, and compare arm MEANS ACROSS RANKS rather than the representative run. The
exported histories cannot do the second -- only the representative run's history is kept -- so an arm
comparison needs the fleet running, or the exporter extended to pull every rank's history.

**Three arms produced one update above 3.275 each** (FitNets 3.37, mutual 3.28, on-policy 3.67) and **no arm
produced a trend above it**. The two arms that merely HELD at v15's level are the two whose design follows the
measurement that the divergence between v15 and the clone is accumulated memory rather than perception
(encoder gap 0.497, GRU gap 0.938). Both had 4 ranks and ~14 updates when the fleet died.

**IBRL's number is not a result.** `bcPick` 54-78% with `q_loss` 2.5-10: the Q head was untrained when it
began gating, so the policy executed the clone's actions on most steps and fell from 2.3 to 0.38 in nine
updates. That is an implementation error of this repo's on-policy adaptation, not a property of the method
(claim R49).

**Arm 3 as launched (config).** `gohma-v20-v15bcd` launch `20261015180929` (12 ranks)
and `gohma-v15bc-v15bcd` launch `20261015181013` (4 ranks): coef **0.1 -> 0.02 over 50 updates, then held
at 0.02 forever**. Decaying to zero was rejected on the author's instruction and is asserted impossible in
`tests/test_bc_regulariser.py`: a coefficient that reaches 0 leaves `--init-from`-only behaviour, where the
demonstrations shape a few early updates and nothing after. 0.02 against the measured `bcL` ~3.0 contributes
~0.06 to the loss, the same order as PPO's own policy loss (0.01-0.1), so the tail keeps real influence.
`bc_demos.coef_at()` is the schedule and `bcC` in the log line is the value actually applied.

---

## 4. WM track

**Result so far: negative, and the negative is the contribution.** This track is older than the RL track and
ran in four distinct phases, each with its own architecture family, its own dataset scale and its own reason for
being abandoned. The phases are what a paper needs; the individual arms are in the docs.

### 4.0 Phase table

| Phase | Dates | Architecture family | Dataset | Verdict | Doc |
|---|---|---|---|---|---|
| **P1 latent** | to 2026-09-19 | frozen/joint BC encoder + latent transition (6 configs, latent swept 32→512) | 203,424 transitions, 398 episodes, four v3 collections | every config **loses to the identity**, best −1.14 | `docs/LATENT_WORLD_MODEL.md` (§1–30, 1,554 lines) |
| **P2 causal / OOCDM** | 2026-09-19/20 | object-oriented causal dynamics (Yu et al., ICML 2024) | same 203k | **the paper's own method was never run**; what trained was its baseline | `docs/CAUSAL_WORLD_MODEL.md`, `docs/CAUSAL_DEBRIF.md` |
| **P3 token, fleet scale** | 2026-09-22 | entity tokens + attention, state in / state out | **20,545,657** transitions, 39 pods, FedAvg on 32 shards | **val loss fell 1.8× while rollout skill collapsed +0.0026 → −0.41** | `docs/TOKEN_WORLD_MODEL.md` |
| **P4 token + JEPA, single H100** | 2026-09-22/23 | same tokens, 3 entity classes, 8-frame history, dual frames; and a latent JEPA beside it | **91,522,473** transitions (packed v5, resident in 80 GB VRAM) | positive to H5, **−2.44 at H1800** | `docs/WM_EXPERIMENT_REGIME.md` |
| **P5 motion post-mortem** | 2026-09-26/27 | diagnosis, not a new model | 1.5–2 M subsets + 150 counterfactual pairs | motion is **not action-conditioned**; 43% of states cannot be | `docs/WM_MOTION.md` |

### 4.1 P1 — the six latent configurations that lose to the identity (M)

Skill is `1 − model_error / identity_error`; the identity predicts no change. **Negative means worse than
predicting nothing.** `scripts/train_joint.py`, `docs/LATENT_WORLD_MODEL.md` §2:

| configuration | skill |
|---|---|
| frozen BC latent, MSE, no residual, stride 5 | −7.24 |
| frozen BC latent, MSE, residual, stride 1 | −1.73 |
| unfrozen, EMA target + stop-gradient, L1 on normalised targets, 2-step rollout | −1.78 |
| joint decode+predict, no BC, lr 1e-4 | −1.37 |
| joint, with exit data (`joint_v3`) | −1.14 |
| three further architecture changes (§29) | all negative |

Swept and recorded: latent width 32→512 (decode saturates at 128, predictability degrades monotonically with
width), BC training length 2→300 epochs. **A**: the selection metric went degenerate — a zero-init residual
predictor *is* the identity, so skill = 0 at epoch 0 beat every learned value (§462).

**The most expensive single lesson on this track (M, `docs/LATENT_WORLD_MODEL.md` §1, and AGENTS.md §1):** a
competent PPO policy's `proj` latent scored skill **+0.742** as a predictive surface and was nearly adopted as
the encoder. Controls destroyed it: random untrained weights of the same architecture **+0.617**, minus the
control echo **+0.240**, minus velocity **+0.053**, minus `time_left` and the goal block **−0.012**. The whole
number was artefacts.

### 4.2 P2 — the causal phase, and why it is not evidence for OOCDM (M/A)

Stage 0 (schema + census) and Stage 1 (per-field model at the full graph) were built as planned. Then six
unrequested experiments added a change gate, per-field `pos_weight` oversampling, binned direction+magnitude
heads and firing-threshold calibration — **none of which appear in the paper**, and all four of which came from
the *failed* latent line OOCDM was adopted to escape. Meanwhile the paper's actual contribution never ran: no
masking, no CMI, no causal discovery, no `Ĝ` vs `G₁` vs empty-graph comparison, no shuffled-class control.

**So P2 is evidence about OOCDM's baseline, not about OOCDM.** It is the origin of `AGENTS.md` §13. Recorded in
full in `docs/CAUSAL_DEBRIF.md`; deviations are listed in the header of `zelda_rl/oocdm.py`.

### 4.3 P3 — fleet scale, and the anti-correlation result (M)

20,545,657 transitions collected on 39 RunPod pods with the `badplay` mix (61% success, 5% death, 5% room-exit,
28% timeout), then data-parallel FedAvg on the 32 pods that had a shard, `lr=1e-5`, batch 512, averaged every
epoch.

| Group | What | Exported |
|---|---|---|
| `v4-collect-wm` | the collection + first WM training | `results/ut6gdkgb/` (57 ranks) |
| `v4-collect-wm100` | **the 100-epoch run: val loss fell 1.8× while rollout skill went +0.0026 → −0.41** | `results/c3zurq8g/` (32 ranks) |

That single pair of curves is the strongest evidence on this track for **W2 (loss is not the metric)** and is
why `AGENTS.md` §8 exists. Local copies of the metric histories: `runs/wm5_synced/`, `runs/wm100_synced/`.

### 4.4 P4 — 91.5 M transitions on one H100 (M)

Dataset packed once to 579 B/frame (49.4 GiB, 91,626,189 frames) and held **resident in VRAM** — 49.7 GiB of
79.2, peak 52.0 with training. Packed vs in-memory paths verified **bit-identical**. LR 3e-5 chosen by a sweep
in which loss decreased monotonically across the whole range while rollout skill peaked in the middle — the
cleanest loss/task anti-correlation in the project.

| Group | Runs | Exported |
|---|---|---|
| `v5-h100` | `wm_v2`/`wm_v3` token WM and `jepa_v2`/`jepa_v3` | `results/gymco2ix/` |
| (ungrouped) | the LR sweep and one-off arms | `results/di0qmlz3/` |

**The two-stage regime is the finding:** teacher-forced pretrain, then an autoregressive fine-tune over a fixed
depth mixture (60% of batches at H=1 but only ~3% of the compute; 0.15% at H=1800 is ~15% of it), step decay as
a half-life, drift-correcting targets, truncated BPTT with backward per window.

| H | stage 1 | stage 2 mean (28 epochs) | epochs negative |
|---|---|---|---|
| 1 | +0.370 | **+0.333** | 0/28 |
| 2 | **−2.63** | **+0.251** | 0/28 |
| 5 | **−2.65** | **+0.091** | 0/28 |
| 10 | −1.29 | −0.002 | 16/28 |
| 20 | −0.83 | −0.055 | 25/28 |
| 100 / 400 / 1800 | — | **−0.289 / −1.753 / −2.442** (one eval, ep 60) | 1/1 |

**Read the mean column, not the best.** A mid-run report of epoch 56 as "all positive through H20" was wrong:
H20 was positive in 3 epochs of 28.

Architecture, shared by both models via `zelda_rl/wm_context.py` so they cannot drift: three entity classes
(Link / actor / **static environment**, the last keys-values-only — 49 env tokens against ≤10 entity tokens),
8-frame history summarised locally per entity into one token before global attention, dual rotation frames
(camera for movement, Link-facing for combat), position only as nav-grid cell **embeddings**, HP classified and
input-binned, zero-init heads + logit residuals so skill starts at exactly 0.000.

**Utilisation, corrected (A):** I claimed autoregressive rollout caps GPU at ~27%. True of the thin model; with
3 blocks, 8-frame history and 49 env keys the token WM held **96%** for a whole run. The JEPA sat at **33%** and
that gap was never explained.

### 4.5 P5 — the motion post-mortem (M), and the instrument errors it exposed

| Finding | Number | Class |
|---|---|---|
| Position is **not action-conditioned** at one step | action ablation +0.002 (+0.4143 vs +0.4127 zeroed) | M |
| Action/displacement cosine by lag | 0.006 (lag 1) → 0.135 (lag 5); 0.26 conditioned on a clean move | M |
| **43% of states are action-INSENSITIVE** | 150 counterfactual pairs; the other 57% separate by a mean 21.4 world units | M |
| A 1.35 M-param MLP matches both token models at one step | +0.393 vs JEPA +0.379 / token +0.188 | M |
| **L1 vs MSE**, nothing else changed | aggregate skill **−0.379 → +0.392** | M |
| Change gate on sparse regression heads | `a_yaw` −3.71 → **+0.00**, 3 others likewise, at full scale | M |
| Motion loss ×10 (MOBA's ego-motion fix) | `l_doff` **−0.193 → −0.079** — the only thing that has moved motion | U (3 epochs, 1 seed) |

**Three conclusions on this track were artefacts of the instrument, not the model, and all three are recorded
with their retraction in `docs/WM_MOTION.md`:**

1. Scoring a change gate on **position skill** when the gate deliberately excludes position.
2. Measuring the nav grid on a **re-fit** rather than the grid stored in the checkpoint — published 142×66, the
   real cell is **28.8 × 57.1**, and the original estimate had been right.
3. A "gate below 4% change rate" threshold rule derived from arms using the **wrong gate normalisation**
   (changed-only instead of MOBA's `--gate-value-norm all`). Withdrawn.

### 4.6 WM checkpoints

| File | md5 | Bytes | What | Class |
|---|---|---|---|---|
| `runs/wm_v2/ck_0080.pt` | `16419e8d337d` | 84,319,348 | token WM, teacher-forced. **Best 1-step skill +0.188** — prefer over v3 | M |
| `runs/wm_v3/ck_0085.pt` | `5ef096a872e3` | 84,364,980 | token WM after AR fine-tune; 1-step skill went **negative** (~−0.13) | M |
| `runs/jepa_v2/ck_0045.pt` | `cf2373f7535d` | 137,650,436 | JEPA stage 1, the resume point for any stage-2 rerun | M |
| `runs/jepa_v3/ck_0070.pt` | `502308296220` | 138,565,124 | JEPA stage 2. H1 +0.333 mean, H1800 −2.44 | M |

Every WM checkpoint uses nav grid n=32 with lo `[-389.96, -911.47]`, hi `[531.36, 917.12]` → a **28.8 × 57.1**
world-unit cell, stored in `ck["grid"]`. **Read it; do not re-fit** (see 4.5).

**Inspecting any of them:** `scripts/wm_replay.py --ckpt <ck> --html out.html --open` rolls the model
autoregressively over a real episode and writes a self-contained viewer (`docs/WM_VIEWER.md`). This is the only
instrument that shows the failure directly — a dreamed Link 429 world units adrift.

## 5. Open items — the U list

Close these before any of them enters a paper.

| Item | Why it is not M | Track |
|---|---|---|
| The five v11 architecture changes were deployed **as one set** | no attribution; `scripts/ablate_classes.py` has never been run on these checkpoints | WM |
| Grid-96 result | +0.068 at 60 k frames / dim 128, **−0.012 at 2 M / dim 384**; one seed each, and every arm but one degraded by the last epoch | WM |
| `--walk-weight 10` (motion −0.193 → −0.079) | 3 epochs, one seed, one data subset | WM |
| `--act-in-token` | appeared to cancel the walk-weight gain; one seed, 3 epochs | WM |
| The "gate below 4%" threshold rule | **withdrawn** — derived from arms with the wrong gate normalisation | WM |
| Spacing term (`g_spacing`) | `close_attack_steps` never moved off ~21% against the human win's 9.3%; the v11 death-rate fall is **not** attributable to it | RL |
| v13's distance wall | slot 8 (stunned, 448 units) scored 0/225 at 20% death; one run | RL |
| Per-slot Deku Baba result | one episode per slot, watched | RL |
| Wall map | 209 of 1,024 cells, only where Link bumped into something | both |
| H=1800 rollout numbers | ~31 k of 52 M starts qualify, and they are all timeout episodes by construction | WM |
| `puffertank-moba-wm` pod | terminated by me 2026-09-27 during a fleet teardown; **not part of this project** | — |

---

## 6. How to regenerate this file's inputs

```sh
.venv/bin/python scripts/export_wandb.py --history      # results/<run_id>/ for all 23 groups
.venv/bin/python scripts/make_samples.py               # samples/ excerpts + artifacts/MANIFEST.json
```
