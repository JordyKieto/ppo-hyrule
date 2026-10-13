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
