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

**Result so far: negative, and the negative is the contribution.** Six latent architectures lose to the
identity (`docs/LATENT_WORLD_MODEL.md`); the token model is positive to about 5 steps and worse than
doing nothing beyond ~100 (`docs/WM_EXPERIMENT_REGIME.md`).

| Group | What | Exported | Class |
|---|---|---|---|
| `v4-collect-wm` | 20.5 M-transition fleet collection + FedAvg WM training | `results/ut6gdkgb/` (57 ranks) | M |
| `v4-collect-wm100` | the 100-epoch run where **val loss fell 1.8× while rollout skill collapsed +0.0026 → −0.41** | `results/c3zurq8g/` (32 ranks) | M |
| `v5-h100` | the 91.5 M single-H100 runs: `wm_v2`/`wm_v3` token WM and `jepa_v2`/`jepa_v3` | `results/gymco2ix/` | M |
| (ungrouped) | LR sweep and one-off arms | `results/di0qmlz3/` | M |

**Checkpoints:**

| File | md5 | Bytes | What | Class |
|---|---|---|---|---|
| `runs/wm_v2/ck_0080.pt` | `16419e8d337d` | 84,319,348 | token WM, teacher-forced. **Best 1-step skill +0.188** | M |
| `runs/wm_v3/ck_0085.pt` | `5ef096a872e3` | 84,364,980 | token WM after autoregressive fine-tune. 1-step skill went **negative** | M |
| `runs/jepa_v2/ck_0045.pt` | `cf2373f7535d` | 137,650,436 | latent JEPA, teacher-forced stage 1 | M |
| `runs/jepa_v3/ck_0070.pt` | `502308296220` | 138,565,124 | JEPA stage 2, AR. H1 +0.333 mean, H1800 −2.44 | M |

Grid: every WM checkpoint uses n=32 with lo `[-389.96, -911.47]`, hi `[531.36, 917.12]` → a **28.8 × 57.1**
world-unit cell (stored in `ck["grid"]`; read it, do not re-fit — see the retraction in `docs/WM_MOTION.md`).

---

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
