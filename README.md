# zelda-RL

RL agents that play *The Legend of Zelda: Ocarina of Time* (Wii VC) in Dolphin, observing **live game RAM** (zero-copy, ~25 µs per step) instead of pixels. Develop and evaluate on a Mac; train on a RunPod CPU fleet coordinated over Upstash.

[**Paper draft**](paper/paper.md) · [**Claim audit**](paper/CLAIMS.md) · [**Dataset card**](datasets/transitions/deku-baba/README.md) · [**Figures**](results/)

## Research direction

Learn a **transition model over structured state**, optimise policies entirely inside it, and transfer them to the emulator — which is used only as an authoritative judge, never for policy gradients. The organising idea is that the right thing to predict is not raw next state but state compressed through a representation **already shaped by competent behaviour**: freeze a behaviour-cloning encoder `E`, then learn `ẑ_{t+1} = M(E(s_t), a_t)` against `E(s_{t+1})`, with auxiliary heads keeping the quantities optimisation needs. The encoder discards what competent play never depended on, so the dynamics model is never charged with reconstructing the game.

The claim is stated as a comparison, not a preference: the same dynamics model is trained over a behaviour-shaped latent, a reconstruction-shaped one of equal width, a random projection, and raw state. See [paper/paper.md](paper/paper.md) §7. The encoder half is **built and measured** (§6); the dynamics model is not — read [paper/CLAIMS.md](paper/CLAIMS.md) before treating anything as a result.

## Status

| Piece | State |
|---|---|
| Deku Baba combat, model-free PPO | **Solved.** G150 clears **10/10** start states locally, 0 deaths; 77% fleet training success; 40.7 M env steps ([docs/COMBAT_TASK.md](docs/COMBAT_TASK.md)) |
| Transition dataset | **Released.** 109,817 transitions, 168 episodes, six competence tiers ([dataset card](datasets/transitions/deku-baba/README.md)) |
| Frozen BC encoder + state decoder | **Done.** 128-d latent reconstructs the reward function's inputs on held-out slots (Baba behaviour state 0.999, row HP 0.908, lock-on 1.000); reward events recovered at AP 0.89 ([paper §6](paper/paper.md)) |
| Latent dynamics model | Next ([protocol fixed](paper/paper.md) §7) |
| Dream-trained policy | Not started |
| Queen Gohma (boss), model-free | Task, reward, obs v7 and slingshot/nut actions built and validated on recorded human fights; not solved ([docs/GOHMA_TASK.md](docs/GOHMA_TASK.md)) |

## Artifacts

| What | Where |
|---|---|
| Start savestates + manifest | [`datasets/zelda/combat/deku-baba/`](datasets/zelda/combat/deku-baba/README.md) → `s3://zelda-rl/zelda/combat/deku-baba/states/` |
| Checkpoints | `s3://zelda-rl/zelda/combat/deku-baba/checkpoints/` (final: `G150_deku-combat-gru-v2b_u00120.pt`) |
| Training curves | wandb [`drift-ai-models/zelda-rl-combat-ppo`](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo): groups `deku-combat-gru-v2` (G1–30), `deku-combat-gru-v2b` (G31–154) |
| **Transition dataset** | [`datasets/transitions/deku-baba/20260918-134229/`](datasets/transitions/deku-baba/README.md) — 109,817 transitions, 168 episodes, 17 MB, committed · mirror `s3://zelda-rl/zelda/combat/deku-baba/transitions/20260918-134229/` |
| **Frozen BC encoder** | `s3://zelda-rl/zelda/combat/deku-baba/encoders/bc-decode-final.pt` — 128-d latent + state decoder ([paper §6](paper/paper.md)) |

The bucket is private (aws profile `race-wm`). Game data (savestates, NAND, ROM) is never committed to git.

## Layout

| Path | What |
|---|---|
| `zelda_rl/` | Dolphin control (pipes, hotkeys), live RAM attach (macOS mach / Linux shm), observations, combat task + reward tracker, PPO (entity + GRU) |
| `scripts/` | `train_combat.py`, `train_ppo.py` (goal/rupees), `collect_transitions.py` (dataset), `train_bc_decode.py` (frozen encoder + state decoder), `eval_decode_events.py` (event detection), `eval_bc.py` (BC in the emulator), `play.py`, `fleet_watch.py`, probes |
| `fleet/` | `python -m fleet`: RunPod create/bootstrap/launch/status/kill, Upstash weight averaging |
| `paper/` | [paper.md](paper/paper.md) the draft · [CLAIMS.md](paper/CLAIMS.md) the per-claim evidence audit |
| `results/` | `make_figures.py` regenerates every paper table and figure from committed artifacts; `figures/`, `tables/` |
| `datasets/` | savestate manifests (files in S3) and transition datasets with cards |
| `docs/` | [FINDINGS](docs/FINDINGS.md) · [RAM_MAP](docs/RAM_MAP.md) · [OBSERVATION_SCHEMA](docs/OBSERVATION_SCHEMA.md) · [ADDING_OBSERVATIONS](docs/ADDING_OBSERVATIONS.md) · [COMBAT_TASK](docs/COMBAT_TASK.md) · [RUNPOD](docs/RUNPOD.md) |

## Quick start (Mac)

```sh
scripts/setup_macos_memaccess.sh          # one-time: Developer Mode, signed Dolphin copy + python-dbg (docs/FINDINGS.md §3.1)
# per-task start states go in their own dir, NOT the live Dolphin slots (see "Savestate slots" below)
aws s3 sync s3://zelda-rl/zelda/combat/deku-baba/states/ datasets/zelda/combat/deku-baba/states/ --profile race-wm
aws s3 cp s3://zelda-rl/zelda/combat/deku-baba/checkpoints/G150_deku-combat-gru-v2b_u00120.pt checkpoints/kept/ --profile race-wm
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/play.py checkpoints/kept/G150_deku-combat-gru-v2b_u00120.pt \
    --slots-dir datasets/zelda/combat/deku-baba/states --slot 1 --slot 7
```

### Savestate slots

`~/Library/Application Support/Dolphin/StateSaves/NACE01.s0{1..10}` is the **live** slot set the user plays with, and it gets overwritten whenever a new task's states are made — it currently holds **Queen Gohma** states, not Deku Baba. Always pass `--slots-dir` pointing at the per-task directory.

This matters more than it sounds. Pointing a Deku Baba policy at the Gohma slots does not crash: it loads, runs, and reports every episode as a zero-target timeout, which looks exactly like a weak policy. One dataset collection was silently ruined this way. `scripts/collect_transitions.py` now refuses to start unless each start state contains a live actor of the expected id.

## Collect a transition dataset

```sh
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/collect_transitions.py --transitions 100000 --workers 10
.venv/bin/python results/make_figures.py      # regenerate every paper table and figure
```

Replays checkpoints from six points on the training curve so the data spans competence tiers rather than expert play only. ~104 steps/s across 10 headless Dolphin instances; 110 k transitions in ~18 min on an 8-core laptop.

## Record a video (fullscreen + game audio)

```sh
scripts/record_play.sh                      # final policy through slots 1-10 -> recordings/deku-baba_<time>.mp4
```

- **How:** a ScreenCaptureKit recorder (`tools/screenrec.swift`) captures only Dolphin's window and Dolphin's audio, with no loopback driver.
- **Stops by itself** when the last slot ends; Ctrl+C also stops and still saves the file.
- **First run:** allow Screen Recording for your terminal app, restart it, and run again.

