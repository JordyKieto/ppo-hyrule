# zelda-RL

PPO agents that play *The Legend of Zelda: Ocarina of Time* (Wii VC) in Dolphin, observing **live game RAM** (zero-copy, µs per step) instead of pixels. Develop and evaluate on a Mac; train on a RunPod CPU fleet coordinated over Upstash.

**Current milestone (2026-09-13): the Deku Tree Deku Baba fight is solved.**
- The final GRU policy (global update 150, 41.8 M env steps on 20 pods) clears the room from **10/10** start savestates locally.
- It reaches **77%** success over the fleet's sampled training episodes.
- See [docs/COMBAT_TASK.md](docs/COMBAT_TASK.md).

**Next:** Gohma (boss).

## Artifacts

| What | Where |
|---|---|
| Start savestates + manifest | [`datasets/zelda/combat/deku-baba/`](datasets/zelda/combat/deku-baba/README.md) → `s3://zelda-rl/zelda/combat/deku-baba/states/` |
| Checkpoints | `s3://zelda-rl/zelda/combat/deku-baba/checkpoints/` (final: `G150_deku-combat-gru-v2b_u00120.pt`) |
| Training curves | wandb [`drift-ai-models/zelda-rl-combat-ppo`](https://wandb.ai/drift-ai-models/zelda-rl-combat-ppo): groups `deku-combat-gru-v2` (G1–30), `deku-combat-gru-v2b` (G31–154) |

The bucket is private (aws profile `race-wm`). Game data (savestates, NAND, ROM) is never committed to git.

## Layout

| Path | What |
|---|---|
| `zelda_rl/` | Dolphin control (pipes, hotkeys), live RAM attach (macOS mach / Linux shm), observations, combat task + reward tracker, PPO (entity + GRU) |
| `scripts/` | `train_combat.py`, `train_ppo.py` (goal/rupees), `play.py` (watch a checkpoint), `fleet_watch.py`, probes and benchmarks |
| `fleet/` | `python -m fleet`: RunPod create/bootstrap/launch/status/kill, Upstash weight averaging |
| `datasets/` | savestate manifests (files in S3) |
| `docs/` | [FINDINGS](docs/FINDINGS.md) · [RAM_MAP](docs/RAM_MAP.md) · [OBSERVATION_SCHEMA](docs/OBSERVATION_SCHEMA.md) · [ADDING_OBSERVATIONS](docs/ADDING_OBSERVATIONS.md) · [COMBAT_TASK](docs/COMBAT_TASK.md) · [RUNPOD](docs/RUNPOD.md) |

## Quick start (Mac)

```sh
scripts/setup_macos_memaccess.sh          # one-time: Developer Mode, signed Dolphin copy + python-dbg (docs/FINDINGS.md §3.1)
aws s3 sync s3://zelda-rl/zelda/combat/deku-baba/states/ "$HOME/Library/Application Support/Dolphin/StateSaves/" --profile race-wm
aws s3 cp s3://zelda-rl/zelda/combat/deku-baba/checkpoints/G150_deku-combat-gru-v2b_u00120.pt checkpoints/kept/ --profile race-wm
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/play.py checkpoints/kept/G150_deku-combat-gru-v2b_u00120.pt --slot 1 --slot 7
```

## Record a video (fullscreen + game audio)

```sh
scripts/record_play.sh                      # final policy through slots 1-10 -> recordings/deku-baba_<time>.mp4
```

- **How:** a ScreenCaptureKit recorder (`tools/screenrec.swift`) captures only Dolphin's window and Dolphin's audio, with no loopback driver.
- **Stops by itself** when the last slot ends; Ctrl+C also stops and still saves the file.
- **First run:** allow Screen Recording for your terminal app, restart it, and run again.

