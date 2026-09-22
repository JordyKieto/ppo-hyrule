# Repository map — read this before searching for anything

There are 15 documents and ~5,600 lines of notes in `docs/`. Without a map an agent greps for words it already
suspects, finds a plausible file, reads the top of it and proceeds — which is exactly how a documented 3.5-minute
fleet fan-out became a one-hour outage (see `AGENTS.md` §14). This file is the index. Keep it current.

## "I am about to ..." → read this first

| Task | Read | Why |
|---|---|---|
| add a RAM field / observation | `ADDING_OBSERVATIONS.md`, `RAM_MAP.md` | offsets come from the OoT decomp, never recall; retail Player = decomp − 0x10 |
| anything with the world model | `LATENT_WORLD_MODEL.md` (§1–§30), `JEPA_PLAN.md` | 30 sections of what was already tried and what it measured. Most ideas here have been tried. |
| run anything on RunPod | `RUNPOD.md` **all of it**, then `NEXT_SESSION_RUNPOD.md` | the fast paths and the measured timings are in the tables, not the top-of-file examples |
| collect a dataset | `COMBAT_TASK.md`, then `scripts/collect_transitions.py` docstring | mixes, slots, savestate provenance |
| change the reward | `COMBAT_TASK.md`, `zelda_rl/combat.py::CombatTracker.step` | 18 terms; the dream calls the same tracker, so parity is structural |
| touch the encoder/decoder | `zelda_rl/bc.py` docstring, `LATENT_WORLD_MODEL.md` §26–§30 | control echo, unified decoder, reward-only spec |
| a causal / object-centric model | `CAUSAL_WORLD_MODEL.md`, `CAUSAL_DEBRIF.md` | 1221 lines incl. an OOCDM attempt and why it went wrong |
| report results | `FINDINGS.md`, `FIELD_SCORECARD.md`, `BENCHMARK.md` | the numbers that are already established |
| the Gohma task | `GOHMA_TASK.md` | separate task, separate savestates |

## Code layout

| Path | What |
|---|---|
| `zelda_rl/env.py`, `dolphin.py`, `live.py` | Dolphin process, RAM attach (macOS mach / Linux `linuxmem`), env step |
| `zelda_rl/combat.py` | `CombatState`, `CombatTracker` (the reward), RAM field offsets, Baba states |
| `zelda_rl/state_layout.py` | the ONE shared state layout; collector and model both import it so they cannot drift |
| `zelda_rl/bc.py` | featuriser, `BCNet` encoder, `UnifiedDecoder`, `REWARD_GLOBAL/REWARD_ROW` |
| `zelda_rl/dynamics.py` | every predictor (`Residual`, `Factored`, `FiLM`, `Categorical`) + `TransitionHead` |
| `zelda_rl/reward_decode.py` | reward-field targets + loss, shared by every stage so they cannot drift |
| `zelda_rl/navgrid.py` | deterministic position ↔ grid-cell + offset codec, and the static wall map |
| `zelda_rl/tokens.py` | entity tokens: typed, egocentric, no raw coordinates, no action echo |
| `zelda_rl/dream_state.py` | the dream env: `apply_transition`, `DreamTracker` |
| `zelda_rl/ppo.py` | `EntityActorCritic`, `RecurrentEntityActorCritic`, PPO update |
| `fleet/cli.py` | RunPod fleet: create / bootstrap / assets / verify / launch / collected / sync / kill |

## Scripts that matter

| Script | What |
|---|---|
| `collect_transitions.py` | dataset collection; `--mix-preset badplay` for deaths/damage |
| `collect_scenarios.py` | scripted walkouts, suicides, wall collisions, flails |
| `train_wm_tokens.py` | current world model: entity tokens + nav grid, next-state deltas |
| `train_joint.py` | the older joint encoder+decoder+predictor stack |
| `train_dream_ppo.py` | PPO inside the dream |
| `eval_rollout_space.py` | **the barometer**: ground-truth rollout in world units vs a do-nothing control |
| `eval_event_windows.py`, `eval_dream_rollout.py` | event-window and long-horizon evaluation |

## Non-obvious environment facts that have each cost a session

- **Live RAM attach needs the signed interpreter on macOS**: `.venv/bin/python-dbg`, not `.venv/bin/python`.
  Plain `python` fails with `N64 RAM not found in Dolphin process`. On Linux pods it is `linuxmem` + `/dev/shm`
  and needs nothing special.
- **Deku Baba savestates are NOT the default slots.** Local: `datasets/zelda/combat/deku-baba/states`.
  In S3: `zelda/combat/deku-baba/states`. `fleet`'s default `S3_STATES_PREFIX` is the **gohma** task — using it
  for a Baba run silently gives 10 slots with zero live Babas.
- **A collector is not a trainer.** `fleet launch` passes `--run-name/--num-envs/--fleet-*`; use `--collect`.
- **`DOLPHIN_BIN`** must be exported: `~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin` locally.
- **A dead collector leaves its Dolphins running.** `pkill -f Dolphin` after any failed launch, and check
  `pgrep -f Dolphin` before blaming performance on anything else.
- **The dataset tree is ~52 GB.** Never rsync the repo to a pod without excluding `datasets/` and `runs/`.

## Fleet quick reference

```sh
python -m fleet create    --experiment E --count N --vcpu 16 --confirm
python -m fleet bootstrap --experiment E                       # seeds first, no --via-seeds
python -m fleet bootstrap --experiment E --via-seeds 0,2 --ranks <rest>   # ALL of them at once: ~3.5 min/19 pods
python -m fleet assets    --experiment E --states-only --states-prefix zelda/combat/deku-baba/states
python -m fleet verify    --experiment E --instances 12
python -m fleet launch    --experiment E --collect --script collect_transitions.py --train-args "..."
python -m fleet collected --experiment E [--verbose]           # total transitions + outcome mix
python -m fleet sync      --experiment E --datasets <remote dir>
python -m fleet kill      --experiment E --confirm              # pods bill by the hour
```

Do **not** throttle the fan-out into small sequential batches when it reports transient SSH errors: it retries,
and serialising it turns 3.5 minutes into an hour.

## Fleet-scale WM training (current)

Data-parallel FedAvg: each pod trains on ITS OWN collected shard and weights average every epoch, so the fleet
sees the whole dataset with no data movement. `FleetLink.average_weights` is generic over any model with a
state_dict; it was only ever used by `train_ppo.py`.

```sh
# 1. collect (writes datasets/transitions/<name> on each pod)
python -m fleet launch --experiment E --collect --script collect_transitions.py \
  --train-args "--transitions 700000 --workers 12 --mix-preset badplay --slots-dir assets/states --out datasets/transitions/v4-collect"
python -m fleet collected --experiment E            # total transitions + outcome mix

# 2. fit the shared preprocessing ONCE, locally (scales, nav grid, split)
python scripts/train_wm_tokens.py --data <local data> --epochs 1 --save-prep configs/wm_prep_v4.json

# 3. train (normal launch path: train_wm_tokens.py accepts the --fleet-* flags)
python -m fleet launch --experiment E --script train_wm_tokens.py --run-tag wm --wandb-project zelda-wm \
  --train-args "--data datasets/transitions/v4-collect --prep configs/wm_prep_v4.json --epochs 100 --lr 1e-5 \
                --rollout-every 5 --ckpt-dir runs/wm100 --ckpt-every 5 [--resume runs/wm100/ck_0005.pt]"
python -m fleet metrics --experiment E --dir runs/wm100     # train/val/skill/dcell/rollout per epoch
```

**`--prep` is mandatory with FedAvg and the code refuses without it.** Every fitted quantity (17 feature
scales, nav-grid bounds, the y delta scale) comes from data; fitting them per-pod averages weights across ranks
whose inputs mean different things, which is silently wrong and presents as a training failure.

**`lr=1e-5` is the only stable setting.** The sweep: 1e-5 → skill −0.002 with best == last; 1e-4 → best −0.024,
last −9.919; 3e-4 → oscillates between −0.07 and −5.3. Reading the last epoch of a higher-LR run is how several
"degrades with training" conclusions got made. Best-epoch selection is in the script now.

**Pods with no shard cannot train.** 7 of 39 collected zero episodes and had to be terminated; check
`fleet collected --verbose` before launching training, not after.

## Collecting at fleet scale — the three things that silently destroy a dataset

All three were hit in one session. Each reported SUCCESS while the data was not where it needed to be.

1. **Pod-local storage as the only copy.** 20M transitions were lost by terminating a fleet after syncing only
   checkpoints. Always `--s3-prefix s3://zelda-rl/zelda/transitions/<run>`; the collector then uploads each
   episode as it is written. Verify with `fleet verify-s3` and kill with
   `fleet kill --check-s3 s3://.../<run>`, which REFUSES while any pod has unmirrored episodes.
2. **A flat S3 prefix across pods.** Every collector numbers episodes from `ep00001`, so 109 pods overwrote each
   other: 79,694 episodes on disk became ~20,000 objects in S3, a 75% loss that looked like a complete dataset.
   `S3Streamer` now namespaces by `--s3-shard` (default: hostname). Never share a shard.
3. **A missing dependency disabling the upload silently.** `boto3` was absent from `PY_DEPS`, so ~66 pods
   collected happily and uploaded nothing. `fleet preflight` now treats missing boto3 as a hard failure, and
   `fleet launch` skips pods that fail preflight.

Order of operations for a collection run:

```sh
python -m fleet create    --experiment E --count N --vcpu 16 --confirm
python -m fleet bootstrap --experiment E --assets-s3 --workers 50     # Dolphin, NAND, venv all from S3
python -m fleet assets    --experiment E --states-only --states-prefix zelda/combat/deku-baba/states
python -m fleet preflight --experiment E                              # MUST be 100% before launching
python -m fleet launch    --experiment E --collect --script collect_transitions.py \
    --train-args "... --out datasets/transitions/E --s3-prefix s3://zelda-rl/zelda/transitions/E"
python -m fleet audit     --experiment E                              # nothing idle, nothing billing for free
python -m fleet verify-s3 --experiment E --s3-prefix s3://zelda-rl/zelda/transitions/E
python -m fleet kill      --experiment E --check-s3 s3://zelda-rl/zelda/transitions/E --confirm
```

Bring a dataset local (per-rank prefixes, so pull whichever shards you need):

```sh
aws s3 sync s3://zelda-rl/zelda/transitions/v5/ datasets/transitions/v5/ --profile race-wm
aws s3 sync s3://zelda-rl/zelda/transitions/v5/rank000/ datasets/transitions/v5/rank000/ --profile race-wm
```

`load_many()` takes a list of directories and only the first needs `dataset.json`, so per-rank directories
concatenate with no merge step.
