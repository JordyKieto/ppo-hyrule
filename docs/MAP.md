# Repository map — read this before searching for anything

There are 17 documents and ~5,600 lines of notes in `docs/`. Without a map an agent greps for words it already
suspects, finds a plausible file, reads the top of it and proceeds — which is exactly how a documented 3.5-minute
fleet fan-out became a one-hour outage (see `AGENTS.md` §14). This file is the index. Keep it current.

## "I am about to ..." → read this first

| Task | Read | Why |
|---|---|---|
| add a RAM field / observation | `ADDING_OBSERVATIONS.md`, `RAM_MAP.md` | offsets come from the OoT decomp, never recall; retail Player = decomp − 0x10 |
| the **current** world model (tokens + attention) | `TOKEN_WORLD_MODEL.md` | `train_wm_tokens.py`: state in, state out. What is measured, what is confounded, what to run next |
| train a WM at horizon / pick the next experiment | `WM_EXPERIMENT_REGIME.md` | the concluded teacher-forced → autoregressive regime: what it bought (positive through H5, nothing past it), the token WM regression, the saved checkpoints, and the next four experiments in order |
| a **latent** world model | `LATENT_WORLD_MODEL.md` (§1–§30), `JEPA_PLAN.md` | 30 sections of next-latent prediction. Six architectures, all lose to the identity. Most ideas here have been tried. |
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

Bring a dataset local — **use the script, not a bare `aws s3 sync`**:

```sh
scripts/pull_dataset.sh s3://zelda-rl/zelda/transitions/v5 datasets/transitions/v5    # ~9 min for 17 GB
aws s3 sync s3://zelda-rl/zelda/transitions/v5/rank000/ datasets/transitions/v5/rank000/ --profile race-wm  # one shard
```

The obvious one-line sync of the whole prefix is **12x slower** and it is not a bandwidth limit. 80k objects
of ~200 KB is request-latency bound, and `aws s3 sync` defaults to 10 concurrent requests in ONE process, so
the link sits idle. Measured on v5 (79,694 episodes, 17 GB): **588 files/min (126 min) flat, 7,164 files/min
(9 min)** fanned out 16-way over the per-rank prefixes. The script does the fan-out and then compares the
remote and local `.npz` counts, exiting non-zero on a gap (`sync` resumes, so the fix is to rerun it).

This is the same lesson as `AGENTS.md` §15 in the other direction: per-rank prefixes exist for correctness
(§2 above) but they are also what makes a parallel pull possible. A flat prefix would have no shards to
fan out over.

`load_many()` takes a list of directories and only the first needs `dataset.json`, so per-rank directories
concatenate with no merge step.

## Fast iteration: pack ONCE, then every run starts in ~2.5 minutes

The packed dataset is the only artefact a training run needs. Keep it in S3 and NOTHING ever repacks.

```sh
# ONCE per dataset (~40 min for v5). Streams episodes in chunks; peak RSS a few GB.
python scripts/pack_dataset.py /workspace/v5/rank000 --extra /workspace/v5 '/workspace/v5/rank*' \
    --out /workspace/packed/v5 --rows 9
aws s3 sync /workspace/packed/v5/ s3://zelda-rl/zelda/packed/v5/ --profile race-wm   # make it durable

# EVERY run after that, on any pod:
aws s3 sync s3://zelda-rl/zelda/packed/v5/ /workspace/packed/v5/ --profile race-wm   # ~49 GiB
python scripts/train_wm_tokens.py --packed /workspace/packed/v5 --rows 9 --device cuda --amp ...
```

| | before | after |
|---|---|---|
| time to first epoch | ~25 min (103k .npz opens + decompress + tokenise) | **~2.5 min** (mmap 49 GiB at 347 MB/s) |
| peak host RAM | ~266 GB — **OOM-killed on a 188 GB pod** | a few GB |
| what a fresh pod needs | 26 GB of .npz + the whole tokenisation pipeline | 49 GiB of .npy + meta.json |

**`/workspace` is a POD volume and is destroyed when the pod is terminated.** That is how a 20M-transition
dataset was lost once already (`AGENTS.md` §15-16). The packed set goes to S3 the moment it is built, verified
by comparing every file's size remotely, not by trusting the exit code.

**`--max-frames` is not how you iterate quickly any more.** It exists for a laptop; on a pod, load the whole
packed set and cut epochs instead, so the numbers stay comparable to the full run.

Measured on the H100, allocating the real arrays rather than estimating: the full 91,626,189-frame set is
**48.6 GiB resident**, and a real training step at batch 16384 / dim 384 peaks at **52.0 GiB of 79.2**, i.e.
activations cost only 3.4 GiB and there is ~25 GiB spare for a bigger model.

## Training the world model on ONE GPU pod (current fastest path)

Measured: the model in isolation does 9,667 frames/s at batch 2048 on 6 Mac CPU threads, and real end-to-end
training does 9,400 — so **CPU training is ~100% model forward/backward and data loading is noise.** Packing
the data to parquet or any other on-disk format buys nothing for throughput; the only lever is the GPU, where
the same model does **193k frames/s at dim 128 and ~290k at dim 384.**

```sh
python -m fleet create --experiment wm --count 1   --gpu "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3"   --container-disk 60 --volume-gb 200 --confirm            # priority list: a sold-out A100 falls through
# on the pod: pull the data straight from S3 (never push it from the Mac -- AGENTS.md 15)
scripts/pull_dataset.sh s3://zelda-rl/zelda/transitions/v5 /workspace/v5 32 race-wm
python scripts/rebuild_manifests.py /workspace/v5           # S3's manifests may be stale; see below
python scripts/train_wm_tokens.py --shards '/workspace/v5/rank*' --rows 9 --device cuda --amp   --dim 384 --heads 6 --batch 16384 --lr <swept> --epochs N --ckpt-dir runs/... --ckpt-every 5
```

**The whole dataset fits in VRAM, and that is the point.** 91.5M transitions at the old 3,260 B/frame is
298 GB; at ~590 B/frame it is ~54 GB, resident on an 80 GB card with room for activations, so training does
**zero host-device transfer**. Three changes get it there, none lossy:
- `--rows 9`: actor slots 9-15 are occupied in **0.000%** of v5 frames (at most 9 actors ever present). Slots
  are STABLE ACROSS TIME and the per-actor targets are per-slot, so present actors cannot be sorted forward —
  only always-empty trailing slots may be dropped. The trainer hard-errors if `--rows` would drop an occupied
  slot.
- f16 for continuous fields (already scaled to [-1,1], so 3 decimal digits are ample), uint8 for every index
  and mask, cast back per batch on the device.
- the action stays an **index**; its 63-wide f32 one-hot was 252 B/frame, more than Link's whole token.

**Re-sweep the LR whenever the batch size changes.** "lr=1e-5 is the only stable setting" was measured at
batch 512 on 203k frames. At batch 16384 that is 32x the batch and 450x the data; carrying the old number over
is unjustified.

**Two traps hit on the first GPU run:**
- `python … | grep …` into a file **buffers**. Without `python -u` and `grep --line-buffered` the log stays
  empty for the whole run, which is indistinguishable from a hang — 9.5 minutes were spent believing a healthy
  job (GPU 97%) was stuck.
- RunPod returns "no instances available" and "your account balance is too low" in the **same** 500. To tell
  them apart, try to create a tiny pod: if it is really money, the message drops the capacity clause.
