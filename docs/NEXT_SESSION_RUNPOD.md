# Handoff: deploy the 20M-transition collection on a RunPod fleet

Everything below is committed and verified locally. The task is to run it on ~39 pods.
Read `docs/RUNPOD.md` first — it is the authoritative runbook and has measured numbers.

## Goal

**20M transitions** of mostly-BAD Deku Baba play, to scale the world model's training set from the current
203k. Bad play is the point: the reward terminates on death, room-exit and damage, and competent play produces
almost none of those (measured: 61 episodes of the default mix gave 47 success, 11 timeout, **2 deaths**, 1 exit).

## What is already done

- **`--mix-preset badplay`** in `scripts/collect_transitions.py`: 85% of episodes from G90 or below, including
  two **random-init** policies (`checkpoints/random/random-init-s{0,1}.pt`, untrained weights, committed).
  Weighted to the MIDDLE of the skill ladder — the lowest checkpoint never dies, it times out at 0/3 kills
  because it never engages. Deaths come from G30–G90.
- **`fleet sync --datasets <dir>`** pulls each pod's episodes into `datasets/transitions/<name>/rank<NN>/`.
  Per-rank subdirs because `load_many()` takes a directory list and only the first needs `dataset.json`.
- `fleet --script` already accepts `collect_transitions.py`.
- Linux RAM attach already works (`zelda_rl/linuxmem.py`, 0.8 s, `/dev/shm` mmap) — the macOS `python-dbg`
  signing requirement does NOT apply on pods.

## The arithmetic (from docs/RUNPOD.md, measured 2026-09-12)

| | |
|---|---|
| one 16-vCPU pod | 214–224 env steps/s at 12 envs |
| this Mac, 10 envs | ~140 steps/s (so one pod ≈ 1.5× the Mac) |
| 39 pods | ~8,580 steps/s → **20M in ~39 min** |
| cost | $0.48–0.56/hr per pod → **~$22/hr, ~$30 for the run** |
| dataset size | 0.11 KB/frame compressed → **2.2 GB total**, ~56 MB per pod |

## Commands

```sh
python -m fleet create    --experiment v4-collect --count 39 --vcpu 16 --confirm
python -m fleet bootstrap --experiment v4-collect --via-seeds 0,2     # pod->pod rsync; 39x upload from the Mac is too slow
python -m fleet verify    --experiment v4-collect --instances 12
python -m fleet launch    --experiment v4-collect --script collect_transitions.py \
    --train-args "--transitions 512000 --episodes 20000 --workers 12 --torch-threads 1 --mix-preset badplay"
python -m fleet status    --experiment v4-collect
python -m fleet sync      --experiment v4-collect --datasets datasets/transitions/<the out dir the pods used>
python -m fleet kill      --experiment v4-collect --confirm     # DO NOT FORGET: pods bill by the hour
```

512k transitions per pod × 39 = 20M.

## Things that will bite

1. **Pod creation fails intermittently.** `docs/RUNPOD.md` records 2 of 4 creates hitting RunPod
   `500 Something went wrong`, both fine on retry ~20 s later. Expect a few of 39 to need re-creating.
2. **Bootstrap is the long pole, not collection.** First pod ~9 min (builds Dolphin 2603a, 269 s, then cached
   as a 42 MB tarball); asset push is ~3 min/pod bound by the Mac's uplink, which is why `--via-seeds` exists.
3. **The mix checkpoints must reach the pods.** `BAD_PLAY_MIX` references `checkpoints/inbox/*` and
   `checkpoints/random/*`. Confirm bootstrap ships those, or the run dies on the first `resolve_checkpoint`.
4. **`--out` differs per pod** unless pinned. Pass an explicit `--out` in `--train-args` so `fleet sync
   --datasets` has one known path, otherwise each pod timestamps its own directory.
5. **Kill the fleet when done.** 39 idle pods is ~$22/hr.
6. **Orphaned Dolphins.** A collector that dies on startup leaves its Dolphins running; `pkill -f Dolphin` on
   the pod (this happened locally this session).

## After the data lands

The consumer is `scripts/train_wm_tokens.py` (entity tokens + nav-grid position codec, starts at persistence).
Point `--data` at the first rank dir and `--extra-data` at the rest.

**Open question it should settle:** every architecture this session starts at the persistence/identity baseline
and then degrades with training. The untested suspect is `--clip 1.0` against a gradient norm of 11–15 — every
step clipped ~12×, so each update is a fixed-size direction step. Try `--clip 20 --lr 1e-4` on the current 203k
BEFORE assuming more data is the fix; it is a 3-minute run and would explain a lot of this session if right.
