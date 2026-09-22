#!/usr/bin/env python3
"""Pack a collected dataset into per-field .npy arrays that a training run can mmap in seconds.

WHY THIS EXISTS. Two separate limits, and only one of them is the GPU's.

  * HOST RAM during load. `load_many` materialises the whole f32 state array (1,452 B/frame -> 133 GB for
    v5's 91.6M frames) and `np.concatenate` doubles it, then `tokens.build` adds 16-row f32 tokens on top.
    Peak was ~266 GB. The H100 pod's cgroup limit is 188 GB and the run was OOM-killed after 20 minutes of
    loading. (`free` reports the HOST's 1.5 TB, not the container limit -- read /sys/fs/cgroup/memory.max.)
  * STARTUP TIME. That load takes ~25 minutes off a pod's network filesystem, every single run.

Packing in chunks fixes both. Never more than one chunk of f32 exists at a time, so peak RSS is a few GB over
the output size, and the output is ~583 B/frame = ~50 GiB for v5 -- small enough to sit resident in an 80 GB
GPU, so training does no host-device transfer either.

The token and target maths is NOT reimplemented here: it comes from `zelda_rl.wm_data`, which the trainer's
in-memory path uses too, so the two cannot drift.

    scripts/pack_dataset.py '/workspace/v5/rank000' --extra '/workspace/v5' '/workspace/v5/rank0*' \
        --out /workspace/packed/v5 --rows 9
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import load_many  # noqa: E402
from zelda_rl import wm_data  # noqa: E402
from zelda_rl.tokens import ACTOR_DIM, LINK_DIM  # noqa: E402


def episode_index(dirs):
    """(dir, manifest row) for every episode, plus the frame total to preallocate."""
    idx, frames = [], 0
    for d in dirs:
        rows = [json.loads(l) for l in (Path(d) / "manifest.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:
            idx.append((d, r))
            frames += int(r["steps"]) + 1        # load_many keeps min(len(action)+1, len(state)) frames
    return idx, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="first dataset dir; must contain dataset.json")
    ap.add_argument("--extra", nargs="*", default=[], help="more dirs or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=3000, help="episodes per chunk")
    ap.add_argument("--fit-episodes", type=int, default=4000, help="episodes sampled to fit scales/grid")
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt", help="carries the slot split")
    ap.add_argument("--prep", default=None, help="reuse an existing prep bundle instead of fitting")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dirs = [a.data]
    for e in a.extra:
        dirs += sorted(g for g in glob.glob(e)) if any(c in e for c in "*?[") else [e]
    dirs = list(dict.fromkeys(d.rstrip("/") for d in dirs))
    dirs = [d for d in dirs if (Path(d) / "manifest.jsonl").exists()]
    if not (Path(dirs[0]) / "dataset.json").exists():
        sys.exit(f"{dirs[0]} has no dataset.json (load_many reads it from the FIRST dir)")
    meta = json.loads((Path(dirs[0]) / "dataset.json").read_text())
    NA = meta["n_actions"]

    import torch
    split = (json.load(open(a.prep))["split"] if a.prep
             else torch.load(a.ref, weights_only=False, map_location="cpu")["split"])
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))

    index, n_alloc = episode_index(dirs)
    print(f"{len(dirs)} dirs | {len(index):,} episodes | allocating for {n_alloc:,} frames")

    # ---- fit scales / grid on a SAMPLE (see wm_data.fit_prep for why a sample is enough) ----
    if a.prep:
        pj = json.load(open(a.prep))
        from zelda_rl.navgrid import NavGrid
        sc, grid, sdy = pj["scales"], NavGrid.load(pj["grid"]), pj["d_link_y"]
        print(f"prep reused from {a.prep}")
    else:
        rng = np.random.default_rng(a.seed)
        pick = [index[i] for i in rng.choice(len(index), min(a.fit_episodes, len(index)), replace=False)]
        t0 = time.time()
        st, ac, sl, ep = _read(pick)
        tr_m = ~np.isin(sl, list(test | tune))
        sc, grid, sdy = wm_data.fit_prep(st, ep, tr_m, a.rows, a.grid)
        print(f"fitted prep on {len(st):,} frames from {len(pick):,} episodes in {time.time()-t0:.0f}s | "
              f"grid {grid.n}x{grid.n} cell {np.round(grid.cell,1)} | d_link_y {sdy:.2f}")
        del st, ac, sl, ep

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shapes = wm_data.field_shapes(a.rows, LINK_DIM, ACTOR_DIM)
    mm = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=wm_data.dtype_of(k),
                                       shape=(n_alloc, *sh)) for k, sh in shapes.items()}
    total = sum(v.dtype.itemsize * int(np.prod(sh or (1,))) for (k, sh), v in zip(shapes.items(), mm.values()))
    print(f"{len(mm)} fields | {total} B/frame | {total * n_alloc / 2**30:.1f} GiB output")

    w = 0
    t0 = time.time()
    for s in range(0, len(index), a.chunk):
        part = index[s:s + a.chunk]
        st, ac, sl, ep = _read(part)
        P = wm_data.tokens_and_targets(st, ac, sl, ep, grid=grid, scales=sc, d_link_y=sdy,
                                       rows_keep=a.rows, n_actions=NA)
        n = len(st)
        for k, v in P.items():
            mm[k][w:w + n] = v
        w += n
        del st, ac, sl, ep, P
        el = time.time() - t0
        done = s + len(part)
        print(f"  {done:,}/{len(index):,} episodes  {w:,} frames  {el:.0f}s  "
              f"eta {(len(index)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    for v in mm.values():
        v.flush()
    json.dump({"n_frames": int(w), "rows": a.rows, "n_actions": NA, "link_dim": LINK_DIM,
               "actor_dim": ACTOR_DIM, "scales": sc, "grid": grid.state_dict(), "d_link_y": sdy,
               "split": split, "bytes_per_frame": int(total), "dirs": dirs},
              open(out / "meta.json", "w"), indent=1)
    print(f"\nwrote {w:,} frames to {out} ({total * w / 2**30:.1f} GiB). "
          f"Allocated for {n_alloc:,}; the loader slices [:n_frames] from meta.json.")


def _read(part):
    """load_many over an explicit episode list, by grouping rows back into their directories."""
    from collections import defaultdict
    by_dir = defaultdict(list)
    for d, r in part:
        by_dir[d].append(r)
    S, A, SL, EP = [], [], [], []
    off = 0
    for d, rows in by_dir.items():
        for r in rows:
            arr = np.load(Path(d) / r["file"])
            a_, st = arr["action"], arr["state"]
            n = min(len(a_) + 1, len(st))
            S.append(st[:n])
            A.append(np.r_[a_[:n - 1], a_[n - 2]] if n > 1 else a_[:1])
            SL.append(np.full(n, r["slot"], np.int64))
            EP.append(np.full(n, off, np.int64))
            off += 1
    return np.concatenate(S), np.concatenate(A), np.concatenate(SL), np.concatenate(EP)


if __name__ == "__main__":
    main()
