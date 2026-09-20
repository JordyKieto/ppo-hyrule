"""The barometer: replay real action sequences through the OOCDM and compare to ground truth.

    .venv/bin/python scripts/eval_oocdm_rollout.py --ckpt runs/paper-fast-s0/model.pt

AGENTS.md section 8: anything claimed about a world model is validated by replaying REAL action sequences
from REAL start states and comparing decoded trajectories against ground truth at multiple horizons. Every
number in docs/CAUSAL_WORLD_MODEL.md sections 11 and 14 is single-step; none of them is this.

FULL-STATE ROLLOUT. Every one of the 77 state fields is predicted each step and fed back:

    s_0 = real start state
    s_{t+1} = recompose( model( decompose(s_t, a_t) ) )        with a_t the REAL action taken

so errors compound across fields exactly as they would in a dream. Section 22 is the record of what happens
when this is skipped: every single-step metric improved while the dream produced zero damage and zero kills
on real action sequences, and position ended worse than assuming Link never moved.

CONTROLS
  do-nothing   the state frozen at s_0 forever. This is the floor; section 22's latent model lost to it at
               every horizon past 25 steps, and b88868f measured the latent recursion 3x worse at h=10.
  velocity     s_0 position advanced by k * vel_0 each step, everything else frozen -- the free-signal
               control from AGENTS.md section 1, which beats the trained single-step head (+0.257 vs +0.064).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import OOCDM, make_batch, pack, predict_raw  # noqa: E402
from zelda_rl.oomdp import (SCHEMA_V3, decompose, err, prep, recompose)  # noqa: E402
from zelda_rl.state_layout import STATE_DIM_V3, link_index  # noqa: E402

HORIZONS = (1, 5, 10, 25, 50, 100)


def load_episodes(dirs, limit, seed, slots):
    cat = []
    for d in dirs:
        d = Path(d)
        cat += [(d, r) for r in (json.loads(l) for l in
                                 (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    cat = [(d, r) for d, r in cat if r["slot"] in slots]
    keep = sorted(np.random.default_rng(seed).permutation(len(cat))[:limit])
    out = []
    for i in keep:
        d, r = cat[i]
        z = np.load(d / r["file"])
        st, a = z["state"], z["action"]
        if len(st) >= max(HORIZONS) + 2:
            out.append((st, np.r_[a, a[-1]][: len(st)]))
    return out


@torch.no_grad()
def roll(model, codecs, state0, actions, horizon, device):
    """One episode, full-state autoregressive rollout. Returns (H+1, width) predicted states."""
    cur = state0.copy()[None]                                  # (1, width)
    traj = [cur[0].copy()]
    for t in range(horizon):
        a = actions[t: t + 1]
        p = pack([(cur, a)], SCHEMA_V3)
        p.pad_to(model.capacity)
        if len(p.pairs) == 0:                                  # a 1-step pack has no (t, t+1) pair
            p.pairs = np.array([0])
        b = make_batch(p, codecs, model, np.array([0]), device)
        pred = predict_raw(model, model(b), b)
        groups = decompose(cur, a, SCHEMA_V3)
        for cname, grp in groups.items():
            for f in grp.cls.state_fields():
                key = f"{cname}|{f.name}"
                if key in pred and f.name in grp.attrs:
                    grp.attrs[f.name][:] = pred[key][:, : grp.n_obj]
            if "present" in grp.attrs:
                grp.mask[:] = grp.attrs["present"][:, :, 0] > 0.5
        cur = recompose(groups, state0.shape[-1], SCHEMA_V3)
        traj.append(cur[0].copy())
    return np.stack(traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", default=[f"datasets/transitions/v3-{k}/20260919-084052"
                                                  for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=max(HORIZONS))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    eps = load_episodes(a.data, a.episodes, 0, {7, 9})
    print(f"{len(eps)} held-out episodes (test slots), horizon {a.horizon}, ckpt {a.ckpt}")

    if "codecs" not in ck:
        raise SystemExit(f"{a.ckpt} predates codec saving; retrain it. Codecs are fitted statistics and "
                         f"refitting them on a different sample changes head widths.")
    codecs = ck["codecs"]
    dev = torch.device(a.device)
    model = OOCDM(codecs, ck["capacity"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    LI = link_index(STATE_DIM_V3)
    px, pz = LI["pos_x"], LI["pos_z"]
    vx, vz = LI["vel_x"], LI["vel_z"]
    acc = {h: {"model": [], "none": [], "vel": []} for h in HORIZONS if h <= a.horizon}
    for st, act in eps:
        traj = roll(model, codecs, st[0], act, a.horizon, dev)
        k = 1.7
        for h in acc:
            truth = st[h][[px, pz]]
            acc[h]["model"].append(np.linalg.norm(traj[h][[px, pz]] - truth))
            acc[h]["none"].append(np.linalg.norm(st[0][[px, pz]] - truth))
            acc[h]["vel"].append(np.linalg.norm(st[0][[px, pz]] + h * k * st[0][[vx, vz]] - truth))
    print(f"\nLink XZ position error, world units (mean over {len(eps)} episodes)")
    print(f"  {'h':>4} {'do-nothing':>11} {'vel*1.7':>9} {'OOCDM':>9}   verdict")
    for h in sorted(acc):
        m, n, v = (float(np.mean(acc[h][k])) for k in ("model", "none", "vel"))
        best = min(m, n, v)
        who = "OOCDM" if m == best else ("do-nothing" if n == best else "vel")
        print(f"  {h:>4} {n:11.2f} {v:9.2f} {m:9.2f}   {who} wins"
              + ("" if m == best else f"  (OOCDM {m / n:.2f}x do-nothing)"))


if __name__ == "__main__":
    main()
