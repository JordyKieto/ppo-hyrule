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
  velocity     s_0 position advanced by k * vel_0 each step, everything else frozen. NOTE: episodes start
               from savestates where Link is stationary, so vel_0 is zero and this control is IDENTICAL to
               do-nothing in a rollout. It is retained only so that fact stays visible.

AGENTS.md section 2: predictability and informativeness are reported TOGETHER. An error equal to do-nothing
is ambiguous -- the model may be tracking the truth as well as freezing does, or it may simply be producing
no motion. So DISTANCE TRAVELLED is reported next to error, as section 22 did, and a dream that travels ~0
while the truth travels tens of units is not a good model, whatever its error column says.
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
        # make_batch reads a (t, t+1) PAIR; a rollout step only has t. The current state is duplicated so
        # the pair exists -- the t+1 slot is only used to build loss targets, which a rollout never computes.
        a = np.repeat(actions[t: t + 1], 2)
        p = pack([(np.repeat(cur, 2, axis=0), a)], SCHEMA_V3).pad_to(model.capacity)
        b = make_batch(p, codecs, model, np.array([0]), device)
        from zelda_rl.oocdm import GraphMask
        pred = predict_raw(model, model(b, GraphMask(model.graph)), b)
        groups = decompose(cur, actions[t: t + 1], SCHEMA_V3)
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
    ap.add_argument("--episodes", type=int, default=40, help="candidates sampled; those shorter than the "
                                                                   "horizon are dropped, so fewer are scored")
    ap.add_argument("--horizon", type=int, default=max(HORIZONS))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    graph_sd = ck.get("graph")
    eps = load_episodes(a.data, a.episodes, 0, {7, 9})
    print(f"{len(eps)} held-out episodes (test slots), horizon {a.horizon}, ckpt {a.ckpt}")

    if "codecs" not in ck:
        raise SystemExit(f"{a.ckpt} predates codec saving; retrain it. Codecs are fitted statistics and "
                         f"refitting them on a different sample changes head widths.")
    codecs = ck["codecs"]
    dev = torch.device(a.device)
    model = OOCDM(codecs, ck["capacity"]).to(dev)
    model.load_state_dict(ck["model"])
    if graph_sd is not None:
        from zelda_rl.oocdm import OOCGraph
        model.set_graph(OOCGraph(model.order, model.field_names, model.target_names,
                                 full=False).load_state_dict(graph_sd))
    print(f"graph: {model.graph.n_edges()} edges")
    model.eval()

    LI = link_index(STATE_DIM_V3)
    px, pz = LI["pos_x"], LI["pos_z"]
    vx, vz = LI["vel_x"], LI["vel_z"]
    acc = {h: {"model": [], "none": [], "vel": [], "trav_m": [], "trav_r": []}
           for h in HORIZONS if h <= a.horizon}
    for st, act in eps:
        traj = roll(model, codecs, st[0], act, a.horizon, dev)
        k = 1.7
        for h in acc:
            truth = st[h][[px, pz]]
            acc[h]["model"].append(np.linalg.norm(traj[h][[px, pz]] - truth))
            acc[h]["none"].append(np.linalg.norm(st[0][[px, pz]] - truth))
            acc[h]["vel"].append(np.linalg.norm(st[0][[px, pz]] + h * k * st[0][[vx, vz]] - truth))
            # path length actually walked, dream vs real -- the informativeness half of AGENTS.md section 2
            acc[h]["trav_m"].append(float(np.abs(np.diff(traj[: h + 1][:, [px, pz]], axis=0)).sum()))
            acc[h]["trav_r"].append(float(np.abs(np.diff(st[: h + 1][:, [px, pz]], axis=0)).sum()))
    print(f"\nLink XZ position error, world units (mean over {len(eps)} episodes)")
    print(f"  {'h':>4} {'do-nothing':>11} {'OOCDM':>9} {'ratio':>7} | {'travelled dream/real':>22}")
    for h in sorted(acc):
        m, n, v = (float(np.mean(acc[h][k])) for k in ("model", "none", "vel"))
        tm, tr = float(np.mean(acc[h]["trav_m"])), float(np.mean(acc[h]["trav_r"]))
        print(f"  {h:>4} {n:11.2f} {m:9.2f} {m / n if n > 1e-9 else float('nan'):7.2f} | "
              f"{tm:10.1f} / {tr:<10.1f}")
    if False:
        pass


if __name__ == "__main__":
    main()
