"""Can the OBSERVATION predict a damage event before it happens? A linear probe, not a policy.

Two questions the whole project rests on and neither has been measured:
  1. does obs_t carry "Gohma is about to lose HP" -- if a linear probe can read it, the information the
     reward needs is present and the policy's failure is credit assignment, not perception;
  2. does obs_t carry "LINK is about to be hurt" -- 81-90% of clone episodes end in death, so a readable
     danger signal is the difference between a policy that can learn to survive and one that cannot.

Trained on the same 91/9 episode split the clones use, so the number is held out the same way.

    .venv/bin/python scripts/probe_events.py --horizon 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tbh", REPO / "scripts" / "train_bc_human.py")
tbh = importlib.util.module_from_spec(spec); sys.modules["tbh"] = tbh; spec.loader.exec_module(tbh)


def fields(z):
    off, idx = 0, {}
    for n, w in json.loads(str(z["obs_fields"]))["global"]:
        idx[n] = off
        off += w
    return idx


def build(eps, horizon):
    X, Yb, Yl = [], [], []
    for e in eps:
        z = np.load(e["z"], allow_pickle=False)
        idx = fields(z)
        o = e["obs"]
        ghp, lhp = o[:, idx["gohma_hp_frac"]], o[:, idx["link_hp_frac"]]
        T = len(o)
        yb = np.zeros(T, np.float32)
        yl = np.zeros(T, np.float32)
        bd = np.where(ghp[:-1] - ghp[1:] > 1e-6)[0] + 1
        ld = np.where(lhp[:-1] - lhp[1:] > 1e-6)[0] + 1
        for m in bd:
            yb[max(0, m - horizon):m] = 1.0
        for m in ld:
            yl[max(0, m - horizon):m] = 1.0
        X.append(o); Yb.append(yb); Yl.append(yl)
    return np.concatenate(X), np.concatenate(Yb), np.concatenate(Yl)


def auc(y, s):
    o = np.argsort(s)
    y = y[o]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit(Xtr, ytr, Xte, yte, l2=10.0):
    """Ridge on centred/scaled features: a LINEAR read of the observation, so a high AUC means the signal is
    there for the taking rather than that a big model can tease it out."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    return auc(yte, B @ w), float(ytr.mean()), float(yte.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human/gohma")
    ap.add_argument("--horizon", type=int, default=10, help="an event counts if it happens within N steps")
    a = ap.parse_args()
    eps = tbh.load(Path(a.data), False, 0)
    by_slot = {}
    for i, e in enumerate(eps):
        by_slot.setdefault(e["slot"], []).append(i)
    val = set()
    for sl in sorted(by_slot, key=lambda k: (-len(by_slot[k]), k)):
        if len(val) >= 9:
            break
        if len(by_slot[sl]) > 1:
            val.add(by_slot[sl][-1])
    tr = [e for i, e in enumerate(eps) if i not in val]
    te = [e for i, e in enumerate(eps) if i in val]
    print(f"train {len(tr)} episodes / held out {len(te)}, horizon {a.horizon} steps ({a.horizon/10:.1f}s)\n")
    Xtr, Btr, Ltr = build(tr, a.horizon)
    Xte, Bte, Lte = build(te, a.horizon)
    for name, ytr, yte in (("GOHMA about to lose HP", Btr, Bte), ("LINK about to be hurt", Ltr, Lte)):
        a_, base_tr, base_te = fit(Xtr, ytr, Xte, yte)
        print(f"{name:<26} AUC {a_:.3f}   base rate train {100*base_tr:.1f}% / held out {100*base_te:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
