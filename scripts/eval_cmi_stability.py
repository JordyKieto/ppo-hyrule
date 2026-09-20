"""Is the discovered causal graph reproducible, or is the threshold deciding?

    .venv/bin/python scripts/eval_cmi_stability.py --ckpt runs/p1-paper-obj/model.pt

The paper does not run this control. We must: at 256 transitions two discovery passes over different
gameplay agreed on only 12 of 37 edges (Jaccard 0.194), so roughly two thirds of the discovered graph was
sampling noise -- and it showed, with edges like `Link.floor_height -> En_Dekubaba.pos'` that are not physics.

Feeding an unreproducible graph back into training would bake that noise in and LOOK like progress, which is
worse than failing loudly. So this sweeps the CMI sample size and reports, at each size, how much two
independent passes agree:

    Jaccard = |edges(A) & edges(B)| / |edges(A) | edges(B)|

If agreement climbs toward ~0.8 the estimator is simply sample-starved and we size there. If it plateaus
low, these effects are not identifiable at `causal_threshold` on the data we have, which is a finding about
the method here rather than a knob to turn -- it would mean recording more gameplay, not spending more
compute.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import OOCDM, discover_graph, pack  # noqa: E402


def episodes(cat, seed, n):
    eps = []
    for i in np.random.default_rng(seed).permutation(len(cat))[:n]:
        d, r = cat[i]
        z = np.load(d / r["file"])
        eps.append((z["state"], np.r_[z["action"], z["action"][-1]][: len(z["state"])]))
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/p1-paper-obj/model.pt")
    ap.add_argument("--data", nargs="+", default=[f"datasets/transitions/v3-{k}/20260919-084052"
                                                  for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--out", default="runs/cmi_stability.json")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cat = []
    for d in a.data:
        d = Path(d)
        cat += [(d, r) for r in (json.loads(l) for l in
                                 (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = OOCDM(ck["codecs"], ck["capacity"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()

    print(f"ckpt {a.ckpt} | threshold {a.threshold} | {m.graph.n_in} input fields, {m.graph.n_out} targets")
    print(f"\n  {'transitions':>12} {'edges A':>8} {'edges B':>8} {'shared':>7} {'Jaccard':>8} {'secs':>6}")
    rows = []
    for nb in a.batches:
        sets, t0 = [], time.time()
        for sd in (0, 1):
            p = pack(episodes(cat, 100 + sd, max(25, 12 * nb))).pad_to(ck["capacity"])
            _, cmi = discover_graph(m, p, ck["codecs"], dev, a.threshold, n_batches=nb,
                                    batch_size=a.batch_size, seed=sd)
            sets.append({k for k, v in cmi.items() if v > a.threshold})
        A, B = sets
        j = len(A & B) / max(len(A | B), 1)
        rows.append({"transitions": nb * a.batch_size, "a": len(A), "b": len(B),
                     "shared": len(A & B), "jaccard": j, "secs": time.time() - t0,
                     "stable_edges": sorted(A & B)})
        print(f"  {nb * a.batch_size:>12} {len(A):>8} {len(B):>8} {len(A & B):>7} {j:>8.3f} "
              f"{time.time() - t0:>6.0f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    best = rows[-1]
    print(f"\nedges reproduced at the largest size ({best['transitions']} transitions):")
    for e in best["stable_edges"][:20]:
        print(f"    {e}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
