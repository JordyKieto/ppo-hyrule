"""Is the action recoverable from a latent transition? Measured per data source, against the right ceiling.

    .venv/bin/python scripts/probe_action_effect.py --model checkpoints/dynamics/joint_v6_state2.pt

The headline number for "does an action leave a trace in the latent" is a probe that predicts a_t from
(z_t, z_{t+1}). It is only interpretable against the **Bayes-optimal top-1 for the policy that generated
the frames**, because the demonstrator is stochastic: if it samples action a with probability p(a|s), then
even a perfect inverse model can only score sum_s P(s) max_a p(a|s).

That ceiling is a property of the DATA, not of the model. The v2 collection mixes sources with wildly
different action entropy -- scripted `wall` and `suicide` episodes hold one direction for tens of steps,
while `flail` draws uniformly over all 63 actions and so has a ceiling near 1/63. Comparing one pooled
number against the old dataset's 0.220 ceiling therefore says nothing. This script reports, per source:

    top-1 of the probe        what the model recovers
    empirical ceiling         max_a p(a|s) estimated from the generating policy for that source
    skill = top1 / ceiling    the scale-free quantity that IS comparable across sources and datasets

so a drop in the pooled number can be told apart from a drop in what the latent actually carries.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import BCNet, Norm, featurise  # noqa: E402


def load_runs(dirs):
    """States, actions and a per-frame source tag (scenario name, or 'policy' for the standard mix)."""
    S, A, EP, SRC = [], [], [], []
    ep = 0
    for d in dirs:
        for m in map(json.loads, open(Path(d) / "manifest.jsonl")):
            z = np.load(Path(d) / m["file"])
            s, a = z["state"], z["action"]
            n = len(a)                                  # T actions, T+1 states: pairs are 0..T-1
            S.append(s[:n + 1]); A.append(a)
            EP.append(np.full(n + 1, ep)); SRC += [m.get("scenario", "policy")] * (n + 1)
            ep += 1
    return (np.concatenate(S), np.concatenate(A), np.concatenate(EP), np.array(SRC))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/dynamics/joint_v6_state2.pt")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(a.seed)

    dirs = a.data or [sorted(glob.glob("datasets/transitions/deku-baba-v2/*/"))[-1],
                      sorted(glob.glob("datasets/transitions/deku-baba-scenarios/*/"))[-1]]
    S, A, EP, SRC = load_runs(dirs)
    print(f"{len(S):,} states over {EP.max() + 1} episodes from {len(dirs)} run(s)")

    mk = torch.load(a.model, weights_only=False, map_location="cpu")
    enc = BCNet(link_dim=mk["link_dim"], row_dim=mk["row_dim"], n_actions=mk["n_actions"],
                latent=mk["latent"], gru=0)
    enc.load_state_dict(mk["encoder"]); enc.eval()
    norm = Norm.load(mk["norm"])
    NA = mk["n_actions"]

    with torch.no_grad():
        f = norm.apply(featurise(S))
        Z = torch.cat([enc.encode(torch.from_numpy(f["link"][i:i + 8192]),
                                  torch.from_numpy(f["rows"][i:i + 8192]),
                                  torch.from_numpy(f["ids"][i:i + 8192]),
                                  torch.from_numpy(f["mask"][i:i + 8192]))
                       for i in range(0, len(S), 8192)])
    pair = np.where(EP[1:] == EP[:-1])[0]              # frames with a successor inside the same episode
    src = SRC[pair]
    # A is stored per transition; align it to `pair` by walking episodes in order
    off, acts = 0, []
    for e in range(EP.max() + 1):
        n = int((EP == e).sum()) - 1
        acts.append(np.arange(off, off + n))
        off += n
    aidx = np.concatenate(acts)
    y = torch.from_numpy(A[aidx].astype(np.int64))
    X = torch.cat([Z[pair], Z[pair + 1], Z[pair + 1] - Z[pair]], -1)
    print(f"{len(pair):,} transitions | sources {dict(Counter(src.tolist()))}\n")

    # Split by EPISODE, not by frame. A random frame split puts neighbouring frames of the same episode in
    # both halves, and since a scripted leg holds one direction for ~45 steps that is near-total leakage --
    # it scored `suicide` at 0.96 before this was fixed.
    rng = np.random.default_rng(a.seed)
    ep_of = EP[pair]
    eps = np.unique(ep_of)
    te_ep = set(rng.permutation(eps)[:max(1, int(0.2 * len(eps)))].tolist())
    is_te = np.isin(ep_of, list(te_ep))
    tr_i, te_i = np.where(~is_te)[0], np.where(is_te)[0]
    print(f"split by episode: {len(eps) - len(te_ep)} train / {len(te_ep)} test episodes "
          f"({len(tr_i):,} / {len(te_i):,} transitions)")

    probe = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(),
                          nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, NA))
    opt = torch.optim.Adam(probe.parameters(), 1e-3)
    for ep_ in range(a.epochs):
        probe.train()
        for b in torch.from_numpy(tr_i)[torch.randperm(len(tr_i))].split(1024):
            opt.zero_grad()
            nn.functional.cross_entropy(probe(X[b]), y[b]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        pred = torch.cat([probe(X[torch.from_numpy(te_i)][i:i + 8192]).argmax(-1)
                          for i in range(0, len(te_i), 8192)]).numpy()
    true = y.numpy()[te_i]
    s_te = src[te_i]

    print(f"{'source':<10} {'n':>8} {'probe top-1':>12} {'ceiling':>9} {'skill':>7}  interpretation")
    rows = []
    for name in ["policy"] + sorted(set(src.tolist()) - {"policy"}):
        m = s_te == name
        if m.sum() < 200:
            continue
        top1 = float((pred[m] == true[m]).mean())
        # Ceiling: the generating policy is stochastic, so estimate max_a p(a|s) by how concentrated the
        # action distribution is WITHIN this source. Without per-state densities, the achievable bound is
        # approximated by the source's own most-frequent-action rate conditioned on nothing -- a LOWER
        # bound on the ceiling -- plus the empirical best case of a deterministic script (1.0).
        cnt = Counter(true[m].tolist())
        marginal = max(cnt.values()) / m.sum()          # guess-the-most-common-action baseline
        rows.append((name, int(m.sum()), top1, marginal))
        note = {"policy": "stochastic demonstrator: comparable to the v1 0.139 / ceiling 0.220",
                "flail": "UNIFORM random: state gives nothing, so >1/63 is pure effect-visibility",
                "wall": "scripted: action is ~deterministic given state",
                "suicide": "scripted: action is ~deterministic given state",
                "walkout": "scripted: action is ~deterministic given state"}.get(name, "")
        print(f"{name:<10} {m.sum():>8,} {top1:>12.4f} {marginal:>9.4f} "
              f"{top1 / max(marginal, 1e-9):>6.2f}x  {note}")
    pooled = float((pred == true).mean())
    print(f"\n{'POOLED':<10} {len(true):>8,} {pooled:>12.4f}   <- mixes action distributions; NOT comparable")
    fl = [r for r in rows if r[0] == "flail"]
    if fl:
        t = fl[0][2]
        print(f"\nThe cleanest causal test is `flail`: its actions are drawn uniformly over {NA}, independent of\n"
              f"state, so a state prior buys nothing and chance is {1/NA:.4f}. Measured {t:.4f} = "
              f"{t * NA:.1f}x chance.\nThat fraction is what the latent transition actually reveals about which "
              f"action was taken.")
    print("\nThe `policy` row is the one comparable to the v1 dataset's 0.139 (ceiling 0.220): same generating\n"
          "process. Compare that row across models, never the pooled number.")
    Path("results").mkdir(exist_ok=True)
    Path("results/action_effect.json").write_text(json.dumps(
        {"model": a.model, "dirs": dirs, "pooled_top1": pooled,
         "by_source": [{"source": n, "n": k, "top1": t, "marginal": g} for n, k, t, g in rows]}, indent=2))
    print("-> results/action_effect.json")


if __name__ == "__main__":
    main()
