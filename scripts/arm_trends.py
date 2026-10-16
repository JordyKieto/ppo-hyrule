"""Per-arm boss-damage trends for the 2026-10-15 BC/latent arms, from the EXPORTED W&B histories.

The fleet was lost with every arm's checkpoints, so `results/<run_id>/history.csv` is the only surviving
record and this script is how its numbers are regenerated -- no pods, no manual transcription.

It reports the MIX-INVARIANT `combat/boss_damage_balanced` as the headline, with the episode-weighted
`reward/boss_damage` beside it, because a cross-arm claim on an episode-weighted average is a claim about the
curriculum as much as the policy (AGENTS 29). Slopes are over ALL points with the standard error and the
window printed, never a last-k mean (AGENTS 35).

    .venv/bin/python scripts/arm_trends.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BAR = 3.275          # v15, trained with no human data at all
ARMS = [
    ("FitNets projection", "0elfmc4m", 4, "latent: learned linear map before the MSE, rollout states"),
    ("latent transition", "w3wsholh", 4, "latent: MSE on h_t - h_(t-1)"),
    ("mutual alignment", "k3kooogu", 4, "latent: both nets trainable, contrastive"),
    ("one-way latent", "vc48mqx8", 4, "latent: MSE to the frozen clone, demo frames"),
    ("Q-filter", "vj96cuw5", 4, "action BC on the 12% of frames that damage her"),
    ("decay 0.1->0.02", "81mre82x", 12, "action BC, decaying to a floor"),
    ("decay replicate", "un4fkisv", 4, "as above"),
    ("constant 0.1", "73jjd4mu", 12, "action BC, fixed"),
    ("on-policy distil", "e6fssffk", 2, "KL to the clone at the policy's own states"),
    ("IBRL (broken)", "xb2e5oli", 12, "clone proposes, COLD Q head picks: bcPick 54-78%"),
    ("inverse", "72uww7w7", 4, "v15's representation into the clone"),
    ("BC init", "g6w9mnu3", 12, "the clone as the starting policy"),
]


def source_run(rid: str) -> str:
    """Which run the history came from. ONE RANK, not the fleet mean -- the export keeps the group's
    representative run only, and per-rank damage on this task spans 1.36-3.37, so a single rank's mean is not
    the arm's mean. Printed on every row so nobody reads one rank as a result (AGENTS 7)."""
    import json
    f = REPO / "results" / rid / "run.json"
    return json.load(open(f)).get("name", rid) if f.exists() else rid


def series(rid: str, col: str):
    f = REPO / "results" / rid / "history.csv"
    if not f.exists():
        return None
    vals = []
    for r in csv.DictReader(open(f)):
        v = r.get(col)
        if v not in (None, "", "NaN"):
            try:
                vals.append((float(r["_step"]), float(v)))
            except ValueError:
                pass
    return np.array([v for _, v in sorted(vals)]) if len(vals) >= 3 else None


def fit(y):
    x = np.arange(1, len(y) + 1)
    if len(y) < 5:
        return None
    b, c = np.polyfit(x, y, 1)
    r = y - (c + b * x)
    se = float(np.sqrt((r @ r) / (len(y) - 2) / ((x - x.mean()) @ (x - x.mean()))))
    return float(b), se, float(b / se)


def main() -> int:
    print(f"MIX-INVARIANT boss damage out of 10 (combat/boss_damage_balanced).\n"
          f"Each row is ONE RANK -- the W&B group's representative run -- not the arm's fleet mean.\n"
          f"v15's {BAR} was recorded earlier and NOT re-derived on this column from the same aggregation,\n"
          f"so treat it as a landmark, not a matched baseline (AGENTS 12).\n")
    print(f"{'arm':<20}{'rk':>3}{'n':>4}{'balanced':>10}{'max':>6}{'slope':>10}{'SE':>8}{'t':>7}"
          f"{'verdict':>11}  source rank")
    rows = []
    for name, rid, ranks, _ in ARMS:
        bal = series(rid, "combat/boss_damage_balanced")
        if bal is None:
            print(f"{name:<20}{ranks:>3}{'-':>4}{'no exported history':>35}")
            continue
        f = fit(bal)
        s = f"{f[0]:+.4f}{f[1]:>8.4f}{f[2]:>7.2f}" if f else f"{'<5 points':>25}"
        v = ("DECLINING" if f and f[2] < -2 else ("RISING" if f and f[2] > 2 else "flat")) if f else "-"
        print(f"{name:<20}{ranks:>3}{len(bal):>4}{bal.mean():>10.2f}{bal.max():>6.2f}{s}{v:>11}"
              f"  {source_run(rid)}")
        rows.append((name, bal))
    best = max(rows, key=lambda r: r[1].mean()) if rows else None
    print(f"\nsingle-rank means above v15's {BAR} landmark: "
          f"{[n for n, y in rows if y.mean() > BAR] or 'NONE'}  "
          f"<- ONE RANK EACH, not a result")
    print(f"one update above it: {len([n for n, y in rows if y.max() > BAR])} of {len(rows)} arms")
    if best:
        print(f"highest mean: {best[0]} at {best[1].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
