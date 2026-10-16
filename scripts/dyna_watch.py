#!/usr/bin/env python3
"""The online readout for the Dyna loop: per-snapshot damage, behaviour and the slope with its SE.

The loop's online evaluator (`eval_fleet.py --watch`) appends one row per REAL episode to
`runs/<run>/online.jsonl` as each dream snapshot lands. This turns that into the thing worth looking at:

  * per snapshot: balanced boss damage, kill rate, and the BEHAVIOUR mix -- how it ends (death / timeout /
    success), hearts lost, episode length. A damage number without behaviour hides the difference between
    "fights longer" and "dies slower";
  * across snapshots: the damage SLOPE with its standard error, because "improving" is a claim about a
    slope and a slope without an SE is an impression (AGENTS 34);
  * mix-invariant: damage is the unweighted mean over slots, so it cannot move with the slot mix (AGENTS 29).

    .venv/bin/python scripts/dyna_watch.py --run gohma-dyna            # once
    .venv/bin/python scripts/dyna_watch.py --run gohma-dyna --follow   # every 60 s
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                                   # a row still being written
    return out


def balanced(rows: list[dict]) -> float:
    per: dict[int, list[float]] = {}
    for r in rows:
        per.setdefault(int(r.get("slot", -1)), []).append(float(r.get("boss_damage", 0)))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def report(run: str) -> None:
    p = ROOT / "runs" / run / "online.jsonl"
    rows = load(p)
    if not rows:
        print(f"{p}: no rows yet"); return
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get("policy") or r.get("ckpt") or "?"), []).append(r)
    order = sorted(by)
    print(f"{'snapshot':34}{'eps':>5}{'dmgB':>7}{'kill%':>7}{'death%':>8}{'timeout%':>9}"
          f"{'hpLost':>8}{'len_s':>7}")
    xs, ys = [], []
    for i, k in enumerate(order):
        rs = by[k]
        ends = Counter(str(r.get("end", "?")) for r in rs)
        n = len(rs)
        b = balanced(rs)
        kills = sum(1 for r in rs if r.get("boss_defeated"))
        hp = float(np.mean([float(r.get("hp_lost", 0)) for r in rs]))
        ln = float(np.mean([float(r.get("length", 0)) for r in rs])) / 10.0
        print(f"{Path(k).name[:34]:34}{n:>5}{b:>7.2f}{100*kills/n:>6.1f}%"
              f"{100*ends['death']/n:>7.1f}%{100*ends['timeout']/n:>8.1f}%{hp:>8.2f}{ln:>7.1f}")
        xs.append(i); ys.append(b)
    if len(xs) >= 3:
        x = np.asarray(xs, float); y = np.asarray(ys, float)
        A = np.vstack([x, np.ones_like(x)]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        se = float(np.sqrt((res @ res / max(len(x) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = beta[0] / se if se > 0 else float("nan")
        verdict = "RISING" if t > 2 else "FALLING" if t < -2 else "flat"
        print(f"\ndamage slope over {len(xs)} snapshots: {beta[0]:+.4f} +- {se:.4f} per snapshot  "
              f"t={t:+.2f}  {verdict}")
    else:
        print(f"\n{len(xs)} snapshot(s): need 3+ for a slope")
    print(f"total {len(rows)} real episodes evaluated online")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    while True:
        print(f"\n===== {time.strftime('%H:%M:%S')}  run {a.run}")
        report(a.run)
        if not a.follow:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
