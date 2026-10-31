"""Pooled per-update signal for the full-episode-credit run, straight from W&B.

AGENTS 31: the fast instrument is the training log, not a held-out eval fleet. AGENTS 38: a single rank's
update is a sample -- the claim is the pooled mean over ranks with its SE. AGENTS 34: the headline is boss
damage (graded), mix-invariant (`combat/boss_damage_balanced`), with the slope reported as slope +- SE and
no trend language under |t|>2 or fewer than 5 updates.

EXPERIMENT-CONTRACT: exempt -- read-only analysis of rows W&B already holds; writes nothing.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

V15_BALANCED = 3.275            # the level to beat, PROVENANCE 3.2
V14_SLOPE = 0.012               # the trend to beat, as slope +- SE with |t|>2

KEY = "combat/boss_damage_balanced"
ALT = "combat/boss_damage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="gohma-cred-v20")
    ap.add_argument("--project", default="drift-ai-models/zelda-rl-gohma-ppo")
    ap.add_argument("--min-ranks", type=int, default=4, help="ignore updates measured by fewer ranks")
    a = ap.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    api = wandb.Api(timeout=60)
    allr = [r for r in api.runs(a.project, order="-created_at", per_page=100) if a.group in r.name]
    # ONE RUN PER RANK, newest first. The 4-pod launch and the 18-pod relaunch share this group, and
    # pooling a dead rank's rows beside a live one's is the duplicate-counting error of AGENTS 32 -- which
    # moved three p-values toward significance the last time it happened.
    runs, seen = [], set()
    for r in allr:                                    # api.runs is ordered newest-first
        rank = r.name.rsplit("-r", 1)[-1]
        if rank in seen:
            continue
        seen.add(rank); runs.append(r)
    print(f"{len(runs)} live ranks in {a.group} ({len(allr) - len(runs)} superseded runs ignored)")
    per_upd: dict[int, list[float]] = {}
    kills = eps = 0
    for r in runs:
        # scan_history returns dicts and does not need pandas; history(pandas=True) silently returns a
        # list when pandas is absent, which is how the first version of this crashed on .columns
        # NO key filter: scan_history(keys=[...]) returns nothing when any requested key is missing from a
        # row, which is how the first version of this reported "no update measured yet" against 18 runs
        # that each held 14 updates of data. Read the rows, then pick the fields.
        try:
            hist = list(r.scan_history())
        except Exception:
            hist = []
        for row in hist:
            u = row.get("update", row.get("_step"))
            v = row.get(KEY, row.get(ALT))
            if u is None or v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            per_upd.setdefault(int(u), []).append(float(v))
        s = r.summary or {}
        kills += int(s.get("combat/defeats", 0) or 0)
        eps += int(s.get("combat/episodes", 0) or 0)

    rows = [(u, np.array(v)) for u, v in sorted(per_upd.items()) if len(v) >= a.min_ranks]
    if not rows:
        print("no update has been measured by enough ranks yet")
        return 0
    print(f"\n{'upd':>4} {'ranks':>6} {'balanced damage':>17}   vs v15 {V15_BALANCED}")
    for u, v in rows:
        se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        d = v.mean() - V15_BALANCED
        print(f"{u:>4} {len(v):>6}   {v.mean():6.3f} +- {se:5.3f}      {d:+.3f}")
    if len(rows) >= 5:
        x = np.array([u for u, _ in rows], float)
        y = np.array([v.mean() for _, v in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ b
        se = float(np.sqrt((res @ res / max(len(x) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = b[0] / se if se else float("nan")
        verdict = "RISING" if t > 2 else "FALLING" if t < -2 else "flat"
        print(f"\nslope {b[0]:+.4f} +- {se:.4f} /update  t={t:+.2f} -> {verdict} over {len(rows)} updates")
        print(f"the number to beat is v14's {V14_SLOPE:+.3f}/update, judged |t|>2")
    else:
        print(f"\n{len(rows)} update(s) pooled -- a slope needs 5; level only until then (AGENTS 38)")
    if eps:
        print(f"defeats so far: {kills}/{eps} = {100*kills/eps:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
