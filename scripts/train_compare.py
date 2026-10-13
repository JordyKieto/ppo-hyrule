"""Compare two training runs on their OWN per-slot episode outcomes, at matched updates.

The eval fleet scores published checkpoints on all ten starts, which is the right instrument for a hold-out
claim -- and it is slow: 40 episodes per evaluation, one checkpoint per 5 updates. Meanwhile the training run
has already played thousands of episodes on the six trained slots and logged every one. Waiting 40 minutes
for the evaluator while that sits unread is the mistake this script exists to stop.

What it does NOT do: say anything about held-out starts. The trained set is slots 3-8 only. It answers "is
this version beating the boss more often than the last one, on the distribution they both trained on", at an
n the evaluator cannot reach for hours.

Two things it enforces:

- **Matched updates** (AGENTS 12). A run 40 updates in is not comparable to one 6 updates in; the window is
  clipped to the shorter run unless --updates says otherwise, and the window used is printed.
- **Equal weight per rank-update, identically on both arms.** W&B logs `by_slot/sN/success_rate` as a mean
  over that rank-update's episodes without the count, so a perfectly episode-weighted pool is not
  recoverable from W&B alone. Both arms get the same treatment, so the comparison is fair even though the
  absolute rate carries a small bias; `--exact` re-derives true per-episode counts from the live pods' logs
  when they still exist.

    .venv/bin/python scripts/train_compare.py --baseline gohma-kl-v15 --compare gohma-kl-v16
    .venv/bin/python scripts/train_compare.py --compare gohma-kl-v16 --exact gohma-kl   # exact, live pods
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLOT_DIST = {1: 473, 2: 368, 3: 188, 4: 541, 5: 472, 6: 357, 7: 38, 8: 448, 9: 394, 10: 490}
SLOT_HP = {1: 5, 2: 5, 3: 10, 4: 8, 5: 10, 6: 6, 7: 5, 8: 10, 9: 5, 10: 5}
BANDS = (("near  <200u", 0, 200), ("mid 200-400u", 200, 400), ("far   >400u", 400, 10 ** 9))
ENTITY_PROJECT = "drift-ai-models/zelda-rl-gohma-ppo"


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0 or (a + c) == 0:
        return 1.0
    return sum(comb(a + c, x) * comb(b + d, a + b - x) / comb(n, a + b)
               for x in range(a, min(a + b, a + c) + 1))


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wandb_slots(group: str, max_update: int | None):
    """{slot: [sum of success_rate, count of rank-updates]} plus the highest update seen."""
    import wandb
    api = wandb.Api(timeout=90)
    runs = list(api.runs(ENTITY_PROJECT, {"group": group}))
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0])
    top = 0
    for r in runs:
        # Ask only for keys THIS run logged. W&B returns an empty history when any requested key is absent,
        # and a run trained on slots 3-8 never logs by_slot/s9 -- so requesting all ten silently yielded
        # nothing and the comparison printed an empty table with "updates 1-0".
        have = [f"by_slot/s{x}/success_rate" for x in SLOT_DIST
                if f"by_slot/s{x}/success_rate" in r.summary]
        if not have:
            continue
        for row in r.history(keys=have + ["_step"], pandas=False, samples=100000):
            u = row.get("_step")
            if u is None:
                continue
            top = max(top, int(u))
            if max_update is not None and int(u) > max_update:
                continue
            for s in SLOT_DIST:
                v = row.get(f"by_slot/s{s}/success_rate")
                if isinstance(v, (int, float)):
                    acc[s][0] += float(v)
                    acc[s][1] += 1
    return acc, top, len(runs)


EP = re.compile(r"\bslot(\d+)\s+(SUCCESS|DEATH|TIMEOUT)\b")


def exact_slots(experiment: str, group: str):
    """Exact per-episode counts, straight from every pod's log. Only works while the pods exist."""
    from fleet.cli import R_REPO, for_pods, load_state, run, select
    st = load_state(experiment)
    probe = (f"grep -hoE 'slot[0-9]+ (SUCCESS|DEATH|TIMEOUT)' {R_REPO}/logs/*{group.split('-')[-1]}*.log "
             f"2>/dev/null | sort | uniq -c | tr '\\n' ';'")

    def one(pod):
        try:
            return run(pod, probe, capture=True).strip()
        except Exception:                                                        # noqa: BLE001
            return ""
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for out in for_pods(select(st, None), one, workers=24):
        for chunk in (out or "").split(";"):
            m = re.match(r"\s*(\d+)\s+slot(\d+)\s+(\w+)", chunk)
            if not m:
                continue
            cnt, slot, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            acc[slot][1] += cnt
            if kind == "SUCCESS":
                acc[slot][0] += cnt
    return acc


def bands_of(per_slot: dict[int, list], scale: float = 1.0):
    """(label, wins, n) per band, then full HP, then overall. `scale` turns rate-sums into pseudo-counts."""
    def tot(keep):
        w = sum(per_slot[s][0] for s in per_slot if keep(s))
        n = sum(per_slot[s][1] for s in per_slot if keep(s))
        return w * scale, n * scale
    for name, lo, hi in BANDS:
        w, n = tot(lambda s: lo <= SLOT_DIST.get(s, -1) < hi)
        if n:
            yield name, w, n
    w, n = tot(lambda s: SLOT_HP.get(s) == 10)
    if n:
        yield "FULL HP (10)", w, n
    w, n = tot(lambda s: True)
    if n:
        yield "OVERALL", w, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="wandb group to judge, e.g. gohma-kl-v16")
    ap.add_argument("--baseline", default=None, help="wandb group to compare against, e.g. gohma-kl-v15")
    ap.add_argument("--updates", type=int, default=None,
                    help="clip BOTH runs to this many updates (default: the shorter run's length)")
    ap.add_argument("--exact", default=None,
                    help="fleet experiment whose live pod logs give exact per-episode counts for --compare")
    a = ap.parse_args()

    if a.exact:
        per = exact_slots(a.exact, a.compare)
        n = sum(v[1] for v in per.values())
        w = sum(v[0] for v in per.values())
        print(f"EXACT per-episode counts for {a.compare} from the live pods of {a.exact}: "
              f"{int(w)}/{int(n)} = {100 * w / max(n, 1):.2f}%\n")
        print(f"{'slot':>5}{'HP':>4}{'dist':>6}{'n':>7}{'wins':>6}{'rate':>8}")
        for s in sorted(per, key=lambda k: SLOT_DIST.get(k, 0)):
            wi, ni = per[s]
            print(f"{s:>5}{SLOT_HP.get(s, 0):>4}{SLOT_DIST.get(s, 0):>6}{ni:>7}{wi:>6}{100 * wi / ni:>7.2f}%")
        print()
        for name, wi, ni in bands_of(per):
            lo, hi = wilson(wi, ni)
            print(f"  {name:<14}{int(wi):>5}/{int(ni):<6}{100 * wi / ni:>6.2f}%   95% CI {100 * lo:.1f}-{100 * hi:.1f}%")
        if not a.baseline:
            return 0

    cmp_acc, cmp_top, cmp_n = wandb_slots(a.compare, None)
    if not a.baseline:
        print(f"{a.compare}: {cmp_n} ranks, highest update {cmp_top}; no --baseline given")
        return 0
    base_acc, base_top, base_n = wandb_slots(a.baseline, None)
    win = a.updates or min(cmp_top, base_top)
    print(f"matched window: updates 1-{win}  ({a.compare} has {cmp_top} on {cmp_n} ranks, "
          f"{a.baseline} has {base_top} on {base_n})")
    cmp_acc, _, _ = wandb_slots(a.compare, win)
    base_acc, _, _ = wandb_slots(a.baseline, win)
    print("weight: one rank-update per sample, identically on both arms (W&B does not log the per-slot "
          "episode count, so this is a rate-of-rates, not an episode pool)\n")

    print(f"{'band':<14}{a.compare[-3:]+' rate':>12}{'95% CI':>17}{a.baseline[-3:]+' rate':>12}{'95% CI':>17}{'p':>8}")
    bb = {k: (w, n) for k, w, n in bands_of(base_acc)}
    verdict = []
    for name, w, n in bands_of(cmp_acc):
        if name not in bb:
            continue
        w0, n0 = bb[name]
        lo, hi = wilson(w, n)
        lo0, hi0 = wilson(w0, n0)
        # Fisher needs integers; the rate-sums are rounded to the nearest pseudo-episode. With hundreds of
        # rank-updates per band the rounding is far below the sampling noise.
        p = fisher_greater(round(w), round(n - w), round(w0), round(n0 - w0))
        print(f"{name:<14}{100 * w / n:>10.2f}% {100 * lo:>7.1f}-{100 * hi:<8.1f}"
              f"{100 * w0 / n0:>10.2f}% {100 * lo0:>7.1f}-{100 * hi0:<8.1f}{p:>7.3f}")
        verdict.append((name, p, w / n, w0 / n0, n))

    sig = [v for v in verdict if v[1] < 0.05]
    up = [v for v in verdict if v[2] > v[3]]
    print()
    if sig:
        print("SIGNIFICANT at p<0.05: " + ", ".join(f"{v[0].strip()} (p={v[1]:.3f})" for v in sig))
    else:
        print(f"NOT SIGNIFICANT in any band. {len(up)}/{len(verdict)} bands point the right way.")
    print("NOTE: trained slots only (3-8). This says nothing about held-out starts -- that is the eval fleet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
