"""Is each watched-behaviour term BITING? Per-arm behaviour counters at matched updates, straight from the pods.

EXPERIMENT-CONTRACT: exempt -- a read-only reporter; it trains nothing and evaluates nothing.

WHY THIS EXISTS, rather than reading boss damage. A defeat is a 1-10% binary and boss damage needs five
complete points before a slope is quotable (AGENTS 48). The behaviour counters are DENSE and the untreated
policy scores ~0 on most of them, so they answer "did this term do anything" in a handful of updates. On this
task a graded signal showed an 8.7x improvement across six versions that the binary reported as six
consecutive failures (AGENTS 34). If a term's own counter has not moved, that term is not working and waiting
will not fix it.

WHAT IT DOES NOT DO. It does not rank arms on boss damage -- `scripts/train_compare.py --trend` is for that,
and it refuses under five points for a reason. This script prints the counters and, for each, the treatment
mean against the control mean with an SE taken ACROSS PODS, because with `--fleet-sync none` every pod is an
independent seed and the pod is the unit of replication. Under FedAvg it would NOT be, and the script says so
rather than silently computing a wrong SE.

THE CONTROL'S n IS THE BINDING CONSTRAINT. At the measured seed spread (sd 0.380 on balanced boss damage over
five identical controls) three control pods give SE 0.219 whatever the treatment's n is, so the smallest
resolvable difference is ~0.46 and no amount of treatment pods changes that. Printed on every run so the
number is never read as more precise than it is.

    .venv/bin/python scripts/gohma_beh_signal.py --experiment gohma-beh
    .venv/bin/python scripts/gohma_beh_signal.py --experiment gohma-beh --pull   # refetch from the pods
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fleet.cli import R_REPO, for_pods, load_state, run  # noqa: E402

# (metric key, label, format, the measured baseline it has to beat). Baselines are from 5,610 logged v10-era
# episodes (runs/gohma-r*/episodes.jsonl); "-" means no counter existed before this session.
COUNTERS = [
    ("gohma/items_picked",        "items picked/ep",      "{:7.3f}", "0.878"),
    ("gohma/hearts_picked",       "hearts picked/ep",     "{:7.3f}", "~0.14"),
    ("gohma/grass_cut",           "grass cut/ep",         "{:7.3f}", "0.0000 (0% of eps)"),
    ("gohma/item_approach_units", "units closed to loot", "{:7.1f}", "-"),
    ("gohma/facing_frac",         "facing fraction",      "{:7.3f}", "-"),
    ("gohma/facing_cos",          "mean facing align",    "{:7.3f}", "-"),
    ("gohma/lock_frac",           "lock-on fraction",     "{:7.3f}", "0.223 (corpus 0.63)"),
    ("gohma/nuts_thrown",         "nuts thrown/ep",       "{:7.3f}", "5.64"),
    ("gohma/nuts_wasted",         "nuts wasted/ep",       "{:7.3f}", "-"),
    ("gohma/nut_waste_frac",      "nut waste fraction",   "{:7.3f}", "0.53 (outcome defn)"),
    ("gohma/red_eye_seeds",       "red-eye seeds/ep",     "{:7.3f}", "2.81"),
    ("gohma/red_eye_seed_conv",   "red-eye conversion",   "{:7.3f}", "0.048"),
    ("gohma/stun_to_sword_s",     "stun->sword (s)",      "{:7.2f}", "-"),
    ("gohma/stun_reach_frac",     "stun windows reached", "{:7.3f}", "-"),
    ("combat/boss_damage_balanced", "BOSS DAMAGE (bal)",  "{:7.3f}", "3.275 (v15)"),
]
SEED_SD = 0.380      # balanced boss damage, five identical controls, 2.00-2.97


def pull(exp: str, out: Path) -> None:
    """One thread pool over every pod, never a loop (the fleet ops rule). Writes <out>/<run_name>.jsonl."""
    st = load_state(exp)
    out.mkdir(parents=True, exist_ok=True)

    def one(pod):
        name = pod.get("run_name")
        if not name:
            return (pod["rank"], "no run_name in fleet state -- not launched from this machine")
        try:
            txt = run(pod, f"cat {R_REPO}/runs/{name}/updates.jsonl 2>/dev/null || true",
                      capture=True, timeout=120)
        except Exception as e:                                                   # noqa: BLE001
            return (pod["rank"], f"ssh failed: {e}")
        if not txt.strip():
            return (pod["rank"], "no updates.jsonl yet (no update has completed)")
        (out / f"{name}.jsonl").write_text(txt)
        return (pod["rank"], f"{len(txt.strip().splitlines())} updates")

    for rank, msg in sorted(for_pods(st["pods"], one, workers=33)):
        print(f"  rank{rank:<3} {msg}")


def load(out: Path, ctl_tag: str, beh_tag: str) -> dict[str, dict[int, list[dict]]]:
    """{arm: {update: [one row per POD]}}. Keyed on the run-name tag, which is what distinguishes the arms."""
    arms: dict[str, dict[int, list[dict]]] = {"control": {}, "treatment": {}}
    # *.withdrawn.jsonl holds rows from a run that was superseded in place: the trainer appends and restarts
    # its update counter at 1, so an unfiltered glob pools two rewards under one update number. That is
    # AGENTS 49 fault 6, which was recorded as open and then repeated. Rows are split out rather than
    # deleted, and this glob is what keeps them out of a comparison.
    for f in sorted(f for f in out.glob("*.jsonl") if not f.name.endswith(".withdrawn.jsonl")):
        stem = f.stem
        arm = ("control" if f"-{ctl_tag}-r" in stem else "treatment" if f"-{beh_tag}-r" in stem else None)
        if arm is None:
            print(f"  (skipping {stem}: name matches neither arm tag)")
            continue
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = d.get("update")
            if isinstance(u, int):
                arms[arm].setdefault(u, []).append(d)
    return arms


def mean_se(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return v[0], float("nan"), 1
    return statistics.fmean(v), statistics.stdev(v) / math.sqrt(len(v)), len(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="gohma-beh")
    ap.add_argument("--ctl-tag", default="ctl")
    ap.add_argument("--beh-tag", default="beh")
    ap.add_argument("--pull", action="store_true", help="refetch updates.jsonl from every pod first")
    ap.add_argument("--dir", default=None, help="where the pulled rows live (default results/<experiment>)")
    a = ap.parse_args()
    out = Path(a.dir) if a.dir else REPO / "results" / a.experiment
    if a.pull:
        print(f"pulling updates.jsonl from every pod of {a.experiment} ...")
        pull(a.experiment, out)
    if not out.exists():
        print(f"no rows at {out}; run with --pull")
        return 2

    arms = load(out, a.ctl_tag, a.beh_tag)
    # Pods per arm = the most rows any single update has. Taken from the data rather than from the fleet
    # state, so a pod that never produced an update cannot inflate the denominator (AGENTS 19).
    pods = {k: max((len(rows) for rows in v.values()), default=0) for k, v in arms.items()}
    print(f"\npods reporting: control {pods['control']}, treatment {pods['treatment']}")
    if not arms["control"] or not arms["treatment"]:
        print("one arm has no rows yet -- nothing comparable")
        return 1

    # AGENTS 48: admit an update only when BOTH arms have it, and only at full pod coverage for that arm.
    # A partial update is a different population, not a smaller sample -- the pods that finish first are not
    # a random subset. Three slope claims were reversed in consecutive messages for exactly this.
    usable = []
    for u in sorted(set(arms["control"]) & set(arms["treatment"])):
        if len(arms["control"][u]) == pods["control"] and len(arms["treatment"][u]) == pods["treatment"]:
            usable.append(u)
    shared = sorted(set(arms["control"]) & set(arms["treatment"]))
    if len(shared) > len(usable):
        print(f"dropping {len(shared) - len(usable)} in-progress update(s): not every pod has reported them")
    if not usable:
        print("no update is complete on every pod of both arms yet; nothing quotable")
        return 1
    u = usable[-1]
    print(f"matched update {u} (complete on all pods of both arms; {len(usable)} complete updates so far)\n")

    w = max(len(lbl) for _, lbl, _, _ in COUNTERS)
    print(f"{'counter':<{w}} {'control':>16} {'treatment':>16}   {'delta':>9}  {'|t|':>5}  baseline (v10-era)")
    print("-" * (w + 70))
    for key, lbl, fmt, base in COUNTERS:
        cm, cse, cn = mean_se([r.get(key) for r in arms["control"][u]])
        tm, tse, tn = mean_se([r.get(key) for r in arms["treatment"][u]])
        if cn == 0 and tn == 0:
            print(f"{lbl:<{w}} {'(metric absent)':>16}")
            continue
        d = tm - cm
        se = math.sqrt((cse if cse == cse else 0) ** 2 + (tse if tse == tse else 0) ** 2)
        t = abs(d) / se if se > 0 else float("nan")
        cs = (fmt.format(cm) + (f"+-{cse:.3f}" if cse == cse else "   n=1")) if cn else " " * 16
        ts = (fmt.format(tm) + (f"+-{tse:.3f}" if tse == tse else "   n=1")) if tn else " " * 16
        mark = " <<<" if t == t and t > 2 else ""
        print(f"{lbl:<{w}} {cs:>16} {ts:>16}   {d:>+9.3f}  {t:>5.2f}  {base}{mark}")

    print(f"\nSE is across PODS, which are independent seeds only under --fleet-sync none. "
          f"control n={pods['control']}, treatment n={pods['treatment']}.")
    cse_floor = SEED_SD / math.sqrt(max(pods["control"], 1))
    tse_floor = SEED_SD / math.sqrt(max(pods["treatment"], 1))
    tot = math.sqrt(cse_floor ** 2 + tse_floor ** 2)
    print(f"At the measured seed spread sd={SEED_SD} on balanced boss damage, the control's n gives "
          f"SE {cse_floor:.3f} and the treatment's {tse_floor:.3f}: this design resolves about "
          f"+-{2 * tot:.2f} HP and NO SMALLER. The CONTROL's n is the binding constraint.")
    print("Do NOT quote a boss-damage slope from fewer than five complete points (AGENTS 48); "
          "use scripts/train_compare.py --trend for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
