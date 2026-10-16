"""Export W&B run metadata and metric histories into `results/`, so the paper never depends on W&B staying up.

Format follows ../neural-simulators: one directory per exported run, holding run.json (identity and lifetime),
config.json (the exact flags), summary.json (final metrics) and history.csv (the full metric history). A paper
claim cites `results/<run_id>/`, not a W&B URL.

Every fleet run is 20 ranks in one group. Exporting all 505 runs is pointless -- FedAvg keeps them on one set
of weights -- so per group this exports the AGGREGATOR rank (the one the others sync to, rank 4 for the Gohma
fleet), falling back to the rank with the most steps. The group's full rank list is recorded in run.json so
nothing is silently dropped.

    .venv/bin/python scripts/export_wandb.py                       # every group, metadata only
    .venv/bin/python scripts/export_wandb.py --history             # + full metric histories (slower)
    .venv/bin/python scripts/export_wandb.py --groups gohma-band-v13,deku-combat-gru-v2b --history
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# zelda-rl-goal-ppo is where `fleet launch` sends combat runs by default -- it was missing from this
# list, so the BC/latent arms of 2026-10-15 were not exported until the fleet had already been lost.
PROJECTS = ("zelda-rl-combat-ppo", "zelda-rl-gohma-ppo", "zelda-rl-goal-ppo", "zelda-wm")
ENTITY = "drift-ai-models"


def pick(runs):
    """The run that represents a fleet group: the aggregator if its name says so, else the longest."""
    for r in runs:
        if r.config.get("fleet_aggregator") is not None and \
           r.config.get("fleet_rank") == r.config.get("fleet_aggregator"):
            return r, "aggregator"
    return max(runs, key=lambda r: (r.summary.get("_step") or 0)), "longest"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", action="store_true", help="also export the full metric history (slow)")
    p.add_argument("--groups", default=None, help="comma-separated group names; default every group")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    want = set(a.groups.split(",")) if a.groups else None

    import wandb
    api = wandb.Api(timeout=60)
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    index = []

    for proj in PROJECTS:
        try:
            runs = list(api.runs(f"{ENTITY}/{proj}"))
        except Exception as e:
            print(f"{proj}: {str(e)[:100]}")
            continue
        groups = {}
        for r in runs:
            groups.setdefault(r.group or "(ungrouped)", []).append(r)
        for grp, rs in sorted(groups.items()):
            if want and grp not in want:
                continue
            rs.sort(key=lambda r: r.created_at)
            rep, how = pick(rs)
            d = out / rep.id
            d.mkdir(exist_ok=True)
            steps = max((r.summary.get("_step") or 0) for r in rs)
            meta = {
                "entity_project_run": f"{ENTITY}/{proj}/{rep.id}",
                "url": rep.url, "name": rep.name, "group": grp, "state": rep.state,
                "created_at": rep.created_at, "heartbeat_at": getattr(rep, "heartbeat_at", None),
                "representative_of_group": how,
                "group_size": len(rs),
                "group_run_ids": [r.id for r in rs],
                "group_run_names": [r.name for r in rs],
                "max_step_in_group": steps,
            }
            (d / "run.json").write_text(json.dumps(meta, indent=2, default=str))
            (d / "config.json").write_text(json.dumps(dict(rep.config), indent=2, default=str))
            (d / "summary.json").write_text(json.dumps(
                {k: v for k, v in rep.summary.items() if not k.startswith("_wandb")}, indent=2, default=str))
            rows = 0
            if a.history:
                try:
                    # pandas is not a dependency of this repo, so take the list-of-dicts form and write the
                    # CSV directly. The union of keys is the header, because W&B rows are sparse: a metric
                    # logged only on eval steps is absent from the others.
                    h = rep.history(pandas=False, samples=100000)
                    if h:
                        import csv
                        cols = []
                        seen = set()
                        for row in h:
                            for k in row:
                                if k not in seen:
                                    seen.add(k); cols.append(k)
                        with open(d / "history.csv", "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                            w.writeheader()
                            w.writerows(h)
                        rows = len(h)
                except Exception as e:
                    print(f"    history failed for {rep.id}: {str(e)[:70]}")
            index.append({"project": proj, "group": grp, "run_id": rep.id, "runs_in_group": len(rs),
                          "created_at": str(rep.created_at), "state": rep.state,
                          "max_step": steps, "history_rows": rows})
            print(f"{proj:<22}{grp:<26}{rep.id:<10}{len(rs):>3} ranks  step {steps:<9} "
                  f"{'hist ' + str(rows) if rows else ''}")

    (out / "INDEX.json").write_text(json.dumps(index, indent=2))
    print(f"\n{len(index)} groups exported to {out}/  (INDEX.json lists them)")


if __name__ == "__main__":
    main()
