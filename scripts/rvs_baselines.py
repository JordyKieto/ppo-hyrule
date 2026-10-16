#!/usr/bin/env python3
"""Capture the STATIC RvS mixes' pooled online damage into results/rvs_baselines.json, from the pods' logs.

Written because the Dyna arms run on pods whose static predecessor is the thing they must beat, and those
pods are destroyed or repurposed before the comparison happens. On 2026-10-16 M1/M2/M4 were repurposed onto
Dyna while their own `rvs_M*.log` files were still sitting on the same pods -- the only copy of their eval
history. A baseline that lives in a chat message is not a baseline (AGENTS 32: the pooling step is code).

    .venv/bin/python scripts/rvs_baselines.py --experiment gohma-rvs            # capture
    .venv/bin/python scripts/rvs_baselines.py --compare dyna_M5_s0,dyna_M4_s4   # Dyna vs its baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fleet.cli import R_REPO, for_pods, load_state, run   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "rvs_baselines.json"
EV = re.compile(r"EVAL epoch (\d+): dmgB\(balanced\) ([\d.]+) \| pooled ([\d.]+) \| wins (\d+)/(\d+)")


def pooled(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    return float(a.mean()), (float(a.std(ddof=1) / len(a) ** 0.5) if len(a) > 1 else float("nan"))


def capture(exp: str) -> dict:
    pods = load_state(exp)["pods"]

    def grab(pod):
        out = run(pod, f"for f in {R_REPO}/logs/rvs_*.log {R_REPO}/logs/dyna_*.log "
                       f"{R_REPO}/logs/m5_*.log; do [ -f \"$f\" ] && echo \"FILE $(basename $f .log)\" "
                       f"&& grep -hE 'EVAL epoch' \"$f\"; done 2>/dev/null", capture=True, check=False,
                  timeout=300)
        return pod["rank"], out

    arms: dict[str, dict] = {}
    for rank, out in for_pods(pods, grab, workers=max(4, len(pods))):
        cur = None
        for line in out.splitlines():
            if line.startswith("FILE "):
                cur = line.split(None, 1)[1].strip()
                arms.setdefault(cur, {"arm": cur, "rank": rank, "evals": [], "wins": 0, "episodes": 0})
                continue
            m = EV.search(line)
            if m and cur:
                arms[cur]["evals"].append(float(m.group(2)))
                arms[cur]["wins"] += int(m.group(4))
                arms[cur]["episodes"] += int(m.group(5))
    for a_ in arms.values():
        a_["n_evals"] = len(a_["evals"])
        a_["pooled"], a_["se"] = pooled(a_["evals"]) if a_["evals"] else (float("nan"), float("nan"))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="gohma-rvs")
    ap.add_argument("--compare", default=None,
                    help="comma-separated arm names to score against their static baseline")
    a = ap.parse_args()
    arms = capture(a.experiment)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    # an arm's history only ever GROWS; never let a re-capture shrink a recorded baseline
    for k, v in arms.items():
        if k in prev and prev[k].get("n_evals", 0) > v["n_evals"]:
            arms[k] = prev[k]
    OUT.write_text(json.dumps(arms, indent=2, default=float))
    print(f"{len(arms)} arm(s) -> {OUT}")
    for k, v in sorted(arms.items(), key=lambda kv: -(kv[1]["pooled"] if kv[1]["pooled"] == kv[1]["pooled"]
                                                       else -9)):
        print(f"  {k:16} n={v['n_evals']:2d} evals  pooled {v['pooled']:.2f} +- {v['se']:.2f}  "
              f"kills {v['wins']}/{v['episodes']}")
    if a.compare:
        print("\n=== DYNA vs its STATIC baseline (same mix) ===")
        for name in a.compare.split(","):
            name = name.strip()
            d = arms.get(name)
            mix = re.search(r"(M\d)", name)
            base = arms.get(f"rvs_{mix.group(1)}") if mix else None
            if not d or not base:
                print(f"  {name}: no baseline found (need rvs_{mix.group(1) if mix else '?'})"); continue
            if d["n_evals"] < 3:
                print(f"  {name}: {d['n_evals']} eval(s) -- NO VERDICT, need 3+"); continue
            gap = d["pooled"] - base["pooled"]
            comb = (d["se"] ** 2 + base["se"] ** 2) ** 0.5
            t = gap / comb if comb > 0 else float("nan")
            word = ("DYNA HELPS" if t > 2 else "DYNA HURTS" if t < -2 else "no effect")
            print(f"  {name} {d['pooled']:.2f} +- {d['se']:.2f} (n={d['n_evals']}) vs "
                  f"rvs_{mix.group(1)} {base['pooled']:.2f} +- {base['se']:.2f} (n={base['n_evals']}): "
                  f"{gap:+.2f} +- {comb:.2f}  t={t:+.2f}  {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
