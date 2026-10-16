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
EV = re.compile(r"EVAL epoch (\d+): dmgB\(balanced\) ([\d.]+) \| pooled ([\d.]+) \|.*?wins (\d+)/(\d+)")


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
                arms[cur].setdefault("epochs", []).append(int(m.group(1)))
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
        print("  (a fresh arm's first evals are its EARLY training curve; the static reference rose then\n"
              "   flattened, so every comparison below is ALSO made at MATCHED epochs -- AGENTS 12)")
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
            print(f"  {name} DAMAGE {d['pooled']:.2f} +- {d['se']:.2f} (n={d['n_evals']}) vs "
                  f"rvs_{mix.group(1)} {base['pooled']:.2f} +- {base['se']:.2f} (n={base['n_evals']}): "
                  f"{gap:+.2f} +- {comb:.2f}  t={t:+.2f}  {word}")
            # KILLS too, because damage and defeats rank the arms DIFFERENTLY: w495_s0.3 has the highest
            # kill rate of any arm (4.76%) on damage of 1.40, while rvs_M5 has the highest damage (1.76)
            # at 3.33%. Damage is what the reward pays for (34) but a defeat is the task, so both are
            # reported. A two-proportion z on counts, which is the right test for a rate -- and 30 says to
            # state the n it needs: separating 4% from 2% at p<0.05 needs ~1,200 episodes per arm.
            k1, n1_ = d["wins"], d["episodes"]
            k2, n2_ = base["wins"], base["episodes"]
            # MATCHED EPOCHS: the same epoch numbers on both sides. Comparing a 3-eval arm's pooled mean
            # to a 10-eval reference's charges the new arm for not having converged yet, and comparing its
            # SLOPE to the reference's flat one credits it with the early rise every arm shows (rvs_M5's
            # own first three evals were 1.54, 2.00, 1.33 before it flattened near 1.76).
            de, be = d.get("epochs") or [], base.get("epochs") or []
            # Exact epoch numbers do not line up -- the Dyna arms eval every 3 epochs and the static ones
            # every 4, giving ZERO shared epochs. So match on the TRAINING BUDGET: take both arms' evals
            # inside the epoch span the younger arm has reached. That is the comparison AGENTS 12 asks for
            # (equal epochs), not equal eval indices.
            span = max(de) if de else 0
            shared = [e for e in de if e <= span]
            bsel = [e for e in be if e <= span + 1]
            if len(shared) >= 2 and len(bsel) >= 2:
                dv = np.array([d["evals"][de.index(e)] for e in shared])
                bv = np.array([base["evals"][be.index(e)] for e in bsel])
                g2 = float(dv.mean() - bv.mean())
                s2 = float(np.sqrt(dv.var(ddof=1) / len(dv) + bv.var(ddof=1) / len(bv))) if len(dv) > 1 \
                    else float("nan")
                t2 = g2 / s2 if s2 == s2 and s2 > 0 else float("nan")
                w2 = ("DYNA HELPS" if t2 > 2 else "DYNA HURTS" if t2 < -2 else "no effect")
                print(f"  {name} MATCHED to epoch {span} (dyna {shared} vs static {bsel}): "
                      f"{dv.mean():.2f} vs {bv.mean():.2f} "
                      f"= {g2:+.2f} +- {s2:.2f}  t={t2:+.2f}  {w2}")
            else:
                print(f"  {name} MATCHED: not enough evals inside the shared span yet "
                      f"(dyna {len(shared)}, static {len(bsel)})")
            if n1_ and n2_:
                p1, p2 = k1 / n1_, k2 / n2_
                pp = (k1 + k2) / (n1_ + n2_)
                sd = (pp * (1 - pp) * (1 / n1_ + 1 / n2_)) ** 0.5
                z = (p1 - p2) / sd if sd > 0 else float("nan")
                need = int(2 * 7.85 * pp * (1 - pp) / max((p1 - p2) ** 2, 1e-9)) if p1 != p2 else -1
                kw = ("MORE KILLS" if z > 2 else "FEWER KILLS" if z < -2 else "no effect")
                print(f"  {name} KILLS  {k1}/{n1_} = {100*p1:.2f}% vs {k2}/{n2_} = {100*p2:.2f}%: "
                      f"z={z:+.2f}  {kw}"
                      + (f"  (needs ~{need} episodes/arm to separate this gap)" if need > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
