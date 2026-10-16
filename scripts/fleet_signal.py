#!/usr/bin/env python3
"""ONE command that answers "is any arm doing anything" from the pods' own logs. Seconds, not minutes.

    .venv/bin/python scripts/fleet_signal.py                       # every live experiment
    .venv/bin/python scripts/fleet_signal.py --experiment gohma-col --control 'sw_c0_'

Why this exists (AGENTS.md 37). On 2026-10-16 the question "has any formula done anything" was asked four
times and answered with "~45 more minutes" three times, while the answer was already sitting in every pod's
log. Reconstructing it by hand cost ~40 minutes and three self-inflicted parse bugs:

  * `grep 'upd '` matched only the repeating HEADER line, so the first attempt reported `n=0` for all 23
    arms and looked like "no data yet";
  * the `steps` column is printed with thousands separators (`566,400`), so a `(\\S+)` field regex silently
    failed on every data row;
  * the throwaway script was called `signal.py`, which shadowed the stdlib module and broke `subprocess`.

Each was a minute of debugging against a furious user watching 35 pods bill. A question asked more than
once is a script (AGENTS.md 31: use the instrument you already have).

What it prints, per arm: updates parsed, latest `boss_damage_balanced`, and the SLOPE WITH ITS SE and t --
"flat" and "rising" are claims about a slope and a slope without an SE is an impression (AGENTS.md 34).
Then the only honest cross-arm read: every non-control arm group against the control arms over the SAME
update numbers, with the control group's own spread printed beside it, because five replicates of one
config spanned 2.00-2.97 on this metric and that spread is the floor for any claim (AGENTS.md 35).

FedAvg ranks are pooled into one row per group, never ranked individually: "rank r0 is climbing" has been
retracted twice in this project (AGENTS.md 36, PROVENANCE.md 5).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fleet.cli import R_REPO, STATE_DIR, for_pods, load_state, run   # noqa: E402

# `upd  steps(with commas)  eps  succ%  dmgB  dmgH` -- boss damage leads the trainer's line (AGENTS.md 34)
ROW = re.compile(r"INFO\s+(\d+)\s+([\d,]+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s")
NAME = re.compile(r"[a-z0-9]+_c[0-9.]+_w[0-9.]+_d[01]|[a-z0-9]+[0-9]*-r[0-9]+")


def fit(u: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """OLS slope and its standard error over ALL updates, not a last-k mean (AGENTS.md 35)."""
    if len(u) < 3:
        return float("nan"), float("nan")
    A = np.vstack([u, np.ones_like(u)]).T.astype(float)
    beta, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ beta
    se = float(np.sqrt((r @ r / max(len(u) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
    return float(beta[0]), se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="append", default=None, help="repeatable; default every live one")
    ap.add_argument("--control", default="_c0_",
                    help="substring identifying the CONTROL arms, e.g. 'sw_c0_' (default '_c0_')")
    ap.add_argument("--tail", type=int, default=800, help="log lines to read per pod")
    a = ap.parse_args()

    exps = a.experiment or sorted(p.stem for p in STATE_DIR.glob("*.json") if "terminated" not in p.stem)
    # A state file outlives its pods by weeks, so the API is the truth. `RunPod.list_pods` does not exist
    # -- the first version called it, the exception was swallowed, and the fallback "trust the state files"
    # then reported 64 live pods across 8 retired experiments. Same call fleet_health.py uses, and an empty
    # answer now means SKIP, not "probe everything".
    live = set()
    try:
        from fleet.runpod_api import RunPod
        live = {p.get("id") for p in (RunPod()._req("GET", "/pods") or [])}
    except Exception as e:                   # noqa: BLE001
        print(f"(could not list live pods: {str(e)[:70]})")
    if not live:
        print("no live pods on the account")
        return 0
    jobs = []
    for e in exps:
        for p in load_state(e)["pods"]:
            if not p.get("ip") or not p.get("port"):
                continue                     # never bootstrapped, or already terminated
            if p.get("id") not in live:
                continue
            p = dict(p); p["_exp"] = e; jobs.append(p)
    if not jobs:
        print("no live pods in any experiment state file")
        return 0
    print(f"{len(jobs)} live pod(s) across {len({j['_exp'] for j in jobs})} experiment(s)")

    def grab(pod):
        body = run(pod, f"for f in {R_REPO}/logs/*.log; do tail -{a.tail} \"$f\" 2>/dev/null; done",
                   capture=True, check=False, timeout=300)
        label = run(pod, f"grep -hoE '{NAME.pattern}' {R_REPO}/logs/*.log 2>/dev/null | tail -1",
                    capture=True, check=False, timeout=200).strip()
        return pod["_exp"], pod["rank"], label, body

    arms = []
    for exp, rank, label, body in for_pods(jobs, grab, workers=max(8, len(jobs))):
        seen: dict[int, float] = {}
        for line in body.splitlines():
            m = ROW.search(line)
            if m:
                seen[int(m.group(1))] = float(m.group(5))
        arms.append({"exp": exp, "rank": rank, "name": label or "?", "u": seen})

    print(f"{'arm':24} {'pod':12} {'n':>3} {'dmgB':>6} {'slope':>9} {'SE':>7} {'t':>6}")
    for arm in sorted(arms, key=lambda x: x["name"]):
        u = np.array(sorted(arm["u"]))
        if not len(u):
            print(f"{arm['name']:24} {arm['exp'][-4:]+' r'+str(arm['rank']):12}   0      -  (no update line yet)")
            continue
        d = np.array([arm["u"][k] for k in u])
        sl, se = fit(u, d)
        t = sl / se if se == se and se > 0 else float("nan")
        print(f"{arm['name']:24} {arm['exp'][-4:]+' r'+str(arm['rank']):12} {len(u):3d} "
              f"{d[-1]:6.2f} {sl:9.4f} {se:7.4f} {t:6.2f}")

    ctrl = [x for x in arms if a.control in x["name"] and x["u"]]
    rest = [x for x in arms if a.control not in x["name"] and x["u"]]
    if not ctrl or not rest:
        print(f"\n(no matched comparison: {len(ctrl)} control arms, {len(rest)} others)")
        return 0
    k = min(len(x["u"]) for x in rest)
    if k < 3:
        print(f"\n(matched comparison needs 3+ updates on every arm; the shortest has {k})")
        return 0
    upds = sorted(set.intersection(*[set(x["u"]) for x in rest]))[:k]

    def mean_over(x):
        v = [x["u"][i] for i in upds if i in x["u"]]
        return float(np.mean(v)) if v else float("nan")

    c = np.array([mean_over(x) for x in ctrl])
    c = c[~np.isnan(c)]
    print(f"\n=== MATCHED updates {upds[0]}..{upds[-1]} ({len(upds)}), boss_damage_balanced ===")
    print(f"control            {len(c)} arms  mean {c.mean():.3f}  sd {c.std(ddof=1):.3f}  "
          f"range {c.min():.2f}-{c.max():.2f}   <- the floor for ANY claim below")
    groups: dict[str, list[float]] = {}
    for x in rest:
        key = re.sub(r"-r\d+$", "", x["name"])            # FedAvg ranks pool into ONE row
        groups.setdefault(key, []).append(mean_over(x))
    for key, vals in sorted(groups.items()):
        g = np.array([v for v in vals if v == v])
        if not len(g):
            continue
        diff = g.mean() - c.mean()
        se = float(np.sqrt(g.var(ddof=1) / len(g) + c.var(ddof=1) / len(c))) if len(g) > 1 else float("nan")
        t = diff / se if se == se and se > 0 else float("nan")
        verdict = "no effect" if not (t == t) or abs(t) < 2 else ("BETTER" if t > 0 else "WORSE")
        print(f"{key:18} {len(g):2d} arms  mean {g.mean():.3f}  vs control {diff:+.3f} +- {se:.3f}  "
              f"t={t:+.2f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
