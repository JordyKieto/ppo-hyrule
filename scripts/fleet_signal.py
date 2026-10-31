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
# scripts/train_rvs.py is a PURE-RvS arm: it has no PPO updates, it evaluates itself online every few
# epochs. Same headline (balanced boss damage), different x-axis, so the slope is per EPOCH there.
EVAL = re.compile(r"EVAL epoch (\d+): dmgB\(balanced\) ([\d.]+)")
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
        # ONLY the newest log. Reading every log on the pod made a retooled pod's new cell inherit the
        # previous arm's update rows: after gohma-la was retooled, `m5_w0_s0.3` reported the old
        # `la_B_s0` slope verbatim (-0.1068, t=-5.50) on data from an arm that no longer existed.
        body = run(pod, f"tail -{a.tail} \"$(ls -t {R_REPO}/logs/*.log 2>/dev/null | head -1)\" "
                        f"2>/dev/null", capture=True, check=False, timeout=300)
        # The arm's identity is its LOG FILE, not a pattern grepped out of the log: the first version
        # matched `v15-r4` out of the --init-from path and labelled every latent-alignment arm after the
        # checkpoint it started from. One log per arm is how every launcher here writes it.
        label = run(pod, f"ls -t {R_REPO}/logs/*.log 2>/dev/null | head -1 | xargs -r basename",
                    capture=True, check=False, timeout=200).strip().removesuffix(".log")
        return pod["_exp"], pod["rank"], label, body

    arms = []
    for exp, rank, label, body in for_pods(jobs, grab, workers=max(8, len(jobs))):
        seen: dict[int, float] = {}
        unit = "upd"
        for line in body.splitlines():
            m = ROW.search(line)
            if m:
                seen[int(m.group(1))] = float(m.group(5))
        if not seen:                                  # a pure-RvS arm reports online evals, not updates
            for line in body.splitlines():
                m = EVAL.search(line)
                if m:
                    seen[int(m.group(1))] = float(m.group(2))
                    unit = "epoch"
        arms.append({"exp": exp, "rank": rank, "name": label or "?", "u": seen, "unit": unit})

    # POOL THE FedAvg RANKS. This file's own docstring has always said "FedAvg ranks are pooled into one row
    # per group, never ranked individually" -- and it printed one row PER RANK, because an arm's label is its
    # log filename and every launcher ends that with `-r<N>`. So the table invited exactly the per-rank
    # reading the docstring forbids ("rank r0 is climbing" has been retracted twice here). Ranks of one
    # FedAvg group share a weight average: they are one model, and their evals are one sample.
    def group_of(name: str) -> str:
        return re.sub(r"[-_]r\d+$", "", name)

    groups: dict[str, list] = {}
    for arm in arms:
        groups.setdefault((arm["exp"], group_of(arm["name"])), []).append(arm)
    pooled_arms = []
    for (exp, gname), members in groups.items():
        if len(members) == 1:
            pooled_arms.append(members[0])
            continue
        merged: dict[int, list] = {}
        for m in members:
            for k, v in m["u"].items():
                merged.setdefault(k, []).append(v)
        # DROP THE IN-PROGRESS EVAL. Ranks finish an eval at different times, so the newest point is
        # usually a partial: at epoch 20, two of seven ranks had reported and the pooled value read 1.105;
        # with all six it was 1.250, which moved the slope from -0.0274 (t=-2.84, "falling") to -0.0165
        # (t=-1.99, borderline). Quoting a partial as a measurement happened three times in one session --
        # it is AGENTS §38 ("an eval is a sample, not a measurement") with the extra twist that the sample
        # was not even complete. A point is admitted only once as many ranks have reported as the
        # best-covered point has.
        if merged:
            full = max(len(v) for v in merged.values())
            partial = {k: len(v) for k, v in merged.items() if len(v) < full}
            if partial:
                print(f"  (dropping {len(partial)} in-progress point(s) {sorted(partial)} with "
                      f"{sorted(partial.values())} of {full} ranks reported)")
            merged = {k: v for k, v in merged.items() if len(v) == full}
        pooled_arms.append({"exp": exp, "rank": f"x{len(members)}", "name": gname,
                            "u": {k: sum(v) / len(v) for k, v in merged.items()},
                            # the unit comes from a member that HAS data: members[0] may be a rank whose
                            # first eval has not landed, and its default label ("upd") then mislabels the
                            # whole pooled row as update-indexed when it is epoch-indexed
                            "unit": next((m["unit"] for m in members if m["u"]), members[0]["unit"]),
                            "n_ranks": len(members),
                            "spread": {k: v for k, v in merged.items()}})
    arms = pooled_arms

    print(f"{'arm':24} {'pod':12} {'n':>3} {'last':>6} {'POOLED':>6} {'+-SE':>6} "
          f"{'slope':>9} {'SE':>7} {'t':>6}")
    for arm in sorted(arms, key=lambda x: x["name"]):
        u = np.array(sorted(arm["u"]))
        if not len(u):
            print(f"{arm['name']:24} {arm['exp'][-4:]+' r'+str(arm['rank']):12}   0      -       -      -"
                  f"          -       -      -  (no update line yet)")
            continue
        d = np.array([arm["u"][k] for k in u])
        sl, se = fit(u, d)
        t = sl / se if se == se and se > 0 else float("nan")
        # POOLED mean +- SE over every eval, not the latest one. A single 24-episode eval swings +-0.75
        # on this metric, which is larger than any between-arm difference measured today: M2 read 1.50 at
        # three evals and 0.75 at four, and M5 read 2.00 then 1.33. Ranking on the last value is ranking
        # on noise (AGENTS 35 rule 2); the pooled mean uses n_evals x 24 episodes.
        pooled = float(d.mean())
        pse = float(d.std(ddof=1) / len(d) ** 0.5) if len(d) > 1 else float("nan")
        print(f"{arm['name']:24} {arm['exp'][-4:]+' '+str(arm['rank']):12} {len(u):3d} "
              f"{d[-1]:6.2f} {pooled:6.2f} {pse:6.2f} {sl:9.4f} {se:7.4f} {t:6.2f}  /{arm['unit']}")

    # THE DECISION RULE, pre-registered in code so a ranking cannot be read off the table by eye.
    # A 24-episode eval of boss damage swings +-0.75 here. Ranked on the latest eval, M5 read 2.00 and
    # "led", then 1.33 and was retracted, then 1.50; M2 read 1.50 and "led", then 0.75. Both the claim and
    # the retraction were single-eval artefacts. Pooled over 5 evals (120 episodes) M5 is 1.60 +- 0.11 and
    # M2 0.93 +- 0.17: a real 0.67 +- 0.20 gap, t=3.3. So this prints the verdict itself, and prints TIE
    # whenever the gap to the runner-up is inside the combined SE (AGENTS 35 rule 5, 38).
    ranked = sorted(([x, float(np.mean([x["u"][k] for k in x["u"]])),
                      (float(np.std([x["u"][k] for k in x["u"]], ddof=1) / len(x["u"]) ** 0.5)
                       if len(x["u"]) > 1 else float("nan"))]
                     for x in arms if x["u"]), key=lambda r: -r[1])
    if len(ranked) >= 2:
        (a1, m1, e1), (a2, m2, e2) = ranked[0], ranked[1]
        comb = (e1 ** 2 + e2 ** 2) ** 0.5 if e1 == e1 and e2 == e2 else float("nan")
        gap = m1 - m2
        n1 = len(a1["u"])
        if not (comb == comb) or len(ranked[0][0]["u"]) < 3:
            verdict = f"NO VERDICT: {n1} eval(s) on the top arm, need 3+ to pool an SE"
        elif gap > 2 * comb:
            verdict = (f"LEADER {a1['name']} {m1:.2f} +- {e1:.2f} beats {a2['name']} {m2:.2f} +- {e2:.2f} "
                       f"by {gap:.2f}, t={gap/comb:+.2f}")
        elif gap > comb:
            # The first threshold here was `gap > comb`, i.e. t > 1, taken from 35 rule 5's "more than the
            # combined standard error". It printed LEADER at t=+1.80, which 34 rule 3 forbids calling an
            # improvement at all ("|t|>2 or do not say improving"). t>1 is ~68% one-sided: a coin with a
            # lean. LEADER now needs t>2 and this band is named for what it is.
            verdict = (f"WEAK: {a1['name']} {m1:.2f} +- {e1:.2f} leads {a2['name']} {m2:.2f} +- {e2:.2f} "
                       f"by {gap:.2f} but only t={gap/comb:+.2f} -- NOT separable at |t|>2, keep running")
        else:
            verdict = (f"TIE: {a1['name']} {m1:.2f} and {a2['name']} {m2:.2f} differ by {gap:.2f}, "
                       f"inside the combined SE {comb:.2f} -- NOT a ranking")
        print(f"\n=== VERDICT (pooled over every eval, {n1} evals on the top arm) ===\n{verdict}")

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
