#!/usr/bin/env python3
"""DEFEATS first: per-arm kill rate with a Wilson CI, group z-tests, and the within-arm early-vs-late split.

Damage led the reporting because at a 2-4% defeat rate a binary carries p(1-p) ~ 0.03 of variance per
episode against a graded 0-10 quantity (AGENTS 34) -- the choice that once hid an 8.7x improvement behind
six "flat" readings. But the TASK is defeating her, and with ~1,300 pooled episodes per group the rate is
measurable on its own. So defeats are the headline with damage beside them, and when the two disagree --
as on training length, where two of three arms kill MORE after epoch 40 while doing LESS damage -- that
disagreement is the finding, not something to resolve by picking the friendlier number.

    .venv/bin/python scripts/arm_defeats.py
"""
import re, sys
import numpy as np
sys.path.insert(0, "/Users/jordy/Documents/dev/zelda-RL")
from fleet.cli import R_REPO, for_pods, load_state, run
EV = re.compile(r"EVAL epoch (\d+): dmgB\(balanced\) ([\d.]+).*?wins (\d+)/(\d+)")

def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def ztest(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    sd = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return (p1 - p2) / sd if sd > 0 else float("nan")

jobs = []
for exp in ("gohma-la", "gohma-rvs"):
    for p in load_state(exp)["pods"]:
        p = dict(p); p["_exp"] = exp; jobs.append(p)
def grab(pod):
    out = run(pod, f"for f in {R_REPO}/logs/ep*_s*.log; do [ -f \"$f\" ] && echo \"F $(basename $f .log)\" "
                   f"&& grep -hE 'EVAL epoch' \"$f\"; done 2>/dev/null", capture=True, check=False, timeout=300)
    return pod["_exp"], pod["rank"], out
arms = []
for exp, rank, out in for_pods(jobs, grab, workers=8):
    cur, data = None, {}
    for line in out.splitlines():
        if line.startswith("F "):
            cur = line.split(None, 1)[1].strip(); data.setdefault(cur, []); continue
        m = EV.search(line)
        if m and cur:
            data[cur].append((int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))))
    for name, ev in data.items():
        if ev:
            arms.append((name, f"{exp.split('-')[-1]}{rank}", sorted(ev)))

print(f"{'arm':14}{'pod':8}{'kills':>10}{'rate':>8}{'95% CI':>18}{'dmgB':>7}")
for name, pod, ev in sorted(arms, key=lambda a: -sum(e[2] for e in a[2]) / max(sum(e[3] for e in a[2]), 1)):
    k = sum(e[2] for e in ev); n = sum(e[3] for e in ev)
    lo, hi = wilson(k, n)
    d = float(np.mean([e[1] for e in ev]))
    print(f"{name:14}{pod:8}{f'{k}/{n}':>10}{100*k/n:>7.2f}%{f'[{100*lo:.2f}, {100*hi:.2f}]':>18}{d:>7.2f}")

groups = {}
for name, pod, ev in arms:
    g = re.match(r"ep(\d+)_", name).group(1)
    k = sum(e[2] for e in ev); n = sum(e[3] for e in ev)
    a, b = groups.get(g, (0, 0))
    groups[g] = (a + k, b + n)
print("\n=== BY TRAINING LENGTH, defeats pooled ===")
for g in sorted(groups, key=int):
    k, n = groups[g]
    lo, hi = wilson(k, n)
    print(f"  {g:>4} epochs: {k}/{n} = {100*k/n:.2f}%  95% CI [{100*lo:.2f}, {100*hi:.2f}]")
if "40" in groups and "160" in groups:
    k1, n1 = groups["160"]; k2, n2 = groups["40"]
    z = ztest(k1, n1, k2, n2)
    print(f"  160 vs 40 epochs: z={z:+.2f}  "
          f"{'MORE TRAINING KILLS MORE' if z > 2 else 'FEWER' if z < -2 else 'not separable'}")
# the reference: rvs_M5
print(f"  reference rvs_M5: 8/240 = 3.33%  95% CI "
      f"[{100*wilson(8,240)[0]:.2f}, {100*wilson(8,240)[1]:.2f}]")
print(f"  reference BC clones: 0/400 = 0.00%  95% CI "
      f"[{100*wilson(0,400)[0]:.2f}, {100*wilson(0,400)[1]:.2f}]")

print("\n=== WITHIN-ARM on DEFEATS: early vs late (seed cancels) ===")
for name, pod, ev in sorted(arms):
    x = np.array([e[0] for e in ev])
    if x.max() < 60 or len(ev) < 8:
        continue
    cut = 40
    e_k = sum(e[2] for e in ev if e[0] <= cut); e_n = sum(e[3] for e in ev if e[0] <= cut)
    l_k = sum(e[2] for e in ev if e[0] > cut); l_n = sum(e[3] for e in ev if e[0] > cut)
    if min(e_n, l_n) < 50:
        continue
    z = ztest(l_k, l_n, e_k, e_n)
    print(f"  {name:14} <=40ep {e_k}/{e_n} = {100*e_k/e_n:.2f}%   >40ep {l_k}/{l_n} = {100*l_k/l_n:.2f}%   "
          f"z={z:+.2f}  {'MORE' if z > 2 else 'FEWER' if z < -2 else 'no effect'}")
