#!/usr/bin/env python3
"""Tabular metrics for the token world model: per-epoch, per-ENTITY, per-HORIZON, plus the rollout skills."""
import json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "/root/zelda-RL/runs/wm_v5_100"
f = os.path.join(d, "metrics.jsonl")
rows = [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
if not rows:
    sys.exit(f"no metrics yet at {f}")
G, R, B, X, DIM = "\033[32m", "\033[31m", "\033[1m", "\033[0m", "\033[2m"
HS = sorted({int(k.split("/")[0][1:]) for r in rows for k in r
             if k.startswith("h") and k.split("/")[0][1:].isdigit() and k.endswith("/link")})
RH = sorted({int(k.split("roll/h")[1].split("_")[0]) for r in rows for k in r if k.startswith("roll/h")
             and k.endswith("_skill")})
def col(v, w=8, good_high=True):
    if v is None: return f"{'-':>{w}}"
    c = (G if (v >= 0) == good_high else R)
    return f"{c}{v:>+{w}.3f}{X}"
print(f"{B}{os.path.basename(d)}   {len(rows)} epochs{X}")
print(f"{DIM}  train loss per ENTITY at each autoregressive step, then rollout skill vs persistence{X}")
hdr = f"{B}{'ep':>4} {'loss':>7} {'grad':>5} {'1step':>8} |"
for h in HS: hdr += f"{'H'+str(h)+' link':>10}{'H'+str(h)+' act':>10} |"
hdr += "".join(f"{'roll H'+str(h):>10}" for h in RH) + X
print(hdr)
print("-" * (len(hdr) - len(B) - len(X)))
for r in rows:
    line = f"{r['epoch']:>4} {r.get('train', float('nan')):>7.3f} {r.get('grad_mean', 0):>5.0f} "
    line += col(r.get("skill"), 8) + " |"
    for h in HS:
        line += f"{r.get(f'h{h}/link', float('nan')):>10.4f}{r.get(f'h{h}/actor', float('nan')):>10.4f} |"
    for h in RH:
        line += col(r.get(f"roll/h{h}_skill"), 10)
    print(line)
last = rows[-1]
heads = sorted(k for k in last if k.startswith("h1/") and not k.endswith(("/link", "/actor", "/other")))
if heads:
    print(f"\n{B}per-head loss, step h1, epoch {last['epoch']}{X}")
    for k in heads:
        base = k.split("/", 1)[1]
        deep = [last.get(f"h{h}/{base}") for h in HS]
        trend = "  ".join(f"h{h} {v:.4f}" if v is not None else f"h{h} -" for h, v in zip(HS, deep))
        print(f"  {base:<10} {trend}")
