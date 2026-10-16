"""Which OFFLINE metric predicts a BC clone's ONLINE boss damage?

Uniform validation accuracy does not (measured r = -0.11 across five arms, PROVENANCE 3.3), so every cheap
offline comparison this project could make was measuring the wrong thing. If some offline metric tracks online
damage, a BC change can be judged in seconds instead of an 80-episode eval.

Candidates, all computed on the SAME held-out episodes the arms validated on:
  * overall action accuracy -- the known-bad baseline
  * per-class DECISION accuracy: accuracy restricted to the steps of each event class
  * behavioural rates: how often the clone would fire / equip / swing, against the human's own rates
    (a clone can be accurate frame by frame and still never pull the slingshot out)

    .venv/bin/python scripts/offline_probes.py --arms <dir> --damage uniform=0.33,s12=0.53,...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location("tbh", REPO / "scripts" / "train_bc_human.py")
tbh = importlib.util.module_from_spec(spec)
sys.modules["tbh"] = tbh
spec.loader.exec_module(tbh)

from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTONS_V5  # noqa: E402
from zelda_rl.policy_io import load_policy  # noqa: E402

CLASSES = ["seed_hit", "boss_hp", "stun", "sword", "seed_fire", "sling_out", "larva"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="directory of <arm>.pt checkpoints")
    ap.add_argument("--data", default="datasets/human/gohma")
    ap.add_argument("--damage", required=True, help="arm=damage,arm=damage -- the ONLINE numbers to correlate")
    ap.add_argument("--val-episodes", type=int, default=9)
    a = ap.parse_args()
    dmg = {k.split("=")[0]: float(k.split("=")[1]) for k in a.damage.split(",") if "=" in k}

    eps = tbh.load(Path(a.data), False, 0)
    # the same split the arms used: the last episode of each slot with a spare
    by_slot = {}
    for i, e in enumerate(eps):
        by_slot.setdefault(e["slot"], []).append(i)
    val_idx = set()
    for sl in sorted(by_slot, key=lambda k: (-len(by_slot[k]), k)):
        if len(val_idx) >= a.val_episodes:
            break
        if len(by_slot[sl]) > 1:
            val_idx.add(by_slot[sl][-1])
    va = [eps[i] for i in sorted(val_idx)]
    print(f"held-out: {[e['file'] for e in va]}\n")

    btn = [set(COMBAT_BUTTONS_V5[b]) for _, b in COMBAT_ACTIONS_V5]
    fires = np.array([1.0 if "CD" in s else 0.0 for s in btn])
    swings = np.array([1.0 if "B" in s else 0.0 for s in btn])

    # per-episode class masks, from the same code the trainer weights with
    masks = []
    for e in va:
        z = np.load(e["z"], allow_pickle=False)
        w, _ = tbh.event_weights(z, e["obs"], e["act"], 12, 1.0)      # scale 1 -> weight 2.0 on class steps
        per = {}
        for c in CLASSES:
            wc, _ = tbh.event_weights(z, e["obs"], e["act"], 12, 1.0, {c})
            per[c] = wc > 1.0
        masks.append(per)

    rows = {}
    for f in sorted(Path(a.arms).glob("*.pt")):
        arm = f.stem
        model, cfg, _ = load_policy(f)
        model.eval()
        hit_all = n_all = 0
        hit_c = {c: 0 for c in CLASSES}
        n_c = {c: 0 for c in CLASSES}
        fire_pred = swing_pred = fire_true = swing_true = n_steps = 0
        with torch.no_grad():
            for e, per in zip(va, masks):
                o = torch.from_numpy(e["obs"]); y = torch.from_numpy(e["act"])
                h = model.init_hidden(1)
                first = torch.ones(1)
                pred = []
                for t in range(len(y)):
                    lg, _, h = model.step(o[t:t + 1], h, first)
                    first = torch.zeros(1)
                    pred.append(int(lg.argmax(-1)))
                pred = np.array(pred)
                ok = pred == e["act"]
                hit_all += int(ok.sum()); n_all += len(ok)
                for c in CLASSES:
                    m = per[c][:len(ok)]
                    hit_c[c] += int(ok[m].sum()); n_c[c] += int(m.sum())
                fire_pred += float(fires[pred].sum()); fire_true += float(fires[e["act"]].sum())
                swing_pred += float(swings[pred].sum()); swing_true += float(swings[e["act"]].sum())
                n_steps += len(ok)
        rows[arm] = {"acc": hit_all / max(n_all, 1),
                     **{f"acc_{c}": hit_c[c] / max(n_c[c], 1) for c in CLASSES},
                     "fire_rate_pred": fire_pred / max(n_steps, 1),
                     "fire_rate_human": fire_true / max(n_steps, 1),
                     "swing_rate_pred": swing_pred / max(n_steps, 1),
                     "swing_rate_human": swing_true / max(n_steps, 1),
                     "fire_ratio": fire_pred / max(fire_true, 1e-9),
                     "swing_ratio": swing_pred / max(swing_true, 1e-9)}
        print(f"{arm:<12} acc {rows[arm]['acc']*100:5.1f}%  " +
              "  ".join(f"{c[:5]} {rows[arm]['acc_'+c]*100:4.0f}%" for c in CLASSES) +
              f"  fire x{rows[arm]['fire_ratio']:.2f}  swing x{rows[arm]['swing_ratio']:.2f}")

    print("\nCORRELATION WITH ONLINE BOSS DAMAGE (n = arms with a final 80-episode number)")
    have = [k for k in rows if k in dmg]
    d = np.array([dmg[k] for k in have])
    print(f"  arms: {', '.join(f'{k}={dmg[k]:.2f}' for k in have)}\n")
    keys = ["acc"] + [f"acc_{c}" for c in CLASSES] + ["fire_ratio", "swing_ratio", "fire_rate_pred"]
    out = []
    for k in keys:
        v = np.array([rows[x][k] for x in have])
        r = float(np.corrcoef(d, v)[0, 1]) if v.std() > 1e-12 else float("nan")
        out.append((abs(r) if r == r else -1, r, k))
    for _, r, k in sorted(out, reverse=True):
        flag = "  <-- tracks damage" if abs(r) > 0.8 else ""
        print(f"  {k:<16}{r:+.3f}{flag}")
    print(f"\n  n = {len(have)} arms: a correlation on five points is a hint, not a result (AGENTS 7)")
    Path("results/offline_probes.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
