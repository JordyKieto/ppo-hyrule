"""Audit: are the fields the reward function reads actually modelled? Scored on EVENT-RICH windows.

    .venv/bin/python scripts/eval_reward_fields.py --ckpt runs/p2-loop/loop_state.pt

WHY THE WINDOW SELECTION MATTERS -- the mistake this script exists to avoid.

The first version of this audit scored every reward field over whole held-out episodes and reported
"+0.000 on 7 of 10 fields". En_Dekubaba.hp changed 8 times in ~25,000 transitions, Room.scene 4 times. A
skill computed over 8 samples is not a measurement, and section 23 of LATENT_WORLD_MODEL.md says exactly this:

    "Validated on event-rich windows, which is the ONLY place this shows, because whole held-out episodes are
     event-poor -- identity scores 0.92-1.00 there, so a frozen model looks fine."

So this selects, per field, only the (object, step) pairs where the field ACTUALLY CHANGED, and reports the
count alongside every number so a low-power cell is visible rather than quietly averaged in.

WHAT COMBATTRACKER.STEP() READS (zelda_rl/combat.py), and which model field supplies it:

  enemy hp            -> En_Dekubaba.hp          enemy_damage, kill, kill_speed_bonus, stun_hit
  enemy behaviour     -> En_Dekubaba.state_idx   stun_hit, and the lunge/block/dodge cycles
  enemy present       -> En_Dekubaba.present     target_lost (despawn that is NOT a kill)
  enemy position      -> En_Dekubaba.pos         approach shaping, shield range, facing
  Link hp             -> Link.hp                 link_damage, link_heal, death
  Link position       -> Link.pos                approach shaping, shield range
  Link facing         -> Link.yaw                _facing() -> shield_facing
  Link state flags    -> Link.sf1                SF1_SHIELD -> shield_facing; the dead bit
  lock-on             -> En_Dekubaba.is_targeted lockon, lock_hold
  scene               -> Room.scene              left_room penalty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import (OOCDM, GraphMask, OOCGraph, make_batch, pack, predict_raw,  # noqa: E402
                            velocity_baseline)
from zelda_rl.oomdp import err, prep, unit  # noqa: E402

REWARD_FIELDS = [
    ("En_Dekubaba|hp", "enemy_damage, kill, stun_hit"),
    ("En_Dekubaba|state_idx", "stun_hit, lunge/block/dodge cycles"),
    ("En_Dekubaba|present", "target_lost"),
    ("En_Dekubaba|pos", "approach, shield range, facing"),
    ("Link|hp", "link_damage, link_heal, death"),
    ("Link|pos", "approach, shield range"),
    ("Link|yaw", "facing -> shield_facing"),
    ("Link|sf1", "shield bit, dead bit"),
    ("En_Dekubaba|is_targeted", "lockon, lock_hold"),
    ("Room|scene", "left_room"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/p2-loop/loop_state.pt")
    ap.add_argument("--data", nargs="+", default=[f"datasets/transitions/v3-{k}/20260919-084052"
                                                  for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--episodes", type=int, default=0, help="0 = every test-slot episode available")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cat = []
    for d in a.data:
        d = Path(d)
        cat += [(d, r) for r in (json.loads(l) for l in
                                 (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    cat = [(d, r) for d, r in cat if r["slot"] in (7, 9)]
    if a.episodes:
        cat = cat[: a.episodes]
    eps = []
    for d, r in cat:
        z = np.load(d / r["file"])
        eps.append((z["state"], np.r_[z["action"], z["action"][-1]][: len(z["state"])]))
    p = pack(eps).pad_to(ck["capacity"])
    dev = torch.device(a.device)
    model = OOCDM(ck["codecs"], ck["capacity"]).to(dev)
    model.load_state_dict(ck["model"])
    if "graph" in ck:
        model.set_graph(OOCGraph(model.order, model.field_names, model.target_names,
                                 full=False).load_state_dict(ck["graph"]))
    model.eval()
    print(f"{len(eps)} test-slot episodes, {len(p.pairs)} transitions, graph {model.graph.n_edges()} edges")

    acc = {}
    with torch.no_grad():
        for s in range(0, len(p.pairs), 512):
            b = make_batch(p, ck["codecs"], model, p.pairs[s: s + 512], dev)
            pr = predict_raw(model, model(b, GraphMask(model.graph)), b)
            for c in model.order:
                for t in model.target_names[c.name]:
                    key, f = f"{c.name}|{t}", c.field(t)
                    v = b["valid"][c.name][t].cpu().numpy()
                    if not v.any():
                        continue
                    cur = b["raw"][c.name][t].cpu().numpy()[v]
                    nxt = p.attrs[c.name][t][b["t"] + 1][v]
                    ei = err(f, prep(f, nxt), prep(f, cur))
                    em = err(f, prep(f, nxt), prep(f, pr[key][v]))
                    ch = ei > 0
                    d = acc.setdefault(key, [0.0, 0.0, 0, 0])
                    d[0] += float(ei[ch].sum()); d[1] += float(em[ch].sum())
                    d[2] += int(ch.sum()); d[3] += len(ei)

    print(f"\n  {'field':<26} {'unit':>6} {'chg%':>6} {'n_chg':>7} {'identity':>9} {'model':>9} "
          f"{'skill':>8}  power  reward terms")
    n_zero = n_low = 0
    for key, why in REWARD_FIELDS:
        cls, fld = key.split("|")
        u = unit(next(c for c in model.order if c.name == cls).field(fld))
        if key not in acc:
            print(f"  {key.replace('|', '.'):<26} {'--':>6}  NOT A TARGET")
            continue
        d = acc[key]
        sk = 1 - d[1] / d[0] if d[0] else float("nan")
        # Section 23: a cell computed on a handful of changed samples is not a measurement.
        power = "OK" if d[2] >= 200 else ("thin" if d[2] >= 50 else "NONE")
        n_low += power == "NONE"
        n_zero += abs(sk) < 1e-9
        print(f"  {key.replace('|', '.'):<26} {u:>6} {100 * d[2] / d[3]:6.2f} {d[2]:>7} "
              f"{d[0] / max(d[2], 1):9.4f} {d[1] / max(d[2], 1):9.4f} {sk:+8.3f}  {power:>5}  {why}")
    print(f"\n  {n_zero}/{len(REWARD_FIELDS)} reward fields at exactly zero skill; "
          f"{n_low} have too few changed samples (<50) for the number to mean anything.")
    vb = velocity_baseline(p)
    print(f"  control: Link.pos = pos + {vb['k']:.2f}*vel scores {vb['val_chg']:+.3f} on {vb['n']} changed steps")


if __name__ == "__main__":
    main()
