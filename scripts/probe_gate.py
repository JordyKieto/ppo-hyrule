"""Fast diagnostic: is the change gate informative, and is the value head any good? Two minutes, not thirty.

    .venv/bin/python scripts/probe_gate.py                    # ~2 min
    .venv/bin/python scripts/probe_gate.py --episodes 200 --epochs 3

WHY THIS EXISTS
---------------
The first gated run conflated three changes in one 30-minute experiment -- the gate, its `pos_weight`, and
threshold calibration -- and then could not say which one produced the outcome. That is the mistake
AGENTS.md section 9 names. Worse, two of the three never needed a training run at all:

  * the THRESHOLD is applied at evaluation time, so one trained model yields every threshold for free;
  * GATE QUALITY is a property of a binary classifier and is measured with AP, not with end-to-end error.

So this script trains once on a subset and then reports the three things separately:

  gate AP vs base rate   is "did this field change" predictable at all from (state, action)?
                         Section 29 measured TransitionHead's `moved` AP at 0.435 against a 6.8x-lower base
                         rate, so that is the number this is comparable to.
  value | changed        given that the field DID change, is the value head better than do-nothing?
                         Measured on changed steps only, so the gate cannot flatter or spoil it.
  threshold curve        total raw error as the firing threshold varies, including never-fire. This shows
                         WHY calibration chooses what it chooses instead of only reporting the choice.

A field can then be classified honestly: gate uninformative (AP ~ base rate), gate fine but value head bad
(AP high, value|changed <= 0), or both working.

SAMPLING, WHICH THE FIRST VERSION OF THIS SCRIPT GOT WRONG
-----------------------------------------------------------
Episodes are drawn at RANDOM across ALL the given collections, not as the first N of one. Taking the first 60
episodes of v3-main alone gave Link.pos a 0.629% change rate against 8.3% on the real test split -- v3-main is
combat, where Link is often standing still, while v3-approach and v3-moves are where he walks. A probe whose
subset makes the target event 13x rarer is measuring a different problem, and its numbers cannot be compared
with a full run's. `--episodes` is now a TOTAL across collections, sampled with a fixed seed, and the script
prints the realised change rate for a few reference fields so the subset can be checked against the census
before anything else in the table is believed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import (OOCDM, change_rates, field_losses, fit_codecs, make_batch,  # noqa: E402
                            pack, predict_raw, velocity_baseline)
from zelda_rl.oomdp import decompose, err, prep, unit  # noqa: E402

GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)     # 1.01 = never fire


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """AP = area under precision-recall, computed directly so this has no sklearn dependency."""
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    o = np.argsort(-score)
    tp = np.cumsum(label[o])
    prec = tp / np.arange(1, len(label) + 1)
    return float((prec * label[o]).sum() / n_pos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+",
                    default=[f"datasets/transitions/v3-{k}/20260919-084052"
                             for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--episodes", type=int, default=120, help="TOTAL episodes, sampled at random across all "
                                                              "collections (not the first N of one)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-gate", type=float, default=1.0)
    ap.add_argument("--pos-weight-cap", type=float, default=200.0)
    ap.add_argument("--gate", dest="gate", action="store_true", default=True)
    ap.add_argument("--gate-attached", dest="gate_detach", action="store_false", default=True,
                    help="let the gate loss backprop into the shared trunk (measured harmful)")
    ap.add_argument("--vector-bins", action="store_true",
                    help="binned XZ direction + magnitude heads for 3-vector fields (section 12.2)")
    ap.add_argument("--value-on-changed", action="store_true",
                    help="restrict the value loss to changed steps (the original design; measured harmful)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="train the value head on ALL steps, as the ungated arm does. The ONE variable that "
                         "separates the two regimes; gate columns are then meaningless and read as nan.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    t0 = time.time()
    catalogue = []
    for d in a.data:
        d = Path(d)
        catalogue += [(d, r) for r in
                      (json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    pick = np.random.default_rng(0).permutation(len(catalogue))[: a.episodes]
    tr_eps, te_eps = [], []
    for i in sorted(pick):
        d, r = catalogue[i]
        z = np.load(d / r["file"])
        e = (z["state"], np.r_[z["action"], z["action"][-1]][: len(z["state"])])
        (te_eps if r["slot"] in (7, 9) else tr_eps).append(e)

    # The delta scale must match the steps the value head trains on, or the targets are out of scale.
    codecs = fit_codecs([decompose(s, x) for s, x in tr_eps[:160]], changed_only=a.value_on_changed,
                        vector_bins=a.vector_bins)
    tr, te = pack(tr_eps), pack(te_eps)
    cap = {c: max(tr.capacity[c], te.capacity[c]) for c in tr.capacity}
    tr, te = tr.pad_to(cap), te.pad_to(cap)
    dev = torch.device(a.device)
    model = OOCDM(codecs, cap, use_gate=a.gate, value_on_changed=a.value_on_changed,
                  gate_detach=a.gate_detach).to(dev)
    rates = change_rates(tr, model)
    model.set_change_rates(rates, cap=a.pos_weight_cap)
    print(f"train {len(tr)} / test {len(te)} transitions, {sum(x.numel() for x in model.parameters())/1e6:.2f}M "
          f"params, load+pack {time.time()-t0:.0f}s")
    # Is this subset representative? Compare against the pooled census before believing anything below.
    ref = {"Link|pos": 21.6, "Camera|eye": 26.4, "En_Dekubaba|pos": 6.2, "En_Dekubaba|hp": 0.021}
    print("  subset change rates vs pooled census:  " + "   ".join(
        f"{k.replace('|', '.')} {100 * rates.get(k, 0):.2f}% (census {v}%)" for k, v in ref.items()))

    vb = velocity_baseline(te)
    if vb:
        print(f"  CONTROL  Link.pos = pos + {vb['k']:.2f}*vel  ->  val|chg {vb['val_chg']:+.3f} "
              f"(XZ, {vb['n']} changed steps). A value head that does not beat this has earned nothing.")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        t1, perm, tot, nb = time.time(), rng.permutation(tr.pairs), 0.0, 0
        for s in range(0, len(perm) - a.batch, a.batch):
            b = make_batch(tr, codecs, model, perm[s: s + a.batch], dev)
            L = field_losses(model, model(b), b)
            loss = a.w_gate * sum(v for k, v in L.items() if k.startswith("gate:")) \
                + sum(v for k, v in L.items() if not k.startswith("gate:"))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"  epoch {ep+1}/{a.epochs}  loss {tot/max(nb,1):8.2f}  ({time.time()-t1:.0f}s)")

    # ---- one pass over test, collecting everything the three questions need -----------------------------
    model.eval()
    acc: dict[str, dict] = {}
    with torch.no_grad():
        for s in range(0, len(te.pairs), 512):
            idx = te.pairs[s: s + 512]
            b = make_batch(te, codecs, model, idx, dev)
            o = model(b)
            if not a.gate:                       # no gate: score the value head's raw output everywhere.
                # NaN, not a constant: AP over constant scores ranks by array order, which is temporal, and
                # changes cluster in time -- a constant fill produced a bogus "40/47 gate informative".
                o = {"value": o["value"], "gate": {k: torch.full_like(v[..., 0], float("nan"))
                                                   for k, v in o["value"].items()}}
            for c in model.order:
                for t in model.target_names[c.name]:
                    key, f = f"{c.name}|{t}", c.field(t)
                    v = b["valid"][c.name][t].cpu().numpy()
                    if not v.any():
                        continue
                    cur = b["raw"][c.name][t].cpu().numpy()[v]
                    nxt = te.attrs[c.name][t][b["t"] + 1][v]
                    val = codecs[c.name][t].apply(b["raw"][c.name][t].cpu().numpy(),
                                                  o["value"][key].float().cpu().numpy())[v]
                    pr = torch.sigmoid(o["gate"][key]).cpu().numpy()[v]
                    e_id = err(f, prep(f, nxt), prep(f, cur))
                    e_md = err(f, prep(f, nxt), prep(f, val))
                    d = acc.setdefault(key, {"p": [], "y": [], "eid": [], "emd": [], "dir": [], "mag": []})
                    d["p"].append(pr); d["y"].append((e_id > 0).astype(np.float32))
                    d["eid"].append(e_id); d["emd"].append(e_md)
                    # For vector fields, split the error into DIRECTION and MAGNITUDE. A delta head can be
                    # worse than predicting zero for two opposite reasons -- too small (regression to the
                    # conditional mean) or pointing the wrong way -- and they need different fixes. Section
                    # 29 of the latent trail hit this and binned both, because "a mode does not shrink".
                    if f.vtype.kind == "real" and f.vtype.size >= 2:
                        tv, pv = (nxt - cur), (val - cur)
                        # For 3-vectors this is measured in the XZ PLANE. Link's vertical motion is ~0, so a
                        # spurious y component inflates a 3D angle while being irrelevant to walking -- the
                        # first version of this reported 142.8 deg for Link.pos on that basis, against a
                        # ground-truth check showing Link.vel[t] sits 11.3 deg from the next displacement in
                        # XZ. The plane is what the task cares about and what the transition head bins.
                        if f.vtype.size == 3:
                            tv, pv = tv[..., [0, 2]], pv[..., [0, 2]]
                        tn, pn = np.linalg.norm(tv, axis=-1), np.linalg.norm(pv, axis=-1)
                        ok = (tn > 1e-6) & (pn > 1e-6)
                        if ok.any():
                            cos = (tv[ok] * pv[ok]).sum(-1) / (tn[ok] * pn[ok])
                            d["dir"].append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                            d["mag"].append(pn[ok] / tn[ok])

    rows = []
    for key, d in acc.items():
        p, y = np.concatenate(d["p"]), np.concatenate(d["y"])
        have_gate = bool(np.isfinite(p).all())
        eid, emd = np.concatenate(d["eid"]), np.concatenate(d["emd"])
        base = float(y.mean())
        if base == 0:
            continue
        curve = ([float(np.where(p >= g, emd, eid).mean()) for g in GRID] if have_gate
                 else [float(emd.mean())] * (len(GRID) - 1) + [float(eid.mean())])
        m = y > 0.5
        rows.append({"field": key.replace("|", "."), "unit": unit(next(c for c in model.order
                                                                      if c.name == key.split("|")[0]).field(key.split("|")[1])),
                     "base": base,
                     "ap": average_precision(p, y) if have_gate else float("nan"),
                     "lift": (average_precision(p, y) / base) if have_gate else float("nan"),
                     "val_skill_chg": float(1 - emd[m].mean() / eid[m].mean()) if eid[m].mean() else float("nan"),
                     "best_thr": float(GRID[int(np.argmin(curve))]), "curve": curve,
                     "err_never": curve[-1], "err_best": float(min(curve)),
                     "dir_err_deg": float(np.median(np.concatenate(d["dir"]))) if d["dir"] else None,
                     "dir_within_45": float((np.concatenate(d["dir"]) < 45).mean()) if d["dir"] else None,
                     "mag_ratio": float(np.median(np.concatenate(d["mag"]))) if d["mag"] else None})

    rows.sort(key=lambda r: -(r["lift"] if np.isfinite(r["lift"]) else 0))
    print(f"\n{'field':<28} {'unit':>6} {'base%':>7} {'gateAP':>7} {'lift':>6} {'val|chg':>8} {'bestThr':>8} "
          f"{'err@best':>9} {'err@never':>9}")
    for r in rows:
        print(f"{r['field']:<28} {r['unit']:>6} {100*r['base']:7.3f} {r['ap']:7.3f} {r['lift']:6.1f} "
              f"{r['val_skill_chg']:+8.3f} {r['best_thr']:8.2f} {r['err_best']:9.4f} {r['err_never']:9.4f}")

    inf = [r for r in rows if np.isfinite(r["lift"]) and r["lift"] > 2]
    good_val = [r for r in rows if r["val_skill_chg"] > 0]
    both = [r for r in inf if r["val_skill_chg"] > 0]
    fires = [r for r in rows if r["best_thr"] <= 1.0]
    vec = [r for r in rows if r["dir_err_deg"] is not None]
    if vec:
        print(f"\nvector fields -- why the value head is wrong (median over changed steps):")
        print(f"  {'field':<28} {'val|chg':>8} {'dir err':>9} {'<45deg':>7} {'mag ratio':>10}   "
              f"(90 deg = no directional info; 1.0 = right size; XZ plane for 3-vectors)")
        for r in sorted(vec, key=lambda r: r["val_skill_chg"]):
            print(f"  {r['field']:<28} {r['val_skill_chg']:+8.3f} {r['dir_err_deg']:8.1f}d "
                  f"{100*r['dir_within_45']:6.1f}% {r['mag_ratio']:10.3f}")
    print(f"\ngate informative (AP > 2x base): {len(inf)}/{len(rows)}")
    print(f"value head beats do-nothing given a change: {len(good_val)}/{len(rows)}")
    print(f"BOTH: {len(both)}/{len(rows)}   calibration would fire at all: {len(fires)}/{len(rows)}")
    print(f"total {time.time()-t0:.0f}s")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
