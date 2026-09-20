"""THE BENCHMARK: roll the model on windows where a competency actually happens, and score % of real.

    .venv/bin/python scripts/eval_competencies.py --ckpt runs/p2-loop-oversample/loop_state.pt
    .venv/bin/python scripts/eval_competencies.py --ckpt ... --trace walk    # step-by-step side by side

WHY THIS IS THE BENCHMARK AND PER-FIELD SKILL IS NOT
-----------------------------------------------------
`Link.pos` scores skill +0.144 -- on the steps where Link moves, the model's SINGLE-STEP error is 14% better
than predicting no movement. Roll that same model forward on its own output for 20 steps with the real
actions and it travels **6% of the real distance**, ending further from the truth than freezing Link at the
start. Both numbers are correct. They measure different things, and only this one says whether the model can
be used as a dream.

So the headline metric is **% of real**: what fraction of the real quantity does the dream produce.

  walk      distance travelled, dream / real
  lockon    Z-target acquisitions, dream / real
  shield    shield RAISE events, dream / real (steps-with-shield-up is free to a frozen model)
  approach  distance closed to the nearest Deku Baba, dream / real
  combat    enemy HP removed, dream / real

Each window is selected because the REAL trajectory contains that competency -- so "real" is 1.00 by
construction and a window definition that is broken shows up as a real column that is not ~1.00. Each is
rolled from its own real start state with the real action sequence, full-state autoregressive: all 77 fields
predicted each step and fed back, exactly as a dream would.

The do-nothing control is reported alongside. A dream that produces 6% of real motion is not meaningfully
different from one that produces 0%.

SELECTION uses the REAL trajectory only, never the model, so the same windows score any checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import SF1_SHIELD, SF1_ZTARGET  # noqa: E402
from zelda_rl.oocdm import (OOCDM, GraphMask, OOCGraph, make_batch, pack, predict_raw)  # noqa: E402
from zelda_rl.oomdp import SCHEMA_V3, decompose, recompose  # noqa: E402
from zelda_rl.state_layout import ENEMY_FIELDS, N_ROWS, ROW_DIM, STATE_DIM_V3, link_index  # noqa: E402

LI = link_index(STATE_DIM_V3)
RI = {k: i for i, k in enumerate(ENEMY_FIELDS)}
PX, PZ, SC, SF1, HP = LI["pos_x"], LI["pos_z"], LI["scene"], LI["sf1"], LI["hp"]
DL = STATE_DIM_V3 - N_ROWS * ROW_DIM
BABA = 0x0055


def rows(st):
    return st[:, DL:].reshape(len(st), N_ROWS, ROW_DIM)


def travelled(st):
    return float(np.hypot(*np.diff(st[:, [PX, PZ]], axis=0).T).sum())


def lockons(st):
    z = (st[:, SF1].astype(np.int64) >> SF1_ZTARGET) & 1
    return float(((z[1:] == 1) & (z[:-1] == 0)).sum())


def shield_raises(st):
    """Shield RAISE events, not steps-with-shield-up.

    Counting steps was the first version and the do-nothing control killed it: if the shield is already up
    at step 0, freezing the state holds it up for the whole window, so do-nothing scored 21 of a real 23 and
    the metric was nearly free. Transitions cannot be obtained by freezing."""
    b = (st[:, SF1].astype(np.int64) >> SF1_SHIELD) & 1
    return float(((b[1:] == 1) & (b[:-1] == 0)).sum())


def _baba_dist(st):
    r = rows(st)
    live = (r[:, :, RI["present"]] > 0.5) & (r[:, :, RI["id"]].astype(np.int64) == BABA)
    d = np.where(live, np.hypot(r[:, :, RI["pos_x"]] - st[:, PX, None],
                                r[:, :, RI["pos_z"]] - st[:, PZ, None]), np.inf)
    return d.min(1)


def approach(st):
    d = _baba_dist(st)
    ok = np.isfinite(d[0]) and np.isfinite(d[-1])
    return float(max(d[0] - d[-1], 0.0)) if ok else 0.0


def hp_removed(st):
    r = rows(st)
    live = (r[:, :, RI["present"]] > 0.5) & (r[:, :, RI["id"]].astype(np.int64) == BABA)
    hp = np.where(live, r[:, :, RI["hp"]], np.nan)
    d = np.nan_to_num(hp[:-1] - hp[1:])
    return float(np.clip(d, 0, None).sum())


COMPETENCIES = {
    "walk":     (travelled,     lambda w: travelled(w) > 60 and (w[:, SC] == w[0, SC]).all(), "world units"),
    "lockon":   (lockons,       lambda w: lockons(w) >= 1, "acquisitions"),
    "shield":   (shield_raises, lambda w: shield_raises(w) >= 1, "raise events"),
    "approach": (approach,      lambda w: approach(w) > 30, "units closed"),
    "combat":   (hp_removed,    lambda w: hp_removed(w) >= 1, "enemy HP"),
}


@torch.no_grad()
def roll(model, codecs, state0, actions, horizon, dev):
    cur, traj = state0[None].copy(), [state0.copy()]
    for t in range(horizon):
        p = pack([(np.repeat(cur, 2, axis=0), np.repeat(actions[t:t + 1], 2))], SCHEMA_V3).pad_to(model.capacity)
        b = make_batch(p, codecs, model, np.array([0]), dev)
        pred = predict_raw(model, model(b, GraphMask(model.graph)), b)
        g = decompose(cur, actions[t:t + 1], SCHEMA_V3)
        for cn, grp in g.items():
            for f in grp.cls.state_fields():
                k = f"{cn}|{f.name}"
                if k in pred and f.name in grp.attrs:
                    grp.attrs[f.name][:] = pred[k][:, : grp.n_obj]
            if "present" in grp.attrs:
                grp.mask[:] = grp.attrs["present"][:, :, 0] > 0.5
        cur = recompose(g, state0.shape[-1], SCHEMA_V3)
        traj.append(cur[0].copy())
    return np.stack(traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/p2-loop-oversample/loop_state.pt")
    ap.add_argument("--data", nargs="+", default=[f"datasets/transitions/v3-{k}/20260919-084052"
                                                  for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=20, help="per competency")
    ap.add_argument("--trace", default="", help="print a step-by-step side-by-side for this competency")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    codecs, dev = ck["codecs"], torch.device(a.device)
    model = OOCDM(codecs, ck["capacity"]).to(dev)
    model.load_state_dict(ck["model"])
    if "graph" in ck:
        model.set_graph(OOCGraph(model.order, model.field_names, model.target_names,
                                 full=False).load_state_dict(ck["graph"]))
    model.eval()

    cat = []
    for d in a.data:
        d = Path(d)
        cat += [(d, r) for r in (json.loads(l) for l in
                                 (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    H = a.horizon
    wins = {k: [] for k in COMPETENCIES}
    for d, r in cat:
        if r["slot"] not in (7, 9):
            continue
        z = np.load(d / r["file"])
        st, act = z["state"], np.r_[z["action"], z["action"][-1]][: len(z["state"])]
        for t0 in range(0, len(st) - H - 1, H):
            w = st[t0: t0 + H + 1]
            for name, (_, sel, _) in COMPETENCIES.items():
                if len(wins[name]) < a.max_windows and sel(w):
                    wins[name].append((st[t0], act[t0: t0 + H], w, r["file"]))

    print(f"ckpt {a.ckpt} | graph {model.graph.n_edges()} edges | horizon {H} | test slots {{7,9}}\n")
    print(f"  {'competency':<10} {'windows':>8} {'real':>10} {'dream':>10} {'% of real':>10} "
          f"{'do-nothing':>11}  unit")
    out = {}
    for name, (measure, _, unit) in COMPETENCIES.items():
        ws = wins[name]
        if not ws:
            print(f"  {name:<10} {'0':>8}   (no windows found)")
            continue
        R = D = N = 0.0
        traces = []
        for s0, acts, w, fn in ws:
            traj = roll(model, codecs, s0, acts, H, dev)
            R += measure(w); D += measure(traj); N += measure(np.repeat(s0[None], H + 1, 0))
            traces.append((w, traj, acts, fn))
        pct = 100 * D / R if R else float("nan")
        print(f"  {name:<10} {len(ws):>8} {R:>10.1f} {D:>10.1f} {pct:>9.1f}% {N:>11.1f}  {unit}")
        out[name] = {"windows": len(ws), "real": R, "dream": D, "pct": pct, "do_nothing": N}
        if a.trace == name and traces:
            w, traj, acts, fn = traces[0]
            from zelda_rl.env import action_name
            print(f"\n  --- trace: {fn} ---")
            print(f"  {'t':>3} {'action':<12} {'REAL x':>9} {'REAL z':>9} {'DREAM x':>9} {'DREAM z':>9} "
                  f"{'err':>7}")
            for t in range(H + 1):
                an = action_name(int(acts[t]), "combat") if t < H else ""
                e = np.hypot(traj[t][PX] - w[t][PX], traj[t][PZ] - w[t][PZ])
                print(f"  {t:>3} {an:<12} {w[t][PX]:9.2f} {w[t][PZ]:9.2f} {traj[t][PX]:9.2f} "
                      f"{traj[t][PZ]:9.2f} {e:7.2f}")
            print()
    Path("runs/competencies.json").write_text(json.dumps(out, indent=1))
    print(f"\n  wrote runs/competencies.json")


if __name__ == "__main__":
    main()
