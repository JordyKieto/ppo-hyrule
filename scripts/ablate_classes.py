#!/usr/bin/env python3
"""OOCDM-style CLASS-LEVEL ablation: which input classes does each prediction head actually use?

The question this answers is not "is the model accurate" but "is the model reading the information the task
requires". HP has sat at exactly its persistence baseline in every run tonight -- if ablating the Baba's attack
timer and behaviour state does not change the HP head's loss at all, the model is not reading the lunge cycle,
which is the signal that determines when Link takes damage. That is a falsifiable explanation rather than
another hyperparameter guess.

Following OOCDM (object-oriented causal dynamics), influence is measured at the CLASS level, not per instance:
"does the class Baba-position influence the class Link-HP", not "does Baba #3 influence Link's HP". Instances
come and go, so per-instance edges cannot be estimated from finite data; classes can.

    I(C -> T) = E[ log p(T | all) - log p(T | ablate C) ]

Ablation is by SHUFFLING the class across the batch, not zeroing it. Zeroing moves the input off-distribution
and the resulting loss increase conflates "this input matters" with "the model has never seen zeros here".
Shuffling preserves the marginal distribution exactly and destroys only the correlation with the target, which
is what the conditional-mutual-information framing actually asks for.

    scripts/ablate_classes.py --ckpt runs/wm_v5_100/ck_0020.pt --packed /workspace/packed/v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_wm_tokens import CENTRE, DC, DW, LOGIT_K, N_STATES, TokenWM  # noqa: E402
from zelda_rl import wm_data  # noqa: E402
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.tokens import ACTOR_DIM, LINK_DIM, SF1_KEEP  # noqa: E402

NB = len(SF1_KEEP)
# Input CLASSES: (name, token field, column slice or None for the whole field).
# Grouped the way the game groups them, not the way the tensor is laid out -- "velocity" is one class even
# though it is three columns, because the question is whether the model uses velocity at all.
LINK_COLS = {"l_rel_y": (0, 1), "l_hp": (1, 3), "l_yaw": (3, 5), "l_speed": (5, 6),
             "l_vel": (6, 9), "l_cam": (9, 13), "l_floor": (13, 14), "l_counts": (14, 16),
             "l_sf1_bits": (16, 16 + NB), "l_bg_bits": (16 + NB, LINK_DIM)}
ACTOR_COLS = {"a_rel_y": (0, 1), "a_bearing": (1, 3), "a_hp": (3, 4), "a_yaw": (4, 8),
              "a_yaw_to_player": (8, 10), "a_timer": (10, 11), "a_vel": (11, ACTOR_DIM)}
CLASSES = ([(n, "link", c) for n, c in LINK_COLS.items()]
           + [(n, "actors", c) for n, c in ACTOR_COLS.items()]
           + [("link_position", "link_cell+link_off", None),
              ("actor_position", "actor_rcell+actor_off+cell_xy", None),
              ("actor_identity", "ids+cats", None),
              ("actor_behaviour_state", "states", None),
              ("actor_presence", "mask", None),
              ("ACTION", "action", None)])
HEADS = ("l_dcell", "l_doff", "l_y", "l_yaw", "l_hp", "l_bits", "l_scene",
         "a_dcell", "a_doff", "a_hp", "a_pres", "a_state")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--n", type=int, default=32768, help="frames to evaluate per ablation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the influence matrix as JSON")
    a = ap.parse_args()

    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    pm = json.load(open(Path(a.packed) / "meta.json"))
    N, NA = int(pm["n_frames"]), pm["n_actions"]
    grid = NavGrid.load(pm["grid"])
    net = TokenWM(NA, cfg["dim"], cfg["heads"], n_rows=a.rows).to(dev).eval()
    net.load_state_dict(ck["model"])
    print(f"{a.ckpt}: epoch {ck['epoch']}, dim {cfg['dim']} heads {cfg['heads']}, "
          f"recorded skill {ck['metrics'].get('skill', float('nan')):+.4f}")

    # held-out TEST slots -- never trained on, never used for model selection tonight
    test = set(pm["split"]["val_slots"])
    slot = np.asarray(np.load(Path(a.packed) / "slot.npy", mmap_mode="r")[:N])
    ok = np.asarray(np.load(Path(a.packed) / "ok.npy", mmap_mode="r")[:N])
    pool = np.flatnonzero(ok & np.isin(slot, list(test)))
    sel = np.sort(pool[rng.choice(len(pool), min(a.n, len(pool)), replace=False)])
    print(f"{len(sel):,} frames from held-out TEST slots {sorted(test)}")

    NEED = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
            "actor_rcell", "actor_off", "cell_xy", "act", "dcell", "doff", "dy", "dyaw", "hp_ev",
            "bits_now", "bits_nxt", "scene_now", "scene_nxt", "r_dcell", "r_doff", "r_hp_ev",
            "pres_nxt", "st_nxt")
    B = {k: torch.from_numpy(np.ascontiguousarray(np.load(Path(a.packed) / f"{k}.npy", mmap_mode="r")[sel])).to(dev)
         for k in NEED}
    LONG = ("ids", "cats", "states", "link_cell", "actor_rcell")
    YL = ("dcell", "hp_ev", "scene_now", "scene_nxt", "r_dcell", "r_hp_ev", "st_nxt")

    def losses(T, act):
        """The same twelve typed heads train_wm_tokens.py trains, with the same logit residuals."""
        n = T["link"].shape[0]
        ol, oa = net(T, F.one_hot(act.long(), NA).float())
        j = 0
        dcl, j = ol[:, j:j + DC], j + DC
        dof, j = ol[:, j:j + 2], j + 2
        dyv, j = ol[:, j:j + 1], j + 1
        dyw, j = ol[:, j:j + 2], j + 2
        hp, j = ol[:, j:j + 3], j + 3
        bits, j = ol[:, j:j + NB], j + NB
        scn = ol[:, j:j + 2]
        dcl = dcl + LOGIT_K * F.one_hot(torch.full((n,), CENTRE, device=dev), DC).float()
        bits = bits + LOGIT_K * (2 * B["bits_now"].float() - 1)
        scn = scn + LOGIT_K * F.one_hot(B["scene_now"].long(), 2).float()
        hp = hp + LOGIT_K * F.one_hot(torch.zeros(n, dtype=torch.long, device=dev), 3).float()
        j = 0
        rdcl, j = oa[..., j:j + DC], j + DC
        rdof, j = oa[..., j:j + 2], j + 2
        rhp, j = oa[..., j:j + 3], j + 3
        rpr, j = oa[..., j:j + 1], j + 1
        rst = oa[..., j:j + N_STATES]
        rhp = rhp + LOGIT_K * F.one_hot(torch.zeros(oa.shape[:2], dtype=torch.long, device=dev), 3).float()
        rpr = rpr + LOGIT_K * (2 * B["mask"].float()[..., None] - 1)
        rst = rst + LOGIT_K * F.one_hot(B["states"].long(), N_STATES).float()
        rdcl = rdcl + LOGIT_K * F.one_hot(torch.full(oa.shape[:2], CENTRE, device=dev), DC).float()
        m = B["mask"].float(); ms = m.sum().clamp(min=1)
        return {
            "l_dcell": F.cross_entropy(dcl, B["dcell"].long()),
            "l_doff": F.l1_loss(dof, B["doff"].float()),
            "l_y": F.l1_loss(dyv, B["dy"].float()),
            "l_yaw": F.l1_loss(dyw, B["dyaw"].float()),
            "l_hp": F.cross_entropy(hp, B["hp_ev"].long()),
            "l_bits": F.binary_cross_entropy_with_logits(bits, B["bits_nxt"].float()),
            "l_scene": F.cross_entropy(scn, B["scene_nxt"].long()),
            "a_dcell": ((F.cross_entropy(rdcl.reshape(-1, DC), B["r_dcell"].long().reshape(-1),
                                         reduction="none").reshape(m.shape) * m).sum() / ms),
            "a_doff": ((rdof - B["r_doff"].float()).abs().mean(-1) * m).sum() / ms,
            "a_hp": ((F.cross_entropy(rhp.reshape(-1, 3), B["r_hp_ev"].long().reshape(-1),
                                      reduction="none").reshape(m.shape) * m).sum() / ms),
            "a_pres": F.binary_cross_entropy_with_logits(rpr.squeeze(-1), B["pres_nxt"].float()),
            "a_state": ((F.cross_entropy(rst.reshape(-1, N_STATES), B["st_nxt"].long().reshape(-1),
                                         reduction="none").reshape(m.shape) * m).sum() / ms),
        }

    def tokens(mod=None):
        TK = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
              "actor_rcell", "actor_off", "cell_xy")
        o = {k: (B[k].long() if k in LONG else B[k].float()).clone() for k in TK}
        o["rel_xz"] = o["actors"][:, :, :2]
        act = B["act"].clone()
        if mod is not None:
            act = mod(o, act)
        o["rel_xz"] = o["actors"][:, :, :2]          # re-derive AFTER any ablation of `actors`
        return o, act

    n = len(sel)
    with torch.no_grad():
        base_T, base_a = tokens()
        base = {k: float(v) for k, v in losses(base_T, base_a).items()}
    print("\nbaseline per-head loss on held-out test:")
    print("  " + "  ".join(f"{k} {base[k]:.4f}" for k in HEADS))

    def shuffler(field, cols):
        """Ablate a class by PERMUTING it across the batch: marginal preserved, correlation destroyed."""
        def mod(o, act):
            p = torch.from_numpy(rng.permutation(n)).to(dev)
            if field == "action":
                return act[p]
            for f in field.split("+"):
                if cols is None:
                    o[f] = o[f][p]
                elif o[f].dim() == 2:                       # link: (B, LINK_DIM)
                    o[f][:, cols[0]:cols[1]] = o[f][p][:, cols[0]:cols[1]]
                else:                                       # actors: (B, R, ACTOR_DIM) -- slice the FEATURE dim
                    o[f][:, :, cols[0]:cols[1]] = o[f][p][:, :, cols[0]:cols[1]]
            return act
        return mod

    rowsout = {}
    with torch.no_grad():
        for name, field, cols in CLASSES:
            T, act = tokens(shuffler(field, cols))
            L = {k: float(v) for k, v in losses(T, act).items()}
            rowsout[name] = {k: L[k] - base[k] for k in HEADS}

    w = max(len(c[0]) for c in CLASSES) + 1
    print(f"\nINFLUENCE  I(class -> head) = loss increase when the class is shuffled.")
    print(f"{'class':<{w}}" + "".join(f"{h:>10}" for h in HEADS))
    print("-" * (w + 10 * len(HEADS)))
    for name, _, _ in CLASSES:
        r = rowsout[name]
        cells = "".join(f"{r[h]:>+10.4f}" if abs(r[h]) >= 5e-4 else f"{'.':>10}" for h in HEADS)
        print(f"{name:<{w}}" + cells)
    print("-" * (w + 10 * len(HEADS)))
    print(f"{'(baseline)':<{w}}" + "".join(f"{base[h]:>10.4f}" for h in HEADS))
    print("\n'.' = |delta| < 0.0005, i.e. the head does not use that class at all.")
    if a.out:
        json.dump({"baseline": base, "influence": rowsout, "ckpt": a.ckpt, "n": int(n)},
                  open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
