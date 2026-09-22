"""Flat MLP + nav-grid positions + reconstruction + JEPA prediction. Trained from scratch.

    .venv/bin/python scripts/train_jepa_nav.py

Three changes from `train_jepa_pure.py --flat`, all measured against it:

1. POSITION IS NO LONGER A RAW COORDINATE, in OR out. `zelda_rl.navgrid.NavGrid` splits world XZ into a
   classified CELL plus a bounded within-cell OFFSET, losslessly (round trip 6e-5 world units, every offset
   inside [-0.5, 0.5], nothing clipped). Input: per-axis one-hots + offsets, for Link and all 16 actor rows.
   Output: cell logits (cross-entropy) + offset regression, so the multi-modal "where in the room" part is a
   classification and only the tiny sub-cell remainder is a mean.

   This is the same move that worked every other time it was made here -- Link HP regressed gave damage F1
   0.286 and classified gave 0.906; displacement magnitude regressed came out 0.11x of true and binned 1.04x.
   At n=64 the cell is 14.5 x 22.3 world units and cell-only error is 7.90, already below the 9-11 the
   regressed head achieves, before the offset is added.

2. RECONSTRUCTION PRESSURE IS BACK. `zelda_rl.reward_decode.RewardTargets` supplies every reward field except
   position (`skip=("link_pos","row_pos")`), and the grid heads own position. Pure prediction alone was
   measured WORSE than a random encoder at holding the state -- link_hp 0.786 vs 0.862, link_pos 35.1 vs 29.0
   world units -- so reconstruction is not optional if the reward has to be readable.

3. NO ENTITY ENCODER. A plain MLP over the flattened observation, bottlenecked width -> width -> width/2 ->
   latent. The entity encoder's weight sharing was never measured to help: on the flat baseline held-out
   prediction error IMPROVED over training (0.0118 -> 0.0083) where the entity encoder's doubled.

Kept from JEPA: EMA target encoder with a stop-gradient (I-JEPA App. A), L1 on layer-normalised targets,
teacher forcing plus a rollout term, stride 1 to match deployment. Collapse is printed every epoch, because
the flat run showed a slow contraction -- effective rank 9.5 -> 6.7 and move/rand 0.045 -> 0.024 over 20
epochs -- that made the identity baseline stronger faster than the model improved.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import DEFAULT_DATA, load_many  # noqa: E402
from zelda_rl.bc import (LINK_FEAT_NAMES, LINK_HP_VALUES, N_BABA_STATES, ROW_FEAT_NAMES,  # noqa: E402
                         ROW_HP_VALUES, LI, RI, Norm, _split, featurise)
from zelda_rl.dynamics import build_predictor  # noqa: E402
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.reward_decode import RewardTargets  # noqa: E402

SKIP_POS = ("link_pos", "row_pos")


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


class FlatEncoder(nn.Module):
    """Plain MLP over the flattened observation, bottlenecked down to the latent."""

    def __init__(self, in_dim: int, latent: int = 128, width: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, width // 2), nn.ReLU(),
                                 nn.Linear(width // 2, latent))
        self.ln = nn.LayerNorm(latent)

    def forward(self, x):
        return self.ln(self.net(x))


class NavDecoder(nn.Module):
    """One trunk, one output layer, sliced -- same design as `UnifiedDecoder`, with grid position heads.

    Position slices are: link cell-x logits (n), cell-z logits (n), offset (2), y (1); and the same per row.
    Everything else keeps the widths `reward_decode` expects, so the shared loss owns those fields unchanged.
    """

    def __init__(self, latent: int, n: int, n_rows: int = 16, hidden: int = 1024):
        super().__init__()
        self.n, self.n_rows = n, n_rows
        self.gspec = [("link_hp", len(LINK_HP_VALUES)), ("link_yaw", 2), ("link_bits", 3),
                      ("targeted", n_rows + 1), ("scene", 2), ("room", 8), ("baba_count", 4),
                      ("near_dist", 1), ("link_cx", n), ("link_cz", n), ("link_off", 2), ("link_y", 1)]
        self.rspec = [("row_present", 1), ("row_is_baba", 1), ("row_hp", len(ROW_HP_VALUES)),
                      ("row_state", N_BABA_STATES), ("link_facing_target", 2), ("row_dist", 1),
                      ("row_cx", n), ("row_cz", n), ("row_off", 2), ("row_y", 1)]
        self.gw = sum(w for _, w in self.gspec)
        self.rw = sum(w for _, w in self.rspec)
        self.trunk = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, self.gw + n_rows * self.rw)

    def forward(self, z):
        h = self.out(self.trunk(z))
        g, r = h[:, :self.gw], h[:, self.gw:].reshape(-1, self.n_rows, self.rw)
        o, i = {}, 0
        for name, w in self.gspec:
            v = g[:, i:i + w]
            o[name] = torch.tanh(v) if name == "link_yaw" else v
            i += w
        j = 0
        for name, w in self.rspec:
            v = r[..., j:j + w]
            if name == "link_facing_target":
                v = torch.tanh(v)
            o[name] = v.squeeze(-1) if w == 1 and name in ("row_present", "row_is_baba") else v
            j += w
        return o


def build_inputs(state, grid: NavGrid):
    """featurise(), with absolute pos_x/pos_z replaced by nav-grid encodings for Link and every row."""
    Fr = featurise(state, link_v2=True, control_echo=False)
    lk, rows, _ = _split(state)
    lx = [i for i, nm in enumerate(LINK_FEAT_NAMES) if nm in ("pos_x", "pos_z")]
    rx = [i for i, nm in enumerate(ROW_FEAT_NAMES) if nm in ("pos_x", "pos_z")]
    assert len(lx) == 2 and len(rx) == 2, (lx, rx)
    link_keep = np.delete(Fr["link"], lx, axis=1)
    rows_keep = np.delete(Fr["rows"], rx, axis=2)
    link_g = grid.onehot(lk[:, [LI["pos_x"], LI["pos_z"]]])
    n, R = len(state), rows.shape[1]
    rows_g = grid.onehot(rows[:, :, [RI["pos_x"], RI["pos_z"]]].reshape(-1, 2)).reshape(n, R, grid.feat_dim)
    rows_g = rows_g * (rows[:, :, [RI["present"]]] > 0.5)           # absent rows carry no stale geometry
    return {"link": np.concatenate([link_keep, link_g], 1).astype(np.float32),
            "rows": np.concatenate([rows_keep, rows_g], 2).astype(np.float32),
            "ids": Fr["ids"], "mask": Fr["mask"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt", help="slot split only")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--pred-width", type=int, default=1024)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--unroll-decay", type=float, default=0.97)
    ap.add_argument("--unroll-batch", type=int, default=128)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-off", type=float, default=1.0, help="within-cell offset regression weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/jepa/nav.pt")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    split = torch.load(a.ref, weights_only=False, map_location="cpu")["split"]
    meta, state, act, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    NA = meta["n_actions"]
    lk, rows, _ = _split(state)
    pres_np = rows[:, :, RI["present"]] > 0.5
    allxz = np.concatenate([lk[:, [LI["pos_x"], LI["pos_z"]]],
                            rows[:, :, [RI["pos_x"], RI["pos_z"]]][pres_np]])
    grid = NavGrid.fit(allxz, n=a.grid)
    print(f"nav grid {a.grid}x{a.grid} | cell {np.round(grid.cell,1)} world units | "
          f"cell-only error {grid.quantisation_error(allxz):.2f} | {grid.feat_dim} dims/entity")

    Fraw = build_inputs(state, grid)
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_m = ~np.isin(slot, list(test | tune))
    norm = Norm.fit({k: v[tr_m] for k, v in Fraw.items()})
    T = {k: torch.from_numpy(v) for k, v in norm.apply(Fraw).items()}
    X = torch.cat([T["link"], T["rows"].flatten(1)], 1)           # flat: no entity pooling
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1), NA).float()

    # grid targets
    li, lo_ = grid.to_cell(lk[:, [LI["pos_x"], LI["pos_z"]]])
    ri, ro_ = grid.to_cell(rows[:, :, [RI["pos_x"], RI["pos_z"]]].reshape(-1, 2))
    R = rows.shape[1]
    G = {"link_c": torch.from_numpy(li), "link_off": torch.from_numpy(lo_),
         "link_y": torch.from_numpy(lk[:, [LI["pos_y"]]].astype(np.float32) / 100.0),
         "row_c": torch.from_numpy(ri.reshape(len(state), R, 2)),
         "row_off": torch.from_numpy(ro_.reshape(len(state), R, 2)),
         "row_y": torch.from_numpy(rows[:, :, [RI["pos_y"]]].astype(np.float32) / 100.0)}
    RT = RewardTargets(state, obs, epid)

    same = np.zeros(len(state), bool)
    same[:-1] = epid[1:] == epid[:-1]

    def idx(mask, need):
        return np.flatnonzero(mask & same & np.r_[np.ones(len(state) - need, bool), np.zeros(need, bool)])

    tr, tu, te = idx(tr_m, 1), idx(np.isin(slot, list(tune)), 1), idx(np.isin(slot, list(test)), 1)
    deep = idx(tr_m, a.unroll + 1)
    enc = FlatEncoder(X.shape[1], a.latent, a.width)
    tgt = copy.deepcopy(enc)
    for q in tgt.parameters():
        q.requires_grad_(False)
    dec = NavDecoder(a.latent, a.grid, R, a.width)
    pred = build_predictor(a.latent, NA, hidden=a.pred_width)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(pred.parameters()), lr=a.lr)
    ps = [sum(q.numel() for q in m.parameters()) for m in (enc, dec, pred)]
    print(f"{len(state):,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | input {X.shape[1]} "
          f"| enc {ps[0]/1e6:.2f}M dec {ps[1]/1e6:.2f}M pred {ps[2]/1e6:.2f}M = {sum(ps)/1e6:.2f}M")

    def pos_loss(p, i):
        m = RT.Y["row_present"][i]
        L = (F.cross_entropy(p["link_cx"], G["link_c"][i][:, 0])
             + F.cross_entropy(p["link_cz"], G["link_c"][i][:, 1])
             + a.w_off * F.mse_loss(p["link_off"], G["link_off"][i])
             + F.mse_loss(p["link_y"], G["link_y"][i]))
        cx = F.cross_entropy(p["row_cx"].reshape(-1, a.grid), G["row_c"][i][..., 0].reshape(-1),
                             reduction="none").reshape(m.shape)
        cz = F.cross_entropy(p["row_cz"].reshape(-1, a.grid), G["row_c"][i][..., 1].reshape(-1),
                             reduction="none").reshape(m.shape)
        L = L + ((cx + cz) * m).sum() / m.sum().clamp(min=1)
        L = L + a.w_off * (((p["row_off"] - G["row_off"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        L = L + (((p["row_y"] - G["row_y"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        return L

    def recon(p, i):
        return RT.loss(p, i, skip=SKIP_POS) + pos_loss(p, i)

    def world_err(p, i):
        """Decoded link position error in WORLD units, so it is comparable to the regressed 9-11."""
        cx, cz = p["link_cx"].argmax(-1).numpy(), p["link_cz"].argmax(-1).numpy()
        xz = grid.from_cell(np.stack([cx, cz], 1), p["link_off"].numpy())
        true = lk[i.numpy()][:, [LI["pos_x"], LI["pos_z"]]]
        return float(np.linalg.norm(xz - true, axis=1).mean())

    hist = []
    for ep in range(a.epochs):
        enc.train(); dec.train(); pred.train()
        perm = rng.permutation(tr)
        t0, tl, nb = time.time(), 0.0, 0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = enc(X[i])
            with torch.no_grad():
                zt = tgt(X[i + 1])
            loss = a.w_recon * recon(dec(z), i) + a.w_pred * F.l1_loss(
                ln(pred(torch.cat([z, A[i]], -1))), ln(zt))
            if a.unroll > 1:
                zi = torch.from_numpy(rng.choice(deep, min(a.unroll_batch, len(deep)), replace=False))
                zz, rl, ws = enc(X[zi]), torch.zeros(()), 0.0
                for j in range(a.unroll):
                    zz = pred(torch.cat([zz, A[zi + j]], -1))
                    with torch.no_grad():
                        zn = tgt(X[zi + j + 1])
                    w = a.unroll_decay ** j
                    rl = rl + w * (F.l1_loss(ln(zz), ln(zn)) + a.w_recon * recon(dec(zz), zi + j + 1))
                    ws += w
                loss = loss + a.w_pred * rl / max(ws, 1e-9)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters())
                                     + list(pred.parameters()), 1.0)
            opt.step()
            with torch.no_grad():
                for qt, qs in zip(tgt.parameters(), enc.parameters()):
                    qt.mul_(a.ema).add_(qs, alpha=1 - a.ema)
            tl += float(loss.detach()); nb += 1
        enc.eval(); dec.eval(); pred.eval()
        with torch.no_grad():
            iu = torch.from_numpy(tu[:20000])
            z0, z1 = enc(X[iu]), enc(X[iu + 1])
            pm = float(F.l1_loss(ln(pred(torch.cat([z0, A[iu]], -1))), ln(z1)))
            im = float(F.l1_loss(ln(z0), ln(z1)))
            we = world_err(dec(z0), iu)
            zc = z0 - z0.mean(0, keepdim=True)
            evs = torch.linalg.eigvalsh(((zc.T @ zc) / len(zc)).double()).clamp(min=1e-12)
            pp = evs / evs.sum()
            er = float(torch.exp(-(pp * pp.log()).sum()))
            mr = float((z1 - z0).abs().mean()) / max(float((z0 - z0[torch.randperm(len(z0))]).abs().mean()), 1e-9)
        hist.append({"epoch": ep + 1, "train": tl / nb, "tune": pm, "identity": im, "pos_world": we,
                     "erank": er, "move_ratio": mr})
        print(f"  ep {ep+1:>3}/{a.epochs} train {tl/nb:.4f} | pred {pm:.5f} id {im:.5f} skill "
              f"{1-pm/max(im,1e-9):+.3f} | link_pos {we:>6.2f} world units | erank {er:.1f} "
              f"move/rand {mr:.3f}  ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        it = torch.from_numpy(te)
        p = dec(enc(X[it]))
        print(f"\n=== TEST SLOTS ===\n  link_pos {world_err(p, it):.2f} world units "
              f"(regressed baseline 9-11; cell-only floor {grid.quantisation_error(allxz):.2f})")
        for name, v, _, kind in RT.report(p, None, it):
            if name not in SKIP_POS:
                print(f"  {name:<20} {v:.4f} ({'acc' if kind == 'acc' else 'MAE world units'})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "pred": pred.state_dict(),
                "pred_residual": True, "norm": norm.state_dict(), "grid": grid.state_dict(),
                "split": split, "latent": a.latent, "n_actions": NA, "in_dim": X.shape[1],
                "config": vars(a), "history": hist}, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
