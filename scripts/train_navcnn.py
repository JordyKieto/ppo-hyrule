"""Deterministic actor nav grid -> CNN encoder -> latent prediction -> CNN decoder. Trained from scratch.

    .venv/bin/python scripts/train_navcnn.py

The grid is NOT learned. `zelda_rl.navgrid` turns world XZ into a fixed n x n grid with one channel per actor
TYPE (Link, Deku Baba, Baba HP, plus a static wall-contact map), deterministically and losslessly given the
within-cell offsets. Only the CNN over it is trained.

    grid (4 x n x n)  --CNN--\
                              fuse -> z_t --M(z_t,a_t)--> z_hat        L1 on layer-normalised targets
    flat non-spatial features-/                                        EMA target encoder, stop-gradient
                                   z_t --CNN decoder--> grid channels  + MLP heads for the reward's fields

WHY A GRID AND A CONVOLUTION. Raw XZ is a bad input: unbounded, no natural scale, and as an MSE target it is
minimised by the conditional mean, which for "kept walking vs hit the wall" is a point the game cannot occupy.
On a grid, position becomes a location in an image, and convolution is translation-equivariant -- "Link moved
one cell east" is a SHIFT, not an unrelated change in two floats. Every previous move from regression to
classification in this project produced a large gain (Link HP damage F1 0.286 -> 0.906; displacement magnitude
0.11x of true regressed -> 1.04x binned), and position was the last big regressed field.

WHY TYPES, NOT SLOTS. A convolution cannot carry "actor 7" through a pooling stack, and it does not need to:
the reward reads per-row HP and behaviour state, so those travel alongside as a flat vector. The grid carries
WHERE, the flat vector carries WHAT. Nothing is lost and the spatial part gets the right inductive bias.

WALLS. The dataset has no collision mesh, but every frame records whether Link is touching a wall, so
accumulating those contacts over all 203k frames traces the walkable boundary (120 cells against 447 Link ever
visits, at n=32). Static, so it is an input the model never has to predict.

Reconstruction is back in the loss, unlike `train_jepa_pure.py`: pure prediction was measured WORSE than a
random encoder at holding the state (link_hp 0.786 vs 0.862, link_pos 35.1 vs 29.0 world units), so it is not
optional if the reward must stay readable.
"""
from __future__ import annotations

import argparse
import copy
import json
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
                         ROW_HP_VALUES, BABA_ID, LI, RI, Norm, _split, featurise)
from zelda_rl.combat import BG_WALL  # noqa: E402
from zelda_rl.dynamics import build_predictor  # noqa: E402
from zelda_rl.navgrid import CH_BABA, CH_BABA_HP, CH_LINK, CH_WALL, N_CH, NavGrid, wall_map  # noqa: E402
from zelda_rl.reward_decode import RewardTargets  # noqa: E402

SKIP_POS = ("link_pos", "row_pos")


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


class GridEncoder(nn.Module):
    """CNN over the type-channel grid, fused with the flat non-spatial features."""

    def __init__(self, n: int, flat_dim: int, latent: int = 128, ch: int = 32, width: int = 512):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(N_CH, ch, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.ReLU())
        self.sp = (n // 8) ** 2 * ch * 4
        self.flat = nn.Sequential(nn.Linear(flat_dim, width), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(self.sp + width, width), nn.ReLU(), nn.Linear(width, latent))
        self.ln = nn.LayerNorm(latent)

    def forward(self, g, f):
        return self.ln(self.fuse(torch.cat([self.conv(g).flatten(1), self.flat(f)], -1)))


class GridDecoder(nn.Module):
    """latent -> grid channels (transposed convolution) + MLP heads for the reward's non-spatial fields."""

    def __init__(self, latent: int, n: int, n_rows: int = 16, ch: int = 32, width: int = 512):
        super().__init__()
        self.n, self.n_rows, self.ch = n, n_rows, ch
        self.s = n // 8
        self.to_sp = nn.Linear(latent, ch * 4 * self.s * self.s)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(ch, N_CH, 4, 2, 1))
        self.trunk = nn.Sequential(nn.Linear(latent, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.gspec = [("link_hp", len(LINK_HP_VALUES)), ("link_yaw", 2), ("link_bits", 3),
                      ("targeted", n_rows + 1), ("scene", 2), ("room", 8), ("baba_count", 4),
                      ("near_dist", 1), ("link_off", 2), ("link_y", 1)]
        self.rspec = [("row_present", 1), ("row_is_baba", 1), ("row_hp", len(ROW_HP_VALUES)),
                      ("row_state", N_BABA_STATES), ("link_facing_target", 2), ("row_dist", 1)]
        self.gw, self.rw = sum(w for _, w in self.gspec), sum(w for _, w in self.rspec)
        self.head = nn.Linear(width, self.gw + n_rows * self.rw)

    def forward(self, z):
        grid = self.deconv(self.to_sp(z).view(-1, self.ch * 4, self.s, self.s))
        h = self.head(self.trunk(z))
        g, r = h[:, :self.gw], h[:, self.gw:].reshape(-1, self.n_rows, self.rw)
        o, i = {"grid": grid}, 0
        for name, w in self.gspec:
            v = g[:, i:i + w]
            o[name] = torch.tanh(v) if name in ("link_yaw", "link_off") else v
            i += w
        j = 0
        for name, w in self.rspec:
            v = r[..., j:j + w]
            if name == "link_facing_target":
                v = torch.tanh(v)
            o[name] = v.squeeze(-1) if w == 1 and name in ("row_present", "row_is_baba") else v
            j += w
        return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--pred-width", type=int, default=1024)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-grid", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/jepa/navcnn.pt")
    ap.add_argument("--ckpt-dir", default=None,
                    help="save a checkpoint every --ckpt-every epochs here, plus best.pt (lowest tune pred) "
                         "and last.pt, plus metrics.jsonl appended live. Without this only the final model is "
                         "written -- which meant every rollout number so far came from the LAST epoch, and the "
                         "last epoch was repeatedly the worst held-out one.")
    ap.add_argument("--ckpt-every", type=int, default=1, help="in epochs")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    split = torch.load(a.ref, weights_only=False, map_location="cpu")["split"]
    meta, state, act, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    NA, n = meta["n_actions"], a.grid
    lk, rows, _ = _split(state)
    R = rows.shape[1]
    pres = rows[:, :, RI["present"]] > 0.5
    link_xz = lk[:, [LI["pos_x"], LI["pos_z"]]]
    row_xz = rows[:, :, [RI["pos_x"], RI["pos_z"]]]
    grid = NavGrid.fit(np.concatenate([link_xz, row_xz[pres]]), n=n)
    wall = torch.from_numpy(wall_map(grid, link_xz, ((lk[:, LI["bg_flags"]].astype(np.int64) >> BG_WALL) & 1) > 0))
    print(f"nav grid {n}x{n} | cell {np.round(grid.cell,1)} world units | {N_CH} type channels "
          f"| wall cells {int(wall.sum())}")

    li, loff = grid.to_cell(link_xz)
    ri, roff = grid.to_cell(row_xz.reshape(-1, 2))
    ri, roff = ri.reshape(len(state), R, 2), roff.reshape(len(state), R, 2)
    is_baba = (rows[:, :, RI["id"]].astype(np.int64) == BABA_ID) & pres
    LC = torch.from_numpy(li)                                  # (N,2) x,z cell
    RC = torch.from_numpy(ri)
    BB = torch.from_numpy(is_baba.astype(np.float32))
    RHP = torch.from_numpy((rows[:, :, RI["hp"]] / 2.0).astype(np.float32))
    LOFF = torch.from_numpy(loff)

    # flat features: everything featurise() gives EXCEPT absolute pos, plus the within-cell offsets
    Fr = featurise(state, link_v2=True, control_echo=False)
    lx = [i for i, nm in enumerate(LINK_FEAT_NAMES) if nm in ("pos_x", "pos_z")]
    rx = [i for i, nm in enumerate(ROW_FEAT_NAMES) if nm in ("pos_x", "pos_z")]
    Fr["link"] = np.concatenate([np.delete(Fr["link"], lx, 1), loff], 1).astype(np.float32)
    Fr["rows"] = np.concatenate([np.delete(Fr["rows"], rx, 2), roff.astype(np.float32)], 2).astype(np.float32)
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_m = ~np.isin(slot, list(test | tune))
    norm = Norm.fit({k: v[tr_m] for k, v in Fr.items()})
    Tn = norm.apply(Fr)
    FLAT = torch.cat([torch.from_numpy(Tn["link"]), torch.from_numpy(Tn["rows"]).flatten(1)], 1)
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1), NA).float()

    RFLAT = RC[..., 1] * n + RC[..., 0]                         # (frames, rows) flat cell index, precomputed
    LFLAT = LC[:, 1] * n + LC[:, 0]

    def make_grid(i):
        """Build the (B, 4, n, n) grid from precomputed cell indices. Deterministic, no learned part.

        Fully vectorised: the obvious per-row Python loop called 16 scatters per channel per batch and, at ~5
        grid builds per training step, dominated the whole step -- more than the convolutions it feeds.
        `scatter_reduce_` with amax also handles two actors sharing a cell correctly, where a plain scatter
        would let an absent row's zero overwrite a present one.
        """
        B = len(i)
        g = torch.zeros(B, N_CH, n * n)
        g[torch.arange(B), CH_LINK, LFLAT[i]] = 1.0
        rf, bb = RFLAT[i], BB[i]
        g[:, CH_BABA].scatter_reduce_(1, rf, bb, reduce="amax")
        g[:, CH_BABA_HP].scatter_reduce_(1, rf, RHP[i] * bb, reduce="amax")
        g = g.view(B, N_CH, n, n)
        g[:, CH_WALL] = wall
        return g

    RT = RewardTargets(state, obs, epid)
    same = np.zeros(len(state), bool)
    same[:-1] = epid[1:] == epid[:-1]

    def idx(mask, need):
        return np.flatnonzero(mask & same & np.r_[np.ones(len(state) - need, bool), np.zeros(need, bool)])

    tr, tu, te = idx(tr_m, 1), idx(np.isin(slot, list(tune)), 1), idx(np.isin(slot, list(test)), 1)
    deep = idx(tr_m, a.unroll + 1)
    enc = GridEncoder(n, FLAT.shape[1], a.latent, a.ch, a.width)
    tgt = copy.deepcopy(enc)
    for q in tgt.parameters():
        q.requires_grad_(False)
    dec = GridDecoder(a.latent, n, R, a.ch, a.width)
    pred = build_predictor(a.latent, NA, hidden=a.pred_width)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(pred.parameters()), lr=a.lr)
    ps = [sum(q.numel() for q in m.parameters()) for m in (enc, dec, pred)]
    print(f"{len(state):,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | flat {FLAT.shape[1]} "
          f"| enc {ps[0]/1e6:.2f}M dec {ps[1]/1e6:.2f}M pred {ps[2]/1e6:.2f}M = {sum(ps)/1e6:.2f}M")

    def grid_loss(p, i):
        """Link cell as cross-entropy over the whole grid (it is ONE location); Baba occupancy as BCE; Baba HP
        masked to occupied cells. The wall channel is static, so it is not a target."""
        gt = make_grid(i)
        gl = p["grid"]
        lce = F.cross_entropy(gl[:, CH_LINK].flatten(1), gt[:, CH_LINK].flatten(1).argmax(-1))
        bce = F.binary_cross_entropy_with_logits(gl[:, CH_BABA], gt[:, CH_BABA])
        occ = gt[:, CH_BABA]
        hp = ((torch.sigmoid(gl[:, CH_BABA_HP]) - gt[:, CH_BABA_HP]) ** 2 * occ).sum() / occ.sum().clamp(min=1)
        return lce + bce + hp

    def recon(p, i):
        return (RT.loss(p, i, skip=SKIP_POS) + a.w_grid * grid_loss(p, i)
                + F.mse_loss(p["link_off"], LOFF[i])
                + F.mse_loss(p["link_y"], torch.from_numpy(lk[i.numpy()][:, [LI["pos_y"]]] / 100.0)))

    def world_err(p, i):
        """Decoded Link position in WORLD units: argmax cell of the link channel plus the offset head."""
        flat = p["grid"][:, CH_LINK].flatten(1).argmax(-1).numpy()
        cz, cx = flat // n, flat % n
        xz = grid.from_cell(np.stack([cx, cz], 1), p["link_off"].numpy())
        return float(np.linalg.norm(xz - link_xz[i.numpy()], axis=1).mean())

    cdir = Path(a.ckpt_dir) if a.ckpt_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    meta_keys = {"norm": norm.state_dict(), "grid": grid.state_dict(), "wall": wall.numpy(),
                 "split": split, "latent": a.latent, "n_actions": NA, "config": vars(a)}

    def save(path, ep, m, full=False):
        d = {"encoder": enc.state_dict(), "decoder": dec.state_dict(), "pred": pred.state_dict(),
             "pred_residual": True, "epoch": ep, "metrics": m, **meta_keys}
        if full:
            d["target"] = tgt.state_dict()          # only the resumable ones carry the EMA copy
            d["optimizer"] = opt.state_dict()
        torch.save(d, path)

    start_ep, best = 0, float("inf")
    if a.resume:
        rk = torch.load(a.resume, weights_only=False, map_location="cpu")
        enc.load_state_dict(rk["encoder"]); dec.load_state_dict(rk["decoder"]); pred.load_state_dict(rk["pred"])
        if "target" in rk:
            tgt.load_state_dict(rk["target"])
        if "optimizer" in rk:
            opt.load_state_dict(rk["optimizer"])
        start_ep = int(rk.get("epoch", 0))
        print(f"resumed from {a.resume} at epoch {start_ep}")

    hist = []
    for ep in range(start_ep, a.epochs):
        enc.train(); dec.train(); pred.train()
        perm = rng.permutation(tr)
        t0, tl, nb = time.time(), 0.0, 0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = enc(make_grid(i), FLAT[i])
            with torch.no_grad():
                zt = tgt(make_grid(i + 1), FLAT[i + 1])
            loss = (a.w_recon * recon(dec(z), i)
                    + a.w_pred * F.l1_loss(ln(pred(torch.cat([z, A[i]], -1))), ln(zt)))
            if a.unroll > 1:
                zi = torch.from_numpy(rng.choice(deep, min(256, len(deep)), replace=False))
                zz = enc(make_grid(zi), FLAT[zi])
                for j in range(a.unroll):
                    zz = pred(torch.cat([zz, A[zi + j]], -1))
                    with torch.no_grad():
                        zn = tgt(make_grid(zi + j + 1), FLAT[zi + j + 1])
                    loss = loss + a.w_pred * (0.97 ** j) * F.l1_loss(ln(zz), ln(zn)) / a.unroll
                loss = loss + a.w_recon * recon(dec(zz), zi + a.unroll) / a.unroll
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
            iu = torch.from_numpy(tu[:8192])
            z0 = enc(make_grid(iu), FLAT[iu])
            z1 = enc(make_grid(iu + 1), FLAT[iu + 1])
            pm = float(F.l1_loss(ln(pred(torch.cat([z0, A[iu]], -1))), ln(z1)))
            im = float(F.l1_loss(ln(z0), ln(z1)))
            we = world_err(dec(z0), iu)
            zc = z0 - z0.mean(0, keepdim=True)
            evs = torch.linalg.eigvalsh(((zc.T @ zc) / len(zc)).double()).clamp(min=1e-12)
            pp = evs / evs.sum()
            er = float(torch.exp(-(pp * pp.log()).sum()))
            mr = float((z1 - z0).abs().mean()) / max(float((z0 - z0[torch.randperm(len(z0))]).abs().mean()), 1e-9)
        row = {"epoch": ep + 1, "train": tl / nb, "tune": pm, "identity": im,
               "skill": 1 - pm / max(im, 1e-9), "pos_world": we, "erank": er, "move_ratio": mr,
               "secs": round(time.time() - t0)}
        hist.append(row)
        if cdir:
            with open(cdir / "metrics.jsonl", "a") as fh:      # appended live, so the curve is watchable
                fh.write(json.dumps(row) + "\n")
            save(cdir / "last.pt", ep + 1, row, full=True)
            if (ep + 1) % a.ckpt_every == 0:
                save(cdir / f"ck_{ep+1:04d}.pt", ep + 1, row)
            if pm < best:
                best = pm
                save(cdir / "best.pt", ep + 1, row, full=True)
        print(f"  ep {ep+1:>3}/{a.epochs} train {tl/nb:.4f} | pred {pm:.5f} id {im:.5f} "
              f"skill {1-pm/max(im,1e-9):+.3f} | link_pos {we:>6.2f} world | erank {er:.1f} "
              f"move/rand {mr:.3f}  ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        it = torch.from_numpy(te[:8192])
        p = dec(enc(make_grid(it), FLAT[it]))
        print(f"\n=== TEST ===\n  link_pos {world_err(p, it):.2f} world units "
              f"(regressed baseline 9-11; grid cell {np.round(grid.cell,1)})")
        for name, v, _, kind in RT.report(p, None, it):
            if name not in SKIP_POS:
                print(f"  {name:<20} {v:.4f} ({'acc' if kind == 'acc' else 'MAE world units'})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "pred": pred.state_dict(),
                "pred_residual": True, "norm": norm.state_dict(), "grid": grid.state_dict(),
                "wall": wall.numpy(), "split": split, "latent": a.latent, "n_actions": NA,
                "config": vars(a), "history": hist}, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
