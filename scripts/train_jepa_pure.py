"""Pure JEPA: train the encoder BY the predictive objective, no reconstruction, then probe what survived.

    .venv/bin/python scripts/train_jepa_pure.py

Reconstruction is deliberately PINNED here. Stage 1 + 2 (`docs/JEPA_PLAN.md`) trained the encoder to rebuild
the reward's fields and then fitted a transition model on it; that tied do-nothing at H=10 and lost beyond.
This asks the opposite question, which is the one the JEPA papers actually pose:

    if the ONLY pressure is "the next latent must be predictable from this one and the action",
    what representation appears, and can the true state still be read out of it?

    encoder (trainable) ---> z_t ---> M(z_t, a_t) ---> z_hat
    EMA copy (frozen)   ---> z_{t+1}  [stop-gradient TARGET]
    loss = L1( layer_norm(z_hat), layer_norm(z_{t+1}) )      teacher forcing + a rollout term

TWO THINGS THAT MAKE OR BREAK THIS, both measured rather than assumed:

1. COLLAPSE. A constant encoder solves prediction perfectly -- it is the global optimum of this loss. I-JEPA
   names the EMA target encoder with a stop-gradient as the mechanism that prevents it (2301.08243 App. A), so
   that is what is used, and `collapse_stats` prints latent std, effective rank and move/rand EVERY epoch so a
   collapse is visible as it happens instead of being discovered in the final table. Our own VQ run scored
   skill +0.000 with a predictor that had provably learned to copy z_t, so this is not hypothetical here.

   Note on why the loss is layer-normalised ONLY, with no un-normalised term: stage 2 needed an un-normalised
   L1 because the layer-normalised loss is exactly scale-invariant, so nothing bounded ||z|| and an
   autoregressive rollout reached 2.2e11. That fix is correct for a FROZEN encoder and actively harmful here --
   with a trainable encoder, `L1(z_hat, z_target)` is minimised by driving both to zero, i.e. it pays for
   collapse. Rollout scale is therefore handled at inference (renormalise the predicted latent to the
   encoder's own typical norm) rather than in the loss.

2. IS THE STATE STILL THERE. After training, the encoder is frozen and a decoder is fitted on top for the
   reward fields ONLY (`zelda_rl.reward_decode`, the same loss stage 1 used, so the numbers are comparable).
   Nothing from this probe reaches the encoder -- DINO-WM 2411.04983 reports decoder gradients hurt, and
   keeping it post-hoc is what makes the answer interpretable: whatever the probe cannot read, the predictive
   objective discarded.

Controls: an untrained-encoder probe (does training help the readout at all?) and, because a residual
predictor is zero-initialised, an untrained predictor that is EXACTLY the identity.
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
from zelda_rl.bc import BCNet, Norm, UnifiedDecoder, featurise  # noqa: E402
from zelda_rl.dynamics import build_predictor  # noqa: E402
from zelda_rl.reward_decode import RewardTargets  # noqa: E402


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


def collapse_stats(z: torch.Tensor) -> dict:
    """A constant encoder solves the prediction task perfectly, so measure spread and effective rank."""
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / max(len(z) - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp(min=1e-12)
    p = ev / ev.sum()
    return {"std": float(z.std(0).mean()), "min_std": float(z.std(0).min()),
            "erank": float(torch.exp(-(p * p.log()).sum())), "norm": float(z.norm(dim=-1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt", help="slot split only")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--probe-epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--ema", type=float, default=0.996, help="target-encoder momentum (I-JEPA App. A)")
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--unroll-decay", type=float, default=0.97)
    ap.add_argument("--unroll-batch", type=int, default=128)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 40, 100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/jepa/pure.pt")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    split = torch.load(a.ref, weights_only=False, map_location="cpu")["split"]
    meta, state, act, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    NA = meta["n_actions"]
    Fraw = featurise(state, link_v2=True, control_echo=False)          # no control echo, ever
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_m = ~np.isin(slot, list(test | tune))
    norm = Norm.fit({k: v[tr_m] for k, v in Fraw.items()})
    T = {k: torch.from_numpy(v) for k, v in norm.apply(Fraw).items()}
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1), NA).float()

    same = np.zeros(len(state), bool)
    same[:-1] = epid[1:] == epid[:-1]

    def idx(mask, need):
        return np.flatnonzero(mask & same & np.r_[np.ones(len(state) - need, bool), np.zeros(need, bool)])

    tr, tu, te = idx(tr_m, 1), idx(np.isin(slot, list(tune)), 1), idx(np.isin(slot, list(test)), 1)
    deep = idx(tr_m, a.unroll + 1)
    print(f"{len(state):,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | latent {a.latent} "
          f"| EMA {a.ema} | unroll {a.unroll} | NO reconstruction, NO BC in the loss")

    enc = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=NA, latent=a.latent, gru=0)
    tgt = copy.deepcopy(enc)                                  # EMA target encoder, never receives gradient
    for q in tgt.parameters():
        q.requires_grad_(False)
    pred = build_predictor(a.latent, NA)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=a.lr)

    def E(m, i):
        return m.encode(T["link"][i], T["rows"][i], T["ids"][i], T["mask"][i])

    hist = []
    for ep in range(a.epochs):
        enc.train(); pred.train()
        perm = rng.permutation(tr)
        t0, tl, nb = time.time(), 0.0, 0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            with torch.no_grad():
                zt = E(tgt, i + 1)                            # stop-gradient target
            loss = F.l1_loss(ln(pred(torch.cat([E(enc, i), A[i]], -1))), ln(zt))
            if a.unroll > 1:
                zi = torch.from_numpy(rng.choice(deep, min(a.unroll_batch, len(deep)), replace=False))
                zz, rl, ws = E(enc, zi), torch.zeros(()), 0.0
                for j in range(a.unroll):
                    zz = pred(torch.cat([zz, A[zi + j]], -1))
                    with torch.no_grad():
                        zn = E(tgt, zi + j + 1)
                    w = a.unroll_decay ** j
                    rl = rl + w * F.l1_loss(ln(zz), ln(zn))
                    ws += w
                loss = loss + rl / max(ws, 1e-9)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            with torch.no_grad():                             # EMA update
                for qt, qs in zip(tgt.parameters(), enc.parameters()):
                    qt.mul_(a.ema).add_(qs, alpha=1 - a.ema)
            tl += float(loss.detach()); nb += 1
        enc.eval(); pred.eval()
        with torch.no_grad():
            iu = torch.from_numpy(tu[:20000])
            z0, z1 = E(enc, iu), E(enc, iu + 1)
            pm = float(F.l1_loss(ln(pred(torch.cat([z0, A[iu]], -1))), ln(z1)))
            im = float(F.l1_loss(ln(z0), ln(z1)))
            cs = collapse_stats(z0)
            mr = float((z1 - z0).abs().mean()) / max(float((z0 - z0[torch.randperm(len(z0))]).abs().mean()), 1e-9)
        hist.append({"epoch": ep + 1, "train": tl / nb, "tune": pm, "identity": im, **cs, "move_ratio": mr})
        print(f"  ep {ep+1:>3}/{a.epochs}  train {tl/nb:.5f} | tune {pm:.5f} id {im:.5f} "
              f"skill {1-pm/max(im,1e-9):+.3f} | COLLAPSE std {cs['std']:.4f} min {cs['min_std']:.5f} "
              f"erank {cs['erank']:.1f} |z| {cs['norm']:.2f} move/rand {mr:.3f}  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- probe: is the state still readable?
    print("\n=== PROBE: freeze the encoder, fit a reward-field decoder on top (no gradient to the encoder) ===")
    RT = RewardTargets(state, obs, epid)
    with torch.no_grad():
        Z = torch.cat([E(enc, torch.arange(i, min(i + 8192, len(state))))
                       for i in range(0, len(state), 8192)])
    rnd = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=NA, latent=a.latent, gru=0)
    with torch.no_grad():                                     # CONTROL: an untrained encoder's latent
        Zr = torch.cat([E(rnd, torch.arange(i, min(i + 8192, len(state))))
                        for i in range(0, len(state), 8192)])

    def fit_probe(Zsrc, tag):
        d = UnifiedDecoder(latent=a.latent, reward_only=True)
        o = torch.optim.Adam(d.parameters(), lr=1e-3)
        for _ in range(a.probe_epochs):
            pp = rng.permutation(tr)
            for s in range(0, len(pp), a.batch):
                i = torch.from_numpy(pp[s:s + a.batch])
                l = RT.loss(d(Zsrc[i]), i)
                o.zero_grad(); l.backward(); o.step()
        with torch.no_grad():
            rows = RT.report(d(Zsrc[torch.from_numpy(te)]), None, torch.from_numpy(te))
        print(f"\n  --- {tag} ---")
        for name, v, _, kind in rows:
            print(f"    {name:<20} {v:.4f}  ({'accuracy' if kind == 'acc' else 'MAE, world units'})")
        return {r[0]: r[1] for r in rows}

    trained = fit_probe(Z, "JEPA-trained encoder (frozen)")
    untrained = fit_probe(Zr, "CONTROL: untrained encoder (frozen)")
    print(f"\n  {'field':<20}{'JEPA':>10}{'untrained':>12}   verdict")
    for k in trained:
        better = trained[k] > untrained[k] if k in ("link_hp", "targeted", "scene", "baba_count", "row_hp",
                                                    "row_state") else trained[k] < untrained[k]
        print(f"  {k:<20}{trained[k]:>10.4f}{untrained[k]:>12.4f}   "
              f"{'JEPA better' if better else 'untrained better or equal'}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "target": tgt.state_dict(), "pred": pred.state_dict(),
                "pred_residual": True, "norm": norm.state_dict(), "split": split, "latent": a.latent,
                "link_dim": T["link"].shape[1], "row_dim": T["rows"].shape[2], "n_actions": NA,
                "control_echo": False, "history": hist, "config": vars(a),
                "metrics": {"probe_trained": trained, "probe_untrained": untrained}}, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
