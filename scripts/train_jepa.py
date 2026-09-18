"""Train the encoder and the predictor together, the way the JEPA papers actually do it.

    .venv/bin/python scripts/train_jepa.py

The frozen behaviour-cloned encoder failed as a dynamics space: an inverse-dynamics probe recovered the
action from a pair of latents at 0.059 against a 0.056 single-latent baseline, i.e. the transition carried
almost no trace of what the action did. BC is discriminative -- it learns which action to take, never what
an action changes -- so this trains the encoder BY the predictive objective instead.

Faithful to the papers where it matters:
  * context encoder trainable; **target encoder is an EMA copy** with a stop-gradient, the mechanism
    I-JEPA calls essential and the only thing preventing collapse once the encoder is no longer frozen
    (I-JEPA 2301.08243 App. A; V-JEPA 2404.08471 Eq. 1)
  * **L1** on **layer-normalised targets** (I-JEPA reference implementation; V-JEPA Eq. 1)
  * a short **rollout loss** as well as teacher forcing (V-JEPA 2-AC 2506.09985 Eq. 2-3)
  * **stride** above one frame -- at 10 Hz this game is unchanged in 74% of steps

No decoder is in the loss path: DINO-WM (2411.04983) reports decoder gradients hurt. Instead the learned
latent is probed AFTER training, so we can see what the predictive objective kept and what it discarded --
in particular whether enemy behaviour state survives, since Link does not control it and it is what the
reward depends on.

Collapse is measured, not assumed: a constant encoder solves the prediction task perfectly.
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
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)
from zelda_rl.dynamics import LatentDynamics, norm_target  # noqa: E402
from scripts.train_bc_decode import DEFAULT_DATA, load  # noqa: E402


def collapse_stats(z):
    """A constant encoder solves prediction perfectly, so measure spread and effective rank."""
    zc = z - z.mean(0, keepdim=True)
    std = zc.std(0)
    cov = (zc.T @ zc) / max(len(z) - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp(min=1e-12)
    p = ev / ev.sum()
    return {"mean_std": float(std.mean()), "min_std": float(std.min()),
            "effective_rank": float(torch.exp(-(p * p.log()).sum()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--init-from", default="checkpoints/bc/bc-decode-final.pt",
                    help="warm-start the encoder from the BC one (and take its normaliser/split); "
                         "pass 'none' to start the encoder from scratch")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--unroll", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--ema-final", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--probe-epochs", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/dynamics/jepa.pt")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ref = torch.load(a.init_from, weights_only=False, map_location="cpu") if a.init_from != "none" else None
    if ref is None:
        raise SystemExit("--init-from is needed for the normaliser and slot split; pass the BC checkpoint")
    norm, split, LAT = Norm.load(ref["norm"]), ref["split"], ref["latent"]

    meta, state, action, slot, epid = load(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    Fn = norm.apply(featurise(state))
    T = {k: torch.from_numpy(v) for k, v in Fn.items()}
    act = torch.from_numpy(action)

    ctx = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=meta["n_actions"],
                latent=LAT, gru=0)
    if a.init_from != "none":
        ctx.load_state_dict(ref["model"])
        print(f"encoder warm-started from {a.init_from}")
    tgt = copy.deepcopy(ctx)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = LatentDynamics(latent=LAT, n_actions=meta["n_actions"], hidden=a.hidden, layers=a.layers)
    opt = torch.optim.Adam(list(ctx.parameters()) + list(pred.parameters()), lr=a.lr)

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    def starts(mask, k):
        ok = mask.copy()
        for j in range(1, k + 1):
            ok &= (np.r_[epid[j:], np.full(j, -1)] == epid)
        return np.where(ok)[0]
    need = a.unroll * a.stride
    tr_idx = starts(~np.isin(slot, list(test | tune)), need)
    tu_idx = starts(np.isin(slot, list(tune)), need)
    te_idx = starts(np.isin(slot, list(test)), 10 * a.stride)
    print(f"train {len(tr_idx):,} | tune {len(tu_idx):,} | test {len(te_idx):,} | stride {a.stride} "
          f"({a.stride * 100} ms) | unroll {a.unroll}")

    def enc(model, i):
        return model.encode(T["link"][i], T["rows"][i], T["ids"][i], T["mask"][i])

    @torch.no_grad()
    def embed(model, idx, bs=8192):
        model.eval()
        return torch.cat([enc(model, torch.from_numpy(idx[s:s + bs])) for s in range(0, len(idx), bs)])

    steps_total = a.epochs * max(len(tr_idx) // a.batch, 1)
    step, best, best_state, bad = 0, float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        ctx.train()
        perm = rng.permutation(tr_idx)
        tot = n = 0.0
        for s in range(0, len(perm) - a.batch, a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = enc(ctx, i)
            loss = 0.0
            for k in range(a.unroll):
                z = pred(z, act[i + k * a.stride])
                with torch.no_grad():                        # stop-gradient onto the EMA target
                    y = norm_target(enc(tgt, i + (k + 1) * a.stride))
                loss = loss + F.l1_loss(norm_target(z), y)
            loss = loss / a.unroll
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(ctx.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            m = a.ema + (a.ema_final - a.ema) * step / max(steps_total, 1)      # momentum -> 1.0
            with torch.no_grad():
                for pt, pc in zip(tgt.parameters(), ctx.parameters()):
                    pt.mul_(m).add_(pc.detach(), alpha=1 - m)
            step += 1; tot += loss.item(); n += 1
        ctx.eval()
        with torch.no_grad():
            zt = enc(ctx, torch.from_numpy(tu_idx))
            zp = pred(zt, act[tu_idx])
            y = norm_target(enc(tgt, torch.from_numpy(tu_idx + a.stride)))
            tl = float(F.l1_loss(norm_target(zp), y))
            ident = float(F.l1_loss(norm_target(zt), y))
            cs = collapse_stats(zt[:4096])
        skill = 1 - tl / max(ident, 1e-9)
        if tl < best - 1e-5:
            best, bad = tl, 0
            best_state = {"ctx": {k: v.clone() for k, v in ctx.state_dict().items()},
                          "pred": {k: v.clone() for k, v in pred.state_dict().items()}}
        else:
            bad += 1
        if ep % 2 == 0 or ep == 1:
            print(f"  ep {ep:>3}/{a.epochs} train {tot/n:.4f} | tune L1 {tl:.4f} vs identity {ident:.4f} "
                  f"(skill {skill:+.3f}) | z std {cs['mean_std']:.3f} eff-rank {cs['effective_rank']:.1f}"
                  f"{'  *' if bad == 0 else ''} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= 5:
            print(f"  early stop at epoch {ep}")
            break
    ctx.load_state_dict(best_state["ctx"]); pred.load_state_dict(best_state["pred"])

    # ---------------- what did the learned latent keep?
    ctx.eval()
    print("\n--- probes on the LEARNED latent (test slots) ---")
    Zte = embed(ctx, te_idx)
    cs = collapse_stats(Zte[:8192])
    print(f"collapse check: mean per-dim std {cs['mean_std']:.4f}, min {cs['min_std']:.4f}, "
          f"effective rank {cs['effective_rank']:.1f} / {LAT}")

    same = lambda k: np.where((np.r_[epid[k:], np.full(k, -1)] == epid) & np.isin(slot, list(test)))[0]
    print("\nlatent movement (consecutive vs random pairs, L1 on normalised latents)")
    for k in (a.stride, 10 * a.stride):
        i = same(k)
        zi, zj = embed(ctx, i), embed(ctx, i + k)
        d = float(F.l1_loss(norm_target(zi), norm_target(zj)))
        r = float(F.l1_loss(norm_target(zi), norm_target(zi[torch.randperm(len(zi))])))
        print(f"  k={k:>3} ({k*100:>4} ms)  consecutive {d:.4f}  random {r:.4f}  ratio {d/r:.3f}")

    print("\ninverse dynamics: can the action be read from a pair of latents?")
    for k in (a.stride,):
        i = same(k)
        itr = i[~np.isin(slot[i], list(test))] if False else None
        tr_i = starts(~np.isin(slot, list(test | tune)), k)
        Ztr, Ztr2 = embed(ctx, tr_i), embed(ctx, tr_i + k)
        Zte1, Zte2 = embed(ctx, i), embed(ctx, i + k)
        f_tr = torch.cat([Ztr, Ztr2, Ztr2 - Ztr], -1)
        f_te = torch.cat([Zte1, Zte2, Zte2 - Zte1], -1)
        net = nn.Sequential(nn.Linear(LAT * 3, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                            nn.Linear(512, meta["n_actions"]))
        o2 = torch.optim.Adam(net.parameters(), 1e-3)
        for e in range(a.probe_epochs):
            pm = rng.permutation(len(f_tr))
            for s in range(0, len(pm), 1024):
                b = torch.from_numpy(pm[s:s + 1024])
                o2.zero_grad(); F.cross_entropy(net(f_tr[b]), act[tr_i][b]).backward(); o2.step()
        with torch.no_grad():
            lg = net(f_te)
            t1 = float((lg.argmax(-1) == act[i]).float().mean())
            t3 = float((lg.topk(3, -1).indices == act[i][:, None]).any(1).float().mean())
        maj = float((act[i] == torch.mode(act[torch.from_numpy(tr_i)]).values).float().mean())
        print(f"  k={k}: top-1 {t1:.4f}  top-3 {t3:.4f}   (majority {maj:.4f}, ceiling ~0.220; "
              f"the frozen BC latent scored 0.059)")

    print("\nreward-relevant fields, decoder trained post-hoc on the FROZEN learned latent")
    tr_i = starts(~np.isin(slot, list(test | tune)), 1)
    Ztr = embed(ctx, tr_i)
    dec = StateDecoder(latent=LAT)
    o3 = torch.optim.Adam(dec.parameters(), 1e-3)
    ytr = {k: torch.from_numpy(v[tr_i]) for k, v in Y.items()}
    for e in range(a.probe_epochs):
        pm = rng.permutation(len(Ztr))
        for s in range(0, len(pm), 1024):
            b = torch.from_numpy(pm[s:s + 1024])
            p = dec(Ztr[b])
            m, mb = ytr["row_present"][b], ytr["row_is_baba"][b]
            l = F.cross_entropy(p["link_hp"], ytr["link_hp_c"][b])
            l = l + (F.cross_entropy(p["row_hp"].reshape(-1, 3), ytr["row_hp_c"][b].reshape(-1),
                                     reduction="none").reshape(m.shape) * m).sum() / m.sum().clamp(min=1)
            ce = F.cross_entropy(p["row_state"].reshape(-1, p["row_state"].shape[-1]),
                                 ytr["row_state"][b].reshape(-1), reduction="none").reshape(m.shape)
            l = l + (ce * mb).sum() / mb.sum().clamp(min=1)
            o3.zero_grad(); l.backward(); o3.step()
    dec.eval()
    ite = starts(np.isin(slot, list(test)), 1)
    with torch.no_grad():
        p = dec(embed(ctx, ite))
    m, mb = Y["row_present"][ite] > .5, Y["row_is_baba"][ite] > .5
    res = {"link_hp": float((p["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][ite]).mean()),
           "row_hp": float((p["row_hp"].argmax(-1).numpy()[m] == Y["row_hp_c"][ite][m]).mean()),
           "row_state": float((p["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][ite][mb]).mean())}
    print(f"  link_hp {res['link_hp']:.4f}   row_hp {res['row_hp']:.4f}   row_state {res['row_state']:.4f}")
    print(f"  (frozen BC latent, same fields: link_hp 0.9839  row_hp 0.9073  row_state 0.9992)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ctx": ctx.state_dict(), "pred": pred.state_dict(), "norm": ref["norm"], "split": split,
                "latent": LAT, "link_dim": T["link"].shape[1], "row_dim": T["rows"].shape[2],
                "n_actions": meta["n_actions"], "config": vars(a),
                "metrics": {"collapse": cs, "fields": res, "inverse_top1": t1}}, a.out)
    Path("results/jepa.json").write_text(json.dumps(
        {"config": vars(a), "collapse": cs, "fields": res, "inverse_top1": t1}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
