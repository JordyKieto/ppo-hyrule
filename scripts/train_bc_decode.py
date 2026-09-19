"""BC + state reconstruction, trained jointly: does z retain the fields the reward function reads?

    .venv/bin/python scripts/train_bc_decode.py

The reward is never regressed. CombatTracker.step() computes it from raw state, so what decides whether a
latent world model can work here is whether those raw fields survive `encode()`'s masked mean+max pooling.

HP is discrete (Link: 7 half-heart values; Deku Baba: 0/1/2) and is therefore CLASSIFIED. Regressing it and
rounding turns numeric jitter into fabricated damage events, which is fatal: damage happens once in ~253
transitions, so spurious events swamp real ones. Frames at an HP transition are additionally upweighted,
because those few frames are what the reward is made of.

Feedforward (no GRU head): the decoder reads z, not the recurrent state, so recurrence is irrelevant here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)

DEFAULT_DATA = "datasets/transitions/deku-baba/20260918-134229"
DYING = 8                    # index of "dying" in dataset.json["baba_states"]
CONT = ("link_pos", "link_yaw", "row_pos")


def load_many(dirs):
    """Concatenate several collection runs, keeping episode ids unique. Only the first needs dataset.json."""
    meta = json.loads((Path(dirs[0]) / "dataset.json").read_text())
    S, A, SL, EP = [], [], [], []
    off = 0
    for d in dirs:
        d = Path(d)
        rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
        for i, r in enumerate(rows):
            arr = np.load(d / r["file"])
            a_, st = arr["action"], arr["state"]
            n = min(len(a_) + 1, len(st))
            S.append(st[:n])
            A.append(np.r_[a_[:n - 1], a_[n - 2]] if n > 1 else a_[:1])
            SL.append(np.full(n, r["slot"], np.int64)); EP.append(np.full(n, off + i, np.int64))
        off += len(rows)
    return meta, np.concatenate(S), np.concatenate(A), np.concatenate(SL), np.concatenate(EP)


def episode_targets(data_dir: Path, kind: str):
    """One scalar per episode, broadcast to every frame in it."""
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        d = np.load(data_dir / r["file"])
        t = len(d["action"])
        v = float(r["return"]) if kind == "return" else float(d["reward"].mean())
        out.append(np.full(t, v, np.float32))
    return np.concatenate(out)


def load(data_dir: Path):
    """-> meta, state, action, slot, episode-id, with T+1 states per episode.

    The final state of each episode is KEPT. An earlier version stored state[:T] for T actions, which
    silently dropped the last transition -- the one carrying the death, the room clear, or the room exit.
    For the room-exit episodes that was 100% of the signal: `scene` flips 0->85 at exactly index T, so all
    27 exits sat one index past everything training could see.

    Actions are padded by repeating the last one so every array aligns; the padded frame is never a
    transition source, because a transition needs episode-id equality with the following frame.
    """
    meta = json.loads((data_dir / "dataset.json").read_text())
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    S, act, slot, ep = [], [], [], []
    for i, r in enumerate(rows):
        d = np.load(data_dir / r["file"])
        a_, st = d["action"], d["state"]
        n = min(len(a_) + 1, len(st))                       # T+1 states where they were stored
        S.append(st[:n])
        act.append(np.r_[a_[:n - 1], a_[n - 2]] if n > 1 else a_[:1])
        slot.append(np.full(n, r["slot"], np.int64)); ep.append(np.full(n, i, np.int64))
    return (meta, np.concatenate(S), np.concatenate(act), np.concatenate(slot), np.concatenate(ep))


def transition_frames(y: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Frames on either side of a change in `y`, within an episode. Shape follows y (frames, ...)."""
    same = ep[1:] == ep[:-1]
    ch = np.zeros_like(y, dtype=bool)
    d = (y[1:] != y[:-1]) & same.reshape(-1, *([1] * (y.ndim - 1)))
    ch[:-1] |= d
    ch[1:] |= d
    return ch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--val-slots", default="7,9", help="TEST slots: reported, never used to fit anything")
    ap.add_argument("--tune-slots", default="4,8", help="threshold-selection slots: held out of training, "
                                                       "used only to pick detector operating points")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--w-decode", type=float, default=1.0)
    ap.add_argument("--w-action", type=float, default=1.0)
    ap.add_argument("--w-hp", type=float, default=3.0, help="weight on the two HP classification losses")
    ap.add_argument("--w-pos", type=float, default=3.0,
                    help="weight on Link/actor position regression. Positions need finer resolution than "
                         "action selection ever required, so BC alone under-determines them.")
    ap.add_argument("--w-transition", type=float, default=20.0, help="extra weight on HP-transition frames")
    ap.add_argument("--class-weight-pow", type=float, default=0.5, help="class weight ~ freq^-p (0 = off)")
    ap.add_argument("--oversample", type=float, default=0.15,
                    help="fraction of each batch drawn from HP-transition frames (0 = off). Reweighting alone "
                         "leaves ~3-6 such frames in a 1024 batch; this puts ~150 in.")
    ap.add_argument("--w-consistency", type=float, default=0.5,
                    help="penalise HP=0 on a Baba row that is neither dying nor despawned")
    ap.add_argument("--out", default="checkpoints/bc/bc-decode.pt")
    ap.add_argument("--cond", choices=["none", "return", "mean"], default="none",
                    help="condition the ACTION head on one number per episode: its total return, or its "
                         "mean reward per step. The encoder is untouched, so z stays a function of state "
                         "alone. Without this, BC sees the same state in a +86 game and a -25 game and is "
                         "asked to predict both actions.")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also save <out>.ep<N>.pt every N epochs, so one run yields the whole training "
                         "curve instead of re-training from scratch per point")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)

    meta, state, action, slot, epid = load(Path(a.data))
    val_slots = {int(s) for s in a.val_slots.split(",")}
    tune_slots = {int(s) for s in a.tune_slots.split(",") if s.strip()}
    held = list(val_slots | tune_slots)
    va = np.where(np.isin(slot, list(val_slots)))[0]
    tu = np.where(np.isin(slot, list(tune_slots)))[0]
    tr = np.where(~np.isin(slot, held))[0]
    print(f"{len(state):,} frames | train {len(tr):,} (slots {sorted(set(slot[tr].tolist()))}) | "
          f"tune {len(tu):,} (slots {sorted(tune_slots)}) | test {len(va):,} (slots {sorted(val_slots)})")

    Fa = featurise(state)
    norm = Norm.fit({k: v[tr] for k, v in Fa.items()})
    T = {k: torch.from_numpy(v) for k, v in norm.apply(Fa).items()}
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    sc = {k: (Y[k][tr].mean(0, keepdims=True), Y[k][tr].std(0, keepdims=True) + 1e-6) for k in CONT}
    Yt = {k: torch.from_numpy(((Y[k] - sc[k][0]) / sc[k][1]).astype(np.float32)) if k in CONT
          else torch.from_numpy(Y[k]) for k in Y}
    act = torch.from_numpy(action)

    # per-frame weights: HP transitions are the frames the reward is made of
    wl = 1.0 + a.w_transition * transition_frames(Y["link_hp_c"], epid).astype(np.float32)
    wr = 1.0 + a.w_transition * transition_frames(Y["row_hp_c"], epid).astype(np.float32)
    W = {"link": torch.from_numpy(wl), "row": torch.from_numpy(wr)}

    def cw(y, n, mask=None):
        c = np.bincount(y[tr].ravel() if mask is None else y[tr][mask[tr]].ravel(), minlength=n).astype(np.float64)
        w = (c.sum() / np.maximum(c, 1)) ** a.class_weight_pow
        return torch.tensor((w / w.mean()).astype(np.float32))
    pres_m = Y["row_present"] > 0.5
    CW = {"link_hp": cw(Y["link_hp_c"], len(LINK_HP_VALUES)),
          "row_hp": cw(Y["row_hp_c"], len(ROW_HP_VALUES), pres_m)}
    print(f"class weights  link_hp {CW['link_hp'].numpy().round(2)}  row_hp {CW['row_hp'].numpy().round(2)}")
    print(f"HP-transition frames: link {(wl > 1).mean():.3%}  row {(wr > 1).any(1).mean():.3%}")

    trans_any = (wl > 1) | (wr > 1).any(1)
    pool = tr[trans_any[tr]]
    print(f"oversample pool: {len(pool):,} transition frames of {len(tr):,} train "
          f"({len(pool) / len(tr):.2%}) -> {a.oversample:.0%} of each batch")

    net = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=meta["n_actions"],
                latent=a.latent, gru=0)
    dec = StateDecoder(latent=a.latent)
    extra = []
    if a.cond != "none":
        ct = episode_targets(Path(a.data), a.cond)
        cmu, csd = float(ct[tr].mean()), float(ct[tr].std() + 1e-6)
        C = torch.from_numpy(((ct - cmu) / csd).astype(np.float32))[:, None]
        cond_head = torch.nn.Linear(a.latent + 1, meta["n_actions"])
        torch.nn.init.orthogonal_(cond_head.weight, 0.01); torch.nn.init.zeros_(cond_head.bias)
        extra = list(cond_head.parameters())
        print(f"conditioning the action head on per-episode {a.cond}: "
              f"mean {cmu:.2f} sd {csd:.2f} over train")
    opt = torch.optim.Adam(list(net.parameters()) + list(dec.parameters()) + extra, lr=a.lr)
    act_logits = (lambda z, i: cond_head(torch.cat([z, C[i]], -1))) if a.cond != "none" \
        else (lambda z, i: net.action(z))
    rng = np.random.default_rng(a.seed)

    def fwd(i):
        z = net.encode(T["link"][i], T["rows"][i], T["ids"][i], T["mask"][i])
        return act_logits(z, i), dec(z)

    def decode_loss(p, i):
        m, mb = Yt["row_present"][i], Yt["row_is_baba"][i]
        hp_l = (F.cross_entropy(p["link_hp"], Yt["link_hp_c"][i], weight=CW["link_hp"],
                                reduction="none") * W["link"][i]).mean()
        rh = (F.cross_entropy(p["row_hp"].reshape(-1, p["row_hp"].shape[-1]), Yt["row_hp_c"][i].reshape(-1),
                              weight=CW["row_hp"], reduction="none").reshape(m.shape) * m * W["row"][i])
        hp_r = rh.sum() / (m * W["row"][i]).sum().clamp(min=1)
        L = (a.w_pos * F.mse_loss(p["link_pos"], Yt["link_pos"][i]) + F.mse_loss(p["link_yaw"], Yt["link_yaw"][i])
             + F.binary_cross_entropy_with_logits(p["link_bits"], Yt["link_bits"][i])
             + F.cross_entropy(p["targeted"], Yt["targeted"][i])
             + F.binary_cross_entropy_with_logits(p["row_present"], m)
             + F.binary_cross_entropy_with_logits(p["row_is_baba"], mb))
        L = L + a.w_pos * (((p["row_pos"] - Yt["row_pos"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        ce = F.cross_entropy(p["row_state"].reshape(-1, p["row_state"].shape[-1]),
                             Yt["row_state"][i].reshape(-1), reduction="none").reshape(m.shape)
        L = L + (ce * mb).sum() / mb.sum().clamp(min=1) + a.w_hp * (hp_l + hp_r)
        if a.w_consistency:
            # a Baba at 0 HP is dying or already despawned; anything else is the heads contradicting each other
            p0 = p["row_hp"].softmax(-1)[..., 0]
            pdy = p["row_state"].softmax(-1)[..., DYING]
            L = L + a.w_consistency * (p0 * p["row_present"].sigmoid() * (1 - pdy) * mb).mean()
        return L

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); dec.train()
        perm = rng.permutation(tr)
        n = na = nd = 0.0
        k = int(a.batch * a.oversample) if len(pool) else 0
        for s in range(0, len(perm), a.batch - k):
            b = perm[s:s + a.batch - k]
            if k:
                b = np.concatenate([b, rng.choice(pool, k, replace=True)])
            i = torch.from_numpy(b)
            logits, p = fwd(i)
            la = F.cross_entropy(logits, act[i])
            ld = decode_loss(p, i)
            loss = a.w_action * la + a.w_decode * ld
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            n += 1; na += la.item(); nd += ld.item()
        if ep % max(a.epochs // 12, 1) == 0 or ep == 1:
            print(f"  epoch {ep}/{a.epochs}  action CE {na / n:.3f}  decode {nd / n:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if a.snapshot_every and ep % a.snapshot_every == 0:
            snap = Path(str(a.out).replace(".pt", f".ep{ep:04d}.pt"))
            snap.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "decoder": dec.state_dict(), "norm": norm.state_dict(),
                        "decode_scale": {k: [v[0], v[1]] for k, v in sc.items()}, "config": vars(a) | {"epoch": ep},
                        "latent": a.latent, "link_dim": T["link"].shape[1], "row_dim": T["rows"].shape[2],
                        "n_actions": meta["n_actions"],
                        "split": {"val_slots": sorted(val_slots), "tune_slots": sorted(tune_slots),
                                  "train_slots": sorted(set(slot[tr].tolist()))}}, snap)

    # ---------------- report
    net.eval(); dec.eval()
    P = {}
    with torch.no_grad():
        for s in range(0, len(va), 8192):
            logits, p = fwd(torch.from_numpy(va[s:s + 8192]))
            for k, v in list(p.items()) + [("logits", logits)]:
                P.setdefault(k, []).append(v)
    P = {k: torch.cat(v) for k, v in P.items()}
    yv = {k: v[va] for k, v in Y.items()}
    raw = lambda k: P[k].numpy() * sc[k][1] + sc[k][0]
    out = []

    def cls(pred, true, name, n_cls, extra=""):
        base = float((true == np.bincount(true.ravel(), minlength=n_cls).argmax()).mean())
        out.append({"field": name, "metric": "accuracy", "model": float((pred == true).mean()),
                    "baseline": base, "unit": "majority" + extra})

    cls(P["link_hp"].argmax(-1).numpy(), yv["link_hp_c"], "link_hp (7-class)", len(LINK_HP_VALUES))
    out.append({"field": "link_pos", "metric": "MAE", "unit": "world units",
                "model": float(np.abs(raw("link_pos") - yv["link_pos"]).mean()),
                "baseline": float(np.abs(yv["link_pos"] - Y["link_pos"][tr].mean(0)).mean())})
    ang = lambda p, t: np.minimum(*(lambda e: (e, 360 - e))(
        np.degrees(np.abs(np.arctan2(p[:, 0], p[:, 1]) - np.arctan2(t[:, 0], t[:, 1])))))
    out.append({"field": "link_yaw", "metric": "mean abs error", "unit": "degrees",
                "model": float(ang(raw("link_yaw"), yv["link_yaw"]).mean()),
                "baseline": float(ang(np.repeat(Y["link_yaw"][tr].mean(0, keepdims=True), len(va), 0),
                                      yv["link_yaw"]).mean())})
    for j, nm in enumerate(["sf1_dead", "sf1_shield", "sf1_damaged"]):
        cls((P["link_bits"][:, j].numpy() > 0).astype(np.int64), yv["link_bits"][:, j].astype(np.int64), nm, 2,
            f", pos rate {yv['link_bits'][:, j].mean():.3%}")
    cls(P["targeted"].argmax(-1).numpy(), yv["targeted"], "targeted_row", 17)
    cls((P["row_present"].numpy() > 0).astype(np.int64), yv["row_present"].astype(np.int64), "row_present", 2)
    cls((P["row_is_baba"].numpy() > 0).astype(np.int64), yv["row_is_baba"].astype(np.int64), "row_is_baba", 2)
    m = yv["row_present"] > 0.5
    ph = P["row_hp"].argmax(-1).numpy()
    out.append({"field": "row_hp (3-class, present)", "metric": "accuracy", "unit": "majority",
                "model": float((ph[m] == yv["row_hp_c"][m]).mean()),
                "baseline": float((yv["row_hp_c"][m] == np.bincount(Y["row_hp_c"][tr][pres_m[tr]]).argmax()).mean())})
    rp, tp = raw("row_pos")[m], yv["row_pos"][m]
    out.append({"field": "row_pos (present)", "metric": "MAE", "unit": "world units",
                "model": float(np.abs(rp - tp).mean()),
                "baseline": float(np.abs(tp - Y["row_pos"][tr][pres_m[tr]].mean(0)).mean())})
    for nm, e in [("link_pos", np.linalg.norm(raw("link_pos") - yv["link_pos"], axis=-1)),
                  ("row_pos (present)", np.linalg.norm(rp - tp, axis=-1))]:
        for q in (50, 90):
            out.append({"field": f"{nm} p{q}", "metric": "euclidean err", "unit": "world units",
                        "model": float(np.percentile(e, q)), "baseline": float("nan")})
    mb = yv["row_is_baba"] > 0.5
    tb = Y["row_state"][tr][Y["row_is_baba"][tr] > 0.5]
    out.append({"field": "row_state (Baba)", "metric": "accuracy", "unit": "majority",
                "model": float((P["row_state"].argmax(-1).numpy()[mb] == yv["row_state"][mb]).mean()),
                "baseline": float((yv["row_state"][mb] == np.bincount(tb, minlength=10).argmax()).mean())})

    print(f"\n{'field':<28} {'metric':<16} {'model':>9} {'baseline':>9}  unit")
    print("-" * 82)
    for r in out:
        print(f"{r['field']:<28} {r['metric']:<16} {r['model']:>9.4f} {r['baseline']:>9.4f}  {r['unit']}")

    # ---------------- the metric that decides it: reward-critical events, from DECODED state
    e = epid[va]
    same = e[1:] == e[:-1]
    dec_rh = np.where(P["row_present"].numpy() > 0, ph, 0)
    tru_rh = np.where(m, yv["row_hp_c"], 0)
    dec_lh = P["link_hp"].argmax(-1).numpy()

    def prf(t, p, name):
        tp_, fp_, fn_ = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        pr = tp_ / max(tp_ + fp_, 1); rc = tp_ / max(tp_ + fn_, 1)
        print(f"  {name:<26} true={int(t.sum()):>5}  P={pr:.3f}  R={rc:.3f}  F1={2*pr*rc/max(pr+rc,1e-9):.3f}"
              f"   (FP={fp_}, FN={fn_})")

    def cummin_ep(x, ep):
        """HP never rises: a Deku Baba cannot heal (0 of 891,459 true row-steps) and Link healed once in
        109,649. Running-min over the episode removes decoded flicker, which is most of the false events."""
        out = x.copy()
        cuts = np.r_[0, np.where(ep[1:] != ep[:-1])[0] + 1, len(ep)]
        for i, j in zip(cuts[:-1], cuts[1:]):
            out[i:j] = np.minimum.accumulate(out[i:j], axis=0)
        return out

    bb = mb[:-1][same]
    for tag, rh, lh in [("raw", dec_rh, dec_lh),
                        ("+ monotone HP", np.where(P["row_present"].numpy() > 0, cummin_ep(ph, e), 0),
                         cummin_ep(dec_lh, e))]:
        print(f"\nreward-critical EVENTS from decoded state, {tag} (held-out slots)")
        print("-" * 82)
        prf(((tru_rh[1:] - tru_rh[:-1])[same] < 0) & bb, ((rh[1:] - rh[:-1])[same] < 0) & bb, "Baba damage")
        prf((tru_rh[:-1][same] > 0) & (tru_rh[1:][same] == 0) & bb,
            (rh[:-1][same] > 0) & (rh[1:][same] == 0) & bb, "Baba kill")
        prf((yv["link_hp_c"][1:] - yv["link_hp_c"][:-1])[same] < 0, (lh[1:] - lh[:-1])[same] < 0, "Link damage")
    top1 = float((P["logits"].argmax(-1).numpy() == action[va]).mean())
    top3 = float((P["logits"].topk(3, -1).indices.numpy() == action[va][:, None]).any(1).mean())
    print(f"\naction top-1 {top1:.4f}  top-3 {top3:.4f}   (Bayes ceiling 0.220, majority 0.039)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "decoder": dec.state_dict(), "norm": norm.state_dict(),
                "decode_scale": {k: [v[0], v[1]] for k, v in sc.items()}, "config": vars(a),
                "latent": a.latent, "link_dim": T["link"].shape[1], "row_dim": T["rows"].shape[2],
                "n_actions": meta["n_actions"],
                "split": {"val_slots": sorted(val_slots), "tune_slots": sorted(tune_slots),
                          "train_slots": sorted(set(slot[tr].tolist()))},
                "metrics": {"fields": out, "top1": top1, "top3": top3}}, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
