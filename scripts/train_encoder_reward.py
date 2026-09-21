"""JEPA stage 1: an encoder trained to reconstruct ONLY the reward's own fields, with light BC pressure.

    .venv/bin/python scripts/train_encoder_reward.py --out checkpoints/jepa/enc_s1.pt

Plan and rationale: `docs/JEPA_PLAN.md`. Stage 2 freezes this encoder and fits a transition model on it;
stage 3 is dream PPO. Nothing here predicts anything -- there is no predictor and no action-conditioning.

    loss = w_recon * recon(D(E(s)), reward_fields(s))  +  w_bc * CE(action_head(E(s)), a)

WHY ONLY THE REWARD'S FIELDS. The full decoder reconstructs 27 global + 10 per-row heads, i.e. camera,
velocity, collision bits, action state, melee, doors, ledges. None of that is read by `CombatTracker.step`.
Every dynamics attempt so far has been at or below the identity (-0.776 deep+LN, -2.023 shallow+tanh, -2.412
transformer, +0.000 VQ), and the recorded diagnosis is a latent asked to carry far more than the reward needs
while carrying too little of what an action does. `bc.REWARD_GLOBAL` / `bc.REWARD_ROW` are the exact fields
that function touches, derived from it rather than recalled.

NO CONTROL LEAKAGE. BC means Link sees the state and predicts the controls, so he must not see the controls:
`featurise(control_echo=False)` zeroes `CONTROL_ECHO_FEATS`. Two reward fields are themselves action-states
and are NOT in that list -- `sf1_bit22` (shielding = R held) and `has_lockon` (Z pressed) -- so `--drop-leak`
zeroes those too. They stay reconstruction TARGETS either way; the point is to measure how much of the BC
number depends on reading them, because accuracy that comes from "I am shielding" is not behaviour cloning.

Head kinds follow what is already validated here rather than what is convenient: HP is CLASSIFIED, never
regressed (damage F1 0.286 -> 0.906 in section 3), `link_facing_target` is masked to frames where the Baba is
out of the ground (64.7% of Baba frames are `hidden`, where the angle is near-uniform, R=0.12), and the real
fields are standardised before the MSE so one head in raw world units cannot swamp the rest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import DEFAULT_DATA, load_many, transition_frames  # noqa: E402
from zelda_rl.bc import (LINK_FEAT_NAMES, LINK_HP_VALUES, N_BABA_STATES, REWARD_GLOBAL,  # noqa: E402
                         REWARD_ROW, ROW_HP_VALUES, BCNet, Norm, UnifiedDecoder, decode_targets,
                         featurise, hp_class)

REAL_FIELDS = ("link_pos", "near_dist", "row_pos", "row_dist")     # standardised before the MSE
LEAK_FEATS = ("sf1_bit22", "has_lockon")                           # shielding (R held), lock-on (Z pressed)


def leak_idx():
    """Link-feature columns for the two reward fields that are also action-states. Derived, never counted."""
    idx = [i for i, n in enumerate(LINK_FEAT_NAMES) if n in LEAK_FEATS]
    assert len(idx) == len(LEAK_FEATS), f"expected {LEAK_FEATS} in LINK_FEAT_NAMES, found {idx}"
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt", help="source of the slot split only")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=0.05, help="deliberately small: BC is the pressure, not the task")
    ap.add_argument("--w-pos", type=float, default=6.0)
    ap.add_argument("--drop-leak", action="store_true",
                    help="also zero sf1_bit22 / has_lockon in the INPUT (they stay targets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/jepa/enc_s1.pt")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ref = torch.load(a.ref, weights_only=False, map_location="cpu")
    split = ref["split"]
    meta, state, act, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    n = len(state)

    # ---- inputs: no control echo, ever
    Fraw = featurise(state, link_v2=True, control_echo=False)
    if a.drop_leak:
        Fraw["link"][:, leak_idx()] = 0.0
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_m = ~np.isin(slot, list(test | tune))
    norm = Norm.fit({k: v[tr_m] for k, v in Fraw.items()})     # fit on TRAIN only; never inherit a stale norm
    Fn = norm.apply(Fraw)
    T = {k: torch.from_numpy(v) for k, v in Fn.items()}

    # ---- targets: only the reward's fields
    Y = decode_targets(state, obs, epid=epid)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    scale = {}
    for k in REAL_FIELDS:
        v = Y[k].reshape(-1, Y[k].shape[-1])
        mu, sd = v[np.isfinite(v).all(1)].mean(0), v[np.isfinite(v).all(1)].std(0) + 1e-6
        Y[k] = ((Y[k] - mu) / sd).astype(np.float32)
        scale[k] = (mu.tolist(), sd.tolist())
    Yt = {k: torch.from_numpy(np.asarray(v)) for k, v in Y.items()
          if k in set(REWARD_GLOBAL) | set(REWARD_ROW) | {"link_hp_c", "row_hp_c", "row_state"}}
    A = torch.from_numpy(act.astype(np.int64)).clamp(0, meta["n_actions"] - 1)

    from zelda_rl.combat import BABA_STATES
    HIDDEN = BABA_STATES.index("hidden") if "hidden" in BABA_STATES else 0
    VIS = torch.from_numpy((Y["row_state"] != HIDDEN).astype(np.float32))

    def cw(lab, k, m=None):
        v = lab.reshape(-1)[(m.reshape(-1) if m is not None else slice(None))]
        c = np.bincount(np.asarray(v).astype(np.int64), minlength=k).astype(np.float64)
        w = np.sqrt(c.sum() / np.maximum(c, 1.0))               # sqrt: a raw inverse collapsed the rare class
        return torch.from_numpy((w / w.mean()).astype(np.float32))

    pres = Y["row_present"] > 0.5
    CW = {"link_hp": cw(Y["link_hp_c"], len(LINK_HP_VALUES)),
          "row_hp": cw(Y["row_hp_c"], len(ROW_HP_VALUES), pres)}
    # HP-transition frames are what damage detection lives or dies on, and they are a tiny minority
    wl = torch.from_numpy(1 + 20.0 * transition_frames(Y["link_hp_c"], epid).astype(np.float32))
    wr = torch.from_numpy(1 + 20.0 * transition_frames(Y["row_hp_c"], epid).astype(np.float32))

    tr = np.flatnonzero(tr_m)
    tu = np.flatnonzero(np.isin(slot, list(tune)))
    te = np.flatnonzero(np.isin(slot, list(test)))
    print(f"{n:,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | "
          f"link feats {T['link'].shape[1]} | w_recon {a.w_recon} w_bc {a.w_bc} "
          f"| leak fields {'ZEROED' if a.drop_leak else 'present'} in the input")
    print(f"reconstructing {len(REWARD_GLOBAL)} global + {len(REWARD_ROW)} row heads "
          f"(the full decoder does 27 + 10)")

    enc = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=meta["n_actions"],
                latent=a.latent, gru=0)
    dec = UnifiedDecoder(latent=a.latent, reward_only=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)

    def E(i):
        return enc.encode(T["link"][i], T["rows"][i], T["ids"][i], T["mask"][i])

    def recon(p, i):
        m, mb = Yt["row_present"][i], Yt["row_is_baba"][i]
        L = (a.w_pos * F.mse_loss(p["link_pos"], Yt["link_pos"][i])
             + F.mse_loss(p["link_yaw"], Yt["link_yaw"][i])
             + F.mse_loss(p["near_dist"], Yt["near_dist"][i])
             + F.binary_cross_entropy_with_logits(p["link_bits"], Yt["link_bits"][i])
             + F.cross_entropy(p["targeted"], Yt["targeted"][i])
             + F.cross_entropy(p["scene"], Yt["scene"][i])
             + F.cross_entropy(p["room"], Yt["room"][i])
             + F.cross_entropy(p["baba_count"], Yt["baba_count"][i])
             + F.binary_cross_entropy_with_logits(p["row_present"], m)
             + F.binary_cross_entropy_with_logits(p["row_is_baba"], mb)
             + (F.cross_entropy(p["link_hp"], Yt["link_hp_c"][i], weight=CW["link_hp"],
                                reduction="none") * wl[i]).mean())
        rh = (F.cross_entropy(p["row_hp"].reshape(-1, len(ROW_HP_VALUES)), Yt["row_hp_c"][i].reshape(-1),
                              weight=CW["row_hp"], reduction="none").reshape(m.shape) * m * wr[i])
        L = L + rh.sum() / m.sum().clamp(min=1)
        L = L + a.w_pos * (((p["row_pos"] - Yt["row_pos"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        L = L + (((p["row_dist"] - Yt["row_dist"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        ce = F.cross_entropy(p["row_state"].reshape(-1, N_BABA_STATES), Yt["row_state"][i].reshape(-1),
                             reduction="none").reshape(m.shape)
        L = L + (ce * mb).sum() / mb.sum().clamp(min=1)
        vis = mb * VIS[i]                       # the facing angle is only defined while the Baba is visible
        L = L + (((p["link_facing_target"] - Yt["link_facing_target"][i]) ** 2).mean(-1)
                 * vis).sum() / vis.sum().clamp(min=1)
        return L

    hist = []
    for ep in range(a.epochs):
        enc.train(); dec.train()
        perm = rng.permutation(tr)
        tre = trb = nb = 0.0
        t0 = time.time()
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = E(i)
            lr_ = recon(dec(z), i)
            lb = F.cross_entropy(enc.step(z)[0], A[i])
            loss = a.w_recon * lr_ + a.w_bc * lb
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            tre += float(lr_.detach()); trb += float(lb.detach()); nb += 1
        enc.eval(); dec.eval()
        with torch.no_grad():
            iu = torch.from_numpy(tu)
            zu = E(iu)
            vr, vb = float(recon(dec(zu), iu)), float(F.cross_entropy(enc.step(zu)[0], A[iu]))
        hist.append({"epoch": ep + 1, "recon": tre / nb, "bc": trb / nb, "tune_recon": vr, "tune_bc": vb})
        print(f"  ep {ep+1:>3}/{a.epochs}  recon {tre/nb:.4f}  bc {trb/nb:.4f} | tune recon {vr:.4f} "
              f"bc {vb:.4f} | contribution recon {a.w_recon*tre/nb:.3f} vs bc {a.w_bc*trb/nb:.3f} "
              f"({a.w_recon*tre/nb/max(a.w_bc*trb/nb,1e-9):.0f}x)  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ controls (docs/JEPA_PLAN.md stage 1)
    with torch.no_grad():
        it = torch.from_numpy(te)
        zt = E(it)
        p = dec(zt)
        m = Yt["row_present"][it] > 0.5
        mb = Yt["row_is_baba"][it] > 0.5

        def marg(lab, k):
            """Predict-the-marginal floor: always answer the training split's most common class."""
            c = np.bincount(np.asarray(lab).reshape(-1).astype(np.int64), minlength=k)
            return int(c.argmax())

        print("\n=== CONTROL 1: every field against its predict-the-marginal floor ===")
        rows = []
        # Rare terminals: accuracy against a 99.97% marginal says nothing, so score them the way section 6
        # did -- recall on the event frames and the false-positive rate on the rest.
        for name, k in (("scene", 2), ("room", 8)):
            lab = Yt[name][it].reshape(-1)
            pc = p[name].argmax(-1).reshape(-1)
            base = marg(Yt[name][torch.from_numpy(tr)], k)
            pos = lab != base
            npos = int(pos.sum())
            if npos == 0:
                print(f"  {name:<18} no off-baseline frames in the test split -- UNTESTABLE here "
                      f"(train split has {int((Yt[name][torch.from_numpy(tr)] != base).sum())})")
                continue
            rec = float((pc[pos] == lab[pos]).float().mean())
            fpr = float((pc[~pos] != base).float().mean())
            rows.append((name, rec, fpr, rec > fpr))
            print(f"  {name:<18} recall {rec:.4f} on {npos} event frames | false-positive {fpr:.4f}  "
                  f"{'OK' if rec > fpr else '** FAILS **'}")
        for name, pred_c, lab, k, msk in (
            ("link_hp", p["link_hp"].argmax(-1), Yt["link_hp_c"][it], len(LINK_HP_VALUES), None),
            ("targeted", p["targeted"].argmax(-1), Yt["targeted"][it], 17, None),
            ("baba_count", p["baba_count"].argmax(-1), Yt["baba_count"][it], 4, None),
            ("row_hp", p["row_hp"].argmax(-1), Yt["row_hp_c"][it], len(ROW_HP_VALUES), m),
            ("row_state", p["row_state"].argmax(-1), Yt["row_state"][it], N_BABA_STATES, mb),
        ):
            lab = lab.reshape(-1) if msk is None else lab[msk].reshape(-1)
            pc = pred_c.reshape(-1) if msk is None else pred_c[msk].reshape(-1)
            base = marg(Yt[{"link_hp": "link_hp_c", "row_hp": "row_hp_c"}.get(name, name)]
                        [torch.from_numpy(tr)], k)
            acc, bacc = float((pc == lab).float().mean()), float((lab == base).float().mean())
            rows.append((name, acc, bacc, acc > bacc))
            print(f"  {name:<18} acc {acc:.4f}  marginal {bacc:.4f}  {'OK' if acc > bacc else '** FAILS **'}")
        for name, key, msk in (("link_pos", "link_pos", None), ("near_dist", "near_dist", None),
                               ("row_pos", "row_pos", m), ("row_dist", "row_dist", m)):
            t_ = Yt[key][it]
            pr = p[name]
            if msk is not None:
                t_, pr = t_[msk], pr[msk]
            mae = float((pr - t_).abs().mean())
            bmae = float((t_ - Yt[key][torch.from_numpy(tr)].reshape(-1, t_.shape[-1]).mean(0)).abs().mean())
            rows.append((name, mae, bmae, mae < bmae))
            print(f"  {name:<18} MAE {mae:.4f} (std units)  mean-predictor {bmae:.4f}  "
                  f"{'OK' if mae < bmae else '** FAILS **'}")

        print("\n=== CONTROL 3: collapse (a constant encoder is stage 2's degenerate optimum) ===")
        zc = zt - zt.mean(0, keepdim=True)
        ev = torch.linalg.eigvalsh(((zc.T @ zc) / max(len(zc) - 1, 1)).double()).clamp(min=1e-12)
        pr_ = ev / ev.sum()
        erank = float(torch.exp(-(pr_ * pr_.log()).sum()))
        d_con = float((zt[1:] - zt[:-1]).abs().mean())
        d_rnd = float((zt - zt[torch.randperm(len(zt))]).abs().mean())
        print(f"  latent std mean {float(zt.std(0).mean()):.4f} min {float(zt.std(0).min()):.5f} | "
              f"effective rank {erank:.1f}/{a.latent} | move/rand {d_con/max(d_rnd,1e-9):.3f}")
        bc_top1 = float((enc.step(zt)[0].argmax(-1) == A[it]).float().mean())
        print(f"\n=== CONTROL 2 (half): BC top-1 {bc_top1:.4f} with leak fields "
              f"{'ZEROED' if a.drop_leak else 'PRESENT'} -- run the other arm to get the leak ===")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "norm": norm.state_dict(),
                "split": split, "latent": a.latent, "link_dim": T["link"].shape[1],
                "row_dim": T["rows"].shape[2], "n_actions": meta["n_actions"], "decode_scale": scale,
                "reward_only": True, "control_echo": False, "drop_leak": a.drop_leak,
                "enc_out": "ln", "enc_shallow": False, "enc_kind": "mlp",
                "config": vars(a), "history": hist,
                "metrics": {"bc_top1": bc_top1, "effective_rank": erank, "move_ratio": d_con / max(d_rnd, 1e-9),
                            "fields": {r[0]: {"value": r[1], "floor": r[2], "beats_floor": bool(r[3])}
                                       for r in rows}}}, a.out)
    print(f"\n-> {a.out}")
    print(json.dumps({r[0]: round(r[1], 4) for r in rows}))


if __name__ == "__main__":
    main()
