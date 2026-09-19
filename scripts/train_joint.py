"""Train an encoder to be decodable AND predictable at once. No behaviour cloning.

    .venv/bin/python scripts/train_joint.py --epochs 10        # smoke test

Three parts trained together from scratch:

    z_t      = E(s_t)                       encoder
    fields   = D(z_t)                       decoder -> the raw state the reward function reads
    z_hat    = M(z_t, a_t)                  predictor

    loss = w_decode * decode(D(z_t), s_t)  +  w_pred * MSE(M(z_t, a_t), sg[E(s_{t+1})])

The diagnosis this responds to: a behaviour-cloned encoder was never asked to represent what an action
*changes*, only which action to take, so its transitions carried almost no action signal (inverse-dynamics
probe 0.059 from a latent pair vs 0.056 from a single latent). Here the predictive objective shapes the
encoder directly, while the decode term holds on to the fields the reward depends on.

Two design notes:
  * **the decoder is the anti-collapse mechanism.** A latent that must reconstruct HP, positions and enemy
    behaviour cannot go constant, so no EMA teacher or stop-gradient pair is needed -- unlike a pure
    predictive objective, where a constant encoder is the global optimum.
  * the prediction **target is stop-gradiented**, so the encoder cannot cheat by making its own future
    easier to predict (smoothing itself), which is the degenerate direction observed when it could.
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
from zelda_rl.bc import (LINK_HP_VALUES, MELEE_STATE_VALUES, N_LINK_ACTIONS, ROW_HP_VALUES,  # noqa: E402
                         BCNet, Norm,
                         StateDecoder, decode_targets,
                         featurise, hp_class)
from scripts.train_bc_decode import DEFAULT_DATA, CONT, load_many, transition_frames  # noqa: E402

DYING = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="extra collection runs to fold in, e.g. the room-exit episodes")
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt", help="normaliser + slot split only")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--unroll", type=int, default=1,
                    help="steps to roll the predictor on its OWN output during training. Single-step "
                         "training is teacher forcing: every input is a real encoder output, so the model "
                         "is never penalised for error it feeds itself, and it collapses past ~10 steps.")
    ap.add_argument("--unroll-detach", type=int, default=8,
                    help="cut the gradient every N unroll steps: the model still consumes its own output "
                         "for the full length, but backprop depth stays bounded")
    ap.add_argument("--unroll-decay", type=float, default=0.97,
                    help="weight step j by decay^j, so late steps inform without dominating")
    ap.add_argument("--unroll-batch", type=int, default=96,
                    help="how many of each batch to unroll; cost is linear in unroll length")
    ap.add_argument("--w-decode", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-hp", type=float, default=3.0)
    ap.add_argument("--w-pos", type=float, default=3.0)
    ap.add_argument("--w-v2", type=float, default=1.0,
                    help="weight on the v2 Link body/collision/action heads. Lower it to test whether a "
                         "fixed-width latent is spending capacity on them at the expense of the transition.")
    ap.add_argument("--w-scene", type=float, default=2.0,
                    help="weight on the in-room / left-room head; it drives the -20 room-exit terminal")
    ap.add_argument("--whiten", action="store_true",
                    help="standardise each latent dim before the prediction loss, so a dim holding 0.5%% of "
                         "the variance counts as much as one holding 60%%")
    ap.add_argument("--patience", type=int, default=6, help="early stop after this many epochs with no gain")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="checkpoints/dynamics/joint.pt")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ref = torch.load(a.ref, weights_only=False, map_location="cpu")
    norm, split = Norm.load(ref["norm"]), ref["split"]
    meta, state, action, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    if a.extra_data:
        print(f"data: {a.data} + {len(a.extra_data)} extra run(s) -> {len(state):,} frames")
    Y = decode_targets(state, obs)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    Fraw = featurise(state)
    act = torch.from_numpy(action)
    sc = {k: (Y[k].mean(0, keepdims=True), Y[k].std(0, keepdims=True) + 1e-6) for k in CONT}
    Yt = {k: torch.from_numpy(((Y[k] - sc[k][0]) / sc[k][1]).astype(np.float32)) if k in CONT
          else torch.from_numpy(Y[k]) for k in Y}

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    def starts(mask, k):
        ok = mask.copy()
        for j in range(1, k + 1):
            ok &= (np.r_[epid[j:], np.full(j, -1)] == epid)
        return np.where(ok)[0]
    K = a.stride
    # The decoder can learn from every frame; the predictor needs a successor. The room-exit frame is the
    # LAST of its episode, so a shared pair-index silently excludes exactly the label we collected.
    # Exits only occur on training slots (7 and 9 never reach a door), so the slot split cannot measure
    # the scene head. Hold out a quarter of the exit EPISODES instead, and report recall on those.
    exit_ep = np.unique(epid[Y["scene"] != 0])
    exit_eval_ep = exit_ep[::4]
    tr_mask = ~np.isin(slot, list(test | tune)) & ~np.isin(epid, exit_eval_ep)
    tr_all = np.where(tr_mask)[0]

    # The reference checkpoint's Norm was fitted on the v1 feature width. v2 adds Link's body/collision
    # block, so refit -- on the TRAINING frames only, or test statistics leak into the standardisation.
    if norm.link_mean.shape[0] != Fraw["link"].shape[1]:
        print(f"  refitting Norm: ref has {norm.link_mean.shape[0]} link features, data has "
              f"{Fraw['link'].shape[1]} (v2 block added); fitted on {len(tr_all):,} training frames")
        norm = Norm.fit({k: v[tr_all] for k, v in Fraw.items()})
    T = {k: torch.from_numpy(v) for k, v in norm.apply(Fraw).items()}

    tr = starts(tr_mask, K)
    has_next = np.zeros(len(slot), bool); has_next[tr] = True
    deep = starts(tr_mask, K * a.unroll)            # frames with the whole unroll window inside an episode
    print(f"unroll {a.unroll} (detach every {a.unroll_detach}, decay {a.unroll_decay}): "
          f"{len(deep):,} usable starts")
    tu = starts(np.isin(slot, list(tune)), K)
    te = starts(np.isin(slot, list(test)), 10 * K)
    print(f"stride {K} ({K*100} ms) | train {len(tr):,} tune {len(tu):,} test {len(te):,} | "
          f"w_decode {a.w_decode} w_pred {a.w_pred} | whiten {a.whiten}")

    enc = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=meta["n_actions"],
                latent=a.latent, gru=0)
    dec = StateDecoder(latent=a.latent)
    NA = meta["n_actions"]
    pred = nn.Sequential(nn.Linear(a.latent + NA, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, a.latent))
    A = F.one_hot(act, NA).float()
    params = list(enc.parameters()) + list(dec.parameters()) + list(pred.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)

    from zelda_rl.combat import BABA_STATES
    HIDDEN = BABA_STATES.index("hidden")
    VIS = torch.from_numpy((Y["row_state"] != HIDDEN).astype(np.float32))
    OBM = torch.from_numpy(has_obs.astype(np.float32))      # exit runs were saved without obs
    V2M = torch.from_numpy(Y["v2_mask"].astype(np.float32))  # v1 episodes carry no Link body/collision state
    MELEE = torch.from_numpy(Y["link_melee"].astype(np.int64))
    LACT = torch.from_numpy(Y["link_action"].astype(np.int64))
    print(f"obs available on {has_obs.mean():.1%} of frames (camera yaw + Link velocity come from it)")
    pos = Y["link_sf1"][tr_all].mean(0)
    SF1_PW = torch.from_numpy(((1 - pos) / np.maximum(pos, 1e-4)).clip(1, 50).astype(np.float32))
    pres_m = Y["row_present"] > 0.5
    def cw(y, n, mask=None):
        c = np.bincount(y[tr_all].ravel() if mask is None else y[tr_all][mask[tr_all]].ravel(),
                        minlength=n).astype(float)
        w = (c.sum() / np.maximum(c, 1)) ** 0.5
        return torch.tensor((w / w.mean()).astype(np.float32))
    CW = {"link_hp": cw(Y["link_hp_c"], 7), "row_hp": cw(Y["row_hp_c"], 3, pres_m),
          # a swing is live in a small minority of frames, so weight the melee classes by frequency
          "link_melee": cw(Y["link_melee"], len(MELEE_STATE_VALUES), Y["v2_mask"] > 0),
          # jump_attack, hurt and knocked_air are each under 5% of frames but are what the dream needs
          "link_action": cw(Y["link_action"], N_LINK_ACTIONS, Y["v2_mask"] > 0)}
    # Leaving the room is a scene change and it is the rarest label in the data -- 27 frames in 154,805.
    # Unweighted it is simply never predicted, and the -20 terminal can never fire inside a dream.
    n_sc = np.bincount(Y["scene"][tr_all], minlength=2).astype(float)
    w_sc = (n_sc.sum() / np.maximum(n_sc, 1)) ** 0.5           # sqrt: a raw 3633:1 ratio zeroes class 0
    CW["scene"] = torch.tensor((w_sc / w_sc.mean()).astype(np.float32))
    print(f"scene class counts (train): {n_sc.astype(int).tolist()}  "
          f"weights {CW['scene'].numpy().round(2)}  | held-out exit episodes: {len(exit_eval_ep)}")
    wl = torch.from_numpy(1 + 20.0 * transition_frames(Y["link_hp_c"], epid).astype(np.float32))
    wr = torch.from_numpy(1 + 20.0 * transition_frames(Y["row_hp_c"], epid).astype(np.float32))

    def E(i):
        return enc.encode(T["link"][i], T["rows"][i], T["ids"][i], T["mask"][i])

    def decode_loss(p, i):
        m, mb = Yt["row_present"][i], Yt["row_is_baba"][i]
        hp_l = (F.cross_entropy(p["link_hp"], Yt["link_hp_c"][i], weight=CW["link_hp"],
                                reduction="none") * wl[i]).mean()
        rh = (F.cross_entropy(p["row_hp"].reshape(-1, 3), Yt["row_hp_c"][i].reshape(-1), weight=CW["row_hp"],
                              reduction="none").reshape(m.shape) * m * wr[i])
        L = (a.w_pos * F.mse_loss(p["link_pos"], Yt["link_pos"][i])
             + F.mse_loss(p["link_yaw"], Yt["link_yaw"][i])
             + F.binary_cross_entropy_with_logits(p["link_bits"], Yt["link_bits"][i])
             + F.cross_entropy(p["targeted"], Yt["targeted"][i])
             + F.binary_cross_entropy_with_logits(p["row_present"], m)
             + F.binary_cross_entropy_with_logits(p["row_is_baba"], mb))
        L = L + a.w_pos * (((p["row_pos"] - Yt["row_pos"][i]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        ce = F.cross_entropy(p["row_state"].reshape(-1, p["row_state"].shape[-1]),
                             Yt["row_state"][i].reshape(-1), reduction="none").reshape(m.shape)
        L = L + (ce * mb).sum() / mb.sum().clamp(min=1)
        # live-Baba count as one head: the timeout penalty and the clear condition both read it, and a
        # single 4-way choice is far steadier than counting 16 independent per-row predictions.
        L = L + F.cross_entropy(p["baba_count"], Yt["baba_count"][i])
        # Orientation and timing -- the lunge telegraph. The facing angle is only defined while the Baba
        # is OUT of the ground: 64.7% of Baba frames are `hidden`, where shape_yaw is arbitrary and the
        # angle is near-uniform (R=0.12). Fitting it there is fitting noise, and it swamped the 35% that
        # carries signal (R=0.84-0.99 in every visible state).
        vis = mb * VIS[i]
        for k, w in (("row_facing_link", 1.0), ("link_facing_target", 1.0)):
            L = L + w * (((p[k] - Yt[k][i]) ** 2).mean(-1) * vis).sum() / vis.sum().clamp(min=1)
        for k, w in (("row_timer", 1.0), ("row_vel", 0.3)):
            L = L + w * (((p[k] - Yt[k][i]) ** 2).mean(-1) * mb).sum() / mb.sum().clamp(min=1)
        L = L + F.binary_cross_entropy_with_logits(p["link_sf1"], Yt["link_sf1"][i], pos_weight=SF1_PW)
        om = OBM[i]                                          # camera / velocity only where obs exists
        for k in ("cam_yaw", "link_vel"):
            L = L + (((p[k] - Yt[k][i]) ** 2).mean(-1) * om).sum() / om.sum().clamp(min=1)
        # v2: what Link is DOING and TOUCHING. Masked to v2 episodes, so a v1 episode contributes nothing
        # to these heads rather than teaching them zeros.
        vm = V2M[i]
        if float(vm.sum()) > 0 and a.w_v2 > 0:
            den = vm.sum().clamp(min=1) / a.w_v2
            for k, w in (("link_speed", 1.0), ("link_wall_yaw", 1.0), ("link_floor_dy", 0.5),
                         ("link_shape_yaw", 1.0), ("link_parallel_yaw", 0.5), ("link_fall", 0.3)):
                L = L + w * (((p[k] - Yt[k][i]) ** 2).mean(-1) * vm).sum() / den
            for k in ("link_bg", "link_ztarget", "link_airborne"):
                bce = F.binary_cross_entropy_with_logits(p[k], Yt[k][i], reduction="none")
                L = L + (bce.mean(-1) * vm).sum() / den
            ce_m = F.cross_entropy(p["link_melee"], MELEE[i], weight=CW["link_melee"], reduction="none")
            L = L + (ce_m * vm).sum() / den
            ce_a = F.cross_entropy(p["link_action"], LACT[i], weight=CW["link_action"], reduction="none")
            L = L + (ce_a * vm).sum() / den
        L = L + a.w_scene * F.cross_entropy(p["scene"], Yt["scene"][i], weight=CW["scene"])
        L = L + 0.1 * F.cross_entropy(p["room"], Yt["room"][i])
        L = L + a.w_hp * (hp_l + rh.sum() / (m * wr[i]).sum().clamp(min=1))
        p0, pdy = p["row_hp"].softmax(-1)[..., 0], p["row_state"].softmax(-1)[..., DYING]
        return L + 0.5 * (p0 * p["row_present"].sigmoid() * (1 - pdy) * mb).mean()

    mu = sd = None
    best_skill, best_pm, best_state, bad = -1e9, None, None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        enc.train(); dec.train(); pred.train()
        perm = rng.permutation(tr_all)
        td = tp = n = 0.0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            z = E(i)
            ld = decode_loss(dec(z), i)                          # every frame, exit frames included
            if a.unroll > 1:
                zi = torch.from_numpy(rng.choice(deep, min(a.unroll_batch, len(deep)), replace=False))
                zz, lp, wsum = E(zi), torch.zeros(()), 0.0
                for j in range(a.unroll):
                    zz = pred(torch.cat([zz, A[zi + j * K]], -1))
                    with torch.no_grad():
                        zt = E(zi + (j + 1) * K)                 # stop-gradient target, as in JEPA
                    w = a.unroll_decay ** j
                    lp = lp + w * F.mse_loss(zz, zt)
                    wsum += w
                    if a.unroll_detach and (j + 1) % a.unroll_detach == 0:
                        zz = zz.detach()                         # bound backprop depth, keep the exposure
                lp = lp / max(wsum, 1e-9)
            else:
                nx = torch.from_numpy(has_next[i.numpy()])
                if nx.any():
                    zi, zp = i[nx], z[nx]
                    with torch.no_grad():
                        zt = E(zi + K)
                    zh = pred(torch.cat([zp, A[zi]], -1))
                    lp = F.mse_loss(zh, zt)
                else:
                    lp = torch.zeros((), requires_grad=True)
            loss = a.w_decode * ld + a.w_pred * lp
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            td += ld.item(); tp += lp.item(); n += 1
        enc.eval(); dec.eval(); pred.eval()
        with torch.no_grad():
            zt = E(torch.from_numpy(tu)); zn = E(torch.from_numpy(tu + K))
            mu, sd = zt.mean(0, keepdim=True), zt.std(0, keepdim=True) + 1e-6
            zh = pred(torch.cat([zt, A[tu]], -1))
            f = (lambda x, y: F.mse_loss((x - mu) / sd, (y - mu) / sd)) if a.whiten else F.mse_loss
            pm, im = float(f(zh, zn)), float(f(zt, zn))
        skill = 1 - pm / im
        if skill > best_skill + 1e-4:
            best_skill, best_pm, bad = skill, pm, 0
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                          "dec": {k: v.clone() for k, v in dec.state_dict().items()},
                          "pred": {k: v.clone() for k, v in pred.state_dict().items()}}
        else:
            bad += 1
        if not a.quiet:
            print(f"  ep {ep:>3}/{a.epochs}  decode {td/n:.4f}  pred {tp/n:.5f} | tune pred {pm:.5f} "
                  f"identity {im:.5f}  skill {skill:+.3f}{'  *' if bad == 0 else ''}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience:
            break
    enc.load_state_dict(best_state["enc"]); dec.load_state_dict(best_state["dec"])
    pred.load_state_dict(best_state["pred"])
    print(f"  best tune skill {best_skill:+.3f} (pred {best_pm:.5f})")

    # ---------------- probes
    print("\n--- probes (test slots) ---")
    with torch.no_grad():
        Zte = E(torch.from_numpy(te)); Zte2 = E(torch.from_numpy(te + K))
        p = dec(Zte)
    m, mb = Y["row_present"][te] > .5, Y["row_is_baba"][te] > .5
    ho = np.where(np.isin(epid, exit_eval_ep))[0]
    with torch.no_grad():
        p_ho = dec(E(torch.from_numpy(ho)))
    sc_ho = p_ho["scene"].argmax(-1).numpy(); sc_ho_true = Y["scene"][ho]
    mbt = Y["row_is_baba"][te] > 0.5
    ang = lambda a_, b_: float(np.degrees(np.abs(np.arctan2(a_[..., 0], a_[..., 1])
                                                 - np.arctan2(b_[..., 0], b_[..., 1]))).clip(0, 360).mean())
    vist = mbt & (Y["row_state"][te] != HIDDEN)
    face_err = ang(p["row_facing_link"].numpy()[vist], Y["row_facing_link"][te][vist])
    lface_err = ang(p["link_facing_target"].numpy()[vist], Y["link_facing_target"][te][vist])
    timer_mae = float(np.abs(p["row_timer"].numpy()[mbt] - Y["row_timer"][te][mbt]).mean() * 100)
    sc_pred = p["scene"].argmax(-1).numpy(); sc_true = Y["scene"][te]
    fields = {"scene_recall": float((sc_pred[sc_true != 0] != 0).mean()) if (sc_true != 0).any() else float("nan"),
              "scene_acc": float((sc_pred == sc_true).mean()),
              "baba_count": float((p["baba_count"].argmax(-1).numpy() == Y["baba_count"][te]).mean()),
              "link_hp": float((p["link_hp"].argmax(-1).numpy() == Y["link_hp_c"][te]).mean()),
              "row_hp": float((p["row_hp"].argmax(-1).numpy()[m] == Y["row_hp_c"][te][m]).mean()),
              "row_state": float((p["row_state"].argmax(-1).numpy()[mb] == Y["row_state"][te][mb]).mean()),
              "link_pos_mae": float(np.abs(p["link_pos"].numpy() * sc["link_pos"][1] + sc["link_pos"][0]
                                           - Y["link_pos"][te]).mean())}
    import collections
    bc_major = collections.Counter(Y["baba_count"][te].tolist()).most_common(1)[0][1] / len(te)
    ho_rec = float((sc_ho[sc_ho_true != 0] != 0).mean()) if (sc_ho_true != 0).any() else float("nan")
    ho_fp = float((sc_ho[sc_ho_true == 0] != 0).mean())
    fields["exit_recall_heldout"] = ho_rec; fields["exit_false_pos"] = ho_fp
    print(f"decode: LEFT-ROOM on {len(exit_eval_ep)} held-out exit episodes -> "
          f"recall {ho_rec:.3f} on {int((sc_ho_true != 0).sum())} exit frames, "
          f"false-positive rate {ho_fp:.5f} on the {int((sc_ho_true == 0).sum()):,} in-room frames")
    from zelda_rl.bc import SF1_NAMED
    from zelda_rl.obs import SF1_BITS
    nm = {b: n for n, b in SF1_BITS}
    sp = (p["link_sf1"].numpy() > 0); stt = Y["link_sf1"][te] > 0.5
    parts = [f"{nm.get(b, b)} {float((sp[:, j] == stt[:, j]).mean()):.3f}"
             f"/{float(sp[stt[:, j], j].mean()) if stt[:, j].any() else float('nan'):.2f}"
             for j, b in enumerate(SF1_NAMED) if stt[:, j].mean() > 0.001]
    ov = has_obs[te]
    if ov.any():
        cy = p["cam_yaw"].numpy()[ov]; cyt = Y["cam_yaw"][te][ov]
        _d = np.degrees(np.abs(np.arctan2(cy[:, 0], cy[:, 1]) - np.arctan2(cyt[:, 0], cyt[:, 1])))
        cerr = float(np.minimum(_d, 360 - _d).mean())        # wrap: an angular error is at most 180 deg
        verr = float(np.abs(p["link_vel"].numpy()[ov] - Y["link_vel"][te][ov]).mean())
        print(f"        CAMERA/MOTION: camera yaw err {cerr:.1f} deg | Link velocity MAE {verr:.3f} "
              f"(obs units, |v| up to 4)")
    print(f"        LINK STATE (acc/recall): {'  '.join(parts)}")
    print(f"        TELEGRAPH (visible Babas only, {int(vist.sum()):,} frames): baba-faces-Link "
          f"{face_err:.1f} deg | Link-faces-baba {lface_err:.1f} deg | timer MAE {timer_mae:.1f}")
    print(f"        baba_count {fields['baba_count']:.4f} (majority {bc_major:.4f})  "
          f"link_hp {fields['link_hp']:.4f}  row_hp {fields['row_hp']:.4f}  "
          f"row_state {fields['row_state']:.4f}  link_pos MAE {fields['link_pos_mae']:.2f}")

    # --- v2: what Link is DOING and TOUCHING. Every head is reported against the majority-class baseline,
    # because most of these are rare: airborne is 10% of frames, a live swing 14%, knocked-airborne 0.1%.
    vmask = Y["v2_mask"][te] > 0
    if vmask.any():
        from zelda_rl.bc import BG_BITS, LINK_ACTION_NAMES
        bgn = ("ground", "grnd_touch", "grnd_leave", "wall", "ceiling", "water", "grnd_strict",
               "crushed", "player_wall")
        bgp = (p["link_bg"].numpy() > 0)[vmask]; bgt = (Y["link_bg"][te] > 0.5)[vmask]
        bits = [f"{bgn[j]} {float((bgp[:, j] == bgt[:, j]).mean()):.3f}/"
                f"{float(bgp[bgt[:, j], j].mean()) if bgt[:, j].any() else float('nan'):.2f}"
                for j in range(len(BG_BITS)) if bgt[:, j].mean() > 0.0005]
        print(f"        COLLISION (acc/recall): {'  '.join(bits)}")

        def acc(key, tgt):
            pr = p[key].numpy().argmax(-1)[vmask]; tt = tgt[te][vmask]
            maj = float(np.bincount(tt, minlength=p[key].shape[-1]).max() / max(len(tt), 1))
            return float((pr == tt).mean()), maj, pr, tt
        a_acc, a_maj, apr, att = acc("link_action", Y["link_action"])
        m_acc, m_maj, _, _ = acc("link_melee", Y["link_melee"])
        print(f"        ACTION STATE: link_action {a_acc:.4f} (majority {a_maj:.4f}, "
              f"{len(set(att.tolist()))} classes present)  melee {m_acc:.4f} (majority {m_maj:.4f})")
        # per-class recall on the states the dream was blind to
        # the input-verified states; the dream was blind to all of these before state v2
        WATCH = ("attack", "roll", "shield", "airborne", "airborne_z", "hurt",
                 "hurt_still", "hurt_airborne", "falling", "locomotion")
        rec = []
        for name in WATCH:
            if name not in LINK_ACTION_NAMES:
                continue
            k = LINK_ACTION_NAMES.index(name)
            if (att == k).any():
                rec.append(f"{name} {float((apr[att == k] == k).mean()):.2f} (n={int((att == k).sum()):,})")
        print("          recall: " + "  ".join(rec))
        for key, lbl, unit in (("link_speed", "speed", "x10 world u/step"), ("link_floor_dy", "floor_dy", "x100"),
                               ("link_fall", "fall_dist", "x100")):
            e = float(np.abs(p[key].numpy()[vmask] - Y[key][te][vmask]).mean())
            print(f"          {lbl} MAE {e:.4f} ({unit})")
        for key, lbl, m in (("link_shape_yaw", "shape yaw", None),
                            ("link_wall_yaw", "wall yaw", (Y["link_bg"][te][:, 3] > 0.5)),
                            ("link_parallel_yaw", "parallel yaw", (Y["link_ztarget"][te][:, 0] > 0.5))):
            sel = vmask if m is None else (vmask & m)     # wall yaw only means anything while touching one
            if not sel.any():
                continue
            q, qt = p[key].numpy()[sel], Y[key][te][sel]
            _d = np.degrees(np.abs(np.arctan2(q[:, 0], q[:, 1]) - np.arctan2(qt[:, 0], qt[:, 1])))
            print(f"          {lbl} median err {float(np.median(np.minimum(_d, 360 - _d))):.1f} deg "
                  f"(n={int(sel.sum()):,})")
        zp = (p["link_ztarget"].numpy()[:, 0] > 0)[vmask]; zt_ = (Y["link_ztarget"][te][:, 0] > 0.5)[vmask]
        ap_ = (p["link_airborne"].numpy()[:, 0] > 0)[vmask]; at_ = (Y["link_airborne"][te][:, 0] > 0.5)[vmask]
        print(f"          z_target acc {float((zp == zt_).mean()):.4f} recall {float(zp[zt_].mean()):.3f} | "
              f"airborne acc {float((ap_ == at_).mean()):.4f} recall {float(ap_[at_].mean()):.3f} "
              f"(base rate {float(at_.mean()):.3f})")
    print(f"  (BC encoder: 0.9839 / 0.9073 / 0.9992 / 4.22)")
    zc = Zte - Zte.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(zc.double()) ** 2; ev = ev / ev.sum()
    print(f"spread: top PC {float(ev[0]):.1%} of variance, PCs for 99%: {int((torch.cumsum(ev,0)<.99).sum())+1}"
          f"  (BC encoder: 61.5%, 63)")
    d = float(F.l1_loss(F.layer_norm(Zte, (a.latent,)), F.layer_norm(Zte2, (a.latent,))))
    r = float(F.l1_loss(F.layer_norm(Zte, (a.latent,)),
                        F.layer_norm(Zte[torch.randperm(len(Zte))], (a.latent,))))
    print(f"latent movement: consecutive {d:.4f}  random {r:.4f}  ratio {d/r:.3f}  (BC encoder: 0.099)")

    tr_i = starts(~np.isin(slot, list(test | tune)), K)
    with torch.no_grad():
        Ztr, Ztr2 = E(torch.from_numpy(tr_i)), E(torch.from_numpy(tr_i + K))
    f_tr = torch.cat([Ztr, Ztr2, Ztr2 - Ztr], -1); f_te = torch.cat([Zte, Zte2, Zte2 - Zte], -1)
    probe = nn.Sequential(nn.Linear(a.latent * 3, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                          nn.Linear(512, NA))
    o2 = torch.optim.Adam(probe.parameters(), 1e-3)
    for e in range(8):
        pm_ = rng.permutation(len(f_tr))
        for s in range(0, len(pm_), 1024):
            b = torch.from_numpy(pm_[s:s + 1024])
            o2.zero_grad(); F.cross_entropy(probe(f_tr[b]), act[tr_i][b]).backward(); o2.step()
    with torch.no_grad():
        lg = probe(f_te)
        t1 = float((lg.argmax(-1) == act[te]).float().mean())
    print(f"inverse dynamics (action from a latent pair): top-1 {t1:.4f}   "
          f"(BC encoder 0.059, single-latent 0.056, ceiling 0.220)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "pred": pred.state_dict(),
                # the norm ACTUALLY used for training, not ref["norm"]: with the v2 feature block the two
                # differ, and saving the reference silently mis-normalises every consumer of this checkpoint
                # (train_dream_ppo, play_dream_policy, the probes).
                "norm": norm.state_dict(), "split": split, "latent": a.latent, "link_dim": T["link"].shape[1],
                "row_dim": T["rows"].shape[2], "n_actions": NA, "decode_scale": {k: [v[0], v[1]] for k, v in sc.items()},
                "config": vars(a), "metrics": {"fields": fields, "inverse_top1": t1, "move_ratio": d / r,
                                               "skill": best_skill}}, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
