"""JEPA stage 2: how good a predictive surface is the frozen stage-1 encoder?

    .venv/bin/python scripts/train_jepa_stage2.py --enc checkpoints/jepa/enc_s1_s0.pt

Plan: `docs/JEPA_PLAN.md`. The encoder AND decoder are frozen from stage 1 -- only `M` is trained here:

    z_{t+1} = M(z_t, a_t)          L1 on LAYER-NORMALISED targets, plus a rollout term

A frozen target is the stop-gradient, so there is no EMA copy and no encoder-collapse risk (I-JEPA's EMA
exists to stop a *trainable* encoder collapsing; ours cannot move). No decoder gradient reaches the encoder,
which DINO-WM 2411.04983 asks for and which is free here.

FOUR MEASUREMENTS, REPORTED TOGETHER, because skill alone is how VQ scored +0.000 with a predictor that had
provably learned to copy `z_t` (`AGENTS.md` §2):

  1. skill = 1 - err(M) / err(identity), on the same L1-on-layer-normalised metric the loss uses, and on raw
     MSE for comparability with every earlier number in LATENT_WORLD_MODEL.md.
  2. decode-of-predicted: the FROZEN stage-1 decoder run on M's output, scored on the reward fields against
     the same decoder run on the real latent. This is the number that matters -- a predicted latent is only
     useful if the reward can still be read out of it.
  3. ground-truth rollout in WORLD UNITS at several horizons against the do-nothing control (§8). The
     training loss is not a result.
  4. inverse dynamics: can the action be recovered from (z_t, z_{t+1})? This is the diagnostic that motivated
     JEPA in the first place -- the BC encoder scored 0.059 against a 0.056 single-latent floor and a 0.220
     ceiling, i.e. the transition carried almost no trace of what the action did.

DEVIATIONS FROM THE PAPERS, named here rather than buried (`AGENTS.md` §13.3):

  * RESIDUAL predictor, `z + f(z,a)`, not a direct map. Measured here as skill -7.24 -> -1.73, and it makes an
    untrained predictor exactly the identity, which is the control the skill metric is defined against.
  * STRIDE 1 by default, where the papers use stride > 1. Their rationale (adjacent frames are nearly
    identical) holds -- this game is unchanged in 74% of steps at 10 Hz -- but with action conditioning a_t is
    only the FIRST of `stride` actions, so stride > 1 predicts the consequence of an action sequence it was
    never shown. The dream also steps one frame at a time, so stride > 1 would mismatch deployment. Exposed as
    `--stride` so the paper's setting is one flag away, and reported when used.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import DEFAULT_DATA, load_many  # noqa: E402
from zelda_rl.bc import (LINK_HP_VALUES, N_BABA_STATES, ROW_HP_VALUES, BCNet, Norm,  # noqa: E402
                         UnifiedDecoder, decode_targets, enc_arch, featurise, hp_class)
from zelda_rl.dynamics import build_predictor  # noqa: E402


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc", default="checkpoints/jepa/enc_s1_s0.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stride", type=int, default=1, help="see the DEVIATIONS note; >1 breaks action conditioning")
    ap.add_argument("--unroll", type=int, default=8)
    ap.add_argument("--unroll-decay", type=float, default=0.97)
    ap.add_argument("--unroll-batch", type=int, default=256)
    ap.add_argument("--w-roll", type=float, default=1.0)
    ap.add_argument("--factored", action="store_true", help="factored action input instead of a flat one-hot")
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 40, 100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/jepa/pred_s2.pt")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.enc, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0, **enc_arch(ck))
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = UnifiedDecoder(latent=LAT, reward_only=ck.get("reward_only", True))
    dec.load_state_dict(ck["decoder"]); dec.eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    norm, split = Norm.load(ck["norm"]), ck["split"]

    meta, state, act, slot, epid, obs, has_obs = load_many([a.data, *a.extra_data])
    # control_echo=False, matching stage 1 exactly. Mismatching this silently zeroes 11 features the encoder
    # relies on -- it drove a P(moved) measurement to 0.021 and produced a retracted conclusion.
    Fn = norm.apply(featurise(state, link_v2=True, control_echo=ck.get("control_echo", False)))
    T = {k: torch.from_numpy(v) for k, v in Fn.items()}

    with torch.no_grad():
        Z = torch.cat([enc.encode(T["link"][i:i + 8192], T["rows"][i:i + 8192],
                                  T["ids"][i:i + 8192], T["mask"][i:i + 8192])
                       for i in range(0, len(state), 8192)])
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1), NA).float()

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    K = a.stride
    same = np.zeros(len(state), bool)
    same[:-K] = epid[K:] == epid[:-K]                       # t and t+K inside one episode

    def idx(mask, need):
        return np.flatnonzero(mask & same & np.r_[np.ones(len(state) - need, bool), np.zeros(need, bool)])

    tr = idx(~np.isin(slot, list(test | tune)), K)
    tu = idx(np.isin(slot, list(tune)), K)
    te = idx(np.isin(slot, list(test)), K)
    deep = idx(~np.isin(slot, list(test | tune)), K * a.unroll)
    print(f"{len(state):,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | latent {LAT} "
          f"| stride {K} | unroll {a.unroll} ({len(deep):,} starts)")

    pred = build_predictor(LAT, NA, factored=a.factored)
    opt = torch.optim.Adam(pred.parameters(), lr=a.lr)

    def step_loss(i):
        zp = pred(torch.cat([Z[i], A[i]], -1))
        return F.l1_loss(ln(zp), ln(Z[i + K]))               # L1 on layer-normalised targets

    for ep in range(a.epochs):
        pred.train()
        perm = rng.permutation(tr)
        t0, tl, nb = time.time(), 0.0, 0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            loss = step_loss(i)
            if a.unroll > 1 and a.w_roll > 0:
                zi = torch.from_numpy(rng.choice(deep, min(a.unroll_batch, len(deep)), replace=False))
                zz, rl, wsum = Z[zi], torch.zeros(()), 0.0
                for j in range(a.unroll):
                    zz = pred(torch.cat([zz, A[zi + j * K]], -1))
                    w = a.unroll_decay ** j
                    rl = rl + w * F.l1_loss(ln(zz), ln(Z[zi + (j + 1) * K]))
                    wsum += w
                loss = loss + a.w_roll * rl / max(wsum, 1e-9)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            opt.step()
            tl += float(loss.detach()); nb += 1
        pred.eval()
        with torch.no_grad():
            iu = torch.from_numpy(tu)
            zp = pred(torch.cat([Z[iu], A[iu]], -1))
            pm = float(F.l1_loss(ln(zp), ln(Z[iu + K])))
            im = float(F.l1_loss(ln(Z[iu]), ln(Z[iu + K])))
        print(f"  ep {ep+1:>3}/{a.epochs}  train {tl/nb:.5f} | tune {pm:.5f} identity {im:.5f} "
              f"skill {1-pm/max(im,1e-9):+.3f}  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- 1. skill, both metrics
    with torch.no_grad():
        it = torch.from_numpy(te)
        zp = pred(torch.cat([Z[it], A[it]], -1))
        s_ln = 1 - float(F.l1_loss(ln(zp), ln(Z[it + K]))) / max(float(F.l1_loss(ln(Z[it]), ln(Z[it + K]))), 1e-9)
        s_mse = 1 - float(F.mse_loss(zp, Z[it + K])) / max(float(F.mse_loss(Z[it], Z[it + K])), 1e-9)
    print(f"\n=== 1. SKILL on held-out slots ===\n  L1/layer-norm {s_ln:+.4f}   raw MSE {s_mse:+.4f}"
          f"   (0 = identical to 'assume no change')")

    # ---------------------------------------------------------------- 2. decode the PREDICTED latent
    Y = decode_targets(state, obs, epid=epid)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    sc = ck["decode_scale"]
    for k, (mu, sd) in sc.items():                          # the STORED scale, never refit
        Y[k] = ((Y[k] - np.asarray(mu)) / np.asarray(sd)).astype(np.float32)
    print("\n=== 2. DECODE OF THE PREDICTED LATENT vs of the real one (frozen stage-1 decoder) ===")
    with torch.no_grad():
        dp, dr = dec(zp), dec(Z[it + K])
        tgt = it.numpy() + K
        m = torch.from_numpy(Y["row_present"][tgt]) > 0.5
        mb = torch.from_numpy(Y["row_is_baba"][tgt]) > 0.5
        print(f"  {'field':<16}{'predicted':>11}{'real':>9}   metric")
        for name, key, k, msk in (("link_hp", "link_hp_c", len(LINK_HP_VALUES), None),
                                  ("targeted", "targeted", 17, None),
                                  ("baba_count", "baba_count", 4, None),
                                  ("row_hp", "row_hp_c", len(ROW_HP_VALUES), m),
                                  ("row_state", "row_state", N_BABA_STATES, mb)):
            lab = torch.from_numpy(np.asarray(Y[key])[tgt])
            pc, rc = dp[name].argmax(-1), dr[name].argmax(-1)
            if msk is not None:
                lab, pc, rc = lab[msk], pc[msk], rc[msk]
            print(f"  {name:<16}{float((pc==lab).float().mean()):>11.4f}"
                  f"{float((rc==lab).float().mean()):>9.4f}   accuracy")
        for name, world in (("link_pos", np.asarray(sc["link_pos"][1])),
                            ("near_dist", np.asarray(sc["near_dist"][1]) * 100.0),
                            ("row_pos", np.asarray(sc["row_pos"][1]))):
            t_ = torch.from_numpy(np.asarray(Y[name])[tgt])
            pv, rv = dp[name], dr[name]
            if t_.dim() == 3:
                t_, pv, rv = t_[m], pv[m], rv[m]
            w = float(np.mean(world))
            print(f"  {name:<16}{float((pv-t_).abs().mean())*w:>11.2f}{float((rv-t_).abs().mean())*w:>9.2f}"
                  f"   MAE, world units")

    # ---------------------------------------------------------------- 3. ground-truth rollout, world units
    H = max(a.horizons)
    eps = np.unique(epid)
    from zelda_rl.bc import LI
    all_starts = np.array([np.flatnonzero(epid == e)[0] for e in eps
                           if np.flatnonzero(epid == e).size > H + 1])
    held = all_starts[np.isin(slot[all_starts], list(test))]      # score on HELD-OUT episodes only
    starts, where = (held, "held-out") if len(held) >= 8 else (all_starts, "ALL (too few held-out)")
    pmu, psd = np.asarray(sc["link_pos"][0]), np.asarray(sc["link_pos"][1])
    pos_i = [LI["pos_x"], LI["pos_y"], LI["pos_z"]]
    truth = {h: state[starts + h][:, pos_i] for h in a.horizons}
    start_pos = state[starts][:, pos_i]
    print(f"\n=== 3. GROUND-TRUTH ROLLOUT from {len(starts)} real {where} start frames (world units, XZ) ===")
    print(f"  {'H':>5}{'do-nothing':>12}{'JEPA rollout':>14}{'true moved':>12}{'|z|':>8}")
    with torch.no_grad():
        zr = Z[torch.from_numpy(starts)].clone()
        for t in range(H):
            oh = A[torch.from_numpy(starts + t)]
            zr = pred(torch.cat([zr, oh], -1))
            if (t + 1) in truth:
                p = dec(zr)["link_pos"].numpy() * psd + pmu
                e_m = np.linalg.norm(p[:, [0, 2]] - truth[t + 1][:, [0, 2]], axis=1).mean()
                e_0 = np.linalg.norm(start_pos[:, [0, 2]] - truth[t + 1][:, [0, 2]], axis=1).mean()
                mv = np.linalg.norm(truth[t + 1][:, [0, 2]] - start_pos[:, [0, 2]], axis=1).mean()
                print(f"  {t+1:>5}{e_0:>12.1f}{e_m:>14.1f}{mv:>12.1f}{float(zr.norm(dim=-1).mean()):>8.1f}")

    # ---------------------------------------------------------------- 4. inverse dynamics
    probe = nn.Sequential(nn.Linear(LAT * 3, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                          nn.Linear(512, NA))
    po = torch.optim.Adam(probe.parameters(), lr=1e-3)
    av = torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1)

    def pair(i):
        return torch.cat([Z[i], Z[i + K], Z[i + K] - Z[i]], -1)

    for _ in range(3):
        pp = rng.permutation(tr)
        for s in range(0, len(pp) - a.batch, a.batch):
            i = torch.from_numpy(pp[s:s + a.batch])
            l = F.cross_entropy(probe(pair(i)), av[i])
            po.zero_grad(); l.backward(); po.step()
    with torch.no_grad():
        t1 = float((probe(pair(it)).argmax(-1) == av[it]).float().mean())
    print(f"\n=== 4. INVERSE DYNAMICS: action from (z_t, z_t+1) top-1 {t1:.4f} ===")
    print(f"  chance {1/NA:.4f} | BC encoder 0.059 | single-latent floor 0.056 | Bayes ceiling 0.220")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pred": pred.state_dict(), "pred_residual": True, "pred_factored": a.factored,
                "enc_from": a.enc, "latent": LAT, "n_actions": NA, "stride": K,
                "config": vars(a),
                "metrics": {"skill_ln": s_ln, "skill_mse": s_mse, "inverse_top1": t1}}, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
