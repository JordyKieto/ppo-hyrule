"""Latent-space rollout vs state-space rollout, on Link's position, against ground truth. Minutes, no training.

    .venv/bin/python scripts/eval_rollout_space.py

Two ways to roll a dream forward from the same real start frame, using the SAME trained encoder, decoder and
transition head out of one checkpoint. Only the recursion differs:

  LATENT   z_{t+1} = M(z_t, a_t);  pos_t = decode(z_t).xyz
           what the dream does today. z is fed into itself for the whole episode.

  STATE    pos_{t+1} = pos_t + moved * mag * dir   from T(E(s_t), a_t)
           z_{t+1} = E(s_{t+1}), recomputed, never accumulated.

Scored as mean XZ error against Link's real position at the same horizon, with the do-nothing control (Link
never moves from the start frame) as the floor. If STATE does not beat LATENT by H=100 there is nothing here.

Only position is rolled. Every other state field is frozen at its step-0 value, which is exactly the known
limitation of the state-space proposal -- camera, velocity and collision have no heads yet -- so this is a
LOWER bound on the state arm, not a tuned result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import load_many  # noqa: E402
from zelda_rl.bc import (BCNet, LI, MOVED_THRESH, Norm, StateDecoder, UnifiedDecoder,  # noqa: E402
                         enc_arch, featurise, wants_v2)
from zelda_rl.dynamics import (TransitionHead, dir_unit_vectors, load_predictor,  # noqa: E402
                               mag_centers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ck", default="checkpoints/dynamics/wm_fu_dist.pt")
    ap.add_argument("--data", nargs="+", default=["datasets/transitions/v3-main/20260919-084052"])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 40, 100])
    ap.add_argument("--recalib", action="store_true",
                    help="refit the moved Platt pair on the rollout population (disjoint episodes)")
    ap.add_argument("--freeze-vel", action="store_true",
                    help="leave velocity/speed frozen at their step-0 values (the original arm)")
    a = ap.parse_args()

    ck = torch.load(a.ck, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0, **enc_arch(ck))
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = (UnifiedDecoder(latent=LAT) if ck.get("unified_decoder") else StateDecoder(latent=LAT))
    dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)
    thead = TransitionHead(LAT, NA, n_rows=16); thead.load_state_dict(ck["thead"]); thead.eval()
    norm = Norm.load(ck["norm"])
    v2 = wants_v2(ck["link_dim"])
    scale = ck["decode_scale"]
    mag_c, dir_u = mag_centers(), dir_unit_vectors()
    # The `moved` gate, copied from dream_state.apply_transition rather than re-derived: Platt-calibrate the
    # logit on event_calib["moved"], then SAMPLE. moved_thresh (2.838) is a LOGIT threshold, not a probability
    # -- comparing a sigmoid against it never fires, and Link never moves. Sampling, not scaling by P, because
    # E[dpos] = P * magnitude is systematically short for a discrete event (measured ~8x too small).
    ca, cb = (ck.get("event_calib") or {}).get("moved", (1.0, 0.0))
    ev_rng = np.random.default_rng(0)

    meta, state, act, slot, epid, obs, has = load_many(a.data)

    H = max(a.horizons)
    # one start per episode, far enough from the end to score the longest horizon
    eps = np.unique(epid)[: a.episodes]
    starts = []
    for e in eps:
        idx = np.flatnonzero(epid == e)
        if len(idx) > H + 1:
            starts.append(idx[0])
    starts = np.asarray(starts)
    print(f"{len(starts)} episodes, horizons {a.horizons}, checkpoint {Path(a.ck).name}")
    fit_starts, starts = starts[::2], starts[1::2]      # disjoint episodes: refit on one half, score the other
    print(f"  refit on {len(fit_starts)} episodes, scored on the other {len(starts)}")

    def Z(st):
        # control_echo MUST match training. train_joint defaults ECHO=True, and train_dream_ppo /
        # eval_dream_rollout both call featurise with the default -- passing False here zeroed 11 features the
        # encoder relies on and drove P(moved) to 0.021, which I then misread as a calibration bug in the gate.
        T = norm.apply(featurise(st, link_v2=v2))
        with torch.no_grad():
            return enc.encode(torch.from_numpy(T["link"]), torch.from_numpy(T["rows"]),
                              torch.from_numpy(T["ids"]), torch.from_numpy(T["mask"]))

    # POPULATION-MATCHED RECALIBRATION. The shipped Platt pair is fitted on the tuning slots, where Link's
    # true move rate is 0.119; we roll from episode starts, where it is 0.411. Correctly calibrated to the wrong
    # population. Refit a,b by BCE on held-out-from-scoring episodes and see whether firing at the right rate
    # makes the rollout walk. Refit on DISJOINT episodes so this is not fitted to what it is scored on.
    if a.recalib:
        lgs, ys = [], []
        with torch.no_grad():
            for t in range(H):
                zr = Z(state[fit_starts + t])
                ohr = F.one_hot(torch.from_numpy(act[fit_starts + t].astype(np.int64)).clamp(0, NA - 1),
                                NA).float()
                lgs.append(thead(zr, ohr)["tr_moved"].squeeze(-1).numpy())
                ys.append((np.linalg.norm(state[fit_starts + t + 1][:, [LI["pos_x"], LI["pos_z"]]]
                                          - state[fit_starts + t][:, [LI["pos_x"], LI["pos_z"]]], axis=1)
                           > MOVED_THRESH).astype(np.float32))
        lg_t = torch.from_numpy(np.concatenate(lgs)).float()
        y_t = torch.from_numpy(np.concatenate(ys)).float()
        pa, pb = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
        o = torch.optim.LBFGS([pa, pb], max_iter=120)
        def _cl():
            o.zero_grad()
            l = F.binary_cross_entropy_with_logits(pa * lg_t + pb, y_t)
            l.backward()
            return l
        o.step(_cl)
        with torch.no_grad():
            pc = torch.sigmoid(pa * lg_t + pb).numpy()
        print(f"  recalibrated on {len(y_t):,} frames: P = sigmoid({float(pa):.3f}*logit {float(pb):+.3f}) | "
              f"mean P {pc.mean():.3f} vs true rate {float(y_t.mean()):.3f} "
              f"(shipped pair gave mean P {(1/(1+np.exp(-(ca*lg_t.numpy()+cb)))).mean():.3f})")
        ca, cb = float(pa), float(pb)

    pos_i = [LI["pos_x"], LI["pos_y"], LI["pos_z"]]
    true0 = state[starts][:, pos_i].copy()
    # ground truth position at every horizon
    truth = {h: state[starts + h][:, pos_i].copy() for h in a.horizons}

    # ---- LATENT arm: z fed into itself; position read by decoding z each step
    zl = Z(state[starts]).clone()
    pm, ps = scale.get("link_pos", ([0, 0, 0], [1, 1, 1]))
    pm, ps = np.asarray(pm, np.float32), np.asarray(ps, np.float32)
    lat_err, lat_norm = {}, {}
    with torch.no_grad():
        for t in range(H):
            oh = F.one_hot(torch.from_numpy(act[starts + t].astype(np.int64)).clamp(0, NA - 1), NA).float()
            zl = pred(torch.cat([zl, oh], -1))
            if (t + 1) in truth:
                d = dec(zl)
                p = (d["link_pos"].numpy() if isinstance(d, dict) else d[0].numpy()) * ps + pm
                lat_err[t + 1] = np.linalg.norm(p[:, [0, 2]] - truth[t + 1][:, [0, 2]], axis=1)
                lat_norm[t + 1] = float(zl.norm(dim=-1).mean())

    # ---- STATE arm: position integrated from the head's events; z re-encoded from the state each step
    s = state[starts].copy()
    st_err, st_norm, pmoved_log = {}, {}, []
    with torch.no_grad():
        for t in range(H):
            z = Z(s)
            oh = F.one_hot(torch.from_numpy(act[starts + t].astype(np.int64)).clamp(0, NA - 1), NA).float()
            q = thead(z, oh)
            lg = q["tr_moved"].squeeze(-1).numpy()
            p_moved = 1.0 / (1.0 + np.exp(-(ca * lg + cb)))
            moved = ev_rng.random(len(p_moved)) < p_moved
            pmoved_log.append(float(p_moved.mean()))
            di = q["tr_dir"].argmax(-1).numpy()
            mi = q["tr_mag"].argmax(-1).numpy()
            step = dir_u[di] * mag_c[mi][:, None] * moved[:, None]        # (n, 2) in XZ world units
            s[:, LI["pos_x"]] += step[:, 0]
            s[:, LI["pos_z"]] += step[:, 1]
            s[:, LI["pos_y"]] += q["tr_dy"].squeeze(-1).numpy() * moved
            if not a.freeze_vel:
                # Velocity needs NO new head: it is the displacement we just applied. Without this the head
                # reads a Link whose velocity says "standing still" at every step, predicts no movement, and
                # the freeze is self-fulfilling -- measured P(moved) 0.021 against a true rate of ~0.17.
                s[:, LI["vel_x"]] = step[:, 0]
                s[:, LI["vel_z"]] = step[:, 1]
                s[:, LI["speed"]] = np.linalg.norm(step, axis=1)
            if (t + 1) in truth:
                st_err[t + 1] = np.linalg.norm(s[:, [LI["pos_x"], LI["pos_z"]]] - truth[t + 1][:, [0, 2]], axis=1)
                st_norm[t + 1] = float(Z(s).norm(dim=-1).mean())

    print(f"\n{'H':>5} {'do-nothing':>11} {'LATENT':>9} {'STATE':>9} {'winner':>8} "
          f"{'true moved':>11} {'|z| latent':>11} {'|z| state':>10}")
    # Is the gate low because the ROLLED state goes stale, or because it is miscalibrated for this use?
    # Same head, same frames, but reading the REAL state at each step separates the two.
    with torch.no_grad():
        pr = []
        for t in range(H):
            zr = Z(state[starts + t])
            ohr = F.one_hot(torch.from_numpy(act[starts + t].astype(np.int64)).clamp(0, NA - 1), NA).float()
            lgr = thead(zr, ohr)["tr_moved"].squeeze(-1).numpy()
            pr.append(float((1.0 / (1.0 + np.exp(-(ca * lgr + cb)))).mean()))
    # ground truth: did Link actually move between t and t+1, on these same frames
    true_rate = float(np.mean([np.linalg.norm(state[starts + t + 1][:, [LI["pos_x"], LI["pos_z"]]]
                                             - state[starts + t][:, [LI["pos_x"], LI["pos_z"]]], axis=1)
                               > MOVED_THRESH
                               for t in range(H)]))
    print(f"      P(moved): {np.mean(pmoved_log):.3f} on the ROLLED state | {np.mean(pr):.3f} on the REAL "
          f"state | {true_rate:.3f} actually moved (MOVED_THRESH={MOVED_THRESH})")
    for h in a.horizons:
        stay = np.linalg.norm(true0[:, [0, 2]] - truth[h][:, [0, 2]], axis=1).mean()
        le, se = lat_err[h].mean(), st_err[h].mean()
        moved_true = np.linalg.norm(truth[h][:, [0, 2]] - true0[:, [0, 2]], axis=1).mean()
        print(f"{h:>5} {stay:>11.1f} {le:>9.1f} {se:>9.1f} {('STATE' if se < le else 'latent'):>8} "
              f"{moved_true:>11.1f} {lat_norm[h]:>11.1f} {st_norm[h]:>10.1f}")
    print("\nnumbers are mean |XZ| error in world units against Link's real position at that horizon.")
    print("'do-nothing' = Link never leaves the start frame: the floor any real model must beat.")
    print("'true moved' = how far Link actually went, so an error above it means worse than not moving.")


if __name__ == "__main__":
    main()
