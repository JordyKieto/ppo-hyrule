"""Smoke test: how far does a PLAIN MLP get on one-step next-state prediction?

No attention, no tokens, no nav grid, no history: just concat(state[t], one_hot(action[t])) -> state[t+1],
trained on the flat 363-dim state vector the collector writes. It exists to put a floor under the
architectures in docs/WM_EXPERIMENT_REGIME.md -- if an MLP scores what they score, the tokens are not
paying for themselves.

Two design points make the comparison honest rather than flattering:

  * The head predicts a DELTA and its final layer is ZERO-INITIALISED, so an untrained model IS the
    persistence baseline and skill starts at exactly 0.000. Any number above zero was learned.
  * Scoring is per FIELD, not on the 363-dim mean. Most dimensions barely move at 10 Hz, so an aggregate
    MSE is dominated by fields where doing nothing is already correct, and a model that learns nothing
    at all reads as "close". Fields whose persistence error is ~0 on the eval split are reported as
    CONSTANT and excluded from the averages, because 1 - 0/0 is not a score.

The controls are persistence (predict no change) and best-constant-delta (predict the training mean
delta), which catches a model that has only learned the average drift.

usage: python scripts/smoke_mlp.py [--transitions 1500000] [--epochs 5] [--hidden 1024]
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.state_layout import ENEMY_FIELDS, N_ROWS, ROW_DIM, link_fields

# Field groups to score. name -> column indices, filled in once the state width is known.
def field_groups(width: int) -> dict[str, list[int]]:
    lf = link_fields(width)
    li = {n: i for i, n in enumerate(lf)}
    L = len(lf)
    g: dict[str, list[int]] = {}
    for n in ("pos_x", "pos_y", "pos_z"):
        g.setdefault("link_pos", []).append(li[n])
    for n, k in (("link_yaw", "yaw"), ("link_hp", "hp"), ("link_speed", "speed"),
                 ("link_action_fn", "action_fn_rel"), ("link_melee", "melee_state"),
                 ("link_bg_flags", "bg_flags"), ("link_floor_y", "floor_height"),
                 ("link_ztarget", "ztarget_timer"), ("link_cam_yaw", "cam_yaw")):
        if k in li:
            g[n] = [li[k]]
    g["link_vel"] = [li[n] for n in ("vel_x", "vel_y", "vel_z") if n in li]
    # enemy rows: every row's copy of the field, so one score per field across all 16 slots
    ei = {n: i for i, n in enumerate(ENEMY_FIELDS)}
    for n in ("present", "hp", "state_idx", "timer", "yaw_to_player"):
        g[f"enemy_{n}"] = [L + r * ROW_DIM + ei[n] for r in range(N_ROWS)]
    g["enemy_pos"] = [L + r * ROW_DIM + ei[n] for r in range(N_ROWS) for n in ("pos_x", "pos_y", "pos_z")]
    g["enemy_vel"] = [L + r * ROW_DIM + ei[n] for r in range(N_ROWS) for n in ("vel_x", "vel_y", "vel_z")]
    return g


def load(target: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Concatenate episodes until `target` transitions. Returns (states, actions, ep_id, n_actions).

    Episodes are kept contiguous and tagged with an id so the split can be by EPISODE: splitting by
    transition would put frame t in train and t+1 in val, which at 10 Hz is a near-duplicate.
    """
    files = sorted(glob.glob("datasets/transitions/v5/episodes/*.npz"))
    files += sorted(glob.glob("datasets/transitions/v5/rank*/episodes/*.npz"))
    random.Random(seed).shuffle(files)
    print(f"  {len(files):,} episode files available")

    S, A, E, tot = [], [], [], 0
    for k, f in enumerate(files):
        d = np.load(f)
        s, a = d["state"], d["action"]
        if len(s) != len(a) + 1 or len(a) < 2:
            continue
        S.append(s.astype(np.float32)); A.append(a.astype(np.int64))
        E.append(np.full(len(a), k, np.int32))
        tot += len(a)
        if tot >= target:
            break
    print(f"  loaded {len(S):,} episodes, {tot:,} transitions")
    # flat index: for episode i, transitions map to rows [off_i, off_i + T_i) of the stacked states
    return S, A, np.concatenate(E), int(max(int(a.max()) for a in A) + 1)


def flatten(S, A, E):
    """Stack per-episode arrays into (X_state, action, Y_next_state) with no episode-crossing pairs."""
    xs = np.concatenate([s[:-1] for s in S])
    ys = np.concatenate([s[1:] for s in S])
    ac = np.concatenate(A)
    return xs, ac, ys, E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=1_500_000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-action", action="store_true",
                   help="zero the action one-hot. The control for 'did it learn DYNAMICS or just "
                        "momentum?' -- state[t] alone already implies most of state[t+1] through velocity, "
                        "so a model can look good with the action channel dead.")
    p.add_argument("--loss", choices=("mse", "l1"), default="mse",
                   help="l1 trains toward the conditional MEDIAN. Most deltas are exactly zero, so the "
                        "median is zero and an L1-trained head can output exact no-change; MSE trains "
                        "toward the MEAN, which a rare spike drags off zero forever. The metric is MAE, "
                        "so mse/l1 is the objective-vs-metric question, not a tuning knob.")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {dev}")

    t0 = time.time()
    S, A, E, n_act = load(a.transitions, a.seed)
    X, AC, Y, E = flatten(S, A, E)
    del S, A
    W = X.shape[1]
    print(f"  state width {W}, {n_act} actions, load {time.time()-t0:.1f}s")

    # split by episode
    eps = np.unique(E)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * a.val_frac))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.fromiter((e in val_eps for e in E), bool, len(E))
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"  train {len(tr):,} transitions / {len(eps)-n_val:,} episodes | "
          f"val {len(va):,} / {n_val:,} episodes")

    # normalise inputs and DELTAS from the train split only
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.maximum(sd, 1e-6)
    D = Y; D -= X                                # the thing the head predicts, built in place over Y
    del Y
    dmu, dsd = D[tr].mean(0), D[tr].std(0)
    dsd = np.maximum(dsd, 1e-6)

    # in place, and torch.from_numpy shares the buffer: three float32 copies of 363 dims x millions of
    # rows does not fit in 17 GB, and a smoke test that OOMs the laptop teaches nothing (AGENTS.md 23).
    X -= mu; X /= sd
    D /= dsd                                     # scale only: the head must still predict zero for "no change"
    Xt, Dt, At = torch.from_numpy(X), torch.from_numpy(D), torch.from_numpy(AC)
    trt, vat = torch.from_numpy(tr), torch.from_numpy(va)

    net = nn.Sequential(nn.Linear(W + n_act, a.hidden), nn.GELU(),
                        nn.Linear(a.hidden, a.hidden), nn.GELU(),
                        nn.Linear(a.hidden, W)).to(dev)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)   # start AS persistence
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"  MLP {W}+{n_act} -> {a.hidden} -> {a.hidden} -> {W}   {nparam/1e6:.2f}M params, "
          f"{a.loss} loss, lr {a.lr}\n")

    def batch(idx):
        x = Xt[idx].to(dev)
        oh = torch.zeros(len(idx), n_act, device=dev)
        if not a.no_action:
            oh.scatter_(1, At[idx].to(dev).unsqueeze(1), 1.0)
        return torch.cat([x, oh], 1), Dt[idx].to(dev)

    groups = field_groups(W)
    dsd_t = torch.from_numpy(dsd).to(dev)

    @torch.no_grad()
    def evaluate():
        """Per-field MAE in the state's own units for: the MLP, persistence, best-constant-delta."""
        net.eval()
        n = torch.zeros(W, device=dev)
        err = torch.zeros(W, device=dev); per = torch.zeros(W, device=dev); con = torch.zeros(W, device=dev)
        cmean = torch.from_numpy(dmu).to(dev)
        pos_e = pos_p = 0.0
        li = {nm: i for i, nm in enumerate(link_fields(W))}
        pcols = [li["pos_x"], li["pos_y"], li["pos_z"]]
        for st in range(0, len(vat), a.batch):
            idx = vat[st:st + a.batch]
            xb, db = batch(idx)
            pred = net(xb) * dsd_t                      # predicted delta, real units
            true = db * dsd_t
            err += (pred - true).abs().sum(0)
            per += true.abs().sum(0)                    # persistence predicts zero delta
            con += (cmean - true).abs().sum(0)
            n += len(idx)
            # position as a WORLD DISTANCE, which is the metric eval_rollout_space uses
            pe = (pred[:, pcols] - true[:, pcols]).norm(dim=1).sum()
            pp = true[:, pcols].norm(dim=1).sum()
            pos_e += float(pe); pos_p += float(pp)
        net.train()
        return (err / n).cpu().numpy(), (per / n).cpu().numpy(), (con / n).cpu().numpy(), pos_e, pos_p

    def report(e, pr, co, pos_e, pos_p, tag):
        print(f"  {tag}")
        print(f"    {'field':<16} {'MLP err':>10} {'persist':>10} {'const-d':>10} {'skill':>8} {'vs const':>9}")
        rows, sk_all = [], []
        for nm, cols in groups.items():
            pe_, pp_, pc_ = e[cols].mean(), pr[cols].mean(), co[cols].mean()
            if pp_ < 1e-8:
                rows.append((nm, pe_, pp_, pc_, None, None)); continue
            sk = 1 - pe_ / pp_
            skc = 1 - pe_ / max(pc_, 1e-12)
            sk_all.append(sk)
            rows.append((nm, pe_, pp_, pc_, sk, skc))
        for nm, pe_, pp_, pc_, sk, skc in rows:
            if sk is None:
                print(f"    {nm:<16} {pe_:>10.4f} {pp_:>10.4f} {pc_:>10.4f} {'CONSTANT':>8} {'':>9}")
            else:
                print(f"    {nm:<16} {pe_:>10.4f} {pp_:>10.4f} {pc_:>10.4f} {sk:>+8.4f} {skc:>+9.4f}")
        # all 363 dims, excluding the ones nothing ever changes
        live = pr > 1e-8
        agg = 1 - e[live].sum() / pr[live].sum()
        print(f"    {'':<16} {'':>10} {'':>10} {'':>10}")
        print(f"    mean skill over {len(sk_all)} scored field groups: {np.mean(sk_all):+.4f}")
        print(f"    aggregate skill over {int(live.sum())} live dims of {len(pr)}: {agg:+.4f}")
        print(f"    POSITION, world distance: MLP {pos_e/max(pos_p,1e-9)*100:5.1f}% of persistence "
              f"-> skill {1 - pos_e/max(pos_p,1e-9):+.4f}")

    e, pr, co, pe_, pp_ = evaluate()
    report(e, pr, co, pe_, pp_, "epoch 0 (zero-init == persistence, so skill MUST be 0.000)")

    for ep in range(1, a.epochs + 1):
        t = time.time()
        perm = trt[torch.randperm(len(trt))]
        tot, nb = 0.0, 0
        for st in range(0, len(perm) - a.batch, a.batch):
            xb, db = batch(perm[st:st + a.batch])
            pred = net(xb)
            loss = (nn.functional.mse_loss(pred, db) if a.loss == "mse"
                    else nn.functional.l1_loss(pred, db))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        e, pr, co, pe_, pp_ = evaluate()
        print(f"\nepoch {ep}  train {a.loss} {tot/max(nb,1):.5f}  {time.time()-t:.0f}s")
        report(e, pr, co, pe_, pp_, "")


if __name__ == "__main__":
    main()
