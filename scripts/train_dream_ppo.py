"""PPO trained entirely inside the learned latent dynamics. The emulator is never touched.

    .venv/bin/python scripts/train_dream_ppo.py --transitions 100000

Reward comes from the event head (`scripts/train_event_head.py`): a calibrated distribution over damage,
kills, Link damage and terminal, read from the latent transition. It is NOT obtained by decoding HP and
subtracting. That earlier scheme was unsound in both directions -- clamping HP monotone turned model drift
into permanent unearned kills, and leaving it unclamped let a policy farm one enemy by steering into a
region where the decode oscillates. Expected reward under a bounded probability cannot exceed the real
payout for an event, however wrong the latent is.

Two further corrections over the first version:
  * episodes start from real **episode-start** frames, not a uniform sample of every frame. Sampling
    uniformly began fights midway with about two Babas already engaged, so "cleared the room" meant
    finishing a partial fight and could not be compared with a baseline measured over whole episodes.
  * the horizon is bounded. Decoded behaviour state holds to roughly ten steps and decays after, so
    dreaming 200 steps spent most of its time where the model cannot predict at all.
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
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, decode_targets, featurise,  # noqa: E402
                         hp_class)
from zelda_rl.ppo import ActorCritic, gae, ppo_update  # noqa: E402
from scripts.train_bc_decode import DEFAULT_DATA  # noqa: E402
from scripts.train_event_head import EventHead, labels, load_with_terminal  # noqa: E402

W_DMG, W_KILL, W_LINK, HP_UNIT, DEATH, SUCCESS, TIME_PEN = 1.0, 5.0, 0.25, 8.0, 20.0, 20.0, 0.001


class Dream:
    """Batched latent rollout. Reward and termination both come from the event head."""

    def __init__(self, pred, head, starts, n_actions, max_steps):
        self.pred, self.head, self.starts = pred, head, starts
        self.NA, self.max_steps = n_actions, max_steps

    @torch.no_grad()
    def reset(self, n, rng):
        self.z = self.starts[torch.from_numpy(rng.integers(0, len(self.starts), n))].clone()
        self.t = torch.zeros(n)
        return self.z

    @torch.no_grad()
    def reset_at(self, k, rng):
        self.z[k] = self.starts[torch.from_numpy(rng.integers(0, len(self.starts), len(k)))]
        self.t[k] = 0

    @torch.no_grad()
    def step(self, a):
        a1h = F.one_hot(a, self.NA).float()
        z2 = self.pred(torch.cat([self.z, a1h], -1))
        d, k, l, t = self.head(self.z, a1h, z2)
        e_d = (d.softmax(-1) * torch.arange(d.shape[-1], dtype=torch.float)).sum(-1)
        e_k = (k.softmax(-1) * torch.arange(k.shape[-1], dtype=torch.float)).sum(-1)
        pt = t.softmax(-1)
        r = (W_DMG * e_d + W_KILL * e_k - W_LINK * HP_UNIT * l.sigmoid() - TIME_PEN
             - DEATH * pt[..., 1] + SUCCESS * pt[..., 2])
        term = torch.multinomial(pt, 1).squeeze(-1)          # sampled, consistent with the expected reward
        self.z, self.t = z2, self.t + 1
        dead, cleared = term == 1, term == 2
        done = dead | cleared | (self.t >= self.max_steps)
        return self.z, r, done, {"dead": dead, "cleared": cleared, "e_kill": e_k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/tmp/joint_3e-4.pt")
    ap.add_argument("--event-head", default="checkpoints/dynamics/event_head.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--transitions", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=100, help="dream episode cap")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--out", default="checkpoints/dynamics/dream_ppo.pt")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    pred = nn.Sequential(nn.Linear(LAT + NA, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, LAT))
    pred.load_state_dict(ck["pred"]); pred.eval()
    hk = torch.load(a.event_head, weights_only=False, map_location="cpu")
    head = EventHead(LAT, NA, hk["config"]["hidden"])
    head.load_state_dict(hk["head"]); head.eval()
    for p in list(enc.parameters()) + list(pred.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    meta, state, frm, action, slot = load_with_terminal(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    L = labels(Y, frm, action)
    norm = Norm.load(ck["norm"])
    Fn = norm.apply(featurise(state))
    with torch.no_grad():
        Z = torch.cat([enc.encode(torch.from_numpy(Fn["link"][s:s + 8192]),
                                  torch.from_numpy(Fn["rows"][s:s + 8192]),
                                  torch.from_numpy(Fn["ids"][s:s + 8192]),
                                  torch.from_numpy(Fn["mask"][s:s + 8192]))
                       for s in range(0, len(state), 8192)])

    first = np.r_[True, np.diff(frm) != 1]                       # first transition of each episode
    held = list(set(ck["split"]["val_slots"]) | set(ck["split"]["tune_slots"]))
    ep_start = frm[first][~np.isin(slot[first], held)]
    starts = Z[torch.from_numpy(ep_start)]

    # real-data control: the SAME event reward, using true labels, per held-out episode
    te = np.isin(slot, list(set(ck["split"]["val_slots"])))
    r_real = (W_DMG * L["dmg"][te] + W_KILL * L["kill"][te] - W_LINK * HP_UNIT * L["link"][te] - TIME_PEN
              - DEATH * (L["term"][te] == 1) + SUCCESS * (L["term"][te] == 2))
    n_ep_te = max(int((L["term"][te] != 0).sum()), 1)
    base = float(r_real.sum() / n_ep_te)

    env = Dream(pred, head, starts, NA, a.max_steps)
    model = ActorCritic(LAT, NA, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    N, T = a.envs, a.steps
    updates = max(a.transitions // (N * T), 1)
    print(f"dreaming {updates} updates x {N} envs x {T} steps = {updates*N*T:,} transitions")
    print(f"start states: {len(starts)} real episode starts (training slots only)")
    print(f"real-data control, same event reward on TRUE labels: {base:.2f} per held-out episode\n")
    print(f"{'upd':>4} {'transitions':>12} {'return':>8} {'len':>6} {'clear%':>7} {'death%':>7} "
          f"{'E[kill]':>8} {'sps':>8}")
    print("-" * 70)

    obs = env.reset(N, rng)
    ep_r, ep_l = torch.zeros(N), torch.zeros(N)
    done_ret, done_len, ncl, nde, neps, ekill, hist = [], [], 0, 0, 0, 0.0, []
    t0 = time.time()
    for u in range(1, updates + 1):
        O = torch.zeros(T, N, LAT); A_ = torch.zeros(T, N, dtype=torch.long)
        LP = torch.zeros(T, N); V = torch.zeros(T, N); R = torch.zeros(T, N); D = torch.zeros(T, N)
        for t in range(T):
            with torch.no_grad():
                act_, lp, _, v = model.act(obs)
            O[t], A_[t], LP[t], V[t] = obs, act_, lp, v
            obs, r, done, info = env.step(act_)
            R[t], D[t] = r, done.float()
            ep_r += r; ep_l += 1
            ekill += float(info["e_kill"].sum())
            if done.any():
                k = done.nonzero(as_tuple=True)[0]
                done_ret += ep_r[k].tolist(); done_len += ep_l[k].tolist()
                ncl += int(info["cleared"][k].sum()); nde += int(info["dead"][k].sum()); neps += len(k)
                ep_r[k] = 0; ep_l[k] = 0
                env.reset_at(k, rng)
                obs = env.z
        with torch.no_grad():
            nv = model.value(obs)
        adv, ret = gae(R.numpy(), V.numpy(), D.numpy(), nv.numpy(), a.gamma, a.lam)
        ppo_update(model, opt, {"obs": O.reshape(-1, LAT).numpy(), "act": A_.reshape(-1).numpy(),
                                "logp": LP.reshape(-1).numpy(), "adv": adv.reshape(-1),
                                "ret": ret.reshape(-1), "val": V.reshape(-1).numpy()},
                   epochs=4, minibatches=8, clip=a.clip, ent_coef=a.ent, vf_coef=0.5, max_grad_norm=0.5)
        if u % max(updates // 20, 1) == 0 or u == 1:
            mr = float(np.mean(done_ret[-200:])) if done_ret else float("nan")
            ml = float(np.mean(done_len[-200:])) if done_len else float("nan")
            print(f"{u:>4} {u*N*T:>12,} {mr:>8.2f} {ml:>6.0f} {100*ncl/max(neps,1):>6.1f}% "
                  f"{100*nde/max(neps,1):>6.1f}% {ekill/(N*T):>8.4f} "
                  f"{u*N*T/(time.time()-t0):>8,.0f}", flush=True)
            hist.append({"update": u, "transitions": u * N * T, "return": mr, "length": ml,
                         "clear_pct": 100 * ncl / max(neps, 1), "death_pct": 100 * nde / max(neps, 1)})
            ncl = nde = neps = 0; ekill = 0.0
    el = time.time() - t0
    print(f"\n{updates*N*T:,} dream transitions in {el:.0f}s ({updates*N*T/el:,.0f}/s), 0 emulator steps")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(a), "history": hist, "baseline": base}, a.out)
    Path("results/dream_ppo.json").write_text(json.dumps({"config": vars(a), "history": hist,
                                                          "real_baseline": base}, indent=2))


if __name__ == "__main__":
    main()
