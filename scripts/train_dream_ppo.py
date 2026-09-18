"""PPO trained entirely inside the learned latent dynamics. The emulator is never touched.

    .venv/bin/python scripts/train_dream_ppo.py --transitions 100000

Episodes start from real latents sampled from the dataset, then roll forward with the learned predictor.
Reward is not regressed: the decoder reads the raw fields the reward function keys on out of each predicted
latent, and the outcome terms of `CombatTracker` are applied to them directly.

Implemented outcome terms (weights from `CombatConfig`): enemy damage, kill, Link damage, death,
room-clear success with its time bonus, time penalty. The shaping terms -- lock-on, approach, shield,
block/dodge -- are omitted: they depend on relative geometry, which is the part the decoder reconstructs
least well, so including them would measure decoder error rather than policy quality.

Because the reward here differs from the one the model-free run optimised, the same outcome-only reward is
also computed over real dataset transitions using true (not decoded) fields. That is the honest baseline.
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
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)
from zelda_rl.ppo import ActorCritic, gae, ppo_update  # noqa: E402
from scripts.train_bc_decode import DEFAULT_DATA, load  # noqa: E402

W_DMG, W_KILL, W_LINK_DMG, DEATH, SUCCESS, SUCCESS_TIME, TIME_PEN = 1.0, 5.0, 0.25, 20.0, 20.0, 15.0, 0.001
HP_UNIT = 8.0            # Link HP classes are half-hearts; the reward weights raw HP units


class Dream:
    """Batched rollout in latent space. Everything the reward needs is decoded, nothing is simulated."""

    def __init__(self, enc, dec, pred, Z0, sc, n_actions, max_steps):
        self.dec, self.pred, self.Z0, self.sc = dec, pred, Z0, sc
        self.NA, self.max_steps = n_actions, max_steps

    @torch.no_grad()
    def _read(self, z):
        p = self.dec(z)
        pres = p["row_present"] > 0
        baba = (p["row_is_baba"] > 0) & pres
        hp = p["row_hp"].argmax(-1) * baba              # 0 where not a live Baba row
        return p["link_hp"].argmax(-1), hp, baba

    @torch.no_grad()
    def reset(self, n, rng):
        i = torch.from_numpy(rng.integers(0, len(self.Z0), n))
        self.z = self.Z0[i].clone()
        self.lhp, self.rhp, self.baba = self._read(self.z)
        self.start_targets = (self.rhp > 0).sum(-1).clamp(min=1)
        self.t = torch.zeros(n)
        return self.z

    @torch.no_grad()
    def reset_at(self, k, rng):
        """Restart only the finished envs, in place."""
        i = torch.from_numpy(rng.integers(0, len(self.Z0), len(k)))
        self.z[k] = self.Z0[i]
        lhp, rhp, baba = self._read(self.z[k])
        self.lhp[k], self.rhp[k], self.baba[k] = lhp, rhp, baba
        self.start_targets[k] = (rhp > 0).sum(-1).clamp(min=1)
        self.t[k] = 0

    @torch.no_grad()
    def step(self, a):
        self.z = self.pred(torch.cat([self.z, F.one_hot(a, self.NA).float()], -1))
        lhp, rhp, baba = self._read(self.z)
        self.t += 1
        # HP never rises: clamp both, which is the monotonicity constraint that fixed event decoding
        rhp = torch.minimum(rhp, self.rhp)
        lhp = torch.minimum(lhp, self.lhp)
        dmg = (self.rhp - rhp).clamp(min=0).sum(-1).float()
        kill = ((self.rhp > 0) & (rhp == 0)).sum(-1).float()
        ldmg = (self.lhp - lhp).clamp(min=0).float() * HP_UNIT
        r = W_DMG * dmg + W_KILL * kill - W_LINK_DMG * ldmg - TIME_PEN
        dead = lhp <= 0
        cleared = (rhp.sum(-1) == 0) & (self.start_targets > 0)
        r = r - DEATH * dead.float()
        r = r + (SUCCESS + SUCCESS_TIME * (1 - self.t / self.max_steps)).clamp(min=0) * cleared.float()
        done = dead | cleared | (self.t >= self.max_steps)
        self.lhp, self.rhp, self.baba = lhp, rhp, baba
        return self.z, r, done, {"dead": dead, "cleared": cleared}


def real_baseline(Y, action, slot, epid, split, rng):
    """The same outcome-only reward, applied to real transitions with TRUE fields."""
    test = set(split["val_slots"])
    lhp = Y["link_hp_c"]
    rhp = np.where((Y["row_present"] > .5) & (Y["row_is_baba"] > .5), Y["row_hp_c"], 0)
    same = epid[1:] == epid[:-1]
    rets, ep_r, start = [], 0.0, 0
    for t in range(len(same)):
        if not np.isin(slot[t], list(test)):
            continue
        if not same[t]:
            if ep_r:
                rets.append(ep_r)
            ep_r = 0.0
            continue
        d = np.clip(rhp[t] - rhp[t + 1], 0, None).sum()
        k = ((rhp[t] > 0) & (rhp[t + 1] == 0)).sum()
        ld = max(lhp[t] - lhp[t + 1], 0) * HP_UNIT
        ep_r += W_DMG * d + W_KILL * k - W_LINK_DMG * ld - TIME_PEN
        if lhp[t + 1] <= 0:
            ep_r -= DEATH
        if rhp[t + 1].sum() == 0 and rhp[t].sum() > 0:
            ep_r += SUCCESS
    return float(np.mean(rets)) if rets else float("nan"), len(rets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/tmp/joint_3e-4.pt", help="joint encoder/decoder/predictor")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--transitions", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=32, help="rollout length per update")
    ap.add_argument("--max-steps", type=int, default=200, help="dream episode cap")
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
    dec = StateDecoder(latent=LAT); dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = nn.Sequential(nn.Linear(LAT + NA, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, LAT))
    pred.load_state_dict(ck["pred"]); pred.eval()
    for p in list(enc.parameters()) + list(dec.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)

    meta, state, action, slot, epid = load(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    norm = Norm.load(ck["norm"])
    Fn = norm.apply(featurise(state))
    train_mask = ~np.isin(slot, list(set(ck["split"]["val_slots"]) | set(ck["split"]["tune_slots"])))
    idx = np.where(train_mask)[0]
    with torch.no_grad():
        Z0 = torch.cat([enc.encode(torch.from_numpy(Fn["link"][idx[s:s + 8192]]),
                                   torch.from_numpy(Fn["rows"][idx[s:s + 8192]]),
                                   torch.from_numpy(Fn["ids"][idx[s:s + 8192]]),
                                   torch.from_numpy(Fn["mask"][idx[s:s + 8192]]))
                        for s in range(0, len(idx), 8192)])

    base, nb = real_baseline(Y, action, slot, epid, ck["split"], rng)
    env = Dream(enc, dec, pred, Z0, ck["decode_scale"], NA, a.max_steps)
    model = ActorCritic(LAT, NA, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    N, T = a.envs, a.steps
    updates = max(a.transitions // (N * T), 1)
    print(f"dreaming {updates} updates x {N} envs x {T} steps = {updates * N * T:,} transitions")
    print(f"real-data baseline, same outcome-only reward on TRUE fields: {base:.2f} mean return "
          f"over {nb} held-out episodes\n")
    print(f"{'upd':>4} {'transitions':>12} {'return':>8} {'len':>6} {'clear%':>7} {'death%':>7} {'kills':>6} {'sps':>8}")
    print("-" * 66)

    obs = env.reset(N, rng)
    hist, ep_r, ep_l, done_ret, done_len, ncl, nde, nkill, neps = [], torch.zeros(N), torch.zeros(N), [], [], 0, 0, 0.0, 0
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
            if done.any():
                k = done.nonzero(as_tuple=True)[0]
                done_ret += ep_r[k].tolist(); done_len += ep_l[k].tolist()
                ncl += int(info["cleared"][k].sum()); nde += int(info["dead"][k].sum()); neps += len(k)
                nkill += float((env.start_targets[k] - (env.rhp[k] > 0).sum(-1)).clamp(min=0).float().sum())
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
            sps = u * N * T / (time.time() - t0)
            print(f"{u:>4} {u * N * T:>12,} {mr:>8.2f} {ml:>6.0f} {100*ncl/max(neps,1):>6.1f}% "
                  f"{100*nde/max(neps,1):>6.1f}% {nkill/max(neps,1):>6.2f} {sps:>8,.0f}", flush=True)
            hist.append({"update": u, "transitions": u * N * T, "return": mr, "length": ml,
                         "clear_pct": 100 * ncl / max(neps, 1), "death_pct": 100 * nde / max(neps, 1)})
            ncl = nde = neps = 0; nkill = 0.0
    el = time.time() - t0
    print(f"\n{updates * N * T:,} dream transitions in {el:.0f}s ({updates*N*T/el:,.0f}/s), 0 emulator steps")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(a), "history": hist, "baseline": base}, a.out)
    Path("results/dream_ppo.json").write_text(json.dumps({"config": vars(a), "history": hist,
                                                          "real_baseline": base}, indent=2))


if __name__ == "__main__":
    main()
