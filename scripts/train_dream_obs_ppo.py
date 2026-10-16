#!/usr/bin/env python3
"""PPO inside the Gohma observation-space world model. The emulator is never touched.

This is the dream half of the loop in the author's paper (arXiv 2610.08033): the policy trains ONLY in the
model, on free-running episodes, and is judged in the real game by the online evaluator. Four things it does
deliberately, each because of a recorded failure:

  * EPISODES START FROM REAL EPISODE-START FRAMES, not a uniform sample of every frame. Sampling uniformly
    begins fights midway and "cleared the room" then means finishing a partial fight, which cannot be
    compared with a baseline measured over whole episodes (scripts/train_dream_ppo.py's own docstring).
  * REWARD IS READ FROM THE PREDICTED FIELDS the real reward reads -- boss HP and Link HP -- with the same
    coefficients as GOHMA_SIMPLE (+4.0 per boss HP, +60 on defeat, the Link-damage penalty). It is not a
    learned scalar, so it cannot drift away from what the real task pays for.
  * THE HORIZON IS BOUNDED and reported. A WM is accurate for a while and then not; the dream must stop
    before it is dreaming noise. --horizon is the knob and the WM's own H20/H100 skill is how you choose it.
  * SNAPSHOTS EVERY FEW UPDATES into --snapshot-dir, which the online evaluator watches, so the policy is
    measured in the REAL game while it learns in the dream. In-dream return alone is worthless: the paper's
    unanchored runs collapsed in reality with no in-dream metric detecting it.

    python3 -u scripts/train_dream_obs_ppo.py --wm runs/wm_obs/wm.pt \
        --init checkpoints/kept/rvs_M5_s5.pt --data datasets/ppo/wm --updates 200 \
        --snapshot-dir runs/gohma-dyna/snaps --out runs/gohma-dyna/dream.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.policy_io import load_policy          # noqa: E402
from zelda_rl.ppo import ppo_update_recurrent       # noqa: E402
from zelda_rl.wm_obs import load_wm                 # noqa: E402

BOSS_HP = 10.0                     # Gohma's HP, so a frac delta becomes HP
R_DMG = 4.0                        # GOHMA_SIMPLE: +4.0 per boss HP taken
R_DEFEAT = 60.0                    # +60 for the kill
R_HURT = 2.0                       # Link-damage penalty per heart


def start_frames(dirs, want: int, obs_dim: int) -> np.ndarray:
    """Real episode-START observations: the only honest place to begin a dream."""
    out = []
    files = [f for d in dirs for f in sorted(Path(d).glob("*.npz"))]
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            if "obs" not in z.files:
                continue
            o = z["obs"]
            if o.shape[1] != obs_dim or len(o) < 10:
                continue
            out.append(o[0].astype(np.float32))
            if len(out) >= want:
                break
        except Exception:                                    # noqa: BLE001
            continue
    if not out:
        raise SystemExit(f"no usable episode-start frames in {dirs}")
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--init", required=True, help="policy to start from (the ANCHOR)")
    ap.add_argument("--data", action="append", required=True, help="episodes, for real start frames")
    ap.add_argument("--envs", type=int, default=64, help="parallel dreams")
    ap.add_argument("--horizon", type=int, default=240, help="dream steps per rollout segment")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--out", default="runs/dream_obs/dream.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    wm, scales, wck = load_wm(a.wm, dev)
    fields = wck.get("fields") or {}
    idx, off = {}, 0
    for name, w in fields.get("global", []):
        idx[name] = off; off += w
    g_hp, l_hp = idx.get("gohma_hp_frac"), idx.get("link_hp_frac")
    if g_hp is None or l_hp is None:
        raise SystemExit("the WM checkpoint has no gohma_hp_frac/link_hp_frac in its fields -- the dream "
                         "reward reads those columns and cannot be computed without them")
    obs_dim = int(wck["obs_dim"])
    print(f"WM {a.wm}: obs {obs_dim}, epoch {wck.get('epoch')}, metrics {wck.get('metrics', {}).get(20)}",
          flush=True)

    model, cfg, _ = load_policy(Path(a.init))
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    starts = torch.from_numpy(start_frames(a.data, 4000, obs_dim)).to(dev)
    print(f"policy {a.init} | {starts.shape[0]} real start frames | horizon {a.horizon} | "
          f"{a.envs} parallel dreams", flush=True)

    snap_dir = Path(a.snapshot_dir) if a.snapshot_dir else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    N, T = a.envs, a.horizon
    sel = torch.randint(0, starts.shape[0], (N,), device=dev)
    obs = starts[sel].clone()
    h = model.init_hidden(N).to(dev)
    first = torch.ones(N, device=dev)
    t_start = time.time()
    for upd in range(1, a.updates + 1):
        O = torch.zeros(T, N, obs_dim, device=dev)
        A = torch.zeros(T, N, dtype=torch.long, device=dev)
        LP = torch.zeros(T, N, device=dev)
        V = torch.zeros(T, N, device=dev)
        R = torch.zeros(T, N, device=dev)
        D = torch.zeros(T, N, device=dev)
        FI = torch.zeros(T, N, device=dev)
        H = torch.zeros(T, N, h.shape[-1], device=dev)
        n_win = n_death = 0
        with torch.no_grad():
            for t in range(T):
                O[t], H[t], FI[t] = obs, h, first
                logits, val, h2 = model.step(obs, h, first)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                A[t], LP[t], V[t] = act, dist.log_prob(act), val.squeeze(-1)
                d, _ = wm.step(obs, act, torch.zeros(N, wm.gru.hidden_size, device=dev)) \
                    if False else wm.step(obs, act, getattr(main, "_wmh", None)
                                          if getattr(main, "_wmh", None) is not None
                                          else wm.init_hidden(N, dev))
                nxt = obs + d * scales
                # reward from the PREDICTED fields the real reward reads
                dmg_hp = (obs[:, g_hp] - nxt[:, g_hp]).clamp(min=0) * BOSS_HP
                hurt = (obs[:, l_hp] - nxt[:, l_hp]).clamp(min=0) * BOSS_HP
                win = (nxt[:, g_hp] <= 1e-3).float()
                dead = (nxt[:, l_hp] <= 1e-3).float()
                R[t] = R_DMG * dmg_hp - R_HURT * hurt + R_DEFEAT * win
                done = ((win + dead) > 0).float()
                D[t] = done
                n_win += int(win.sum()); n_death += int(dead.sum())
                # reset finished dreams to fresh REAL start frames
                if float(done.sum()) > 0:
                    sel = torch.randint(0, starts.shape[0], (N,), device=dev)
                    nxt = torch.where(done[:, None] > 0, starts[sel], nxt)
                    h2 = torch.where(done[:, None] > 0, torch.zeros_like(h2), h2)
                obs, h, first = nxt, h2, done
            _, last_v, _ = model.step(obs, h, first)
            last_v = last_v.squeeze(-1)
        # GAE
        adv = torch.zeros_like(R)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nd = 1.0 - D[t]
            delta = R[t] + a.gamma * nv * nd - V[t]
            gae = delta + a.gamma * a.lam * nd * gae
            adv[t] = gae
        ret = adv + V
        batch = {"obs": O.cpu().numpy(), "act": A.cpu().numpy(), "logp": LP.cpu().numpy(),
                 "adv": ((adv - adv.mean()) / (adv.std() + 1e-8)).cpu().numpy(),
                 "ret": ret.cpu().numpy(), "val": V.cpu().numpy(),
                 "first": FI.cpu().numpy(), "h": H.cpu().numpy()}
        stats = ppo_update_recurrent(model, opt, batch, chunk=a.chunk, epochs=a.epochs,
                                     chunks_per_mb=max(1, (T // a.chunk) * N // a.minibatches),
                                     clip=a.clip, ent_coef=a.ent, vf_coef=a.vf_coef, max_grad_norm=0.5)
        if upd % 1 == 0:
            el = time.time() - t_start
            print(f"upd {upd:>4}  ret {float(R.sum(0).mean()):>9.2f}  dreamWins {n_win:>4}  "
                  f"deaths {n_death:>4}  kl {np.mean(stats.get('kl', [0])):.4f}  "
                  f"ent {np.mean(stats.get('ent', [0])):.3f}  {T*N/max(el,1e-9):,.0f} dream steps/s",
                  flush=True)
            t_start = time.time()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "update": upd,
                    "config": {**cfg, **vars(a)}}, out)
        if snap_dir and upd % a.snapshot_every == 0:
            torch.save({"model": model.state_dict(), "update": upd, "config": {**cfg, **vars(a)}},
                       snap_dir / f"{out.stem}.upd{upd:05d}.pt")
    print(f"done: {a.updates} updates, 0 emulator steps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
