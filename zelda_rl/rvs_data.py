"""Return-conditioned windows for the PPO loop: (obs | g_actual, action), with wins oversampled.

The corpus is mostly losses, and that is the point (zelda_rl/rvs.py): a losing episode labelled with the
outcome it produced teaches the policy what the fatal action looks like under a LOW prompt, while the rare
wins teach the winning action under a HIGH prompt. Acting with the target prompt then selects the latter.

Wins are ~1% of a PPO corpus, so without `win_weight` the gradient is almost entirely losing play.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from zelda_rl.rvs import COND_DIM, conditioning, field_offsets


def solve_win_weight(win_steps: int, all_steps: int, share: float) -> float:
    """Multiplier that puts winning play at `share` of the gradient: w = s(1-p) / (p(1-s)).

    A fixed multiplier is wrong because it encodes an assumption about corpus composition. The same 8x that
    is about right on a 4%-win corpus puts wins at 83% of the gradient on a 48%-win one, and the whole point
    of RvS is that the LOSSES carry weight -- they are what label the fatal action under a low outcome.
    """
    p = win_steps / max(all_steps, 1)
    if not (0.0 < p < 1.0) or not (0.0 < share < 1.0):
        return 1.0
    return (share * (1 - p)) / (p * (1 - share))


def load_rvs_windows(dirs, chunk: int, *, obs_dim: int, n_actions: int, win_weight: float = 0.0,
                     win_share: float = 0.3, max_windows: int = 12000) -> dict:
    # EP/POS let the consumer replay an episode's windows IN ORDER with the hidden state carried, instead of
    # cold-starting every 60-step slice. The label is already episode-length (rvs.conditioning), so a cold
    # context was the one short horizon left in the loop.
    # ONE pass over the corpus, bounded memory, every file seen. 9,340 episodes as 60-step windows is
    # ~20 GB of float32, which a 32 GB pod cannot hold alongside training, so loss windows are strided to a
    # budget while EVERY winning window is kept -- wins are 5% of the corpus and are the scarce thing.
    files_all = [f for d in dirs for f in sorted(Path(d).glob("*.npz"))]
    O, A, F, W, EP, POS = [], [], [], [], [], []
    ep_id = 0
    files = n_win = n_ep = steps = 0
    w_steps = a_steps = 0
    loss_keep_every = max(1, int(len(files_all) * 8 / max(max_windows, 1)))   # ~8 windows per episode
    for fi, f in enumerate(files_all):
        z = np.load(f, allow_pickle=False)
        if "obs" not in z.files:
            continue
        obs = z["obs"].astype(np.float32)
        if obs.shape[1] + COND_DIM != obs_dim:
            raise SystemExit(f"{f.name}: obs {obs.shape[1]} + cond {COND_DIM} != model's {obs_dim}")
        won = bool(z["boss_defeated"]) if "boss_defeated" in z.files else False
        act = z["action"].astype(np.int64)
        a_steps += len(act)
        if won:
            w_steps += len(act)
        elif fi % loss_keep_every:
            n_ep += 1                                   # counted for composition, windows not kept
            continue
        g = conditioning(obs, field_offsets(str(z["obs_fields"])), won)
        x = np.concatenate([obs, g], 1)
        for k_, i in enumerate(range(0, len(act) - chunk + 1, chunk)):
            O.append(x[i:i + chunk]); A.append(act[i:i + chunk])
            f0 = np.zeros(chunk, np.float32)
            if i == 0:
                f0[0] = 1.0
            F.append(f0); W.append(np.float32(1.0))     # filled in below once the multiplier is solved
            EP.append(ep_id); POS.append(k_)
        W[-len(range(0, len(act) - chunk + 1, chunk)):] = [np.float32(1.0 if not won else -1.0)] * \
            len(range(0, len(act) - chunk + 1, chunk))   # -1 marks "winning", resolved after solving
        files += 1; n_ep += 1; n_win += int(won); steps += len(act); ep_id += 1
    solved = win_weight if win_weight > 0 else solve_win_weight(w_steps, a_steps, win_share)
    W = [np.full(chunk, solved if float(w) < 0 else 1.0, np.float32) for w in W]
    if not O:
        raise SystemExit(f"--rvs-data {dirs}: no usable episodes")
    out = {"obs": torch.from_numpy(np.stack(O)), "act": torch.from_numpy(np.stack(A)),
           "first": torch.from_numpy(np.stack(F)), "keep": torch.from_numpy(np.stack(W)),
           "ep": torch.tensor(EP), "pos": torch.tensor(POS)}
    wsum = float(out["keep"].sum())
    win_share_actual = float(out["keep"][out["keep"] > 1.0].sum()) / max(wsum, 1e-9)
    out["n_static"] = len(O)          # anchored: never evicted by the Dyna refresh
    out["desc"] = (f"{n_ep} episodes ({n_win} wins, {100*n_win/max(n_ep,1):.1f}%), {steps:,} steps -> "
                   f"{len(O)} windows of {chunk}; wins are {100*w_steps/max(a_steps,1):.1f}% of steps, "
                   f"solved multiplier {solved:.2f} -> winning play is {100*win_share_actual:.1f}% of the "
                   f"gradient")
    return out


def rollout_windows(batch, idx: dict, chunk: int, cond_dim: int, win_weight: float = 20.0):
    """Turn the policy's OWN just-collected rollout into return-conditioned windows.

    Dyna-style: the corpus is not a fixed dataset but a buffer that keeps absorbing the newest episodes, so
    the conditioning the policy is trained on reflects what IT currently does, not only what v15 and a human
    did hours ago. The rollout observations arrive carrying the inference prompt g=(1,1,0) in their last
    `cond_dim` columns; those columns are OVERWRITTEN here with the outcome each step actually led to, which
    is the whole mechanism -- a fatal action gets labelled with the low outcome that followed it.

    Episode boundaries come from `first`; `won` is read from her HP reaching zero, not from a reward, so it
    means the same thing as in the offline corpora.
    """
    obs = np.asarray(batch["obs"], np.float32)        # (T, N, obs_dim) with the prompt in the tail
    act = np.asarray(batch["act"], np.int64)
    first = np.asarray(batch["first"], np.float32)
    T, N, D = obs.shape
    g_hp, l_hp = idx["gohma_hp_frac"], idx["link_hp_frac"]
    O, A, Fl, W, EP, POS = [], [], [], [], [], []
    ep_id = 0
    for n in range(N):
        starts = [t for t in range(T) if first[t, n] > 0.5] or [0]
        bounds = starts + [T]
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            seg = obs[s0:s1, n]
            if len(seg) < chunk:
                continue
            ghp, lhp = seg[:, g_hp], seg[:, l_hp]
            won = bool(ghp[-1] <= 1e-3)
            dmg = np.clip(ghp - ghp[-1], 0.0, 1.0)
            hurt = np.clip(lhp - lhp[-1], 0.0, 1.0)
            g = np.stack([dmg, np.full(len(seg), 1.0 if won else 0.0, np.float32), hurt], 1)
            x = seg.copy()
            x[:, D - cond_dim:] = g[:, :cond_dim]      # replace the prompt with the ACTUAL outcome
            w = win_weight if won else 1.0
            for k_, i in enumerate(range(0, len(x) - chunk + 1, chunk)):
                O.append(x[i:i + chunk]); A.append(act[s0 + i:s0 + i + chunk, n])
                f0 = np.zeros(chunk, np.float32)
                if i == 0:
                    f0[0] = 1.0
                Fl.append(f0); W.append(np.full(chunk, w, np.float32))
                EP.append(ep_id); POS.append(k_)
            ep_id += 1
    if not O:
        return None
    return {"obs": torch.from_numpy(np.stack(O)), "act": torch.from_numpy(np.stack(A)),
            "first": torch.from_numpy(np.stack(Fl)), "keep": torch.from_numpy(np.stack(W)),
            "ep": torch.tensor(EP), "pos": torch.tensor(POS)}


def merge_capped(rvs: dict, fresh: dict, cap: int, fresh_share: float = 0.5) -> dict:
    """Append fresh windows, capped, evicting only the oldest FRESH ones -- the static corpus is anchored.

    The human wins are the only episodes in this project that actually beat Gohma, and they are the anchor:
    if eviction were a plain ring buffer the corpus sits at the front and would be the FIRST thing dropped,
    so within a few updates the policy would be imitating only its own recent play, which is how
    self-imitation collapses. `n_static` windows are therefore never evictable, and the log reports the
    anchor's share of the gradient every update.
    """
    if fresh is None:
        return rvs
    n_static = int(rvs.get("n_static", rvs["obs"].shape[0]))
    out = {"n_static": n_static}
    for k in ("obs", "act", "first", "keep", "ep", "pos"):
        if k not in rvs or k not in fresh:
            continue
        static, prior = rvs[k][:n_static], rvs[k][n_static:]
        fk = fresh[k]
        if k == "ep":                        # keep fresh episode ids distinct from everything already held
            fk = fk + int(rvs["ep"].max().item()) + 1 if rvs.get("ep") is not None and len(rvs["ep"]) else fk
        tail = torch.cat([prior, fk], 0)
        room = max(cap - n_static, 0)
        if room and tail.shape[0] > room:
            tail = tail[-room:]
        elif not room:
            tail = tail[:0]
        out[k] = torch.cat([static, tail], 0)
    # HOW MUCH fresh data, not whether: the fresh windows' weights are rescaled so they carry exactly
    # `fresh_share` of the conditioning gradient. Controlling the COUNT would not control the influence,
    # because winning windows already carry a solved multiplier -- and "on/off" answers a cruder question
    # than the one that matters.
    w = out["keep"]
    n_now = w.shape[0]
    if n_now > n_static and 0.0 <= fresh_share < 1.0:
        sw = float(w[:n_static].sum())
        fw = float(w[n_static:].sum())
        if fw > 0:
            want = (fresh_share / max(1.0 - fresh_share, 1e-6)) * sw
            out["keep"][n_static:] = w[n_static:] * (want / fw)
    w = out["keep"]
    out["desc"] = rvs.get("desc", "")
    out["fresh"] = int(fresh["obs"].shape[0])
    out["fresh_share"] = float(w[n_static:].sum() / w.sum().clamp(min=1e-9))
    out["anchor_share"] = float(w[:n_static].sum() / w.sum().clamp(min=1e-9))
    return out
