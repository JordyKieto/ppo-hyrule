"""Human demonstrations as a PPO regulariser (DAPG-style): initialise from BC, then keep a BC loss on.

`--init-from` alone decays: PPO's own gradient walks the policy away from the demonstrated behaviour within a
few updates and the only trace left is wherever the weights started. Keeping a cross-entropy term on the human
corpus for the whole run bounds that drift against the DATA, which is a different instrument from
`--kl-anchor`, which bounds it against a POLICY (zelda_rl/ppo.py:170).

Not to be confused with `zelda_rl/bc.py`, which is the WM track's behaviour-cloned ENCODER.

The chunks here are sampled WINDOWS, not whole episodes, so a window that starts mid-episode gets a cold GRU
(hidden zeros, `first`=1 at its first step) where the real episode would have carried state in. That is the
standard approximation for this term and it is why `scripts/train_bc_human.py` -- which carries hidden across
windows inside an episode -- remains the instrument for measuring clone quality. This one is a regulariser.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_bc_chunks(data: str | Path, chunk: int, *, obs_dim: int, n_actions: int) -> dict:
    """Pre-chunk every recorded episode into (M, chunk) windows. Asserts the spec matches the live model.

    A width or action-count mismatch here is the failure that trains silently on the wrong layout, so it is a
    hard error rather than a warning (AGENTS 3, 16).
    """
    d = Path(data)
    files = sorted(d.glob("ep*.npz"))
    if not files:
        raise SystemExit(f"--bc-data {d}: no ep*.npz")
    O, A, F = [], [], []
    wins = steps = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        obs, act = z["obs"].astype(np.float32), z["action"].astype(np.int64)
        if obs.shape[1] != obs_dim:
            raise SystemExit(f"--bc-data {f.name}: obs_dim {obs.shape[1]} but the model builds {obs_dim}")
        if int(z["n_actions"]) != n_actions:
            raise SystemExit(f"--bc-data {f.name}: n_actions {int(z['n_actions'])} but the model has {n_actions}")
        wins += bool(z["boss_defeated"]); steps += len(act)
        for i in range(0, len(act) - chunk + 1, chunk):           # drop the ragged tail
            O.append(obs[i:i + chunk]); A.append(act[i:i + chunk])
            f0 = np.zeros(chunk, np.float32)
            if i == 0:
                f0[0] = 1.0                                        # a true episode start resets the GRU
            F.append(f0)
    if not O:
        raise SystemExit(f"--bc-data {d}: every episode is shorter than chunk={chunk}")
    out = {"obs": torch.from_numpy(np.stack(O)), "act": torch.from_numpy(np.stack(A)),
           "first": torch.from_numpy(np.stack(F))}
    out["meta"] = {"files": len(files), "wins": int(wins), "steps": int(steps), "chunks": len(O), "chunk": chunk}
    return out


def describe(bc: dict) -> str:
    m = bc["meta"]
    return (f"{m['files']} episodes ({m['wins']} wins, {m['steps']:,} steps) -> {m['chunks']} windows "
            f"of {m['chunk']}")


def coef_at(update: int, start: float, final: float, updates: int) -> float:
    """DAPG-style decay with a FLOOR: `start` at update 1, reaching `final` at `updates`, then held there.

    The floor is the point. Decaying to 0 removes the anchor entirely and the policy is free to drift back
    wherever PPO's gradient takes it -- the demonstrations then only shaped the early updates, which is the
    `--init-from`-only behaviour this term exists to avoid. A long tail at a small coefficient keeps paying a
    little attention to the human corpus for the whole run.

    Linear rather than exponential so the coefficient at any update is readable from the flags alone.
    """
    if updates <= 1:
        return float(final)
    f = min(1.0, max(0.0, (update - 1) / float(updates - 1)))
    if f >= 1.0:
        return float(final)              # exact, so the floor is comparable and not start + delta*1.0
    return float(start + (final - start) * f)
