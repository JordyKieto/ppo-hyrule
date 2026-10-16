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


def hp_column(fields: dict) -> int:
    """Offset of `gohma_hp_frac` in the observation, derived from the recording's OWN field list.

    Never hand-counted: AGENTS 3 exists because a hand-counted slice zeroed half an ablation and produced a
    false result. Verified on the corpus: offset 100, monotone non-increasing, 10 distinct levels = her 10 HP.
    """
    off = 0
    for name, w in fields["global"]:
        if name == "gohma_hp_frac":
            return off
        off += w
    raise SystemExit("gohma_hp_frac is not in the recording's obs_fields; cannot build a damage filter")


def load_bc_chunks(data: str | Path, chunk: int, *, obs_dim: int, n_actions: int,
                   qfilter_steps: int = 0) -> dict:
    """Pre-chunk every recorded episode into (M, chunk) windows. Asserts the spec matches the live model.

    A width or action-count mismatch here is the failure that trains silently on the wrong layout, so it is a
    hard error rather than a warning (AGENTS 3, 16).
    """
    d = Path(data)
    files = sorted(d.glob("ep*.npz"))
    if not files:
        raise SystemExit(f"--bc-data {d}: no ep*.npz")
    O, A, F, K = [], [], [], []
    wins = steps = 0
    hp_idx = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        obs, act = z["obs"].astype(np.float32), z["action"].astype(np.int64)
        if obs.shape[1] != obs_dim:
            raise SystemExit(f"--bc-data {f.name}: obs_dim {obs.shape[1]} but the model builds {obs_dim}")
        if int(z["n_actions"]) != n_actions:
            raise SystemExit(f"--bc-data {f.name}: n_actions {int(z['n_actions'])} but the model has {n_actions}")
        if hp_idx is None:
            hp_idx = hp_column(json.loads(str(z["obs_fields"])))
        wins += bool(z["boss_defeated"]); steps += len(act)
        # The FILTER (`--bc-qfilter H`). Deviation from the Q filter of Nair et al. 2018, stated plainly
        # because deviations belong in the docstring and not in a commit (AGENTS 13): their filter keeps a
        # demonstration frame when Q(s, a_demo) > Q(s, a_policy), and this project's critic is a V head, so
        # there is no Q to compare. What is available is the OUTCOME the human actually achieved, in the units
        # the reward pays for: boss HP. A frame is kept when her HP falls within the next H steps, i.e. when
        # the human's play from that state actually damaged her. That drops the walking-around frames, which
        # is where "53% action accuracy, 1.4% of her HP" comes from.
        keep = None
        if qfilter_steps > 0:
            hp = obs[:, hp_idx]
            fut = np.minimum.accumulate(hp[::-1])[::-1]            # min HP from t to the end
            ahead = np.empty_like(hp)
            for t in range(len(hp)):
                e = min(len(hp) - 1, t + qfilter_steps)
                ahead[t] = hp[t:e + 1].min()
            keep = (hp - ahead > 1e-6).astype(np.float32)
        for i in range(0, len(act) - chunk + 1, chunk):           # drop the ragged tail
            O.append(obs[i:i + chunk]); A.append(act[i:i + chunk])
            f0 = np.zeros(chunk, np.float32)
            if i == 0:
                f0[0] = 1.0                                        # a true episode start resets the GRU
            F.append(f0)
            K.append(np.ones(chunk, np.float32) if keep is None else keep[i:i + chunk])
    if not O:
        raise SystemExit(f"--bc-data {d}: every episode is shorter than chunk={chunk}")
    out = {"obs": torch.from_numpy(np.stack(O)), "act": torch.from_numpy(np.stack(A)),
           "first": torch.from_numpy(np.stack(F)), "keep": torch.from_numpy(np.stack(K))}
    kept = float(out["keep"].mean())
    out["meta"] = {"files": len(files), "wins": int(wins), "steps": int(steps), "chunks": len(O),
                   "chunk": chunk, "qfilter_steps": int(qfilter_steps), "kept_frac": kept}
    if qfilter_steps > 0 and kept <= 0.0:
        raise SystemExit(f"--bc-qfilter {qfilter_steps} keeps NO frames; the term would be a silent no-op")
    return out


def describe(bc: dict) -> str:
    m = bc["meta"]
    base = (f"{m['files']} episodes ({m['wins']} wins, {m['steps']:,} steps) -> {m['chunks']} windows "
            f"of {m['chunk']}")
    if m.get("qfilter_steps"):
        base += (f"; qfilter {m['qfilter_steps']} steps keeps {100 * m['kept_frac']:.1f}% of frames "
                 f"(those where her HP falls within {m['qfilter_steps']} steps)")
    return base


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
