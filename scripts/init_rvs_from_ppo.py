"""Widen a PPO policy into a return-conditioned (RvS) checkpoint, so RvS can START from v15.

v15 is a 1033-d PPO policy; RvS needs 1036 = global(297) + cond(3) + actors(16x46). Only one tensor sees
the global block -- `proj.0.weight`, whose input is [global | actor_mean | actor_max] -- so widening is
three ZERO columns inserted at the end of the global slice. Zeros mean the initialised model is EXACTLY the
PPO policy for any prompt: the conditioning starts with no effect and has to be learned, which is the only
honest starting point (and is asserted below rather than assumed).

This only works with the corrected prompt placement (zelda_rl.rvs.cond_offset). Under the old tail
placement the actor table shifts by three columns and PPO weights would be misaligned garbage.

EXPERIMENT-CONTRACT: exempt -- a one-shot checkpoint converter: it reads one file, writes one file, runs no
training and no evaluation. The run that consumes the output is under the contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.ppo import RecurrentEntityActorCritic          # noqa: E402
from zelda_rl.rvs import COND_DIM, TARGET                    # noqa: E402

torch.set_num_threads(1)


def widen(ppo_ck: dict, cond_at: int) -> tuple[dict, dict]:
    cfg = dict(ppo_ck["config"])
    gd, na, ad = int(cfg["global_dim"]), int(cfg["n_actors"]), int(cfg["actor_dim"])
    n_actions = int(cfg["n_actions"])
    hidden = int(cfg.get("hidden", 256)); gru = int(cfg.get("gru", 256))
    if not 0 < cond_at <= gd:
        raise SystemExit(f"cond_at {cond_at} is outside the global block of width {gd}")
    src = ppo_ck["model"]
    new = RecurrentEntityActorCritic(gd + COND_DIM, na, ad, n_actions, hidden=hidden, gru=gru)
    tgt = new.state_dict()
    out, widened = {}, []
    for k, v in tgt.items():
        s = src.get(k)
        if s is None:
            raise SystemExit(f"the PPO checkpoint has no {k}; architectures do not correspond")
        if s.shape == v.shape:
            out[k] = s.clone()
            continue
        if s.dim() == 2 and v.dim() == 2 and s.shape[0] == v.shape[0] and v.shape[1] - s.shape[1] == COND_DIM:
            w = torch.zeros_like(v)
            w[:, :cond_at] = s[:, :cond_at]
            w[:, cond_at + COND_DIM:] = s[:, cond_at:]       # the three inserted columns stay ZERO
            out[k] = w
            widened.append(f"{k} {tuple(s.shape)} -> {tuple(v.shape)} at col {cond_at}")
            continue
        raise SystemExit(f"cannot map {k}: {tuple(s.shape)} -> {tuple(v.shape)}")
    new.load_state_dict(out)
    cfg.update({"global_dim": gd + COND_DIM, "obs_dim": gd + COND_DIM + na * ad,
                "raw_dim": gd + na * ad, "rvs_cond": True, "rvs_cond_dim": COND_DIM,
                "rvs_cond_at": cond_at, "rvs_target": list(TARGET),
                "init_from_ppo": True})
    return {"model": new.state_dict(), "optimizer": None, "config": cfg,
            "update": ppo_ck.get("update", 0), "global_step": ppo_ck.get("global_step", 0),
            "episode_count": ppo_ck.get("episode_count", 0), "widened": widened}, cfg


def check_identical(ppo_ck: dict, rvs_ck: dict, cond_at: int, n: int = 4) -> float:
    """The widened model must reproduce the PPO policy EXACTLY, for any prompt."""
    pc, rc = ppo_ck["config"], rvs_ck["config"]
    old = RecurrentEntityActorCritic(int(pc["global_dim"]), int(pc["n_actors"]), int(pc["actor_dim"]),
                                     int(pc["n_actions"]), hidden=int(pc.get("hidden", 256)),
                                     gru=int(pc.get("gru", 256)))
    old.load_state_dict(ppo_ck["model"]); old.eval()
    new = RecurrentEntityActorCritic(int(rc["global_dim"]), int(rc["n_actors"]), int(rc["actor_dim"]),
                                     int(rc["n_actions"]), hidden=int(rc.get("hidden", 256)),
                                     gru=int(rc.get("gru", 256)))
    new.load_state_dict(rvs_ck["model"]); new.eval()
    torch.manual_seed(0)
    raw = torch.randn(n, int(pc["obs_dim"]))
    raw[:, int(pc["global_dim"])::int(pc["actor_dim"])] = 3.0        # plausible actor ids
    worst = 0.0
    with torch.no_grad():
        for prompt in ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (7.0, -7.0, 0.5)):
            g = torch.tensor(prompt).expand(n, COND_DIM)
            wide = torch.cat([raw[:, :cond_at], g, raw[:, cond_at:]], -1)
            h0 = torch.zeros(n, old.hidden_size); first = torch.ones(n)
            lo, vo, _ = old.step(raw, h0, first)
            ln, vn, _ = new.step(wide, h0.clone(), first)
            worst = max(worst, float((lo - ln).abs().max()), float((vo - vn).abs().max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()
    ck = torch.load(a.ppo, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cond_at = int(cfg["global_dim"])          # the prompt goes at the END of the global block
    rvs, newcfg = widen(ck, cond_at)
    worst = check_identical(ck, rvs, cond_at)
    print(f"widened: {rvs['widened']}")
    print(f"obs_dim {cfg['obs_dim']} -> {newcfg['obs_dim']} | global {cfg['global_dim']} -> "
          f"{newcfg['global_dim']} | prompt at column {cond_at}")
    print(f"max |PPO - widened| over logits and value, across three prompts: {worst:.3e}")
    if worst > a.tol:
        raise SystemExit(f"REFUSING: the widened model is not the PPO policy (worst {worst:.3e} > {a.tol})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rvs, a.out)
    print(f"wrote {a.out} -- identical to {Path(a.ppo).name} for any prompt, so the conditioning starts "
          f"with zero effect and must be learned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
