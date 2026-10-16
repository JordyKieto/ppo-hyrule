"""Return-conditioned BC over the WHOLE corpus: 100 human wins plus thousands of PPO losses.

See zelda_rl/rvs.py for why the conditioning is outcome-based rather than reward-based (two reward profiles
in one corpus, and the human's own return is ~80% damage-independent).

Wins are rare and must be oversampled or the gradient is 99% losing play: `--win-weight` multiplies a winning
episode's loss weight. The log prints the resulting share of the gradient so the balance is visible and not
assumed.

    .venv/bin/python scripts/train_rvs.py --data datasets/human/gohma --data datasets/ppo/gohma_steps \\
        --win-weight 20 --epochs 40 --out checkpoints/kept/rvs.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tbh", REPO / "scripts" / "train_bc_human.py")
tbh = importlib.util.module_from_spec(spec); sys.modules["tbh"] = tbh; spec.loader.exec_module(tbh)

from zelda_rl import ckpt_s3  # noqa: E402
from zelda_rl.ppo import RecurrentEntityActorCritic  # noqa: E402
from zelda_rl.rvs import COND_DIM, TARGET, conditioning, field_offsets  # noqa: E402


def load_all(dirs, limit_per_dir=0):
    eps = []
    for d in dirs:
        files = sorted(Path(d).glob("*.npz"))
        if limit_per_dir:
            files = files[:limit_per_dir]
        for f in files:
            z = np.load(f, allow_pickle=False)
            if "obs" not in z.files or "action" not in z.files:
                continue
            obs = z["obs"].astype(np.float32)
            idx = field_offsets(str(z["obs_fields"]))
            won = bool(z["boss_defeated"]) if "boss_defeated" in z.files else False
            g = conditioning(obs, idx, won)
            eps.append({"file": f.name, "src": str(d), "obs": np.concatenate([obs, g], 1),
                        "act": z["action"].astype(np.int64), "win": won,
                        "dmg": int(z["boss_damage"]) if "boss_damage" in z.files else 0,
                        "obs_dim": obs.shape[1] + COND_DIM, "raw_dim": obs.shape[1],
                        "n_actions": int(z["n_actions"]) if "n_actions" in z.files else 108,
                        "fields": json.loads(str(z["obs_fields"]))})
    if not eps:
        raise SystemExit(f"no episodes in {dirs}")
    dims = {e["obs_dim"] for e in eps}
    if len(dims) != 1:
        raise SystemExit(f"mixed observation widths: {dims}")
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="repeatable: corpus directories")
    ap.add_argument("--limit-per-dir", type=int, default=0)
    ap.add_argument("--win-share", type=float, default=0.3,
                    help="TARGET fraction of the gradient that winning play should carry; the multiplier is "
                         "solved from the corpus to hit it. A fixed multiplier is the wrong knob: 20x was "
                         "sized for a ~1%%-win PPO corpus and on a 48%%-win mix it put wins at 93%% of the "
                         "gradient, which is plain BC with extra steps")
    ap.add_argument("--win-weight", type=float, default=0.0,
                    help="override the solved multiplier (0 = solve it from --win-share)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-episodes", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--episode-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/kept/rvs.pt")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)

    eps = load_all(a.data, a.limit_per_dir)
    wins = [e for e in eps if e["win"]]
    print(f"{len(eps)} episodes, {len(wins)} wins ({100*len(wins)/len(eps):.1f}%), "
          f"{sum(len(e['act']) for e in eps):,} steps")
    for d in a.data:
        sub = [e for e in eps if e["src"] == str(d)]
        print(f"  {d}: {len(sub)} episodes, {sum(1 for e in sub if e['win'])} wins, "
              f"{sum(len(e['act']) for e in sub):,} steps")

    # Solve the multiplier so winning play lands at --win-share of the gradient. w = s(1-p)/(p(1-s)) where
    # p is the winning fraction of STEPS, so the knob is the share we want and not a guess that only
    # happens to be right for one corpus composition.
    win_steps = sum(len(e["act"]) for e in eps if e["win"])
    all_steps = sum(len(e["act"]) for e in eps)
    p_ = win_steps / max(all_steps, 1)
    if a.win_weight > 0:
        solved = a.win_weight
    elif 0.0 < a.win_share < 1.0 and 0.0 < p_ < 1.0:
        solved = (a.win_share * (1 - p_)) / (p_ * (1 - a.win_share))
    else:
        solved = 1.0
    print(f"  winning play is {100*p_:.1f}% of steps; multiplier {solved:.2f} to reach "
          f"{100*a.win_share:.0f}% of the gradient")
    tot_w = win_w = 0.0
    for e in eps:
        w = solved if e["win"] else 1.0
        e["w"] = np.full(len(e["act"]), w, np.float32)
        tot_w += w * len(e["act"])
        win_w += (w * len(e["act"]) if e["win"] else 0.0)
    print(f"  -> winning play is {100*win_w/max(tot_w,1e-9):.1f}% of the gradient")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    n_val = max(2, int(a.val_frac * len(eps)))
    val_idx = set(order[:n_val].tolist())
    tr = [e for i, e in enumerate(eps) if i not in val_idx]
    va = [e for i, e in enumerate(eps) if i in val_idx]
    print(f"  train {len(tr)} / val {len(va)} (val wins: {sum(1 for e in va if e['win'])})")

    f0 = eps[0]["fields"]
    n_actors, actor_dim = f0["n_actors"], sum(w for _, w in f0["actor"])
    raw_global = eps[0]["raw_dim"] - n_actors * actor_dim
    global_dim = raw_global + COND_DIM          # the conditioning rides in the global block
    n_actions = eps[0]["n_actions"]
    print(f"  obs {eps[0]['obs_dim']} = global {raw_global} + cond {COND_DIM} + {n_actors} x {actor_dim}")

    model = RecurrentEntityActorCritic(global_dim, n_actors, actor_dim, n_actions, hidden=a.hidden, gru=a.gru)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = (0.0, -1)
    print(f"\n{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val acc':>9}{'val WIN acc':>13}")
    for ep_i in range(1, a.epochs + 1):
        model.train()
        tl, ta, _ = tbh.run_split(model, tr, a.chunk, opt, a.batch_episodes)
        model.eval()
        with torch.no_grad():
            _, vacc, _ = tbh.run_split(model, va, a.chunk, None, a.batch_episodes)
            vw = [e for e in va if e["win"]]
            _, vwacc, _ = (tbh.run_split(model, vw, a.chunk, None, a.batch_episodes) if vw
                           else (0.0, float("nan"), 0))
        print(f"{ep_i:>6}{tl:>12.4f}{100*ta:>10.1f}%{100*vacc:>8.1f}%{100*vwacc:>12.1f}%")
        score = vwacc if vwacc == vwacc else vacc        # accuracy ON WINNING play is what we want
        if score > best[0]:
            best = (score, ep_i)
            out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
            cfg = {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
                   "n_actions": n_actions, "obs_dim": eps[0]["obs_dim"], "hidden": a.hidden, "gru": a.gru,
                   "recurrent": True, "obs_version": 7, "task": "combat", "profile": "gohma",
                   "run_name": "rvs", "episode_seconds": a.episode_seconds, "vi_per_step": 6,
                   "goal_slot": 3, "steps_per_rollout": 0,
                   "rvs_cond": True, "rvs_target": list(TARGET), "rvs_cond_dim": COND_DIM,
                   "rvs_win_weight": solved, "rvs_win_share": a.win_share, "rvs_episodes": len(eps), "rvs_wins": len(wins),
                   "rvs_val_win_acc": float(score), "rvs_epoch": ep_i, "bc_seed": a.seed}
            from dataclasses import asdict
            from zelda_rl.combat import CombatConfig, GOHMA_SIMPLE
            cfg["combat_cfg"] = asdict(CombatConfig(profile="gohma", max_seconds=a.episode_seconds,
                                                    **GOHMA_SIMPLE))
            torch.save({"model": model.state_dict(), "optimizer": None, "update": 0, "global_step": 0,
                        "episode_count": 0, "config": cfg}, out)
            ckpt_s3.upload(out, "rvs", ep_i)
    print(f"\nbest val WIN accuracy {100*best[0]:.1f}% at epoch {best[1]} -> {a.out}")
    print(f"  inference prompt is g = {TARGET} (take all her HP, win, take no damage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
