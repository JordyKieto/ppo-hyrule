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
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)   # AGENTS 28: this script now opens Dolphins for its online eval
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
        n_before = len(eps)
        bad = 0
        for f in files:
            # A corpus is downloaded, and a download can be truncated. This is the SECOND loader to die on
            # `EOFError: No data left in file` today -- zelda_rl.rvs_data was fixed this morning and this
            # function has its own np.load, so it crashed the M2-shape cell (gohma-la r0). Skip and REPORT.
            try:
                z = np.load(f, allow_pickle=False)
                if "obs" not in z.files or "action" not in z.files:
                    continue
                z["obs"].shape, z["action"].shape      # force the read: a lazy npz raises here
            except Exception:                          # noqa: BLE001
                bad += 1
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
        load_all.skipped = getattr(load_all, "skipped", 0) + bad
        if bad:
            print(f"  {d}: SKIPPED {bad} unreadable of {len(files)} file(s), kept "
                  f"{len(eps) - n_before}", flush=True)
    if not eps:
        raise SystemExit(f"no episodes in {dirs} ({getattr(load_all, 'skipped', 0)} unreadable)")
    dims = {e["obs_dim"] for e in eps}
    if len(dims) != 1:
        raise SystemExit(f"mixed observation widths: {dims}")
    return eps


def online_eval(ckpt: Path, a, ep_i: int, hist: list[tuple[int, float]]) -> float | None:
    """Play the game with the TARGET prompt and print balanced boss damage with its slope.

    Reuses scripts/eval_fleet.evaluate rather than reimplementing a rollout -- one tested path, and the
    checkpoint is loaded through policy_io, which wraps an `rvs_cond` checkpoint with the target prompt
    g = (1, 1, 0) automatically. Balanced = the UNWEIGHTED mean of the per-slot means, because an
    episode-weighted average moves with the slot mix and cannot compare arms (AGENTS 29).
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("eval_fleet", Path(__file__).with_name("eval_fleet.py"))
    ef = _ilu.module_from_spec(spec); spec.loader.exec_module(ef)
    slots = [int(x) for x in a.eval_slots.split(",") if x.strip()]
    try:
        eps, _ = ef.evaluate(ckpt, slots, a.eval_episodes, a.eval_envs, a.slots_dir, a.episode_seconds)
    except Exception as e:  # noqa: BLE001
        print(f"  EVAL epoch {ep_i}: FAILED ({type(e).__name__}: {str(e)[:90]})", flush=True)
        return None
    if not eps:
        print(f"  EVAL epoch {ep_i}: no episodes", flush=True)
        return None
    per = {}
    for e in eps:
        per.setdefault(e["slot"], []).append(float(e.get("boss_damage", 0)))
    bal = float(np.mean([np.mean(v) for v in per.values()]))
    pooled = float(np.mean([d for v in per.values() for d in v]))
    wins = sum(1 for e in eps if e.get("boss_defeated"))
    hist.append((ep_i, bal))
    trend = ""
    if len(hist) >= 3:
        u = np.array([h[0] for h in hist], float); d = np.array([h[1] for h in hist], float)
        A = np.vstack([u, np.ones_like(u)]).T
        beta, *_ = np.linalg.lstsq(A, d, rcond=None)
        r = d - A @ beta
        se = float(np.sqrt((r @ r / max(len(u) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = beta[0] / se if se > 0 else float("nan")
        trend = (f" | slope {beta[0]:+.4f} +- {se:.4f} /epoch  t={t:+.2f}  "
                 f"{'RISING' if t > 2 else 'FALLING' if t < -2 else 'flat'} over {len(hist)} evals")
    print(f"  EVAL epoch {ep_i}: dmgB(balanced) {bal:.3f} | pooled {pooled:.3f} | wins {wins}/{len(eps)} | "
          f"slots {len(per)}{trend}", flush=True)
    return bal


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
    # ONLINE eval inside the arm. Offline accuracy is not the headline: the clone reached 53% val accuracy
    # and 0.33 boss damage, and the only number that ranks a policy is the damage it does in the game
    # (AGENTS 34). Evaluating here means one pod gives training AND its own feedback loop, with a slope.
    ap.add_argument("--eval-every", type=int, default=4, help="online eval every N epochs (0 = never)")
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--eval-slots", default="3,4,5,6,7,8")
    ap.add_argument("--slots-dir", default="assets/states")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)

    eps = load_all(a.data, a.limit_per_dir)
    wins = [e for e in eps if e["win"]]
    print(f"{len(eps)} episodes, {len(wins)} wins ({100*len(wins)/len(eps):.1f}%), "
          f"{sum(len(e['act']) for e in eps):,} steps")
    if getattr(load_all, "skipped", 0):
        print(f"  SKIPPED {load_all.skipped} unreadable .npz file(s)", flush=True)
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
    best, best_dmg, hist = (0.0, -1), (0.0, -1), []
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
        do_eval = a.eval_every and (ep_i % a.eval_every == 0 or ep_i == a.epochs)
        if do_eval or score > best[0]:
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
        if do_eval:
            ev = online_eval(Path(a.out), a, ep_i, hist)
            if ev is not None and ev > best_dmg[0]:
                best_dmg = (ev, ep_i)
                keep = Path(a.out).with_name(Path(a.out).stem + "_bestdmg.pt")
                shutil.copy(Path(a.out), keep)
                ckpt_s3.upload(keep, "rvs_bestdmg", ep_i)
    print(f"\nbest val WIN accuracy {100*best[0]:.1f}% at epoch {best[1]} -> {a.out}")
    if best_dmg[1] > 0:
        print(f"best ONLINE boss damage (balanced) {best_dmg[0]:.3f} at epoch {best_dmg[1]}")
    print(f"  inference prompt is g = {TARGET} (take all her HP, win, take no damage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
