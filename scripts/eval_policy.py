"""Evaluate a trained goal-PPO checkpoint on local Dolphin (visible window, real-time by default).

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/eval_policy.py --checkpoint runs/kokiri-goal-v1/checkpoints/latest.pt --episodes 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import N_ACTIONS, DolphinGoalEnv, EnvConfig  # noqa: E402
from zelda_rl.goal import GoalConfig  # noqa: E402
from zelda_rl.obs import OBS_DIM, read_state_file  # noqa: E402
from zelda_rl.plots import plot_trajectories  # noqa: E402
from zelda_rl.ppo import ActorCritic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3, help="episodes per start slot")
    ap.add_argument("--start-slots", default=None, help="default: from the checkpoint config")
    ap.add_argument("--slots-dir", default=os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
    ap.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--fast", action="store_true", help="unlimited emulation speed")
    ap.add_argument("--out", default=None, help="output dir (default: <run>/eval/<timestamp>)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    cfg = ck["config"]
    assert cfg["obs_dim"] == OBS_DIM and cfg["n_actions"] == N_ACTIONS, "checkpoint obs/action spec mismatch"
    model = ActorCritic(OBS_DIM, N_ACTIONS, cfg["hidden"])
    model.load_state_dict(ck["model"])
    model.eval()

    start_slots = [int(s) for s in (args.start_slots or cfg["start_slots"]).split(",")]
    goal_slot = cfg["goal_slot"]
    files = {k: Path(args.slots_dir) / f"NACE01.s{k:02d}" for k in sorted(set(start_slots + [goal_slot]))}
    states = {k: read_state_file(f) for k, f in files.items()}
    goal = states[goal_slot]
    gcfg = GoalConfig(**cfg["goal_cfg"])
    ecfg = EnvConfig(episode_vi=int(cfg["episode_seconds"] * 60), vi_per_step=cfg["vi_per_step"],
                     video_backend="Metal", unlimited_speed=args.fast, goal=gcfg)
    out = Path(args.out or Path(args.checkpoint).parents[1] / "eval" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    env = DolphinGoalEnv(0, out / "instance", files, states, goal, ecfg)
    print(f"checkpoint update {ck['update']} (global_step {ck['global_step']:,}) | goal slot {goal_slot} "
          f"pos {tuple(round(v) for v in goal.pos)} rupees {goal.rupees} | {'deterministic' if args.deterministic else 'sampled'}")
    episodes = []
    try:
        for slot in start_slots:
            for n in range(args.episodes):
                obs = env.reset(slot)
                m0 = env.metrics()
                ep = {"slot": slot, "return": 0.0, "min_pos_dist": m0.pos_dist, "start_pos_dist": m0.pos_dist,
                      "traj_x": [env.state.pos[0]], "traj_z": [env.state.pos[2]], "success": False}
                for t in range(ecfg.max_steps):
                    with torch.no_grad():
                        a, *_ = model.act(torch.as_tensor(obs[None]), deterministic=args.deterministic)
                    env.send_action(int(a[0]))
                    while not env.ready():
                        time.sleep(0.0005)
                    obs, r, term, trunc, info = env.observe()
                    m = info["metrics"]
                    ep["return"] += r
                    ep["min_pos_dist"] = min(ep["min_pos_dist"], m.pos_dist)
                    ep["success"] |= m.success
                    ep["traj_x"].append(env.state.pos[0]); ep["traj_z"].append(env.state.pos[2])
                    if term or trunc:
                        break
                m = env.metrics()
                ep.update(final_pos_dist=m.pos_dist, final_rupees=env.state.rupees, length=t + 1)
                episodes.append(ep)
                print(f"slot {slot} ep {n}: success={ep['success']} dist {ep['start_pos_dist']:.0f} → {m.pos_dist:.0f} "
                      f"(min {ep['min_pos_dist']:.0f}) rupees {env.state.rupees}/{goal.rupees} return {ep['return']:.2f}")
    finally:
        env.close()
    with open(out / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps({k: v for k, v in ep.items() if not k.startswith("traj_")}) + "\n")
    items = {tuple(round(c) for c in p): (v, p) for k in start_slots for v, p in states[k].rupee_items}
    plot_trajectories(episodes, goal.pos, list(items.values()), gcfg.success_radius, out / "trajectories.png",
                      f"eval update {ck['update']} | success {np.mean([e['success'] for e in episodes]):.0%}")
    print(f"success {np.mean([e['success'] for e in episodes]):.0%} | mean final dist "
          f"{np.mean([e['final_pos_dist'] for e in episodes]):.0f} | wrote {out}")


if __name__ == "__main__":
    main()
