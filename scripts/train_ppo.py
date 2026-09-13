"""Goal-conditioned PPO on OoT in Dolphin: start from savestate slots, learn to reach the goal slot's state.

Rollout = one 60 s (game time) episode per env, all envs reset at the start of each rollout, then one PPO update.
Logs: console + runs/<run>/train.log, episodes.jsonl, updates.jsonl, plots/, checkpoints/, and wandb.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/train_ppo.py --num-envs 2 --run-name kokiri-goal-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DIR_NAMES, N_ACTIONS, DolphinGoalEnv, EnvConfig, EnvError  # noqa: E402
from zelda_rl.goal import GoalConfig  # noqa: E402
from zelda_rl.obs import (ACTOR_DIM_V2, ACTOR_FIELDS_V2, GLOBAL_DIM_V2, GLOBAL_FIELDS_V2, N_ACTORS, OBS_DIM,  # noqa: E402
                          OBS_DIM_V2, OBS_FIELDS, read_state_file, read_state_file_v2)
from zelda_rl.plots import plot_trajectories  # noqa: E402
from zelda_rl.ppo import build_model, gae, ppo_update  # noqa: E402
from zelda_rl.runlog import LOGGER, RunLogger  # noqa: E402

DEFAULT_SLOTS_DIR = (os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves")
                     if sys.platform == "darwin" else "assets/states")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=time.strftime("goal-%Y%m%d-%H%M%S"))
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/<run-name>/checkpoints/latest.pt")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--total-updates", type=int, default=300)
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR, help="dir with NACE01.s01..s05")
    ap.add_argument("--start-slots", default="1,2,3")
    ap.add_argument("--goal-slot", type=int, default=5)
    ap.add_argument("--episode-seconds", type=float, default=60.0, help="game time per episode")
    ap.add_argument("--vi-per-step", type=int, default=6, help="60 Hz VIs per decision (6 = 10 Hz)")
    ap.add_argument("--video-backend", default="Null", help="Null = no window (training); Metal to watch")
    ap.add_argument("--headless", action="store_true", help="shorthand for --video-backend Null")
    ap.add_argument("--realtime", action="store_true", help="cap emulation at 1x (default: unlimited speed)")
    ap.add_argument("--obs-version", type=int, default=2, choices=[1, 2],
                    help="1: 44-d MLP obs, camera-relative actions; 2: actor table + state flags, world-relative actions")
    ap.add_argument("--torch-threads", type=int, default=2, help="keep torch from starving the Dolphin instances")
    # PPO
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    # goal / reward
    ap.add_argument("--success-radius", type=float, default=75.0)
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-rupee", type=float, default=1.0)
    ap.add_argument("--success-bonus", type=float, default=10.0)
    ap.add_argument("--w-flags", type=float, default=1.0)
    ap.add_argument("--goal-rupees", type=int, default=None, help="override the goal state's rupee count (e.g. 7)")
    ap.add_argument("--no-rupees-first", dest="rupees_first", action="store_false",
                    help="reward goal-position progress even before the rupee target is met")
    ap.add_argument("--w-rupee-approach", type=float, default=1.0)
    # logging
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "zelda-rl-goal-ppo"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-artifacts", action="store_true", help="also upload checkpoints as wandb artifacts")
    ap.add_argument("--wandb-group", default=None, help="group runs of one experiment (fleet sets this)")
    # fleet (RunPod): Upstash heartbeats, published policies, cross-pod weight averaging
    ap.add_argument("--fleet-experiment", default=None)
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world-size", type=int, default=1)
    ap.add_argument("--fleet-launch-id", default="0", help="namespaces sync keys per launch (set by fleet launch)")
    ap.add_argument("--fleet-sync", default="avg", choices=["avg", "none"], help="average weights across pods each update")
    ap.add_argument("--publish-every", type=int, default=5, help="publish policy to Upstash every N updates")
    ap.add_argument("--plot-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()
    if args.headless:
        args.video_backend = "Null"
    return args


def main():
    args = parse_args()
    run_dir = Path(args.out_dir) / args.run_name
    for sub in ("checkpoints", "plots", "instances"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    # ---- savestates -> start/goal game states ---------------------------------------------
    start_slots = [int(s) for s in args.start_slots.split(",")]
    slot_ids = sorted(set(start_slots + [args.goal_slot]))
    slot_files = {k: Path(args.slots_dir) / f"NACE01.s{k:02d}" for k in slot_ids}
    reader = read_state_file_v2 if args.obs_version >= 2 else read_state_file
    slot_states = {k: reader(f) for k, f in slot_files.items()}
    if args.goal_rupees is not None:
        import dataclasses
        slot_states[args.goal_slot] = dataclasses.replace(slot_states[args.goal_slot], rupees=args.goal_rupees)
    goal = slot_states[args.goal_slot]
    goal_cfg = GoalConfig(success_radius=args.success_radius, w_pos=args.w_pos, w_rupee=args.w_rupee,
                          success_bonus=args.success_bonus, w_flags=args.w_flags,
                          rupees_first=args.rupees_first, w_rupee_approach=args.w_rupee_approach)
    env_cfg = EnvConfig(episode_vi=int(args.episode_seconds * 60), vi_per_step=args.vi_per_step,
                        video_backend=args.video_backend, unlimited_speed=not args.realtime,
                        obs_version=args.obs_version, goal=goal_cfg)
    T, N = env_cfg.max_steps, args.num_envs

    v2 = args.obs_version >= 2
    obs_dim = OBS_DIM_V2 if v2 else OBS_DIM
    obs_spec = ({"global_dim": GLOBAL_DIM_V2, "n_actors": N_ACTORS, "actor_dim": ACTOR_DIM_V2,
                 "obs_fields": GLOBAL_FIELDS_V2, "actor_fields": ACTOR_FIELDS_V2,
                 "action_frame": "world"} if v2 else {"obs_fields": OBS_FIELDS, "action_frame": "camera"})
    config = {**vars(args), **obs_spec, "obs_dim": obs_dim, "n_actions": N_ACTIONS,
              "steps_per_episode": T, "batch_size": T * N, "goal_cfg": asdict(goal_cfg),
              "slots": {k: {"scene": s.scene, "rupees": s.rupees, "pos": [round(v, 1) for v in s.pos]}
                        for k, s in slot_states.items()}}
    log = RunLogger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                    run_name=args.run_name, resume=args.resume, tags=["goal-ppo", f"goal-slot-{args.goal_slot}"],
                    group=args.wandb_group or args.fleet_experiment)
    link = None
    if args.fleet_experiment:
        from zelda_rl.fleet_link import FleetLink
        link = FleetLink(args.fleet_experiment, args.fleet_rank, args.fleet_world_size, args.fleet_launch_id)
        LOGGER.info("fleet: experiment %s rank %d/%d sync=%s", args.fleet_experiment, args.fleet_rank,
                    args.fleet_world_size, args.fleet_sync)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    LOGGER.info("run dir %s | obs v%d (%d-d) | actions %d | %d envs x %d steps (%.0fs game @ %d Hz) | backend %s | %s speed",
                run_dir, args.obs_version, obs_dim, N_ACTIONS, N, T, args.episode_seconds, 60 // args.vi_per_step,
                args.video_backend, "1x" if args.realtime else "unlimited")
    for k, s in slot_states.items():
        role = "GOAL " if k == args.goal_slot else "start"
        LOGGER.info("  slot %d %s scene=%d rupees=%d pos=(%.0f, %.0f, %.0f)", k, role, s.scene, s.rupees, *s.pos)

    # ---- envs -----------------------------------------------------------------------------
    t_boot = time.time()
    def make_env(i: int) -> DolphinGoalEnv:
        return DolphinGoalEnv(i, run_dir / "instances" / f"env{i}", slot_files, slot_states, goal, env_cfg)

    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(make_env, range(N)))
    LOGGER.info("booted %d Dolphin instances in %.1fs", N, time.time() - t_boot)
    usable = []
    for k in start_slots:
        envs[0].load_slot(k)
        chk = envs[0].input_check(wait_vi=60)
        LOGGER.info("  slot %d input check: %s", k, chk)
        (usable.append(k) if chk["game_sees_stick"] else LOGGER.warning("  slot %d ignores input (frozen?) -> excluded", k))
    if not usable:
        raise SystemExit("no usable start slots")

    # ---- model ----------------------------------------------------------------------------
    torch.manual_seed(1234)                      # identical initial weights on every fleet rank
    model = build_model(config)
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start_update, global_step, episode_count = 1, 0, 0
    best = {"min_pos_dist": float("inf"), "success_rate": 0.0}
    ckpt_latest = run_dir / "checkpoints" / "latest.pt"
    if args.resume and ckpt_latest.exists():
        ck = torch.load(ckpt_latest, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_update, global_step, episode_count = ck["update"] + 1, ck["global_step"], ck["episode_count"]
        best = ck.get("best", best)
        LOGGER.info("resumed from update %d (global_step %d)", ck["update"], global_step)

    obs_buf = np.zeros((T, N, obs_dim), np.float32)
    act_buf = np.zeros((T, N), np.int64)
    logp_buf, val_buf, rew_buf, done_buf = (np.zeros((T, N), np.float32) for _ in range(4))
    header = (f"{'upd':>4} {'steps':>8} {'eps':>5} {'succ%':>6} {'start→final dist':>17} {'min dist':>8} "
              f"{'rupees':>8} {'return':>7} {'kl':>6} {'ent':>5} {'sps':>5} {'emu':>5}")

    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)       # `kill -USR1 <pid>` dumps all stacks to the log
    try:
        for update in range(start_update, args.total_updates + 1):
            faulthandler.dump_traceback_later(900, repeat=True)      # an update taking >15 min dumps stacks (hang post-mortem)
            for attempt in range(3):
                try:
                    t_roll = time.perf_counter()
                    eps_done: list[dict] = []
                    trackers = [None] * N

                    def begin(i):
                        slot = int(rng.choice(usable))
                        o = envs[i].reset(slot)
                        m = envs[i].metrics()
                        s = envs[i].state
                        trackers[i] = {"slot": slot, "env": i, "start_step": t_idx, "return": 0.0, "length": 0,
                                       "start_pos_dist": m.pos_dist, "min_pos_dist": m.pos_dist, "min_goal_dist": m.goal_dist,
                                       "start_rupees": s.rupees, "max_rupees": s.rupees, "success": False,
                                       "traj_x": [s.pos[0]], "traj_z": [s.pos[2]], "actions": np.zeros(N_ACTIONS, np.int64),
                                       "vi_overshoot": 0, "open": True, "talk_steps": 0, "ztarget_steps": 0}
                        return o

                    def finish(i, terminated):
                        tr, e = trackers[i], envs[i]
                        tr["open"] = False
                        m = e.metrics()
                        ep = {k: v for k, v in tr.items() if k not in ("actions", "open")}
                        ep.update(final_pos_dist=m.pos_dist, final_goal_dist=m.goal_dist, final_rupees=e.state.rupees,
                                  final_rupee_gap=m.rupee_gap, final_flags_missing=m.flags_missing,
                                  talking_frac=tr["talk_steps"] / max(tr["length"], 1),
                                  ztarget_frac=tr["ztarget_steps"] / max(tr["length"], 1),
                                  terminated=terminated, left_scene=not m.same_scene,
                                  partial=tr["start_step"] > 0, update=update,
                                  pos_progress=(tr["start_pos_dist"] - m.pos_dist) / max(tr["start_pos_dist"], 1e-6),
                                  action_counts=tr["actions"].tolist())
                        eps_done.append(ep)
                        return ep

                    t_idx = 0
                    obs = np.stack([begin(i) for i in range(N)])
                    model.eval()
                    for t_idx in range(T):
                        obs_buf[t_idx] = obs
                        with torch.no_grad():
                            a, logp, _, v = model.act(torch.as_tensor(obs))
                        a = a.numpy()
                        act_buf[t_idx], logp_buf[t_idx], val_buf[t_idx] = a, logp.numpy(), v.numpy()
                        for i, e in enumerate(envs):
                            e.send_action(int(a[i]))
                        pending = set(range(N))
                        results = [None] * N
                        while pending:
                            for i in list(pending):
                                if envs[i].ready():
                                    results[i] = envs[i].observe()
                                    pending.discard(i)
                            if pending:
                                time.sleep(0.0005)
                        for i, (o, r, term, trunc, info) in enumerate(results):
                            tr = trackers[i]
                            m, s = info["metrics"], info["state"]
                            rew_buf[t_idx, i], done_buf[t_idx, i] = r, float(term)
                            tr["return"] += r; tr["length"] += 1; tr["actions"][a[i]] += 1
                            tr["min_pos_dist"] = min(tr["min_pos_dist"], m.pos_dist)
                            tr["min_goal_dist"] = min(tr["min_goal_dist"], m.goal_dist)
                            tr["max_rupees"] = max(tr["max_rupees"], s.rupees)
                            tr["success"] |= m.success
                            tr["vi_overshoot"] += info["vi_elapsed"] - args.vi_per_step
                            sf1 = getattr(s, "sf1", 0)
                            tr["talk_steps"] += sf1 >> 6 & 1
                            tr["ztarget_steps"] += sf1 >> 15 & 1
                            if t_idx % 2 == 0:
                                tr["traj_x"].append(s.pos[0]); tr["traj_z"].append(s.pos[2])
                            if term:
                                ep = finish(i, True)
                                LOGGER.info("  env%d slot%d %s at step %d | dist %.0f | rupees %d", i, ep["slot"],
                                            "SUCCESS" if ep["success"] else "left scene", ep["length"], ep["final_pos_dist"],
                                            ep["final_rupees"])
                                o = begin(i) if t_idx < T - 1 else o
                            obs[i] = o
                    for i in range(N):
                        if trackers[i]["open"] and trackers[i]["length"] > 0:
                            finish(i, False)
                    rollout_s = time.perf_counter() - t_roll
                    emu_speed = sum(ep["length"] for ep in eps_done) * args.vi_per_step / N / 60 / rollout_s
                    break
                except EnvError as err:
                    i_bad = err.idx if err.idx >= 0 else 0
                    LOGGER.warning("rollout aborted (%s) -> diagnosing, restarting env%d, retrying update %d", err, i_bad, update)
                    diag = envs[i_bad].diagnose()
                    LOGGER.warning("  stall diagnosis env%d: %s", i_bad, json.dumps(diag, default=str))
                    with open(run_dir / "stalls.jsonl", "a") as f:
                        f.write(json.dumps({"update": update, "error": str(err), **diag}, default=str) + "\n")
                    envs[i_bad].close()
                    envs[i_bad] = make_env(i_bad)
            else:
                raise RuntimeError(f"update {update}: 3 rollout attempts failed")

            # ---- PPO update -----------------------------------------------------------------
            with torch.no_grad():
                next_v = model.value(torch.as_tensor(obs)).numpy()
            obs_abs_max = float(np.abs(obs_buf).max())
            if not (np.isfinite(obs_buf).all() and np.isfinite(rew_buf).all() and np.isfinite(val_buf).all()):
                dump = run_dir / f"bad_batch_u{update:05d}.npz"
                np.savez_compressed(dump, obs=obs_buf, rew=rew_buf, val=val_buf, act=act_buf)
                LOGGER.warning("non-finite rollout data at update %d (dumped %s) -> sanitizing", update, dump)
                for buf in (obs_buf, rew_buf, val_buf):
                    np.nan_to_num(buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            adv, ret = gae(rew_buf, val_buf, done_buf, next_v, args.gamma, args.gae_lambda)
            batch = {"obs": obs_buf.reshape(T * N, -1), "act": act_buf.reshape(-1), "logp": logp_buf.reshape(-1),
                     "adv": adv.reshape(-1), "ret": ret.reshape(-1), "val": val_buf.reshape(-1)}
            t_upd = time.perf_counter()
            model.train()
            stats = ppo_update(model, opt, batch, epochs=args.epochs, minibatches=args.minibatches, clip=args.clip,
                               ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                               target_kl=args.target_kl)
            update_s = time.perf_counter() - t_upd
            if stats["skipped_minibatches"]:
                LOGGER.warning("update %d: skipped %d non-finite minibatches", update, stats["skipped_minibatches"])
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError(f"update {update}: model parameters became non-finite")
            sync_stats = {}
            if link is not None and args.fleet_world_size > 1 and args.fleet_sync == "avg":
                try:
                    sync_stats = link.average_weights(model, update, timeout=120)
                except Exception as e:  # noqa: BLE001 -- a relay hiccup must not kill the run
                    LOGGER.warning("fleet weight sync failed at update %d: %s", update, e)
            global_step += T * N

            # ---- metrics / logs -------------------------------------------------------------
            full = [ep for ep in eps_done if not ep["partial"]] or eps_done
            episode_count += len(eps_done)
            for ep in eps_done:
                log.episode({k: v for k, v in ep.items() if not k.startswith("traj_")}, global_step)
            mean = lambda key, eps=full: float(np.mean([ep[key] for ep in eps])) if eps else float("nan")
            succ = mean("success", eps_done) * 100
            metrics = {
                "episode/count": episode_count, "episode/return": mean("return"), "episode/length": mean("length"),
                "episode/left_scene_rate": mean("left_scene", eps_done),
                "goal/success_rate": succ / 100, "goal/start_pos_dist": mean("start_pos_dist"),
                "goal/final_pos_dist": mean("final_pos_dist"), "goal/min_pos_dist": mean("min_pos_dist"),
                "goal/pos_progress": mean("pos_progress"), "goal/final_rupees": mean("final_rupees"),
                "goal/max_rupees": mean("max_rupees"), "goal/final_rupee_gap": mean("final_rupee_gap"),
                "goal/final_goal_dist": mean("final_goal_dist"), "goal/min_goal_dist": mean("min_goal_dist"),
                "goal/final_flags_missing": mean("final_flags_missing"),
                "episode/talking_frac": mean("talking_frac", eps_done), "episode/ztarget_frac": mean("ztarget_frac", eps_done),
                "goal/best_min_pos_dist_ever": min(best["min_pos_dist"], min(ep["min_pos_dist"] for ep in eps_done)),
                **{f"ppo/{k}": v for k, v in stats.items()},
                "ppo/lr": opt.param_groups[0]["lr"],
                "perf/env_steps_per_sec": T * N / rollout_s, "perf/rollout_seconds": rollout_s,
                "perf/update_seconds": update_s, "perf/emu_speed_x": emu_speed, "perf/obs_abs_max": obs_abs_max,
                "perf/vi_overshoot_per_step": mean("vi_overshoot", eps_done) / max(mean("length", eps_done), 1),
                "update": update,
                **{f"fleet/{k}": v for k, v in sync_stats.items() if k != "weight_hash"},
            }
            for k in usable:
                se = [ep for ep in full if ep["slot"] == k]
                if se:
                    metrics[f"by_slot/s{k}/final_pos_dist"] = mean("final_pos_dist", se)
                    metrics[f"by_slot/s{k}/min_pos_dist"] = mean("min_pos_dist", se)
                    metrics[f"by_slot/s{k}/final_rupees"] = mean("final_rupees", se)
                    metrics[f"by_slot/s{k}/success_rate"] = mean("success", se)
            acts = np.sum([ep["action_counts"] for ep in eps_done], axis=0)
            top = ", ".join(f"{DIR_NAMES[a // 3]}{['', '+A', '+B'][a % 3]}:{c / acts.sum():.0%}"
                            for a, c in sorted(enumerate(acts), key=lambda x: -x[1])[:3])
            log.update(metrics, global_step)
            if link is not None:
                hb = {k.split("/")[-1]: v for k, v in metrics.items() if k in (
                    "goal/success_rate", "goal/final_pos_dist", "goal/min_pos_dist", "goal/final_rupees",
                    "episode/return", "perf/env_steps_per_sec", "perf/emu_speed_x", "ppo/approx_kl", "ppo/entropy")}
                try:
                    link.heartbeat({"update": update, "global_step": global_step, "run": args.run_name,
                                    "weight_hash": sync_stats.get("weight_hash"), "sync_ranks": sync_stats.get("sync_ranks"),
                                    "sync_wait_s": sync_stats.get("sync_wait_s"), **hb})
                    if update % args.publish_every == 0:
                        link.publish_policy(model, config, update, global_step,
                                            {"episode_count": episode_count, **hb})
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("fleet heartbeat/publish failed at update %d: %s", update, e)

            best["min_pos_dist"] = metrics["goal/best_min_pos_dist_ever"]
            best["success_rate"] = max(best["success_rate"], succ / 100)
            log.summary({"best/min_pos_dist": best["min_pos_dist"], "best/success_rate": best["success_rate"]})

            if update == start_update or update % 20 == 0:
                LOGGER.info(header)
            LOGGER.info(f"{update:>4} {global_step:>8,} {episode_count:>5} {succ:>5.0f}% "
                        f"{metrics['goal/start_pos_dist']:>7.0f} → {metrics['goal/final_pos_dist']:<7.0f} "
                        f"{metrics['goal/min_pos_dist']:>8.0f} {metrics['goal/final_rupees']:>4.1f}/{goal.rupees:<3d} "
                        f"{metrics['episode/return']:>7.2f} {stats['approx_kl']:>6.3f} {stats['entropy']:>5.2f} "
                        f"{metrics['perf/env_steps_per_sec']:>5.0f} {emu_speed:>4.1f}x  top: {top}"
                        + (f"  sync {sync_stats['sync_ranks']}r {sync_stats['weight_hash']}" if sync_stats else ""))

            if update % args.plot_every == 0 or update == start_update:
                p = run_dir / "plots" / f"traj_u{update:05d}.png"
                items = {tuple(round(c) for c in pos): (v, pos) for k in usable for v, pos in slot_states[k].rupee_items}
                plot_trajectories(eps_done, goal.pos, list(items.values()), args.success_radius, p,
                                  f"{args.run_name} update {update} | success {succ:.0f}% | "
                                  f"final dist {metrics['goal/final_pos_dist']:.0f}")
                log.image("charts/trajectories", p, global_step, caption=f"update {update}")

            if update % args.save_every == 0 or update == args.total_updates:
                ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": update,
                      "global_step": global_step, "episode_count": episode_count, "best": best, "config": config}
                path = run_dir / "checkpoints" / f"update_{update:05d}.pt"
                torch.save(ck, path)
                torch.save(ck, ckpt_latest)
                if args.wandb_artifacts:
                    log.artifact(path)
    finally:
        for e in envs:
            e.close()
        log.close()


if __name__ == "__main__":
    main()
