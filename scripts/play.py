"""Watch a PPO checkpoint play OoT in a local Dolphin window, with a live terminal HUD.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin .venv/bin/python-dbg scripts/play.py [CKPT]

CKPT: a .pt file | a run name (runs/<name>/checkpoints/latest.pt) | omitted = newest .pt in checkpoints/inbox/
(then runs/*/checkpoints/). Examples:
    play.py                                   # newest dropped/synced checkpoint, all start slots, real time
    play.py kokiri-goal-v2 --slot 2 -n 5      # a run's latest, slot 2, 5 episodes
    play.py ckpt.pt --fast --deterministic    # uncapped speed, argmax actions
    play.py --watch                           # loop forever, hot-swap to newer checkpoints between episodes
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
from zelda_rl.env import BUTTONS, DIR_NAMES  # noqa: E402
from zelda_rl.policy_io import (DEFAULT_SLOTS_DIR, INBOX, load_policy, make_env, newest_checkpoint,  # noqa: E402
                                resolve_checkpoint)


def action_name(a: int) -> str:
    return f"{DIR_NAMES[a // 3]}{'+' + BUTTONS[a % 3] if a % 3 else ''}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--slot", type=int, action="append", help="start slot(s); repeatable (default: checkpoint's)")
    ap.add_argument("-n", "--episodes", type=int, default=1, help="episodes per start slot")
    ap.add_argument("--fast", action="store_true", help="uncapped emulation speed")
    ap.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--watch", action="store_true", help="loop forever; reload when a newer checkpoint appears")
    ap.add_argument("--headless", action="store_true", help="no Dolphin window (Null backend)")
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    ap.add_argument("--hud-hz", type=float, default=2.0)
    ap.add_argument("--episode-seconds", type=float, default=None,
                    help="override the checkpoint's episode length (game time), e.g. 180 for long-horizon starts")
    ap.add_argument("--upstash", metavar="EXPERIMENT", default=None,
                    help="pull the newest policy the RunPod fleet published for EXPERIMENT (into checkpoints/inbox/)")
    args = ap.parse_args()

    def pull_fleet():
        from zelda_rl.fleet_link import fetch_latest_policy
        got = fetch_latest_policy(args.upstash, INBOX)
        if got:
            print(f"⇣ fleet policy {got[1]['key']} (update {got[1]['update']}, rank {got[1]['rank']})")
        return got

    if args.upstash and not args.checkpoint:
        got = pull_fleet()
        if not got:
            raise SystemExit(f"no published policy for experiment {args.upstash!r} yet")
        args.checkpoint = got[0]
    path = resolve_checkpoint(args.checkpoint)
    model, cfg, ck = load_policy(path)
    if args.episode_seconds:
        cfg["episode_seconds"] = args.episode_seconds
    slots = args.slot or [int(s) for s in str(cfg["start_slots"]).split(",")]
    if cfg.get("task") == "combat":
        return play_combat(args, path, model, cfg, ck, slots)
    out = Path("runs/play") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    env, states = make_env(cfg, out / "instance", slots=slots, slots_dir=args.slots_dir,
                           video_backend="Null" if args.headless else "Metal", unlimited_speed=args.fast)
    goal = env.goal
    print(f"▶ {path}  (update {ck['update']}, {ck['global_step']:,} steps, obs v{cfg.get('obs_version', 1)})\n"
          f"  goal slot {cfg['goal_slot']}: pos ({goal.pos[0]:.0f}, {goal.pos[1]:.0f}, {goal.pos[2]:.0f}) rupees {goal.rupees}"
          f" | start slots {slots} | {'argmax' if args.deterministic else 'sampled'} | {'uncapped' if args.fast else 'real time'}")
    results = []
    loaded_mtime = path.stat().st_mtime
    try:
        while True:
            for slot in slots:
                for n in range(args.episodes):
                    if args.watch:
                        if args.upstash:
                            pull_fleet()
                        newest = newest_checkpoint()
                        if newest and (newest != path or newest.stat().st_mtime > loaded_mtime):
                            path = newest
                            model, cfg2, ck = load_policy(path)
                            loaded_mtime = path.stat().st_mtime
                            print(f"↻ reloaded {path} (update {ck['update']})")
                    obs = env.reset(slot)
                    m0 = env.metrics()
                    ep = {"slot": slot, "return": 0.0, "min_dist": m0.pos_dist, "start_dist": m0.pos_dist,
                          "start_rupees": env.state.rupees, "success": False}
                    t_hud = 0.0
                    print(f"\n── slot {slot} ep {n + 1}/{args.episodes}: start dist {m0.pos_dist:.0f}, rupees {env.state.rupees}")
                    for t in range(env.cfg.max_steps):
                        with torch.no_grad():
                            x = torch.as_tensor(obs[None])
                            a, _, _, v = model.act(x, deterministic=args.deterministic)
                            if hasattr(model, "_features"):
                                probs = torch.softmax(model.actor(model._features(x)), -1)[0]
                            else:
                                probs = torch.softmax(model.actor(x), -1)[0]
                        a = int(a[0])
                        env.send_action(a)
                        while not env.ready():
                            time.sleep(0.0005)
                        obs, r, term, trunc, info = env.observe()
                        m, s = info["metrics"], info["state"]
                        ep["return"] += r
                        ep["min_dist"] = min(ep["min_dist"], m.pos_dist)
                        ep["success"] |= m.success
                        now = time.perf_counter()
                        if now - t_hud >= 1.0 / args.hud_hz or term:
                            t_hud = now
                            top = torch.topk(probs, 3)
                            top_s = " ".join(f"{action_name(int(i))}:{float(p):.0%}" for p, i in zip(top.values, top.indices))
                            extra = ""
                            if hasattr(s, "sf1"):
                                from zelda_rl.obs import SF1_BITS
                                bits = [nm for nm, b in SF1_BITS if s.sf1 >> b & 1]
                                near = s.actors[0] if s.actors else None
                                extra = (f" | state {','.join(bits) or '-'}"
                                         + (f" | nearest id {near.id:#06x} cat {near.category} @{near.dist:.0f}" if near else ""))
                            print(f"  t={t * env.cfg.vi_per_step / 60:5.1f}s  {action_name(a):6s} V={float(v[0]):6.2f}  "
                                  f"dist {m.pos_dist:6.0f} (min {ep['min_dist']:.0f})  rupees {s.rupees}/{goal.rupees}  "
                                  f"R={ep['return']:6.2f}  top[{top_s}]{extra}")
                        if term or trunc:
                            break
                    ep.update(final_dist=env.metrics().pos_dist, final_rupees=env.state.rupees, steps=t + 1)
                    results.append(ep)
                    print(f"■ slot {slot}: {'SUCCESS' if ep['success'] else 'no success'} | dist {ep['start_dist']:.0f} → "
                          f"{ep['final_dist']:.0f} (min {ep['min_dist']:.0f}) | rupees {ep['start_rupees']} → "
                          f"{ep['final_rupees']}/{goal.rupees} | return {ep['return']:.2f}")
            if not args.watch:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        with open(out / "episodes.jsonl", "w") as f:
            for ep in results:
                f.write(json.dumps(ep) + "\n")
        if results:
            print(f"\nsummary: {len(results)} episodes | success {np.mean([e['success'] for e in results]):.0%} | "
                  f"mean final dist {np.mean([e['final_dist'] for e in results]):.0f} | "
                  f"mean rupees gained {np.mean([e['final_rupees'] - e['start_rupees'] for e in results]):.1f} | {out}")


def play_combat(args, path, model, cfg, ck, slots):
    """Combat checkpoints (Deku Babas): HUD shows Link HP, targets, lock-on, events; carries GRU memory if recurrent."""
    from zelda_rl.env import action_name as aname
    out = Path("runs/play") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if args.episode_seconds:
        cfg["episode_seconds"] = args.episode_seconds
    env, _ = make_env(cfg, out / "instance", slots=slots, slots_dir=args.slots_dir,
                      video_backend="Null" if args.headless else "Metal", unlimited_speed=args.fast)
    rec = bool(getattr(model, "recurrent", False))
    if not args.headless and sys.platform == "darwin":        # a hidden Metal window stalls Dolphin: bring it to the front
        import subprocess
        subprocess.run(["osascript", "-e", f'tell application "System Events" to set frontmost of (first process whose unix id is {env.inst.proc.pid}) to true'],
                       capture_output=True)
    print(f"▶ {path} (update {ck['update']}, combat, {'GRU memory' if rec else 'stateless'}) | slots {slots} | "
          f"{'argmax' if args.deterministic else 'sampled'} | {'uncapped' if args.fast else 'real time'}")
    results = []
    try:
        for slot in slots:
            for n in range(args.episodes):
                obs = env.reset(slot)
                h = model.init_hidden(1) if rec else None
                first = torch.ones(1)
                tr = env.tracker
                print(f"\n── slot {slot} ep {n + 1}: Link {env.cstate.hp / 16:.1f} hearts | targets {tr.targets_start}")
                t_hud = 0.0
                for t in range(env.cfg.max_steps):
                    with torch.no_grad():
                        x = torch.as_tensor(obs[None])
                        if rec:
                            a, _, _, v, h = model.act(x, h, first, deterministic=args.deterministic)
                        else:
                            a, _, _, v = model.act(x, deterministic=args.deterministic)
                    first = torch.zeros(1)
                    env.send_action(int(a[0]))
                    while not env.ready():
                        time.sleep(0.0005)
                    obs, r, term, trunc, info = env.observe()
                    ev, ep, c = info["events"], info["episode"], env.cstate
                    now = time.perf_counter()
                    if ev or term or now - t_hud >= 1.0 / args.hud_hz:
                        t_hud = now
                        live = tr.live_targets(c)
                        near = min((round(float(np.linalg.norm(np.subtract(e.pos, c.pos)))) for e in live), default=None)
                        print(f"  t={t * env.cfg.vi_per_step / 60:5.1f}s {aname(int(a[0]), 'combat'):12s} V={float(v[0]):6.2f} "
                              f"hp {c.hp / 16:.1f} | babas {len(live)}/{tr.targets_start} nearest {near} | "
                              f"locked {c.targeted in {e.ptr for e in live}} shield {bool(c.sf1 >> 22 & 1)} | R={ep['return']:6.2f}"
                              + (f" | EVENTS {ev}" if ev else ""))
                    if term or trunc:
                        break
                results.append(dict(slot=slot, **{k: ep[k] for k in ("end", "defeated", "targets_start", "hp_lost", "return",
                                                                     "kill_times_s", "lockons", "blocks")}))
                print(f"■ slot {slot}: {ep['end'].upper()} | killed {ep['defeated']}/{ep['targets_start']} at {ep['kill_times_s']} s | "
                      f"hearts lost {ep['hp_lost'] / 16:.1f} | return {ep['return']:.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))


if __name__ == "__main__":
    main()
