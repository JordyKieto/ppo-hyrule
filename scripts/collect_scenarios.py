"""Collect the outcomes and contacts the trained demonstrators never produce, by scripting them.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/collect_scenarios.py --scenarios walkout,suicide,wall,flail

A world model can only represent outcomes that occur in its data, and the released dataset is expert combat:
168 episodes with **0 room exits and 0 deaths**. Every demonstrator was trained with a -20 penalty on both,
so it learned never to produce either. The consequence was measured: a policy dreamed inside that model
walked out of the room for free, because leaving was an outcome the model could not represent.

So each terminal and each contact event gets its own scripted scenario. They overlap on purpose -- walking
into a Baba without shielding yields Link damage, enemy contact and eventually death from one episode:

  walkout   hold a direction AWAY from the nearest Baba          -> left_room terminal, scene 0 -> 85
  suicide   walk INTO the nearest Baba, never shield, never hit  -> Link damage, contact, death terminal
  wall      hold each compass direction into the geometry        -> bgCheckFlags WALL, wallYaw, speed -> 0
  flail     uniform random actions                               -> jumps, rolls, whiffed swings, Z with no target

Output is the same .npz + manifest format as scripts/collect_transitions.py, so all of it concatenates with
the standard collection. States are v2 (zelda_rl/state_layout.py), so the collision and action fields are
present -- which is what makes the wall and damage scenarios worth recording at all.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, EnemyRows, check_slots, encode_state  # noqa: E402
from zelda_rl.combat import BG_WALL  # noqa: E402
from zelda_rl.env import EnvError, action_name  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"
N_DIRS, N_BTN = 9, 7          # 63 combat actions = 9 stick directions x 7 button combos
BTN_NONE, BTN_A, BTN_B, BTN_Z, BTN_SHIELD = 0, 1, 2, 3, 4

# Per-scenario episode cap. A walkout takes a few seconds; the 180 s default would waste minutes per episode.
SECONDS = {"walkout": 25.0, "suicide": 90.0, "wall": 40.0, "flail": 180.0}


def _nearest(c):
    live = [e for e in c.enemies.values() if e.hp > 0]
    return min(live, key=lambda e: math.dist(e.pos, c.pos)) if live else None


def _dir_toward(c, tx, tz) -> int:
    """Stick direction index (1..8) pointing from Link at a world point. 0 is neutral; 1..8 are 45-degree
    steps from +x, matching zelda_rl.env DIRECTIONS (world angles, atan2(dz, dx))."""
    return (int(round(math.atan2(tz - c.pos[2], tx - c.pos[0]) / (math.pi / 4))) % 8) + 1


def act_walkout(env, c, rng, t, st):
    """Away from the nearest Baba. The exit is BEHIND Link: every start state faces him into the room."""
    if t < st["warm"]:
        return int(rng.integers(0, N_DIRS * N_BTN))
    if st.get("dir") is None:
        n = _nearest(c)
        away = (math.atan2(n.pos[2] - c.pos[2], n.pos[0] - c.pos[0]) + math.pi) if n \
            else rng.uniform(0, 2 * math.pi)
        k = (int(round(away / (math.pi / 4))) + int(rng.integers(-1, 2))) % 8
        st["dir"] = k + 1
    return st["dir"] * N_BTN + BTN_NONE


def act_suicide(env, c, rng, t, st):
    """Straight into the nearest Baba, no shield and no sword, so its lunge always connects."""
    n = _nearest(c)
    if n is None:
        return int(rng.integers(0, N_DIRS * N_BTN))
    return _dir_toward(c, n.pos[0], n.pos[2]) * N_BTN + BTN_NONE


def act_wall(env, c, rng, t, st):
    """Hold one direction long enough to pin Link against the geometry, then rotate to the next.

    Each leg is held for `leg` steps. Legs are walked in order rather than sampled so a single episode
    covers the whole compass and is guaranteed to meet a wall whatever the start position.
    """
    leg = st["leg"]
    return (((t // leg) % 8) + 1) * N_BTN + BTN_NONE


def act_flail(env, c, rng, t, st):
    """Uniform over all 63 actions: the only scenario that produces jumps, rolls and whiffed swings."""
    return int(rng.integers(0, N_DIRS * N_BTN))


SCENARIOS = {"walkout": act_walkout, "suicide": act_suicide, "wall": act_wall, "flail": act_flail}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="walkout,suicide,wall,flail")
    ap.add_argument("--episodes", default="40", help="episodes per scenario: one int for all, or a "
                    "comma list matching --scenarios (e.g. 60,60,40,30)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--slots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--warmup-max", type=int, default=25, help="random steps before a scripted walkout")
    ap.add_argument("--leg", type=int, default=45, help="wall scenario: steps held per compass direction")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)

    names = [s for s in args.scenarios.split(",") if s]
    if bad := [s for s in names if s not in SCENARIOS]:
        raise SystemExit(f"unknown scenarios {bad}; have {list(SCENARIOS)}")
    eps = [int(x) for x in str(args.episodes).split(",")]
    if len(eps) == 1:
        eps *= len(names)
    if len(eps) != len(names):
        raise SystemExit(f"--episodes has {len(eps)} values but --scenarios has {len(names)}")
    per = dict(zip(names, eps))
    slots = [int(s) for s in args.slots.split(",")]
    babas = check_slots(args.slots_dir, slots)
    if bad := [k for k, n in babas.items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba in {args.slots_dir}")

    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    out = Path(args.out or f"datasets/transitions/deku-baba-scenarios/{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    N = min(args.workers, sum(eps))
    print(f"▶ {sum(eps)} episodes over {len(names)} scenarios | {N} headless envs | slots {slots} -> {out}")
    for s in names:
        print(f"    {s:<9} {per[s]:>3} eps  cap {SECONDS[s]:>5.0f}s")

    # One env per worker, but the episode cap differs per scenario, so the cap is applied per episode by
    # bounding the step loop rather than by rebuilding the env.
    cfg["episode_seconds"] = max(SECONDS[s] for s in names)

    def build(i):
        env, _ = make_env(cfg, out / "instances" / f"env{i}", slots=slots, slots_dir=args.slots_dir,
                          video_backend="Null", unlimited_speed=True, idx=i, audio=False)
        return env

    t0 = time.perf_counter()
    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(build, range(N)))
    print(f"  {N} envs up in {time.perf_counter() - t0:.0f}s", flush=True)

    # Round-robin the scenarios so a crash partway through still leaves every scenario represented, and so
    # the 10 workers are never all running the same scenario (which would bunch all the long flails together).
    byname = {s: [(s, slots[i % len(slots)]) for i in range(per[s])] for s in names}
    jobs = [j for grp in zip_longest(*byname.values()) for j in grp if j is not None]
    it, lock, tally, idx = iter(jobs), threading.Lock(), Counter(), [0]
    manifest = (out / "manifest.jsonl").open("w")
    n_trans = [0]

    def worker(w: int):
        env, rng = envs[w], np.random.default_rng(args.seed + w)
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            name, slot = job
            policy = SCENARIOS[name]
            try:
                rows = EnemyRows()
                env.reset(slot)
                # env was built with the LONGEST scenario's cap; scale it down for the shorter ones
                cap = max(1, round(env.cfg.max_steps * SECONDS[name] / max(SECONDS[n] for n in names)))
                st = {"warm": int(rng.integers(3, args.warmup_max)), "dir": None, "leg": args.leg}
                S, A, R = [encode_state(env.cstate, rows)], [], []
                ep, n_wall, n_hit = None, 0, 0
                hp0 = env.cstate.hp
                for t in range(cap):
                    a = policy(env, env.cstate, rng, t, st)
                    A.append(a)
                    env.send_action(a)
                    while not env.ready():
                        time.sleep(0.0005)
                    _, r, term, trunc, info = env.observe()
                    R.append(r)
                    c = env.cstate
                    S.append(encode_state(c, rows))
                    n_wall += c.bg(BG_WALL)
                    n_hit += "link_damage" in info["events"]      # tracker events are a dict, see pay()
                    if term or trunc:
                        ep = dict(info["episode"])
                        break
                ep = ep or dict(env.tracker.ep)
                with lock:
                    idx[0] += 1
                    i = idx[0]
                    n_trans[0] += len(A)
                    tally[f"{name}/{ep['end']}"] += 1
                    fn = f"{name}{i:05d}_s{slot:02d}_{ep['end']}.npz"
                    np.savez_compressed(out / "episodes" / fn, action=np.asarray(A, np.int64),
                                        reward=np.asarray(R, np.float32), state=np.asarray(S, np.float32))
                    manifest.write(json.dumps({
                        "file": f"episodes/{fn}", "index": i, "checkpoint": f"scripted-{name}",
                        "scenario": name, "global_update": -1, "update": -1, "slot": slot, "steps": len(A),
                        "end": ep["end"], "success": ep["end"] == "success", "died": ep["died"],
                        "defeated": ep["defeated"], "targets_start": ep["targets_start"],
                        "hp_start": ep["hp_start"], "hp_end": ep["hp_end"], "hp_lost": ep["hp_lost"],
                        "damage_dealt": ep["damage_dealt"], "kill_times_s": ep["kill_times_s"],
                        "lockons": ep["lockons"], "blocks": ep["blocks"], "dodges": ep["dodges"],
                        "return": round(ep["return"], 3), "enemy_rows_used": len(rows.rows),
                        "actors_dropped": len(rows.dropped), "wall_contact_frames": int(n_wall),
                        "link_damage_events": int(n_hit),
                        "left_reason": ep.get("left_reason", "")}) + "\n")
                    manifest.flush()
                    print(f"  [{i}/{len(jobs)}] {name:<8} s{slot:<2} {ep['end']:<9} {len(A):>4} steps  "
                          f"hp {hp0}->{ep['hp_end']}  hits {n_hit:>2}  wall {n_wall:>4}f  "
                          f"{n_trans[0]:>7,} trans  {ep.get('left_reason', '')}", flush=True)
            except EnvError as err:
                print(f"  ! env{w}: {err} -> rebuilding", flush=True)
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
                env = envs[w] = build(w)

    try:
        with ThreadPoolExecutor(N) as pool:
            list(pool.map(worker, range(N)))
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        manifest.close()
        print(f"\noutcomes: {dict(tally)}")
        print(f"{n_trans[0]:,} transitions -> {out}")


if __name__ == "__main__":
    main()
