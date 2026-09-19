"""Collect episodes that END BY LEAVING THE ROOM, which the main dataset contains none of.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/collect_exits.py --episodes 12

`left_room` is a terminal worth -20 in the real reward -- the same as dying -- and exists so a losing fight
can't be escaped. Every demonstrator in the released dataset was trained with that penalty and learned never
to leave, so the dataset has 0 such episodes out of 168. A world model built on it therefore cannot
represent the outcome at all, and a policy dreamed inside it walks out of the room for free. That was
observed directly: the dream policy left in 23 steps.

Rather than guessing what the run-up to an exit looks like, this drives the real emulator out of the room:
a few steps of random play so the states are not all identical, then a held direction until the episode
terminates. Output is the same .npz + manifest format as scripts/collect_transitions.py, so the two
datasets concatenate.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import math

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import (DEKU_STATES_DIR, MAX_ENEMIES, EnemyRows, check_slots,  # noqa: E402
                                         encode_state)
from zelda_rl.env import EnvError, action_name  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"
N_DIRS, N_BTN = 9, 7            # 63 combat actions = 9 stick directions x 7 button combos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", default="1,2,3,6,7,10", help="slots whose exit is reachable")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--warmup-max", type=int, default=25,
                    help="random steps before walking, so the run-up states vary")
    ap.add_argument("--jitter", type=int, default=1,
                    help="allow the walk direction to be +/- this many 45-degree steps off straight back")
    ap.add_argument("--episode-seconds", type=float, default=25.0,
                    help="short cap: a walkout takes a few seconds, so the 180 s default wastes minutes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)

    slots = [int(s) for s in a.slots.split(",")]
    babas = check_slots(a.slots_dir, slots)
    if bad := [k for k, n in babas.items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba in {a.slots_dir}")
    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    cfg["episode_seconds"] = a.episode_seconds
    out = Path(a.out or f"datasets/transitions/deku-baba-exits/{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    N = min(a.workers, a.episodes)
    print(f"▶ {a.episodes} exit episodes | {N} headless envs | slots {slots} -> {out}")

    def build(i):
        env, _ = make_env(cfg, out / "instances" / f"env{i}", slots=slots, slots_dir=a.slots_dir,
                          video_backend="Null", unlimited_speed=True, idx=i, audio=False)
        return env

    t0 = time.perf_counter()
    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(build, range(N)))
    print(f"  {N} envs up in {time.perf_counter() - t0:.0f}s")

    jobs = [slots[i % len(slots)] for i in range(a.episodes)]
    it, lock, tally, idx = iter(list(enumerate(jobs))), threading.Lock(), Counter(), [0]
    manifest = (out / "manifest.jsonl").open("w")

    def worker(w: int):
        env, rng = envs[w], np.random.default_rng(a.seed + w)
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            n, slot = job
            try:
                rows = EnemyRows()
                env.reset(slot)
                warm = int(rng.integers(3, a.warmup_max))
                # The exit is BEHIND Link: every start state faces him into the room with the Babas ahead.
                # So walk directly away from the nearest one. Stick directions are world angles
                # (atan2(dz, dx)), index 0 is neutral and 1..8 are 45-degree steps from +x.
                c = env.cstate
                live = [e for e in c.enemies.values() if e.hp > 0]
                if live:
                    n_ = min(live, key=lambda e: math.dist(e.pos, c.pos))
                    away = math.atan2(n_.pos[2] - c.pos[2], n_.pos[0] - c.pos[0]) + math.pi
                else:
                    away = rng.uniform(0, 2 * math.pi)
                k = int(round(away / (math.pi / 4))) % 8
                k = (k + int(rng.integers(-a.jitter, a.jitter + 1))) % 8
                walk = (k + 1) * N_BTN
                S, A, R = [encode_state(env.cstate, rows)], [], []
                ep = None
                for t in range(env.cfg.max_steps):
                    act = int(rng.integers(0, N_DIRS * N_BTN)) if t < warm else walk
                    A.append(act)
                    env.send_action(act)
                    while not env.ready():
                        time.sleep(0.0005)
                    _, r, term, trunc, info = env.observe()
                    R.append(r)
                    S.append(encode_state(env.cstate, rows))
                    if term or trunc:
                        ep = dict(info["episode"])
                        break
                ep = ep or dict(env.tracker.ep)
                with lock:
                    idx[0] += 1
                    i = idx[0]
                    tally[ep["end"]] += 1
                    name = f"exit{i:05d}_s{slot:02d}_{ep['end']}.npz"
                    np.savez_compressed(out / "episodes" / name, action=np.asarray(A, np.int64),
                                        reward=np.asarray(R, np.float32), state=np.asarray(S, np.float32))
                    manifest.write(json.dumps({
                        "file": f"episodes/{name}", "index": i, "checkpoint": "scripted-walkout",
                        "global_update": -1, "update": -1, "slot": slot, "steps": len(A),
                        "end": ep["end"], "success": ep["end"] == "success", "died": ep["died"],
                        "defeated": ep["defeated"], "targets_start": ep["targets_start"],
                        "hp_start": ep["hp_start"], "hp_end": ep["hp_end"], "hp_lost": ep["hp_lost"],
                        "damage_dealt": ep["damage_dealt"], "kill_times_s": ep["kill_times_s"],
                        "lockons": ep["lockons"], "blocks": ep["blocks"], "dodges": ep["dodges"],
                        "return": round(ep["return"], 3), "enemy_rows_used": len(rows.rows),
                        "actors_dropped": len(rows.dropped), "warmup": warm,
                        "walk_dir": action_name(walk, "combat"),
                        "left_reason": ep.get("left_reason", "")}) + "\n")
                    manifest.flush()
                    print(f"  [{i}/{a.episodes}] s{slot} {ep['end']:<9} {len(A):>4} steps  "
                          f"warmup {warm:>2} walk {action_name(walk, 'combat'):<6} "
                          f"{ep.get('left_reason', '')}", flush=True)
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
        print(f"-> {out}")


if __name__ == "__main__":
    main()
