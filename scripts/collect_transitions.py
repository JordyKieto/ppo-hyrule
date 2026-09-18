"""Collect a (state, action, next_state) dataset from trained Deku Baba policies for a transition model.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/collect_transitions.py --episodes 100 --workers 4

Runs headless Dolphin instances at uncapped speed, one batch of episodes per checkpoint (early -> late), and writes
one .npz per episode plus a manifest. Each episode stores T actions and T+1 game states, so transition t is
(state[t], action[t], state[t+1]).

Enemy rows are per-episode STABLE: an actor keeps its row for the whole episode (first-seen order), so row j at t and
t+1 is the same Deku Baba. `targeted`/`arrow` are stored as row indices (-1 = none), not raw pointers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import BABA_STATES, DEKU_BABA, CombatState, read_combat_state  # noqa: E402
from zelda_rl.env import EnvError, action_name  # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR, load_policy, make_env, resolve_checkpoint  # noqa: E402

# Default mix: early (flailing, dies a lot) -> final (77% clear). Global update per docs/COMBAT_TASK.md
# (gru-v2 G1-30, gru-v2b G31-154 so gru-v2b u -> G(30+u)).
DEFAULT_MIX = [
    ("checkpoints/inbox/deku-combat_r5_u00005.pt", 5, 10),
    ("checkpoints/inbox/deku-combat-gru-v2_u00030.pt", 30, 15),
    ("checkpoints/inbox/deku-combat_r2_u00030.pt", 30, 10),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00060.pt", 90, 15),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00080.pt", 110, 15),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00100.pt", 130, 15),
    ("checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt", 150, 20),
]
MAX_ENEMIES = 16          # slots start with 9 cat-5 actors (3 Baba + 3 0x0095 + 3 0x0037); headroom for mid-fight spawns
DEKU_STATES_DIR = "datasets/zelda/combat/deku-baba/states"   # NOT the live Dolphin slots: those hold Gohma states now
LINK_FIELDS = ["scene", "room", "hp", "hp_cap", "sf1", "pos_x", "pos_y", "pos_z", "yaw",
               "targeted_row", "arrow_row", "held_item", "seeds", "nuts", "grass"]
ENEMY_FIELDS = ["present", "id", "hp", "pos_x", "pos_y", "pos_z", "fn_rel", "state_idx", "shape_yaw",
                "yaw_to_player", "timer", "cat", "params", "yaw", "vel_x", "vel_y", "vel_z",
                "focus_x", "focus_y", "focus_z"]
LINK_DIM, ENEMY_DIM = len(LINK_FIELDS), len(ENEMY_FIELDS)
STATE_DIM = LINK_DIM + MAX_ENEMIES * ENEMY_DIM


def check_slots(slots_dir: str, slots: list[int]) -> dict[int, int]:
    """Deku Babas per slot, read straight from the savestate files. Guards against pointing at the wrong states
    (the live Dolphin StateSaves were overwritten with Gohma savestates on 2026-09-14)."""
    from zelda_rl.savestate import decompress_state, find_n64_ram
    counts = {}
    for k in slots:
        f = Path(slots_dir) / f"NACE01.s{k:02d}"
        if not f.exists():
            raise SystemExit(f"missing savestate {f}\n  fetch: aws s3 sync s3://zelda-rl/zelda/combat/deku-baba/states/ "
                             f"{DEKU_STATES_DIR}/ --profile race-wm")
        raw = decompress_state(f.read_bytes())
        base = find_n64_ram(raw)
        c = read_combat_state(memoryview(raw)[base:base + 0x400000])
        counts[k] = sum(1 for e in c.enemies.values() if e.id == DEKU_BABA and e.hp > 0)
    return counts


class EnemyRows:
    """Per-episode actor pointer -> stable row index (first-seen order)."""

    def __init__(self):
        self.rows: dict[int, int] = {}
        self.dropped: set[int] = set()

    def row(self, ptr: int) -> int:
        if ptr not in self.rows:
            if len(self.rows) >= MAX_ENEMIES:
                self.dropped.add(ptr)
                return -1
            self.rows[ptr] = len(self.rows)
        return self.rows[ptr]


def encode_state(c: CombatState, rows: EnemyRows) -> np.ndarray:
    """Flatten a CombatState into STATE_DIM float32s with stable enemy rows."""
    out = np.zeros(STATE_DIM, np.float32)
    for ptr in c.enemies:
        rows.row(ptr)
    out[:LINK_DIM] = [c.scene, c.room, c.hp, c.hp_cap, c.sf1, *c.pos, c.yaw,
                      rows.row(c.targeted) if c.targeted else -1, rows.row(c.arrow) if c.arrow else -1,
                      c.held_item, c.seeds, c.nuts, c.grass]
    for ptr, e in c.enemies.items():
        j = rows.row(ptr)
        if j < 0:
            continue
        base = LINK_DIM + j * ENEMY_DIM
        st = BABA_STATES.index(e.state) if e.state in BABA_STATES else BABA_STATES.index("other")
        out[base:base + ENEMY_DIM] = [1.0, e.id, e.hp, *e.pos, e.fn_rel, st, e.shape_yaw, e.yaw_to_player,
                                      e.timer, e.cat, e.params, e.yaw, *e.vel, *e.focus]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transitions", type=int, default=100_000,
                    help="stop once this many transitions are collected (0 = go by --episodes instead)")
    ap.add_argument("--episodes", type=int, default=400, help="episode budget; the cap when --transitions is 0")
    ap.add_argument("--workers", type=int, default=10, help="parallel Dolphin instances")
    ap.add_argument("--out", default=None, help="output dir (default datasets/transitions/deku-baba/<timestamp>)")
    ap.add_argument("--slots", default="1,2,3,4,5,6,7,8,9,10", help="start slots to cycle")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR, help=f"default {DEKU_STATES_DIR}")
    ap.add_argument("--torch-threads", type=int, default=1, help="keep at 1: Dolphin needs the cores")
    ap.add_argument("--episode-seconds", type=float, default=None, help="override episode length (default 180)")
    ap.add_argument("--deterministic", action="store_true", help="argmax actions (default: sampled, more diverse)")
    ap.add_argument("--windowed", action="store_true", help="show Dolphin windows (slow; debugging only)")
    ap.add_argument("--realtime", action="store_true", help="cap emulation at 1x (debugging only)")
    ap.add_argument("--no-obs", action="store_true", help="skip the 483-d policy obs (game state + action only)")
    ap.add_argument("--mix", default=None, help="JSON [[path, global_update, episodes], ...] instead of the default mix")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    mix = json.loads(args.mix) if args.mix else DEFAULT_MIX
    scale = args.episodes / sum(n for _, _, n in mix)
    quota = [(p, g, max(1, round(n * scale))) for p, g, n in mix]
    slots = [int(s) for s in args.slots.split(",")]
    babas = check_slots(args.slots_dir, slots)
    if bad := [k for k, n in babas.items() if n == 0]:
        raise SystemExit(f"slots {bad} in {args.slots_dir} have no live Deku Baba -> wrong savestates.\n"
                         f"  per-slot babas: {babas}\n"
                         f"  fetch the right ones: aws s3 sync s3://zelda-rl/zelda/combat/deku-baba/states/ "
                         f"{DEKU_STATES_DIR}/ --profile race-wm")
    out = Path(args.out or f"datasets/transitions/deku-baba/{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    ck_paths = [resolve_checkpoint(p) for p, _, _ in quota]
    models = [load_policy(p) for p in ck_paths]
    cfg = models[-1][1]
    if args.episode_seconds:
        for _, c, _ in models:
            c["episode_seconds"] = args.episode_seconds
    total = sum(n for _, _, n in quota)
    goal = f"{args.transitions:,} transitions (<={total} fights)" if args.transitions else f"{total} fights"
    print(f"▶ collecting {goal} from {len(quota)} checkpoints | {args.workers} headless envs | "
          f"state {STATE_DIM}d (link {LINK_DIM} + {MAX_ENEMIES}x{ENEMY_DIM}) | obs {cfg['obs_dim']}d | -> {out}\n"
          f"  slots {args.slots_dir}: babas per slot {babas}")
    for (p, g, n), (_, c, ck) in zip(quota, models):
        print(f"    G{g:<4} u{ck['update']:<4} {n:>3} eps  {Path(p).name}")

    N = args.workers
    backend = "Metal" if args.windowed else "Null"

    def build(i: int):
        env, _ = make_env(cfg, out / "instances" / f"env{i}", slots=slots, slots_dir=args.slots_dir,
                          video_backend=backend, unlimited_speed=not args.realtime, idx=i, audio=False)
        return env

    t_boot = time.perf_counter()
    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(build, range(N)))
    print(f"  {N} envs up in {time.perf_counter() - t_boot:.0f}s")

    # Job queue: every episode is independent, so workers never wait for each other (a lockstep batch would make
    # every tick wait for the slowest env, which is what caps throughput as workers are added). Shuffled so that
    # stopping on a transition target still leaves the checkpoint mix and slot coverage proportional.
    jobs = [(qi, slots[j % len(slots)]) for qi, (_, _, n) in enumerate(quota) for j in range(n)]
    random.Random(0).shuffle(jobs)
    job_iter = iter(jobs)
    manifest = (out / "manifest.jsonl").open("w")
    tally: Counter = Counter()
    lock = threading.Lock()
    done_per_ck: Counter = Counter()
    trans_per_ck: Counter = Counter()
    ep_index = n_trans = in_flight = 0
    t_start = time.perf_counter()

    def run_episode(env, qi: int, slot: int) -> tuple[dict, dict]:
        """One episode, solo on this env. Returns (arrays, episode metrics)."""
        nonlocal ep_index, n_trans
        model, _, ck = models[qi]
        rec = bool(getattr(model, "recurrent", False))
        rows = EnemyRows()
        obs = env.reset(slot)
        acts, rews, states, obss = [], [], [encode_state(env.cstate, rows)], []
        h = model.init_hidden(1) if rec else None
        first = torch.ones(1)
        ep = None
        for t in range(env.cfg.max_steps):
            with torch.no_grad():
                x = torch.as_tensor(obs[None])
                if rec:
                    a, _, _, _, h = model.act(x, h, first, deterministic=args.deterministic)
                else:
                    a, _, _, _ = model.act(x, deterministic=args.deterministic)
            first = torch.zeros(1)
            a = int(a[0])
            if not args.no_obs:
                obss.append(obs.copy())
            acts.append(a)
            env.send_action(a)
            while not env.ready():
                time.sleep(0.0005)
            obs, r, term, trunc, info = env.observe()
            rews.append(r)
            states.append(encode_state(env.cstate, rows))
            if term or trunc:
                ep = dict(info["episode"])
                break
        ep = ep or dict(env.tracker.ep)
        arrs = {"action": np.asarray(acts, np.int64), "reward": np.asarray(rews, np.float32),
                "state": np.asarray(states, np.float32)}
        if not args.no_obs:
            arrs["obs"] = np.asarray(obss, np.float32)
        with lock:
            ep_index += 1
            idx = ep_index
            n_trans += len(acts)
            tally[ep["end"]] += 1
            done_per_ck[qi] += 1
            trans_per_ck[qi] += len(acts)
            name = f"ep{idx:05d}_G{quota[qi][1]:03d}_s{slot:02d}_{ep['end']}.npz"
            np.savez_compressed(out / "episodes" / name, **arrs)
            manifest.write(json.dumps({
                "file": f"episodes/{name}", "index": idx, "checkpoint": str(quota[qi][0]),
                "global_update": quota[qi][1], "update": ck["update"], "slot": slot, "steps": len(acts),
                "end": ep["end"], "success": ep["end"] == "success", "died": ep["died"],
                "defeated": ep["defeated"], "targets_start": ep["targets_start"],
                "hp_start": ep["hp_start"], "hp_end": ep["hp_end"], "hp_lost": ep["hp_lost"],
                "damage_dealt": ep["damage_dealt"], "kill_times_s": ep["kill_times_s"],
                "lockons": ep["lockons"], "blocks": ep["blocks"], "dodges": ep["dodges"],
                "return": round(ep["return"], 3), "enemy_rows_used": len(rows.rows),
                "actors_dropped": len(rows.dropped),
            }) + "\n")
            manifest.flush()
            el = time.perf_counter() - t_start
            drop = f" !{len(rows.dropped)} DROPPED" if rows.dropped else ""
            rate = n_trans / max(el, 1e-9)
            eta = f" eta {(args.transitions - n_trans) / rate / 60:4.1f}m" if args.transitions and rate else ""
            pct = f"{n_trans / args.transitions:4.0%}" if args.transitions else f"ep{idx}"
            print(f"  [{pct}] ep{idx:<3} G{quota[qi][1]:<4} s{slot:<2} {ep['end']:<9} "
                  f"{ep['defeated']}/{ep['targets_start']} kills {len(acts):>4} steps | {n_trans:>7,} trans | "
                  f"{rate:5.0f} steps/s{eta} | succ {tally['success']} death {tally['death']} "
                  f"left {tally['left_room']} to {tally['timeout']}{drop}", flush=True)
        return arrs, ep

    def worker(i: int):
        nonlocal in_flight
        env = envs[i]
        while True:
            with lock:
                # Reserve what the episodes still running are likely to add, so the target isn't badly overshot.
                mean_len = n_trans / ep_index if ep_index else 0
                projected = n_trans + in_flight * mean_len
                job = None if args.transitions and projected >= args.transitions else next(job_iter, None)
                if job is not None:
                    in_flight += 1
            if job is None:
                return
            qi, slot = job
            try:
                for attempt in range(3):
                    try:
                        run_episode(env, qi, slot)
                        break
                    except EnvError as err:
                        print(f"  ! env{i}: {err} -> rebuilding", flush=True)
                        try:
                            env.close()
                        except Exception:  # noqa: BLE001
                            pass
                        env = envs[i] = build(i)
                else:
                    print(f"  ! env{i} gave up on G{quota[qi][1]} slot {slot}", flush=True)
            finally:
                with lock:
                    in_flight -= 1

    for m, _, _ in models:
        m.eval()
    try:
        with ThreadPoolExecutor(N) as pool:
            list(pool.map(worker, range(N)))
    except KeyboardInterrupt:
        print("\n interrupted")
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        manifest.close()
        el = time.perf_counter() - t_start
        (out / "dataset.json").write_text(json.dumps({
            "task": "deku_baba_combat", "episodes": ep_index, "transitions": n_trans,
            "outcomes": dict(tally), "wall_seconds": round(el, 1),
            "state_dim": STATE_DIM, "link_dim": LINK_DIM, "max_enemies": MAX_ENEMIES, "enemy_dim": ENEMY_DIM,
            "link_fields": LINK_FIELDS, "enemy_fields": ENEMY_FIELDS, "baba_states": BABA_STATES,
            "n_actions": cfg["n_actions"], "action_names": [action_name(a, "combat") for a in range(cfg["n_actions"])],
            "obs_dim": None if args.no_obs else cfg["obs_dim"],
            "obs_fields": None if args.no_obs else cfg["obs_fields"],
            "obs_version": cfg["obs_version"], "vi_per_step": cfg["vi_per_step"], "steps_per_second": 10.0,
            "episode_seconds": cfg["episode_seconds"], "slots": slots, "deterministic": args.deterministic,
            "checkpoints": [{"path": str(p), "global_update": g, "episodes": done_per_ck[qi],
                             "transitions": trans_per_ck[qi]} for qi, (p, g, _) in enumerate(quota)],
            "layout": "each episode .npz has action[T], reward[T], state[T+1, state_dim] (+ obs[T, obs_dim]); "
                      "transition t = (state[t], action[t], state[t+1])",
        }, indent=2))
        print(f"\n■ {ep_index} episodes, {n_trans:,} transitions in {el / 60:.1f} min "
              f"({n_trans / max(el, 1e-9):.0f} steps/s)\n  outcomes: {dict(tally)}\n  -> {out}")
        if ep_index:
            print(f"  success {tally['success'] / ep_index:.0%} | death {tally['death'] / ep_index:.0%} | "
                  f"left {tally['left_room'] / ep_index:.0%} | timeout {tally['timeout'] / ep_index:.0%}")


if __name__ == "__main__":
    main()
