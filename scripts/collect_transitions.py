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
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import BABA_STATES, DEKU_BABA, GOHMA_ID, CombatState, read_combat_state  # noqa: E402
from zelda_rl.gohma import GOHMA_STATES  # noqa: E402

STATE_VOCAB = BABA_STATES          # replaced by --profile before any row is built
from zelda_rl.state_layout import (ENEMY_FIELDS as ENEMY_FIELDS_,  # noqa: E402
                                    LINK_FIELDS_V3, MAX_ENEMIES as MAX_ENEMIES_)
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
# Equal-parts-BAD-PLAY mix. The default mix is weighted toward competent play, which is the wrong shape for a
# world model: a policy that clears the room never takes damage and never dies, so the rare outcomes the reward
# actually terminates on are absent. Measured on 61 episodes of the default mix: 47 success, 11 timeout, only
# 2 deaths, 1 room-exit.
#
# Two corrections are baked in here:
#   * RANDOM-INIT policies, not just early checkpoints. `deku-combat_r5_u00005` has already had 5 PPO updates
#     and prefers some actions; an untrained net samples near-uniformly, so it wanders, walks out of the room
#     and blunders into Babas with no learned avoidance.
#   * Weighted toward the MIDDLE. The lowest-skill checkpoint does not die -- it times out at 0/3 kills after
#     1800 steps because it never engages. Deaths and Link damage come from G30-G90: good enough to approach a
#     Baba, bad enough to lose to it.
BAD_PLAY_MIX = [
    ("checkpoints/random/random-init-s0.pt", 0, 12),
    ("checkpoints/random/random-init-s1.pt", 0, 8),
    ("checkpoints/inbox/deku-combat_r5_u00005.pt", 5, 20),
    ("checkpoints/inbox/deku-combat-gru-v2_u00030.pt", 30, 13),
    ("checkpoints/inbox/deku-combat_r2_u00030.pt", 30, 12),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00060.pt", 90, 20),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00080.pt", 110, 6),
    ("checkpoints/inbox/deku-combat-gru-v2b_u00100.pt", 130, 5),
    ("checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt", 150, 4),
]
MIX_PRESETS = {"default": None, "badplay": BAD_PLAY_MIX}


class S3Streamer:
    """Upload each episode to S3 the moment it is written, in a background thread pool.

    WHY: pod-local storage must never be the only copy. A 20M-transition collection was lost because it lived
    only in 32 ephemeral pod volumes and terminating the fleet deleted all of them at once. Uploading as we go
    makes the dataset durable from the first episode instead of at the end, and survives any pod dying.
    Credentials arrive in the pod's .fleet_secrets (AWS_* env vars); with none present this is a silent no-op
    so local runs are unaffected.
    """

    def __init__(self, bucket_prefix: str | None, workers: int = 4, shard: str | None = None):
        self.ex = self.s3 = self.bucket = self.prefix = None
        self.n = self.failed = 0
        if not bucket_prefix:
            return
        try:
            import boto3
            from concurrent.futures import ThreadPoolExecutor
            b = bucket_prefix.replace("s3://", "").split("/", 1)
            self.bucket, self.prefix = b[0], (b[1].rstrip("/") if len(b) > 1 else "")
            # PER-SHARD NAMESPACE, or 109 pods silently overwrite each other. Every collector numbers episodes
            # from ep00001, so a flat prefix means rank 7's ep00001 clobbers rank 3's. Measured: 79,694 episodes
            # on disk across the fleet collapsed to ~20,000 objects in S3 -- a 75% loss that looked like a
            # complete dataset. The shard defaults to the hostname so it is always distinct.
            import socket
            self.prefix = f"{self.prefix}/{shard or socket.gethostname()}".strip("/")
            # pods get AWS_* from .fleet_secrets; locally fall back to the named profile
            import os as _os
            self.s3 = (boto3.client("s3") if _os.environ.get("AWS_ACCESS_KEY_ID")
                       else boto3.Session(profile_name="race-wm", region_name="us-west-1").client("s3"))
            self.ex = ThreadPoolExecutor(workers)
            print(f"  S3 streaming -> s3://{self.bucket}/{self.prefix}/ ({workers} uploaders)")
        except Exception as e:  # noqa: BLE001
            print(f"  S3 streaming disabled: {e}")
            self.ex = None

    def put(self, path: "Path", key_rel: str) -> None:
        if self.ex is None:
            return
        self.ex.submit(self._one, str(path), f"{self.prefix}/{key_rel}")

    def _one(self, path: str, key: str) -> None:
        err = None
        for attempt in range(3):
            try:
                self.s3.upload_file(path, self.bucket, key)
                self.n += 1
                return
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 * (attempt + 1))
        self.failed += 1
        # SAY WHY. A silently-swallowed upload error is how pod-local storage ends up being the only copy.
        if self.failed <= 3:
            print(f"  S3 upload FAILED {key}: {type(err).__name__}: {str(err)[:160]}", flush=True)

    def close(self) -> None:
        if self.ex is not None:
            self.ex.shutdown(wait=True)
            print(f"  S3: {self.n} objects uploaded, {self.failed} failed")

MAX_ENEMIES = MAX_ENEMIES_  # 16: slots start with 9 cat-5 actors (3 Baba + 3 0x0095 + 3 0x0037); headroom for spawns
DEKU_STATES_DIR = "datasets/zelda/combat/deku-baba/states"   # NOT the live Dolphin slots: those hold Gohma states now
# Layout lives in zelda_rl/state_layout.py so the collector and the world model cannot drift apart.
LINK_FIELDS, ENEMY_FIELDS = LINK_FIELDS_V3, ENEMY_FIELDS_
LINK_DIM, ENEMY_DIM = len(LINK_FIELDS), len(ENEMY_FIELDS)
STATE_DIM = LINK_DIM + MAX_ENEMIES * ENEMY_DIM


def check_slots(slots_dir: str, slots: list[int], target_ids=(DEKU_BABA,), label="Deku Baba") -> dict[int, int]:
    """Live TARGET actors per slot, read straight from the savestate files.

    `target_ids` comes from the profile, not from a constant: this guard exists to catch pointing at the
    wrong savestates, and hard-coding Deku Baba made it reject a perfectly good set of GOHMA states with
    "no live Deku Baba" -- which is what stopped the Dyna loop collecting (2026-10-16)."""
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
        counts[k] = sum(1 for e in c.enemies.values() if e.id in target_ids and e.hp > 0)
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
                      c.held_item, c.seeds, c.nuts, c.grass,
                      c.shape_yaw, *c.vel, c.speed, c.bg_flags, c.wall_yaw, c.floor_height,
                      c.action_fn_rel, c.sf2, c.sf3, c.melee_state, c.melee_anim, c.ztarget_timer,
                      c.parallel_yaw, c.fall_distance, c.floor_pitch, c.dist_interact_wall,
                      c.y_dist_to_ledge, c.door_type, c.door_direction, c.door_timer, c.knockback_type,
                      c.cam_yaw, c.cam_pitch, *c.cam_eye]
    for ptr, e in c.enemies.items():
        j = rows.row(ptr)
        if j < 0:
            continue
        base = LINK_DIM + j * ENEMY_DIM
        st = (STATE_VOCAB.index(e.state) if e.state in STATE_VOCAB
              else STATE_VOCAB.index("other") if "other" in STATE_VOCAB else 0)
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
    ap.add_argument("--profile", choices=("deku_baba", "gohma"), default="deku_baba",
                    help="which TARGET the savestates and the enemy state column belong to. The env itself "
                         "comes from the policy's own config, so this only picks the savestate guard and "
                         "the behaviour-state vocabulary")
    ap.add_argument("--torch-threads", type=int, default=1, help="keep at 1: Dolphin needs the cores")
    ap.add_argument("--episode-seconds", type=float, default=None, help="override episode length (default 180)")
    ap.add_argument("--deterministic", action="store_true", help="argmax actions (default: sampled, more diverse)")
    ap.add_argument("--windowed", action="store_true", help="show Dolphin windows (slow; debugging only)")
    ap.add_argument("--realtime", action="store_true", help="cap emulation at 1x (debugging only)")
    ap.add_argument("--no-obs", action="store_true", help="skip the 483-d policy obs (game state + action only)")
    ap.add_argument("--mix", default=None, help="JSON [[path, global_update, episodes], ...] instead of the default mix")
    ap.add_argument("--s3-prefix", default=None,
                    help="stream every episode to S3 as it is written, e.g. s3://zelda-rl/zelda/transitions/v5. "
                         "Pod-local storage as the ONLY copy is how a 20M collection was lost.")
    ap.add_argument("--s3-workers", type=int, default=4)
    ap.add_argument("--s3-shard", default=None,
                    help="S3 sub-prefix for THIS collector (default: hostname). Never share one across pods: "
                         "all collectors number episodes from ep00001 and would overwrite each other.")
    ap.add_argument("--mix-preset", default="default", choices=sorted(MIX_PRESETS),
                    help="named mix; 'badplay' is equal-parts low-skill incl. random-init, for deaths/damage. "
                         "A preset rather than a JSON path so a 39-pod fleet needs no shared file.")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    mix = (json.loads(args.mix) if args.mix else (MIX_PRESETS[args.mix_preset] or DEFAULT_MIX))
    s3 = S3Streamer(args.s3_prefix, args.s3_workers, shard=args.s3_shard)
    global STATE_VOCAB
    STATE_VOCAB = GOHMA_STATES if args.profile == "gohma" else BABA_STATES
    scale = args.episodes / sum(n for _, _, n in mix)
    quota = [(p, g, max(1, round(n * scale))) for p, g, n in mix]
    slots = [int(s) for s in args.slots.split(",")]
    TARGETS = {"deku_baba": ((DEKU_BABA,), "Deku Baba"), "gohma": ((GOHMA_ID,), "Gohma")}
    tgt_ids, tgt_label = TARGETS[args.profile]
    found = check_slots(args.slots_dir, slots, tgt_ids, tgt_label)
    if bad := [k for k, n in found.items() if n == 0]:
        raise SystemExit(f"slots {bad} in {args.slots_dir} have no live {tgt_label} -> wrong savestates "
                         f"for --profile {args.profile}.\n  per-slot {tgt_label}: {found}")
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
        print(f"    G{g:<4} u{ck.get('update', 0):<4} {n:>3} eps  {Path(p).name}")

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
    # APPEND, and resume the episode numbering from what is already on disk. Mode "w" here truncated the
    # manifest every time a pod was relaunched to fill its backlog: the earlier run's episodes stayed on disk
    # and in S3 but vanished from the index, and load_many() enumerates the index, so 28.8% of the v5
    # collection was silently invisible until scripts/rebuild_manifests.py recovered it. Restarting the
    # numbering at 0 additionally let a relaunch overwrite the first run's episodes whenever the generated
    # name collided.
    prior = sorted((out / "episodes").glob("*.npz"))
    resume_at = 0
    for q in prior:
        m = re.match(r"^ep(\d+)_", q.name)
        if m:
            resume_at = max(resume_at, int(m.group(1)))
    if resume_at:
        print(f"  {len(prior)} episodes already here; numbering resumes at ep{resume_at + 1:05d}")
    manifest = (out / "manifest.jsonl").open("a")
    tally: Counter = Counter()
    lock = threading.Lock()
    done_per_ck: Counter = Counter()
    trans_per_ck: Counter = Counter()
    ep_index, n_trans, in_flight = resume_at, 0, 0
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
            s3.put(out / "episodes" / name, f"episodes/{name}")     # durable immediately, not at the end
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
            if idx % 25 == 0:
                # The manifest used to be uploaded ONLY in the finally block, so a pod killed mid-run left
                # its episodes durable in S3 and its index stale. It is a few hundred KB; upload it as we go.
                s3.put(out / "manifest.jsonl", "manifest.jsonl")
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
        # dataset.json used to be written ONLY here, so a collector killed mid-run left 39 pods with episodes
        # and a manifest but no dataset.json -- and load_many() requires it, making the whole collection
        # unreadable. It is now also written at startup (see `write_meta` below) and refreshed here.
        for e in envs:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        manifest.flush()
        manifest.close()
        el = time.perf_counter() - t_start
        s3.put(out / "manifest.jsonl", "manifest.jsonl")
        _S3_FINALISE = True
        (out / "dataset.json").write_text(json.dumps({
            "task": "deku_baba_combat", "episodes": ep_index, "transitions": n_trans,
            "outcomes": dict(tally), "wall_seconds": round(el, 1),
            "state_dim": STATE_DIM, "link_dim": LINK_DIM, "max_enemies": MAX_ENEMIES, "enemy_dim": ENEMY_DIM,
            "link_fields": LINK_FIELDS, "enemy_fields": ENEMY_FIELDS, "baba_states": STATE_VOCAB,
            "profile": args.profile,
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
        s3.put(out / "dataset.json", "dataset.json")
        s3.close()                       # wait for in-flight uploads so nothing is dropped on exit


if __name__ == "__main__":
    main()
