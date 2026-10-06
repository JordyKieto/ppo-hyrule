"""Held-out evaluation of Gohma checkpoints, on slots the policy was NOT trained on.

Why this exists: every Gohma number this project has reported is IN-TRAINING, measured on the same slots the
policy trains on, with no holdout. `gohma-band-v13` peaked at updates 41-50 and drifted down for 23 more, and
nothing noticed -- there was no evaluator and the saved checkpoint was `latest.pt`, so the peak was lost when
the pods were terminated. This is the instrument that makes `best` mean something, and it is what the KL anchor
anchors TO.

Adopted from puffertank E39, where the anchor advances only on seeds never used to choose a checkpoint
(`../puffertank/docs/moba/experiments/E39-continuous-dyna.md`).

    # one checkpoint, held-out slots
    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/eval_fleet.py --ckpt runs/.../latest.pt --episodes 40

    # watch a run directory and evaluate every new checkpoint, writing a leaderboard
    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/eval_fleet.py --watch runs/gohma-band-v14-r0/checkpoints \
        --out results/eval/v14.jsonl --episodes 40

HONEST LIMIT on what "held out" means here. The task has 10 fixed savestates, not a seed distribution, so a
slot held out of THIS run may still have been trained on by an earlier run in the resume chain. The script
records which slots the checkpoint's own config trained on and warns when the holdout is only w.r.t. the
current run. Stochasticity across episodes comes from sampled actions and the boss's own RNG, not from the
start state.
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

# ONE THREAD. Dolphin needs ~2 threads per instance (emulation + the Null GPU thread) and this Mac has 4
# performance cores; torch defaults to taking all 4 for a forward pass, which starves the emulation thread past
# the 10 s Null-backend stall timeout and raises "emulation stalled (VI stuck)". 27 scripts in this repo
# already set this -- the three I added did not, and I spent three runs blaming the hardware. AGENTS.md 28.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass            # already initialised in this process; the intra-op setting is the one that matters

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zelda_rl.env import EnvError  # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR, load_policy, make_env  # noqa: E402

ALL_SLOTS = list(range(1, 11))


def held_out(cfg) -> list[int]:
    """Slots this checkpoint's own run did NOT train on."""
    trained = cfg.get("start_slots") or cfg.get("usable_slots") or []
    if isinstance(trained, str):
        trained = [int(x) for x in trained.split(",") if x.strip()]
    return [s for s in ALL_SLOTS if s not in set(trained)], list(trained)


@torch.no_grad()
def evaluate(ckpt: Path, slots, episodes, envs_n, slots_dir, seconds, deterministic=False):
    model, cfg, _ = load_policy(ckpt)
    rec = bool(cfg.get("recurrent"))
    if seconds:
        cfg = {**cfg, "episode_seconds": seconds}
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"evalf_{os.getpid()}"
    envs = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(envs_n) as pool:
        envs = list(pool.map(lambda i: make_env(cfg, root / f"i{i}", slots=slots, slots_dir=slots_dir,
                                                video_backend="Null", unlimited_speed=True, idx=i)[0],
                             range(envs_n)))
    out = []
    mk = lambda i: make_env(cfg, root / f"i{i}", slots=slots, slots_dir=slots_dir, video_backend="Null",
                            unlimited_speed=True, idx=i)[0]                            # noqa: E731
    stalls = 0
    try:
        while len(out) < episodes:
            slot = int(slots[len(out) % len(slots)])          # round-robin, so every slot gets equal weight
            try:
                obs = np.stack([e.reset(slot) for e in envs])
                h = model.init_hidden(len(envs)) if rec else None
                first = torch.ones(len(envs))
                alive = [True] * len(envs)
                while any(alive):
                    x = torch.as_tensor(obs, dtype=torch.float32)
                    if rec:
                        a, _, _, _, h = model.act(x, h, first, deterministic=deterministic)
                    else:
                        a, _, _, _ = model.act(x, deterministic=deterministic)
                    first = torch.zeros(len(envs))
                    live = [i for i in range(len(envs)) if alive[i]]
                    for i in live:
                        envs[i].send_action(int(a[i]))
                    for i in live:
                        while not envs[i].ready():
                            time.sleep(0.0005)
                    for i in live:
                        o, r, term, trunc, info = envs[i].observe()
                        obs[i] = o
                        if term or trunc:
                            alive[i] = False
                            out.append(dict(info["episode"], slot=slot))
            except EnvError as err:
                # This Mac stalls a Dolphin every so often (10 s Null-backend stall timeout). Rebuild just the
                # bad env and drop this round's in-flight episodes rather than losing the whole evaluation --
                # the same recovery train_combat.py does. Stalls are counted so a flaky run is visible.
                stalls += 1
                i = err.idx if err.idx >= 0 else 0
                print(f"  stall on env{i} ({stalls}) -> rebuilding, dropping this round")
                try:
                    envs[i].close()
                except Exception:
                    pass
                envs[i] = mk(i)
                if stalls > 3 + episodes // 4:
                    print("  too many stalls; reporting what completed")
                    break
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
    return out, cfg


def score(eps):
    n = max(len(eps), 1)
    full = [e for e in eps if e.get("boss_hp_start", 0) >= 10]
    return {
        "episodes": len(eps),
        "defeat_rate": sum(1 for e in eps if e.get("boss_defeated")) / n,
        "death_rate": sum(1 for e in eps if e.get("died")) / n,
        "boss_hp_taken": sum(e.get("boss_damage", 0) for e in eps) / n,
        "return": sum(e.get("return", 0) for e in eps) / n,
        "full_hp_episodes": len(full),
        "full_hp_defeats": sum(1 for e in full if e.get("boss_defeated")),
        "by_slot": {str(s): {"n": sum(1 for e in eps if e["slot"] == s),
                             "defeats": sum(1 for e in eps if e["slot"] == s and e.get("boss_defeated"))}
                    for s in sorted({e["slot"] for e in eps})},
    }


def run_one(ckpt, a):
    slots = [int(s) for s in a.slots.split(",")] if a.slots else None
    model_cfg = load_policy(ckpt)[1]
    ho, trained = held_out(model_cfg)
    if slots is None:
        slots = ho or ALL_SLOTS
        if not ho:
            print("  !! this checkpoint trained on EVERY slot -- nothing is held out; using all 10 (class: "
                  "in-distribution, NOT a holdout)")
    print(f"  trained on {trained or 'unknown'} | evaluating on {slots}"
          f"{' (HELD OUT of this run)' if set(slots).isdisjoint(trained) else ' (OVERLAPS training)'}")
    eps, _ = evaluate(ckpt, slots, a.episodes, a.envs, a.slots_dir, a.episode_seconds, a.deterministic)
    s = score(eps)
    s.update(checkpoint=str(ckpt), update=model_cfg.get("update"), slots=slots, trained_slots=trained,
             held_out=bool(set(slots).isdisjoint(trained)), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"  defeat {s['defeat_rate']*100:5.2f}%  death {s['death_rate']*100:5.1f}%  "
          f"bossHP {s['boss_hp_taken']:4.2f}  return {s['return']:+7.1f}  "
          f"fullHP {s['full_hp_defeats']}/{s['full_hp_episodes']}  ({s['episodes']} eps)")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--watch", default=None, help="a checkpoints/ directory; evaluate every new .pt forever")
    p.add_argument("--slots", default=None, help="default: whatever the checkpoint did NOT train on")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--episode-seconds", type=float, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    p.add_argument("--out", default=None, help="append each result as JSON to this file")
    p.add_argument("--poll", type=float, default=60.0)
    a = p.parse_args()

    def emit(s):
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(s) + "\n")

    if a.ckpt:
        emit(run_one(Path(a.ckpt), a))
        return
    if not a.watch:
        raise SystemExit("pass --ckpt or --watch")

    d = Path(a.watch)
    seen, best = set(), None
    print(f"watching {d} every {a.poll:.0f}s")
    while True:
        for f in sorted(d.glob("update_*.pt")):
            if f.name in seen:
                continue
            seen.add(f.name)
            print(f"\n{f.name}")
            try:
                s = run_one(f, a)
            except Exception as e:
                print(f"  eval failed: {str(e)[:120]}")
                continue
            emit(s)
            if best is None or s["defeat_rate"] > best["defeat_rate"]:
                best = s
                # THE POINT: `best` is by HELD-OUT defeat rate, not by the last checkpoint written.
                print(f"  >>> NEW BEST on held-out slots: {s['defeat_rate']*100:.2f}% "
                      f"({f.name}) -- this is the KL anchor candidate")
                if a.out:
                    Path(a.out).with_suffix(".best.json").write_text(json.dumps(best, indent=2))
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
