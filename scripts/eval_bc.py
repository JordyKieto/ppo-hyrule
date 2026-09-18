"""Play a behaviour-cloning checkpoint in Dolphin and measure success rate.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/eval_bc.py checkpoints/bc/*.pt --episodes 10 --workers 6

This is the "behaviour cloning alone" row of paper/paper.md Table 3: the bar any dream-trained policy has to
clear. Reports success separately on the slots the encoder trained on and the held-out slots in its own
split, because only the latter measures generalisation.

Action accuracy is not success. A BC policy can predict the demonstrator's action 60% of the time and still
never clear a room, because errors compound over ~450 decisions. This script measures the thing that counts.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, EnemyRows, check_slots, encode_state  # noqa: E402
from zelda_rl.bc import BCNet, Norm, featurise  # noqa: E402
from zelda_rl.env import EnvError  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"   # env config only; its weights are unused


def load_bc(path: Path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    c = ck["config"]
    net = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=ck["n_actions"], latent=ck["latent"],
                hidden=c.get("hidden", 256), row_hidden=c.get("row_hidden", 96),
                layernorm=c.get("layernorm", True), dropout=0.0, gru=c.get("gru", 0))
    net.load_state_dict(ck["model"])
    net.eval()
    return net, Norm.load(ck["norm"]), ck


@torch.no_grad()
def act(net, norm, state: np.ndarray, deterministic: bool, h=None, first: float = 0.0):
    """-> (action, next hidden). The GRU head carries memory across the episode, as the demonstrator did."""
    f = norm.apply(featurise(state[None]))
    _, logits, _, h = net(torch.from_numpy(f["link"]), torch.from_numpy(f["rows"]),
                          torch.from_numpy(f["ids"]), torch.from_numpy(f["mask"]),
                          h, torch.tensor([first]))
    a = int(logits.argmax(-1)[0]) if deterministic else int(torch.distributions.Categorical(logits=logits).sample()[0])
    return a, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="BC .pt files (globs ok)")
    ap.add_argument("--episodes", type=int, default=10, help="episodes per checkpoint, spread over slots")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--deterministic", action="store_true", help="argmax instead of sampling from the BC policy")
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--out", default="results/bc_eval.json")
    args = ap.parse_args()
    torch.set_num_threads(1)

    paths = sorted({p for g in args.checkpoints for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints}")
    slots = [int(s) for s in args.slots.split(",")]
    babas = check_slots(args.slots_dir, slots)
    if bad := [k for k, n in babas.items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba -> wrong savestates in {args.slots_dir}")

    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))       # env spec must match the collection env exactly
    if args.episode_seconds:
        cfg["episode_seconds"] = args.episode_seconds
    scratch = Path("runs/bc_eval") / time.strftime("%Y%m%d-%H%M%S")
    N = min(args.workers, args.episodes)
    print(f"▶ {len(paths)} checkpoint(s) x {args.episodes} episodes | {N} headless envs | "
          f"{'argmax' if args.deterministic else 'sampled'} | slots {slots}")

    def build(i: int):
        env, _ = make_env(cfg, scratch / f"env{i}", slots=slots, slots_dir=args.slots_dir,
                          video_backend="Null", unlimited_speed=True, idx=i, audio=False)
        return env

    t0 = time.perf_counter()
    with ThreadPoolExecutor(N) as pool:
        envs = list(pool.map(build, range(N)))
    print(f"  {N} envs up in {time.perf_counter() - t0:.0f}s")

    results = {}
    try:
        for path in paths:
            net, norm, ck = load_bc(Path(path))
            val_slots = set(ck.get("split", {}).get("val_slots", []))
            m = ck.get("metrics", {})
            print(f"\n── {Path(path).name}  latent {ck['latent']}  held-out slots {sorted(val_slots) or '?'}"
                  + (f"  (val top1 {m.get('top1', float('nan')):.1%})" if "top1" in m else ""))
            jobs = [slots[i % len(slots)] for i in range(args.episodes)]
            it, lock, eps = iter(jobs), threading.Lock(), []

            def worker(i: int):
                env = envs[i]
                while True:
                    with lock:
                        slot = next(it, None)
                    if slot is None:
                        return
                    try:
                        rows = EnemyRows()
                        env.reset(slot)
                        ep, h = None, None
                        for t in range(env.cfg.max_steps):
                            a, h = act(net, norm, encode_state(env.cstate, rows), args.deterministic,
                                       h, 1.0 if t == 0 else 0.0)
                            env.send_action(a)
                            while not env.ready():
                                time.sleep(0.0005)
                            _, _, term, trunc, info = env.observe()
                            if term or trunc:
                                ep = dict(info["episode"])
                                break
                        ep = ep or dict(env.tracker.ep)
                        with lock:
                            eps.append(dict(slot=slot, end=ep["end"], defeated=ep["defeated"],
                                            targets=ep["targets_start"], steps=ep["length"],
                                            hp_lost=ep["hp_lost"], ret=round(ep["return"], 2),
                                            held_out=slot in val_slots))
                            e = eps[-1]
                            print(f"   s{slot:<2} {e['end']:<9} {e['defeated']}/{e['targets']} kills "
                                  f"{e['steps']:>4} steps  ({len(eps)}/{args.episodes})", flush=True)
                    except EnvError as err:
                        print(f"   ! env{i}: {err} -> rebuilding", flush=True)
                        try:
                            env.close()
                        except Exception:  # noqa: BLE001
                            pass
                        env = envs[i] = build(i)

            with ThreadPoolExecutor(N) as pool:
                list(pool.map(worker, range(N)))

            def summarise(sub):
                if not sub:
                    return None
                n = len(sub)
                return {"episodes": n, "success": sum(e["end"] == "success" for e in sub) / n,
                        "death": sum(e["end"] == "death" for e in sub) / n,
                        "timeout": sum(e["end"] == "timeout" for e in sub) / n,
                        "kill_frac": sum(e["defeated"] for e in sub) / max(sum(e["targets"] for e in sub), 1),
                        "mean_steps": sum(e["steps"] for e in sub) / n,
                        "mean_return": sum(e["ret"] for e in sub) / n}

            results[Path(path).name] = {
                "checkpoint": str(path), "latent": ck["latent"], "val_slots": sorted(val_slots),
                "train_metrics": {k: m.get(k) for k in ("top1", "top3", "majority") if k in m},
                "all": summarise(eps),
                "train_slots": summarise([e for e in eps if not e["held_out"]]),
                "held_out_slots": summarise([e for e in eps if e["held_out"]]),
                "episodes": eps,
            }
            a = results[Path(path).name]["all"]
            print(f"  ■ success {a['success']:.0%} | kills {a['kill_frac']:.0%} | death {a['death']:.0%} | "
                  f"timeout {a['timeout']:.0%} | mean {a['mean_steps']:.0f} steps")
            for k in ("train_slots", "held_out_slots"):
                s = results[Path(path).name][k]
                if s:
                    print(f"      {k:<16} {s['success']:>4.0%} success over {s['episodes']:>2} eps "
                          f"({s['kill_frac']:.0%} of targets killed)")
    except KeyboardInterrupt:
        print("\n interrupted")
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n-> {args.out}")
        if results:
            print(f"\n{'checkpoint':<34} {'all':>6} {'train sl':>9} {'held-out':>9} {'kills':>6}")
            for k, v in results.items():
                t, h = v["train_slots"], v["held_out_slots"]
                print(f"{k:<34} {v['all']['success']:>5.0%} {(t or {}).get('success', float('nan')):>8.0%} "
                      f"{(h or {}).get('success', float('nan')):>8.0%} {v['all']['kill_frac']:>5.0%}")
            print("\nreference: model-free PPO G150 = 10/10 slots (docs/COMBAT_TASK.md); "
                  "dataset G150 replay = 81% (paper Table 2)")


if __name__ == "__main__":
    main()
