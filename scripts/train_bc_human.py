"""Behaviour-clone the human Gohma demonstrations into a checkpoint `train_combat.py --init-from` accepts.

The point is a PPO starting point that already knows roughly what to do, instead of random weights (which take
~400 updates to reach 1.5 of her 10 HP) or v15 (which is stuck at 3.3 and resists every change).

It trains the SAME `RecurrentEntityActorCritic` the PPO trainer builds, with the same `global_dim`,
`actor_dim` and `n_actions`, so the resulting checkpoint passes `--init-from`'s spec check and loads with no
surgery. Only the architecture fields in `config` matter for that check; the rest are recorded for provenance.

Two things that are easy to get wrong and are handled explicitly:

- **Split by EPISODE, never by frame.** Consecutive frames at 10 Hz are near-duplicates, so a random frame
  split leaks the answer and reports a fake accuracy. `scripts/train_bc.py` carries the same warning about
  the Deku dataset.
- **The policy is recurrent**, so it is trained on contiguous chunks with the hidden state carried inside an
  episode and reset at its start -- the same `chunk` discipline `ppo_update_recurrent` uses. Shuffling frames
  would make the GRU untrainable.

    .venv/bin/python scripts/train_bc_human.py --data datasets/human/gohma --epochs 30
    .venv/bin/python scripts/train_bc_human.py --data datasets/human/gohma --wins-only --out checkpoints/kept/bc_human.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))

from zelda_rl import ckpt_s3
from zelda_rl.ppo import RecurrentEntityActorCritic  # noqa: E402


def load(data: Path, wins_only: bool, max_episodes: int = 0):
    """`max_episodes` takes the FIRST N in filename order -- deterministic, so a dataset-size ablation is
    reproducible and each size is a prefix of the next rather than a different random draw."""
    eps = []
    files = sorted(data.glob("ep*.npz"))
    if max_episodes:
        files = files[:max_episodes]
    for f in files:
        z = np.load(f, allow_pickle=False)
        if wins_only and not bool(z["boss_defeated"]):
            continue
        eps.append({"file": f.name, "obs": z["obs"].astype(np.float32), "act": z["action"].astype(np.int64),
                    "win": bool(z["boss_defeated"]), "slot": int(z["slot"]),
                    "obs_dim": int(z["obs_dim"]), "n_actions": int(z["n_actions"]),
                    "fields": json.loads(str(z["obs_fields"]))})
    if not eps:
        raise SystemExit(f"no episodes in {data}" + (" with a win" if wins_only else ""))
    dims = {e["obs_dim"] for e in eps}
    nacts = {e["n_actions"] for e in eps}
    if len(dims) != 1 or len(nacts) != 1:
        raise SystemExit(f"inconsistent dataset: obs_dim {dims}, n_actions {nacts}")
    return eps


def batches(eps, batch_episodes: int):
    """Group episodes into padded batches, longest-first so padding waste is small.

    The GRU is stepped over the time axis with `batch_episodes` episodes IN PARALLEL, each carrying its own
    hidden state, so per-episode continuity is preserved exactly -- this is still the instrument that measures
    clone quality, not the windowed approximation the PPO regulariser uses (zelda_rl/bc_demos.py).

    Why this shape: the previous version stepped ONE episode with BATCH SIZE 1, which is 54,626 sequential
    Python calls per epoch on the 100-game corpus and took ~1 minute per epoch. The work per step is a few
    small matmuls, so the cost is almost entirely per-call overhead, and batching 16 episodes removes 15/16ths
    of it without changing the maths.
    """
    order = sorted(range(len(eps)), key=lambda i: -len(eps[i]["act"]))
    for b0 in range(0, len(order), batch_episodes):
        sel = [eps[i] for i in order[b0:b0 + batch_episodes]]
        B, T = len(sel), max(len(e["act"]) for e in sel)
        obs = np.zeros((T, B, sel[0]["obs"].shape[1]), np.float32)
        act = np.zeros((T, B), np.int64)
        valid = np.zeros((T, B), np.float32)
        for j, e in enumerate(sel):
            n = len(e["act"])
            obs[:n, j] = e["obs"][:n]
            act[:n, j] = e["act"][:n]
            valid[:n, j] = 1.0
        yield (torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(valid))


def run_split(model, eps, chunk, opt=None, batch_episodes: int = 16):
    """One pass. With `opt`, trains; without, evaluates. Returns (loss, accuracy, n).

    Backprop is truncated every `chunk` steps (the hidden state is detached), which is the same BPTT horizon
    as before -- only the batch dimension changed.
    """
    tot_loss = tot_ok = tot_n = 0.0
    for obs, act, valid in batches(eps, batch_episodes):
        T, B = act.shape
        h = model.init_hidden(B)
        first = torch.zeros(B)
        first[:] = 1.0                                  # every episode in the batch starts at t=0
        seg_logits, seg_act, seg_valid = [], [], []
        for t in range(T):
            lg, _, h = model.step(obs[t], h, first)
            first = torch.zeros(B)
            seg_logits.append(lg); seg_act.append(act[t]); seg_valid.append(valid[t])
            if len(seg_logits) == chunk or t == T - 1:
                L = torch.stack(seg_logits, 0).reshape(-1, lg.shape[-1])
                y = torch.stack(seg_act, 0).reshape(-1)
                w = torch.stack(seg_valid, 0).reshape(-1)
                ce = F.cross_entropy(L, y, reduction="none") * w
                n = w.sum().clamp(min=1.0)
                loss = ce.sum() / n
                if opt is not None:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                with torch.no_grad():
                    tot_loss += float(loss.detach()) * float(n)
                    tot_ok += float((((L.argmax(-1) == y).float()) * w).sum())
                    tot_n += float(n)
                h = h.detach()
                seg_logits, seg_act, seg_valid = [], [], []
    return tot_loss / max(tot_n, 1), tot_ok / max(tot_n, 1), int(tot_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human/gohma")
    ap.add_argument("--out", default="checkpoints/kept/bc_human.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=60, help="BPTT window, matching the PPO trainer's default")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--val-episodes", type=int, default=2, help="held out BY EPISODE, never by frame")
    ap.add_argument("--wins-only", action="store_true", help="clone only the episodes that killed her")
    ap.add_argument("--batch-episodes", type=int, default=16,
                    help="episodes stepped IN PARALLEL. Each keeps its own hidden state, so this changes "
                         "speed and not the maths; 1 reproduces the old one-episode-at-a-time loop")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="train on the FIRST N episodes only (0 = all). For the dataset-size ablation: each "
                         "size is a prefix of the next, so the curve is not confounded by which episodes "
                         "were drawn")
    ap.add_argument("--episode-seconds", type=float, default=120.0,
                    help="recorded in the checkpoint so an evaluator rebuilds the same fight; match the PPO run")
    a = ap.parse_args()

    eps = load(Path(a.data), a.wins_only, a.max_episodes)
    fields = eps[0]["fields"]
    n_actors, actor_dim = fields["n_actors"], sum(w for _, w in fields["actor"])
    global_dim = eps[0]["obs_dim"] - n_actors * actor_dim
    n_actions = eps[0]["n_actions"]
    wins = sum(e["win"] for e in eps)
    steps = sum(len(e["act"]) for e in eps)
    print(f"{len(eps)} episodes, {wins} wins, {steps:,} steps")
    print(f"  obs {eps[0]['obs_dim']} = global {global_dim} + {n_actors} x {actor_dim} | actions {n_actions}")
    if wins == 0:
        print("  WARNING: no winning episodes -- this will clone losing play")

    # Split BY EPISODE, and across DIFFERENT START SLOTS. Taking the first N in file order held out both
    # slot-1 episodes at once, so the model saw no slot-1 play and was scored only on slot 1 -- the val number
    # then measures transfer to an unseen start rather than imitation quality. One episode per slot, from the
    # slots that have a spare, keeps both sides representative.
    by_slot: dict[int, list[int]] = {}
    for i, e in enumerate(eps):
        by_slot.setdefault(e["slot"], []).append(i)
    val_idx = set()
    for sl in sorted(by_slot, key=lambda k: (-len(by_slot[k]), k)):
        if len(val_idx) >= a.val_episodes:
            break
        if len(by_slot[sl]) > 1:                       # never hold out a slot's ONLY episode
            val_idx.add(by_slot[sl][-1])
    if not val_idx:                                    # every slot has exactly one episode
        order = sorted(range(len(eps)), key=lambda i: (not eps[i]["win"], i))
        val_idx = set(order[:a.val_episodes]) if len(eps) > a.val_episodes else set()
    tr = [e for i, e in enumerate(eps) if i not in val_idx]
    va = [e for i, e in enumerate(eps) if i in val_idx]
    print(f"  train {len(tr)} episodes / val {len(va)} ({[e['file'] for e in va]})")

    model = RecurrentEntityActorCritic(global_dim, n_actors, actor_dim, n_actions,
                                       hidden=a.hidden, gru=a.gru)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    freq = Counter(int(x) for e in eps for x in e["act"])
    base = max(freq.values()) / steps
    print(f"  majority-action baseline accuracy {100 * base:.1f}% (action {freq.most_common(1)[0][0]})\n")

    best = (0.0, -1)
    print(f"{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val loss':>11}{'val acc':>9}")
    for ep_i in range(1, a.epochs + 1):
        model.train()
        tl, ta, _ = run_split(model, tr, a.chunk, opt, a.batch_episodes)
        model.eval()
        with torch.no_grad():
            vl, vacc, _ = run_split(model, va, a.chunk, None, a.batch_episodes) if va else (float("nan"), float("nan"), 0)
        print(f"{ep_i:>6}{tl:>12.4f}{100 * ta:>10.1f}%{vl:>11.4f}{100 * vacc:>8.1f}%")
        score = vacc if va else ta
        if score > best[0]:
            best = (score, ep_i)
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            # `config` must carry what --init-from's spec check compares (global_dim, n_actions, actor_dim)
            # AND what policy_io.make_env needs to rebuild the fight (combat_cfg, episode_seconds,
            # vi_per_step). Without the second group the trainer loads the clone fine and the EVALUATOR dies
            # on `KeyError: 'combat_cfg'` -- which is how the first eval of this checkpoint failed.
            from dataclasses import asdict
            from zelda_rl.combat import CombatConfig, GOHMA_SIMPLE
            ccfg = CombatConfig(profile="gohma", max_seconds=a.episode_seconds, **GOHMA_SIMPLE)
            ckpt = {"model": model.state_dict(), "optimizer": None, "update": 0, "global_step": 0,
                        "episode_count": 0,
                        "config": {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
                                   "n_actions": n_actions, "obs_dim": eps[0]["obs_dim"], "hidden": a.hidden,
                                   "gru": a.gru, "recurrent": True, "obs_version": 7, "task": "combat",
                                   "combat_cfg": asdict(ccfg), "episode_seconds": a.episode_seconds,
                                   "vi_per_step": 6, "goal_slot": 3, "steps_per_rollout": 0,
                                   "profile": "gohma", "run_name": "bc_human",
                                   "bc_source": str(a.data), "bc_episodes": len(eps), "bc_wins": wins,
                                   "bc_steps": steps, "bc_val_acc": float(vacc), "bc_epoch": ep_i,
                                   "bc_wins_only": bool(a.wins_only)}}
            torch.save(ckpt, out)
            ckpt_s3.upload(out, f"bc_human_n{len(eps)}", ep_i)
    print(f"\nbest {'val' if va else 'train'} accuracy {100 * best[0]:.1f}% at epoch {best[1]} -> {a.out}")
    print(f"  majority baseline was {100 * base:.1f}% -- beat that or the clone learned nothing")
    print(f"\nuse it:  fleet ckpt --experiment E --path {a.out}")
    print(f"         ... --train-args \"... --init-from {a.out} ...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
