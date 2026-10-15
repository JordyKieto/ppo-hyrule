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

from zelda_rl.ppo import RecurrentEntityActorCritic  # noqa: E402


def load(data: Path, wins_only: bool):
    eps = []
    for f in sorted(data.glob("ep*.npz")):
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


def chunks(ep, chunk: int):
    """Contiguous (obs, act, first) windows. `first` marks the episode start so the GRU resets there."""
    T = len(ep["act"])
    for i in range(0, T, chunk):
        o = ep["obs"][i:i + chunk]
        a = ep["act"][i:i + chunk]
        f = np.zeros(len(a), np.float32)
        if i == 0:
            f[0] = 1.0
        yield o, a, f


def run_split(model, eps, chunk, opt=None):
    """One pass. With `opt`, trains; without, evaluates. Returns (loss, accuracy, n)."""
    tot_loss = tot_ok = tot_n = 0
    for ep in eps:
        h = model.init_hidden(1)
        for o, a, f in chunks(ep, chunk):
            x = torch.from_numpy(o).unsqueeze(1)                 # (T, 1, obs)
            y = torch.from_numpy(a)
            first = torch.from_numpy(f).unsqueeze(1)
            logits = []
            hh = h
            for t in range(len(y)):
                lg, _, hh = model.step(x[t], hh, first[t])
                logits.append(lg)
            lg = torch.cat(logits, 0)
            loss = F.cross_entropy(lg, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            h = hh.detach()
            tot_loss += float(loss) * len(y)
            tot_ok += int((lg.argmax(-1) == y).sum())
            tot_n += len(y)
    return tot_loss / max(tot_n, 1), tot_ok / max(tot_n, 1), tot_n


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
    a = ap.parse_args()

    eps = load(Path(a.data), a.wins_only)
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

    # Split BY EPISODE. Prefer holding out a win so the val number means something.
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
        tl, ta, _ = run_split(model, tr, a.chunk, opt)
        model.eval()
        with torch.no_grad():
            vl, vacc, _ = run_split(model, va, a.chunk) if va else (float("nan"), float("nan"), 0)
        print(f"{ep_i:>6}{tl:>12.4f}{100 * ta:>10.1f}%{vl:>11.4f}{100 * vacc:>8.1f}%")
        score = vacc if va else ta
        if score > best[0]:
            best = (score, ep_i)
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            # `config` must carry what --init-from's spec check compares: global_dim, n_actions, actor_dim.
            torch.save({"model": model.state_dict(), "optimizer": None, "update": 0, "global_step": 0,
                        "episode_count": 0,
                        "config": {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
                                   "n_actions": n_actions, "obs_dim": eps[0]["obs_dim"], "hidden": a.hidden,
                                   "gru": a.gru, "recurrent": True, "obs_version": 7, "task": "combat",
                                   "profile": "gohma", "run_name": "bc_human",
                                   "bc_source": str(a.data), "bc_episodes": len(eps), "bc_wins": wins,
                                   "bc_steps": steps, "bc_val_acc": float(vacc), "bc_epoch": ep_i,
                                   "bc_wins_only": bool(a.wins_only)}}, out)
    print(f"\nbest {'val' if va else 'train'} accuracy {100 * best[0]:.1f}% at epoch {best[1]} -> {a.out}")
    print(f"  majority baseline was {100 * base:.1f}% -- beat that or the clone learned nothing")
    print(f"\nuse it:  fleet ckpt --experiment E --path {a.out}")
    print(f"         ... --train-args \"... --init-from {a.out} ...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
