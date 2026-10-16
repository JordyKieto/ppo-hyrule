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
                    "fields": json.loads(str(z["obs_fields"])), "z": f})
    if not eps:
        raise SystemExit(f"no episodes in {data}" + (" with a win" if wins_only else ""))
    dims = {e["obs_dim"] for e in eps}
    nacts = {e["n_actions"] for e in eps}
    if len(dims) != 1 or len(nacts) != 1:
        raise SystemExit(f"inconsistent dataset: obs_dim {dims}, n_actions {nacts}")
    return eps


FACTORS = {"kill": 3.0, "sword": 2.0, "seed_hit": 1.5, "boss_hp": 1.5, "stun": 1.2,
           "seed_fire": 0.8, "sling_out": 0.8, "larva": 0.3}


def event_weights(z, obs, act, span: int, scale: float, only=None,
                  factors=None) -> tuple[np.ndarray, dict]:
    """Per-step loss weights keyed on the DECISIONS that matter, so positioning cannot dominate the gradient.

    Uniform imitation spends its gradient in proportion to how often something happens, and in this corpus
    that means walking: the human's first damage lands at step 190 of a 546-step episode, and only 2.99 seeds
    are fired per win. The classes below are the moments where the human chose something consequential --
    measured, not guessed (scripts/mine_corpus.py, scripts/mine_slingshot.py) -- and each is up-weighted by
    `scale x class factor` over a `span` leading up to it, which is where the decision was actually made.

    Classes, highest first. A step takes the MAX of the classes covering it, so overlaps do not compound.

      the KILL         3.0  her last two HP drops: the killing blow and the hit that set it up
      sword swing      2.0  B pressed while she is stunned and within sword range -- offline accuracy on
                            these steps is the only metric measured to track online damage (r = +0.83)
      successful seed  1.5  fired and an impact followed within 1.2 s -- the skill the fight turns on
      her HP falls     1.5  the payoff moment
      stun onset       1.2  the window that permits sword damage (gates 100% of recorded hits)
      seed fired      0.8  any shot, landed or not
      slingshot out   0.8  the equip decision that precedes every shot
      larva hit       0.3  kept low on purpose: larva hits correlate -0.598 with damage dealt
      everything else 0    positioning, approach, waiting -> weight 1.0
    """
    import json as _json
    f = _json.loads(str(z["obs_fields"]))
    off, idx = 0, {}
    for name, w in f["global"]:
        idx[name] = off
        off += w
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTONS_V5
    btn = [set(COMBAT_BUTTONS_V5[b]) for _, b in COMBAT_ACTIONS_V5]

    T = len(obs)
    w = np.ones(T, np.float32)
    ghp = obs[:, idx["gohma_hp_frac"]]
    stun = (obs[:, idx["gohma_stunned"]] > 0.5).astype(np.int8)
    dist = obs[:, idx["gohma_eye_dist"]] * 1000.0
    sling = (obs[:, idx["held_slingshot"]] > 0.5).astype(np.int8)
    fires = np.where((obs[:-1, idx["since_last_seed"]] > 0.05) &
                     (obs[1:, idx["since_last_seed"]] <= 0.05))[0] + 1 if "since_last_seed" in idx else []
    imps = set((np.where((obs[:-1, idx["since_last_impact"]] > 0.05) &
                         (obs[1:, idx["since_last_impact"]] <= 0.05))[0] + 1).tolist()
               ) if "since_last_impact" in idx else set()
    larva = []
    for n in ("larva0_hp", "larva1_hp", "larva2_hp"):
        if n in idx:
            c = obs[:, idx[n]]
            larva += (np.where(c[:-1] - c[1:] > 1e-6)[0] + 1).tolist()

    hp_drops = (np.where(ghp[:-1] - ghp[1:] > 1e-6)[0] + 1).tolist()
    # THE KILL: her last HP drops of the episode. Every episode in this corpus is a win, so the final drop is
    # the killing blow and the one before it is the hit that set it up. They are weighted ABOVE everything
    # else -- the reward already believes this asymmetry (`g_boss_hp_progress` pays 3x for late hits) and
    # until now the clone did not: all 678 drops shared one factor, including the 100 that ended a fight.
    kills = hp_drops[-2:] if len(hp_drops) >= 2 else hp_drops
    F = dict(FACTORS, **(factors or {}))
    marks = {
        "kill": kills,
        "sword": [t for t in range(T) if "B" in btn[int(act[t])] and stun[t] and dist[t] < 120],
        "seed_hit": [i for i in fires if any(i < j <= i + 12 for j in imps)],
        "boss_hp": hp_drops,
        "stun": (np.where((stun[1:] == 1) & (stun[:-1] == 0))[0] + 1).tolist(),
        "seed_fire": list(fires),
        "sling_out": (np.where((sling[1:] == 1) & (sling[:-1] == 0))[0] + 1).tolist(),
        "larva": larva,
    }
    # A class whose factor is 0 contributes nothing, which is how a single class is isolated without also
    # removing it from the event bookkeeping.
    classes = {k: (F.get(k, 0.0), v) for k, v in marks.items() if F.get(k, 0.0) > 0}
    if only:
        classes = {k: v for k, v in classes.items() if k in only}
    share = {}
    for name, (factor, marks) in sorted(classes.items(), key=lambda kv: kv[1][0]):
        hit = np.zeros(T, bool)
        for m in marks:
            hit[max(0, m - span):m + 1] = True
        w[hit] = np.maximum(w[hit], 1.0 + scale * factor)
        share[name] = (len(marks), int(hit.sum()))
    return w, share


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
        swm = np.zeros((T, B), np.float32)          # sword-step mask; `sel` above is the episode list
        for j, e in enumerate(sel):
            n = len(e["act"])
            obs[:n, j] = e["obs"][:n]
            act[:n, j] = e["act"][:n]
            valid[:n, j] = e.get("w", np.ones(n, np.float32))[:n]
            swm[:n, j] = e.get("sel", np.zeros(n, np.float32))[:n]
        yield (torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(valid),
               torch.from_numpy(swm))


def run_split(model, eps, chunk, opt=None, batch_episodes: int = 16, want_sword: bool = False):
    """One pass. With `opt`, trains; without, evaluates. Returns (loss, accuracy, n).

    Backprop is truncated every `chunk` steps (the hidden state is detached), which is the same BPTT horizon
    as before -- only the batch dimension changed.
    """
    tot_loss = tot_ok = tot_n = 0.0
    sw_hit = sw_n = 0.0
    for obs, act, valid, swmask in batches(eps, batch_episodes):
        T, B = act.shape
        h = model.init_hidden(B)
        first = torch.zeros(B)
        first[:] = 1.0                                  # every episode in the batch starts at t=0
        seg_logits, seg_act, seg_valid, seg_sel = [], [], [], []
        for t in range(T):
            lg, _, h = model.step(obs[t], h, first)
            first = torch.zeros(B)
            seg_logits.append(lg); seg_act.append(act[t]); seg_valid.append(valid[t])
            seg_sel.append(swmask[t])
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
                    ok = (L.argmax(-1) == y).float()
                    tot_ok += float((ok * w).sum())
                    tot_n += float(n)
                    if want_sword:
                        sm = torch.stack(seg_sel, 0).reshape(-1)
                        sw_hit += float((ok * sm).sum()); sw_n += float(sm.sum())
                h = h.detach()
                seg_logits, seg_act, seg_valid, seg_sel = [], [], [], []
    if want_sword:
        return tot_loss / max(tot_n, 1), tot_ok / max(tot_n, 1), int(tot_n), sw_hit / max(sw_n, 1.0)
    return tot_loss / max(tot_n, 1), tot_ok / max(tot_n, 1), int(tot_n)


def ckpt_payload(model, a, eps, global_dim, n_actors, actor_dim, n_actions, wins, steps, vacc, ep_i,
                 vsw=float("nan")):
    """What goes in a checkpoint: the architecture fields `--init-from` compares, the env fields
    `policy_io.make_env` needs to rebuild the fight, and the provenance of this particular clone."""
    from dataclasses import asdict
    from zelda_rl.combat import CombatConfig, GOHMA_SIMPLE
    ccfg = CombatConfig(profile="gohma", max_seconds=a.episode_seconds, **GOHMA_SIMPLE)
    return {"model": model.state_dict(), "optimizer": None, "update": 0, "global_step": 0,
            "episode_count": 0,
            "config": {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
                       "n_actions": n_actions, "obs_dim": eps[0]["obs_dim"], "hidden": a.hidden,
                       "gru": a.gru, "recurrent": True, "obs_version": 7, "task": "combat",
                       "combat_cfg": asdict(ccfg), "episode_seconds": a.episode_seconds,
                       "vi_per_step": 6, "goal_slot": 3, "steps_per_rollout": 0,
                       "profile": "gohma", "run_name": "bc_human",
                       "bc_source": str(a.data), "bc_episodes": len(eps), "bc_wins": wins,
                       "bc_steps": steps, "bc_val_acc": float(vacc), "bc_sword_acc": float(vsw),
                       "bc_epoch": ep_i, "bc_event_weight": a.event_weight, "bc_event_span": a.event_span,
                       "bc_event_classes": a.event_classes, "bc_event_factors": a.event_factors,
                       "bc_select_by": a.select_by, "bc_seed": a.seed,
                       "bc_wins_only": bool(a.wins_only)}}


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
    ap.add_argument("--event-weight", type=float, default=1.0,
                    help="EVENT STRATIFICATION: loss weight on steps in the damage chain -- within "
                         "--event-span of a seed fire, a stun onset or one of her HP drops. 1.0 = uniform "
                         "imitation, which spends most of its gradient on the two thirds of each episode "
                         "that is walking around")
    ap.add_argument("--snapshot-every", type=int, default=0, metavar="N",
                    help="also save an UNCONDITIONAL checkpoint every N epochs to <out>.snaps/. Point "
                         "`eval_fleet.py --watch <out>.snaps` at that directory and every snapshot is scored "
                         "online as it lands, which answers whether more training keeps buying damage or "
                         "only keeps buying validation accuracy")
    ap.add_argument("--seed", type=int, default=None,
                    help="torch seed. Unset = nondeterministic, which is what every arm so far used, so "
                         "repeated runs of one config are independent draws (AGENTS 7)")
    ap.add_argument("--event-factors", default="",
                    help="override per-class factors, e.g. 'sword=1.0,kill=0' -- the weight on a class is "
                         "1 + scale x factor, so 0 switches a class off. Use it to isolate ONE class at a "
                         "time; the defaults are in docs/BC_EVENT_CLASSES.md")
    ap.add_argument("--select-by", choices=["sword", "acc"], default="sword",
                    help="which validation metric keeps the checkpoint. `sword` is accuracy restricted to the "
                         "steps where B is pressed while she is stunned and in range -- the only offline "
                         "metric measured to track ONLINE damage (r=+0.83). `acc` is overall accuracy, which "
                         "correlates -0.43 and is kept only to reproduce older runs")
    ap.add_argument("--event-classes", default="",
                    help="comma list restricting WHICH decision classes are up-weighted: seed_hit, boss_hp, "
                         "stun, sword, seed_fire, sling_out, larva. Empty = all of them. Use it to ask which "
                         "events matter rather than assuming the whole set does")
    ap.add_argument("--event-span", type=int, default=15,
                    help="steps BEFORE a chain event that count as part of it (15 = 1.5 s)")
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

    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
    eps = load(Path(a.data), a.wins_only, a.max_episodes)
    only = set(x.strip() for x in a.event_classes.split(",") if x.strip()) if a.event_classes else None
    fac = {}
    for kv in a.event_factors.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            fac[k.strip()] = float(v)
    if fac:
        print(f"  factor overrides: {fac}")
    # The SWORD mask is built regardless of stratification: accuracy on these steps is the only offline
    # metric measured to track online boss damage (r = +0.83 over five arms), so it is what selects the
    # checkpoint. Overall accuracy correlates -0.43 and had been choosing the checkpoint until now.
    for e in eps:
        z = np.load(e["z"], allow_pickle=False)
        wsw, _ = event_weights(z, e["obs"], e["act"], a.event_span, 1.0, {"sword"}, {"sword": 1.0})
        e["sel"] = (wsw > 1.0).astype(np.float32)
    if a.event_weight != 1.0:
        tot = chain = 0
        if only:
            print(f"  only up-weighting: {sorted(only)}")
        agg = {}
        for e in eps:
            z = np.load(e["z"], allow_pickle=False)
            e["w"], sh = event_weights(z, e["obs"], e["act"], a.event_span, a.event_weight, only, fac)
            tot += len(e["w"]); chain += int((e["w"] > 1.0).sum())
            for k, (n, cov) in sh.items():
                p0, p1 = agg.get(k, (0, 0))
                agg[k] = (p0 + n, p1 + cov)
        print(f"  event stratification at scale {a.event_weight}, span {a.event_span}: "
              f"{chain:,}/{tot:,} steps up-weighted ({100*chain/max(tot,1):.1f}%); "
              f"{100-100*chain/max(tot,1):.1f}% stays at 1.0 (positioning)")
        for k in ("kill", "sword", "seed_hit", "boss_hp", "stun", "seed_fire", "sling_out", "larva"):
            n, cov = agg.get(k, (0, 0))
            print(f"    {k:<11}{n:>6} events  {cov:>7,} steps  {100*cov/max(tot,1):>5.1f}% of the corpus")
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
    print(f"{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val loss':>11}{'val acc':>9}"
          f"{'SWORD acc':>9}")
    for ep_i in range(1, a.epochs + 1):
        model.train()
        tl, ta, _ = run_split(model, tr, a.chunk, opt, a.batch_episodes)
        model.eval()
        with torch.no_grad():
            vl, vacc, _, vsw = (run_split(model, va, a.chunk, None, a.batch_episodes, True)
                                if va else (float("nan"), float("nan"), 0, float("nan")))
        print(f"{ep_i:>6}{tl:>12.4f}{100 * ta:>10.1f}%{vl:>11.4f}{100 * vacc:>8.1f}%{100 * vsw:>9.1f}%")
        if a.snapshot_every and (ep_i % a.snapshot_every == 0 or ep_i == a.epochs):
            snap = Path(str(a.out) + ".snaps")
            snap.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_payload(model, a, eps, global_dim, n_actors, actor_dim, n_actions, wins, steps,
                                    vacc, ep_i, vsw), snap / f"snap_e{ep_i:04d}.pt")
        score = (vsw if a.select_by == "sword" else vacc) if va else ta
        if score > best[0]:
            best = (score, ep_i)
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_payload(model, a, eps, global_dim, n_actors, actor_dim, n_actions, wins,
                                    steps, vacc, ep_i, vsw), out)
            ckpt_s3.upload(out, f"bc_human_n{len(eps)}", ep_i)
    print(f"\nbest {a.select_by if va else 'train'} accuracy {100 * best[0]:.1f}% at epoch {best[1]} "
          f"-> {a.out}")
    print(f"  majority baseline was {100 * base:.1f}% -- beat that or the clone learned nothing")
    print(f"\nuse it:  fleet ckpt --experiment E --path {a.out}")
    print(f"         ... --train-args \"... --init-from {a.out} ...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
