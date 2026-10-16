"""Return-conditioned BC over the WHOLE corpus: 100 human wins plus thousands of PPO losses.

See zelda_rl/rvs.py for why the conditioning is outcome-based rather than reward-based (two reward profiles
in one corpus, and the human's own return is ~80% damage-independent).

Wins are rare and must be oversampled or the gradient is 99% losing play: `--win-weight` multiplies a winning
episode's loss weight. The log prints the resulting share of the gradient so the balance is visible and not
assumed.

    .venv/bin/python scripts/train_rvs.py --data datasets/human/gohma --data datasets/ppo/gohma_steps \\
        --win-weight 20 --epochs 40 --out checkpoints/kept/rvs.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)   # AGENTS 28: this script now opens Dolphins for its online eval
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tbh", REPO / "scripts" / "train_bc_human.py")
tbh = importlib.util.module_from_spec(spec); sys.modules["tbh"] = tbh; spec.loader.exec_module(tbh)

from zelda_rl import ckpt_s3  # noqa: E402
from zelda_rl.ppo import RecurrentEntityActorCritic  # noqa: E402
from zelda_rl.rvs import COND_DIM, TARGET, conditioning, field_offsets  # noqa: E402


def _mem_limit() -> int | None:
    """The limit that actually applies, from the cgroup -- not what `free` says (AGENTS 23).

    ZRL_MEM_LIMIT_BYTES overrides it: needed to TEST the guard (a dev box has no cgroup, so the first
    version of this skipped the check silently and its control reported "FAIL: no refusal"), and useful on
    any host where the cgroup files cannot be read.
    """
    import os
    env = os.environ.get("ZRL_MEM_LIMIT_BYTES")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from zelda_rl.wm_data import host_mem_limit
        return host_mem_limit()
    except Exception:                                        # noqa: BLE001
        return None


def load_all(dirs, limit_per_dir=0, mem_frac: float = 0.55):
    """Load every episode, REFUSING before the OOM killer gets involved.

    `la3` was OOM-killed at 8,345 episodes on 2026-10-16: each episode holds ~2.9 MB of float32
    observations, the Dyna refresh reloads the whole corpus while the previous copy is still referenced,
    and the pod's cgroup limit was 32 GB while `free` reported 503 GB. The run died with `Killed` -- no
    traceback -- and the health check then reported it as "FINISHED (clean exit)". A silent death is worse
    than a crash, so this aborts with a number and an instruction instead.
    """
    eps = []
    lim = _mem_limit()
    budget = int(lim * mem_frac) if lim else None
    used = 0
    for d in dirs:
        files = sorted(Path(d).glob("*.npz"))
        if limit_per_dir:
            files = files[:limit_per_dir]
        n_before = len(eps)
        bad = 0
        for f in files:
            # A corpus is downloaded, and a download can be truncated. This is the SECOND loader to die on
            # `EOFError: No data left in file` today -- zelda_rl.rvs_data was fixed this morning and this
            # function has its own np.load, so it crashed the M2-shape cell (gohma-la r0). Skip and REPORT.
            try:
                z = np.load(f, allow_pickle=False)
                if "obs" not in z.files or "action" not in z.files:
                    continue
                z["obs"].shape, z["action"].shape      # force the read: a lazy npz raises here
            except Exception:                          # noqa: BLE001
                bad += 1
                continue
            obs = z["obs"].astype(np.float32)
            used += obs.nbytes + obs.shape[0] * COND_DIM * 4
            if budget and used > budget:
                g_ = 2 ** 30
                raise SystemExit(
                    f"CORPUS TOO LARGE FOR THIS POD: {len(eps)} episodes already hold "
                    f"{used / g_:.1f} GiB and the cgroup limit is {lim / g_:.1f} GiB "
                    f"(budget {mem_frac:.0%} = {budget / g_:.1f} GiB, and a Dyna refresh needs a second "
                    f"copy). Reduce the corpus -- cap the directory, use --limit-per-dir, or lower "
                    f"--dyna-dir's contents. Refusing rather than being OOM-killed with no traceback.")
            idx = field_offsets(str(z["obs_fields"]))
            won = bool(z["boss_defeated"]) if "boss_defeated" in z.files else False
            g = conditioning(obs, idx, won)
            eps.append({"file": f.name, "src": str(d), "obs": np.concatenate([obs, g], 1),
                        "act": z["action"].astype(np.int64), "win": won,
                        "dmg": int(z["boss_damage"]) if "boss_damage" in z.files else 0,
                        "obs_dim": obs.shape[1] + COND_DIM, "raw_dim": obs.shape[1],
                        "n_actions": int(z["n_actions"]) if "n_actions" in z.files else 108,
                        "fields": json.loads(str(z["obs_fields"]))})
        load_all.skipped = getattr(load_all, "skipped", 0) + bad
        load_all.gib = used / 2 ** 30
        if bad:
            print(f"  {d}: SKIPPED {bad} unreadable of {len(files)} file(s), kept "
                  f"{len(eps) - n_before}", flush=True)
    if not eps:
        raise SystemExit(f"no episodes in {dirs} ({getattr(load_all, 'skipped', 0)} unreadable)")
    dims = {e["obs_dim"] for e in eps}
    if len(dims) != 1:
        raise SystemExit(f"mixed observation widths: {dims}")
    return eps


CORPUS = {"episodes": 0, "wins": 0}          # updated by prepare(), printed beside every eval


def online_eval(ckpt: Path, a, ep_i: int, hist: list[tuple[int, float]]) -> float | None:
    """Play the game with the TARGET prompt and print balanced boss damage with its slope.

    Reuses scripts/eval_fleet.evaluate rather than reimplementing a rollout -- one tested path, and the
    checkpoint is loaded through policy_io, which wraps an `rvs_cond` checkpoint with the target prompt
    g = (1, 1, 0) automatically. Balanced = the UNWEIGHTED mean of the per-slot means, because an
    episode-weighted average moves with the slot mix and cannot compare arms (AGENTS 29).
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("eval_fleet", Path(__file__).with_name("eval_fleet.py"))
    ef = _ilu.module_from_spec(spec); spec.loader.exec_module(ef)
    slots = [int(x) for x in a.eval_slots.split(",") if x.strip()]
    if getattr(a, "dyna_dir", None):
        ef.LIVE["dump"] = a.dyna_dir          # every episode this eval plays joins the corpus
        ef.LIVE["policy"] = str(ckpt)
    try:
        eps, _ = ef.evaluate(ckpt, slots, a.eval_episodes, a.eval_envs, a.slots_dir, a.episode_seconds)
    except Exception as e:  # noqa: BLE001
        print(f"  EVAL epoch {ep_i}: FAILED ({type(e).__name__}: {str(e)[:90]})", flush=True)
        return None
    if not eps:
        print(f"  EVAL epoch {ep_i}: no episodes", flush=True)
        return None
    per = {}
    for e in eps:
        per.setdefault(e["slot"], []).append(float(e.get("boss_damage", 0)))
    bal = float(np.mean([np.mean(v) for v in per.values()]))
    pooled = float(np.mean([d for v in per.values() for d in v]))
    wins = sum(1 for e in eps if e.get("boss_defeated"))
    # RETURN as well as damage. 34 says the headline is what the reward pays for, and the first version of
    # this line printed damage and kills only -- so "what is the reward trend" could not be answered for an
    # arm that had already run forty epochs. It is free: eval_fleet hands back each episode's return.
    ret = float(np.mean([float(e.get("return", 0.0)) for e in eps]))
    hurt = float(np.mean([float(e.get("hp_lost", 0.0)) for e in eps]))
    hist.append((ep_i, bal))
    trend = ""
    if len(hist) >= 3:
        u = np.array([h[0] for h in hist], float); d = np.array([h[1] for h in hist], float)
        A = np.vstack([u, np.ones_like(u)]).T
        beta, *_ = np.linalg.lstsq(A, d, rcond=None)
        r = d - A @ beta
        se = float(np.sqrt((r @ r / max(len(u) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = beta[0] / se if se > 0 else float("nan")
        trend = (f" | slope {beta[0]:+.4f} +- {se:.4f} /epoch  t={t:+.2f}  "
                 f"{'RISING' if t > 2 else 'FALLING' if t < -2 else 'flat'} over {len(hist)} evals")
    print(f"  EVAL epoch {ep_i}: dmgB(balanced) {bal:.3f} | pooled {pooled:.3f} | ret {ret:.1f} | "
          f"hpLost {hurt:.2f} | wins {wins}/{len(eps)} | slots {len(per)} | "
          f"corpus {CORPUS['episodes']}eps/{CORPUS['wins']}wins{trend}", flush=True)
    return bal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="repeatable: corpus directories")
    ap.add_argument("--limit-per-dir", type=int, default=0)
    ap.add_argument("--win-share", type=float, default=0.3,
                    help="TARGET fraction of the gradient that winning play should carry; the multiplier is "
                         "solved from the corpus to hit it. A fixed multiplier is the wrong knob: 20x was "
                         "sized for a ~1%%-win PPO corpus and on a 48%%-win mix it put wins at 93%% of the "
                         "gradient, which is plain BC with extra steps")
    ap.add_argument("--win-weight", type=float, default=0.0,
                    help="override the solved multiplier (0 = solve it from --win-share)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-episodes", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--episode-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/kept/rvs.pt")
    # ONLINE eval inside the arm. Offline accuracy is not the headline: the clone reached 53% val accuracy
    # and 0.33 boss damage, and the only number that ranks a policy is the damage it does in the game
    # (AGENTS 34). Evaluating here means one pod gives training AND its own feedback loop, with a slope.
    ap.add_argument("--eval-every", type=int, default=4, help="online eval every N epochs (0 = never)")
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--eval-slots", default="3,4,5,6,7,8")
    ap.add_argument("--slots-dir", default="assets/states")
    # DYNA: the online eval already plays the game with this policy, and eval_fleet writes every episode as
    # a corpus-format .npz when its dump sink is set. So the evaluator IS the collector -- the arm absorbs
    # its OWN current episodes, each labelled by rvs.conditioning() with the outcome it actually produced,
    # and keeps training on the grown corpus. The static corpus dirs are never evicted, so the 100 human
    # wins stay anchored (the self-imitation failure mode is a buffer that drops them).
    ap.add_argument("--fresh", action="store_true",
                    help="deliberately discard an existing checkpoint at --out and start from scratch")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out if it exists (weights + optimiser + epoch). Without this a "
                         "restart begins from a random init and every epoch already paid for is lost -- "
                         "which happened three times on 2026-10-16, once costing an arm that had reached "
                         "1.54 online damage")
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="cap optimiser steps per epoch, so the EVAL CADENCE stays constant while the "
                         "corpus grows. With --dyna-dir the corpus went 792 -> 5,608 episodes and an "
                         "'epoch' got 7x slower, so --eval-every 1 still meant one point every ~8 min")
    # DISTRIBUTED: one model across every pod, not N independent seeds. FleetLink.average_weights is the
    # same aggregator-pattern sync train_combat uses -- every rank uploads, the aggregator averages what
    # arrived within straggler_wait and publishes, followers load it. Six pods then give 6x the gradient
    # throughput on ONE model, and their online evals pool into one kill-rate measurement instead of six
    # under-powered ones (separating 3.78% from 2.64% needs ~2,900 episodes per group).
    ap.add_argument("--fleet-experiment", default=None)
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world-size", type=int, default=1)
    ap.add_argument("--fleet-aggregator", type=int, default=0)
    ap.add_argument("--fleet-launch-id", default="0")
    ap.add_argument("--fleet-sync-every", type=int, default=4, help="average weights every N epochs")
    ap.add_argument("--dyna-dir", default=None,
                    help="absorb the online eval's own episodes from this directory into the corpus "
                         "(Dyna). Off when unset.")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)

    def prepare(dirs):
        """Load the corpus, solve the win multiplier for THIS composition, and split train/val.

        Re-run whenever Dyna grows the corpus: the multiplier must be re-solved, because absorbing the
        policy's own episodes changes the winning fraction of steps and a fixed multiplier would silently
        drift away from --win-share (the whole reason solve_win_weight exists).
        """
        eps = load_all(dirs, a.limit_per_dir)
        # REAL DATA PARALLELISM, not an ensemble. Averaging weights across ranks that each train on the
        # WHOLE corpus costs 6x the compute for ~1x the speed: every rank still does a full epoch. So the
        # LOSING episodes -- the bulk -- are sharded by rank, and an epoch over the full corpus takes 1/N
        # the wall clock with the average stitching it back together.
        #
        # The WINNING episodes are NOT sharded. 100 human wins are the scarce anchor and the win class is
        # what drives damage (R72/R77/R88 from three directions); giving each rank ~17 of them would dilute
        # the one thing that matters. Every rank keeps every win and 1/N of the failures.
        if a.fleet_world_size > 1:
            n_before = len(eps)
            # STRATIFIED sharding: every CLASS is split round-robin, so each rank holds the same
            # COMPOSITION as the whole corpus -- its own slice of the human wins, of the PPO wins and of
            # the failures. Anchoring is preserved by composition, not by duplication: the win multiplier
            # is re-solved per rank, so human play carries the same share of each rank's gradient, and the
            # weight average recombines all 100 human wins every round. Keeping them whole on every rank
            # capped the speedup at ~2x for no statistical benefit; this is the full ~Nx.
            def klass(e):
                src = (str(e.get("src", "")) + " " + str(e.get("file", ""))).lower()
                return "human" if "human" in src else ("win" if e["win"] else "loss")
            buckets: dict[str, list] = {}
            for e in eps:
                buckets.setdefault(klass(e), []).append(e)
            eps = [e for k in sorted(buckets)
                   for e in buckets[k][a.fleet_rank::a.fleet_world_size]]
            got = {k: len(buckets[k][a.fleet_rank::a.fleet_world_size]) for k in sorted(buckets)}
            print(f"  shard {a.fleet_rank}/{a.fleet_world_size} (stratified): {got} -> {len(eps)} of "
                  f"{n_before} episodes (~{n_before/max(len(eps),1):.1f}x faster epoch; every class keeps "
                  f"its proportion and the average recombines all ranks)", flush=True)
        wins = [e for e in eps if e["win"]]
        lim_ = _mem_limit()
        print(f"corpus holds {getattr(load_all, 'gib', 0):.1f} GiB in RAM"
              + (f" of a {lim_ / 2 ** 30:.0f} GiB cgroup limit" if lim_ else " (no cgroup limit visible)"),
              flush=True)
        print(f"corpus: {len(eps)} episodes, {len(wins)} wins "
              f"({100*len(wins)/max(len(eps),1):.1f}%), {sum(len(e['act']) for e in eps):,} steps",
              flush=True)
        if getattr(load_all, "skipped", 0):
            print(f"  SKIPPED {load_all.skipped} unreadable .npz file(s) in total", flush=True)
        for d in dirs:
            sub = [e for e in eps if e["src"] == str(d)]
            print(f"  {d}: {len(sub)} episodes, {sum(1 for e in sub if e['win'])} wins, "
                  f"{sum(len(e['act']) for e in sub):,} steps")

        # Solve the multiplier so winning play lands at --win-share of the gradient. w = s(1-p)/(p(1-s)) where
        # p is the winning fraction of STEPS, so the knob is the share we want and not a guess that only
        # happens to be right for one corpus composition.
        win_steps = sum(len(e["act"]) for e in eps if e["win"])
        all_steps = sum(len(e["act"]) for e in eps)
        p_ = win_steps / max(all_steps, 1)
        if a.win_weight > 0:
            solved = a.win_weight
        elif 0.0 < a.win_share < 1.0 and 0.0 < p_ < 1.0:
            solved = (a.win_share * (1 - p_)) / (p_ * (1 - a.win_share))
        else:
            solved = 1.0
        print(f"  winning play is {100*p_:.1f}% of steps; multiplier {solved:.2f} to reach "
              f"{100*a.win_share:.0f}% of the gradient")
        tot_w = win_w = 0.0
        for e in eps:
            w = solved if e["win"] else 1.0
            e["w"] = np.full(len(e["act"]), w, np.float32)
            tot_w += w * len(e["act"])
            win_w += (w * len(e["act"]) if e["win"] else 0.0)
        print(f"  -> winning play is {100*win_w/max(tot_w,1e-9):.1f}% of the gradient")

        rng = np.random.default_rng(0)
        order = rng.permutation(len(eps))
        n_val = max(2, int(a.val_frac * len(eps)))
        val_idx = set(order[:n_val].tolist())
        tr = [e for i, e in enumerate(eps) if i not in val_idx]
        va = [e for i, e in enumerate(eps) if i in val_idx]
        print(f"  train {len(tr)} / val {len(va)} (val wins: {sum(1 for e in va if e['win'])})")
        CORPUS["episodes"], CORPUS["wins"] = len(eps), len(wins)
        return eps, tr, va, solved

    eps, tr, va, solved = prepare(list(a.data))
    wins = [e for e in eps if e["win"]]          # read by the checkpoint config, below

    f0 = eps[0]["fields"]
    n_actors, actor_dim = f0["n_actors"], sum(w for _, w in f0["actor"])
    raw_global = eps[0]["raw_dim"] - n_actors * actor_dim
    global_dim = raw_global + COND_DIM          # the conditioning rides in the global block
    n_actions = eps[0]["n_actions"]
    print(f"  obs {eps[0]['obs_dim']} = global {raw_global} + cond {COND_DIM} + {n_actors} x {actor_dim}")

    model = RecurrentEntityActorCritic(global_dim, n_actors, actor_dim, n_actions, hidden=a.hidden, gru=a.gru)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    start_ep = 1
    resume_from = None
    for cand in (Path(str(a.out) + ".last.pt"), Path(a.out)):
        if a.resume and cand.is_file():
            resume_from = cand
            break
    if resume_from:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_ep = int(ck.get("config", {}).get("rvs_epoch", 0)) + 1
        print(f"RESUMED from {resume_from} at epoch {start_ep} "
              f"(val-win acc was {100*float(ck.get('config', {}).get('rvs_val_win_acc', 0)):.1f}%)",
              flush=True)
    elif a.resume:
        print(f"--resume given but no checkpoint at {a.out}[.last.pt]: starting fresh", flush=True)
    elif not a.fresh and (Path(str(a.out) + ".last.pt").is_file() or Path(a.out).is_file()):
        # A DESTRUCTIVE RESTART IS NOW IMPOSSIBLE BY ACCIDENT. Three arms were restarted on 2026-10-16 to
        # change a flag, and each silently began from a random init, discarding every epoch already paid
        # for -- one of them had reached 1.54 online damage. Pass --resume to continue or --fresh to
        # deliberately discard.
        raise SystemExit(f"{a.out} already holds training progress. Pass --resume to continue from it, "
                         f"or --fresh to deliberately start over and overwrite it.")
    from dataclasses import asdict
    from zelda_rl.combat import CombatConfig, GOHMA_SIMPLE

    def save_ckpt(path: Path, ep_i: int, score: float) -> None:
        """One writer for every checkpoint this script produces, so the per-epoch file and the
        'improved' file can never carry different metadata or a different optimiser state."""
        cfg = {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
               "n_actions": n_actions, "obs_dim": eps[0]["obs_dim"], "hidden": a.hidden, "gru": a.gru,
               "recurrent": True, "obs_version": 7, "task": "combat", "profile": "gohma",
               "run_name": "rvs", "episode_seconds": a.episode_seconds, "vi_per_step": 6,
               "goal_slot": 3, "steps_per_rollout": 0,
               "rvs_cond": True, "rvs_target": list(TARGET), "rvs_cond_dim": COND_DIM,
               "rvs_win_weight": solved, "rvs_win_share": a.win_share, "rvs_episodes": len(eps),
               "rvs_wins": len(wins), "rvs_val_win_acc": float(score), "rvs_epoch": ep_i,
               "bc_seed": a.seed}
        cfg["combat_cfg"] = asdict(CombatConfig(profile="gohma", max_seconds=a.episode_seconds,
                                                **GOHMA_SIMPLE))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "update": 0,
                    "global_step": 0, "episode_count": 0, "config": cfg}, path)
        ckpt_s3.upload(path, "rvs", ep_i)

    best, best_dmg, hist = (0.0, -1), (0.0, -1), []
    print(f"\n{'epoch':>6}{'train loss':>12}{'train acc':>11}{'val acc':>9}{'val WIN acc':>13}")
    link = None
    if a.fleet_experiment and a.fleet_world_size > 1:
        try:
            from zelda_rl.fleet_link import FleetLink
            link = FleetLink(a.fleet_experiment, a.fleet_rank, a.fleet_world_size, a.fleet_launch_id)
            print(f"fleet sync: {a.fleet_experiment} rank {a.fleet_rank}/{a.fleet_world_size}, "
                  f"averaging every {a.fleet_sync_every} epoch(s)", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"fleet sync unavailable ({str(e)[:80]}): training alone", flush=True)

    for ep_i in range(start_ep, a.epochs + 1):
        model.train()
        tr_ep = tr
        if a.steps_per_epoch:
            # a fixed number of episodes per epoch, resampled each time, so every episode is still
            # eventually seen but the wall-clock between evals does not grow with the corpus
            take = min(len(tr), a.steps_per_epoch * a.batch_episodes)
            sel = np.random.default_rng(ep_i).permutation(len(tr))[:take]
            tr_ep = [tr[i] for i in sel]
        tl, ta, _ = tbh.run_split(model, tr_ep, a.chunk, opt, a.batch_episodes)
        model.eval()
        with torch.no_grad():
            _, vacc, _ = tbh.run_split(model, va, a.chunk, None, a.batch_episodes)
            vw = [e for e in va if e["win"]]
            _, vwacc, _ = (tbh.run_split(model, vw, a.chunk, None, a.batch_episodes) if vw
                           else (0.0, float("nan"), 0))
        print(f"{ep_i:>6}{tl:>12.4f}{100*ta:>10.1f}%{100*vacc:>8.1f}%{100*vwacc:>12.1f}%")
        # ACCURACY IS AN ANTI-SELECTOR HERE. Measured across 14 arms on 2026-10-16: val WIN accuracy vs
        # pooled online damage r = -0.875 (t=-6.25), val acc r = -0.599, train acc r = -0.626, train loss
        # r = +0.624. The arms that fit their corpus best have the EASIEST corpus -- rvs_M3 reaches 56.7%
        # val-win accuracy and 0.82 damage, rvs_M5 reaches 27.4% and 1.76. Accuracy measures corpus
        # homogeneity, and diversity is what produces damage. So accuracy is kept as a DIAGNOSTIC and the
        # file this script promotes is chosen by ONLINE damage (AGENTS 30: select on a statistic that
        # tracks the goal). Within one arm accuracy does track damage weakly (r=+0.44), which is why it is
        # still printed -- it is a progress signal, not a selector.
        score = vwacc if vwacc == vwacc else vacc        # DIAGNOSTIC only -- see above
        # EVERY epoch, unconditionally, to <out>.last.pt. The only files written before this were
        # "improved" ones, so killing a run between improvements threw away everything since the last one
        # -- and a restart then began from a random init. An interruption must never cost more than one
        # epoch (AGENTS 39).
        save_ckpt(Path(str(a.out) + ".last.pt"), ep_i, score)
        # HEARTBEAT EVERY EPOCH, BEFORE the sync. average_weights finds its peers through
        # live_peers(), which reads heartbeats -- and the heartbeat used to be sent only AFTER a
        # successful sync, so at every sync no rank had ever announced itself, the aggregator saw no live
        # peers, waited ~6 s and averaged ITSELF (sync_ranks=1 on all six ranks, five sixths of the work
        # overwritten). A rank must announce itself before it asks who else is there.
        if link is not None:
            try:
                link.heartbeat({"epoch": ep_i, "val_win_acc": float(score)})
            except Exception:                                    # noqa: BLE001
                pass
        if link is not None and ep_i % a.fleet_sync_every == 0:
            try:
                # KEY THE SYNC ON A ROUND, NOT AN EPOCH. average_weights keys its blobs by the
                # number passed here; with the epoch number, ranks that drift apart (shards differ in
                # size, so one rank is at 36 while another is at 40) NEVER occupy the same key and each
                # one averages alone -- measured: sync_ranks 1 on all six, and the followers loading the
                # aggregator's weights without contributing, i.e. 5/6 of the work discarded. A round
                # absorbs +-sync_every epochs of drift, and the wait is long enough for a slow rank.
                _round = ep_i // max(a.fleet_sync_every, 1)
                sync = link.average_weights(model, _round, aggregator=a.fleet_aggregator,
                                            straggler_wait=120.0, timeout=600.0)
                if sync:
                    print(f"  fleet sync @epoch {ep_i}: {sync}", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"  fleet sync FAILED @epoch {ep_i}: {str(e)[:100]} -- training alone", flush=True)
            try:
                link.heartbeat({"epoch": ep_i, "val_win_acc": float(score)})
            except Exception:                                    # noqa: BLE001
                pass
        do_eval = a.eval_every and (ep_i % a.eval_every == 0 or ep_i == a.epochs)
        if do_eval or score > best[0]:
            if score > best[0]:
                best = (score, ep_i)
            out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
            save_ckpt(out, ep_i, score)
        if do_eval:
            ev = online_eval(Path(a.out), a, ep_i, hist)
            if a.dyna_dir and Path(a.dyna_dir).is_dir():
                # DYNA refresh: the episodes just played are now on disk in corpus format. Reload with the
                # static dirs FIRST (they are the anchor and are never dropped) plus the fresh dir, and
                # re-solve the win multiplier for the new composition.
                n_fresh = len(list(Path(a.dyna_dir).glob("*.npz")))
                print(f"  DYNA: absorbing {n_fresh} of the policy's own episodes from {a.dyna_dir}",
                      flush=True)
                eps, tr, va, solved = prepare(list(a.data) + [a.dyna_dir])
                wins = [e for e in eps if e["win"]]
            if ev is not None and ev > best_dmg[0]:
                best_dmg = (ev, ep_i)
                keep = Path(a.out).with_name(Path(a.out).stem + "_bestdmg.pt")
                shutil.copy(Path(a.out), keep)
                ckpt_s3.upload(keep, "rvs_bestdmg", ep_i)
    print(f"\nbest val WIN accuracy {100*best[0]:.1f}% at epoch {best[1]} (DIAGNOSTIC: r=-0.875 against "
          f"online damage across arms -- not a selector)")
    if best_dmg[1] > 0:
        keep = Path(a.out).with_name(Path(a.out).stem + "_bestdmg.pt")
        print(f"SELECTED by ONLINE boss damage: {best_dmg[0]:.3f} at epoch {best_dmg[1]} -> {keep}")
        print(f"  ({a.out} is the accuracy-selected file and is kept only for comparison)")
    else:
        print(f"no online eval ran, so nothing was selected on damage -> {a.out} is accuracy-selected")
    print(f"  inference prompt is g = {TARGET} (take all her HP, win, take no damage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
