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
# Gohma HP at each start, decoded from the savestates (PROVENANCE.md 3.2). The eval reports by TIER as well as
# by slot, because "defeat rate" over a mixed suite hides which difficulty actually moved -- and the full-HP
# tier is the task, the rest are rungs.
SLOT_HP = {1: 5, 2: 5, 3: 10, 4: 8, 5: 10, 6: 6, 7: 5, 8: 10, 9: 5, 10: 5}
# Imported, not copied: the horizontal distances live in zelda_rl/combat.py, measured by
# scripts/audit_gohma_slots.py. Four separate copies of this table existed and the one here was 3D,
# which put slot 2 in the wrong band (AGENTS 3).
from zelda_rl.combat import GOHMA_SLOT_DIST as SLOT_DIST  # noqa: E402
TIERS = {"full_hp_10": [3, 5, 8], "mid_hp_6_8": [4, 6], "half_hp_5": [1, 2, 7, 9, 10]}

# The engagement DISTANCE bands. v13 established distance as the real wall (PROVENANCE 3.2, claim R9), and
# under v13's curriculum the held-out slots [1,2,9,10] are ALL 5 HP and ALL 368-490 units out -- perfectly
# collinear with "far". So a held-out rate on this curriculum cannot separate generalisation from the
# distance wall, and reading it as generalisation is exactly the free-signal error of AGENTS 1. Every row
# carries the band breakdown and the collinearity warning so that confound cannot be read past.
BANDS = {"near_lt200": lambda d: d < 200, "mid_200_400": lambda d: 200 <= d <= 400, "far_gt400": lambda d: d > 400}

# Measured mean episode length per slot, over 519 evaluation episodes on disk. Slots differ by 6x: slot 8
# runs to the 300 s cap 75% of the time (251 s mean) while slot 9 dies in 41 s. Sharding by `slots[rank::N]`
# put slots 6 (210 s) and 8 (251 s) on the SAME pod, so one suite took 7.4 min there and 3.0 min on the
# other -- and slots 4/8/10, three of the four starts beyond 400 units, were all on the slow pod. The far
# band is the band the approach-reward result depends on, so the imbalance was throttling exactly the
# measurement being waited on.
SLOT_SECS = {8: 251.0, 6: 210.0, 4: 81.0, 10: 76.0, 3: 70.0, 5: 60.0, 7: 55.0, 2: 51.0, 1: 42.0, 9: 41.0}


def shard_slots(slots, rank, world):
    """Split slots across eval ranks by measured COST, not by count (longest-processing-time first).

    Greedy LPT: hand each slot, heaviest first, to whichever rank has the least work so far. On the ten
    slots this balances 479 s against 458 s (1.05x) where round-robin gave 2.5x.
    """
    if world <= 1:
        return list(slots)
    bins = [[] for _ in range(world)]
    load = [0.0] * world
    for sl in sorted(slots, key=lambda x: -SLOT_SECS.get(x, 100.0)):
        i = min(range(world), key=lambda j: load[j])
        bins[i].append(sl)
        load[i] += SLOT_SECS.get(sl, 100.0)
    return sorted(bins[rank])


def held_out(cfg) -> list[int]:
    """Slots this checkpoint's own run did NOT train on."""
    trained = cfg.get("start_slots") or cfg.get("usable_slots") or []
    if isinstance(trained, str):
        trained = [int(x) for x in trained.split(",") if x.strip()]
    return [s for s in ALL_SLOTS if s not in set(trained)], list(trained)


@torch.no_grad()
def evaluate(ckpt: Path, slots, episodes, envs_n, slots_dir, seconds, deterministic=False):
    """Run `episodes` episodes over `slots`, exactly episodes/len(slots) on each, on `envs_n` emulators.

    Work-QUEUE, not rounds. Two bugs lived in the round version and neither was visible in the output. A
    round fixed one slot for every env, so `--episodes-per-slot 8` on 12 envs actually ran 12 episodes on
    slot 1, 12 on slot 3, 12 on slot 5 and 4 on slot 7 -- the per-slot table was real but its counts were
    not the requested ones, and two slots were never evaluated. And a round ended only when the SLOWEST env
    finished, so one episode running to the 300 s cap idled eleven emulators. Here every env carries its own
    slot and pulls the next the instant it terminates: coverage is exact and the only idle time is the tail.
    """
    model, cfg, _ = load_policy(ckpt)
    rec = bool(cfg.get("recurrent"))
    if seconds:
        cfg = {**cfg, "episode_seconds": seconds}
    envs_n = max(1, min(envs_n, episodes))
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"evalf_{os.getpid()}"
    mk = lambda i: make_env(cfg, root / f"i{i}", slots=slots, slots_dir=slots_dir, video_backend="Null",
                            unlimited_speed=True, idx=i)[0]                            # noqa: E731
    from concurrent.futures import ThreadPoolExecutor

    def build_all():
        """Build every env, and on ANY failure close the ones that DID come up before re-raising.

        Construction sat outside the try/finally, so one Dolphin failing to expose its RAM ("N64 RAM not
        found in pid N after 60s") orphaned the other eleven: they stayed alive, owned by nobody, and the
        next checkpoint started twelve more. Observed live -- 12 orphans, all the same age, on a pod whose
        evaluator was idle and polling.
        """
        made = {}
        err = None
        with ThreadPoolExecutor(envs_n) as pool:
            futs = {pool.submit(mk, i): i for i in range(envs_n)}
            for fu, i in futs.items():
                try:
                    made[i] = fu.result()
                except Exception as e:                                          # noqa: BLE001
                    err = err or e
        if err is not None:
            for e in made.values():
                try:
                    e.close()
                except Exception:                                               # noqa: BLE001
                    pass
            raise err
        return [made[i] for i in range(envs_n)]

    # A Dolphin that has not finished booting is a transient error, and the documented response to a
    # transient error here is to re-run it, not to redesign around it (AGENTS 14). One retry, then give up
    # and let the watcher move to the next checkpoint.
    for attempt in (1, 2):
        try:
            envs = build_all()
            break
        except Exception as e:                                                  # noqa: BLE001
            print(f"  env build failed, attempt {attempt}/2: {str(e)[:100]}", flush=True)
            if attempt == 2:
                raise
            time.sleep(15)

    # Interleaved, so partial results already cover every slot, and exact because k % len(slots) visits each
    # slot equally often when episodes is a multiple of len(slots) -- which the caller guarantees by
    # computing episodes = episodes_per_slot * len(slots).
    queue = [int(slots[k % len(slots)]) for k in range(episodes)]
    cur = [queue.pop(0) for _ in range(envs_n)]
    h = model.init_hidden(envs_n) if rec else None
    first = torch.ones(envs_n)
    obs = np.stack([e.reset(s) for e, s in zip(envs, cur)])
    out, stalls, t0, nxt = [], 0, time.time(), time.time() + 60.0

    def start(i):
        """Hand env i the next slot off the queue, or retire it when the queue is empty."""
        cur[i] = queue.pop(0) if queue else None
        if cur[i] is not None:
            obs[i] = envs[i].reset(cur[i])
            first[i] = 1.0

    try:
        while any(c is not None for c in cur):
            live = [i for i in range(envs_n) if cur[i] is not None]
            x = torch.as_tensor(obs, dtype=torch.float32)
            if rec:
                a, _, _, _, h = model.act(x, h, first, deterministic=deterministic)
            else:
                a, _, _, _ = model.act(x, deterministic=deterministic)
            first = torch.zeros(envs_n)
            try:
                for i in live:
                    envs[i].send_action(int(a[i]))
                for i in live:
                    while not envs[i].ready():
                        time.sleep(0.0005)
                for i in live:
                    o, r, term, trunc, info = envs[i].observe()
                    obs[i] = o
                    if term or trunc:
                        out.append(dict(info["episode"], slot=cur[i]))
                        start(i)                      # straight into the next episode, no round barrier
            except EnvError as err:
                # A stalled Dolphin (10 s Null-backend timeout) now costs exactly the one episode that env
                # was running: rebuild it and give it fresh work instead of dropping every env's progress.
                stalls += 1
                i = err.idx if 0 <= err.idx < envs_n else live[0]
                print(f"  stall on env{i} ({stalls}) -> rebuilt, lost 1 episode", flush=True)
                try:
                    envs[i].close()
                except Exception:
                    pass
                envs[i] = mk(i)
                start(i)
                if stalls > 3 + episodes // 4:
                    print("  too many stalls; reporting what completed", flush=True)
                    break
            if time.time() > nxt:                     # AGENTS 24: no silent minutes
                d = sum(1 for e in out if e.get("defeat"))
                print(f"  {len(out)}/{episodes} done, {len(queue)} queued, {d} defeats, "
                      f"{time.time()-t0:.0f}s", flush=True)
                nxt = time.time() + 60.0
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
    return out, cfg

def score(eps):
    n = max(len(eps), 1)
    def agg(sel):
        m = max(len(sel), 1)
        return {"n": len(sel),
                "defeat_rate": sum(1 for e in sel if e.get("boss_defeated")) / m,
                "death_rate": sum(1 for e in sel if e.get("died")) / m,
                "timeout_rate": sum(1 for e in sel if e.get("end") == "timeout") / m,
                "boss_hp_taken": sum(e.get("boss_damage", 0) for e in sel) / m,
                "return": sum(e.get("return", 0) for e in sel) / m,
                "length_s": sum(e.get("length", 0) for e in sel) / m / 10.0}
    out = dict(agg(eps), episodes=len(eps))
    out["by_slot"] = {str(sl): dict(agg([e for e in eps if e["slot"] == sl]), hp=SLOT_HP.get(sl),
                                    dist=SLOT_DIST.get(sl))
                      for sl in sorted({e["slot"] for e in eps})}
    out["by_tier"] = {t: agg([e for e in eps if e["slot"] in sl]) for t, sl in TIERS.items()
                      if any(e["slot"] in sl for e in eps)}
    out["by_distance"] = {b: agg([e for e in eps if f(SLOT_DIST.get(e["slot"], -1))])
                          for b, f in BANDS.items() if any(f(SLOT_DIST.get(e["slot"], -1)) for e in eps)}
    # THE HEADLINE. A full-HP kill is the task; everything else is a rung on the way to it, and a defeat rate
    # pooled over a mixed suite can rise while this stays at zero -- which is exactly what v11->v13 did.
    full = [e for e in eps if e.get("boss_hp_start", 0) >= 10]
    out["full_hp_episodes"] = len(full)
    out["full_hp_defeats"] = sum(1 for e in full if e.get("boss_defeated"))
    out["full_hp_defeat_rate"] = (out["full_hp_defeats"] / len(full)) if full else float("nan")
    return out


def run_one(ckpt, a):
    # `update` lives at the TOP level of the checkpoint, not inside ck["config"], so every row written so far
    # has `"update": null`. The published_update field carried the real number; this makes the row agree.
    _, model_cfg, _ck = load_policy(ckpt)
    ckpt_update = _ck.get("update")
    ho, trained = held_out(model_cfg)
    if a.slots:
        slots = [int(x) for x in a.slots.split(",")]
        label = "explicit"
    elif a.suite == "holdout":
        slots, label = (ho or ALL_SLOTS), ("held-out only" if ho else "NOTHING held out, falling back to all")
    else:
        slots, label = ALL_SLOTS, "FULL suite, all 10 starts"
    # Shard the SLOTS across eval ranks, not the checkpoints. Sharding checkpoints makes each pod wait a full
    # suite before the next policy is scored; sharding slots makes every pod work on the SAME policy and cuts
    # the latency per checkpoint, which is what decides whether the signal arrives before the next publish.
    mine = shard_slots(slots, a.fleet_rank, a.fleet_world_size)
    if not mine:
        mine = slots[:1]
    eps_total = a.episodes_per_slot * len(mine)
    print(f"  run {model_cfg.get('run_name')} (group {model_cfg.get('wandb_group')}), its update "
          f"{ckpt_update}")
    print(f"  trained on {trained or 'unknown'} | suite: {label}")
    print(f"  rank {a.fleet_rank}/{a.fleet_world_size} evaluating slots {mine} "
          f"x {a.episodes_per_slot} = {eps_total} episodes")
    eps, _ = evaluate(ckpt, mine, eps_total, a.envs, a.slots_dir, a.episode_seconds, a.deterministic)
    s = score(eps)
    hos = [x for x in mine if x not in set(trained)]
    # The HELD-OUT-ONLY sub-score. Selection used to read `s["defeat_rate"]` while printing "on held-out
    # slots", which under --suite full is the rate pooled over all ten starts, TRAINED ONES INCLUDED. The
    # label was wrong, not the arithmetic, and a mislabelled selection metric silently chooses the wrong
    # checkpoint. `s["holdout"]` is now computed from the held-out episodes alone and is what `best` reads.
    s["holdout"] = score([e for e in eps if e["slot"] in set(hos)]) if hos else None
    # WHICH RUN produced this checkpoint. v15 resumes with --init-from, so its `update` counter restarts at
    # 0 while the published filename stays gohma-kl_r{rank}_u{update}.pt -- v14 u00005 and v15 u00005 are
    # different policies with the same name, and pooling them in one jsonl would be unrecoverable later.
    # The checkpoint carries run_name and wandb_group, so the row records them and is self-describing.
    s.update(run_name=model_cfg.get("run_name"), wandb_group=model_cfg.get("wandb_group"))
    s.update(checkpoint=str(ckpt), update=ckpt_update, slots=mine, trained_slots=trained,
             suite=a.suite, rank=a.fleet_rank, world_size=a.fleet_world_size,
             held_out_slots=hos, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    fh = s["full_hp_defeat_rate"]
    print(f"  ALL   defeat {s['defeat_rate']*100:5.2f}%  death {s['death_rate']*100:5.1f}%  "
          f"timeout {s['timeout_rate']*100:5.1f}%  bossHP {s['boss_hp_taken']:4.2f}  "
          f"len {s['length_s']:3.0f}s  ({s['episodes']} eps)")
    print(f"  FULL-HP (the task): {s['full_hp_defeats']}/{s['full_hp_episodes']}"
          + (f" = {fh*100:.2f}%" if fh == fh else " -- none in this slice"))
    for t, v in s["by_tier"].items():
        print(f"    {t:<11} n={v['n']:<3} defeat {v['defeat_rate']*100:5.2f}%  death {v['death_rate']*100:5.1f}%"
              f"  bossHP {v['boss_hp_taken']:4.2f}")
    for b, v in s["by_distance"].items():
        print(f"    {b:<11} n={v['n']:<3} defeat {v['defeat_rate']*100:5.2f}%  death {v['death_rate']*100:5.1f}%"
              f"  bossHP {v['boss_hp_taken']:4.2f}")
    # The held-out rate is only a generalisation number if the held-out slots span the same difficulty as the
    # trained ones. On v13's ladder they do not -- [1,2,9,10] are all 5 HP and all >=368 units -- so the
    # warning fires rather than letting the number be read as generalisation (AGENTS 1, free-signal control).
    hos = [x for x in mine if x not in set(trained)]
    if hos and trained:
        hd = [SLOT_DIST[x] for x in hos if x in SLOT_DIST]
        td = [SLOT_DIST[x] for x in mine if x in set(trained) and x in SLOT_DIST]
        hh, th = {SLOT_HP.get(x) for x in hos}, {SLOT_HP.get(x) for x in mine if x in set(trained)}
        if hd and td and (min(hd) > min(td) + 100 or not (hh & th)):
            print(f"  !! CONFOUNDED: held-out slots {hos} are {min(hd)}-{max(hd)}u / {sorted(hh)} HP against "
                  f"trained {min(td)}-{max(td)}u / {sorted(th)} HP. This held-out rate measures DISTANCE and "
                  f"HP as much as generalisation; do not cite it as a generalisation gap.")
            s["holdout_confounded"] = True
    for sl, v in s["by_slot"].items():
        print(f"      slot {sl:>2} ({v['hp']:>2} HP, {v['dist']:>3}u)  {v['defeat_rate']*100:5.1f}%  "
              f"death {v['death_rate']*100:5.0f}%  n={v['n']}")
    return s




def wb_payload(s):
    """Flatten a row for W&B. The held-out sub-score is a nested dict, and the previous comprehension kept
    only top-level scalars -- so the one metric the eval fleet exists to produce never reached W&B at all."""
    out = {f"eval/{k}": v for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k, v in (s.get("holdout") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"holdout/{k}"] = v
    for grp in ("by_tier", "by_distance"):
        for name, d in (s.get(grp) or {}).items():
            out[f"{grp}/{name}/defeat_rate"] = d["defeat_rate"]
            out[f"{grp}/{name}/boss_hp_taken"] = d["boss_hp_taken"]
    return out

def sel_key(s):
    """What `best` is ranked on: the HELD-OUT defeat rate, with held-out boss HP taken as the tiebreak.

    The user's criterion is "keep the policy that is best on unseen starts". Taken literally it has no
    resolution here: every v14 checkpoint scores 0/32 on the held-out slots, so a strict `>` never fires and
    the first checkpoint seen stays `best` forever -- the selector was choosing by arrival order while
    printing a percentage. Boss HP taken is the same quantity measured with resolution (it ranges 1.0-4.7
    across these rows), so it breaks the ties without changing the criterion. It is a TIEBREAK only: a
    checkpoint that defeats her more often on unseen starts always wins.
    """
    h = s.get("holdout") or s
    return (h["defeat_rate"], h["boss_hp_taken"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--watch", default=None, help="a checkpoints/ directory; evaluate every new .pt forever")
    p.add_argument("--watch-published", default=None, metavar="EXPERIMENT",
                   help="watch the FLEET's published policies for EXPERIMENT instead of a local directory. "
                        "Checkpoints live on the training pod's own disk, so an evaluator on a different pod "
                        "cannot see them; train_combat.py already publishes every --publish-every updates "
                        "through zelda_rl.fleet_link, and this pulls from there. This is the mode to use on an "
                        "eval fleet.")
    p.add_argument("--slots", default=None, help="default: whatever the checkpoint did NOT train on")
    p.add_argument("--suite", choices=["full", "holdout"], default="full",
                   help="`full` evaluates ALL 10 starts -- the holistic view, and the only way to see full-HP "
                        "games, which the 4 unseen slots cannot show (none of them start Gohma at 10 HP). "
                        "`holdout` restricts to the slots this checkpoint did not train on.")
    p.add_argument("--episodes-per-slot", type=int, default=8,
                   help="episodes per slot, so coverage is explicit instead of a total that one slot can eat")
    p.add_argument("--episodes", type=int, default=None, help="(legacy) total episodes; --episodes-per-slot wins")
    p.add_argument("--envs", type=int, default=12,
                   help="12 on a 16-vCPU pod: ~2 threads per Dolphin and 12 envs saturated a 16-vCPU pod in "
                        "the measurements in docs/RUNPOD.md. Measured episode length is ~109s (only 6.3% hit "
                        "the 300s cap), so 2 pods x 12 envs cover all 10 slots in ~6 min -- faster than the "
                        "~15 min between published policies, which is what makes the signal usable.")
    p.add_argument("--episode-seconds", type=float, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    p.add_argument("--out", default=None, help="append each result as JSON to this file")
    p.add_argument("--poll", type=float, default=60.0)
    # ---- flags `fleet launch` passes to every script. Accepting them is what makes this launchable the same
    # way training is, instead of needing a hand-written wrapper on the pod (which is not reproducible and was
    # how I first ran it).
    p.add_argument("--num-envs", type=int, default=None, help="alias for --envs; fleet launch passes this")
    p.add_argument("--run-name", default=None, help="names the output file when --out is not given")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fleet-rank", type=int, default=0)
    p.add_argument("--fleet-world-size", type=int, default=1)
    p.add_argument("--wandb-project", default=None, help="log the HELD-OUT metrics beside the training curve")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--video-backend", default="Null", help="accepted for fleet compatibility; always Null here")
    p.add_argument("--fleet-experiment", default=None)
    p.add_argument("--fleet-launch-id", default=None)
    a = p.parse_args()
    if a.num_envs:
        a.envs = a.num_envs
    if a.episodes:                      # legacy --episodes: spread it over the suite
        a.episodes_per_slot = max(1, a.episodes // 10)
    if a.out is None and a.run_name:
        a.out = f"results/eval/{a.run_name}.jsonl"
    if a.watch is None and a.run_name:
        # fleet launch gives every rank the same command, so derive the directory to watch from the run name:
        # `gohma-kl-v14-r3` -> watch rank 0's checkpoints, since FedAvg keeps every rank on the same weights.
        base = a.run_name.rsplit("-r", 1)[0]
        a.watch = f"runs/{base}-r0/checkpoints"
        print(f"derived --watch {a.watch} from --run-name {a.run_name}")

    def emit(s):
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(s) + "\n")

    if a.ckpt:
        emit(run_one(Path(a.ckpt), a))
        return

    if not a.watch and not a.watch_published:
        raise SystemExit("pass --ckpt, --watch or --watch-published")
    d = Path(a.watch) if a.watch else None
    seen, best = set(), None
    wb = None
    if a.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, group=a.wandb_group,
                            name=f"eval-{a.run_name or 'holdout'}", job_type="eval",
                            config={"slots": a.slots, "episodes": a.episodes, "envs": a.envs,
                                    "watch": str(d), "rank": a.fleet_rank})
            print(f"wandb: {wb.url}")
        except Exception as e:
            print(f"wandb init failed ({str(e)[:60]}); continuing without it")
    print(f"watching {d} every {a.poll:.0f}s"
          + (f" | rank {a.fleet_rank}/{a.fleet_world_size}, taking every {a.fleet_world_size}th checkpoint"
             if a.fleet_world_size > 1 else ""))
    inbox = ROOT / "checkpoints" / "eval_inbox"
    while True:
        if a.watch_published:
            from zelda_rl.fleet_link import fetch_latest_policy
            try:
                got = fetch_latest_policy(a.watch_published, inbox)
            except Exception as e:
                print(f"  fetch failed ({str(e)[:70]}); retrying in {a.poll:.0f}s")
                got = None
            if got:
                f, ptr = Path(got[0]), got[1]
                if f.name not in seen:
                    seen.add(f.name)
                    print(f"\npublished update {ptr['update']} (rank {ptr['rank']}) -> {f.name}")
                    try:
                        s_ = run_one(f, a)
                        s_["published_update"] = ptr["update"]
                        emit(s_)
                        if wb is not None:
                            wb.log(wb_payload(s_), step=ptr["update"])
                        if best is None or sel_key(s_) > sel_key(best):
                            best = s_
                            hh = s_.get("holdout") or s_
                            print(f"  >>> NEW BEST on held-out slots {s_['held_out_slots']}: "
                                  f"{hh['defeat_rate']*100:.2f}% defeat, bossHP {hh['boss_hp_taken']:.2f} "
                                  f"(published update {ptr['update']}) -- the KL anchor candidate")
                            if a.out:
                                Path(a.out).with_suffix(".best.json").write_text(json.dumps(best, indent=2))
                    except Exception as e:
                        print(f"  eval failed: {str(e)[:120]}")
            time.sleep(a.poll)
            continue
        for n_ck, f in enumerate(sorted(d.glob("update_*.pt"))):
            if f.name in seen:
                continue
            seen.add(f.name)
            # Shard across eval ranks: each pod takes every world_size-th checkpoint, so N pods evaluate N
            # times as many without duplicating work. Every rank still sees every file, so nothing is missed
            # if one pod dies -- it just stops being evaluated twice as fast.
            if a.fleet_world_size > 1 and (n_ck % a.fleet_world_size) != a.fleet_rank:
                continue
            print(f"\n{f.name}")
            try:
                s = run_one(f, a)
            except Exception as e:
                print(f"  eval failed: {str(e)[:120]}")
                continue
            emit(s)
            if wb is not None:
                upd = s.get("update") or 0
                wb.log(wb_payload(s), step=upd)
            if best is None or sel_key(s) > sel_key(best):
                best = s
                # THE POINT: `best` is by HELD-OUT defeat rate, not by the last checkpoint written.
                hh = s.get("holdout") or s
                print(f"  >>> NEW BEST on held-out slots {s['held_out_slots']}: "
                      f"{hh['defeat_rate']*100:.2f}% defeat, bossHP {hh['boss_hp_taken']:.2f} "
                      f"({f.name}) -- this is the KL anchor candidate")
                if a.out:
                    Path(a.out).with_suffix(".best.json").write_text(json.dumps(best, indent=2))
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
