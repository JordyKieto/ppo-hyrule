#!/usr/bin/env python3
"""The continuous Dyna loop: real games -> world model -> dream-PPO -> real eval -> ANCHOR -> repeat.

The architecture is the one in the author's own paper, "Learning in Dreams, Winning in Reality: A Continuous
Dyna Loop for a Ten-Hero MOBA" (arXiv 2610.08033). Four of its findings are load-bearing here and are the
reason this orchestrator exists instead of a shell script:

  1. THE POLICY TRAINS ONLY INSIDE THE MODEL. `scripts/train_dream_ppo.py` never touches the emulator.
  2. THE MODEL IS POOR ON THE POLICY'S OWN GAMES. It is accurate on whatever corpus it was fitted to, so the
     loop must keep feeding it the CURRENT policy's games -- that is the whole point of being continuous.
  3. ANCHORING IS LOAD-BEARING. Unanchored runs collapsed within a few updates and NO IN-DREAM METRIC
     DETECTED IT; only the real-game eval caught it. So every iteration's dream-PPO resumes from the
     best REAL-EVAL policy, not from the newest one, and a regression is rolled back.
  4. DREAM TRAINING ALONE IS WORTH 0%. The real-game eval is not a progress report, it is part of the
     algorithm: it selects.

Phases per iteration, each resumable and each verified before the next starts:

    collect   scripts/collect_transitions.py with the ANCHOR policy   -> datasets/transitions/<run>/iterN
    wm        scripts/train_wm_tokens.py (resume)                     -> runs/<run>/wm.pt
    event     scripts/train_event_head.py                             -> runs/<run>/event.pt
    dream     scripts/train_dream_ppo.py --resume <anchor>            -> runs/<run>/dream_iterN.pt
    eval      scripts/eval_fleet.py --ckpt <dream policy>             -> real boss damage + kills
    anchor    keep the better of (anchor, candidate) BY REAL EVAL

Run:
    .venv/bin/python scripts/dyna_loop.py --run gohma-dyna --anchor checkpoints/kept/rvs_M5_s5.pt --iters 6
    .venv/bin/python scripts/dyna_loop.py --run gohma-dyna --resume      # picks up mid-iteration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_ = sys.executable


def sh(args: list[str], log: Path, env_extra: dict | None = None) -> int:
    """Run a phase, streaming to its own log with -u so a long phase is never silent (AGENTS 24)."""
    import os
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_extra:
        env.update(env_extra)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(args)}\n")
        f.flush()
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        return p.wait()


def read_eval(path: Path) -> dict | None:
    """The real-game eval's verdict: balanced boss damage and the kill rate."""
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    per: dict[int, list[float]] = {}
    kills = 0
    for r in rows:
        per.setdefault(int(r.get("slot", -1)), []).append(float(r.get("boss_damage", 0)))
        kills += int(bool(r.get("boss_defeated")))
    bal = sum(sum(v) / len(v) for v in per.values()) / max(len(per), 1)
    return {"dmg_balanced": bal, "kills": kills, "episodes": len(rows),
            "kill_rate": kills / max(len(rows), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under runs/")
    ap.add_argument("--anchor", default="checkpoints/kept/rvs_M5_s5.pt",
                    help="the policy the loop starts anchored to (the best REAL policy you have)")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes-collect", type=int, default=600, help="real games per iteration")
    ap.add_argument("--workers", type=int, default=10, help="parallel Dolphins for collect/eval")
    ap.add_argument("--wm-epochs", type=int, default=12, help="WM epochs per iteration (it RESUMES)")
    ap.add_argument("--dream-transitions", type=int, default=400_000)
    ap.add_argument("--eval-episodes", type=int, default=120,
                    help="real episodes per candidate. The selector's n: 120 separates a 2x kill-rate "
                         "difference far better than the 16 that AGENTS 30 showed was a lottery")
    ap.add_argument("--slots", default="3,4,5,6,7,8")
    ap.add_argument("--profile", default="gohma", choices=("gohma", "deku_baba"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    run = ROOT / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    state_p = run / "loop_state.json"
    st = json.loads(state_p.read_text()) if (a.resume and state_p.is_file()) else {
        "iter": 0, "anchor": a.anchor, "anchor_score": None, "history": []}
    if not a.resume and state_p.is_file():
        raise SystemExit(f"{state_p} exists. Pass --resume to continue this loop, or use a new --run. "
                         f"(AGENTS 39: a restart must never silently discard progress.)")

    data_dir = ROOT / "datasets" / "transitions" / a.run
    wm = run / "wm.pt"
    ev_head = run / "event.pt"
    snaps = run / "snaps"
    snaps.mkdir(exist_ok=True)

    # THE ONLINE HALF, RUNNING CONTINUOUSLY. The paper's eval is part of the algorithm, and a batch eval at
    # the end of an iteration tells you nothing for the twenty minutes the dream is training. eval_fleet
    # --watch evaluates every new .pt the moment it appears, and train_dream_ppo --snapshot-every drops one
    # every few updates, so the policy is measured in the REAL game the whole time it is learning in the
    # dream -- behaviour, damage and kills per snapshot, into one jsonl that scripts/dyna_watch.py reads as
    # a damage slope with its SE.
    online_log = run / "online.jsonl"
    online = subprocess.Popen(
        [PY_, "-u", "scripts/eval_fleet.py", "--watch", str(snaps), "--suite", "full",
         "--episodes-per-slot", "4", "--episode-seconds", "120", "--num-envs", str(a.workers),
         "--slots-dir", "assets/states", "--video-backend", "Null",
         "--per-episode-out", str(online_log), "--out", str(run / "online.json"),
         "--fleet-world-size", "1", "--run-name", f"{a.run}_online",
         "--wandb-project", "zelda-rl-dyna"],
        stdout=(run / "online.log").open("a"), stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"online evaluator pid {online.pid}: watching {snaps} -> {online_log}", flush=True)
    try:

        while st["iter"] < a.iters:
            i = st["iter"]
            tag = f"iter{i:02d}"
            print(f"\n########## {tag}  anchor={st['anchor']}  score={st['anchor_score']}", flush=True)
            rec: dict = {"iter": i, "anchor_in": st["anchor"], "t0": time.time()}

            # 1. REAL GAMES with the anchor policy -- finding 2: the model needs the current policy's games
            shard = data_dir / tag
            if not (shard.exists() and any(shard.rglob("*.npz"))):
                rc = sh([PY_, "-u", "scripts/collect_transitions.py",
                         "--transitions", str(a.episodes_collect * 1200), "--workers", str(a.workers),
                         "--mix", json.dumps([[st["anchor"], 0, a.episodes_collect]]),
                         "--profile", a.profile,
                         "--slots", a.slots, "--slots-dir", "assets/states", "--out", str(shard)],
                        run / f"{tag}_collect.log")
                if rc != 0:
                    raise SystemExit(f"{tag}: collect failed rc={rc}; see {run / f'{tag}_collect.log'}")
            n_npz = len(list(shard.rglob("*.npz")))
            rec["collected_files"] = n_npz
            if n_npz == 0:
                raise SystemExit(f"{tag}: collect produced no .npz -- refusing to train a WM on nothing")

            # 2. WORLD MODEL, resumed so each iteration improves the same model rather than restarting it
            # THE PREP BUNDLE carries the per-field scales, the nav-grid bounds and the train/val SLOT
            # split. configs/wm_prep_v4.json is Deku Baba's (63 actions, that room's grid) and would be
            # wrong for Gohma, so iteration 0 FITS one from our own data with --save-prep and every later
            # iteration reuses it -- otherwise the normalisation drifts as the corpus grows and the
            # metrics stop being comparable across iterations (AGENTS 25, same lesson as the packer).
            prep = run / "prep.json"
            prep_args = (["--prep", str(prep)] if prep.is_file() else ["--save-prep", str(prep)])
            rc = sh([PY_, "-u", "scripts/train_wm_tokens.py", "--data", str(data_dir),
                     "--epochs", str(a.wm_epochs), "--device", "cuda", "--amp", *prep_args,
                     *(["--resume", str(wm)] if wm.is_file() else []),
                     "--out", str(wm)], run / f"{tag}_wm.log")
            if rc != 0 or not wm.is_file():
                raise SystemExit(f"{tag}: WM training failed rc={rc}; see {run / f'{tag}_wm.log'}")
            if not prep.is_file():
                raise SystemExit(f"{tag}: no prep bundle at {prep} -- the WM fitted no scales/split, so "
                                 f"later iterations could not be compared with this one")

            # 3. EVENT HEAD -- the dream's reward. Not decoded HP (see train_dream_ppo's docstring).
            rc = sh([PY_, "-u", "scripts/train_event_head.py", "--data", str(data_dir),
                     "--model", str(wm), "--out", str(ev_head), "--device", "cuda"],
                    run / f"{tag}_event.log")
            if rc != 0 or not ev_head.is_file():
                raise SystemExit(f"{tag}: event head failed rc={rc}; see {run / f'{tag}_event.log'}")

            # 4. DREAM-PPO from the ANCHOR (finding 3), never from the newest policy
            # snapshots are written as <out>.updNNNNN.pt beside --out, so --out goes in the WATCHED
            # dir and the online evaluator picks up every snapshot as it lands. No new flag needed.
            cand = snaps / f"dream_{tag}.pt"
            rc = sh([PY_, "-u", "scripts/train_dream_ppo.py", "--model", str(wm),
                     "--event-head", str(ev_head), "--data", str(data_dir),
                     "--transitions", str(a.dream_transitions), "--resume", st["anchor"],
                     "--seed-mode", "frames", "--out", str(cand),
                 "--snapshot-every", "5", "--snapshot-dir", str(snaps),
                     "--run-name", f"{a.run}_{tag}", "--wandb-project", "zelda-rl-dyna"],
                    run / f"{tag}_dream.log")
            if rc != 0 or not cand.is_file():
                raise SystemExit(f"{tag}: dream-PPO failed rc={rc}; see {run / f'{tag}_dream.log'}")

            # 5. REAL EVAL -- finding 4: this is part of the algorithm, not a progress report
            per_ep = run / f"{tag}_eval.jsonl"
            rc = sh([PY_, "-u", "scripts/eval_fleet.py", "--ckpt", str(cand), "--suite", "full",
                     "--episodes-per-slot", str(max(1, a.eval_episodes // 6)), "--episode-seconds", "120",
                     "--num-envs", str(a.workers), "--slots-dir", "assets/states",
                     "--video-backend", "Null", "--per-episode-out", str(per_ep),
                     "--out", str(run / f"{tag}_eval.json"), "--fleet-world-size", "1",
                     "--run-name", f"{a.run}_{tag}_eval",
                 "--wandb-project", "zelda-rl-dyna", "--wandb-group", f"{a.run}_eval"], run / f"{tag}_evalreal.log")
            cand_score = read_eval(per_ep)
            if rc != 0 or cand_score is None:
                raise SystemExit(f"{tag}: real eval failed rc={rc}; see {run / f'{tag}_evalreal.log'}")
            rec["candidate"] = {"path": str(cand), **cand_score}

            # 6. ANCHOR: keep the better policy BY REAL EVAL. A regression is rolled back, because the paper's
            #    unanchored runs collapsed with no in-dream signal that anything was wrong.
            base = st.get("anchor_score")
            if base is None or cand_score["dmg_balanced"] > base["dmg_balanced"]:
                print(f"{tag}: ANCHOR MOVES to the candidate "
                      f"({cand_score['dmg_balanced']:.2f} vs {base and base['dmg_balanced']})", flush=True)
                st["anchor"], st["anchor_score"] = str(cand), cand_score
                rec["decision"] = "accepted"
            else:
                print(f"{tag}: REJECTED, anchor holds "
                      f"({cand_score['dmg_balanced']:.2f} <= {base['dmg_balanced']:.2f})", flush=True)
                rec["decision"] = "rejected"

            rec["seconds"] = time.time() - rec["t0"]
            st["history"].append(rec)
            st["iter"] = i + 1
            state_p.write_text(json.dumps(st, indent=2))
            print(f"{tag} done in {rec['seconds'] / 60:.1f} min; state -> {state_p}", flush=True)

    finally:
        online.terminate()
        try:
            online.wait(timeout=30)
        except Exception:                                        # noqa: BLE001
            online.kill()
        print(f"online evaluator stopped; its rows are in {online_log}", flush=True)

    print("\n=== LOOP HISTORY ===")
    for r in st["history"]:
        c = r.get("candidate", {})
        print(f"  iter {r['iter']:>2}  {r.get('decision','?'):>8}  dmg {c.get('dmg_balanced', float('nan')):.2f}"
              f"  kills {c.get('kills','?')}/{c.get('episodes','?')}  {r['seconds'] / 60:.0f} min")
    print(f"final anchor: {st['anchor']}  {st['anchor_score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
