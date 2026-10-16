#!/usr/bin/env python3
"""One board for the whole Dyna loop, separating WHERE any improvement came from.

Four questions, four sections, because a single number cannot tell them apart:

  1. IS THE WORLD MODEL GETTING BETTER?  From pure transition training on the corpus. The honest measure is
     not its loss -- it is H20 skill against PERSISTENCE and the SHUFFLED-ACTION penalty. A model that does
     not beat persistence has learned nothing, and one that does not get worse under shuffled actions has
     not learned the action (docs/WM_MOTION.md).
  2. IS THE POLICY GETTING BETTER IN THE DREAM?  dream-PPO's own return and event rates, inside the model.
     The paper warns this can rise while the real policy collapses, which is why it is never read alone.
  3. IS THE POLICY GETTING BETTER IN REAL LIFE?  The online evaluator's real episodes per snapshot: balanced
     boss damage, kill rate, and the behaviour mix. This is the only one that counts as a result.
  4. WHAT IS THE LOOP DOING?  Per iteration: accepted or rolled back, and the anchor's score history --
     i.e. whether real games are improving the model and the policy, or whether the loop is treading water.

    python3 scripts/dyna_board.py --run gohma-dyna --wm-log logs/wm_obs.log          # on the pod
    .venv/bin/python scripts/dyna_board.py --remote gohma-wm --run gohma-dyna        # from the Mac
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WM_EPOCH = re.compile(r"^\s*(\d+)\s+(\d)\s+([\d.]+)\s+([\d.]+)\s")
WM_VERDICT = re.compile(r"H20 skill vs persistence ([-+\d.]+) \| shuffled-action penalty x([\d.]+)")
DREAM = re.compile(r"upd\s+(\d+).*?ret\s+([-\d.]+)")


def remote(exp: str, cmd: str) -> str:
    r = subprocess.run([sys.executable, "-m", "fleet", "ssh", "--experiment", exp, "--rank", "0", "--", cmd],
                       cwd=ROOT, capture_output=True, text=True, timeout=400)
    return (r.stdout or "") + (r.stderr or "")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def wm_section(text: str) -> None:
    section("1. WORLD MODEL -- pure transition training on the corpus")
    # ONLY THE CURRENT RUN. Every launch appends to the same log, so the newest verdict line can belong to
    # a run that is already dead: a killed ungated run's "-3.653" was reported as the live gated run's
    # number. "device cuda" is printed once per launch, so everything before the LAST one is history.
    marks = [i for i, l in enumerate(text.splitlines()) if l.startswith("device ")]
    if marks:
        text = "\n".join(text.splitlines()[marks[-1]:])
        print(f"   (reading the current run only -- {len(marks) - 1} earlier run(s) in this log ignored)")
    eps = [m.groups() for m in (WM_EPOCH.match(l) for l in text.splitlines()) if m]
    vs = [m.groups() for m in (WM_VERDICT.search(l) for l in text.splitlines()) if m]
    corp = [l.strip() for l in text.splitlines() if l.startswith("loaded ") or l.startswith("corpus:")]
    for c in corp[-2:]:
        print("   " + c[:110])
    if not eps:
        print("   no epoch yet")
        return
    print(f"   {'epoch':>6}{'stage':>6}{'train':>10}{'val':>10}")
    for e in eps[-4:]:
        print(f"   {e[0]:>6}{e[1]:>6}{float(e[2]):>10.4f}{float(e[3]):>10.4f}")
    if vs:
        sk, ab = float(vs[-1][0]), float(vs[-1][1])
        verdict = ("LEARNED SOMETHING" if sk > 0.05 else "NOT beating persistence")
        act = ("ACTION-CONDITIONED" if ab > 1.1 else "action is DECORATION")
        print(f"   H20 skill vs persistence {sk:+.3f} -> {verdict}")
        print(f"   shuffled-action penalty  x{ab:.2f}      -> {act}")
        if len(vs) >= 3:
            s = np.array([float(v[0]) for v in vs], float)
            x = np.arange(len(s), dtype=float)
            A = np.vstack([x, np.ones_like(x)]).T
            b, *_ = np.linalg.lstsq(A, s, rcond=None)
            r = s - A @ b
            se = float(np.sqrt((r @ r / max(len(x) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
            t = b[0] / se if se > 0 else float("nan")
            print(f"   skill trend {b[0]:+.4f} +- {se:.4f}/epoch t={t:+.2f} "
                  f"{'IMPROVING' if t > 2 else 'FALLING' if t < -2 else 'flat'}")


def dream_section(text: str) -> None:
    section("2. POLICY IN THE DREAM -- dream-PPO inside the model (never read alone)")
    rows = [m.groups() for m in (DREAM.search(l) for l in text.splitlines()) if m]
    if not rows:
        print("   no dream updates yet")
        return
    u = np.array([float(r[0]) for r in rows]); ret = np.array([float(r[1]) for r in rows])
    print(f"   {len(rows)} updates, dream return {ret[0]:.1f} -> {ret[-1]:.1f}")
    if len(u) >= 3:
        A = np.vstack([u, np.ones_like(u)]).T
        b, *_ = np.linalg.lstsq(A, ret, rcond=None)
        r_ = ret - A @ b
        se = float(np.sqrt((r_ @ r_ / max(len(u) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = b[0] / se if se > 0 else float("nan")
        print(f"   dream return slope {b[0]:+.3f} +- {se:.3f}/update t={t:+.2f}  "
              f"(rising in-dream means NOTHING about the real game -- see section 3)")


def real_section(rows: list[dict]) -> None:
    section("3. POLICY IN REAL LIFE -- the online evaluator (the only result that counts)")
    if not rows:
        print("   no real episodes evaluated yet")
        return
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get("policy") or r.get("ckpt") or "?"), []).append(r)
    ks = sorted(by)
    print(f"   {'snapshot':30}{'eps':>5}{'dmgB':>7}{'kill%':>7}{'death%':>8}{'hpLost':>8}")
    bals = []
    for k in ks[-6:]:
        rs = by[k]
        per: dict[int, list[float]] = {}
        for r in rs:
            per.setdefault(int(r.get("slot", -1)), []).append(float(r.get("boss_damage", 0)))
        bal = float(np.mean([np.mean(v) for v in per.values()]))
        ends = Counter(str(r.get("end", "?")) for r in rs)
        kills = sum(1 for r in rs if r.get("boss_defeated"))
        hp = float(np.mean([float(r.get("hp_lost", 0)) for r in rs]))
        print(f"   {Path(k).name[:30]:30}{len(rs):>5}{bal:>7.2f}{100*kills/len(rs):>6.1f}%"
              f"{100*ends['death']/len(rs):>7.1f}%{hp:>8.2f}")
    for k in ks:
        rs = by[k]
        per2: dict[int, list[float]] = {}
        for r in rs:
            per2.setdefault(int(r.get("slot", -1)), []).append(float(r.get("boss_damage", 0)))
        bals.append(float(np.mean([np.mean(v) for v in per2.values()])))
    if len(bals) >= 3:
        y = np.array(bals); x = np.arange(len(y), dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r_ = y - A @ b
        se = float(np.sqrt((r_ @ r_ / max(len(x) - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0]))
        t = b[0] / se if se > 0 else float("nan")
        print(f"   REAL damage slope {b[0]:+.4f} +- {se:.4f} per snapshot  t={t:+.2f}  "
              f"{'RISING' if t > 2 else 'FALLING' if t < -2 else 'flat'}  over {len(y)} snapshots")
    print(f"   {len(rows)} real episodes total")


def loop_section(state: dict | None) -> None:
    section("4. THE DYNA LOOP -- is real play improving the model and the policy?")
    if not state:
        print("   loop has not completed an iteration yet")
        return
    print(f"   anchor: {Path(str(state.get('anchor'))).name}   score: {state.get('anchor_score')}")
    for h in state.get("history", [])[-6:]:
        c = h.get("candidate", {})
        print(f"   iter {h['iter']:>2}  {h.get('decision','?'):>8}  dmg {c.get('dmg_balanced', float('nan')):.2f}"
              f"  kills {c.get('kills','?')}/{c.get('episodes','?')}  {h.get('seconds',0)/60:.0f} min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="gohma-dyna")
    ap.add_argument("--wm-log", default="logs/wm_obs.log")
    ap.add_argument("--remote", default=None, help="fleet experiment to read from over ssh")
    a = ap.parse_args()
    if a.remote:
        wm = remote(a.remote, f"tail -400 /workspace/zelda-RL/{a.wm_log} 2>/dev/null")
        dream = remote(a.remote, f"tail -300 /workspace/zelda-RL/runs/{a.run}/*dream*.log 2>/dev/null")
        onl = remote(a.remote, f"tail -4000 /workspace/zelda-RL/runs/{a.run}/online.jsonl 2>/dev/null")
        stj = remote(a.remote, f"cat /workspace/zelda-RL/runs/{a.run}/loop_state.json 2>/dev/null")
    else:
        rd = ROOT / "runs" / a.run
        wm = (ROOT / a.wm_log).read_text() if (ROOT / a.wm_log).is_file() else ""
        dream = "\n".join(p.read_text() for p in rd.glob("*dream*.log")) if rd.is_dir() else ""
        onl = (rd / "online.jsonl").read_text() if (rd / "online.jsonl").is_file() else ""
        stj = (rd / "loop_state.json").read_text() if (rd / "loop_state.json").is_file() else ""
    rows = []
    for line in onl.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    state = None
    if stj.strip().startswith("{"):
        try:
            state = json.loads(stj[stj.index("{"):stj.rindex("}") + 1])
        except Exception:                                    # noqa: BLE001
            state = None
    wm_section(wm)
    dream_section(dream)
    real_section(rows)
    loop_section(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
