"""The human anchor: whole on every rank, and pinned to a fixed share of the gradient as Dyna grows.

Measured on `gohma-rvs2` (2026-10-31), the previous behaviour:
  * stratified sharding gave each of 7 ranks 13-15 of the 100 human episodes, DISJOINT from its peers';
    val-WIN accuracy spread 3.4x across ranks at matched epochs (9.7% to 33.2%) and balanced boss damage
    fell -0.0639 +- 0.0109/epoch (t=-5.86) while train accuracy rose.
  * --win-share pinned the WIN class, with human a sub-class of it, so human's share was only stable
    because the policy's own play won almost nothing (1 win in 15 episodes). Dyna wins would take share
    from the demonstrations exactly when the policy is least worth imitating.

Both are asserted here against the real solver arithmetic, including the growth that is supposed to break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shard(eps, rank, world):
    """The sharding rule from scripts/train_rvs.py: human whole, every other class round-robin."""
    def klass(e):
        src = (str(e.get("src", "")) + " " + str(e.get("file", ""))).lower()
        return "human" if "human" in src else ("win" if e["win"] else "loss")
    buckets = {}
    for e in eps:
        buckets.setdefault(klass(e), []).append(e)
    shardable = [k for k in sorted(buckets) if k != "human"]
    return list(buckets.get("human", [])) + [e for k in shardable
                                             for e in buckets[k][rank::world]]


def solve(eps, win_share, human_share):
    """The multiplier arithmetic from scripts/train_rvs.py, returning the ACHIEVED shares."""
    def is_human(e):
        return "human" in (str(e.get("src", "")) + " " + str(e.get("file", ""))).lower()
    n = lambda sel: sum(len(e["act"]) for e in eps if sel(e))        # noqa: E731
    hum, oth = n(is_human), n(lambda e: e["win"] and not is_human(e))
    loss = n(lambda e: not e["win"])
    T = loss / (1.0 - win_share)
    mh = human_share * T / hum
    mw = ((win_share - human_share) * T / oth) if oth else 0.0
    tot = hum * mh + oth * mw + loss
    return hum * mh / tot, (hum * mh + oth * mw) / tot


def corpus(n_human=100, n_ppo_win=200, n_loss=500, n_dyna=0, n_dyna_win=0, steps=700):
    eps = [{"src": "datasets/human/gohma", "file": f"h{i}.npz", "win": True,
            "act": [0] * steps} for i in range(n_human)]
    eps += [{"src": "datasets/ppo/mix", "file": f"w{i}.npz", "win": True,
             "act": [0] * steps} for i in range(n_ppo_win)]
    eps += [{"src": "datasets/ppo/mix", "file": f"l{i}.npz", "win": False,
             "act": [0] * steps} for i in range(n_loss)]
    eps += [{"src": "datasets/dyna_live2", "file": f"d{i}.npz", "win": i < n_dyna_win,
             "act": [0] * steps} for i in range(n_dyna)]
    return eps


def main() -> None:
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{('  <- ' + detail) if (detail and not cond) else ''}")
        if not cond:
            fails.append(name)

    # ---------- the anchor is whole on every rank
    base = corpus()
    world = 7
    shards = [shard(base, r, world) for r in range(world)]
    hums = [sum(1 for e in s if "human" in e["src"]) for s in shards]
    ck(f"every rank holds all 100 human episodes (got {hums})", all(h == 100 for h in hums))
    ck("and the old rule did not (13-15 each)",
       all(13 <= len(base[0:100][r::world]) <= 15 for r in range(world)))
    losses = [sum(1 for e in s if not e["win"]) for s in shards]
    ck(f"failures are still sharded ~1/7 (got {losses})", all(60 <= l <= 80 for l in losses))
    seen = [{e["file"] for e in s if not e["win"]} for s in shards]
    ck("and the failure shards are disjoint", not (seen[0] & seen[1]) and not (seen[2] & seen[3]))

    # ---------- the human share is PINNED as Dyna grows -- the claim that matters
    print("\n  human % of gradient as Dyna grows (target 20%):")
    rows = []
    for n_dyna, n_win in [(0, 0), (200, 0), (1000, 0), (1000, 300), (4000, 1500)]:
        eps = shard(corpus(n_dyna=n_dyna, n_dyna_win=n_win), 0, world)
        h, w = solve(eps, win_share=0.30, human_share=0.20)
        rows.append((n_dyna, n_win, h, w))
        print(f"      dyna {n_dyna:>5} eps ({n_win:>4} wins): human {100*h:5.2f}%   all wins {100*w:5.2f}%")
    ck("human stays at 20.00% of the gradient for every corpus",
       all(abs(h - 0.20) < 1e-9 for _, _, h, _ in rows), str([round(h, 4) for *_, h, _ in rows]))
    ck("and all wins stay at 30.00%", all(abs(w - 0.30) < 1e-9 for *_, w in rows))

    # ---------- the control: without --human-share, Dyna WINS erode the demonstrations
    def solve_old(eps, win_share):
        def is_human(e):
            return "human" in str(e.get("src", "")).lower()
        n = lambda sel: sum(len(e["act"]) for e in eps if sel(e))     # noqa: E731
        win, allst = n(lambda e: e["win"]), n(lambda e: True)
        p = win / allst
        m = (win_share * (1 - p)) / (p * (1 - win_share))
        tot = win * m + (allst - win)
        return n(is_human) * m / tot
    a = solve_old(shard(corpus(), 0, world), 0.30)
    b = solve_old(shard(corpus(n_dyna=4000, n_dyna_win=1500), 0, world), 0.30)
    print(f"\n  control, --human-share off: human {100*a:.2f}% -> {100*b:.2f}% once Dyna wins 1500")
    ck("the old weighting DOES erode human when Dyna wins (so the fix is load-bearing)", b < a * 0.5,
       f"{a:.4f} -> {b:.4f}")

    # ---------- the guard
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "train_rvs.py"),
                        "--data", "datasets/human/gohma", "--human-share", "0.4", "--win-share", "0.3",
                        "--out", "checkpoints/kept/_ha_test.pt"],
                       capture_output=True, text=True, timeout=300)
    ck("--human-share >= --win-share is refused", "must be BELOW" in (r.stdout + r.stderr),
       (r.stdout + r.stderr)[-200:])

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'ALL PASS'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
