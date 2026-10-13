"""The adaptive start-slot distribution: the spread must be a STATED quantity, and nothing may be starved.

PPO optimises expected return under the sampling distribution, so these weights are part of the objective.
Two failure modes have already cost this project a run each, and both are asserted against here:

- v16 spent 30% of its episodes on the slot it already wins 28% of the time. That is what `deficit` fixes.
- v12 hand-inverted the curriculum, starved the slots it had already learned, and stalled at +0.098/update.
  That is what `min_share` prevents, so it is tested rather than hoped for.

    .venv/bin/python tests/test_deficit_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import deficit_weights      # noqa: E402

V15 = {7: 0.2846, 3: 0.0383, 6: 0.0407, 8: 0.0051, 5: 0.0604, 4: 0.0198}


def main() -> int:
    fails = []

    def ok(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    print("on v15's real measured per-slot rates")
    for ratio in (2.0, 4.0, 8.0):
        p = deficit_weights(V15, ratio=ratio, min_share=0.0)
        got = max(p.values()) / min(p.values())
        ok(f"ratio={ratio}: worst/best weight is exactly {ratio}", abs(got - ratio) < 1e-6, f"got {got:.4f}")
        ok(f"ratio={ratio}: sums to 1", abs(sum(p.values()) - 1) < 1e-9)
        ok(f"ratio={ratio}: the least-solved slot (8, 0.51%) gets the most weight",
           max(p, key=p.get) == 8)
        ok(f"ratio={ratio}: the most-solved slot (7, 28.5%) gets the least", min(p, key=p.get) == 7)

    print("\nthe anti-forgetting floor (the v12 failure)")
    p = deficit_weights(V15, ratio=8.0, min_share=0.06)
    ok("no slot below the floor", min(p.values()) >= 0.06 - 1e-9, f"min {100 * min(p.values()):.2f}%")
    ok("still sums to 1", abs(sum(p.values()) - 1) < 1e-9)
    ok("the solved slot is held AT the floor, not starved", abs(p[7] - 0.06) < 1e-9,
       f"slot 7 = {100 * p[7]:.2f}%")
    # At the shipped ratio of 4, slot 7 would get 5.9% unaided, so the floor BINDS and lifts it to 6%.
    # Worth pinning: it means the floor is load-bearing at the default settings, not decorative.
    bare = deficit_weights(V15, ratio=4.0, min_share=0.0)
    p2 = deficit_weights(V15, ratio=4.0, min_share=0.06)
    ok("at the shipped ratio 4 the floor BINDS on slot 7",
       bare[7] < 0.06 and abs(p2[7] - 0.06) < 1e-9, f"{100 * bare[7]:.2f}% -> {100 * p2[7]:.2f}%")
    ok("raising slot 7 to the floor does not push any other slot below it",
       min(p2.values()) >= 0.06 - 1e-9)

    print("\nthe spread is scale-free: it does not depend on the absolute rates")
    halved = {k: v / 2 for k, v in V15.items()}
    a = deficit_weights(V15, ratio=4.0, min_share=0.0)
    b = deficit_weights(halved, ratio=4.0, min_share=0.0)
    ok("halving every rate keeps worst/best at 4x",
       abs(max(b.values()) / min(b.values()) - 4.0) < 1e-6)
    ok("...and the ORDER is unchanged",
       sorted(a, key=a.get) == sorted(b, key=b.get))

    print("\ndegenerate inputs")
    ok("empty -> empty", deficit_weights({}) == {})
    flat = deficit_weights({3: 0.1, 7: 0.1, 8: 0.1}, ratio=4.0)
    ok("all slots equal -> uniform", all(abs(v - 1 / 3) < 1e-9 for v in flat.values()))
    ok("ratio<=1 -> uniform", all(abs(v - 1 / 6) < 1e-9
                                  for v in deficit_weights(V15, ratio=1.0).values()))
    allzero = deficit_weights({3: 0.0, 7: 0.0}, ratio=4.0)
    ok("nothing solved yet -> uniform, not a divide by zero",
       all(abs(v - 0.5) < 1e-9 for v in allzero.values()))
    solved = deficit_weights({3: 1.0, 7: 0.5}, ratio=4.0, min_share=0.0)
    ok("a fully solved slot still gets the LEAST weight", min(solved, key=solved.get) == 3)

    print("\nwhat it actually does to v16's mix")
    cur = {3: 0.133, 4: 0.15, 5: 0.133, 6: 0.15, 7: 0.3, 8: 0.133}
    new = deficit_weights(V15, ratio=4.0, min_share=0.06)
    hard = sum(new[k] for k in (3, 4, 5, 6, 8))
    hard0 = sum(cur[k] for k in (3, 4, 5, 6, 8))
    print(f"  episodes on the five unsolved slots: {100 * hard0:.1f}% -> {100 * hard:.1f}%")
    ok("more than 85% of episodes go to the unsolved slots", hard > 0.85, f"{100 * hard:.1f}%")

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
