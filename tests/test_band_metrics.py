"""`success_balanced` must be invariant to the curriculum mix. That is the whole point of it.

Pooled `combat/success_rate` is episode-weighted, so it moves when the slot weights move and cannot be
compared across versions. This project launched gohma-band-v12 because v11 showed +0.270/update of
"momentum" -- which was slot 7 alone (7.32% -> 14.00%) amplified by that run's 50% weighting on slot 7, with
all nine other slots at exactly 0.00% for the entire run. The same shape then made v14/v15 look flat while
they were the first versions to win from ANY distance.

So the property is asserted, not assumed: two rollouts with identical per-slot skill and different mixes
must give the same `success_balanced` and different `success_rate`.

    .venv/bin/python tests/test_band_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import GOHMA_SLOT_DIST, band_metrics      # noqa: E402

SLOT_HP = {1: 5, 2: 5, 3: 10, 4: 8, 5: 10, 6: 6, 7: 5, 8: 10, 9: 5, 10: 5}


def eps_for(mix: dict[int, int], skill: dict[int, float]) -> list[dict]:
    """`mix` = episodes per slot; `skill` = that slot's true win rate. Wins are placed deterministically."""
    out = []
    for slot, n in mix.items():
        wins = round(skill[slot] * n)
        for i in range(n):
            out.append({"slot": slot, "success": i < wins, "died": i >= wins,
                        "boss_hp_start": SLOT_HP[slot], "boss_damage": 2.0})
    return out


def main() -> int:
    fails = []

    def check(name, got, want, tol=1e-9):
        ok = abs(got - want) < tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got:.6f} (expected {want:.6f})")
        if not ok:
            fails.append(name)

    # The real thing: v11's skill profile. Slot 7 competent, everything else zero.
    v11_skill = {7: 0.14, 3: 0.0, 6: 0.0, 5: 0.0, 4: 0.0, 8: 0.0}
    # Episode counts must make each slot's rate EXACTLY realisable, or the two arms have different true
    # skill and the invariance check fails on integer rounding rather than on the metric. 0.14 needs a
    # multiple of 50; the first version of this test used 70 and read 10/70 = 0.1429.
    heavy7 = {7: 200, 3: 50, 6: 50, 5: 50, 4: 50, 8: 50}        # v11-like: ~44% of episodes on slot 7
    even = {s: 100 for s in heavy7}                              # a flat mix

    a = band_metrics(eps_for(heavy7, v11_skill))
    b = band_metrics(eps_for(even, v11_skill))
    print("identical per-slot skill, two different curriculum mixes:")
    print(f"  pooled success_rate   heavy-slot-7 mix would read higher (it is episode-weighted)")
    ra = sum(1 for e in eps_for(heavy7, v11_skill) if e["success"]) / sum(heavy7.values())
    rb = sum(1 for e in eps_for(even, v11_skill) if e["success"]) / sum(even.values())
    print(f"    heavy7 {100 * ra:.2f}%   even {100 * rb:.2f}%   -> differ by {100 * abs(ra - rb):.2f} points")
    if abs(ra - rb) < 0.01:
        fails.append("pooled rate should differ between mixes")
        print("  FAIL  pooled rate did not move with the mix; the test fixture is wrong")
    check("success_balanced is MIX-INVARIANT", a["combat/success_balanced"], b["combat/success_balanced"])
    check("success_hard is MIX-INVARIANT", a["combat/success_hard"], b["combat/success_hard"])

    print("\nsuccess_hard must exclude the near band, where slot 7 lives")
    check("v11 profile scores 0 on hard", a["combat/success_hard"], 0.0)
    check("v11 profile scores 0.14/6 balanced", a["combat/success_balanced"], 0.14 / 6)

    print("\nslots_nonzero: the measure v11 scores 1 on and v15 scores 6 on")
    check("v11 profile: 1 slot non-zero", a["combat/slots_nonzero"], 1.0)
    v15_skill = {7: 0.2846, 3: 0.0383, 6: 0.0407, 8: 0.0051, 5: 0.0604, 4: 0.0198}
    c = band_metrics(eps_for({s: 200 for s in v15_skill}, v15_skill))
    check("v15 profile: 6 slots non-zero", c["combat/slots_nonzero"], 6.0)
    print(f"  and on the HARD metric: v11 {100 * a['combat/success_hard']:.2f}%  vs  "
          f"v15 {100 * c['combat/success_hard']:.2f}%   <- the comparison pooled success rate hid")
    if not c["combat/success_hard"] > a["combat/success_hard"]:
        fails.append("v15 should beat v11 on success_hard")

    print("\nbands are assigned by engagement distance")
    bands = band_metrics(eps_for({7: 10, 3: 10, 6: 10, 5: 10}, {7: 1.0, 3: 0.0, 6: 0.0, 5: 0.0}))
    check("near band = slots 7 and 3 (38u, 188u), half winning", bands["by_band/near/success_rate"], 0.5)
    check("near band n = 20", bands["by_band/near/n"], 20.0)
    check("mid band = slot 6 (357u)", bands["by_band/mid/n"], 10.0)
    check("far band = slot 5 (472u)", bands["by_band/far/n"], 10.0)

    print("\nedge cases")
    check("empty rollout returns nothing", float(len(band_metrics([]))), 0.0)
    one = band_metrics(eps_for({7: 8}, {7: 0.5}))
    check("a single near-only rollout has no hard metric", float("combat/success_hard" in one), 0.0)
    check("  ...but still reports balanced", one["combat/success_balanced"], 0.5)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
