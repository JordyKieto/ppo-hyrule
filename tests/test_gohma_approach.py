"""g_approach: it must pay for net progress toward Gohma, and must NOT be farmable by oscillating.

Every shaping term this project has had to remove was farmable (g_sling_ready milked for ~150 steps without
starting the fight, g_spacing's floor banking the whole cap with zero boss damage). So the farming property is
tested before the term is trained on, not after -- AGENTS 9, ablate before you scale.

    .venv/bin/python tests/test_gohma_approach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import GOHMA_SIMPLE, CombatConfig, CombatState, Enemy      # noqa: E402
from zelda_rl.gohma import GOHMA_STATE, GohmaTracker              # noqa: E402

UPD = 0x801E75F8
FN = {v: UPD + k for k, v in GOHMA_STATE.items()}                 # state name -> absolute action-function ptr


def st(dist: float, state: str = "floor_walk", hp: int = 10) -> CombatState:
    """Link `dist` units from a Gohma at the origin, with her in `state`."""
    e = Enemy(0x1000, 0x28, hp, (0.0, 0.0, 0.0), fn=FN[state], yaw_to_player=0, shape_yaw=0, cat=9,
              params=0xFFFF, upd=UPD, focus=(0.0, 60.0, 0.0), yaw=0)   # kind() needs cat 9 + params 0xFFFF
    return CombatState(scene=10, room=0, hp=48, hp_cap=48, sf1=0, pos=(dist, 0.0, 0.0), yaw=0,
                       targeted=0, arrow=0, enemies={e.ptr: e}, held_item=0, seeds=30, nuts=10)


def run(path, g_approach=0.5, state="floor_walk"):
    """Walk Link through `path` (a list of distances) and return what `approach` paid in total."""
    t = GohmaTracker(CombatConfig(profile="gohma", g_approach=g_approach))
    prev = st(path[0], state)
    t.reset(prev, max_steps=3000)
    for d in path[1:]:
        cur = st(d, state)
        t.step(prev, cur)
        prev = cur
    return t.ep["r_parts"].get("approach", 0.0)


def main():
    fails = []

    def check(name, got, want, tol=1e-6):
        ok = abs(got - want) < tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got:+.4f} (expected {want:+.4f})")
        if not ok:
            fails.append(name)

    print("closing pays proportionally to NET progress")
    check("490 -> 90, straight in", run([490, 390, 290, 190, 90]), 0.5 * 4.0)
    check("490 -> 290, halfway", run([490, 390, 290]), 0.5 * 2.0)
    check("standing still pays nothing", run([490, 490, 490]), 0.0)
    check("retreating pays nothing", run([200, 300, 400, 490]), 0.0)

    print("\nTHE FARMING TEST: oscillating must pay the NET, never the sum of the inward legs")
    osc = [490] + [v for _ in range(12) for v in (390, 490)]          # 12 round trips of 100 units
    check("12x (in 100, out 100) == one 100-unit gain", run(osc), 0.5 * 1.0)
    check("in 400 / out 400 / in 400 == one 400-unit gain", run([490, 90, 490, 90]), 0.5 * 4.0)
    deep = [490, 90] + [v for _ in range(20) for v in (300, 150)]     # wander after reaching the best
    check("wandering after the best pays nothing more", run(deep), 0.5 * 4.0)

    print("\nphase separation: silent during her attack frames, where g_spacing owns the step")
    check("floor_eye_red closing pays 0", run([490, 290, 90], state="floor_eye_red"), 0.0)
    check("floor_prepare closing pays 0", run([490, 290, 90], state="floor_prepare"), 0.0)
    check("floor_walk closing pays", run([490, 290, 90], state="floor_walk"), 0.5 * 4.0)

    # Tested at the SHIPPED coefficient, not the function default: the cap only means anything against the
    # value the fleet actually trains with, and when v16 raised 0.5 -> 1.5 this assertion was the thing that
    # noticed the cap no longer bound.
    live = GOHMA_SIMPLE["g_approach"]
    print(f"\nthe cap, at the coefficient the `simple` preset ships ({live})")
    check("a 1350-unit walk hits the cap", run(list(range(1400, 0, -50)), g_approach=live),
          CombatConfig().g_approach_cap)

    # THE DESIGN PROPERTY: the cap must NOT bind on a real crossing, or the gradient dies partway across the
    # room -- which is exactly what raising the coefficient alone would have done.
    far = CombatConfig()
    crossing = run([541, 450, 350, 250, 150, 90], g_approach=live)
    ok = crossing < far.g_approach_cap
    print(f"  {'PASS' if ok else 'FAIL'}  the longest real crossing (541 -> 90) stays UNDER the cap: "
          f"{crossing:+.2f} < {far.g_approach_cap:+.2f}")
    if not ok:
        fails.append("crossing under cap")

    print("\nreproducibility of every earlier run")
    check("g_approach=0 pays nothing (v11-v14 reproducible)", run([490, 290, 90], g_approach=0.0), 0.0)

    c = CombatConfig()
    print("\nscale, against the terms it competes with")
    print(f"  a full 490 -> 90 crossing pays {run([490, 390, 290, 190, 90], g_approach=live):+.2f} at the "
          f"shipped {live}  (cap {c.g_approach_cap:+.2f}, g_spacing_cap {c.g_spacing_cap:+.2f}, "
          f"boss_defeat {c.g_boss_defeat:+.0f})")
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
