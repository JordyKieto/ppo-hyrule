"""The human recorder's pad -> action mapping must be the exact inverse of what the env sends.

If it is not, the recording looks perfect and teaches the policy the WRONG DIRECTION, silently. The policy's
action direction is a WORLD angle (`atan2(dz, dx)`) and the env rotates it into stick space by the live camera
yaw (`DolphinGoalEnv._stick`); the human's stick is already in that rotated space. So the test is a round
trip: take every action the policy can emit, compute the stick the env would send, feed that stick back to
`to_action` with the same camera yaw, and require the original action back.

The first version of this mapping treated the direction as an octant INDEX rather than a world angle in
radians and ignored the camera entirely. It produced plausible-looking output and was wrong for every
non-neutral direction (AGENTS 3: derive the transform from the code, assert it).

    .venv/bin/python tests/test_record_human.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_human import BITS, to_action                                      # noqa: E402
from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTONS_V5, STICK_HANDEDNESS, action_name  # noqa: E402

NAME_TO_BIT = {"A": BITS["A"], "B": BITS["B"], "L": BITS["L"], "R": BITS["R"],
               "CD": BITS["CD"], "CL": BITS["CL"]}


def env_stick(d: float, cam_yaw: float) -> tuple[int, int]:
    """What the env would put on the stick for world angle `d`, as a pad reading in [-127, 127]."""
    stick = math.pi / 2 + STICK_HANDEDNESS * (d - cam_yaw)
    return int(round(100 * math.cos(stick))), int(round(100 * math.sin(stick)))


def main() -> int:
    fails = []
    print("ROUND TRIP: every action -> the stick the env would send -> back through to_action\n")
    for cam_yaw in (0.0, 0.7, -1.9, math.pi, 2.6):
        bad = 0
        for a, (d, bi) in enumerate(COMBAT_ACTIONS_V5):
            combo = COMBAT_BUTTONS_V5[bi]
            if any(n not in NAME_TO_BIT for n in combo):
                continue
            mask = 0
            for n in combo:
                mask |= NAME_TO_BIT[n]
            sx, sy = (0, 0) if d is None else env_stick(d, cam_yaw)
            got = to_action(mask, sx, sy, cam_yaw)
            if got != a:
                bad += 1
                if bad <= 3:
                    print(f"  FAIL cam_yaw {cam_yaw:+.2f}: action {a} ({action_name(a, 'combat')}) "
                          f"-> stick({sx},{sy}) -> action {got} ({action_name(got, 'combat')})")
        print(f"  cam_yaw {cam_yaw:+.2f}: {'PASS' if bad == 0 else f'{bad} MISMATCHES'}")
        if bad:
            fails.append(f"cam_yaw {cam_yaw:+.2f}")

    print("\nthe camera actually matters: the same stick is a different world direction per yaw")
    seen = {}
    for cam_yaw in (0.0, 1.57, 3.14, 4.71):
        a = to_action(0, 100, 0, cam_yaw)
        seen[cam_yaw] = a
        print(f"  stick right at cam_yaw {cam_yaw:.2f} -> action {a:>3} {action_name(a, 'combat')}")
    ok = len(set(seen.values())) == len(seen)
    print(f"  {'PASS' if ok else 'FAIL'}  four yaws give four different world directions")
    if not ok:
        fails.append("camera yaw ignored")

    print("\nbuttons, independent of the stick")
    cases = [(0, "neutral"), (BITS["B"], "B"), (BITS["L"] | BITS["B"], "L+B"),
             (BITS["L"] | BITS["R"], "L+R"), (BITS["CD"], "CD"), (BITS["CL"], "CL"),
             (BITS["L"] | BITS["CL"], "L+CL"), (BITS["L"] | BITS["CD"], "L+CD")]
    for mask, label in cases:
        a = to_action(mask, 0, 0, 0.0)
        got = set(COMBAT_BUTTONS_V5[COMBAT_ACTIONS_V5[a][1]])
        want = {n for n in NAME_TO_BIT if mask & NAME_TO_BIT[n]}
        good = got == want
        print(f"  {'PASS' if good else 'FAIL'}  {label:<6} -> {action_name(a, 'combat'):<14} {sorted(got)}")
        if not good:
            fails.append(f"buttons {label}")

    print("\nunmapped buttons are ignored rather than breaking the match")
    a = to_action(BITS["A"] | BITS["START"] | BITS["DU"], 0, 0, 0.0)
    good = set(COMBAT_BUTTONS_V5[COMBAT_ACTIONS_V5[a][1]]) == {"A"}
    print(f"  {'PASS' if good else 'FAIL'}  A+START+DUp -> {action_name(a, 'combat')}")
    if not good:
        fails.append("unmapped buttons")

    print("\ndeadzone")
    for sx, sy, want_neutral in ((0, 0, True), (10, 10, True), (100, 0, False)):
        a = to_action(0, sx, sy, 0.0)
        is_neutral = COMBAT_ACTIONS_V5[a][0] is None
        good = is_neutral == want_neutral
        print(f"  {'PASS' if good else 'FAIL'}  stick({sx},{sy}) neutral={is_neutral}")
        if not good:
            fails.append(f"deadzone {sx},{sy}")

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
