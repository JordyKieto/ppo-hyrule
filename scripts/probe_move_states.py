"""Which actionFunc values are rolls, backflips and sidehops? Drive the inputs and read them off.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/probe_move_states.py

`zelda_rl.bc.LINK_ACTION_VALUES` was built by taking the 20 most frequent `actionFunc` values in the
collection and naming each from what co-occurs with it (airborne %, melee %, wall %, speed). That found
jump and jump_attack, but it never asked whether rolls, backflips and sidehops are in the table at all --
they could be inside a generic label, or in the 0.2% "other" bucket.

In OoT these are separate action routines with specific inputs:

    roll      run in a direction, then A
    backflip  Z-target held, stick BACK, A
    sidehop   Z-target held, stick LEFT or RIGHT, A

So drive each one and record which `actionFunc` appears, then check it against the decoded table. Naming a
state from a deliberate input is stronger evidence than naming it from a correlation.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, check_slots  # noqa: E402
from zelda_rl.bc import LINK_ACTION_NAMES, LINK_ACTION_VALUES  # noqa: E402
from zelda_rl.combat import BG_GROUND, LINK_FIELD_OFFSETS  # noqa: E402
from zelda_rl.obs import PLAY_STATE  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"
N_BTN = 7
BTN_NONE, BTN_A, BTN_B, BTN_Z, BTN_SHIELD, BTN_ZB, BTN_ZA = range(7)
# stick direction indices: 1..8 are 45-degree steps from +x (world angles), 0 is neutral
N, NE, E, SE, S, SW, W, NW = 3, 4, 5, 6, 7, 8, 1, 2

# name -> list of (direction, button) held in sequence, each for `--hold` steps
MOVES = {
    "run":       [(N, BTN_NONE)] * 3,
    "roll":      [(N, BTN_NONE), (N, BTN_A)],          # momentum, then A while moving = roll
    "backflip":  [(0, BTN_Z), (S, BTN_ZA)],            # Z held, stick back, A
    "sidehop_l": [(0, BTN_Z), (W, BTN_ZA)],
    "sidehop_r": [(0, BTN_Z), (E, BTN_ZA)],
    "jump_atk":  [(0, BTN_Z), (N, BTN_ZA)],            # Z held, stick toward target, A
    "shield":    [(0, BTN_SHIELD)] * 2,
}
ACT_OFF, ACT_FMT = LINK_FIELD_OFFSETS["action_fn"]
UPD_OFF, UPD_FMT = LINK_FIELD_OFFSETS["upd_fn"]
BG_OFF, BG_FMT = LINK_FIELD_OFFSETS["bg_flags"]
SPD_OFF, SPD_FMT = LINK_FIELD_OFFSETS["speed"]
SPAN = 0x8B4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", default="1,2,3,6")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--hold", type=int, default=12, help="steps per leg of a move")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]
    if bad := [k for k, n in check_slots(a.slots_dir, slots).items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba in {a.slots_dir}")
    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    cfg["episode_seconds"] = 200.0
    env, _ = make_env(cfg, Path("runs/moves") / time.strftime("%H%M%S"), slots=slots,
                      slots_dir=a.slots_dir, video_backend="Null", unlimited_speed=True, audio=False)
    ram = env.ram

    seen: dict[str, Counter] = {k: Counter() for k in MOVES}
    seq: dict = {}                 # (move, slot) -> per-frame action_fn_rel through the move
    air: dict[str, list] = {k: [] for k in MOVES}
    spd: dict[str, list] = {k: [] for k in MOVES}
    try:
        for slot in slots:
            for name, legs in MOVES.items():
                env.reset(slot)
                for li, (d, b) in enumerate(legs):
                    for _ in range(a.hold):
                        env.send_action(d * N_BTN + b)
                        while not env.ready():
                            time.sleep(0.0005)
                        env.observe()
                        pl = struct.unpack_from(">I", ram, (PLAY_STATE + 0x1C44) & 0x3FFFFF)[0]
                        if not 0x80000000 <= pl < 0x80400000:
                            break
                        buf = bytes(ram[pl & 0x3FFFFF:(pl + SPAN) & 0x3FFFFF])
                        rel = (struct.unpack_from(ACT_FMT, buf, ACT_OFF)[0]
                               - struct.unpack_from(UPD_FMT, buf, UPD_OFF)[0])
                        bg = struct.unpack_from(BG_FMT, buf, BG_OFF)[0]
                        if li == len(legs) - 1:          # only the leg that performs the move
                            seen[name][rel] += 1
                            air[name].append(1 - ((bg >> BG_GROUND) & 1))
                            spd[name].append(struct.unpack_from(SPD_FMT, buf, SPD_OFF)[0])
                            seq.setdefault((name, slot), []).append(rel)
    finally:
        env.close()

    tbl = {v: LINK_ACTION_NAMES[i] for i, v in enumerate(LINK_ACTION_VALUES)}
    print(f"\n{'move':<11} {'action_fn_rel':>14} {'frames':>7} {'air%':>6} {'speed':>7}  decoded as")
    missing = set()
    for name in MOVES:
        if not seen[name]:
            print(f"{name:<11} {'(no frames)':>14}")
            continue
        for rel, n in seen[name].most_common(3):
            lbl = tbl.get(rel)
            mark = lbl if lbl else "** NOT IN TABLE -> falls into 'other' **"
            if not lbl:
                missing.add(rel)
            print(f"{name:<11} {rel:>14} {n:>7} {100*np.mean(air[name]):>5.0f}% "
                  f"{np.mean(spd[name]):>7.2f}  {mark}")
    print("\n=== per-frame sequence through the move (slot 1), so the move's OWN routine is visible ===")
    print("    the value on the frame the input changes is the move; later frames are the recovery/walk")
    for name in MOVES:
        v = seq.get((name, slots[0]))
        if v:
            print(f"  {name:<11} " + " ".join(
                f"{tbl.get(x, 'other?')[:9]}" if i == 0 else ("." if x == v[i-1] else tbl.get(x, f"NEW:{x}"))
                for i, x in enumerate(v)))
    print(f"\ntable covers {len(LINK_ACTION_VALUES)} values; "
          f"{len(missing)} value(s) produced by these moves are NOT in it: {sorted(missing)}")
    if missing:
        print("Those frames decode as 'other', so the dream cannot tell them apart from each other.")


if __name__ == "__main__":
    main()
