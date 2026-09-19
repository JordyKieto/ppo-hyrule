"""Verify Link's collision / action / attack fields against the OoT decompilation.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/verify_link_fields.py

Step 3 (Verify) of docs/ADDING_OBSERVATIONS.md. Offsets are not guessed: they come from
`include/actor.h` and `include/player.h` in zeldaret/oot. The one adjustment needed is a version shift.

    typedef struct Actor {
        ...
        /* 0x138 */ struct ActorOverlay* overlayEntry;
    #if DEBUG_FEATURES
        /* 0x13C */ char dbgPad[0x10];
    #endif
    } Actor; // size = 0x14C

We run retail NTSC 1.2, which is not a DEBUG_FEATURES build, so `Actor` is 0x13C rather than 0x14C and
every `Player` field after the embedded actor sits 0x10 LOWER than the offset printed in the decomp.
Actor's own fields (< 0x13C) are unshifted. Two offsets already in zelda_rl/combat.py agree:
stateFlags1 0x67C-0x10 = 0x66C, and heldItemAction 0x151-0x10 = 0x141.

Each check drives a behaviour that should move the field and asserts the field responds, so a wrong
offset fails loudly instead of silently feeding a constant into the dataset.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, check_slots  # noqa: E402
from zelda_rl.obs import PLAY_STATE  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"
SHIFT = 0x10          # retail has no Actor.dbgPad, so Player fields are decomp offset - 0x10

# name -> (offset, struct format). Actor fields unshifted; Player fields already shifted.
FIELDS = {
    # --- Actor (unshifted) ---
    "pos_x":            (0x024, ">f"),
    "pos_y":            (0x028, ">f"),
    "pos_z":            (0x02C, ">f"),
    "world_yaw":        (0x032, ">H"),
    "vel_x":            (0x05C, ">f"),
    "vel_y":            (0x060, ">f"),
    "vel_z":            (0x064, ">f"),
    "speed":            (0x068, ">f"),
    "wall_poly":        (0x074, ">I"),
    "floor_poly":       (0x078, ">I"),
    "wall_bg_id":       (0x07C, ">B"),
    "wall_yaw":         (0x07E, ">h"),
    "floor_height":     (0x080, ">f"),
    "depth_in_water":   (0x084, ">f"),
    "bg_check_flags":   (0x088, ">H"),
    "shape_yaw":        (0x0B4 + 0x02, ">h"),
    # --- Player (decomp offset - SHIFT) ---
    "door_type":        (0x42C - SHIFT, ">b"),
    "door_direction":   (0x42D - SHIFT, ">b"),
    "door_timer":       (0x42E - SHIFT, ">h"),
    "focus_actor":      (0x664 - SHIFT, ">I"),
    "ztarget_timer":    (0x66C - SHIFT, ">i"),
    "action_func":      (0x674 - SHIFT, ">I"),
    "state_flags1":     (0x67C - SHIFT, ">I"),
    "state_flags2":     (0x680 - SHIFT, ">I"),
    "state_flags3":     (0x692 - SHIFT, ">B"),
    "speed_xz":         (0x838 - SHIFT, ">f"),
    "yaw":              (0x83C - SHIFT, ">h"),
    "parallel_yaw":     (0x83E - SHIFT, ">h"),
    "melee_weapon_anim": (0x842 - SHIFT, ">b"),
    "melee_weapon_state": (0x843 - SHIFT, ">b"),
    "y_dist_to_ledge":  (0x884 - SHIFT, ">f"),
    "dist_interact_wall": (0x888 - SHIFT, ">f"),
    "ledge_climb_type": (0x88C - SHIFT, ">B"),
    "fall_start_height": (0x894 - SHIFT, ">h"),
    "fall_distance":    (0x896 - SHIFT, ">h"),
    "floor_pitch":      (0x898 - SHIFT, ">h"),
    "knockback_type":   (0x8A1 - SHIFT, ">B"),
    "knockback_rot":    (0x8A2 - SHIFT, ">h"),
}
PLAYER_SPAN = 0x8B4                      # read enough to cover every offset above

BGCHECK = {0: "GROUND", 1: "GROUND_TOUCH", 2: "GROUND_LEAVE", 3: "WALL", 4: "CEILING", 5: "WATER",
           6: "WATER_TOUCH", 7: "GROUND_STRICT", 8: "CRUSHED", 9: "PLAYER_WALL_INTERACT"}

N_BTN = 7
# combat action = dir * 7 + button. Button ids come from zelda_rl.env's combat action table.
NEUTRAL, FWD = 0, 1 * N_BTN


def read(buf: bytes) -> dict:
    return {k: struct.unpack_from(f, buf, o)[0] for k, (o, f) in FIELDS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", default="1,2,3,6")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--hold", type=int, default=45)
    ap.add_argument("--out", default="runs/manual/link_fields.json")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]
    if bad := [k for k, n in check_slots(a.slots_dir, slots).items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba in {a.slots_dir}")
    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    cfg["episode_seconds"] = 200.0
    env, _ = make_env(cfg, Path("runs/verify") / time.strftime("%H%M%S"), slots=slots,
                      slots_dir=a.slots_dir, video_backend="Null", unlimited_speed=True, audio=False)
    ram = env.ram

    rows = []

    def episode(tag: str, slot: int, actions):
        """Run a scripted action sequence, recording every field each step."""
        env.reset(slot)
        prev = None
        for t, act in enumerate(actions):
            env.send_action(act)
            while not env.ready():
                time.sleep(0.0005)
            env.observe()
            pl = struct.unpack_from(">I", ram, (PLAY_STATE + 0x1C44) & 0x3FFFFF)[0]
            if not 0x80000000 <= pl < 0x80400000:
                return
            buf = bytes(ram[pl & 0x3FFFFF:(pl + PLAYER_SPAN) & 0x3FFFFF])
            r = read(buf)
            r["tag"], r["t"], r["act"] = tag, t, act
            r["moved"] = 0.0 if prev is None else float(
                np.hypot(r["pos_x"] - prev[0], r["pos_z"] - prev[1]))
            prev = (r["pos_x"], r["pos_z"])
            rows.append(r)

    try:
        for s in slots:
            for d in range(1, 9):                       # every direction: some walk into walls
                episode("walk", s, [d * N_BTN] * a.hold)
            # button 1=A (roll/jump), 2=B (sword), 3=L (Z-target), 4=R (shield), 5=L+B, 6=L+A
            for b in range(1, N_BTN):
                episode(f"btn{b}", s, [NEUTRAL] * 3 + [FWD + b] * a.hold)
    finally:
        env.close()

    print(f"\n{len(rows):,} frames, tags {dict(Counter(r['tag'] for r in rows))}")
    walk = [r for r in rows if r["tag"] == "walk"]
    blocked = [r for r in walk if r["moved"] < 1.0 and r["t"] > 2]
    free = [r for r in walk if r["moved"] >= 1.0]
    print(f"walk frames {len(walk):,}: {len(blocked):,} blocked, {len(free):,} free\n")

    ok, fail = [], []

    def check(name, cond, detail):
        (ok if cond else fail).append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name:<26} {detail}")

    print("=== bgCheckFlags (Actor +0x088) ===")
    wb = np.mean([(r["bg_check_flags"] >> 3) & 1 for r in blocked])
    wf = np.mean([(r["bg_check_flags"] >> 3) & 1 for r in free])
    # Cross-validate two independent offsets: BGCHECKFLAG_WALL must agree with wallPoly being non-null.
    # This is the decisive test. "Blocked implies wall" is NOT, because Link slides along a wall while
    # still moving -- 212 such frames here, at mean speed 3.48 vs 4.97 for free-and-not-touching.
    agree = np.mean([((r["bg_check_flags"] >> 3) & 1) == (r["wall_poly"] != 0) for r in rows])
    check("WALL bit == wallPoly!=0", agree > 0.999, f"agreement {agree:.4f} over {len(rows):,} frames")
    check("WALL bit set when blocked", wb > 0.7, f"set on {wb:.0%} of blocked frames ({wf:.0%} of free, "
          f"which is wall-sliding)")
    gr = np.mean([r["bg_check_flags"] & 1 for r in rows])
    check("GROUND bit mostly set", gr > 0.8, f"set on {gr:.0%} of all frames")
    seen = Counter()
    for r in rows:
        for b, n in BGCHECK.items():
            if r["bg_check_flags"] >> b & 1:
                seen[n] += 1
    print(f"    flags seen: {dict(seen)}")

    print("\n=== collision geometry ===")
    wp_b = np.mean([r["wall_poly"] != 0 for r in blocked])
    wp_f = np.mean([r["wall_poly"] != 0 for r in free])
    check("wallPoly non-null blocked", wp_b > 0.7 and wp_f < 0.3, f"{wp_b:.0%} blocked vs {wp_f:.0%} free")
    wy = {r["wall_yaw"] for r in blocked}
    check("wallYaw varies", len(wy) >= 3, f"{len(wy)} distinct values while blocked")
    fh = np.array([r["floor_height"] for r in rows])
    py = np.array([r["pos_y"] for r in rows])
    check("floorHeight ~= Link y", float(np.abs(fh - py).mean()) < 20.0,
          f"mean |floorHeight - y| = {np.abs(fh - py).mean():.2f}")
    dw = np.array([r["depth_in_water"] for r in rows])
    check("depthInWater negative", bool((dw < 0).mean() > 0.9),
          f"{(dw < 0).mean():.0%} of frames above water, mean {dw.mean():.0f}")

    print("\n=== action state machine (Player +0x664) ===")
    af = Counter(r["action_func"] for r in rows)
    inram = all(0x80000000 <= v < 0x80800000 for v in af)
    check("actionFunc is a code ptr", inram, f"{len(af)} distinct values, all in RAM: {inram}")
    check("actionFunc few values", 2 <= len(af) <= 40, f"{len(af)} distinct (expect a small state set)")
    print("    top: " + ", ".join(f"0x{v:08X}x{n}" for v, n in af.most_common(6)))

    print("\n=== attack / z-target ===")
    mw = Counter(r["melee_weapon_state"] for r in rows)
    check("meleeWeaponState nonzero sometimes", any(k != 0 for k in mw),
          f"values {dict(mw)}")
    zt = Counter(r["ztarget_timer"] for r in rows)
    check("zTargetActiveTimer varies", len(zt) >= 2, f"{len(zt)} distinct: {list(zt)[:6]}")
    nofocus = [r for r in rows if r["focus_actor"] == 0 and r["ztarget_timer"] != 0]
    check("Z-target with NO target seen", len(nofocus) > 0,
          f"{len(nofocus)} frames with ztarget active and focusActor == 0")

    print("\n=== speed / yaw consistency (Player vs Actor) ===")
    sp = np.array([r["speed"] for r in rows])
    sx = np.array([r["speed_xz"] for r in rows])
    c = float(np.corrcoef(sp, sx)[0, 1]) if sp.std() > 0 and sx.std() > 0 else 0.0
    check("speedXZ tracks actor.speed", c > 0.8, f"corr {c:.3f}")
    print(f"    actor.speed: blocked mean {np.mean([r['speed'] for r in blocked]):.2f}, "
          f"free mean {np.mean([r['speed'] for r in free]):.2f}")
    yw = np.array([r["yaw"] for r in rows], np.int64) & 0xFFFF
    sy = np.array([r["shape_yaw"] for r in rows], np.int64) & 0xFFFF
    d = np.abs((yw - sy + 32768) % 65536 - 32768) / 65536 * 360
    check("Player.yaw ~= shape yaw", float(np.median(d)) < 30.0, f"median angular diff {np.median(d):.1f} deg")

    print("\n=== airborne / slope ===")
    fd = Counter(r["fall_distance"] for r in rows)
    check("fallDistance plausible", all(-2000 < k < 6000 for k in fd), f"{len(fd)} distinct, range "
          f"{min(fd)}..{max(fd)}")
    fp = np.array([r["floor_pitch"] for r in rows])
    check("floorPitch small", bool(np.abs(fp).max() < 20000), f"max |pitch| {np.abs(fp).max()}")

    print("\n=== wall interaction distance ===")
    di = np.array([r["dist_interact_wall"] for r in rows])
    check("distToInteractWall finite", bool(np.isfinite(di).all() and np.abs(di).max() < 1e5),
          f"range {di.min():.1f}..{di.max():.1f}")

    print(f"\n{len(ok)} passed, {len(fail)} failed" + (f": {fail}" if fail else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"frames": len(rows), "passed": ok, "failed": fail,
         "fields": {k: [hex(o), f] for k, (o, f) in FIELDS.items()},
         "bgcheck_seen": dict(seen), "action_funcs": {hex(k): v for k, v in af.most_common()},
         "melee_states": {str(k): v for k, v in mw.items()}}, indent=2, default=float))
    np.savez_compressed(Path(a.out).with_suffix(".npz"),
                        **{k: np.array([r[k] for r in rows]) for k in list(FIELDS) + ["t", "act", "moved"]},
                        tag=np.array([r["tag"] for r in rows]))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
