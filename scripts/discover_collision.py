"""Find Link's collision fields by driving him into walls and scanning the Actor header.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
    .venv/bin/python-dbg scripts/discover_collision.py

Step 1-2 of docs/ADDING_OBSERVATIONS.md, with the human replaced by a script. The behaviour we want
contrast on -- "Link is pressed against a wall" -- labels itself: hold one direction and compare how far he
actually moves. Held a direction and went nowhere = blocked. Held it and moved = free.

Existing traces cover the Player region +0x3F0..+0x900 (colliders) and the Baba's +0x14C..+0x3A0. Neither
covers the **Actor header** +0x00..+0x14C, which is where OoT keeps bgCheckFlags, wallYaw and floorHeight --
so those fields have never been searched for. Guessed offsets (+0x84, +0x88, +0x90) read constants across
savestates, which is why they were rejected.

Scans every u16 / s16 / u32 / f32 in the header for the signatures in ADDING_OBSERVATIONS.md step 2:
  event flag  a bit that is set on blocked frames and clear on free ones
  direction   s16 spanning the circle, correlated with the wall we are pushing into
  height      f32 that tracks Link's y
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_transitions import DEKU_STATES_DIR, check_slots  # noqa: E402
from zelda_rl.obs import PLAY_STATE  # noqa: E402
from zelda_rl.policy_io import load_policy, make_env, resolve_checkpoint  # noqa: E402

ENV_REF = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"
HDR = 0x14C                      # Actor header size: everything before the actor-specific region
N_DIRS, N_BTN = 9, 7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", default="1,2,3,6")
    ap.add_argument("--slots-dir", default=DEKU_STATES_DIR)
    ap.add_argument("--hold", type=int, default=60, help="steps to hold each direction")
    ap.add_argument("--blocked-thresh", type=float, default=1.0,
                    help="world units of movement per step below which a held direction counts as blocked")
    ap.add_argument("--out", default="runs/manual/collision_scan.json")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]
    if bad := [k for k, n in check_slots(a.slots_dir, slots).items() if n == 0]:
        raise SystemExit(f"slots {bad} have no live Deku Baba in {a.slots_dir}")
    _, cfg, _ = load_policy(resolve_checkpoint(ENV_REF))
    cfg["episode_seconds"] = 200.0
    env, _ = make_env(cfg, Path("runs/collision") / time.strftime("%H%M%S"), slots=slots,
                      slots_dir=a.slots_dir, video_backend="Null", unlimited_speed=True, audio=False)
    ram = env.ram

    rows = []
    try:
        for slot in slots:
            for d in range(1, N_DIRS):                 # every compass direction, so some hit walls
                env.reset(slot)
                act = d * N_BTN
                prev = None
                for t in range(a.hold):
                    env.send_action(act)
                    while not env.ready():
                        time.sleep(0.0005)
                    env.observe()
                    pl = struct.unpack_from(">I", ram, (PLAY_STATE + 0x1C44) & 0x3FFFFF)[0]
                    if not 0x80000000 <= pl < 0x80400000:
                        break
                    hdr = bytes(ram[pl & 0x3FFFFF:(pl + HDR) & 0x3FFFFF])
                    pos = struct.unpack_from(">fff", hdr, 0x24)
                    moved = 0.0 if prev is None else float(np.hypot(pos[0] - prev[0], pos[2] - prev[2]))
                    if prev is not None:
                        rows.append({"slot": slot, "dir": d, "t": t, "hdr": hdr.hex(),
                                     "moved": moved, "blocked": moved < a.blocked_thresh})
                    prev = pos
                print(f"  slot {slot} dir {d}: {sum(r['blocked'] for r in rows if r['slot'] == slot and r['dir'] == d)}"
                      f"/{a.hold - 1} blocked frames", flush=True)
    finally:
        env.close()

    H = np.frombuffer(b"".join(bytes.fromhex(r["hdr"]) for r in rows), np.uint8).reshape(len(rows), HDR)
    blocked = np.array([r["blocked"] for r in rows])
    # Keep the raw header bytes: interpreting candidate offsets takes several passes and re-driving the
    # emulator for each one is minutes per pass.
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    trace = Path(a.out).with_suffix(".npz")
    np.savez_compressed(trace, hdr=H, blocked=blocked,
                        slot=np.array([r["slot"] for r in rows]), dir=np.array([r["dir"] for r in rows]),
                        t=np.array([r["t"] for r in rows]), moved=np.array([r["moved"] for r in rows]))
    print(f"raw trace -> {trace}")
    ypos = np.array([struct.unpack_from(">f", H[i].tobytes(), 0x28)[0] for i in range(len(H))])
    print(f"\n{len(rows):,} frames, {blocked.mean():.1%} blocked\n")

    B = bytes(H.tobytes())
    field = lambda fmt, off, w: np.array(  # noqa: E731 -- one column of the trace, big-endian
        [struct.unpack_from(fmt, B, i * HDR + off)[0] for i in range(len(H))])

    print("=== BIT FLAGS: bits that separate blocked from free ===")
    hits = []
    for off in range(HDR - 2):
        v = field(">H", off, 2)
        for b in range(16):
            bit = (v >> b) & 1
            if bit.std() == 0:
                continue
            sep = abs(bit[blocked].mean() - bit[~blocked].mean())
            if sep > 0.6:
                hits.append((sep, off, b, float(bit[blocked].mean()), float(bit[~blocked].mean())))
    for sep, off, b, pb, pf in sorted(hits, reverse=True)[:10]:
        print(f"  +0x{off:03X} bit {b:>2}: set on {pb:.0%} of blocked frames, {pf:.0%} of free  (sep {sep:.2f})")
    if not hits:
        print("  none")

    print("\n=== FLOATS tracking Link's height (floor height candidates) ===")
    fl = []
    for off in range(0, HDR - 4, 2):
        v = field(">f", off, 4)
        if not np.isfinite(v).all() or np.abs(v).max() > 1e5 or v.std() == 0:
            continue
        c = abs(float(np.corrcoef(v, ypos)[0, 1])) if ypos.std() > 0 else 0.0
        if c > 0.9 and off != 0x28:
            fl.append((c, off, float(v.mean())))
    for c, off, m in sorted(fl, reverse=True)[:6]:
        print(f"  +0x{off:03X}: corr {c:.3f} with Link y, mean {m:.1f}")
    if not fl:
        print("  none")

    print("\n=== s16 DIRECTION fields that vary only while blocked (wall yaw candidates) ===")
    dr = []
    for off in range(0, HDR - 2):
        v = field(">h", off, 2)
        if v.std() == 0:
            continue
        vb, vf = v[blocked], v[~blocked]
        if len(vb) < 20 or len(vf) < 20:
            continue
        if len(set(vb.tolist())) >= 3 and vb.std() > 3 * max(vf.std(), 1e-6):
            dr.append((vb.std() / max(vf.std(), 1e-6), off, len(set(vb.tolist()))))
    for r, off, n in sorted(dr, reverse=True)[:8]:
        print(f"  +0x{off:03X}: {n} distinct values while blocked, {r:.0f}x the variance of free frames")
    if not dr:
        print("  none")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"frames": len(rows), "blocked_frac": float(blocked.mean()),
                                       "bit_hits": hits[:20], "float_hits": fl[:10]}, indent=2, default=float))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
