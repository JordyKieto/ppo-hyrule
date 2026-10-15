"""Record HUMAN demonstrations of the Gohma fight with the controller inputs, for behaviour cloning.

The existing `runs/manual/boss_*.jsonl` traces are unusable for BC: their `buttons` field reads
`SAVE_CTX + 0x68`, which is the **equipped C-button items** (`monitor_boss.py` says so in its own log line),
so it is constant for every frame of every trace. Full state, no actions.

This reads the real controller out of `gPlayState + 0x14` -- `Input[4]`, 0x14 bytes each,
`cur.button` u16 / `cur.stick_x` s8 / `cur.stick_y` s8 -- which `docs/RAM_MAP.md` line 101 records as
VERIFIED (`00007f00` with the stick right, zeros in a frozen slot). Offsets come from that map, not recall
(AGENTS 3).

Each episode loads a savestate, you play until Gohma dies / you die / the cap, and it writes one `.npz` whose
`action` column is in the SAME 108-action space the policy uses (`COMBAT_ACTIONS_V5`), so the result drops
straight into behaviour cloning with no relabelling.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
      .venv/bin/python-dbg scripts/record_human.py --slots 3,4,5,6,7,8 --episodes 10 --out datasets/human/gohma

    ... --probe        # 10 s input check: hold buttons, confirm they are being read, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(1)                      # AGENTS 28: never starve the emulator

from zelda_rl.combat import CombatConfig, SF1_DEAD, read_combat_state  # noqa: E402
from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTONS_V5, action_name  # noqa: E402
from zelda_rl.obs import PLAY_STATE, read_state_file_v2  # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR  # noqa: E402

# gPlayState + 0x14 = Input[4], 0x14 bytes each. docs/RAM_MAP.md section 4, status VERIFIED.
INPUT0 = PLAY_STATE + 0x14

# N64 button bits as the VC build reports them in cur.button. Verified empirically by --probe before any
# recording is kept: a constant or all-zero reading means the offset is wrong and the run aborts.
BITS = {"A": 0x8000, "B": 0x4000, "Z": 0x2000, "START": 0x1000,
        "DU": 0x0800, "DD": 0x0400, "DL": 0x0200, "DR": 0x0100,
        "L": 0x0020, "R": 0x0010, "CU": 0x0008, "CD": 0x0004, "CL": 0x0002, "CR": 0x0001}


def read_pad(ram) -> tuple[int, int, int]:
    """(button bitmask, stick_x, stick_y) from controller 0."""
    b = int(np.frombuffer(ram[INPUT0:INPUT0 + 2], ">u2")[0])
    x = int(np.frombuffer(ram[INPUT0 + 2:INPUT0 + 3], np.int8)[0])
    y = int(np.frombuffer(ram[INPUT0 + 3:INPUT0 + 4], np.int8)[0])
    return b, x, y


def held(mask: int) -> list[str]:
    return [n for n, bit in BITS.items() if mask & bit]


def to_action(mask: int, sx: int, sy: int, cam_yaw: float, dead: int = 20) -> int:
    """Map a real pad reading onto the policy's 108-action space, inverting `DolphinGoalEnv._stick`.

    The policy's direction is a **WORLD angle** in the x/z plane (`atan2(dz, dx)`), not a stick octant. The env
    converts world -> stick with

        stick = pi/2 + STICK_HANDEDNESS * (d - cam_yaw)

    so recovering the world angle the human asked for means inverting that with the live camera yaw:

        d = cam_yaw + (stick - pi/2) / STICK_HANDEDNESS

    Getting this wrong is silent -- the recording would look fine and teach the policy the wrong direction,
    which is exactly the class of error AGENTS 3 is about, so it is derived from `env.py` rather than guessed
    and checked by a round-trip test in `tests/test_record_human.py`.

    The stick is then snapped to the nearest of the 8 world DIRECTIONS the action space offers, and the button
    combo is the longest held combo that COMBAT_BUTTONS_V5 contains.
    """
    import math

    from zelda_rl.env import DIRECTIONS, STICK_HANDEDNESS

    d = None
    if abs(sx) > dead or abs(sy) > dead:
        stick = math.atan2(sy, sx)                       # pad axes, same convention as SET MAIN x/y
        world = cam_yaw + (stick - math.pi / 2) / STICK_HANDEDNESS
        # DIRECTIONS includes None (the neutral action); only the real angles are snap targets.
        angles = [c for c in DIRECTIONS if c is not None]
        d = min(angles, key=lambda c: abs((world - c + math.pi) % (2 * math.pi) - math.pi))

    # COMBAT_ACTIONS_V5 entries are (world angle or None, INDEX into COMBAT_BUTTONS_V5). Longest matching
    # combo wins, so a held L+B maps to ('L','B') rather than to ('B',).
    names = set(held(mask))
    best, best_len = None, -1
    for i, (dir_i, bi) in enumerate(COMBAT_ACTIONS_V5):
        same = (dir_i is None and d is None) or (dir_i is not None and d is not None and abs(dir_i - d) < 1e-6)
        if not same:
            continue
        want = set(COMBAT_BUTTONS_V5[bi])
        if want <= names and len(want) > best_len:
            best, best_len = i, len(want)
    return best if best is not None else 0


def build_env(slots, slots_dir, seconds, visible):
    from zelda_rl.env import DolphinGoalEnv, EnvConfig
    files = {k: Path(slots_dir) / f"NACE01.s{k:02d}" for k in slots}
    states = {k: read_state_file_v2(f) for k, f in files.items()}
    cfg = EnvConfig(episode_vi=int(seconds * 60), vi_per_step=6,
                    video_backend="Metal" if visible else "Null", unlimited_speed=False,
                    audio=visible, obs_version=7, task="combat", stall_timeout=600.0,
                    combat=CombatConfig(profile="gohma", max_seconds=seconds, target_ids=(0x28, 0x55)))
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"human_{os.getpid()}"
    return DolphinGoalEnv(0, root / "i0", files, states, states[min(slots)], cfg)


def probe(env, seconds=10.0):
    """Ten seconds of live pad reading. If this is constant, the offset is wrong -- stop before recording."""
    print(f"\nPROBE: reading gPlayState+0x14 for {seconds:.0f}s. Press buttons and move the stick.\n")
    seen, t0, last = set(), time.time(), None
    while time.time() - t0 < seconds:
        b, x, y = read_pad(env.ram)
        cy = env.state.cam_yaw
        seen.add((b, x, y))
        if (b, x, y) != last:
            print(f"  button 0x{b:04X} {held(b) or '-':<40} stick ({x:>4},{y:>4})  -> action "
                  f"{to_action(b, x, y, cy):>3} {action_name(to_action(b, x, y, cy), 'combat')}")
            last = (b, x, y)
        time.sleep(0.05)
    print(f"\n  {len(seen)} distinct pad readings in {seconds:.0f}s")
    if len(seen) < 3:
        print("  FAIL: the pad reading barely changed. Either nothing was pressed, or the offset is wrong.")
        print("  Do NOT record until this shows your presses. docs/RAM_MAP.md section 4.")
        return False
    print("  OK: inputs are being captured.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="3,4,5,6,7,8", help="start slots to cycle through")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=240.0, help="per-episode cap; the human win took 167 s")
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    ap.add_argument("--out", default="datasets/human/gohma")
    ap.add_argument("--probe", action="store_true", help="10 s input check, record nothing")
    ap.add_argument("--headless", action="store_true", help="Null backend (for testing only -- you need video)")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]
    env = build_env(slots, a.slots_dir, a.seconds, not a.headless)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    try:
        if not probe(env, 10.0):
            return 1
        if a.probe:
            return 0
        for ep in range(a.episodes):
            slot = slots[ep % len(slots)]
            obs = env.reset(slot)
            O, A, R, PAD = [], [], [], []
            print(f"\n=== episode {ep + 1}/{a.episodes}, slot {slot}. Play. Ctrl-C to stop recording.")
            t0, info = time.time(), {}
            while True:
                b, x, y = read_pad(env.ram)
                act = to_action(b, x, y, env.state.cam_yaw)
                O.append(np.asarray(obs, np.float32)); A.append(act); PAD.append((b, x, y))
                env.send_action(act)                      # mirrors what the human pressed, so state advances
                while not env.ready():
                    time.sleep(0.0005)
                obs, r, term, trunc, info = env.observe()
                R.append(r)
                if term or trunc:
                    break
            st = read_combat_state(env.ram)
            ep_info = info.get("episode", {})
            won = bool(ep_info.get("boss_defeated"))
            died = bool(st.bit(SF1_DEAD))
            f = out / f"ep{ep:02d}_slot{slot}_{'WIN' if won else 'death' if died else 'timeout'}.npz"
            np.savez_compressed(f, obs=np.stack(O), action=np.array(A, np.int64),
                                reward=np.array(R, np.float32), pad=np.array(PAD, np.int32),
                                slot=slot, boss_damage=ep_info.get("boss_damage", 0),
                                boss_defeated=won, length_s=len(A) / 10.0)
            kept.append((f.name, won, ep_info.get("boss_damage", 0), len(A) / 10.0))
            print(f"  -> {f.name}  {len(A)} steps, {len(A)/10.0:.0f}s, boss damage "
                  f"{ep_info.get('boss_damage', 0)}/10, {'WIN' if won else 'no kill'}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            env.close()
        except Exception:                                                        # noqa: BLE001
            pass
    if kept:
        wins = sum(1 for _, w, _, _ in kept if w)
        (out / "MANIFEST.json").write_text(json.dumps(
            {"episodes": [{"file": n, "win": w, "boss_damage": d, "length_s": s} for n, w, d, s in kept],
             "wins": wins, "total": len(kept),
             "action_space": "COMBAT_ACTIONS_V5 (108)", "obs_version": 7,
             "pad_source": f"gPlayState+0x14 Input[0] (docs/RAM_MAP.md section 4)"}, indent=2))
        print(f"\n{len(kept)} episodes, {wins} wins -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
