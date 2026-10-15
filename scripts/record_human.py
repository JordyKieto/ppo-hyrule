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
      .venv/bin/python-dbg scripts/record_human.py --probe          # 10 s input check first, records nothing

    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/record_human.py    # 20 episodes from the pre-fight states

Defaults are the five recovered PRE-FIGHT savestates in assets/prefight/ (Gohma in `ceiling_wait`, so you
look up to trigger the cutscene and fight her end to end). Slot numbers must be 1-10 because the env loads
via Dolphin's "Load State Slot N" hotkey, which only exists for those; the number is just which of the
INSTANCE'S OWN StateSaves to load, so any state can sit at any number and your real StateSaves is untouched.

    ... --probe        # 10 s input check: hold buttons, confirm they are being read, write nothing
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import struct
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

OBS_FIELDS: list = []          # filled in main() once the layout is known
ACTION_SPEC: list = []
try:
    GIT = __import__("subprocess").run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
except Exception:              # noqa: BLE001
    GIT = "unknown"

# gPlayState + 0x14 = Input[4], 0x14 bytes each. docs/RAM_MAP.md section 4, status VERIFIED.
INPUT0 = PLAY_STATE + 0x14

# N64 button bits as the VC build reports them in cur.button. Verified empirically by --probe before any
# recording is kept: a constant or all-zero reading means the offset is wrong and the run aborts.
# VERIFIED BY --selftest, which drives the pad through the env's own pipe and reads these back. The names are
# the env's button names (COMBAT_BUTTONS_V5), NOT the N64 pad's:
#
#   "L" is the env's Z-TARGET button and the game sees it on bit 0x2000 -- the N64 Z bit. docs/RAM_MAP.md:249
#   says so in words ("L held = Z-target"), and the first version of this table mapped "L" to 0x0020 and
#   called 0x2000 "Z". Every Z-targeted action -- which is most of this fight -- would have decoded as
#   no-button, silently, and the dataset would have taught the policy never to lock on.
BITS = {"A": 0x8000, "B": 0x4000, "L": 0x2000, "START": 0x1000,
        "DU": 0x0800, "DD": 0x0400, "DL": 0x0200, "DR": 0x0100,
        "R": 0x0010, "CU": 0x0008, "CD": 0x0004, "CL": 0x0002, "CR": 0x0001}


def read_pad(ram) -> tuple[int, int, int]:
    """(button bitmask, stick_x, stick_y) from controller 0.

    `env.ram` is indexed by ARENA OFFSET, not by absolute address -- every reader in zelda_rl/obs.py masks
    with `& 0x3FFFFF`. Slicing with the raw 0x801C8D74 returned an empty buffer and the probe crashed on an
    empty array, which is the loud version of this mistake; the quiet version would have been reading garbage
    from a wrapped offset. Uses struct.unpack_from with the same mask as _u32/_s16 rather than re-deriving it.
    """
    b = struct.unpack_from(">H", ram, INPUT0 & 0x3FFFFF)[0]
    x, y = struct.unpack_from(">bb", ram, (INPUT0 + 2) & 0x3FFFFF)
    return int(b), int(x), int(y)


def read_cam_yaw(ram) -> float:
    """Camera yaw read FRESH from RAM, the same way zelda_rl/obs.py:178 computes it.

    `env.state` is None until the first reset() -- the probe crashed on that -- and after a reset it is only
    as fresh as the last observe(), i.e. one step stale. The action decode depends on the yaw AT THE MOMENT
    the stick is read, so it is read directly every frame rather than taken from the cached state.
    """
    from zelda_rl.obs import MAIN_CAMERA
    at = struct.unpack_from(">fff", ram, (MAIN_CAMERA + 0x50) & 0x3FFFFF)
    eye = struct.unpack_from(">fff", ram, (MAIN_CAMERA + 0x5C) & 0x3FFFFF)
    import math
    return math.atan2(at[2] - eye[2], at[0] - eye[0])


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


def selftest(env, slot) -> bool:
    """Prove the input offset WITHOUT a controller, by driving the pad through the env's own pipe.

    `send_action` writes PRESS/RELEASE and SET MAIN to Dolphin's pad pipe. If gPlayState+0x14 is the real
    Input[0], those presses must appear there on the next frames. This closes the loop end to end -- offset,
    button bits and the world/stick direction inversion -- with nothing but the emulator, so a wrong offset
    can never reach a recording session.
    """
    import time as _t

    from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTONS_V5
    env.reset(slot)
    print("\nSELFTEST: driving the pad through the pipe and reading it back from gPlayState+0x14\n")
    cases = []
    for want in (("B",), ("L",), ("R",), ("A",), ("L", "B"), ("L", "R"), ("CD",), ("CL",),
                 ("L", "CD"), ("L", "CL")):
        a = next(i for i, (d, bi) in enumerate(COMBAT_ACTIONS_V5)
                 if d is None and set(COMBAT_BUTTONS_V5[bi]) == set(want))
        cases.append((a, set(want)))
    ok = True
    for a, want in cases:
        for _ in range(3):                       # a few frames for the press to register
            env.send_action(a)
            while not env.ready():
                _t.sleep(0.0005)
            env.observe()
        b, x, y = read_pad(env.ram)
        got = set(held(b)) & {"A", "B", "L", "R", "CD", "CL"}
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  sent {sorted(want)!s:<16} read 0x{b:04X} -> {sorted(got)}")
    # and a direction: send a world angle, confirm the decoded action round-trips
    import math
    for a, lbl in [(next(i for i, (d, bi) in enumerate(COMBAT_ACTIONS_V5)
                         if d is not None and abs(d) < 1e-6 and bi == 0), "world E")]:
        for _ in range(3):
            env.send_action(a)
            while not env.ready():
                _t.sleep(0.0005)
            env.observe()
        b, x, y = read_pad(env.ram)
        back = to_action(b, x, y, read_cam_yaw(env.ram))
        good = back == a
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  sent action {a} ({lbl}) -> stick({x},{y}) -> decoded {back}")
    env.send_action(0)
    print(f"\n  {'OFFSET AND MAPPING VERIFIED' if ok else 'SELFTEST FAILED -- do not record'}")
    return ok


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


def probe(env, slot, seconds=10.0):
    """Ten seconds of live pad reading. If this is constant, the offset is wrong -- stop before recording."""
    env.reset(slot)              # also initialises env.state, and puts the game in the recorded start state
    from zelda_rl.gohma import find_gohma, gohma_state
    g = find_gohma(read_combat_state(env.ram))
    print(f"\nPROBE on slot {slot}: gohma={gohma_state(g) if g else 'ABSENT'}"
          + ("  <-- PRE-CUTSCENE" if g is not None and gohma_state(g) == "ceiling_wait" else ""))
    print(f"reading gPlayState+0x14 for {seconds:.0f}s. Press buttons and move the stick.\n")
    seen, t0, last = set(), time.time(), None
    while time.time() - t0 < seconds:
        b, x, y = read_pad(env.ram)
        cy = read_cam_yaw(env.ram)
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
    ap.add_argument("--slots", default="1,2,3,4,5",
                    help="start slots to cycle through. MUST be 1-10: Dolphin only exposes "
                         "'Load State Slot 1..10' as hotkeys, and the env loads by hotkey. The slot NUMBER is "
                         "just which of the instance's own StateSaves to load -- the file can be any state, "
                         "and your real ~/Library/.../StateSaves is never touched (env.py:109 copies into a "
                         "per-instance user dir)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=240.0, help="per-episode cap; the human win took 167 s")
    ap.add_argument("--slots-dir", default="assets/prefight",
                    help="default: the recovered PRE-FIGHT states (Gohma in ceiling_wait, cutscene not yet "
                         "triggered). Pass the live dir for mid-fight engaged starts")
    ap.add_argument("--out", default="datasets/human/gohma")
    ap.add_argument("--probe", action="store_true", help="10 s input check, record nothing")
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the input gate. ONLY for testing the save path headless -- a real session must "
                         "pass the probe, or you can record 20 episodes of neutral input")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the input offset with NO controller: drive the pad through the env's pipe and "
                         "read it back from gPlayState+0x14. Records nothing")
    ap.add_argument("--headless", action="store_true", help="Null backend (for testing only -- you need video)")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]

    # Fail here, with the reason, rather than after Dolphin has booted. Dolphin exposes load-state hotkeys for
    # slots 1-10 only; asking for 11 crashed with a bare KeyError after a 60 s boot.
    from zelda_rl.dolphin import HOTKEY_MAP
    have = {int(k.split()[-1]) for k in HOTKEY_MAP if "Load State Slot" in k}
    bad = [k for k in slots if k not in have]
    if bad:
        print(f"slots {bad} cannot be loaded: Dolphin only has Load State Slot hotkeys for {sorted(have)}.")
        print("Rename your savestates to NACE01.s01..s10 inside --slots-dir; the slot number is only which of")
        print("the instance's own StateSaves to load, so any state can live at any number 1-10.")
        return 2
    missing = [k for k in slots if not (Path(a.slots_dir) / f"NACE01.s{k:02d}").exists()]
    if missing:
        print(f"missing savestate files for slots {missing} in {a.slots_dir}")
        return 2

    # The observation layout and the action table, taken from the SAME definitions train_combat.py uses, so
    # the dataset describes itself and a remote job cannot drift from this repo's source (AGENTS 25).
    global OBS_FIELDS, ACTION_SPEC
    from zelda_rl.combat import COMBAT_FIELDS
    from zelda_rl.ego import ACTOR_FIELDS_V7, EGO_V7_FIELDS
    from zelda_rl.gohma import BOSS_FIELDS
    from zelda_rl.obs import GLOBAL_FIELDS_V2, N_ACTORS
    OBS_FIELDS = {"global": [list(x) for x in GLOBAL_FIELDS_V2 + COMBAT_FIELDS + BOSS_FIELDS + EGO_V7_FIELDS],
                  "n_actors": N_ACTORS, "actor": [list(x) for x in ACTOR_FIELDS_V7]}
    ACTION_SPEC = [{"i": i, "world_dir_rad": d, "buttons": list(COMBAT_BUTTONS_V5[b]),
                    "name": action_name(i, "combat")} for i, (d, b) in enumerate(COMBAT_ACTIONS_V5)]

    env = build_env(slots, a.slots_dir, a.seconds, not a.headless)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    try:
        if a.selftest:
            return 0 if selftest(env, slots[0]) else 1
        if not a.skip_probe and not probe(env, slots[0], 10.0):
            return 1
        if a.probe:
            return 0
        for ep in range(a.episodes):
            slot = slots[ep % len(slots)]
            obs = env.reset(slot)
            # Report the start state. A pre-cutscene slot reads `ceiling_wait`; an engaged one does not. This
            # is recorded per episode so a remote trainer can filter by it without re-deriving anything.
            from zelda_rl.gohma import find_gohma, gohma_state, hdist
            st0 = read_combat_state(env.ram)
            g0 = find_gohma(st0)
            start_state = gohma_state(g0) if g0 else "ABSENT"
            start_dist = round(hdist(g0.pos, st0.pos), 1) if g0 else None
            start_hp = g0.hp if g0 else 0
            O, A, R, PAD, CY = [], [], [], [], []
            print(f"\n=== episode {ep + 1}/{a.episodes}  slot {slot}  gohma={start_state} "
                  f"hp={start_hp} dist={start_dist}u"
                  + ("   <-- PRE-CUTSCENE: look up to start the fight" if start_state == "ceiling_wait" else ""))
            t0, info = time.time(), {}
            while True:
                b, x, y = read_pad(env.ram)
                act = to_action(b, x, y, read_cam_yaw(env.ram))
                O.append(np.asarray(obs, np.float32)); A.append(act); PAD.append((b, x, y))
                CY.append(env.state.cam_yaw)
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
            # SELF-DESCRIBING, because this is going to S3 and being trained on remotely: the obs field
            # names, the action-space definition and the pad decoding all travel WITH the data, so a remote
            # job never has to reconstruct the layout from this repo's current source (AGENTS 25, 27).
            np.savez_compressed(
                f,
                obs=np.stack(O).astype(np.float32),          # (T, obs_dim) v7
                action=np.array(A, np.int64),                # (T,) index into COMBAT_ACTIONS_V5
                reward=np.array(R, np.float32),              # (T,)
                pad=np.array(PAD, np.int32),                 # (T, 3) raw (button mask, stick_x, stick_y)
                cam_yaw=np.array(CY, np.float32),            # (T,) the yaw used to decode each action
                slot=np.int32(slot),
                start_gohma_state=start_state, start_gohma_hp=np.int32(start_hp),
                start_dist=np.float32(-1.0 if start_dist is None else start_dist),
                boss_damage=np.int32(ep_info.get("boss_damage", 0)),
                boss_defeated=np.bool_(won), died=np.bool_(died),
                length_s=np.float32(len(A) / 10.0), hz=np.float32(10.0),
                obs_version=np.int32(7), obs_dim=np.int32(len(O[0])),
                n_actions=np.int32(len(COMBAT_ACTIONS_V5)),
                obs_fields=json.dumps(OBS_FIELDS), action_space=json.dumps(ACTION_SPEC),
                pad_bits=json.dumps(BITS),
                pad_source="gPlayState+0x14 Input[0], docs/RAM_MAP.md section 4",
                recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                git_commit=GIT,
            )
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            kept.append((f.name, won, ep_info.get("boss_damage", 0), len(A) / 10.0, slot,
                         start_state, md5, f.stat().st_size))
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
        wins = sum(1 for k in kept if k[1])
        (out / "MANIFEST.json").write_text(json.dumps({
            "dataset": "human Gohma demonstrations with controller inputs",
            "episodes": [{"file": n, "win": w, "boss_damage": d, "length_s": s, "slot": sl,
                          "start_gohma_state": ss, "md5": m, "bytes": by}
                         for n, w, d, s, sl, ss, m, by in kept],
            "total": len(kept), "wins": wins,
            "steps": sum(round(k[3] * 10) for k in kept),
            "obs_version": 7, "hz": 10.0,
            "action_space": "COMBAT_ACTIONS_V5", "n_actions": len(COMBAT_ACTIONS_V5),
            "pad_source": "gPlayState+0x14 Input[0] (docs/RAM_MAP.md section 4)",
            "git_commit": GIT,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verify": "scripts/verify_human_dataset.py <dir>  -- run BEFORE uploading to S3",
        }, indent=2))
        print(f"\n{len(kept)} episodes, {wins} wins, {sum(round(k[3]*10) for k in kept):,} steps -> {out}/")
        print(f"  NEXT: .venv/bin/python scripts/verify_human_dataset.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
