"""Make a new start state by loading an existing slot, walking Link around for a few seconds, and saving.

Runs a headless Dolphin. Walks a sequence of world directions (camera-relative stick, as the RL env does), stops if a
cutscene starts, and saves only if Link is alive, out of cutscenes, has moved at least --min-move units and the game
still reads stick input. The old destination slot file is backed up first.

    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/make_walk_state.py --source-slot 2 --dest-slot 10
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import SF1_CUTSCENE, read_combat_state  # noqa: E402
from zelda_rl.dolphin import HOTKEY_MAP  # noqa: E402
from zelda_rl.env import DolphinGoalEnv, EnvConfig  # noqa: E402
from zelda_rl.gohma import find_gohma, gohma_state  # noqa: E402
from zelda_rl.obs import read_state_file_v2, read_state_v2  # noqa: E402

D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-slot", type=int, required=True)
    ap.add_argument("--dest-slot", type=int, required=True)
    ap.add_argument("--path", default="N:1.5,E:1.0,S:0.8,W:1.2", help="world direction:seconds, comma separated")
    ap.add_argument("--min-move", type=float, default=150.0)
    args = ap.parse_args()
    dirs = {"E": 0.0, "NE": math.pi / 4, "N": math.pi / 2, "NW": 3 * math.pi / 4, "W": math.pi, "SW": 5 * math.pi / 4,
            "S": 3 * math.pi / 2, "SE": 7 * math.pi / 4}

    files = {args.source_slot: D / f"NACE01.s{args.source_slot:02d}"}
    states = {k: read_state_file_v2(f) for k, f in files.items()}
    env = DolphinGoalEnv(0, Path("runs/instances/make_walk_state"), files, states, states[args.source_slot],
                         EnvConfig(video_backend="Null", unlimited_speed=False, obs_version=2))
    try:
        env.load_slot(args.source_slot)
        chk = env.input_check(wait_vi=60)
        if not chk["game_sees_stick"]:
            sys.exit(f"source slot {args.source_slot} ignores stick input: {chk}")
        env.load_slot(args.source_slot)
        if read_combat_state(env.ram).held_item == 15:          # aiming the slingshot: the stick aims instead of walking
            env.inst.pad.send("PRESS B"); time.sleep(0.1); env.inst.pad.send("RELEASE B"); time.sleep(0.6)
        s0 = read_combat_state(env.ram)
        for leg in args.path.split(","):
            name, secs = leg.split(":")
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < float(secs):
                env.state = read_state_v2(env.ram)
                x, y = env._stick(dirs[name])
                env.inst.pad.send(f"SET MAIN {x:.3f} {y:.3f}")
                s = read_combat_state(env.ram)
                if s.dead or s.bit(SF1_CUTSCENE):
                    break
                time.sleep(0.05)
            if s.dead or s.bit(SF1_CUTSCENE):
                break
        env.inst.pad.send("SET MAIN 0.5 0.5")
        time.sleep(0.3)
        s = read_combat_state(env.ram)
        g = find_gohma(s)
        moved = math.dist(s.pos, s0.pos)
        info = (f"Link {s.hp / 16:.1f} hearts at {[round(v) for v in s.pos]} (moved {moved:.0f}), cutscene {s.bit(SF1_CUTSCENE)}, "
                f"Gohma {g.hp if g else '-'} HP {gohma_state(g) if g else '-'} d {math.dist(g.pos, s.pos) if g else 0:.0f}")
        print("after walk:", info, flush=True)
        if s.dead or s.bit(SF1_CUTSCENE) or moved < args.min_move:
            sys.exit("not saving: Link dead / in a cutscene / did not move enough")
        key = HOTKEY_MAP["Save State/Save State Slot 1"]
        saved_file = env.inst.user_dir / "StateSaves" / "NACE01.s01"
        before = saved_file.stat().st_mtime if saved_file.exists() else 0
        env.inst.hk.send(f"PRESS {key}"); time.sleep(0.1); env.inst.hk.send(f"RELEASE {key}")
        t0 = time.time()
        while time.time() - t0 < 20 and (not saved_file.exists() or saved_file.stat().st_mtime == before):
            time.sleep(0.2)
        time.sleep(2.0)
        # verify the new file loads and reads input before replacing the user's slot
        env.slot_states[1] = read_state_file_v2(saved_file)
        env.load_slot(1)
        chk = env.input_check(wait_vi=60)
        if not chk["game_sees_stick"]:
            sys.exit(f"new state ignores stick input: {chk}")
        dest = D / f"NACE01.s{args.dest_slot:02d}"
        backup = Path("runs/manual/replaced_states") / f"NACE01.s{args.dest_slot:02d}.{time.strftime('%Y%m%d-%H%M%S')}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.copy(dest, backup)
            print(f"backed up old slot {args.dest_slot} -> {backup}", flush=True)
        shutil.copy(saved_file, dest)
        print(f"wrote slot {args.dest_slot}: {dest} ({info}); input check ok", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
