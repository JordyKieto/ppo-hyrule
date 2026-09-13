"""Verify savestate slots on this machine's Dolphin: each loads (RAM matches the file) and the game sees stick input.

    [xvfb-run -a] python scripts/verify_slots.py [--slots-dir DIR] [--slots 1,2,3,...]   -> prints a table, writes slots_ok.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.obs import read_state, read_state_file_v2
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
ap.add_argument("--slots", default=None, help="default: every NACE01.sNN present (1-10)")
ap.add_argument("--out", default="slots_ok.json")
a = ap.parse_args()
D = Path(a.slots_dir)
slots = [int(x) for x in a.slots.split(",")] if a.slots else [k for k in range(1, 11) if (D / f"NACE01.s{k:02d}").exists()]
files = {k: D / f"NACE01.s{k:02d}" for k in slots}
states = {k: read_state_file_v2(f) for k, f in files.items()}
env = DolphinGoalEnv(0, Path("runs/instances/verify_slots"), files, states, states[slots[-1]],
                     EnvConfig(video_backend="Null", unlimited_speed=True, obs_version=2))
results = {}
for k in slots:
    exp = states[k]
    row = {"slot": k, "scene": exp.scene, "rupees": exp.rupees, "pos": [round(v) for v in exp.pos]}
    try:
        t = env.load_slot(k)
        s = read_state(env.ram)
        row["load_ms"] = round(t * 1000)
        row["ram_matches_file"] = s.scene == exp.scene and s.rupees == exp.rupees and math.dist(s.pos, exp.pos) < 2.0
        chk = env.input_check(wait_vi=60)
        row["game_sees_stick"] = chk["game_sees_stick"]
        p0 = read_state(env.ram).pos
        env.inst.pad.send("SET MAIN 0.5 1.0"); time.sleep(0.4); env.inst.pad.send("SET MAIN 0.5 0.5")
        row["link_moved"] = round(math.dist(read_state(env.ram).pos, p0), 1)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)
    row["ok"] = bool(row.get("ram_matches_file") and row.get("game_sees_stick"))
    results[k] = row
    print(json.dumps(row), flush=True)
env.close()
ok = [k for k, r in results.items() if r["ok"]]
print(f"\nOK slots: {ok} | FAILED: {[k for k in slots if k not in ok]}")
Path(a.out).write_text(json.dumps({"ok": ok, "results": results}, indent=2))
