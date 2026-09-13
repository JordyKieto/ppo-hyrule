"""Calibrate camera-relative stick -> world movement: hold stick up / right, compare Link's world motion to camera yaw."""
import math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.obs import read_state_file_v2, read_state_v2
D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
files = {k: D / f"NACE01.s{k:02d}" for k in (2, 5)}
states = {k: read_state_file_v2(f) for k, f in files.items()}
env = DolphinGoalEnv(0, Path("runs/instances/calib"), files, states, states[5], EnvConfig(video_backend="Null", obs_version=2))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
for name, (x, y), stick_angle in [("up", (0.5, 1.0), math.pi / 2), ("right", (1.0, 0.5), 0.0), ("left", (0.0, 0.5), math.pi), ("down", (0.5, 0.0), -math.pi / 2)]:
    env.load_slot(2); time.sleep(0.2)
    s0 = read_state_v2(env.ram)
    env.inst.pad.send(f"SET MAIN {x} {y}")
    vi0 = s0.vi
    while read_state_v2(env.ram).vi < vi0 + 20: time.sleep(0.002)
    env.inst.pad.send("SET MAIN 0.5 0.5")
    s1 = read_state_v2(env.ram)
    move = math.atan2(s1.pos[2] - s0.pos[2], s1.pos[0] - s0.pos[0])
    rel = wrap(move - s0.cam_yaw)
    k = wrap(rel) / wrap(stick_angle - math.pi / 2) if abs(wrap(stick_angle - math.pi / 2)) > 0.1 else float("nan")
    print(f"stick {name:5s}: moved {math.dist(s0.pos, s1.pos):5.1f} units, world angle {math.degrees(move):7.1f}, cam yaw {math.degrees(s0.cam_yaw):7.1f}, "
          f"move-cam {math.degrees(rel):7.1f} deg (stick-up offset {math.degrees(wrap(stick_angle - math.pi/2)):6.1f}) -> handedness ~{k:+.2f}")
env.close()
