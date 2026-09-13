import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.obs import read_state_file, read_state
from zelda_rl.dolphin import HOTKEY_MAP
D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
files = {k: D / f"NACE01.s{k:02d}" for k in (1, 2, 3, 5)}
states = {k: read_state_file(f) for k, f in files.items()}
print({k: s.vi for k, s in states.items()})
cfg = EnvConfig(unlimited_speed=True)
env = DolphinGoalEnv(0, Path("runs/instances/dbg0"), files, states, states[5], cfg)
print("boot load ok; vi now", read_state(env.ram).vi, "scene", read_state(env.ram).scene)
for slot in (1, 2, 1, 3):
    time.sleep(1.0)
    key = HOTKEY_MAP[f"Load State/Load State Slot {slot}"]
    pre = read_state(env.ram).vi
    env.inst.hk.send(f"PRESS {key}"); t0 = time.perf_counter(); trace = []
    while time.perf_counter() - t0 < 0.8:
        trace.append((round((time.perf_counter()-t0)*1000), read_state(env.ram).vi)); time.sleep(0.01)
    env.inst.hk.send(f"RELEASE {key}")
    jumps = [(ms, v) for (ms, v), (_, pv) in zip(trace[1:], trace) if abs(v - pv) > 30]
    print(f"slot {slot}: target vi {states[slot].vi} pre {pre} first {trace[0]} last {trace[-1]} jumps {jumps[:3]}", flush=True)
env.close()
