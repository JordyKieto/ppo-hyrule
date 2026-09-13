"""Reset reliability + latency, verified from live RAM: walk away, load state, time until Link is back at spawn."""
import os, sys, time, shutil, struct, statistics as st
from pathlib import Path
S = "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad"
os.environ.setdefault("DOLPHIN_BIN", f"{S}/Dolphin.app/Contents/MacOS/Dolphin")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance, HOTKEY_MAP
from zelda_rl.machmem import MachTask, remap, find_mem1_views
ud = Path("runs/instances/mach"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud); t = None
while t is None or not find_mem1_views(t):
    time.sleep(0.5)
    try: t = t or MachTask(inst.proc.pid)
    except PermissionError: pass
mv = memoryview(remap(t, find_mem1_views(t)[0] + 0xF64120, 0x400000)).cast("B")
def pos():
    p = struct.unpack_from(">I", mv, (0x801C8D60 + 0x1C44) & 0x3FFFFF)[0]
    return struct.unpack_from(">fff", mv, (p & 0x3FFFFF) + 0x24) if p >> 24 == 0x80 else (9e9, 9e9, 9e9)
at_spawn = lambda: (lambda x, y, z: abs(x - 1.0) < 0.5 and abs(z - 95.0) < 0.5)(*pos())
key = HOTKEY_MAP["Load State/Load State Slot 1"]
def reset(open_loop_hold=None, timeout=3.0):
    """open_loop_hold: fixed pulse; None = closed loop (hold until RAM confirms)."""
    t0 = time.perf_counter(); inst.hk.send(f"PRESS {key}")
    released = False
    while time.perf_counter() - t0 < timeout:
        if open_loop_hold is not None and not released and time.perf_counter() - t0 >= open_loop_hold:
            inst.hk.send(f"RELEASE {key}"); released = True
        if at_spawn():
            if not released: inst.hk.send(f"RELEASE {key}")
            return time.perf_counter() - t0
        time.sleep(0.0005)
    if not released: inst.hk.send(f"RELEASE {key}")
    return None
assert reset(None, 20) is not None, "initial load failed"
time.sleep(0.5)
for mode in (0.02, 0.2, None):
    lat, fails = [], 0
    for i in range(15):
        inst.pad.send("SET MAIN 0.5 0.0"); time.sleep(0.6); inst.pad.send("SET MAIN 0.5 0.5")
        assert not at_spawn()
        r = reset(mode)
        if r is None: fails += 1
        else: lat.append(r)
        time.sleep(0.3 if r else 1.0)
        if r is None: reset(None, 5)
    name = "closed-loop" if mode is None else f"pulse {mode*1e3:.0f}ms"
    print(f"{name:12s}: {15-fails}/15 ok, press->Link at spawn p50={st.median(lat)*1e3:.0f}ms p90={sorted(lat)[int(len(lat)*.9)]*1e3 if lat else 0:.0f}ms max={max(lat)*1e3 if lat else 0:.0f}ms", flush=True)
inst.stop()
