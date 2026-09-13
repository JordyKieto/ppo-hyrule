import os, sys, time, shutil
from pathlib import Path
S = "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad"
os.environ.setdefault("DOLPHIN_BIN", f"{S}/Dolphin.app/Contents/MacOS/Dolphin")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.machmem import MachTask, remap, find_mem1_views
from zelda_rl.savestate import N64Ram
from zelda_rl.oot_ram import read_features
ud = Path("runs/instances/mach"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud)
t0 = time.time(); t = None
while t is None or not find_mem1_views(t):
    time.sleep(0.5)
    try: t = t or MachTask(inst.proc.pid)
    except PermissionError: pass
mv = memoryview(remap(t, find_mem1_views(t)[0] + 0xF64120, 0x400000)).cast("B")
print(f"MEM1 found at {time.time()-t0:.1f}s")
scene = lambda: read_features(N64Ram(mv))["scene"]
last = None
for i in range(40):
    s = scene()
    if s != last: print(f"{time.time()-t0:5.1f}s scene={s}", flush=True); last = s
    if s == 52: break
    if i % 4 == 0:
        print(f"{time.time()-t0:5.1f}s press load (hold 0.2)", flush=True)
        inst.hotkey("Load State/Load State Slot 1", 0.2)
    time.sleep(0.5)
print("final scene", scene(), "log tail:", (ud / "dolphin_stdout.log").read_text()[-500:])
inst.stop()
