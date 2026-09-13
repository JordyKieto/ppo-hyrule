"""Why do state loads fail? Log Dolphin core messages; try house.sav vs user slot; optional Null video backend."""
import os, sys, time, shutil, struct
from pathlib import Path
S = "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad"
os.environ.setdefault("DOLPHIN_BIN", f"{S}/Dolphin.app/Contents/MacOS/Dolphin")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance, HOTKEY_MAP
from zelda_rl.machmem import MachTask, remap, find_mem1_views
backend = sys.argv[1]; state = sys.argv[2]
ud = Path(f"runs/instances/load_{backend}"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
(ud / "Config/Logger.ini").write_text("[Options]\nWriteToFile = True\nVerbosity = 3\n[Logs]\nCORE = True\nBOOT = True\nMASTER = True\n")
shutil.copy(os.path.expanduser(state), ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud, video_backend=backend); t0 = time.time(); t = None
while t is None or not find_mem1_views(t):
    time.sleep(0.25)
    try: t = t or MachTask(inst.proc.pid)
    except PermissionError: pass
    if time.time() - t0 > 40: print("no MEM1"); inst.stop(); sys.exit()
view = remap(t, find_mem1_views(t)[0] + 0xF64120, 0x400000)
scene = lambda: struct.unpack_from(">h", view, (0x801C8D60 + 0xA4) & 0x3FFFFF)[0]
vi = lambda: struct.unpack_from(">I", view, 0x8011DBB0 & 0x3FFFFF)[0]
print(f"MEM1 at {time.time()-t0:.1f}s", flush=True)
k = HOTKEY_MAP["Load State/Load State Slot 1"]; ok = None
for i in range(25):
    v0 = vi(); time.sleep(0.5); rate = (vi() - v0) * 2
    print(f"{time.time()-t0:5.1f}s scene={scene()} vi_rate={rate}/s", flush=True)
    if scene() == 85 or scene() == 52:
        ok = time.time() - t0; break
    inst.hk.send(f"PRESS {k}"); time.sleep(0.2); inst.hk.send(f"RELEASE {k}")
print("loaded at", ok)
log = ud / "Logs/dolphin.log"
if log.exists():
    lines = [l for l in log.read_text(errors="replace").splitlines() if any(w in l for w in ("State", "state", "Load", "rror"))]
    print("\n".join(lines[-25:]))
inst.stop()
