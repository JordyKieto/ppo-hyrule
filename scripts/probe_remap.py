import os, sys, time, shutil, struct, statistics as st
from pathlib import Path
import numpy as np
S = "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad"
os.environ.setdefault("DOLPHIN_BIN", f"{S}/Dolphin.app/Contents/MacOS/Dolphin")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.machmem import MachTask, remap, find_mem1_views
from zelda_rl.savestate import N64Ram
from zelda_rl.oot_ram import read_features
MAGIC = bytes.fromhex("3c1a8000275a260003400008")
ud = Path("runs/instances/mach"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud); time.sleep(18)
t = MachTask(inst.proc.pid)
cands = find_mem1_views(t)
print("MEM1 views:", [hex(a) for a in cands])
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(3)
views = [remap(t, a + 0xF64120, 0x400000) for a in cands]
print("post-load scene via view:", read_features(N64Ram(memoryview(views[0]).cast("B")))["scene"])
arrs = [np.frombuffer(v, np.uint8) for v in views]
live = [i for i, a in enumerate(arrs) if bytes(a[:12]) == MAGIC]
print("remapped views with magic:", live)
ram = views[live[0]]
# is it live? read Link pos while walking, via zero-copy view
mv = memoryview(ram).cast("B")
def pos_fast():
    player = struct.unpack_from(">I", mv, (0x801C8D60 + 0x1C44) & 0x3FFFFF)[0]
    return struct.unpack_from(">h", mv, (0x8011AC80 + 0x34) & 0x3FFFFF)[0], struct.unpack_from(">fff", mv, (player & 0x3FFFFF) + 0x24)
lat = []
for k in range(40):
    inst.pad.send("SET MAIN 0.5 1.0" if k < 20 else "SET MAIN 0.0 0.5")
    a = time.perf_counter(); r = pos_fast(); lat.append(time.perf_counter() - a)
    if k % 10 == 0: print(k, r, "| all views agree:", len({bytes(x[0x11AC80:0x11AD00]) for x in (arrs[i] for i in live)}) == 1, flush=True)
    time.sleep(0.05)
inst.pad.send("SET MAIN 0.5 0.5")
print(f"zero-copy rupees+pos read: p50={st.median(lat)*1e6:.1f}us p90={sorted(lat)[36]*1e6:.1f}us")
ts = []
for _ in range(200):
    a = time.perf_counter(); f = read_features(N64Ram(mv)); ts.append(time.perf_counter() - a)
print(f"full read_features() incl. actor-list walk: p50={st.median(ts)*1e6:.0f}us -> {f['scene']=} {f['rupees']=} {f['link_pos']=} {f['actor_counts']['PLAYER']=}")
ts = []
for _ in range(200):
    a = time.perf_counter(); snap = arrs[live[0]].copy(); ts.append(time.perf_counter() - a)
print(f"4MB snapshot memcpy from view: p50={st.median(ts)*1e3:.2f}ms")
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(2)
print("after another state load, view still live:", pos_fast(), "scene", read_features(N64Ram(mv))["scene"])
inst.stop()
