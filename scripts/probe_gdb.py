import sys, time, shutil, subprocess, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.gdb import GDB
ud = Path("runs/instances/gdb"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud, extra=["-C", "Dolphin.General.GDBPort=2345"])
g = None
for _ in range(60):
    try: g = GDB(port=2345); break
    except OSError: time.sleep(0.25)
print("connected:", g is not None, flush=True)
g.s.settimeout(3)
print("? ->", g.cmd("?"), flush=True)
try: print("m80000000 ->", g.read(0x80000000, 8).hex(), flush=True)
except Exception as e: print("early read:", e)
g.cont(); time.sleep(12)
print("interrupt ->", g.interrupt(), flush=True)
print("N64 @0x80F64120:", g.read(0x80F64120, 12).hex(), " @0x80003180:", g.read(0x80003180, 4), flush=True)
g.cont()
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(3)
N64 = 0x80F64120
def feats():
    sc = g.read(N64 + 0x11AC80, 0x40); ps_player = int.from_bytes(g.read(N64 + 0x1C8D60 + 0x1C44, 4), "big")
    import struct
    pos = struct.unpack(">fff", g.read(N64 + (ps_player & 0x3FFFFF) + 0x24, 12)) if ps_player >> 24 == 0x80 else None
    return sc[0x1C:0x22], struct.unpack_from(">h", sc, 0x34)[0], pos
lat = []
for k in range(20):
    inst.pad.send(f"SET MAIN 0.5 1.0" if k < 10 else "SET MAIN 0.5 0.0")
    t0 = time.perf_counter(); g.interrupt(); t1 = time.perf_counter()
    magic, rupees, pos = feats(); t2 = time.perf_counter(); g.cont(); t3 = time.perf_counter()
    lat.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
    print(f"k={k} {magic} rupees={rupees} pos={pos} halt={1e3*(t1-t0):.1f}ms read={1e3*(t2-t1):.1f}ms", flush=True)
    time.sleep(0.05)
inst.pad.send("SET MAIN 0.5 0.5")
for name, i in (("halt", 0), ("read 3 regions", 1), ("continue", 2), ("total", 3)):
    print(f"{name}: p50={st.median([x[i] for x in lat])*1e3:.2f}ms")
g.interrupt(); t = time.perf_counter(); blob = g.read(N64, 0x400000); print(f"full 4MB RAM read: {time.perf_counter()-t:.2f}s ok={blob[:4].hex()}"); g.cont()
inst.stop()
