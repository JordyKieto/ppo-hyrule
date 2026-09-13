import sys, time, shutil, struct, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.gdb import GDB
N64 = 0x80F64120
mode = sys.argv[1]  # "noload" or "load"
ud = Path("runs/instances/gdb"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud, extra=["-C", "Dolphin.General.GDBPort=2345"])
for _ in range(60):
    try: g = GDB(port=2345); break
    except OSError: time.sleep(0.25)
g.s.settimeout(3); g.cmd("?"); g.cont(); time.sleep(12)
if mode == "load":
    inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(3)
lat = []
for k in range(30):
    inst.pad.send("SET MAIN 0.5 1.0" if k < 15 else "SET MAIN 0.5 0.0")
    try:
        t0 = time.perf_counter(); g.interrupt(); t1 = time.perf_counter()
    except TimeoutError:
        print(f"k={k} interrupt timed out"); break
    sc = g.read(N64 + 0x11AC80, 0x40)
    player = struct.unpack(">I", g.read(N64 + 0x1C8D60 + 0x1C44, 4))[0]
    pos = struct.unpack(">fff", g.read(N64 + (player & 0x3FFFFF) + 0x24, 12)) if player >> 24 == 0x80 else None
    t2 = time.perf_counter(); g.cont(); t3 = time.perf_counter()
    lat.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
    if k % 5 == 0 or k == 29:
        print(f"k={k} magic={sc[0x1C:0x22]} scene?rupees={struct.unpack_from('>h', sc, 0x34)[0]} pos={tuple(round(v,1) for v in pos) if pos else None}", flush=True)
    time.sleep(0.05)
if lat:
    for name, i in (("halt (\\x03 -> stop reply)", 0), ("3 reads (0x40+4+12 B)", 1), ("continue", 2), ("total", 3)):
        print(f"{name}: p50={st.median([x[i] for x in lat])*1e3:.2f}ms p90={sorted(x[i] for x in lat)[int(len(lat)*.9)]*1e3:.2f}ms")
    g.interrupt(); t = time.perf_counter(); blob = g.read(N64, 0x400000); print(f"full 4MB N64 RAM read: {time.perf_counter()-t:.2f}s head={blob[:4].hex()}")
inst.stop()
