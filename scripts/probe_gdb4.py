import sys, time, shutil, struct, socket
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.gdb import GDB
N64 = 0x80F64120
ud = Path("runs/instances/gdb"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud, extra=["-C", "Dolphin.General.GDBPort=2345"])
for _ in range(60):
    try: g = GDB(port=2345); break
    except OSError: time.sleep(0.25)
g.s.settimeout(2); g.cmd("?"); g.cont(); time.sleep(12)
def pos():
    player = struct.unpack(">I", g.read(N64 + 0x1C8D60 + 0x1C44, 4))[0]
    return struct.unpack_from(">h", g.read(N64 + 0x11AC80 + 0x34, 2))[0], struct.unpack(">fff", g.read(N64 + (player & 0x3FFFFF) + 0x24, 12))
print("before load:", g.interrupt(), pos())
# Variant A: load state while halted (hotkey thread), then continue
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(1)
g.cont(); time.sleep(2)
try:
    print("A: interrupt after load-while-halted:", g.interrupt(), pos()); g.cont()
except Exception as e:
    print("A failed:", type(e).__name__, e)
    try:
        g.s.close(); g = GDB(port=2345); g.s.settimeout(2); print("reconnect ?:", g.cmd("?"))
        print("after reconnect interrupt:", g.interrupt(), pos()); g.cont()
    except Exception as e2:
        print("reconnect failed:", type(e2).__name__, e2)
print("dolphin alive:", inst.proc.poll() is None)
inst.stop()
