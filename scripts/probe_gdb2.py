import sys, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.gdb import GDB
ud = Path("runs/instances/gdb"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
inst = DolphinInstance(ud, extra=["-C", "Dolphin.General.GDBPort=2345"])
for _ in range(60):
    try: g = GDB(port=2345); break
    except OSError: time.sleep(0.25)
g.s.settimeout(3)
g.cmd("?"); g.cont(); time.sleep(12)
print("interrupt ->", g.interrupt())
for a in (0x80F64120, 0x80003180, 0x80000000, 0x90000000, 0x80F64120):
    r = g.cmd(f"m{a:x},c"); print(hex(a), r)
inst.stop()
