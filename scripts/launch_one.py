import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
ud = build_user_dir(Path("runs/instances/i0").resolve())
inst = DolphinInstance(ud)
print("pid", inst.proc.pid, flush=True)
time.sleep(20)
