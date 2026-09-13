"""Boot OoT in a sandboxed instance, load File 1 until the HUD is visible, save state -> runs/states/<name>.sav"""
import sys, time, shutil
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.frames import DumpDirFrames
from zelda_rl.hud import in_gameplay

name = sys.argv[1] if len(sys.argv) > 1 else "house"
ud = Path("runs/instances/mkstate")
shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve())
inst = DolphinInstance(ud)
frames = DumpDirFrames(ud)
t0 = time.time()
time.sleep(10)
hud_seen = 0
while time.time() - t0 < 120:
    f = frames.latest()
    if f and in_gameplay(f[1]):
        hud_seen += 1
        if hud_seen >= 3:
            break
    else:
        hud_seen = 0
        inst.tap("A", 0.3)   # A skips attract/title, opens file 1, confirms "Yes"
    time.sleep(2.5)
time.sleep(2)
inst.hotkey("Save State/Save State Slot 1", 0.3); time.sleep(3)
states = list((ud / "StateSaves").glob("*.s01"))
print("states:", states, f"{time.time()-t0:.1f}s")
out = Path("runs/states"); out.mkdir(parents=True, exist_ok=True)
if states:
    shutil.copy(states[0], out / f"{name}.sav")
f = frames.latest()
Image.fromarray(f[1]).save(out / f"{name}.png")
inst.stop()
