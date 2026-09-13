"""Tiny CLI to poke a running instance: ctl.py <inst> <cmd...>  then saves latest frame.
cmds: tap:BUTTON[:secs]  hk:NAME  set:MAIN:x:y[:secs]  wait:secs"""
import sys, time, os, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import Pipe, HOTKEY_MAP
ud = Path("runs/instances") / sys.argv[1]
pad, hk = Pipe(ud / "Pipes/pad"), Pipe(ud / "Pipes/hk")
frames = ud / "Dump/Frames"
def latest():
    fs = sorted(frames.glob("framedump_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    return fs
for c in sys.argv[2:]:
    parts = c.split(":")
    if parts[0] == "tap":
        pad.send(f"PRESS {parts[1]}"); time.sleep(float(parts[2]) if len(parts) > 2 else 0.1); pad.send(f"RELEASE {parts[1]}")
    elif parts[0] == "hk":
        b = HOTKEY_MAP[parts[1]]; hk.send(f"PRESS {b}"); time.sleep(0.15); hk.send(f"RELEASE {b}")
    elif parts[0] == "set":
        pad.send(f"SET {parts[1]} {parts[2]} {parts[3]}"); time.sleep(float(parts[4]) if len(parts) > 4 else 0.5); pad.send(f"SET {parts[1]} 0.5 0.5")
    elif parts[0] == "wait":
        time.sleep(float(parts[1]))
time.sleep(0.3)
fs = latest()
if fs:
    shutil.copy(fs[-2] if len(fs) > 1 else fs[-1], "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad/latest.png")
    for f in fs[:-5]: f.unlink()
print(len(fs), "frames seen; latest", fs[-1].name if fs else None)
