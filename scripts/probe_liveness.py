"""Is the game simulating? Count changed RAM bytes per second (no input) + Link motion under input, per config."""
import os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0], "x"]
exec(open("scripts/probe_frame_counter.py").read().split('mode = sys.argv[1]')[0])
import struct
VI = 0x8011DBB0
configs = [tuple(c.split(":")) for c in os.environ.get("CONFIGS","Metal:0,Null:0,Metal:1,Null:1").split(",")]
configs = [(b, u == "1") for b, u in configs]
for backend, unlimited in configs:
    inst, ram = boot(backend, unlimited, f"live_{backend}_{int(unlimited)}")
    time.sleep(1)
    a = ram[0x100000:].copy(); v0 = be_u32(ram, VI); time.sleep(1.0); b = ram[0x100000:].copy(); v1 = be_u32(ram, VI)
    changed = int((a != b).sum())
    p = be_u32(ram, 0x801C8D60 + 0x1C44)
    pos0 = struct.unpack_from(">fff", ram, (p & 0x3FFFFF) + 0x24)
    inst.pad.send("SET MAIN 0.5 1.0"); time.sleep(1.5); inst.pad.send("SET MAIN 0.5 0.5")
    pos1 = struct.unpack_from(">fff", ram, (p & 0x3FFFFF) + 0x24)
    # controller state as the game sees it: PlayState input[0].cur (GameState+0x14: u16 button, s8 x, s8 y)
    inst.pad.send("SET MAIN 1.0 0.5"); time.sleep(0.5)
    cur = ram[(0x801C8D60 + 0x14) & 0x3FFFFF:(0x801C8D60 + 0x18) & 0x3FFFFF].tobytes().hex()
    inst.pad.send("SET MAIN 0.5 0.5")
    print(f"{backend:5s} unlimited={unlimited!s:5s}: VI/s {v1-v0:4d}  RAM bytes changed/s {changed:7d}  Link {tuple(round(x) for x in pos0)} -> {tuple(round(x) for x in pos1)}  pad(game view, stick right)={cur}", flush=True)
    inst.stop(); time.sleep(1.5)
