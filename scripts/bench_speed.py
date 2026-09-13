"""Emulation speed per video backend and speed limit, measured from RAM: VI rate + Link displacement per VI."""
import os, sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0], "x"]
exec(open("scripts/probe_frame_counter.py").read().split('mode = sys.argv[1]')[0])
import struct
def link_pos(ram):
    p = be_u32(ram, 0x801C8D60 + 0x1C44)
    return struct.unpack_from(">fff", ram, (p & 0x3FFFFF) + 0x24)
VI = 0x8011DBB0
for backend in ("Metal", "Null"):
    for unlimited in (False, True):
        inst, ram = boot(backend, unlimited, f"spd_{backend}_{int(unlimited)}")
        time.sleep(0.5)
        # walk in a slow circle so Link keeps moving; measure distance travelled per VI and VI rate
        v0, t0, p0 = be_u32(ram, VI), time.perf_counter(), link_pos(ram); dist = 0.0; prev = p0; changes = 0
        while time.perf_counter() - t0 < 6:
            a = (time.perf_counter() - t0) * 1.5
            inst.pad.send(f"SET MAIN {0.5+0.5*math.cos(a):.2f} {0.5+0.5*math.sin(a):.2f}")
            time.sleep(0.01)
            p = link_pos(ram)
            if p != prev: changes += 1; dist += math.dist(p[::2], prev[::2]); prev = p
        v1, t1 = be_u32(ram, VI), time.perf_counter()
        inst.pad.send("SET MAIN 0.5 0.5")
        vi_rate = (v1 - v0) / (t1 - t0)
        print(f"{backend:5s} unlimited={unlimited!s:5s}: VI {vi_rate:6.1f}/s ({vi_rate/60*100:4.0f}%)  Link moved {dist:7.0f} units ({dist/max(v1-v0,1):.2f}/VI), pos updates seen {changes}", flush=True)
        inst.stop(); time.sleep(1.5)
