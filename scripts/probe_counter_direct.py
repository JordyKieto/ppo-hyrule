import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0], "x"]
exec(open("scripts/probe_frame_counter.py").read().split('mode = sys.argv[1]')[0])
inst, ram = boot(os.environ.get("BACKEND","Null"), False, "fc")
import struct
for addr in (0x801C8DFC, 0x8011DBB0, 0x8011AC80 + 0x0C):
    vals = []
    for i in range(9):
        vals.append((time.perf_counter(), be_u32(ram, addr) if addr != 0x8011AC80 + 0x0C else struct.unpack_from(">H", ram, addr & 0x3FFFFF)[0])); time.sleep(0.25)
    rates = [(b[1]-a[1])/(b[0]-a[0]) for a, b in zip(vals, vals[1:])]
    print(hex(addr), "values", [v for _, v in vals][:5], "rates", [round(r,1) for r in rates])
# sample at 1ms to see per-update step pattern
seq = []
t0 = time.perf_counter()
while time.perf_counter() - t0 < 0.5:
    seq.append((round((time.perf_counter()-t0)*1000), be_u32(ram, 0x801C8DFC))); time.sleep(0.001)
ch = [s for a, s in zip([None]+seq, seq) if a is None or a[1] != s[1]]
print("frames changes (ms, value):", ch[:20])
inst.stop()
