import os, sys, time, shutil, struct, statistics as st
from pathlib import Path
S = "/private/tmp/claude-501/-Users-jordy-Documents-dev-zelda-RL/a9a13dc7-3c2e-4ab4-a237-bc4417e8ccaa/scratchpad"
os.environ["DOLPHIN_BIN"] = f"{S}/Dolphin.app/Contents/MacOS/Dolphin"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.machmem import MachTask
MAGIC = bytes.fromhex("3c1a8000275a260003400008")
ud = Path("runs/instances/mach"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud); time.sleep(12)
print("pid", inst.proc.pid, "alive", inst.proc.poll() is None, flush=True)
try:
    t = MachTask(inst.proc.pid)
except PermissionError as e:
    print(e); inst.stop(); sys.exit(1)
# find MEM1 mappings: N64 magic at +0xF64120 and 'NACE' at +0x3180
t0 = time.time(); cands = []
for addr, size, prot in t.regions():
    if size < 0xF64120 + 12 or not (prot & 1): continue
    try:
        if t.read(addr + 0xF64120, 12) == MAGIC and t.read(addr + 0x3180, 4) == b"NACE":
            cands.append((addr, size))
    except OSError:
        pass
print(f"MEM1 candidates {[(hex(a), hex(s)) for a, s in cands]} scan {time.time()-t0:.2f}s", flush=True)
mem1 = cands[0][0]; N64 = mem1 + 0xF64120
def feats():
    sc = t.read(N64 + 0x11AC80, 0x40)
    player = struct.unpack(">I", t.read(N64 + 0x1C8D60 + 0x1C44, 4))[0]
    pos = struct.unpack(">fff", t.read(N64 + (player & 0x3FFFFF) + 0x24, 12)) if player >> 24 == 0x80 else None
    return sc[0x1C:0x22], struct.unpack_from(">h", sc, 0x34)[0], pos
print("attract:", feats())
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(3)
print("after state load:", feats(), flush=True)
lat = []
for k in range(40):
    inst.pad.send("SET MAIN 0.5 1.0" if k < 20 else "SET MAIN 1.0 0.5")
    a = time.perf_counter(); f = feats(); lat.append(time.perf_counter() - a)
    if k % 8 == 0: print(k, f, flush=True)
    time.sleep(0.05)
inst.pad.send("SET MAIN 0.5 0.5")
print(f"feature read (3 small reads): p50={st.median(lat)*1e6:.0f}us p90={sorted(lat)[36]*1e6:.0f}us")
buf = bytearray(0x400000); ts = []
for _ in range(50):
    a = time.perf_counter(); t.read_into(N64, buf); ts.append(time.perf_counter() - a)
print(f"full 4MB N64 RAM snapshot: p50={st.median(ts)*1e3:.2f}ms -> {1/st.median(ts):.0f}/s, head={bytes(buf[:4]).hex()}")
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(2)
print("2nd load, mapping still valid:", t.read(N64, 4).hex(), feats())
inst.stop()
