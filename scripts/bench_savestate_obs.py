"""Savestate-as-observation benchmark: hotkey save -> file -> LZ4 decompress -> N64 RAM -> features."""
import os, sys, time, shutil, statistics as st, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.savestate import decompress_state, find_n64_ram, N64Ram
from zelda_rl.oot_ram import read_features

state_dir = sys.argv[1] if len(sys.argv) > 1 else None   # optional RAM disk dir for StateSaves
ud = Path("runs/instances/ssbench"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
ss = ud / "StateSaves"
if state_dir:
    shutil.rmtree(ss); Path(state_dir).mkdir(parents=True, exist_ok=True); os.symlink(state_dir, ss)
slot = ss / "NACE01.s01"
shutil.copy("runs/states/house.sav", slot)
inst = DolphinInstance(ud)
time.sleep(12)
inst.hotkey("Load State/Load State Slot 1"); time.sleep(3)

def wait_new(prev_mtime, timeout=5):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        try:
            m = os.stat(slot).st_mtime_ns
            if m != prev_mtime:
                buf = open(slot, "rb").read()
                raw = decompress_state(buf)   # raises if partially written
                return m, buf, raw
        except (OSError, ValueError, Exception):
            pass
        time.sleep(0.002)
    raise TimeoutError

save_lat, read_t, dec_t, feat_t, periods = [], [], [], [], []
mt = os.stat(slot).st_mtime_ns; base = None; tprev = time.perf_counter()
for k in range(25):
    ang = k * 0.5
    inst.pad.send(f"SET MAIN {0.5 + 0.5*math.cos(ang):.3f} {0.5 + 0.5*math.sin(ang):.3f}")
    t0 = time.perf_counter()
    inst.hotkey("Save State/Save State Slot 1", 0.15)
    # time to a complete new file (includes our read+decompress polling)
    t = time.perf_counter()
    def mtime():
        try: return os.stat(slot).st_mtime_ns
        except FileNotFoundError: return mt
    while mtime() == mt and time.perf_counter() - t < 5: time.sleep(0.001)
    t_file = time.perf_counter()
    time.sleep(0.0)
    while True:
        try:
            ta = time.perf_counter(); buf = open(slot, "rb").read(); tb = time.perf_counter()
            raw = decompress_state(buf); tc = time.perf_counter(); break
        except Exception:
            time.sleep(0.005)
    mt = mtime()
    base = find_n64_ram(raw, base)
    td = time.perf_counter(); f = read_features(N64Ram(raw[base:base + 0x400000])); te = time.perf_counter()
    save_lat.append(t_file - t0); read_t.append(tb - ta); dec_t.append(tc - tb); feat_t.append(te - td)
    periods.append(te - tprev); tprev = te
    print(f"k={k:2d} pos={f.get('link_pos')} yaw={f.get('link_yaw')} rupees={f['rupees']} scene={f['scene']} size={len(buf)/1e6:.1f}MB", flush=True)
inst.pad.send("SET MAIN 0.5 0.5")
ms = lambda xs: f"p50={st.median(xs)*1e3:.0f}ms p90={np.percentile(xs,90)*1e3:.0f}ms"
print("hotkey -> state file mtime changes:", ms(save_lat))
print("read file:", ms(read_t)); print("LZ4 decompress (python):", ms(dec_t)); print("find RAM + features:", ms(feat_t))
print(f"end-to-end obs period: {ms(periods[1:])} -> {1/st.median(periods[1:]):.1f} obs/s")
inst.stop()
