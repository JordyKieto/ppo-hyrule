import os, sys, time, shutil, threading, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
ud = Path("runs/instances/ssbench"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False); slot = ud / "StateSaves/NACE01.s01"
shutil.copy("runs/states/house.sav", slot)
inst = DolphinInstance(ud); time.sleep(12)
inst.hotkey("Load State/Load State Slot 1", 0.2); time.sleep(3)
def mtime():
    try: return os.stat(slot).st_mtime_ns
    except FileNotFoundError: return None
for hold in (0.01, 0.02, 0.035, 0.05, 0.08):
    ok, lat = 0, []
    for _ in range(6):
        m0 = mtime(); t0 = time.perf_counter(); inst.hk.send("PRESS L")
        released = False; got = None
        while time.perf_counter() - t0 < 1.0:
            if not released and time.perf_counter() - t0 >= hold: inst.hk.send("RELEASE L"); released = True
            m = mtime()
            if m is not None and m != m0:
                # wait until parseable size stops changing
                got = time.perf_counter() - t0; break
            time.sleep(0.0005)
        if not released: inst.hk.send("RELEASE L")
        if got: ok += 1; lat.append(got)
        time.sleep(0.3)
    print(f"hold {hold*1e3:4.0f}ms: {ok}/6 saved, press->new file p50={st.median(lat)*1e3 if lat else float('nan'):.0f}ms min={min(lat)*1e3 if lat else float('nan'):.0f}ms", flush=True)
inst.stop()
