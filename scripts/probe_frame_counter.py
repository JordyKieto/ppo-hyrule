"""Find RAM counters ticking at game rate in gameplay; measure emulation speed per backend/speed limit."""
import os, sys, time, shutil, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance, HOTKEY_MAP
from zelda_rl.live import attach_n64_ram, be_s16, be_u32
SLOT = os.path.expanduser(os.environ.get("SLOT", "~/Library/Application Support/Dolphin/StateSaves/NACE01.s04"))

def boot(backend, unlimited, name):
    ud = Path(f"runs/instances/{name}"); shutil.rmtree(ud, ignore_errors=True)
    ud = build_user_dir(ud.resolve(), dump_frames=False, unlimited_speed=unlimited)
    shutil.copy(SLOT, ud / "StateSaves/NACE01.s01")
    inst = DolphinInstance(ud, video_backend=backend)
    ram = attach_n64_ram(inst.proc.pid)
    k = HOTKEY_MAP["Load State/Load State Slot 1"]
    for _ in range(40):
        if be_s16(ram, 0x801C8D60 + 0xA4) in (52, 85): break
        inst.hk.send(f"PRESS {k}"); time.sleep(0.15); inst.hk.send(f"RELEASE {k}"); time.sleep(0.35)
    assert be_s16(ram, 0x801C8D60 + 0xA4) in (52, 85), "load failed"
    return inst, ram

mode = sys.argv[1]
if mode == "find":
    inst, ram = boot("Null", False, "fc"); time.sleep(1)
    snaps, ts = [], []
    for i in range(11):
        snaps.append(ram.copy()); ts.append(time.perf_counter()); time.sleep(0.25)
    d = np.stack([s.view(">u4").astype(np.int64) for s in snaps]); steps = np.diff(d, axis=0); rate = steps / np.diff(ts)[:, None]
    steady = (steps > 0).all(0) & (rate.std(0) < 0.2 * rate.mean(0) + 1)
    for i in np.nonzero(steady)[0]:
        r = rate[:, i].mean()
        if 5 < r < 100: print(f"  u32 {0x80000000 + i*4:#010x} rate {r:6.1f}/s value {d[-1, i]}")
    d16 = np.stack([s[:len(s)//2*2].view(">u2").astype(np.int64) for s in snaps]); s16 = np.diff(d16, axis=0)
    steady16 = (s16 > 0).all(0) & (s16 < 100).all(0)
    for i in np.nonzero(steady16)[0]:
        r = (s16[:, i] / np.diff(ts)).mean()
        if 15 < r < 25: print(f"  u16 {0x80000000 + i*2:#010x} rate {r:6.1f}/s value {d16[-1, i]}")
    inst.stop()
else:
    addr = int(sys.argv[2], 16)
    for backend in ("Metal", "Null"):
        for unlimited in (False, True):
            inst, ram = boot(backend, unlimited, f"spd_{backend}_{int(unlimited)}")
            time.sleep(1); c0, t0 = be_u32(ram, addr), time.perf_counter(); v0 = be_u32(ram, 0x8011DBB0)
            time.sleep(5); c1, t1 = be_u32(ram, addr), time.perf_counter(); v1 = be_u32(ram, 0x8011DBB0)
            print(f"{backend:5s} unlimited={unlimited!s:5s}: game counter {(c1-c0)/(t1-t0):6.1f}/s  VI {(v1-v0)/(t1-t0):6.1f}/s  -> {(v1-v0)/(t1-t0)/60*100:4.0f}% speed", flush=True)
            inst.stop(); time.sleep(1)
