"""Measure the live loop: frame-on-disk -> decode -> preprocess -> CNN -> pipe write, plus input->screen delay and reset time."""
import os, sys, time, shutil, statistics as st
from pathlib import Path
import numpy as np, cv2, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
from zelda_rl.frames import DumpDirFrames
from zelda_rl.hud import in_gameplay
from zelda_rl.policy import NatureCNN

def ms(xs): return f"p50={st.median(xs)*1e3:.2f}ms p90={np.percentile(xs,90)*1e3:.2f}ms"

ud = Path("runs/instances/bench"); shutil.rmtree(ud, ignore_errors=True); ud = build_user_dir(ud.resolve())
shutil.copy("runs/states/house.sav", ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud); fr = DumpDirFrames(ud)
t0 = time.time()
while not (fr.latest() and fr.last_idx > 100): time.sleep(0.2)
print(f"boot->frames flowing: {time.time()-t0:.1f}s")

# --- reset via savestate load hotkey
t1 = time.time(); inst.hotkey("Load State/Load State Slot 1")
while True:
    f = fr.latest()
    if f and in_gameplay(f[1]): break
    time.sleep(0.01)
print(f"reset (hotkey press -> HUD frame on disk): {time.time()-t1:.2f}s")
time.sleep(1.5)

# --- per-stage timings over ~200 new frames
nets = {"cpu": NatureCNN().eval()}
if torch.backends.mps.is_available(): nets["mps"] = NatureCNN().to("mps").eval()
stack = np.zeros((4, 84, 84), np.float32)
age, dec, pre, inf, act, loop = [], [], [], [], [], []
inf_dev = {k: [] for k in nets}
last = fr.last_idx; n = 0; tl = time.perf_counter()
while n < 200:
    names = sorted(fr._indices())
    if len(names) < 2 or names[-2] == last:
        time.sleep(0.001); continue
    i = names[-2]; last = i; p = fr.dir / f"framedump_{i}.png"
    ta = time.time(); age.append(ta - os.stat(p).st_mtime)
    t = time.perf_counter(); img = cv2.imread(str(p), cv2.IMREAD_COLOR); dec.append(time.perf_counter() - t)
    t = time.perf_counter()
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
    stack = np.roll(stack, -1, 0); stack[-1] = g / 255.0
    x = torch.from_numpy(stack[None]); pre.append(time.perf_counter() - t)
    with torch.no_grad():
        for dev, net in nets.items():
            t = time.perf_counter(); logits, _ = net(x.to(dev)); a = int(logits.argmax()); 
            if dev == "mps": torch.mps.synchronize()
            inf_dev[dev].append(time.perf_counter() - t)
    t = time.perf_counter(); inst.pad.send(f"SET MAIN {0.5 + 0.3*np.sin(n/10):.3f} 0.5"); act.append(time.perf_counter() - t)
    loop.append(time.perf_counter() - tl); tl = time.perf_counter()
    for j in names[:-4]: (fr.dir / f"framedump_{j}.png").unlink(missing_ok=True)
    n += 1
inst.pad.send("SET MAIN 0.5 0.5")
print("frame age at read (dolphin write -> we see it):", ms(age))
print("decode PNG (cv2):", ms(dec))
print("preprocess (gray, 84x84, stack):", ms(pre))
for d, v in inf_dev.items(): print(f"NatureCNN inference bs=1 {d}:", ms(v[5:]))
print("pipe write:", ms(act))
print(f"loop period (new unique frames): {ms(loop[1:])} -> {1/st.median(loop[1:]):.1f} Hz")

# --- input -> visible effect: idle, then push stick; first frame whose diff exceeds idle noise
time.sleep(2)
def gray_small(): 
    f = fr.latest(); return f[0], cv2.resize(cv2.cvtColor(f[1], cv2.COLOR_RGB2GRAY), (160, 120)).astype(np.float32)
delays = []
for trial in range(6):
    inst.hotkey("Load State/Load State Slot 1"); time.sleep(2.5)
    idle = []; i0, prev = gray_small()
    for _ in range(15):
        time.sleep(0.05); i1, cur = gray_small(); idle.append(np.abs(cur - prev).mean()); prev = cur
    thr = max(idle) * 3 + 1.0
    base_idx, base = gray_small()
    t = time.perf_counter(); inst.pad.send("SET MAIN 0.5 1.0")  # walk away from camera
    while time.perf_counter() - t < 2:
        idx, cur = gray_small()
        if idx != base_idx and np.abs(cur - base).mean() > thr:
            delays.append((time.perf_counter() - t, idx - base_idx)); break
        time.sleep(0.002)
    inst.pad.send("SET MAIN 0.5 0.5")
print("input -> first changed frame on disk:", [f"{d*1e3:.0f}ms/{k}fr" for d, k in delays])
inst.stop()
