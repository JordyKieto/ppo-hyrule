"""Smoke test: per-slot load + input check, then random-policy stepping timing, for 1x and unlimited speed."""
import argparse, os, sys, time, statistics as st
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig, N_ACTIONS
from zelda_rl.obs import read_state_file, OBS_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--slots-dir", default=os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
ap.add_argument("--unlimited", action="store_true")
ap.add_argument("--backend", default="Metal")
ap.add_argument("--check-slots", default="1,2,3,4,5")
ap.add_argument("--wait-vi", type=int, default=12)
args = ap.parse_args()
files = {k: Path(args.slots_dir) / f"NACE01.s{k:02d}" for k in range(1, 6)}
states = {k: read_state_file(f) for k, f in files.items()}
for k, s in states.items():
    print(f"slot {k}: scene={s.scene} rupees={s.rupees} pos={tuple(round(v) for v in s.pos)} vi={s.vi} rupee_items={len(s.rupee_items)}")
cfg = EnvConfig(video_backend=args.backend, unlimited_speed=args.unlimited)
env = DolphinGoalEnv(0, Path("runs/instances/smoke"), files, states, states[5], cfg)
print(f"OBS_DIM={OBS_DIM} N_ACTIONS={N_ACTIONS} backend={args.backend} unlimited={args.unlimited}")
for k in [int(x) for x in args.check_slots.split(",")]:
    try:
        t = env.load_slot(k); time.sleep(0.3)
        print(f"slot {k}: load {t*1e3:.0f} ms, input {env.input_check(args.wait_vi)}", flush=True)
    except Exception as e:
        print(f"slot {k}: FAILED {e}", flush=True)
rng = np.random.default_rng(0)
for slot in (2, 3):
    obs = env.reset(slot); wait, lat, vi_el = [], [], []
    t_ep = time.perf_counter()
    for i in range(100):
        a = int(rng.integers(N_ACTIONS)) if i % 5 == 0 else a if i else 0
        t0 = time.perf_counter(); env.send_action(a)
        while not env.ready(): time.sleep(0.0005)
        t1 = time.perf_counter(); obs, r, term, trunc, info = env.observe(); t2 = time.perf_counter()
        wait.append(t1 - t0); lat.append(t2 - t1); vi_el.append(info["vi_elapsed"])
        if term: break
    m = info["metrics"]
    print(f"slot {slot}: {i+1} steps in {time.perf_counter()-t_ep:.1f}s | step wait p50 {st.median(wait)*1e3:.1f} ms | observe+reward p50 {st.median(lat)*1e6:.0f} us | VI/step {st.mean(vi_el):.2f} | pos_dist {m.pos_dist:.0f} rupees {info['state'].rupees}", flush=True)
env.close()
