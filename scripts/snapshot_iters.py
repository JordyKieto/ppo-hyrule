"""Copy a loop run's rolling `loop_state.pt` to a per-iteration `iterNN.pt` as each iteration lands.

train_oocdm_loop.py writes per-iteration snapshots itself now; this exists for a run that was LAUNCHED
before that change, where the process already holds the old code. Point it at the run dir and leave it.

Errors are printed, never swallowed: an earlier version of this wrapped the copy in `except Exception: pass`
and sat alive for 16 minutes producing nothing, which is worse than crashing.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

rd = Path(sys.argv[1])
src, hist = rd / "loop_state.pt", rd / "history.json"
done = set()
while True:
    n = len(json.loads(hist.read_text())) if hist.exists() else 0
    if n and n not in done and src.exists():
        dst = rd / f"iter{n:02d}.pt"
        shutil.copyfile(src, dst)                      # raises loudly if it cannot write
        done.add(n)
        print(f"{time.strftime('%H:%M:%S')}  iteration {n} -> {dst}", flush=True)
    alive = subprocess.run(["pgrep", "-f", "train_oocdm_loop"], capture_output=True).returncode == 0
    if not alive:
        print(f"{time.strftime('%H:%M:%S')}  loop finished; {len(done)} snapshots", flush=True)
        break
    time.sleep(15)
