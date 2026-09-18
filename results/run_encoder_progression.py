"""Reproduce every encoder/decoder number in the paper, end to end.

    .venv/bin/python results/run_encoder_progression.py

Trains each configuration in the progression, evaluates event detection with the honest tune/test split, and
writes results/encoder_progression.json. The notebook and the paper tables read only that file, so no number
is typed by hand. ~8 minutes on CPU.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
SCRATCH = Path("/tmp/zelda_progression")

# (label, extra train args) -- the order is the story: what each lever added.
CONFIGS = [
    ("classification (base)", []),
    ("+ oversample 5%", ["--oversample", "0.05"]),
    ("+ consistency", ["--w-consistency", "0.5"]),
    ("+ both (final)", ["--oversample", "0.05", "--w-consistency", "0.5"]),
    ("oversample 15% (too much)", ["--oversample", "0.15", "--w-consistency", "0.5"]),
]
BASE = ["--epochs", "14", "--oversample", "0", "--w-consistency", "0"]


def run(label, extra):
    SCRATCH.mkdir(exist_ok=True)
    ckpt = SCRATCH / (label.replace(" ", "_").replace("%", "") + ".pt")
    args = [PY, "scripts/train_bc_decode.py", *BASE, "--out", str(ckpt)]
    for i in range(0, len(extra), 2):                       # later flags win over BASE
        if extra[i].startswith("--") and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
            args += [extra[i], extra[i + 1]]
        else:
            args.append(extra[i])
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit(f"train failed for {label}")
    ev = subprocess.run([PY, "scripts/eval_decode_events.py", str(ckpt), "--out", str(ckpt) + ".json"],
                        capture_output=True, text=True, cwd=ROOT)
    if ev.returncode:
        print(ev.stdout[-2000:], ev.stderr[-2000:])
        raise SystemExit(f"eval failed for {label}")
    res = json.loads((Path(str(ckpt) + ".json")).read_text())
    print(f"  {label:<28} {time.time() - t0:>5.0f}s", flush=True)
    return {"label": label, "args": extra, "train_stdout": p.stdout, "events": res["events"],
            "split": res["split"]}


def main():
    out = {"configs": [], "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"running {len(CONFIGS)} configurations")
    for label, extra in CONFIGS:
        out["configs"].append(run(label, extra))
    dst = ROOT / "results/encoder_progression.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    print(f"\n{'configuration':<28} {'damage AP':>10} {'kill AP':>9} {'link AP':>9}")
    print("-" * 60)
    for c in out["configs"]:
        g = lambda e, d: c["events"].get(e, {}).get(d, {}).get("ap", float("nan"))
        print(f"{c['label']:<28} {g('damage', 'HP-drop probability'):>10.3f} "
              f"{g('kill', 'HP->0 probability'):>9.3f} {g('link_damage', 'HP-drop probability'):>9.3f}")


if __name__ == "__main__":
    main()
