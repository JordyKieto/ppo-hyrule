"""Verify a human-demonstration dataset BEFORE it is uploaded to S3 and trained on remotely.

A bad demonstration dataset is worse than none: it trains a policy toward the wrong behaviour and nothing in
the loss says so. The checks here are the ones that would have caught every mistake already made while
building the recorder -- constant inputs (the old boss_*.jsonl traces), a direction mapping that ignored the
camera, and an observation whose width does not match the model the remote job will build.

    .venv/bin/python scripts/verify_human_dataset.py datasets/human/gohma
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    files = sorted(d.glob("ep*.npz"))
    if not files:
        print(f"no ep*.npz in {d}")
        return 1
    man = json.loads((d / "MANIFEST.json").read_text()) if (d / "MANIFEST.json").exists() else {}
    by_md5 = {e["file"]: e.get("md5") for e in man.get("episodes", [])}

    fails, warns = [], []
    tot_steps = wins = 0
    dims, nacts = set(), set()
    act_hist: Counter = Counter()
    print(f"{'file':<34}{'steps':>7}{'dmg':>5}{'win':>5}{'uniq act':>9}{'pad vals':>9}{'md5':>7}")
    for f in files:
        z = np.load(f, allow_pickle=False)
        obs, act, pad = z["obs"], z["action"], z["pad"]
        T = len(act)
        tot_steps += T
        won = bool(z["boss_defeated"])
        wins += won
        dims.add(int(z["obs_dim"])); nacts.add(int(z["n_actions"]))
        act_hist.update(act.tolist())
        upad = len({tuple(r) for r in pad})
        uact = len(set(act.tolist()))
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        md5ok = (by_md5.get(f.name) is None) or (by_md5[f.name] == md5)
        print(f"{f.name:<34}{T:>7}{int(z['boss_damage']):>5}{'Y' if won else '-':>5}{uact:>9}{upad:>9}"
              f"{'ok' if md5ok else 'BAD':>7}")
        # the checks
        if len(obs) != T + 0 and len(obs) != T:
            fails.append(f"{f.name}: obs rows {len(obs)} != action rows {T}")
        if upad < 3:
            fails.append(f"{f.name}: only {upad} distinct pad readings -- inputs are CONSTANT, the old bug")
        if uact < 3:
            fails.append(f"{f.name}: only {uact} distinct actions -- the mapping is probably broken")
        if not np.isfinite(obs).all():
            fails.append(f"{f.name}: non-finite values in obs")
        if act.min() < 0 or act.max() >= int(z["n_actions"]):
            fails.append(f"{f.name}: action out of range [0,{int(z['n_actions'])})")
        if "cam_yaw" in z and len(z["cam_yaw"]) != T:
            fails.append(f"{f.name}: cam_yaw length {len(z['cam_yaw'])} != {T}")
        if "obs_fields" not in z or "action_space" not in z:
            fails.append(f"{f.name}: not self-describing (missing obs_fields / action_space)")
        if not md5ok:
            fails.append(f"{f.name}: md5 does not match MANIFEST")
        if T < 50:
            warns.append(f"{f.name}: only {T} steps ({T/10:.1f}s) -- very short")

    print()
    if len(dims) != 1:
        fails.append(f"inconsistent obs_dim across files: {sorted(dims)}")
    if len(nacts) != 1:
        fails.append(f"inconsistent n_actions across files: {sorted(nacts)}")
    print(f"  {len(files)} episodes, {tot_steps:,} steps, {wins} wins, obs_dim {sorted(dims)}, "
          f"n_actions {sorted(nacts)}")
    cov = 100 * len(act_hist) / max(nacts and list(nacts)[0] or 1, 1)
    print(f"  action coverage: {len(act_hist)}/{list(nacts)[0] if nacts else '?'} distinct ({cov:.0f}%)")
    top = ", ".join(f"{a}:{c}" for a, c in act_hist.most_common(5))
    print(f"  most used actions: {top}")
    if act_hist and act_hist.most_common(1)[0][1] / tot_steps > 0.8:
        warns.append(f"one action is {100*act_hist.most_common(1)[0][1]/tot_steps:.0f}% of all steps")
    if wins == 0:
        warns.append("NO WINS in the dataset -- behaviour cloning will learn losing play")

    for w in warns:
        print(f"  WARN  {w}")
    for e in fails:
        print(f"  FAIL  {e}")
    print()
    if fails:
        print(f"NOT SAFE TO UPLOAD: {len(fails)} failure(s). Fix before pushing to S3.")
        return 1
    print("OK to upload. Suggested:")
    print(f"  aws --profile race-wm s3 sync {d} s3://zelda-rl/zelda/human/gohma/ --exclude '*' --include '*.npz' --include 'MANIFEST.json'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
