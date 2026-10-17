#!/usr/bin/env python3
"""Resume must CONTINUE training, and a restart must never silently discard it.

2026-10-16: three RvS arms were restarted to change a flag. `scripts/train_rvs.py` had no resume, so each
began from a random init and discarded every epoch already paid for -- one had reached 1.54 online damage.
The fix is only worth something if it is tested, so this asserts the three things that matter:

  1. a resumed run's loss CONTINUES the curve instead of jumping back to a fresh-init loss;
  2. the resumed weights stay close to the checkpoint they came from, and far from a fresh init;
  3. starting over on top of existing progress is REFUSED unless --fresh is passed.

Run: .venv/bin/python tests/test_rvs_resume.py
"""
from __future__ import annotations

import json
import shutil
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN = str(ROOT / "scripts" / "train_rvs.py")


def _corpus(d: Path, n: int = 6, steps: int = 200, dim: int = 24) -> None:
    """A corpus the trainer accepts: obs/action/boss_defeated/obs_fields, half of them wins."""
    # the global block must name the fields rvs.conditioning reads, or load_all dies on KeyError
    fields = json.dumps({"n_actors": 2, "actor": [["x", 4]],
                         "global": [["gohma_hp_frac", 1], ["link_hp_frac", 1], ["g", dim - 2 * 4 - 2]]})
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        np.savez(d / f"ep{i:02d}.npz",
                 obs=np.clip(rng.random((steps, dim)).astype(np.float32), 0.0, 1.0),
                 action=rng.integers(0, 4, steps).astype(np.int64),
                 boss_defeated=np.array(i % 2 == 0),
                 boss_damage=np.int32(3 * (i % 2)),
                 n_actions=np.int32(8),
                 obs_fields=np.array(fields))


def _run(args: list[str], expect_ok: bool = True) -> str:
    # ZRL_ALLOW_NO_WANDB is the contract's named override (zelda_rl/eval_log.py): the trainer REFUSES to
    # start without a durable eval sink, which is the point, and a test is the one place that is fine --
    # the sink still writes its jsonl, so resume behaviour is unaffected.
    env = {**os.environ, "ZRL_ALLOW_NO_WANDB": "1"}
    r = subprocess.run([PY, "-u", TRAIN, *args], capture_output=True, text=True, timeout=1800, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    if expect_ok and r.returncode != 0:
        raise AssertionError(f"train_rvs failed ({r.returncode}) on args {args}:\n{out[-900:]}")
    if not expect_ok and r.returncode == 0:
        raise AssertionError(f"expected a refusal, got success:\n{out[-800:]}")
    return out


def _losses(out: str) -> list[float]:
    vals = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            try:
                vals.append(float(p[1]))
            except ValueError:
                pass
    return vals


def _flat(path: Path) -> np.ndarray:
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return np.concatenate([v.detach().numpy().ravel() for v in sd.values()])


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "corpus"
    _corpus(data)
    base = ["--data", str(data), "--batch-episodes", "2", "--chunk", "60", "--val-frac", "0.34",
            "--eval-every", "0", "--seed", "0", "--lr", "1e-3"]

    # --- 1. two epochs, then resume for two more -------------------------------------------------
    out_a = tmp / "a.pt"
    first = _run(base + ["--epochs", "2", "--out", str(out_a)])
    l_first = _losses(first)
    assert len(l_first) == 2, f"expected 2 epoch rows, got {l_first}"
    mid = _flat(Path(str(out_a) + ".last.pt"))

    second = _run(base + ["--epochs", "4", "--out", str(out_a), "--resume"])
    assert "RESUMED" in second, f"no RESUMED line:\n{second[-600:]}"
    l_second = _losses(second)
    assert len(l_second) == 2, f"resume should run epochs 3-4 only, got {l_second}"

    # the continuation must not look like a fresh start
    fresh_out = tmp / "fresh.pt"
    fresh = _run(base + ["--epochs", "1", "--out", str(fresh_out)])
    l_fresh = _losses(fresh)[0]
    assert l_second[0] < l_first[0], (
        f"resumed epoch-3 loss {l_second[0]:.4f} is not below the first epoch's {l_first[0]:.4f} "
        f"-- it restarted rather than continued")
    assert l_second[0] < l_fresh, (
        f"resumed loss {l_second[0]:.4f} >= a fresh init's first-epoch loss {l_fresh:.4f}")
    print(f"ok  loss continues: {l_first[0]:.3f} -> {l_first[1]:.3f} -> {l_second[0]:.3f} -> "
          f"{l_second[1]:.3f}   (fresh init starts at {l_fresh:.3f})")

    # --- 2. the weights came from the checkpoint, not from a new init -----------------------------
    after = _flat(Path(str(out_a) + ".last.pt"))
    fresh_w = _flat(Path(str(fresh_out) + ".last.pt"))
    d_cont = float(np.linalg.norm(after - mid))
    d_fresh = float(np.linalg.norm(after - fresh_w))
    assert d_cont < d_fresh, (f"resumed weights are closer to a FRESH init ({d_fresh:.3f}) than to the "
                             f"checkpoint they resumed from ({d_cont:.3f})")
    print(f"ok  weights continue: |after-checkpoint| {d_cont:.3f} < |after-freshinit| {d_fresh:.3f}")

    # --- 3. a restart on top of progress is REFUSED ----------------------------------------------
    refused = _run(base + ["--epochs", "2", "--out", str(out_a)], expect_ok=False)
    assert "already holds training progress" in refused, f"wrong refusal:\n{refused[-500:]}"
    print("ok  silent overwrite refused:", refused.strip().splitlines()[-1][:95])

    # --- 4. --fresh overrides it deliberately ----------------------------------------------------
    _run(base + ["--epochs", "1", "--out", str(out_a), "--fresh"])
    print("ok  --fresh overrides the refusal")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS  tests/test_rvs_resume.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
