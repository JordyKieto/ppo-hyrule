#!/usr/bin/env python3
"""The RvS loader must REFUSE a corpus that will not fit, instead of being OOM-killed silently.

2026-10-16: the spout arm `la3` grew its corpus to 8,345 episodes and was OOM-killed. Each episode holds
~2.9 MB of float32 observations and the Dyna refresh reloads the whole corpus while the previous copy is
still referenced, so it needed ~24 GB against the pod's 32 GB cgroup limit -- while `free` reported 503 GB
(AGENTS 23). The run died with `Killed`, no traceback, and `fleet_health` then reported it as
"FINISHED (clean exit)". A silent death is worse than a crash.

Run: .venv/bin/python tests/test_rvs_memguard.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("train_rvs", ROOT / "scripts" / "train_rvs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _corpus(d: Path, n: int = 5, steps: int = 300, dim: int = 24) -> None:
    fields = json.dumps({"n_actors": 2, "actor": [["x", 4]],
                         "global": [["gohma_hp_frac", 1], ["link_hp_frac", 1], ["g", dim - 2 * 4 - 2]]})
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        np.savez(d / f"ep{i:02d}.npz", obs=rng.random((steps, dim)).astype(np.float32),
                 action=rng.integers(0, 4, steps).astype(np.int64),
                 boss_defeated=np.array(i % 2 == 0), boss_damage=np.int32(i),
                 n_actions=np.int32(8), obs_fields=np.array(fields))


def test_refuses_when_over_budget():
    tmp = Path(tempfile.mkdtemp())
    _corpus(tmp / "c")
    m = _load_module()
    os.environ["ZRL_MEM_LIMIT_BYTES"] = str(64 * 1024)          # 64 KiB: everything is over budget
    try:
        m.load_all([tmp / "c"])
    except SystemExit as e:
        assert "CORPUS TOO LARGE" in str(e), f"wrong refusal: {e}"
        assert "cgroup limit" in str(e), f"refusal must quote the limit: {e}"
        print("ok  refuses over budget:", str(e)[:100].replace("\n", " "))
    else:
        raise AssertionError("loaded a corpus that does not fit -- the guard did not fire")
    finally:
        os.environ.pop("ZRL_MEM_LIMIT_BYTES", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_loads_when_within_budget():
    tmp = Path(tempfile.mkdtemp())
    _corpus(tmp / "c")
    m = _load_module()
    os.environ["ZRL_MEM_LIMIT_BYTES"] = str(4 * 1024 ** 3)      # 4 GiB: plenty
    try:
        eps = m.load_all([tmp / "c"])
        assert len(eps) == 5, f"expected 5 episodes, got {len(eps)}"
        assert getattr(m.load_all, "gib", None) is not None, "the loader must report its RAM use"
        print(f"ok  loads within budget: {len(eps)} episodes, {m.load_all.gib:.4f} GiB reported")
    finally:
        os.environ.pop("ZRL_MEM_LIMIT_BYTES", None)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_refuses_when_over_budget()
    test_loads_when_within_budget()
    print("\nALL PASS  tests/test_rvs_memguard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
