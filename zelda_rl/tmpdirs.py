"""Per-instance Dolphin user dirs, cleaned up when the process exits AND reaped when it did not.

Each Dolphin instance needs its own user dir (NAND, savestates, config): ~800 MB, and an evaluator with 8
of them is ~6.5 GB. `scripts/eval_fleet.py` and `scripts/record_human.py` created
`/tmp/evalf_<pid>` / `/tmp/human_<pid>` and never removed them, so every run leaked its whole working set.
Measured on the gohma-s3 pods 2026-10-16: **five `evalf_*` dirs, 6.5 GB each, 33 GB of a 40 GB overlay**,
which is what produced `OSError: [Errno 28] No space left on device` writing a savestate and killed five
training arms. The arms were blamed on the pod being small; the pods were fine.

atexit alone is not enough -- a SIGKILLed or OOM-killed run leaves its dir behind forever, and that is how
the 33 GB accumulated. So every call also REAPS the dirs of processes that no longer exist.
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
from pathlib import Path


def _tmp() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp"))


def reap_orphans(prefix: str) -> tuple[int, float]:
    """Remove `<tmp>/<prefix>_<pid>` dirs whose pid is gone. Returns (dirs, GB) reclaimed."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    n = gb = 0
    for d in _tmp().glob(f"{prefix}_*"):
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        pid = int(m.group(1))
        if pid == os.getpid() or Path(f"/proc/{pid}").exists():
            continue                                   # still running: not ours to remove
        try:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except OSError:
            size = 0
        shutil.rmtree(d, ignore_errors=True)
        n += 1
        gb += size / 1e9
    if n:
        print(f"reclaimed {n} orphaned {prefix}_* dir(s) in {_tmp()}: {gb:.1f} GB", flush=True)
    return n, gb


def instance_root(prefix: str) -> Path:
    """`<tmp>/<prefix>_<pid>`, reaping dead runs' dirs first and removing ours at exit."""
    reap_orphans(prefix)
    root = _tmp() / f"{prefix}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root
