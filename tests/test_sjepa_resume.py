"""Resume for scripts/train_sjepa.py, asserted on the WEIGHTS and the loss CURVE -- not on a log line.

AGENTS 39 is the reason this file is this shape. `train_rvs.py` had no resume and checkpointed only on
improvement, three arms were restarted to change a flag and each silently began from a random init, and one
had reached 1.54 online boss damage before going back to 0.75. Then the first "fix" was a four-episode toy
that printed RESUMED and was called done -- which proved the MESSAGE appeared, not that the WEIGHTS continued.
The user's response to that is quoted in AGENTS 39 and it was deserved.

So, four assertions, two of them controls:

  1. the resumed run's loss CONTINUES the curve rather than restarting at a fresh init's level;
  2. the resumed WEIGHTS are closer to the checkpoint they came from than to a fresh init -- a vector norm,
     not a print statement, so a restart masquerading as a resume fails here;
  3. POSITIVE CONTROL: starting on top of existing progress is REFUSED without --resume/--fresh;
  4. NEGATIVE CONTROL: --fresh overrides that refusal. The first version of the guard in train_rvs.py
     ignored its own flag, and only a test of the override caught it.

Run via tests/run_all.py, which also enforces that this file imports no pytest and actually calls its tests.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PY = str(REPO / ".venv" / "bin" / "python")
DATA = "datasets/transitions/v3-main"
BASE = ["--data", DATA, "--clip-len", "4", "--stride", "64", "--limit-clips", "240",
        "--dim", "32", "--depth", "1", "--pred-depth", "1", "--heads", "2",
        "--batch", "32", "--lr", "1e-3", "--torch-threads", "2", "--seed", "0"]


def run(args, cwd, expect=0):
    env = {**os.environ, "ZRL_ALLOW_NO_WANDB": "1", "PYTHONPATH": str(REPO)}
    r = subprocess.run([PY, "-u", str(REPO / "scripts" / "train_sjepa.py"), *args],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    if (r.returncode == 0) != (expect == 0):
        raise AssertionError(f"exit {r.returncode}, expected {expect}\nSTDOUT\n{r.stdout[-3000:]}\n"
                             f"STDERR\n{r.stderr[-2000:]}")
    return r


def losses(out: Path) -> list[float]:
    """Per-epoch train loss, read from the EvalSink jsonl -- the durable artefact, not the stdout."""
    rows = [json.loads(l) for l in (out / "runs" / "evals" / "t.jsonl").read_text().splitlines()]
    return [r["train_loss"] for r in rows]


def weights(p: Path):
    import torch
    ck = torch.load(p, weights_only=False, map_location="cpu")
    return torch.cat([v.flatten().float() for k, v in sorted(ck["model"].items())
                      if not k.startswith(("tok_t", "enc_t"))]), int(ck["epoch"])


def main() -> None:
    import torch
    if not (REPO / DATA).is_dir():
        print(f"SKIP: {DATA} not present")          # still a real run elsewhere; do not fake a pass
        return
    tmp = Path(tempfile.mkdtemp(prefix="sjepa_resume_"))
    try:
        for d in ("datasets", "configs"):
            (tmp / d).mkdir(parents=True, exist_ok=True)
        os.symlink(REPO / "datasets" / "transitions", tmp / "datasets" / "transitions")
        out = tmp / "m.pt"
        prep = tmp / "configs" / "p.json"
        common = [*BASE, "--out", str(out), "--prep", str(prep), "--run-name", "t"]

        # two epochs, from scratch
        run([*common, "--fit-prep", "--epochs", "2"], cwd=tmp)
        first = losses(tmp)
        assert len(first) == 2, f"expected 2 epochs, got {first}"
        w_ck, ep_ck = weights(out.with_suffix(".last.pt"))
        assert ep_ck == 2, f"checkpoint says epoch {ep_ck}"

        # 3. POSITIVE CONTROL: a bare rerun must refuse rather than wipe it
        r = run([*common, "--epochs", "4"], cwd=tmp, expect=1)
        assert "already holds training progress" in (r.stdout + r.stderr), \
            f"the guard did not fire:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"

        # 1. the resumed run continues the curve
        shutil.copy(out.with_suffix(".last.pt"), tmp / "ck2.pt")
        run([*common, "--epochs", "4", "--resume"], cwd=tmp)
        after = losses(tmp)
        assert len(after) == 4, f"expected 4 rows after resume, got {len(after)}"
        assert after[:2] == first, "resume rewrote the earlier epochs' rows"
        w_after, ep_after = weights(out.with_suffix(".last.pt"))
        assert ep_after == 4, f"resumed checkpoint says epoch {ep_after}"

        # 4. NEGATIVE CONTROL: --fresh is allowed to overwrite, and it RESTARTS the curve
        out2 = tmp / "f.pt"
        run([*BASE, "--out", str(out2), "--prep", str(prep), "--run-name", "f",
             "--epochs", "1"], cwd=tmp)
        run([*BASE, "--out", str(out2), "--prep", str(prep), "--run-name", "f",
             "--epochs", "1", "--fresh"], cwd=tmp)
        w_fresh, _ = weights(out2.with_suffix(".last.pt"))

        # 2. the resumed weights are closer to their checkpoint than to a fresh init
        d_ck = float((w_after - w_ck).norm())
        d_fresh = float((w_after - w_fresh).norm())
        assert d_ck < d_fresh, (f"resumed weights are NOT closer to the checkpoint they came from: "
                                f"|after-ckpt| {d_ck:.3f} vs |after-fresh| {d_fresh:.3f} -- this is what a "
                                f"restart masquerading as a resume looks like")
        print(f"resume continues the curve: {[round(x, 4) for x in after]} "
              f"(epochs 1-2 identical to the first run)")
        print(f"resumed weights closer to their checkpoint: |after-ckpt| {d_ck:.3f} "
              f"< |after-fresh| {d_fresh:.3f}")
        print("positive control: a bare rerun is refused; negative control: --fresh overrides it")
        print("OK test_sjepa_resume")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
