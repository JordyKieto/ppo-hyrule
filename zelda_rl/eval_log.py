"""Make a non-durable evaluation impossible.

WHY THIS EXISTS, in one sentence: on 2026-10-16 the consolidated RvS run played **3,118 held-out episodes**
across six pods, its only record was `print()` into `logs/consolidated_r*.log` on the pod volumes, the pods
were terminated, and the measurement that answered "does offline RvS match our best PPO at killing" is now
unrecoverable -- `train_rvs.py` and `train_wm_obs.py` contained **zero** W&B calls between them, and the
pre-shutdown S3 save globbed `logs/rvs*.log`, which did not match `consolidated_r*.log`.

The weights survived; the evidence did not. An eval that exists only in a process's stdout has already been
lost, so this module is the only way the trainers in this repo are allowed to report one:

  sink = EvalSink(run_label="consolidated_r0", project="zelda-rl-rvs", group="rvs-consolidated", rank=0)
  sink.log(step=epoch, metrics={...}, rows=episode_dicts)

`log` writes THREE copies before it returns: the per-episode rows appended to a local jsonl, that jsonl
uploaded to S3, and the metrics into W&B. Construction FAILS LOUDLY if W&B is unusable, because a trainer
that evaluates into the void should not start -- `ZRL_ALLOW_NO_WANDB=1` is the deliberate, named override
(tests and offline smoke runs use it), and it still writes the jsonl and the S3 copy.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from zelda_rl import ckpt_s3

ALLOW_NO_WANDB = "ZRL_ALLOW_NO_WANDB"


class EvalLogUnavailable(RuntimeError):
    """Raised at construction: refuse to train rather than evaluate into a pod-local log file."""


class EvalSink:
    def __init__(self, run_label: str, project: str, group: str | None = None, rank: int = 0,
                 out_dir: str | Path = "runs/evals", config: dict | None = None,
                 wandb_mod=None, uploader=None):
        self.label = run_label
        self.rank = rank
        self.path = Path(out_dir) / f"{run_label}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_rows = 0
        self._upload = uploader if uploader is not None else ckpt_s3.upload
        self._wandb = None
        self._run = None
        allow = os.environ.get(ALLOW_NO_WANDB) == "1"
        mod = wandb_mod
        if mod is None and not allow:
            try:
                import wandb as mod                              # noqa: PLC0415
            except Exception as e:                               # noqa: BLE001
                raise EvalLogUnavailable(
                    f"wandb is not importable ({str(e)[:80]}). Every evaluation in this repo is logged to "
                    f"W&B; a run whose evals live only in a pod log is a run whose results die with the "
                    f"pod (3,118 episodes were lost that way on 2026-10-16). Install wandb, or set "
                    f"{ALLOW_NO_WANDB}=1 deliberately to accept jsonl+S3 only.") from e
        if mod is not None:
            try:
                # NO settings= here. Passing Settings(start_method="thread") made wandb 0.30 raise
                # "Extra inputs are not permitted", wandb.init failed, and this class then refused to let
                # the trainer start -- the gate working exactly as designed, against a bug of its own.
                self._run = mod.init(project=project, group=group, name=run_label,
                                     config=config or {}, reinit=True)
                self._wandb = mod
            except Exception as e:                               # noqa: BLE001
                if not allow:
                    raise EvalLogUnavailable(
                        f"wandb.init failed ({str(e)[:90]}). Fix the credentials or set {ALLOW_NO_WANDB}=1 "
                        f"to accept jsonl+S3 only -- but then nothing aggregates across pods.") from e
        print(f"eval sink: {self.path}"
              f"{' + wandb ' + project if self._wandb else ' (NO WANDB -- jsonl + S3 only)'}", flush=True)

    def log(self, step: int, metrics: dict, rows: list[dict] | None = None) -> None:
        """Append rows, upload the jsonl, log the metrics. Durable before it returns."""
        if rows:
            with self.path.open("a") as f:
                for r in rows:
                    f.write(json.dumps({"step": step, "rank": self.rank, "run_label": self.label,
                                        "ts": time.time(), **r}) + "\n")
            self.n_rows += len(rows)
        try:
            self._upload(self.path, f"evals/{self.label}", step)
        except Exception as e:                                   # noqa: BLE001
            print(f"  EVAL ROWS NOT BACKED UP to S3: {str(e)[:90]} -- they exist only on this machine",
                  flush=True)
        if self._wandb is not None:
            try:
                self._wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            except Exception as e:                               # noqa: BLE001
                print(f"  EVAL METRICS NOT IN WANDB: {str(e)[:90]}", flush=True)

    def close(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:                                    # noqa: BLE001
                pass
