"""Run logging: console + train.log, episodes.jsonl / updates.jsonl, and wandb (sunnypilot conventions)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

LOGGER = logging.getLogger("zelda_rl")


def setup_logging(run_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        LOGGER.addHandler(h)


class RunLogger:
    def __init__(self, run_dir: Path, config: dict, *, use_wandb: bool, project: str, entity: str | None,
                 run_name: str, resume: bool, tags: list[str] | None = None, group: str | None = None):
        self.run_dir = run_dir
        setup_logging(run_dir)
        self._episodes = open(run_dir / "episodes.jsonl", "a")
        self._updates = open(run_dir / "updates.jsonl", "a")
        self.run = None
        if use_wandb:
            if wandb is None:
                raise RuntimeError("wandb not installed (pip install wandb) or pass --no-wandb")
            id_path = run_dir / "wandb_run_id.txt"
            run_id = id_path.read_text().strip() if resume and id_path.exists() else None
            if run_id is None:
                from wandb.sdk.lib import runid
                run_id = runid.generate_id()
                id_path.write_text(run_id)
            self.run = wandb.init(project=project, entity=entity, name=run_name, id=run_id,
                                  resume="allow" if resume else None, config=config, tags=tags, group=group,
                                  dir=str(run_dir))
            wandb.define_metric("global_step")
            for prefix in ("episode", "goal", "combat", "reward", "by_slot", "ppo", "perf", "fleet", "charts"):
                wandb.define_metric(f"{prefix}/*", step_metric="global_step")
            LOGGER.info("wandb run: %s", self.run.url)

    def episode(self, ep: dict, global_step: int) -> None:
        self._episodes.write(json.dumps({"global_step": global_step, **ep}) + "\n")
        self._episodes.flush()

    def update(self, metrics: dict, global_step: int) -> None:
        self._updates.write(json.dumps({"global_step": global_step, **metrics}) + "\n")
        self._updates.flush()
        if self.run is not None:
            wandb.log({"global_step": global_step, **metrics})

    def image(self, key: str, path: Path, global_step: int, caption: str = "") -> None:
        if self.run is not None:
            wandb.log({"global_step": global_step, key: wandb.Image(str(path), caption=caption)})

    def summary(self, values: dict) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def artifact(self, path: Path, ttl_days: int = 7) -> None:
        if self.run is None:
            return
        try:
            import datetime
            art = wandb.Artifact(name=path.name.replace(".", "_"), type="model")
            art.add_file(str(path))
            art.ttl = datetime.timedelta(days=ttl_days)
            self.run.log_artifact(art)
        except Exception:
            LOGGER.exception("wandb artifact upload failed (non-fatal; local checkpoint is the durable copy)")

    def close(self) -> None:
        self._episodes.close()
        self._updates.close()
        if self.run is not None:
            wandb.finish()
