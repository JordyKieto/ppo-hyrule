"""Upload a checkpoint to S3 the moment it is written. No flag, no operator step, no end-of-session sync.

WHY THIS EXISTS: on 2026-10-15 a 34-pod fleet vanished without warning and took the checkpoints of FIFTEEN
arms with it, because `fleet sync` is a manual command and nobody had run it. Eleven hours of training
survived only as W&B metric curves -- no weights, so not one arm could be resumed or re-evaluated. A rule that
says "remember to sync" is not a rule (AGENTS 21); the write itself has to do it.

Pods already hold AWS credentials for exactly this reason (`fleet/secrets.py` ships them so a pod can stream
its dataset to S3 as it collects), so there is nothing to plumb. Failure is logged LOUDLY and never raises:
losing a checkpoint copy must not kill a training run, but it must not be silent either.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

BUCKET = os.environ.get("ZRL_S3_BUCKET", "zelda-rl")
REGION = os.environ.get("ZRL_S3_REGION", "us-west-1")
PREFIX = os.environ.get("ZRL_CKPT_PREFIX", "fleet/ckpt_auto")
LOGGER = logging.getLogger("ckpt_s3")
_state = {"client": None, "warned": False, "ok": 0, "fail": 0}


def _client():
    if _state["client"] is not None:
        return _state["client"]
    try:
        import boto3
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            s = boto3.Session(region_name=REGION)                 # pod: creds in the environment
        else:
            s = boto3.Session(profile_name=os.environ.get("ZRL_S3_PROFILE", "race-wm"), region_name=REGION)
        _state["client"] = s.client("s3")
    except Exception as e:                                        # noqa: BLE001
        if not _state["warned"]:
            LOGGER.warning("CHECKPOINTS ARE NOT BEING BACKED UP: no S3 client (%s). If this is a pod, the "
                           "run's weights die with it.", str(e)[:120])
            _state["warned"] = True
        _state["client"] = False
    return _state["client"]


def upload(path: Path, run_name: str, update: int, *, blocking: bool = False) -> None:
    """Copy `path` to s3://BUCKET/PREFIX/<run_name>/<file>. Called from the trainer's save, never by hand."""
    c = _client()
    if not c:
        return
    key = f"{PREFIX}/{run_name}/{Path(path).name}"

    def _do():
        try:
            c.upload_file(str(path), BUCKET, key)
            _state["ok"] += 1
            LOGGER.info("checkpoint -> s3://%s/%s (update %d)", BUCKET, key, update)
        except Exception as e:                                    # noqa: BLE001
            _state["fail"] += 1
            LOGGER.error("CHECKPOINT BACKUP FAILED for %s: %s -- this run's weights exist ONLY on this pod",
                         key, str(e)[:160])

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()


def summary() -> str:
    return f"{_state['ok']} checkpoint uploads ok, {_state['fail']} failed"
