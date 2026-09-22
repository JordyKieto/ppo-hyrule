"""Credential loading. Values are only ever returned to code — never printed or logged.

Sources: KEY=VALUE lines in $ZRL_SECRETS_FILE (default: sunnypilot's secrets.yaml) for RUNPOD /
UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN; WANDB_API_KEY from env or ~/.netrc.
"""
from __future__ import annotations

import netrc
import os
from pathlib import Path

DEFAULT_SECRETS_FILE = Path.home() / "Documents/dev/sunnypilot/secrets.yaml"


def load_secrets() -> dict[str, str]:
    path = Path(os.environ.get("ZRL_SECRETS_FILE", DEFAULT_SECRETS_FILE))
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    for k in ("RUNPOD", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "WANDB_API_KEY"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    if "WANDB_API_KEY" not in out:
        try:
            auth = netrc.netrc().authenticators("api.wandb.ai")
            if auth and auth[2]:
                out["WANDB_API_KEY"] = auth[2]
        except (FileNotFoundError, netrc.NetrcParseError):
            pass
    return out


def require(key: str) -> str:
    val = load_secrets().get(key)
    if not val:
        raise KeyError(f"missing credential {key!r} (set it in $ZRL_SECRETS_FILE or the environment)")
    return val


def pod_secrets_file_content() -> str:
    """Contents for the mode-600 env file shipped to pods (sourced by the launch command, never on a command line)."""
    s = load_secrets()
    keys = ("WANDB_API_KEY", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")
    out = "".join(f"export {k}='{s[k]}'\n" for k in keys if s.get(k))
    # AWS creds so a pod can stream its dataset to S3 AS IT COLLECTS. Without them pod-local storage is the only
    # copy, and terminating the fleet destroys it -- which is exactly how a 20M-transition collection was lost.
    try:
        import boto3
        c = boto3.Session(profile_name="race-wm").get_credentials().get_frozen_credentials()
        if c.access_key and c.secret_key:
            out += (f"export AWS_ACCESS_KEY_ID='{c.access_key}'\n"
                    f"export AWS_SECRET_ACCESS_KEY='{c.secret_key}'\n"
                    f"export AWS_DEFAULT_REGION='us-west-1'\n")
            if getattr(c, "token", None):
                out += f"export AWS_SESSION_TOKEN='{c.token}'\n"
    except Exception:
        pass
    return out
