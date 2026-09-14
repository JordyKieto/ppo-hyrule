"""Checkpoint discovery/loading and env construction from a checkpoint's own config (v1 and v2 policies)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.goal import GoalConfig
from zelda_rl.ppo import build_model

REPO = Path(__file__).resolve().parents[1]
INBOX = REPO / "checkpoints" / "inbox"
DEFAULT_SLOTS_DIR = (os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves")
                     if sys.platform == "darwin" else str(REPO / "assets" / "states"))


def newest_checkpoint() -> Path | None:
    for pattern_root, pattern in ((INBOX, "*.pt"), (REPO / "runs", "*/checkpoints/*.pt")):
        cands = sorted(pattern_root.glob(pattern), key=lambda p: p.stat().st_mtime)
        if cands:
            return cands[-1]
    return None


def resolve_checkpoint(arg: str | None) -> Path:
    """None -> newest in inbox/runs; a .pt path; a run name/dir -> its checkpoints/latest.pt."""
    if not arg:
        p = newest_checkpoint()
        if p is None:
            raise FileNotFoundError(f"no checkpoints in {INBOX} or runs/*/checkpoints")
        return p
    p = Path(arg)
    if p.suffix == ".pt" and p.exists():
        return p
    for run_dir in (p, REPO / "runs" / arg):
        if (run_dir / "checkpoints" / "latest.pt").exists():
            return run_dir / "checkpoints" / "latest.pt"
    raise FileNotFoundError(f"checkpoint not found: {arg}")


def load_policy(path: Path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def make_env(cfg: dict, instance_dir: Path, *, slots: list[int], slots_dir: str = DEFAULT_SLOTS_DIR,
             video_backend: str = "Metal", unlimited_speed: bool = False, idx: int = 0, audio: bool = False,
             fullscreen: bool = False):
    """Build a DolphinGoalEnv matching the checkpoint (obs version, goal slot, reward config, step timing)."""
    version = cfg.get("obs_version", 1)
    if cfg.get("task") == "combat":
        from zelda_rl.combat import CombatConfig
        from zelda_rl.obs import read_state_file_v2
        files = {k: Path(slots_dir) / f"NACE01.s{k:02d}" for k in sorted(set(slots))}
        states = {k: read_state_file_v2(f) for k, f in files.items()}
        ccfg = CombatConfig(**{k: (tuple(v) if k == "target_ids" else v) for k, v in cfg["combat_cfg"].items()})
        ecfg = EnvConfig(episode_vi=int(cfg["episode_seconds"] * 60), vi_per_step=cfg["vi_per_step"],
                         video_backend=video_backend, unlimited_speed=unlimited_speed, obs_version=cfg.get("obs_version", 3),
                         task="combat", combat=ccfg, stall_timeout=300.0 if video_backend != "Null" else 10.0,
                         audio=audio, fullscreen=fullscreen)
        return DolphinGoalEnv(idx, instance_dir, files, states, states[min(slots)], ecfg), states
    from zelda_rl.obs import read_state_file, read_state_file_v2
    reader = read_state_file_v2 if version >= 2 else read_state_file
    goal_slot = cfg["goal_slot"]
    files = {k: Path(slots_dir) / f"NACE01.s{k:02d}" for k in sorted(set(slots + [goal_slot]))}
    states = {k: reader(f) for k, f in files.items()}
    if cfg.get("goal_rupees") is not None:           # checkpoint trained against an overridden rupee target
        import dataclasses
        states[goal_slot] = dataclasses.replace(states[goal_slot], rupees=cfg["goal_rupees"])
    gcfg = GoalConfig(**cfg["goal_cfg"])
    ecfg = EnvConfig(episode_vi=int(cfg["episode_seconds"] * 60), vi_per_step=cfg["vi_per_step"],
                     video_backend=video_backend, unlimited_speed=unlimited_speed, obs_version=version, goal=gcfg,
                     audio=audio, fullscreen=fullscreen)
    env = DolphinGoalEnv(idx, instance_dir, files, states, states[goal_slot], ecfg)
    return env, states
