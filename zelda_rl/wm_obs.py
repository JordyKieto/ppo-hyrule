"""The Gohma observation-space world model, shared by its trainer and the dream.

Defined here rather than in the training script so `scripts/train_dream_obs_ppo.py` reconstructs the SAME
architecture the checkpoint was saved from -- a dream that silently builds a different model would load the
state dict and roll garbage.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


N_DIRS = 9                      # 8 compass directions + neutral; the action index is dir * n_btn + btn


def load_episodes(dirs, limit=0, min_steps=120):
    eps, bad = [], 0
    files = [f for d in dirs for f in sorted(Path(d).glob("*.npz"))]
    if limit:
        files = files[:limit]
    fields = None
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            if "obs" not in z.files or "action" not in z.files:
                continue
            obs = z["obs"].astype(np.float32)
            act = z["action"].astype(np.int64)
            if len(obs) < min_steps:
                continue
            if fields is None and "obs_fields" in z.files:
                fields = json.loads(str(z["obs_fields"]))
            eps.append({"obs": obs, "act": act, "file": f.name,
                        "win": bool(z["boss_defeated"]) if "boss_defeated" in z.files else False})
        except Exception:                                    # noqa: BLE001
            bad += 1
    print(f"loaded {len(eps)} episodes ({bad} unreadable), "
          f"{sum(len(e['obs']) for e in eps):,} steps", flush=True)
    return eps, fields


def field_index(fields: dict) -> dict:
    """name -> column, for the global block (where hp lives)."""
    idx, off = {}, 0
    for name, w in fields["global"]:
        idx[name] = off
        off += w
    return idx


class ObsWM(nn.Module):
    """GRU over (obs, action) predicting the normalised delta to the next obs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 1024, gru: int = 1024, n_btn: int = 12):
        super().__init__()
        self.obs_dim, self.n_actions, self.n_btn = obs_dim, n_actions, n_btn
        self.act_emb = nn.Embedding(n_actions, 64)
        # the action's GEOMETRY, not just its index (docs/WM_MOTION.md change 2)
        self.enc = nn.Sequential(nn.Linear(obs_dim + 64 + 2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, gru)
        self.head = nn.Linear(gru, obs_dim)
        nn.init.zeros_(self.head.weight)         # start AT persistence: zero delta (AGENTS 1's identity)
        nn.init.zeros_(self.head.bias)

    def act_feat(self, a: torch.Tensor) -> torch.Tensor:
        d = (a // self.n_btn).clamp(max=N_DIRS - 1).float() * (2 * math.pi / N_DIRS)
        return torch.stack([torch.sin(d), torch.cos(d)], -1)

    def step(self, obs: torch.Tensor, a: torch.Tensor, h: torch.Tensor):
        x = torch.cat([obs, self.act_emb(a), self.act_feat(a)], -1)
        h = self.gru(self.enc(x), h)
        return self.head(h), h

    def init_hidden(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.gru.hidden_size, device=device)




def load_wm(path, device="cpu"):
    """Reconstruct the WM and its per-field scales from a checkpoint written by train_wm_obs.py."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ObsWM(int(ck["obs_dim"]), int(ck["n_actions"]), int(ck["hidden"]), int(ck["gru"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    scales = torch.as_tensor(ck["scales"], dtype=torch.float32, device=device)
    return m, scales, ck
