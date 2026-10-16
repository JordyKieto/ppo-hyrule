"""Return-conditioned supervised learning: P(a | s, g) instead of P(a | s).

WHY, for this corpus specifically. Behaviour cloning on a mixed corpus averages over outcomes: a state that
appears in a win and in a death pulls the loss toward an action between the escape and the fatal mistake.
Conditioning on what FOLLOWED separates the modes -- (s, high g) -> the action that won, (s, low g) -> the
action that died -- and the thousands of losing episodes become coverage instead of corruption. At inference
the policy is prompted with the outcome we want.

THE CONDITIONING IS OUTCOME-BASED, NOT REWARD-BASED, and that is deliberate. The human corpus was recorded
under the FULL gohma reward profile and every PPO episode under `--reward simple` (39 terms zeroed), so a
return-to-go computed from the recorded reward is on two different scales and not comparable across sources.
The human's own return is also ~80% damage-independent (PROVENANCE 3.3), so it is a poor target even within
one source. These three numbers are read from the OBSERVATION and mean the same thing in both corpora:

    g[0]  damage_to_come   (gohma_hp_frac[t] - gohma_hp_frac[end])      in [0, 1]
    g[1]  win              1.0 if the episode ended in her defeat        in {0, 1}
    g[2]  hurt_to_come     (link_hp_frac[t] - link_hp_frac[end])         in [0, 1]

At inference the prompt is `TARGET = [1.0, 1.0, 0.0]`: take all her remaining HP, win, lose none of ours.
"""
from __future__ import annotations

import json

import numpy as np
import torch

COND_DIM = 3
TARGET = (1.0, 1.0, 0.0)          # damage_to_come, win, hurt_to_come


def field_offsets(obs_fields_json: str) -> dict:
    off, idx = 0, {}
    for name, w in json.loads(obs_fields_json)["global"]:
        idx[name] = off
        off += w
    return idx


def conditioning(obs: np.ndarray, idx: dict, won: bool) -> np.ndarray:
    """Per-step g for one episode, from the episode's own observations."""
    ghp = obs[:, idx["gohma_hp_frac"]]
    lhp = obs[:, idx["link_hp_frac"]]
    dmg = np.clip(ghp - ghp[-1], 0.0, 1.0)
    hurt = np.clip(lhp - lhp[-1], 0.0, 1.0)
    w = np.full(len(obs), 1.0 if won else 0.0, np.float32)
    return np.stack([dmg, w, hurt], 1).astype(np.float32)


class Conditioned(torch.nn.Module):
    """Wraps a policy trained on [obs | g] so a plain env observation can drive it.

    `policy_io.load_policy` returns this automatically when a checkpoint declares `rvs_cond`, which keeps
    `eval_fleet.py` and every other consumer working without knowing about conditioning at all.
    """

    recurrent = True

    def __init__(self, inner, target=TARGET):
        super().__init__()
        self.inner = inner
        self.register_buffer("g", torch.tensor(target, dtype=torch.float32))
        self.hidden_size = getattr(inner, "hidden_size", 256)

    def _cat(self, x):
        return torch.cat([x, self.g.expand(x.shape[0], self.g.shape[0])], dim=-1)

    def init_hidden(self, n):
        return self.inner.init_hidden(n)

    def encode(self, x):
        return self.inner.encode(self._cat(x))

    def step(self, x, h, first):
        return self.inner.step(self._cat(x), h, first)

    def act(self, x, h=None, first=None, action=None, deterministic=False):
        return self.inner.act(self._cat(x), h, first, action=action, deterministic=deterministic)

    def value(self, x, h, first):
        return self.inner.value(self._cat(x), h, first)
