"""The latent predictor, in one place, because five scripts were each constructing it by hand.

`scripts/train_joint.py`, `scripts/train_dream_ppo.py`, `scripts/eval_rollout.py`,
`scripts/diagnose_dream.py` and `scripts/probe_action_effect.py` each wrote out the same
`nn.Sequential(Linear(LAT+NA, 512), ...)` and called it the same way. That is the duplication that produced
the loader bugs, so the definition lives here and they import it.

RESIDUAL, which the flat version was not. The predictor used to map (z, a) -> z_{t+1} directly, rebuilding
all 128 latent dimensions from scratch every step. Measured consequence: decoded per-step displacement had
mean 12.11 world units against a true mean of 0.58 -- Link is stationary on 95% of real steps and the model
moved him every single step, 20.9x too far, in incoherent directions (so net drift stayed near zero and he
never reached the door). Any error in reconstructing 128 dims shows up as motion.

With z + f(z, a), "nothing changed" is f = 0, which is what a zero-initialised last layer already outputs.
This project's own dynamics sweep measured the same change on the frozen-BC latent as skill -7.24 -> -1.73.

`build_predictor` therefore zero-initialises the output layer, so an untrained predictor is exactly the
identity -- the control that the whole skill metric is defined against.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualPredictor(nn.Module):
    """z_{t+1} = z_t + f(z_t, a_t). Action arrives one-hot, concatenated."""

    def __init__(self, latent: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.latent, self.n_actions = latent, n_actions
        self.net = nn.Sequential(nn.Linear(latent + n_actions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)      # untrained == identity == the skill metric's control

    def forward(self, za: torch.Tensor) -> torch.Tensor:
        """`za` is cat([z, one_hot(a)], -1), matching the old flat predictor's call convention."""
        z = za[..., :self.latent]
        return z + self.net(za)

    def delta(self, za: torch.Tensor) -> torch.Tensor:
        return self.net(za)


class FlatPredictor(nn.Module):
    """The pre-residual predictor, kept only so checkpoints written before this change still load."""

    def __init__(self, latent: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent + n_actions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))

    def forward(self, za: torch.Tensor) -> torch.Tensor:
        return self.net(za)


def build_predictor(latent: int, n_actions: int, residual: bool = True, hidden: int = 512) -> nn.Module:
    return (ResidualPredictor if residual else FlatPredictor)(latent, n_actions, hidden)


def load_predictor(ck: dict, latent: int, n_actions: int) -> nn.Module:
    """Rebuild the predictor a checkpoint was saved with.

    Checkpoints from before the residual change stored a bare `nn.Sequential` state dict with `0.weight`
    style keys and no `residual` flag; new ones are `ResidualPredictor` with `net.0.weight`. Detect by key
    shape rather than trusting the flag to be present, so old artefacts keep working.
    """
    sd = ck["pred"]
    residual = ck.get("pred_residual")
    if residual is None:
        residual = any(k.startswith("net.") for k in sd)
    m = build_predictor(latent, n_actions, residual=residual)
    if not residual and not any(k.startswith("net.") for k in sd):
        m.net.load_state_dict(sd)           # bare Sequential keys: 0.weight, 2.weight, ...
    else:
        m.load_state_dict(sd)
    m.eval()
    return m
