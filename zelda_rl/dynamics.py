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


# --- What an action DOES: motion and combat events, conditioned on (z_t, a_t) -------------------------
#
# The audit that motivates this: across all 63 constant actions in the dream, max kills 0.00, max damage
# dealt 0.70 of the 6 HP available, and the damage that did appear came from SHIELD actions -- which cannot
# deal damage in the real game. `approach` paid <= 0 for every action and the best Z-press acquired 0.90
# lock-ons against a cap of 6. None of it is earnable, because the reward reads HP and position changes out
# of a decoded state whose predictor ties the identity: nothing changes, so the only changes are decode noise.
#
# Two design points, both patterns that already worked here:
#
#   * CONDITION ON THE ACTION DIRECTLY. Displacement direction is close to a function of the stick direction
#     and camera, and a sword hit is a function of swinging while near and facing a Baba -- but the action
#     barely survives into the latent (inverse dynamics 0.089 against a 0.220 ceiling). Reading these out of
#     z_{t+1} routes the quantities that need the action through the weakest channel. This is the event-head
#     pattern: predict events from a latent transition rather than differencing a decoded state.
#   * CLASSIFY, DO NOT REGRESS. On truly-moving steps the regressed magnitude was 1.19 against a real 11.03,
#     9.3x too small -- regression to the conditional mean under uncertainty. The same failure made HP
#     unusable until it was classified (damage F1 0.286 -> 0.906). Direction and magnitude are binned, and a
#     mode does not shrink. Sectors are 16 so they align with the 9 stick directions of the action space.
N_DIR_SECTORS = 16
MAG_EDGES = (1.0, 3.0, 6.0, 9.0, 12.0, 16.0, 25.0, 1e9)      # 8 bins over |XZ| in world units
N_MAG_BINS = len(MAG_EDGES)


def mag_centers() -> "np.ndarray":
    import numpy as np
    lo = (0.0,) + MAG_EDGES[:-1]
    c = [(a + b) / 2 for a, b in zip(lo, MAG_EDGES)]
    c[-1] = 35.0                                              # open top bin: a representative, not (25+1e9)/2
    return np.asarray(c, dtype="float32")


def dir_unit_vectors() -> "np.ndarray":
    import numpy as np
    a = (np.arange(N_DIR_SECTORS) + 0.5) * (2 * np.pi / N_DIR_SECTORS)
    return np.stack([np.cos(a), np.sin(a)], 1).astype("float32")   # (x, z) unit vectors at sector centres


class TransitionHead(nn.Module):
    """(z_t, one_hot(a_t)) -> what this action does over one step.

    moved     1 logit    does Link move at all (he does not, on 95% of real steps)
    dir       16 logits  displacement direction sector
    mag       8 logits   displacement magnitude bin
    dy        1          vertical displacement, regressed (small and nearly always 0)
    hit_row   n_rows+1   which actor row loses HP this step, or "none" (the last index)
    link_hit  1 logit    Link loses HP this step
    gone_row  n_rows+1   which actor row DESPAWNS this step, or "none"

    `gone_row` exists because a Deku Baba that dies is removed from the actor list, and without it a killed
    Baba lingers at HP 0 forever -- so `row_present` and `baba_count` disagree with the enemy dict, and the
    room can never actually empty.
    """

    def __init__(self, latent: int, n_actions: int, n_rows: int = 16, hidden: int = 512):
        super().__init__()
        self.latent, self.n_actions, self.n_rows = latent, n_actions, n_rows
        self.trunk = nn.Sequential(nn.Linear(latent + n_actions, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.moved = nn.Linear(hidden, 1)
        self.dir = nn.Linear(hidden, N_DIR_SECTORS)
        self.mag = nn.Linear(hidden, N_MAG_BINS)
        self.dy = nn.Linear(hidden, 1)
        self.hit_row = nn.Linear(hidden, n_rows + 1)
        self.link_hit = nn.Linear(hidden, 1)
        self.gone_row = nn.Linear(hidden, n_rows + 1)

    def forward(self, z: torch.Tensor, a_onehot: torch.Tensor) -> dict:
        h = self.trunk(torch.cat([z, a_onehot], -1))
        return {"tr_moved": self.moved(h), "tr_dir": self.dir(h), "tr_mag": self.mag(h),
                "tr_dy": self.dy(h), "tr_hit_row": self.hit_row(h), "tr_link_hit": self.link_hit(h),
                "tr_gone_row": self.gone_row(h)}
