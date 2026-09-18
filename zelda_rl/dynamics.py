"""Latent dynamics over a frozen behaviour-shaped encoder.

    z_t     = E(s_t)                  frozen, state-only
    z_hat   = M(z_t, a_t)             this module
    target  = E(s_{t+k})              frozen encoder applied to the true state k frames later

No decoder is in the loss path: the model is never charged with reconstructing the game, only with
predicting the part of it the encoder kept. Reward-relevant fields are read back afterwards through the
*same* frozen decoder used in paper/paper.md Table 5, so these numbers sit directly against that bound.

Anchored on the papers rather than guessed (I-JEPA arXiv 2301.08243 and its reference implementation,
V-JEPA 2404.08471 Eq. 1, V-JEPA 2 / V-JEPA 2-AC 2506.09985 Eq. 2-3, DINO-WM 2411.04983):

  * the loss is **L1**, not MSE -- MSE over a high-dimensional latent is dominated by a few large dims;
  * normalisation goes on the **target** (I-JEPA's `train.py` layer-norms `h` before the loss), not on the
    prediction alone. Normalising only the output makes the loss metric asymmetric;
  * **no residual step** by default. A residual makes the identity the initialisation, so a model that loses
    to the do-nothing baseline has learned a delta anti-correlated with the true one;
  * a **rollout loss** over a short horizon, which V-JEPA 2-AC adds explicitly to cut error accumulation;
  * a **stride** well above one frame. V-JEPA 2-AC trains at 4 fps, V-JEPA at temporal stride 4, and DINO-WM
    uses frameskip because "nearby observations can be highly similar"; V-JEPA's own ablation found that too
    little coverage "results in a trivial prediction task". At 10 Hz with stride 1 this game's state is
    unchanged in 74% of steps -- exactly that trivial task.

Freezing E (as V-JEPA 2-AC and DINO-WM also do) makes representation collapse structurally impossible, so
no EMA teacher or stop-gradient is needed. The open risk is different: E was trained by behaviour cloning,
which is discriminative, so it may have discarded dynamics information that action selection never used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zelda_rl.bc import _layer


def norm_target(z):
    """I-JEPA normalises the target representation before the loss; apply to both sides symmetrically."""
    return F.layer_norm(z, (z.shape[-1],))


class LatentDynamics(nn.Module):
    def __init__(self, latent: int = 128, n_actions: int = 63, hidden: int = 512, act_emb: int = 32,
                 layers: int = 3, dropout: float = 0.0, residual: bool = False):
        super().__init__()
        self.latent, self.n_actions, self.residual = latent, n_actions, residual
        self.act = nn.Embedding(n_actions, act_emb)
        blocks, d = [], latent + act_emb
        for _ in range(layers):
            blocks += [_layer(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*blocks)
        self.head = _layer(hidden, latent, 0.01)

    def forward(self, z, a):
        y = self.head(self.trunk(torch.cat([z, self.act(a)], -1)))
        return z + y if self.residual else y

    @torch.no_grad()
    def rollout(self, z0, actions):
        """z0 (B, latent), actions (B, T) -> (B, T, latent): iterate the model on its own predictions."""
        z, out = z0, []
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            out.append(z)
        return torch.stack(out, 1)
