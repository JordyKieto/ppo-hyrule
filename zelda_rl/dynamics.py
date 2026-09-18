"""Latent dynamics over a frozen behaviour-shaped encoder (JEPA-style: the target is an embedding).

    z_t     = E(s_t)                  frozen, state-only
    z_hat   = M(z_t, a_t)             this module
    target  = E(s_{t+1})              frozen encoder applied to the true next state

No decoder is in the loss path by default: the model is never charged with reconstructing the game, only
with predicting the part of it the encoder kept. Reward-relevant fields are read back afterwards through the
*same* frozen decoder used in paper/paper.md Table 5, so latent-dynamics numbers are directly comparable to
the autoencoding upper bound measured there.

The encoder ends in a LayerNorm, so predictions are put back on that manifold rather than allowed to drift
off it; and the step is residual because consecutive latents are close at 10 Hz.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from zelda_rl.bc import _layer


class LatentDynamics(nn.Module):
    def __init__(self, latent: int = 128, n_actions: int = 63, hidden: int = 512, act_emb: int = 32,
                 layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.latent, self.n_actions = latent, n_actions
        self.act = nn.Embedding(n_actions, act_emb)
        blocks, d = [], latent + act_emb
        for _ in range(layers):
            blocks += [_layer(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*blocks)
        self.delta = _layer(hidden, latent, 0.01)      # small init: start near the identity step
        self.ln = nn.LayerNorm(latent)

    def forward(self, z, a):
        d = self.delta(self.trunk(torch.cat([z, self.act(a)], -1)))
        return self.ln(z + d)

    @torch.no_grad()
    def rollout(self, z0, actions):
        """z0 (B, latent), actions (B, T) -> (B, T, latent): iterate the model on its own predictions."""
        z, out = z0, []
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            out.append(z)
        return torch.stack(out, 1)
