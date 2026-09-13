import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Atari-style CNN actor-critic (Mnih et al. 2015) on stacked 84x84 frames."""

    def __init__(self, in_ch: int = 4, n_actions: int = 12):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.pi = nn.Linear(512, n_actions)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h)
