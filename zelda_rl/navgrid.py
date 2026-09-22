"""Deterministic position <-> navigation-grid codec, so position stops being a raw regressed coordinate.

Raw XZ position is a terrible model input and a worse output. It spans hundreds of world units with no natural
scale, it is unbounded, and as a regression target under MSE it is minimised by the conditional MEAN -- which
for a bimodal next state (kept walking / hit the wall and stopped) is a point the game cannot occupy. Every
time this project moved a quantity from regression to classification it got dramatically better:

    Link HP, regressed then differenced -> damage F1 0.286
    Link HP as classification            -> 0.906
    displacement magnitude, regressed    -> 0.11x of true on moving frames
    displacement magnitude, binned       -> 1.04x

This is the same move for position: a coarse CELL INDEX (classified) plus a small bounded WITHIN-CELL OFFSET
(regressed over [-0.5, 0.5]). The cell carries the multi-modal "where in the room" decision where a mean is
meaningless; the offset carries sub-cell precision where a mean is fine because the range is tiny.

The codec is exactly LOSSLESS -- `decode(encode(p)) == p` to float precision -- so nothing is thrown away by
adopting it, and the grid resolution only bounds the error of a PREDICTED cell, never of the representation.

Bounds are fitted from the data (robust percentiles), never guessed; `AGENTS.md` §3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NavGrid:
    """An n x n grid over world XZ. `y` is deliberately not gridded: it varies by ~7 units against ~200 for
    x and z (one flat room), so a grid there would be all one cell."""

    lo: np.ndarray            # (2,) world units, x and z
    hi: np.ndarray            # (2,)
    n: int

    @staticmethod
    def fit(xz: np.ndarray, n: int = 64, pad: float = 0.05, pct: float = 0.0) -> "NavGrid":
        """Bounds from the observed extremes, padded.

        `pct` defaults to 0 (true min/max) deliberately. With percentile bounds, 4.55% of real points fell
        outside the grid, their cell index clipped, and the offset absorbed the overflow -- so offsets ranged
        [-3.08, 7.32] instead of [-0.5, 0.5] and stopped being the small bounded quantity the whole split is
        for. Covering the full observed range costs a slightly larger cell and keeps every offset in range.
        """
        p = np.asarray(xz, np.float64).reshape(-1, 2)
        p = p[np.isfinite(p).all(1)]
        lo, hi = np.percentile(p, pct, axis=0), np.percentile(p, 100 - pct, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        return NavGrid(lo - pad * span, hi + pad * span, n)

    @property
    def cell(self) -> np.ndarray:
        return (self.hi - self.lo) / self.n

    def to_cell(self, xz: np.ndarray):
        """world XZ -> (integer cell index per axis, offset in [-0.5, 0.5] per axis). Clipped to the grid."""
        p = np.asarray(xz, np.float32).reshape(-1, 2)
        u = (p - self.lo) / (self.hi - self.lo) * self.n          # continuous cell coordinate
        idx = np.clip(np.floor(u), 0, self.n - 1)
        return idx.astype(np.int64), (u - idx - 0.5).astype(np.float32)

    def from_cell(self, idx: np.ndarray, off: np.ndarray) -> np.ndarray:
        """(cell index, offset) -> world XZ. Exact inverse of `to_cell` inside the grid."""
        u = np.asarray(idx, np.float32) + np.asarray(off, np.float32) + 0.5
        return (u / self.n * (self.hi - self.lo) + self.lo).astype(np.float32)

    def onehot(self, xz: np.ndarray) -> np.ndarray:
        """world XZ -> (N_points, 2n + 2): one-hot of the x cell, one-hot of the z cell, then the 2 offsets.

        Per-AXIS one-hots rather than one n^2 one-hot: 2n inputs instead of n^2, and adjacent cells on one axis
        stay adjacent in the encoding, so 'one cell east' is a small change rather than an unrelated index.
        """
        idx, off = self.to_cell(xz)
        m = len(idx)
        out = np.zeros((m, 2 * self.n + 2), np.float32)
        out[np.arange(m), idx[:, 0]] = 1.0
        out[np.arange(m), self.n + idx[:, 1]] = 1.0
        out[:, 2 * self.n:] = off
        return out

    @property
    def feat_dim(self) -> int:
        return 2 * self.n + 2

    def quantisation_error(self, xz: np.ndarray) -> float:
        """Mean world-unit error if only the CELL were kept and the offset discarded -- the floor on how well
        a predicted cell can do. The codec itself is lossless; this bounds a prediction."""
        idx, _ = self.to_cell(xz)
        centres = self.from_cell(idx, np.zeros_like(idx, np.float32))
        return float(np.linalg.norm(np.asarray(xz, np.float32).reshape(-1, 2) - centres, axis=1).mean())

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "n": self.n}

    @staticmethod
    def load(d):
        return NavGrid(np.asarray(d["lo"], np.float64), np.asarray(d["hi"], np.float64), int(d["n"]))


# --- Actor nav grid: one channel per TYPE, which is what a convolution can actually use -------------------
#
# Channels are TYPES, not actor slots. A convolution over "where is Link / where are the Babas" is
# translation-equivariant, so "Link moved one cell east" is a shift of the input rather than an unrelated
# change in two float coordinates. Actor slot identity is deliberately NOT in the grid -- the reward needs
# per-row hp and behaviour state, and those travel alongside as a flat vector. The grid carries WHERE; the
# flat vector carries WHAT.
CH_LINK, CH_BABA, CH_BABA_HP, CH_WALL, N_CH = 0, 1, 2, 3, 4


def wall_map(grid: "NavGrid", link_xz: np.ndarray, wall_hit: np.ndarray) -> np.ndarray:
    """(n, n) static map of cells where Link's BGCHECKFLAG_WALL bit ever fired.

    The dataset has no room geometry -- no collision mesh, no occupancy map -- but it does record, per frame,
    whether Link is touching a wall. Accumulating those contacts over every frame traces the boundary of the
    walkable region: 120 cells against 447 that Link ever visits, at n=32. Static, so the model never has to
    predict it, and it gives a displacement head a reason not to walk through a wall.
    """
    idx, _ = grid.to_cell(link_xz)
    m = np.zeros((grid.n, grid.n), np.float32)
    m[idx[wall_hit, 1], idx[wall_hit, 0]] = 1.0
    return m
