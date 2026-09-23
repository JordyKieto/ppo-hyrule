"""Static environment tokens and temporal windows, shared by both world models so they cannot drift apart.

Two refinements taken from how Wayformer and AlphaStar actually structure scene input, adapted to this game
rather than copied:

STATIC ENVIRONMENT AS KEYS/VALUES ONLY. Walls are context, not agents. At a 7x7 neighbourhood there are 49
environment tokens against at most 10 entity tokens, so they are 83% of the pool. If they were queries too the
attention would grow from 10x11 to 59x60 pairs, we would compute 49 query outputs that no prediction head
reads, and every one of those rows would contribute gradient paths with no causal bearing on the dynamics. A
wall has no dynamics to predict. `Attn` therefore takes them through its `ctx` argument, which is keys/values
only -- the bipartite constraint both systems impose, and the reason their entity transformers stay stable at
entity counts where a dense block does not.

TEMPORAL WINDOW. Wayformer's central trick is that an agent's history is summarised by a LOCAL encoder into one
token before the global attention runs -- a vehicle's 20 ticks become one agent token, not 20 tokens. Our
tokenizer was strictly Markovian on a single frame, which withholds exactly the information the task needs: the
OOCDM ablation measured a Baba's HEIGHT (`a_rel_y`, i.e. how far it has emerged from the ground) as the dominant
predictor of its behaviour state at ~90x baseline loss. That is a phase variable, and its RATE is only
observable across frames. Velocity is in the state; acceleration -- which is what a button press physically
controls -- is not.
"""
from __future__ import annotations

import numpy as np
import torch

from zelda_rl.combat import BG_WALL


def wall_map_from_packed(link_cell: np.ndarray, link_bits: np.ndarray, bg_offset: int, n: int) -> np.ndarray:
    """(n, n) static wall map, rebuilt from the PACKED arrays so no repacking is needed.

    `link_bits` is the bg-flag block of the link token (already 0/1 floats) and `bg_offset` is BG_WALL's index
    within it. Accumulating Link's wall contacts over the whole dataset traces the walkable boundary; the
    dataset has no collision mesh, so this is the only geometry available.
    """
    hit = np.asarray(link_bits[:, bg_offset]) > 0.5
    m = np.zeros((n, n), np.float32)
    c = np.asarray(link_cell)[hit]
    if len(c):
        m[np.clip(c[:, 1], 0, n - 1), np.clip(c[:, 0], 0, n - 1)] = 1.0
    return m


def patch_offsets(radius: int, device) -> torch.Tensor:
    """The fixed (P, 2) grid of cell offsets in a (2r+1)^2 neighbourhood, ordered by distance from the centre."""
    r = radius
    g = torch.arange(-r, r + 1, device=device)
    dy, dx = torch.meshgrid(g, g, indexing="ij")
    off = torch.stack([dx.reshape(-1), dy.reshape(-1)], -1)
    d = off.float().norm(dim=-1)
    return off[d.argsort()]


def wall_patch(link_cell: torch.Tensor, wmap: torch.Tensor, offsets: torch.Tensor):
    """Local environment tokens around Link.

    Returns (occ, dist):
      occ  (B, P) 1.0 where that neighbouring cell is a wall (or out of bounds, which is also impassable)
      dist (B, P) distance of each offset from Link's cell, in cells, for the attention distance bias
    Content is the OFFSET plus occupancy, never an absolute coordinate -- the patch moves with Link, so it is
    translation-invariant by construction.
    """
    B = link_cell.shape[0]
    n = wmap.shape[0]
    cells = link_cell[:, None, :].long() + offsets[None]                       # (B, P, 2)
    oob = ((cells < 0) | (cells >= n)).any(-1)
    c = cells.clamp(0, n - 1)
    occ = wmap[c[..., 1], c[..., 0]]
    occ = torch.where(oob, torch.ones_like(occ), occ)                          # off-grid counts as wall
    dist = offsets.float().norm(dim=-1)[None].expand(B, -1)
    return occ, dist


def window_index(idx: torch.Tensor, K: int, run_start: torch.Tensor) -> torch.Tensor:
    """(B, K) indices for a K-frame history ending at `idx`, clamped so it never crosses an episode start.

    Clamping rather than masking: repeating the first frame of the episode is a truthful statement that nothing
    earlier exists, whereas a zero row would be a state the game can never be in and the encoder would have to
    learn to recognise it.
    """
    off = torch.arange(K - 1, -1, -1, device=idx.device)[None]                 # oldest .. newest
    return torch.maximum(idx[:, None] - off, run_start[:, None])


def local_frames(rel_cell, actor_off, link_off, yaw_sc, cam_sc, cell_wh, scale=1.0):
    """Each actor's offset from Link expressed in BOTH local frames. (B, R, 4): fwd/right in Link's FACING
    frame, then fwd/right in the CAMERA frame.

    Two frames because two different mechanics need them:
      * the CAMERA frame is what movement lives in -- the stick is camera-relative, so whether a given input
        moves Link toward or away from a Baba is only defined once you know the camera. The ablation measured
        `l_cam` at +21% on the position head, so the model is already reaching for this.
      * LINK'S FACING frame is what combat lives in -- the sword hits in front of him, so "is the Baba in
        front or behind" decides whether an attack connects at all. `bearing` only gives the absolute angle;
        the model would have to do the trig itself to recover this.

    Translation invariance came from being egocentric; this adds ROTATION invariance, which is the part
    Wayformer treats as essential and we did not have. Derived from already-packed fields, so no repacking:
    relative cell + offsets give the displacement, and the link token already carries sin/cos of both yaws.

    OoT convention (see combat.py::_facing): yaw 0 = +z, so the facing unit vector is (sin, cos) in (x, z).
    """
    d = (rel_cell.float() + actor_off - link_off[:, None, :]) * cell_wh        # (B, R, 2) world units
    dx, dz = d[..., 0], d[..., 1]
    out = []
    for sc in (yaw_sc, cam_sc):
        sn, cs = sc[:, 0:1], sc[:, 1:2]
        out += [dx * sn + dz * cs,                                            # along the facing direction
                dx * cs - dz * sn]                                            # perpendicular (right)
    return (torch.stack(out, -1) / scale).clamp(-1, 1)
