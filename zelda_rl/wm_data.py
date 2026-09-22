"""Tokens + next-step targets for the world model, packed to the narrowest dtype each field admits.

ONE implementation, used by both paths:
  * `scripts/train_wm_tokens.py` in memory, for datasets small enough to load at once;
  * `scripts/pack_dataset.py` a chunk at a time, for ones that are not.

That matters more than it looks. A packer that re-derived these targets would drift from the trainer, and
numbers from a sweep would stop being comparable to numbers from a run without anything looking wrong -- the
same class of bug as the encoder/collector state layouts drifting, which `zelda_rl/state_layout.py` exists to
prevent.

WHY PACKED. The full v5 dataset is 91.6M frames. As f32 tokens plus f32 targets plus a f32 one-hot action that
is 3,260 B/frame = 298 GB; packed it is ~583 B/frame = ~50 GiB, which fits in an 80 GB GPU with room for
activations, so training does no host-device transfer at all. Nothing is lost doing it:
  * continuous fields are ALREADY scaled to [-1,1] by `tokens.build`, so f16's ~3 decimal digits are ample;
  * every index and mask is small (cells 0-63, rel-cells 0-30, ids 0-255, states 0-9, masks 0/1) -> uint8;
  * the action is kept as an INDEX and one-hot'd on the device; its 63-wide f32 form was 252 B/frame, more
    than Link's entire token.
`pres_now` and `st_now` are NOT stored: they are bit-identical to `mask` and `states`, and the trainer aliases
them. `rel_xz` is not stored either -- it was a verbatim copy of `actors[:, :, :2]`.
"""
from __future__ import annotations

import pathlib

import numpy as np

from zelda_rl.bc import LI, RI, _split
from zelda_rl.navgrid import NavGrid
from zelda_rl.tokens import SF1_KEEP, build

DW = 5                       # 5x5 neighbourhood of cells a body can move to in one step
DC = DW * DW
CENTRE = (DW // 2) * DW + DW // 2

# link_off stays f32: besides feeding the model (as f16, cast when it goes to the device) it is the
# HOST-side term in the grid decode that turns a prediction back into world units for the eval and
# rollout metrics. Rounding it to f16 there shifts dpos by ~0.014 world units against a ~2.2 baseline,
# which moved measured skill from -0.0024 to -0.0007 -- small, but it is the reported number.
F16 = ("link", "actors", "actor_off", "cell_xy", "doff", "dy", "dyaw", "r_doff")
F32 = ("link_off",)
U8 = ("ids", "cats", "states", "mask", "link_cell", "actor_rcell", "dcell", "hp_ev",
      "bits_now", "bits_nxt", "scene_now", "scene_nxt", "r_dcell", "r_hp_ev", "pres_nxt", "st_nxt", "act")

def field_shapes(R: int, link_dim: int, actor_dim: int) -> dict[str, tuple]:
    nb = len(SF1_KEEP)
    return {
        "link": (link_dim,), "actors": (R, actor_dim),
        "ids": (R,), "cats": (R,), "states": (R,), "mask": (R,),
        "link_cell": (2,), "link_off": (2,), "actor_rcell": (R, 2), "actor_off": (R, 2), "cell_xy": (R, 2),
        "dcell": (), "doff": (2,), "dy": (1,), "dyaw": (2,), "hp_ev": (),
        "bits_now": (nb,), "bits_nxt": (nb,), "scene_now": (), "scene_nxt": (),
        "r_dcell": (R,), "r_doff": (R, 2), "r_hp_ev": (R,), "pres_nxt": (R,), "st_nxt": (R,),
        "act": (), "ok": (), "slot": (), "lpos": (2,),
    }


def dtype_of(name: str):
    if name in F16:
        return np.float16
    if name in F32:
        return np.float32
    if name in U8:
        return np.uint8
    if name == "ok":
        return np.bool_
    if name == "slot":
        return np.uint8
    if name == "lpos":
        return np.float32
    raise KeyError(name)


def fit_prep(state: np.ndarray, epid: np.ndarray, train_mask: np.ndarray, rows_keep: int, grid_n: int):
    """Fit the per-field scales, the nav grid and the y-delta scale. Run this on a SAMPLE: every fitted
    quantity is a 99.5th percentile or a min/max over positions, and a couple of million frames pins those as
    well as ninety does -- while fitting on the whole set would need the whole set in memory, which is the
    problem this module exists to avoid."""
    lk, rows, _ = _split(state)
    rows = rows[:, :rows_keep]
    pres = rows[:, :, RI["present"]] > 0.5
    grid = NavGrid.fit(np.concatenate([lk[:, [LI["pos_x"], LI["pos_z"]]],
                                       rows[:, :, [RI["pos_x"], RI["pos_z"]]][pres]]), n=grid_n)
    _, sc = build(state, epid, train_mask=train_mask, grid=grid)
    nxt = np.minimum(np.arange(len(state)) + 1, len(state) - 1)
    ok = np.r_[epid[1:] == epid[:-1], False]
    rdy = lk[nxt, LI["pos_y"]] - lk[:, LI["pos_y"]]
    m = train_mask & ok
    sdy = float(max(np.percentile(np.abs(rdy[m]), 99.5), 1e-3)) if m.any() else 1.0
    sc["d_link_y"] = sdy
    return sc, grid, sdy


def tokens_and_targets(state, act, slot, epid, *, grid: NavGrid, scales: dict, d_link_y: float,
                       rows_keep: int, n_actions: int) -> dict[str, np.ndarray]:
    """One chunk (or a whole dataset) -> dict of PACKED arrays. Chunks must contain WHOLE EPISODES: Link's
    token position is relative to his episode start and every target is a within-episode next-step delta."""
    lk, rows, _ = _split(state)
    rows = rows[:, :rows_keep].copy()
    T, _ = build(state, epid, grid=grid, scales=scales)
    for k in ("actors", "ids", "cats", "states", "mask", "actor_rcell", "actor_off", "cell_xy"):
        T[k] = T[k][:, :rows_keep]

    n = len(state)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    ok = np.r_[epid[1:] == epid[:-1], False]
    lpos = lk[:, [LI["pos_x"], LI["pos_z"]]]

    lc_n, lo_n = grid.to_cell(lpos[nxt])
    dcell = np.clip(lc_n - T["link_cell"], -(DW // 2), DW // 2) + DW // 2
    out = {
        "dcell": (dcell[:, 0] * DW + dcell[:, 1]),
        "doff": (lo_n - T["link_off"]),
        "dy": np.clip((lk[nxt, LI["pos_y"]] - lk[:, LI["pos_y"]]) / d_link_y, -1, 1)[:, None],
    }
    ys, yc = np.sin(lk[:, LI["yaw"]] * 2 * np.pi / 65536.0), np.cos(lk[:, LI["yaw"]] * 2 * np.pi / 65536.0)
    out["dyaw"] = np.stack([np.sin(lk[nxt, LI["yaw"]] * 2 * np.pi / 65536.0) - ys,
                            np.cos(lk[nxt, LI["yaw"]] * 2 * np.pi / 65536.0) - yc], 1)
    out["hp_ev"] = np.where(lk[nxt, LI["hp"]] < lk[:, LI["hp"]], 1,
                            np.where(lk[nxt, LI["hp"]] > lk[:, LI["hp"]], 2, 0))
    sf1 = lk[:, LI["sf1"]].astype(np.int64)
    out["bits_now"] = np.stack([((sf1 >> i) & 1) for i in SF1_KEEP], 1)
    out["bits_nxt"] = np.stack([((sf1[nxt] >> i) & 1) for i in SF1_KEEP], 1)
    out["scene_now"] = (lk[:, LI["scene"]] != 0)
    out["scene_nxt"] = (lk[nxt, LI["scene"]] != 0)

    rpos = rows[:, :, [RI["pos_x"], RI["pos_z"]]]
    rc_n, ro_n = grid.to_cell(rpos[nxt].reshape(-1, 2))
    rc_c, ro_c = grid.to_cell(rpos.reshape(-1, 2))
    rc_n, ro_n = rc_n.reshape(n, -1, 2), ro_n.reshape(n, -1, 2)
    rc_c, ro_c = rc_c.reshape(n, -1, 2), ro_c.reshape(n, -1, 2)
    rdc = np.clip(rc_n - rc_c, -(DW // 2), DW // 2) + DW // 2
    out["r_dcell"] = rdc[..., 0] * DW + rdc[..., 1]
    out["r_doff"] = ro_n - ro_c
    out["r_hp_ev"] = np.where(rows[nxt, :, RI["hp"]] < rows[:, :, RI["hp"]], 1,
                              np.where(rows[nxt, :, RI["hp"]] > rows[:, :, RI["hp"]], 2, 0))
    out["pres_nxt"] = (rows[nxt, :, RI["present"]] > 0.5)
    out["st_nxt"] = rows[nxt, :, RI["state_idx"]].clip(0, 9)

    out.update(T)
    out["act"] = np.clip(act, 0, n_actions - 1)
    out["ok"] = ok
    out["slot"] = slot
    out["lpos"] = lpos
    return {k: v.astype(dtype_of(k), copy=False) for k, v in out.items() if k != "start_idx"}


# ---- host memory -------------------------------------------------------------------------------------------
# `free` reports the HOST inside a container: on a 188 GB RunPod pod it cheerfully says 1511 GB. Reading it and
# concluding memory was a non-issue is how a 100-epoch run was OOM-killed 20 minutes into loading. The cgroup
# is the only honest source.
LOAD_BYTES_PER_FRAME = 3300      # f32 state (1452) + 16-row f32 tokens (2032), measured
CONCAT_FACTOR = 1.9              # np.concatenate copies, so the arrays exist twice at the seam


def host_mem_limit() -> int | None:
    """Bytes this process may actually use, from cgroup v2 then v1. None if unlimited/unknown."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = pathlib.Path(path).read_text().strip()
        except OSError:
            continue
        if v == "max":
            return None
        try:
            n = int(v)
        except ValueError:
            continue
        return None if n > (1 << 62) else n       # v1 uses a huge sentinel for "unlimited"
    return None


def check_load_fits(n_frames: int, *, note: str = "") -> tuple[bool, str]:
    """Would loading n_frames the un-packed way fit? Returns (ok, message)."""
    lim = host_mem_limit()
    need = int(n_frames * LOAD_BYTES_PER_FRAME * CONCAT_FACTOR)
    g = 2 ** 30
    if lim is None:
        return True, f"peak host RAM ~{need / g:.0f} GiB; no cgroup limit visible{note}"
    ok = need < 0.85 * lim
    return ok, (f"peak host RAM ~{need / g:.0f} GiB vs cgroup limit {lim / g:.0f} GiB "
                f"({need / lim:.0%} of it){note}")
