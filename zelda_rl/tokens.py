"""Entity tokens for the world model: typed, egocentric, every continuous field scaled to [-1, 1].

Built to the constraints that make structured-telemetry attention train cleanly, rather than to NLP defaults:

  * NO RAW COORDINATES anywhere. Actors are egocentric to Link; Link is relative to his EPISODE START, which
    is also exactly what the reward's `left_room` checks compare against (radius from start, fall from start y).
    Absolute world position never enters a projection -- unnormalised coordinates leaking into a query/key dot
    product is what drives logits to +-50 and saturates the softmax.
  * PER-FIELD SCALING. Every continuous field is divided by a scale FITTED from the data (99.5th percentile of
    |x| on the training split) and clipped to [-1, 1]. Discrete things -- actor id, category, behaviour state
    -- are left as integer indices for embedding lookups, never as floats in a projection.
  * HETEROGENEOUS PROJECTIONS. Link and actors are different token types with different field lists and
    different encoders; they do not share one generic linear layer.
  * NO ACTION LEAK. The state at step t is read BEFORE action t is applied, so anything reflecting the
    controller belongs to a_{t-1}, not to the world. `bc.CONTROL_ECHO_FEATS` names those and none of them are
    here: no action_fn_rel, no melee state, no ztarget timer, no parallel yaw, no sf1 bits 11/15/18/19. The
    action reaches the model only as a conditioning vector on the query side.

The field lists are explicit and auditable on purpose: `LINK_FIELDS` / `ACTOR_FIELDS` below are the whole
contract, so what the model can and cannot see is readable in one place instead of inferred from a featuriser.
"""
from __future__ import annotations

import numpy as np

from zelda_rl.bc import BABA_ID, LI, RI, _split
from zelda_rl.combat import BG_BITS

TURNS = 65536.0
# sf1 bits worth seeing, with the control-echo bits (11 carrying, 15 z-target, 18 jump, 19 freefall) removed.
SF1_KEEP = (6, 7, 10, 13, 14, 16, 21, 22, 26, 27, 29)

# (name, kind) where kind is "cont" (scaled to [-1,1]), "angle" (emitted as sin/cos), "bit", or "idx".
# NO raw or scaled coordinate appears here. Position enters ONLY as a grid cell (an embedding lookup, so it has
# no magnitude to leak into a dot product) plus a within-cell offset bounded to [-0.5, 0.5]. `rel_y` survives
# because y is not gridded: it spans ~22 units against 420/810 for x/z in one flat room, and the reward's `fell`
# terminal reads it.
LINK_FIELDS = [("rel_y", "cont"),
               ("hp", "cont"), ("hp_frac", "cont"), ("yaw", "angle"), ("speed", "cont"),
               ("vel_x", "cont"), ("vel_z", "cont"), ("vel_y", "cont"),
               ("cam_yaw", "angle"), ("heading_rel_cam", "angle"), ("floor_dy", "cont"),
               ("n_present", "cont"), ("n_baba", "cont")]
ACTOR_FIELDS = [("rel_y", "cont"),
                ("bearing", "angle"), ("hp", "cont"), ("yaw", "angle"), ("shape_yaw", "angle"),
                ("yaw_to_player", "angle"), ("timer", "cont"),
                ("vel_x", "cont"), ("vel_z", "cont"), ("vel_y", "cont")]


def _ang(u):
    a = u.astype(np.float32) * (2 * np.pi / TURNS)
    return np.sin(a), np.cos(a)


def width(fields, n_bits=0):
    return sum(2 if k == "angle" else 1 for _, k in fields) + n_bits


LINK_DIM = width(LINK_FIELDS, len(SF1_KEEP) + len(BG_BITS))
ACTOR_DIM = width(ACTOR_FIELDS)


REL_CLIP = 15          # actor cell offsets relative to Link, clipped to +-15 -> vocabulary 31 per axis


def build(state: np.ndarray, epid: np.ndarray, scales: dict | None = None, train_mask=None, grid=None):
    """state -> dict of link (N, LINK_DIM), actors (N, 16, ACTOR_DIM), ids/cats/states (N, 16) int64,
    mask (N, 16), plus `dist` (N, 16) in scaled units for the attention distance bias, and the scales used."""
    lk, rows, _ = _split(state)
    n, R = len(state), rows.shape[1]
    pres = (rows[:, :, RI["present"]] > 0.5).astype(np.float32)

    # episode-start reference, so Link's own position is never absolute
    starts = np.zeros(n, np.int64)
    b = np.r_[0, np.flatnonzero(np.diff(epid) != 0) + 1, n]
    for lo, hi in zip(b[:-1], b[1:]):
        starts[lo:hi] = lo
    p0 = lk[starts][:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    lpos = lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    rel = lpos - p0

    raw_l = {"rel_x": rel[:, 0], "rel_z": rel[:, 2], "rel_y": rel[:, 1],
             "dist_from_start": np.linalg.norm(rel[:, [0, 2]], axis=1),
             "hp": lk[:, LI["hp"]], "hp_frac": lk[:, LI["hp"]] / np.maximum(lk[:, LI["hp_cap"]], 1),
             "yaw": lk[:, LI["yaw"]], "speed": lk[:, LI["speed"]],
             "vel_x": lk[:, LI["vel_x"]], "vel_z": lk[:, LI["vel_z"]], "vel_y": lk[:, LI["vel_y"]],
             "cam_yaw": lk[:, LI["cam_yaw"]] * (TURNS / (2 * np.pi)),
             "heading_rel_cam": lk[:, LI["yaw"]] - lk[:, LI["cam_yaw"]] * (TURNS / (2 * np.pi)),
             "floor_dy": lk[:, LI["floor_height"]] - lk[:, LI["pos_y"]],
             "n_present": pres.sum(1), "n_baba": (((rows[:, :, RI["id"]].astype(np.int64) == BABA_ID))
                                                  * pres).sum(1)}
    d3 = rows[:, :, [RI["pos_x"], RI["pos_y"], RI["pos_z"]]] - lpos[:, None, :]   # EGOCENTRIC to Link
    raw_a = {"rel_x": d3[:, :, 0], "rel_z": d3[:, :, 2], "rel_y": d3[:, :, 1],
             "dist": np.linalg.norm(d3[:, :, [0, 2]], axis=2),
             "bearing": np.arctan2(d3[:, :, 0], d3[:, :, 2]) * (TURNS / (2 * np.pi)),
             "hp": rows[:, :, RI["hp"]], "yaw": rows[:, :, RI["yaw"]],
             "shape_yaw": rows[:, :, RI["shape_yaw"]], "yaw_to_player": rows[:, :, RI["yaw_to_player"]],
             "timer": rows[:, :, RI["timer"]],
             "vel_x": rows[:, :, RI["vel_x"]], "vel_z": rows[:, :, RI["vel_z"]],
             "vel_y": rows[:, :, RI["vel_y"]]}

    m = np.ones(n, bool) if train_mask is None else train_mask
    fit = scales is None
    sc = {} if fit else dict(scales)

    def scale(key, v, is_row):
        if fit:
            x = np.abs(v[m][pres[m] > 0] if is_row else v[m])
            sc[key] = float(max(np.percentile(x[np.isfinite(x)], 99.5), 1e-3)) if x.size else 1.0
        return np.clip(v / sc[key], -1.0, 1.0).astype(np.float32)

    def pack(fields, raw, prefix, is_row):
        out = []
        for nm, kind in fields:
            v = raw[nm]
            if kind == "angle":
                s, c = _ang(v)
                out += [s, c]
            else:
                out.append(scale(prefix + nm, v, is_row))
        return out

    link = np.stack(pack(LINK_FIELDS, raw_l, "l_", False), -1)
    sf1 = lk[:, LI["sf1"]].astype(np.int64)
    bg = lk[:, LI["bg_flags"]].astype(np.int64)
    link = np.concatenate([link] + [((sf1 >> i) & 1).astype(np.float32)[:, None] for i in SF1_KEEP]
                          + [((bg >> i) & 1).astype(np.float32)[:, None] for i in BG_BITS], 1)
    actors = np.stack(pack(ACTOR_FIELDS, raw_a, "a_", True), -1) * pres[:, :, None]
    # --- grid codec for every position: Link absolute cell, actors RELATIVE to Link's cell (egocentric AND
    # discrete), both with their within-cell offsets. No coordinate is ever projected.
    lc, lo = grid.to_cell(lpos[:, [0, 2]])
    rc, ro = grid.to_cell(rows[:, :, [RI["pos_x"], RI["pos_z"]]].reshape(-1, 2))
    rc, ro = rc.reshape(n, R, 2), ro.reshape(n, R, 2)
    rel_cell = np.clip(rc - lc[:, None, :], -REL_CLIP, REL_CLIP) + REL_CLIP        # -> [0, 2*REL_CLIP]
    extra = {"link_cell": lc.astype(np.int64), "link_off": lo.astype(np.float32),
             "actor_rcell": (rel_cell * pres[:, :, None].astype(np.int64) + REL_CLIP
                             * (1 - pres[:, :, None]).astype(np.int64)).astype(np.int64),
             "actor_off": (ro * pres[:, :, None]).astype(np.float32),
             "cell_xy": (rc - lc[:, None, :]).astype(np.float32)}       # for the attention distance bias
    return {**extra, "link": link.astype(np.float32), "actors": actors.astype(np.float32),
            "ids": (rows[:, :, RI["id"]].astype(np.int64) * pres.astype(np.int64)),
            "cats": (rows[:, :, RI["cat"]].astype(np.int64).clip(0, 15) * pres.astype(np.int64)),
            "states": (rows[:, :, RI["state_idx"]].astype(np.int64).clip(0, 9) * pres.astype(np.int64)),
            "mask": pres, "start_idx": starts}, sc
