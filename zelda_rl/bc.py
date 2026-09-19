"""Behaviour cloning over the structured game state (335-d v1 / 358-d v2, see zelda_rl.state_layout), to produce a frozen encoder for a latent world model.

Featuriser: raw state -> (link block, 16 actor rows, actor ids, present mask). Angles become sin/cos, the Link
state-flags word becomes bits, actor rows get Link-relative geometry and one-hot behaviour state, and lock-on /
Navi-arrow row indices become per-row flags (so the row ordering carries no information).

Encoder: shared per-row MLP -> masked mean+max pool -> concat Link block -> MLP -> z, mirroring the entity
encoder of the PPO policy that generated the data (`zelda_rl.ppo.EntityActorCritic._features`).

`encode()` is a pure function of the current state and is the artefact the latent world model consumes:
z_t = E(s_t), then z_{t+1} = M(z_t, a_t). The demonstrator policies were recurrent, so an optional GRUCell
sits *after* the encoder in the action head only (h_t = GRU(z_t, h_{t-1}) -> logits); the auxiliary heads
(reward / terminal / Link HP / live targets) hang off z_t so they stay decodable from a predicted latent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from zelda_rl.combat import BG_BITS, BG_GROUND, BG_WALL
from zelda_rl.state_layout import (ENEMY_FIELDS as ROW_F, LINK_FIELDS_V1, LINK_FIELDS_V2, N_ROWS, ROW_DIM,
                                  link_dim)

LINK_DIM, N_ACTIONS = len(LINK_FIELDS_V1), 63     # v1 width; real width is read per call, see _split()
N_BABA_STATES = 10
SF1_BITS = 32
BABA_ID = 0x0055
TURNS = 65536.0  # OoT binary angle unit

LINK_F = LINK_FIELDS_V1
LI = {k: i for i, k in enumerate(LINK_FIELDS_V2)}   # v1 names keep their v1 indices: v2 only appends
RI = {k: i for i, k in enumerate(ROW_F)}


def _split(state: np.ndarray):
    """Flat state -> (link block, row block, is_v2). Width decides the version; v2 only APPENDS Link
    fields, so every v1 index in `LI` stays valid for both."""
    state = np.asarray(state, np.float32)
    n, w = len(state), state.shape[-1]
    dl = link_dim(w)
    return state[:, :dl], state[:, dl:].reshape(n, N_ROWS, ROW_DIM), dl == len(LINK_FIELDS_V2)


LINK_FEAT_NAMES = (["hp", "hp_frac", "pos_x", "pos_y", "pos_z", "yaw_sin", "yaw_cos", "held_item", "nuts",
                    "has_lockon", "has_arrow", "n_present", "n_baba"] + [f"sf1_bit{i}" for i in range(SF1_BITS)])
# v2 block: Link's body, collision and action state. ALWAYS emitted, zero-filled for a v1 state, so the
# encoder input width does not depend on the dataset version and v1/v2 episodes can be mixed in one batch.
LINK_V2_FEAT_NAMES = (["speed", "vel_x", "vel_y", "vel_z", "vel_mag"]
                      + [f"bg_{n}" for n in ("ground", "ground_touch", "ground_leave", "wall", "ceiling",
                                             "water", "ground_strict", "crushed", "player_wall")]
                      + ["wall_yaw_sin", "wall_yaw_cos", "floor_dy", "shape_yaw_sin", "shape_yaw_cos",
                         "turn_sin", "turn_cos", "action_fn_rel", "melee_active", "melee_anim",
                         "ztarget_active", "ztarget_timer", "parallel_yaw_sin", "parallel_yaw_cos",
                         "fall_distance", "floor_pitch_sin", "floor_pitch_cos", "has_interact_wall",
                         "dist_interact_wall", "y_dist_to_ledge", "door_type", "door_timer", "knockback"]
                      + [f"sf2_bit{i}" for i in range(SF1_BITS)] + [f"sf3_bit{i}" for i in range(8)])
ROW_FEAT_NAMES = (["present", "hp", "dx", "dy", "dz", "dist", "log_dist", "bearing_sin", "bearing_cos",
                   "pos_x", "pos_y", "pos_z", "yaw_sin", "yaw_cos", "shape_yaw_sin", "shape_yaw_cos",
                   "yaw_to_player_sin", "yaw_to_player_cos", "timer", "cat", "params", "fn_rel",
                   "vel_x", "vel_y", "vel_z", "speed", "focus_dx", "focus_dy", "focus_dz",
                   "is_baba", "is_targeted", "is_arrow"] + [f"state_{i}" for i in range(N_BABA_STATES)])
LINK_FEAT_NAMES = LINK_FEAT_NAMES + LINK_V2_FEAT_NAMES
LINK_FEAT_DIM, ROW_FEAT_DIM = len(LINK_FEAT_NAMES), len(ROW_FEAT_NAMES)
LINK_V2_FEAT_DIM = len(LINK_V2_FEAT_NAMES)


def _sincos(units):
    a = units.astype(np.float32) * (2.0 * np.pi / TURNS)
    return np.sin(a), np.cos(a)


def _link_v2_feats(lk: np.ndarray, v2: bool, n: int) -> np.ndarray:
    """Link body / collision / action features, or zeros for a v1 state.

    Angles are sin/cos, never raw units. `wall_yaw` is MASKED by actual wall contact: with no wall touched
    the field holds a stale value, and feeding that taught the Baba-facing head a 103-degree error before
    the same masking fixed it to 5 degrees.
    """
    if not v2:
        return np.zeros((n, LINK_V2_FEAT_DIM), np.float32)
    col = lambda k: lk[:, LI[k]]                                            # noqa: E731
    bg = col("bg_flags").astype(np.int64)
    vel = lk[:, [LI["vel_x"], LI["vel_y"], LI["vel_z"]]]
    touching_wall = ((bg >> BG_WALL) & 1).astype(np.float32)
    wys, wyc = _sincos(col("wall_yaw"))
    sys_, syc = _sincos(col("shape_yaw"))
    tus, tuc = _sincos(col("shape_yaw") - col("yaw"))                       # mid-turn body-vs-heading offset
    pys, pyc = _sincos(col("parallel_yaw"))
    fps, fpc = _sincos(col("floor_pitch"))
    zt = col("ztarget_timer")
    diw = col("dist_interact_wall")
    sf2 = col("sf2").astype(np.int64)
    sf3 = col("sf3").astype(np.int64)
    return np.concatenate([
        (col("speed") / 10.0)[:, None], vel / 10.0, (np.linalg.norm(vel, axis=1) / 10.0)[:, None],
        np.stack([((bg >> b) & 1).astype(np.float32) for b in BG_BITS], 1),
        (wys * touching_wall)[:, None], (wyc * touching_wall)[:, None],
        ((col("floor_height") - col("pos_y")) / 100.0)[:, None],
        sys_[:, None], syc[:, None], tus[:, None], tuc[:, None],
        (col("action_fn_rel") * 1e-4)[:, None],
        (col("melee_state") != 0).astype(np.float32)[:, None], (col("melee_anim") / 10.0)[:, None],
        (zt != 0).astype(np.float32)[:, None], (np.clip(zt, 0, 10) / 10.0)[:, None],
        pys[:, None], pyc[:, None],
        (np.clip(col("fall_distance"), -100, 2000) / 100.0)[:, None], fps[:, None], fpc[:, None],
        (diw > 0).astype(np.float32)[:, None], (np.clip(diw, 0, 200) / 100.0)[:, None],
        (np.clip(col("y_dist_to_ledge"), -200, 200) / 100.0)[:, None],
        col("door_type")[:, None], (np.clip(col("door_timer"), 0, 200) / 100.0)[:, None],
        (col("knockback_type") != 0).astype(np.float32)[:, None],
        np.stack([((sf2 >> b) & 1).astype(np.float32) for b in range(SF1_BITS)], 1),
        np.stack([((sf3 >> b) & 1).astype(np.float32) for b in range(8)], 1),
    ], 1).astype(np.float32)


def featurise(state: np.ndarray) -> dict[str, np.ndarray]:
    """state: (N, 335) v1 or (N, 358) v2 float32 -> dict of link (N, Dl), rows (N, 16, Dr), ids (N, 16) int64, mask (N, 16)."""
    state = np.asarray(state, dtype=np.float32)
    n = len(state)
    lk, rows, v2 = _split(state)
    present = (rows[:, :, RI["present"]] > 0.5).astype(np.float32)
    ids = rows[:, :, RI["id"]].astype(np.int64)
    is_baba = ((ids == BABA_ID) & (present > 0)).astype(np.float32)

    lpos = lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    ys, yc = _sincos(lk[:, LI["yaw"]])
    sf1 = lk[:, LI["sf1"]].astype(np.int64)
    bits = np.stack([((sf1 >> i) & 1).astype(np.float32) for i in range(SF1_BITS)], 1)
    link = np.concatenate([
        lk[:, [LI["hp"]]],
        (lk[:, [LI["hp"]]] / np.maximum(lk[:, [LI["hp_cap"]]], 1.0)),
        lpos, ys[:, None], yc[:, None],
        lk[:, [LI["held_item"], LI["nuts"]]],
        (lk[:, [LI["targeted_row"]]] >= 0).astype(np.float32),
        (lk[:, [LI["arrow_row"]]] >= 0).astype(np.float32),
        present.sum(1, keepdims=True), is_baba.sum(1, keepdims=True),
        bits,
        _link_v2_feats(lk, v2, n),
    ], 1).astype(np.float32)

    rpos = rows[:, :, [RI["pos_x"], RI["pos_y"], RI["pos_z"]]]
    d = rpos - lpos[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    bs, bc = _sincos(np.arctan2(d[:, :, 0], d[:, :, 2]) * (TURNS / (2 * np.pi)))
    yws, ywc = _sincos(rows[:, :, RI["yaw"]])
    sys_, syc = _sincos(rows[:, :, RI["shape_yaw"]])
    tps, tpc = _sincos(rows[:, :, RI["yaw_to_player"]])
    vel = rows[:, :, [RI["vel_x"], RI["vel_y"], RI["vel_z"]]]
    focus = rows[:, :, [RI["focus_x"], RI["focus_y"], RI["focus_z"]]] - rpos
    idx = np.arange(N_ROWS)[None, :]
    targeted = (lk[:, [LI["targeted_row"]]] == idx).astype(np.float32)
    arrow = (lk[:, [LI["arrow_row"]]] == idx).astype(np.float32)
    sidx = np.clip(rows[:, :, RI["state_idx"]].astype(np.int64), 0, N_BABA_STATES - 1)
    onehot = np.eye(N_BABA_STATES, dtype=np.float32)[sidx]
    row_feat = np.concatenate([
        present[:, :, None], rows[:, :, [RI["hp"]]], d, dist[:, :, None], np.log1p(dist)[:, :, None],
        bs[:, :, None], bc[:, :, None], rpos,
        yws[:, :, None], ywc[:, :, None], sys_[:, :, None], syc[:, :, None], tps[:, :, None], tpc[:, :, None],
        rows[:, :, [RI["timer"], RI["cat"], RI["params"]]], rows[:, :, [RI["fn_rel"]]] * 1e-6,
        vel, np.linalg.norm(vel, axis=2)[:, :, None], focus,
        is_baba[:, :, None], targeted[:, :, None], arrow[:, :, None], onehot,
    ], 2).astype(np.float32)
    row_feat *= present[:, :, None]  # absent rows carry no stale geometry
    return {"link": link, "rows": row_feat, "ids": ids * present.astype(np.int64), "mask": present}


@dataclass
class Norm:
    """Per-field standardisation. Row stats are computed over present rows only."""
    link_mean: np.ndarray
    link_std: np.ndarray
    row_mean: np.ndarray
    row_std: np.ndarray
    dead_link: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dead_row: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @staticmethod
    def fit(f: dict[str, np.ndarray], eps: float = 1e-6) -> "Norm":
        lm, ls = f["link"].mean(0), f["link"].std(0)
        m = f["mask"].reshape(-1) > 0
        r = f["rows"].reshape(-1, f["rows"].shape[-1])[m]
        rm, rs = r.mean(0), r.std(0)
        return Norm(lm, np.where(ls < eps, 1.0, ls).astype(np.float32), rm,
                    np.where(rs < eps, 1.0, rs).astype(np.float32), (ls < eps), (rs < eps))

    def apply(self, f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        link = (f["link"] - self.link_mean) / self.link_std
        rows = ((f["rows"] - self.row_mean) / self.row_std) * f["mask"][:, :, None]
        return {"link": link.astype(np.float32), "rows": rows.astype(np.float32), "ids": f["ids"], "mask": f["mask"]}

    def state_dict(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}

    @staticmethod
    def load(d):
        return Norm(**{k: np.asarray(v) for k, v in d.items()})


def _layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class BCNet(nn.Module):
    """Structured-state encoder + BC action head (optionally recurrent) + auxiliary heads.

    `encode()` is the artefact that matters and never sees the hidden state. `gru=0` gives the feedforward
    ablation; `gru>0` puts a GRUCell between z and the action logits, as the demonstrator had.
    """

    AUX = ("reward", "terminal", "hp", "targets")

    def __init__(self, link_dim: int = LINK_FEAT_DIM, row_dim: int = ROW_FEAT_DIM, n_actions: int = N_ACTIONS,
                 latent: int = 128, hidden: int = 256, row_hidden: int = 96, id_vocab: int = 256, id_emb: int = 8,
                 layernorm: bool = True, dropout: float = 0.0, gru: int = 0):
        super().__init__()
        self.latent = latent
        self.id_vocab = id_vocab
        self.hidden_size = gru
        self.id_embedding = nn.Embedding(id_vocab, id_emb)
        self.row_enc = nn.Sequential(_layer(row_dim + id_emb, row_hidden), nn.ReLU(),
                                     _layer(row_hidden, row_hidden), nn.ReLU())
        self.trunk = nn.Sequential(_layer(link_dim + 2 * row_hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                   _layer(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), _layer(hidden, latent))
        self.ln = nn.LayerNorm(latent) if layernorm else nn.Identity()
        self.gru = nn.GRUCell(latent, gru) if gru else None
        self.action = _layer(gru or latent, n_actions, 0.01)
        self.aux = nn.ModuleDict({k: _layer(latent, 1, 0.1) for k in self.AUX})

    def encode(self, link, rows, ids, mask):
        """z_t = E(s_t). Pure function of the current state; no recurrence, no hidden state."""
        e = self.id_embedding(ids.clamp(0, self.id_vocab - 1))
        m = mask.unsqueeze(-1)
        h = self.row_enc(torch.cat([rows, e], -1)) * m
        n = m.sum(-2).clamp(min=1.0)
        mean = h.sum(-2) / n
        mx = (h - (1 - m) * 1e4).max(-2).values * (m.sum(-2) > 0)
        return self.ln(self.trunk(torch.cat([link, mean, mx], -1)))

    def heads(self, z):
        return {k: self.aux[k](z).squeeze(-1) for k in self.AUX}

    def init_hidden(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden_size)

    def step(self, z, h=None, first=None):
        """Action logits from a latent. With a GRU head, h is carried and zeroed where `first` = 1."""
        if self.gru is None:
            return self.action(z), None
        if h is None:
            h = self.init_hidden(z.shape[0])
        if first is not None:
            h = h * (1.0 - first.unsqueeze(-1))
        h = self.gru(z, h)
        return self.action(h), h

    def forward(self, link, rows, ids, mask, h=None, first=None):
        """-> (z, action logits, aux dict, next hidden state or None)."""
        z = self.encode(link, rows, ids, mask)
        logits, h = self.step(z, h, first)
        return z, logits, self.heads(z), h


# ---------------------------------------------------------------- state decoder
# The reward is not regressed: it is COMPUTED by CombatTracker.step() from raw state. So the decoder's job is
# to make exactly the fields that function reads readable from z. Everything below is one of its inputs:
#   Link      hp, pos (approach / room radius / fall), yaw (facing for shield+block), sf1 bits 7/22/26
#             (dead / shielding / damaged), targeted row (lock-on)
#   per row   present (despawn != kill), is_baba (is it a target), hp (damage / kills), pos (distances),
#             state_idx (stunned|hit -> stun bonus; lunge|pull_back -> block/dodge cycles)
SF1_DEAD, SF1_SHIELD, SF1_DAMAGED = 7, 22, 26
DECODE_CONT = ("link_pos", "link_yaw", "row_pos")   # HP is classified, not regressed


OBS_CAM_YAW = slice(12, 14)     # GLOBAL_FIELDS_V2: camera_yaw_sin_cos
OBS_LINK_VEL = slice(14, 17)    # GLOBAL_FIELDS_V2: link_vel


def decode_targets(state: np.ndarray, obs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Raw state (v1 335-d or v2 358-d) -> the fields CombatTracker.step() reads. Continuous fields stay in raw units.

    `obs` is the 483-d observation the demonstrator saw, stored alongside every transition. Camera yaw and
    Link's velocity live only there -- the 335-d state was built from CombatState, the combat-task reader,
    which has no camera and no Link velocity. Both were in the original PPO's observation, so decoding
    them closes that parity gap without re-collecting anything.
    """
    state = np.asarray(state, np.float32)
    n = len(state)
    lk, rows, v2 = _split(state)
    sf1 = lk[:, LI["sf1"]].astype(np.int64)
    ys, yc = _sincos(lk[:, LI["yaw"]])
    tr = lk[:, LI["targeted_row"]].astype(np.int64)
    present = (rows[:, :, RI["present"]] > 0.5).astype(np.float32)
    out = {
        "link_hp": lk[:, [LI["hp"]]],
        "link_pos": lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]],
        "link_yaw": np.stack([ys, yc], 1),
        "link_bits": np.stack([((sf1 >> b) & 1).astype(np.float32)
                               for b in (SF1_DEAD, SF1_SHIELD, SF1_DAMAGED)], 1),
        # jumping, freefall, z_targeting, ledge_*, in_water, ... -- the jump-attack phase lives here
        "link_sf1": np.stack([((sf1 >> b) & 1).astype(np.float32) for b in SF1_NAMED], 1),
        "targeted": np.where((tr >= 0) & (tr < N_ROWS), tr, N_ROWS),          # N_ROWS = "no lock-on"
        "row_present": present,
        "row_is_baba": ((rows[:, :, RI["id"]].astype(np.int64) == BABA_ID) & (present > 0)).astype(np.float32),
        "row_hp": rows[:, :, RI["hp"]],
        "row_pos": rows[:, :, [RI["pos_x"], RI["pos_y"], RI["pos_z"]]],
        "row_state": np.clip(rows[:, :, RI["state_idx"]].astype(np.int64), 0, N_BABA_STATES - 1),
        # Orientation and timing: whether a Baba is turned toward Link, and how long until its action
        # fires. The real policy reads both (t1_facing_link_cos_sin, t1_timer) and its kill move is a
        # timed Z+A against that telegraph. The encoder is given them; nothing checked z kept them.
        "row_facing_link": np.stack(_sincos((rows[:, :, RI["shape_yaw"]]
                                             - rows[:, :, RI["yaw_to_player"]])), -1),
        "row_timer": (np.clip(rows[:, :, RI["timer"]], 0, 100) / 100.0)[..., None],
        "row_vel": rows[:, :, [RI["vel_x"], RI["vel_y"], RI["vel_z"]]],
        "link_facing_target": np.stack(_sincos(
            np.arctan2(rows[:, :, RI["pos_x"]] - lk[:, [LI["pos_x"]]],
                       rows[:, :, RI["pos_z"]] - lk[:, [LI["pos_z"]]]) * (TURNS / (2 * np.pi))
            - lk[:, [LI["yaw"]]]), -1),
        # the room-exit terminal reads scene and room; without them nothing downstream can tell Link left.
        "scene": np.clip(lk[:, LI["scene"]].astype(np.int64), 0, 1),
        "room": np.clip(lk[:, LI["room"]].astype(np.int64), 0, MAX_ROOM - 1),
        # live Baba count as ONE low-cardinality head, rather than counting 16 noisy per-row predictions
        "baba_count": np.clip(((present > 0.5) & (rows[:, :, RI["id"]].astype(np.int64) == BABA_ID)
                               & (rows[:, :, RI["hp"]] > 0)).sum(1), 0, N_BABA_COUNT - 1).astype(np.int64),
    }
    if obs is not None:
        out["cam_yaw"] = np.asarray(obs[:, OBS_CAM_YAW], np.float32)
        out["link_vel"] = np.asarray(obs[:, OBS_LINK_VEL], np.float32)
    out.update(_v2_targets(lk, v2, n))
    return out


def _v2_targets(lk: np.ndarray, v2: bool, n: int) -> dict[str, np.ndarray]:
    """v2 decoder targets: what Link is doing and what he is touching.

    Emitted for v1 states too, zero-filled, alongside `v2_mask`. The trainer multiplies these losses by the
    mask the same way it already does for the obs-only camera and velocity heads, so a v1 episode
    contributes nothing to them instead of teaching zeros.
    """
    if not v2:
        z = lambda *s: np.zeros((n, *s), np.float32)                        # noqa: E731
        return {"v2_mask": z(), "link_bg": z(len(BG_BITS)),
                "link_melee": np.zeros(n, np.int64) + MELEE_STATE_VALUES.index(0),
                "link_action": np.full(n, N_LINK_ACTIONS - 1, np.int64),
                "link_speed": z(1), "link_wall_yaw": z(2), "link_floor_dy": z(1), "link_shape_yaw": z(2),
                "link_parallel_yaw": z(2), "link_fall": z(1), "link_ztarget": z(1), "link_airborne": z(1)}
    col = lambda k: lk[:, LI[k]]                                            # noqa: E731
    bg = col("bg_flags").astype(np.int64)
    wall = ((bg >> BG_WALL) & 1).astype(np.float32)
    wys, wyc = _sincos(col("wall_yaw"))
    sys_, syc = _sincos(col("shape_yaw"))
    pys, pyc = _sincos(col("parallel_yaw"))
    ms = np.abs(col("melee_state").astype(np.int64)[:, None]
                - np.asarray(MELEE_STATE_VALUES)).argmin(-1)
    return {
        "v2_mask": np.ones(n, np.float32),
        "link_bg": np.stack([((bg >> b) & 1).astype(np.float32) for b in BG_BITS], 1),
        "link_melee": ms.astype(np.int64),
        "link_action": np.array([_ACTION_INDEX.get(int(v), N_LINK_ACTIONS - 1)
                                 for v in col("action_fn_rel")], np.int64),
        "link_speed": (col("speed") / 10.0)[:, None],
        # masked by contact: with no wall touched the field is stale, which is the mistake that put the
        # Baba-facing head at 103 degrees until the same masking took it to 5.
        "link_wall_yaw": np.stack([wys * wall, wyc * wall], 1),
        "link_floor_dy": ((col("floor_height") - col("pos_y")) / 100.0)[:, None],
        "link_shape_yaw": np.stack([sys_, syc], 1),
        "link_parallel_yaw": np.stack([pys, pyc], 1),
        "link_fall": (np.clip(col("fall_distance"), -100, 2000) / 100.0)[:, None],
        "link_ztarget": (col("ztarget_timer") != 0).astype(np.float32)[:, None],
        "link_airborne": (1.0 - ((bg >> BG_GROUND) & 1)).astype(np.float32)[:, None],
    }


LINK_HP_VALUES = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0)   # half-heart steps; 48 = 3 full hearts
ROW_HP_VALUES = (0.0, 1.0, 2.0)                             # Deku Baba has 2 HP
N_BABA_COUNT = 4                                            # 0..3 live Babas
# The 14 stateFlags1 bits the original PPO observed (obs.SF1_BITS). They are already in the recorded
# state; only 3 were ever decoded, so jumping / freefall / z_targeting were never checked to survive z.
SF1_NAMED = [6, 7, 10, 11, 13, 14, 15, 16, 18, 19, 21, 26, 27, 29]
MAX_ROOM = 8                                                # room ids seen in this scene
# Link's action state machine (Player actionFunc, stored relative to his update fn so it survives a
# different load address). These 20 values cover 99.8% of 187,111 recorded frames. Labels are behavioural,
# read off what co-occurs with each value rather than guessed from a name -- see docs/COMBAT_TASK.md.
#                  value      share  evidence in the collected data
LINK_ACTION_VALUES = (
    3208332,   # 19.7%  melee live 68%, grounded            -> sword attack
    3151228,   # 19.0%  wall 37%, speed 2.4, Z-target 6%    -> free walk (the one that meets walls)
    3143728,   # 15.5%  Z 100%, speed 0.2                   -> locked-on idle
    3149108,   #  8.7%  Z 100%, speed 0.6                   -> locked-on step
    3160928,   #  7.4%  speed 3.8                           -> run
    3159404,   #  5.4%  AIRBORNE 100%, speed 3.9            -> jump
    3144620,   #  5.3%  Z 99%, speed 0.2                    -> locked-on idle (2nd routine)
    3145640,   #  5.0%  speed 1.1, Z 9%                     -> idle
    3161932,   #  4.5%  AIRBORNE 100%, melee 9%, Z 90%      -> jump attack (the timed Z+A kill move)
    3155380,   #  2.5%  speed 1.2, Z 6%                     -> slow walk / turn
    3201864,   #  2.2%  speed 0, Z 0                        -> standing
    3189068,   #  1.3%  speed 0                             -> standing (2nd routine)
    3157388,   #  1.0%  DAMAGED 100%, Z 99%                 -> hurt
    3162828,   #  0.9%  Z 97%, speed 0                      -> locked-on standing
    3151488,   #  0.6%  Z 100%, speed 1.3                   -> locked-on walk
    3157620,   #  0.3%  DAMAGED 100%, speed 0               -> hurt, recovering
    3166492,   #  0.2%  WALL 44%, speed 4.8                 -> pushing into a wall
    3156912,   #  0.1%  AIRBORNE + DAMAGED 100%             -> knocked airborne
    3148004,   #  0.1%  Z 100%                              -> locked-on, rare routine
    3191624,   #  0.1%  airborne 90%, speed 0               -> falling
)
LINK_ACTION_NAMES = ("attack", "walk", "z_idle", "z_step", "run", "jump", "z_idle2", "idle", "jump_attack",
                     "slow_walk", "stand", "stand2", "hurt", "z_stand", "z_walk", "hurt_recover",
                     "wall_push", "knocked_air", "z_rare", "falling", "other")
N_LINK_ACTIONS = len(LINK_ACTION_VALUES) + 1                # + "other"
assert len(LINK_ACTION_NAMES) == N_LINK_ACTIONS
_ACTION_INDEX = {v: i for i, v in enumerate(LINK_ACTION_VALUES)}

MELEE_STATE_VALUES = (-1, 0, 1)                             # meleeWeaponState seen live; 0 = no swing
N_MELEE_STATES = len(MELEE_STATE_VALUES)
# The v2 body head is one linear layer split into named pieces, so adding a field is one entry here.
V2_BODY = ("link_speed", "link_wall_yaw", "link_floor_dy", "link_shape_yaw", "link_parallel_yaw",
           "link_fall", "link_ztarget", "link_airborne")
V2_BODY_SPLIT = (1, 2, 1, 2, 2, 1, 1, 1)


def hp_class(v: np.ndarray, values) -> np.ndarray:
    """Nearest-value class index. HP is discrete, so it is decoded by classification, never regression:
    a regressed HP rounded to an integer turns numeric jitter into fabricated damage events."""
    return np.abs(np.asarray(v, np.float32)[..., None] - np.asarray(values, np.float32)).argmin(-1)


class StateDecoder(nn.Module):
    """z -> the raw state variables the reward function reads.

    HP (Link and per row) is CLASSIFIED over its discrete value set; everything continuous stays regression.
    Per-row fields are decoded through a shared head off a learned per-row query, which is the part of this
    that can fail: `encode()` pools rows with masked mean+max, so row identity must have survived the pool.
    """

    def __init__(self, latent: int = 128, hidden: int = 256, row_hidden: int = 96, n_rows: int = N_ROWS,
                 n_states: int = N_BABA_STATES):
        super().__init__()
        self.n_rows, self.n_states, self.row_hidden = n_rows, n_states, row_hidden
        self.n_link_hp, self.n_row_hp = len(LINK_HP_VALUES), len(ROW_HP_VALUES)
        self.link = nn.Sequential(_layer(latent, hidden), nn.ReLU(), _layer(hidden, hidden), nn.ReLU())
        self.link_out = _layer(hidden, self.n_link_hp + 3 + 2 + 3, 0.1)   # hp logits, pos, yaw sin/cos, sf1 bits
        self.targeted = _layer(latent, n_rows + 1, 0.1)
        self.scene = _layer(latent, 2, 0.1)
        self.room = _layer(latent, MAX_ROOM, 0.1)
        self.baba_count = _layer(latent, N_BABA_COUNT, 0.1)
        self.link_sf1 = _layer(latent, len(SF1_NAMED), 0.1)
        self.cam_yaw = _layer(latent, 2, 0.1)          # camera yaw sin/cos: Z-target recentres it
        self.link_vel = _layer(latent, 3, 0.1)         # Link's momentum, which a jump attack depends on
        # --- v2: what Link is DOING and what he is TOUCHING. Without these the dream cannot tell a jump
        # from a roll from a whiffed swing, which is why an action left almost no trace in the latent.
        self.link_bg = _layer(latent, len(BG_BITS), 0.1)        # ground / wall / ceiling / water / crushed
        self.link_melee = _layer(latent, N_MELEE_STATES, 0.1)   # a sword swing is live
        self.link_action = _layer(latent, N_LINK_ACTIONS, 0.1)  # Link's action state machine
        self.link_body = _layer(latent, 11, 0.1)   # speed, wall_yaw(2), floor_dy, shape_yaw(2),
        #                                            parallel_yaw(2), fall_distance, ztarget, airborne
        self.row_expand = _layer(latent, n_rows * row_hidden)
        self.row_mlp = nn.Sequential(nn.ReLU(), _layer(row_hidden, row_hidden), nn.ReLU())
        # present, is_baba, hp, pos(3), state, facing-link sin/cos, timer, vel(3), link-facing sin/cos
        self.row_out = _layer(row_hidden, 1 + 1 + self.n_row_hp + 3 + n_states + 2 + 1 + 3 + 2, 0.1)

    def forward(self, z):
        lo = self.link_out(self.link(z))
        k = self.n_link_hp
        r = self.row_out(self.row_mlp(self.row_expand(z).reshape(-1, self.n_rows, self.row_hidden)))
        j = 2 + self.n_row_hp
        return {"link_hp": lo[:, :k], "link_pos": lo[:, k:k + 3], "link_yaw": lo[:, k + 3:k + 5],
                "link_bits": lo[:, k + 5:k + 8], "targeted": self.targeted(z),
                "scene": self.scene(z), "room": self.room(z), "baba_count": self.baba_count(z),
                "link_sf1": self.link_sf1(z), "cam_yaw": self.cam_yaw(z), "link_vel": self.link_vel(z),
                "link_bg": self.link_bg(z), "link_melee": self.link_melee(z),
                "link_action": self.link_action(z),
                **dict(zip(V2_BODY, self.link_body(z).split(V2_BODY_SPLIT, dim=-1))),
                "row_present": r[..., 0], "row_is_baba": r[..., 1], "row_hp": r[..., 2:j],
                "row_pos": r[..., j:j + 3],
                "row_state": r[..., j + 3:j + 3 + self.n_states],
                "row_facing_link": r[..., j + 3 + self.n_states:j + 5 + self.n_states],
                "row_timer": r[..., j + 5 + self.n_states:j + 6 + self.n_states],
                "row_vel": r[..., j + 6 + self.n_states:j + 9 + self.n_states],
                "link_facing_target": r[..., j + 9 + self.n_states:j + 11 + self.n_states]}
