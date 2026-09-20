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

from zelda_rl.combat import BG_BITS, BG_DECODE_BITS, BG_GROUND, BG_WALL
from zelda_rl.state_layout import (ENEMY_FIELDS as ROW_F, LINK_FIELDS_V1, LINK_FIELDS_V2,
                                  LINK_FIELDS_V3, N_ROWS, ROW_DIM, link_dim)

LINK_DIM, N_ACTIONS = len(LINK_FIELDS_V1), 63     # v1 width; real width is read per call, see _split()
N_BABA_STATES = 10
SF1_BITS = 32
BABA_ID = 0x0055
TURNS = 65536.0  # OoT binary angle unit

LINK_F = LINK_FIELDS_V1
LI = {k: i for i, k in enumerate(LINK_FIELDS_V3)}   # each version only APPENDS, so indices are stable
RI = {k: i for i, k in enumerate(ROW_F)}


def _split(state: np.ndarray):
    """Flat state -> (link block, row block, is_v2). Width decides the version; v2 only APPENDS Link
    fields, so every v1 index in `LI` stays valid for both."""
    state = np.asarray(state, np.float32)
    n, w = len(state), state.shape[-1]
    dl = link_dim(w)
    return state[:, :dl], state[:, dl:].reshape(n, N_ROWS, ROW_DIM), dl >= len(LINK_FIELDS_V2)


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
# v3: the CAMERA. A stick direction moves Link relative to it, so the world direction of a walk is
# unknowable without it. Zero-filled for v1/v2 states so the encoder width stays version-independent.
LINK_V3_FEAT_NAMES = ["cam_yaw_sin", "cam_yaw_cos", "cam_pitch_sin", "cam_pitch_cos",
                      "cam_rel_x", "cam_rel_y", "cam_rel_z", "cam_dist",
                      "heading_rel_cam_sin", "heading_rel_cam_cos"]
ROW_FEAT_NAMES = (["present", "hp", "dx", "dy", "dz", "dist", "log_dist", "bearing_sin", "bearing_cos",
                   "pos_x", "pos_y", "pos_z", "yaw_sin", "yaw_cos", "shape_yaw_sin", "shape_yaw_cos",
                   "yaw_to_player_sin", "yaw_to_player_cos", "timer", "cat", "params", "fn_rel",
                   "vel_x", "vel_y", "vel_z", "speed", "focus_dx", "focus_dy", "focus_dz",
                   "is_baba", "is_targeted", "is_arrow"] + [f"state_{i}" for i in range(N_BABA_STATES)])
LINK_FEAT_NAMES = LINK_FEAT_NAMES + LINK_V2_FEAT_NAMES + LINK_V3_FEAT_NAMES
LINK_FEAT_DIM, ROW_FEAT_DIM = len(LINK_FEAT_NAMES), len(ROW_FEAT_NAMES)
LINK_V2_FEAT_DIM, LINK_V3_FEAT_DIM = len(LINK_V2_FEAT_NAMES), len(LINK_V3_FEAT_NAMES)


def _link_v3_feats(lk: np.ndarray, n: int) -> np.ndarray:
    """Camera features, or zeros when the state predates v3.

    `heading_rel_cam` is the one that matters most: the stick is interpreted in camera space, so Link's
    heading MINUS the camera yaw is what a held direction actually controls.
    """
    if lk.shape[1] < len(LINK_FIELDS_V3):
        return np.zeros((n, LINK_V3_FEAT_DIM), np.float32)
    cy, cp = lk[:, LI["cam_yaw"]], lk[:, LI["cam_pitch"]]
    eye = lk[:, [LI["cam_eye_x"], LI["cam_eye_y"], LI["cam_eye_z"]]]
    rel = (lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]] - eye) / 100.0
    yaw_rad = lk[:, LI["yaw"]] * (2 * np.pi / TURNS)
    hrel = yaw_rad - cy
    return np.stack([np.sin(cy), np.cos(cy), np.sin(cp), np.cos(cp),
                     rel[:, 0], rel[:, 1], rel[:, 2], np.linalg.norm(rel, axis=1),
                     np.sin(hrel), np.cos(hrel)], 1).astype(np.float32)


def wants_v2(link_dim_of_checkpoint: int) -> bool:
    """Whether a checkpoint's encoder was trained with the v2 Link block, from the width it stored."""
    if link_dim_of_checkpoint == LINK_FEAT_DIM:
        return True
    if link_dim_of_checkpoint == LINK_FEAT_DIM_V1:
        return False
    raise ValueError(f"checkpoint link_dim {link_dim_of_checkpoint} is neither "
                     f"{LINK_FEAT_DIM_V1} (pre-v2) nor {LINK_FEAT_DIM} (v2)")


# Encoder-input features that are near-copies of the CONTROLLER rather than of the world. The state at step t
# is read BEFORE action t is applied, so these reflect action t-1 -- and because the demonstrator holds buttons
# for many steps in a row, knowing t-1 predicts t. So behaviour cloning fed these can score well by reading
# "what I was just doing" instead of learning anything about the world, and the same echo inflated the
# inverse-dynamics floor that was used to argue the action leaves no trace in a latent transition.
CONTROL_ECHO_FEATS = ("action_fn_rel", "melee_active", "melee_anim", "ztarget_active", "ztarget_timer",
                      "parallel_yaw_sin", "parallel_yaw_cos",
                      "sf1_bit11", "sf1_bit15", "sf1_bit18", "sf1_bit19")   # carrying / z_target / jump / freefall


def control_echo_idx():
    """Column indices of the control-echo features inside the link feature block."""
    return [i for i, n in enumerate(LINK_FEAT_NAMES) if n in CONTROL_ECHO_FEATS]


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
    zact = (zt != 0).astype(np.float32)          # parallelYaw is the aim yaw only while Z is held
    diw = col("dist_interact_wall")
    sf2 = col("sf2").astype(np.int64)
    sf3 = col("sf3").astype(np.int64)
    return np.concatenate([
        (col("speed") / 10.0)[:, None], vel / 10.0, (np.linalg.norm(vel, axis=1) / 10.0)[:, None],
        np.stack([((bg >> b) & 1).astype(np.float32) for b in BG_BITS], 1),   # input: all 9 bits
        (wys * touching_wall)[:, None], (wyc * touching_wall)[:, None],
        ((col("floor_height") - col("pos_y")) / 100.0)[:, None],
        sys_[:, None], syc[:, None], tus[:, None], tuc[:, None],
        (col("action_fn_rel") * 1e-4)[:, None],
        (col("melee_state") != 0).astype(np.float32)[:, None], (col("melee_anim") / 10.0)[:, None],
        (zt != 0).astype(np.float32)[:, None], (np.clip(zt, 0, 10) / 10.0)[:, None],
        (pys * zact)[:, None], (pyc * zact)[:, None],
        (np.clip(col("fall_distance"), -100, 2000) / 100.0)[:, None], fps[:, None], fpc[:, None],
        (diw > 0).astype(np.float32)[:, None], (np.clip(diw, 0, 200) / 100.0)[:, None],
        (np.clip(col("y_dist_to_ledge"), -200, 200) / 100.0)[:, None],
        col("door_type")[:, None], (np.clip(col("door_timer"), 0, 200) / 100.0)[:, None],
        (col("knockback_type") != 0).astype(np.float32)[:, None],
        np.stack([((sf2 >> b) & 1).astype(np.float32) for b in range(SF1_BITS)], 1),
        np.stack([((sf3 >> b) & 1).astype(np.float32) for b in range(8)], 1),
    ], 1).astype(np.float32)


LINK_FEAT_DIM_V1 = LINK_FEAT_DIM - LINK_V2_FEAT_DIM - LINK_V3_FEAT_DIM   # 45: pre-v2 checkpoint width


def featurise(state: np.ndarray, link_v2: bool = True,
              control_echo: bool = True) -> dict[str, np.ndarray]:
    """state: (N, 335) v1 or (N, 358) v2 float32 -> dict of link (N, Dl), rows (N, 16, Dr), ids (N, 16) int64, mask (N, 16).

    `link_v2=False` omits the Link body/collision/action block, giving the 45-wide link vector that every
    checkpoint trained before state v2 expects. Pass it when loading such a checkpoint -- its encoder's first
    layer is 45 wide and cannot consume the 122-d vector. `link_feat_dim(ckpt["link_dim"])` picks for you.
    """
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
    ] + ([_link_v2_feats(lk, v2, n), _link_v3_feats(lk, n)] if link_v2 else []), 1).astype(np.float32)
    if not control_echo:
        link[:, control_echo_idx()] = 0.0      # zeroed, not removed, so the encoder width is unchanged

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


def enc_arch(ck: dict) -> dict:
    """Encoder architecture flags out of a checkpoint, defaulting to the deep + LayerNorm original.

    Every BCNet rebuild site should splat this, so a checkpoint carries its own architecture instead of each
    caller assuming one. Checkpoints written before §29 have neither key and rebuild exactly as before.
    """
    return {"out_act": ck.get("enc_out", "ln"), "shallow": bool(ck.get("enc_shallow", False)),
            "arch": ck.get("enc_kind", "mlp")}


class BCNet(nn.Module):
    """Structured-state encoder + BC action head (optionally recurrent) + auxiliary heads.

    `encode()` is the artefact that matters and never sees the hidden state. `gru=0` gives the feedforward
    ablation; `gru>0` puts a GRUCell between z and the action logits, as the demonstrator had.
    """

    AUX = ("reward", "terminal", "hp", "targets")

    def __init__(self, link_dim: int = LINK_FEAT_DIM, row_dim: int = ROW_FEAT_DIM, n_actions: int = N_ACTIONS,
                 latent: int = 128, hidden: int = 256, row_hidden: int = 96, id_vocab: int = 256, id_emb: int = 8,
                 layernorm: bool = True, dropout: float = 0.0, gru: int = 0,
                 out_act: str | None = None, shallow: bool = False, arch: str = "mlp",
                 tf_dim: int = 128, tf_layers: int = 2, tf_heads: int = 4):
        """`out_act` and `shallow` default to the pre-existing deep + LayerNorm encoder, so every checkpoint
        written before they existed rebuilds identically. See `enc_arch()` and §29 for why they exist: measured
        untrained on identical features and at matched latent movement (move/rand ~0.098 for all six), the
        latent LayerNorm costs 0.73 of predictive skill and the deep trunk another 0.36, and both also hurt
        linear decodability (pos MAE 7.7 shallow+tanh vs 150.8 deep+LN)."""
        super().__init__()
        self.latent = latent
        self.id_vocab = id_vocab
        self.hidden_size = gru
        self.id_embedding = nn.Embedding(id_vocab, id_emb)
        self.row_enc = nn.Sequential(_layer(row_dim + id_emb, row_hidden), nn.ReLU(),
                                     _layer(row_hidden, row_hidden), nn.ReLU())
        self.arch = arch
        if arch == "transformer":
            # The state is ALREADY tokens: 16 actor rows + 1 Link token. The MLP path destroys that structure by
            # mean/max pooling the rows into two fixed vectors before the trunk ever sees them, so which actor
            # is where survives only as a pooled average. Self-attention over 17 tokens keeps them addressable.
            self.row_tok = nn.Linear(row_dim + id_emb, tf_dim)
            self.link_tok = nn.Linear(link_dim, tf_dim)
            self.type_emb = nn.Parameter(torch.zeros(2, tf_dim))
            layer = nn.TransformerEncoderLayer(tf_dim, tf_heads, dim_feedforward=4 * tf_dim,
                                               dropout=dropout, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.tf = nn.TransformerEncoder(layer, tf_layers)
            self.tf_out = nn.Linear(tf_dim, latent)
        self.trunk = (nn.Sequential(_layer(link_dim + 2 * row_hidden, latent)) if shallow else
                      nn.Sequential(_layer(link_dim + 2 * row_hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                    _layer(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                    _layer(hidden, latent)))
        self.out_act, self.shallow = out_act or ("ln" if layernorm else "none"), shallow
        self.ln = {"ln": nn.LayerNorm(latent), "tanh": nn.Tanh(), "none": nn.Identity()}[self.out_act]
        self.gru = nn.GRUCell(latent, gru) if gru else None
        self.action = _layer(gru or latent, n_actions, 0.01)
        self.aux = nn.ModuleDict({k: _layer(latent, 1, 0.1) for k in self.AUX})

    def encode(self, link, rows, ids, mask):
        """z_t = E(s_t). Pure function of the current state; no recurrence, no hidden state."""
        e = self.id_embedding(ids.clamp(0, self.id_vocab - 1))
        m = mask.unsqueeze(-1)
        if self.arch == "transformer":
            tok = torch.cat([(self.link_tok(link) + self.type_emb[0]).unsqueeze(-2),
                             self.row_tok(torch.cat([rows, e], -1)) + self.type_emb[1]], -2)
            pad = torch.cat([torch.zeros_like(mask[..., :1]), 1.0 - mask], -1).bool()   # absent rows masked out
            return self.ln(self.tf_out(self.tf(tok, src_key_padding_mask=pad)[..., 0, :]))
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


def decode_targets(state: np.ndarray, obs: np.ndarray | None = None,
                   epid: np.ndarray | None = None) -> dict[str, np.ndarray]:
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
    if lk.shape[1] >= len(LINK_FIELDS_V3):
        out["cam_yaw"] = np.stack([np.sin(lk[:, LI["cam_yaw"]]), np.cos(lk[:, LI["cam_yaw"]])], 1)
        out["has_cam"] = np.ones(n, np.float32)
    else:
        out["has_cam"] = np.zeros(n, np.float32)
    if obs is not None:
        # v3 states carry the camera directly, so obs is only a fallback for older collections
        out.setdefault("cam_yaw", np.asarray(obs[:, OBS_CAM_YAW], np.float32))
        out["link_vel"] = np.asarray(obs[:, OBS_LINK_VEL], np.float32)
    # distance and local geometry, in units of 100 so they sit on the same scale as the other real heads
    rp = rows[:, :, [RI["pos_x"], RI["pos_y"], RI["pos_z"]]]
    lp3 = lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    rd = np.linalg.norm(rp - lp3[:, None, :], axis=2) / 100.0
    live = (present > 0.5) & (rows[:, :, RI["id"]].astype(np.int64) == BABA_ID) & (rows[:, :, RI["hp"]] > 0)
    out["row_dist"] = (rd * present)[..., None].astype(np.float32)
    out["near_dist"] = np.where(live.any(1), np.where(live, rd, np.inf).min(1), 0.0)[:, None].astype(np.float32)
    if lk.shape[1] >= len(LINK_FIELDS_V2):
        out["link_dist_wall"] = (np.clip(lk[:, [LI["dist_interact_wall"]]], 0, 400) / 100.0).astype(np.float32)
        out["link_ledge_dy"] = (np.clip(lk[:, [LI["y_dist_to_ledge"]]], -400, 400) / 100.0).astype(np.float32)
    else:
        out["link_dist_wall"] = np.zeros((n, 1), np.float32)
        out["link_ledge_dy"] = np.zeros((n, 1), np.float32)
    out.update(_v2_targets(lk, v2, n))
    out.update(_dpos_targets(lk, epid, n))
    out.update(transition_targets(state, epid))
    return out


MOVED_THRESH = 1.0          # world units of XZ travel in one 100 ms step that counts as "moved"


def transition_targets(state: np.ndarray, epid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """What the action at step t DID, indexed at t and describing t -> t+1.

    These train `zelda_rl.dynamics.TransitionHead`, which is conditioned on (z_t, a_t). Everything here is
    classified rather than regressed, because the regressed version shrank: on truly-moving steps the
    regressed magnitude was 1.19 against a real 11.03. Direction goes to 16 sectors and magnitude to 8 bins.

    `tr_hit_row` is which actor row lost HP over the step, or `N_ROWS` for "none" -- the analogue of decoding
    the damage event rather than differencing two noisy HP decodes. A row that DESPAWNS while it still had HP
    is not a hit (death reload despawns everything at once), so only rows present at both ends count.

    `tr_mask` is 0 on the last frame of each episode, which has no successor.
    """
    from zelda_rl.dynamics import MAG_EDGES, N_DIR_SECTORS
    state = np.asarray(state, np.float32)
    n = len(state)
    lk, rows, _ = _split(state)
    nxt = np.arange(1, n + 1).clip(max=n - 1)
    mask = np.ones(n, np.float32)
    mask[-1] = 0.0
    if epid is not None:
        last = np.r_[epid[1:] != epid[:-1], True]
        mask[last] = 0.0

    pos = lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    d = pos[nxt] - pos
    mag = np.linalg.norm(d[:, [0, 2]], axis=1)
    moved = ((mag > MOVED_THRESH) & (mask > 0)).astype(np.float32)
    ang = np.arctan2(d[:, 2], d[:, 0]) % (2 * np.pi)
    sector = np.floor(ang / (2 * np.pi / N_DIR_SECTORS)).astype(np.int64) % N_DIR_SECTORS
    mbin = np.searchsorted(np.asarray(MAG_EDGES), mag, side="right").clip(0, len(MAG_EDGES) - 1)

    hp = rows[:, :, RI["hp"]]
    present = rows[:, :, RI["present"]] > 0.5
    both = present & present[nxt]
    dropped = (hp[nxt] < hp) & both                     # HP fell while the row stayed present
    hit_row = np.where(dropped.any(1), dropped.argmax(1), N_ROWS).astype(np.int64)
    link_hit = ((lk[nxt, LI["hp"]] < lk[:, LI["hp"]]) & (mask > 0)).astype(np.float32)
    # Despawn: present now, absent next. A death/reload despawns EVERY enemy at once and is not a kill
    # (combat.py documents that trap), so frames where more than half the live rows vanish are excluded.
    vanished = present & ~present[nxt]
    bulk = vanished.sum(1) > np.maximum(present.sum(1) * 0.5, 1)
    vanished = vanished & ~bulk[:, None]
    gone_row = np.where(vanished.any(1), vanished.argmax(1), N_ROWS).astype(np.int64)
    sc = lk[:, LI["scene"]].astype(np.int64)
    exit_ev = ((sc == 0) & (sc[nxt] != 0) & (mask > 0)).astype(np.float32)   # the frame the room is left
    return {"tr_mask": mask, "tr_moved": moved[:, None], "tr_dir": sector, "tr_mag": mbin,
            "tr_dy": d[:, [1]].astype(np.float32), "tr_hit_row": hit_row, "tr_link_hit": link_hit[:, None],
            "tr_gone_row": gone_row, "tr_exit": exit_ev[:, None],
            "tr_mag_units": mag.astype(np.float32)}


def _dpos_targets(lk: np.ndarray, epid: np.ndarray | None, n: int) -> dict[str, np.ndarray]:
    """Displacement INTO this frame, plus whether Link moved at all.

    Absolute position is decoded to ~9-13 world units of error while the true per-step step is 0.58, so the
    signal is ~20x smaller than the noise it is measured through and no predictor improvement can show up in
    it. This head learns the step directly instead. The dream integrates it (pos_t + dpos) rather than
    reading the absolute head, so motion accumulates coherently.

    First frame of each episode has no predecessor: marked not-moved with zero displacement and excluded
    from the magnitude loss by `dpos_mask`.
    """
    pos = lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]]
    d = np.zeros_like(pos)
    d[1:] = pos[1:] - pos[:-1]
    valid = np.ones(n, np.float32)
    valid[0] = 0.0
    if epid is not None:
        boundary = np.r_[True, epid[1:] != epid[:-1]]
        d[boundary] = 0.0
        valid[boundary] = 0.0
    moved = ((np.linalg.norm(d[:, [0, 2]], axis=1) > MOVED_THRESH) & (valid > 0)).astype(np.float32)
    return {"link_dpos": d.astype(np.float32), "link_moved": moved[:, None],
            "dpos_mask": (valid * moved).astype(np.float32)}   # magnitude learned only where it moved


def _v2_targets(lk: np.ndarray, v2: bool, n: int) -> dict[str, np.ndarray]:
    """v2 decoder targets: what Link is doing and what he is touching.

    Emitted for v1 states too, zero-filled, alongside `v2_mask`. The trainer multiplies these losses by the
    mask the same way it already does for the obs-only camera and velocity heads, so a v1 episode
    contributes nothing to them instead of teaching zeros.
    """
    if not v2:
        z = lambda *s: np.zeros((n, *s), np.float32)                        # noqa: E731
        return {"v2_mask": z(), "link_bg": z(len(BG_DECODE_BITS)),
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
    zact = (col("ztarget_timer") != 0).astype(np.float32)
    ms = np.abs(col("melee_state").astype(np.int64)[:, None]
                - np.asarray(MELEE_STATE_VALUES)).argmin(-1)
    return {
        "v2_mask": np.ones(n, np.float32),
        "link_bg": np.stack([((bg >> b) & 1).astype(np.float32) for b in BG_DECODE_BITS], 1),
        "link_melee": ms.astype(np.int64),
        "link_action": np.array([_ACTION_INDEX.get(int(v), N_LINK_ACTIONS - 1)
                                 for v in col("action_fn_rel")], np.int64),
        "link_speed": (col("speed") / 10.0)[:, None],
        # masked by contact: with no wall touched the field is stale, which is the mistake that put the
        # Baba-facing head at 103 degrees until the same masking took it to 5.
        "link_wall_yaw": np.stack([wys * wall, wyc * wall], 1),
        "link_floor_dy": ((col("floor_height") - col("pos_y")) / 100.0)[:, None],
        "link_shape_yaw": np.stack([sys_, syc], 1),
        # masked like wall_yaw: parallelYaw holds a stale value when Z is not held, and scoring a head on
        # stale targets is what kept the Baba-facing head at 103 degrees until masking took it to 5.
        "link_parallel_yaw": np.stack([pys * zact, pyc * zact], 1),
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
# Link's action state machine (Player actionFunc, relative to his update fn so it survives relocation).
# ALL 34 values observed across the three collections are listed, so no real state collapses into
# "other" -- an earlier top-20 cut silently truncated a sidehop transition state into it.
#
# NAMES ARE FROM DRIVING THE INPUT (scripts/probe_move_states.py, and the `moves` scenario whose cycle phase
# is known from t), or from a flag that is definitive (damaged 100%, airborne 100%). An earlier attempt named
# them from marginal co-occurrence and got several wrong: the roll state was called "wall_push" because
# rolling into walls is common in `flail`, and the shield state was called "slow_walk".
#
# Decomp function names could NOT be recovered: zeldaret/oot supports ntsc-1.2 but ships only checksums and
# segment maps, and per-function symbol addresses come from building it. Rank-aligning our 31 observed
# addresses against the 89 Player_Action_* definitions fails because the gaps span unobserved functions.
#
# Finding worth keeping: rolls, backflips and sidehops are NOT separate action functions. Backflip and both
# sidehops all produce 3159404 at 100% airborne -- in OoT they ARE jumps. What separates them is velocity
# direction relative to facing, which the v2 state carries (vel, speed, shape_yaw, turn_sin/cos).
LINK_ACTION_VALUES = (
      3151228,  # 18.675%  locomotion     DRIVEN: constant through run; dominant in roll_run/swing_move
      3208332,  # 18.303%  attack         melee live 68%; dominant in the swing phases
      3143728,  # 15.562%  fn_3143728     unverified
      3149108,  #  9.665%  fn_3149108     unverified
      3160928,  #  6.900%  fn_3160928     unverified
      3144620,  #  5.977%  fn_3144620     unverified
      3159404,  #  5.682%  airborne       airborne 100%; produced by backflip AND both sidehops AND jump
      3145640,  #  4.853%  fn_3145640     unverified
      3161932,  #  4.121%  airborne_z     airborne 100%, Z 90%, melee 9%
      3155380,  #  3.138%  shield         DRIVEN: 20-37% of the two shield phases, speed 0
      3201864,  #  1.982%  fn_3201864     unverified
      3162828,  #  1.331%  fn_3162828     unverified
      3189068,  #  1.148%  fn_3189068     unverified
      3157388,  #  0.897%  hurt           damaged flag 100%
      3151488,  #  0.538%  fn_3151488     unverified
      3166492,  #  0.389%  roll           DRIVEN: constant through the roll leg, speed 5.6
      3157620,  #  0.239%  hurt_still     damaged flag 100%, speed 0
      3146184,  #  0.132%  airborne_x     DRIVEN: appears only in sidehop legs; brief transition
      3156912,  #  0.107%  hurt_airborne  damaged + airborne 100%
      3148004,  #  0.089%  fn_3148004     unverified
      3191624,  #  0.070%  falling        airborne 90%, speed 0
            0,  #  0.043%  no_player      player pointer invalid
      3165496,  #  0.037%  fn_3165496     unverified
      3168412,  #  0.026%  fn_3168412     unverified
      3156796,  #  0.023%  fn_3156796     unverified
      3148788,  #  0.021%  fn_3148788     unverified
      3148548,  #  0.013%  fn_3148548     unverified
      3156464,  #  0.011%  fn_3156464     unverified
      3158316,  #  0.011%  fn_3158316     unverified
      3193384,  #  0.008%  fn_3193384     unverified
      3191336,  #  0.005%  fn_3191336     unverified
      3164012,  #  0.001%  fn_3164012     unverified
      3163236,  #  0.000%  fn_3163236     unverified
      3149720,  #  0.000%  fn_3149720     unverified
)
LINK_ACTION_NAMES = ("locomotion", "attack", "fn_3143728", "fn_3149108", "fn_3160928", "fn_3144620", "airborne", "fn_3145640", "airborne_z", "shield", "fn_3201864", "fn_3162828", "fn_3189068", "hurt", "fn_3151488", "roll", "hurt_still", "airborne_x", "hurt_airborne", "fn_3148004", "falling", "no_player", "fn_3165496", "fn_3168412", "fn_3156796", "fn_3148788", "fn_3148548", "fn_3156464", "fn_3158316", "fn_3193384", "fn_3191336", "fn_3164012", "fn_3163236", "fn_3149720", "other")
N_LINK_ACTIONS = len(LINK_ACTION_VALUES) + 1                # + "other"
assert len(LINK_ACTION_NAMES) == N_LINK_ACTIONS
# Merge routine values that are ONE semantic state, per docs/ADDING_OBSERVATIONS.md step 2: "Some
# behaviours have two routine values (e.g. two idle, two dying). Merge them into one semantic state."
# 3146184 is a 1-2 frame transition inside a sidehop -- it is airborne like 3159404, and as its own class it
# scored recall 0.12 on 32 test frames even after the `moves` scenario raised its density 30x.
# `hurt_still` and `hurt_airborne` are the same semantic state as `hurt` -- reacting to damage -- and split
# three ways `hurt` scored recall 0.71 on 34 test frames. Merged, per the same rule that fixed the sidehop
# transition. Nothing in the reward needs the distinction: the DAMAGED stateFlags1 bit decodes at 1.000/1.00.
LINK_ACTION_MERGE = {3146184: 3159404, 3157620: 3157388, 3156912: 3157388}
_ACTION_INDEX = {v: i for i, v in enumerate(LINK_ACTION_VALUES)}
_ACTION_INDEX.update({src: _ACTION_INDEX[dst] for src, dst in LINK_ACTION_MERGE.items()
                      if dst in _ACTION_INDEX})

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
        self.link_bg = _layer(latent, len(BG_DECODE_BITS), 0.1)        # ground / wall / ceiling / water / crushed
        self.link_melee = _layer(latent, N_MELEE_STATES, 0.1)   # a sword swing is live
        self.link_action = _layer(latent, N_LINK_ACTIONS, 0.1)  # Link's action state machine
        # Displacement is ZERO-INFLATED: Link does not move at all on 95% of real steps (true per-step XZ
        # mean 0.58, median 0.00). A regression head cannot emit an exact zero, so it emitted 12.11 every
        # step -- 20.9x too much, in incoherent directions. Split it the way HP was split (0.286 -> 0.906):
        # a classifier for "did it change", and a magnitude trained ONLY where it did.
        self.link_moved = _layer(latent, 1, 0.1)
        self.link_dpos = _layer(latent, 3, 0.1)
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
                "link_moved": self.link_moved(z), "link_dpos": self.link_dpos(z),
                **dict(zip(V2_BODY, self.link_body(z).split(V2_BODY_SPLIT, dim=-1))),
                "row_present": r[..., 0], "row_is_baba": r[..., 1], "row_hp": r[..., 2:j],
                "row_pos": r[..., j:j + 3],
                "row_state": r[..., j + 3:j + 3 + self.n_states],
                "row_facing_link": r[..., j + 3 + self.n_states:j + 5 + self.n_states],
                "row_timer": r[..., j + 5 + self.n_states:j + 6 + self.n_states],
                "row_vel": r[..., j + 6 + self.n_states:j + 9 + self.n_states],
                "link_facing_target": r[..., j + 9 + self.n_states:j + 11 + self.n_states]}


# --- One decoder, heterogeneous targets, differentiated by output activation ------------------------------
#
# StateDecoder grew into ~18 separate nn.Linear heads hanging off z, plus a per-row branch. That is a lot of
# independent output layers competing for the same latent, and nothing shares structure between, say, Link's
# position and an actor's position. This replaces them with ONE trunk and ONE output layer, sliced by a spec
# that records each field's width and what kind of quantity it is; the activation is what differentiates them:
#
#   logits  raw, cross-entropy applies the softmax        (HP classes, behaviour state, action state)
#   bits    raw, BCE applies the sigmoid                  (stateFlags, collision flags, moved)
#   unit    TANH, because these are sin/cos pairs and belong in [-1, 1]
#   real    raw                                           (positions, velocities, displacements)
#
# Output keys are identical to StateDecoder's, so every consumer -- decoded_to_states, the training losses, the
# probes -- works unchanged.
def _decode_spec(n_rows=N_ROWS):
    g = [("link_hp", len(LINK_HP_VALUES), "logits"), ("link_pos", 3, "real"), ("link_yaw", 2, "unit"),
         ("link_bits", 3, "bits"), ("targeted", n_rows + 1, "logits"), ("scene", 2, "logits"),
         ("room", MAX_ROOM, "logits"), ("baba_count", N_BABA_COUNT, "logits"),
         ("link_sf1", len(SF1_NAMED), "bits"), ("cam_yaw", 2, "unit"), ("link_vel", 3, "real"),
         ("link_bg", len(BG_DECODE_BITS), "bits"), ("link_melee", N_MELEE_STATES, "logits"),
         ("link_action", N_LINK_ACTIONS, "logits"), ("link_moved", 1, "bits"), ("link_dpos", 3, "real"),
         ("link_speed", 1, "real"), ("link_wall_yaw", 2, "unit"), ("link_floor_dy", 1, "real"),
         ("link_shape_yaw", 2, "unit"), ("link_parallel_yaw", 2, "unit"), ("link_fall", 1, "real"),
         ("link_ztarget", 1, "bits"), ("link_airborne", 1, "bits"),
         # DISTANCE, decoded directly rather than left as the difference of two absolute positions. It is what
         # `approach` pays on and what `shield_facing` gates on (dist < shield_radius), and section 22 showed
         # absolute position diverges, so a derived distance compounds two diverging errors.
         ("near_dist", 1, "real"),
         # LOCAL GEOMETRY. The model had no representation of the room: it learned only "I am touching a wall"
         # after contact, so a predicted displacement could point straight through one at no cost. These are
         # the geometry fields the v3 state already records and that were encoder inputs but never decoded.
         ("link_dist_wall", 1, "real"), ("link_ledge_dy", 1, "real")]
    r = [("row_present", 1, "bits"), ("row_is_baba", 1, "bits"), ("row_hp", len(ROW_HP_VALUES), "logits"),
         ("row_pos", 3, "real"), ("row_state", N_BABA_STATES, "logits"), ("row_facing_link", 2, "unit"),
         ("row_timer", 1, "real"), ("row_vel", 3, "real"), ("link_facing_target", 2, "unit"),
         ("row_dist", 1, "real")]
    return g, r


class UnifiedDecoder(nn.Module):
    """z -> every decoded field, from one trunk and one output layer, sliced by `_decode_spec`."""

    def __init__(self, latent: int = 128, hidden: int = 512, n_rows: int = N_ROWS):
        super().__init__()
        self.n_rows = n_rows
        self.gspec, self.rspec = _decode_spec(n_rows)
        self.gw = sum(w for _, w, _ in self.gspec)
        self.rw = sum(w for _, w, _ in self.rspec)
        self.trunk = nn.Sequential(_layer(latent, hidden), nn.ReLU(), _layer(hidden, hidden), nn.ReLU())
        self.out = _layer(hidden, self.gw + n_rows * self.rw, 0.1)

    def forward(self, z):
        h = self.out(self.trunk(z))
        g, r = h[:, :self.gw], h[:, self.gw:].reshape(-1, self.n_rows, self.rw)
        o, i = {}, 0
        for name, w, kind in self.gspec:
            v = g[:, i:i + w]
            o[name] = torch.tanh(v) if kind == "unit" else v
            i += w
        j = 0
        for name, w, kind in self.rspec:
            v = r[..., j:j + w]
            if kind == "unit":
                v = torch.tanh(v)
            o[name] = v.squeeze(-1) if w == 1 and name in ("row_present", "row_is_baba") else v
            j += w
        return o
