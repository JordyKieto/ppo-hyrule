"""Observation schema: raw OoT game state read from N64 RAM, and the fixed-size policy vector built from it.

All addresses are NTSC 1.2 (Wii VC NACE); see docs/OBSERVATION_SCHEMA.md.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAVE_CTX = 0x8011AC80
PLAY_STATE = 0x801C8D60
VI_COUNTER = 0x8011DBB0          # 60 Hz emulated video-interrupt counter (restored by savestates)
ACTOR_EN_ITEM00 = 0x0015
RUPEE_VALUE = {0x00: 1, 0x01: 5, 0x02: 20, 0x13: 50, 0x14: 200}   # EnItem00 params -> rupee value
N_ITEMS = 4                      # nearest rupee actors exposed to the policy


def _u32(r, a): return struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
def _s16(r, a): return struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
def _vec3(r, a): return struct.unpack_from(">fff", r, a & 0x3FFFFF)


@dataclass
class GameState:
    vi: int                                   # emulated time (1/60 s)
    scene: int
    rupees: int
    health: int
    health_cap: int
    pos: tuple[float, float, float]           # Link world position
    yaw: int                                  # Link facing, 0..65535
    rupee_items: list[tuple[int, tuple[float, float, float]]] = field(default_factory=list)  # (value, pos)
    player_ok: bool = True

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray(self.pos, np.float32)


def read_state(ram) -> GameState:
    """~25 us on a zero-copy view. `ram` is any buffer of the 4 MB N64 RAM."""
    ps, sc = PLAY_STATE, SAVE_CTX
    player = _u32(ram, ps + 0x1C44)
    ok = 0x80000000 <= player < 0x80400000
    pos = _vec3(ram, player + 0x24) if ok else (math.nan,) * 3
    yaw = _s16(ram, player + 0x32) & 0xFFFF if ok else 0
    items = []
    for cat in range(12):
        a, guard = _u32(ram, ps + 0x1C24 + 0x10 + cat * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 128:
            if (_u32(ram, a) >> 16) == ACTOR_EN_ITEM00:
                value = RUPEE_VALUE.get(_s16(ram, a + 0x1C) & 0xFF)
                if value:
                    items.append((value, _vec3(ram, a + 0x24)))
            a = _u32(ram, a + 0x124)
            guard += 1
    return GameState(
        vi=_u32(ram, VI_COUNTER), scene=_s16(ram, ps + 0xA4), rupees=_s16(ram, sc + 0x34),
        health=_s16(ram, sc + 0x30), health_cap=_s16(ram, sc + 0x2E), pos=pos, yaw=yaw,
        rupee_items=items, player_ok=ok,
    )


def read_state_file(path: str | Path) -> GameState:
    """GameState of a Dolphin savestate file (offline; ~150 ms)."""
    from zelda_rl.savestate import decompress_state, find_n64_ram
    raw = decompress_state(Path(path).read_bytes())
    base = find_n64_ram(raw)
    return read_state(raw[base:base + 0x400000])


# ---- policy vector --------------------------------------------------------------------------
POS_SCALE = 1000.0
FEATURE_CLIP = 10.0      # garbage actor positions / teleport velocities must not blow up the policy
RUPEE_SCALE = 20.0

OBS_FIELDS = (
    [("link_pos", 3), ("goal_pos", 3), ("goal_delta", 3), ("goal_dist", 1), ("link_yaw_sin_cos", 2),
     ("link_vel", 3), ("rupees", 1), ("goal_rupees", 1), ("rupee_gap", 1), ("health_frac", 1),
     ("time_left", 1), ("prev_action_dir_sin_cos", 2), ("prev_action_button", 2)]
    + [(f"item{i}_delta_value_present", 5) for i in range(N_ITEMS)]
)
OBS_DIM = sum(n for _, n in OBS_FIELDS)


def featurize(s: GameState, goal: GameState, prev_pos: np.ndarray, time_left: float,
              prev_dir: float | None, prev_button: int) -> np.ndarray:
    """Fixed-size float32 vector (OBS_DIM); layout = OBS_FIELDS. Distances in units of POS_SCALE."""
    xyz, g = s.xyz, goal.xyz
    delta = (g - xyz) / POS_SCALE
    vel = (xyz - prev_pos) / 10.0
    yaw = s.yaw / 65536.0 * 2 * math.pi
    items = sorted(s.rupee_items, key=lambda it: (it[1][0] - xyz[0]) ** 2 + (it[1][2] - xyz[2]) ** 2)[:N_ITEMS]
    item_feats = []
    for i in range(N_ITEMS):
        if i < len(items):
            v, p = items[i]
            item_feats += [(p[0] - xyz[0]) / POS_SCALE, (p[1] - xyz[1]) / POS_SCALE, (p[2] - xyz[2]) / POS_SCALE,
                           v / RUPEE_SCALE, 1.0]
        else:
            item_feats += [0.0, 0.0, 0.0, 0.0, 0.0]
    vec = [
        *(xyz / POS_SCALE), *(g / POS_SCALE), *delta, float(np.linalg.norm(g - xyz)) / POS_SCALE,
        math.sin(yaw), math.cos(yaw), *vel,
        s.rupees / RUPEE_SCALE, goal.rupees / RUPEE_SCALE, max(goal.rupees - s.rupees, 0) / RUPEE_SCALE,
        s.health / max(s.health_cap, 1), time_left,
        *( (math.sin(prev_dir), math.cos(prev_dir)) if prev_dir is not None else (0.0, 0.0) ),
        float(prev_button == 1), float(prev_button == 2),
        *item_feats,
    ]
    out = np.asarray(vec, np.float32)
    assert out.shape == (OBS_DIM,)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)


# ==== v2: nearest actors of any type, camera, player state flags, save-progress flags ================
PLAYER_SF1 = 0x66C
MAIN_CAMERA = PLAY_STATE + 0x1E0
ACTOR_EN_ELF = 0x0018              # Navi: always next to Link, excluded from the actor table
N_ACTORS = 16
ACTOR_ID_VOCAB = 512

# stateFlags1 bits exposed to the policy (decomp names; 15/16 verified on slot 1)
SF1_BITS = [("talking", 6), ("dead", 7), ("getting_item", 10), ("carrying", 11), ("ledge_hang", 13),
            ("ledge_climb", 14), ("z_targeting", 15), ("friendly_focus", 16), ("jumping", 18),
            ("freefall", 19), ("climbing_ladder", 21), ("damaged", 26), ("in_water", 27), ("in_cutscene", 29)]

# save-progress flag blob: (offset in SaveContext, length)
FLAG_BLOCKS = [("items", 0x74, 24), ("equipment", 0x9C, 2), ("upgrades", 0xA0, 4), ("quest", 0xA4, 4),
               ("eventChkInf", 0xED4, 28), ("itemGetInf", 0xEF0, 8), ("infTable", 0xEF8, 60)]
SCENE_FLAGS_SIZE = 0x1C


@dataclass
class ActorInfo:
    id: int
    category: int
    params: int
    flags: int
    health: int
    pos: tuple[float, float, float]
    dist: float


@dataclass
class GameStateV2(GameState):
    cam_yaw: float = 0.0               # radians, math convention in the x/z plane: atan2(dz, dx) of eye->at
    sf1: int = 0
    equipment: int = 0
    flags: np.ndarray | None = None    # uint8 blob: FLAG_BLOCKS + current scene's saved flags
    actors: list[ActorInfo] = field(default_factory=list)   # sorted by distance, Link and Navi excluded


def read_state_v2(ram) -> GameStateV2:
    base = read_state(ram)
    ps, sc = PLAY_STATE, SAVE_CTX
    player = _u32(ram, ps + 0x1C44)
    ok = base.player_ok
    at, eye = _vec3(ram, MAIN_CAMERA + 0x50), _vec3(ram, MAIN_CAMERA + 0x5C)
    cam_yaw = math.atan2(at[2] - eye[2], at[0] - eye[0])
    sf1 = _u32(ram, player + PLAYER_SF1) if ok else 0
    blob = [bytes(ram[(sc + off) & 0x3FFFFF:(sc + off + n) & 0x3FFFFF]) for _, off, n in FLAG_BLOCKS]
    if 0 <= base.scene < 124:
        so = (sc + 0xD4 + base.scene * SCENE_FLAGS_SIZE) & 0x3FFFFF
        blob.append(bytes(ram[so:so + SCENE_FLAGS_SIZE]))
    else:
        blob.append(bytes(SCENE_FLAGS_SIZE))
    lp = base.pos
    actors = []
    for cat in range(12):
        a, guard = _u32(ram, ps + 0x1C24 + 0x10 + cat * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 128:
            aid = struct.unpack_from(">H", ram, a & 0x3FFFFF)[0]
            if a != player and aid != ACTOR_EN_ELF:
                p = _vec3(ram, a + 0x24)
                d = math.dist(p, lp) if ok else math.inf
                actors.append(ActorInfo(aid, ram[(a + 2) & 0x3FFFFF], _s16(ram, a + 0x1C) & 0xFFFF,
                                        _u32(ram, a + 4), ram[(a + 0xAF) & 0x3FFFFF], p, d))
            a = _u32(ram, a + 0x124)
            guard += 1
    actors.sort(key=lambda x: x.dist)
    return GameStateV2(**vars(base), cam_yaw=cam_yaw, sf1=sf1, equipment=_s16(ram, sc + 0x9C) & 0xFFFF,
                       flags=np.frombuffer(b"".join(blob), np.uint8), actors=actors[:N_ACTORS])


def read_state_file_v2(path: str | Path) -> GameStateV2:
    from zelda_rl.savestate import decompress_state, find_n64_ram
    raw = decompress_state(Path(path).read_bytes())
    base = find_n64_ram(raw)
    return read_state_v2(raw[base:base + 0x400000])


def flag_progress(cur: np.ndarray, goal: np.ndarray, start: np.ndarray) -> tuple[int, int]:
    """(missing, extra): goal bits not yet set that the start lacked; bits set now that neither goal nor start has."""
    need = goal & ~start
    missing = int(np.unpackbits(need & ~cur).sum())
    extra = int(np.unpackbits(cur & ~goal & ~start).sum())
    return missing, extra


GLOBAL_FIELDS_V2 = [("link_pos", 3), ("goal_pos", 3), ("goal_delta", 3), ("goal_dist", 1), ("link_yaw_sin_cos", 2),
                    ("camera_yaw_sin_cos", 2), ("link_vel", 3), ("rupees", 1), ("goal_rupees", 1), ("rupee_gap", 1),
                    ("health_frac", 1), ("time_left", 1), ("prev_action_world_dir_sin_cos", 2),
                    ("prev_action_button", 2), ("state_flags1_bits", len(SF1_BITS)), ("owns_sword_shield", 2),
                    ("goal_flags_missing_extra", 2), ("same_scene_as_goal", 1)]
ACTOR_FIELDS_V2 = [("id", 1), ("category_onehot", 12), ("rel_pos", 3), ("dist", 1), ("flag_targetable_hostile_friendly", 3),
                   ("health", 1), ("rupee_value", 1), ("params_low", 1), ("present", 1)]
GLOBAL_DIM_V2 = sum(n for _, n in GLOBAL_FIELDS_V2)
ACTOR_DIM_V2 = sum(n for _, n in ACTOR_FIELDS_V2)
OBS_DIM_V2 = GLOBAL_DIM_V2 + N_ACTORS * ACTOR_DIM_V2


def featurize_v2(s: GameStateV2, goal: GameStateV2, prev_pos: np.ndarray, time_left: float,
                 prev_dir: float | None, prev_button: int, start_flags: np.ndarray) -> np.ndarray:
    """float32 [GLOBAL_DIM_V2 global | N_ACTORS x ACTOR_DIM_V2 actor table]. Actor `id` is a raw index (model embeds it)."""
    xyz, g = s.xyz, goal.xyz
    yaw = s.yaw / 65536.0 * 2 * math.pi
    missing, extra = flag_progress(s.flags, goal.flags, start_flags)
    glob = [
        *(xyz / POS_SCALE), *(g / POS_SCALE), *((g - xyz) / POS_SCALE), float(np.linalg.norm(g - xyz)) / POS_SCALE,
        math.sin(yaw), math.cos(yaw), math.sin(s.cam_yaw), math.cos(s.cam_yaw), *((xyz - prev_pos) / 10.0),
        s.rupees / RUPEE_SCALE, goal.rupees / RUPEE_SCALE, max(goal.rupees - s.rupees, 0) / RUPEE_SCALE,
        s.health / max(s.health_cap, 1), time_left,
        *((math.sin(prev_dir), math.cos(prev_dir)) if prev_dir is not None else (0.0, 0.0)),
        float(prev_button == 1), float(prev_button == 2),
        *[float(s.sf1 >> b & 1) for _, b in SF1_BITS],
        float(s.equipment & 0x000F != 0), float(s.equipment & 0x00F0 != 0),
        missing / 8.0, extra / 8.0, float(s.scene == goal.scene),
    ]
    table = np.zeros((N_ACTORS, ACTOR_DIM_V2), np.float32)
    for i, a in enumerate(s.actors):
        row = table[i]
        row[0] = min(a.id, ACTOR_ID_VOCAB - 1)
        if 0 <= a.category < 12:
            row[1 + a.category] = 1.0
        row[13:16] = [(a.pos[0] - xyz[0]) / POS_SCALE, (a.pos[1] - xyz[1]) / POS_SCALE, (a.pos[2] - xyz[2]) / POS_SCALE]
        row[16] = a.dist / POS_SCALE
        row[17:20] = [a.flags & 1, a.flags >> 2 & 1, a.flags >> 3 & 1]
        row[20] = a.health / 20.0
        row[21] = (RUPEE_VALUE.get(a.params & 0xFF, 0) / RUPEE_SCALE) if a.id == ACTOR_EN_ITEM00 else 0.0
        row[22] = (a.params & 0xFF) / 255.0
        row[23] = 1.0
    out = np.concatenate([np.asarray(glob, np.float32), table.reshape(-1)])
    assert out.shape == (OBS_DIM_V2,), out.shape
    ids = out[GLOBAL_DIM_V2::ACTOR_DIM_V2].copy()          # actor id column is an index, not a feature: don't clip it
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)
    out[GLOBAL_DIM_V2::ACTOR_DIM_V2] = ids
    return out
