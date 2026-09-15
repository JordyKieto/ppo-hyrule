"""Observation v6: raw aim + egocentric geometry, for walking as well as first-person aiming.

Link acts in a camera-relative way (stick directions are rotated by the camera; in first person the stick rotates
the view and the camera eye sits at Link's head, verified live), but v5 described the world only in world-frame
offsets plus Link's body yaw. v6 gives the policy the same frames it acts in (no hand-crafted aim-error features):

global AIM/EGO block
  aim_forward_world      camera eye->at unit vector (first person: the aim ray)
  aim_pitch / aim_yaw    raw Player focus.rot.x / .y (sin, cos)
  cam_eye_rel_link       camera eye relative to Link
  link_facing_vs_camera  Link body yaw relative to the camera yaw (sin, cos)
  link_velocity_cam      Link's displacement since the last step, in the camera frame (right, up, forward)
  gohma_eye_cam/_body    Gohma's eye in the camera frame and in Link's body frame (right, up, forward)
  larva1_cam/_body       nearest live larva, both frames
actor table (all 16 nearest actors, any type), inserted before `present`
  ego_cam_rel_pos        position in the camera frame
  ego_body_rel_pos       position in Link's body frame
"""
from __future__ import annotations

import math

import numpy as np

from zelda_rl.obs import ACTOR_DIM_V2, ACTOR_FIELDS_V2, FEATURE_CLIP, N_ACTORS, POS_SCALE

AIM_FIELDS = [("aim_forward_world", 3), ("aim_pitch_sin_cos", 2), ("aim_yaw_sin_cos", 2), ("cam_eye_rel_link", 3),
              ("link_facing_vs_camera_sin_cos", 2), ("link_velocity_cam", 3),
              ("gohma_eye_cam", 3), ("gohma_eye_body", 3), ("larva1_cam", 3), ("larva1_body", 3)]
AIM_DIM = sum(n for _, n in AIM_FIELDS)
ACTOR_FIELDS_V6 = ACTOR_FIELDS_V2[:-1] + [("ego_cam_rel_pos", 3), ("ego_body_rel_pos", 3)] + ACTOR_FIELDS_V2[-1:]
ACTOR_DIM_V6 = ACTOR_DIM_V2 + 6
EGO_INSERT = ACTOR_DIM_V2 - 1          # new actor columns go right before `present`
UP = (0.0, 1.0, 0.0)


def _frame(forward) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(forward, np.float64)
    n = np.linalg.norm(f)
    f = f / n if np.isfinite(n) and n > 1e-6 else np.array([0.0, 0.0, 1.0])
    r = np.cross(f, UP)
    rn = np.linalg.norm(r)
    r = r / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
    return r, np.cross(r, f), f


def camera_frame(s):
    """(right, up, forward) of the camera (walking: behind Link; first person: along the aim)."""
    f = np.subtract(s.cam_at, s.cam_eye)
    if not np.all(np.isfinite(f)) or np.linalg.norm(f) < 1e-3:
        return body_frame(s)
    return _frame(f)


def body_frame(s):
    """(right, up, forward) of Link's body (yaw only). OoT yaw 0 = +z, 0x4000 = +x."""
    yaw = s.yaw / 65536.0 * 2 * math.pi
    return _frame((math.sin(yaw), 0.0, math.cos(yaw)))


def to_frame(frame, origin, point) -> list[float]:
    d = np.subtract(point, origin)
    return [float(np.dot(d, b)) / POS_SCALE for b in frame]


def aim_features(s, cstate, prev_pos=None) -> np.ndarray:
    from zelda_rl.gohma import find_gohma, larvae
    cam, body = camera_frame(s), body_frame(s)
    pitch, yaw = s.aim_pitch / 32768.0 * math.pi, s.aim_yaw / 32768.0 * math.pi
    body_yaw = s.yaw / 65536.0 * 2 * math.pi
    cam_yaw = math.atan2(cam[2][0], cam[2][2])                           # same convention as body yaw
    rel = body_yaw - cam_yaw
    vel = [0.0, 0.0, 0.0]
    if prev_pos is not None and np.all(np.isfinite(prev_pos)):
        d = np.subtract(s.pos, prev_pos)
        vel = [float(np.dot(d, b)) / 10.0 for b in cam]
    out = [*cam[2], math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw),
           *[(s.cam_eye[i] - s.pos[i]) / POS_SCALE for i in range(3)], math.sin(rel), math.cos(rel), *vel]
    g = find_gohma(cstate) if cstate is not None else None
    out += (to_frame(cam, s.cam_eye, g.focus) + to_frame(body, s.pos, g.focus)) if g is not None else [0.0] * 6
    lv = sorted((e for e in larvae(cstate) if e.hp > 0), key=lambda e: math.dist(e.pos, cstate.pos)) if cstate is not None else []
    out += (to_frame(cam, s.cam_eye, lv[0].pos) + to_frame(body, s.pos, lv[0].pos)) if lv else [0.0] * 6
    arr = np.asarray(out, np.float32)
    assert arr.shape == (AIM_DIM,), arr.shape
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)


def splice_actor_table(table_v2_flat: np.ndarray, s) -> np.ndarray:
    """v2 actor table (N_ACTORS x 24, flat) -> v6 (N_ACTORS x 30, flat): + camera-frame and body-frame positions."""
    t = table_v2_flat.reshape(N_ACTORS, ACTOR_DIM_V2)
    ego = np.zeros((N_ACTORS, 6), np.float32)
    cam, body = camera_frame(s), body_frame(s)
    for i, a in enumerate(s.actors[:N_ACTORS]):
        ego[i] = to_frame(cam, s.cam_eye, a.pos) + to_frame(body, s.pos, a.pos)
    ego = np.clip(np.nan_to_num(ego, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)
    ego[t[:, -1] == 0] = 0.0
    return np.concatenate([t[:, :EGO_INSERT], ego, t[:, EGO_INSERT:]], 1).reshape(-1)


# ======================================================================================================================
# Observation v7: + motion, facing, size for every actor; every larva / egg; seeds in flight and the last impact point
# ======================================================================================================================
from zelda_rl.obs import ACTOR_SEED  # noqa: E402

ACTOR_FIELDS_V7 = ACTOR_FIELDS_V2[:-1] + [
    ("ego_cam_rel_pos", 3), ("ego_body_rel_pos", 3),        # where it is (camera frame, Link body frame)
    ("ego_cam_vel", 3), ("ego_body_vel", 3),                 # raw actor velocity (+0x5C) in both frames
    ("facing_cam_fwd_right", 2), ("facing_body_fwd_right", 2),   # which way it faces (its forward in each frame)
    ("facing_vs_link_cos_sin", 2),                           # its facing relative to the direction to Link
    ("scale", 1), ("cyl_radius_height", 2), ("mass", 1),     # size / collision cylinder / movable
] + ACTOR_FIELDS_V2[-1:]
ACTOR_DIM_V7 = sum(n for _, n in ACTOR_FIELDS_V7)
N_LARVA_SLOTS, N_SEED_SLOTS = 3, 2
_LARVA_SLOT = [("present", 1), ("hp", 1), ("state_onehot", 11), ("cam", 3), ("body", 3), ("vel_cam", 3), ("facing_vs_link_cos_sin", 2)]
_SEED_SLOT = [("present", 1), ("cam", 3), ("vel_cam", 3), ("vel_body", 3)]
EGO_V7_FIELDS = AIM_FIELDS[:6] + [
    ("gohma_eye_cam", 3), ("gohma_eye_body", 3), ("gohma_vel_cam", 3), ("gohma_facing_cam_fwd_right", 2),
    ("gohma_facing_body_fwd_right", 2)] + \
    [(f"larva{i}_{n}", k) for i in range(N_LARVA_SLOTS) for n, k in _LARVA_SLOT] + \
    [(f"seed{i}_{n}", k) for i in range(N_SEED_SLOTS) for n, k in _SEED_SLOT] + \
    [("last_impact_present", 1), ("since_last_impact", 1), ("last_impact_cam", 3), ("last_impact_minus_gohma_eye_cam", 3),
     ("last_impact_to_gohma_eye_dist", 1)]
EGO_V7_DIM = sum(n for _, n in EGO_V7_FIELDS)


def _facing(frame, yaw_s16: int) -> list[float]:
    y = yaw_s16 / 32768.0 * math.pi
    v = (math.sin(y), 0.0, math.cos(y))
    return [float(np.dot(v, frame[2])), float(np.dot(v, frame[0]))]


def _vel(frame, d, per: float = 10.0) -> list[float]:
    return [float(np.dot(d, b)) / per for b in frame]


class EgoMemory:
    """Per-episode memory for v7: seeds tracked EVERY emulated frame (polled from env.ready), so the impact point is
    where a flying seed actually vanished (a seed moves ~50 units/frame: 10 Hz sampling would be ~300 units off).
    A cocked (not yet fired) seed is also an En_Arrow actor, at Link's hand with zero velocity: not an impact."""

    FLYING_SPEED = 5.0

    def reset(self) -> None:
        self.seeds: dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        self.last_impact = None
        self.last_impact_t = None
        self.t = 0

    def __init__(self):
        self.reset()

    def poll(self, ram) -> None:
        """Scan the item-action actor list for seeds (cheap: a handful of actors) and record impacts."""
        from zelda_rl.obs import PLAY_STATE, _u32, _vec3
        import struct as _struct
        live, a, guard = {}, _u32(ram, PLAY_STATE + 0x1C24 + 0x10 + 7 * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 64:
            if _struct.unpack_from(">H", ram, a & 0x3FFFFF)[0] == ACTOR_SEED:
                live[a] = (_vec3(ram, a + 0x24), _vec3(ram, a + 0x5C))
            a, guard = _u32(ram, a + 0x124), guard + 1
        for ptr, (pos, vel) in self.seeds.items():
            if ptr not in live and math.sqrt(sum(v * v for v in vel)) > self.FLYING_SPEED:
                self.last_impact, self.last_impact_t = tuple(np.add(pos, vel)), self.t   # one more update of travel
        self.seeds = live

    def step(self, s) -> None:
        self.t += 1


def ego_v7_features(s, cstate, prev_pos, mem: EgoMemory) -> np.ndarray:
    from zelda_rl.gohma import LARVA_STATES, find_gohma, larva_state, larvae
    cam, body = camera_frame(s), body_frame(s)
    out = list(aim_features(s, None, prev_pos)[:15])          # aim + camera eye + facing-vs-camera + Link velocity
    g = find_gohma(cstate) if cstate is not None else None
    if g is not None:
        out += to_frame(cam, s.cam_eye, g.focus) + to_frame(body, s.pos, g.focus) + _vel(cam, g.vel) + _facing(cam, g.yaw) + _facing(body, g.yaw)
    else:
        out += [0.0] * 13
    lv = sorted(larvae(cstate) if cstate is not None else [], key=lambda e: (e.params, math.dist(e.pos, s.pos)))[:N_LARVA_SLOTS]
    for i in range(N_LARVA_SLOTS):
        if i < len(lv):
            e = lv[i]
            onehot = [0.0] * len(LARVA_STATES)
            onehot[LARVA_STATES.index(larva_state(e))] = 1.0
            ang = ((e.yaw - e.yaw_to_player + 32768) % 65536 - 32768) / 32768.0 * math.pi
            out += [1.0, e.hp / 2.0, *onehot, *to_frame(cam, s.cam_eye, e.pos), *to_frame(body, s.pos, e.pos),
                    *_vel(cam, e.vel), math.cos(ang), math.sin(ang)]
        else:
            out += [0.0] * sum(k for _, k in _LARVA_SLOT)
    seeds = sorted(s.seeds, key=lambda a: a.dist)[:N_SEED_SLOTS]
    for i in range(N_SEED_SLOTS):
        if i < len(seeds):
            a = seeds[i]
            out += [1.0, *to_frame(cam, s.cam_eye, a.pos), *_vel(cam, a.vel), *_vel(body, a.vel)]
        else:
            out += [0.0] * sum(k for _, k in _SEED_SLOT)
    if mem.last_impact is not None:
        imp = mem.last_impact
        out += [1.0, min((mem.t - mem.last_impact_t) / 50.0, 1.0), *to_frame(cam, s.cam_eye, imp)]
        if g is not None:
            out += [float(np.dot(np.subtract(imp, g.focus), b)) / POS_SCALE for b in cam] + [math.dist(imp, g.focus) / POS_SCALE]
        else:
            out += [0.0] * 4
    else:
        out += [0.0] * 9
    arr = np.asarray(out, np.float32)
    assert arr.shape == (EGO_V7_DIM,), (arr.shape, EGO_V7_DIM)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)


def actor_table_v7(table_v2_flat: np.ndarray, s, mem: EgoMemory) -> np.ndarray:
    """v2 actor table -> v7: + frames, velocities, facing, size per actor (inserted before `present`)."""
    t = table_v2_flat.reshape(N_ACTORS, ACTOR_DIM_V2)
    extra_n = ACTOR_DIM_V7 - ACTOR_DIM_V2
    extra = np.zeros((N_ACTORS, extra_n), np.float32)
    cam, body = camera_frame(s), body_frame(s)
    for i, a in enumerate(s.actors[:N_ACTORS]):
        d = a.vel                                                   # raw actor velocity (+0x5C)
        ang = ((a.yaw - a.yaw_to_player + 32768) % 65536 - 32768) / 32768.0 * math.pi
        extra[i] = (to_frame(cam, s.cam_eye, a.pos) + to_frame(body, s.pos, a.pos) + _vel(cam, d) + _vel(body, d)
                    + _facing(cam, a.yaw) + _facing(body, a.yaw) + [math.cos(ang), math.sin(ang)]
                    + [a.scale * 100.0, a.cyl_radius / 100.0, a.cyl_height / 100.0, a.mass / 255.0])
    extra = np.clip(np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0), -FEATURE_CLIP, FEATURE_CLIP)
    extra[t[:, -1] == 0] = 0.0
    return np.concatenate([t[:, :EGO_INSERT], extra, t[:, EGO_INSERT:]], 1).reshape(-1)
