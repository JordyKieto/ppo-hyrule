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
