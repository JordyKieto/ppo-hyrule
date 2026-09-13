"""Goal-conditioned ("JEPA-style") distance to a target game state, and the shaped reward.

The state embedding is e(s) = [x, y, z] / POS_SCALE  ++  rupees / goal_rupees  (position + rupees are what matter).
goal_dist = ||e_pos(s) - e_pos(g)|| + RUPEE_WEIGHT * rupee_gap_frac. Reward = progress on each term + success bonus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zelda_rl.obs import POS_SCALE, GameState


@dataclass
class GoalConfig:
    success_radius: float = 75.0      # world units (~Link is ~50 tall)
    rupee_weight: float = 1.0         # weight of the rupee term in goal_dist
    w_pos: float = 1.0                # reward per 100 units of progress toward the goal
    w_rupee: float = 1.0              # reward per rupee gained (only up to the goal count)
    success_bonus: float = 10.0
    time_penalty: float = 0.001
    scene_exit_penalty: float = 5.0
    w_flags: float = 1.0              # reward per goal flag bit newly satisfied (v2)
    rupees_first: bool = True         # while rupee_gap > 0: reward approaching the nearest rupee, not the goal spot
    w_rupee_approach: float = 1.0     # reward per 100 units closer to the nearest rupee (rupee phase)


@dataclass
class GoalMetrics:
    pos_dist: float                   # world units, 3D
    rupee_gap: int                    # rupees still missing vs goal (>= 0)
    goal_dist: float                  # combined embedding distance
    success: bool
    same_scene: bool
    flags_missing: int = 0            # goal save-flag bits not yet set (v2; 0 when unused)


def goal_metrics(s: GameState, goal: GameState, cfg: GoalConfig, start_flags=None) -> GoalMetrics:
    pos_dist = float(np.linalg.norm(s.xyz - goal.xyz))
    gap = max(goal.rupees - s.rupees, 0)
    same = s.scene == goal.scene
    missing = 0
    if start_flags is not None and getattr(s, "flags", None) is not None:
        from zelda_rl.obs import flag_progress
        missing = flag_progress(s.flags, goal.flags, start_flags)[0]
    gd = pos_dist / POS_SCALE + cfg.rupee_weight * gap / max(goal.rupees, 1) + missing / 8.0
    ok = same and pos_dist < cfg.success_radius and gap == 0 and missing == 0
    return GoalMetrics(pos_dist, gap, gd, ok, same, missing)


def nearest_rupee_dist(s: GameState) -> float:
    """3D distance from Link to the closest rupee actor (inf if none are loaded)."""
    if not s.rupee_items:
        return float("inf")
    return float(min(np.linalg.norm(np.asarray(p, np.float32) - s.xyz) for _, p in s.rupee_items))


def reward(prev: GameState, cur: GameState, goal: GameState, cfg: GoalConfig,
           start_flags=None) -> tuple[float, GoalMetrics, bool]:
    """Returns (reward, metrics, terminated)."""
    mp, mc = goal_metrics(prev, goal, cfg, start_flags), goal_metrics(cur, goal, cfg, start_flags)
    if not mc.same_scene:
        return -cfg.scene_exit_penalty, mc, True
    gained = min(cur.rupees, goal.rupees) - min(prev.rupees, goal.rupees)
    r = cfg.w_rupee * gained
    if cfg.rupees_first and mp.rupee_gap > 0:
        # Rupee phase: shape toward the nearest rupee. Skip the step a rupee is collected (the nearest one vanishes,
        # so the distance jumps) and the step the target count is reached (potential switches to the goal spot).
        if gained == 0:
            dp, dc = nearest_rupee_dist(prev), nearest_rupee_dist(cur)
            if np.isfinite(dp) and np.isfinite(dc):
                r += float(np.clip(cfg.w_rupee_approach * (dp - dc) / 100.0, -1.0, 1.0))
    elif mc.rupee_gap == 0 or not cfg.rupees_first:
        r += cfg.w_pos * (mp.pos_dist - mc.pos_dist) / 100.0
    r += cfg.w_flags * (mp.flags_missing - mc.flags_missing)
    r -= cfg.time_penalty
    if mc.success:
        r += cfg.success_bonus
    if not np.isfinite(r):          # e.g. no player pointer during a transition -> NaN distances
        r = 0.0
    return float(r), mc, mc.success
