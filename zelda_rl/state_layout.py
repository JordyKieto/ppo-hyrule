"""The flat game-state vector layout: one definition, shared by the collector and the world model.

`scripts/collect_transitions.py` writes these arrays and `zelda_rl/bc.py` reads them back as decoder
targets. Those two had the layout written out separately, and every loader bug so far came from the copies
drifting (the exit frames that trained on nothing, the predictor index that dropped them again). So the
field order lives here and both import it.

Two versions exist in the datasets:

  v1  link 15 + 16 rows x 20 = 335   the released deku-baba dataset and its exits
  v2  link 38 + 16 rows x 20 = 358   adds Link's body, collision and action state

v2 exists because the dream could not represent what an action *does*. The v1 state carries Link's position
and a few flags, so a jump, a roll, a sword swing and a Z-press with no target are nearly indistinguishable
in it -- which is one reason the action was barely recoverable from a latent transition (0.139 against a
0.220 ceiling). The added fields are the engine's own answer to "what is Link doing and what is he touching":
`action_fn_rel` (the action state machine), `melee_state` (a swing is live), `bg_flags` (ground / wall /
ceiling contact), `speed`, `vel`, `wall_yaw`, `floor_height`, `ztarget_timer` and `parallel_yaw`.

Offsets and provenance for every v2 field are in `zelda_rl.combat.LINK_FIELD_OFFSETS`; they come from the OoT
decompilation and were verified live by `scripts/verify_link_fields.py`. See docs/ADDING_OBSERVATIONS.md.

Read the width, not a constant: `link_dim(state.shape[-1])` tells you which version an array is.
"""
from __future__ import annotations

N_ROWS, ROW_DIM = 16, 20
MAX_ENEMIES = N_ROWS          # slots start with 9 cat-5 actors; headroom for mid-fight spawns

LINK_FIELDS_V1 = ["scene", "room", "hp", "hp_cap", "sf1", "pos_x", "pos_y", "pos_z", "yaw",
                  "targeted_row", "arrow_row", "held_item", "seeds", "nuts", "grass"]

# Link body / collision / action state. Verified live; see zelda_rl.combat.LINK_FIELD_OFFSETS.
LINK_FIELDS_V2_EXTRA = ["shape_yaw", "vel_x", "vel_y", "vel_z", "speed", "bg_flags", "wall_yaw",
                        "floor_height", "action_fn_rel", "sf2", "sf3", "melee_state", "melee_anim",
                        "ztarget_timer", "parallel_yaw", "fall_distance", "floor_pitch",
                        "dist_interact_wall", "y_dist_to_ledge", "door_type", "door_direction",
                        "door_timer", "knockback_type"]
LINK_FIELDS_V2 = LINK_FIELDS_V1 + LINK_FIELDS_V2_EXTRA

# v3 adds the CAMERA. A stick direction moves Link relative to the camera, so without it the world direction
# of a walk cannot be known -- which is what capped the transition head's direction accuracy at 0.532 over 16
# sectors and left the dream unable to approach a Baba. camera_yaw_sin_cos was in the original PPO's
# observation, so this is a parity restoration, not an addition.
LINK_FIELDS_V3_EXTRA = ["cam_yaw", "cam_pitch", "cam_eye_x", "cam_eye_y", "cam_eye_z"]
LINK_FIELDS_V3 = LINK_FIELDS_V2 + LINK_FIELDS_V3_EXTRA

ENEMY_FIELDS = ["present", "id", "hp", "pos_x", "pos_y", "pos_z", "fn_rel", "state_idx", "shape_yaw",
                "yaw_to_player", "timer", "cat", "params", "yaw", "vel_x", "vel_y", "vel_z",
                "focus_x", "focus_y", "focus_z"]
assert len(ENEMY_FIELDS) == ROW_DIM

STATE_DIM_V1 = len(LINK_FIELDS_V1) + N_ROWS * ROW_DIM      # 335
STATE_DIM_V2 = len(LINK_FIELDS_V2) + N_ROWS * ROW_DIM      # 358
STATE_DIM_V3 = len(LINK_FIELDS_V3) + N_ROWS * ROW_DIM      # 363
LINK_DIM_V1, LINK_DIM_V2 = len(LINK_FIELDS_V1), len(LINK_FIELDS_V2)
LINK_DIM_V3 = len(LINK_FIELDS_V3)
_KNOWN = (LINK_DIM_V1, LINK_DIM_V2, LINK_DIM_V3)


def link_dim(width: int) -> int:
    """Link-block width implied by a flat state vector's total width."""
    d = width - N_ROWS * ROW_DIM
    if d not in _KNOWN:
        raise ValueError(f"state width {width} implies link block {d}, expected one of {_KNOWN} "
                         f"(v1 / v2 / v3)")
    return d


def link_fields(width: int) -> list[str]:
    d = link_dim(width)
    return {LINK_DIM_V1: LINK_FIELDS_V1, LINK_DIM_V2: LINK_FIELDS_V2, LINK_DIM_V3: LINK_FIELDS_V3}[d]


def link_index(width: int) -> dict[str, int]:
    """field name -> column index in the Link block, for the version this width implies."""
    return {n: i for i, n in enumerate(link_fields(width))}


def has_v2(width: int) -> bool:
    return link_dim(width) >= LINK_DIM_V2


def has_v3(width: int) -> bool:
    return link_dim(width) >= LINK_DIM_V3
