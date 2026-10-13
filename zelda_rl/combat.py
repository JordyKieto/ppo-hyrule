"""Combat task: defeat the target enemies (Deku Babas in the Deku Tree's first room) without dying or leaving the room.

Rewards are per-step deltas of RAM state; episode outcomes are logged separately. Addresses verified on the user's
Deku Tree savestates and a live recording (runs/manual/trace_*.jsonl) unless marked heuristic:

  enemy HP            actor +0xAF        (Deku Baba 0x0055: 2 HP, one sword hit = 1)
  lock-on target      gPlayState+0x1CCC  (TargetContext.targetedActor; non-zero exactly while Z-locked on an actor)
  arrow target        gPlayState+0x1CC8  (actor Navi points at = lockable in range)
  room number         gPlayState+0x11CBC (reads room 0 in all slots; change-on-exit not yet observed -> combined w/ fall + scene)
  Link HP / dead      save ctx +0x30 / stateFlags1 bit 7 (both flip in the same frame at death)
  damaged / shielding stateFlags1 bit 26 / bit 22

Traps handled (both seen live): a death/reload despawns every enemy at once (not kills), and newly spawned actors
report HP 8 for one frame before initialising (not hits).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from zelda_rl.obs import PLAY_STATE, SAVE_CTX, POS_SCALE

TARGETED_ACTOR = PLAY_STATE + 0x1C24 + 0x6C + 0x3C
ARROW_ACTOR = PLAY_STATE + 0x1C24 + 0x6C + 0x38
ROOM_NUM = PLAY_STATE + 0x11CBC
DEKU_BABA = 0x0055
SF1_DEAD, SF1_ZTARGET, SF1_SHIELD, SF1_DAMAGED, SF1_CUTSCENE = 7, 15, 22, 26, 29


def _u32(r, a): return struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
def _u16(r, a): return struct.unpack_from(">H", r, a & 0x3FFFFF)[0]
def _s16(r, a): return struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
def _vec3(r, a): return struct.unpack_from(">fff", r, a & 0x3FFFFF)


@dataclass
class Enemy:
    ptr: int
    id: int
    hp: int
    pos: tuple[float, float, float]
    fn: int = 0                 # Deku Baba action routine (+0x1B0) -> behaviour state
    shape_yaw: int = 0          # +0xB6: where the head points
    yaw_to_player: int = 0      # +0x8A
    timer: int = 0              # +0x1B6: action countdown
    cat: int = 5                # actor category (5 enemy, 9 boss)
    params: int = 0             # +0x1C
    upd: int = 0                # +0x130 update function: states of relocatable overlays are decoded relative to it
    focus: tuple[float, float, float] = (0.0, 0.0, 0.0)   # +0x38 focus point (lock-on / aim point: Gohma's eye)
    yaw: int = 0                # +0x32 world rotation y (body facing)
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)   # +0x5C velocity per game update (verified live for Gohma, larvae)

    @property
    def state(self) -> str:
        return DEKU_BABA_STATE.get(self.fn, "other") if self.id == DEKU_BABA else "other"

    @property
    def fn_rel(self) -> int:
        """Action function relative to the actor's own update function: stable wherever the overlay is loaded."""
        return self.fn - self.upd if self.upd else 0


# Deku Baba action routines (+0x1B0), decoded from a live fight recording and present in all 10 savestates.
DEKU_BABA_STATE = {0x801EDF30: "hidden", 0x801EDFDC: "emerge_retract", 0x801EE444: "emerge_retract",
                   0x801EEE94: "idle", 0x801EE910: "idle", 0x801EEDC8: "prepare_lunge", 0x801EEB40: "lunge",
                   0x801EF220: "pull_back", 0x801EF318: "hit", 0x801EF428: "stunned", 0x801EF5EC: "dying",
                   0x801EF950: "dying"}
# Action-function field per (actor id, category); default = Deku Baba's +0x1B0. Found by scanning 20 Hz fight traces
# (runs/manual/boss_*.jsonl) for code-pointer fields with few distinct values (docs/ADDING_OBSERVATIONS.md step 2).
GRASS_ID = 0x0125                     # En_Kusa grass tuft (8 in Gohma's room; cut -> removed, may drop seeds/nuts/hearts)
GOHMA_ID = 0x0028                     # Boss_Goma (category 9) and its larvae/eggs/debris En_Goma share this id in RAM
ACTION_FN_OFFSET = {(GOHMA_ID, 9): 0x180, (GOHMA_ID, 5): 0x2A0}
BABA_STATES = ["hidden", "emerge_retract", "idle", "prepare_lunge", "lunge", "pull_back", "hit", "stunned", "dying", "other"]


@dataclass
class CombatState:
    scene: int
    room: int
    hp: int
    hp_cap: int
    sf1: int
    pos: tuple[float, float, float]
    yaw: int
    targeted: int
    arrow: int
    enemies: dict[int, Enemy]           # by actor pointer (ENEMY + BOSS categories)
    held_item: int = 0                  # Player +0x141 held item action (verified: 15 slingshot, 4 sword, 0 none)
    seeds: int = 0                      # slingshot seeds (save ctx ammo slot 6)
    nuts: int = 0                       # Deku nuts (ammo slot 1)
    grass: int = -1                     # grass tufts (En_Kusa 0x0125, prop list) still standing; -1 = not read
    # --- Link body / collision / action state. Offsets from the OoT decomp (see LINK_FIELD_OFFSETS).
    # Defaults are inert so a CombatState rebuilt from decoder output (zelda_rl/dream_state.py) still works.
    shape_yaw: int = 0                  # Actor +0xB6: rendered facing, differs from world yaw mid-turn
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed: float = 0.0                  # Actor +0x68 speed. ~0 when pressed into a wall, ~4.7 walking free
    bg_flags: int = 0                   # Actor +0x88 bgCheckFlags: GROUND / WALL / CEILING / WATER / CRUSHED
    wall_yaw: int = 0                   # Actor +0x7E: facing of the wall being touched
    floor_height: float = 0.0           # Actor +0x80: y of the floor poly below Link
    action_fn: int = 0                  # Player +0x664 actionFunc: Link's action state machine
    upd_fn: int = 0                     # Actor +0x130: Link's own update fn, to make action_fn relocation-proof
    sf2: int = 0                        # Player +0x670 stateFlags2
    sf3: int = 0                        # Player +0x682 stateFlags3
    melee_state: int = 0                # Player +0x833 meleeWeaponState: nonzero while a swing is live
    melee_anim: int = 0                 # Player +0x832 meleeWeaponAnimation
    ztarget_timer: int = 0              # Player +0x65C: Z-target held, nonzero even with NO lock-on target
    parallel_yaw: int = 0               # Player +0x82E: aim yaw when Z-held without a target
    fall_distance: int = 0              # Player +0x886: how far Link has fallen so far (positive = down)
    floor_pitch: int = 0                # Player +0x888: floor slope along current heading
    dist_interact_wall: float = 0.0     # Player +0x878: xyz distance to the interact wall
    y_dist_to_ledge: float = 0.0        # Player +0x874: height of ground above an interact wall
    door_type: int = 0                  # Player +0x41C..0x41E: door interaction (how a room is actually left)
    door_direction: int = 0
    door_timer: int = 0
    knockback_type: int = 0             # Player +0x891: set when Link is knocked back
    # Camera. NOT optional: a stick direction moves Link relative to the CAMERA, so without it the world
    # direction of a walk is unknowable. It was in the original PPO's observation (camera_yaw_sin_cos) but
    # never in the recorded state, and the audit traced the dream's failure to approach a Baba to exactly
    # this -- displacement direction top-1 stalled at 0.532 over 16 sectors with no camera input at all.
    cam_yaw: float = 0.0                # radians, atan2(dz, dx) of eye->at, same convention as obs.py
    cam_pitch: float = 0.0              # radians, elevation of eye->at
    cam_eye: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def action_fn_rel(self) -> int:
        """actionFunc relative to Link's own update fn, so the value survives a different load address."""
        return self.action_fn - self.upd_fn if self.upd_fn else 0

    def bg(self, b: int) -> bool:
        return bool(self.bg_flags >> b & 1)

    @property
    def dead(self) -> bool:
        return self.hp <= 0 or bool(self.sf1 >> SF1_DEAD & 1)

    def bit(self, b: int) -> bool:
        return bool(self.sf1 >> b & 1)


def read_combat_state(ram) -> CombatState:
    ps = PLAY_STATE
    player = _u32(ram, ps + 0x1C44)
    ok = 0x80000000 <= player < 0x80400000
    enemies = {}
    for cat in (5, 9):
        a, guard = _u32(ram, ps + 0x1C24 + 0x10 + cat * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 64:
            aid = _u16(ram, a)
            fn_off = ACTION_FN_OFFSET.get((aid, cat), 0x1B0)
            enemies[a] = Enemy(a, aid, int(ram[(a + 0xAF) & 0x3FFFFF]), _vec3(ram, a + 0x24),
                               fn=_u32(ram, a + fn_off), shape_yaw=_s16(ram, a + 0xB6),
                               yaw_to_player=_s16(ram, a + 0x8A), timer=_s16(ram, a + 0x1B6), cat=cat,
                               params=_u16(ram, a + 0x1C), upd=_u32(ram, a + 0x130), focus=_vec3(ram, a + 0x38),
                               yaw=_s16(ram, a + 0x32), vel=_vec3(ram, a + 0x5C))
            a = _u32(ram, a + 0x124)
            guard += 1
    return CombatState(
        scene=_s16(ram, ps + 0xA4), room=int(ram[ROOM_NUM & 0x3FFFFF]), hp=_s16(ram, SAVE_CTX + 0x30),
        hp_cap=_s16(ram, SAVE_CTX + 0x2E), sf1=_u32(ram, player + 0x66C) if ok else 0,
        pos=_vec3(ram, player + 0x24) if ok else (math.nan,) * 3, yaw=_u16(ram, player + 0x32) if ok else 0,
        targeted=_u32(ram, TARGETED_ACTOR), arrow=_u32(ram, ARROW_ACTOR), enemies=enemies,
        held_item=struct.unpack_from(">b", ram, (player + 0x141) & 0x3FFFFF)[0] if ok else 0,
        seeds=int(ram[(SAVE_CTX + 0x8C + 6) & 0x3FFFFF]), nuts=int(ram[(SAVE_CTX + 0x8C + 1) & 0x3FFFFF]),
        grass=_count_actors(ram, 6, GRASS_ID),
        **(_read_link_body(ram, player) if ok else {}), **_read_camera(ram),
    )


# Link body / collision / action state. Offsets come from the OoT decompilation, NOT from recall:
# `Actor` from include/actor.h and `Player` from include/player.h in github.com/zeldaret/oot.
#
# The decomp prints offsets for a DEBUG_FEATURES build, where `Actor` ends with `char dbgPad[0x10]` and so
# `size = 0x14C`. Retail NTSC 1.2 has no pad, so `Actor` is 0x13C and every *Player* field sits 0x10 lower
# than the decomp says; Actor's own fields are unshifted. The two offsets that predate this table agree:
# stateFlags1 0x67C-0x10 = 0x66C (sf1 above) and heldItemAction 0x151-0x10 = 0x141 (held_item above).
#
# Verified live by scripts/verify_link_fields.py over 2,510 frames (16/16 checks), the decisive one being a
# redundant pair: BGCHECKFLAG_WALL (bit 3 of bgCheckFlags) == (wallPoly != 0) at 1.0000 agreement.
# See docs/ADDING_OBSERVATIONS.md step 0.
PLAYER_SHIFT = 0x10
LINK_FIELD_OFFSETS = {                      # name: (offset, struct format)
    "speed": (0x068, ">f"), "wall_yaw": (0x07E, ">h"), "floor_height": (0x080, ">f"),
    "bg_flags": (0x088, ">H"), "shape_yaw": (0x0B6, ">h"), "upd_fn": (0x130, ">I"),
    "door_type": (0x42C - PLAYER_SHIFT, ">b"), "door_direction": (0x42D - PLAYER_SHIFT, ">b"),
    "door_timer": (0x42E - PLAYER_SHIFT, ">h"), "ztarget_timer": (0x66C - PLAYER_SHIFT, ">i"),
    "action_fn": (0x674 - PLAYER_SHIFT, ">I"), "sf2": (0x680 - PLAYER_SHIFT, ">I"),
    "sf3": (0x692 - PLAYER_SHIFT, ">B"), "parallel_yaw": (0x83E - PLAYER_SHIFT, ">h"),
    "melee_anim": (0x842 - PLAYER_SHIFT, ">b"), "melee_state": (0x843 - PLAYER_SHIFT, ">b"),
    "y_dist_to_ledge": (0x884 - PLAYER_SHIFT, ">f"), "dist_interact_wall": (0x888 - PLAYER_SHIFT, ">f"),
    "fall_distance": (0x896 - PLAYER_SHIFT, ">h"), "floor_pitch": (0x898 - PLAYER_SHIFT, ">h"),
    "knockback_type": (0x8A1 - PLAYER_SHIFT, ">B"),
}
# bgCheckFlags bit -> name (include/actor.h BGCHECKFLAG_*)
BG_GROUND, BG_GROUND_TOUCH, BG_GROUND_LEAVE, BG_WALL, BG_CEILING = 0, 1, 2, 3, 4
BG_WATER, BG_WATER_TOUCH, BG_GROUND_STRICT, BG_CRUSHED, BG_PLAYER_WALL = 5, 6, 7, 8, 9
BG_BITS = [BG_GROUND, BG_GROUND_TOUCH, BG_GROUND_LEAVE, BG_WALL, BG_CEILING,
           BG_WATER, BG_GROUND_STRICT, BG_CRUSHED, BG_PLAYER_WALL]
# Bits worth DECODING. GROUND_TOUCH and GROUND_LEAVE are commented "only active for 1 frame" in the decomp,
# and the env samples every 6 VI frames (10 Hz), so ~5 of 6 occurrences are not in the recorded state at all
# -- GROUND_TOUCH scored recall 0.02. Per docs/ADDING_OBSERVATIONS.md ("a flag that flickers for 1-2 frames
# is invisible at 10 Hz; derive the event from a slower signal instead", which is how blocks were recovered
# from lunge -> pull_back cycles), landing is derived from the airborne transition instead: see `landed()`.
# GROUND_STRICT is dropped as well: it tracks GROUND almost exactly, so it adds a head and no information.
BG_DECODE_BITS = [BG_GROUND, BG_WALL, BG_CEILING, BG_WATER, BG_CRUSHED, BG_PLAYER_WALL]


def landed(prev_airborne, cur_airborne):
    """Touched down this step: was off the ground, now on it.

    Replaces decoding BGCHECKFLAG_GROUND_TOUCH. `airborne` (the GROUND bit) decodes at 1.000 accuracy /
    1.000 recall, so a transition on it is reliable where the 1-frame flag is not.
    """
    return (prev_airborne > 0.5) & (cur_airborne <= 0.5)


def left_ground(prev_airborne, cur_airborne):
    """Took off this step. Replaces BGCHECKFLAG_GROUND_LEAVE, same reasoning as `landed`."""
    return (prev_airborne <= 0.5) & (cur_airborne > 0.5)


def _read_link_body(ram, player: int) -> dict:
    d = {k: struct.unpack_from(f, ram, (player + o) & 0x3FFFFF)[0]
         for k, (o, f) in LINK_FIELD_OFFSETS.items()}
    d["vel"] = _vec3(ram, player + 0x5C)
    return d


def _read_camera(ram) -> dict:
    """Main camera, same offsets obs.read_state_v2 uses (MAIN_CAMERA +0x50 `at`, +0x5C `eye`)."""
    from zelda_rl.obs import MAIN_CAMERA
    at, eye = _vec3(ram, MAIN_CAMERA + 0x50), _vec3(ram, MAIN_CAMERA + 0x5C)
    dx, dy, dz = at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]
    return {"cam_yaw": math.atan2(dz, dx),
            "cam_pitch": math.atan2(dy, math.hypot(dx, dz)),
            "cam_eye": eye}


def _count_actors(ram, cat: int, actor_id: int) -> int:
    a, n, guard = _u32(ram, PLAY_STATE + 0x1C24 + 0x10 + cat * 8), 0, 0
    while 0x80000000 <= a < 0x80400000 and guard < 128:
        n += _u16(ram, a) == actor_id
        a, guard = _u32(ram, a + 0x124), guard + 1
    return n


@dataclass
class CombatConfig:
    profile: str = "deku_baba"          # "deku_baba" (CombatTracker) | "gohma" (zelda_rl.gohma.GohmaTracker)
    target_ids: tuple[int, ...] = (DEKU_BABA,)
    max_seconds: float = 180.0
    # outcomes
    w_enemy_damage: float = 1.0        # per HP point removed from a target
    w_defeat: float = 5.0              # per target defeated (x speed multiplier below)
    speed_mult_max: float = 2.0        # kill within ~0 s of the previous kill/start: x(1 + 2) = 15; after speed_ref_s: x1
    speed_ref_s: float = 30.0
    steps_per_second: float = 10.0
    success_bonus: float = 20.0        # all targets defeated
    success_time_bonus: float = 15.0   # x fraction of time remaining at success (clearing fast pays)
    w_link_damage: float = 0.25        # per Link HP unit lost (16 = 1 heart -> -2 per half heart)
    w_link_heal: float = 0.05          # per HP unit regained (recovery hearts)
    death_penalty: float = 20.0
    leave_room_penalty: float = 20.0   # terminal like death: must never be an escape hatch from a losing fight
    timeout_penalty_per_target: float = 3.0   # at the time cap, per target still alive (no reward for stalling)
    # behaviour shaping (all capped per episode so they can't be farmed)
    w_lockon: float = 0.3              # acquiring a Z lock on a live target
    lockon_cap: int = 6
    w_lock_step: float = 0.005         # per step locked on a live target
    lock_step_cap: float = 1.0
    w_approach: float = 0.5            # per 100 units closer to the nearest live target (when not locked)
    w_shield_face: float = 0.01        # per step shielding while facing a live target within shield_radius
    shield_cap: float = 0.5
    w_block: float = 1.0               # lunge cycle (lunge->pull_back) survived while shielding + facing + in range
    block_cap: int = 8
    w_dodge: float = 0.2               # lunge cycle survived in range without shielding
    dodge_cap: int = 5
    w_stun_hit: float = 0.5            # extra per damage dealt while the Baba is stunned / recoiling
    block_range: float = 220.0         # max base distance during a lunge for block/dodge credit
    shield_radius: float = 150.0
    lunge_radius: float = 110.0
    time_penalty: float = 0.001
    # episode
    fall_drop: float = 150.0           # Link this far below his start height = fell through the floor web
    room_radius: float = 1400.0        # Link this far (x/z) from his start = left the room area
    spawn_grace_steps: int = 2         # ignore HP changes of actors younger than this (spawn HP=8 frame)
    # ---- Gohma profile (zelda_rl/gohma.py); ignored by the Deku Baba tracker
    g_boss_damage: float = 4.0         # per Gohma HP (10 HP -> 40)
    g_boss_defeat: float = 60.0        # outsized: x speed multiplier (1 + speed_mult_max * (1 - minutes/g_speed_ref_min))
    g_speed_ref_min: float = 3.0
    g_stun_hit: float = 2.0            # extra per HP dealt while she is knocked down / stunned
    g_boss_hp_progress: float = 2.0    # boss_damage x (1 + this * fraction of her HP gone after the hit): last hits pay 3x
    g_seed_stun: float = 15.0          # a red-eye seed that actually reaches the eye: THE key skill (long sword window)
    g_seed_stun_cap: float = 60.0
    g_red_eye_seed: float = 4.0        # every seed fired while the eye is red (hit or not): firing clearly beats waiting
    g_red_eye_seed_cap: float = 12.0
    g_unfired_red_eye: float = 1.5     # penalty: a red-eye window closes with the slingshot cocked but no seed fired
    g_sling_ready: float = 1.0         # ONCE per red-eye window (pre-red + red span): slingshot cocked and aimed at her
    g_sling_ready_cap: float = 6.0     #   (not per step: waiting can't be milked)
    g_drop_red_eye_shot: float = 1.0   # penalty: slingshot put away (no seed fired) while her eye is red
    g_elevated_seed: float = 0.5       # per seed fired at her (aimed / locked) right before or during the red eye
    g_elevated_seed_cap: float = 5.0
    g_nut_stun: float = 2.0            # Deku Nut stun: works in any floor state but only ~2 s (weaker than the seed)
    g_nut_stun_cap: float = 10.0
    g_nut_window_steps: int = 6        # stun within 0.6 s of a nut throw (measured 0.22 s)
    g_nut_miss: float = 0.3            # per nut that stuns nothing, x (20 + 4) / (nuts left + 4): 0.3 at 20 nuts ... 1.8 at 0
    g_nut_scarcity: float = 20.0
    g_seed_window_steps: int = 10      # stun within this many steps of a seed counts as the seed's hit
    g_seed_miss: float = 0.15          # per seed that hits nothing (no stun, no larva damage within the window), scaled by
    g_seed_scarcity: float = 30.0      #   scarcity: x (30 + 5) / (seeds left + 5) -> 0.15 at 30 seeds ... 1.05 at 0
    g_seed_window_hit_steps: int = 12  # a seed resolves as a hit if it stuns Gohma / damages a larva within 1.2 s
    g_lock_red_eye: float = 0.05       # per step locked on (or facing the eye) while the eye is red
    g_lock_red_eye_cap: float = 5.0
    g_sling_red_eye: float = 0.3       # per step with the slingshot out and aimed (Z-lock or free aim) at the RED eye,
    g_sling_red_eye_cap: float = 8.0   #   wherever she is (ceiling or floor)
    g_sword_range: float = 90.0
    g_sword_ready: float = 0.02        # per step sword out within sword range while she is stunned
    g_sword_ready_cap: float = 1.0
    g_larva_damage: float = 0.5        # larvae (sword): uncapped (Gohma reopens her eye only after they die)
    g_larva_kill: float = 2.0
    g_larva_damage_sling: float = 0.25 # larvae taken out with the slingshot pay less than the sword
    g_larva_kill_sling: float = 1.0
    g_link_heal: float = 0.1           # per HP unit healed (recovery hearts: +0.8 per heart picked up)
    g_grass_cut: float = 0.2           # per grass tuft cut (resource gathering: drops seeds / nuts / hearts)
    g_grass_cut_cap: float = 2.0
    g_seed_pickup: float = 0.05        # per seed picked up, x (30 + 5) / (seeds before + 5): worth more when running low
    g_nut_pickup: float = 0.15         # per nut picked up, x (20 + 4) / (nuts before + 4)
    g_pickup_cap: float = 6.0
    g_death_penalty: float = 30.0
    g_block_range: float = 250.0
    g_spacing: float = 0.0             # per step in the BAND [g_spacing_range, g_spacing_far] while she is in
    g_spacing_cap: float = 1.5         #   an attack state. A band, not a floor: a floor paid a policy that
    g_spacing_range: float = 150.0     #   stood off all episode doing nothing exactly as much as it paid the
    g_spacing_far: float = 600.0       #   human win -- observed live, a smoke-run episode on slot 6 banked the
    #   whole cap over 3,000 attack steps with zero boss damage. The band is the WIN's own distribution during
    #   her attack frames: median 311, p10 162, p90 533. The cap is deliberately SMALL (1.5 against
    #   boss_defeat's 60 and death's -30) because this is a nudge on positioning, not a strategy -- the signal
    #   that stalling is bad already exists in timeout + time_penalty, worth about -13.   This is the defence
    #   the recordings actually show. Measured over her
    #   attack frames only: the WIN held a median 311 units and was inside
    #   150 for 8.7% of them, losing half a heart; the DEATH held a median 131 and was inside 150 for 61.8%,
    #   losing three hearts and dying -- same player, same boss. 150 separates them by 7x, which is why it is
    #   the threshold. It pairs with approach_stunned / sword_ready_stunned, which pay for closing while she is
    #   DOWN: the skill is phase-dependent spacing, not distance for its own sake, and the cap (reached only by
    #   spacing correctly through essentially every attack step) stops it becoming a reason to never engage.
    g_guard: float = 0.0               # per step LOCKED ON Gohma with the shield up WHILE SHE IS ATTACKING.
    g_guard_cap: float = 6.0           #   The simplest defensive decomposition there is, and it was not even
    #   expressible until ("L", "R") was added to the action space: the policy could lock on OR shield, never
    #   both. Gated on GOHMA_ATTACK so it pays for defence rather than for standing still. Default 0.0 keeps
    #   the `full` baseline reproducible; the `simple` preset turns it on.
    g_approach: float = 0.0            # per 100 units of NET progress toward Gohma, outside her attack frames.
    g_approach_cap: float = 7.0        #   The gap the 10-slot eval exposed: GohmaTracker.step fully replaces
    #   CombatTracker.step and never calls super(), so the generic `approach` term (w_approach, paid in
    #   CombatTracker) has NEVER fired on Gohma. The only distance term that ran was `approach_stunned`, gated
    #   on GOHMA_STUNNED -- it pays for closing the last few units AFTER she is already down, and pays nothing
    #   for crossing the room to reach her. Measured on 80 episodes at v14 update 20: 18.75% defeat from under
    #   200 units, 4.17% at 200-400, 2.50% beyond 400. The policy can fight her and cannot reach her.
    #
    #   Paid on NET progress (closer than the closest it has been this episode), not on the per-step delta.
    #   A signed per-step delta is farmable by oscillating in and out, and `_capped` ignores negative values
    #   so the retreat half would have been free -- the exact shape of the g_sling_ready and g_spacing farming
    #   this project has already paid for twice. Net progress is monotone: the total is bounded by the start
    #   distance regardless of the path, so there is nothing to farm.
    #
    #   NOT paid during GOHMA_ATTACK, where g_spacing pays for holding the 150-600 band. The two terms would
    #   otherwise bid against each other on the same step. Phase separation is the existing design intent
    #   ("close when she is not swinging, hold off when she is"), not a new idea.
    #
    #   Scale: the furthest slot starts 541 units out and sword range is 90, so the most useful net progress
    #   any episode can bank is ~451 units. At 1.5 that is +6.77, just under the 7.0 cap -- deliberately, so
    #   the gradient survives all the way to contact. Against boss_defeat's +60 and boss_damage's +4/HP.
    #
    #   v15 ran 0.5 / cap 3.0 and measured `reward/approach` at mean +0.609 per episode over 18 ranks, i.e.
    #   the policy banked ~120 of those 451 units while the recorded human win banks the whole cap. The term
    #   was active and the behaviour it pays for was ~20% realised, so v16 raises the coefficient 3x. The CAP
    #   had to move with it: at 1.5 against a 3.0 cap the cap binds after 200 units, which would zero the
    #   gradient over exactly the 200-450 unit range the term exists to reward. Coefficient and cap are one
    #   knob -- the term's strength -- not two variables.
    #
    #   Default 0.0 keeps every earlier run reproducible.
    g_stun_on_convert: bool = False    # pay seed_stun / nut_stun only when that stun is CONVERTED into damage.
    #   Default False keeps the measured baseline reproducible; the `simple` preset turns it on. Paying at the
    #   stun TRANSITION rewards the stun itself, so a policy can farm stuns it never cashes in: measured
    #   40,353 nut stuns over 15,560 episodes while Gohma lost a mean 2.06 of 10 HP.



@dataclass
class Track:
    target: bool
    hp: int
    age: int = 0
    defeated: bool = False


@dataclass
class CombatTracker:
    cfg: CombatConfig
    tracks: dict[int, Track] = field(default_factory=dict)

    def reset(self, s: CombatState, max_steps: int) -> None:
        self.max_steps = max_steps
        self.t = 0
        self.start_pos = s.pos
        self.start_scene, self.start_room = s.scene, s.room
        self.tracks = {p: Track(target=e.id in self.cfg.target_ids and e.hp > 0, hp=e.hp, age=self.cfg.spawn_grace_steps)
                       for p, e in s.enemies.items()}
        self.targets_start = sum(t.target for t in self.tracks.values())
        self.ep = {"targets_start": self.targets_start, "defeated": 0, "damage_dealt": 0, "hp_start": s.hp, "hp_lost": 0,
                   "hp_healed": 0, "died": False, "left_room": False, "success": False, "lockons": 0, "lock_steps": 0,
                   "shield_steps": 0, "blocks": 0, "time_to_clear_s": None, "return": 0.0, "length": 0,
                   "r_outcome": 0.0, "r_shaping": 0.0, "end": "timeout"}
        self._lock_r = self._shield_r = 0.0
        self._last_block = -10 ** 9
        self.last_kill_t, self.last_hit_t, self.last_damage_t = 0, None, None     # episode memory
        self.ep.update(kill_times_s=[], kill_mults=[], hp_end=s.hp, r_parts={}, dodges=0, lunges=0, lunge_hits_taken=0,
                       stun_hits=0)
        self.cycles: dict[int, dict] = {}         # target ptr -> open lunge cycle

    # -- helpers --
    def live_targets(self, s: CombatState) -> list[Enemy]:
        return [s.enemies[p] for p, t in self.tracks.items() if t.target and not t.defeated and p in s.enemies]

    @staticmethod
    def _facing(s: CombatState, e: Enemy) -> float:
        """cos(angle) between Link's facing and the direction to the enemy (x/z). OoT yaw 0 = +z, 0x4000 = +x."""
        yaw = s.yaw / 65536.0 * 2 * math.pi
        fx, fz = math.sin(yaw), math.cos(yaw)
        dx, dz = e.pos[0] - s.pos[0], e.pos[2] - s.pos[2]
        n = math.hypot(dx, dz) or 1.0
        return (fx * dx + fz * dz) / n

    def _left_reason(self, cur: CombatState) -> str | None:
        c = self.cfg
        if cur.scene != self.start_scene:
            return f"scene {self.start_scene}->{cur.scene}"
        if cur.room != self.start_room:
            return f"room {self.start_room}->{cur.room}"
        if math.isfinite(cur.pos[1]) and cur.pos[1] < self.start_pos[1] - c.fall_drop:
            return f"fell y {self.start_pos[1]:.0f}->{cur.pos[1]:.0f}"
        if math.hypot(cur.pos[0] - self.start_pos[0], cur.pos[2] - self.start_pos[2]) > c.room_radius:
            return "radius"
        return None

    def nearest_target(self, s: CombatState):
        live = self.live_targets(s)
        return min(live, key=lambda e: math.dist(e.pos, s.pos)) if live else None

    # -- the reward --
    def step(self, prev: CombatState, cur: CombatState) -> tuple[float, bool, dict]:
        c, ep = self.cfg, self.ep
        self.t += 1
        ep["length"] = self.t
        parts = ep["r_parts"]
        acc = {"out": 0.0, "shape": 0.0}

        def pay(name: str, value: float, kind: str) -> None:
            """Every reward term goes through here -> per-component episode totals (wandb reward/<name>)."""
            parts[name] = parts.get(name, 0.0) + value
            acc[kind] += value

        ev: dict = {}
        alive_now = not cur.dead
        same_place = cur.scene == self.start_scene

        # enemy tracking: hits / defeats (only while Link is alive and in the room; ignore spawn frames)
        for p, e in cur.enemies.items():
            tr = self.tracks.get(p)
            if tr is None:
                self.tracks[p] = Track(target=False, hp=e.hp)        # mid-episode spawn: observe, never a target
                continue
            tr.age += 1
            if tr.age <= c.spawn_grace_steps:
                tr.hp = e.hp
                continue
            if e.hp < tr.hp and alive_now and same_place and tr.target and not tr.defeated:
                dmg = tr.hp - e.hp
                pay("enemy_damage", c.w_enemy_damage * dmg, "out")
                if prev.enemies.get(p) is not None and prev.enemies[p].state in ("stunned", "hit"):
                    pay("stun_hit", c.w_stun_hit * dmg, "shape")
                    ep["stun_hits"] += 1
                    ev["stun_hit"] = True
                ep["damage_dealt"] += dmg
                ev["hit"] = ev.get("hit", 0) + dmg
                self.last_hit_t = self.t
                if e.hp == 0:
                    tr.defeated = True
                    dt_s = (self.t - self.last_kill_t) / c.steps_per_second
                    mult = 1.0 + c.speed_mult_max * max(0.0, 1.0 - dt_s / c.speed_ref_s)
                    pay("kill", c.w_defeat, "out")
                    pay("kill_speed_bonus", c.w_defeat * (mult - 1.0), "out")
                    self.last_kill_t = self.t
                    ep["defeated"] += 1
                    ep["kill_times_s"].append(round(self.t / c.steps_per_second, 1))
                    ep["kill_mults"].append(round(mult, 3))
                    ev["defeat"] = ev.get("defeat", 0) + 1
                    ev["kill_mult"] = round(mult, 2)
            tr.hp = e.hp
        for p in [p for p in self.tracks if p not in cur.enemies]:
            tr = self.tracks.pop(p)
            if tr.target and not tr.defeated:
                ev["target_lost"] = ev.get("target_lost", 0) + 1       # despawned without HP 0: not a kill

        # Link health
        ep["hp_end"] = max(cur.hp, 0)
        dhp = cur.hp - prev.hp
        if dhp < 0:
            pay("link_damage", -c.w_link_damage * (-dhp), "out")
            ep["hp_lost"] += -dhp
            self.last_damage_t = self.t
            ev["link_damage"] = -dhp
        elif dhp > 0 and not prev.dead:
            pay("link_heal", c.w_link_heal * dhp, "out")
            ep["hp_healed"] += dhp

        terminated = False
        if cur.dead:
            pay("death", -c.death_penalty, "out")
            ep["died"], ep["end"], terminated = True, "death", True
        elif (reason := self._left_reason(cur)):
            pay("left_room", -c.leave_room_penalty, "out")
            ep["left_room"], ep["end"], terminated = True, "left_room", True
            ep["left_reason"] = reason
        elif self.targets_start > 0 and ep["defeated"] >= self.targets_start:
            pay("success", c.success_bonus, "out")
            pay("success_time_bonus", c.success_time_bonus * max(0.0, 1 - self.t / self.max_steps), "out")
            ep["success"], ep["end"], terminated = True, "success", True
            ep["time_to_clear_s"] = self.t
        elif self.t >= self.max_steps:
            alive = len(self.live_targets(cur))
            pay("timeout", -c.timeout_penalty_per_target * alive, "out")
            ep["end"], ep["targets_alive_at_timeout"] = "timeout", alive     # truncation (the env bootstraps value)

        if not terminated:
            live = {e.ptr for e in self.live_targets(cur)}
            # Z-target on a live target
            if cur.targeted in live:
                ep["lock_steps"] += 1
                if prev.targeted not in live and ep["lockons"] < c.lockon_cap:
                    pay("lockon", c.w_lockon, "shape")
                    ep["lockons"] += 1
                    ev["lockon"] = True
                if self._lock_r < c.lock_step_cap:
                    pay("lock_hold", c.w_lock_step, "shape")
                    self._lock_r += c.w_lock_step
            else:
                nt_prev, nt_cur = self.nearest_target(prev), self.nearest_target(cur)
                if nt_prev is not None and nt_cur is not None and nt_prev.ptr == nt_cur.ptr:
                    d0, d1 = math.dist(nt_prev.pos, prev.pos), math.dist(nt_cur.pos, cur.pos)
                    if math.isfinite(d0) and math.isfinite(d1):
                        pay("approach", float(np.clip(c.w_approach * (d0 - d1) / 100.0, -0.5, 0.5)), "shape")
            # shield: facing a live target in range
            nt = self.nearest_target(cur)
            if cur.bit(SF1_SHIELD) and nt is not None:
                if self._facing(cur, nt) > 0.7 and math.dist(nt.pos, cur.pos) < c.shield_radius:
                    ep["shield_steps"] += 1
                    if self._shield_r < c.shield_cap:
                        pay("shield_facing", c.w_shield_face, "shape")
                        self._shield_r += c.w_shield_face
            # lunge cycles (decoded Baba states): block / dodge / hit-by-lunge
            for e in self.live_targets(cur):
                in_attack = e.state in ("lunge", "pull_back")
                cyc = self.cycles.get(e.ptr)
                if in_attack and cyc is None and e.state == "lunge":
                    cyc = self.cycles[e.ptr] = {"shield": False, "dmg": 0, "dmin": math.inf}
                    ep["lunges"] += 1
                if cyc is not None:
                    d = math.dist(e.pos, cur.pos)
                    cyc["dmin"] = min(cyc["dmin"], d) if math.isfinite(d) else cyc["dmin"]
                    cyc["shield"] |= cur.bit(SF1_SHIELD) and self._facing(cur, e) > 0.5
                    cyc["dmg"] += max(0, -dhp)
                    if not in_attack:                       # cycle over
                        del self.cycles[e.ptr]
                        if cyc["dmg"] > 0:
                            ep["lunge_hits_taken"] += 1
                            ev["lunge_hit"] = True
                        elif cyc["dmin"] < c.block_range and cyc["shield"] and ep["blocks"] < c.block_cap:
                            pay("block", c.w_block, "shape")
                            ep["blocks"] += 1
                            ev["block"] = True
                        elif cyc["dmin"] < c.block_range * 0.75 and not cyc["shield"] and ep["dodges"] < c.dodge_cap:
                            pay("dodge", c.w_dodge, "shape")
                            ep["dodges"] += 1
                            ev["dodge"] = True
        pay("time_penalty", -c.time_penalty, "shape")
        r = acc["out"] + acc["shape"]
        ep["return"] += r
        ep["r_outcome"] += acc["out"]
        ep["r_shaping"] += acc["shape"]
        return float(r), terminated, ev


def make_tracker(cfg: "CombatConfig"):
    if cfg.profile == "gohma":
        from zelda_rl.gohma import GohmaTracker
        return GohmaTracker(cfg)
    return CombatTracker(cfg)


# ---------------------------------------------------------------------------------------------------------
# GOHMA_SIMPLE: the reward with the entrance-era shaping removed.
#
# The slingshot-shaping cluster (sling_ready, unfired_red_eye, drop_red_eye_shot, elevated_seed,
# slingshot_on_red_eye, red_eye_seed) was added across gohma-v5..v7 to stop the ENTRANCE slots being farmed
# -- v5 measured the policy milking `sling_ready` for ~150 steps without ever starting the fight -- and to
# unstick slot 6, which waited cocked for ~1,400 steps. With the pre-fight slots gone from the start set,
# those terms patch a problem that no longer exists. On slots 1/2/10 they now sum to +0.011 per episode.
#
# Worse, measurement over 10,929 engaged episodes of gohma-v10 showed the cluster had become the dominant
# reward source: shaping was 88% of reward magnitude, `red_eye_seed` paid +8.34/episode against
# `boss_defeat`'s +1.98, and the skill it pays for appears in none of the 360 wins (seed_stuns 0.05 per
# defeat, against nut_stuns 2.53 and stun_hits 5.23). The policy optimised it correctly -- red-eye aim
# reached 3.36x chance and kept rising -- and still died 64.5% of the time with her at 8 of 10 HP.
#
# So this preset keeps only the chain the wins actually used: stun -> sword while stunned -> HP -> defeat,
# and it gates the stun credit on conversion (g_stun_on_convert) so stunning without cashing in pays zero.
# Every OUTCOME term is untouched: boss_damage, hp progress, defeat, speed bonus, success, larva, link
# damage, death, timeout.
GOHMA_SIMPLE: dict[str, float | bool] = {
    # the entrance-era slingshot cluster: gone
    "g_red_eye_seed": 0.0, "g_red_eye_seed_cap": 0.0,
    "g_sling_red_eye": 0.0, "g_sling_red_eye_cap": 0.0,
    "g_sling_ready": 0.0, "g_sling_ready_cap": 0.0,
    "g_unfired_red_eye": 0.0,
    "g_drop_red_eye_shot": 0.0,
    "g_elevated_seed": 0.0, "g_elevated_seed_cap": 0.0,
    "g_lock_red_eye": 0.0, "g_lock_red_eye_cap": 0.0,
    # distractions from a boss fight
    "g_grass_cut": 0.0, "g_grass_cut_cap": 0.0,
    "g_seed_pickup": 0.0, "g_nut_pickup": 0.0, "g_pickup_cap": 0.0,
    "w_lockon": 0.0, "w_lock_step": 0.0,
    # stun credit survives at full value, but only when it is cashed in
    "g_stun_on_convert": True,
    # the defensive channel. g_guard stays OFF: 17,182 frames of human play set the shield bit 0.00% of the
    # time, so paying for "locked on and shielding while she attacks" rewards standing inside the range the
    # winning run stayed out of. The counters stay on, so if the policy discovers the shield anyway we see it.
    "g_guard": 0.0,
    "g_spacing": 0.02,
    # the approach channel, new in v15 at 0.5 / cap 3.0, raised 3x for v16 after v15 realised only ~20% of
    # what it pays for. See g_approach in CombatConfig for why it was absent until v15 at all.
    "g_approach": 1.5,
}


# Start-slot engagement distance, in world units, Link to Gohma at the savestate.
GOHMA_SLOT_DIST = {1: 473, 2: 368, 3: 188, 4: 541, 5: 472, 6: 357, 7: 38, 8: 448, 9: 394, 10: 490}
GOHMA_BANDS = (("near", 0, 200), ("mid", 200, 400), ("far", 400, 10 ** 9))


def band_metrics(eps: list[dict]) -> dict[str, float]:
    """Banded and MIX-FREE outcome metrics for a rollout's episodes.

    `combat/success_rate` is episode-weighted, so it moves when the CURRICULUM moves and cannot be compared
    across versions with different slot weights. That has misled every cross-version decision on this task:
    v11's +0.270/update "momentum", which is why v12 was launched, was slot 7 alone (7.32% -> 14.00%)
    amplified by that run's 50% weighting on slot 7, while all nine other slots sat at exactly 0.00% for the
    whole run.

    - `success_balanced` is the unweighted mean of the per-slot rates. Changing the mix cannot move it, so a
      version-to-version claim belongs on this number. Asserted mix-invariant in tests/test_band_metrics.py.
    - `success_hard` is the same with the near band dropped. The near band is slot 7 at 38 units, where Link
      starts on top of her, and it has carried every apparent gain this project has reported.
    - `slots_nonzero` is how many start slots the policy wins from AT ALL -- the measure on which v11 scored
      1 of 10 and v15 scores 6 of 6.

    One implementation, called by the trainer and by the test, so the two cannot drift (AGENTS 25).
    """
    import numpy as _np
    out: dict[str, float] = {}
    if not eps:
        return out
    slots = sorted({e["slot"] for e in eps})
    rate = {k: float(_np.mean([bool(e.get("success")) for e in eps if e["slot"] == k])) for k in slots}
    for name, lo, hi in GOHMA_BANDS:
        bs = [e for e in eps if lo <= GOHMA_SLOT_DIST.get(e["slot"], -1) < hi]
        if bs:
            out[f"by_band/{name}/success_rate"] = float(_np.mean([bool(e.get("success")) for e in bs]))
            out[f"by_band/{name}/death_rate"] = float(_np.mean([bool(e.get("died")) for e in bs]))
            out[f"by_band/{name}/boss_damage"] = float(_np.mean([e.get("boss_damage", 0) for e in bs]))
            out[f"by_band/{name}/n"] = float(len(bs))
    full = [e for e in eps if e.get("boss_hp_start", 0) >= 10]
    if full:
        out["by_hp/full/success_rate"] = float(_np.mean([bool(e.get("success")) for e in full]))
        out["by_hp/full/n"] = float(len(full))
    out["combat/success_balanced"] = float(_np.mean(list(rate.values())))
    hard = [v for k, v in rate.items() if GOHMA_SLOT_DIST.get(k, 0) >= 200]
    if hard:
        out["combat/success_hard"] = float(_np.mean(hard))
    out["combat/slots_nonzero"] = float(sum(1 for v in rate.values() if v > 0))
    out["combat/slots_seen"] = float(len(rate))
    return out

COMBAT_FIELDS = [("link_hp_frac", 1), ("link_hp_cap", 1), ("targets_alive_frac", 1), ("targets_start", 1),
                 ("locked_on_target", 1), ("arrow_on_target", 1), ("shielding", 1), ("damaged", 1),
                 ("nearest_target_rel_pos", 3), ("nearest_target_dist", 1), ("nearest_target_hp", 1),
                 ("nearest_target_facing_cos_sin", 2), ("second_target_rel_pos", 3), ("second_target_dist", 1),
                 ("second_target_hp", 1), ("time_left", 1),
                 # episode memory (deterministic facts; the GRU policy adds learned memory on top)
                 ("mem_kills_frac", 1), ("mem_damage_dealt", 1), ("mem_hearts_lost", 1), ("mem_since_last_hit", 1),
                 ("mem_since_last_damage", 1), ("mem_since_last_kill", 1), ("mem_lockons", 1),
                 # decoded Deku Baba behaviour (nearest, second nearest)
                 ("t1_state_onehot", 10), ("t1_facing_link_cos_sin", 2), ("t1_timer", 1),
                 ("t2_state_onehot", 10), ("t2_facing_link_cos_sin", 2), ("t2_timer", 1)]
COMBAT_DIM = sum(n for _, n in COMBAT_FIELDS)


def combat_features(s: CombatState, tracker: CombatTracker, time_left: float) -> np.ndarray:
    live = tracker.live_targets(s)
    live.sort(key=lambda e: math.dist(e.pos, s.pos))
    ts = max(tracker.targets_start, 1)
    out = [s.hp / max(s.hp_cap, 1), s.hp_cap / 48.0, len(live) / ts, tracker.targets_start / 3.0,
           float(s.targeted in {e.ptr for e in live}), float(s.arrow in {e.ptr for e in live}),
           float(s.bit(SF1_SHIELD)), float(s.bit(SF1_DAMAGED))]
    for i in range(2):
        if i < len(live):
            e = live[i]
            rel = [(e.pos[j] - s.pos[j]) / POS_SCALE for j in range(3)]
            block = rel + [math.dist(e.pos, s.pos) / POS_SCALE, e.hp / 2.0]
            if i == 0:
                f = tracker._facing(s, e)
                block += [f, math.sqrt(max(0.0, 1 - f * f))]
        else:
            block = [0.0] * (5 if i else 7)
        out += block
    out.append(time_left)
    ep, sps = tracker.ep, tracker.cfg.steps_per_second
    since = lambda t0: 1.0 if t0 is None else min((tracker.t - t0) / sps / 30.0, 1.0)
    out += [ep["defeated"] / ts, ep["damage_dealt"] / (2.0 * ts), ep["hp_lost"] / 48.0, since(tracker.last_hit_t),
            since(tracker.last_damage_t), since(tracker.last_kill_t), ep["lockons"] / 6.0]
    for i in range(2):
        onehot = [0.0] * len(BABA_STATES)
        if i < len(live):
            e = live[i]
            onehot[BABA_STATES.index(e.state)] = 1.0
            ang = (e.shape_yaw - e.yaw_to_player) / 32768.0 * math.pi
            out += onehot + [math.cos(ang), math.sin(ang), min(max(e.timer, 0), 100) / 100.0]
        else:
            out += onehot + [0.0, 0.0, 0.0]
    arr = np.asarray(out, np.float32)
    assert arr.shape == (COMBAT_DIM,), arr.shape
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
