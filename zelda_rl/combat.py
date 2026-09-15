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
                               yaw=_s16(ram, a + 0x32))
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
    )


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
    g_red_eye_seed: float = 1.5        # every seed fired while the eye is red (hit or not): shoot at the right time
    g_red_eye_seed_cap: float = 9.0
    g_sling_ready: float = 0.08        # per step holding the slingshot cocked, Z-targeting or free-aiming at Gohma while
    g_sling_ready_cap: float = 6.0     #   her eye is about to turn red or is red (not while she just waits / crawls)
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
