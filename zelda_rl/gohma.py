"""Boss task: Queen Gohma (Inside the Deku Tree boss room, scene 17) and her larvae.

Built on the Deku Baba combat task (zelda_rl/combat.py): same RAM reader, same reward bookkeeping (`pay`), same
core observation block, so a Deku Baba policy transfers (zelda_rl/transfer.py). Gohma is the single *target*
(success = Gohma HP 0); larvae are rewarded per kill, uncapped: Gohma only becomes vulnerable again once her larvae
are dead, so clearing them is progress (the episode time limit bounds it).

RAM (verified on two recorded human fights, runs/manual/boss_20260914-*.jsonl, and all 10 boss-room savestates):
  Gohma        actor id 0x0028, category 9 (boss), params 0xFFFF, HP +0xAF = 10 (one per sword hit), action fn +0x180
  larva / egg  actor id 0x0028, category 5 (enemy), params 0..2 (egg index), HP 2, action fn +0x2A0
  debris       actor id 0x0028, params >= 10 (egg-shell pieces while hatching, body pieces when Gohma dies): ignored
  action fns are decoded RELATIVE to the actor's update function (+0x130): the En_Goma overlay was seen loaded at
  two addresses 0x280 apart, so absolute pointers are not stable across scene reloads.
  Link         held item +0x141 (15 slingshot, 4 sword, 6 Deku Nut throw, 0 none), stateFlags1 bit 20 first-person aim, bit 9 ready to
               fire, bit 3 item out; slingshot seeds = save ctx ammo slot 6
"""
from __future__ import annotations

import math

import numpy as np

from zelda_rl.combat import (DEKU_BABA, GOHMA_ID, SF1_SHIELD, CombatState, CombatTracker, Enemy, Track)
from zelda_rl.obs import POS_SCALE

SF1_ITEM_OUT, SF1_READY_TO_FIRE, SF1_FIRST_PERSON = 3, 9, 20
HELD_SLINGSHOT, HELD_SWORD = 15, 4
GOHMA_FLOOR_Y = -640.0

# Gohma action functions, offset from her update function (+0x130). Semantics from the recorded fights (timeline of
# state vs seed hits / HP changes / height) and the zeldaret/oot BossGoma source order. Grouped into semantic states.
GOHMA_STATE = {
    -0x36FC: "ceiling_wait",       # hangs on the ceiling before the fight (intro)
    -0x0FA8: "ceiling_idle",
    -0x0954: "ceiling_move",       # crawling on the ceiling (seeds here never stun: eye not red)
    -0x0D88: "ceiling_spawn",      # lays eggs from the ceiling
    -0x0E4C: "ceiling_eye_red",    # RED EYE on the ceiling: the seed at 144.1 s knocked her down from here
    -0x1068: "fall",               # knocked off the ceiling
    -0x111C: "fall",
    -0x1348: "land",
    -0x0A20: "climb",              # climbing back up to the ceiling
    -0x0C90: "floor_walk",
    -0x1794: "floor_prepare",      # eye closed wind-up (seed at 103.3 s did nothing; hit Link at 81.6 s)
    -0x168C: "floor_eye_opening",  # 1-frame transition into the red eye
    -0x1630: "floor_eye_red",      # RED EYE on the floor: the seed at 106.6 s stunned her from here (also her attack)
    -0x0DF0: "floor_eye_close",    # after the red-eye window (seed at 92.7 s did nothing)
    -0x1248: "stunned",            # the sword window
    -0x13E0: "stunned_hit",        # HP drops here (hit while stunned)
    -0x12A4: "stunned_recover",
    -0x2604: "defeated",
}
GOHMA_STATES = ["ceiling_wait", "ceiling_idle", "ceiling_move", "ceiling_spawn", "ceiling_eye_red", "fall", "land",
                "climb", "floor_walk", "floor_prepare", "floor_eye_opening", "floor_eye_red", "floor_eye_close",
                "stunned", "stunned_hit", "stunned_recover", "defeated", "other"]
# The eye is ONLY vulnerable in these two states (every stun in the recordings came from a seed fired in one of them;
# a seed in the same state can still miss on angle, so reward requires the knock-down that follows, not just firing).
GOHMA_VULNERABLE = {"ceiling_eye_red", "floor_eye_red"}
GOHMA_STUNNED = {"fall", "land", "stunned", "stunned_hit"}          # knocked down / sword window
GOHMA_ATTACK = {"floor_prepare", "floor_eye_red", "floor_eye_close"}   # Link took hits from her in these (block credit)
GOHMA_CEILING = {"ceiling_wait", "ceiling_idle", "ceiling_move", "ceiling_spawn", "ceiling_eye_red"}
GOHMA_PRE_RED = {"floor_prepare", "floor_eye_opening"}                  # floor wind-up: the eye opens next

# Larva (En_Goma) action functions, offset from its update function.
LARVA_STATE = {
    -0x16A0: "egg", -0x1920: "egg", -0x19C0: "egg",
    -0x110C: "hatching", -0x07A0: "hatching",
    -0x0724: "idle", -0x09A4: "idle",
    -0x0A7C: "prepare_jump", -0x0858: "jump", -0x096C: "land",
    -0x0FD8: "hit", -0x1740: "recover",
    -0x0E90: "dying", -0x0D74: "dead", -0x1258: "hit", -0x138C: "other",
}
LARVA_STATES = ["egg", "hatching", "idle", "prepare_jump", "jump", "land", "hit", "recover", "dying", "dead", "other"]
LARVA_ATTACK = {"prepare_jump", "jump", "land"}


def kind(e: Enemy) -> str:
    if e.id == DEKU_BABA:
        return "deku_baba"
    if e.id == GOHMA_ID:
        if e.cat == 9 and e.params == 0xFFFF:
            return "gohma"
        if e.cat == 5 and e.params < 10:
            return "larva"
        return "gohma_debris"
    return "other"


def gohma_state(e: Enemy) -> str:
    return GOHMA_STATE.get(e.fn_rel, "other")


def larva_state(e: Enemy) -> str:
    return LARVA_STATE.get(e.fn_rel, "other")


def find_gohma(s: CombatState) -> Enemy | None:
    return next((e for e in s.enemies.values() if kind(e) == "gohma"), None)


def larvae(s: CombatState) -> list[Enemy]:
    return [e for e in s.enemies.values() if kind(e) == "larva"]


def _yaw_diff(a: int, b: int) -> float:
    return ((a - b + 32768) % 65536 - 32768) / 32768.0 * math.pi


def link_aim(s: CombatState, point) -> tuple[float, float]:
    """(cos of horizontal angle between Link's facing and the direction to `point`, elevation angle to it / (pi/2))."""
    c, _, elev = link_aim3(s, point)
    return c, elev


def link_aim3(s: CombatState, point) -> tuple[float, float, float]:
    """Link facing vs direction to `point`: (cos, signed sin: + = point is to Link's left, elevation / (pi/2))."""
    yaw = s.yaw / 65536.0 * 2 * math.pi
    fx, fz = math.sin(yaw), math.cos(yaw)
    dx, dy, dz = point[0] - s.pos[0], point[1] - s.pos[1], point[2] - s.pos[2]
    h = math.hypot(dx, dz) or 1.0
    return (fx * dx + fz * dz) / h, (fz * dx - fx * dz) / h, math.atan2(dy, h) / (math.pi / 2)


class GohmaTracker(CombatTracker):
    """Rewards every step toward defeating Gohma. Every term is paid through `pay` (-> reward/<name> in wandb)."""

    def reset(self, s: CombatState, max_steps: int) -> None:
        super().reset(s, max_steps)
        g = find_gohma(s)
        self.tracks = {p: Track(target=(kind(e) == "gohma" and e.hp > 0), hp=e.hp, age=self.cfg.spawn_grace_steps)
                       for p, e in s.enemies.items()}
        self.targets_start = sum(t.target for t in self.tracks.values())
        self.ep["targets_start"] = self.targets_start
        self.g_prev_state = gohma_state(g) if g else "other"
        self.g_state_t = 0
        self.last_seed_t = None
        self.last_nut_t = None
        self.seed_state = "other"
        self.ep.update(boss_hp_start=g.hp if g else 0, boss_hp_max=10, nut_stuns=0, nuts_thrown=0, boss_damage=0, boss_defeated=False, stuns=0, seed_stuns=0,
                       seeds_fired=0, red_eye_seeds=0, seeds_start=s.seeds, larva_kills=0, larva_damage=0, larvae_seen=0,
                       stun_hits=0, boss_attacks=0, boss_blocks=0, vulnerable_steps=0, vulnerable_lock_steps=0,
                       sling_aim_steps=0, sling_ready_steps=0, elevated_seeds=0, time_to_defeat_s=None, boss_hp_end=g.hp if g else 0)
        self._r = {}                              # running totals of capped shaping terms
        self._larva_seen = set()
        self.attack_cycle = None

    def _capped(self, pay, name: str, value: float, cap: float, kind_: str = "shape") -> None:
        got = self._r.get(name, 0.0)
        if value > 0 and got < cap:
            v = min(value, cap - got)
            self._r[name] = got + v
            pay(name, v, kind_)

    def live_targets(self, s: CombatState) -> list[Enemy]:
        return [s.enemies[p] for p, t in self.tracks.items() if t.target and not t.defeated and p in s.enemies]

    def step(self, prev: CombatState, cur: CombatState) -> tuple[float, bool, dict]:
        c, ep = self.cfg, self.ep
        self.t += 1
        ep["length"] = self.t
        parts = ep["r_parts"]
        acc = {"out": 0.0, "shape": 0.0}

        def pay(name: str, value: float, kind_: str) -> None:
            parts[name] = parts.get(name, 0.0) + value
            acc[kind_] += value

        ev: dict = {}
        alive_now = not cur.dead
        g_prev, g = find_gohma(prev), find_gohma(cur)
        gs = gohma_state(g) if g else "other"
        stunned_before = self.g_prev_state in GOHMA_STUNNED

        # ---- enemies: Gohma damage / defeat, larva damage / kills (spawn-grace + despawn traps as in CombatTracker)
        for p, e in cur.enemies.items():
            k = kind(e)
            tr = self.tracks.get(p)
            if tr is None:
                self.tracks[p] = tr = Track(target=False, hp=e.hp)
                if k == "larva" and p not in self._larva_seen:
                    self._larva_seen.add(p)
                    ep["larvae_seen"] += 1
                continue
            tr.age += 1
            if tr.age <= c.spawn_grace_steps:
                tr.hp = e.hp
                continue
            if e.hp < tr.hp and alive_now and not tr.defeated and k in ("gohma", "larva"):
                dmg = tr.hp - e.hp
                if k == "gohma":
                    hp_max = max(ep["boss_hp_max"], 1)
                    pay("boss_damage", c.g_boss_damage * dmg, "out")
                    gone = (hp_max - e.hp) / hp_max                 # 0.1 after the first of 10 hits ... 1.0 on the last
                    pay("boss_hp_progress_bonus", c.g_boss_damage * dmg * c.g_boss_hp_progress * gone, "out")
                    ep["boss_damage"] += dmg
                    ep["damage_dealt"] += dmg
                    ev["boss_hit"] = ev.get("boss_hit", 0) + dmg
                    self.last_hit_t = self.t
                    if stunned_before or gs in GOHMA_STUNNED:
                        pay("stun_hit", c.g_stun_hit * dmg, "shape")
                        ep["stun_hits"] += dmg
                    if e.hp == 0:
                        tr.defeated = True
                        mins = self.t / c.steps_per_second / 60.0
                        mult = 1.0 + c.speed_mult_max * max(0.0, 1.0 - mins / c.g_speed_ref_min)
                        pay("boss_defeat", c.g_boss_defeat, "out")
                        pay("boss_defeat_speed_bonus", c.g_boss_defeat * (mult - 1.0), "out")
                        ep["defeated"] += 1
                        ep["boss_defeated"] = True
                        ep["kill_times_s"].append(round(self.t / c.steps_per_second, 1))
                        ep["kill_mults"].append(round(mult, 3))
                        self.last_kill_t = self.t
                        ev["boss_defeat"] = True
                else:                                               # uncapped: she only reopens her eye once larvae die
                    pay("larva_damage", c.g_larva_damage * dmg, "out")
                    ep["larva_damage"] += dmg
                    ev["larva_hit"] = ev.get("larva_hit", 0) + dmg
                    if e.hp == 0:
                        tr.defeated = True
                        ep["larva_kills"] += 1
                        pay("larva_kill", c.g_larva_kill, "out")
                        ev["larva_kill"] = True
            tr.hp = e.hp
        for p in [p for p in self.tracks if p not in cur.enemies]:
            tr = self.tracks.pop(p)
            if tr.target and not tr.defeated:
                ev["target_lost"] = ev.get("target_lost", 0) + 1

        # ---- Gohma state transitions: stuns (seed to the eye), attacks (block credit)
        if g is not None:
            if gs != self.g_prev_state:
                self.g_state_t = self.t
                if gs in GOHMA_STUNNED and self.g_prev_state not in GOHMA_STUNNED and gs != "stunned_hit":
                    ep["stuns"] += 1
                    ev["boss_stunned"] = True
                    recent = self.last_seed_t is not None and self.t - self.last_seed_t <= c.g_seed_window_steps
                    recent_nut = self.last_nut_t is not None and self.t - self.last_nut_t <= c.g_nut_window_steps
                    if recent and self.seed_state in GOHMA_VULNERABLE:   # knocked down right after a red-eye seed = eye hit
                        ep["seed_stuns"] += 1
                        ev["seed_stun"] = True
                        self._capped(pay, "seed_stun", c.g_seed_stun, c.g_seed_stun_cap, "shape")
                    elif recent_nut:                                     # Deku Nut flash: shorter stun, smaller credit
                        ep["nut_stuns"] += 1
                        ev["nut_stun"] = True
                        self._capped(pay, "nut_stun", c.g_nut_stun, c.g_nut_stun_cap, "shape")
                if gs in GOHMA_ATTACK and self.g_prev_state not in GOHMA_ATTACK:
                    ep["boss_attacks"] += 1
                    self.attack_cycle = {"shield": False, "dmg": 0}
            self.g_prev_state = gs
            ep["boss_hp_end"] = g.hp

        # ---- seeds
        if cur.seeds < prev.seeds and not prev.dead:
            n = prev.seeds - cur.seeds
            ep["seeds_fired"] += n
            self.last_seed_t = self.t
            self.seed_state = gs if g is not None else "other"      # Gohma's state when the seed left the slingshot
            if g is not None and (gs in GOHMA_CEILING or g.pos[1] > GOHMA_FLOOR_Y + 150) and \
                    (cur.targeted == g.ptr or link_aim(cur, g.focus)[0] > 0.85):
                ep["elevated_seeds"] += 1
                self._capped(pay, "elevated_seed", c.g_elevated_seed * n, c.g_elevated_seed_cap)
            if self.seed_state in GOHMA_VULNERABLE:
                ep["red_eye_seeds"] += 1
                self._capped(pay, "red_eye_seed", c.g_red_eye_seed * n, c.g_red_eye_seed_cap)
            ev["seed"] = n
            pay("seed_cost", -c.g_seed_cost * n, "shape")

        if cur.nuts < prev.nuts and not prev.dead:
            n = prev.nuts - cur.nuts
            ep["nuts_thrown"] += n
            self.last_nut_t = self.t
            ev["nut"] = n
            pay("nut_cost", -c.g_nut_cost * n, "shape")

        # ---- Link health
        ep["hp_end"] = max(cur.hp, 0)
        dhp = cur.hp - prev.hp
        if dhp < 0:
            pay("link_damage", -c.w_link_damage * (-dhp), "out")
            ep["hp_lost"] += -dhp
            self.last_damage_t = self.t
            ev["link_damage"] = -dhp
        elif dhp > 0 and not prev.dead:
            pay("link_heal", c.g_link_heal * dhp, "out")
            ep["hp_healed"] += dhp
        if self.attack_cycle is not None:
            self.attack_cycle["dmg"] += max(0, -dhp)
            self.attack_cycle["shield"] |= cur.bit(SF1_SHIELD) and g is not None and self._facing(cur, g) > 0.5
            if gs not in GOHMA_ATTACK:
                cyc, self.attack_cycle = self.attack_cycle, None
                near = g is not None and math.dist(g.pos, cur.pos) < c.g_block_range
                if cyc["dmg"] == 0 and cyc["shield"] and near:
                    ep["boss_blocks"] += 1
                    ev["block"] = True
                    self._capped(pay, "block", c.w_block, c.block_cap * c.w_block)

        # ---- terminal
        terminated = False
        if cur.dead:
            pay("death", -c.g_death_penalty, "out")
            ep["died"], ep["end"], terminated = True, "death", True
        elif (reason := self._left_reason(cur)):
            pay("left_room", -c.g_death_penalty, "out")
            ep["left_room"], ep["end"], terminated = True, "left_room", True
            ep["left_reason"] = reason
        elif self.targets_start > 0 and ep["defeated"] >= self.targets_start:
            pay("success", c.success_bonus, "out")
            pay("success_time_bonus", c.success_time_bonus * max(0.0, 1 - self.t / self.max_steps), "out")
            ep["success"], ep["end"], terminated = True, "success", True
            ep["time_to_clear_s"] = ep["time_to_defeat_s"] = self.t
        elif self.t >= self.max_steps:
            alive = len(self.live_targets(cur))
            pay("timeout", -c.timeout_penalty_per_target * alive, "out")
            ep["end"], ep["targets_alive_at_timeout"] = "timeout", alive

        # ---- shaping toward the kill loop: lock on -> eye red -> slingshot -> stun -> close in -> sword
        if not terminated and g is not None and alive_now:
            locked = cur.targeted == g.ptr
            vulnerable = gs in GOHMA_VULNERABLE
            sling = cur.held_item == HELD_SLINGSHOT
            if locked:
                ep["lock_steps"] += 1
                if prev.targeted != g.ptr and ep["lockons"] < c.lockon_cap:
                    pay("lockon", c.w_lockon, "shape")
                    ep["lockons"] += 1
                    ev["lockon"] = True
                self._capped(pay, "lock_hold", c.w_lock_step, c.lock_step_cap)
            if vulnerable:
                ep["vulnerable_steps"] += 1
                aim_h, _ = link_aim(cur, g.focus)
                if locked or aim_h > 0.9:
                    ep["vulnerable_lock_steps"] += 1
                    self._capped(pay, "lock_on_red_eye", c.g_lock_red_eye, c.g_lock_red_eye_cap)
                    if sling:
                        ep["sling_aim_steps"] += 1
                        self._capped(pay, "slingshot_on_red_eye", c.g_sling_red_eye, c.g_sling_red_eye_cap)
            elevated = gs in GOHMA_CEILING or g.pos[1] > GOHMA_FLOOR_Y + 150
            if sling and (elevated or gs in GOHMA_PRE_RED):         # slingshot cocked at her while she is up (Z or free aim)
                aim_h, _ = link_aim(cur, g.focus)
                if locked or aim_h > 0.85:
                    ep["sling_ready_steps"] += 1
                    self._capped(pay, "sling_ready", c.g_sling_ready, c.g_sling_ready_cap)
            if gs in GOHMA_STUNNED:                                # close the distance for the sword window
                d0, d1 = math.dist(g_prev.pos, prev.pos) if g_prev else math.nan, math.dist(g.pos, cur.pos)
                if math.isfinite(d0) and math.isfinite(d1) and d1 > c.g_sword_range:
                    pay("approach_stunned", float(np.clip(c.w_approach * (d0 - d1) / 100.0, -0.5, 0.5)), "shape")
                if cur.held_item == HELD_SWORD and d1 < c.g_sword_range:
                    self._capped(pay, "sword_ready_stunned", c.g_sword_ready, c.g_sword_ready_cap)
        pay("time_penalty", -c.time_penalty, "shape")
        r = acc["out"] + acc["shape"]
        ep["return"] += r
        ep["r_outcome"] += acc["out"]
        ep["r_shaping"] += acc["shape"]
        return float(r), terminated, ev


# ---- observation block (obs v5): appended after the combat block; all zeros when there is no Gohma / larva
BOSS_FIELDS = [("gohma_present", 1), ("gohma_hp_frac", 1), ("gohma_eye_rel_pos", 3), ("gohma_eye_dist", 1),
               ("gohma_on_ceiling", 1), ("gohma_state_onehot", len(GOHMA_STATES)), ("gohma_vulnerable", 1),
               ("gohma_stunned", 1), ("gohma_time_in_state", 1), ("gohma_facing_link_cos_sin", 2),
               ("link_aim_to_eye_cos_sin_elev", 3), ("locked_on_gohma", 1),
               ("larvae_alive", 1), ("larva1_rel_pos", 3), ("larva1_dist", 1), ("larva1_hp", 1),
               ("larva1_state_onehot", len(LARVA_STATES)), ("larva1_facing_link_cos_sin", 2),
               ("link_facing_larva1_cos_sin", 2), ("larva2_rel_pos", 3), ("larva2_dist", 1),
               ("held_slingshot", 1), ("held_sword", 1), ("first_person_aim", 1), ("ready_to_fire", 1),
               ("seeds", 1), ("since_last_seed", 1), ("mem_stuns", 1), ("mem_larva_kills", 1),
               ("nuts", 1), ("since_last_nut", 1)]
BOSS_DIM = sum(n for _, n in BOSS_FIELDS)


def boss_features(s: CombatState, tracker) -> np.ndarray:
    out: list[float] = []
    g = find_gohma(s)
    if g is not None:
        st = gohma_state(g)
        onehot = [0.0] * len(GOHMA_STATES)
        onehot[GOHMA_STATES.index(st)] = 1.0
        rel = [(g.focus[j] - s.pos[j]) / POS_SCALE for j in range(3)]
        ang = _yaw_diff(g.yaw, g.yaw_to_player)
        aim_c, aim_s, aim_v = link_aim3(s, g.focus)
        t_state = (tracker.t - getattr(tracker, "g_state_t", tracker.t)) / 50.0
        out += [1.0, g.hp / 10.0, *rel, math.dist(g.focus, s.pos) / POS_SCALE, float(g.pos[1] > GOHMA_FLOOR_Y + 150),
                *onehot, float(st in GOHMA_VULNERABLE), float(st in GOHMA_STUNNED), min(t_state, 1.0),
                math.cos(ang), math.sin(ang), aim_c, aim_s, aim_v, float(s.targeted == g.ptr)]
    else:
        out += [0.0] * (1 + 1 + 3 + 1 + 1 + len(GOHMA_STATES) + 1 + 1 + 1 + 2 + 3 + 1)
    lv = sorted((e for e in larvae(s) if e.hp > 0), key=lambda e: math.dist(e.pos, s.pos))
    out.append(len(lv) / 6.0)
    if lv:
        e = lv[0]
        onehot = [0.0] * len(LARVA_STATES)
        onehot[LARVA_STATES.index(larva_state(e))] = 1.0
        la = _yaw_diff(e.yaw, e.yaw_to_player)                     # larva body facing vs direction to Link
        lc, ls, _ = link_aim3(s, e.pos)                            # Link facing vs direction to the larva
        out += [(e.pos[j] - s.pos[j]) / POS_SCALE for j in range(3)] + [math.dist(e.pos, s.pos) / POS_SCALE, e.hp / 2.0] + onehot
        out += [math.cos(la), math.sin(la), lc, ls]
    else:
        out += [0.0] * (3 + 1 + 1 + len(LARVA_STATES) + 4)
    if len(lv) > 1:
        out += [(lv[1].pos[j] - s.pos[j]) / POS_SCALE for j in range(3)] + [math.dist(lv[1].pos, s.pos) / POS_SCALE]
    else:
        out += [0.0] * 4
    ep = getattr(tracker, "ep", {})
    last_seed = getattr(tracker, "last_seed_t", None)
    since = 1.0 if last_seed is None else min((tracker.t - last_seed) / 50.0, 1.0)
    out += [float(s.held_item == HELD_SLINGSHOT), float(s.held_item == HELD_SWORD), float(s.bit(SF1_FIRST_PERSON)),
            float(s.bit(SF1_READY_TO_FIRE)), s.seeds / 30.0, since, ep.get("stuns", 0) / 5.0, ep.get("larva_kills", 0) / 6.0]
    last_nut = getattr(tracker, "last_nut_t", None)
    out += [s.nuts / 20.0, 1.0 if last_nut is None else min((tracker.t - last_nut) / 50.0, 1.0)]
    arr = np.asarray(out, np.float32)
    assert arr.shape == (BOSS_DIM,), (arr.shape, BOSS_DIM)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
