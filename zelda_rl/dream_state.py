"""Rebuild a real `CombatState` from decoder outputs, so the dream can run the REAL reward function.

The dream previously re-implemented a handful of reward terms by hand. That drifted: an audit found only
5 of the 18 terms in `CombatTracker.step` were present, and the four missing *outcome* terms were the ones
that make play urgent -- leaving the room (-20), the timeout penalty (-3 per live target), the kill speed
bonus (up to +10) and the success time bonus (up to +15). The dreamed policy stalled to the step cap and
walked out of the room, which is correct play for the reward it was actually given.

So: do not re-implement it. Decode the state, rebuild a `CombatState`, and call `CombatTracker.step` --
the same object the emulator run uses. Parity is then structural rather than maintained by hand, and the
shaping terms (lock-on, approach, shield, block, dodge) come along for free since they only need geometry
the decoder already produces.

Fields the decoder cannot supply are filled with values that are inert for the Deku Baba profile:
seeds / nuts / grass / held_item are read only by the Gohma tracker, and `arrow` only feeds observations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from zelda_rl.ppo import RecurrentEntityActorCritic, _layer
from zelda_rl.bc import LINK_HP_VALUES, N_BABA_STATES, ROW_HP_VALUES
from zelda_rl.combat import (BABA_STATES, DEKU_BABA, DEKU_BABA_STATE, SF1_DAMAGED, SF1_DEAD, SF1_SHIELD,
                             CombatState, CombatTracker, Enemy)

# BABA_STATES index -> a representative action-routine pointer, so Enemy.state decodes back to the name.
_FN_FOR_STATE = {}
for _fn, _name in DEKU_BABA_STATE.items():
    _FN_FOR_STATE.setdefault(BABA_STATES.index(_name), _fn)
_FN_FOR_STATE.setdefault(BABA_STATES.index("other"), 0)
FN_BY_STATE = [_FN_FOR_STATE.get(i, 0) for i in range(N_BABA_STATES)]
ROW_PTR_BASE = 0x80300000          # synthetic, stable per row: the tracker only uses pointers as identity


def apply_transition(prev: list[CombatState], cur: list[CombatState], q: dict,
                     calib: dict | None = None, rng=None, hp_per_hit: int = 8,
                     hp0_despawn_steps: int = 12, _timers: dict | None = None) -> list[CombatState]:
    """Edit the decoded next state with what the ACTION did, from `TransitionHead(z_t, a_t)`.

    The audit this answers: across all 63 constant actions the dream managed max 0.00 kills and max 0.70 of
    the 6 HP available, and the damage it did show came from SHIELD actions, which cannot deal damage at all.
    The reward reads HP out of a decoded state whose predictor ties the identity, so nothing changed and the
    only changes were decode noise -- billed as combat.

    Position comes from the classified direction and magnitude (a mode, so it does not shrink the way the
    regressed magnitude did, 0.11x of true on moving steps).

    Damage and despawn are SAMPLED from calibrated probabilities, not thresholded. A threshold tuned to
    maximise F1 on a 0.02%-base-rate event is necessarily precision-heavy, so it fired ~0.9 times an episode
    -- and a 2 HP Deku Baba needs TWO hits on the same row, so kills stayed at exactly 0.00 for every action.
    Sampling a calibrated probability reproduces the real event RATE instead, which is what has to be right
    for two hits to accumulate. The dream becomes stochastic, which PPO handles and the real game is anyway.

    Despawn matters because a kill is booked when HP reaches 0 while the actor is still present, and the real
    game then REMOVES it; without that a dead Baba lingers at HP 0 forever, so `baba_count` and `row_present`
    disagree with the enemy dict and the room never empties. `hp0_despawn_steps` is a backstop: a row held at
    0 HP longer than the dying animation is removed regardless of what the head says.

    Reward PARITY is preserved: this edits the STATE, and `CombatTracker` still computes all 18 terms from
    states exactly as the emulator run does. Nothing here computes reward.
    """
    import torch as _t
    from zelda_rl.dynamics import dir_unit_vectors, mag_centers
    calib = calib or {}
    rng = rng or np.random.default_rng()
    uv, mc = dir_unit_vectors(), mag_centers()

    def prob(logits, key, softmax_none=None):
        """Platt-calibrated probability; identity when the checkpoint carries no calibration for this head."""
        a, b = calib.get(key, (1.0, 0.0))
        sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))   # clipped: raw logits overflow exp
        if softmax_none is None:
            return sig(a * logits + b)
        p_none = _t.softmax(_t.as_tensor(logits), -1).numpy()[:, softmax_none]
        pe = 1.0 - p_none
        lg = np.log(np.clip(pe, 1e-6, 1 - 1e-6) / np.clip(1 - pe, 1e-6, 1 - 1e-6))
        return sig(a * lg + b)

    gate = prob(q["tr_moved"].numpy()[:, 0], "moved")
    d = q["tr_dir"].numpy().argmax(-1)
    mg = q["tr_mag"].numpy().argmax(-1)
    step = uv[d] * (mc[mg] * gate)[:, None]
    dy = q["tr_dy"].numpy()[:, 0]
    n_rows = q["tr_hit_row"].shape[1] - 1
    p_hit = prob(q["tr_hit_row"].numpy(), "hit", softmax_none=n_rows)
    p_gone = prob(q["tr_gone_row"].numpy(), "gone", softmax_none=n_rows)
    hit_row = q["tr_hit_row"].numpy()[:, :n_rows].argmax(-1)
    gone_row = q["tr_gone_row"].numpy()[:, :n_rows].argmax(-1)
    p_lhit = prob(q["tr_link_hit"].numpy()[:, 0], "link_hit")
    timers = _timers if _timers is not None else {}

    out = []
    for i, (pv, cu) in enumerate(zip(prev, cur)):
        cu.pos = (pv.pos[0] + float(step[i, 0]), pv.pos[1] + float(dy[i]), pv.pos[2] + float(step[i, 1]))
        if rng.random() < p_hit[i]:
            e = cu.enemies.get(ROW_PTR_BASE + int(hit_row[i]) * 0x100)
            if e is not None and e.hp > 0:
                e.hp = max(0, e.hp - 1)
        if rng.random() < p_lhit[i] and cu.hp > 0:
            cu.hp = max(0, cu.hp - hp_per_hit)      # a Deku Baba lunge costs half a heart
        # despawn: the head's choice, or any row stuck at 0 HP past the dying animation
        drop = {ROW_PTR_BASE + int(gone_row[i]) * 0x100} if rng.random() < p_gone[i] else set()
        for ptr, e in list(cu.enemies.items()):
            k = (i, ptr)
            timers[k] = timers.get(k, 0) + 1 if e.hp <= 0 else 0
            if timers[k] > hp0_despawn_steps:
                drop.add(ptr)
        for ptr in drop:
            cu.enemies.pop(ptr, None)
            timers.pop((i, ptr), None)
        out.append(cu)
    return out


def decoded_to_states(p: dict, scale: dict, prev: list[CombatState] | None = None,
                      moved_thresh: float | None = None,
                      moved_calib: tuple[float, float] = (1.0, 0.0)) -> list[CombatState]:
    """Decoder output (batched) -> one CombatState per batch element.

    With `prev`, Link's position is INTEGRATED (prev.pos + decoded displacement) instead of read from the
    absolute head. Absolute position decodes to ~12 world units of error while a real step is 0.58, so the
    per-step signal is ~20x smaller than the noise it would be measured through; reading it absolutely is why
    the dream's motion was incoherent -- 12.11 units of jitter per step against a true 0.58, which averaged
    out to almost no net drift, so Link never reached the door and the -20 exit never fired in training.

    `link_moved` gates it, because Link is stationary on 95% of real steps and a displacement head that
    cannot emit an exact zero will always move him. The gate is SOFT by default:

        E[dpos] = P(moved) * E[dpos | moved]

    which is just the expectation of the two-part model, and needs no threshold. That matters because a hard
    threshold did not transfer: tuned on the tuning slots (moved base rate 0.182) it ran at precision 0.295
    on the test slots (base rate 0.085). A soft gate is also self-scaling -- an uncertain frame contributes a
    small step instead of a full one, which is what took predicted displacement from 2.52x of true to ~1x.
    Pass `moved_thresh` to force the old hard gate instead.
    """
    lhp = np.asarray(LINK_HP_VALUES)[p["link_hp"].argmax(-1).numpy()]
    lpos = p["link_pos"].numpy() * scale["link_pos"][1] + scale["link_pos"][0]
    if prev is not None and "link_dpos" in p and "link_moved" in p:
        d = p["link_dpos"].numpy()
        if "link_dpos" in scale:
            d = d * scale["link_dpos"][1] + scale["link_dpos"][0]
        mlog = p["link_moved"].numpy()[:, 0]
        ca, cb = moved_calib                                    # Platt scaling fitted on the tuning slots
        gate = ((mlog > moved_thresh) if moved_thresh is not None
                else 1.0 / (1.0 + np.exp(-(ca * mlog + cb))))[:, None]
        lpos = np.asarray([list(s.pos) for s in prev], np.float32) + d * gate
    lyaw = p["link_yaw"].numpy()
    yaw = (np.arctan2(lyaw[:, 0], lyaw[:, 1]) / (2 * np.pi) * 65536.0).astype(np.int64) % 65536
    bits = (p["link_bits"].numpy() > 0)
    tgt_row = p["targeted"].argmax(-1).numpy()
    scene = p["scene"].argmax(-1).numpy()
    room = p["room"].argmax(-1).numpy()
    present = p["row_present"].numpy() > 0
    is_baba = p["row_is_baba"].numpy() > 0
    rhp = np.asarray(ROW_HP_VALUES)[p["row_hp"].argmax(-1).numpy()]
    rpos = p["row_pos"].numpy() * scale["row_pos"][1] + scale["row_pos"][0]
    rstate = p["row_state"].argmax(-1).numpy()
    n, n_rows = len(lhp), present.shape[1]

    out = []
    for b in range(n):
        sf1 = 0
        for bit, on in ((SF1_DEAD, bits[b, 0]), (SF1_SHIELD, bits[b, 1]), (SF1_DAMAGED, bits[b, 2])):
            if on:
                sf1 |= 1 << bit
        enemies = {}
        for j in range(n_rows):
            if not present[b, j]:
                continue
            ptr = ROW_PTR_BASE + j * 0x100
            enemies[ptr] = Enemy(ptr=ptr, id=DEKU_BABA if is_baba[b, j] else 0x0095,
                                 hp=int(rhp[b, j]), pos=tuple(float(v) for v in rpos[b, j]),
                                 fn=FN_BY_STATE[int(rstate[b, j])], cat=5)
        t = int(tgt_row[b])
        out.append(CombatState(
            scene=int(scene[b]), room=int(room[b]), hp=int(lhp[b]), hp_cap=48, sf1=sf1,
            pos=tuple(float(v) for v in lpos[b]), yaw=int(yaw[b]),
            targeted=(ROW_PTR_BASE + t * 0x100) if t < n_rows else 0, arrow=0,
            enemies=enemies, held_item=4, seeds=0, nuts=0, grass=-1))
    return out


def memory_features(trk, time_left: float) -> np.ndarray:
    """The episode-memory block the real policy sees, computed with the formulas in `combat_features`.

    The reward terms are stateful -- lock-on is capped at 6 per episode, blocks at 8, the kill bonus decays
    with time since the last kill -- so a policy without these cannot see why the same action pays
    differently at different moments.
    """
    ep, sps = trk.ep, trk.cfg.steps_per_second
    ts = max(trk.targets_start, 1)
    since = lambda t0: 1.0 if t0 is None else min((trk.t - t0) / sps / 30.0, 1.0)   # noqa: E731
    return np.array([ep["defeated"] / ts, ep["damage_dealt"] / (2.0 * ts), ep["hp_lost"] / 48.0,
                     since(trk.last_hit_t), since(trk.last_damage_t), since(trk.last_kill_t),
                     ep["lockons"] / 6.0, time_left], np.float32)


MEM_DIM = 8


class DreamPolicy(RecurrentEntityActorCritic):
    """The emulator's recurrent policy, with the entity encoder replaced by the frozen latent.

    Everything downstream -- proj, GRUCell, actor and critic heads, their widths and initialisations,
    act()/value()/init_hidden(), and `ppo_update_recurrent` -- is the class the model-free run used,
    unchanged. Only `_features` is bypassed, because `z` is already the state encoding.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, gru: int = 256):
        super().__init__(global_dim=obs_dim, n_actors=1, actor_dim=1, n_actions=n_actions,
                         hidden=hidden, gru=gru)
        self.proj = nn.Sequential(_layer(obs_dim, gru), nn.Tanh())
        del self.enc

    def step(self, x, h, first):
        h = h * (1.0 - first.unsqueeze(-1))
        h = self.gru(self.proj(x), h)
        return self.pi(h), self.v(h).squeeze(-1), h


class DreamTracker(CombatTracker):
    """CombatTracker with a monotone HP memory, for use on DECODED states only.

    On real states HP never rises -- a Deku Baba's HP increases in 0 of 891,459 recorded row-steps -- so
    the base class can safely sync its record to whatever it observes:

        if e.hp < tr.hp: pay(enemy_damage, tr.hp - e.hp)
        tr.hp = e.hp                      # also runs when HP ROSE

    A decoded HP can wobble. The sync then resets the payment counter, so 2 -> 1 -> 2 -> 1 bills the same
    enemy twice, and a policy can farm one Baba indefinitely by steering into a region where the decode
    oscillates. Measured: 8.94 damage per dream episode against a physical maximum of 5.58, with kills
    flat at 0.22 -- more damage than the enemies have HP, and nothing dying.

    Clamping the *record* rather than the state is the narrow fix. A spurious drop still pays once, which
    is unavoidable, but it cannot be re-billed. Clamping the decoded state instead was tried earlier and
    is worse: it makes a spurious drop permanent, so drift toward the dataset mean hands out free kills.

    On real states every clamp here is a no-op, so reward parity with the emulator is preserved exactly.
    """

    def step(self, prev: CombatState, cur: CombatState):
        for p, e in cur.enemies.items():
            t = self.tracks.get(p)
            if t is not None and e.hp > t.hp:
                e.hp = t.hp                        # never let a decoded rise reset the counter
        if cur.hp > prev.hp:
            cur.hp = prev.hp                       # same for Link: no healing in a dream (no drops modelled)
        return super().step(prev, cur)


def build_seeds(state, epid, keep, ccfg, dream_max_steps, pool_idx=None):
    """Replay the REAL CombatTracker over real episodes and snapshot it at each wanted frame.

    Tracker state -- kills so far, lock-ons spent against the cap of 6, blocks against 8, time since the
    last kill, damage dealt, open lunge cycles -- is nowhere in the 335-number game state. It is the reward
    function's own bookkeeping, accumulated over an episode, so no decoder reading a single frame can
    recover it. Replaying the real episode up to that frame gives it exactly.

    Without this, seeding a dream mid-fight tells the policy "0 kills, no lock-ons used, full time left"
    while the state shows a half-finished fight, and the reward pays again for progress already made.

    Time is rebased: `t` is reset to 0 so the dream gets a full horizon from the seed, while
    `last_*_t` are shifted to preserve *steps since* each event, which is what the memory block reports.
    """
    import copy

    from zelda_rl.combat import CombatState
    seeds = []
    want = None if pool_idx is None else set(pool_idx.tolist())
    order = np.argsort(epid, kind="stable")
    for e in np.unique(epid):
        idx = order[np.searchsorted(epid[order], e):np.searchsorted(epid[order], e, "right")]
        if not keep[idx[0]]:
            continue
        states = decoded_states_from_raw(state[idx])
        trk = CombatTracker(ccfg)
        trk.reset(states[0], dream_max_steps)
        if want is None or int(idx[0]) in want:
            seeds.append((int(idx[0]), copy.deepcopy(trk)))
        for t in range(1, len(states)):
            trk.step(states[t - 1], states[t])
            if want is None or int(idx[t]) in want:
                s = copy.deepcopy(trk)
                shift = s.t
                s.t = 0
                for f in ("last_kill_t", "last_hit_t", "last_damage_t"):
                    v = getattr(s, f)
                    if v is not None:
                        setattr(s, f, v - shift)
                s.ep["length"] = 0
                seeds.append((int(idx[t]), s))
    return seeds


def decoded_states_from_raw(raw):
    """Real 335-d states -> CombatState objects, for replaying the tracker on ground truth."""
    from zelda_rl.bc import decode_targets
    import torch as _t
    Y = decode_targets(raw)
    n = len(raw)
    p = {"link_hp": _t.nn.functional.one_hot(_t.from_numpy(
             np.abs(Y["link_hp"][:, 0:1] - np.asarray(LINK_HP_VALUES)).argmin(-1)), len(LINK_HP_VALUES)).float(),
         "link_pos": _t.from_numpy(Y["link_pos"]), "link_yaw": _t.from_numpy(Y["link_yaw"]),
         "link_bits": _t.from_numpy(Y["link_bits"] * 2 - 1),
         "targeted": _t.nn.functional.one_hot(_t.from_numpy(Y["targeted"]), 17).float(),
         "scene": _t.nn.functional.one_hot(_t.from_numpy(Y["scene"]), 2).float(),
         "room": _t.nn.functional.one_hot(_t.from_numpy(Y["room"]), 8).float(),
         "row_present": _t.from_numpy(Y["row_present"] * 2 - 1),
         "row_is_baba": _t.from_numpy(Y["row_is_baba"] * 2 - 1),
         "row_hp": _t.nn.functional.one_hot(_t.from_numpy(
             np.abs(Y["row_hp"][..., None] - np.asarray(ROW_HP_VALUES)).argmin(-1)), len(ROW_HP_VALUES)).float(),
         "row_pos": _t.from_numpy(Y["row_pos"]),
         "row_state": _t.nn.functional.one_hot(_t.from_numpy(Y["row_state"]), N_BABA_STATES).float()}
    return decoded_to_states(p, {"link_pos": (0.0, 1.0), "row_pos": (0.0, 1.0)})


def _fresh(self, max_steps):
    """A DreamTracker copy of this replayed tracker, ready to continue for `max_steps` more."""
    import copy
    t = copy.deepcopy(self)
    t.__class__ = DreamTracker
    t.max_steps = max_steps
    return t


CombatTracker.fresh = _fresh
