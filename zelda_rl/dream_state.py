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
                             CombatState, Enemy)

# BABA_STATES index -> a representative action-routine pointer, so Enemy.state decodes back to the name.
_FN_FOR_STATE = {}
for _fn, _name in DEKU_BABA_STATE.items():
    _FN_FOR_STATE.setdefault(BABA_STATES.index(_name), _fn)
_FN_FOR_STATE.setdefault(BABA_STATES.index("other"), 0)
FN_BY_STATE = [_FN_FOR_STATE.get(i, 0) for i in range(N_BABA_STATES)]
ROW_PTR_BASE = 0x80300000          # synthetic, stable per row: the tracker only uses pointers as identity


def decoded_to_states(p: dict, scale: dict) -> list[CombatState]:
    """Decoder output (batched) -> one CombatState per batch element."""
    lhp = np.asarray(LINK_HP_VALUES)[p["link_hp"].argmax(-1).numpy()]
    lpos = p["link_pos"].numpy() * scale["link_pos"][1] + scale["link_pos"][0]
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
