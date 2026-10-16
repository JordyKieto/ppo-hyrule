"""The key-event penalties and the missed/hallucinated measurement.

Every assertion fails against the previous loss, which weighted all 1033 columns alike and used plain BCE
for the firing decision -- under which a gate that never opens is near-optimal, measured live as boss-HP
recall 0.000.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.wm_obs import (KEY_EVENTS, ObsWM, column_weight, event_accuracy,  # noqa: E402
                             gate_pos_weight, key_event_masks, step_weight)

D, G_HP, L_HP = 12, 3, 7


def _episode(n=40, boss_hits=(10, 20), link_hits=(15,), boss_dies_at=None, link_dies_at=None):
    obs = np.zeros((n, D), np.float32)
    obs[:, G_HP] = 1.0
    obs[:, L_HP] = 1.0
    obs[:, 0] = np.arange(n) / n                       # a moving non-key field
    for t in boss_hits:
        obs[t:, G_HP] -= 0.1
    for t in link_hits:
        obs[t:, L_HP] -= 0.25
    if boss_dies_at:
        obs[boss_dies_at:, G_HP] = 0.0
    if link_dies_at:
        obs[link_dies_at:, L_HP] = 0.0
    return {"obs": obs, "act": np.zeros(n, np.int64), "file": "t.npz", "win": bool(boss_dies_at)}


def test_masks_find_all_four_events():
    e = _episode(boss_hits=(10,), link_hits=(15,), boss_dies_at=30, link_dies_at=35)
    obs = torch.from_numpy(e["obs"])[None]
    m = key_event_masks(obs, G_HP, L_HP)
    assert set(m) == set(KEY_EVENTS), set(m)
    assert int(m["boss_damage"][0].sum()) == 2, "the hit and the death are both boss damage"
    assert int(m["boss_death"][0].sum()) == 1 and int(m["boss_death"][0].nonzero()[0, 0]) == 29
    assert int(m["link_death"][0].sum()) == 1 and int(m["link_death"][0].nonzero()[0, 0]) == 34
    assert int(m["link_damage"][0].sum()) == 2


def test_death_ticks_weigh_more_than_damage_ticks():
    e = _episode(boss_hits=(10,), link_hits=(), boss_dies_at=30)
    obs = torch.from_numpy(e["obs"])[None]
    msk = torch.ones(1, obs.shape[1])
    w = step_weight(msk, key_event_masks(obs, G_HP, L_HP), event_w=5.0, death_w=50.0)
    assert abs(float(w[0, 9]) - 5.0) < 1e-6, "a damage tick carries the damage weight"
    assert abs(float(w[0, 29]) - 50.0) < 1e-6, "a death tick carries the death weight"
    assert abs(float(w[0, 0]) - 1.0) < 1e-6, "a quiet tick is unweighted"


def test_missing_boss_damage_costs_more_than_missing_a_normal_field():
    """The whole point of --key-weight: the same miss on the boss-HP column must cost far more."""
    col_w = column_weight(D, [G_HP, L_HP], 20.0)
    pos_w = gate_pos_weight(D, [G_HP, L_HP], 50.0)
    fired = torch.zeros(1, 1, D); fired[0, 0, G_HP] = 1.0; fired[0, 0, 1] = 1.0
    tgt = torch.zeros(1, 1, D); tgt[0, 0, G_HP] = -1.0; tgt[0, 0, 1] = -1.0
    pred = torch.zeros(1, 1, D)                      # predicts NOTHING moved: missed both
    logits = torch.full((1, 1, D), -6.0)             # gate shut: missed both firings
    bce = F.binary_cross_entropy_with_logits(logits, fired, reduction="none", pos_weight=pos_w)
    mag = F.smooth_l1_loss(pred, tgt, reduction="none") * (0.1 + 0.9 * fired) * col_w
    key_cost = float((bce + mag)[0, 0, G_HP]); normal_cost = float((bce + mag)[0, 0, 1])
    assert key_cost > 15 * normal_cost, f"key {key_cost:.3f} vs normal {normal_cost:.3f}"


def test_hallucinating_boss_damage_is_penalised():
    col_w = column_weight(D, [G_HP, L_HP], 20.0)
    halluc_w = column_weight(D, [G_HP, L_HP], 5.0)
    fired = torch.zeros(1, 1, D)                     # nothing happened
    tgt = torch.zeros(1, 1, D)
    pred = torch.zeros(1, 1, D); pred[0, 0, G_HP] = -1.0; pred[0, 0, 1] = -1.0   # invents movement
    cw = col_w * fired + halluc_w * (1.0 - fired)
    mag = F.smooth_l1_loss(pred, tgt, reduction="none") * (0.1 + 0.9 * fired) * cw
    assert float(mag[0, 0, G_HP]) > 4 * float(mag[0, 0, 1]), mag[0, 0, :2]


def test_tiny_drift_is_not_counted_as_a_hallucinated_hit():
    """A predicted -1e-4 per tick is phantom DRIFT, not a hallucinated hit, and is reported separately."""
    eps_ = [_episode(boss_hits=(10,), link_hits=())]
    m = ObsWM(D, 4, hidden=16, gru=16)
    with torch.no_grad():
        m.head.bias[G_HP] = -1e-4
        m.gate.bias.fill_(6.0)                             # gate open, so the tiny value passes through
    r = event_accuracy(m, torch.ones(D), eps_, G_HP, L_HP, torch.device("cpu"), max_steps=39, batch=1,
                       obs_dim=D)
    assert r["boss_damage"]["halluc_per_1k_quiet"] < 1.0, r["boss_damage"]["halluc_per_1k_quiet"]
    assert 5e-5 < r["boss_damage"]["phantom_drift_per_tick"] < 5e-4, r["boss_damage"]


def test_event_accuracy_separates_missed_from_hallucinated():
    eps = [_episode(boss_hits=(10, 20), link_hits=(15,), boss_dies_at=30) for _ in range(2)]
    dev = torch.device("cpu")
    scales = torch.ones(D)
    m = ObsWM(D, 4, hidden=16, gru=16)               # zero head + shut gate == persistence
    r = event_accuracy(m, scales, eps, G_HP, L_HP, dev, max_steps=39, batch=2, obs_dim=D)
    assert r["boss_damage"]["n_events"] == 6, r["boss_damage"]            # 2 hits + death, x2 episodes
    assert r["boss_damage"]["recall"] == 0.0, "persistence must MISS every event"
    assert r["boss_damage"]["missed"] == 1.0
    assert r["boss_damage"]["halluc_per_1k_quiet"] == 0.0, "and hallucinate none"
    assert r["boss_damage"]["phantom_drift_per_tick"] < 1e-3, "persistence does not drift"
    assert r["boss_death"]["n_events"] == 2 and r["boss_death"]["recall"] == 0.0
    # a model that always predicts a big negative delta on the boss column: perfect recall, hallucinates
    with torch.no_grad():
        m.head.bias[G_HP] = -0.5
        m.gate.bias.fill_(6.0)
    r2 = event_accuracy(m, scales, eps, G_HP, L_HP, dev, max_steps=39, batch=2, obs_dim=D)
    assert r2["boss_damage"]["recall"] == 1.0, r2["boss_damage"]
    assert r2["boss_damage"]["halluc_per_1k_quiet"] > 900, r2["boss_damage"]
    assert r2["boss_damage"]["precision"] < 0.1, r2["boss_damage"]


if __name__ == "__main__":
    fails = 0
    for fn in (test_masks_find_all_four_events,
               test_death_ticks_weigh_more_than_damage_ticks,
               test_missing_boss_damage_costs_more_than_missing_a_normal_field,
               test_hallucinating_boss_damage_is_penalised,
               test_tiny_drift_is_not_counted_as_a_hallucinated_hit,
               test_event_accuracy_separates_missed_from_hallucinated):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
