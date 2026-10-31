"""Where the RvS prompt sits, and that PPO weights survive the widening.

The bug these cover: `train_rvs` built its model as global(297)+cond(3)+actors but built its CORPUS as
concat([obs, g]) -- prompt at the tail. Since the model slices global = x[:, :300], that pushed three
actor-table columns into the global block, shifted every actor row by three (so the id-embedding read the
wrong field) and buried the prompt in the last actor row.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zelda_rl.ppo import RecurrentEntityActorCritic          # noqa: E402
from zelda_rl.rvs import COND_DIM, Conditioned, cond_offset  # noqa: E402

spec = importlib.util.spec_from_file_location("initrvs", ROOT / "scripts" / "init_rvs_from_ppo.py")
initrvs = importlib.util.module_from_spec(spec)
sys.modules["initrvs"] = initrvs
spec.loader.exec_module(initrvs)

FIELDS = {"n_actors": 16, "actor": [("a", 40), ("b", 5), ("present", 1)]}   # 16 x 46 = 736


def test_cond_offset_is_the_end_of_the_global_block():
    assert cond_offset(FIELDS, 1033) == 297, cond_offset(FIELDS, 1033)
    assert cond_offset(FIELDS, 1036 - COND_DIM) == 297


def test_conditioned_inserts_at_cond_at_and_appends_without_it():
    inner = torch.nn.Identity()
    inner.hidden_size = 8
    x = torch.arange(10, dtype=torch.float32)[None]           # 0..9
    ins = Conditioned(inner, target=(1.0, 2.0, 3.0), cond_at=4)._cat(x)[0].tolist()
    assert ins == [0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9], ins
    leg = Conditioned(inner, target=(1.0, 2.0, 3.0), cond_at=None)._cat(x)[0].tolist()
    assert leg == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], leg


def _ppo_ck(global_dim=297, n_actors=16, actor_dim=46, n_actions=108):
    torch.manual_seed(0)
    m = RecurrentEntityActorCritic(global_dim, n_actors, actor_dim, n_actions)
    return {"model": m.state_dict(), "update": 3,
            "config": {"global_dim": global_dim, "n_actors": n_actors, "actor_dim": actor_dim,
                       "n_actions": n_actions, "obs_dim": global_dim + n_actors * actor_dim,
                       "hidden": 256, "gru": 256}}


def test_widening_preserves_the_ppo_policy_exactly():
    ck = _ppo_ck()
    rvs, cfg = initrvs.widen(ck, cond_at=297)
    assert cfg["obs_dim"] == 1036 and cfg["global_dim"] == 300 and cfg["rvs_cond_at"] == 297
    assert cfg["rvs_cond"] is True
    worst = initrvs.check_identical(ck, rvs, 297)
    assert worst == 0.0, f"widened model differs from the PPO policy by {worst:.3e}"


def test_the_inserted_columns_are_zero():
    ck = _ppo_ck()
    rvs, _ = initrvs.widen(ck, cond_at=297)
    w = rvs["model"]["proj.0.weight"]
    assert w.shape == (256, 428), tuple(w.shape)
    assert torch.count_nonzero(w[:, 297:300]) == 0, "the conditioning columns must start at zero"
    # and the surrounding weights are the originals, not shifted
    src = ck["model"]["proj.0.weight"]
    assert torch.equal(w[:, :297], src[:, :297])
    assert torch.equal(w[:, 300:], src[:, 297:])


def test_widening_refuses_a_mismatched_architecture():
    ck = _ppo_ck()
    del ck["model"]["proj.0.weight"]
    try:
        initrvs.widen(ck, cond_at=297)
        raise AssertionError("should have refused a checkpoint missing a tensor")
    except SystemExit as e:
        assert "proj.0.weight" in str(e), str(e)


if __name__ == "__main__":
    fails = 0
    for fn in (test_cond_offset_is_the_end_of_the_global_block,
               test_conditioned_inserts_at_cond_at_and_appends_without_it,
               test_widening_preserves_the_ppo_policy_exactly,
               test_the_inserted_columns_are_zero,
               test_widening_refuses_a_mismatched_architecture):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
