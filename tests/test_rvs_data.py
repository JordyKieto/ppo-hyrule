"""The RvS loader must return equal-length keys even when an episode is shorter than one window.

2026-10-16, gohma-col: all 18 RvS arms died in their first updates with
`IndexError: index 6538 is out of bounds for dimension 0 with size 19`. The loader slice-assigned each
episode's weights with `W[-n:] = ...`; an episode shorter than `chunk` gives n == 0, `W[-0:]` is `W[0:]`, and
every weight collected so far was replaced by an empty list. `keep` ended shorter than `obs`, and the sampler
indexed past it. Only the coef-0 control arms survived, because they never load the corpus.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl import rvs_data  # noqa: E402

KEYS = ("obs", "act", "first", "keep", "ep", "pos")
CHUNK = 10


def _episode(tmp: Path, name: str, steps: int, won: bool, dim: int):
    np.savez(tmp / name, obs=np.zeros((steps, dim), np.float32), action=np.zeros(steps, np.int64),
             boss_defeated=np.array(won), obs_fields=np.array(""))


def test_short_episode_does_not_wipe_weights():
    dim = 8
    tmp_path = Path(tempfile.mkdtemp())
    rvs_data.field_offsets = lambda s: {}
    rvs_data.conditioning = lambda obs, off, won: np.zeros((len(obs), rvs_data.COND_DIM), np.float32)
    # long, SHORT (< one chunk: the n == 0 case), long, short, long win
    for i, (steps, won) in enumerate([(35, False), (4, False), (25, False), (7, False), (30, True)]):
        _episode(tmp_path, f"ep{i:02d}.npz", steps, won, dim)
    out = rvs_data.load_rvs_windows([tmp_path], CHUNK, obs_dim=dim + rvs_data.COND_DIM, n_actions=4,
                                    win_weight=5.0, max_windows=10_000)
    n = {k: int(out[k].shape[0]) for k in KEYS}
    assert len(set(n.values())) == 1, n
    assert n["obs"] == 3 + 2 + 3                     # 35 -> 3, 25 -> 2, 30 -> 3; the short ones give none
    assert torch.allclose(out["keep"][-3:], torch.full((3, CHUNK), 5.0))   # the win kept its multiplier
    assert torch.allclose(out["keep"][:-3], torch.ones(5, CHUNK))


def test_merge_stays_equal_length():
    def blk(n, w):
        return {"obs": torch.zeros(n, CHUNK, 3), "act": torch.zeros(n, CHUNK, dtype=torch.long),
                "first": torch.zeros(n, CHUNK), "keep": torch.full((n, CHUNK), w),
                "ep": torch.arange(n), "pos": torch.zeros(n, dtype=torch.long)}
    rvs = {**blk(6, 1.0), "n_static": 6}
    for _ in range(5):
        rvs = rvs_data.merge_capped(rvs, blk(4, 2.0), cap=14, fresh_share=0.5)
        n = {k: int(rvs[k].shape[0]) for k in KEYS}
        assert len(set(n.values())) == 1, n
    assert rvs["obs"].shape[0] == 14
    assert abs(rvs["fresh_share"] - 0.5) < 1e-6


if __name__ == "__main__":
    test_short_episode_does_not_wipe_weights()
    test_merge_stays_equal_length()
    print("OK  tests/test_rvs_data.py: short-episode weights + merge length")
