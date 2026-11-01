"""The structured-state JEPA's schema and mask sampler, tested against the states they can reach.

Each assertion here exists for a failure this repo has already paid for:

  * the grid must cover EXACTLY the flat state's width, for all three versions. `state_layout.py` exists
    because the collector and the model had the layout written out twice and drifted; a (entity, field) grid
    is a third copy unless every index is derived from the first one (AGENTS 3).
  * a mask may never leave the context or the target side empty. `W[-0:]` is `W[0:]`, and that one n==0 edge
    case wiped the weights of 17 of 22 sweep arms (AGENTS 36 row 1). Here the equivalent is a draw that
    covers the whole grid: the loss reduces over nothing, or the encoder reads nothing.
  * the achieved mask ratio must be near the requested one. Unclamped, the sampler produced draws with ONE
    context cell and reported them as a 0.997 mask. A number a program prints about its own output is a
    claim (AGENTS 36).
  * tube masks must be TUBES. A per-frame random mask is solvable by temporal interpolation on a 60 Hz game,
    so if `entity_tube` leaves that entity visible in some frame the pretext is not the one intended.

The negative controls at the bottom break the code and require the check to fire, per AGENTS 36 rule 1 --
a check that has not been seen to fail is decoration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zelda_rl.sjepa import BLOCKS, GROUPS, MaskSampler, Schema  # noqa: E402
from zelda_rl.state_layout import (ENEMY_FIELDS, N_ROWS, ROW_DIM, STATE_DIM_V1,  # noqa: E402
                                   STATE_DIM_V2, STATE_DIM_V3, link_fields)

WIDTHS = (STATE_DIM_V1, STATE_DIM_V2, STATE_DIM_V3)


def test_schema_covers_every_width() -> None:
    for w in WIDTHS:
        s = Schema(w)
        assert int(s.valid.sum()) == w, f"width {w}: grid covers {int(s.valid.sum())}"
        assert s.n_ent == 1 + N_ROWS
        # Link's cells are its own field count; every actor row is ROW_DIM wide.
        assert int(s.valid[0].sum()) == len(link_fields(w))
        for r in range(1, s.n_ent):
            assert int(s.valid[r].sum()) == ROW_DIM
        # and the columns are a permutation of 0..w-1, so no cell maps to two fields or none
        cols = np.sort(s.cols[s.valid])
        assert (cols == np.arange(w)).all(), f"width {w}: columns are not a permutation of the state"
    print(f"schema ok for widths {WIDTHS}")


def test_unpack_roundtrips_real_columns() -> None:
    s = Schema(STATE_DIM_V3)
    rng = np.random.default_rng(0)
    state = rng.standard_normal((5, STATE_DIM_V3)).astype(np.float32)
    grid = s.unpack(state)
    assert grid.shape == (5, s.n_ent, s.n_field)
    # every valid cell equals the column it came from; every invalid cell is zero
    e, f = np.nonzero(s.valid)
    assert np.allclose(grid[:, e, f], state[:, s.cols[e, f]])
    assert (grid[:, ~s.valid] == 0).all()
    # a named spot-check that does not go through s.cols, so a wrong cols table cannot pass this
    li = {n: i for i, n in enumerate(link_fields(STATE_DIM_V3))}
    assert np.allclose(grid[:, 0, s.fidx["pos_x"]], state[:, li["pos_x"]])
    row3_hp = len(link_fields(STATE_DIM_V3)) + 3 * ROW_DIM + ENEMY_FIELDS.index("hp")
    assert np.allclose(grid[:, 1 + 3, s.fidx["hp"]], state[:, row3_hp])
    print("unpack matches hand-derived columns for link pos_x and row 3 hp")


def test_groups_partition_the_state() -> None:
    s = Schema(STATE_DIM_V3)
    total = sum(int(s.group_cells(g).sum()) for g in s.groups)
    assert total == STATE_DIM_V3, f"groups cover {total} of {STATE_DIM_V3} cells"
    # no cell in two groups
    stack = np.stack([s.group_cells(g) for g in s.groups])
    assert stack.sum(0).max() <= 1, "a cell belongs to more than one group"
    print(f"{len(s.groups)} groups partition all {STATE_DIM_V3} cells")


def test_mask_invariants_including_T_equals_one() -> None:
    s = Schema(STATE_DIM_V3)
    ms = MaskSampler(s, ratio=0.75, rng=np.random.default_rng(1))
    valid = s.valid
    for T in (1, 2, 3, 8, 16):
        for _ in range(60):
            tgt, info = ms.sample(T)
            vt = np.broadcast_to(valid, tgt.shape)
            ctx = vt & ~tgt
            assert tgt.any(), f"T={T}: empty target side"
            assert ctx.any(), f"T={T}: empty context side"
            assert not (tgt & ~vt).any(), f"T={T}: targeted an invalid cell"
            assert ((ctx | tgt) == vt).all(), f"T={T}: context+target != valid"
            assert info["n_target"] + info["n_context"] == info["n_valid"]
            assert 0.60 <= info["ratio_achieved"] <= 0.90, \
                f"T={T}: achieved {info['ratio_achieved']:.3f} for a 0.75 request"
    print("mask invariants hold for T in 1,2,3,8,16 (300 draws)")


def test_entity_and_group_masks_are_tubes() -> None:
    """A tube covers ALL frames, or a neighbouring frame answers the target by interpolation."""
    s = Schema(STATE_DIM_V3)
    T = 8
    rng = np.random.default_rng(2)
    only_ent = MaskSampler(s, ratio=0.2, weights={"entity_tube": 1, "group_tube": 0,
                                                  "time_block": 0, "cell": 0}, rng=rng)
    tgt, _ = only_ent.sample(T)
    hit = tgt.any(axis=(0, 2))
    assert hit.any(), "entity_tube masked no entity"
    for e in np.flatnonzero(hit):
        per_frame = tgt[:, e, :][:, s.valid[e]]
        assert per_frame.all(), f"entity {e} is visible in some frame -- not a tube"
    only_grp = MaskSampler(s, ratio=0.2, weights={"entity_tube": 0, "group_tube": 1,
                                                  "time_block": 0, "cell": 0}, rng=rng)
    tgt, _ = only_grp.sample(T)
    for g in s.groups:
        cells = s.group_cells(g)
        inside = tgt[:, cells]
        if inside.any():
            assert inside.all(), f"group {g} is partly visible -- not a tube"
    print("entity_tube and group_tube cover every frame")


def test_weights_and_ratio_are_validated() -> None:
    s = Schema(STATE_DIM_V3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            MaskSampler(s, ratio=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ratio={bad} was accepted")
    try:
        MaskSampler(s, ratio=0.5, weights={"entity_tube": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("a weights dict missing three block names was accepted")
    try:
        MaskSampler(s, ratio=0.5, weights={b: 0.0 for b in BLOCKS})
    except ValueError:
        pass
    else:
        raise AssertionError("all-zero block weights were accepted")
    print("ratio and weight validation fires on every bad input")


def test_negative_controls_a_broken_schema_is_caught() -> None:
    """Break it, and watch the check fail. AGENTS 36 rule 1."""
    import zelda_rl.sjepa as sj

    # 1. a field dropped from GROUPS must fail _assert_groups, not silently become unmaskable
    saved = dict(sj.GROUPS)
    try:
        sj.GROUPS["pos"] = ("pos_x", "pos_y")            # drop pos_z
        try:
            Schema(STATE_DIM_V3)
        except AssertionError as e:
            assert "pos_z" in str(e), f"wrong error: {e}"
        else:
            raise AssertionError("a state field in no GROUP was accepted")
        # 2. the same field in two groups must fail too
        sj.GROUPS.clear(); sj.GROUPS.update(saved)
        sj.GROUPS["vel"] = tuple(saved["vel"]) + ("pos_x",)
        try:
            Schema(STATE_DIM_V3)
        except AssertionError as e:
            assert "two groups" in str(e), f"wrong error: {e}"
        else:
            raise AssertionError("a field in two groups was accepted")
        # 3. a typo'd name that exists in NO state version must fail
        sj.GROUPS.clear(); sj.GROUPS.update(saved)
        sj.GROUPS["pos"] = tuple(saved["pos"]) + ("pos_w",)
        try:
            Schema(STATE_DIM_V3)
        except AssertionError as e:
            assert "pos_w" in str(e), f"wrong error: {e}"
        else:
            raise AssertionError("a nonexistent field name was accepted")
    finally:
        sj.GROUPS.clear(); sj.GROUPS.update(saved)
    Schema(STATE_DIM_V3)                                  # restored, still builds
    print("negative controls: missing / duplicated / nonexistent field names all fire")


def test_negative_control_unclamped_overshoot_would_be_caught() -> None:
    """The bug that WAS there: blocks overlap, so the ratio overshot to 0.997 with one context cell.

    Reproduced by raising ratio_max back to its unclamped behaviour and asserting the invariant test's bound
    would reject it. This is the positive control for the clamp -- without it, this test fails.
    """
    s = Schema(STATE_DIM_V3)
    loose = MaskSampler(s, ratio=0.75, ratio_max=0.999, rng=np.random.default_rng(0))
    worst = max(loose.sample(1)[1]["ratio_achieved"] for _ in range(200))
    assert worst > 0.90, f"could not reproduce the overshoot (worst {worst:.3f})"
    tight = MaskSampler(s, ratio=0.75, rng=np.random.default_rng(0))
    worst_t = max(tight.sample(1)[1]["ratio_achieved"] for _ in range(200))
    assert worst_t <= 0.90, f"clamp did not hold (worst {worst_t:.3f})"
    print(f"overshoot control: unclamped reaches {worst:.3f}, clamped stays at {worst_t:.3f}")


def main() -> None:
    test_schema_covers_every_width()
    test_unpack_roundtrips_real_columns()
    test_groups_partition_the_state()
    test_mask_invariants_including_T_equals_one()
    test_entity_and_group_masks_are_tubes()
    test_weights_and_ratio_are_validated()
    test_negative_controls_a_broken_schema_is_caught()
    test_negative_control_unclamped_overshoot_would_be_caught()
    print("OK test_sjepa_mask")


if __name__ == "__main__":
    main()
