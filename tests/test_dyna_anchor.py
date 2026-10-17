"""The Dyna loop's accept rule. Fails against the version that accepted on `base is None`.

That version cost the loop its anchor on 2026-10-16: v15 kills 5.00% of episodes (6/120, measured the same
day) and was replaced, unmeasured, by an iteration-0 dream candidate whose nine iterations across six pods
scored 0 defeats in 607 real episodes.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dyna_loop", ROOT / "scripts" / "dyna_loop.py")
dyna = importlib.util.module_from_spec(spec)
sys.modules["dyna_loop"] = dyna
spec.loader.exec_module(dyna)


def test_never_accepts_against_an_unmeasured_anchor():
    d = dyna.anchor_decision(None, {"dmg_balanced": 2.90, "kills": 0, "episodes": 18})
    assert d == "needs_baseline", f"got {d}: an unmeasured anchor must be SCORED, never replaced blind"


def test_rejects_a_worse_candidate():
    base = {"dmg_balanced": 2.88, "kills": 6, "episodes": 120}        # v15, measured
    assert dyna.anchor_decision(base, {"dmg_balanced": 2.19}) == "rejected"
    assert dyna.anchor_decision(base, {"dmg_balanced": 2.88}) == "rejected", "a tie is not an improvement"


def test_accepts_a_better_candidate():
    base = {"dmg_balanced": 2.19, "kills": 0, "episodes": 73}
    assert dyna.anchor_decision(base, {"dmg_balanced": 2.90}) == "accepted"


def test_the_loop_scores_the_anchor_before_the_first_decision():
    """A structural check: the accept branch must be preceded by an anchor evaluation."""
    src = (ROOT / "scripts" / "dyna_loop.py").read_text()
    i = src.index("base = st.get(\"anchor_score\")")
    j = src.index("anchor_decision(base, cand_score)")
    between = src[i:j]
    assert "eval_fleet.py" in between and "anchor_eval" in between, \
        "nothing scores the anchor between reading it and deciding against it"
    assert "st[\"anchor_score\"] = base" in between, "the measured anchor score must be persisted"


if __name__ == "__main__":
    fails = 0
    for fn in (test_never_accepts_against_an_unmeasured_anchor,
               test_rejects_a_worse_candidate,
               test_accepts_a_better_candidate,
               test_the_loop_scores_the_anchor_before_the_first_decision):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
