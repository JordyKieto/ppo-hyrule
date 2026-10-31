"""Does the credit horizon actually reach the kill? Asserted on the real `gae()`, not on arithmetic.

docs/GOHMA_FOCUS.md names this as the root cause of the 3.2-3.4/10 damage ceiling: `gamma 0.995 /
gae_lambda 0.95` are identical in all 19 exported RL runs and were tuned when a Deku Baba died in 15 s.
Gohma takes 45-154 s. These tests measure, on `zelda_rl.ppo.gae`, how much of a terminal +60 reaches the
first step of a long episode under v15's settings and under the full-episode settings, and they fail if
anyone sets the horizons back.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.ppo import gae  # noqa: E402

HZ = 10.0
V15 = dict(gamma=0.995, lam=0.95)
FULL = dict(gamma=1 - 1 / 3000, lam=1.0)     # value horizon 3000 steps = the 300 s episode cap


def _terminal_reward_episode(seconds=154.0, bonus=60.0):
    """One episode of `seconds`, zero reward until a +bonus kill on the final step."""
    T = int(seconds * HZ)
    rew = np.zeros((T, 1), np.float64)
    rew[-1, 0] = bonus
    val = np.zeros((T, 1), np.float64)       # a zero critic isolates the discounting
    done = np.zeros((T, 1), np.float64)
    done[-1, 0] = 1.0
    return rew, val, done, np.zeros((1,), np.float64)


def credit_at_start(seconds, **hp):
    rew, val, done, nv = _terminal_reward_episode(seconds)
    adv, _ = gae(rew, val, done, nv, hp["gamma"], hp["lam"])
    return float(adv[0, 0]) / 60.0           # fraction of face value seen by the first decision


def test_v15_cannot_see_the_kill_even_in_a_SHORT_fight():
    """TWO different discounts, and conflating them is an error I made in writing this file.

    The VALUE target is discounted by gamma^t: gamma 0.995 keeps 47% of a terminal +60 over 15 s, which is
    where "these settings were fine for Deku Baba" comes from. The POLICY GRADIENT is weighted by
    (gamma*lambda)^t = 0.94525^t, which keeps 0.023% over the same 15 s. So the terminal bonus has never
    reached the first decision through GAE on ANY task in this project -- Deku Baba was solved by dense
    shaping inside the 1.8 s window, not by propagated credit for the kill. That is the strongest available
    argument for the horizon change, and it only appears if the two discounts are kept apart.
    """
    assert credit_at_start(154.0, **V15) < 1e-5, "154 s: effectively zero"
    short = credit_at_start(15.0, **V15)
    assert short < 0.001, f"15 s should also be ~0.02% through the ADVANTAGE, got {100*short:.3f}%"
    # the value discount over the same 15 s is the number that looks reassuring, and is a different quantity
    assert 0.4 < V15["gamma"] ** 150 < 0.5


def test_full_episode_settings_reach_the_first_decision():
    for secs, floor in ((154.0, 0.5), (100.0, 0.6), (300.0, 0.3)):
        frac = credit_at_start(secs, **FULL)
        assert frac > floor, f"at {secs:.0f} s the first step sees {100*frac:.1f}%, want >{100*floor:.0f}%"


def test_the_change_is_worth_orders_of_magnitude():
    gain = credit_at_start(154.0, **FULL) / max(credit_at_start(154.0, **V15), 1e-12)
    assert gain > 1000, f"the horizon change must move credit by >1000x at 154 s, got {gain:.0f}x"


def test_gae_still_stops_at_an_episode_boundary():
    """Full-episode credit must NOT leak across a reset: two episodes in one rollout stay separate."""
    T = 200
    rew = np.zeros((T, 1), np.float64); val = np.zeros((T, 1), np.float64); done = np.zeros((T, 1), np.float64)
    done[99, 0] = 1.0                      # first episode ends at t=99
    rew[199, 0] = 60.0                     # the SECOND episode's terminal reward
    done[199, 0] = 1.0
    adv, _ = gae(rew, val, done, np.zeros((1,), np.float64), FULL["gamma"], FULL["lam"])
    assert abs(float(adv[50, 0])) < 1e-9, "credit leaked backwards across a done flag"
    assert float(adv[100, 0]) > 30.0, "the second episode's own steps must see its reward"


def test_lambda_one_is_monte_carlo_over_the_episode():
    """lambda=1 means the advantage is the full return minus the baseline -- no bootstrap truncation."""
    T = 50
    rng = np.random.default_rng(0)
    rew = rng.normal(size=(T, 1))
    val = rng.normal(size=(T, 1))
    done = np.zeros((T, 1)); done[-1, 0] = 1.0
    adv, ret = gae(rew, val, done, np.zeros((1,)), 1.0, 1.0)
    mc = np.cumsum(rew[::-1, 0])[::-1]                         # undiscounted return-to-go
    assert np.allclose(adv[:, 0] + val[:, 0], mc, atol=1e-9), "lambda=1, gamma=1 must give the MC return"


if __name__ == "__main__":
    fails = 0
    for fn in (test_v15_cannot_see_the_kill_even_in_a_SHORT_fight,
               test_full_episode_settings_reach_the_first_decision,
               test_the_change_is_worth_orders_of_magnitude,
               test_gae_still_stops_at_an_episode_boundary,
               test_lambda_one_is_monte_carlo_over_the_episode):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {e}")
    print()
    for secs in (15, 45, 100, 154, 300):
        print(f"  kill credit at the FIRST decision of a {secs:>3} s fight:  "
              f"v15 {100*credit_at_start(secs, **V15):7.3f}%   full-episode "
              f"{100*credit_at_start(secs, **FULL):7.2f}%")
    print("\nALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
