"""Ground-truth action-effect pairs: the SAME state, two different actions, both real outcomes.

This is the test the motion work needs before any model change, and the criterion the MOBA world-model study
used ("restore one state, apply two different action sets, check the model predicts both"). Without it,
"is motion action-conditioned" is an argument; with it, it is a number.

Method: load a savestate slot, replay a fixed random PREFIX of actions to diversify the state, record the
state, apply action A for one step and record the outcome; then reload the same slot, replay the identical
prefix, apply action B and record. The emulator is deterministic given identical inputs, so the two branches
share a state exactly -- verified per pair, and any pair whose branch point differs is discarded rather than
kept and averaged over.

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
      .venv/bin/python-dbg scripts/collect_counterfactual.py --pairs 200 --out datasets/cf/pairs.npz

What it writes: s0 (state at the branch point), a1/a2 (the two actions), s1/s2 (the two outcomes). Then
`--report` prints how far apart the two outcomes are, which is the ceiling any model can be measured against:
if two actions from one state produce the same next state, no model can be scored on telling them apart.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import SF1_CUTSCENE, SF1_DEAD, read_combat_state  # noqa: E402
from zelda_rl.env import COMBAT_ACTIONS_V5, action_name  # noqa: E402
from zelda_rl.obs import read_state_file_v2  # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR  # noqa: E402
from zelda_rl.state_layout import link_fields  # noqa: E402

# encode_state is the collector's OWN flattener (scripts/collect_transitions.py), so these pairs land in
# exactly the 363-column layout every model here already reads. Re-deriving it would be a second copy of the
# layout, which zelda_rl/state_layout.py exists to prevent.
from scripts.collect_transitions import EnemyRows, encode_state  # noqa: E402

MOVE_ONLY = [a for a, (d, b) in enumerate(COMBAT_ACTIONS_V5) if b == 0 and d is not None]   # 8 directions


def build(idx, slots, slots_dir, seconds):
    from zelda_rl.combat import CombatConfig
    from zelda_rl.env import DolphinGoalEnv, EnvConfig
    files = {k: Path(slots_dir) / f"NACE01.s{k:02d}" for k in slots}
    states = {k: read_state_file_v2(f) for k, f in files.items()}
    # task="combat" is what selects the 108-action COMBAT_ACTIONS_V5 space in send_action; "goal" has 27.
    cfg = EnvConfig(episode_vi=int(seconds * 60), vi_per_step=6, video_backend="Null",
                    unlimited_speed=True, obs_version=5, task="combat",
                    combat=CombatConfig(profile="gohma", max_seconds=seconds, target_ids=(0x28, 0x55)),
                    stall_timeout=30.0)
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"cf_{os.getpid()}"
    return DolphinGoalEnv(idx, root / f"i{idx}", files, states, states[min(slots)], cfg)


def step(env, a, settle=0.0005):
    env.send_action(int(a))
    while not env.ready():
        time.sleep(settle)
    return env.observe()


def flat(env, rows):
    """The 363-column state row, via the collector's own encoder so the layout cannot drift."""
    return encode_state(read_combat_state(env.ram), rows).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--slots", default="3,4,5,7,9")
    p.add_argument("--prefix", type=int, default=12, help="random steps replayed before the branch point")
    p.add_argument("--lag", type=int, default=3, help="how many steps the branch action is HELD. The measured "
                   "action/displacement alignment is 0.006 at one step and 0.24 by three, so branching one "
                   "step apart would measure nothing (docs/WM_MOTION.md).")
    p.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    p.add_argument("--out", default="datasets/cf/pairs.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", action="store_true", help="only re-report an existing --out")
    a = p.parse_args()

    if a.report:
        return report(a.out)

    slots = [int(s) for s in a.slots.split(",")]
    rng = np.random.default_rng(a.seed)
    env = build(0, slots, a.slots_dir, 60.0)
    S0, S1, S2, A1, A2, SL = [], [], [], [], [], []
    kept = dropped = 0
    t0 = time.time()
    try:
        while kept < a.pairs:
            slot = int(rng.choice(slots))
            pre = rng.integers(0, len(COMBAT_ACTIONS_V5), a.prefix)
            a1, a2 = rng.choice(MOVE_ONLY, 2, replace=False)
            branch = []
            for aa in (a1, a2):
                env.reset(slot)
                for x in pre:
                    step(env, x)
                s0 = read_combat_state(env.ram)
                if s0.bit(SF1_DEAD) or s0.bit(SF1_CUTSCENE):
                    branch = None
                    break
                rows = EnemyRows()
                pre_state = flat(env, rows)
                for _ in range(a.lag):                     # the action is applied for `lag` steps: the lag
                    step(env, aa)                          # table says its effect is ~0 at one step
                branch.append((pre_state, flat(env, rows)))
            if branch is None or len(branch) != 2:
                dropped += 1
                continue
            (p1, o1), (p2, o2) = branch
            if not np.allclose(p1, p2, atol=1e-3):        # the replay did not reproduce the branch point
                dropped += 1
                continue
            S0.append(p1); S1.append(o1); S2.append(o2); A1.append(int(a1)); A2.append(int(a2)); SL.append(slot)
            kept += 1
            if kept % 20 == 0:
                print(f"  {kept}/{a.pairs} pairs ({dropped} dropped), {time.time()-t0:.0f}s", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, s0=np.stack(S0), s1=np.stack(S1), s2=np.stack(S2),
                        a1=np.array(A1), a2=np.array(A2), slot=np.array(SL), lag=a.lag)
    print(f"wrote {out}  {kept} pairs, {dropped} dropped, {time.time()-t0:.0f}s")
    report(a.out)


def report(path):
    d = np.load(path)
    s0, s1, s2 = d["s0"], d["s1"], d["s2"]
    lf = link_fields(s0.shape[1]); li = {n: i for i, n in enumerate(lf)}
    P = [li["pos_x"], li["pos_z"]]
    d1 = np.linalg.norm(s1[:, P] - s0[:, P], axis=1)          # how far A moved Link
    d2 = np.linalg.norm(s2[:, P] - s0[:, P], axis=1)
    sep = np.linalg.norm(s1[:, P] - s2[:, P], axis=1)          # how far apart the two outcomes are
    print(f"\n{len(s0)} pairs from {len(set(d['slot'].tolist()))} slots")
    print(f"  displacement under action A   mean {d1.mean():6.2f}  median {np.median(d1):6.2f}  "
          f"zero on {100*(d1 < .5).mean():.0f}% of pairs")
    print(f"  displacement under action B   mean {d2.mean():6.2f}  median {np.median(d2):6.2f}")
    print(f"  SEPARATION of the outcomes    mean {sep.mean():6.2f}  median {np.median(sep):6.2f}  "
          f"below 0.5 units on {100*(sep < .5).mean():.0f}% of pairs")
    print(f"\n  A model that ignores the action scores at best {sep.mean()/2:.2f} mean error on this set;")
    print(f"  a model that reads it perfectly scores 0. That is the whole measurable range.")
    m = sep > 0.5
    if m.any():
        print(f"  on the {m.sum()} pairs the actions DO separate: separation mean {sep[m].mean():.2f}, "
              f"max {sep[m].max():.2f}")


if __name__ == "__main__":
    main()
