"""The BC penalty must change the policy toward the demonstrated action, and do nothing when its coef is 0.

Without this, `--bc-coef` could be plumbed through, logged, and silently multiply a detached tensor -- the
run would look regularised and not be. The demonstrations here are synthetic with ONE correct action, so
"the term works" has an unambiguous signature: bc accuracy climbs toward 100%.

    .venv/bin/python tests/test_bc_regulariser.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

from zelda_rl.ppo import RecurrentEntityActorCritic, ppo_update_recurrent  # noqa: E402

G, NA, AD, A, CH = 12, 3, 5, 9, 8
OBS = G + NA * AD
TARGET = 4


def rollout(T=16, N=2, seed=0):
    r = np.random.default_rng(seed)
    model_h = 16
    b = {"obs": r.standard_normal((T, N, OBS)).astype(np.float32),
         "act": r.integers(0, A, (T, N)).astype(np.int64),
         "logp": -np.log(A) * np.ones((T, N), np.float32),
         "adv": r.standard_normal((T, N)).astype(np.float32),
         "ret": r.standard_normal((T, N)).astype(np.float32),
         "val": np.zeros((T, N), np.float32),
         "first": np.zeros((T, N), np.float32),
         "h": np.zeros((T, N, model_h), np.float32)}
    b["first"][0] = 1.0
    return b


def bc_data(M=24, seed=1):
    r = np.random.default_rng(seed)
    f = np.zeros((M, CH), np.float32); f[:, 0] = 1.0
    return {"obs": torch.from_numpy(r.standard_normal((M, CH, OBS)).astype(np.float32)),
            "act": torch.full((M, CH), TARGET, dtype=torch.long),
            "first": torch.from_numpy(f),
            "meta": {"files": 0, "wins": 0, "steps": M * CH, "chunks": M, "chunk": CH}}


def run(bc_coef: float, iters: int = 25):
    torch.manual_seed(0)
    model = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    bc = bc_data()
    accs = []
    for i in range(iters):
        st = ppo_update_recurrent(model, opt, rollout(seed=i), chunk=CH, epochs=1, chunks_per_mb=4,
                                  clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                                  bc=bc, bc_coef=bc_coef)
        accs.append(st.get("bc_acc", float("nan")))
    # Measure the policy's agreement with the demonstrations the same way for BOTH arms, so the coef-0 control
    # is comparable even though it reports no bc stats of its own.
    with torch.no_grad():
        h = model.init_hidden(bc["obs"].shape[0])
        hits = []
        for k in range(CH):
            lg, _, h = model.step(bc["obs"][:, k], h, bc["first"][:, k])
            hits.append((lg.argmax(-1) == TARGET).float().mean().item())
    return accs, float(np.mean(hits))


def schedule_checks() -> list[str]:
    """The decay must hold a FLOOR forever, be monotone, and start where the flag says.

    A coefficient that decays to 0 leaves `--init-from`-only behaviour -- the demonstrations shape a few early
    updates and nothing after -- which is the thing the regulariser exists to avoid. So "never reaches 0" is
    asserted, not trusted to the arithmetic.
    """
    from zelda_rl.bc_demos import coef_at
    bad = []
    start, final, n = 0.1, 0.02, 50
    if coef_at(1, start, final, n) != start:
        bad.append("update 1 is not --bc-coef")
    if coef_at(n, start, final, n) != final:
        bad.append("the floor is not exact at --bc-coef-updates")
    for u in (n, n + 1, 200, 10_000):
        if coef_at(u, start, final, n) != final:
            bad.append(f"floor not held at update {u}")
    if any(coef_at(u, start, final, n) < coef_at(u + 1, start, final, n) for u in range(1, 2 * n)):
        bad.append("not monotone")
    if min(coef_at(u, start, final, n) for u in range(1, 10_000)) <= 0.0:
        bad.append("decays to zero -- the tail is gone")
    print(f"\ndecay schedule {start} -> {final} over {n} updates, then held")
    for u in (1, 10, 25, 50, 100, 400):
        print(f"  update {u:>3}  coef {coef_at(u, start, final, n):.4f}")
    print(f"  {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return bad


def main() -> int:
    fails = []
    on_accs, on_final = run(1.0)
    off_accs, off_final = run(0.0)
    chance = 1.0 / A

    print(f"chance = {100 * chance:.1f}%  (one correct action out of {A})\n")
    print(f"{'arm':<14}{'bc_acc first':>14}{'bc_acc last':>13}{'agreement after':>17}")
    print(f"{'coef 1.0':<14}{100 * on_accs[0]:>13.1f}%{100 * on_accs[-1]:>12.1f}%{100 * on_final:>16.1f}%")
    off0 = "n/a" if off_accs[0] != off_accs[0] else f"{100 * off_accs[0]:.1f}%"
    print(f"{'coef 0.0':<14}{off0:>14}{off0:>13}{100 * off_final:>16.1f}%")

    if not (on_final > 0.9):
        fails.append(f"coef 1.0 only reached {100 * on_final:.1f}% agreement -- the term is not training")
    if not (on_final > off_final + 0.3):
        fails.append(f"coef 1.0 ({100 * on_final:.1f}%) is not clearly above coef 0.0 ({100 * off_final:.1f}%)")
    if off_accs[-1] == off_accs[-1]:
        fails.append("coef 0.0 reported bc stats -- the term ran when it was switched off")
    if not (on_accs[-1] > on_accs[0]):
        fails.append("bc_acc did not rise over the iterations")

    fails += schedule_checks()

    print()
    for f in fails:
        print("  FAIL " + f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0




if __name__ == "__main__":
    sys.exit(main())
