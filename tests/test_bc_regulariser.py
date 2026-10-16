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



def filter_checks() -> list[str]:
    """The Q-filter must train on KEPT frames and ignore dropped ones.

    Built so a no-op is impossible to miss: kept frames demand action TARGET, dropped frames demand a
    different action DECOY. If the filter works the policy learns TARGET and not DECOY; if `keep` were
    ignored it would see a 50/50 mixture and learn neither.
    """
    import torch
    from zelda_rl.ppo import RecurrentEntityActorCritic, ppo_update_recurrent
    DECOY = 7
    torch.manual_seed(0)
    r = np.random.default_rng(3)
    M = 24
    f = np.zeros((M, CH), np.float32); f[:, 0] = 1.0
    keep = np.zeros((M, CH), np.float32); keep[:, : CH // 2] = 1.0        # first half kept
    act = np.full((M, CH), DECOY, dtype=np.int64); act[:, : CH // 2] = TARGET
    bc = {"obs": torch.from_numpy(r.standard_normal((M, CH, OBS)).astype(np.float32)),
          "act": torch.from_numpy(act), "first": torch.from_numpy(f),
          "keep": torch.from_numpy(keep)}
    model = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for i in range(30):
        ppo_update_recurrent(model, opt, rollout(seed=i), chunk=CH, epochs=1, chunks_per_mb=4, clip=0.2,
                             ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, bc=bc, bc_coef=1.0)
    with torch.no_grad():
        h = model.init_hidden(M); pred = []
        for k in range(CH):
            lg, _, h = model.step(bc["obs"][:, k], h, bc["first"][:, k])
            pred.append(lg.argmax(-1))
        pred = torch.stack(pred, 1)
    on_kept = float((pred[:, : CH // 2] == TARGET).float().mean())
    decoy_learned = float((pred[:, CH // 2:] == DECOY).float().mean())
    bad = []
    print(f"\nQ-filter: kept frames demand action {TARGET}, dropped frames demand {DECOY}")
    print(f"  kept frames predicted {TARGET}: {100 * on_kept:.1f}%   dropped frames predicted {DECOY}: "
          f"{100 * decoy_learned:.1f}%")
    if on_kept < 0.8:
        bad.append(f"kept frames only reached {100 * on_kept:.0f}% -- the filter dropped what it should keep")
    if decoy_learned > 0.5:
        bad.append(f"learned the DECOY on {100 * decoy_learned:.0f}% of dropped frames -- keep was ignored")
    print(f"  {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return bad


def adaptive_checks() -> list[str]:
    """The adaptive weight must track a fraction of |policy loss|, not sit at --bc-coef."""
    import torch
    from zelda_rl.ppo import RecurrentEntityActorCritic, ppo_update_recurrent
    torch.manual_seed(0)
    model = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    bc = bc_data()
    applied, accs = [], []
    for i in range(20):
        st = ppo_update_recurrent(model, opt, rollout(seed=i), chunk=CH, epochs=1, chunks_per_mb=4, clip=0.2,
                                  ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                                  bc=bc, bc_coef=0.0, bc_adaptive_frac=0.1)
        applied.append(st.get("bc_coef_applied", float("nan")))
        accs.append(st.get("bc_acc", float("nan")))
    bad = []
    # BY CONSTRUCTION the BC term contributes `frac * |policy loss|`, so at frac 0.1 it is a tenth of the
    # policy term and the resulting coefficient is TINY (policy loss ~1e-4 here / 0.01-0.1 in the real run,
    # against a BC loss near 3). What must be asserted at this frac is the RATIO behaviour -- reported,
    # non-zero, bounded, and moving with the losses -- not that it overwhelms PPO. A second arm at a strong
    # frac asserts the term can still train, so "it is adaptive" cannot pass while "it does nothing" hides.
    print("\nadaptive weight, frac 0.1 (--bc-coef is 0, so any report proves the adaptive path ran)")
    print(f"  applied coef: first {applied[0]:.5f}  last {applied[-1]:.5f}  "
          f"min {min(applied):.5f}  max {max(applied):.5f}")
    print(f"  bc accuracy:  first {100 * accs[0]:.1f}%  last {100 * accs[-1]:.1f}%")
    if not all(a == a for a in applied):
        bad.append("no coefficient was reported")
    elif min(applied) <= 0.0:
        bad.append("applied coefficient hit zero")
    elif len(set(round(a, 7) for a in applied)) < 2:
        bad.append("applied coefficient never changed -- it is not adaptive")

    torch.manual_seed(0)
    strong = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
    opt2 = torch.optim.Adam(strong.parameters(), lr=3e-3)
    s_acc, s_coef = [], []
    for i in range(25):
        st = ppo_update_recurrent(strong, opt2, rollout(seed=i), chunk=CH, epochs=1, chunks_per_mb=4, clip=0.2,
                                  ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                                  bc=bc, bc_coef=0.0, bc_adaptive_frac=200.0)
        s_acc.append(st.get("bc_acc", float("nan"))); s_coef.append(st.get("bc_coef_applied", float("nan")))
    print(f"  frac 200: coef {s_coef[0]:.4f} -> {s_coef[-1]:.4f}, accuracy "
          f"{100 * s_acc[0]:.1f}% -> {100 * s_acc[-1]:.1f}%")
    if s_acc[-1] <= s_acc[0]:
        bad.append("even at a strong frac the adaptive term does not train")
    print(f"  {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return bad


def latent_checks() -> list[str]:
    """Latent alignment must pull the student's GRU state toward a frozen teacher's, and do nothing at coef 0."""
    import torch
    from zelda_rl.ppo import RecurrentEntityActorCritic, ppo_update_recurrent
    bc = bc_data()

    def arm(coef):
        torch.manual_seed(0)
        student = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
        torch.manual_seed(1)
        teacher = RecurrentEntityActorCritic(G, NA, AD, A, hidden=16, gru=16)
        for q in teacher.parameters():
            q.requires_grad_(False)
        teacher.eval()
        opt = torch.optim.Adam(student.parameters(), lr=3e-3)
        for i in range(25):
            st = ppo_update_recurrent(student, opt, rollout(seed=i), chunk=CH, epochs=1, chunks_per_mb=4,
                                      clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, bc=bc,
                                      bc_coef=0.0, bc_teacher=teacher, bc_latent_coef=coef)
        with torch.no_grad():                        # measure the gap the same way for both arms
            sh = student.init_hidden(bc["obs"].shape[0]); th = teacher.init_hidden(bc["obs"].shape[0])
            gaps = []
            for k in range(bc["obs"].shape[1]):
                _, _, sh = student.step(bc["obs"][:, k], sh, bc["first"][:, k])
                _, _, th = teacher.step(bc["obs"][:, k], th, bc["first"][:, k])
                gaps.append(float(((sh - th) ** 2).mean()))
        return st.get("bc_latent", float("nan")), float(np.mean(gaps))

    on_reported, on_gap = arm(1.0)
    off_reported, off_gap = arm(0.0)
    bad = []
    print("\nlatent alignment to a frozen teacher (random teacher, so the gap is meaningful)")
    print(f"  coef 1.0: reported latent loss {on_reported:.4f}   final hidden-state gap {on_gap:.4f}")
    off_r = "n/a" if off_reported != off_reported else f"{off_reported:.4f}"
    print(f"  coef 0.0: reported latent loss {off_r:>14}   final hidden-state gap {off_gap:.4f}")
    if on_gap >= off_gap:
        bad.append(f"alignment did not reduce the gap ({on_gap:.4f} vs {off_gap:.4f} without it)")
    if off_reported == off_reported:
        bad.append("coef 0.0 reported a latent loss -- the term ran when switched off")
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
    fails += filter_checks()
    fails += adaptive_checks()
    fails += latent_checks()

    print()
    for f in fails:
        print("  FAIL " + f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0




if __name__ == "__main__":
    sys.exit(main())
