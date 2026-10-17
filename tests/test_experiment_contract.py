"""The experiment contract, and the proof that it both FIRES and STAYS QUIET (AGENTS 36).

The contract itself lives in zelda_rl/experiment_contract.py and is enforced by fleet.cli.rsync, so a
non-compliant trainer cannot be pushed to a pod. These tests check the three things that make a check
trustworthy: it fails on the broken state, it passes on the healthy one, and the enforcement is actually
wired into the deploy path rather than sitting in a module nobody calls.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zelda_rl.experiment_contract import (OVERRIDE, check, enforce, legacy,  # noqa: E402
                                          opens_dolphin, runs_evals, saves_checkpoints, violations)

ACTIVE = {"train_rvs.py", "train_wm_obs.py", "train_dream_obs_ppo.py", "dyna_loop.py", "eval_fleet.py",
          "collect_transitions.py", "wm_status.py", "train_combat.py"}


def _tmp(src: str, name="train_thing.py") -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / name
    p.write_text(src)
    return p


def test_it_fires_on_a_trainer_with_no_s3_backup():
    p = _tmp("import torch\nap.add_argument('--out')\ntorch.save({}, out)\n")
    v = violations(p)
    assert any("S3" in x for x in v), v


def test_it_fires_on_an_eval_that_only_prints():
    p = _tmp("print(f'  EVAL epoch {ep}: dmgB 1.0 | wins 2/24')\n")
    v = violations(p)
    assert any("stdout" in x for x in v), v


def test_it_fires_on_a_dolphin_script_without_thread_pinning():
    p = _tmp("env, _ = make_env(cfg, dir)\n")
    v = violations(p)
    assert any("set_num_threads" in x for x in v), v


def test_a_compliant_script_is_quiet():
    p = _tmp("import torch\nfrom zelda_rl import ckpt_s3\nfrom zelda_rl.eval_log import EvalSink\n"
             "torch.set_num_threads(1)\nap.add_argument('--out')\n"
             "torch.save({}, out)\nckpt_s3.upload(out, 'x', 1)\n"
             "env, _ = make_env(cfg, d)\nprint('  EVAL epoch 1: ok')\nsink.log(1, {}, rows=[])\n")
    assert violations(p) == [], violations(p)


def test_the_exemption_marker_is_honoured():
    p = _tmp("# EXPERIMENT-CONTRACT: exempt -- reads rows written elsewhere\n"
             "print('  EVAL epoch 1: x')\n")
    assert violations(p) == []


def test_the_active_experiment_surface_complies():
    bad = check([p for p in (ROOT / "scripts").glob("*.py") if p.name in ACTIVE])
    assert not bad, f"the scripts the current experiments run break the contract: {bad}"


def test_legacy_scripts_are_grandfathered_not_silently_ignored():
    lg = legacy()
    assert lg, "the legacy snapshot is empty: either every script complies (delete the file) or it was lost"
    assert not (lg & ACTIVE), f"active scripts must not be grandfathered: {lg & ACTIVE}"


def test_enforce_raises_and_the_override_works():
    p = _tmp("import torch\nap.add_argument('--out')\ntorch.save({}, out)\n")
    try:
        enforce([p]); raise AssertionError("enforce() did not raise on a violating file")
    except SystemExit as e:
        assert "REFUSING TO PUSH" in str(e), str(e)
    os.environ[OVERRIDE] = "1"
    try:
        enforce([p])                                   # must not raise with the deliberate override
    finally:
        os.environ.pop(OVERRIDE, None)


def test_the_gate_is_wired_into_the_deploy_path():
    src = (ROOT / "fleet" / "cli.py").read_text()
    i = src.index("def rsync(")
    body = src[i:i + 1800]
    assert "experiment_contract" in body and "enforce(" in body, \
        "fleet.cli.rsync does not enforce the contract: the check exists but nothing calls it on deploy"


if __name__ == "__main__":
    fails = 0
    for fn in (test_it_fires_on_a_trainer_with_no_s3_backup,
               test_it_fires_on_an_eval_that_only_prints,
               test_it_fires_on_a_dolphin_script_without_thread_pinning,
               test_a_compliant_script_is_quiet,
               test_the_exemption_marker_is_honoured,
               test_the_active_experiment_surface_complies,
               test_legacy_scripts_are_grandfathered_not_silently_ignored,
               test_enforce_raises_and_the_override_works,
               test_the_gate_is_wired_into_the_deploy_path):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                               # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {e}")
    print("ALL PASS" if not fails else f"\n{fails} FAILED")
    raise SystemExit(1 if fails else 0)
