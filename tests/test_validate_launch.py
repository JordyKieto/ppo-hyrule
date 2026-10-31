"""Does the validator actually catch the three failures of 2026-10-31? Reproduce each and watch it fail.

A check that cannot fail is decoration (AGENTS §36). Each case below recreates the real broken state, runs
the validator, and asserts it reports the problem; then the tree is restored and it must go quiet again.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN_ARGS = ("--data datasets/human/gohma --epochs 10 --out checkpoints/kept/_vl_test.pt "
              "--eval-every 0 --fleet-aggregator 0")


def run(script="train_rvs.py"):
    r = subprocess.run([PY, "-u", str(REPO / "scripts" / "validate_launch.py"),
                        "--script", script, "--train-args", TRAIN_ARGS],
                       capture_output=True, text=True, timeout=600, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def main() -> None:
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{('  <- ' + detail) if (detail and not cond) else ''}")
        if not cond:
            fails.append(name)

    # --- healthy tree: must PASS, or the check gets ignored within a day
    rc, out = run()
    ck("healthy tree passes", rc == 0, out[-400:])

    with tempfile.TemporaryDirectory() as bak:
        bak = Path(bak)

        # === failure 1 (AGENTS §44): an exempt marker pasted mid-function breaks the module
        tgt = REPO / "zelda_rl" / "policy_io.py"
        shutil.copy2(tgt, bak / "policy_io.py")
        src = tgt.read_text()
        tgt.write_text(src.replace("def resolve_checkpoint(arg: str | None) -> Path:",
                                   "def resolve_checkpoint(arg: str | None) -> Path:\n\n"
                                   "EXPERIMENT-CONTRACT: exempt -- pasted in the wrong place\n", 1))
        rc, out = run()
        ck("catches the unparseable module that killed 10 ranks", rc == 1 and "does not parse" in out,
           out[-300:])
        shutil.copy2(bak / "policy_io.py", tgt)

        # === failure 2 (AGENTS §44b): a vendored tree under the push root must NOT be flagged
        vend = REPO / "zelda_rl" / "_vl_vendor_test"
        vend.mkdir(exist_ok=True)
        (vend / "storage.py").write_text("import torch\ndef save(p):\n    torch.save({}, p)  # --out\n")
        try:
            rc, out = run()
            ck("an untracked vendored file does not block the push", rc == 0, out[-300:])
        finally:
            shutil.rmtree(vend)

        # === failure 3 (AGENTS §45): the trainer rejects the args the launcher injects
        tgt = REPO / "scripts" / "train_rvs.py"
        shutil.copy2(tgt, bak / "train_rvs.py")
        src = tgt.read_text()
        assert '    ap.add_argument("--run-name", default=None,' in src
        broken = src.replace('    ap.add_argument("--run-name", default=None,',
                             '    _removed = (None,) and (lambda **k: None)(\n        default=None,', 1)
        broken = broken.replace("    a, _unknown = ap.parse_known_args()", "    a = ap.parse_args()", 1)
        tgt.write_text(broken)
        rc, out = run()
        ck("catches the trainer rejecting its own launch argv",
           rc == 1 and ("REJECTED" in out or "unrecognized" in out), out[-300:])
        shutil.copy2(bak / "train_rvs.py", tgt)

        # === and it must be WIRED: the failure mode of every tool in this repo is that nothing calls it
        cli = (REPO / "fleet" / "cli.py").read_text()
        ck("fleet launch calls the validator", "validate_launch.py" in cli, "not wired into cmd_launch")
        ck("and refuses the launch on failure",
           "launch validation failed" in cli and "Nothing was pushed" in cli)

        # --- and the tree is clean again
        rc, out = run()
        ck("tree restored, validator quiet again", rc == 0, out[-300:])

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
