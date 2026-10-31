"""The gate let a file that does not parse onto 18 pods. These assert it cannot happen again.

2026-10-31: an `EXPERIMENT-CONTRACT: exempt` note was pasted into the middle of a function in
`zelda_rl/policy_io.py`. That did two things at once -- the module became unimportable, and because the
exemption was a regex over raw text, it silenced every other check in the contract. All ten ranks of
`gohma-rvsv15` trained four epochs, synced ten-of-ten, and died on IndentationError at their first online
eval, which imports that module through eval_fleet.

Two bugs, one shape (AGENTS §36: a check that cannot fail is a check that lies):
  1. the contract never syntax-checked anything;
  2. `fleet.cli.rsync` filtered its argv for names ending ".py" -- and a code push sends DIRECTORIES, so
     the gate had inspected zero files on every real push since it was written.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zelda_rl.experiment_contract import HARD, check, enforce, expand_py, violations

BROKEN = '''"""A trainer."""
def f():

EXPERIMENT-CONTRACT: exempt -- a marker in a position that breaks the parse
    """doc"""
    return 1
'''
FINE = '''"""A trainer.

EXPERIMENT-CONTRACT: exempt -- a marker in a real docstring
"""
def f():
    return 1
'''


def main() -> None:
    fails = []

    def ck(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pkg = d / "pkg"
        pkg.mkdir()
        bad = pkg / "broken.py"
        bad.write_text(BROKEN)
        good = pkg / "ok.py"
        good.write_text(FINE)

        # --- positive control: the broken state must FAIL (it did not, which is why 10 ranks died)
        v = violations(bad)
        ck("a file that does not parse is a violation", len(v) == 1 and "does not parse" in v[0])
        ck("an exempt marker cannot silence a parse failure", v and v[0].startswith(HARD))

        # --- negative control: correct work must PASS, or the check gets ignored within a day
        ck("a marker in a real docstring still exempts", violations(good) == [])

        # --- the actual deploy bug: rsync is handed DIRECTORIES, not files
        ck("expand_py walks a directory", {p.name for p in expand_py([pkg])} == {"broken.py", "ok.py"})
        ck("the old filter saw nothing", [s for s in [str(pkg)] if s.endswith(".py")] == [])
        ck("check() on the directory finds the broken file", "broken.py" in check(expand_py([pkg])))

        # --- the override is loud, but a file that cannot run is not deployable at any volume
        os.environ["ZRL_FORCE_CONTRACT"] = "1"
        try:
            try:
                enforce(expand_py([pkg]), what="push")
                ck("ZRL_FORCE_CONTRACT=1 does NOT wave through a parse failure", False)
            except SystemExit:
                ck("ZRL_FORCE_CONTRACT=1 does NOT wave through a parse failure", True)
        finally:
            os.environ.pop("ZRL_FORCE_CONTRACT", None)

        bad.unlink()
        enforce(expand_py([pkg]), what="push")
        ck("a clean directory pushes", True)

    # --- and the real tree, which is what a push actually sends
    repo = Path(__file__).resolve().parents[1]
    tree = check(expand_py([repo / "zelda_rl", repo / "scripts", repo / "fleet"]))
    ck(f"the repo's own push passes ({len(expand_py([repo / 'zelda_rl', repo / 'scripts', repo / 'fleet']))} files)",
       tree == {})
    if tree:
        for k, vs in sorted(tree.items()):
            print(f"      {k}: {vs}")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
