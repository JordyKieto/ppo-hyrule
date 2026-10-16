#!/usr/bin/env python3
"""Run every test in tests/, and FAIL a test that would silently run nothing.

2026-10-16: `tests/test_rvs_data.py` was written to catch the ragged-RvS-buffer crash that had just killed
17 sweep arms. It used pytest fixtures (`tmp_path`, `monkeypatch`), and there is no pytest in this venv --
`python tests/test_rvs_data.py` exited 0 having executed not one assertion, and the fix shipped unverified
(AGENTS.md 36, row 2). A check that cannot fail is worse than no check, so this runner enforces the two
things that make a test file in this repo real:

  1. it does not import pytest (nothing in this venv can run it), and
  2. everything it defines as a test -- a `test_*` function, or the `main()` the older files use -- is
     actually CALLED from its `__main__` block.

Then it runs each file and requires exit 0 with some output. Run it before any commit that claims a fix:

    .venv/bin/python tests/run_all.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).name


def defects(path: Path) -> list[str]:
    src = path.read_text()
    tree = ast.parse(src)
    bad: list[str] = []
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) and "pytest" in ast.unparse(n) for n in ast.walk(tree)):
        bad.append("imports pytest, which is not installed in this venv -- it would run NOTHING")
    tests = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
             and (n.name.startswith("test_") or n.name == "main")]
    if not tests:
        bad.append("defines neither a test_* function nor main()")
    main = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    if not main:
        bad.append("has no `if __name__ == \"__main__\":` block, so it runs nothing when executed")
    else:
        # a test counts as reached if its name is REFERENCED anywhere outside its OWN definition: called
        # from __main__, called from main(), or passed as a function object to a runner loop. Scanning only
        # the __main__ block's text reported test_rvs_memguard.py as a failure while it was passing.
        for t in tests:
            refs: set[str] = set()
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == t:
                    continue
                refs |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if t not in refs:
                bad.append(f"{t}() is defined but nothing references it -- it would never run")
    return bad


def main() -> int:
    files = sorted(f for f in HERE.glob("test_*.py"))
    if not files:
        print("FAIL  tests/: no test_*.py found")
        return 1
    fails = 0
    for f in files:
        bad = defects(f)
        if bad:
            fails += 1
            print(f"FAIL  {f.name}")
            for b in bad:
                print(f"        {b}")
            continue
        r = subprocess.run([sys.executable, str(f)], text=True, capture_output=True, timeout=1800)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            fails += 1
            print(f"FAIL  {f.name} (exit {r.returncode})")
            print("        " + "\n        ".join(out.strip().splitlines()[-12:]))
        elif not out.strip():
            fails += 1
            print(f"FAIL  {f.name}: exited 0 and printed NOTHING -- it ran no assertion")
        else:
            print(f"ok    {f.name}  {out.strip().splitlines()[-1][:88]}")
    print(f"\n{len(files) - fails}/{len(files)} test files pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
