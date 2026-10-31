"""Run the launch LOCALLY, as far as the first optimiser step, before any pod is touched.

EXPERIMENT-CONTRACT: exempt -- a validator; it trains nothing and evaluates nothing.

WHY THIS EXISTS. On 2026-10-31 three separate failures each killed a whole fleet and each was found by hand,
on live pods, minutes apart:

  1. an `EXPERIMENT-CONTRACT: exempt` note pasted mid-function made `zelda_rl/policy_io.py` unimportable.
     Ten ranks trained four epochs and died at their first online eval. (AGENTS §44)
  2. the contract gate, newly made to walk directories, walked into `.venv/`, flagged torch's own
     `storage.py` and REFUSED THE BOOTSTRAP. Seven pods had no Dolphin. (AGENTS §44b)
  3. `fleet launch` injects five args `train_rvs.py` did not accept, so all seven ranks died on argparse in
     under a second -- and the launcher logged "launched" for every one. (AGENTS §45)

Every one was visible on the Mac, in seconds, before anything was pushed. Nothing looked. Each cost a
bring-up: ~30 minutes of wall clock and a fleet's hourly rate, repeatedly, for errors a parse would catch.

WHAT IT CHECKS, in the order the failures happened:

  A. every tracked .py PARSES, and the experiment contract passes on the REAL push argument (`REPO/`)
  B. every module under zelda_rl/ IMPORTS -- which is what #1 broke and what no test covered
  C. the trainer accepts the EXACT argv `fleet launch` will compose, including the five it injects

(C) runs the real script through runpy with `argparse.ArgumentParser.parse_args` patched to exit the moment
it succeeds, so it needs no cooperation from the trainer and no dataset: an unrecognised argument, a missing
import and a bad default all surface, and nothing trains.
"""
from __future__ import annotations

import argparse
import io
import os
import pkgutil
import runpy
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class _Parsed(SystemExit):
    """Raised instead of returning, so the trainer stops the instant its args are known to be valid."""


def check_parses() -> list[str]:
    from zelda_rl.experiment_contract import check, expand_py
    pushed = expand_py([Path(str(REPO) + "/")])        # exactly what fleet.cli.rsync passes
    bad = check(pushed)
    out = [f"{len(pushed)} tracked .py files inspected (the real push argument)"]
    for name, vs in sorted(bad.items()):
        out += [f"  VIOLATION {name}: {v}" for v in vs]
    return out if bad else out + ["  contract clean"]


def check_imports() -> list[str]:
    """Import every zelda_rl module. A syntax error here is what killed ten ranks at their first eval."""
    import zelda_rl
    bad, n = [], 0
    for m in pkgutil.iter_modules(zelda_rl.__path__, "zelda_rl."):
        n += 1
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                __import__(m.name)
        except Exception as e:                                           # noqa: BLE001
            bad.append(f"  IMPORT FAILED {m.name}: {type(e).__name__}: {e}")
    return [f"{n} zelda_rl modules imported"] + (bad or ["  all import"])


def injected_argv(script: str, train_args: str, *, rank: int = 0, world: int = 1,
                  envs: int = 8, experiment: str = "validate", project: str = "zelda-rl-rvs") -> list[str]:
    """The argv `fleet launch` composes. Kept beside the launcher's own string so they cannot drift."""
    run_name = f"{experiment}-{Path(script).stem}-r{rank}"
    return ["--run-name", run_name, "--num-envs", str(envs),
            "--slots-dir", "assets/states", "--video-backend", "Null", "--seed", str(rank),
            "--wandb-project", project, "--wandb-group", experiment,
            "--fleet-experiment", experiment, "--fleet-rank", str(rank),
            "--fleet-world-size", str(world), "--fleet-launch-id", "0"] + train_args.split()


def check_args(script: str, train_args: str) -> list[str]:
    path = REPO / "scripts" / script
    if not path.is_file():
        return [f"  NO SUCH SCRIPT: {path}"]
    argv = injected_argv(script, train_args)
    real_parse, real_known = argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args

    def stop(self, args=None, namespace=None):                           # noqa: ANN001
        # REPLICATE argparse's own strictness. The first version of this patch called parse_known_args and
        # returned -- which TOLERATES unrecognised arguments, i.e. it could never detect the one failure it
        # was written for (the test caught this; it is the same "a check that cannot fail" shape as §36 #2).
        ns, argv = real_known(self, args, namespace)
        if argv:
            self.error("unrecognized arguments: %s" % " ".join(argv))     # exits 2, as the real one does
        raise _Parsed(0)

    def stop_known(self, args=None, namespace=None):                     # noqa: ANN001
        real_known(self, args, namespace)
        raise _Parsed(0)

    argparse.ArgumentParser.parse_args = stop
    argparse.ArgumentParser.parse_known_args = stop_known
    old_argv, buf = sys.argv, io.StringIO()
    sys.argv = [str(path)] + argv
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_path(str(path), run_name="__main__")
        return [f"  {script}: parsed its args but never reached parse_args (no argparse?)"]
    except _Parsed:
        return [f"  {script}: accepts all {len(argv)} args, including the 13 fleet injects"]
    except SystemExit as e:                                              # argparse's own exit(2)
        msg = [l for l in buf.getvalue().splitlines() if "error:" in l or "unrecognized" in l]
        return [f"  {script}: REJECTED ITS OWN LAUNCH ARGV (exit {e.code})"] + [f"      {m}" for m in msg[:3]]
    except Exception as e:                                               # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        return [f"  {script}: FAILED BEFORE PARSING: {type(e).__name__}: {e}"] + [f"      {l}" for l in tb[-3:]]
    finally:
        sys.argv = old_argv
        argparse.ArgumentParser.parse_args = real_parse
        argparse.ArgumentParser.parse_known_args = real_known


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=None, help="scripts/<name>.py the launch will run")
    ap.add_argument("--train-args", default="", help="the --train-args string, verbatim")
    a = ap.parse_args()

    sections = [("A. parses + experiment contract", check_parses()),
                ("B. zelda_rl imports", check_imports())]
    if a.script:
        sections.append((f"C. {a.script} accepts the launcher's argv", check_args(a.script, a.train_args)))

    bad = 0
    for title, lines in sections:
        print(f"\n{title}")
        for l in lines:
            print(l if l.startswith(("  ", "\t")) else f"  {l}")
            if any(k in l for k in ("VIOLATION", "FAILED", "REJECTED", "NO SUCH")):
                bad += 1
    if bad:
        print(f"\nLAUNCH VALIDATION FAILED: {bad} problem(s). Nothing was pushed. Fix these first -- each "
              f"one of them has already cost a fleet bring-up (see the module docstring).")
        return 1
    print("\nlaunch validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
