"""The gate that stops a non-durable experiment reaching a pod.

Twice this project paid for results and then lost them, and in both cases the tool that would have saved
them already existed -- the script just did not call it:

  2026-10-15  a 34-pod fleet vanished with FIFTEEN arms' checkpoints        -> zelda_rl/ckpt_s3.py existed
  2026-10-16  3,118 held-out RvS episodes lived only in print() on a pod    -> W&B and S3 both existed

Rules in a document did not prevent either (AGENTS 21/33/36 say exactly that). So the contract is checked
where code CROSSES ONTO A POD: `fleet.cli.rsync` calls `enforce()` on every `scripts/*.py` it pushes and
refuses the push on a violation. There is no step to remember, and the override is named and loud.

    parses at all      -> ast.parse, before anything else, and not exemptable
    saves checkpoints  -> must back them up with ckpt_s3
    runs evaluations   -> must report through zelda_rl.eval_log.EvalSink (or eval_fleet's own wandb path)
    opens Dolphin      -> must pin torch to one thread (AGENTS 28)

Scripts written before the contract are listed in `experiment_contract_legacy.txt`: they are reported, not
enforced, so the check passes on a healthy tree and only fires on code that is new or touched. A check that
flags correct work is ignored within a day (AGENTS 36), and a check that cannot fire is decoration -- both
failure modes are covered by tests/test_experiment_contract.py.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_FILE = Path(__file__).with_name("experiment_contract_legacy.txt")
EXEMPT = re.compile(r"EXPERIMENT-CONTRACT:\s*exempt")
OVERRIDE = "ZRL_FORCE_CONTRACT"
VENDORED = {".venv", "venv", "site-packages", "node_modules", "__pycache__",
            ".git", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache"}
HARD = "[not overridable] "                 # a file that cannot parse cannot run; see syntax_error()


def legacy() -> set[str]:
    if not LEGACY_FILE.is_file():
        return set()
    return {l.split("#")[0].strip() for l in LEGACY_FILE.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")} - {""}


def saves_checkpoints(src: str) -> bool:
    return "torch.save(" in src and ("--out" in src or "checkpoint" in src.lower())


def runs_evals(src: str) -> bool:
    return ("EVAL epoch" in src or "per_episode_out" in src or "per-episode-out" in src
            or re.search(r"\bef\.evaluate\(|\bevaluate\(ckpt", src) is not None)


def opens_dolphin(src: str) -> bool:
    return "make_env(" in src or "DOLPHIN_BIN" in src


def reports_metrics(src: str) -> bool:
    """A trainer that prints per-epoch held-out numbers is reporting an evaluation, whatever it calls it.

    The first version of this file only looked for "EVAL epoch", so it passed `train_wm_obs.py` -- whose
    per-epoch skill-vs-persistence and per-event accuracy lines were, that same day, lost with the pods
    they were printed on. If a number is worth printing every epoch it is worth keeping.
    """
    if not saves_checkpoints(src):
        return False
    return bool(re.search(r"skill vs persistence|better than doing nothing|val[_ ]?(acc|loss)|"
                          r"held-?out|balanced", src))


def syntax_error(src: str, name: str) -> str | None:
    """The cheapest check there is, and the one that was missing.

    An `EXPERIMENT-CONTRACT: exempt` note was pasted into the MIDDLE of a function in
    `zelda_rl/policy_io.py`, which did two things at once: it made the module unimportable, and -- because
    the exemption was a regex over raw text -- it silenced every other check in this file. All ten ranks of
    `gohma-rvsv15` trained four epochs, synced ten ranks, and then died on `IndentationError` at their first
    online eval, because the eval path imports that module. The gate they crossed read the broken file and
    said nothing.

    So syntax is checked FIRST, it is not exemptable, and it is not overridable: a file that does not parse
    cannot run, and there is no reason to put it on a pod. It also makes the exemption marker honest --
    a marker in a position that breaks the parse now fails here instead of disabling the contract.
    """
    try:
        ast.parse(src, filename=name)
    except SyntaxError as e:                                     # noqa: PERF203
        return f"does not parse: {type(e).__name__}: {e.msg} (line {e.lineno})"
    return None


def violations(path: Path) -> list[str]:
    """What this one file breaks. Empty list = compliant (or exempt/legacy)."""
    try:
        src = path.read_text()
    except Exception:                                            # noqa: BLE001
        return []
    syn = syntax_error(src, path.name)
    if syn is not None:
        return [HARD + syn]
    if EXEMPT.search(src) or path.name in legacy():
        return []
    out = []
    if saves_checkpoints(src) and "ckpt_s3" not in src:
        out.append("writes checkpoints with no S3 backup: add `from zelda_rl import ckpt_s3` and "
                   "`ckpt_s3.upload(path, run_name, step)` beside every torch.save")
    if (runs_evals(src) or reports_metrics(src)) and "EvalSink" not in src and "wandb" not in src:
        out.append("evaluates into stdout only: use zelda_rl.eval_log.EvalSink (jsonl + S3 + W&B in one "
                   "call) so the measurement outlives the pod")
    if opens_dolphin(src) and "set_num_threads" not in src and "torch-threads" not in src:
        out.append("opens Dolphin without torch.set_num_threads(1) (AGENTS 28): the emulator starves and "
                   "the env raises 'emulation stalled'")
    return out


def expand_py(paths):
    """Every .py under `paths` that is OURS. See the comment at the fleet.cli.rsync call site.

    Scoped to git-tracked files, because the code push is `rsync [REPO/]` -- the whole repo root -- and the
    first version of this walker went into `.venv/`, flagged **torch's own storage.py** for not backing its
    checkpoints up to S3, and REFUSED THE BOOTSTRAP. Seven pods sat with no Dolphin and no venv as a result.
    That is §36's "a check that flags correct work is ignored within a day", one step worse: it blocked a
    deploy instead of merely being noise.

    rsync drops `.venv/` via CODE_EXCLUDES and git ignores it, so "tracked by git" is the same set as "code
    we wrote and are sending", and it cannot drift from the excludes the way a second hand-kept list would.
    VENDORED is a belt-and-braces fallback for a tree git cannot answer for (the tests use a temp dir).
    """
    import subprocess
    out = []
    for p in paths:
        p = Path(str(p).rstrip("/"))
        if p.is_dir():
            got = []
            try:
                r = subprocess.run(["git", "-C", str(p), "ls-files", "-z", "--", "*.py"],
                                   capture_output=True, text=True, timeout=60)
                if r.returncode == 0:
                    got = [p / f for f in r.stdout.split("\0") if f.endswith(".py")]
            except Exception:                                    # noqa: BLE001
                got = []
            if not got:                                          # not a git tree (or git unavailable)
                got = [f for f in sorted(p.rglob("*.py"))
                       if not VENDORED.intersection(f.parts)]
            out += [f for f in got if f.is_file()]
        elif p.suffix == ".py":
            out.append(p)
    return out


def check(paths) -> dict[str, list[str]]:
    bad = {}
    for p in paths:
        p = Path(p)
        if p.suffix != ".py":
            continue
        v = violations(p)
        if v:
            bad[p.name] = v
    return bad


def enforce(paths, *, what: str = "push") -> None:
    """Raise unless every path satisfies the contract. Called from the deploy path, not from a checklist."""
    bad = check(paths)
    if not bad:
        return
    lines = [f"REFUSING TO {what.upper()}: {len(bad)} file(s) break the experiment contract "
             f"(zelda_rl/experiment_contract.py)"]
    for name, vs in sorted(bad.items()):
        lines.append(f"  {name}")
        for v in vs:
            lines.append(f"      - {v}")
    lines.append(f"  Fix it, add an `EXPERIMENT-CONTRACT: exempt -- <reason>` line, or set {OVERRIDE}=1 to "
                 f"override deliberately ({HARD.strip()} findings ignore the override).")
    msg = "\n".join(lines)
    hard = any(v.startswith(HARD) for vs in bad.values() for v in vs)
    if os.environ.get(OVERRIDE) == "1" and not hard:
        print(msg + f"\n  {OVERRIDE}=1 -- proceeding anyway.", flush=True)
        return
    raise SystemExit(msg)
