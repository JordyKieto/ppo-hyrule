"""Pool the held-out evaluator's rows and test one run against another. The answer to "is it better yet".

This existed as inline python retyped on every status request, which is how a comparison gets done slightly
differently each time and how a p-value never gets computed at all. AGENTS 17 and 21: the question you ask
repeatedly is a command, not a habit.

What it enforces, because each was got wrong by hand at least once:

- **Rows are grouped by `wandb_group`, never by update number.** A run resumed with `--init-from` restarts
  its update counter at 0, so v14's u00025 and v15's u00025 are different policies with the same name.
- **Episodes are pooled from the per-slot sub-rows**, whose `n` is exact, not from the row aggregate.
- **Predictability is reported with the denominator**: every rate carries n and a Wilson interval, and every
  comparison carries a one-sided Fisher exact p. A rate without n is not a measurement (AGENTS 1, 2, 7).
- **The hold-out rate is NOT reported as generalisation.** On the Gohma ladder the held-out slots are by
  construction all 5 HP and all >=368 units, so the number is collinear with distance. The band table is the
  honest decomposition and is what this prints.

    .venv/bin/python scripts/eval_compare.py --pull gohma-eval              # pull from the live eval pods
    .venv/bin/python scripts/eval_compare.py                               # whatever is already on disk
    .venv/bin/python scripts/eval_compare.py --baseline gohma-kl-v15 --compare gohma-kl-v16
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Kept next to the evaluator's own copy on purpose: this script must read rows written by an OLDER evaluator
# that did not record hp/dist in by_slot, so it cannot depend on those fields being present.
SLOT_HP = {1: 5, 2: 5, 3: 10, 4: 8, 5: 10, 6: 6, 7: 5, 8: 10, 9: 5, 10: 5}
# Imported, not copied: the horizontal distances live in zelda_rl/combat.py, measured by
# scripts/audit_gohma_slots.py. Four separate copies of this table existed and the one here was 3D,
# which put slot 2 in the wrong band (AGENTS 3).
from zelda_rl.combat import GOHMA_SLOT_DIST as SLOT_DIST  # noqa: E402
BANDS = (("near  <200u", 0, 200), ("mid 200-400u", 200, 400), ("far   >400u", 400, 10 ** 9))


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided P(arm1 >= observed) under equal rates. Table [[a wins, b losses], [c wins, d losses]]."""
    n = a + b + c + d
    if n == 0 or (a + c) == 0:
        return 1.0
    return sum(comb(a + c, x) * comb(b + d, a + b - x) / comb(n, a + b)
               for x in range(a, min(a + b, a + c) + 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pull(experiment: str, dest: Path) -> int:
    """rsync every eval pod's rows into dest/rank<N>.jsonl. Needs the fleet state file."""
    sys.path.insert(0, str(ROOT))
    from fleet.cli import R_REPO, SSH_KEY, load_state
    st = load_state(experiment)
    dest.mkdir(parents=True, exist_ok=True)
    got = 0
    for pod in st["pods"]:
        rsh = (f"ssh -i {SSH_KEY} -p {pod['port']} -o StrictHostKeyChecking=accept-new "
               f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20")
        r = subprocess.run(["rsync", "-rlptz", "--no-owner", "--no-group", "-e", rsh,
                            f"root@{pod['ip']}:{R_REPO}/results/eval/", str(dest / f"rank{pod['rank']}") + "/"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            got += 1
        else:
            print(f"  rank{pod['rank']}: no rows yet ({r.stderr.strip().splitlines()[-1][:60] if r.stderr else ''})")
    return got


def load(patterns: list[str]) -> list[dict]:
    """Read rows, honouring two directory markers.

    `.withdrawn` -- the rows in this directory are a RETRACTED measurement and are skipped. They stay on disk
    because a retraction is only useful next to the thing it retracts (AGENTS 27), and the first version of
    this script happily pooled 2831 of them into an aggregate, which is the precise failure the marker stops.

    `.run_label` -- names the run for rows written before the evaluator recorded `wandb_group`. Without it
    those rows land in one "unlabelled" bucket together regardless of which policy produced them.
    """
    rows, skipped = [], []
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            d = Path(f).parent
            mark = next((x / ".withdrawn" for x in (d, d.parent) if (x / ".withdrawn").exists()), None)
            if mark is not None:
                skipped.append(f)
                continue
            lab = next((x / ".run_label" for x in (d, d.parent) if (x / ".run_label").exists()), None)
            label = lab.read_text().strip() if lab is not None else None
            for line in open(f):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if label and not r.get("wandb_group"):
                        r["wandb_group"] = label
                        r["_labelled_by"] = str(lab.relative_to(ROOT))
                    rows.append(r)
    if skipped:
        print(f"skipped {len(skipped)} file(s) in .withdrawn directories (retracted measurements): "
              f"{', '.join(sorted({str(Path(f).parent.relative_to(ROOT)) for f in skipped}))}")

    # REJECT ROWS FROM THE ROUND-BASED EVALUATOR, BY CONTENT.
    #
    # Directory markers are not enough: `results/` was never in CODE_EXCLUDES, so every `fleet launch`
    # rsynced the Mac's withdrawn rows ONTO the pods, and pulling the pods' results back nested them under a
    # path the marker did not cover. Whack-a-mole with markers would have silently re-admitted retracted data.
    #
    # The round version round-robined the slot per ROUND and ran `--num-envs` episodes per round, so it leaves
    # a signature the work-queue version cannot produce: unequal per-slot counts, or fewer slots measured than
    # the row claims. The work-queue version runs exactly `episodes_per_slot` on every slot it lists.
    def round_based(r):
        bs = r.get("by_slot") or {}
        ns = {v["n"] for v in bs.values()}
        return (not bs) or len(ns) > 1 or len(bs) < len(r.get("slots") or bs)

    bad = [r for r in rows if round_based(r)]
    rows = [r for r in rows if not round_based(r)]
    if bad:
        print(f"rejected {len(bad)} row(s) from the ROUND-BASED evaluator (unequal per-slot counts, or fewer "
              f"slots measured than listed) -- a retracted measurement, PROVENANCE.md 3.2")

    # DEDUPLICATE. The same evaluation lands in several files -- a committed copy, a later re-pull from the
    # pod, and the pod's own jsonl -- so a naive pool counted v14 at 1551 episodes where it ran 559. That
    # inflates n and moves every p-value in the CONFIDENT direction, which is the worst way for a statistic
    # to be wrong. One evaluation is uniquely (checkpoint, rank, ts): same checkpoint re-evaluated later is a
    # genuinely new sample and keeps its own row.
    seen, uniq, dupes = set(), [], 0
    for r in rows:
        k = (r.get("checkpoint"), r.get("rank"), r.get("ts"))
        if k in seen:
            dupes += 1
            continue
        seen.add(k)
        uniq.append(r)
    if dupes:
        print(f"dropped {dupes} duplicate row(s) present in more than one file")
    return uniq


def pool(rows: list[dict]) -> dict[int, list]:
    """Episodes per slot, from the per-slot sub-rows whose n is exact."""
    out: dict[int, list] = {}
    for r in rows:
        for s, v in (r.get("by_slot") or {}).items():
            a = out.setdefault(int(s), [0, 0.0, 0.0, 0.0])
            a[0] += v["n"]
            a[1] += v["n"] * v.get("defeat_rate", 0.0)
            a[2] += v["n"] * v.get("boss_hp_taken", 0.0)
            a[3] += v["n"] * v.get("death_rate", 0.0)
    return out


def group_label(r: dict) -> str:
    """The run a row belongs to. Rows from before run_name was recorded fall back to a dated bucket."""
    return r.get("wandb_group") or f"<unlabelled {str(r.get('ts'))[:10]}>"


def band_rows(p: dict[int, list]):
    for name, lo, hi in BANDS:
        n = sum(p[s][0] for s in p if lo <= SLOT_DIST.get(s, -1) < hi)
        w = round(sum(p[s][1] for s in p if lo <= SLOT_DIST.get(s, -1) < hi))
        if n:
            yield name, w, n
    nf = sum(p[s][0] for s in p if SLOT_HP.get(s) == 10)
    wf = round(sum(p[s][1] for s in p if SLOT_HP.get(s) == 10))
    if nf:
        yield "FULL HP (10)", wf, nf
    n = sum(v[0] for v in p.values())
    w = round(sum(v[1] for v in p.values()))
    if n:
        yield "OVERALL", w, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default=None, help="rsync rows from this eval fleet first, e.g. gohma-eval")
    ap.add_argument("--rows", action="append", default=None,
                    help="glob for jsonl rows (repeatable); default: everything under results/eval/")
    ap.add_argument("--baseline", default=None, help="wandb_group to compare AGAINST (default: the 2nd newest)")
    ap.add_argument("--compare", default=None, help="wandb_group to judge (default: the newest)")
    ap.add_argument("--slots", action="store_true", help="also print the per-slot table")
    a = ap.parse_args()

    if a.pull:
        d = ROOT / "results/eval/pulled" / a.pull
        n = pull(a.pull, d)
        print(f"pulled from {n} pod(s) of {a.pull} -> {d.relative_to(ROOT)}\n")

    rows = load(a.rows or [str(ROOT / "results/eval/**/*.jsonl")])
    if not rows:
        print("no evaluation rows found")
        return 1
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(group_label(r), []).append(r)
    order = sorted(by, key=lambda g: max(str(r.get("ts") or "") for r in by[g]))

    print(f"{len(rows)} evaluation rows in {len(by)} run(s)\n")
    print(f"{'run':<26}{'evals':>6}{'episodes':>10}{'wins':>6}{'rate':>9}   newest row")
    for g in order:
        p = pool(by[g])
        n = sum(v[0] for v in p.values())
        w = round(sum(v[1] for v in p.values()))
        ts = max(str(r.get("ts") or "") for r in by[g])
        print(f"{g:<26}{len(by[g]):>6}{n:>10}{w:>6}{100 * w / max(n, 1):>8.2f}%   {ts}")

    comp = a.compare or order[-1]
    base = a.baseline or (order[-2] if len(order) > 1 else None)
    if base is None or base == comp:
        print(f"\nonly one run present ({comp}); nothing to compare against")
        return 0
    if comp not in by or base not in by:
        print(f"\nunknown run; available: {', '.join(order)}")
        return 1

    pc, pb = pool(by[comp]), pool(by[base])
    print(f"\n{comp}  vs  {base}\n")
    print(f"{'band':<14}{'new':>10}{'95% CI':>18}{'base':>11}{'95% CI':>18}{'p':>8}")
    bb = {k: (w, n) for k, w, n in band_rows(pb)}
    verdict = []
    for name, w, n in band_rows(pc):
        if name not in bb:
            continue
        w0, n0 = bb[name]
        lo, hi = wilson(w, n)
        lo0, hi0 = wilson(w0, n0)
        p = fisher_greater(w, n - w, w0, n0 - w0)
        print(f"{name:<14}{w:>4}/{n:<5}{100 * lo:>7.1f}-{100 * hi:<9.1f}{w0:>4}/{n0:<6}"
              f"{100 * lo0:>7.1f}-{100 * hi0:<9.1f}{p:>7.3f}")
        verdict.append((name, p, w / n, w0 / max(n0, 1), n))

    sig = [v for v in verdict if v[1] < 0.05]
    up = [v for v in verdict if v[2] > v[3]]
    print()
    if sig:
        print(f"SIGNIFICANT at p<0.05: {', '.join(f'{v[0].strip()} (p={v[1]:.3f})' for v in sig)}")
    else:
        print(f"NOT SIGNIFICANT in any band. {len(up)}/{len(verdict)} bands point the right way.")
        ov = next((v for v in verdict if v[0] == "OVERALL"), None)
        if ov:
            # How many more episodes, at the observed rate, before the overall band could reach p<0.05.
            w0, n0 = bb["OVERALL"]
            for mult in (2, 3, 5, 8, 12):
                n1 = ov[4] * mult
                if fisher_greater(round(ov[2] * n1), n1 - round(ov[2] * n1), w0, n0 - w0) < 0.05:
                    print(f"  at the observed {100 * ov[2]:.2f}%, ~{n1} episodes ({mult}x the current "
                          f"{ov[4]}) would reach p<0.05")
                    break
            else:
                print(f"  the observed {100 * ov[2]:.2f}% would need more than {ov[4] * 12} episodes to separate")

    if a.slots:
        print(f"\n{'slot':>5}{'HP':>4}{'dist':>6}{'n':>6}{'wins':>6}{'rate':>8}{'bossHP':>8}{'death':>8}")
        for s in sorted(pc, key=lambda k: SLOT_DIST.get(k, 0)):
            n, w, bh, dr = pc[s]
            print(f"{s:>5}{SLOT_HP.get(s, 0):>4}{SLOT_DIST.get(s, 0):>6}{n:>6}{w:>6.0f}"
                  f"{100 * w / n:>7.1f}%{bh / n:>8.2f}{100 * dr / n:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
