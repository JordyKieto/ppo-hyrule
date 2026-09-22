#!/usr/bin/env python3
"""Recompute per-episode combat analytics from the stored state trajectory.

WHY. `manifest.jsonl` carried per-episode combat metadata -- damage_dealt, defeated, kill_times_s, hp_lost,
lockons, targets_start -- written by `CombatTracker` as the episode was collected. The collector opened the
manifest with mode "w", so every backlog-fill relaunch truncated it; `scripts/rebuild_manifests.py` recovered
the episode LIST from filenames, but the filename encodes only index, checkpoint tier, slot and outcome. The
analytics were gone from the index.

They are not gone from the DATA. Every episode .npz stores state[T+1, 363], which contains Link's HP, every
actor row's HP and presence, the targeted row, and the scene/room -- everything those counters are computed
from. So they are recomputed here directly from the arrays rather than by replaying CombatTracker: the
tracker needs Link's absolute `action_fn`, and the collector stored only `action_fn_rel` (fn minus the actor's
own update fn), so a faithful replay is not reconstructible. Direct formulas are also checkable, which a
replay would not be.

VALIDATION IS THE POINT. 69,917 of 103,716 episodes still have their ORIGINAL manifest metadata, so every
formula here is scored against ground truth on those before being trusted on the rest. --validate prints an
exact-match rate per field, writes nothing, and exits 1 if any field is below 99%. Fields that do not
validate are reported and left ABSENT rather than filled with a plausible-looking guess.

Measured on a 20,000-episode sample (`--validate --sample 20000`), every written field is exact:

    field           exact     best constant     exact on non-trivial rows
    targets_start   100.00%   59.47%            100.00% (n=19,984)
    defeated        100.00%   32.44%            100.00% (n=16,500)
    damage_dealt    100.00%   28.32%            100.00% (n=16,892)
    hp_start        100.00%   80.05%            100.00% (n=20,000)
    hp_end          100.00%   36.62%            100.00% (n=19,044)
    hp_lost         100.00%   42.71%            100.00% (n=11,458)
    died            100.00%   95.22%            100.00% (n=956)
    success         100.00%   61.16%            100.00% (n=12,232)
    lockons         100.00%   65.83%            100.00% (n=16,908)
    kill_times_s    100.00%   17.50%            100.00% (n=16,500)
    steps           100.00%   29.24%            100.00% (n=20,000)

The "best constant" column is AGENTS.md §1's do-nothing control: 65.83% of episodes never lock on, so a
formula that always answered 0 would have looked two thirds right. `hp_healed` is NOT written -- the tracker
gates it on `not prev.dead` and no original row carries it, so it has no ground truth; `blocks`, `dodges` and
`return` need Link's absolute `action_fn` and the reward weights, so they are not recoverable at all.

    scripts/recover_analytics.py datasets/transitions/v5 --validate          # score formulas, write nothing
    scripts/recover_analytics.py datasets/transitions/v5 --workers 8         # fill in the recovered rows
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import BABA_ID, LI, RI, _split  # noqa: E402

SPS = 10.0            # steps_per_second, from dataset.json
LOCKON_CAP = 6        # CombatConfig.lockon_cap
FIELDS = ("targets_start", "defeated", "damage_dealt", "hp_start", "hp_end", "hp_lost",
          "died", "success", "lockons", "kill_times_s", "steps")


def analytics(state: np.ndarray, n_act: int, end: str | None = None) -> dict:
    """Per-episode counters from the state trajectory alone."""
    lk, rows, _ = _split(state)
    hp = lk[:, LI["hp"]].astype(np.int64)
    d = np.diff(hp)
    pres = rows[:, :, RI["present"]] > 0.5
    rhp = rows[:, :, RI["hp"]].astype(np.int64)
    baba = rows[:, :, RI["id"]].astype(np.int64) == BABA_ID

    # A target is a Baba alive in its row at episode start. Rows are stable across time, so a row index
    # identifies the same actor throughout.
    tgt = pres[0] & baba[0] & (rhp[0] > 0)
    # Damage dealt: HP a target row LOST while present. A row going absent is a despawn, not damage, so
    # transitions into absence contribute nothing.
    both = pres[:-1] & pres[1:] & tgt[None, :]
    drop = np.where(both, np.maximum(rhp[:-1] - rhp[1:], 0), 0)
    # Defeated: a target row that reaches 0 HP while present, counted once.
    dead = (rhp <= 0) & pres & tgt[None, :]
    ever = dead.any(0)
    kill_t = [float(np.argmax(dead[:, j]) / SPS) for j in np.flatnonzero(ever)]

    # Lock-ons, matching CombatTracker exactly: it counts a step where `cur.targeted` IS a live target and
    # `prev.targeted` is NOT -- both tested against the LIVE SET AT THE CURRENT STEP -- and caps at
    # cfg.lockon_cap. A live target is a start-of-episode target that is present and not yet defeated, so
    # holding lock on the same Baba counts once, and locking a non-target or a corpse never counts. Counting
    # "targeted_row went from -1 to >=0" instead scored 32% against ground truth.
    #
    # The last step is the subtle part. `combat.py` puts the whole lock-on block behind `if not terminated:`,
    # so on the step that ends the episode by death, left_room or success the lock-on is NOT counted; a
    # timeout does not set `terminated`, so there it IS. Ignoring that scored 99.96%, and every single
    # mismatch was an `end=death` episode recomputed exactly one too high. `end` comes from the filename, so
    # recovered rows have it too.
    tr = lk[:, LI["targeted_row"]].astype(np.int64)
    defeated_before = np.cumsum(dead, 0) > 0                      # sticky, like Track.defeated
    live = tgt[None, :] & pres & (rhp > 0) & ~np.r_[np.zeros((1, dead.shape[1]), bool), defeated_before[:-1]]
    R = rows.shape[1]
    def is_live(row_idx, t):
        ok = (row_idx >= 0) & (row_idx < R)
        return ok & live[t, np.clip(row_idx, 0, R - 1)]
    last = len(state) - (1 if end in ("death", "left_room", "success") else 0)
    n_lock = 0
    for t in range(1, last):
        if is_live(tr[t], t) and not is_live(tr[t - 1], t):
            n_lock += 1
            if n_lock >= LOCKON_CAP:
                break
    return {"targets_start": int(tgt.sum()),
            "defeated": int(ever.sum()),
            "damage_dealt": int(drop.sum()),
            "hp_start": int(hp[0]), "hp_end": int(hp[-1]),
            # NOT hp_healed: the tracker gates it on `not prev.dead`, and no original row carries it, so
            # there is no ground truth to score it against. Doctrine: unvalidatable means unwritten.
            "hp_lost": int(np.maximum(-d, 0).sum()),
            "died": bool(hp[-1] <= 0),
            "success": bool(ever.sum() == tgt.sum() and tgt.sum() > 0),
            "lockons": int(n_lock),
            "kill_times_s": sorted(kill_t),
            "steps": int(n_act)}


def one(args):
    d, row = args
    try:
        with np.load(Path(d) / row["file"]) as z:
            return row, analytics(z["state"], len(z["action"]), row.get("end"))
    except Exception as e:  # noqa: BLE001
        return row, {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--validate", action="store_true", help="score against rows that kept their originals")
    ap.add_argument("--sample", type=int, default=4000, help="episodes to score under --validate")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root = Path(a.root)
    dirs = ([root] if (root / "episodes").is_dir() else []) + \
           sorted(p for p in root.iterdir() if p.is_dir() and (p / "episodes").is_dir())
    jobs, n_orig, n_rec = [], 0, 0
    for d in dirs:
        for l in (d / "manifest.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            r["_dir"] = str(d)
            # A row has ORIGINALS only if it carries the collector's metadata and was NOT filled by this
            # script. Keying on "damage_dealt" alone was right exactly once: after the first write every row
            # has it, and --validate would then score the recovered rows against their own output and report a
            # meaningless 100%. The post-write re-validation is the check that the originals survived, so it
            # has to keep selecting the same 69,917 rows.
            orig = "damage_dealt" in r and "analytics_recovered" not in r
            n_orig += orig
            n_rec += not orig
            if a.validate:
                if orig:
                    jobs.append((str(d), r))
            else:
                jobs.append((str(d), r))
    print(f"{len(dirs)} dirs | {n_orig:,} rows with originals | {n_rec:,} rows missing analytics")

    if a.validate:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in rng.choice(len(jobs), min(a.sample, len(jobs)), replace=False)]
        print(f"scoring {len(jobs):,} episodes that kept their originals...")
        # AGENTS.md §1: an exact-match rate is not a result until its trivial control is run. A formula that
        # always returns 0 already scores well on `lockons` because most episodes never lock on, so every rate is
        # printed next to the best CONSTANT predictor's rate and to the rate on the non-trivial rows only.
        pairs: dict[str, list] = {f: [] for f in FIELDS}
        bad: dict[str, list] = {f: [] for f in FIELDS}
        tot = err = 0
        with ProcessPoolExecutor(a.workers) as ex:
            for row, got in ex.map(one, jobs, chunksize=16):
                if "error" in got:
                    err += 1
                    continue
                tot += 1
                for f in FIELDS:
                    if f in row and f in got:
                        pairs[f].append((row[f], got[f]))
                        if not bool(row[f] == got[f]) and len(bad[f]) < 6:
                            bad[f].append((row["file"], row[f], got[f], row.get("end"), row.get("steps")))
        print(f"\n{tot:,} scored ({err} unreadable)")
        print(f"  {'field':16s} {'n':>7s} {'exact':>8s} {'const':>8s} {'nontrivial':>12s}")
        worst = 1.0
        for f in FIELDS:
            p = pairs[f]
            if not p:
                print(f"  {f:16s} {'-':>7s}   no originals to score against")
                continue
            o = [repr(x) for x, _ in p]
            ok = sum(1 for x, y in p if bool(x == y)) / len(p)
            const = max(o.count(v) for v in set(o)) / len(o)          # best constant predictor
            nt = [(x, y) for x, y in p if x not in (0, False, None) and x != []]
            ntr = sum(1 for x, y in nt if bool(x == y)) / len(nt) if nt else float("nan")
            print(f"  {f:16s} {len(p):7,d} {ok:8.2%} {const:8.2%} "
                  f"{ntr:11.2%} (n={len(nt):,})")
            worst = min(worst, ok)
        for f in FIELDS:
            for file, x, y, end, steps in bad[f]:
                print(f"    MISMATCH {f}: {file} original={x!r} recomputed={y!r} end={end} steps={steps}")
        # §21: the doctrine is a check, not a paragraph. A field below this bar must not be written as
        # recovered, so fail loudly rather than leaving the caller to read the table.
        low = [f for f in FIELDS if pairs[f] and
               sum(1 for x, y in pairs[f] if bool(x == y)) / len(pairs[f]) < 0.99]
        if low:
            print(f"\nFAIL: below 99% exact match, must not be written as recovered: {', '.join(low)}")
            return 1
        print("\nOK: every field above validates at >=99% exact match")
        return 0

    out_by_dir: dict[str, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        done = 0
        for row, got in ex.map(one, jobs, chunksize=32):
            d = row.pop("_dir")
            if "error" not in got:
                # never overwrite an original: only fill what is absent, and tag ONLY the rows that were
                # actually filled. Tagging every row would put analytics_recovered on the 69,917 rows whose
                # numbers are the collector's own -- a false provenance claim in the file the pod trusts.
                filled = [k for k in got if k not in row]
                for k in filled:
                    row[k] = got[k]
                if filled:
                    row["analytics_recovered"] = True
            out_by_dir.setdefault(d, []).append(row)
            done += 1
            if done % 20000 == 0:
                print(f"  {done:,}/{len(jobs):,}")
    for d, rows in out_by_dir.items():
        rows.sort(key=lambda r: r["file"])
        (Path(d) / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {sum(len(v) for v in out_by_dir.values()):,} rows across {len(out_by_dir)} manifests")


if __name__ == "__main__":
    raise SystemExit(main())
