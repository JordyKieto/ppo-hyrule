#!/usr/bin/env python3
"""Rebuild manifest.jsonl for a collected dataset from the episodes actually on disk.

WHY THIS IS NEEDED. `load_many()` enumerates episodes from manifest.jsonl, not from the directory, so an
episode that exists but is unlisted is silently invisible -- the load succeeds, reports a smaller number, and
nothing says data was dropped. Two separate bugs in the v5 collection left 27% of it in that state:

  1. The collector opened manifest.jsonl with mode "w". Relaunching a pod to fill its backlog TRUNCATED the
     manifest, so the first run's episodes -- still on disk and still in S3 -- vanished from the listing.
     Measured on v5: 9,775 of 79,692 sharded episodes (12.3%), across 43 of 109 shards.
  2. manifest.jsonl was uploaded to S3 only in the collector's `finally` block, while episodes uploaded as
     they were written. A pod killed before it finished left its episodes durable and its manifest stale or
     absent. The 16,415 episodes under the flat pre-sharding prefix have no manifest at all.

Both are fixed in collect_transitions.py, but the data already collected still needs its index rebuilt.

The filename carries what the loader needs: ep<index>_G<global_update>_s<slot>_<outcome>.npz. `slot` and
`file` are the only fields load_many reads; the rest are recovered for the reporting paths. Rows already in
the manifest are KEPT AS IS -- they carry per-episode combat metadata (damage dealt, kill times, lockons)
that the filename does not encode and that no rebuild can invent.

    scripts/rebuild_manifests.py datasets/transitions/v5            # every shard under it, plus a flat tree
    scripts/rebuild_manifests.py datasets/transitions/v5 --check    # report only, exit 1 if anything is unlisted
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

NAME = re.compile(r"^ep(\d+)_G(\d+)_s(\d+)_(.+)\.npz$")


def rebuild(d: Path, check: bool, quiet: bool = False) -> tuple[int, int]:
    """Returns (kept, recovered) for one dataset directory."""
    eps = d / "episodes"
    if not eps.is_dir():
        return (0, 0)
    man = d / "manifest.jsonl"
    rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()] if man.exists() else []
    listed = {r["file"].split("/")[-1] for r in rows}
    disk = sorted(p.name for p in eps.glob("*.npz"))
    missing = [n for n in disk if n not in listed]
    gone = [n for n in listed if not (eps / n).exists()]
    if gone and not quiet:
        print(f"  {d.name}: {len(gone)} manifest rows point at files NOT on disk (incomplete pull?)")
    if not missing:
        return (len(rows), 0)
    if check:
        return (len(rows), len(missing))

    new = []
    for n in missing:
        m = NAME.match(n)
        if not m:
            print(f"  {d.name}: cannot parse {n!r}, skipping")
            continue
        idx, gupd, slot, end = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        try:                                      # steps is the only field worth opening the file for
            with np.load(eps / n) as z:
                steps = int(len(z["action"]))
        except Exception as e:                    # noqa: BLE001  a truncated download, not a reason to stop
            print(f"  {d.name}: {n} unreadable ({e}), skipping")
            continue
        new.append({"file": f"episodes/{n}", "index": idx, "global_update": gupd, "slot": slot,
                    "steps": steps, "end": end, "success": end == "success", "died": end == "death",
                    "recovered": True})
    # rewrite whole-file rather than append: the manifest must not contain the truncated prefix twice if
    # this is run again, and rows are matched by filename.
    man.write_text("".join(json.dumps(r) + "\n" for r in rows + new))
    return (len(rows), len(new))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset dir; every subdir with an episodes/ is rebuilt, and the root too")
    ap.add_argument("--check", action="store_true", help="report only, exit 1 if any episode is unlisted")
    a = ap.parse_args()

    root = Path(a.root)
    dirs = ([root] if (root / "episodes").is_dir() else []) + \
           sorted(p for p in root.iterdir() if p.is_dir() and (p / "episodes").is_dir())
    if not dirs:
        sys.exit(f"no dataset directory with an episodes/ under {root}")

    kept = rec = 0
    for d in dirs:
        k, r = rebuild(d, a.check)
        kept += k; rec += r
        if r:
            print(f"  {d.name if d != root else '<root>'}: {k} listed + {r} {'unlisted' if a.check else 'recovered'}")
    tot = kept + rec
    verb = "would recover" if a.check else "recovered"
    print(f"\n{len(dirs)} dirs | manifest had {kept:,} | {verb} {rec:,} | total {tot:,} episodes"
          + (f"  ({rec / max(tot, 1):.1%} of the dataset was invisible to load_many)" if rec else ""))
    if a.check and rec:
        sys.exit(1)


if __name__ == "__main__":
    main()
