"""Build the auditable, versioned manifest for the human demonstration corpus.

Every claim about this dataset has to be re-derivable months later from the manifest alone (AGENTS 27), and
the corpus grew in place -- the 13-episode version produced published numbers (53.4% clone accuracy, 0.14
held-out damage) and the 100-episode version supersedes it. Both must stay identifiable, so each VERSION
records exactly which files it contains and a single digest over their md5s.

Per episode it records what the recorder already stamped into the npz -- `recorded_utc`, `git_commit`,
`obs_version`, the action-space size, the pad source -- plus the start state, start distance, outcome and an
md5. Nothing here is typed by hand; it is all read from the files.

    .venv/bin/python scripts/dataset_manifest.py                  # rebuild + print the digests
    .venv/bin/python scripts/dataset_manifest.py --upload         # and push to S3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# A version is a PREFIX of the corpus in filename order, which is also recording order. v1 is the 13 episodes
# that produced the first clone and the 0.14 held-out damage measurement; v2 is the full 100-game corpus.
VERSIONS = {"v1": 13, "v2": 100}


def row(f: Path) -> dict:
    z = np.load(f, allow_pickle=False)
    def g(k, default=None):
        return z[k].item() if k in z.files and z[k].shape == () else (z[k] if k in z.files else default)
    return {
        "file": f.name,
        "md5": hashlib.md5(f.read_bytes()).hexdigest(),
        "bytes": f.stat().st_size,
        "slot": int(z["slot"]),
        "start_gohma_state": str(g("start_gohma_state", "?")),
        "start_gohma_hp": int(g("start_gohma_hp", -1)),
        "start_dist_u": float(g("start_dist", -1.0)),
        "boss_damage": int(z["boss_damage"]),
        "win": bool(z["boss_defeated"]),
        "died": bool(g("died", False)),
        "steps": int(len(z["action"])),
        "length_s": float(g("length_s", -1.0)),
        "hz": float(g("hz", 10.0)),
        "obs_version": int(g("obs_version", -1)),
        "obs_dim": int(g("obs_dim", -1)),
        "n_actions": int(g("n_actions", -1)),
        "distinct_actions": int(len(np.unique(z["action"]))),
        "distinct_pad": int(len(np.unique(z["pad"], axis=0))),
        "recorded_utc": str(g("recorded_utc", "?")),
        "git_commit": str(g("git_commit", "?")),
        "pad_source": str(g("pad_source", "?")),
    }


def digest(rows: list[dict]) -> str:
    """One hash identifying a version: md5 over the sorted per-file md5s."""
    h = hashlib.md5()
    for m in sorted(r["md5"] for r in rows):
        h.update(m.encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human/gohma")
    ap.add_argument("--upload", action="store_true", help="also push the manifest to S3")
    a = ap.parse_args()
    d = Path(a.data)
    files = sorted(d.glob("ep*.npz"))
    if not files:
        print(f"no ep*.npz in {d}")
        return 1
    rows = [row(f) for f in files]

    versions = {}
    for name, n in VERSIONS.items():
        if len(rows) >= n:
            sub = rows[:n]
            versions[name] = {
                "episodes": n, "wins": sum(1 for r in sub if r["win"]),
                "steps": sum(r["steps"] for r in sub),
                "slots": sorted({r["slot"] for r in sub}),
                "prefight_starts": sum(1 for r in sub if r["start_gohma_state"] == "ceiling_wait"),
                "first_recorded_utc": min(r["recorded_utc"] for r in sub),
                "last_recorded_utc": max(r["recorded_utc"] for r in sub),
                "git_commits": sorted({r["git_commit"] for r in sub}),
                "digest_md5": digest(sub),
                "files": [r["file"] for r in sub],
            }
    man = {
        "dataset": "human Gohma demonstrations with controller inputs",
        "s3": "s3://zelda-rl/zelda/human/gohma/",
        "recorder": "scripts/record_human.py",
        "verifier": "scripts/verify_human_dataset.py",
        "built_by": "scripts/dataset_manifest.py",
        "total": len(rows), "wins": sum(1 for r in rows if r["win"]),
        "steps": sum(r["steps"] for r in rows),
        "digest_md5": digest(rows),
        "versions": versions,
        "episodes": rows,
    }
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"{man['total']} episodes, {man['wins']} wins, {man['steps']:,} steps")
    print(f"corpus digest {man['digest_md5']}")
    for name, v in versions.items():
        print(f"  {name}: {v['episodes']:>3} episodes, {v['wins']:>3} wins, {v['steps']:>6,} steps, "
              f"slots {v['slots']}, {v['prefight_starts']} pre-fight, digest {v['digest_md5']}")
        print(f"      recorded {v['first_recorded_utc']} .. {v['last_recorded_utc']}")
        print(f"      recorder commits: {', '.join(v['git_commits'])}")
    if a.upload:
        import boto3
        boto3.Session(profile_name="race-wm", region_name="us-west-1").client("s3").upload_file(
            str(d / "MANIFEST.json"), "zelda-rl", "zelda/human/gohma/MANIFEST.json")
        print("manifest -> s3://zelda-rl/zelda/human/gohma/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
