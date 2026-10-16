"""Upload a pod's checkpoints, result rows and any un-uploaded episodes to S3, and VERIFY each landed.

AGENTS 25: /workspace is a pod volume and dies with the pod -- a 20M-transition dataset was already lost
that way. AGENTS 16: a success log line is not evidence, so every upload is re-HEADed and the printed size
comes from S3, not from the local file. Runs ON the pod (needs boto3 + creds from .fleet_secrets).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

_tl = threading.local()


def cli():
    c = getattr(_tl, "c", None)
    if c is None:
        c = _tl.c = boto3.Session(region_name="us-west-1").client("s3")
    return c


def md5(p: Path, limit=8 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while chunk := f.read(limit):
            h.update(chunk)
    return h.hexdigest()


def put(bucket: str, key: str, path: Path, skip_same_size: bool = True) -> tuple[str, int, str]:
    try:
        head = cli().head_object(Bucket=bucket, Key=key)
        if skip_same_size and int(head["ContentLength"]) == path.stat().st_size:
            return key, int(head["ContentLength"]), "already there"
    except Exception:                                        # noqa: BLE001
        pass
    cli().upload_file(str(path), bucket, key)
    size = int(cli().head_object(Bucket=bucket, Key=key)["ContentLength"])   # VERIFY, not assume
    return key, size, "uploaded" if size == path.stat().st_size else f"SIZE MISMATCH {size}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="zelda-rl")
    ap.add_argument("--prefix", required=True, help="e.g. zelda/save/rvs_consolidated/r0")
    ap.add_argument("--glob", action="append", required=True,
                    help="repo-relative glob(s) to save, e.g. 'checkpoints/kept/rvs_consolidated*.pt'")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--manifest", default="")
    a = ap.parse_args()

    files = sorted({f for g in a.glob for f in Path(".").glob(g) if f.is_file()})
    if not files:
        print(f"NOTHING MATCHED {a.glob} -- nothing saved", flush=True)
        return 2
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} file(s), {total/1e6:.1f} MB -> s3://{a.bucket}/{a.prefix}/", flush=True)

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(put, a.bucket, f"{a.prefix}/{f.as_posix()}", f): f for f in files}
        for fut, f in futs.items():
            try:
                key, size, how = fut.result()
                rows.append({"file": f.as_posix(), "key": key, "size": size, "status": how})
            except Exception as e:                           # noqa: BLE001
                rows.append({"file": f.as_posix(), "key": None, "size": 0, "status": f"FAILED {e}"})

    ok = [r for r in rows if r["status"] in ("uploaded", "already there")]
    bad = [r for r in rows if r not in ok]
    for r in rows[:12]:
        print(f"  {r['status']:<14} {r['size']/1e6:8.2f} MB  {r['file']}", flush=True)
    if len(rows) > 12:
        print(f"  ... {len(rows)-12} more", flush=True)
    print(f"VERIFIED IN S3: {len(ok)}/{len(rows)} file(s), {sum(r['size'] for r in ok)/1e6:.1f} MB",
          flush=True)
    for r in bad:
        print(f"  FAILED: {r['file']}: {r['status']}", flush=True)
    if a.manifest:
        import json
        Path(a.manifest).write_text(json.dumps(
            {"prefix": a.prefix, "rows": rows,
             "md5": {r["file"]: md5(Path(r["file"])) for r in ok if Path(r["file"]).stat().st_size < 2e8}},
            indent=2))
        print(f"manifest {a.manifest}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
