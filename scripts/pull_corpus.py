"""Pull a corpus shard from S3 in PARALLEL. One-at-a-time downloads cost ~3 minutes per pod per 600 files.

boto3 is not thread-safe at the client level across threads for all operations, but `download_file` on a
client created per thread is, so each worker gets its own client. 16 workers takes a 600-file pull from
~3 minutes to ~15 seconds, and a 1,830-file shard from ~10 minutes to ~45 seconds.

    .venv/bin/python scripts/pull_corpus.py --prefix zelda/ppo/gohma_steps/ --out datasets/ppo/pooled \
        --wins 120 --losses 500 --stride-seed 3 --workers 16
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

_local = threading.local()


def client(region: str):
    if not hasattr(_local, "c"):
        _local.c = boto3.Session(region_name=region).client("s3")
    return _local.c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="zelda-rl")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default="us-west-1")
    ap.add_argument("--wins", type=int, default=0, help="take this many files whose name marks a win")
    ap.add_argument("--losses", type=int, default=0, help="take this many non-win files, strided")
    ap.add_argument("--stride-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c0 = boto3.Session(region_name=a.region).client("s3")
    keys = [o["Key"] for p in c0.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=a.prefix)
            for o in p.get("Contents", []) if o["Key"].endswith(".npz")]
    wins = [k for k in keys if "success" in k]
    loss = [k for k in keys if "success" not in k]
    take = (wins[:a.wins] if a.wins else wins) + \
           (loss[a.stride_seed::max(1, len(loss) // max(a.losses, 1))][:a.losses] if a.losses else loss)
    todo = [k for k in take if not (out / k.split("/")[-1]).exists()]
    print(f"corpus {len(keys)} files; taking {len(take)}; {len(todo)} to download with {a.workers} workers",
          flush=True)
    t0, done, lock = time.time(), [0], threading.Lock()

    def get(k):
        try:
            client(a.region).download_file(a.bucket, k, str(out / k.split("/")[-1]))
        except Exception as e:                                                   # noqa: BLE001
            print("fail", k.split("/")[-1], str(e)[:60], flush=True)
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(get, todo))
    have = len(list(out.glob("*.npz")))
    print(f"done: {have} files in {a.out}, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
