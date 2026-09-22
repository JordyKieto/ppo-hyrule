#!/bin/bash
# Pull a fleet-collected dataset from S3 to local, fast, and VERIFY the object count.
#
# Why this exists: `aws s3 sync s3://.../v5/ datasets/transitions/v5/` is the obvious command and it is
# 12x too slow. The dataset is ~80k objects averaging ~200 KB, so the wall clock is request latency, not
# bandwidth, and the default `max_concurrent_requests = 10` on ONE process leaves the link idle.
# Measured on the v5 dataset (79,694 episodes, 17 GB): single sync 588 files/min (126 min),
# fanned out over the per-rank prefixes 7,164 files/min (9 min).
#
# usage: scripts/pull_dataset.sh <s3-uri> <local-dir> [parallel] [profile]
set -uo pipefail
SRC="${1:?s3://bucket/prefix}"; DST="${2:?local dir}"; PAR="${3:-16}"; PROF="${4:-race-wm}"
SRC="${SRC%/}"

# Per-rank prefixes are what makes the fan-out possible; the collector namespaces by --s3-shard.
# no mapfile: macOS ships bash 3.2.
RANKS=$(mktemp); trap 'rm -f "$RANKS"' EXIT
aws s3 ls "$SRC/" --profile "$PROF" | awk '/\/$/{print $2}' | tr -d / > "$RANKS"
n=$(wc -l < "$RANKS" | tr -d ' ')
export AWS_MAX_CONCURRENT_REQUESTS=24 AWS_MAX_QUEUE_SIZE=2000
if [ "$n" -eq 0 ]; then
  echo "no sub-prefixes under $SRC -- not a sharded dataset, syncing flat"
  aws s3 sync "$SRC/" "$DST/" --profile "$PROF" --only-show-errors
else
  echo "$SRC -> $DST : $n shards, $PAR parallel"
  xargs -P "$PAR" -I{} aws s3 sync "$SRC/{}/" "$DST/{}/" --profile "$PROF" --only-show-errors < "$RANKS"
fi

# AGENTS.md 16: verify the artefact landed, not that the command ran.
remote=$(aws s3 ls "$SRC/" --recursive --profile "$PROF" | grep -c '\.npz$')
local_n=$(find "$DST" -name '*.npz' | wc -l | tr -d ' ')
echo "episodes: remote=$remote local=$local_n"
[ "$remote" -eq "$local_n" ] || { echo "MISMATCH: $((remote - local_n)) episodes missing -- rerun (sync resumes)"; exit 1; }
echo "OK: dataset complete ($(du -sh "$DST" | cut -f1))"
