#!/bin/bash
# Builds the arXiv submission tarball from paper/paper.tex, and proves it compiles
# standalone in an empty directory, which is the only thing arXiv will do with it.
#
# paper.tex is NOT kept in this directory: a second copy drifts from the one
# scripts/md_to_tex.py writes, and then the submitted PDF is not the repo's PDF.
set -euo pipefail
cd "$(dirname "$0")"
test -f ../paper.tex || { echo "../paper.tex missing: run scripts/md_to_tex.py first"; exit 1; }

tar czf paper.tar.gz -C .. paper.tex
echo "built paper.tar.gz: $(tar tzf paper.tar.gz | tr '\n' ' ')"

T=$(mktemp -d)
tar xzf paper.tar.gz -C "$T"
( cd "$T" && tectonic -X compile paper.tex --outdir . >/dev/null 2>&1 )
PAGES=$(pdfinfo "$T/paper.pdf" | awk '/^Pages:/{print $2}')
echo "standalone compile OK: $PAGES pages, $(du -h "$T/paper.pdf" | cut -f1)"

# metadata.txt's COMMENTS field, which goes in the arXiv submission form, hardcodes the
# page count. If the paper grows, that field is stale and nothing else would say so.
WANT=$(awk '/^  24 pages/{print "24"}' metadata.txt)
if [ "$PAGES" != "$WANT" ]; then
  echo "PAGE COUNT CHANGED: PDF has $PAGES pages, metadata.txt COMMENTS says ${WANT:-nothing}."
  echo "Update the COMMENTS field, then re-run."
  rm -rf "$T"; exit 1
fi

# an overfull box is a line of text printed outside the column. There are now zero, so
# any at all means a layout fault would go out in the submission.
( cd "$T" && tectonic -X compile paper.tex --outdir . --keep-logs >/dev/null 2>&1 )
OVERFULL=$(grep -c '^Overfull' "$T/paper.log" || true)
echo "overfull boxes: $OVERFULL"
test "$OVERFULL" -eq 0 || { echo "LAYOUT FAULT: $OVERFULL overfull boxes"; rm -rf "$T"; exit 1; }
rm -rf "$T"

A=$(wc -c < abstract.txt)
echo "abstract.txt: $A chars (arXiv limit 1920)"
test "$A" -le 1920 || { echo "ABSTRACT TOO LONG"; exit 1; }
