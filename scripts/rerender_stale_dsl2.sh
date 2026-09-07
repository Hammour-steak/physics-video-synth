#!/usr/bin/env bash
# Re-render the cases whose physics moved when the edit DSL switched to
# multiples and division points.
#
# Every other case kept its exact scenario_overrides.json, so its video still
# matches its prompt and is left alone. The list in renders/stale_after_dsl2.txt
# was produced by diffing each case's overrides before and after the migration,
# not predicted -- if it is empty there is nothing to do.
#
# Each case directory is deleted and rebuilt from scratch; --skip-existing then
# re-renders exactly the missing ones and leaves the rest untouched. Roughly 35
# minutes per case, so ~23 hours for the full list. Run it under tmux.
set -euo pipefail

WORKSPACE=/remote-home/chenyuanjie/physics-video-synth
LIST="$WORKSPACE/renders/stale_after_dsl2.txt"

cd "$WORKSPACE"
[ -s "$LIST" ] || { echo "nothing stale in $LIST"; exit 0; }

echo "== dropping $(wc -l < "$LIST") stale case directories and their thumbnails"
while IFS=$'\t' read -r scene case; do
  [ -n "$scene" ] || continue
  rm -rf "renders/pcve_${scene}_suite/cases/${case}"
  # The benchmark copy and its thumbnail are rebuilt from the render, and
  # make_thumbnail skips any file that already exists.
  rm -rf "pcve_benchmark_v1/scenes/${scene}/cases/${case}"
  rm -f  "pcve_benchmark_v1/thumbnails/${scene}/${case}.jpg"
done < "$LIST"

for scene in $(cut -f1 "$LIST" | sort -u); do
  echo "== re-rendering $scene"
  python3 "scripts/${scene}/build_pcve_${scene}.py" --skip-existing
done

echo "== rebuilding the benchmark bundle"
python3 scripts/build_benchmark.py

echo "== done. Every case in $LIST now has a video matching its prompt."
