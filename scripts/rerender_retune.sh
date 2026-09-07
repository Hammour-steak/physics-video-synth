#!/usr/bin/env bash
# Re-render the three cases changed after the first pass of re-renders:
#
#   car_ramp_climb/edit_underrotate_flip   x0.9  -> x0.95  (flip restored)
#   car_gap_jump/edit_slippery_wheels      x0.05 -> x0.3   (car stays in the room)
#   toy_car_ball/edit_heavy_ball           replaces edit_heavy_car entirely
#
# The last one is a rename, so the case it replaced is deleted from the render
# tree, from the benchmark bundle and from the thumbnails before anything is
# built; --clean-stale-cases would catch the render tree on its own, but not
# the copies the benchmark keeps.
#
# About 55 minutes: 9 min for car_ramp_climb, 40 for car_gap_jump (that scene
# renders at 128 samples), 7 for toy_car_ball -- measured from the previous
# batch in rerender_dsl2.log. Run it under tmux anyway.
set -euo pipefail

WORKSPACE=/remote-home/chenyuanjie/physics-video-synth
LIST="$WORKSPACE/renders/stale_retune.txt"
RETIRED=("toy_car_ball edit_heavy_car")

cd "$WORKSPACE"
[ -s "$LIST" ] || { echo "nothing to do in $LIST"; exit 0; }

echo "== retiring cases that were renamed or dropped"
for entry in "${RETIRED[@]}"; do
  read -r scene case <<< "$entry"
  rm -rf "renders/pcve_${scene}_suite/cases/${case}" \
         "pcve_benchmark_v1/scenes/${scene}/cases/${case}"
  rm -f  "pcve_benchmark_v1/thumbnails/${scene}/${case}.jpg"
  echo "   dropped ${scene}/${case}"
done

echo "== dropping $(wc -l < "$LIST") cases to re-render"
while IFS=$'\t' read -r scene case; do
  [ -n "$scene" ] || continue
  rm -rf "renders/pcve_${scene}_suite/cases/${case}" \
         "pcve_benchmark_v1/scenes/${scene}/cases/${case}"
  rm -f  "pcve_benchmark_v1/thumbnails/${scene}/${case}.jpg"
done < "$LIST"

for scene in $(cut -f1 "$LIST" | sort -u); do
  echo "== re-rendering $scene"
  python3 "scripts/${scene}/build_pcve_${scene}.py" --skip-existing --clean-stale-cases
done

echo "== rebuilding the benchmark bundle"
python3 scripts/build_benchmark.py

echo "== done."
