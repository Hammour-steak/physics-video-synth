#!/usr/bin/env bash
# Prepare a clean re-evaluation after the benchmark's prompts and references
# changed. Does no inference itself -- it only moves the old run aside and
# drops the cached tracks that the new renders invalidated.
#
# Why the cache has to be pruned by hand: eval/metrics/physics.py keys a cached
# GroundedSAM2 track on the *text prompt*, its anchor frame and its hint -- not
# on the video. The scenes' text_prompts did not change, so every re-rendered
# case would silently score against a track measured on the video it replaced.
set -euo pipefail

ROOT=/remote-home/chenyuanjie/physics-video-synth
BENCH=$ROOT/pcve_benchmark_v1
STAMP=$(date +%Y%m%d)

cd "$ROOT"

echo "== 1/3 归档旧预测(不删除,留作旧版 benchmark 的记录)"
for m in ditto wan_vace_14b; do
    src=$BENCH/predictions/$m
    [ -d "$src" ] || continue
    dst=$BENCH/predictions/_archive_${STAMP}_$m
    mv "$src" "$dst"
    echo "   $m -> $(basename "$dst")  ($(find "$dst" -name '*.mp4' | wc -l) 个视频)"
done

echo "== 2/3 删除已失效的 GT 轨迹缓存"
python3 - <<'PY'
import glob, os, json
root = "/remote-home/chenyuanjie/physics-video-synth"
phys = set()
for f in ("renders/stale_after_dsl2.txt", "renders/stale_retune.txt"):
    for line in open(os.path.join(root, f)).read().splitlines():
        if line.strip():
            s, c = line.split("\t"); phys.add((s, c))
phys.discard(("toy_car_ball", "edit_heavy_car"))
phys.add(("toy_car_ball", "edit_heavy_ball"))
man = json.load(open(os.path.join(root, "pcve_benchmark_v1/benchmark_manifest.json")))
live = {(e["scene"], e["case_id"]) for e in man["edits"]}

removed = kept = 0
for f in glob.glob(os.path.join(root, "eval/metrics/gt_tracks_gsam2/*.npz")):
    parts = os.path.basename(f)[:-4].split("__")
    if len(parts) == 4 and parts[1] == "SRC":     # source 视频没重渲,缓存有效
        kept += 1; continue
    key = (parts[0], parts[1])
    if key not in live or key in phys:
        os.remove(f); removed += 1
    else:
        kept += 1
print(f"   删除 {removed} 个,保留 {kept} 个")
PY

echo "== 3/3 确认 benchmark 自身是干净的"
python3 -c "
import json
m=json.load(open('$BENCH/benchmark_manifest.json'))
print('   ', m['counts'])
"
echo
echo "准备完成。接着按顺序跑两个模型(每个都要在 tmux 里)。"
