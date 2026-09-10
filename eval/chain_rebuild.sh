#!/bin/bash
# Chained pipeline: wait for table_drop re-render → build benchmark → clear
# GroundedSAM2 cache → precompute all GT tracks. Anything that fails aborts
# the chain and logs the reason; the chain is idempotent, so if it aborts you
# can inspect the log and re-run this script.
set -euo pipefail

WORKSPACE=/remote-home/chenyuanjie/physics-video-synth
BENCH=$WORKSPACE/pcve_benchmark_v1
CACHE=$WORKSPACE/eval/metrics/gt_tracks_gsam2

source /remote-home/chenyuanjie/miniconda/etc/profile.d/conda.sh
conda activate physics
export HF_ENDPOINT=https://hf-mirror.com

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# ---- 1) wait for the table_drop re-render to finish ------------------------
log "waiting for build_pcve_table_drop_collision to finish..."
while pgrep -f build_pcve_table_drop_collision > /dev/null; do
    sleep 30
done
log "rerender process gone; verifying 3 case videos are 4.0s..."

python - <<'PY' || { echo "[chain FAIL] rerender did not produce 3 valid 4s videos; aborting"; exit 1; }
import av, sys
root = '/remote-home/chenyuanjie/physics-video-synth/renders/pcve_table_drop_collision_suite/cases'
bad = []
for c in ('edit_soft_push', 'edit_dead_rolling_ball', 'edit_remove_target_ball'):
    v = f'{root}/{c}/video.mp4'
    try:
        with av.open(v) as x:
            s = x.streams.video[0]
            d = float(s.duration * s.time_base)
        if abs(d - 4.0) > 0.05:
            bad.append(f'{c}: {d:.2f}s')
    except Exception as e:
        bad.append(f'{c}: {e}')
if bad:
    print('BAD:', *bad, sep='\n  ')
    sys.exit(1)
print('all 3 cases are 4.00s')
PY

# ---- 2) rebuild benchmark from renders -------------------------------------
log "running build_benchmark.py to sync renders/ -> pcve_benchmark_v1/..."
cd "$WORKSPACE"
python scripts/build_benchmark.py

# ---- 3) clear old GroundedSAM2 cache ---------------------------------------
log "clearing $CACHE (old tracks reference stale videos)..."
if [ -d "$CACHE" ]; then
    n=$(ls "$CACHE"/*.npz 2>/dev/null | wc -l)
    rm -f "$CACHE"/*.npz
    log "removed $n cached tracks"
fi

# ---- 4) precompute new GT tracks -------------------------------------------
log "running precompute-tracks on the refreshed benchmark..."
cd "$WORKSPACE/eval"
python -u compute_metrics.py \
    --benchmark-root "$BENCH" \
    --precompute-tracks

log "chain complete"
