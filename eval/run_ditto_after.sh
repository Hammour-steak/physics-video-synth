#!/usr/bin/env bash
# Run ditto's ADD shard once this GPU's VACE shard has finished.
#
# Not run concurrently with VACE on purpose: VACE-14B holds this card for
# hours under group offload, and an OOM from a second process would take the
# long run down with it. This waits for the GPU to go idle instead.
set -uo pipefail
ROOT=/remote-home/chenyuanjie/physics-video-synth
SHARD=${SHARD:?set SHARD}
NSHARDS=${NSHARDS:-5}
# Which physical card to watch. Must be the index, not a CUDA_VISIBLE_DEVICES
# alias: nvidia-smi without -i reports every GPU on the box, and cards 1/3/7
# are running someone else's work, so an unscoped check never goes idle.
GPU=${GPU:?set GPU}
LOG=$ROOT/eval/logs/add_eval
mkdir -p "$LOG"

echo "[$(date +%H:%M:%S)] shard $SHARD: waiting for VACE on this GPU to finish"
while nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  sleep 120
done
echo "[$(date +%H:%M:%S)] shard $SHARD: GPU idle, starting ditto"

source /remote-home/chenyuanjie/miniconda/etc/profile.d/conda.sh
conda activate physics
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT/eval"
python3 run_baseline.py \
    --benchmark-root "$ROOT/pcve_benchmark_v1" \
    --baseline ditto \
    --model-id /remote-home/chenyuanjie/models/Wan2.1-VACE-14B \
    --filter-kind ADD \
    --prompt-flavor quantitative --prompt-lang en \
    --height 480 --width 832 --seed 42 \
    --num-shards "$NSHARDS" --shard-index "$SHARD" \
    --skip-existing >>"$LOG/ditto.shard${SHARD}.retry.log" 2>&1
echo "[$(date +%H:%M:%S)] shard $SHARD: ditto finished rc=$?"
