#!/usr/bin/env bash
# Generate the two baselines' predictions for the 7 ADD edits, then score them.
#
# One shard per GPU, ditto first (fast) then wan_vace_14b (slow), so each GPU
# runs one window start to finish. Settings are copied from the shard manifests
# of the existing 110-case runs so the new cases are comparable with them:
# quantitative/en prompts, seed 42, ditto at 480x832 with the LoRA's own step
# count, VACE at 720x1280 / 50 steps / guidance 5.0 / group offload.
#
# ditto's num_inference_steps is deliberately NOT set: the LoRA is not
# step-distilled, and forcing a lower count makes the edit stop taking effect.
set -uo pipefail

ROOT=/remote-home/chenyuanjie/physics-video-synth
BENCH=$ROOT/pcve_benchmark_v1
SHARD=${SHARD:?set SHARD}          # 0..4
NSHARDS=${NSHARDS:-5}
LOG=$ROOT/eval/logs/add_eval
mkdir -p "$LOG"

source /remote-home/chenyuanjie/miniconda/etc/profile.d/conda.sh
conda activate physics
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT/eval"

run () {
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] shard $SHARD: START $name"
  python3 run_baseline.py \
      --benchmark-root "$BENCH" \
      --baseline "$name" \
      --filter-kind ADD \
      --prompt-flavor quantitative --prompt-lang en \
      --num-shards "$NSHARDS" --shard-index "$SHARD" \
      --skip-existing \
      "$@" >>"$LOG/${name}.shard${SHARD}.log" 2>&1
  echo "[$(date +%H:%M:%S)] shard $SHARD: DONE $name rc=$?"
}

# --model-id is not optional here: ditto's own default is the repo id
# `Wan-AI/Wan2.1-VACE-14B`, which sends it to ModelScope to download and
# straight into a broken modelscope_hub version. The 110-case run used
# the local checkout; so does this.
run ditto        --height 480 --width 832  --seed 42 \
                 --model-id /remote-home/chenyuanjie/models/Wan2.1-VACE-14B
run wan_vace_14b --height 720 --width 1280 --seed 42 \
                 --num-inference-steps 50 --guidance-scale 5.0 \
                 --offload-mode group --num-blocks-per-group 1

echo "[$(date +%H:%M:%S)] shard $SHARD: all baselines finished"
