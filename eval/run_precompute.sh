#!/bin/bash
# Precompute wrapper: activates conda env, sets HF mirror, iterates all edits.
source /remote-home/chenyuanjie/miniconda/etc/profile.d/conda.sh
conda activate physics
export HF_ENDPOINT=https://hf-mirror.com
cd /remote-home/chenyuanjie/physics-video-synth/eval
exec python -u compute_metrics.py \
    --benchmark-root /remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1 \
    --precompute-tracks
