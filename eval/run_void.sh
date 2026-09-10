#!/bin/bash
# Run the VOID baseline on PCVE whole_clip DELETE cases.
# Needs the reverse proxy tunnel to be up on 127.0.0.1:8899 (stage 2 talks to Gemini).
set -u
cd /remote-home/chenyuanjie/physics-video-synth/eval
eval "$(grep -E '^export GEMINI_API_KEY=' /home/chenyuanjie/.bashrc)"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export https_proxy=http://127.0.0.1:8899 http_proxy=http://127.0.0.1:8899
export no_proxy=localhost,127.0.0.1
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PY=/remote-home/chenyuanjie/miniconda/envs/physics/bin/python
exec $PY -u run_baseline.py \
  --benchmark-root ../pcve_benchmark_v1 \
  --baseline void \
  --filter-kind DELETE --filter-timing ${TIMING:-whole_clip} \
  --skip-existing "$@"
