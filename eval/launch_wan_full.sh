#!/usr/bin/env bash
# Launch the full wan_vace_14b run over the PCVE benchmark, sharded across GPUs.
# 720p / 50 steps, one process per GPU, resumable via --skip-existing.
#
# Meant to be run inside tmux -- it stays in the foreground until every shard
# finishes, so Ctrl-C stops all of them and detaching leaves them running.
set -euo pipefail

EVAL=/remote-home/chenyuanjie/physics-video-synth/eval
BENCH=/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1
PY=/remote-home/chenyuanjie/miniconda/envs/physics/bin/python
# Cards to shard across. Override for one run without editing this file:
#     PCVE_GPUS="2 5" bash launch_ditto_full.sh
read -ra GPUS <<< "${PCVE_GPUS:-1 3 5 7}"
N=${#GPUS[@]}

cd "$EVAL"
mkdir -p logs

for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    log="logs/wan_shard${i}of${N}_gpu${g}.log"
    CUDA_VISIBLE_DEVICES=$g \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" run_baseline.py \
        --benchmark-root "$BENCH" \
        --baseline wan_vace_14b \
        --prompt-flavor quantitative --prompt-lang en \
        --num-inference-steps 50 \
        --offload-mode group --num-blocks-per-group 1 \
        --skip-existing \
        --num-shards "$N" --shard-index "$i" \
        > "$log" 2>&1 &
    echo "shard $i/$N -> GPU $g  pid=$!  log=$EVAL/$log"
    sleep 10   # stagger the 70 GB checkpoint loads so they do not fight over disk
done

echo
echo "all $N shards launched. follow from another pane with:"
echo "  tail -f $EVAL/logs/wan_shard*.log"
echo "waiting for all shards to finish (Ctrl-C stops every shard)..."
wait
echo "[launch] all shards done."
