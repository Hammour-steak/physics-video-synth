#!/usr/bin/env bash
# Full Ditto run over PCVE, sharded across free GPUs.
# 50 steps (default), 832x480x73, resumable via --skip-existing.
#
# Meant to run inside tmux -- it stays in the foreground until every shard
# finishes, so Ctrl-C stops all of them and detaching leaves them running.
set -euo pipefail

EVAL=/remote-home/chenyuanjie/physics-video-synth/eval
BENCH=/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1
MODEL=/remote-home/chenyuanjie/models/Wan2.1-VACE-14B
PY=/remote-home/chenyuanjie/miniconda/envs/physics/bin/python
# Cards to shard across. Override for one run without editing this file:
#     PCVE_GPUS="2 5" bash launch_ditto_full.sh
read -ra GPUS <<< "${PCVE_GPUS:-1 3 5 7}"
N=${#GPUS[@]}

cd "$EVAL"
mkdir -p logs

for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    log="logs/ditto_shard${i}of${N}_gpu${g}.log"
    CUDA_VISIBLE_DEVICES=$g \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    "$PY" -u run_baseline.py \
        --benchmark-root "$BENCH" \
        --baseline ditto \
        --model-id "$MODEL" \
        --prompt-flavor quantitative --prompt-lang en \
        --skip-existing \
        --num-shards "$N" --shard-index "$i" \
        > "$log" 2>&1 &
    echo "shard $i/$N -> GPU $g  pid=$!  log=$EVAL/$log"
    sleep 15   # stagger heavy checkpoint loads
done

echo
echo "all $N shards launched. follow from another pane with:"
echo "  tail -f $EVAL/logs/ditto_shard*.log"
echo "waiting for all shards to finish (Ctrl-C stops every shard)..."
wait
echo "[launch] all shards done."
