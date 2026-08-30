#!/bin/bash
# Re-render the 3 removed table_drop_collision cases at 4.0s.
# --skip-existing skips the baseline and the 4 already-4s edits.
source /remote-home/chenyuanjie/miniconda/etc/profile.d/conda.sh
conda activate physics
cd /remote-home/chenyuanjie/physics-video-synth
exec python -u scripts/table_drop_collision/build_pcve_table_drop_collision.py \
    --skip-existing \
    --verbose-render
