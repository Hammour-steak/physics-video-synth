# Ramp collision scene

This scene follows the same coupled simulate-render-build pipeline as `scripts/ball_block`:

1. `simulate_ramp_collision.py` runs PyBullet and writes a trajectory JSON.
2. `render_ramp_collision.py` builds the Blender scene, calls the simulator, applies the trajectories as object keyframes, exports `ground_truth_transforms.json`, and renders the video.
3. `batch_render_ramp_collision.py` orchestrates multiple renders.
4. `build_pcve_ramp_collision.py` builds a named PCVE benchmark suite.

## Quick render

```bash
python scripts/ramp_collision/batch_render_ramp_collision.py \
  --mode animation \
  --count 1 \
  --seed-base 2000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_ramp_video
```

The output video is written to `renders/test_ramp_video/sample_0000/ramp_collision.mp4`.

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/ramp_collision/render_ramp_collision.py -- \
  --mode animation \
  --out-dir renders/ramp_collision \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device cpu \
  --seed 2000
```

The output video is written to `renders/ramp_collision/ramp_collision.mp4`.

## Simulate only

```bash
python scripts/ramp_collision/simulate_ramp_collision.py \
  --out renders/test_ramp_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/ramp_collision/build_pcve_ramp_collision.py \
  --out-root renders/pcve_ramp_collision_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 32 \
  --device auto
```
