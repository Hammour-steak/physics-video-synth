# Pool Collision Scene

This directory implements a two-ball billiard collision on the `pool_table.glb` asset. It follows the same simulate → render → batch → PCVE pipeline as `scripts/ramp_collision`.

## Files

| File | Purpose |
|------|---------|
| `simulate_pool_collision.py` | PyBullet physics for the cue ball and target ball on a flat table surface. |
| `render_pool_collision.py` | Blender orchestration: import `pool_table.glb`, set up camera/lights/HDRI, run the simulation, apply keyframes, export ground truth, and render. |
| `batch_render_pool_collision.py` | Render multiple randomized samples. |
| `build_pcve_pool_collision.py` | Build a small PCVE benchmark suite with several collision cases. |

## Quick Start

Activate the physics environment (required by the PyBullet simulation subprocess):

```bash
conda activate physics
```

Preview a single frame:

```bash
/remote-home/chenyuanjie/physics-video-synth/tools/blender-3.6.23-linux-x64/blender \
  -b --python /remote-home/chenyuanjie/physics-video-synth/scripts/pool_collision/render_pool_collision.py -- \
  --mode preview \
  --out-dir /remote-home/chenyuanjie/physics-video-synth/renders/pool_preview \
  --resolution 640 360 \
  --samples 16 \
  --seed 1001
```

Render a full animation/video:

```bash
/remote-home/chenyuanjie/physics-video-synth/tools/blender-3.6.23-linux-x64/blender \
  -b --python /remote-home/chenyuanjie/physics-video-synth/scripts/pool_collision/render_pool_collision.py -- \
  --mode animation \
  --out-dir /remote-home/chenyuanjie/physics-video-synth/renders/pool_collision \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3 \
  --samples 96 \
  --device cpu \
  --seed 1001
```

The default camera/background settings are `--hdri-rotation 230` and `--scene-lower-z 0.25`. Override them if needed, for example:

```bash
  --hdri-rotation 210 \
  --scene-lower-z 0.3
```

## Outputs

Each render produces:

```
{out-dir}/
  pool_collision.mp4
  pool_collision.blend
  scenario_metadata.json
  ground_truth_transforms.json
  preview.png           # if --mode preview
  frame_*.png           # if --mode frames
```

## Batch Rendering

```bash
python3 scripts/pool_collision/batch_render_pool_collision.py \
  --out-root renders/batch_pool \
  --count 4 \
  --mode preview \
  --dry-run
```

Omit `--dry-run` to actually render.

## PCVE Suite

```bash
python3 scripts/pool_collision/build_pcve_pool_collision.py --out-root renders/pcve_pool_collision_suite --resolution 1280 720 --fps 24 --duration-sec 3.0 --samples 32 --device auto --verbose-render
```

The suite defines five cases:

- `pool_baseline` — head-on collision with default parameters.
- `pool_high_restitution` — bouncier balls.
- `pool_heavy_cue` — heavier cue ball.
- `pool_off_center` — off-center impact.
- `pool_fast_break` — faster cue ball.

## Scenario Parameters

The render script samples or loads a `scenario_metadata.json` object. Key fields under `physics`:

| Field | Default | Description |
|-------|---------|-------------|
| `ball_radius` | `0.05715` | Ball radius in meters (auto-updated from the scaled mesh). |
| `ball_mass` | `0.17` | Ball mass in kg. |
| `ball_friction` | `0.15` | Ball-ball and ball-table friction. |
| `ball_restitution` | `0.90` | Collision elasticity. |
| `ball_rolling_friction` | `0.02` | Rolling resistance. |
| `ball_spinning_friction` | `0.02` | Spinning resistance. |
| `table_friction` | `0.08` | Table surface friction. |
| `table_restitution` | `0.10` | Table surface restitution. |
| `gravity` | `[0, 0, -9.81]` | Gravity vector. |
| `cue_initial_location` | `[0.0, -0.6, 0.0]` | Cue ball initial `(x, y, z_offset)`; final `z = surface_z + radius + z_offset`. |
| `target_initial_location` | `[0.0, 0.0, 0.0]` | Target ball initial `(x, y, z_offset)`. |
| `cue_initial_velocity` | `[0.0, 1.0, 0.0]` | Cue ball initial velocity in m/s. |

You can override any field via `--scenario-overrides-json`:

```json
{
  "physics": {
    "cue_initial_velocity": [0.0, 1.5, 0.0]
  }
}
```

## Notes

- `pool_table.glb` is scaled by `1/3` so the balls match a standard `0.057 m` radius.
- The PyBullet table collision body is a large static box at the height of the green play surface; the visual mesh from the glTF is used for rendering.
- The 14 unused balls are hidden from viewport and render.
