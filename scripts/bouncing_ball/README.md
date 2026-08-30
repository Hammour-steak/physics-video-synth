# Bouncing ball scene

This scene demonstrates a **single rigid ball** undergoing free-fall and elastic bouncing on a static floor. It follows the same coupled simulate-render-build pipeline as `scripts/ramp_collision` and `scripts/ball_block`:

1. `simulate_bouncing_ball.py` runs PyBullet and writes a trajectory JSON.
2. `render_bouncing_ball.py` builds the Blender scene, calls the simulator, applies the trajectory as object keyframes, exports `ground_truth_transforms.json`, and renders the video.
3. `batch_render_bouncing_ball.py` orchestrates multiple randomized renders.
4. `build_pcve_bouncing_ball.py` builds a named PCVE benchmark suite with four parameter variations.

## Scene assets

- **Ball**: the exact same scuffed orange rubber ball material as `scripts/ball_block`, including Rubber002 normal/roughness/displacement maps, procedural colour variation, seams, and surface scuffs. The ball radius is `0.34 m`, matching `scripts/ball_block`.
- **Floor**: worn wood floor material using Poly Haven `wood_floor_worn` on a `10.0 m × 10.0 m` floor.
- **Walls**: the same matte plaster wall and off-white baseboard geometry used in `scripts/ball_block`.
- **Camera**: positioned at `(3.9, -5.7, 2.35)` and pointed at `(0.0, 0.0, 0.45)`, identical to `scripts/ball_block`.
- **Background/Lighting**: Poly Haven `brown_photostudio_05` HDRI for environment lighting; the walls form the visible background.

## Physics

- The ball is a PyBullet rigid sphere (`GEOM_SPHERE`).
- The floor is a large static box (`GEOM_BOX`), which is more stable than `GEOM_PLANE` for small objects.
- The floor's `restitution` is set to `1.0` so that the ball's own `restitution` fully controls the bounce height (PyBullet multiplies the two values).
- No soft-body, cloth, or spring objects are used; everything is a rigid body.

## Quick render

```bash
python scripts/bouncing_ball/batch_render_bouncing_ball.py \
  --mode animation \
  --count 1 \
  --seed-base 4000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_bouncing_ball_video
```

The output video is written to `renders/test_bouncing_ball_video/sample_0000/bouncing_ball.mp4`.

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/bouncing_ball/render_bouncing_ball.py -- \
  --mode animation \
  --out-dir renders/test_bouncing_ball_video \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device cpu \
  --seed 4000
```

The output video is written to `renders/test_bouncing_ball_video/bouncing_ball.mp4`.

## Simulate only

```bash
python scripts/bouncing_ball/simulate_bouncing_ball.py \
  --out renders/test_bouncing_ball_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/bouncing_ball/build_pcve_bouncing_ball.py \
  --out-root renders/pcve_bouncing_ball_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `bounce_baseline` | Default free-fall bounce. |
| `bounce_high_restitution` | Higher ball restitution, more bounces. |
| `bounce_heavy_ball` | Heavier ball. |
| `bounce_lateral_velocity` | Ball bounces across the floor with horizontal velocity. |

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when rendering, or directly in `build_pcve_bouncing_ball.py` for suite cases.

- `ball_radius`: radius of the ball in meters (default `0.05`).
- `ball_mass`: mass of the ball in kilograms (default `0.2`).
- `ball_restitution`: controls bounce height (default `0.75`).
- `ball_friction`: lateral friction of the ball (default `0.4`).
- `floor_friction`: lateral friction of the floor (default `0.6`).
- `ball_initial_location`: starting position in meters (default `[0.0, 0.0, 1.0]`).
- `ball_initial_velocity`: starting linear velocity in m/s (default `[0.5, 0.0, 0.0]`).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.81]`).
