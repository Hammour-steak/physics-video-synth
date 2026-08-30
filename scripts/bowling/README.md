# Bowling ball hitting pins

A bowling scene where a ball rolls down a lane and hits three pins.  It uses
the provided `assets/models/bowling_club.glb` as the environment so the lanes,
neon signage and ball stands are visible in the background.

## Files

- `simulate_bowling.py` – PyBullet physics simulation.
- `render_bowling.py` – Blender rendering script that imports the club GLB,
  keeps one ball and three pins, and applies the physics trajectories as
  keyframes.
- `batch_render_bowling.py` – orchestrates multiple randomized renders.
- `build_pcve_bowling.py` – builds a named PCVE benchmark suite with five
  parameter variations.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/bowling/render_bowling.py -- \
    --mode preview \
    --out-dir renders/bowling_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/bowling/render_bowling.py -- \
    --mode animation \
    --out-dir renders/bowling \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/bowling/batch_render_bowling.py \
  --mode animation \
  --count 1 \
  --seed-base 5000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_bowling_video
```

The output video is written to `renders/test_bowling_video/sample_0000/bowling.mp4`.

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/bowling/render_bowling.py -- \
  --mode animation \
  --out-dir renders/test_bowling_video \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device cpu \
  --seed 5000
```

The output video is written to `renders/test_bowling_video/bowling.mp4`.

## Simulate only

```bash
python scripts/bowling/simulate_bowling.py \
  --out renders/test_bowling_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/bowling/build_pcve_bowling.py \
  --out-root renders/pcve_bowling_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `bowling_baseline` | Default bowling strike. |
| `bowling_heavy_ball` | Heavier bowling ball with more momentum. |
| `bowling_off_center` | Ball hits pins off-center with lateral velocity. |
| `bowling_high_speed` | Faster ball speed for more energetic collision. |
| `bowling_high_restitution` | Higher pin restitution makes pins bounce farther. |

## Scene layout

- The action is offset into the bowling-club floor area at `(7, 27)` so the
  ball rolls on the club's lane surface.
- Ball radius: `0.12 m`, pin radius: `0.075 m`, pin height: `0.495 m`.
- Three pins are arranged in a small triangle in front of the incoming ball.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_bowling.py` for suite cases.

- `ball_radius`: radius of the ball in meters (default `0.12`).
- `ball_mass`: mass of the ball in kilograms (default `3.0`).
- `ball_restitution`: controls bounce on impact (default `0.5`).
- `ball_friction`: lateral friction of the ball (default `0.4`).
- `ball_initial_location`: starting position in meters (default `[10.0, 0.0, 0.18]`).
- `ball_initial_velocity`: starting linear velocity in m/s (default `[-8.0, 0.0, 0.0]`).
- `pin_radius`: radius of each pin in meters (default `0.075`).
- `pin_height`: height of each pin in meters (default `0.495`).
- `pin_mass`: mass of each pin in kilograms (default `0.8`).
- `pin_friction`: lateral friction of pins (default `0.4`).
- `pin_restitution`: controls pin bounce (default `0.3`).
- `pin_spacing`: spacing between pins in meters (default `0.28`).
- `floor_friction`: lateral friction of the floor (default `0.5`).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.81]`).

## Outputs

- `bowling.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `bowling.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms.
- `scenario_metadata.json` – seed, camera, lighting and physics parameters.