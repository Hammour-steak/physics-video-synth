# Curling stone collision

Two curling stones slide toward each other along a single line and collide
head-on in the center of the "house" (the painted target). With equal mass,
equal-and-opposite speed, and near-zero restitution, momentum cancels out
exactly: both stones come to rest at the point of impact instead of one
knocking the other away. This is a direct real-world instance of VideoPhy-2's
"something colliding with something and both come to a halt" action category.

Both stones use the same downloaded model,
`assets/models/curling_stone.glb` (Global Digital Heritage photogrammetry
scan, CC-BY-NC -- internal/research use only, no redistribution), scaled to
a regulation `0.292 m` diameter / `0.114 m` height stone. The ice sheet's
"house" rings are painted at regulation curling dimensions (button `0.152 m`,
red 4-ft `0.610 m`, white 8-ft `1.219 m`, blue 12-ft outer `1.829 m` radius)
via a radial-distance shader lookup, with a pebbled-ice bump texture and a
centerline running between the stones' start positions.

## Files

- `simulate_curling_collision.py` – PyBullet physics simulation.
- `render_curling_collision.py` – Blender rendering script that builds the
  ice sheet and house target procedurally, imports the curling stone model
  twice (one hue-shifted for team color), and applies the physics
  trajectories as keyframes.
- `batch_render_curling_collision.py` – orchestrates multiple randomized renders.
- `build_pcve_curling_collision.py` – builds a named PCVE benchmark suite
  with five parameter variations.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/curling_collision/render_curling_collision.py -- \
    --mode preview \
    --out-dir renders/curling_collision_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 6.0 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/curling_collision/render_curling_collision.py -- \
    --mode animation \
    --out-dir renders/curling_collision \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 6.0 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/curling_collision/batch_render_curling_collision.py \
  --mode animation \
  --count 1 \
  --seed-base 9000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 6.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_curling_collision_video
```

The output video is written to
`renders/test_curling_collision_video/sample_0000/curling_collision.mp4`.

## Simulate only

```bash
python scripts/curling_collision/simulate_curling_collision.py \
  --out renders/test_curling_collision_physics.json \
  --fps 24 \
  --duration-sec 6.0
```

## Build PCVE suite

```bash
python scripts/curling_collision/build_pcve_curling_collision.py \
  --out-root renders/pcve_curling_collision_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 6.0 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `curling_collision_baseline` | Equal-mass head-on collision, both stones stop at impact. |
| `curling_collision_unequal_mass` | Heavier incoming stone knocks the lighter one back instead of both stopping. |
| `curling_collision_bouncy` | Higher restitution, stones bounce apart instead of stopping dead. |
| `curling_collision_high_ice_friction` | Stones slow down too much and never actually collide. |
| `curling_collision_fast_launch` | Higher launch speed, harder/more violent impact. |

## Scene layout

- Two stones start at `x = ∓start_separation/2`, launched toward each other
  with mirrored velocities `(±launch_speed, 0, 0)` along a single line (a
  pure head-on impact, no glancing offset).
- `stone_0` (left, launched in `+x`) keeps the scanned model's original red
  handle color; `stone_1` (right) gets an independent hue-shifted copy of
  the same material (`recolor_stone_handle`, `hue=0.667`, red → yellow/gold)
  so the two stones read as different teams without duplicating geometry.
- `start_separation` defaults to `5.0 m`, well outside the house's `1.829 m`
  outer radius, so both stones are visibly traveling before they reach the
  target; `duration_sec` defaults to `6.0 s` to leave room for the travel,
  the collision, and the low-friction settle tail afterward.
- Camera and the `20 m x 14 m` physics-independent visual ice sheet are
  sized so the frame never shows the rink's edge or background -- only ice.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_curling_collision.py` for suite cases.

- `stone_radius` / `stone_height`: regulation stone dimensions in meters
  (default `0.145` / `0.114`).
- `stone_mass` / `stone_2_mass`: mass of the left/right stone in kilograms
  (default `20.0` each -- real curling stones weigh about this much). Making
  these unequal is the main editability hook: with equal mass and opposite
  velocity the post-collision momentum is exactly zero for both, but an
  unequal pair transfers momentum asymmetrically instead.
- `stone_friction` / `stone_restitution`: contact properties between the two
  stones (default `0.15` / `0.0` -- a near-perfectly inelastic collision, so
  both stop rather than bouncing apart).
- `ice_friction`: lateral friction between stones and ice (default `0.015`,
  deliberately low -- real curling ice is built to be nearly frictionless).
  Raising this significantly can bleed off all the stones' speed before they
  ever meet, so the collision doesn't happen at all.
- `launch_speed`: initial speed of each stone in m/s, mirrored so the total
  momentum starts at zero (default `0.9`).
- `start_separation`: initial distance between the two stones in meters
  (default `5.0`).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.8]`,
  standard Earth gravity).
- Numerical stability: `--substeps` defaults to `60` (not the project's
  usual `12`) because at higher approach speeds a coarser substep count let
  the solver's interpenetration-correction leave a small residual bounce
  velocity even at `stone_restitution = 0.0` -- more substeps plus a modest
  `launch_speed` are what actually gets both final speeds under the
  `both_at_rest` quality threshold (`< 0.03 m/s` per stone).

## Outputs

- `curling_collision.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `curling_collision.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms.
- `scenario_metadata.json` – seed, camera, lighting and physics parameters.
