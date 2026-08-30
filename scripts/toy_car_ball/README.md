# Toy car knocks a ball off a shelf

A toy race car drives right-to-left across a white wall-mounted shelf,
parallel to the wall behind it, and rear-ends a toy ball resting near the
shelf's left edge; the ball is knocked off, falls to the room floor below,
and rolls to a stop, while the much heavier car sheds little speed and
stays safely on the shelf. This is a direct synthetic instance of the
"object falls off a shelf when struck" scenario from the Physics-IQ
benchmark's real-filmed reference clips, and of VideoPhy-2's momentum-
transfer action category (here with an asymmetric, unequal-mass pair rather
than curling_collision's equal-mass cancellation). The framing -- white
floating shelf, dark backdrop wall, a sliver of light wood floor at the very
bottom -- matches a real physics-test-bench reference photo rather than a
furnished room.

Three downloaded models: `assets/models/toy_car.glb` (Gabriel Solon, CC-BY,
a 10-part red toy race car), `assets/models/pixar_ball.glb` (Maggatron,
CC-BY, a hand-modeled toy ball), and `assets/models/potted_plant.glb`
(propsworld.3d, CC-BY, a scanned potted succulent) -- the plant is a static
decoration to the right of the car's start position, not physics-driven.
The shelf, backdrop wall, baseboard, and floor are all built procedurally:
a plain white matte shelf, a plain warm-brown matte backdrop wall, a white
baseboard at the wall/floor junction (otherwise that seam looks
unnaturally sharp), and a `wood_floor_worn` Poly Haven PBR floor
(deliberately darkened/scaled so it doesn't compete with the shelf). The
lighting is warm-toned throughout (warm sun/area-light colors, a warm world
background) for a cozy-apartment feel rather than a cold/neutral studio.

## Files

- `simulate_toy_car_ball.py` – PyBullet physics simulation.
- `render_toy_car_ball.py` – Blender rendering script that builds the
  wall-mounted shelf, backdrop wall, and floor, imports the car and ball
  models, and applies the physics trajectories as keyframes.
- `batch_render_toy_car_ball.py` – orchestrates multiple randomized renders.
- `build_pcve_toy_car_ball.py` – builds a named PCVE benchmark suite with
  five parameter variations.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/toy_car_ball/render_toy_car_ball.py -- \
    --mode preview \
    --out-dir renders/toy_car_ball_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.5 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/toy_car_ball/render_toy_car_ball.py -- \
    --mode animation \
    --out-dir renders/toy_car_ball \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/toy_car_ball/batch_render_toy_car_ball.py \
  --mode animation \
  --count 1 \
  --seed-base 11000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.5 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_toy_car_ball_video
```

The output video is written to
`renders/test_toy_car_ball_video/sample_0000/toy_car_ball.mp4`.

## Simulate only

```bash
python scripts/toy_car_ball/simulate_toy_car_ball.py \
  --out renders/test_toy_car_ball_physics.json \
  --fps 24 \
  --duration-sec 3.5
```

## Build PCVE suite

```bash
python scripts/toy_car_ball/build_pcve_toy_car_ball.py \
  --out-root renders/pcve_toy_car_ball_suite \
  --resolution 1280 720 \
  --fps 24 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `toy_car_ball_baseline` | Car knocks the ball off the edge; ball falls and settles on the floor. |
| `toy_car_ball_weak_hit` | Launch speed too low -- car stops from friction before ever reaching the ball. |
| `toy_car_ball_heavy_car` | Much heavier car, ball flies off faster and further. |
| `toy_car_ball_bouncy_ball` | Higher ball restitution, ball keeps bouncing on the floor instead of settling. |
| `toy_car_ball_moved_back` | Ball starts well back from the edge, taking longer to finally tip off. |

## Scene layout

- The shelf sits at a `0.75m` height, `1.0m` wide (x) by `0.3m` deep (y) --
  wide enough for the car to travel and brake, but shallow like a real
  wall-mounted floating shelf rather than a full tabletop. It has no legs;
  a dark backdrop wall sits flush behind its back edge, so it reads as
  wall-mounted rather than an unsupported floating plank.
- The car and ball travel along world X, parallel to the wall, matching the
  reference photo's left-to-right framing (earlier drafts of this scene had
  them travel in depth, toward/away from the camera, which was wrong). The
  car starts at `x=0.1` (close to the ball, leaving room on the shelf's
  right side for the potted plant) and drives toward `-x` with an initial
  launch velocity (like the curling stones, there's no simulated engine --
  friction alone decelerates it after the push). The ball starts resting
  near the shelf's left edge at `x≈-0.428` (edge is at `x=-0.5`). The potted
  plant sits at roughly `x=0.35`, well clear of the car's path.
- The camera looks straight down the shelf's depth axis (perpendicular to
  the backdrop wall) rather than at an angle -- `CAMERA_LOCATION` and
  `CAMERA_TARGET` share the same `x`, so panning the frame left/right means
  translating both together, not yawing the camera to look off to one side
  (yawing was tried first and produced a visibly oblique, not-quite-facing-
  the-wall shot). The shelf's right edge is deliberately left outside the
  frame, matching the reference photo's tight crop.
- The car's collision shape is a single box approximating its silhouette
  (real wheels/steering aren't modeled, consistent with how other scenes in
  this project simplify rigid-body shapes under a detailed render mesh).
  Its downloaded model's nose faces local `-y` by default, so both the
  PyBullet body and the render mesh are given a fixed `-90°` yaw at spawn
  (`CAR_INITIAL_QUAT_XYZW`) to point that nose toward world `-x`, its
  direction of travel -- the same quaternion is reused for both so physics
  and the visible mesh always agree.
- The car is much heavier than the ball (`0.35kg` vs `0.05kg`, a 7:1 ratio),
  so the collision barely slows it down -- the car should stop just short
  of the shelf's true edge rather than following the ball off, which is why
  `car_friction`/`table_friction` are tuned high enough (`0.22`/`0.2`) to
  actually stop it within the short remaining runway after impact. These
  had to be tuned up again after moving `car_start_x` closer to the ball --
  less travel distance means less time for friction to bleed off speed
  before impact, so the car arrives (and leaves) the collision faster and
  needs proportionally more friction to still stop in time.
- `pixar_ball.glb` and `potted_plant.glb` are both authored in unitless raw
  coordinates (not meters), so unlike the car -- which already measures out
  to real-world toy-car dimensions (`0.1008m` wide, `0.2337m` long,
  `0.0651m` tall) directly from the GLB -- both need an explicit target-size
  scale factor. The plant is deliberately scaled to `0.3m` (30cm), a real
  small-houseplant size for an actual room-scale shelf, rather than shrunk
  to match the miniature toy car/ball next to it.
- A separate, much larger floor box below catches the ball once it falls
  past the shelf's edge -- PyBullet has no explicit "hole", so the shelf's
  footprint simply ends at its own edge and gravity takes over from there.
  The floor and backdrop wall are both sized generously past the camera's
  frame in every direction, since anything short of that reveals the flat
  world background behind them as a visible seam.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_toy_car_ball.py` for suite cases.

- `car_mass` / `ball_mass`: default `0.35` / `0.05` kg. Raising `car_mass`
  transfers more momentum to the ball on impact (the main "how hard does it
  land" editability hook); lowering `launch_speed` far enough means the car
  never even reaches the ball at all (see `toy_car_ball_weak_hit`).
- `car_friction` / `table_friction`: combine to decelerate the car after it
  leaves the collision (default `0.22` / `0.2`); tuned so the car
  reliably comes to rest just short of the shelf's true edge instead of
  driving off it too.
- `ball_friction` / `ball_restitution`: default `0.3` / `0.6`. Higher
  restitution makes the ball keep bouncing on the floor after landing
  instead of settling quickly (see `toy_car_ball_bouncy_ball`).
- `floor_friction`: default `0.8`, tuned so the ball settles to rest on the
  room floor within the clip instead of sliding for several more seconds.
- `launch_speed`: initial speed of the car in m/s (default `0.6`).
- `car_start_x` / `ball_start_x`: starting positions along the direction of
  travel, parallel to the wall (defaults `0.1` and `-0.428`). Moving the
  ball further from the edge is the main "change the layout, change the
  outcome" editability hook -- the car has to push it much further before
  it finally tips off (see `toy_car_ball_moved_back`, which also needs a
  longer `duration_sec` to show the whole event).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.8]`,
  standard Earth gravity).

## Outputs

- `toy_car_ball.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `toy_car_ball.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms.
- `scenario_metadata.json` – seed, camera, lighting and physics parameters.
