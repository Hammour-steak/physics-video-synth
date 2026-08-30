# Toy car climbs a ramp (surface friction comparison)

A toy race car is given a running-start push up an inclined ramp set up on a
real house's front driveway, propped on a concrete block and facing the
house's front wall. How far the car climbs -- whether it holds its ground once
it stops, slides back down past its starting point, or launches clean off the
top into the air -- follows from the ramp surface's friction and the strength
of the push. The core comparison keeps the push fixed and varies only the
surface material (and with it the real-world friction value), a direct
synthetic version of a real reference clip: four side-by-side ramps (grip
tape, turf, grey asphalt, dark asphalt) with the same toy car driven up each
one. The PCVE suite adds two cases that instead hold the (slick) surface fixed
and vary the push, to isolate the two speed-driven outcomes the surface sweep
alone doesn't show -- see Build PCVE suite.

The car is `assets/models/nissan_gtr-35_lbworks.glb` (a Liberty Walk widebody
Nissan GT-R35; its render mesh is uniformly scaled to the same real-world
length the physics uses, while the simulation keeps the simplified box
collision shape from `toy_car_ball`). The environment is
`assets/models/modern_house.glb` (a full house-and-yard model, used as static
background at its native scale and materials -- its front wall, window, and
paved driveway are what makes this scene read as a real place instead of an
obvious procedural set). The ramp, side rails, and concrete support block are
built as procedural geometry (beveled boxes) and dropped into the scene via a
single parented placement transform (see Scene layout), but wear real PBR
scans for realism: the side rails use a stained-pine wood scan and the support
block a weathered-concrete scan (`concrete_floor_damaged_01`), both triplanar
box-projected so the texture wraps the boxes without stretching. Of the four
swappable ramp *surface* materials, grey/dark asphalt are the real `Asphalt031`
scan (also box-projected; the dark variant is the same texture darkened via a
Hue/Saturation node rather than a second download), while grip tape and turf
stay stylized procedural materials, since real close-up photo textures at this
scale are hard to source cleanly and the point of the comparison is the
friction value, not photorealistic grass blades.

## Files

- `simulate_car_ramp_climb.py` – PyBullet physics simulation.
- `render_car_ramp_climb.py` – Blender rendering script that builds the
  ramp (with swappable surface material), rails, and concrete support block,
  imports the car and the `modern_house.glb` environment (which supplies the
  wall and ground), and applies the physics trajectory as keyframes.
- `batch_render_car_ramp_climb.py` – orchestrates multiple randomized renders.
- `build_pcve_car_ramp_climb.py` – builds the five-case PCVE suite (surface
  sweep plus two push-driven failure modes) that *is* this scene's whole
  premise.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/car_ramp_climb/render_car_ramp_climb.py -- \
    --mode preview \
    --out-dir renders/car_ramp_climb_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.5 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/car_ramp_climb/render_car_ramp_climb.py -- \
    --mode animation \
    --out-dir renders/car_ramp_climb \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.5 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/car_ramp_climb/batch_render_car_ramp_climb.py \
  --mode animation \
  --count 1 \
  --seed-base 13000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.5 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_car_ramp_climb_video
```

## Simulate only

```bash
python scripts/car_ramp_climb/simulate_car_ramp_climb.py \
  --out renders/test_car_ramp_climb_physics.json \
  --fps 24 \
  --duration-sec 3.5
```

Compare surfaces directly from the physics output without rendering:

```bash
python scripts/car_ramp_climb/simulate_car_ramp_climb.py \
  --out renders/test_grip.json --ramp-friction 0.9
python scripts/car_ramp_climb/simulate_car_ramp_climb.py \
  --out renders/test_dark_asphalt.json --ramp-friction 0.12
```

## Build PCVE suite

```bash
python scripts/car_ramp_climb/build_pcve_car_ramp_climb.py \
  --out-root renders/pcve_car_ramp_climb_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.5 \
  --samples 32 \
  --device auto
```

The suite contains the following cases -- this is the scene's actual point,
not an afterthought the way suite cases are for most other scenes here. Three
of them (grip tape, turf, dark asphalt) share the same `launch_speed=2.6` and
differ only in surface; the other two hold the grey-asphalt surface fixed and
differ only in push (`2.5` vs `2.8`), isolating how launch speed alone flips
the outcome between a roof landing and a clean one. Each failure mode lives on
its natural surface for a reason (documented in the code header): an *in-frame*
stall-and-slide-back only happens on the medium-friction turf -- on a slick
ramp the car keeps enough speed to slide all the way back down and off the
right of frame -- and the under-rotated roof landing only occurs in a narrow
moderate-speed band (grey/0.25 flips at 2.5 but lands upright again by
2.7-2.8):

| case_id | surface | friction | push | outcome |
|---|---|---|---|---|
| `car_ramp_climb_grip_orange` | grip tape | 0.9 | 2.6 | Barely climbs before friction stops it; stays put. |
| `car_ramp_climb_stall_slideback` | turf | 0.5 | 2.6 | Climbs to just below the top, stalls, then slides back down and settles near the start. |
| `car_ramp_climb_underrotate_flip` | grey asphalt | 0.25 | 2.5 | Launches off the top but under-rotates in the air and lands on its roof. |
| `car_ramp_climb_clear_land` | grey asphalt | 0.25 | 2.8 | Same ramp, firmer push: clears the top and completes its rotation to land cleanly bottom-down. |
| `car_ramp_climb_asphalt_dark` | dark asphalt | 0.12 | 2.6 | Slickest surface: flies off the top with the most speed to spare and lands upright farthest out. |

## Scene layout

- The ramp is tilted `20°` about world Y, `0.9m` long, `0.35m` wide, `0.03m`
  thick, resting on the ground at its low end with a procedural stone block
  propped under its high end -- following the same tilt convention as
  `ramp_collision`'s book ramp (local `-X` is the high end, local `+X` is
  the low end/base). Without the stone, an inclined ramp just floating in
  place with nothing visibly holding its elevated end up reads as obviously
  fake.
- The whole ramp assembly (ramp, rails, stone, car) is built in its own
  local frame exactly as described below, then parented as a group to a
  single static empty (`PLACEMENT_LOCATION` / `PLACEMENT_ROTATION_Z_DEG`)
  that drops it into `modern_house.glb`'s front driveway, facing the house's
  front wall (which has a window). A side-yard placement was tried first --
  it put the ramp right behind a hedge planted along that wall, blocking the
  camera entirely -- so the front driveway is used instead, which has clear
  paved ground in front of the wall. Because parenting only changes where
  the assembly's *local* coordinates land in world space, the physics
  simulation and the per-frame keyframes applied to the ramp/car are
  completely unaffected by this placement; only the render-time transform
  changes. `PLACEMENT_LOCATION`'s z is set to the house pavement's actual
  height (`~0.0093m`, measured by ray-casting the model straight down) rather
  than `0`, so the assembly's sim-floor plane (where the physics rests
  everything) sits exactly on the visible driveway. At `z=0` the ramp base,
  the stone, and -- most visibly -- the landed car sank ~1cm into the
  pavement.
- The car starts at the ramp's low end and is launched toward the high end
  with a fixed initial speed (like the curling stones and the toy car in
  `toy_car_ball`, there's no simulated engine -- momentum and friction alone
  decide the outcome after the push). Its collision shape is the same
  simplified box used in `toy_car_ball`. Both the car's PyBullet body and
  its render mesh start with a combined orientation -- the ramp's own tilt
  quaternion composed with a `-90°` yaw -- so the car's nose already points
  up-slope and stays flush against the incline throughout, rather than
  starting flat and clipping into the ramp on the first physics step.
- The floor beyond the ramp's base has its own *fixed* friction
  (`floor_friction`, independent of the varying `ramp_friction`) so that a
  car that slides back down and off the ramp decelerates and settles on the
  ground instead of coasting indefinitely -- an earlier draft tied the floor
  friction to the same value as the ramp's and produced cars that flew off
  into empty space picking up unbounded speed over the render, the same
  failure mode hit and fixed in `toy_car_ball`.
- The outcome splits several ways with the surface friction (and, in the
  suite's last two cases, the push): grip tape (`0.9`) stops the car almost
  at once and holds it; turf (`0.5`) lets it climb to just below the top but
  then loses its grip and the car slides back down; and the slick asphalt
  surfaces (`0.25` / `0.12`) let the car keep enough speed to launch clean off
  the top of the ramp rather than stopping on it at all. The `20°` incline
  (`tan(20°) ≈ 0.36`) is what leaves the lower-friction surfaces unable to
  hold a stopped car. All of this is real emergent physics, not scripted.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_car_ramp_climb.py` for suite cases.
`surface` (a top-level scenario field, not under `physics`) selects which of
the four materials in `RAMP_SURFACES` is used for rendering; it should
always be changed together with `physics.ramp_friction` to keep the visual
material and its physical behavior consistent.

- `ramp_friction`: the surface parameter this scene is built around (default
  `0.25`). Higher values stop the car sooner and hold it in place; lower
  values let it climb further and (below `tan(ramp_angle_deg)`) slide back
  down again once it stops, or -- slick enough -- launch clean off the top.
- `surface`: `"grip_orange"`, `"turf_green"`, `"asphalt_grey"`, or
  `"asphalt_dark"` (default `"asphalt_grey"`) -- the rendered ramp material.
- `launch_speed`: initial push speed up the ramp in m/s (default `2.7`, tuned
  so the single hero render clears the top and lands bottom-down within frame).
  The PCVE suite pins its own per-case speeds (`2.5`/`2.6`/`2.8`) rather than
  inheriting this default -- see Build PCVE suite.
- `ramp_angle_deg`: incline angle (default `20.0`). Steeper ramps need
  proportionally higher friction to hold a stopped car in place.
- `car_mass` / `car_restitution`: default `0.35` kg / `0.05`.
- `floor_friction`: fixed friction of the ground beyond the ramp's base
  (default `0.9`), independent of `ramp_friction` -- see Scene layout above.
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.8]`,
  standard Earth gravity).

## Outputs

- `car_ramp_climb.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `car_ramp_climb.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms,
  including `ramp_local_x` (the car's position projected onto the ramp's
  own up-slope axis, useful for measuring how far it climbed without
  re-deriving the ramp's tilt).
- `scenario_metadata.json` – seed, camera, lighting, surface choice, and
  physics parameters.
