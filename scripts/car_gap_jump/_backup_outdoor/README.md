# Car gap jump (broken bridge)

A sports car speeds along an elevated road, rides up a take-off ramp at the
broken edge of a bridge, launches into the air, and either clears the gap and
lands on the far deck or falls short and plummets into the chasm below. Like
`car_ramp_climb` and `dining_chain`, nothing is scripted mid-air: the whole arc
is emergent from the car's one initial approach speed plus the ramp geometry and
gravity.

The physical crux is a projectile launch. Once the car leaves the ramp lip it is
a projectile whose range is set by its launch speed and angle; whether that
range reaches the far landing deck depends on the speed and the gap width.
Enough speed and the car lands upright on the far deck and rolls to a stop; too
little speed, or too wide a gap, and the arc falls short and the car drops into
the canyon (see the PCVE suite, which contrasts exactly this).

The car is `assets/models/nissan_gtr-35_lbworks.glb`, uniformly rescaled to a
real Nissan GT-R footprint (~4.70 m long) and given a single box collision proxy
for the physics; the detailed mesh is visual-only. The road decks, take-off
ramp, and canyon floor are built procedurally (asphalt-textured slabs over a
procedural rock chasm floor). Motion is simulated with PyBullet and rendered in
Blender Cycles under the Poly Haven `syferfontein_0d_clear_puresky` clear-sky
HDRI plus a sun.

## Files

- `simulate_car_gap_jump.py` – PyBullet physics simulation (a box car proxy
  driven up a ramp between two static decks over a gap, with a chasm floor far
  below).
- `render_car_gap_jump.py` – Blender rendering script: builds the decks/ramp/
  chasm, imports and rescales the GT-R, runs the simulation, applies the
  trajectory as keyframes, lights the scene, and renders.
- `batch_render_car_gap_jump.py` – orchestrates multiple randomized renders.
- `build_pcve_car_gap_jump.py` – builds the PCVE suite (does the car clear the
  gap, or fall short?).

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame (mid-jump, over the gap)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/car_gap_jump/render_car_gap_jump.py -- \
    --mode preview \
    --out-dir renders/car_gap_jump_preview \
    --resolution 1280 720 \
    --samples 96 \
    --device auto \
    --preview-frame 52

# Render the full animation (108 frames at 24 fps = 4.5 s)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/car_gap_jump/render_car_gap_jump.py -- \
    --mode animation \
    --out-dir renders/car_gap_jump \
    --resolution 1280 720 \
    --fps 24 \
    --duration-sec 4.5 \
    --samples 128 \
    --device auto
```

## Batch render

```bash
python scripts/car_gap_jump/batch_render_car_gap_jump.py \
  --mode animation \
  --count 4 \
  --seed-base 31000 \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 4.5 \
  --samples 128 \
  --device auto \
  --out-root renders/batch_car_gap_jump
```

Each sample jitters the launch speed (9.5–13.5 m/s) and gap width (7–11 m) within
ranges that straddle the clear/fall-short threshold, so the batch contains both
outcomes. Each sample lands in `sample_0000/`, `sample_0001/`, etc., with its own
video, `.blend`, `ground_truth_transforms.json`, and `scenario_metadata.json`;
the batch root also holds `batch_manifest.json` with seeds, sampled params, and
the exact commands used.

## Simulate only

```bash
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_gap_jump_physics.json \
  --fps 24 \
  --duration-sec 4.5
```

Compare the outcome directly from the physics output, no rendering:

```bash
# firm launch, standard gap -> clears and lands on the far deck
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_clears.json --launch-speed 12 --gap-width 8
# same launch, wider gap -> falls short into the chasm
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_short.json --launch-speed 12 --gap-width 12
```

The output JSON's `quality.cleared_gap` / `quality.fell_into_chasm` report the
outcome, and `quality.final_x` / `final_z` the resting place.

## Build PCVE suite

```bash
python scripts/car_gap_jump/build_pcve_car_gap_jump.py \
  --out-root renders/pcve_car_gap_jump_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 4.5 \
  --samples 128 \
  --device auto
```

The suite holds the ramp, car, and decks fixed and varies only the two knobs
that gate the jump -- the launch speed and the gap width -- so each case lands on
the far deck or falls into the chasm (all verified against
`simulate_car_gap_jump.py`):

| case_id | speed (m/s) | gap (m) | outcome |
|---|---|---|---|
| `car_gap_jump_clears` | 12 | 8 | Clears the gap, lands upright on the far deck (~19 m past the lip). |
| `car_gap_jump_barely_clears` | 11 | 8 | Just reaches the far deck (~15 m) -- a narrow success. |
| `car_gap_jump_too_slow` | 9 | 8 | Too slow: the arc falls short and the car plummets into the chasm. |
| `car_gap_jump_wide_gap` | 12 | 12 | Same firm launch, but the wider gap is too far -- it drops into the chasm. |

Outputs are written under `cases/<case_id>/` with `video.mp4`,
`ground_truth_transforms.json`, and `scenario_metadata.json`; the suite root
holds `pcve_manifest.json` with case descriptions, params, and commands.

## Scene layout

- The car travels along world `+X`. The approach deck top is at `z = 0`; the car
  starts on it a few metres behind the ramp base, given an initial `+X` velocity
  (`launch_speed`).
- The take-off ramp is a `13 m` slab tilted up at `12°` (longer than the car so
  it rides up and pitches cleanly instead of see-sawing on the lip), with its
  launch lip at `x = 0`, `z ≈ 2.7 m`.
- The gap spans `x = 0` to `x = gap_width`; the far landing deck (top at `z = 0`)
  begins at `x = gap_width` and runs `45 m` so a fast jump lands on it and rolls
  to a stop rather than overshooting the end.
- A procedural rock chasm floor sits `8 m` below the decks, so a short jump
  visibly plummets.
- The car's collision proxy is a box with half-extents
  `[2.35, 0.95, 0.67] m` (a real GT-R footprint); the detailed mesh is parented
  to the physics transform, origin at the box center.

## Rendering notes

- **Clear-sky HDRI.** The world is lit by Poly Haven
  `syferfontein_0d_clear_puresky` plus a sun. Download it (and the ambientCG
  `Asphalt031` road texture) with `scripts/download_render_assets.py` if absent;
  the road/ramp fall back to a flat dark material and the chasm to procedural
  rock when textures are missing.
- **Camera** is a low side view at roughly deck height, looking perpendicular to
  travel: the decks read as a horizontal band low in frame, the gap as a break,
  and the car arcs above the band against the sky -- the iconic gap-jump
  silhouette. The elevation/drop reads through the gap rather than from a
  top-down angle.
- The car's origin is its geometric center so the per-frame physics quaternion
  spins it about its center (a bottom-pivot origin would make the airborne car
  visibly wobble).

## Key parameters

- `launch_speed`: initial approach speed in m/s (default `12.0`). The scene's
  main knob -- faster clears the gap, slower falls short. The clear/fall-short
  threshold for the default 8 m gap is around `10.5–11 m/s`.
- `gap_width`: width of the broken-bridge gap in metres (default `8.0`). Wider
  gaps require more speed; past what the launch can reach, the car falls in.
- `seed`: seeds the render-side RNG for reproducibility.

## Outputs

- `car_gap_jump.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `car_gap_jump.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame world matrix and velocities for the
  car, plus the camera matrix.
- `scenario_metadata.json` – the render/physics parameters used.
