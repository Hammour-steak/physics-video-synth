# Picnic apple-drop ball roll

An apple falls from above and lands an off-center hit on the top of a soccer
ball resting on a grass lawn next to a laid-out picnic. The oblique hit torques
the ball into a right-to-left roll across the grass, and it coasts to a stop
after about half a metre while the apple bounces off and settles nearby. Like
`dining_chain` and `car_ramp_climb`, nothing is scripted per object: the roll is
entirely emergent from the apple's fall plus the off-center contact.

The physical crux is the lever arm. The contact point between two spheres always
lies on the line joining their centers, so an apple falling *dead-center* above
the ball pushes straight through its center and imparts no spin -- the ball is
just pressed into the grass and stays put. Offsetting the apple's start position
horizontally from the ball's center is what makes the hit oblique enough to
actually roll the ball rather than squash it straight down (see the PCVE suite,
which contrasts exactly this).

The lawn is built procedurally -- there is no ground GLB. A grid mesh carries a
hair-particle system whose instance is a small bent, tapered blade, with a
noise-driven ground shader underneath. A density vertex group zeroes out blade
emission in the footprint under the picnic blanket so grass doesn't poke up
through the thin cloth. The soccer ball, apple, and picnic spread are the three
downloaded GLBs, each uniformly rescaled to real-world size:
`assets/models/soccer_ball.glb` (0.22 m FIFA size-5 ball),
`assets/models/apple.glb` (0.075 m apple), and
`assets/models/french_picnic.glb` (blanket + basket + food, scaled to a 2.1 m
family-size blanket). The ball and apple get a spherical collision proxy for the
physics; the detailed meshes are visual-only.

Motion is simulated with PyBullet and rendered in Blender Cycles under the
Poly Haven `syferfontein_0d_clear_puresky` clear-sky HDRI plus a sun and a soft
fill light.

## Files

- `simulate_picnic_apple_ball.py` – PyBullet physics simulation (a ball and an
  apple as sphere proxies on a ground plane; the apple is dropped with an
  off-center XY offset from the ball).
- `render_picnic_apple_ball.py` – Blender rendering script: builds the grass
  lawn, imports and rescales the three GLBs, runs the simulation, applies the
  trajectory as keyframes, lights the scene, and renders.
- `batch_render_picnic_apple_ball.py` – orchestrates multiple randomized renders.
- `build_pcve_picnic_apple_ball.py` – builds the PCVE suite (does the ball roll,
  and how far?).

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame (the moment of impact)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/picnic_apple_ball/render_picnic_apple_ball.py -- \
    --mode preview \
    --out-dir renders/picnic_apple_ball_preview \
    --resolution 1280 720 \
    --samples 96 \
    --device auto \
    --preview-frame 14

# Render the full animation (108 frames at 24 fps = 4.5 s)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/picnic_apple_ball/render_picnic_apple_ball.py -- \
    --mode animation \
    --out-dir renders/picnic_apple_ball \
    --resolution 1280 720 \
    --fps 24 \
    --duration-sec 4.5 \
    --samples 128 \
    --device auto
```

## Batch render

```bash
python scripts/picnic_apple_ball/batch_render_picnic_apple_ball.py \
  --mode animation \
  --count 4 \
  --seed-base 7000 \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 4.5 \
  --samples 128 \
  --device auto \
  --out-root renders/batch_picnic_apple_ball
```

Each sample jitters the hit's lever arm (`apple_offset_x` 0.08–0.125 m), the
drop height (1.1–1.8 m), and the grass friction (0.18–0.32) within ranges that
keep a clearly visible roll but vary its length and timing. Each sample lands in
`sample_0000/`, `sample_0001/`, etc., with its own video, `.blend`,
`ground_truth_transforms.json`, and `scenario_metadata.json`; the batch root
also holds `batch_manifest.json` with seeds, sampled params, and the exact
commands used.

## Simulate only

```bash
python scripts/picnic_apple_ball/simulate_picnic_apple_ball.py \
  --out renders/test_picnic_physics.json \
  --fps 24 \
  --duration-sec 4.5
```

Compare the roll directly from the physics output, no rendering:

```bash
# off-center hit -> the ball rolls
python scripts/picnic_apple_ball/simulate_picnic_apple_ball.py \
  --out renders/test_roll.json --apple-offset-x 0.105
# dead-center hit -> no lever arm, the ball does not roll
python scripts/picnic_apple_ball/simulate_picnic_apple_ball.py \
  --out renders/test_no_roll.json --apple-offset-x 0.0
```

The output JSON's `quality.ball_roll_distance` reports how far the ball traveled.

## Build PCVE suite

```bash
python scripts/picnic_apple_ball/build_pcve_picnic_apple_ball.py \
  --out-root renders/pcve_picnic_apple_ball_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 4.5 \
  --samples 128 \
  --device auto
```

The suite holds the models and placement fixed and varies only what gates the
roll -- the hit's lever arm, the grass friction, and the drop height -- so each
case ends with the ball at a different distance (all verified against
`simulate_picnic_apple_ball.py`):

| case_id | offset (m) | drop (m) | friction | outcome |
|---|---|---|---|---|
| `picnic_apple_ball_baseline` | 0.105 | 1.3 | 0.25 | Off-center hit; ball rolls ~0.54 m right→left and stops. |
| `picnic_apple_ball_centered_no_roll` | 0.0 | 1.3 | 0.25 | Dead-center hit, no lever arm: the ball does not roll at all. |
| `picnic_apple_ball_far_roll` | 0.115 | 1.3 | 0.12 | Slicker grass + bigger offset: the ball rolls ~1.12 m. |
| `picnic_apple_ball_short_roll` | 0.105 | 1.3 | 0.60 | High-friction turf damps the roll to ~0.19 m. |
| `picnic_apple_ball_high_drop` | 0.105 | 2.2 | 0.25 | Nearly double the drop height; harder impact rolls ~0.84 m. |

Outputs are written under `cases/<case_id>/` with `video.mp4`,
`ground_truth_transforms.json`, and `scenario_metadata.json`; the suite root
holds `pcve_manifest.json` with case descriptions, params, and commands.

## Scene layout

- The simulation runs in a world frame with the ground plane at `z = 0`. The
  ball starts at the origin `(0, 0)` resting on the grass (center at
  `z = 0.11`, its radius). The apple starts `apple_offset_x` metres along `+X`
  from the ball's center and `drop_height` metres above the top of the ball,
  then falls under gravity.
- The ball rolls toward `-X`, which reads as right→left across the frame for
  the default camera (looking back along `+Y` from `(0, -2.0, 1.5)`). The apple
  deflects the opposite way and settles to the ball's right.
- The picnic blanket sits centered and set back at `(0.05, 1.55)` with a small
  `-15°` yaw, fully in frame in the background; the ball/apple impact plays out
  in the foreground close to the camera.
- Collision proxies are spheres: ball radius `0.11 m`, apple radius `0.0375 m`.
  Both are visual-mesh-independent -- the detailed GLBs are parented to the
  physics transforms.

## Rendering notes

- **Clear-sky HDRI.** The world is lit by Poly Haven
  `syferfontein_0d_clear_puresky` (a bright, cloudless daytime sky), plus a sun
  aligned to its implied light direction and a soft cool area fill so the shadow
  side of the ball/apple doesn't go black. Download it (and the other CC0
  assets) with `scripts/download_render_assets.py` if it isn't present.
- **Camera** is a near-level front-quarter view tuned so no horizon/sky edge is
  visible -- the grass fills the frame to the top. The ground patch is sized
  (22 m) well past the frame so its edge never shows.
- **Grass under the blanket** is suppressed via a `grass_density` vertex group
  on the emitter so blades don't clip up through the thin cloth mesh; the mask
  is tucked just inside the blanket edge so there's no bald ring around it.
- The physics-driven objects (ball, apple) have their origins set to their true
  geometric center so the per-frame rotation quaternion spins them about their
  center -- setting the origin at the mesh bottom instead makes a rolling sphere
  visibly wobble in and out of the ground.

## Key parameters

- `apple_offset_x`: the apple's horizontal start offset from the ball's center
  in metres (default `0.105`). This is the scene's main knob -- it's the lever
  arm that converts the vertical drop into roll. `0.0` gives a centered hit with
  no roll; larger values (up to ~`0.13`, past which the apple misses the top of
  the ball) give a stronger oblique hit.
- `drop_height`: how far above the ball the apple starts, in metres (default
  `1.3`). Higher means a faster, harder impact and a longer roll.
- `grass_friction`: ground lateral friction (default `0.25`). Lower = the ball
  rolls further (slick grass); higher = it stops sooner.
- `grass_density`: number of hair particles on the lawn (default `260000`).
- `seed`: seeds the render-side RNG for reproducibility.

## Outputs

- `picnic_apple_ball.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `picnic_apple_ball.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame world matrices and velocities for
  the ball and apple, plus the camera matrix.
- `scenario_metadata.json` – the render/physics parameters used.
