# Mahjong dice drop

Two dice tumble down and land in the center tray of an automatic mahjong
table. It uses `assets/models/riichi_mahjong.glb`: the full table (tile
racks, glass cover, dice shaker) is kept as a realistic static background,
and the two decorative dice already sitting in the tray are pulled out and
driven by PyBullet instead.

## Files

- `simulate_mahjong_dice.py` – PyBullet physics simulation.
- `render_mahjong_dice.py` – Blender rendering script that imports the
  mahjong table GLB, keeps the whole table as background, detaches the two
  dice meshes, and applies the physics trajectories as keyframes.
- `batch_render_mahjong_dice.py` – orchestrates multiple randomized renders.
- `build_pcve_mahjong_dice.py` – builds a named PCVE benchmark suite with
  five parameter variations.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/mahjong_dice/render_mahjong_dice.py -- \
    --mode preview \
    --out-dir renders/mahjong_dice_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/mahjong_dice/render_mahjong_dice.py -- \
    --mode animation \
    --out-dir renders/mahjong_dice \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/mahjong_dice/batch_render_mahjong_dice.py \
  --mode animation \
  --count 1 \
  --seed-base 7000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_mahjong_dice_video
```

The output video is written to
`renders/test_mahjong_dice_video/sample_0000/mahjong_dice.mp4`.

## Simulate only

```bash
python scripts/mahjong_dice/simulate_mahjong_dice.py \
  --out renders/test_mahjong_dice_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/mahjong_dice/build_pcve_mahjong_dice.py \
  --out-root renders/pcve_mahjong_dice_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `mahjong_dice_baseline` | Baseline pair of dice tumbling and landing in the tray. |
| `mahjong_dice_high_drop` | Higher drop height, more tumbling before landing. |
| `mahjong_dice_bouncy` | Higher restitution, dice bounce more before settling. |
| `mahjong_dice_heavy` | Heavier dice settle faster with less bounce. |
| `mahjong_dice_low_friction` | Slicker surface, dice slide further after landing. |

## Scene layout

- Geometry comes directly from `riichi_mahjong.glb`: the two decorative dice
  in the table's center tray are `0.0833`-unit cubes resting on a flat
  surface at world Z `0.9382` (both dice share the exact same height,
  confirming the tray is level).
- The dice drop straight down from directly above their original decorative
  spots with no initial spin -- any rotation they show comes only from the
  bounce itself, not a scripted tumble.
- The rest of the table -- tile racks, glass cover, dice shaker -- stays in
  place as a static, detailed background for visual realism.
- Camera is a close macro-style shot centered on the tray, matching the
  small scale of the dice.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_mahjong_dice.py` for suite cases.

- `die_edge`: cube edge length in meters (default `0.0833`, taken from the asset).
- `die_mass`: mass of each die in kilograms (default `0.006`).
- `die_friction`: lateral friction of the dice (default `0.5`).
- `die_restitution`: controls bounce on impact (default `0.72`). The floor
  body's own restitution is set to match this value automatically --
  PyBullet combines both contacting bodies' restitution, so a low floor
  restitution silently caps the bounce no matter how high `die_restitution`
  is set (this was a real bug here: the floor was hardcoded to `0.2`).
  `0.72` produces a bounce that rises about `0.18` units above the tray (as
  measured from the ground-truth trajectory) and lands again right around
  the end of a 3-second clip -- at this camera's distance/angle, smaller
  bounces (under roughly `0.1` units) don't read as visibly airborne on
  screen even though the underlying physics is correct.
- `drop_height`: how far above the tray surface the dice start (default
  `0.6`; kept low enough that the dice are still inside the camera frame at
  frame 1, instead of falling in from above the shot -- `0.8` was already
  clipped at the top edge in testing).
- `floor_friction`: lateral friction of the tray surface (default `0.55`).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -0.6]`, well
  below Earth gravity so the fall + bounce + settle plays out over about 3
  seconds instead of resolving almost instantly. This value and
  `die_restitution` are tuned together: gravity sets the timescale of the
  bounce, restitution sets its height, and this pair was chosen so the
  bounce cycle completes right around the end of a 3-second clip instead of
  either finishing too early (looks like nothing happened) or still being
  airborne when the video cuts off.

## Outputs

- `mahjong_dice.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `mahjong_dice.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms.
- `scenario_metadata.json` – seed, camera, lighting and physics parameters.
