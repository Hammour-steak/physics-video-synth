# Domino chain reaction

A row of dominoes standing on a tabletop topples one after another. It uses
`assets/models/domino_test.glb` for the tabletop and a single domino tile
mesh; the tile is duplicated into a straight row. The first tile starts
frame 1 already pre-tilted just past its critical tipping angle, at rest
(zero velocity) -- not kicked with an injected impulse -- so its fall is
pure, unforced gravity + contact physics from the very first frame, exactly
like every later tile's contact-triggered fall (no per-tile scripting).
Spacing is wide enough that each tile can only knock over its immediate
neighbor: remove one tile from the row and the chain reaction stops dead,
it cannot jump the gap.

## Files

- `simulate_domino_chain.py` – PyBullet physics simulation.
- `render_domino_chain.py` – Blender rendering script that imports the
  domino GLB, keeps the tabletop plane and one domino tile, duplicates the
  tile into a row, and applies the physics trajectories as keyframes.
- `batch_render_domino_chain.py` – orchestrates multiple randomized renders.
- `build_pcve_domino_chain.py` – builds a named PCVE benchmark suite with
  five parameter variations.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/domino_chain/render_domino_chain.py -- \
    --mode preview \
    --out-dir renders/domino_chain_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 96 \
    --device cpu

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/domino_chain/render_domino_chain.py -- \
    --mode animation \
    --out-dir renders/domino_chain \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 256 \
    --device auto
```

## Batch render

```bash
python scripts/domino_chain/batch_render_domino_chain.py \
  --mode animation \
  --count 1 \
  --seed-base 6000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_domino_chain_video
```

The output video is written to
`renders/test_domino_chain_video/sample_0000/domino_chain.mp4`.

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/domino_chain/render_domino_chain.py -- \
  --mode animation \
  --out-dir renders/test_domino_chain_video \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device cpu \
  --seed 6000
```

The output video is written to `renders/test_domino_chain_video/domino_chain.mp4`.

## Simulate only

```bash
python scripts/domino_chain/simulate_domino_chain.py \
  --out renders/test_domino_chain_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/domino_chain/build_pcve_domino_chain.py \
  --out-root renders/pcve_domino_chain_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 32 \
  --device auto
```

The suite contains the following cases:

| case_id | description |
|---|---|
| `domino_baseline` | Baseline row of four dominoes toppling in sequence. |
| `domino_long_chain` | Longer row of sixteen dominoes for an extended cascade. |
| `domino_wide_spacing` | Wider spacing between tiles, closer to the topple-reach limit. |
| `domino_light_push` | Smaller initial push angle giving a slower first fall. |
| `domino_bouncy` | Higher restitution makes toppled tiles bounce more before settling. |

## Scene layout

- The row is laid out along world X, centered on the origin, on the tabletop
  plane imported from `domino_test.glb`.
- Domino tile size (from the asset): thickness (row axis) `0.20`, width
  `0.70`, height `1.30`.
- The first tile starts frame 1 already tilted just past its critical
  tipping angle, at rest -- no velocity impulse is ever injected, so its
  motion looks the same as every later tile's contact-driven fall (a slow
  start that gradually accelerates), not like it was flicked or shoved.
- Tile spacing (`0.8`) is deliberately wider than a single tile's own
  topple reach but well within reach of its immediate neighbor: a tile can
  knock over the one right next to it, but not skip over a missing one.
- Camera views the row side-on so the cascading fall reads left-to-right.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` when
rendering, or directly in `build_pcve_domino_chain.py` for suite cases.

- `domino_count`: number of tiles in the row (default `4`).
- `domino_spacing`: center-to-center spacing along the row in meters
  (default `0.8`). Chosen so a tile can only knock over its immediate
  neighbor -- removing one tile from the row leaves a gap the next tile's
  own topple reach cannot cross (verified: with the second tile removed,
  the third tile stays upright at `0°` tilt for the whole simulation).
  Values below roughly `0.55` let the reach jump a missing tile; values
  above roughly `1.0` start to strain a single hop too.
- `domino_mass`: mass of each tile in kilograms (default `0.12`).
- `domino_friction`: lateral friction of the tiles (default `0.6`).
- `domino_restitution`: controls bounce on impact (default `0.05`).
- `push_angle_deg`: the critical tipping angle the first tile starts at on
  frame 1, at zero velocity, after which it falls under gravity alone like
  every other tile (default `12.0`; must clear roughly `8.7°` --
  `arctan(domino_thickness / domino_height)` -- or the tile settles back
  upright instead of falling).
- `floor_friction`: lateral friction of the tabletop (default `0.6`).
- `gravity`: gravitational acceleration (default `[0.0, 0.0, -9.81]`).

## Outputs

- `domino_chain.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `domino_chain.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame object and camera transforms.
- `scenario_metadata.json` – seed, camera, lighting and physics parameters.
