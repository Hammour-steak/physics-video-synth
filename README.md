# Physics Video Synth

This project generates deterministic physical-interaction videos and matching
ground-truth trajectories for PCVE. PyBullet owns rigid-body simulation and
Blender Cycles owns rendering. The maintained scenes cover ball/block impact,
drop, wall rebound, grounded-incline motion, and a three-domino gravity chain.

The current renderer builds a more lived-in room scene around the impact:
painted walls, baseboards, distant background objects, optional muted surface
marks, seed-controlled physical variation, fixed-camera rendering by default,
4K floor/block PBR texture sets with AO/roughness/normal/height maps,
UV-projected block faces, rubber normal/roughness/height detail on the ball,
calibrated focus distance, and final video noise/vignette/lens post-processing.

## Project Layout

- Source scripts: `scripts/`
- Downloaded CC0 render assets: `assets/polyhaven/`, `assets/ambientcg/`
- Optional local Blender install: `tools/blender-3.6.23-linux-x64/`
- Generated videos, previews, `.blend` files, transforms, and metadata: `renders/`

`assets/`, `tools/`, and `renders/` are local runtime/cache directories. They
are intentionally kept outside the source surface with `.gitignore`; regenerate
assets with `scripts/download_render_assets.py` and choose a fresh `--out-dir`
for new renders. Historical comparison renders can live under `renders/archive/`
without mixing into new batch output roots.

## Default Render Settings

- Duration: 8 seconds
- FPS: 24
- Frames: 192
- Resolution: 1280 x 720
- Renderer: Blender Cycles
- Samples: 32
- Physics: PyBullet, 12 substeps per rendered frame
- Visual assets: Poly Haven `brown_photostudio_05`, `wood_floor_worn`,
  `wood_table`; ambientCG `Rubber002` 4K
- Device: OptiX/GPU when available, CPU fallback otherwise

Suggested segmentation prompt: `ball.block`.

Each animation output directory contains `ball_block_impact.mp4`, a
baseline H.264/yuv420p compatibility video with constant 24 fps, faststart
metadata, and a silent AAC track. The same directory also receives the `.blend`,
ground-truth transforms, and scenario metadata.

## Re-render

Use Blender 3.6 LTS or a compatible Blender build with Cycles enabled. The
batch script resolves Blender in this order: explicit `--blender`, `BLENDER_BIN`,
`tools/blender-3.6.23-linux-x64/blender`, then `blender` on `PATH`. The render
script calls `python3` for the PyBullet simulation, so `pybullet` must be
available in that Python environment.

All commands below assume the current working directory is the repository root.

Install the Python runtime dependency:

```bash
python3 -m pip install -r requirements.txt
```

Download or refresh the local render assets. The render script requires these
Poly Haven and ambientCG assets so every render uses the same PBR inputs:

```bash
python3 scripts/download_render_assets.py
```

```bash
${BLENDER_BIN:-blender} -b \
  --python scripts/render_ball_block_impact.py -- \
  --mode animation \
  --out-dir renders/current_side_impact \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 8 \
  --samples 32 \
  --device auto \
  --seed 7 \
  --motion side_impact \
  --block-texture-asset wood_table \
  --camera-jitter 0 \
  --surface-marks none
```

Use `--motion drop_onto_block` for a top-down drop where the ball falls onto
the wooden block instead of launching from the side. Add
`--drop-x-velocity 0.55 --drop-y-velocity 0.06` to give that falling ball a
small horizontal initial velocity. Use `--block-texture-asset stained_pine`
only when you explicitly want the warmer alternate block maps.

Deterministic scenarios can also be replayed from metadata:

```bash
${BLENDER_BIN:-blender} -b \
  --python scripts/render_ball_block_impact.py -- \
  --mode animation \
  --out-dir renders/replay_case \
  --scenario-json renders/example/scenario_metadata.json
```

Use `--scenario-overrides-json overrides.json` to recursively override a sampled
scenario while preserving the normal render pipeline. This is how benchmark
suites pin exact object locations, velocities, masses, and camera settings.

For PCVE benchmark examples, keep `--block-texture-asset wood_table` across all
cases. The motion suite builder treats other block textures as stale outputs so
comparisons do not mix texture quality with motion-fitting behavior.

## PCVE Motion Suite

The named PCVE suite builder creates a small deterministic set of ball/block
motion cases with standardized output names:

```bash
python3 scripts/build_pcve_motion_suite.py \
  --out-root renders/pcve_general_motion_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 8 \
  --samples 32 \
  --device auto \
  --skip-existing
```

Outputs are written under `cases/<case_id>/` with `video.mp4`,
`ground_truth_transforms.json`, `scenario_metadata.json`, and, for newly
rendered cases, `scenario_overrides.json`. The suite root also contains
`suite_manifest.json` with case descriptions, commands, and output paths.

## Reproduce The Six PCVE Cases

The PCVE repository publishes a compact six-case snapshot under
`benchmarks/pcve_six_cases/`. Reproduce its five ball/wood cases with one suite
command (repeat `--case-id` exactly as shown):

```bash
python3 scripts/build_pcve_motion_suite.py \
  --out-root renders/pcve_six_cases \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 8 \
  --samples 32 \
  --device auto \
  --case-id drop_centered_soft \
  --case-id drop_lateral_mild \
  --case-id existing_side_impact_wood_table \
  --case-id wall_bounce \
  --case-id wood_incline_grounded_slide_falloff_v3
```

Generate the three-domino case with its dedicated scene entrypoint:

```bash
${BLENDER_BIN:-blender} -b \
  --python scripts/render_domino_chain.py -- \
  --mode animation \
  --out-dir renders/pcve_six_cases/cases/domino \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 8 \
  --samples 32 \
  --device auto \
  --seed 3501 \
  --spacing 0.68 \
  --first-tilt-deg 12
```

Each command is deterministic when Blender, PyBullet, render assets, and GPU
renderer are held fixed. Small floating-point or ray-tracing differences may
remain across Blender, CUDA, and driver versions. The committed videos and GT
JSON files are therefore the evaluation reference; regenerated outputs are a
reproducibility check rather than a byte-for-byte fixture.

Run the pure-Python physics tests without scanning the local Blender tree:

```bash
python3 -m pytest -q
```

## Batch Render

For a quick smoke test, render several deterministic preview samples:

```bash
python3 scripts/batch_render_ball_block_impact.py \
  --out-root renders/batch_preview \
  --count 4 \
  --mode preview \
  --resolution 320 180 \
  --samples 8
```

For dataset-style videos, use animation mode:

```bash
python3 scripts/batch_render_ball_block_impact.py \
  --out-root renders/batch_animation \
  --count 16 \
  --mode animation \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 8 \
  --samples 32 \
  --device auto \
  --motion side_impact \
  --block-texture-asset wood_table \
  --camera-jitter 0 \
  --surface-marks none
```

Each sample is written to `sample_0000/`, `sample_0001/`, etc. Every sample
contains its own video or preview, `.blend`, `ground_truth_transforms.json`, and
`scenario_metadata.json`. The batch root also contains `batch_manifest.json`
with seeds, output paths, and the exact Blender commands used.
