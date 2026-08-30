# Tennis flight scene

A tennis ball is launched from `(-4, 0, 1.5)` with initial velocity
`(4.5, 0, 4.5)` m/s. The parabola apexes near `z=2.5 m`, clears the net
at `x=0` (net height ≈ 0.91 m), first touches down at `x≈1.05` (≈1.17 s),
then rolls out to `x≈3.3` before stopping. The camera is a tight
broadcast-style side view at `(0, -10, 2.5)`, 35 mm, targeting the net.
The ball is inside the frame from the very first rendered frame.

This scene follows the same coupled simulate-render-build pipeline as `scripts/ramp_collision`:

1. `simulate_tennis_flight.py` runs PyBullet and writes a trajectory JSON.
2. `render_tennis_flight.py` builds the Blender scene, calls the simulator, applies the trajectories as object keyframes, exports `ground_truth_transforms.json`, and renders the video.
3. `batch_render_tennis_flight.py` orchestrates multiple renders.
4. `build_pcve_tennis_flight.py` builds a named PCVE benchmark suite.

## Quick render

```bash
python scripts/tennis_flight/batch_render_tennis_flight.py \
  --mode animation \
  --count 1 \
  --seed-base 3000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device cpu \
  --out-root renders/test_tennis_video
```

The output video is written to `renders/test_tennis_video/sample_0000/tennis_flight.mp4`.

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/tennis_flight/render_tennis_flight.py -- \
  --mode animation \
  --out-dir renders/test_tennis_video \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device cpu \
  --seed 3000
```

The output video is written to `renders/test_tennis_video/tennis_flight.mp4`.

## Simulate only

```bash
python scripts/tennis_flight/simulate_tennis_flight.py \
  --out renders/test_tennis_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

## Build PCVE suite

```bash
python scripts/tennis_flight/build_pcve_tennis_flight.py \
  --out-root renders/pcve_tennis_flight_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 4.0 \
  --samples 32 \
  --device auto
```

Edit cases (all deltas measured against the baseline above, at
`--duration-sec 4.0`):

| case | DSL | outcome |
|---|---|---|
| `edit_slick_ball` | `SET ball.friction FROM 0.5 TO 0.1` | same landing at x≈1.06; still rolling at end, x≈8.8 (vs 3.3 baseline) |
| `edit_grippy_ball` | `SET ball.friction FROM 0.5 TO 1.5` | same landing at x≈1.05; stops almost on the spot at x≈1.38 |
| `edit_bouncy_ball` | `SET ball.restitution FROM 0.05 TO 0.9` | same landing at x≈1.05; several visible hops before settling at x≈3.48 |
| `edit_soft_serve` | `SET ball.initial_velocity FROM 4.5 TO 2.5` | ball fails to clear the net, lands short at x≈-1.16, finishes at x≈-0.49 |
| `edit_hard_serve` | `SET ball.initial_velocity FROM 4.5 TO 6.5` | flies further, lands at x≈3.23, rolls out to x≈7.11 |
