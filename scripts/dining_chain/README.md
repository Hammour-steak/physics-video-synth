# Dining-table sliding collision chain

A cola can is given a single push across a dining tabletop; it slides into a
soda cup, and the cup in turn slides into a small milk carton -- a three-object
momentum relay where each object is knocked into the next. Like `domino_chain`
and `car_ramp_climb`, nothing is scripted per object: the whole chain is
emergent from the can's one initial velocity plus contact physics. The three
containers are low, flat-based drinks, so they *slide* along the table rather
than roll or topple.

The environment is `assets/models/dining_room__kichen_baked.glb` (a full baked
dining-room/kitchen interior, used as static background at its native metre
scale -- its table, chairs, wall, window and floor are what make the scene read
as a real room). The three drinks are separate downloaded models:
`assets/models/simple_cola_can.glb`, `assets/models/fast_food_soda_cup.glb`,
and `assets/models/milk_packaging.glb`, each uniformly rescaled to its
real-world size (the raw GLBs range from ~2-5 blender-units for the can/cup to
near-metres for the carton) and given a simple cylinder/box collision proxy for
the physics; the detailed meshes are visual-only.

The collisions are **mostly inelastic** (restitution `0.1`): a struck object
does not bring the striker to a stop and fly off with its velocity (that
"velocity exchange" needs a near-elastic restitution and equal masses). Instead
the striker keeps roughly half its speed and the two slide on together, losing
energy -- the realistic behaviour for solid drink containers thudding into each
other.

## Files

- `simulate_dining_chain.py` – PyBullet physics simulation (three proxies on a
  tabletop plane; the can gets the only initial velocity).
- `render_dining_chain.py` – Blender rendering script: imports the room and the
  three drink GLBs, rescales/places them, runs the simulation, applies the
  trajectory as keyframes, lights the interior (no HDRI), and renders.
- `batch_render_dining_chain.py` – orchestrates multiple randomized renders.
- `build_pcve_dining_chain.py` – builds the PCVE suite (does the chain reach the
  end, or die short?).

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/dining_chain/render_dining_chain.py -- \
    --mode preview \
    --out-dir renders/dining_chain_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 96 \
    --device auto \
    --preview-frame 30

# Render the full animation (72 frames at 24 fps = 3 s)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/dining_chain/render_dining_chain.py -- \
    --mode animation \
    --out-dir renders/dining_chain \
    --resolution 1280 720 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 128 \
    --device auto
```

## Batch render

```bash
python scripts/dining_chain/batch_render_dining_chain.py \
  --mode animation \
  --count 4 \
  --seed-base 27000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 96 \
  --device auto \
  --out-root renders/batch_dining_chain
```

Each sample jitters `launch_speed` (3.0-3.6 m/s) and `table_friction`
(0.28-0.32) within ranges that keep the full chain but vary the timing/spread.

## Simulate only

```bash
python scripts/dining_chain/simulate_dining_chain.py \
  --out renders/test_dining_chain_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

Compare the chain's reach directly from the physics output, no rendering:

```bash
# default grippy top + firm push -> full chain reaches the carton
python scripts/dining_chain/simulate_dining_chain.py \
  --out renders/test_full.json
# tablecloth (high friction) -> chain dies before the carton
python scripts/dining_chain/simulate_dining_chain.py \
  --out renders/test_cloth.json --table-friction 0.90
```

## Build PCVE suite

```bash
python scripts/dining_chain/build_pcve_dining_chain.py \
  --out-root renders/pcve_dining_chain_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device auto
```

The suite holds the objects and their spacing fixed and varies only what gates
the chain -- the tabletop friction and the push -- so each case reaches a
different point before it dies:

| case_id | friction | push | outcome |
|---|---|---|---|
| `dining_chain_full` | 0.30 | 3.3 | Full chain: can → cup → milk carton, all three slide. |
| `dining_chain_stops_at_cup` | 0.30 | 1.8 | Reaches the cup, but the cup stalls before the carton -- dies short for lack of speed. |
| `dining_chain_cloth_stops` | 0.90 | 3.3 | Same firm push, but a cloth damps every slide -- reaches the cup, dies before the carton. |
| `dining_chain_no_chain` | 0.30 | 0.8 | The can stops short of the cup; no collision happens at all. |

## Scene layout

- The simulation runs directly in the room's world frame: the tabletop surface
  is at `z = 0.778 m` (measured from the table mesh), and the chain is laid out
  along `+Y` at `x = 1.15` -- a clear stretch of tabletop off to the side of the
  centre vase. Because the physics already runs in world coordinates, the
  render applies the per-frame transforms directly (no placement indirection).
- Start positions along `+Y`: can `-1.20`, cup `-0.70`, milk `-0.20` (gaps of
  0.50 m each). The gaps are deliberately wide so each object visibly slides
  before the next hit (can hits cup ~frame 5, cup hits milk ~frame 11,
  everything settled ~frame 36) instead of firing in the first few frames.
- The tabletop friction is grippy (`0.30`, effective ~`0.09` after PyBullet
  combines it with the object friction) -- a realistic finished-wood top where
  the objects clearly decelerate as they slide. Rather than lowering friction
  to stretch the timing (which reads like ice), the action is spread out in
  *distance* via the wide gaps and a firm push (`3.3 m/s`); a cloth-level
  friction damps each slide and the chain dies early (see the PCVE suite).
- Collision proxies: can/cup are cylinders (r 0.034 / 0.044 m, h 0.122 / 0.16
  m), the milk carton a box (~0.055 × 0.056 × 0.125 m, sized close to the can
  rather than a full 20 cm carton). All are low and flat-based so they slide
  upright instead of toppling.

## Rendering notes

- **No HDRI.** The baked interior is lit by a neutral world fill plus a warm key
  light over the table and a cool fill from the window side. The room's baked
  textures are very pale/high-albedo, so it reads bright under any lighting; a
  color-management exposure of `-0.8` stops knocks the whole image down to a
  calmer level.
- **Camera** is a near-level front view facing the chain along `-X`, so the
  three drinks stand side-by-side and slide left→right across the frame. The
  dining chairs on the camera (`+X`) side are culled (their geometry past the
  table edge is deleted) so they don't block this near-level view; the far
  chairs stay as background.

## Key parameters

- `launch_speed`: the can's initial push in m/s (default `3.3`). Firm enough to
  carry the chain across the wide gaps on the grippy top; lower and the chain
  dies short, too high and the carton reaches the table edge.
- `table_friction`: tabletop lateral friction (default `0.30`, effective ~`0.09`
  combined with the object friction) -- the scene's main knob. This grippy value
  makes the objects decelerate realistically; higher (a cloth) damps each slide
  so the chain dies early.
- `restitution`: collision bounciness (default `0.1`, mostly inelastic). Near
  `1.0` with equal masses would instead give a billiard/Newton's-cradle
  velocity exchange, but reads less like real drink containers.
- `object_friction`, `gravity_z`: per-object friction and gravity.

## Outputs

- `dining_chain.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `dining_chain.blend` – saved Blender scene.
