# Car gap jump (indoor tabletop stunt)

A 1:24 die-cast toy sports car rolls along the top of a stack of hardback books
sitting at the edge of a dining-room table, shoots off the front of the stack,
and either clears the gap to a second table of the same height and skids to a
stop on it, or falls short and drops to the room floor below. Like
`car_ramp_climb` and `dining_chain`, nothing is scripted mid-air: the whole arc
is emergent from the car's one initial push speed plus the launch height and
gravity.

The physical crux is a projectile launch. Once the car leaves the front edge of
the book stack it is a projectile whose range is set by its launch speed and
its height above the landing surface. Because both tables are the same height,
the stack's `0.149 m` is the entire drop the car gets to cross the gap in --
the books are the only reason the jump has any range at all. Enough speed and
the car lands upright on the far table and skids to a stop; too little speed,
or too wide a gap, and the arc falls short and the car drops to the floor (see
the PCVE suite, which contrasts exactly this).

The car is `assets/models/mini_cooper_s.glb`, uniformly rescaled to a 1:24
die-cast Mini Cooper S footprint (~16 cm long) and given a single box collision
proxy for the physics; the detailed mesh is visual-only. The launch pad is
`assets/models/harry_potter_books_stack.glb` (four real hardbacks). The room is
the baked `assets/models/dining_room__kichen_baked.glb` interior; the two
tables are built procedurally (wood-textured slabs, legs and aprons), and
`assets/models/flowers_in_vase.glb` dresses the far end of the landing table.
Motion is
simulated with PyBullet and rendered in Blender Cycles under interior area
lights (no HDRI -- the room geometry is the visible background).

## Files

- `simulate_car_gap_jump.py` – PyBullet physics simulation (a box car proxy
  rolling off a static book-stack box at a table's edge, over a gap, onto a
  second table with the room floor far below).
- `render_car_gap_jump.py` – Blender rendering script: imports the dining room,
  builds the two tables, places the book-stack GLB, imports and rescales the
  GT-R, runs the simulation, applies the trajectory as keyframes, lights the
  scene, and renders.
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
    --preview-frame 5

# Render the full animation (38 frames at 24 fps = 1.6 s)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/car_gap_jump/render_car_gap_jump.py -- \
    --mode animation \
    --out-dir renders/car_gap_jump \
    --resolution 1280 720 \
    --fps 24 \
    --duration-sec 1.6 \
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
  --duration-sec 1.6 \
  --samples 128 \
  --device auto \
  --out-root renders/batch_car_gap_jump
```

Each sample jitters the push speed (1.0–2.2 m/s) and gap width (0.22–0.42 m)
within ranges that straddle the clear/fall-short threshold, so the batch
contains both outcomes. Each sample lands in `sample_0000/`, `sample_0001/`,
etc., with its own video, `.blend`, `ground_truth_transforms.json`, and
`scenario_metadata.json`; the batch root also holds `batch_manifest.json` with
seeds, sampled params, and the exact commands used.

## Simulate only

```bash
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_gap_jump_physics.json \
  --fps 24 \
  --duration-sec 1.6
```

Compare the outcome directly from the physics output, no rendering:

```bash
# firm push, standard gap -> clears and lands on the far table
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_clears.json --launch-speed 2.0 --gap-width 0.28
# same push, wider gap -> falls short to the floor
python scripts/car_gap_jump/simulate_car_gap_jump.py \
  --out renders/test_short.json --launch-speed 2.0 --gap-width 0.42
```

The output JSON's `quality.cleared_gap` / `quality.fell_into_chasm` report the
outcome, and `quality.final_x` / `final_z` the resting place.
`quality.overshot_far_end` flags the separate failure of a push so hard the car
skids across the whole landing table and off its far end.

## Build PCVE suite

```bash
python scripts/car_gap_jump/build_pcve_car_gap_jump.py \
  --out-root renders/pcve_car_gap_jump_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 1.6 \
  --samples 128 \
  --device auto
```

The suite holds the book stack, car, tables, and room fixed and varies only the
two knobs that gate the jump -- the push speed and the gap width -- so each case
lands on the far table or falls to the floor (all verified against
`simulate_car_gap_jump.py`):

| case_id | push (m/s) | gap (m) | outcome |
|---|---|---|---|
| `car_gap_jump_clears` | 2.0 | 0.28 | Clears the gap, lands upright on the far table and skids to a stop ~0.9 m past the launch edge. |
| `car_gap_jump_barely_clears` | 1.6 | 0.28 | Just over the threshold: reaches the far table ~0.1 m past its near edge. |
| `car_gap_jump_too_slow` | 1.3 | 0.28 | Too slow: the arc falls short and the car drops to the floor. |
| `car_gap_jump_wide_gap` | 2.0 | 0.42 | Same firm push, but the wider gap is too far -- it drops to the floor. |

Outputs are written under `cases/<case_id>/` with `video.mp4`,
`ground_truth_transforms.json`, and `scenario_metadata.json`; the suite root
holds `pcve_manifest.json` with case descriptions, params, and commands.

## Scene layout

The physics runs in a "sim frame" whose origin sits on the launch tabletop at
its front edge; `WORLD_OFFSET` in the render script maps that to world
`(0, -1.6, 0.75)` inside the room, i.e. a real 75 cm table height. All
coordinates below are sim-frame.

- The car travels along `+X`. Both tabletops are at `z = 0`; the room floor is
  `0.74 m` below them.
- The launch table runs `x = -1.05` to `x = 0`; the gap spans `x = 0` to
  `x = gap_width`; the landing table starts at `x = gap_width` and runs
  `1.50 m`, long enough that even the fastest sampled push lands and skids out
  on it rather than shooting off the far end.
- The book stack sits on the launch table set back `0.045 m` from its front
  edge, not flush with it: flush collapses the book's front edge and the
  table's front edge into a single line and the books stop reading as resting
  *on* the table. So the top cover -- the runway -- spans `x = -0.286` to
  `x = -0.045`, at `z = 0.149 m`, and the car sails over that last strip of
  tabletop before crossing the gap (even the weakest push in range clears the
  strip easily). The car starts at the back of the cover and is given its whole
  push speed up front.
- Launching from there, the car drops `0.149 m` while crossing the gap. That is
  the whole airtime budget: `t = sqrt(2h/g) ≈ 0.174 s`, so the range is roughly
  `0.174 × push_speed`.
- The car's collision proxy is a box with half-extents
  `[0.080, 0.036, 0.030] m` (a 1:24 Mini Cooper S); the detailed mesh is
  parented to the physics transform, origin at the box center.

### Fitting the physics box to the book-stack GLB

The stack is a photogrammetry scan, and two of its quirks drive how it is
placed. First, it is a shallow wedge: its base and its top cover are about two
degrees out of parallel, because the books fan out. Second, its underside is
bumpy by a few mm and its bounding-box bottom is a single low corner rather
than a face.

The wedge means no rotation can give both a stack that rests flat and a top
that looks level -- level the base and the cover slopes; level the cover and
the stack teeters on one corner with daylight underneath. So the wedge is
removed from the mesh instead:

- `import_book_stack()` first rotates the stack by `STACK_BASE_LEVEL_ROT_*` to
  bring its **measured base plane** horizontal;
- `straighten_stack_cover()` then takes out what is left. It fits a plane to
  the cover and pulls every vertex down by that plane's slope at the vertex's
  own x/y, scaled by how far up the stack it sits. The base (weight 0) does not
  move, the cover (weight 1) comes level, and the books between straighten
  proportionally. It is an ~8 mm correction over a 250 mm stack -- invisible in
  the book shapes, and it drops the cover's residual tilt from `2.0°` to
  `0.01°`.

With both surfaces horizontal the physics box is a plain axis-aligned one whose
top face *is* the cover, and `BOOK_STACK_H` is simply the stack's own thickness,
so resting the cover at that height puts the base on the tabletop. All three of
those numbers are measured at render time and checked against what the physics
assumes, with a warning if they drift apart.

Two measurement details matter. Both surfaces are sampled from the **mesh
vertices**, bucketed on an x/y grid, rather than by raycasting: `ray_cast` works
off a cached BVH and keeps reporting the geometry as it was before the
straightening moved anything. And the underside is taken as the *median* of the
lowest vertex per bucket, not the single lowest vertex, because resting a bumpy
scan on one stray low point leaves the rest of it hovering.

Using the cover's own front edge as the placement landmark matters too: a lower
book protrudes a few mm further forward, so the overall bounding box would
launch the car a few mm past where the visible book ends.

## Rendering notes

- **The room is a baked model, and it had to be un-baked.** Every material in
  `dining_room__kichen_baked.glb` feeds the same baked texture into *both* Base
  Color and Emission, so the room lights itself: with every lamp in the scene
  switched off it still renders fully lit. That is why props standing on the
  tables looked pasted on no matter how the lamps were arranged -- a uniformly
  self-lit room is pure ambient, and pure ambient casts no contact shadow.
  `import_room()` scales that emission down to `ROOM_EMISSION_SCALE` (0.15);
  the baked texture stays on Base Color, so no detail is lost, and a little
  residual emission keeps the far side of the room from going black. Everything
  below only works because of this.
- **Interior lighting, no HDRI.** The key is a *sun*, not a lamp: an area light
  close enough to throw a long shadow also burns a hot spot into the tabletop
  under it, while a sun has no falloff and lights both tables evenly with one
  shadow direction. Its elevation (~35°) is the number that matters -- each
  prop's shadow reaches about 1.4x its own height across the table, toward the
  camera, which is what makes the books and the vase read as sitting on the
  surface. Three soft area lights fill in: a cool daylight wash from the window
  wall, a warm rim from the dining-room side, and a weak front fill that exists
  to keep the camera-facing surfaces (the dark book spines above all) from
  going flat. The front fill is deliberately weak: it lands squarely on the
  strip of table where the sun's contact shadows fall, so any more of it erases
  them.
- **Tabletop roughness is boosted** (`roughness_boost` in
  `create_wood_material`). Under a directional key the Poly Haven wood scans go
  glossy enough to throw a broad specular sheen across the tabletop that washes
  the grain out to near-white.
- **Room orientation.** The GLB is yawed `90°` (`ROOM_YAW_DEG`) so the empty
  half of the room hosts the stunt and the dining set + kitchen wall become the
  backdrop behind it.
- **Camera** is a low three-quarter view a little above the launch height,
  yawed off perpendicular so the book stack and the depth of the gap read (a
  dead-side view flattens both). It is framed to span the room floor -- where a
  fall-short car ends up -- up to comfortably above the launch, with a shallow
  `f/2.8` depth of field focused on the action.
- **Car orientation.** The raw GLB's length runs along its local Y with the
  nose at `-Y`, so the import yaws it `+90°` about Z to put the nose along
  world `+X`; a `-90°` yaw drives the car tail-first. (`bake_and_join` leaves
  imported objects in quaternion rotation mode, where assigning
  `rotation_euler` is silently ignored -- set `rotation_mode` first.)
- The car's origin is its geometric center so the per-frame physics quaternion
  spins it about its center (a bottom-pivot origin would make the airborne car
  visibly wobble).
- The vase of flowers (`assets/models/flowers_in_vase.glb`) and the toy ball on
  the floor are decor only (no collision proxy). The vase's position on the
  landing table is pinned between two limits -- further along the table and it
  runs out of frame at the right, nearer and it crowds where the car comes to
  rest -- so it is placed at a fixed sim `x` rather than measured from the
  table's near edge, and set back in Y clear of the car's centre line. The
  batch's top speed is capped to keep the car short of it.

## Key parameters

- `launch_speed`: initial push speed in m/s given to the car on the book stack
  (default `2.0`). The scene's main knob -- faster clears the gap, slower falls
  short. The clear/fall-short threshold for the default `0.28 m` gap is about
  `1.55 m/s`.
- `gap_width`: width of the gap between the two tables in metres (default
  `0.28`). Wider gaps require more speed; past what the launch can reach, the
  car drops to the floor.
- `seed`: seeds the render-side RNG for reproducibility.

## Outputs

- `car_gap_jump.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `car_gap_jump.blend` – saved Blender scene.
- `ground_truth_transforms.json` – per-frame world matrix and velocities for the
  car, plus the camera matrix and the sim-frame -> world `world_offset`.
- `scenario_metadata.json` – the render/physics parameters used.
