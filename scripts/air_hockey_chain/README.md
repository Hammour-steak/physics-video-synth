# Air-hockey mallet relay

Three identical air-hockey mallets sit in a line down the centre of an arcade
air-hockey table. The one at the far end is given a single push; it strikes the
second, the second strikes the third, and the third slides away to the near
rail. Like `dining_chain` and `domino_chain` nothing is scripted per object --
the whole relay is emergent from one initial velocity plus contact physics.

Where `dining_chain` is a *mostly inelastic* chain (the striker keeps half its
speed and the two slide on together), this scene is the opposite limit: the
**equal-mass, near-elastic head-on collision**. For two bodies of equal mass in
one dimension, `v1' = v1(1-e)/2` and `v2' = v1(1+e)/2`, so at `e = 1` the
striker stops dead and hands its entire speed to the target. Air hockey is
about as close to that ideal as everyday objects get: the mallets are identical
by design, the air cushion makes the table nearly frictionless so almost
nothing is lost between collisions, and the moulded plastic faces bounce hard.
All three properties are load-bearing -- break any one and the chain stops
looking like a relay (see the PCVE suite, which breaks them one at a time).

The environment is `assets/models/vintage_modern_living_room_with_arcades.glb`
(a vintage-modern living room with arcade cabinets, used as static background
at its native metre scale). The table is
`assets/models/air_hockey_arcade.glb`, scaled to a standard 0.78 m playing
height and stood in the open middle of the room. The three mallets are not a
separate asset: they are **lifted out of the table's own mesh** -- the model has
two mallets moulded onto its playing surface, one of which is cut out by vertex
position, cloned three times and recoloured, while the other is deleted. Cloning
the table's own mallet is what keeps the three identical to each other (the
condition the velocity swap rests on) and identical in style to the table they
slide on, which a procedural cylinder would not be.

The three are painted **blue, red and white** in the order the relay runs, so a
viewer can follow which disc is carrying the momentum. Three identical white
discs make the handoff unreadable.

## Files

- `simulate_air_hockey_chain.py` – PyBullet physics simulation (three cylinder
  proxies on a plane ringed by rails; the far mallet gets the only initial
  velocity).
- `render_air_hockey_chain.py` – Blender rendering script: imports the room and
  the table, places the table, extracts and recolours the mallets, runs the
  simulation, applies the trajectory as keyframes, lights the room and renders.
- `batch_render_air_hockey_chain.py` – orchestrates multiple randomized renders.
- `edit_vocab.py` – this scene's PCVE edit vocabulary: which objects and
  properties may be edited, how each maps to a simulator parameter, and the
  baseline physics values. Must stay in sync with
  `render_air_hockey_chain.create_scenario()`.
- `build_pcve_air_hockey_chain.py` – builds the PCVE suite: one source video
  plus one variant per edit-DSL string (does the relay behave like a velocity
  swap, or not?).

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
CUDA_VISIBLE_DEVICES=0 ./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/air_hockey_chain/render_air_hockey_chain.py -- \
    --mode preview \
    --out-dir renders/air_hockey_chain_preview \
    --resolution 960 540 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 96 \
    --device auto \
    --preview-frame 31

# Render the full animation (72 frames at 24 fps = 3 s)
CUDA_VISIBLE_DEVICES=0 ./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/air_hockey_chain/render_air_hockey_chain.py -- \
    --mode animation \
    --out-dir renders/air_hockey_chain \
    --resolution 1280 720 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 128 \
    --device auto
```

Useful preview frames: **1** the starting layout, **15** blue strikes red,
**31** red strikes white, **48** white mid-slide, **66** white against the near
rail.

`CUDA_VISIBLE_DEVICES` is worth setting on a shared box. `enable_gpu()` in the
render script turns on every non-CPU device Cycles reports, so without it a
render will happily land on a card someone else is training on.

## Batch render

```bash
python scripts/air_hockey_chain/batch_render_air_hockey_chain.py \
  --mode animation \
  --count 4 \
  --seed-base 41000 \
  --resolution 960 540 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 96 \
  --device auto \
  --out-root renders/batch_air_hockey_chain
```

Each sample jitters `push_speed` (0.72-0.90 m/s) and `mallet_restitution`
(0.92-0.97). Both ranges were checked at their four corners against the
simulator: the relay completes in all of them, every striker is left under an
eighth of its speed, and none rebounds. What varies is the timing of the two
handoffs and how far up the near rail the last mallet finishes.

## Simulate only

```bash
python scripts/air_hockey_chain/simulate_air_hockey_chain.py \
  --out renders/test_air_hockey_physics.json \
  --fps 24 \
  --duration-sec 3.0
```

The `quality` block in the output JSON is what to read rather than the final
frame -- the table is nearly frictionless, so by the last frame the third mallet
has reached the rail and the positions say nothing about the original chain.
`retained_fraction` is each striker's speed at the moment the next one takes
off, divided by its own peak: that is what "stopped dead and handed everything
over" actually means, and it should sit near `(1-e)/2`.

```bash
# equal masses -> striker stops dead
python scripts/air_hockey_chain/simulate_air_hockey_chain.py --out renders/t_equal.json
# heavy middle mallet -> striker rebounds backwards
python scripts/air_hockey_chain/simulate_air_hockey_chain.py \
  --out renders/t_heavy.json --middle-mass-scale 4.0
```

## Build PCVE suite

```bash
python scripts/air_hockey_chain/build_pcve_air_hockey_chain.py \
  --out-root renders/pcve_air_hockey_chain_suite \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 64 \
  --device auto
```

The suite is one source video plus N edited variants. Every edit is declared as
a **single edit-DSL string** in `EDIT_CASES`, and the natural-language prompts
(vague + quantitative, zh + en), the scenario overrides handed to the renderer,
and the manifest's `physics_diff` are all derived from that one string by
`scripts/pcve_edit_dsl.py` against this scene's `edit_vocab.py`. Nothing about
the edit is written twice, so a prompt cannot drift from the physics it
describes.

An edit may only **delete one object** or **set one scalar property** on one
object. The layout, the camera, the seed and the lighting are identical across
every case, so the only thing that varies in the frame is the motion.

Editable objects are `blue_mallet` / `red_mallet` / `white_mallet` (the relay
order) only. The table itself is deliberately **not** editable: an edit names
a moving object, not the surface it slides on. Since Bullet multiplies the
pair, raising one mallet's friction to 1.5 gives exactly the effective 0.09
that raising the table's would (verified identical to the millimetre).
Editable properties are `mass`, `restitution`, `friction`, and
`initial_velocity` on the blue mallet -- the only disc with a non-zero
baseline velocity, so the only one where a scalar speed has a direction to
scale.

Every value below was chosen by sweeping `simulate_air_hockey_chain.py`
directly, not guessed, and the outcomes match the closed-form 1-D prediction
`v1' = v1(m1 - e·m2)/(m1 + m2)`, `v2' = v1·m1(1 + e)/(m1 + m2)`.

| case_id | edit_dsl | outcome |
|---|---|---|
| `air_hockey_chain_baseline` | – (source) | Each striker stops dead (keeps 3%/4%); white carries 88% of the push into the near rail. |
| `edit_heavy_middle_mallet` | `SET red_mallet.mass FROM 0.12 TO 0.48` | Blue **rebounds backwards** up the table instead of stopping; red leaves with 0.30 m/s, so only 31% gets through. |
| `edit_heavy_last_mallet` | `SET white_mallet.mass FROM 0.12 TO 1.2` | First handoff untouched, second one fails: red bounces back off white, which only creeps forward at 0.13 m/s. |
| `edit_heavy_striker` | `SET blue_mallet.mass FROM 0.12 TO 1.2` | Blue keeps 80% and drives on behind red; red leaves at 1.37 m/s, faster than the 0.8 m/s push. All three finish bunched at the near rail. |
| `edit_dead_striker_face` | `SET blue_mallet.restitution FROM 0.95 TO 0.35` | The **first** impact becomes a shove (pair restitution 0.33): blue keeps 35% and slides on behind red. The second handoff is unchanged. |
| `edit_grippy_striker` | `SET blue_mallet.friction FROM 0.06 TO 1.5` | Effective coefficient 0.0036 -> 0.09. Blue stops after 36 cm. **No collision happens at all** -- the disc riding on almost nothing is what makes the relay possible. Red and white still glide. |
| `edit_soft_push` | `SET blue_mallet.initial_velocity FROM 0.8 TO 0.3` | Collisions still clean, but red stops 24 cm short of white. The chain dies one link short. |
| `edit_remove_middle_mallet` | `DELETE red_mallet` | Blue coasts the empty middle and strikes white directly. One handoff instead of two, and a cleaner one: white gets 90%. |
| `edit_remove_last_mallet` | `DELETE white_mallet` | First handoff identical to the baseline, then red runs the rest of the table into the near rail unobstructed. |

Because PyBullet **multiplies** the two bodies' restitutions (and their
frictions), setting one mallet's value changes only the impacts that mallet is
part of: blue's governs the first handoff, white's the second, red's both. That
is what makes `edit_dead_striker_face` a one-object edit rather than a global
one, and it is why the prompt's number (0.35) is not the number that governs
the collision (0.33).

Adding or changing an edit:

```bash
# 1. sweep the sim directly first -- no Blender, no rendering
python scripts/air_hockey_chain/simulate_air_hockey_chain.py \
  --out /tmp/s.json --mallet-masses 0.12 0.48 0.12
# 2. add an EditCase to EDIT_CASES, then check what it generates
python scripts/air_hockey_chain/build_pcve_air_hockey_chain.py \
  --dry-run --out-root /tmp/dryrun
# 3. render only what is new; --clean-stale-cases rm -rf's dropped cases
python scripts/air_hockey_chain/build_pcve_air_hockey_chain.py --skip-existing
```

`--skip-existing` still rewrites `prompts.json` and the manifest, so edits to
prompt wording or an `edit_summary` do not cost a re-render.

## Scene layout

- The room is authored in metres with its floor at `z = 0.1489`. The table
  stands in the open middle of the room rather than against a wall: on the
  clear floor between the TV on the `+Y` wall (its console reaching back to
  `y = 2.65`) and the coffee table in front of the sofa (`y -1.71..-0.70`). That
  gap is 3.35 m deep, and between the arcade cabinets flanking it (the left pair
  reaches `x = -1.74`, the right one starts at `x = 1.95`) it is 3.7 m wide.
  Centring the 1.43 × 2.55 m cabinet at `(0.30, 0.98)` leaves ~0.4 m of floor
  off each end.
- Table constants were measured off the mesh, not guessed: a raycast height map
  found the playing surface as one flat plane at model `z = 30.50` and the only
  two things standing on it. Scaled to a 0.78 m playing height the field is
  1.164 × 2.391 m, with the surface at world `z = 0.9289`.
- The mallets sit on the **quarter, half and three-quarter points** of the
  table's length (sim `x` 1.793 / 1.196 / 0.598) -- as far apart as a
  three-mallet relay can be spread while leaving the last one room to run out.
  The wide gaps are also what make each impact a separate two-body event: a
  sequential-impulse solver resolves a row of *resting* contacts unreliably, so
  the mallets deliberately never touch until they collide.
- The chain steps down `-x`, from the end away from the camera back towards it,
  so the struck disc always comes at the viewer and each handoff lands nearer
  the lens and larger in frame than the one before.
- Collision proxies are plain cylinders (r 0.062 m, h 0.066 m, 120 g) matching
  the mallets measured off the table model. The rails are a backstop only -- the
  chain runs straight down the centre line and never reaches the sides.

## Rendering notes

- **The table GLB is a zero-thickness double-sided export.** Essentially every
  surface is two coincident faces pointing opposite ways, the outward one
  carrying the artwork and the inward one a plain colour: 9049 of the 9051
  coincident face pairs in the file are back-to-back. Every material sets
  `use_backface_culling`, so the tool it was authored in only ever drew the
  outward face -- but that flag is EEVEE-only. Cycles hits both, they are
  exactly coplanar, and which one wins is settled per BVH region, which painted
  the playing surface in stripes. `split_double_sided_sheets()` nudges every
  vertex 0.02 model units (~0.5 mm) along its own normal so each sheet moves the
  way it faces and the artwork ends up outermost. Culling backfaces in the
  shader instead does *not* work: on the playfield it is the artwork sheet
  Blender counts as the back one, so that erases the surface.
- **The playfield is polished** to roughness 0.22 (`polish_playfield()`). The
  glTF import lands every material on 0.6, which is chalk. This matters more
  than it sounds: the discs are only 7 cm tall, so a lamp at ceiling height
  throws a shadow a couple of centimetres long that hides underneath the disc.
  What ties a disc to the surface at this camera angle is the surface
  reflecting it back.
- **No HDRI.** Both GLBs carry emission as a proper mask, so the arcade
  cabinets, the lava lamp and the ceiling fixture glow on their own; that glow
  is the room's character and is left alone. The lamps added on top are placed
  on the room's *own* light sources so the shading stays consistent with what is
  in shot: a warm key at the red pendant where it actually hangs (`0, 0, 3.02`),
  a cool daylight area at the window pane on the `-X` wall, and a wide weak
  bounce for what the tiled floor throws back. The balance is deliberately
  lamp-heavy -- roughly doubling the fill flattens the tiled floor out and the
  cabinet stops sitting in the room.
- **Camera** is a three-quarter view from the room's `-Y`/`+X` corner at
  `(2.44, -2.41, 2.00)` on a 42 mm lens, ~3.9 m back. That corner is the one to
  shoot from because the room model only carries walls on `-X` and `+Y`: looking
  that way stands the Metal Slug cabinet and the TV behind the table and keeps
  the missing `+X` wall, which renders as a grey void, behind the lens. It aims
  at 0.40 of the table's length rather than the middle -- the near end is a
  metre closer to the lens so it subtends more, and aiming at the geometric
  centre runs it off the bottom of frame.

## Key parameters

- `push_speed`: the far mallet's initial push in m/s (default `0.8`) -- the
  scene's main knob, setting how fast the relay runs and how far the last mallet
  gets. This value lands both handoffs on theory (3-4% retained against the 4.9%
  predicted) while still walking the last mallet into the near rail at 1.96 s.
  Slower is not better: at 0.75 and 0.70 the second handoff degrades to 8% and
  10%.
- `mallet_restitution`: default `0.95`. PyBullet multiplies the two bodies'
  values, so mallet-on-mallet is this squared (~0.90), leaving the striker
  `(1-e)/2` ≈ 5% of its speed. Drop it and the pair starts moving off together.
- `surface_friction`: default `0.06`, likewise multiplied by the mallet's own
  friction, so the effective coefficient is a few thousandths -- this is the air
  cushion, and it is why a mallet coasts the length of the table with almost no
  loss.
- `middle_mass_scale`: multiplies the middle mallet's mass (default `1.0`).
  Heavier and the striker rebounds backwards instead of stopping.
- `substeps` (sim only, default `160`): not a quality dial to be turned down.
  Below about 120 the impacts land several substeps deep, Bullet applies
  restitution against an already partly corrected velocity, and the collision
  quietly turns inelastic at some push speeds and not others.
- `settle_sec` (sim only, default `0.25`): the mallets are left to settle and
  then frozen flat before the push. Without it they are still micro-bouncing off
  their initial drop when the first impact lands, so the contact normal is
  tilted and the collision is oblique.

## Outputs

- `air_hockey_chain.mp4` – rendered video (animation mode).
- `preview.png` – preview still (preview mode).
- `ground_truth_transforms.json` – per-frame world matrices for the three
  mallets and the camera, plus velocities and the sim-to-world mapping.
- `scenario_metadata.json` – the seed, render settings and physics parameters.
- `air_hockey_chain.blend` – saved Blender scene.
