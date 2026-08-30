# A big glass marble knocks a small one up a kitchen bar table

A 100 mm glass marble is rolled 0.62 m along the bar table in a modern flat and
strikes a 50 mm one sitting at rest. The small marble leaves at 1.29 m/s -- half
again as fast as anything in the shot was moving -- and runs 0.68 m up the table;
the big one keeps 80 per cent of its speed and follows it, stopping 0.46 m
behind. One push, one contact, nothing after the launch is scripted.

The scene is built around one fact that is easy to state and easy to get wrong:
**the mass ratio of two balls of the same material is the cube of their size
ratio, and it is the mass ratio that decides what a collision does.** The two
marbles are the same glass. One is twice the diameter, so it is eight times the
mass, and eight to one is the whole reason the struck marble comes away faster
than the ball that hit it while that ball barely slows down. Nothing about that
is visible in the two objects other than how big they are.

For an impact between masses `m_A` and `m_B` at coefficient of restitution `e`,
the struck ball leaves along the line joining the two centres at

```
v_B = (1 + e) * m_A / (m_A + m_B) * v_A * cos(obliquity)
```

which at 8:1, `e = 0.76` and 9.2 deg off centre predicts 1.542 times the
approach speed. The solver returns 1.553. The big marble's outgoing speed is
predicted at 0.810 of its approach and measured at 0.797. Both comparisons are
recomputed and checked on every render rather than once during tuning, because a
solver that quietly disagrees with the closed form is the most likely way for
this scene to be wrong -- see `predict_impact()` and the `[WARN]` it raises.

The impact is deliberately not *quite* head-on. Dead centre, the two balls come
away on the same line and the small one just runs away in front of the big one:
a tidy demonstration and a poor picture, because nothing in the frame separates.
With the small marble set 12 mm off the big one's line -- 9.2 deg on a 75 mm
centre separation -- the two leave on headings 11.8 deg apart, the small one
bending toward the front edge and the big one drifting back, and the split is
legible for the rest of the shot.

## The room had to be measured before it could be used

`assets/models/modern_living_room.glb` is a SketchUp-style export. It is not in
metres, and all 124 of its meshes are named `Material2.0NN`, so nothing in it can
be found by name -- the name of the object holding the table top is
`Material3.012`. Everything this scene knows about the flat was measured by
raycasting the imported model.

The unit is fixed by the one dimension in a kitchen that is standardised: the
worktop. It sits at model z = 37.219, and the bar table's top is the *same
surface* -- the two are flush, and a raycast walks from one onto the other with
no step at all -- so taking the worktop as 0.90 m fixes the scale at
`0.90 / 37.219`. Four independent checks agree, which is why this is the number
and not a guess:

- the interior ceiling lands at 2.571 m, and the wall cabinets run exactly up to
  it, as fitted cabinets do;
- the window's sill lands at 1.058 m and its head at 2.010 m;
- the coffee table's top lands at 0.404 m and the sofa's seat at 0.421 m;
- the bar stools' seats land at 0.718 m, which is a bar stool for a 0.90 m
  counter.

Because the names are meaningless, `verify_room()` checks the nine objects this
scene reaches for against the world-metre bounding box each one was measured in,
and warns if any has moved more than 10 mm. A name is no evidence that the right
mesh has been found; the box is.

Three things about the model then mattered:

- **It ships no lights at all.** Not a fake light card -- no light objects of any
  kind. What it does have is a large recessed luminous ceiling panel, 5.92 x
  2.82 m directly above the table, modelled as a housing, a warm yellow face and
  a grey translucent diffuser, all carrying ordinary diffuse materials. Cycles
  renders that as what it geometrically is: a dull mustard slab across the
  ceiling, with the room pitch black underneath. The face is given a real
  emission shader on its own copy of the material -- `material_46` is shared, and
  rewriting the shared datablock would set fire to everything else using it --
  and the diffuser is hidden, because at alpha 0.5 it halves the panel's output
  and greys everything under it, the table top included, for nothing the camera
  can see.
- **Every material is flattened to metallic 0, roughness 0.6.** Harmless on the
  walls; on the one surface the camera is looking straight down at it reads as
  unfinished chipboard. `polish_table_top()` takes the top to roughness 0.28 and
  specular 0.55 and leaves the model's own wood texture alone.
- **The table's dressing is kept, and the action moved instead.** A shallow
  wooden tray with two brass candlesticks stands a third of the way along the
  table, hugging the wall side: 0.27 m of the length and 0.10 m of the depth,
  with the candles 0.256 m tall. Hiding it would have given the whole table to
  aim across, and it was tempting. But it is the only thing in the shot that
  gives the table's surface a scale and a foreground, and moving the collision
  into the table's east half costs nothing -- there is still 1.0 m of clear run
  east of the candlesticks, which is more than the shot needs. The rolling
  marble passes 0.117 m clear of the tray, and touching it is a reported failure
  rather than something that can happen unnoticed.

## The table is a peninsula, and that decides almost everything

The bar table is 1.830 x 0.579 m with its top at 0.900 m, and it is not a table
you can walk around:

- its **south edge is against a wall** for its whole length, so nothing can fall
  off there and no camera can stand there;
- its **east end is 67 mm from an open wardrobe**, which is a slot a 50 mm marble
  fits down and nothing worth photographing;
- its **west end is flush with the kitchen worktop**, so there is no edge there
  at all -- the surface simply carries on for another 3.1 m;
- its **north side is open**, with three bar stools tucked under it. Their backs
  stop 0.106 m *below* the table top, which is why they never cross the line the
  marbles roll along and read as a foreground frieze instead.

So the camera is on the north side or nowhere, and the only edge a ball can
plausibly leave by is the east end. `off_table()` reports which edge was crossed
rather than just that one was, because "rolled onto the worktop" and "fell 0.90 m
onto the parquet" are not the same event and the west one is not a failure.

## Why the table's rolling resistance is what it is

`table_rolling_friction` is 0.0200, and it is worth being straight about what
that is: on this table it works out at 0.82 m/s^2 of deceleration for the big
marble, which is high for glass on wood. It is set there because **1.83 m is not
much table.** A hard ball rolled hard enough for the collision to read at 24 fps
will not stop inside that distance at a textbook coefficient -- it runs off the
end, and the only end it can run off leads down a 67 mm slot beside a wardrobe.
The choice was between a surface that takes energy back faster than oiled wood
really does and a shot whose subject leaves the frame, and this scene takes the
first. It is the same knob every other scene here calls its clock; the
difference is that this one is leaning on it.

What it buys is a shot that is entirely contained: the big marble arrives at
0.830 m/s, both balls come to rest on the table, the struck one stopping 0.19 m
short of the far end, and the last frame still has both of them in it with their
separation legible.

Two consequences are recorded rather than hidden. Bullet's rolling friction acts
as a contact-offset *length* rather than a dimensionless coefficient, so the
deceleration goes as 1/radius and the small marble slows faster than the big one
-- which is exactly why it can leave at 1.5 times the speed and still not reach
the end. And the ratio is not the clean 2:1 that implies: measured over its whole
run the small marble decelerates at about 1.23 m/s^2 rather than 1.66, because it
is struck at rest with no spin and skids for the first few centimetres while
friction spins it up. The big marble's own figure is 0.83 m/s^2, measured the
same either side of the impact.

## Files

- `simulate_table_marble_collision.py` -- PyBullet physics, plus `predict_impact()`,
  the closed form the solver is checked against.
- `render_table_marble_collision.py` -- Blender rendering: imports and calibrates
  the flat, lights the ceiling panel, polishes the table top, sizes the two
  marbles from what the simulation reported, applies the trajectory as keyframes.
  Also owns the collider export.
- `cam_survey.py` -- renders the camera survey (12 framings x 4 key frames).
- `batch_render_table_marble_collision.py` -- orchestrates multiple renders.
- `build_pcve_table_marble_collision.py` -- builds the five-case PCVE suite.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame (the contact)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/table_marble_collision/render_table_marble_collision.py -- \
    --mode preview \
    --out-dir renders/table_marble_collision_preview \
    --resolution 960 540 \
    --samples 64 \
    --device auto

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/table_marble_collision/render_table_marble_collision.py -- \
    --mode animation \
    --out-dir renders/table_marble_collision \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 128 \
    --device auto
```

## Simulate only

The dressing's collision mesh has to exist first. The render script writes it
automatically when it is missing, or on demand:

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/table_marble_collision/render_table_marble_collision.py -- \
    --mode colliders --out-dir /tmp/colliders
```

Then the simulation runs on its own and prints a one-line summary, so aims,
sizes and surfaces can be compared without rendering anything:

```bash
python scripts/table_marble_collision/simulate_table_marble_collision.py \
  --out /tmp/tmc.json \
  --props-collider assets/collision/modern_living_room_table_dressing.obj \
  --duration-sec 2.4
```

```
[SIM] hit=True f=15 run=0.619 | a_in=0.830 a_out=0.662 b_out=1.290 |
      ratio b=1.553/1.542 a=0.797/0.810 | obliq=9.2 sep=11.8/11.4 |
      travel a=0.889 b=0.675 | A_end=(0.268, 0.029) B_end=(0.726, -0.102)
      settle=(34, 40) spin_left=0.00/0.00 dressing=None off_table=None
```

Each ratio is printed as `measured/closed-form`. `--disable-ball-b` leaves the
struck marble out, which is what the run-up was calibrated with: the speed at the
impact point has to be known before there is any point predicting what the impact
does.

## Build PCVE suite

```bash
python scripts/table_marble_collision/build_pcve_table_marble_collision.py \
  --out-root renders/pcve_table_marble_collision_suite \
  --resolution 1280 720 --fps 24 --samples 96 --device auto
```

Four cases hold everything fixed -- same big marble, same 1.31 m/s push from the
same place, same aim, so every one of them arrives at 0.83-0.88 m/s -- and change
only how big the struck marble is. Doubling its diameter multiplies its mass by
eight, and the outcome runs through the whole range a two-body impact has to
offer. The fifth restores the hero's pair and softens the push instead.

| case_id | struck marble | mass ratio | struck ball leaves at | big marble | outcome |
|---|---|---|---|---|---|
| `table_marble_throws_it_clear` | 50 mm | 8.00 : 1 | 1.55x (1.54) | keeps 0.80 | **Hero.** Struck marble runs 0.68 m, big one 0.27 m; they rest 0.46 m apart. |
| `table_marble_middling_ratio` | 70 mm | 2.92 : 1 | 1.29x (1.30) | keeps 0.55 | Visibly checked by the impact rather than sailing through it. 0.48 m / 0.15 m. |
| `table_marble_matched_pair` | 100 mm | 1.00 : 1 | 0.87x (0.87) | keeps 0.16 | Big one stops where it stands; the two leave **45 deg** apart. |
| `table_marble_heavier_target` | 140 mm | 1 : 2.74 | 0.49x (0.47) | **reverses** at 0.33 | Sent back 0.075 m the way it came; the pair leaves 164 deg apart. |
| `table_marble_soft_push` | 50 mm | 8.00 : 1 | 1.56x (1.54) | keeps 0.80 | **Distractor.** Same ratios, 1.08 m/s push. Struck marble runs 0.12 m. |

A model that reads "bigger ball, therefore harder to move, therefore everything
happens a bit less" gets the *direction* of the big marble wrong in the last two
cases and the speed *ordering* wrong in the first.

`soft_push` is why the suite is worth rendering. Its struck marble travels 0.12 m,
near enough the same as `heavier_target`'s 0.09 m, and the resting positions
barely separate them -- but the cause is the opposite, and what tells them apart
is the big marble: in `heavier_target` it retreats 0.075 m from the contact point,
here it carries on 0.054 m forward. The
manifest records each case's *outcome*, read back out of the rendered ground
truth, not just the radius it was asked for.

## Scene layout

Coordinates are metres in the physics frame, which is also the world frame:
z = 0 is the table top, the origin is the centre of that surface, +x runs east
along the length and +y north, out of the wall and toward the stools.

- **The big marble launches at (-0.620, 0.020) at 1.31 m/s along +x** and covers
  0.619 m of clear top before the contact, arriving at 0.830 m/s on frame 15 of
  58. Rolling east means it appears to come out of the kitchen and the collision
  lands in the table's empty east half; rolling the other way would have put both
  resting balls on the worktop join, which is not an edge and not interesting.
  The run-up used to be 0.79 m from x = -0.780; it was shortened when the framing
  came in closer, because at 1.31 m/s over 0.62 m the marble arrives within 2 per
  cent of the same speed and every downstream number moves by under 2 cm.
- **The struck marble waits at (0.060, 0.008)**, 12 mm off the roller's line.
  That offset is the only thing setting how far apart the two leave, and it is
  small on purpose: the shot is about the speed transfer, and a glancing blow
  reduces the transfer by `cos(obliquity)` however heavy the balls are.
- **Both run along y ~ 0**, which keeps the roller 0.117 m clear of the tray and
  0.24 m from the table's open edge. The candlesticks stay in shot, in the
  foreground of the wall side, doing the job of saying how big a 100 mm marble is.
- **They come to rest at (0.268, 0.029) and (0.726, -0.102)**, 0.46 m apart along
  the table and 0.13 m across it, with the struck marble 0.19 m short of the far
  end. Both settle on camera, on frames 34 and 40 of 58.
- **Camera**: a low three-quarter from the north-east, (0.58, 1.40, 0.24), 35 mm
  -- 1.14 m off the floor, 0.24 m above the table. Chosen from a survey of twelve
  angles plus a four-step pull-in ladder (`cam_survey.py`, kept in
  `renders/tmc_camera_survey/`).

  The **angle** was picked for one property the prettier candidates do not have:
  the whole shot stays in frame from the launch to both marbles at rest. The view
  down the table's length from the east end had the best depth of the twelve, with
  the top receding toward the kitchen and a daylit window behind it, but the
  struck marble rolls *toward* that camera and leaves frame before it stops, which
  empties the last 15 frames of half their content. For a reconstruction clip the
  two balls' relative position is the content and it has to be legible in the last
  frame as much as the first.

  The **distance** was set by walking the camera in along that sight line. At
  35 per cent closer the marbles read about twice the size and the frame stops
  being mostly bare wall -- but the launch clipped the right edge, and that was
  fixed by shortening the run-up rather than by backing the camera off. Moving the
  subject is nearly always the cheaper fix: the camera position is load-bearing
  and the run-up was not. Half the standoff was rendered too and is the
  best-looking of the lot, with the impact filling the frame and the top's
  reflection coming alive, but no amount of shortening puts the launch back into
  it, and a collision scene that does not show the ball being sent has given away
  its premise.

  The pull-in also holds the camera **height** near 0.24 m instead of scaling it
  down the sight line. Scaling would keep the grazing angle identical, which
  sounds right and is not: at 0.14 m the camera is a ball and a half off the
  table and the two marbles start overlapping each other instead of reading as
  two objects.

## Physics notes

- **Rolling and spinning friction go on the table, not on the marbles.** Bullet
  does not combine either as a plain product: it uses
  `rf_a * lateral_b + rf_b * lateral_a`. Setting a ball to `rf = 1` -- the obvious
  way to let the surface's value pass through -- therefore multiplies in the
  *table's lateral* friction instead and turns the roll into a skid. With the
  balls at 0, the effective value is exactly the table's own.
- **The table needs spinning friction, and it is not a refinement.** An oblique
  impact puts real spin about the *vertical* axis into both marbles -- surface
  friction at the contact acts tangentially, and that is what tangential friction
  does -- and nothing else in Bullet opposes it: rolling friction resists rolling,
  lateral friction resists sliding. At 0 a marble comes to a dead stop and goes on
  turning where it stands indefinitely, which is plainly visible in the render.
  At 0.006 the spin bleeds away in about a third of a second and nothing else
  moves, because spinning friction acts only about the contact normal.
  `residual_spin` and `spinning_on_the_spot` are carried as a standing check, and
  "settled" means stopped moving *and* stopped turning.
- **The rolling marble is launched already rolling** (`omega = (z_hat x v) / r`).
  From rest it spends its first tenth of a second skidding while friction spins it
  up, which reads as a shove rather than a roll.
- **Bullet multiplies the two restitutions.** The impact sees
  `ball_a_restitution * ball_b_restitution`; 0.87 each for glass gives an
  effective 0.76, and that is the `e` in the closed form. It is written into the
  ground truth as `effective_restitution` rather than left to be inferred.
- **The table top goes in as a box, not as the model's triangles**, and here that
  is the honest choice rather than the cheap one. The top *is* a flat rectangle:
  1701 probe points over the clear area came back at the same z to four decimals,
  and its extent was found by walking rays in 0.2-unit steps until the surface
  dropped away. A trimesh would add the metal frame underneath, which no ball on
  the table can reach. `collisionMargin` is 0.0005 on every static body, because
  Bullet's default 0.04 m would inflate a box whose edges are the thing a ball
  falls off by most of the small marble's diameter.
- **The masses are derived, not configured.** `ball_a_mass` and `ball_b_mass`
  default to a solid glass sphere at 2500 kg/m^3 of whatever radius was asked
  for -- 1.309 kg and 0.164 kg for the hero. They are exposed only so that a case
  can deliberately break the link between the size you can see and the mass that
  does the work, which no video would reveal on its own.
- **The rendered marbles are sized from the simulation's output**, not from the
  scenario's own numbers. Carrying a diameter in two places is how a suite case
  ends up simulating a 70 mm ball and rendering a 50 mm one.
- **The contact is sampled at substep resolution**, not at frame boundaries. At
  24 fps and 60 substeps a single frame is long enough for the table to take a
  measurable slice out of the struck marble before anything is recorded.
- **Motion blur is on at a half-frame shutter.** The struck marble leaves at
  1.29 m/s, which is 54 mm per frame against its own 50 mm diameter; without blur
  the impact reads as a ball teleporting.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json`, or
directly in `build_pcve_table_marble_collision.py` for suite cases.

- `ball_b_radius`: the parameter this scene is built around (default `0.025`
  against the big marble's `0.050`). Because the mass follows from it at fixed
  density, it *is* the mass ratio, and it moves the outcome from "struck ball
  leaves at 1.5x" through "roller stops dead" to "roller reverses".
- `launch_speed`: the push, in m/s (default `1.31`). Changes when everything
  happens and how far each ball runs, but **not** the ratios -- which is exactly
  what makes `soft_push` a useful distractor. About 5 per cent above the default
  the struck marble reaches the far end of the table and falls off it, so it is
  tied to `launch_x`: shortening the run-up without softening the push sends it
  over the edge.
- `ball_b_y` vs `launch_y`: the impact parameter (default 12 mm, i.e. 9.2 deg).
  Sets how far apart the two balls leave; increasing it reduces the speed
  transfer by `cos(obliquity)`.
- `table_rolling_friction`: the scene's clock (default `0.0200`). See above for
  why it is high. Low enough that both marbles still move visibly, high enough
  that neither runs off the end.
- `table_spinning_friction`: what stops a marble turning on the spot after it has
  stopped moving (default `0.006`). Affects nothing else; setting it to 0
  reproduces the bug described above.
- `ball_a_restitution` / `ball_b_restitution`: default `0.87` each, effective
  `0.76`.
- `gravity_z`: default `-9.8`.

Lighting is in the scenario too, under `lighting`, because the flat's lighting
had to be built from nothing and the balance between the ceiling panel and the
window was found by rendering it rather than by reasoning about it:
`ceiling_panel_strength` (default `9.0`) is the one that matters.

## Outputs

- `table_marble_collision.mp4` -- rendered video (animation mode).
- `preview.png` -- preview still (preview mode).
- `table_marble_collision.blend` -- saved Blender scene.
- `ground_truth_transforms.json` -- per-frame transforms, velocities, speeds,
  `spin_z` and an `on_table` flag for both marbles, plus a `phase` flag on the
  roller marking `approach` / `after_impact` and a `moving` flag on the struck
  one. The `collision` block carries the size and mass ratios, the effective
  restitution, the impact parameter and obliquity, and the closed-form prediction
  beside the measured value for both balls' outgoing speed and for the angle they
  separate by. The `quality` block carries the contact frame, point and normal,
  the speeds either side of it, the headings and turns, both travel distances and
  resting positions, both settle frames, and three standing failure checks
  (`spinning_on_the_spot`, `hit_table_dressing`, `left_table`).
- `scenario_metadata.json` -- seed, camera, lighting, and physics parameters.
- `room` block in the ground truth -- the model name, `ROOM_SCALE`, and the model
  units the world frame was built from, so the calibration is recoverable from an
  output rather than only from this file.
