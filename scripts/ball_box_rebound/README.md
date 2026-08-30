# Ball bounces off a toy chest and knocks a second ball away

A toy ball is rolled across a playroom floor into the side of a wooden toy
chest, rebounds off it, and on the way back knocks a second, stationary ball
clear. Three events, one push: after the launch nothing is scripted. The ball
gets a single initial velocity and a matching roll, and everything after that is
the chest's restitution and one ball-ball impulse.

The scene is built around **restitution changing the direction of a bounce, not
just its speed.** A bounce scales the component of the velocity normal to the
wall and leaves the tangential one alone, so a duller chest does not send the
ball back along the same line more slowly -- it sends it back on a *flatter*
line. In this layout that difference decides the outcome: at 0.80 the ball comes
off at 29 deg and crosses to the target, at 0.48 it comes off at 42 deg and goes
wide, at 0.22 it comes off at 61 deg and skims away along the front of the chest
without ever going near it. Every one of those cases hits the panel at the same
point, at the same speed, at the same 30 deg.

Both balls are the room's own toys. `Ball1` is the pink ball with a star printed
on it -- that star is why it is the one that rolls, because it is the only thing
in the shot that makes the ball's spin readable. `Ball2` is the little football
that the model leaves lying inside the chest; here it comes out onto the floor
as the target. Both are re-sized: as authored they are a 0.32 m and a 0.24 m
ball, which is too big for two of them plus a bounce to fit in the room, and at
0.16 m and 0.13 m they are still perfectly ordinary toys.

The environment is `assets/models/toy_box.glb`, used as static background at its
own materials. Four things about that model had to be corrected, and all four
mattered:

- **Its play mat is removed** (this was asked for, and it is also the right
  call). With the mat down the balls cross a surface boundary in the middle of
  the shot, and its printed roads and houses are busy enough at this scale to
  compete with the two things the shot is about. Without it the ground is one
  plane with one set of properties -- easier to read, and easier to simulate
  honestly.
- **Its own floorboards go with it, and are replaced.** They are a single quad
  covering 3.11 x 2.60 m -- the set is a corner, with a west wall, a south wall
  and two open sides -- and they are mapped into a shared texture atlas, so they
  can be neither tiled nor enlarged. The ball's run-up alone is 1.15 m and the
  camera has to stand behind it, which put every usable camera position past the
  edge of that quad with its far side in frame as a hard line with nothing under
  it. `build_hardwood_floor()` lays a 9 m plane with Poly Haven's worn-pine 4K
  PBR set instead, which is also a considerably better floor -- this shot is
  largely a picture of it.
- **Ground level is the mat's surface, not the boards.** The model has the
  chest, the toy basket and the teddy bear all standing on the mat, which is
  0.92 model units (13 mm) above its floorboards. Dropping the mat and keeping
  the boards where they were would leave the whole set hovering 13 mm in the
  air, which at a 0.36 m camera height is plainly visible as a gap and as
  daylight under the chest. The laid-in floor is put at the mat's height
  instead, so nothing the physics is calibrated against moves.
- **It ships two fake lighting cards**, a window-shaped quad and a god-ray wedge
  spanning the whole room, both carrying an ordinary diffuse material meant to
  be composited additively by a game engine. Cycles has no such blend mode and
  renders them as what they geometrically are: two large grey-yellow slabs
  hanging in the room, one of which passes straight through the play area. Both
  are hidden and replaced with real lights.

The model is not authored in metres either: its floor-to-wall-top span is
181.571 units, so the whole room is uniformly scaled by `2.60 / 181.571` and
shifted so that world z = 0 is the floor and world (0, 0) is on the chest's
front panel at the mid-point of its width. That calibration checks out
independently against the model's own window -- the sill lands at 1.02 m and the
head at 2.15 m, which is where a real window sits. Because the physics frame
*is* the world frame, the PyBullet output needs no transform at all before being
keyframed.

## The chest is a mesh, not a box

The one thing this scene cannot get wrong is where the chest's front panel is,
and it is not where a hand-fitted box would put it:

- **The chest is yawed 3.74 deg** relative to the room's axes, so its face is
  the line `y = 0.0653 x`, not `y = 0`. The rebound is not a mirror about the
  world axes.
- **Its front is a framed panel**, and the recess sits about 16 mm behind the
  frame. Which of those two planes a ball meets depends on the ball's radius,
  because a rolling ball touches a vertical wall at its equator.
- **It stands on corner feet.** The body's underside is 50 mm off the floor;
  below that there is nothing but the feet.

So the chest goes into PyBullet as its own triangles. `render_ball_box_rebound.py`
exports the collision geometry to metre-scale OBJ through the same transform it
renders with (`--mode colliders`), and `simulate_ball_box_rebound.py` loads it as
a static concave trimesh. Physics and render read the same geometry through the
same matrices, so they cannot drift apart, and the reflection is measured off the
contact normal Bullet reports rather than assumed.

It goes out as **two** files, and that split is not cosmetic. With the chest, the
open lid, the toy basket and the teddy bear in one body, "the ball bounced off
the chest" and "the ball fetched up against the toy basket" are the same contact
event as far as the metrics can tell -- and an early version of this scene
cheerfully reported a 1.2 m/s rebound out of a 0.8 m/s approach, because the
basket contact 20 frames later had overwritten the real one. The chest is the
scene; the rest is in the simulation only so that a ball pushed harder than the
hero case stops against it the way it visibly would instead of rolling through
the rendered furniture, and touching any of it raises a warning.

## Files

- `simulate_ball_box_rebound.py` -- PyBullet physics simulation.
- `render_ball_box_rebound.py` -- Blender rendering script: imports and
  calibrates the room, lays the floor, lifts the two balls out of the set, and
  applies the physics trajectory as keyframes. Also owns the collider export.
- `cam_survey.py` -- renders the camera survey (10 framings x 4 key frames).
- `batch_render_ball_box_rebound.py` -- orchestrates multiple renders.
- `edit_vocab.py` -- the scene's PCVE edit vocabulary: which objects and
  properties may be edited, how each maps to a simulator parameter, and the
  baseline physics values. Must stay in sync with
  `render_ball_box_rebound.create_scenario()`.
- `build_pcve_ball_box_rebound.py` -- builds the PCVE edit suite: one source
  video plus one variant per edit-DSL string.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame (the ball-ball impact)
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/ball_box_rebound/render_ball_box_rebound.py -- \
    --mode preview \
    --out-dir renders/ball_box_rebound_preview \
    --resolution 960 540 \
    --samples 64 \
    --device auto

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/ball_box_rebound/render_ball_box_rebound.py -- \
    --mode animation \
    --out-dir renders/ball_box_rebound \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 128 \
    --device auto
```

## Simulate only

The collision meshes have to exist first. The render script writes them
automatically when they are missing, or on demand:

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/ball_box_rebound/render_ball_box_rebound.py -- \
    --mode colliders --out-dir /tmp/colliders
```

Then the simulation runs on its own and prints a one-line summary, so aims and
surfaces can be compared without rendering anything:

```bash
python scripts/ball_box_rebound/simulate_ball_box_rebound.py \
  --out /tmp/bbr.json \
  --chest-collider assets/collision/toy_box_chest.obj \
  --props-collider assets/collision/toy_box_props.obj
```

```
[SIM] chest=True f=13 in=2.356 out=1.723 inc=30.0 refl=29.2 | run=0.554 |
      ballB=True f=24 a_in=1.097 a_out=0.272 b_out=0.921 b_travel=0.303 |
      A_end=(0.260, 0.651) B_end=(0.606, 0.834) settle=(37, 45)
      props=None off_floor=None
```

`--disable-ball-b` leaves the target ball out, which is what the aiming was done
with: the rebound line has to be measured before there is anywhere sensible to
put the ball it is supposed to hit.

## Build PCVE suite

```bash
python scripts/ball_box_rebound/build_pcve_ball_box_rebound.py \
  --out-root renders/pcve_ball_box_rebound_suite \
  --resolution 1280 720 --fps 24 --samples 64 --device auto
```

One source video plus N edited variants. Every edit is declared as a **single
edit-DSL string** in `EDIT_CASES`, and the natural-language prompts (vague +
quantitative, zh + en), the scenario overrides handed to the renderer, and the
manifest's `physics_diff` are all derived from that one string by
`scripts/pcve_edit_dsl.py` against `edit_vocab.py`. Nothing about the edit is
written twice, so a prompt cannot drift from the physics it describes.

An edit may only **delete one object** or **set one scalar property** on one
object. The camera, the lighting, the aim, the seed and the layout are
identical across every case, and every case is 3.0 s long -- the only thing
that varies in frame is the motion.

Editable objects are `ball_a` (the star ball), `ball_b` (the football) and
`chest`. Editable properties are `mass`, `friction`, `restitution`, and
`initial_velocity` on `ball_a`, which is the push. The `chest` only exposes
`restitution` -- its friction is saturated and editing it would be invisible.
Only `ball_b` can be deleted. The floor is deliberately **not** editable
(edits name a moving object; nothing is lost -- see the traps below), and
the aim (`launch_heading_deg`) is not editable either: re-aiming the shot is
a different scene, not an edit to this one.

Source: the ball meets the panel on **frame 13 at 2.36 m/s and 30 deg**, leaves
at **29.2 deg and 1.72 m/s**, and hits the football on **frame 24**, knocking it
**0.30 m** clear. Both balls settle by frame 45. Every value below came from
sweeping the simulator directly.

| case_id | edit_dsl | outcome |
|---|---|---|
| `ball_box_rebound_baseline` | – (source) | See above. |
| `edit_dead_chest` | `SET chest.restitution FROM 0.8 TO 0.35` | Same point, same 2.36 m/s, same 30 deg in -- but **50.4 deg out**. Runs along the front of the chest and past the football, which never moves. |
| `edit_dead_ball` | `SET ball_a.restitution FROM 0.88 TO 0.2` | The same loss of bounce put on the ball instead. Pair restitution 0.16 vs the chest edit's 0.28, so flatter still: **64.9 deg out**, missing by more. |
| `edit_soft_push` | `SET ball_a.initial_velocity FROM 2.6 TO 1.95` | **Distractor.** Rebound angle untouched (29.4 deg) -- speed does not bend a rebound. Still reaches the football but nudges it 0.05 m, on frame 38 instead of 24. |
| `edit_draggy_ball` | `SET ball_a.friction FROM 1 TO 5` | A tacky ball, not a changed floor: effective rolling resistance 0.006 -> 0.030. Arrives at the chest on frame 18 at 0.85 m/s. Still bounces at 30.0 deg but dies before the football; settled by frame 22. |
| `edit_heavy_target` | `SET ball_b.mass FROM 0.09 TO 0.9` | Roll, bounce and rebound line identical; the football moves 0.013 m instead of 0.300 and the star ball rebounds back off it at 0.57 m/s. |
| `edit_heavy_ball` | `SET ball_a.mass FROM 0.12 TO 1.2` | Bounce unchanged (restitution sets it, not mass); the football leaves at 1.65 m/s and travels 0.84 m while the star ball carries on through. |
| `edit_remove_target` | `DELETE ball_b` | Roll and bounce identical, then the rebound crosses empty floor and rolls out of the bottom of frame around frame 48. The last second is bare floor. |

Three things worth knowing before adding an edit here:

- **A bounce scales only the normal component.** That is the whole scene: less
  restitution means a *flatter exit angle*, not merely a slower one. It is also
  why `edit_soft_push` is worth rendering -- it produces a weak outcome with the
  angle left alone.
- **Restitutions multiply.** `ball_a x chest = 0.88 x 0.80 = 0.70` at baseline,
  so a prompt's number is never the number that governs the bounce. This is why
  `edit_dead_chest` (0.35 -> pair 0.28) and `edit_dead_ball` (0.2 -> pair 0.16)
  land at different angles rather than duplicating each other.
- **`friction` is one knob per object, and it does the whole story.** The
  vocabulary binds `friction` compound-style; on both balls the rolling
  coefficient baseline is 0, so scaling `friction` scales only the lateral
  side. That is not a problem: Bullet combines rolling resistance as
  `rf_a * lateral_b + rf_b * lateral_a`, so with the ball at `rf = 0` the
  effective rolling resistance on the floor is `floor_rolling_friction *
  ball_a_friction`. Raising `ball_a.friction` from 1 to 5 therefore reproduces
  raising the floor's rolling friction from 0.006 to 0.03 exactly -- verified
  identical to the millimetre -- and, unlike the floor edit, it leaves the
  bounce off the chest untouched.

Adding or changing an edit:

```bash
# 1. sweep the sim directly first -- no Blender, no rendering
python scripts/ball_box_rebound/simulate_ball_box_rebound.py \
    --out /tmp/s.json \
    --chest-collider assets/collision/toy_box_chest.obj \
    --props-collider assets/collision/toy_box_props.obj \
    --launch-x -0.744 --launch-y 1.030 --launch-speed 2.60 \
    --launch-heading-deg 303.74 --ball-b-x 0.358 --ball-b-y 0.660 \
    --ball-a-radius 0.080 --ball-b-radius 0.065 \
    --chest-restitution 0.35
# 2. add an EditCase to EDIT_CASES, then check what it generates
python scripts/ball_box_rebound/build_pcve_ball_box_rebound.py \
    --dry-run --out-root /tmp/dryrun
# 3. render only what is new; --clean-stale-cases rm -rf's dropped cases
python scripts/ball_box_rebound/build_pcve_ball_box_rebound.py --skip-existing
```

The simulator prints a one-line `[SIM]` summary carrying the contact frames,
the incidence and reflection angles, both balls' speeds and travel, and the
resting positions -- that is what a sweep should be read off, and the build
script echoes it plus any `[WARN]`/`[NOTE]` for each rendered case.

## Scene layout

- **The launch corner is the one that exists.** The toy basket blocks everything
  west of x = -0.73 up to y = 0.61, so the run-up has to come in over the
  basket's shoulder from the north-west, and the ball is launched from
  (-0.74, 1.03) heading 303.7 deg. It clears the basket's north edge with 0.24 m
  to spare and covers 1.15 m of open floor before reaching the chest.
- **30 deg off the normal is as square as this room allows**, and squarer would
  be wrong anyway: a genuinely head-on bounce sends the ball straight back up its
  own approach line, and there is then nowhere to stand the target ball that the
  roller did not already run through on the way in. At 30 deg the ball visibly
  bounces off rather than glances, and the rebound crosses the room diagonally
  instead of retracing the approach.
- **The target ball sits 0.40 m past the chest on the rebound line, offset 40 mm
  off centre.** The gap is what makes the shot read as three events rather than
  one scramble -- the bounce lands on frame 13 and the ball-ball hit on frame 24,
  nearly half a second apart. The offset is what keeps both balls alive
  afterwards: dead centre, the roller stops where it stands (0.17 m/s) and only
  the target moves, which is a tidier demonstration of momentum transfer but a
  much less informative one. Offset, the roller comes away at 0.27 m/s on one
  heading and the target at 0.92 m/s on another, and the split is visible.
- **Camera**: a low three-quarter angle from the room's north-east
  (1.20, 1.40, 0.36), 35 mm, chosen from a survey of ten positions kept in
  `renders/bbr_camera_survey/` (`cam_survey.py` regenerates it). It sits barely
  more than two ball diameters off the floor, so the balls read as objects
  rolling across a floor rather than markers on a plan, and the three-quarter
  angle on the panel is what makes the bounce legible -- square-on to it the
  approach and the rebound project onto nearly the same line and the ball just
  appears to stop and start again. The wider north-east positions hold more of
  the room but shrink the balls; the north-west ones give the approach more depth
  at the cost of a foreground ball nearly filling frame 1; the tighter north
  position loses the launch off the edge entirely.

## Physics notes

- **Rolling resistance is put on the floor, not on the ball.** Bullet does not
  combine rolling friction as a plain product: it uses
  `rf_a * lateral_b + rf_b * lateral_a`. Setting the ball to `rf = 1.0` -- the
  obvious way to make the surface's value pass straight through -- therefore
  multiplies in the *floor's lateral friction* instead and turns the roll into a
  skid. With the ball at `rf = 0` and lateral friction 1.0, the effective rolling
  resistance is exactly the floor's own value.
- Both balls are launched already rolling (`omega = (z_hat x v) / r`). Launched
  with zero spin the roller spends its first tenth of a second skidding while
  friction spins it up, which reads as a shove rather than a roll.
- **Bullet multiplies the two bodies' restitutions.** What the bounce actually
  sees is `chest_restitution * ball_a_restitution`, so the hero's 0.80 against
  the ball's 0.88 gives 0.704. Both are exposed separately because only one of
  them is a property of the chest, and the effective product is written into the
  ground truth.
- **The contact-exit reflection and the path a few frames later are not the same
  angle**, and the difference is real rather than solver noise. The ball leaves
  the panel at very nearly the specular angle, and then the floor gets hold
  of it: its spin still points along the *approach*, so for
  the first frame after the bounce friction drags the velocity back toward where
  it was going and the path bends flatter (29 deg at contact exit, about 54 deg
  from the normal by two frames later). The ground truth records the contact-exit
  value, which is the one that is a property of the collision; the target ball's
  placement was measured off the actual path.
- **The floor needs spinning friction, and this is not a refinement.** Bouncing
  off the chest at an angle gives the ball a real 10.5 rad/s of spin about the
  *vertical* axis -- the chest's surface friction acts tangentially at the
  contact, and that is what tangential friction does. Nothing else in Bullet
  opposes that component: rolling friction resists rolling, lateral friction
  resists sliding. With `floor_spinning_friction` at 0 both balls therefore came
  to a dead stop on the floor and went on turning on the spot at 7.7 and
  4.7 rad/s, forever, which is plainly visible in the render. At 0.008 -- an
  8 mm contact patch, the same order as the rolling-friction figure -- the spin
  bleeds away over about a third of a second, and nothing else moves: the
  rebound speed, the reflection angle, the rebound run and the ball-ball contact
  frame are all identical to four figures, because spinning friction acts only
  about the contact normal. The ground truth carries `residual_spin` and
  `spinning_on_the_spot` as a standing check, and "settled" means stopped moving
  *and* stopped turning -- speed alone was what let this through the first time.
- **The ball must carry `spinning_friction = 0`, not 1.0**, for exactly the
  reason `ball_rolling_friction` must, and it is the same trap: Bullet combines
  spinning friction as `sf_a * lateral_b + sf_b * lateral_a` too. Setting the
  ball to 1.0 -- the obvious way to let the floor's value pass through --
  multiplies in the floor's *lateral* friction instead and yields an effective
  0.6, which is enough to bleed linear speed as well as spin: the ball reached
  the target at 0.78 m/s instead of 1.10 and moved it 0.06 m instead of 0.30.
- **Impacts are sampled at substep resolution**, not at frame boundaries. At 24
  fps and 60 substeps a single frame is long enough for the floor to eat a
  measurable slice of the rebound before anything gets recorded.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json`, or
directly in `build_pcve_ball_box_rebound.py` for suite cases.

- `chest_restitution`: the parameter this scene is built around (default
  `0.80`). Multiplied by the ball's own restitution it gives the fraction of the
  *normal* speed component that survives, so it changes the rebound's direction
  as well as its speed. Below about `0.50` the rebound flattens enough to miss
  the target ball entirely.
- `launch_speed`: the push, in m/s (default `2.60`). Changes when everything
  happens and how hard the target is hit, but **not** the rebound angle -- which
  is exactly what makes `soft_push` a useful distractor.
- `launch_heading_deg` / `launch_x` / `launch_y`: the aim. Defaults put the
  contact 0.11 m west of the panel's centre at 30 deg off its normal.
- `ball_b_x` / `ball_b_y`: where the target ball waits. It has to sit on the
  rebound line for the hero restitution; the other suite cases deliberately miss
  it.
- `floor_rolling_friction`: the scene's clock (default `0.0060`). Low enough
  that the ball makes it back off the chest, high enough that neither ball runs
  out past the boards.
- `floor_spinning_friction`: what stops a ball turning on the spot after it has
  stopped moving (default `0.008`). Affects nothing else; setting it to 0
  reproduces the bug described above.
- `ball_a_mass` / `ball_b_mass`: default `0.120` kg / `0.090` kg.
- `gravity_z`: default `-9.8`.

## Outputs

- `ball_box_rebound.mp4` -- rendered video (animation mode).
- `preview.png` -- preview still (preview mode).
- `ball_box_rebound.blend` -- saved Blender scene.
- `ground_truth_transforms.json` -- per-frame transforms, velocities and speeds
  for both balls, plus a `phase` flag on the roller marking `approach` /
  `rebound` / `after_impact`, and a `moving` flag on the target. The `quality`
  block carries the chest contact frame, point and normal, the approach and
  rebound speeds, the incidence and reflection angles, the length of the rebound
  run, the ball-ball contact frame and the speeds either side of it, both settle
  frames, and three standing failure checks (`spinning_on_the_spot`,
  `hit_room_props`, `left_floor`). Each frame also carries `spin_z` per ball.
- `scenario_metadata.json` -- seed, camera, and physics parameters.
