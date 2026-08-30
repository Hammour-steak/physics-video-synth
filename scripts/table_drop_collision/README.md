# Tennis ball rolls off a coffee table onto another (projectile + oblique impact)

A tennis ball is rolled west across a round coffee table, runs off the edge,
falls the height of the table and lands on a second tennis ball waiting on the
floor. Nothing about the outcome is scripted. The ball is given one push and
from then on the scene is two textbook problems bolted together, each of which
settles a different half of what the camera sees.

**1. Where it lands is settled at the lip.** Once the ball leaves the edge
nothing touches it, so it is a projectile: it keeps the horizontal speed it had
at the lip and falls under gravity alone. The fall time depends only on the drop
and gravity -- for this table 0.313 s, and not on the ball, its mass, its size or
its speed. The ball leaves the lip at 1.053 m/s and lands 0.331 m clear of the
edge, at (-0.672, -0.290). The simulation computes that parabola in closed form
and checks the trajectory against it on every run; the two agree to **1.0 mm**.

**2. Where the struck ball goes is settled by the line of centres.** Two spheres
exchange momentum only along the line joining their centres, so with the balls
identical the target leaves along that line and the ball that hit it keeps
whatever was perpendicular to it. The target sits dead in line with the lane, so
the whole exchange stays in one vertical plane: the contact normal comes out with
a y component of -0.0000, the target is driven due west at 1.297 m/s and the ball
that hit it is thrown due east, **exactly 180.0 deg apart**, and both finish on
y = -0.290, the line they started on.

**It is still not a central collision, and it cannot be made into one.** A
collision is central when the approach lies along the line of centres. Here the
ball arrives 68.3 deg below horizontal -- it is falling, not rolling -- while the
line of centres is only 25.2 deg below horizontal. The two are **43.1 deg apart**
and the impact parameter is 0.683 of a full ball width, which is a glancing blow
by any measure. Squaring it up means putting the target where the line of centres
points along the fall, i.e. almost directly under the ball; that works
geometrically (the impact parameter drops to 0.022) and is useless, because the
momentum then goes into the floor and **the target moves 14 mm**. A shallower
arrival needs a much harder push, which walks the whole exchange half a metre
further west: at 2.6 m/s off the lip the collision is crammed against the left
edge of frame, at 3.0 m/s the struck ball ends up inside the glazed wall, and the
closed-form check starts disagreeing with the simulation. The obliquity is the
price of dropping a ball onto another ball. What *can* be removed is the sideways
component, and it has been.

Both balls are `assets/models/tennis_ball-3.glb`, sized to a real ITF Type 2 ball
at 67 mm and 57 g. **The mass ratio is therefore exactly 1**, which is what lets
the collision be checked against a closed form with nothing in it to tune. What
tells the two apart on screen is only age -- the one on the floor is duller,
greener and darker -- so the difference the viewer can see is deliberately not a
difference the physics can see.

**A tennis ball is a hollow shell, not a solid sphere.** Its moment of inertia is
2/3 m r², not the 2/5 m r² of a solid ball -- 67 per cent more resistance to
being spun up. PyBullet computes the inertia of a `GEOM_SPHERE` as a solid, so
`add_ball()` overrides it, and that override is what makes the struck ball skid
before it rolls.

The room is `assets/models/living_room.glb`, used as static background: its
round coffee table, three-seat sofa, oiled board floor and 3.15 m glazed west
wall are what make the shot read as a real room. Three things about that model
had to be corrected, and each of them is a trap that fails quietly:

- **The floor's "board texture" is a lightmap.** The base-colour atlas has an
  afternoon painted into it -- hard diagonal blind bands, a blot under the sofa,
  a ring under a rug that is not in this scene, and a black rectangle. It was
  read as wood grain for a long time because it is grey on grey. The tell is to
  turn the sun off: the stripes are still there. It cannot be lit, occluded or
  softened, because it was painted for a different sun. `unbake_floor()` replaces
  it with the one colour it is actually made of. Nothing is lost doing so: over
  the floor's own UV region the 50th to 99th centile of the atlas spans **six per
  cent**, so there is no grain under the shading to keep. The albedo is measured
  rather than chosen -- baked light only ever subtracts, so the unshaded boards
  are the bright end of the distribution, and the 95th centile and above average
  to linear (0.4645, 0.4579, 0.4496).
- **The glazed west wall has no glass.** `Object_6` is the mullion frame and the
  panes are open holes, so the sun prints the whole window grid across the floor,
  the sofa and the table top. That shadow is real and correct, but at this
  elevation the bars fall straight across the lane the ball is rolled down, and a
  67 mm ball crossing a hard shadow edge every few frames reads as the ball
  changing colour rather than as the light. `lighting.window_frame_shadow`
  defaults to `false`, which takes the frame out of the sun's path and keeps the
  daylight, the raking highlight and every contact shadow.
- **The floor ships at roughness 0.154**, a wet look on what is meant to be an
  oiled board. `temper_surfaces()` takes it to 0.34, which keeps a sheen and a
  soft reflection of the balls without turning the room upside down underneath
  them.

The table top is deliberately *not* adjusted, and the reason is worth recording:
its material drives Roughness and Metallic from a packed texture through a Math
node, and assigning to `default_value` on a socket that already has a link does
nothing at all -- no error, no warning, and a render identical to the one before
it. `report_driven_inputs()` prints which sockets are in that state so the next
person does not have to find out the hard way.

**The ball models are multi-mesh and every mesh is part of the ball.**
`tennis_ball-3.glb` arrives as a 994-vertex green body plus a separate 320-vertex
white seam laid over it. `import_ball()` joins them and treats every material
slot. Keeping only the largest mesh, which is the obvious thing to do, throws the
seam away -- and the seam is the only thing on a sphere that shows it is rolling
rather than sliding. On `tennis_ball-2.glb` the failure is worse and completely
silent: its body and its fuzz shell have *the same 444 vertices*, so `max()`
returns whichever the importer happened to hand over first and the ball renders
as a flat plastic sphere.

The world frame is the natural one: z = 0 is the floor, (0, 0) is the centre of
the round table, +x runs east, +y runs north toward the sofa. The room is scaled
by `0.900 / 1.24350` so the table comes out 0.900 m across; the sofa then lands
at 2.209 m wide, 0.982 m deep, with its seat at 0.421 m, which is a real
three-seater and confirms the scale independently. **The simulation works in the
same frame, so its output needs no transform at all before being keyframed.**

## Files

- `simulate_table_drop_collision.py` -- PyBullet physics simulation, plus the
  closed-form projectile and oblique-impact predictions it checks itself against.
- `render_table_drop_collision.py` -- Blender rendering script: imports and
  calibrates the room, unbakes the floor, builds the light, imports both balls
  and applies the physics trajectory as keyframes.
- `cam_survey.py` -- renders the candidate camera positions the framing was
  chosen from.
- `batch_render_table_drop_collision.py` -- orchestrates multiple renders.
- `build_pcve_table_drop_collision.py` -- builds the five-case PCVE suite.

## Quick start

```bash
# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/table_drop_collision/render_table_drop_collision.py -- \
    --mode preview \
    --out-dir renders/table_drop_collision_preview \
    --resolution 960 540 \
    --samples 64 \
    --device auto

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/table_drop_collision/render_table_drop_collision.py -- \
    --mode animation \
    --out-dir renders/table_drop_collision \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 2.8 \
    --samples 128 \
    --device auto
```

`--preview-frames 0 21 45` renders several stills instead of one, which is what
`cam_survey.py` drives and the quickest way to check a change to the light.

## Simulate only

The renderer shells out to the simulation, but it runs standalone and prints a
summary, so parameters can be compared without rendering anything:

```bash
/remote-home/chenyuanjie/miniconda/envs/physics/bin/python \
  scripts/table_drop_collision/simulate_table_drop_collision.py \
  --out /tmp/tdc.json \
  --props-collider assets/collision/living_room_table_dressing.obj
```

PyBullet lives in the `physics` env only -- neither Blender's bundled Python nor
the `vlm`/`GUI` envs have it. The renderer finds that interpreter itself.

`--props-collider` needs `assets/collision/living_room_table_dressing.obj`, which
the renderer generates from the room model; run the renderer once with
`--mode colliders` if it is missing.

`--disable-ball-b` leaves the target out entirely, which is how the falling
ball's own touchdown was pinned down before anything was placed for it to hit.

## Build PCVE suite

```bash
python scripts/table_drop_collision/build_pcve_table_drop_collision.py \
  --out-root renders/pcve_table_drop_collision_suite \
  --resolution 1280 720 \
  --fps 24 \
  --samples 64 \
  --device auto
```

Four cases hold the push fixed at 1.28 m/s and slide the target across the line
of flight, sweeping **one quantity: the impact parameter**. Because the landing
point is settled at the lip, the falling ball touches down at **(-0.672, -0.290)
in all four** -- the number does not move by a millimetre -- so the only thing
that differs is the line joining the two centres at contact. The fifth is a
deliberate distractor: it restores the hero's target and softens the push
instead, producing a miss for the opposite reason.

| case_id | `ball_b_y` | push (m/s) | off-centre | b/(2r) | outcome |
|---|---|---|---|---|---|
| `table_drop_collision_in_line` | -0.290 | 1.28 | 43.1° | 0.683 | **Hero, and the squarest hit available.** Everything in one vertical plane: target due west at 1.297 m/s running 0.304 m, hitter due east, exactly 180.0° apart, both ending on y = -0.290. |
| `table_drop_collision_offset` | -0.258 | 1.28 | 56.3° | 0.832 | Line of centres swings out of the fall plane. Target leaves on 152.6° at 1.152 m/s, runs 0.236 m; the hitter is thrown back toward the camera and runs 0.965 m. |
| `table_drop_collision_graze` | -0.246 | 1.28 | 75.2° | 0.967 | Clips the crown, within a thirtieth of a miss. Target leaves at 0.673 m/s on 142.5° and stops after 0.078 m; the hitter barely notices and carries on 1.013 m. |
| `table_drop_collision_out_of_line` | -0.235 | 1.28 | — | >1 | Clean miss. Same landing point as the other three, but it passes north of the target; runs out 1.177 m. Target never moves. |
| `table_drop_collision_soft_push` | -0.290 | **1.22** | — | — | **Distractor.** Target still dead in line; the ball lands at **(-0.649, -0.290)**, 73 mm short. Same end state as `out_of_line` -- target untouched, hitter run out west -- from a different cause. |

The sweep runs *upward* from 0.683 rather than through zero, because 0.683 is the
floor: the ball arrives falling and no placement of the target makes the hit
squarer without also making it useless. See above.

The two misses are the pair that makes the suite worth rendering. At rest they
are the same picture. Distinguishing them means noticing either the slower roll
across the table or that the target is 50 mm off the line in one and dead in line
in the other -- the resting positions do not carry it.

The manifest records each case's *outcome* (contact frame, both travels, both
resting positions, settle frames), read back out of the rendered ground truth
rather than from the inputs. It also records each case's landing point and
prints a warning if the four fixed-push cases stop agreeing on it, since that
would mean something had leaked from the target's position back into the launch
and the cases were no longer varying only the geometry.

**Three of the cases need a longer clip.** The graze and both misses leave the
falling ball running across the floor and it does not stop until frame 72-77, so
they ask for 3.4 s where the default is 2.8 s. Rendering them shorter is not
wrong, it just ends on a ball that is still moving.

**A miss produces a spurious settle warning.** The renderer warns when either
ball has no settle frame, and a target that is never touched never registers one
-- so `out_of_line` and `soft_push` both print "a ball was still moving on the
last frame" about a ball that never moved at all. Check `hit_ball_b` before
believing it.

## Scene layout

- **The lane runs across the near, southern third of the table.** The open
  magazine lies across the middle with its south edge at y = -0.174, so a lane at
  y = -0.290 clears it by 82 mm, and it keeps the whole roll on the camera's side
  of the table where the ball is a ball rather than a dot behind a cup. The run
  west from the push to the lip is 0.657 m, which is as much of this table as
  exists on a line that clears the magazine.
- **The ball leaves by the west lip because that is the side with somewhere to
  go.** The floor runs 2.0 m further west before the glazed wall; to the east it
  stops 0.21 m past the table's rim. It also puts the whole shot across the frame
  rather than into the lens, which is the only way a 67 mm ball reads at this
  distance. A first pass had the ball rolling south and falling toward the
  camera; it was 67 mm of ball coming straight at a lens two metres away.
- **The table dressing is kept and is why the lane is where it is.** The cup, the
  magazine and the reading glasses are exported as their own PyBullet body rather
  than merged into the table, so touching any of them is reported as its own
  event -- the hero take passes 44 mm clear of the magazine's east edge. Hiding
  them would have handed the ball the whole 0.9 m of table to aim across, and it
  was tempting, but they are the only thing in the shot that gives the top a
  scale and they cost nothing.
- **The target is dead in line, and the floor's friction is what pays for it.**
  This lane used to sit 22 mm off the line, and the offset was not decoration: it
  threw the struck ball sideways, clear of two failures that dead-in-line walks
  straight into. In line, the hitter is thrown back due east and, at the honest
  `floor_rolling_friction` of 0.0060, comes to rest at radius 0.344 -- under the
  table's overhang, hidden behind the top -- while the struck ball slides to
  x = -1.204, which projects to screen x = -0.016, twenty pixels past the left
  edge of frame. Raising the floor to 0.0110 fixes both without bending the
  exchange: the hitter stops at radius 0.511, a clear 62 mm outside the table's
  rim, and the struck ball at -1.026, comfortably inside.
- **The target sits 50 mm past the falling ball's own touchdown**, at x = -0.722
  against a landing at -0.672, so the ball meets it on the way down rather than
  after bouncing. Move it to the touchdown itself and the ball lands squarely on
  its crown and the exchange is vertical; another 10 mm out and it misses.
- **Camera**: south of the table at (-0.34, -2.05, 0.92) on a 40 mm lens, aimed
  at (-0.30, -0.30, 0.20). Chosen from the survey in `cam_survey.py`. Every
  candidate stands south, which is the only side there is -- the sofa's front
  face is 0.356 m off the table's north rim, so there is no room for a camera
  behind it.
- **Motion blur is on, at half a frame.** This scene needs it more than most: the
  ball hits the floor at 3.25 m/s, which is 135 mm between frames against its own
  67 mm diameter. Without blur the fall is four disconnected copies of a ball
  rather than a ball falling.

## Physics notes

- **Rolling resistance is put on the surfaces, not on the balls.** Bullet does
  not combine rolling friction as a plain product: it uses
  `rf_a * lateral_b + rf_b * lateral_a`. Giving a ball `rf = 1.0` to let each
  surface's value pass through instead multiplies in that surface's *lateral*
  friction and turns the roll into a skid. With the balls at 0 the effective
  value is the surface's own rolling friction times the ball's lateral friction,
  which is what the tuning below assumes.
- **The balls are launched already rolling.** Launched with zero spin the ball
  spends its first tenth of a second skidding while friction spins it up, which
  reads as a shove rather than a roll.
- The table's `0.0037` is felt on a lacquered surface, far lossier than the glass
  marbles this room's other scene rolls; it works out to a rolling coefficient of
  0.041 and costs the ball 0.40 m/s² over the 0.657 m run, so 1.28 m/s at the
  push becomes 1.053 m/s at the lip.
- The floor's `0.0060` is worth 0.65 m/s² to a tennis ball -- perhaps a third
  above what felt on hardwood really is, and chosen for the frame rather than the
  physics: any less and both balls run far enough to walk the struck one toward
  the glazed wall and out of the light.
- **Restitution multiplies.** Bullet multiplies the two bodies' values, so the
  balls' 0.86 each gives 0.74 ball-on-ball, which is a tennis ball; against the
  floor's own 0.86 it is an effective 0.75, which is the ITF rebound spec.
- A ball counts as stopped below `SETTLE_SPEED` = 0.02 m/s.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json` under a
`physics` key, or directly in `build_pcve_table_drop_collision.py` for suite
cases. Note that `--disable-ball-b` is a simulation flag only and is not
reachable through the scenario overrides.

- `ball_b_y`: how far off the rolling ball's line the target sits (default
  `-0.290`, i.e. dead in line). **The parameter the suite is built around** -- it
  is the impact parameter in disguise, rotating the line of centres and with it
  everything about the exchange, while leaving the falling ball's own landing
  point untouched. The window is narrow and one-sided: in line is the squarest
  the hit can be, 32 mm off is a solid oblique strike, 44 mm is a graze, and
  beyond about `-0.240` it misses.
- `ball_b_x`: how far west along the line the target sits (default `-0.722`,
  50 mm past the touchdown at `-0.672`).
- `launch_speed`: the push, in m/s along the lane (default `1.28`). Moves the
  landing point, so a few per cent is enough to turn a hit into a miss; move
  `ball_b_y` with it.
- `launch_x` / `launch_y`: where the push starts (default `0.315`, `-0.290`).
- `table_rolling_friction`: rolling resistance of the lacquered top (default
  `0.0037`), which sets how much of the push survives to the lip. Do not confuse
  it with the 0.041 in its help text -- that is the rolling *coefficient* the
  value works out to, not the value.
- `floor_rolling_friction`: rolling resistance of the bare boards (default
  `0.0060`), which sets how far both balls run after the exchange. Likewise the
  0.65 m/s² quoted for it is the deceleration, not the parameter. **It saturates
  hard**: anywhere above roughly `0.02` the balls stop rolling and start sliding,
  lateral friction takes over, and `0.065`, `0.095` and `0.130` all give bit-for-
  bit identical resting positions.
- `ball_a_restitution` / `ball_b_restitution`: default `0.86` each, i.e. 0.74
  between them.
- `gravity_z`: default `-9.8`. Changes the fall time and therefore the landing
  point, but not by way of anything about the ball.

## Outputs

- `table_drop_collision.mp4` -- rendered video (animation mode).
- `preview.png`, or `preview_f<NNN>.png` with `--preview-frames`.
- `table_drop_collision.blend` -- saved Blender scene.
- `ground_truth_transforms.json` -- per-frame transforms for both balls and the
  camera, plus the launch block (lip frame, lip speed, roll deceleration, the
  closed-form predicted flight and the error against it) and the collision block
  (mass ratio, effective restitution, contact normal, and the predicted versus
  measured speed and heading of the struck ball). The `quality` block carries
  `left_table`, `lip_frame`, `hit_ball_b`, `contact_frame`, `contact_point`,
  `contact_normal`, `airborne_frames`, both finals, both travels, both settle
  frames, `residual_spin`, `hit_table_dressing` and `ran_off_floor`.
- `scenario_metadata.json` -- seed, camera, lighting and physics parameters.
