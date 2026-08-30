# Toy ball rolls onto a rug and stops (floor vs. carpet rolling resistance)

A toy ball is rolled across a living room's bare floor onto the square area rug
in the middle of the room, and is brought to a stop by the rug's pile. Nothing
about the stop is scripted: the ball is given one initial velocity and then
exactly one thing changes under it -- rolling resistance. The floor's is low
enough that the ball crosses the 0.5 m approach almost without slowing; the
rug's is twenty times higher, and that ratio alone is what stops it.

**The rug is flat.** `carpet_thickness` is 0, so the rug's surface is coplanar
with the bare floor and there is no step, lip or bump anywhere in the ball's
path -- it rolls across one continuous flat plane and simply starts shedding
speed the moment it crosses the rug's boundary. This is the point of the scene:
friction is the only variable, so nothing else can be credited for the stop.
The ground truth carries `max_lift_over_carpet` as a standing check on that,
and it reads 0.0000 m.

The ball is `assets/models/volleyball.glb`, scaled to a real 0.21 m FIVB
volleyball and given that ball's 0.27 kg. The environment is
`assets/models/living_room_interior_free.glb`, used as static background at its
own materials -- its polished floor, sectional sofa, coffee table, TV console
and window wall are what make the shot read as a real room rather than a
procedural set. Two things about that model had to be corrected; both are
documented in the code and both mattered:

- **Its floor material ships at Metallic 0.98 / Roughness 0.09**, i.e. as
  polished chrome. In Cycles that turns the floor into a perfect mirror and the
  room renders as though it were flooded. Since the ball rolls across this
  surface for the first half of the shot, `tame_floor_gloss()` de-metalises it
  and floors its roughness at 0.22 (a polished-concrete sheen, not a mirror).
- **Its rug carries a flat grey material** that reads as a painted patch. It
  is hidden and rebuilt by `build_carpet()` on exactly the same footprint with
  a procedural cut-pile material, so the rug stays where the room model put it
  but actually looks like carpet.

The model is also not authored in metres: its floor-to-ceiling span is 20.131
units, so the whole room is uniformly scaled by `2.75 / 20.131` and shifted so
that world z = 0 is the floor and world (0, 0) is the rug's centre. That
calibration checks out independently against the model's furniture (the
sectional comes to 2.20 m wide and 0.97 m deep, the rug to 1.53 m square).
Because the physics frame *is* the world frame, the PyBullet output needs no
transform at all before being keyframed.

## Files

- `simulate_ball_carpet_climb.py` -- PyBullet physics simulation.
- `render_ball_carpet_climb.py` -- Blender rendering script: imports and
  calibrates the room, rebuilds the rug, imports the ball, and applies the
  physics trajectory as keyframes.
- `batch_render_ball_carpet_climb.py` -- orchestrates multiple renders.
- `build_pcve_ball_carpet_climb.py` -- builds the five-case PCVE suite.

## Quick start

```bash
# Activate the environment that has PyBullet installed
conda activate physics

# Render a single preview frame
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/ball_carpet_climb/render_ball_carpet_climb.py -- \
    --mode preview \
    --out-dir renders/ball_carpet_climb_preview \
    --resolution 960 540 \
    --samples 64 \
    --device auto

# Render the full animation
./tools/blender-3.6.23-linux-x64/blender -b \
    --python scripts/ball_carpet_climb/render_ball_carpet_climb.py -- \
    --mode animation \
    --out-dir renders/ball_carpet_climb \
    --resolution 1920 1080 \
    --fps 24 \
    --duration-sec 3.0 \
    --samples 128 \
    --device auto
```

## Simulate only

```bash
python scripts/ball_carpet_climb/simulate_ball_carpet_climb.py \
  --out renders/test_ball_carpet_physics.json
```

The simulation prints a one-line summary (whether the ball reached the rug, its
speed crossing onto it, how far it then ran, where it stopped, and which frame
it settled on), so surfaces can be compared without rendering:

```bash
python scripts/ball_carpet_climb/simulate_ball_carpet_climb.py \
  --out /tmp/thick.json --carpet-rolling-friction 0.130
python scripts/ball_carpet_climb/simulate_ball_carpet_climb.py \
  --out /tmp/slick.json --carpet-rolling-friction 0.018
```

## Build PCVE suite

```bash
python scripts/ball_carpet_climb/build_pcve_ball_carpet_climb.py \
  --out-root renders/pcve_ball_carpet_climb_suite \
  --resolution 1280 720 \
  --fps 24 \
  --samples 64 \
  --device auto
```

Four cases hold the push fixed at 2.05 m/s and vary only the rug's pile, so the
ball crosses onto the rug at the same 2.0 m/s in every one and the surface is
the only difference. The fifth is a deliberate distractor: it restores the hero
rug and softens the push instead, so the ball stops early for an entirely
different reason. Telling those two apart requires watching the approach across
the bare floor -- the resting positions alone do not distinguish them.

| case_id | rug rolling friction | push (m/s) | travel on rug | outcome |
|---|---|---|---|---|
| `ball_carpet_climb_flatweave` | 0.018 | 2.05 | 1.12 m | Slick flatweave barely slows it; crosses the whole rug and rests against the sofa. |
| `ball_carpet_climb_low_pile` | 0.038 | 2.05 | 0.78 m | Thin low pile; runs most of the way across and stops just short of the sofa. |
| `ball_carpet_climb_stops_on_rug` | 0.060 | 2.05 | 0.49 m | **Hero.** Cut-pile wool; stops beside the coffee table. |
| `ball_carpet_climb_soft_push` | 0.060 | 1.70 | 0.29 m | **Distractor.** Same rug as the hero, gentler push; reaches the rug at 1.55 m/s and stops early. |
| `ball_carpet_climb_deep_pile` | 0.130 | 2.05 | 0.22 m | Deep shag; stopped barely two ball-widths past the border. |

The manifest records each case's *outcome* (carpet travel, resting position,
settle frame), read back out of the rendered ground truth, not just its inputs.

**There is a ceiling on the pile axis.** Above roughly `0.13` the ball stops
rolling and starts sliding, lateral friction takes over, and further increases
do nothing: `0.14`, `0.18` and `0.22` all stop it at exactly 0.220 m. Asking for
a thicker rug than that is silent no-op, so `deep_pile` sits at the knee.

## Scene layout

- The rug is 1.531 m square, centred on the world origin, its top face flush
  with the floor at z = 0. Both the collision box and the rendered slab are
  driven from the same `carpet_thickness`, so they cannot drift apart.
- **The bare floor is tiled as four boxes around the rug, not laid down as one
  infinite plane.** A plane would run underneath the rug, and with a flush rug
  the plane wins the ball's contacts, so the rug's friction never applies and
  the ball sails straight across it. The four tiles surround the rug's footprint
  without overlapping it and all have their top faces at exactly z = 0, so the
  ball crosses onto the rug over a continuous surface.
- **The ball's lane (`launch_x = -0.57`) is the one that exists.** The room's
  coffee table stands on the rug, occupying x -0.38..0.53, y -0.52..0.38, which
  leaves a ~0.39 m corridor between the rug's west edge and the table's legs.
  A ball rolled along +Y at x = -0.57 threads that corridor with ~9 cm of
  clearance on the rug side and ~8 cm on the table side, and has 1.2 m of clear
  rug ahead of it. Rolling along +X instead was tried first and does not fit:
  every lane that clears the table's south rail leaves the ball within 1-2 cm of
  falling off the rug's south edge.
- The coffee table's four legs and the sofa's base are in the simulation as
  static boxes. The hero ball touches neither, but the suite's low-friction
  cases reach the sofa, and without it they roll straight through the rendered
  furniture.
- **Camera**: a wide establishing shot from high on the room's south side,
  chosen from a survey of eight positions (kept in
  `renders/bcc_camera_survey/`). It holds the ball's entire path plus the whole
  room. The tighter alternatives all failed on one end or the other: every low
  south-east angle ends with the coffee table's near leg beside the stopped
  ball, and every west-side angle loses the run-up off the bottom of frame.
- `launch_y` is set by the framing rather than by the physics. The floor is
  slick enough that the run-up's length barely changes the speed reaching the
  rug (-1.55 and -1.08 both arrive at ~2.0 m/s and stop within 4 cm of each
  other), so it is set to the longest approach that still has the ball fully
  inside frame 1.
- **The rendered rug is lifted 1.2 mm above the floor**, purely so a mesh lying
  exactly on the room's floor does not z-fight with it across the whole 1.53 m
  square. This is the one place the render deliberately disagrees with the
  physics; 1.2 mm is under 1.2% of the ball's radius. The rug's surface noise is
  one-sided for the same reason -- a symmetric lift puts half the vertices back
  below the floor and the rug renders as a shimmering chequerboard.

## Physics notes

- **Rolling resistance is put on the surfaces, not on the ball.** Bullet does
  not combine rolling friction as a plain product: it uses
  `rf_a * lateral_b + rf_b * lateral_a`. Setting the ball to `rf = 1.0` -- the
  obvious way to make each surface's value pass straight through -- therefore
  multiplies in the *floor's lateral friction* instead, which killed the ball's
  spin on the very first contact and turned the entire roll into a 4.4 m/s²
  skid that stopped it in 0.3 m of bare floor. With the ball at `rf = 0` and
  lateral friction 1.0, the effective rolling resistance is exactly the
  surface's own value, which is what lets the floor be slick and the rug grippy
  in one continuous roll.
- The ball is launched already rolling (`omega = -v/r` about X for travel along
  +Y). Launched with zero spin it spends its first tenth of a second skidding
  while friction spins it up, which reads as a shove rather than a roll.
- **Rolling friction saturates above about `0.13`.** Past that the ball stops
  rolling and starts sliding, lateral friction governs the deceleration instead,
  and raising `carpet_rolling_friction` further has no effect whatsoever.
- The rug's default `0.060` is more than double the `0.028` this scene used when
  the rug was a raised 18 mm slab. That is not a retune for its own sake: back
  then the edge itself ate 0.75 m/s before the pile got the ball at all, so the
  ball arrived on the rug at 1.26 m/s. Flush, it arrives at 2.0 m/s with all its
  speed intact, and the rug has to do roughly twice the work to stop it in the
  same distance.

## Key parameters

All physics parameters can be overridden via `--scenario-overrides-json`, or
directly in `build_pcve_ball_carpet_climb.py` for suite cases.

- `carpet_rolling_friction`: the parameter this scene is built around, and with
  a flush rug the only thing that changes under the ball at all (default
  `0.060`). Higher values stop it nearer the near border; `0.018` lets it coast
  the full width into the sofa, `0.130` stops it in 0.22 m. Saturates above
  `~0.13`.
- `floor_rolling_friction`: fixed at `0.0015`, independent of the rug's, so the
  approach stays a slick hard floor no matter what the rug is set to. This
  separation is the whole point -- tying them together makes the contrast the
  scene is demonstrating disappear.
- `launch_speed`: initial push along +Y in m/s (default `2.05`).
- `carpet_thickness`: height of the rug's top face above the floor, in metres
  (default `0.0`, i.e. flush). Drives both the collision box and the rendered
  slab. Setting it positive restores the raised rug the scene used to have, and
  the ball then has to climb a real step -- which costs it speed by itself and
  stops friction being the only variable.
- `ball_mass` / `ball_restitution`: default `0.27` kg / `0.45`.
- `gravity_z`: default `-9.8`.

## Outputs

- `ball_carpet_climb.mp4` -- rendered video (animation mode).
- `preview.png` -- preview still (preview mode).
- `ball_carpet_climb.blend` -- saved Blender scene.
- `ground_truth_transforms.json` -- per-frame ball and camera transforms, plus
  per-frame speed and an `on_carpet` flag marking exactly which frames the ball
  is on the rug rather than the bare floor. The `quality` block additionally
  carries `carpet_entry_speed`, `carpet_travel`, and `max_lift_over_carpet`
  (the flatness check, 0.0000 m for a flush rug).
- `scenario_metadata.json` -- seed, camera, and physics parameters.
