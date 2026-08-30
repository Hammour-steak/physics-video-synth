# Twin-ramp collision

Two glass marbles are held at the crests of two facing wooden ramps, released at
the same instant, roll down, and meet head-on on the short flat valley between
the ramp toes. They rebound, run part-way back up their ramps, roll down again
and come to rest a short way apart in the valley.

The scene follows the same coupled simulate/render/batch/build pipeline as the
other scenes in `scripts/`:

1. `simulate_twin_ramp_collision.py` runs PyBullet and writes a trajectory JSON.
2. `render_twin_ramp_collision.py` builds the Blender scene, calls the
   simulator, applies the trajectory as keyframes, exports
   `ground_truth_transforms.json`, and renders.
3. `batch_render_twin_ramp_collision.py` orchestrates multiple renders.
4. `build_pcve_twin_ramp_collision.py` builds the named PCVE benchmark suite.
5. `survey_cameras.py` renders the scene from candidate camera positions, for
   choosing a framing. Its output is disposable.

`twin_ramp_geometry.py` holds the trigonometry that places the wedges, and is
imported by both the simulator and the renderer so the physics world and the
Blender scene cannot drift apart.

## The apparatus

A flat base plank with a solid wooden wedge screwed down at each end, sloped
faces towards each other, standing on a table in a daylit room.

| | |
|---|---|
| Ramp | 8&deg;, 0.554 m run, 78 mm rise, 0.24 m wide |
| Valley | 0.18 m of flat between the two toes |
| Crest to crest | 1.288 m |
| Balls | 64 mm glass marbles, 0.343 kg, one amber and one blue, released together |
| Clip | 3.0 s at 24 fps, 1280&times;720 |

Timeline of the default scenario, all measured rather than intended:

| t | what happens |
|---|---|
| 0.00–0.25 s | both balls parked at the crests |
| 0.25 s | released together |
| 1.375 s | last frame before contact, balls 44 mm apart |
| 1.42 s | contact, at x = +2 mm, doing 0.92 m/s each |
| ~1.9 s | each has run 142 mm back up its ramp and stopped |
| ~2.9 s | they roll back down and settle 82 mm apart in the valley |

## Things that are not obvious

**The wedges are wedges for a reason.** A ramp built the way `ramp_collision`
builds its one — a board propped on a block — ends in a board-thickness cliff at
the bottom, and a marble doing 0.9 m/s launches off it instead of running onto the
flat. The sloped face here runs all the way down to the plank's top surface, so
there is no step anywhere on the path. The collider is a plain box tilted so its
top face lies on the wedge's slope, rather than a convex hull of the wedge's own
vertices, because Bullet gives hulls a 40 mm collision margin by default — more
than half this ball's radius.

**The rebound is much weaker than the restitution suggests, and that is
correct.** Each ball arrives rolling, so it carries angular momentum the head-on
impulse does not touch: it rebounds translating backwards while still spinning
forwards, and track friction then has to reverse that spin. For a solid sphere
the settled rebound works out to `(2.5e - 1)/3.5` of the approach speed, where
`e` is the effective ball-on-ball restitution. Note that `e` is *not*
`--ball-restitution`: Bullet multiplies the two bodies' values, so two balls at
0.87 collide at e = 0.76. That predicts 0.255&times;, and the default scenario
measures 0.247 m/s against an approach of 0.920 — 0.268&times;. Around e = 0.4
the prediction reaches zero, and the `soft_balls` case in the PCVE suite (e =
0.16) does barely separate at all before settling against each other.

**The ramp's rise is capped by the frame rate.** Arrival speed is
`sqrt(2*g*rise/1.4)` and the balls close on each other at twice that. The first
cut of this scene (14&deg; over 0.45 m) gave 1.12 m/s — 93 mm of closing per
rendered frame while the balls are only 64 mm across — and the impact fell
entirely between two frames, 72 mm apart in one and already rebounding in the
next. The collision the scene exists to show was never on screen. Going much
past 9&deg; at this run length loses that margin again.

**The ramp's length is set by the clip.** Roll-down time is
`sqrt(2.8*rise/g)/sin(angle)`, so with the rise pinned by the above, the only
way to buy more of it is to go shallower and longer. At 10&deg; over 0.33 m the
balls met 1.04 s in and everything had settled by 2.3 s, leaving the last third
of a 3 s clip dead.

**The impact is sub-frame.** At 0.9 m/s the balls are in contact for well under
a millisecond, so `simulate_twin_ramp_collision.py` looks for contact inside the
substep loop. Sampling only at frame boundaries reports the balls as never
having touched even in runs where they visibly rebound. Motion blur is on for
the same reason: without it the collision reads as two balls teleporting past
each other.

**The room is closed on all six sides, and that is a lighting decision.** Left
open at the top it was still an exterior — every up-facing surface, which here
is both slopes and the whole valley, saw the world HDRI straight overhead and
rendered at 210/255 against an albedo of 120. The ceiling is what makes the
window the actual light source. For the same reason the window's area light
points straight out of the wall rather than at the apparatus: aiming it at the
rig turns a vertical window into an overhead source.

**The balls are one model recoloured, and the recolour is a hue rotation.**
`assets/models/marble_yellow_ball.glb` is a single 0.72-unit sphere carrying a
photographed swirl on a 4K texture. That swirl earns its keep: a featureless
sphere is rotationally invariant on screen, and it is the only thing showing that
these balls arrive *rolling* rather than sliding, which is the whole reason the
rebound is as weak as it is. Colour therefore comes from a Hue/Saturation node on
the texture rather than from overwriting the base colour, which would throw the
swirl away along with the yellow. `balls.a.hue` / `balls.b.hue` live in the
scenario, so a case can recolour without touching code; 0.5 leaves the model as
authored (amber) and 0.12 rotates it to blue. Each ball gets its own copy of the
material — sharing one datablock means recolouring the second recolours the first.

**The table prop is decorative only.** It is not in the physics world and nothing
can reach it. It sits behind the rig and out towards one end, because the camera
shoots the apparatus square on and the valley has to stay clean. Its scale is
measured from the GLB rather than assumed — the models in `assets/models` do not
share a unit, and `import_prop` normalises each one to a real height.

## Camera

The default framing is the `R2` candidate from `survey_cameras.py`: square to
the track at `(0.0, -2.05, 0.40)` on a 48 mm lens. Shooting square on is what
makes the two ramps read as a matched pair and puts the meeting point dead
centre. The standoff is not a taste decision — the rig is 1.288 m crest to
crest, which a 48 mm lens on a 36 mm sensor only covers from about 1.8 m, and a
wide lens up close grows the near ramp so the two stop matching.

To compare framings again after changing the geometry:

```bash
python3 scripts/twin_ramp_collision/survey_cameras.py \
  --out-root renders/trc_camera_survey
```

## Single render

```bash
./tools/blender-3.6.23-linux-x64/blender -b \
  --python scripts/twin_ramp_collision/render_twin_ramp_collision.py -- \
  --mode animation \
  --out-dir renders/twin_ramp_collision \
  --resolution 1280 720 \
  --fps 24 \
  --duration-sec 3.0 \
  --samples 128 \
  --device auto
```

Writes `twin_ramp_collision.mp4`, `twin_ramp_collision.blend`,
`ground_truth_transforms.json` and `scenario_metadata.json` into the output
directory. `--mode preview` renders a single still instead; add
`--preview-frames 4 34 35` to get several.

## Simulate only

```bash
python3 scripts/twin_ramp_collision/simulate_twin_ramp_collision.py \
  --out /tmp/twin_ramp_physics.json --fps 24 --duration-sec 3.0
```

Prints where and when the balls met, and warns if they never touched, met on a
ramp instead of in the valley, or left the plank.

## Batch render

```bash
python3 scripts/twin_ramp_collision/batch_render_twin_ramp_collision.py \
  --out-root renders/batch_twin_ramp_collision \
  --count 4 --mode preview --resolution 960 540 --samples 128 --device auto
```

## PCVE suite

Five cases on one axis: the approach is held constant at 0.92 m/s and only the
collision changes — matched, soft balls, lively, uneven release — plus a
dusty-track distractor that slows the approach instead and ends up looking like
the soft-ball case for an unrelated reason.

```bash
python3 scripts/twin_ramp_collision/build_pcve_twin_ramp_collision.py \
  --out-root renders/pcve_twin_ramp_collision_suite \
  --resolution 1280 720 --fps 24 --duration-sec 3.0 --samples 128 --device auto
```

Each case lands in `cases/<case_id>/` with `video.mp4`,
`ground_truth_transforms.json`, `scenario_metadata.json` and
`scenario_overrides.json`; the suite root gets `suite_manifest.json` recording
what each case actually did, not only what it was asked for.

Suggested segmentation prompt: `marble.ramp.table`.
