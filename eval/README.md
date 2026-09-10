# PCVE eval harness

Runs a baseline over the benchmark, then scores its outputs against the
ground-truth edited videos.

## Layout

```
eval/
├── run_baseline.py         # (source, prompt) -> prediction.mp4
├── compute_metrics.py      # prediction.mp4 vs edited_video.mp4 -> metrics
├── baselines/
│   ├── base.py             # BaselineModel interface
│   ├── stub.py             # copies source verbatim (sanity check)
│   └── wan_vace.py         # Wan 2.1 VACE-14B via diffusers
├── metrics/
│   ├── perceptual.py            # PSNR / SSIM / LPIPS / CLIP frame similarity
│   ├── traj_lib.py              # GT normalising, projection, per-object trajectory error
│   ├── grounded_sam2_tracker.py # GroundingDINO (text -> box) + SAM2 propagator
│   ├── physics.py               # per-object trajectory error vs the edited render
│   ├── text_prompts/            # per-scene {gt_object_name: english_phrase}
│   └── gt_tracks_gsam2/         # cached GroundedSAM2 tracks (source + edited)
└── requirements.txt
```

## Quick start

```bash
cd eval
pip install -r requirements.txt        # metrics deps
# extras for wan_vace (installs matching your CUDA):
pip install "torch>=2.4" "diffusers>=0.33" "transformers>=4.44" accelerate safetensors

BENCH=/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1

# 1) smoke test with the stub (no GPU needed)
python run_baseline.py --benchmark-root $BENCH --baseline stub --limit 3
python compute_metrics.py --benchmark-root $BENCH --baseline stub \
    --no-lpips --no-clip --limit 3   # skip heavy metrics for the smoke run

# 2) real run on the whole benchmark
python run_baseline.py --benchmark-root $BENCH --baseline wan_vace_14b \
    --prompt-flavor quantitative --prompt-lang en \
    --skip-existing
python compute_metrics.py --benchmark-root $BENCH --baseline wan_vace_14b
```

`--skip-existing` picks up after a crash / partial run without redoing
finished cases. Outputs are collected under
`{benchmark_root}/predictions/{baseline}/`:

```
predictions/wan_vace_14b/
├── run_manifest.json       # per-case status + elapsed + config
├── videos/{scene}/{case_id}.mp4
├── metrics.json            # per-case + aggregate (overall / by property / by scene / by kind)
└── metrics.csv             # flat table for pandas / paper tables
```

## Adding a new baseline

Copy `baselines/stub.py`, implement `edit_video`, add an entry to
`BASELINES` in `run_baseline.py`. `setup()` / `teardown()` are called
once per run; heavy model loads should go there so `--baseline stub`
never pays the diffusers import cost.

## Filters

Every filter below applies to `benchmark_manifest.json["edits"]` in the
order listed and can be combined:

```
--filter-scene ramp_collision           # one scene only
--filter-property mass                   # mass / friction / restitution / initial_velocity / presence
--filter-kind SET                        # SET, DELETE or ADD
--limit 5                                # cap after all filters
```

## Prompt selection

Every edit ships four prompt flavors under `prompts`:

- `quantitative.en` -- includes numerical from/to. Default for benchmark
  scoring (unambiguous).
- `quantitative.zh` -- same, Chinese.
- `vague.en`, `vague.zh` -- direction only ("Increase the ball's mass").
  Use these for open-ended eval where you want to test *whether* the model
  grasps the physical intent without exact numbers.

Pick with `--prompt-flavor quantitative --prompt-lang en`.

## Notes on Wan 2.1 VACE-14B

- Model card: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- 48 GB (A6000) is enough for 720p bf16 without quantization. If OOM,
  add `--enable-cpu-offload` or drop to `--height 480 --width 720`.
- Default 50 inference steps, guidance 5.0. Turn steps down to 25 for a
  faster smoke run.
- The exact WanVACEPipeline API in `baselines/wan_vace.py` matches the
  diffusers integration as of Wan 2.1 release notes; if you're on a
  bleeding-edge diffusers version the class name or kwargs may have
  shifted. Adjust in one place -- `WanVACEBaseline.edit_video`.

## Physics metrics

### Trajectory error is per object, not per scene

An edit names one object, but what it produces is a chain reaction: making
`mallet_1` heavier changes where `mallet_0` and `mallet_2` end up. Scoring one
object per case cannot separate "the model understood the edit" from "the model
moved the right ball and froze everything else", so every object in the scene is
tracked and scored separately. Across the 20 scenes that is 42 objects rather
than 20.

`metrics/traj_lib.py` provides the pieces:

- `normalise(gt)` -- the three GT layouts (nested dict / prefix-flat / list) to
  one `{name: {mw, radius}}` table, with removed objects blanked out.
- `resolve_targets(base_gt, edit_gt, object_id)` -- every object in the case,
  tagged `edited` / `affected` / `static`, each with its seed frame, apparent
  radius, and source-vs-edited divergence. Roles come from what the two GT files
  say changed (presence and sim parameters), not from the edit's `object_id`,
  because the names disagree: `domino_1 .. domino_4` in the edits are
  `domino_000 .. domino_003` in the render.
- `added_objects(objs, eobjs)` / `removed_objects(objs, eobjs)` -- the two
  presence diffs. An ADD edit's object is in the edited render and not the
  source, so it is invisible to any pass that walks the source's object list;
  `resolve_targets` walks both.
- `edit_seed(base_gt, edit_gt, obj, eobjs)` -- seed frame for an ADDed object,
  read off the edited render because there is no source trajectory to read.
  The camera still comes from the source clip, so the seed is in the same
  pixel frame as every other target's.
- `placement_error(pred_uv, gt_uv, anchor_uv, other_uv, radius_px)` -- how far
  a predicted placement is from where the instruction asked, split into the
  component along the two-object line and the component across it.
- `traj_error(track_uv, ref_uv, ref_pres, seed, radius_px, resolution)` -- one
  object's error, as `abs_*` and `disp_*` (measured from the anchor frame, the
  number to report), in pixels and in object radii. Frames where the object is
  not fully inside the image are dropped and counted (`n_offscreen`,
  `n_clipped`) rather than scored: a mask cut off by the border has its centroid
  pulled inward while the ground-truth origin is not, so those frames measure
  the crop.

### DELETE and ADD

A DELETE edit leaves no reference trajectory -- the object is gone -- so what
is scored is presence: the prediction's mask against the edited video's own
response to the same prompt (`removal_area_ratio`).

An ADD edit is the mirror, and gives more to measure, because the object the
instruction asks for is really there in the edited render. It is scored as an
ordinary target (full trajectory against the reference) *and* gets a placement
score, since an ADD prompt does not ask for a trajectory -- it asks for a
position:

| field | meaning |
| --- | --- |
| `add_detected` | did the prediction put anything there at all |
| `add_area_ratio` | its mask size against the reference's |
| `add_radii_asked` / `add_radii_from` | what the instruction said |
| `add_err_radii_along` | placement error along the two-object line |
| `add_err_radii` | total placement error |

"Nothing placed" and "placed in the wrong spot" are different failures, so
`add_detected` is reported separately rather than folded into the error.

Placement is measured at the seed frame: it is an initial condition, and by
later frames the added object has been struck and moved.

**The radii are apparent (pixel) radii**, the same scale the rest of these
metrics use -- not the scene-space radius the DSL was written in. Perspective
foreshortens the object and the line together, so the along-line figure is
close to the scene-space error at similar depth, but it is not literally the
instruction's number: pool's line is 20.8 radii in the scene and 15.4 on
screen. Compare predictions against each other on one case; do not read it as
the DSL quantity.

### Both sides are tracked, not just the prediction

The reference is the EDITED render, and SAM2 runs on it too, from the same seed
as the prediction. Comparing a tracked centroid against the projected ground
truth instead would charge every rotating object for a bias the model has no
part in: the GT stores an object's ORIGIN, a mask gives its CENTROID, and for a
toppling domino or bowling pin the gap between them swings by 8-18 px with the
tracker working perfectly. Centroid against centroid, it cancels.

The projection still does the work a centroid cannot:

- **seeding** -- the projected source position, identical for every case in a
  scene, so nothing about the edited answer reaches the tracker
- **the in-frame gate** -- a tracker that has lost an object still returns a
  centroid, so only the projection can say the object left the shot
- **scale** -- the apparent radius, so errors are readable in radii
- **`abs_*_vs_projection`** -- absolute placement, which centroid-vs-centroid
  cancels along with the bias, and which catches correct motion in the wrong
  place

Tracks of the source and edited videos do not depend on the prediction, so they
are cached under `metrics/gt_tracks_gsam2/` and every baseline after the first
reuses them.

### Reading the numbers: `gap_closed`

A raw pixel error is uninterpretable on its own -- the edits in this benchmark
are worth anywhere from 15 to 5900 px, so 60 px is a near miss in one scene and
a total failure in another. Every object therefore also gets `null_disp_px`:
what a model scores by ignoring the prompt and reproducing the source clip,
measured the same way (source track vs edited track).

```
gap_closed = 1 - sum(disp) / sum(null)      1.0 perfect, 0.0 did nothing
```

Summed rather than averaged per object: an object the edit barely moves has a
null of a fraction of a pixel, and its ratio swings to -3 on tracking noise.

### Text prompts (GroundingDINO seeding)

The tracker is `metrics/grounded_sam2_tracker.py`: GroundingDINO
(`IDEA-Research/grounding-dino-tiny`, downloaded on first use to the HF cache)
runs on each object's first-visible frame in the source clip, and the returned
box is what SAM2 propagates from. Nothing depends on the exact first-frame
pixel coordinate of every object -- the projected origin is used only as a
positional hint when two objects share a text.

One English phrase per GT object lives in `metrics/text_prompts/{scene}.json`:

```json
{
  "scene": "air_hockey_chain",
  "objects": {
    "mallet_0": "blue air hockey mallet",
    "mallet_1": "red air hockey mallet",
    "mallet_2": "white air hockey mallet"
  }
}
```

Two rules for the phrases:

- **Visually distinct objects** get a unique phrase (colour, material, brand);
  GDINO returns the highest-confidence box and that goes to SAM2.
- **Visually identical groups** (bowling's three pins, `domino_chain`'s four
  tiles, `table_drop_collision`'s two tennis balls) all get the *same* phrase.
  One detection call returns N boxes, and each box is assigned to a GT identity
  by nearest-neighbour against the projected first-frame origin -- edit-
  invariant and identical for every case in a scene, so nothing about the
  edited answer reaches the tracker. Boxes that land more than two object
  diameters from every hint are dropped rather than stolen from a neighbour.

Add a new scene by dropping a file into `metrics/text_prompts/`; edits that
name an object with no entry in that file are reported `no_text_prompt` rather
than scored against silence.

### Reference track quality

Each object's reference track (GroundedSAM2 on the edited render) is graded
relative to the object's apparent size: `GOOD` < 0.5 radius, `OK` < 1 radius,
else `UNUSABLE`. This verdict is recorded in `ref_verdict` in the per-object
output but does not gate scoring -- all objects are scored regardless of
reference track quality.
