# PCVE-RigidBench

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PCVE--RigidBench-blue)](https://huggingface.co/datasets/ccmoony/PCVE-RigidBench)

A benchmark for evaluating physics-aware video editing models on rigid-body
interactions. The repository contains two parts:

1. **Benchmark construction** (`scripts/`) -- PyBullet simulation + Blender
   Cycles rendering to produce deterministic physical-interaction videos and
   matching ground-truth trajectories.
2. **Evaluation harness** (`eval/`) -- run a baseline model over the benchmark,
   then score its outputs with perceptual metrics (PSNR, SSIM, LPIPS, CLIP,
   FVD) and physics-grounded trajectory error.

The benchmark data (videos, ground-truth trajectories, edit manifests) is
hosted on Hugging Face:
[ccmoony/PCVE-RigidBench](https://huggingface.co/datasets/ccmoony/PCVE-RigidBench).

## Repository Layout

```
├── scripts/                    # Benchmark construction
│   ├── build_benchmark.py      # Build the full benchmark manifest
│   ├── pcve_edit_dsl.py        # Edit DSL: property changes, ADD/DELETE
│   ├── pcve_timed_edits.py     # Timed edits (AT FRAME n)
│   ├── download_render_assets.py
│   ├── <scene_name>/           # Per-scene simulation + render scripts
│   └── ...
├── eval/                       # Evaluation harness
│   ├── run_baseline.py         # (source, prompt) -> prediction.mp4
│   ├── compute_metrics.py      # prediction vs ground-truth -> metrics
│   ├── baselines/              # Baseline model wrappers
│   │   ├── wan_vace.py         #   Wan 2.1 VACE-14B
│   │   ├── ditto.py            #   Ditto
│   │   ├── void.py             #   VOID
│   │   └── stub.py             #   Copy-source sanity check
│   ├── metrics/                # Metric implementations
│   │   ├── perceptual.py       #   PSNR / SSIM / LPIPS / CLIP
│   │   ├── fvd.py              #   Fréchet Video Distance
│   │   ├── physics.py          #   Per-object trajectory error
│   │   ├── traj_lib.py         #   GT normalisation, projection, error
│   │   └── grounded_sam2_tracker.py  # GroundingDINO + SAM2 tracking
│   └── requirements.txt
├── videos/                     # Source + edited render outputs
└── requirements.txt            # Benchmark construction deps
```

## Benchmark Construction

The benchmark covers 20 rigid-body scenes (ball impact, drop, rebound,
incline, domino chain, bowling, curling, pool collision, etc.) with 129 edit
cases across five physical properties: **mass**, **friction**, **restitution**,
**initial velocity**, and **presence** (ADD/DELETE).

### Prerequisites

- Python 3.10+
- Blender 3.6 LTS (with Cycles)
- PyBullet

```bash
pip install -r requirements.txt
python scripts/download_render_assets.py
```

### Build the benchmark

```bash
python scripts/build_benchmark.py \
  --out-root /path/to/pcve_benchmark_v1 \
  --resolution 1280 720 --fps 24 --duration-sec 8 \
  --samples 32 --device auto
```

## Evaluation

### Setup

```bash
cd eval
pip install -r requirements.txt
```

For model-specific dependencies (e.g. Wan VACE, Ditto, VOID), see
[eval/README.md](eval/README.md).

### Run a baseline

```bash
BENCH=/path/to/pcve_benchmark_v1

# Generate predictions
python run_baseline.py --benchmark-root $BENCH \
    --baseline wan_vace_14b \
    --prompt-flavor quantitative --prompt-lang en \
    --skip-existing

# Score predictions
python compute_metrics.py --benchmark-root $BENCH --baseline wan_vace_14b
```

Outputs are written to `{benchmark_root}/predictions/{baseline}/`:

```
predictions/wan_vace_14b/
├── videos/{scene}/{case_id}.mp4
├── metrics.json          # per-case + aggregate
├── metrics.csv           # flat table for pandas
└── metrics_objects.csv   # per-object breakdown
```

### Metrics

| Category | Metrics |
|----------|---------|
| Perceptual | PSNR, SSIM, LPIPS, CLIP similarity, FVD |
| Physics | Trajectory displacement (`disp`), Gap Closed (`gap_closed`), Mask IoU (`mask_iou`) |

#### Trajectory Displacement

Every object in each scene is tracked independently by GroundedSAM2 in both
the prediction and the edited ground-truth video. The per-object trajectory
error is computed as displacement from an anchor frame, which cancels the
constant offset between a mask centroid and the object's true origin:

```
disp(t) = ‖(pred(t) - pred(anchor)) - (ref(t) - ref(anchor))‖
```

where `pred(t)` and `ref(t)` are the tracked centroid positions in pixels at
frame `t`. The anchor is the first frame the object is fully inside the image
after its seed frame. Errors are reported in both pixels (`disp_mean_px`) and
object radii (`disp_mean_radii`), where the radius is the object's apparent
size on screen, so a 16 px marble and an 84 px ball are compared on equal
terms.

Frames where the reference object leaves the image are excluded. Frames where
the prediction's tracker lost the object but the reference is still visible are
penalised with the reference's distance to the nearest image edge (a lower
bound on how far off-screen the model must have driven the object).

#### Gap Closed

Raw pixel error is uninterpretable on its own because edits in this benchmark
range from 15 px to 5900 px of displacement. `gap_closed` normalises against
a null baseline -- the error a model would get by ignoring the edit prompt and
reproducing the source clip unchanged:

```
gap_closed = 1 - Σ disp(pred) / Σ disp(null)
```

where `disp(null)` is the trajectory error of the source clip's tracked path
against the edited ground-truth's tracked path, measured identically. The
summation is over all scored objects in a case (not averaged per object then
combined, which would let a barely-moved bystander with a near-zero denominator
dominate the score).

- **1.0** = the prediction perfectly matches the edited ground-truth trajectory.
- **0.0** = the prediction is indistinguishable from replaying the unedited
  source -- the model ignored the edit entirely.
- **< 0** = the model made the trajectory worse than doing nothing.

#### Mask IoU

Per-frame spatial IoU between the prediction's and the reference's segmentation
masks, averaged over scored frames:

```
IoU(t) = |pred_mask(t) ∩ ref_mask(t)| / |pred_mask(t) ∪ ref_mask(t)|
mask_iou = mean(IoU(t)) over scored frames
```

Both sides are gated by a plausibility check on mask area (0.3x-3.0x the
source clip's median area for that object) to reject spurious background masks.
Frames where both sides show nothing are excluded rather than counted as 1.0.
Mask IoU captures shape and spatial overlap that centroid-based trajectory
metrics cannot -- a correct centroid with the wrong object boundary still
scores poorly.

See [eval/README.md](eval/README.md) for more details on tracking, seeding,
and per-object scoring.

## License

Code in this repository is released under the MIT License.
The benchmark data on Hugging Face follows its own license terms.
