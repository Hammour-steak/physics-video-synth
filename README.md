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
| Physics | Per-object trajectory error (px & radii), `gap_closed`, removal accuracy, onset error, placement error (ADD) |

`gap_closed` is the primary physics metric: 1.0 = perfect match to the
edited render, 0.0 = indistinguishable from replaying the unedited source.

See [eval/README.md](eval/README.md) for detailed metric descriptions.

## License

Code in this repository is released under the MIT License.
The benchmark data on Hugging Face follows its own license terms.
