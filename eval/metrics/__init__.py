"""Metrics that score a prediction against the ground-truth edited video.

Two families:
- perceptual: pixel/feature-level similarity between prediction and target
  edited video (PSNR / SSIM / LPIPS / CLIP frame similarity).
- physics:  compare the prediction's object motion against
  ground_truth_transforms.json. NOT trivial -- requires detecting the same
  objects in the generated video (SAM2 / CoTracker) and projecting the
  ground truth into camera space. See physics.py for the plan.
"""

from .perceptual import PerceptualMetrics, load_video_frames  # noqa: F401
