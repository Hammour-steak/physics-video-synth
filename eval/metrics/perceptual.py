"""Frame-level perceptual metrics: PSNR, SSIM, LPIPS, CLIP-similarity.

Every metric is computed per-frame against the ground-truth edited video,
then averaged across frames. Frame counts are aligned by resampling the
prediction to the ground-truth length via nearest-neighbour indexing
(the ground-truth frame count is authoritative).

Deps: torch, torchmetrics, lpips, open_clip_torch, av (or imageio).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def load_video_frames(path: Path, height: int | None = None,
                      width: int | None = None) -> np.ndarray:
    """Return a uint8 (T, H, W, 3) array. Resizes if height/width given.

    Uses imageio with the ffmpeg plugin; the diffusers env already pulls
    in imageio-ffmpeg.
    """
    import imageio.v3 as iio
    frames = iio.imread(str(path), plugin="pyav")  # (T, H, W, 3) uint8
    if height is not None and width is not None and (
            frames.shape[1] != height or frames.shape[2] != width):
        from PIL import Image
        out = np.empty((frames.shape[0], height, width, 3), dtype=np.uint8)
        for i, fr in enumerate(frames):
            out[i] = np.asarray(Image.fromarray(fr).resize((width, height),
                                                            Image.BILINEAR))
        frames = out
    return frames


def align_lengths(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both to the frames the prediction actually covers.

    Every adapter writes its result at the source clip's own rate, so frame i
    of a prediction is frame i of the source -- a shorter prediction is one
    that stopped early, not one that compressed the clip. ditto is the case
    that matters: its LoRA is trained at 73 frames, so it reads the first 73 of
    the 96 and generates those.

    This used to resample the prediction up to the reference's length
    (`np.linspace(0, tp - 1, tg)`), which stretches those 73 frames across 4
    seconds and lines ditto's frame 72 up against ground-truth frame 95. The
    error that produces grows with time and has nothing to do with edit
    quality. metrics/physics.py has always truncated (`k = min(len(p), len(g))`);
    this makes the two agree.

    The frames the prediction never produced are simply not scored, so a short
    prediction is credited with covering less -- `num_frames` on the result
    records how much was compared, and the coverage difference belongs in the
    experimental setup rather than hidden inside a resample.
    """
    k = min(pred.shape[0], gt.shape[0])
    return pred[:k], gt[:k]


@dataclass
class PerceptualScores:
    psnr:     float | None
    ssim:     float | None
    lpips:    float | None
    clip_sim: float | None
    num_frames: int


class PerceptualMetrics:
    """Loads models once, scores many (pred, gt) pairs.

    Enable only what you need -- LPIPS + CLIP each hold ~1 GB of weights.
    """

    def __init__(
        self,
        device: str = "cuda",
        enable_psnr: bool = True,
        enable_ssim: bool = True,
        enable_lpips: bool = True,
        enable_clip: bool = True,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.enable_psnr = enable_psnr
        self.enable_ssim = enable_ssim
        self.enable_lpips = enable_lpips
        self.enable_clip = enable_clip

        if enable_ssim or enable_psnr:
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device) if enable_psnr else None
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) if enable_ssim else None
        if enable_lpips:
            import lpips  # type: ignore
            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        if enable_clip:
            import open_clip  # type: ignore
            model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained, device=self.device,
            )
            self._clip = model.eval()

    # ------------------------------------------------------------------ eval

    def score(self, pred_video: Path, gt_video: Path) -> PerceptualScores:
        pred = load_video_frames(pred_video)
        gt = load_video_frames(gt_video)
        # Resize pred to gt spatial shape if needed.
        if pred.shape[1:3] != gt.shape[1:3]:
            pred = load_video_frames(pred_video, height=gt.shape[1], width=gt.shape[2])
        pred, gt = align_lengths(pred, gt)

        psnr = self._mean_frame(self._score_psnr, pred, gt) if self.enable_psnr else None
        ssim = self._mean_frame(self._score_ssim, pred, gt) if self.enable_ssim else None
        lpip = self._mean_frame(self._score_lpips, pred, gt) if self.enable_lpips else None
        clip = self._score_clip(pred, gt) if self.enable_clip else None

        return PerceptualScores(psnr=psnr, ssim=ssim, lpips=lpip,
                                clip_sim=clip, num_frames=int(pred.shape[0]))

    # -------------------------------------------------------------- helpers

    def _to_tensor(self, frames: np.ndarray):
        """(T, H, W, 3) uint8 -> (T, 3, H, W) float32 [0, 1] on device."""
        import torch
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return x

    def _mean_frame(self, fn, pred: np.ndarray, gt: np.ndarray) -> float:
        import torch
        pt = self._to_tensor(pred)
        gt_t = self._to_tensor(gt)
        with torch.no_grad():
            values = fn(pt, gt_t)
        return float(values.mean().item())

    def _score_psnr(self, pred, gt):
        import torch
        return torch.stack([self._psnr(pred[i:i+1], gt[i:i+1]) for i in range(pred.shape[0])])

    def _score_ssim(self, pred, gt):
        import torch
        return torch.stack([self._ssim(pred[i:i+1], gt[i:i+1]) for i in range(pred.shape[0])])

    def _score_lpips(self, pred, gt):
        # LPIPS wants inputs in [-1, 1].
        pred_ = pred * 2.0 - 1.0
        gt_   = gt   * 2.0 - 1.0
        # Batched call is fine at 720p on 48 GB; drop to 4-frame chunks
        # if VRAM is tight.
        return self._lpips(pred_, gt_).flatten()

    def _score_clip(self, pred: np.ndarray, gt: np.ndarray) -> float:
        import torch
        from PIL import Image
        pred_feats, gt_feats = [], []
        for i in range(pred.shape[0]):
            pi = self._clip_preprocess(Image.fromarray(pred[i])).unsqueeze(0).to(self.device)
            gi = self._clip_preprocess(Image.fromarray(gt[i])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pf = self._clip.encode_image(pi)
                gf = self._clip.encode_image(gi)
            pred_feats.append(pf / pf.norm(dim=-1, keepdim=True))
            gt_feats.append(gf / gf.norm(dim=-1, keepdim=True))
        p = torch.cat(pred_feats); g = torch.cat(gt_feats)
        cos = (p * g).sum(dim=-1)
        return float(cos.mean().item())
