"""Frechet Video Distance, over a whole prediction set.

FVD is a distribution metric: it compares the set of predicted clips against
the set of reference clips, so it has no per-case value and lives in the
aggregate alone. The feature extractor is the Kinetics-400 I3D TorchScript
module every FVD implementation since StyleGAN-V has used -- a different
backbone gives numbers that cannot be compared with published ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

I3D_PATH = "/remote-home/chenyuanjie/models/i3d/i3d_torchscript.pt"
# What the TorchScript module expects: it does the resize to 224 and the
# [0,255] -> [-1,1] rescale itself when told to.
DETECTOR_KWARGS = dict(rescale=True, resize=True, return_features=True)


class I3DFeatures:
    def __init__(self, path: str = I3D_PATH, device: str = "cuda"):
        self.device = device
        self.model = torch.jit.load(path).eval().to(device)

    @torch.no_grad()
    def __call__(self, video: np.ndarray) -> np.ndarray:
        """(T,H,W,3) uint8 -> (1024,) float64."""
        x = torch.from_numpy(video).to(self.device).float()
        # contiguous(): the module resizes with .view internally, which a
        # permuted tensor cannot satisfy.
        x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1,C,T,H,W)
        feat = self.model(x, **DETECTOR_KWARGS)
        return feat.squeeze(0).double().cpu().numpy()


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root via eigendecomposition.

    scipy.linalg.sqrtm on a 1024x1024 product returns a complex array with
    numerical dust and is markedly slower; the covariance product here is
    symmetric positive semi-definite, so an eigendecomposition is both exact
    and cheap. Negative eigenvalues are numerical noise and clip to zero.
    """
    vals, vecs = np.linalg.eigh(mat)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """FID/FVD formula on two (N, D) feature matrices."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    # sqrtm(A B) computed on the symmetrised A^1/2 B A^1/2, which has the same
    # trace and stays real.
    sa = _sqrtm(cov_a)
    covmean = _sqrtm(sa @ cov_b @ sa)
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b)
                 - 2.0 * np.trace(covmean))
