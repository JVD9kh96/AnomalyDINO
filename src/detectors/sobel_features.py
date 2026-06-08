from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8

SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


@dataclass
class CalibrationStats:
    ref_mean: float
    ref_std: float
    ref_median: float
    ref_q1: float
    ref_q3: float
    ref_percentile: float


@dataclass
class ScoreModeParams:
    score_mode: str = "raw"
    zscore_k: float = 2.0
    iqr_k: float = 1.5
    percentile: float = 95.0


def tokens_to_feature_map(
    tokens: np.ndarray,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """
    Reshape patch tokens (H*W, D) to a feature map (1, D, H, W).
    """
    grid_h, grid_w = grid_size
    expected = grid_h * grid_w
    if tokens.shape[0] != expected:
        raise ValueError(
            f"Token count {tokens.shape[0]} does not match grid {grid_size} "
            f"(expected {expected})."
        )
    feat = tokens.reshape(grid_h, grid_w, -1).transpose(2, 0, 1)
    return torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)


def _reduce_across_channels(grad_mag: torch.Tensor, norm_reduction: str) -> torch.Tensor:
    if norm_reduction == "l2":
        return torch.linalg.vector_norm(grad_mag, ord=2, dim=1)
    if norm_reduction == "mean":
        return grad_mag.mean(dim=1)
    if norm_reduction == "max":
        return grad_mag.amax(dim=1)
    raise ValueError(
        f"Unknown norm_reduction: {norm_reduction!r}. Choose 'l2', 'mean', or 'max'."
    )


def feature_sobel_norm(
    feat_map: torch.Tensor,
    norm_reduction: str = "l2",
) -> torch.Tensor:
    """
    Apply Sobel filters in feature space and aggregate per spatial location.

    Args:
        feat_map: (B, D, H, W) tensor
        norm_reduction: how to aggregate across D — l2 | mean | max

    Returns:
        (B, H, W) Sobel norm map
    """
    if feat_map.ndim != 4:
        raise ValueError(f"feat_map must be 4D (B,D,H,W), got shape {tuple(feat_map.shape)}")

    device = feat_map.device
    dtype = feat_map.dtype
    _, channels, _, _ = feat_map.shape

    sobel_x = SOBEL_X.to(device=device, dtype=dtype).expand(channels, 1, 3, 3)
    sobel_y = SOBEL_Y.to(device=device, dtype=dtype).expand(channels, 1, 3, 3)

    gx = F.conv2d(feat_map, sobel_x, padding=1, groups=channels)
    gy = F.conv2d(feat_map, sobel_y, padding=1, groups=channels)
    grad_mag = torch.sqrt(gx * gx + gy * gy + EPS)
    return _reduce_across_channels(grad_mag, norm_reduction)


def compute_calibration_stats(norms: np.ndarray) -> CalibrationStats:
    """Compute global calibration statistics from reference patch norms."""
    flat = norms.astype(np.float64).ravel()
    if flat.size == 0:
        raise ValueError("Cannot calibrate from empty reference norms.")
    return CalibrationStats(
        ref_mean=float(np.mean(flat)),
        ref_std=float(np.std(flat)),
        ref_median=float(np.median(flat)),
        ref_q1=float(np.percentile(flat, 25)),
        ref_q3=float(np.percentile(flat, 75)),
        ref_percentile=float(np.percentile(flat, 95)),
    )


def apply_calibration(norms: np.ndarray, calib: CalibrationStats | None) -> np.ndarray:
    """Adjust norms using reference mean/std when calibration is available."""
    if calib is None:
        return norms.astype(np.float32)
    return ((norms - calib.ref_mean) / (calib.ref_std + EPS)).astype(np.float32)


def apply_score_mode(
    norms: np.ndarray,
    params: ScoreModeParams,
    calib: CalibrationStats | None = None,
) -> np.ndarray:
    """
    Convert Sobel norm map to continuous patch anomaly scores.

    When calibration is present, norms are first adjusted via
    (norm - ref_mean) / ref_std before per-image transforms.
    """
    values = apply_calibration(norms, calib)
    mode = params.score_mode

    if mode == "raw":
        return values.astype(np.float32)

    if mode == "per_image_zscore":
        if calib is not None:
            return values.astype(np.float32)
        mean = float(np.mean(values))
        std = float(np.std(values))
        return ((values - mean) / (std + EPS)).astype(np.float32)

    if mode == "per_image_iqr":
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        median = float(np.median(values))
        scores = (values - q3) / (iqr + EPS)
        if params.iqr_k != 1.0:
            scores = scores / params.iqr_k
        _ = median  # median available for extensions; scores anchored at Q3
        return scores.astype(np.float32)

    if mode == "per_image_percentile":
        threshold = float(np.percentile(values, params.percentile))
        scores = np.maximum(values - threshold, 0.0)
        scale = float(np.max(scores))
        if scale > EPS:
            scores = scores / scale
        return scores.astype(np.float32)

    raise ValueError(
        f"Unknown score_mode: {mode!r}. "
        "Choose 'raw', 'per_image_zscore', 'per_image_iqr', or 'per_image_percentile'."
    )


def norms_to_numpy(norms: torch.Tensor) -> np.ndarray:
    """Detach torch norm map (B,H,W) or (H,W) to numpy float32."""
    if norms.ndim == 3:
        norms = norms[0]
    return norms.detach().cpu().numpy().astype(np.float32)
