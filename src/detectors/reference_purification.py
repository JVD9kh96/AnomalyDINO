"""Automatic and oracle reference-patch purification for memory banks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from src.detectors.knn_index import knn_distances, pairwise_knn_distances
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import build_gt_patch_labels


@dataclass
class PurificationResult:
    keep_mask: np.ndarray
    scores: np.ndarray
    threshold: float
    n_total: int
    n_kept: int
    kept_fraction: float


def purify_reference_grid(
    features: np.ndarray,
    clean_index,
    threshold: float,
    *,
    knn_metric: str,
    k_neighbors: int = 1,
) -> PurificationResult:
    """
    Retain patches whose distance to the clean bank is <= threshold.

    Higher distance => less normal => more likely excluded.
    Does not read GT masks.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got {features.shape}")

    scores = knn_distances(features, clean_index, knn_metric, k_neighbors)
    keep_mask = scores <= float(threshold)
    n_total = int(features.shape[0])
    n_kept = int(keep_mask.sum())
    return PurificationResult(
        keep_mask=keep_mask,
        scores=scores,
        threshold=float(threshold),
        n_total=n_total,
        n_kept=n_kept,
        kept_fraction=float(n_kept / n_total) if n_total else 0.0,
    )


def purify_reference_features(
    features: np.ndarray,
    clean_bank_features: np.ndarray,
    threshold: float,
    *,
    knn_metric: str,
    k_neighbors: int = 1,
) -> PurificationResult:
    """Convenience wrapper that builds a temporary clean FAISS index."""
    from src.detectors.knn_index import build_faiss_index

    index = build_faiss_index(clean_bank_features, knn_metric, faiss_on_cpu=True)
    return purify_reference_grid(
        features, index, threshold, knn_metric=knn_metric, k_neighbors=k_neighbors
    )


def apply_spatial_cleanup(
    keep_mask: np.ndarray,
    grid_size: tuple[int, int],
    *,
    min_rejected_component_patches: int = 2,
) -> np.ndarray:
    """
    Flip tiny rejected connected components back to keep.

    Reduces isolated false rejections. Disabled by default in configs.
    """
    keep = np.asarray(keep_mask, dtype=bool).reshape(grid_size)
    rejected = ~keep
    if not rejected.any():
        return keep.ravel()

    labeled, n_components = ndimage.label(rejected)
    cleaned = keep.copy()
    for comp_id in range(1, n_components + 1):
        component = labeled == comp_id
        if int(component.sum()) < min_rejected_component_patches:
            cleaned[component] = True
    return cleaned.ravel()


def oracle_keep_mask_from_gt(
    sample: SeverstalSample,
    grid_size: tuple[int, int],
    patch_size: int,
    resolution: int,
    overlap_threshold: float = 0.5,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Keep patches that do NOT overlap GT defects (oracle upper bound).

    Returns a flat bool mask of length grid_h * grid_w.
    """
    native_shape = sample.image.shape[:2]
    gt_labels = build_gt_patch_labels(
        sample.masks_by_class,
        native_shape,
        resolution,
        patch_size,
        overlap_threshold,
        num_classes=num_classes,
    )
    defect = gt_labels["agnostic"].astype(bool)
    if defect.shape != grid_size:
        raise ValueError(
            f"GT patch grid {defect.shape} does not match feature grid {grid_size}"
        )
    return (~defect).ravel()


def mine_suspected_defect_mask(
    features: np.ndarray,
    clean_bank_features: np.ndarray,
    defect_percentile: float,
    *,
    knn_metric: str,
    k_neighbors: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Mark patches with distance_to_clean > defect_percentile of those distances.

    Returns (suspected_mask, distances, threshold).
    """
    distances = pairwise_knn_distances(
        features, clean_bank_features, knn_metric, k_neighbors, faiss_on_cpu=True
    )
    threshold = float(np.percentile(distances, defect_percentile))
    suspected = distances > threshold
    return suspected, distances, threshold


def dual_bank_scores(
    query_features: np.ndarray,
    normal_bank: np.ndarray,
    defect_bank: np.ndarray,
    *,
    knn_metric: str,
    k_neighbors: int = 1,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    score = d_normal - alpha * d_defect.

    Higher scores indicate stronger evidence of a defect.
    If defect_bank is empty, falls back to d_normal only.
    """
    d_normal = pairwise_knn_distances(
        query_features, normal_bank, knn_metric, k_neighbors, faiss_on_cpu=True
    )
    if defect_bank is None or len(defect_bank) == 0:
        return d_normal
    d_defect = pairwise_knn_distances(
        query_features, defect_bank, knn_metric, k_neighbors, faiss_on_cpu=True
    )
    return (d_normal - float(alpha) * d_defect).astype(np.float32)
