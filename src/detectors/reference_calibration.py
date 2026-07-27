"""Leave-one-clean-reference-out normal distance calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.knn_index import pairwise_knn_distances


@dataclass
class NormalDistanceCalibration:
    scores: np.ndarray
    percentiles: dict[float, float]
    mean: float
    std: float
    median: float
    iqr: float

    def threshold_at(self, percentile: float) -> float:
        if percentile in self.percentiles:
            return float(self.percentiles[percentile])
        # Allow float key mismatches like 99 vs 99.0
        for key, value in self.percentiles.items():
            if abs(float(key) - float(percentile)) < 1e-9:
                return float(value)
        return float(np.percentile(self.scores, percentile))


def _summary_stats(scores: np.ndarray, percentile_list: list[float]) -> NormalDistanceCalibration:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("Cannot calibrate from empty score set.")
    q25, q75 = np.percentile(scores, [25, 75])
    percentiles = {float(p): float(np.percentile(scores, p)) for p in percentile_list}
    return NormalDistanceCalibration(
        scores=scores.astype(np.float32),
        percentiles=percentiles,
        mean=float(scores.mean()),
        std=float(scores.std()),
        median=float(np.median(scores)),
        iqr=float(q75 - q25),
    )


def _leave_one_patch_out_scores(
    features: np.ndarray,
    knn_metric: str,
    k_neighbors: int,
) -> np.ndarray:
    """Score each patch against all other patches from the same image."""
    n = features.shape[0]
    if n < 2:
        # Degenerate: self-distance is zero; treat as a single normal score.
        return np.zeros((n,), dtype=np.float32)

    scores = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        bank = np.concatenate([features[:i], features[i + 1 :]], axis=0)
        scores[i] = pairwise_knn_distances(
            features[i : i + 1],
            bank,
            knn_metric,
            k_neighbors,
            faiss_on_cpu=True,
        )[0]
    return scores


def calibrate_normal_distances(
    clean_grids: list,
    knn_metric: str = "L2_normalized",
    k_neighbors: int = 1,
    percentiles: tuple[float, ...] = (95.0, 97.5, 99.0, 99.5),
) -> NormalDistanceCalibration:
    """
    Calibrate normal patch distances from clean reference feature grids only.

    - len(clean_grids) >= 2: leave-one-image-out
    - len(clean_grids) == 1: leave-one-patch-out within that image

    Does not use validation images or defect masks.
    """
    if not clean_grids:
        raise ValueError("clean_grids must contain at least one ReferenceFeatureGrid")

    percentile_list = [float(p) for p in percentiles]
    held_out_scores: list[np.ndarray] = []

    if len(clean_grids) == 1:
        grid = clean_grids[0]
        feats = _active_features(grid)
        held_out_scores.append(
            _leave_one_patch_out_scores(feats, knn_metric, k_neighbors)
        )
    else:
        for hold_idx, hold_grid in enumerate(clean_grids):
            bank_parts = [
                _active_features(g)
                for i, g in enumerate(clean_grids)
                if i != hold_idx
            ]
            bank = np.concatenate(bank_parts, axis=0)
            query = _active_features(hold_grid)
            if query.shape[0] == 0:
                continue
            scores = pairwise_knn_distances(
                query, bank, knn_metric, k_neighbors, faiss_on_cpu=True
            )
            held_out_scores.append(scores)

    if not held_out_scores:
        raise ValueError("No held-out normal distances could be computed.")

    return _summary_stats(np.concatenate(held_out_scores), percentile_list)


def _active_features(grid) -> np.ndarray:
    feats = np.asarray(grid.features, dtype=np.float32)
    if getattr(grid, "patch_keep_mask", None) is not None:
        mask = np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
        if mask.shape[0] != feats.shape[0]:
            raise ValueError("patch_keep_mask length must match feature rows")
        return feats[mask]
    return feats
