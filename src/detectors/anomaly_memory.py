"""GT-guided anomaly feature grids and selection policies (Phases 7–8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.detectors.anomaly_dino import ReferenceFeatureGrid, greedy_coreset_absolute
from src.detectors.reference_purification_metrics import (
    compute_candidate_patch_overlaps,
    oracle_rejection_mask,
)
from src.severstal.dataset import SeverstalSample


OverlapRule = Literal["any_overlap", "at_least_10_percent", "at_least_50_percent"]


@dataclass
class AnomalyFeatureGrid:
    image_id: str
    features: np.ndarray
    grid_size: tuple[int, int]
    overlap_by_class: np.ndarray  # H x W x 4
    d_normal: np.ndarray  # H x W
    source_classes: tuple[int, ...]


@dataclass
class AnomalySelectionResult:
    class_id: int
    selected_indices: np.ndarray  # absolute flat indices into concatenated pool
    n_images: int
    n_patches: int
    patches_per_image: dict[str, int] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def require_gt_anomaly_memory(allow_gt_anomaly_memory: bool) -> None:
    if not allow_gt_anomaly_memory:
        raise RuntimeError(
            "GT anomaly memory is disabled. Set allow_gt_anomaly_memory=true "
            "to construct AnomalyFeatureGrid banks (fail-closed)."
        )


def build_anomaly_feature_grid(
    sample: SeverstalSample,
    feature_grid: ReferenceFeatureGrid,
    *,
    d_normal: np.ndarray,
    allow_gt_anomaly_memory: bool,
    resolution: int,
    patch_size: int,
    num_classes: int = 4,
    held_out_classes: set[int] | frozenset[int] | None = None,
) -> AnomalyFeatureGrid:
    """Build a GT-guided anomaly feature grid from train/reference masks only."""
    require_gt_anomaly_memory(allow_gt_anomaly_memory)
    overlaps = compute_candidate_patch_overlaps(
        sample,
        resolution=resolution,
        patch_size=patch_size,
        num_classes=num_classes,
    )
    h, w = feature_grid.grid_size
    overlap_by_class = np.stack(
        [overlaps.class_overlaps[c] for c in range(1, num_classes + 1)],
        axis=-1,
    )
    if overlap_by_class.shape[:2] != (h, w):
        raise ValueError("Overlap grid does not match feature grid size")
    d_grid = np.asarray(d_normal, dtype=np.float32).reshape(h, w)
    source_classes = tuple(
        class_id
        for class_id in range(1, num_classes + 1)
        if float(np.max(overlaps.class_overlaps[class_id])) > 0.0
        and (held_out_classes is None or class_id not in held_out_classes)
    )
    if held_out_classes:
        for class_id in held_out_classes:
            overlap_by_class[:, :, class_id - 1] = 0.0
    return AnomalyFeatureGrid(
        image_id=sample.image_id,
        features=np.asarray(feature_grid.features, dtype=np.float32),
        grid_size=(h, w),
        overlap_by_class=overlap_by_class.astype(np.float32),
        d_normal=d_grid,
        source_classes=source_classes,
    )


def select_anomaly_patches(
    grids: list[AnomalyFeatureGrid],
    *,
    class_id: int,
    overlap_rule: OverlapRule = "any_overlap",
    d_normal_gate_percentile: float | None = None,
    top_distance_per_image: int | None = None,
    per_image_cap: int | None = 256,
    global_cap: int | None = None,
    class_balanced_coreset: int | None = None,
    seed: int = 42,
    exclude_classes: set[int] | frozenset[int] | None = None,
) -> AnomalySelectionResult:
    """Stage 8A–8C selection for one anomaly class."""
    if exclude_classes and class_id in exclude_classes:
        return AnomalySelectionResult(
            class_id=class_id,
            selected_indices=np.zeros((0,), dtype=np.int64),
            n_images=0,
            n_patches=0,
            extras={"excluded": True},
        )

    selected_feats: list[np.ndarray] = []
    selected_meta: list[tuple[str, int]] = []
    patches_per_image: dict[str, int] = {}
    offset = 0
    absolute_indices: list[int] = []

    gate_threshold = None
    if d_normal_gate_percentile is not None:
        all_d = np.concatenate([g.d_normal.ravel() for g in grids]) if grids else np.zeros(0)
        if all_d.size:
            gate_threshold = float(np.percentile(all_d, d_normal_gate_percentile))

    for grid in grids:
        h, w = grid.grid_size
        class_overlap = grid.overlap_by_class[:, :, class_id - 1]
        mask = oracle_rejection_mask(class_overlap, overlap_rule)
        if gate_threshold is not None:
            mask &= grid.d_normal >= gate_threshold
        idxs = np.flatnonzero(mask.ravel())
        if top_distance_per_image is not None and idxs.size:
            distances = grid.d_normal.ravel()[idxs]
            order = np.argsort(-distances, kind="stable")
            idxs = idxs[order[: int(top_distance_per_image)]]
        if per_image_cap is not None and idxs.size > per_image_cap:
            distances = grid.d_normal.ravel()[idxs]
            order = np.argsort(-distances, kind="stable")
            idxs = idxs[order[: int(per_image_cap)]]
        if idxs.size == 0:
            continue
        feats = grid.features[idxs]
        selected_feats.append(feats)
        for local_i, flat_i in enumerate(idxs):
            selected_meta.append((grid.image_id, int(flat_i)))
            absolute_indices.append(offset + local_i)
        patches_per_image[grid.image_id] = int(idxs.size)
        offset += int(idxs.size)

    if not selected_feats:
        return AnomalySelectionResult(
            class_id=class_id,
            selected_indices=np.zeros((0,), dtype=np.int64),
            n_images=0,
            n_patches=0,
            patches_per_image={},
        )

    features = np.concatenate(selected_feats, axis=0)
    keep_idx = np.arange(features.shape[0], dtype=np.int64)
    extras: dict[str, Any] = {
        "overlap_rule": overlap_rule,
        "d_normal_gate_percentile": d_normal_gate_percentile,
        "gate_threshold": gate_threshold,
        "top_distance_per_image": top_distance_per_image,
        "per_image_cap": per_image_cap,
    }

    if class_balanced_coreset is not None and features.shape[0] > class_balanced_coreset:
        features = greedy_coreset_absolute(
            features, int(class_balanced_coreset), seed=seed
        )
        # Approximate: keep first n after coreset (greedy_coreset_absolute returns features).
        keep_idx = np.arange(features.shape[0], dtype=np.int64)
        extras["coreset_size"] = int(class_balanced_coreset)
    elif global_cap is not None and features.shape[0] > global_cap:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(features.shape[0], int(global_cap), replace=False))
        features = features[keep_idx]
        extras["global_cap"] = int(global_cap)

    return AnomalySelectionResult(
        class_id=class_id,
        selected_indices=keep_idx.astype(np.int64),
        n_images=len(patches_per_image),
        n_patches=int(features.shape[0]),
        patches_per_image=patches_per_image,
        extras={**extras, "features": features, "meta": selected_meta},
    )


def save_anomaly_selection_cache(
    path: str | Path,
    *,
    class_results: dict[int, AnomalySelectionResult],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for class_id, result in class_results.items():
        payload[f"class_{class_id}_indices"] = result.selected_indices
        payload[f"class_{class_id}_n_patches"] = np.asarray(result.n_patches)
        if "features" in result.extras:
            payload[f"class_{class_id}_features"] = np.asarray(result.extras["features"])
    np.savez_compressed(path, **payload)
    return path


def load_anomaly_selection_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}
