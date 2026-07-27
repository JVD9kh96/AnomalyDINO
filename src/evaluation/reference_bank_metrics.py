"""Ranking and operating-point metrics for the reference-bank study."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.patch_metrics import _metrics_from_confusion
from src.evaluation.threshold_tuning import (
    default_threshold_grid,
    find_f1_optimal,
    sweep_patch_thresholds,
)
from src.severstal.transforms import build_gt_patch_labels, scores_to_patch_predictions


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.size == 0 or labels.all() or (~labels).all():
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.size == 0 or not labels.any():
        return float("nan")
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return float("nan")


def stack_patch_arrays(
    per_image: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Stack valid patches across images.

    Each item needs: patch_scores, gt_labels (dict), optional valid_mask,
    optional class_wise gt already inside gt_labels.
    """
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    class_parts: dict[str, list[np.ndarray]] = {}

    for item in per_image:
        scores = np.asarray(item["patch_scores"], dtype=np.float32)
        gt = item["gt_labels"]["agnostic"].astype(bool)
        valid = item.get("valid_mask")
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
            scores = scores[valid]
            gt = gt[valid]
        else:
            scores = scores.ravel()
            gt = gt.ravel()
        score_parts.append(scores.ravel())
        label_parts.append(gt.ravel())

        for key, grid in item["gt_labels"].items():
            if key == "agnostic":
                continue
            arr = grid.astype(bool)
            if valid is not None:
                # valid may already be applied shape; rebuild from item
                raw_valid = item.get("valid_mask")
                arr = arr[raw_valid] if raw_valid is not None else arr.ravel()
            else:
                arr = arr.ravel()
            class_parts.setdefault(key, []).append(arr.ravel())

    scores_all = (
        np.concatenate(score_parts) if score_parts else np.zeros((0,), dtype=np.float32)
    )
    labels_all = (
        np.concatenate(label_parts) if label_parts else np.zeros((0,), dtype=bool)
    )
    class_labels = {
        k: np.concatenate(v) if v else np.zeros((0,), dtype=bool)
        for k, v in class_parts.items()
    }
    return scores_all, labels_all, class_labels


def metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = scores >= float(threshold)
    labels = labels.astype(bool)
    tp = int(np.sum(labels & pred))
    fp = int(np.sum(~labels & pred))
    fn = int(np.sum(labels & ~pred))
    tn = int(np.sum(~labels & ~pred))
    return _metrics_from_confusion({"tp": tp, "fp": fp, "fn": fn, "tn": tn})


def compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fixed_threshold: float | None = None,
    class_labels: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """AUROC / AUPRC / F1-opt / fixed-threshold metrics (+ optional per-class)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()

    per_image_for_sweep = [
        {
            "patch_scores": scores.reshape(1, -1),
            "gt_labels": {"agnostic": labels.reshape(1, -1)},
            "valid_mask": None,
        }
    ]
    grid = default_threshold_grid(scores)
    rows = sweep_patch_thresholds(per_image_for_sweep, grid)
    f1_row = find_f1_optimal(rows)

    fixed_thr = float(fixed_threshold) if fixed_threshold is not None else float(f1_row.threshold)
    fixed_metrics = metrics_at_threshold(scores, labels, fixed_thr)
    f1max_metrics = metrics_at_threshold(scores, labels, f1_row.threshold)

    result: dict[str, Any] = {
        "auroc": _safe_auroc(labels, scores),
        "auprc": _safe_auprc(labels, scores),
        "f1_optimal": {
            "threshold": float(f1_row.threshold),
            "precision": float(f1_row.precision),
            "recall": float(f1_row.recall),
            "f1": float(f1_row.f1),
        },
        "fixed_threshold": {
            "threshold": fixed_thr,
            **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in fixed_metrics.items()},
        },
        "f1_max_metrics": {
            "threshold": float(f1_row.threshold),
            **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in f1max_metrics.items()},
        },
    }

    if class_labels:
        per_class = {}
        for class_id, c_labels in class_labels.items():
            c_labels = np.asarray(c_labels, dtype=bool).ravel()
            if c_labels.shape[0] != scores.shape[0]:
                continue
            per_class[class_id] = {
                "auroc": _safe_auroc(c_labels, scores),
                "auprc": _safe_auprc(c_labels, scores),
                "n_positive": int(c_labels.sum()),
            }
        result["per_class"] = per_class

    return result


def collect_image_eval_item(
    det_out,
    sample,
    *,
    gt_overlap_threshold: float,
    resolution: int,
    num_classes: int = 4,
) -> dict:
    """Build one per-image dict for stacking / threshold sweeps."""
    native_shape = sample.image.shape[:2]
    gt_labels = build_gt_patch_labels(
        sample.masks_by_class,
        native_shape,
        resolution,
        det_out.patch_size,
        gt_overlap_threshold,
        num_classes=num_classes,
    )
    return {
        "image_id": sample.image_id,
        "patch_scores": det_out.patch_scores,
        "gt_labels": gt_labels,
        "valid_mask": det_out.patch_valid_mask,
        "has_defect": sample.has_defect,
    }
