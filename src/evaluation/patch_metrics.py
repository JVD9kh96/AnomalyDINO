from __future__ import annotations

import numpy as np

from src.detectors.base import DetectorOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import (
    build_gt_patch_labels,
    scores_to_patch_predictions,
)


def _compute_confusion(
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, int]:
    if valid_mask is not None:
        gt = gt[valid_mask]
        pred = pred[valid_mask]

    gt = gt.astype(bool).ravel()
    pred = pred.astype(bool).ravel()

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_from_confusion(counts: dict[str, int]) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": counts["tn"],
    }


def compute_single_image_patch_metrics(
    gt_labels: dict[str, np.ndarray],
    pred_labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    supports_class: bool = False,
    pred_class_labels: dict[str, np.ndarray] | None = None,
) -> dict:
    result = {
        "class_agnostic": _metrics_from_confusion(
            _compute_confusion(gt_labels["agnostic"], pred_labels, valid_mask)
        ),
    }

    if supports_class and pred_class_labels:
        class_wise = {}
        for class_id, gt in gt_labels.items():
            if class_id == "agnostic":
                continue
            if class_id in pred_class_labels:
                class_wise[class_id] = _metrics_from_confusion(
                    _compute_confusion(gt, pred_class_labels[class_id], valid_mask)
                )
        result["class_wise"] = class_wise
    else:
        result["class_wise"] = {"skipped": "detector_class_agnostic"}

    return result


def aggregate_global_metrics(per_image_counts: list[dict[str, int]]) -> dict:
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for counts in per_image_counts:
        for k in total:
            total[k] += counts[k]
    return _metrics_from_confusion(total)


def aggregate_image_mean_metrics(per_image_metrics: list[dict]) -> dict:
    keys = ["precision", "recall", "f1"]
    result = {}
    for key in keys:
        values = [m[key] for m in per_image_metrics if np.isfinite(m.get(key, float("nan")))]
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def evaluate_patch_predictions(
    det_out: DetectorOutput,
    sample: SeverstalSample,
    gt_overlap_threshold: float,
    pred_score_threshold: float,
    smaller_edge_size: int,
    num_classes: int = 4,
    supports_class: bool = False,
) -> dict:
    native_shape = sample.image.shape[:2]
    gt_labels = build_gt_patch_labels(
        sample.masks_by_class,
        native_shape,
        smaller_edge_size,
        det_out.patch_size,
        gt_overlap_threshold,
        num_classes=num_classes,
    )
    pred_labels = scores_to_patch_predictions(det_out.patch_scores, pred_score_threshold)
    valid_mask = det_out.patch_valid_mask

    pred_class_labels = None
    if supports_class and det_out.patch_class_scores is not None:
        pred_class_labels = {}
        for c in range(1, num_classes + 1):
            pred_class_labels[str(c)] = scores_to_patch_predictions(
                det_out.patch_class_scores[..., c - 1], pred_score_threshold
            )

    image_metrics = compute_single_image_patch_metrics(
        gt_labels, pred_labels, valid_mask, supports_class, pred_class_labels
    )

    agnostic_counts = _compute_confusion(
        gt_labels["agnostic"], pred_labels, valid_mask
    )

    return {
        "image_id": sample.image_id,
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "image_metrics": image_metrics,
        "agnostic_counts": agnostic_counts,
        "class_wise_counts": {
            k: _compute_confusion(v, pred_labels, valid_mask)
            for k, v in gt_labels.items()
            if k != "agnostic"
        }
        if not supports_class
        else {},
    }


def summarize_fold_patch_metrics(
    per_image_results: list[dict],
    supports_class: bool = False,
) -> dict:
    agnostic_image_metrics = [
        r["image_metrics"]["class_agnostic"] for r in per_image_results
    ]
    agnostic_counts = [r["agnostic_counts"] for r in per_image_results]

    summary = {
        "class_agnostic": {
            "global": aggregate_global_metrics(agnostic_counts),
            "image_mean": aggregate_image_mean_metrics(agnostic_image_metrics),
        },
    }

    if supports_class and per_image_results:
        first = per_image_results[0]["image_metrics"]
        if "class_wise" in first and "skipped" not in first["class_wise"]:
            class_ids = first["class_wise"].keys()
            class_wise_summary = {}
            for class_id in class_ids:
                img_metrics = [
                    r["image_metrics"]["class_wise"][class_id]
                    for r in per_image_results
                    if class_id in r["image_metrics"].get("class_wise", {})
                ]
                counts = [
                    _compute_confusion(
                        r["gt_labels"][class_id],
                        r["pred_labels"],
                        None,
                    )
                    for r in per_image_results
                ]
                class_wise_summary[class_id] = {
                    "global": aggregate_global_metrics(counts),
                    "image_mean": aggregate_image_mean_metrics(img_metrics),
                }
            summary["class_wise"] = class_wise_summary
        else:
            summary["class_wise"] = {"skipped": "detector_class_agnostic"}
    else:
        summary["class_wise"] = {"skipped": "detector_class_agnostic"}

    return summary
