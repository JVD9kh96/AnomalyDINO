from __future__ import annotations

import numpy as np

from src.severstal.rle import union_masks
from src.severstal.dataset import SeverstalSample
from src.segmenters.base import SegmenterOutput


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2 * intersection / denom)


def evaluate_mask_single(
    seg_out: SegmenterOutput,
    sample: SeverstalSample,
    supports_class: bool = False,
) -> dict:
    gt_union = union_masks(list(sample.masks_by_class.values()))

    pred = seg_out.mask
    if pred.shape != gt_union.shape:
        raise ValueError(
            f"Mask shape mismatch: pred {pred.shape} vs gt {gt_union.shape}"
        )

    result = {
        "image_id": sample.image_id,
        "class_agnostic": {
            "iou": compute_iou(pred, gt_union),
            "dice": compute_dice(pred, gt_union),
        },
    }

    if supports_class and seg_out.masks_by_class:
        class_wise = {}
        for class_id, gt_mask in sample.masks_by_class.items():
            pred_mask = seg_out.masks_by_class.get(class_id)
            if pred_mask is not None:
                class_wise[str(class_id)] = {
                    "iou": compute_iou(pred_mask, gt_mask),
                    "dice": compute_dice(pred_mask, gt_mask),
                }
        result["class_wise"] = class_wise
    else:
        result["class_wise"] = {"skipped": "detector_class_agnostic"}

    return result


def _aggregate_global_iou_dice(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> dict:
    intersection = sum(np.logical_and(p, g).sum() for p, g in zip(pred_masks, gt_masks))
    union = sum(np.logical_or(p, g).sum() for p, g in zip(pred_masks, gt_masks))
    pred_sum = sum(p.sum() for p in pred_masks)
    gt_sum = sum(g.sum() for g in gt_masks)

    iou = float(intersection / union) if union > 0 else float("nan")
    dice = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else float("nan")
    return {"iou": iou, "dice": dice}


def summarize_fold_mask_metrics(
    per_image_results: list[dict],
    supports_class: bool = False,
    pred_masks: list[np.ndarray] | None = None,
    gt_masks: list[np.ndarray] | None = None,
) -> dict:
    agnostic_ious = [r["class_agnostic"]["iou"] for r in per_image_results]
    agnostic_dices = [r["class_agnostic"]["dice"] for r in per_image_results]

    summary = {
        "class_agnostic": {
            "image_mean": {
                "iou": float(np.nanmean(agnostic_ious)),
                "dice": float(np.nanmean(agnostic_dices)),
            },
        },
    }

    if pred_masks and gt_masks:
        summary["class_agnostic"]["global"] = _aggregate_global_iou_dice(
            pred_masks, gt_masks
        )

    if supports_class and per_image_results:
        first = per_image_results[0]
        if "skipped" not in first.get("class_wise", {}):
            class_ids = first["class_wise"].keys()
            class_wise = {}
            for class_id in class_ids:
                ious = [r["class_wise"][class_id]["iou"] for r in per_image_results]
                dices = [r["class_wise"][class_id]["dice"] for r in per_image_results]
                class_wise[class_id] = {
                    "image_mean": {
                        "iou": float(np.nanmean(ious)),
                        "dice": float(np.nanmean(dices)),
                    },
                }
            summary["class_wise"] = class_wise
        else:
            summary["class_wise"] = {"skipped": "detector_class_agnostic"}
    else:
        summary["class_wise"] = {"skipped": "detector_class_agnostic"}

    return summary
