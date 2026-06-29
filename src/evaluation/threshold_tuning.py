from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.detectors import build_detector
from src.detectors.base import DetectorOutput
from src.evaluation.mask_metrics import evaluate_mask_single, summarize_fold_mask_metrics
from src.evaluation.patch_metrics import (
    aggregate_global_metrics,
    evaluate_patch_predictions,
)
from src.evaluation.reproducibility import save_json, seed_all
from src.segmenters import build_segmenter
from src.segmenters.base import SegmenterPrompts
from src.severstal.dataset import SeverstalDataset, SeverstalSample
from src.severstal.rle import union_masks
from src.severstal.transforms import patches_to_bboxes, scores_to_patch_predictions


@dataclass
class ThresholdSweepRow:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def default_threshold_grid(
    all_scores: np.ndarray,
    n_points: int = 80,
) -> np.ndarray:
    """Build a threshold grid from score percentiles plus linspace endpoints."""
    scores = all_scores[np.isfinite(all_scores)]
    if scores.size == 0:
        return np.linspace(0.0, 1.0, n_points)

    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.array([lo])

    percentiles = np.linspace(0, 100, max(n_points // 2, 20))
    grid = np.unique(np.concatenate([np.percentile(scores, percentiles), np.linspace(lo, hi, n_points)]))
    return np.sort(grid)


def sweep_patch_thresholds(
    per_image_results: list[dict],
    thresholds: np.ndarray,
) -> list[ThresholdSweepRow]:
    """Sweep thresholds using precomputed GT labels and patch scores per image."""
    rows: list[ThresholdSweepRow] = []

    for threshold in thresholds:
        counts_list = []
        for item in per_image_results:
            pred = scores_to_patch_predictions(item["patch_scores"], float(threshold))
            valid = item.get("valid_mask")
            if valid is not None:
                gt = item["gt_labels"]["agnostic"][valid]
                pred = pred[valid]
            else:
                gt = item["gt_labels"]["agnostic"]
            gt_flat = gt.astype(bool).ravel()
            pred_flat = pred.astype(bool).ravel()
            counts_list.append(
                {
                    "tp": int(np.sum(gt_flat & pred_flat)),
                    "fp": int(np.sum(~gt_flat & pred_flat)),
                    "fn": int(np.sum(gt_flat & ~pred_flat)),
                    "tn": int(np.sum(~gt_flat & ~pred_flat)),
                }
            )

        global_metrics = aggregate_global_metrics(counts_list)
        rows.append(
            ThresholdSweepRow(
                threshold=float(threshold),
                precision=global_metrics["precision"],
                recall=global_metrics["recall"],
                f1=global_metrics["f1"],
                tp=global_metrics["tp"],
                fp=global_metrics["fp"],
                fn=global_metrics["fn"],
                tn=global_metrics["tn"],
            )
        )

    return rows


def find_f1_optimal(rows: list[ThresholdSweepRow]) -> ThresholdSweepRow:
    finite = [r for r in rows if np.isfinite(r.f1)]
    if not finite:
        return rows[0]
    return max(finite, key=lambda r: r.f1)


def find_recall_at(
    rows: list[ThresholdSweepRow],
    target_recall: float = 0.7,
) -> ThresholdSweepRow:
    """Threshold with recall >= target (prefer highest precision among ties)."""
    eligible = [r for r in rows if np.isfinite(r.recall) and r.recall >= target_recall]
    if eligible:
        return min(eligible, key=lambda r: r.threshold)
    return max(rows, key=lambda r: r.recall if np.isfinite(r.recall) else -1.0)


def collect_val_predictions(
    config: dict,
    fold_idx: int,
) -> tuple[list[dict], list[SeverstalSample], list[DetectorOutput], Any]:
    seed = config.get("seed", 42)
    fold_seed = seed + fold_idx
    seed_all(fold_seed)

    data_cfg = config["data"]
    cv_cfg = config["cv"]
    detector_cfg = config["detector"]
    patch_cfg = config["patch_eval"]

    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=seed,
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
    )

    shots = detector_cfg.get("shots", 8)
    ref_sampling = detector_cfg.get("reference_sampling", "class_balanced")
    if detector_cfg.get("scoring_mode") == "prototype":
        ref_sampling = detector_cfg.get(
            "prototype_reference_sampling", "defect_free"
        )
    elif detector_cfg.get("name") == "dino_mahalanobis":
        ref_sampling = detector_cfg.get(
            "prototype_reference_sampling", "defect_free"
        )
    elif detector_cfg.get("name") == "ensemble":
        ref_sampling = detector_cfg.get("reference_sampling", "defect_free")

    ref_ids = dataset.select_reference_ids(
        fold_idx,
        shots,
        fold_seed,
        reference_sampling=ref_sampling,
    )

    _, val_ids = dataset.get_fold_split(fold_idx)
    detector = build_detector(detector_cfg, seed=fold_seed)
    ref_samples = [dataset.load_sample(i) for i in ref_ids]
    detector.fit(ref_samples)

    per_image: list[dict] = []
    samples: list[SeverstalSample] = []
    det_outputs: list[DetectorOutput] = []

    for val_id in val_ids:
        sample = dataset.load_sample(val_id)
        det_out = detector.predict(sample)
        patch_result = evaluate_patch_predictions(
            det_out=det_out,
            sample=sample,
            gt_overlap_threshold=patch_cfg["gt_overlap_threshold"],
            pred_score_threshold=patch_cfg["pred_score_threshold"],
            smaller_edge_size=detector_cfg.get("resolution", 448),
            num_classes=data_cfg.get("num_classes", 4),
            supports_class=detector.supports_class_prediction,
        )
        per_image.append(
            {
                "image_id": sample.image_id,
                "patch_scores": det_out.patch_scores,
                "gt_labels": patch_result["gt_labels"],
                "valid_mask": det_out.patch_valid_mask,
            }
        )
        samples.append(sample)
        det_outputs.append(det_out)

    return per_image, samples, det_outputs, detector


def plot_pr_curve(
    rows: list[ThresholdSweepRow],
    f1_row: ThresholdSweepRow,
    recall_row: ThresholdSweepRow,
    output_path: Path,
) -> None:
    recalls = [r.recall for r in rows if np.isfinite(r.recall)]
    precisions = [r.precision for r in rows if np.isfinite(r.precision)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, "b-", linewidth=2, label="PR curve")
    ax.scatter(
        [f1_row.recall],
        [f1_row.precision],
        c="green",
        s=80,
        zorder=5,
        label=f"F1-opt (t={f1_row.threshold:.4f})",
    )
    ax.scatter(
        [recall_row.recall],
        [recall_row.precision],
        c="orange",
        s=80,
        zorder=5,
        label=f"Recall@0.7 (t={recall_row.threshold:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Patch-level precision-recall")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_operating_points(
    all_scores: np.ndarray,
    f1_row: ThresholdSweepRow,
    recall_row: ThresholdSweepRow,
    output_path: Path,
) -> None:
    scores = all_scores[np.isfinite(all_scores)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(
        f1_row.threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"F1-opt t={f1_row.threshold:.4f}",
    )
    ax.axvline(
        recall_row.threshold,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Recall@0.7 t={recall_row.threshold:.4f}",
    )
    ax.set_xlabel("Patch anomaly score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution with operating points")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def optional_sam2_comparison(
    config: dict,
    samples: list[SeverstalSample],
    det_outputs: list[DetectorOutput],
    f1_row: ThresholdSweepRow,
    recall_row: ThresholdSweepRow,
    max_images: int = 10,
) -> dict:
    segmenter_cfg = config.get("segmenter")
    if not segmenter_cfg:
        return {}

    segmenter = build_segmenter(segmenter_cfg)
    subset = list(zip(samples, det_outputs))[:max_images]
    results: dict[str, dict] = {}

    for label, row in [("f1_optimal", f1_row), ("recall_at_0.7", recall_row)]:
        per_image_mask = []
        pred_masks = []
        gt_masks = []

        for sample, det_out in subset:
            pred_patches = scores_to_patch_predictions(
                det_out.patch_scores, row.threshold
            )
            native_shape = sample.image.shape[:2]
            bboxes = patches_to_bboxes(
                pred_patches,
                det_out.patch_size,
                det_out.processed_shape,
                native_shape,
                min_prompt_area=segmenter_cfg.get("min_prompt_area", 1),
            )
            prompts = SegmenterPrompts(bboxes=bboxes)
            seg_out = segmenter.segment(sample.image, prompts)
            mask_result = evaluate_mask_single(seg_out, sample, False)
            per_image_mask.append(mask_result)
            pred_masks.append(seg_out.mask)
            gt_masks.append(union_masks(list(sample.masks_by_class.values())))

        summary = summarize_fold_mask_metrics(
            per_image_mask, False, pred_masks=pred_masks, gt_masks=gt_masks
        )
        results[label] = {
            "threshold": row.threshold,
            "mask_global": summary["class_agnostic"].get("global", {}),
            "n_images": len(subset),
        }

    return results


def run_threshold_tuning(
    config: dict,
    fold_idx: int = 0,
    output_dir: str | Path | None = None,
    n_thresholds: int = 80,
    target_recall: float = 0.7,
    with_sam2: bool = False,
    sam2_max_images: int = 10,
) -> dict[str, Any]:
    per_image, samples, det_outputs, detector = collect_val_predictions(
        config, fold_idx
    )

    all_scores = np.concatenate(
        [item["patch_scores"].ravel() for item in per_image]
    )
    thresholds = default_threshold_grid(all_scores, n_points=n_thresholds)
    rows = sweep_patch_thresholds(per_image, thresholds)
    f1_row = find_f1_optimal(rows)
    recall_row = find_recall_at(rows, target_recall=target_recall)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir or "results_threshold_tuning") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_pr_curve(rows, f1_row, recall_row, run_dir / "pr_curve.png")
    plot_operating_points(all_scores, f1_row, recall_row, run_dir / "operating_points.png")

    sweep_table = [
        {
            "threshold": r.threshold,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            "tn": r.tn,
        }
        for r in rows
    ]

    result: dict[str, Any] = {
        "fold": fold_idx,
        "detector": config["detector"].get("name"),
        "n_val_images": len(per_image),
        "recommended": {
            "f1_optimal": {
                "pred_score_threshold": f1_row.threshold,
                "precision": f1_row.precision,
                "recall": f1_row.recall,
                "f1": f1_row.f1,
            },
            f"recall_at_{target_recall}": {
                "pred_score_threshold": recall_row.threshold,
                "precision": recall_row.precision,
                "recall": recall_row.recall,
                "f1": recall_row.f1,
            },
        },
        "guidance": (
            "F1-optimal threshold can be misleading under patch imbalance (~30:1). "
            f"For SAM2 downstream, prefer recall_at_{target_recall} when recall matters more."
        ),
        "sweep": sweep_table,
    }

    if with_sam2:
        result["sam2_preview"] = optional_sam2_comparison(
            config,
            samples,
            det_outputs,
            f1_row,
            recall_row,
            max_images=sam2_max_images,
        )

    save_json(result, run_dir / "threshold_tuning.json")
    with open(run_dir / "sweep.csv", "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1,tp,fp,fn,tn\n")
        for row in sweep_table:
            f.write(
                f"{row['threshold']},{row['precision']},{row['recall']},"
                f"{row['f1']},{row['tp']},{row['fp']},{row['fn']},{row['tn']}\n"
            )

    print(f"\nThreshold tuning results (fold {fold_idx}, detector={result['detector']})")
    print(f"  F1-optimal:     t={f1_row.threshold:.6f}  P={f1_row.precision:.4f}  "
          f"R={f1_row.recall:.4f}  F1={f1_row.f1:.4f}")
    print(f"  Recall@{target_recall}: t={recall_row.threshold:.6f}  "
          f"P={recall_row.precision:.4f}  R={recall_row.recall:.4f}  F1={recall_row.f1:.4f}")
    print(f"  Saved to {run_dir}")

    return result


def tune_ensemble_weights(
    config: dict,
    tune_fold: int = 0,
    benchmark_folds: list[int] | None = None,
    weight_steps: int = 11,
    target_recall: float = 0.7,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Tune ensemble weights on tune_fold val; report metrics on benchmark_folds only.

    Avoids circular evaluation by not using tune_fold in benchmark set.
    """
    if config["detector"].get("name") != "ensemble":
        raise ValueError("Config detector.name must be 'ensemble'.")

    if benchmark_folds is None:
        benchmark_folds = [1, 2, 3, 4]

    detector_cfg = config["detector"]
    sub_cfgs = detector_cfg.get("sub_detectors", [])
    if len(sub_cfgs) != 2:
        raise ValueError("Weight grid search currently supports exactly 2 sub-detectors.")

    per_image_tune, _, _, _ = collect_val_predictions(config, tune_fold)
    all_scores_tune = np.concatenate([item["patch_scores"].ravel() for item in per_image_tune])
    thresholds = default_threshold_grid(all_scores_tune)

    best: dict[str, Any] | None = None
    weight_grid = np.linspace(0.0, 1.0, weight_steps)

    for w0 in weight_grid:
        w1 = 1.0 - w0
        trial_cfg = dict(config)
        trial_cfg["detector"] = dict(detector_cfg)
        trial_cfg["detector"]["weights"] = [float(w0), float(w1)]

        per_image, _, _, _ = collect_val_predictions(trial_cfg, tune_fold)
        rows = sweep_patch_thresholds(per_image, thresholds)
        recall_row = find_recall_at(rows, target_recall=target_recall)

        score = recall_row.f1 if np.isfinite(recall_row.f1) else -1.0
        if best is None or score > best["tune_score"]:
            best = {
                "weights": [float(w0), float(w1)],
                "tune_threshold": recall_row.threshold,
                "tune_score": score,
                "tune_recall": recall_row.recall,
                "tune_precision": recall_row.precision,
                "tune_f1": recall_row.f1,
            }

    assert best is not None

    benchmark_results = []
    for fold_idx in benchmark_folds:
        eval_cfg = dict(config)
        eval_cfg["detector"] = dict(detector_cfg)
        eval_cfg["detector"]["weights"] = best["weights"]
        eval_cfg["patch_eval"] = dict(config["patch_eval"])
        eval_cfg["patch_eval"]["pred_score_threshold"] = best["tune_threshold"]

        per_image, _, _, _ = collect_val_predictions(eval_cfg, fold_idx)
        rows = sweep_patch_thresholds(
            per_image,
            np.array([best["tune_threshold"]]),
        )
        row = rows[0]
        benchmark_results.append(
            {
                "fold": fold_idx,
                "precision": row.precision,
                "recall": row.recall,
                "f1": row.f1,
            }
        )

    mean_f1 = float(np.nanmean([r["f1"] for r in benchmark_results]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir or "results_ensemble_tuning") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tune_fold": tune_fold,
        "benchmark_folds": benchmark_folds,
        "best_weights": best["weights"],
        "best_threshold": best["tune_threshold"],
        "tune_fold_metrics": {
            "recall": best["tune_recall"],
            "precision": best["tune_precision"],
            "f1": best["tune_f1"],
        },
        "benchmark_per_fold": benchmark_results,
        "benchmark_mean_f1": mean_f1,
        "note": "Report benchmark_mean_f1 on hold-out folds, not tune_fold metrics.",
    }
    save_json(result, run_dir / "ensemble_tuning.json")

    print(f"\nEnsemble weight tuning (tune fold {tune_fold}, benchmark {benchmark_folds})")
    print(f"  Best weights: {best['weights']}")
    print(f"  Threshold (recall@{target_recall} on tune fold): {best['tune_threshold']:.6f}")
    print(f"  Benchmark mean F1 (folds {benchmark_folds}): {mean_f1:.4f}")
    print(f"  Saved to {run_dir}")

    return result
