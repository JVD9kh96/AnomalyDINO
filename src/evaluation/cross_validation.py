from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from src.detectors import build_detector
from src.evaluation.mask_metrics import evaluate_mask_single, summarize_fold_mask_metrics
from src.evaluation.patch_metrics import (
    evaluate_patch_predictions,
    summarize_fold_patch_metrics,
)
from src.evaluation.reproducibility import save_folds_json, save_json, seed_all
from src.segmenters import build_segmenter
from src.segmenters.base import SegmenterPrompts
from src.severstal.dataset import SeverstalDataset
from src.severstal.rle import union_masks
from src.severstal.transforms import (
    patches_to_bboxes,
    patches_to_points,
    scores_to_patch_predictions,
)
from src.training.data import LazySampleList
from src.training.sampling import resolve_training_ids
from src.visualization.severstal_viz import save_visualizations_pdf


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cross_validation(
    config: dict,
    fold_indices: list[int] | None = None,
    config_path: str | Path | None = None,
) -> dict:
    seed = config.get("seed", 42)
    data_cfg = config["data"]
    cv_cfg = config["cv"]
    patch_cfg = config["patch_eval"]
    detector_cfg = config["detector"]
    segmenter_cfg = config["segmenter"]
    output_cfg = config["output"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_cfg.get("dir", "results_severstal")) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_path:
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    folds_json_path = run_dir / "folds.json"
    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=seed,
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
        folds_json_path=folds_json_path if folds_json_path.exists() else None,
    )

    if not folds_json_path.exists():
        save_folds_json(dataset.fold_splits, folds_json_path)

    n_folds = cv_cfg.get("n_folds", 5)
    if fold_indices is None:
        fold_indices = list(range(n_folds))

    all_fold_results = {}
    fold_summaries = {"patch": [], "mask": []}

    for fold_idx in fold_indices:
        fold_seed = seed + fold_idx
        seed_all(fold_seed)
        print(f"\n=== Fold {fold_idx} (seed={fold_seed}) ===")

        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ids, val_ids = dataset.get_fold_split(fold_idx)
        shots = detector_cfg.get("shots", 8)
        ref_sampling = detector_cfg.get("reference_sampling", "class_balanced")
        reference_mode = detector_cfg.get("reference_mode")
        is_linear_probe = detector_cfg.get("name") == "dino_linear_probe"
        ref_meta: dict | None = None

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

        if is_linear_probe:
            ref_ids = resolve_training_ids(
                dataset,
                fold_idx,
                shots,
                fold_seed,
                reference_sampling=ref_sampling,
            )
            if shots is None:
                print("  Mode: full supervised (all train-fold images)")
            elif shots == -1:
                print(
                    f"  Mode: all eligible train images "
                    f"(shots=-1, sampling={ref_sampling})"
                )
            else:
                print(f"  Mode: k-shot supervised training (shots={shots})")
        elif reference_mode:
            ref_meta = dataset.select_reference_composition(
                fold_idx,
                fold_seed,
                reference_mode=reference_mode,
                clean_shots=int(detector_cfg.get("clean_shots", 2)),
                additional_shots=int(detector_cfg.get("additional_shots", 0)),
                additional_sampling=str(
                    detector_cfg.get("additional_sampling", "class_balanced")
                ),
            )
            ref_ids = list(
                dict.fromkeys(
                    [
                        *ref_meta["clean_reference_ids"],
                        *ref_meta["additional_reference_ids"],
                    ]
                )
            )
            print(
                f"  Mode: reference_mode={reference_mode} "
                f"(clean={len(ref_meta['clean_reference_ids'])}, "
                f"additional={len(ref_meta['additional_reference_ids'])})"
            )
        else:
            if shots is None:
                raise ValueError(
                    "detector.shots=null is only supported for dino_linear_probe"
                )
            ref_ids = dataset.select_reference_ids(
                fold_idx,
                shots,
                fold_seed,
                reference_sampling=ref_sampling,
            )
            if shots == 0:
                print("  Mode: zero-shot (shots=0, no reference calibration)")
            else:
                print(f"  Mode: few-shot calibration (shots={shots})")

        print(
            f"  Train: {len(train_ids)}, Val: {len(val_ids)}, "
            f"{'Fit' if is_linear_probe else 'Ref'}: {len(ref_ids)}"
        )

        fit_detector_cfg = dict(detector_cfg)
        if is_linear_probe:
            fit_detector_cfg["num_classes"] = data_cfg.get("num_classes", 4)
            fit_detector_cfg["gt_overlap_threshold"] = patch_cfg[
                "gt_overlap_threshold"
            ]
            fit_detector_cfg["_train_block"] = config.get("train", {})
        elif reference_mode:
            fit_detector_cfg["num_classes"] = data_cfg.get("num_classes", 4)
            fit_detector_cfg["gt_overlap_threshold"] = patch_cfg.get(
                "gt_overlap_threshold", 0.5
            )

        detector = build_detector(fit_detector_cfg, seed=fold_seed)
        ref_samples = LazySampleList(dataset, ref_ids)

        pred_score_threshold = float(patch_cfg["pred_score_threshold"])
        if is_linear_probe:
            val_samples = LazySampleList(dataset, val_ids)
            train_out_dir = fold_dir / "train"
            detector.fit(
                ref_samples,
                val_samples=val_samples,
                output_dir=train_out_dir,
                train_cfg=config.get("train"),
            )
            if getattr(detector, "optimal_threshold", None) is not None:
                pred_score_threshold = float(detector.optimal_threshold)
                print(f"  Using optimal F1 threshold: {pred_score_threshold:.4f}")
        elif reference_mode and hasattr(detector, "fit_reference_composition"):
            clean_ids = ref_meta["clean_reference_ids"]
            add_ids = ref_meta["additional_reference_ids"]
            bank_stats = detector.fit_reference_composition(
                [dataset.load_sample(i) for i in clean_ids],
                [dataset.load_sample(i) for i in add_ids],
                reference_mode=reference_mode,
            )
            ref_meta["n_memory_patches_before_filtering"] = (
                bank_stats.n_memory_patches_before_filtering
            )
            ref_meta["n_memory_patches_after_filtering"] = (
                bank_stats.n_memory_patches_after_filtering
            )
            save_json(ref_meta, fold_dir / "reference_metadata.json")
        else:
            detector.fit([dataset.load_sample(i) for i in ref_ids])

        segmenter = build_segmenter(segmenter_cfg)

        per_image_patch = []
        per_image_mask = []
        pred_masks = []
        gt_masks = []
        viz_data = []
        max_viz = output_cfg.get("max_viz_images_per_fold", 20)

        for val_id in tqdm(val_ids, desc=f"Fold {fold_idx} eval"):
            sample = dataset.load_sample(val_id)
            det_out = detector.predict(sample)

            patch_result = evaluate_patch_predictions(
                det_out=det_out,
                sample=sample,
                gt_overlap_threshold=patch_cfg["gt_overlap_threshold"],
                pred_score_threshold=pred_score_threshold,
                smaller_edge_size=detector_cfg.get("resolution", 448),
                num_classes=data_cfg.get("num_classes", 4),
                supports_class=detector.supports_class_prediction,
            )
            per_image_patch.append(patch_result)

            pred_patches = scores_to_patch_predictions(
                det_out.patch_scores, pred_score_threshold
            )
            native_shape = sample.image.shape[:2]

            if segmenter_cfg.get("prompt_mode", "bbox") == "point":
                points = patches_to_points(
                    pred_patches,
                    det_out.patch_size,
                    det_out.processed_shape,
                    native_shape,
                    min_prompt_area=segmenter_cfg.get("min_prompt_area", 1),
                )
                prompts = SegmenterPrompts(
                    points=points,
                    point_labels=[1] * len(points),
                )
            else:
                bboxes = patches_to_bboxes(
                    pred_patches,
                    det_out.patch_size,
                    det_out.processed_shape,
                    native_shape,
                    min_prompt_area=segmenter_cfg.get("min_prompt_area", 1),
                )
                prompts = SegmenterPrompts(bboxes=bboxes)

            seg_out = segmenter.segment(sample.image, prompts)
            mask_result = evaluate_mask_single(
                seg_out, sample, detector.supports_class_prediction
            )
            per_image_mask.append(mask_result)
            pred_masks.append(seg_out.mask)
            gt_masks.append(union_masks(list(sample.masks_by_class.values())))

            if output_cfg.get("save_visualizations", True) and len(viz_data) < max_viz:
                viz_data.append(
                    {
                        "sample": sample,
                        "det_out": det_out,
                        "gt_labels": patch_result["gt_labels"],
                        "pred_labels": patch_result["pred_labels"],
                        "seg_out": seg_out,
                    }
                )

        patch_summary = summarize_fold_patch_metrics(
            per_image_patch, detector.supports_class_prediction
        )
        mask_summary = summarize_fold_mask_metrics(
            per_image_mask,
            detector.supports_class_prediction,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
        )

        fold_metrics = {
            "fold": fold_idx,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_ref": len(ref_ids),
            "pred_score_threshold": pred_score_threshold,
            "patch": patch_summary,
            "mask": mask_summary,
        }
        if ref_meta is not None:
            fold_metrics["reference"] = {
                "reference_mode": ref_meta.get("reference_mode"),
                "clean_reference_ids": ref_meta.get("clean_reference_ids", []),
                "additional_reference_ids": ref_meta.get(
                    "additional_reference_ids", []
                ),
                "reference_image_has_defect": ref_meta.get(
                    "reference_image_has_defect", {}
                ),
                "reference_classes": ref_meta.get("reference_classes", {}),
                "n_memory_patches_before_filtering": ref_meta.get(
                    "n_memory_patches_before_filtering", 0
                ),
                "n_memory_patches_after_filtering": ref_meta.get(
                    "n_memory_patches_after_filtering", 0
                ),
            }
        save_json(fold_metrics, fold_dir / "metrics.json")
        all_fold_results[f"fold_{fold_idx}"] = fold_metrics
        fold_summaries["patch"].append(patch_summary)
        fold_summaries["mask"].append(mask_summary)

        if output_cfg.get("save_visualizations", True) and viz_data:
            save_visualizations_pdf(
                viz_data,
                fold_dir / "visualizations.pdf",
                pred_score_threshold,
            )

    summary = _summarize_across_folds(fold_summaries)
    save_json(summary, run_dir / "summary.json")
    all_fold_results["summary"] = summary
    return all_fold_results


def _summarize_across_folds(fold_summaries: dict) -> dict:
    """Compute mean ± std of key metrics across folds."""
    import numpy as np

    def _extract(values, *keys):
        for key in keys:
            if isinstance(values, dict) and key in values:
                values = values[key]
            else:
                return None
        return values

    summary = {}
    for metric_type in ("patch", "mask"):
        folds = fold_summaries[metric_type]
        if not folds:
            continue

        for scheme in ("class_agnostic",):
            f1_globals = []
            f1_image_means = []
            iou_globals = []
            iou_image_means = []
            dice_globals = []
            dice_image_means = []

            for fold in folds:
                if metric_type == "patch":
                    block = _extract(fold, scheme, "global")
                    if block and "f1" in block:
                        f1_globals.append(block["f1"])
                    block_im = _extract(fold, scheme, "image_mean")
                    if block_im and "f1" in block_im:
                        f1_image_means.append(block_im["f1"])
                else:
                    block_g = _extract(fold, scheme, "global")
                    block_im = _extract(fold, scheme, "image_mean")
                    if block_g:
                        iou_globals.append(block_g.get("iou", float("nan")))
                        dice_globals.append(block_g.get("dice", float("nan")))
                    if block_im:
                        iou_image_means.append(block_im.get("iou", float("nan")))
                        dice_image_means.append(block_im.get("dice", float("nan")))

            if metric_type == "patch":
                summary[f"{metric_type}_class_agnostic"] = {
                    "mean_f1_global": float(np.nanmean(f1_globals)) if f1_globals else None,
                    "std_f1_global": float(np.nanstd(f1_globals)) if f1_globals else None,
                    "mean_f1_image": float(np.nanmean(f1_image_means)) if f1_image_means else None,
                    "std_f1_image": float(np.nanstd(f1_image_means)) if f1_image_means else None,
                }
            else:
                summary[f"{metric_type}_class_agnostic"] = {
                    "mean_iou_global": float(np.nanmean(iou_globals)) if iou_globals else None,
                    "std_iou_global": float(np.nanstd(iou_globals)) if iou_globals else None,
                    "mean_dice_global": float(np.nanmean(dice_globals)) if dice_globals else None,
                    "std_dice_global": float(np.nanstd(dice_globals)) if dice_globals else None,
                    "mean_iou_image": float(np.nanmean(iou_image_means)) if iou_image_means else None,
                    "std_iou_image": float(np.nanstd(iou_image_means)) if iou_image_means else None,
                    "mean_dice_image": float(np.nanmean(dice_image_means)) if dice_image_means else None,
                    "std_dice_image": float(np.nanstd(dice_image_means)) if dice_image_means else None,
                }

    return summary
