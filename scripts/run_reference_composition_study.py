#!/usr/bin/env python3
"""Run reference-composition / contamination experiments for AnomalyDINO.

Designed for Kaggle GPU runs. Does not sweep everything automatically unless
you pass multiple CLI invocations (see docs/reference_bank_study.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors import build_detector
from src.detectors.anomaly_dino import AnomalyDINODetector
from src.detectors.reference_purification import oracle_keep_mask_from_gt
from src.evaluation.mask_metrics import evaluate_mask_single, summarize_fold_mask_metrics
from src.evaluation.reference_manifest import (
    load_paired_reference_manifest,
    reference_inputs_for_mode,
)
from src.evaluation.reference_bank_metrics import (
    collect_image_eval_item,
    compute_ranking_metrics,
    stack_patch_arrays,
)
from src.evaluation.reproducibility import save_folds_json, save_json, seed_all
from src.segmenters import build_segmenter
from src.segmenters.base import SegmenterPrompts
from src.severstal.dataset import SeverstalDataset
from src.severstal.rle import union_masks
from src.severstal.transforms import patches_to_bboxes, scores_to_patch_predictions


CONDITIONS = (
    "clean",
    "contaminated_all",
    "auto_purified",
    "random_filtered",
    "fixed_ratio_trim",
    "oracle_purified",
    "class_balanced_all",
    "synthetic_contamination",
    "size_matched_clean",
    "size_matched_purified",
)

CONTAMINATION_RATIOS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Base YAML config path")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="Overrides config seed")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Freeze the CV split independently from the reference-selection seed",
    )
    p.add_argument(
        "--paired-manifest",
        type=str,
        default=None,
        help="Immutable Phase 0 manifest consumed as the reference input pool",
    )
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--clean-shots", type=int, default=None)
    p.add_argument("--additional-shots", type=int, default=None)
    p.add_argument("--additional-sampling", type=str, default=None)
    p.add_argument("--acceptance-percentile", type=float, default=None)
    p.add_argument("--fixed-trim-fraction", type=float, default=None)
    p.add_argument("--coreset-size", type=int, default=None)
    p.add_argument(
        "--budget-policy",
        choices=("greedy_coreset", "random"),
        default=None,
        help="Deterministic final-bank budget policy when --coreset-size is set",
    )
    p.add_argument("--contamination-ratio", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Skip SAM2 mask metrics (faster ablation loops)",
    )
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = deepcopy(config)
    det = cfg.setdefault("detector", {})
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.split_seed is not None:
        cfg.setdefault("cv", {})["split_seed"] = int(args.split_seed)
    if args.clean_shots is not None:
        det["clean_shots"] = int(args.clean_shots)
    if args.additional_shots is not None:
        det["additional_shots"] = int(args.additional_shots)
    if args.additional_sampling is not None:
        det["additional_sampling"] = args.additional_sampling
    if args.acceptance_percentile is not None:
        pur = det.setdefault("reference_purification", {})
        pur["normal_acceptance_percentile"] = float(args.acceptance_percentile)
    if args.fixed_trim_fraction is not None:
        pur = det.setdefault("reference_purification", {})
        pur["fixed_trim_fraction"] = float(args.fixed_trim_fraction)
    if args.coreset_size is not None:
        det["coreset_size"] = int(args.coreset_size)
    if args.budget_policy is not None:
        det["budget_policy"] = args.budget_policy
    if args.device is not None:
        det["device"] = args.device
        if "segmenter" in cfg:
            cfg["segmenter"]["device"] = args.device

    condition = args.condition
    mode_map = {
        "clean": "clean",
        "contaminated_all": "contaminated_all",
        "auto_purified": "auto_purified",
        "random_filtered": "random_filtered",
        "fixed_ratio_trim": "fixed_ratio_trim",
        "oracle_purified": "oracle_purified",
        "class_balanced_all": "class_balanced_all",
        "size_matched_clean": "clean",
        "size_matched_purified": "auto_purified",
        "synthetic_contamination": "clean",
    }
    det["reference_mode"] = mode_map[condition]
    if condition == "oracle_purified":
        det["allow_oracle_reference_filtering"] = True
    return cfg


def resolve_mode_for_condition(condition: str) -> str:
    if condition.startswith("size_matched_clean"):
        return "clean"
    if condition.startswith("size_matched_purified"):
        return "auto_purified"
    if condition == "synthetic_contamination":
        return "clean"
    return condition


def evaluate_detector(
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    val_ids: list[str],
    config: dict,
    *,
    fixed_threshold: float | None,
    skip_sam2: bool,
) -> dict:
    detector_cfg = config["detector"]
    patch_cfg = config["patch_eval"]
    data_cfg = config["data"]
    segmenter_cfg = config.get("segmenter", {})
    resolution = int(detector_cfg.get("resolution", 448))
    num_classes = int(data_cfg.get("num_classes", 4))
    pred_thr = float(
        fixed_threshold
        if fixed_threshold is not None
        else patch_cfg.get("pred_score_threshold", 0.35)
    )

    per_image: list[dict] = []
    per_image_mask = []
    pred_masks = []
    gt_masks = []
    predict_times: list[float] = []

    segmenter = None
    if not skip_sam2:
        segmenter = build_segmenter(segmenter_cfg)

    n_val = len(val_ids)
    progress_every = int(config.get("runtime", {}).get("progress_every_images", 100))
    print(
        f"Starting validation patch evaluation: {n_val} images "
        f"(SAM2={'disabled' if skip_sam2 else 'enabled'})",
        flush=True,
    )
    evaluation_started = time.perf_counter()
    for image_index, val_id in enumerate(val_ids, start=1):
        sample = dataset.load_sample(val_id)
        t0 = time.perf_counter()
        det_out = detector.predict(sample)
        predict_times.append(time.perf_counter() - t0)
        per_image.append(
            collect_image_eval_item(
                det_out,
                sample,
                gt_overlap_threshold=float(patch_cfg["gt_overlap_threshold"]),
                resolution=resolution,
                num_classes=num_classes,
            )
        )

        if segmenter is not None:
            pred_patches = scores_to_patch_predictions(det_out.patch_scores, pred_thr)
            bboxes = patches_to_bboxes(
                pred_patches,
                det_out.patch_size,
                det_out.processed_shape,
                sample.image.shape[:2],
                min_prompt_area=segmenter_cfg.get("min_prompt_area", 1),
            )
            seg_out = segmenter.segment(
                sample.image, SegmenterPrompts(bboxes=bboxes)
            )
            mask_result = evaluate_mask_single(seg_out, sample, False)
            per_image_mask.append(mask_result)
            pred_masks.append(seg_out.mask)
            gt_masks.append(union_masks(list(sample.masks_by_class.values())))

        if (
            image_index == n_val
            or (progress_every > 0 and image_index % progress_every == 0)
        ):
            elapsed = time.perf_counter() - evaluation_started
            rate = image_index / elapsed if elapsed else 0.0
            remaining = (n_val - image_index) / rate if rate else 0.0
            print(
                f"  Validation progress: {image_index}/{n_val} "
                f"({100 * image_index / n_val:.1f}%) | "
                f"elapsed={elapsed / 60:.1f} min | ETA={remaining / 60:.1f} min",
                flush=True,
            )

    scores, labels, class_labels = stack_patch_arrays(per_image)
    patch_metrics = compute_ranking_metrics(
        scores,
        labels,
        fixed_threshold=pred_thr,
        class_labels=class_labels,
    )

    mask_summary = None
    if per_image_mask:
        mask_summary = summarize_fold_mask_metrics(
            per_image_mask,
            False,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
        )

    return {
        "patch": patch_metrics,
        "mask": mask_summary,
        "n_val": len(val_ids),
        "mean_predict_time_s": float(np.mean(predict_times)) if predict_times else None,
        "total_predict_time_s": float(np.sum(predict_times)) if predict_times else None,
        "pred_score_threshold": pred_thr,
    }


def fit_for_condition(
    condition: str,
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    ref_meta: dict,
    config: dict,
    fold_seed: int,
) -> dict:
    clean_ids = ref_meta["clean_reference_ids"]
    add_ids = ref_meta["additional_reference_ids"]
    mode = resolve_mode_for_condition(condition)

    t0 = time.perf_counter()
    if condition == "synthetic_contamination":
        # Fit clean bank features, mine anomalous patches from train defect images,
        # then inject controlled contamination ratios in the caller.
        stats = detector.fit_reference_composition(
            [dataset.load_sample(i) for i in clean_ids],
            [],
            reference_mode="clean",
        )
    else:
        stats = detector.fit_reference_composition(
            [dataset.load_sample(i) for i in clean_ids],
            [dataset.load_sample(i) for i in add_ids],
            reference_mode=mode,
        )
    fit_time = time.perf_counter() - t0

    ref_meta = dict(ref_meta)
    ref_meta["n_memory_patches_before_filtering"] = stats.n_memory_patches_before_filtering
    ref_meta["n_memory_patches_after_filtering"] = stats.n_memory_patches_after_filtering
    ref_meta["bank_stats"] = {
        "n_clean_patches": stats.n_clean_patches,
        "n_candidate_patches": stats.n_candidate_patches,
        "n_accepted_candidate_patches": stats.n_accepted_candidate_patches,
        "n_rejected_candidate_patches": stats.n_rejected_candidate_patches,
        "acceptance_fraction": stats.acceptance_fraction,
        "calibration_percentile": stats.calibration_percentile,
        "calibration_threshold": stats.calibration_threshold,
        "final_memory_bank_size": stats.final_memory_bank_size,
        "n_memory_patches_clean": stats.n_memory_patches_clean,
        "n_candidate_patches_before_filter": stats.n_candidate_patches_before_filter,
        "n_candidate_patches_after_filter": stats.n_candidate_patches_after_filter,
        "n_memory_patches_before_budget": stats.n_memory_patches_before_budget,
        "n_memory_patches_final": stats.n_memory_patches_final,
        "budget_policy": detector.budget_policy,
        "budget_size_requested": detector.coreset_size,
        "extras": stats.extras,
    }
    return {"ref_meta": ref_meta, "fit_time_s": fit_time, "stats": stats}


def collect_anomalous_train_patches(
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    fold_idx: int,
    exclude_ids: set[str],
    max_images: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Gather GT-overlapping patch features from train-fold defect images."""
    train_ids, _ = dataset.get_fold_split(fold_idx)
    defect_ids = sorted(
        i for i in train_ids if dataset._has_defect[i] and i not in exclude_ids
    )
    rng = np.random.default_rng(seed)
    if len(defect_ids) > max_images:
        defect_ids = list(rng.choice(defect_ids, size=max_images, replace=False))

    parts: list[np.ndarray] = []
    for image_id in defect_ids:
        sample = dataset.load_sample(image_id)
        grid = detector.extract_reference_features(sample, use_cache=False)
        keep_normal = oracle_keep_mask_from_gt(
            sample,
            grid.grid_size,
            detector._patch_size,
            detector.resolution,
            overlap_threshold=float(
                detector.gt_overlap_threshold
            ),
            num_classes=detector.num_classes,
        )
        anomalous = ~keep_normal
        if grid.patch_keep_mask is not None:
            anomalous = anomalous & np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
        if anomalous.any():
            parts.append(grid.features[anomalous])
    if not parts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def run_synthetic_contamination(
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    fold_idx: int,
    val_ids: list[str],
    ref_meta: dict,
    config: dict,
    out_dir: Path,
    fold_seed: int,
    skip_sam2: bool,
) -> dict:
    clean_ids = ref_meta["clean_reference_ids"]
    detector.fit_reference_composition(
        [dataset.load_sample(i) for i in clean_ids],
        [],
        reference_mode="clean",
    )
    clean_feats = np.asarray(detector._normal_bank_features, dtype=np.float32)
    anomalous = collect_anomalous_train_patches(
        detector,
        dataset,
        fold_idx,
        exclude_ids=set(clean_ids),
        seed=fold_seed,
    )

    rows = []
    memory_stats = {}
    for ratio in CONTAMINATION_RATIOS:
        detector.inject_contamination(
            clean_feats, anomalous, ratio, seed=fold_seed
        )
        # Use calibration threshold from clean bank if available.
        fixed_thr = None
        if detector._calibration is not None:
            pct = float(
                config["detector"]
                .get("reference_purification", {})
                .get("normal_acceptance_percentile", 99.0)
            )
            fixed_thr = detector._calibration.threshold_at(pct)
        else:
            # Recalibrate from clean features via a fresh temp fit if needed.
            fixed_thr = float(config["patch_eval"]["pred_score_threshold"])

        metrics = evaluate_detector(
            detector,
            dataset,
            val_ids,
            config,
            fixed_threshold=fixed_thr,
            skip_sam2=skip_sam2,
        )
        patch = metrics["patch"]
        row = {
            "contamination_ratio": ratio,
            "auprc": patch.get("auprc"),
            "auroc": patch.get("auroc"),
            "f1_optimal": patch.get("f1_optimal", {}).get("f1"),
            "fixed_f1": patch.get("fixed_threshold", {}).get("f1"),
            "memory_bank_size": int(detector.last_bank_stats.final_memory_bank_size),
            "n_anomalous_available": int(anomalous.shape[0]),
        }
        rows.append(row)
        memory_stats[str(ratio)] = {
            "final_memory_bank_size": row["memory_bank_size"],
            "extras": dict(detector.last_bank_stats.extras),
        }

    csv_path = out_dir / "contamination_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(memory_stats, out_dir / "memory_bank_statistics.json")

    xs = [r["contamination_ratio"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, [r["auprc"] for r in rows], marker="o")
    plt.xlabel("Contamination ratio")
    plt.ylabel("Patch AUPRC")
    plt.title("Contamination vs AUPRC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "contamination_vs_auprc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, [r["f1_optimal"] for r in rows], marker="o", label="F1-max")
    plt.plot(xs, [r["fixed_f1"] for r in rows], marker="s", label="Fixed F1")
    plt.xlabel("Contamination ratio")
    plt.ylabel("F1")
    plt.title("Contamination vs F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "contamination_vs_f1.png", dpi=150)
    plt.close()

    return {"contamination_curve": rows, "memory_bank_statistics": memory_stats}


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 42))
    fold_idx = int(args.fold)
    reference_seed = seed + fold_idx
    seed_all(reference_seed)

    condition = args.condition
    out_dir = Path(
        args.output_dir
        or Path(config.get("output", {}).get("dir", "results_refbank"))
        / f"{condition}_fold{fold_idx}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    data_cfg = config["data"]
    cv_cfg = config["cv"]
    detector_cfg = config["detector"]
    folds_path = out_dir / "folds.json"

    split_seed = int(cv_cfg.get("split_seed", seed))
    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=split_seed,
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
        folds_json_path=folds_path if folds_path.exists() else None,
    )
    if not folds_path.exists():
        save_folds_json(dataset.fold_splits, folds_path)

    train_ids, val_ids = dataset.get_fold_split(fold_idx)
    mode = resolve_mode_for_condition(condition)
    clean_shots = int(detector_cfg.get("clean_shots", 2))
    additional_shots = int(detector_cfg.get("additional_shots", 0))
    additional_sampling = str(detector_cfg.get("additional_sampling", "class_balanced"))

    # For size-matched clean, no additional images.
    if condition == "size_matched_clean":
        additional_shots = 0
        mode = "clean"

    if args.paired_manifest:
        manifest = load_paired_reference_manifest(args.paired_manifest, dataset=dataset)
        if int(manifest["fold"]) != fold_idx or int(manifest["seed"]) != seed:
            raise ValueError(
                "Paired manifest fold/seed does not match this run: "
                f"manifest=({manifest['fold']}, {manifest['seed']}), "
                f"run=({fold_idx}, {seed})"
            )
        selection = manifest["selection"]
        if (
            int(selection["clean_shots"]) != clean_shots
            or int(selection["additional_shots"]) != additional_shots
        ):
            raise ValueError(
                "Paired manifest shot counts do not match the requested study run"
            )
        input_mode = mode if condition != "synthetic_contamination" else "clean"
        inputs = reference_inputs_for_mode(manifest, input_mode)
        all_reference_ids = [
            *inputs["clean_reference_ids"], *inputs["additional_reference_ids"]
        ]
        has_defect, classes = dataset.reference_image_metadata(all_reference_ids)
        ref_meta = {
            "reference_mode": input_mode,
            **inputs,
            "reference_image_has_defect": has_defect,
            "reference_classes": classes,
            "n_memory_patches_before_filtering": 0,
            "n_memory_patches_after_filtering": 0,
            "paired_manifest_id": manifest["manifest_id"],
            "paired_manifest_path": str(Path(args.paired_manifest).resolve()),
        }
    else:
        ref_meta = dataset.select_reference_composition(
            fold_idx,
            reference_seed,
            reference_mode=mode if condition != "synthetic_contamination" else "clean",
            clean_shots=clean_shots,
            additional_shots=additional_shots,
            additional_sampling=additional_sampling,
        )
    # class_balanced_all composition already handled inside select_reference_composition

    fit_cfg = dict(detector_cfg)
    fit_cfg["num_classes"] = data_cfg.get("num_classes", 4)
    fit_cfg["gt_overlap_threshold"] = config["patch_eval"].get(
        "gt_overlap_threshold", 0.5
    )
    if condition == "oracle_purified":
        fit_cfg["allow_oracle_reference_filtering"] = True
    if condition in ("size_matched_clean", "size_matched_purified"):
        if fit_cfg.get("coreset_size") is None:
            raise ValueError(
                "size_matched_* conditions require --coreset-size N"
            )

    detector = build_detector(fit_cfg, seed=reference_seed)
    assert isinstance(detector, AnomalyDINODetector)

    print(
        f"Condition={condition} fold={fold_idx} seed={seed} "
        f"clean={len(ref_meta['clean_reference_ids'])} "
        f"additional={len(ref_meta['additional_reference_ids'])}"
    )

    if condition == "synthetic_contamination":
        # Calibrate on clean refs for fixed threshold.
        detector.fit_reference_composition(
            [dataset.load_sample(i) for i in ref_meta["clean_reference_ids"]],
            [],
            reference_mode="clean",
        )
        from src.detectors.reference_calibration import calibrate_normal_distances

        clean_grids = [
            detector.extract_reference_features(dataset.load_sample(i))
            for i in ref_meta["clean_reference_ids"]
        ]
        detector._calibration = calibrate_normal_distances(
            clean_grids,
            knn_metric=detector.knn_metric,
            k_neighbors=detector.k_neighbors,
        )
        curve = run_synthetic_contamination(
            detector,
            dataset,
            fold_idx,
            val_ids,
            ref_meta,
            config,
            out_dir,
            reference_seed,
            skip_sam2=args.skip_sam2,
        )
        save_json(
            {
                "condition": condition,
                "fold": fold_idx,
                "seed": seed,
                "reference": ref_meta,
                "contamination": curve,
            },
            out_dir / "metrics.json",
        )
        print(f"Wrote contamination curve to {out_dir}")
        return

    fit_info = fit_for_condition(
        condition, detector, dataset, ref_meta, config, reference_seed
    )
    ref_meta = fit_info["ref_meta"]
    save_json(ref_meta, out_dir / "reference_metadata.json")

    fixed_thr = None
    if detector._calibration is not None:
        pct = float(
            detector_cfg.get("reference_purification", {}).get(
                "normal_acceptance_percentile", 99.0
            )
        )
        fixed_thr = detector._calibration.threshold_at(pct)
    elif condition == "clean" and ref_meta["clean_reference_ids"]:
        # Build LOO calibration even for clean mode for fair fixed-threshold F1.
        from src.detectors.reference_calibration import calibrate_normal_distances

        grids = [
            detector.extract_reference_features(dataset.load_sample(i))
            for i in ref_meta["clean_reference_ids"]
        ]
        calib = calibrate_normal_distances(
            grids, knn_metric=detector.knn_metric, k_neighbors=detector.k_neighbors
        )
        detector._calibration = calib
        pct = float(
            detector_cfg.get("reference_purification", {}).get(
                "normal_acceptance_percentile", 99.0
            )
        )
        fixed_thr = calib.threshold_at(pct)

    metrics = evaluate_detector(
        detector,
        dataset,
        val_ids,
        config,
        fixed_threshold=fixed_thr,
        skip_sam2=args.skip_sam2,
    )
    metrics.update(
        {
            "condition": condition,
            "fold": fold_idx,
            "seed": seed,
            "fold_seed": reference_seed,
            "split_seed": split_seed,
            "n_train": len(train_ids),
            "reference": ref_meta,
            "fit_time_s": fit_info["fit_time_s"],
            "memory_bank_size": detector.last_bank_stats.final_memory_bank_size,
            "n_memory_patches_clean": detector.last_bank_stats.n_memory_patches_clean,
            "n_candidate_patches_before_filter": detector.last_bank_stats.n_candidate_patches_before_filter,
            "n_candidate_patches_after_filter": detector.last_bank_stats.n_candidate_patches_after_filter,
            "n_memory_patches_before_budget": detector.last_bank_stats.n_memory_patches_before_budget,
            "n_memory_patches_final": detector.last_bank_stats.n_memory_patches_final,
            "gt_masks_used_in_fitting": condition == "oracle_purified",
            "sam2_skipped": bool(args.skip_sam2),
        }
    )
    save_json(metrics, out_dir / "metrics.json")
    save_json(
        {
            "final_memory_bank_size": detector.last_bank_stats.final_memory_bank_size,
            "before": detector.last_bank_stats.n_memory_patches_before_filtering,
            "after": detector.last_bank_stats.n_memory_patches_after_filtering,
            "n_memory_patches_clean": detector.last_bank_stats.n_memory_patches_clean,
            "n_candidate_patches_before_filter": detector.last_bank_stats.n_candidate_patches_before_filter,
            "n_candidate_patches_after_filter": detector.last_bank_stats.n_candidate_patches_after_filter,
            "n_memory_patches_before_budget": detector.last_bank_stats.n_memory_patches_before_budget,
            "n_memory_patches_final": detector.last_bank_stats.n_memory_patches_final,
            "budget_policy": detector.budget_policy,
            "budget_size_requested": detector.coreset_size,
            "bank_stats": ref_meta.get("bank_stats"),
        },
        out_dir / "memory_bank_statistics.json",
    )
    print(
        f"AUPRC={metrics['patch'].get('auprc')} "
        f"AUROC={metrics['patch'].get('auroc')} "
        f"F1max={metrics['patch'].get('f1_optimal', {}).get('f1')} "
        f"FixedF1={metrics['patch'].get('fixed_threshold', {}).get('f1')} "
        f"bank={metrics['memory_bank_size']} "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
