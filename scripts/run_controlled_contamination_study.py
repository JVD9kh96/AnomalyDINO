#!/usr/bin/env python3
"""Phase 6: controlled contamination / repeated-defect mechanism study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors import build_detector
from src.detectors.anomaly_dino import AnomalyDINODetector
from src.detectors.reference_purification import oracle_keep_mask_from_gt
from src.evaluation.reference_bank_metrics import (
    collect_image_eval_item,
    compute_ranking_metrics,
    stack_patch_arrays,
)
from src.evaluation.reproducibility import save_json, seed_all
from src.severstal.dataset import SeverstalDataset

CONTAMINATION_RATES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SOURCE_COMPOSITIONS = (
    "uniform",
    "class_balanced",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
)
FINAL_BUDGET = 51_200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase6_controlled_contamination.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_refbank/phase6")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="Emit condition plan only")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--only-names",
        nargs="+",
        default=None,
        help="Run only these contamination condition names (space-separated).",
    )
    p.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Cap incomplete conditions launched this session (after resume skips).",
    )
    return p.parse_args()


def enumerate_conditions() -> list[dict]:
    """7 rates × 6 compositions with shared 0% across compositions."""
    conditions = []
    shared_zero = {
        "rate": 0.0,
        "composition": "shared_zero",
        "name": "rate0_shared",
    }
    conditions.append(shared_zero)
    for rate, composition in product(CONTAMINATION_RATES, SOURCE_COMPOSITIONS):
        if rate == 0.0:
            continue
        conditions.append(
            {
                "rate": float(rate),
                "composition": composition,
                "name": f"rate{rate:g}_{composition}",
            }
        )
    return conditions


def sample_anomalous_pool(
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    fold_idx: int,
    exclude_ids: set[str],
    *,
    composition: str,
    seed: int,
    max_images: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Sample anomalous source patches from training/reference images only."""
    train_ids, _ = dataset.get_fold_split(fold_idx)
    defect_ids = sorted(
        i for i in train_ids if dataset._has_defect[i] and i not in exclude_ids
    )
    rng = np.random.default_rng(seed)
    if len(defect_ids) > max_images:
        defect_ids = list(rng.choice(defect_ids, size=max_images, replace=False))

    feat_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    meta: list[dict] = []
    for image_id in defect_ids:
        sample = dataset.load_sample(image_id)
        grid = detector.extract_reference_features(sample, use_cache=True)
        h, w = grid.grid_size
        keep_normal = oracle_keep_mask_from_gt(
            sample,
            grid.grid_size,
            detector._patch_size,
            detector.resolution,
            overlap_threshold=float(detector.gt_overlap_threshold),
            num_classes=detector.num_classes,
        )
        anomalous = ~keep_normal
        if grid.patch_keep_mask is not None:
            anomalous &= np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
        if not anomalous.any():
            continue
        # Assign class by max overlap class among positives.
        from src.severstal.transforms import build_gt_patch_labels

        labels = build_gt_patch_labels(
            sample.masks_by_class,
            sample.image.shape[:2],
            detector.resolution,
            detector._patch_size,
            float(detector.gt_overlap_threshold),
            num_classes=detector.num_classes,
        )
        class_id_map = np.zeros(h * w, dtype=np.int32)
        for class_id in range(1, detector.num_classes + 1):
            key = f"class_{class_id}"
            if key in labels:
                class_id_map[np.asarray(labels[key], dtype=bool).ravel()] = class_id
        idxs = np.flatnonzero(anomalous)
        for idx in idxs:
            class_id = int(class_id_map[idx])
            if composition.startswith("class_"):
                want = int(composition.split("_")[1])
                if class_id != want:
                    continue
            feat_parts.append(grid.features[idx : idx + 1])
            class_parts.append(np.array([class_id], dtype=np.int32))
            meta.append(
                {
                    "image_id": image_id,
                    "grid_rc": (int(idx // w), int(idx % w)),
                    "class_id": class_id,
                }
            )

    if not feat_parts:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            [],
        )
    features = np.concatenate(feat_parts, axis=0)
    classes = np.concatenate(class_parts, axis=0)
    if composition == "class_balanced":
        # Subsample to equal counts per present class.
        present = sorted(set(int(c) for c in classes if c > 0))
        if present:
            counts = [int(np.sum(classes == c)) for c in present]
            n_each = min(counts)
            keep = []
            for class_id in present:
                idxs = np.flatnonzero(classes == class_id)
                keep.extend(rng.choice(idxs, size=n_each, replace=False).tolist())
            keep = np.asarray(keep, dtype=np.int64)
            features = features[keep]
            classes = classes[keep]
            meta = [meta[i] for i in keep]
    elif composition == "uniform":
        pass
    return features.astype(np.float32), classes.astype(np.int32), meta


def evaluate_with_provenance(
    detector: AnomalyDINODetector,
    dataset: SeverstalDataset,
    val_ids: list[str],
    config: dict,
    *,
    fixed_threshold: float | None,
    collect_traces: bool,
) -> dict:
    per_image = []
    all_traces = []
    for val_id in val_ids:
        sample = dataset.load_sample(val_id)
        if collect_traces:
            det_out, traces = detector.predict(sample, return_neighbor_trace=True)
            all_traces.extend(traces)
        else:
            det_out = detector.predict(sample, return_neighbor_trace=False)
        per_image.append(
            collect_image_eval_item(
                det_out,
                sample,
                gt_overlap_threshold=float(config["patch_eval"]["gt_overlap_threshold"]),
                resolution=int(config["detector"].get("resolution", 448)),
                num_classes=int(config["data"].get("num_classes", 4)),
            )
        )
    scores, labels, class_labels = stack_patch_arrays(per_image)
    metrics = compute_ranking_metrics(
        scores, labels, fixed_threshold=fixed_threshold, class_labels=class_labels
    )
    return {
        "patch": metrics,
        "traces": [asdict(t) for t in all_traces] if collect_traces else [],
        "validation_scores": scores,
        "validation_gt_labels": labels,
        "class_labels": {k: v for k, v in class_labels.items()},
    }


def class_k_delta(metrics: dict, injected_class: int | None) -> dict:
    if injected_class is None:
        return {}
    per_class = metrics.get("per_class") or {}
    key = f"class_{injected_class}"
    if key not in per_class:
        return {}
    target = per_class[key]
    others = [
        v for k, v in per_class.items() if k != key and isinstance(v, dict)
    ]
    other_auprc = float(np.nanmean([o.get("auprc", np.nan) for o in others])) if others else float("nan")
    return {
        "injected_class": injected_class,
        "class_k_auprc": target.get("auprc"),
        "non_k_auprc_mean": other_auprc,
        "class_k_minus_non_k_auprc": (
            None
            if target.get("auprc") is None or not np.isfinite(other_auprc)
            else float(target["auprc"]) - other_auprc
        ),
    }


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.device:
        config.setdefault("detector", {})["device"] = args.device

    conditions = enumerate_conditions()
    if args.only_names:
        wanted = list(dict.fromkeys(args.only_names))
        by_name = {row["name"]: row for row in conditions}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(f"Unknown --only-names {missing}. Known: {sorted(by_name)}")
        conditions = [by_name[name] for name in wanted]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_json({"conditions": conditions, "n_conditions": len(conditions)}, out_root / "condition_plan.json")
    if args.dry_run:
        print(f"Phase 6 dry-run: {len(conditions)} conditions planned -> {out_root}")
        return

    seed_all(args.seed)
    data_cfg = config["data"]
    cv_cfg = config["cv"]
    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=cv_cfg.get("split_seed", args.seed),
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
    )
    train_ids, val_ids = dataset.get_fold_split(args.fold)
    det_cfg = dict(config["detector"])
    det_cfg["coreset_size"] = int(config.get("phase6", {}).get("budget", FINAL_BUDGET))
    det_cfg["budget_policy"] = "greedy_coreset"
    det_cfg["reference_mode"] = "clean"
    det_cfg["num_classes"] = data_cfg.get("num_classes", 4)
    detector = build_detector(det_cfg, seed=args.seed)
    assert isinstance(detector, AnomalyDINODetector)

    clean_shots = int(config.get("phase6", {}).get("clean_shots", 8))
    ref_meta = dataset.select_reference_composition(
        args.fold,
        args.seed + args.fold,
        reference_mode="clean",
        clean_shots=clean_shots,
        additional_shots=0,
    )
    clean_samples = [dataset.load_sample(i) for i in ref_meta["clean_reference_ids"]]
    detector.fit_reference_composition(clean_samples, [], reference_mode="clean")
    clean_features = detector._normal_bank_features.copy()
    exclude = set(ref_meta["clean_reference_ids"])

    fixed_thr = None
    if detector._calibration is not None:
        fixed_thr = detector._calibration.threshold_at(99.0)

    aggregate_rows = []
    shared_zero_metrics = None
    launched = 0
    for condition in conditions:
        run_dir = out_root / condition["name"]
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.is_file():
            aggregate_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        if args.max_jobs is not None and launched >= args.max_jobs:
            print(
                f"Reached --max-jobs={args.max_jobs}; remaining Phase 6 conditions deferred.",
                flush=True,
            )
            break
        run_dir.mkdir(parents=True, exist_ok=True)

        composition = condition["composition"]
        rate = float(condition["rate"])
        if composition == "shared_zero":
            pool_feats = np.zeros((0, clean_features.shape[1]), dtype=np.float32)
            pool_classes = np.zeros((0,), dtype=np.int32)
            pool_meta: list[dict] = []
            effective_composition = "uniform"
        else:
            pool_feats, pool_classes, pool_meta = sample_anomalous_pool(
                detector,
                dataset,
                args.fold,
                exclude,
                composition=composition,
                seed=args.seed,
            )
            effective_composition = composition

        extras = detector.inject_contamination_replacement(
            clean_features,
            pool_feats,
            rate,
            seed=args.seed,
            anomalous_classes=pool_classes if pool_classes.size else None,
            anomalous_meta=pool_meta or None,
            target_bank_size=int(config.get("phase6", {}).get("budget", FINAL_BUDGET)),
        )
        result = evaluate_with_provenance(
            detector,
            dataset,
            val_ids,
            config,
            fixed_threshold=fixed_thr,
            collect_traces=True,
        )
        injected_class = (
            int(composition.split("_")[1])
            if composition.startswith("class_")
            else None
        )
        row = {
            "name": condition["name"],
            "rate": rate,
            "composition": effective_composition,
            "fold": args.fold,
            "seed": args.seed,
            "bank_size": detector.last_bank_stats.final_memory_bank_size,
            "insertion_policy": "replacement",
            "injection": extras,
            "patch": result["patch"],
            "class_k_delta": class_k_delta(result["patch"], injected_class),
            "n_neighbor_traces": len(result["traces"]),
        }
        save_json(row, metrics_path)
        save_json({"traces": result["traces"][:5000]}, run_dir / "neighbor_traces.json")
        aggregate_rows.append(row)
        launched += 1
        if composition == "shared_zero":
            shared_zero_metrics = row
        print(
            f"[{condition['name']}] AUPRC={result['patch'].get('auprc')} "
            f"bank={row['bank_size']} injected={extras.get('n_injected')}"
        )

    report = {
        "phase": "phase6_controlled_contamination",
        "fold": args.fold,
        "seed": args.seed,
        "n_conditions": len(aggregate_rows),
        "shared_zero": shared_zero_metrics,
        "rows": aggregate_rows,
        "notes": [
            "0% condition is shared across compositions.",
            "Anomalous patches are sampled from training/reference only.",
            "Insertion policy is replacement at constant bank size.",
        ],
    }
    report_path = out_root / "phase6_controlled_contamination_report.json"
    save_json(report, report_path)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
