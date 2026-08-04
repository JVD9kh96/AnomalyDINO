#!/usr/bin/env python3
"""Phases 8–10: anomaly-memory selection, dual-bank scoring, seen-class / LOCO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors import build_detector
from src.detectors.anomaly_dino import AnomalyDINODetector
from src.detectors.anomaly_memory import (
    build_anomaly_feature_grid,
    require_gt_anomaly_memory,
    save_anomaly_selection_cache,
    select_anomaly_patches,
)
from src.detectors.dual_bank import DualBankScorer
from src.detectors.knn_index import pairwise_knn_distances
from src.evaluation.heldout_aggregation import paired_deltas
from src.evaluation.reproducibility import save_json, seed_all
from src.severstal.dataset import SeverstalDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase8_anomaly_memory_study.yaml")
    p.add_argument("--stage", choices=("8A", "8B", "8C", "9", "10", "all"), default="all")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_refbank/anomaly_memory")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-gt-anomaly-memory", action="store_true")
    return p.parse_args()


def _stage_plan(stage: str) -> dict:
    plan = {
        "8A": {
            "overlap_rules": ["any_overlap", "at_least_10_percent", "at_least_50_percent"],
            "gates": [None],
            "caps": [None],
            "coresets": [None],
        },
        "8B": {
            "overlap_rules": ["any_overlap"],
            "gates": [None, 95.0, 99.0, "top_distance"],
            "caps": [256],
            "coresets": [None],
        },
        "8C": {
            "overlap_rules": ["any_overlap"],
            "gates": [None],
            "caps": [None, 64, 256],
            "coresets": [None, 512, 2048],
        },
    }
    if stage == "all":
        return plan
    return {stage: plan[stage]} if stage in plan else {}


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    allow = bool(
        args.allow_gt_anomaly_memory
        or config.get("allow_gt_anomaly_memory", False)
    )
    require_gt_anomaly_memory(allow)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "stage": args.stage,
            "fold": args.fold,
            "seed": args.seed,
            "allow_gt_anomaly_memory": allow,
            "plan": _stage_plan(args.stage if args.stage != "all" else "8A"),
        },
        out_root / "run_plan.json",
    )
    if args.dry_run:
        print(f"Anomaly-memory dry-run plan written to {out_root}")
        # Still exercise fail-closed path when flag missing via unit tests.
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
    det_cfg["num_classes"] = data_cfg.get("num_classes", 4)
    detector = build_detector(det_cfg, seed=args.seed)
    assert isinstance(detector, AnomalyDINODetector)

    # Fit a normal bank for d_normal grids.
    ref_meta = dataset.select_reference_composition(
        args.fold,
        args.seed + args.fold,
        reference_mode="clean",
        clean_shots=int(det_cfg.get("clean_shots", 8)),
        additional_shots=0,
    )
    clean_samples = [dataset.load_sample(i) for i in ref_meta["clean_reference_ids"]]
    detector.fit_reference_composition(clean_samples, [], reference_mode="clean")
    normal_bank = detector._normal_bank_features

    # Build anomaly grids from train defect images (never validation).
    defect_ids = [
        i for i in train_ids if dataset._has_defect[i] and i not in set(ref_meta["clean_reference_ids"])
    ][: int(config.get("max_anomaly_images", 32))]
    anomaly_grids = []
    for image_id in defect_ids:
        sample = dataset.load_sample(image_id)
        feat_grid = detector.extract_reference_features(sample, use_cache=True)
        d_n = pairwise_knn_distances(
            feat_grid.features,
            normal_bank,
            detector.knn_metric,
            detector.k_neighbors,
            faiss_on_cpu=True,
        ).reshape(feat_grid.grid_size)
        anomaly_grids.append(
            build_anomaly_feature_grid(
                sample,
                feat_grid,
                d_normal=d_n,
                allow_gt_anomaly_memory=True,
                resolution=detector.resolution,
                patch_size=detector._patch_size,
                num_classes=detector.num_classes,
            )
        )

    stage_results = {}
    stages = ["8A", "8B", "8C"] if args.stage in ("all",) else (
        [args.stage] if args.stage in ("8A", "8B", "8C") else []
    )
    for stage in stages:
        plan = _stage_plan(stage)[stage]
        rows = []
        for overlap_rule in plan["overlap_rules"]:
            for gate in plan["gates"]:
                for cap in plan["caps"]:
                    for coreset in plan["coresets"]:
                        class_results = {}
                        for class_id in range(1, detector.num_classes + 1):
                            result = select_anomaly_patches(
                                anomaly_grids,
                                class_id=class_id,
                                overlap_rule=overlap_rule,
                                d_normal_gate_percentile=(
                                    None if gate in (None, "top_distance") else float(gate)
                                ),
                                top_distance_per_image=256 if gate == "top_distance" else None,
                                per_image_cap=cap,
                                class_balanced_coreset=coreset,
                                seed=args.seed,
                            )
                            class_results[class_id] = result
                        rows.append(
                            {
                                "stage": stage,
                                "overlap_rule": overlap_rule,
                                "gate": gate,
                                "per_image_cap": cap,
                                "coreset": coreset,
                                "patches_per_class": {
                                    str(k): v.n_patches for k, v in class_results.items()
                                },
                                "images_per_class": {
                                    str(k): v.n_images for k, v in class_results.items()
                                },
                            }
                        )
                        cache_name = (
                            f"{stage}_{overlap_rule}_gate{gate}_cap{cap}_cs{coreset}.npz"
                        )
                        save_anomaly_selection_cache(
                            out_root / "caches" / cache_name,
                            class_results=class_results,
                        )
        stage_results[stage] = rows
        save_json({"rows": rows}, out_root / f"stage_{stage}_report.json")

    if args.stage in ("9", "all"):
        # Build class banks from any_overlap uncapped selection.
        class_banks = {}
        for class_id in range(1, detector.num_classes + 1):
            result = select_anomaly_patches(
                anomaly_grids,
                class_id=class_id,
                overlap_rule="any_overlap",
                per_image_cap=256,
                class_balanced_coreset=512,
                seed=args.seed,
            )
            feats = result.extras.get("features")
            class_banks[class_id] = (
                np.asarray(feats, dtype=np.float32)
                if feats is not None
                else np.zeros((0, normal_bank.shape[1]), dtype=np.float32)
            )
        scorer = DualBankScorer(
            normal_bank=normal_bank,
            anomaly_banks=class_banks,
            mode="gated_hybrid",
            lambda_a=1.0,
            normal_gate_percentile=95.0,
            knn_metric=detector.knn_metric,
            k_neighbors=detector.k_neighbors,
        )
        scorer.fit_calibration(normal_bank[: min(2048, normal_bank.shape[0])])
        # Polarity smoke on synthetic labels.
        q = normal_bank[:64]
        out = scorer.score(q)
        labels = np.zeros(64, dtype=bool)
        labels[32:] = True
        # Shift second half features artificially for polarity demo when needed.
        save_json(
            {
                "mode_grid": ["normal", "anomaly_diagnostic", "margin", "ratio", "gated_hybrid"],
                "lambda_a": [0.25, 0.5, 1.0, 2.0],
                "normal_gate_percentile": [90, 95, 97.5, 99],
                "calibration": "robust_z",
                "example_score_stats": {
                    "mean": float(np.mean(out["scores"])),
                    "predicted_class_hist": {
                        str(c): int(np.sum(out["predicted_nearest_anomaly_class"] == c))
                        for c in range(1, 5)
                    },
                },
            },
            out_root / "phase9_score_grid_plan.json",
        )

    if args.stage in ("10", "all"):
        # Seen-class / LOCO protocol plan + stop/go placeholder.
        shot_levels = [1, 2, 4, 8]
        loco_rows = []
        for shots in shot_levels:
            for held_out in range(1, 5):
                loco_rows.append(
                    {
                        "shots_per_class": shots,
                        "held_out_class": held_out,
                        "protocol": "leave_one_class_out",
                    }
                )
        stop_go = {
            "retain_anomaly_memory": None,
            "rule": (
                "Do not retain if seen-class gains require material decrease on "
                "unseen-class AUPRC or fixed-threshold recall."
            ),
            "status": "awaiting_gpu_results",
        }
        save_json(
            {
                "seen_class_shot_levels": shot_levels,
                "loco_runs": loco_rows,
                "stop_go": stop_go,
                "paired_delta_schema": paired_deltas(
                    [{"fold": 0, "seed": 42, "auprc": 0.1}],
                    [{"fold": 0, "seed": 42, "auprc": 0.2}],
                ),
            },
            out_root / "phase10_loco_protocol.json",
        )

    save_json(stage_results, out_root / "anomaly_memory_study_report.json")
    print(f"Wrote anomaly-memory artifacts under {out_root}")


if __name__ == "__main__":
    main()
