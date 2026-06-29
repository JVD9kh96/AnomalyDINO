#!/usr/bin/env python3
"""Sweep patch score thresholds on a CV validation fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.evaluation.threshold_tuning import run_threshold_tuning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune pred_score_threshold on a validation fold."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML (e.g. configs/severstal.yaml)",
    )
    parser.add_argument("--fold", type=int, default=0, help="CV fold index")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_threshold_tuning",
        help="Root directory for tuning outputs",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=80,
        help="Number of threshold grid points",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.7,
        help="Target recall for alternative operating point",
    )
    parser.add_argument(
        "--with-sam2",
        action="store_true",
        help="Run SAM2 on a val subset at F1-opt vs recall@target thresholds",
    )
    parser.add_argument(
        "--sam2-max-images",
        type=int,
        default=10,
        help="Max val images for optional SAM2 preview",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_threshold_tuning(
        config=config,
        fold_idx=args.fold,
        output_dir=args.output_dir,
        n_thresholds=args.n_thresholds,
        target_recall=args.target_recall,
        with_sam2=args.with_sam2,
        sam2_max_images=args.sam2_max_images,
    )


if __name__ == "__main__":
    main()
