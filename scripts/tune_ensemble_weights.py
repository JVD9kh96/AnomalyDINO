#!/usr/bin/env python3
"""Tune ensemble sub-detector weights on one fold; benchmark on hold-out folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.evaluation.threshold_tuning import tune_ensemble_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ensemble detector weights.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ensemble config YAML (e.g. configs/severstal_dino_ensemble.yaml)",
    )
    parser.add_argument("--tune-fold", type=int, default=0)
    parser.add_argument(
        "--benchmark-folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Hold-out folds for reporting (exclude tune-fold)",
    )
    parser.add_argument("--weight-steps", type=int, default=11)
    parser.add_argument("--target-recall", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_ensemble_tuning",
    )
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tune_ensemble_weights(
        config=config,
        tune_fold=args.tune_fold,
        benchmark_folds=args.benchmark_folds,
        weight_steps=args.weight_steps,
        target_recall=args.target_recall,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
