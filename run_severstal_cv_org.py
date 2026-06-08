import argparse
from pathlib import Path

import yaml

from src.evaluation.cross_validation import load_config, run_cross_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Severstal steel defect detection cross-validation evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/severstal.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single fold only (0-indexed). Default: all folds.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data.root from config.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Override detector.name from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data_root:
        config.setdefault("data", {})["root"] = args.data_root
    if args.detector:
        config.setdefault("detector", {})["name"] = args.detector

    fold_indices = [args.fold] if args.fold is not None else None

    print("Severstal CV Evaluation")
    print(f"  Config: {args.config}")
    print(f"  Data root: {config['data']['root']}")
    print(f"  Folds: {fold_indices if fold_indices else 'all'}")
    print(f"  Detector: {config['detector']['name']}")

    results = run_cross_validation(
        config,
        fold_indices=fold_indices,
        config_path=args.config,
    )

    summary = results.get("summary", {})
    print("\n=== Summary ===")
    for key, val in summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
