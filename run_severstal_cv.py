import argparse
from pathlib import Path
import itertools
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
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Enable grid search over shots and k_neighbors.",
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

    if args.grid_search:
        # Grid search parameters
        shots_values = [4, 8, 16]
        k_neighbors_values = [1, 3, 5, 7]
        
        results_grid = {}
        total_combinations = len(shots_values) * len(k_neighbors_values)
        current = 0
        
        for shots, k_neighbors in itertools.product(shots_values, k_neighbors_values):
            current += 1
            print(f"\n{'='*60}")
            print(f"Grid Search: {current}/{total_combinations}")
            print(f"  shots: {shots}, k_neighbors: {k_neighbors}")
            print(f"{'='*60}")
            
            # Update config with current parameters
            config["detector"]["shots"] = shots
            config["detector"]["k_neighbors"] = k_neighbors
            
            results = run_cross_validation(
                config,
                fold_indices=fold_indices,
                config_path=args.config,
            )
            
            summary = results.get("summary", {})
            key = f"shots={shots}_k={k_neighbors}"
            results_grid[key] = summary
            
            print("\n=== Summary ===")
            for metric_key, val in summary.items():
                print(f"  {metric_key}: {val}")
        
        # Print overall results
        print(f"\n{'='*60}")
        print("=== Grid Search Results Summary ===")
        print(f"{'='*60}")
        for key, summary in results_grid.items():
            print(f"\n{key}:")
            for metric_key, val in summary.items():
                print(f"  {metric_key}: {val}")
    else:
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