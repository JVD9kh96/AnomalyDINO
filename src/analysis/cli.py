from __future__ import annotations

import argparse

from src.analysis.anomaly_distribution import run_analysis
from src.analysis.config import DEFAULT_SCORERS, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINO patch signal distribution analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/analysis_severstal.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default=None,
        help="Run a single scorer (overrides config scorers list).",
    )
    parser.add_argument(
        "--all-scorers",
        action="store_true",
        help="Run all built-in scorers.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset.root from config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir from config.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Override layers: last, all, or comma-separated indices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images (debug).",
    )
    return parser.parse_args()


def _parse_layers(value: str):
    if value in ("last", "all"):
        return value
    return [int(x.strip()) for x in value.split(",")]


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data_root:
        config.dataset.root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.max_images is not None:
        config.dataset.max_images = args.max_images
    if args.layers:
        config.layers = _parse_layers(args.layers)
    if args.all_scorers:
        config.scorers = list(DEFAULT_SCORERS)
    elif args.scorer:
        config.scorers = [args.scorer]

    run_analysis(config)


if __name__ == "__main__":
    main()
