#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.calibration_report import (  # noqa: E402
    DEFAULT_QUANTILES,
    build_report_from_score_bundle,
    load_phase1_score_bundle,
    save_phase1_calibration_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: build the fixed-threshold calibration report from "
            "cross-fitted clean scores and final-bank validation scores."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase1_fixed_threshold_calibration.yaml",
        help="Phase 1 YAML configuration.",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="NPZ score bundle written after the final bank was constructed.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file with fold, seed, mode, manifest ID, and bank size.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the Phase 1 run output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    phase_cfg = config["phase1"]
    output_cfg = config["output"]

    metadata = {}
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as file:
            metadata = json.load(file)

    bundle = load_phase1_score_bundle(args.scores)
    report = build_report_from_score_bundle(
        bundle,
        candidate_acceptance_percentile=float(
            phase_cfg.get("candidate_acceptance_percentile", 99.0)
        ),
        query_percentile=float(phase_cfg.get("query_percentile", 99.5)),
        quantiles=phase_cfg.get("score_quantiles", DEFAULT_QUANTILES),
        metadata=metadata,
    )
    run_dir = Path(
        args.output_dir
        or output_cfg.get("dir", "results_reference_composition/phase1")
    )
    output_path = save_phase1_calibration_report(
        report,
        run_dir,
        filename=output_cfg.get("filename", "phase1_calibration_report.json"),
    )

    fixed = report["query_operating_point"]
    print("Phase 1 fixed-threshold calibration report created")
    print(f"  Report: {output_path}")
    print(f"  tau_accept: {report['candidate_acceptance']['threshold']:.8g}")
    print(f"  tau_query: {fixed['threshold']:.8g}")
    print(f"  F1-max threshold: {report['f1_max_threshold']:.8g}")
    print(
        "  Fixed operating point: "
        f"P={report['precision']:.4f} R={report['recall']:.4f} "
        f"F1={report['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
