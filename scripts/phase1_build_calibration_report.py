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
    DEFAULT_CALIBRATION_REPORT_FILENAME,
    DEFAULT_QUANTILES,
    assert_tau_query_matches_bank,
    build_report_from_score_bundle,
    discover_phase1_score_bundles,
    final_bank_hash,
    load_phase1_score_bundle,
    save_phase1_calibration_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: build the fixed-threshold calibration report from "
            "cross-fitted clean scores and final-bank validation scores. "
            "Supports analysis-only backfill from cached score NPZs."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase1_fixed_threshold_calibration.yaml",
        help="Phase 1 YAML configuration.",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="NPZ score bundle written after the final bank was constructed.",
    )
    parser.add_argument(
        "--backfill-root",
        default=None,
        help=(
            "Scan this results tree for cached phase1 score NPZs and emit a "
            "calibration_report.json beside each bundle (analysis-only)."
        ),
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
    parser.add_argument(
        "--filename",
        default=None,
        help="Report filename (default: calibration_report.json).",
    )
    return parser.parse_args()


def _load_metadata(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _emit_from_bundle(
    *,
    scores_path: Path,
    phase_cfg: dict,
    output_cfg: dict,
    metadata: dict,
    output_dir: Path | None,
    filename: str,
) -> Path:
    bundle = load_phase1_score_bundle(scores_path)
    composition = metadata.get("bank_composition_parts")
    report = build_report_from_score_bundle(
        bundle,
        candidate_acceptance_percentile=float(
            phase_cfg.get("candidate_acceptance_percentile", 99.0)
        ),
        query_percentile=float(phase_cfg.get("query_percentile", 99.5)),
        quantiles=phase_cfg.get("score_quantiles", DEFAULT_QUANTILES),
        metadata={
            **metadata,
            "score_bundle_path": str(scores_path.resolve()),
            "analysis_only_backfill": True,
        },
        exclusion_mode=str(
            metadata.get("exclusion_mode", phase_cfg.get("exclusion_mode", "leave_one_image_out"))
        ),
        bank_composition_parts=composition,
    )
    expected_hash = final_bank_hash(
        bundle["final_bank_id"],
        *(composition or ()),
    )
    assert_tau_query_matches_bank(report, expected_final_bank_hash=expected_hash)

    run_dir = output_dir or scores_path.parent
    if output_dir is None and output_cfg.get("dir") and not scores_path.parent.exists():
        run_dir = Path(output_cfg["dir"])
    return save_phase1_calibration_report(report, run_dir, filename=filename)


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    phase_cfg = config["phase1"]
    output_cfg = config["output"]
    filename = args.filename or output_cfg.get(
        "filename", DEFAULT_CALIBRATION_REPORT_FILENAME
    )
    metadata = _load_metadata(args.metadata)

    if args.backfill_root:
        bundles = discover_phase1_score_bundles(args.backfill_root)
        if not bundles:
            raise SystemExit(
                f"No Phase-1 score bundles found under {args.backfill_root}"
            )
        written = []
        for scores_path in bundles:
            path = _emit_from_bundle(
                scores_path=scores_path,
                phase_cfg=phase_cfg,
                output_cfg=output_cfg,
                metadata=metadata,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                filename=filename,
            )
            written.append(path)
            print(f"  Backfilled {path}")
        print(f"Phase 1 backfill complete: {len(written)} report(s)")
        return

    if not args.scores:
        raise SystemExit("Provide --scores or --backfill-root")

    output_path = _emit_from_bundle(
        scores_path=Path(args.scores),
        phase_cfg=phase_cfg,
        output_cfg=output_cfg,
        metadata=metadata,
        output_dir=Path(
            args.output_dir
            or output_cfg.get("dir", "results_reference_composition/phase1")
        ),
        filename=filename,
    )
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

    print("Phase 1 fixed-threshold calibration report created")
    print(f"  Report: {output_path}")
    print(f"  tau_accept: {report['candidate_acceptance']['threshold']:.8g}")
    print(f"  tau_query: {report['query_operating_point']['threshold']:.8g}")
    print(f"  final_bank_hash: {report['final_bank_hash']}")
    print(f"  F1-max threshold: {report['f1_max_threshold']:.8g}")
    print(
        "  Fixed operating point: "
        f"P={report['fixed_metrics']['precision']:.4f} "
        f"R={report['fixed_metrics']['recall']:.4f} "
        f"F1={report['fixed_metrics']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
