#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.reference_purification_metrics import (  # noqa: E402
    build_phase2_purification_report,
    compute_candidate_patch_overlaps,
    load_auto_rejection_bundle,
    save_phase2_purification_artifacts,
)
from src.evaluation.reference_manifest import (  # noqa: E402
    load_paired_reference_manifest,
)
from src.severstal.dataset import SeverstalDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: audit any/10%/50% oracle overlap and purification quality."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase2_oracle_purification_quality.yaml",
        help="Phase 2 YAML configuration.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override the frozen Phase 0 paired manifest.",
    )
    parser.add_argument(
        "--auto-rejections",
        default=None,
        help="Optional NPZ bundle of actual auto-purifier rejection masks.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the Phase 2 output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_cfg = config["data"]
    cv_cfg = config["cv"]
    phase_cfg = config["phase2"]
    output_cfg = config["output"]

    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=cv_cfg.get("split_seed", 42),
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
    )
    manifest_path = Path(args.manifest or phase_cfg["manifest"])
    manifest = load_paired_reference_manifest(manifest_path, dataset=dataset)

    resolution = int(phase_cfg.get("resolution", 448))
    patch_size = int(phase_cfg.get("patch_size", 14))
    records = [
        compute_candidate_patch_overlaps(
            dataset.load_sample(image_id),
            resolution=resolution,
            patch_size=patch_size,
            num_classes=data_cfg.get("num_classes", 4),
        )
        for image_id in manifest["additional_reference_ids"]
    ]

    auto_rejected = None
    auto_method = None
    if args.auto_rejections:
        auto_rejected, auto_method = load_auto_rejection_bundle(
            args.auto_rejections
        )

    report, distribution_arrays = build_phase2_purification_report(
        records,
        auto_rejected_by_image=auto_rejected,
        metadata={
            "phase0_manifest_id": manifest["manifest_id"],
            "fold": manifest["fold"],
            "seed": manifest["seed"],
            "resolution": resolution,
            "patch_size": patch_size,
            "auto_method": auto_method,
        },
    )
    output_dir = Path(
        args.output_dir
        or output_cfg.get("dir", "results_reference_composition/phase2")
    )
    report_path, distribution_path = save_phase2_purification_artifacts(
        report=report,
        distribution_arrays=distribution_arrays,
        output_dir=output_dir,
        report_filename=output_cfg.get(
            "report_filename",
            "phase2_purification_quality_report.json",
        ),
        distribution_filename=output_cfg.get(
            "distribution_filename",
            "phase2_patch_overlap_distribution.npz",
        ),
    )

    print("Phase 2 oracle overlap and purification audit created")
    print(f"  Report: {report_path}")
    print(f"  Raw overlap distribution: {distribution_path}")
    for rule, result in report["oracle_removal"].items():
        print(
            f"  {rule}: removed={result['n_removed_patches']} "
            f"({100.0 * result['removal_rate']:.2f}%)"
        )
    print(f"  Auto purification: {report['auto_status']}")


if __name__ == "__main__":
    main()
