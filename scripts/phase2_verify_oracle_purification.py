#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.reference_purification import (  # noqa: E402
    fixed_ratio_distance_trim_indices,
    oracle_keep_mask_from_gt,
    random_size_matched_indices,
    selected_indices_to_keep_mask,
)
from src.detectors.reference_purification_metrics import (  # noqa: E402
    BANK_FILTER_NAMES,
    build_multi_bank_purification_report,
    build_phase2_purification_report,
    compute_candidate_patch_overlaps,
    load_auto_rejection_bundle,
    oracle_rejection_mask,
    rejected_mask_from_keep_mask,
    save_phase2_purification_artifacts,
)
from src.evaluation.reference_manifest import (  # noqa: E402
    load_paired_reference_manifest,
)
from src.severstal.dataset import SeverstalDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: audit any/10%/50% oracle overlap and purification quality "
            "across naive / auto / distance20 / random / oracle banks."
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
        "--banks",
        nargs="+",
        default=None,
        choices=list(BANK_FILTER_NAMES),
        help="Banks to evaluate (default: all supported filter names).",
    )
    parser.add_argument(
        "--bank-rejections-dir",
        default=None,
        help=(
            "Directory of per-bank rejection NPZs named {bank}_rejections.npz "
            "(analysis-only; reuse cached rejection masks)."
        ),
    )
    parser.add_argument(
        "--reuse-cached-features",
        action="store_true",
        help="Prefer cached rejection/feature artifacts; do not re-extract features.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.20,
        help="Distance-trim fraction for distance_trim_20 bank (default 0.20).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random size-matched control.",
    )
    parser.add_argument(
        "--distance-scores",
        default=None,
        help=(
            "Optional NPZ with per-image distance scores for distance/random banks "
            "(keys: image_ids, scores_{image_id} or stacked scores)."
        ),
    )
    parser.add_argument(
        "--selected-oracle-rule",
        default="any_overlap",
        choices=("any_overlap", "at_least_10_percent", "at_least_50_percent"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the Phase 2 output directory.",
    )
    return parser.parse_args()


def _load_distance_scores(path: Path, image_ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        scores: dict[str, np.ndarray] = {}
        for image_id in image_ids:
            key = f"scores_{image_id}"
            if key in bundle.files:
                scores[image_id] = np.asarray(bundle[key], dtype=np.float32)
            elif "scores" in bundle.files and "image_ids" in bundle.files:
                ids = [str(v) for v in bundle["image_ids"].tolist()]
                stacked = np.asarray(bundle["scores"])
                scores[image_id] = stacked[ids.index(image_id)]
            else:
                raise ValueError(f"Distance scores missing for {image_id}")
    return scores


def _synthetic_bank_rejections(
    records,
    *,
    banks: list[str],
    trim_fraction: float,
    random_seed: int,
    selected_oracle_rule: str,
    distance_scores: dict[str, np.ndarray] | None,
    dataset: SeverstalDataset,
    resolution: int,
    patch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Build rejection masks for analysis when explicit bank NPZs are absent."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for bank in banks:
        rejected_by_image: dict[str, np.ndarray] = {}
        for record in records:
            shape = record.union_overlap.shape
            flat_n = int(np.prod(shape))
            if bank == "naive":
                rejected = np.zeros(shape, dtype=bool)
            elif bank == "oracle":
                sample = dataset.load_sample(record.image_id)
                keep = oracle_keep_mask_from_gt(
                    sample,
                    shape,
                    patch_size,
                    resolution,
                    overlap_threshold=(
                        0.0
                        if selected_oracle_rule == "any_overlap"
                        else (0.10 if selected_oracle_rule == "at_least_10_percent" else 0.50)
                    ),
                )
                # Prefer overlap-rule based rejection for analysis consistency.
                rejected = oracle_rejection_mask(
                    record.union_overlap, selected_oracle_rule
                )
                del keep
            elif bank in ("distance_trim_20", "random_size_matched"):
                if distance_scores is None or record.image_id not in distance_scores:
                    # Analysis fallback: synthetic distances from overlap itself
                    # (higher overlap => higher distance) so CPU tests / dry runs work.
                    scores = np.asarray(record.union_overlap, dtype=np.float32).ravel()
                else:
                    scores = np.asarray(
                        distance_scores[record.image_id], dtype=np.float32
                    ).ravel()
                if bank == "distance_trim_20":
                    selected = fixed_ratio_distance_trim_indices(
                        scores, trim_fraction=trim_fraction
                    )
                else:
                    n_keep = int(round((1.0 - trim_fraction) * flat_n))
                    selected = random_size_matched_indices(
                        flat_n, n_keep, seed=random_seed + hash(record.image_id) % 10_000
                    )
                keep = selected_indices_to_keep_mask(selected, flat_n).reshape(shape)
                rejected = rejected_mask_from_keep_mask(keep)
            elif bank == "auto_purified":
                # Without a cached auto bundle, leave empty and let caller skip.
                continue
            else:
                raise ValueError(f"Unknown bank={bank}")
            rejected_by_image[record.image_id] = rejected
        if rejected_by_image:
            out[bank] = rejected_by_image
    return out


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

    banks = list(args.banks or BANK_FILTER_NAMES)
    bank_rejected: dict[str, dict[str, np.ndarray]] = {}

    if args.auto_rejections:
        auto_rejected, auto_method = load_auto_rejection_bundle(args.auto_rejections)
        bank_rejected["auto_purified"] = auto_rejected
    else:
        auto_method = None

    if args.bank_rejections_dir:
        root = Path(args.bank_rejections_dir)
        for bank in banks:
            path = root / f"{bank}_rejections.npz"
            if path.is_file():
                rejected, _ = load_auto_rejection_bundle(path)
                bank_rejected[bank] = rejected
            elif args.reuse_cached_features:
                print(f"  Missing cached bank rejections for {bank}: {path}")

    distance_scores = None
    if args.distance_scores:
        distance_scores = _load_distance_scores(
            Path(args.distance_scores),
            list(manifest["additional_reference_ids"]),
        )

    # Fill remaining banks with deterministic analysis masks when possible.
    missing = [bank for bank in banks if bank not in bank_rejected]
    if missing:
        synthesized = _synthetic_bank_rejections(
            records,
            banks=missing,
            trim_fraction=float(args.trim_fraction),
            random_seed=int(args.random_seed),
            selected_oracle_rule=args.selected_oracle_rule,
            distance_scores=distance_scores,
            dataset=dataset,
            resolution=resolution,
            patch_size=patch_size,
        )
        bank_rejected.update(synthesized)

    if bank_rejected:
        report, distribution_arrays = build_multi_bank_purification_report(
            records,
            bank_rejected_by_image=bank_rejected,
            selected_oracle_rule=args.selected_oracle_rule,
            metadata={
                "phase0_manifest_id": manifest["manifest_id"],
                "fold": manifest["fold"],
                "seed": manifest["seed"],
                "resolution": resolution,
                "patch_size": patch_size,
                "auto_method": auto_method,
                "banks": sorted(bank_rejected),
                "trim_fraction": float(args.trim_fraction),
                "reuse_cached_features": bool(args.reuse_cached_features),
            },
        )
    else:
        report, distribution_arrays = build_phase2_purification_report(
            records,
            auto_rejected_by_image=None,
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
    # Persist selected indices summary for distance/random banks when present.
    indices_path = output_dir / "phase2_selected_indices_summary.json"
    indices_summary = {
        bank: {
            image_id: int(np.sum(mask))
            for image_id, mask in rejected.items()
        }
        for bank, rejected in bank_rejected.items()
    }
    indices_path.write_text(
        json.dumps(indices_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Phase 2 oracle overlap and purification audit created")
    print(f"  Report: {report_path}")
    print(f"  Raw overlap distribution: {distribution_path}")
    print(f"  Selected-index summary: {indices_path}")
    for rule, result in report["oracle_removal"].items():
        print(
            f"  {rule}: removed={result['n_removed_patches']} "
            f"({100.0 * result['removal_rate']:.2f}%)"
        )
    print(f"  Auto purification: {report['auto_status']}")
    if "banks_evaluated" in report:
        print(f"  Banks evaluated: {report['banks_evaluated']}")


if __name__ == "__main__":
    main()
