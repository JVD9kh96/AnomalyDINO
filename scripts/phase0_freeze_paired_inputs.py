#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reference_manifest import (  # noqa: E402
    build_paired_reference_manifest,
    save_paired_reference_manifest,
)
from src.severstal.dataset import SeverstalDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 0: freeze paired clean/additional reference IDs and validate "
            "that no validation image enters either pool."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase0_paired_reference_manifest.yaml",
        help="Phase 0 YAML configuration.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Override phase0.fold.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "--clean-shots", type=int, default=None, help="Override phase0.clean_shots."
    )
    parser.add_argument(
        "--additional-shots", type=int, default=None, help="Override phase0.additional_shots."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the exact output manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_cfg = config["data"]
    cv_cfg = config["cv"]
    phase_cfg = config["phase0"]
    output_cfg = config["output"]

    fold = int(phase_cfg.get("fold", 0) if args.fold is None else args.fold)
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    clean_shots = int(
        phase_cfg.get("clean_shots", 2)
        if args.clean_shots is None
        else args.clean_shots
    )
    additional_shots = int(
        phase_cfg.get("additional_shots", 8)
        if args.additional_shots is None
        else args.additional_shots
    )

    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=cv_cfg.get("split_seed", config.get("seed", 42)),
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
    )
    manifest = build_paired_reference_manifest(
        dataset,
        fold=fold,
        seed=seed,
        clean_shots=clean_shots,
        additional_shots=additional_shots,
        additional_sampling=phase_cfg.get(
            "additional_sampling", "class_balanced"
        ),
        resolution=int(phase_cfg.get("resolution", 448)),
        patch_size=int(phase_cfg.get("patch_size", 14)),
    )

    if args.output:
        output_path = Path(args.output)
    else:
        filename = output_cfg.get(
            "filename", "phase0_fold{fold}_seed{seed}_paired_manifest.json"
        ).format(fold=fold, seed=seed)
        output_dir = Path(
            output_cfg.get("dir", "results_reference_composition/phase0")
        )
        output_path = output_dir / filename

    saved_path = save_paired_reference_manifest(manifest, output_path)
    defect_candidates = sum(manifest["additional_reference_has_defect"].values())
    print("Phase 0 paired inputs frozen")
    print(f"  Manifest: {saved_path}")
    print(f"  Manifest ID: {manifest['manifest_id']}")
    print(f"  Fold / seed: {fold} / {seed}")
    print(f"  Clean references: {len(manifest['clean_reference_ids'])}")
    print(f"  Additional references: {len(manifest['additional_reference_ids'])}")
    print(f"  Defect-bearing candidates: {defect_candidates}")
    print(f"  Candidate patches: {manifest['candidate_patch_count_total']}")
    print("  Leakage check: passed")


if __name__ == "__main__":
    main()
