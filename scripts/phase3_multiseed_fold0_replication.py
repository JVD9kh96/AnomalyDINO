#!/usr/bin/env python3
"""Run and aggregate the paired, SAM2-free Phase 3 fold-0 replication.

By default this script only aggregates existing results.  Pass ``--run`` on a
CUDA-capable environment to execute the 20 prescribed runs sequentially, then
write the complete Phase 3 report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reference_replication import (
    PHASE3_CONDITIONS,
    build_phase3_report,
    phase3_run_dir,
    save_phase3_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_refbank/phase3")
    parser.add_argument("--config-dir", default="configs/reference_bank")
    parser.add_argument("--fold", type=int, default=0, choices=(0,))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--clean-shots", type=int, default=2)
    parser.add_argument("--additional-shots", type=int, default=8)
    parser.add_argument("--python", default=sys.executable, help="Python executable for study runs")
    parser.add_argument("--device", default=None, help="Optional detector device override")
    parser.add_argument(
        "--phase0-config",
        default="configs/phase0_paired_reference_manifest.yaml",
        help="Config used to freeze one Phase 0 manifest for each reference seed",
    )
    parser.add_argument("--run", action="store_true", help="Execute missing paired runs before aggregation")
    parser.add_argument("--rerun", action="store_true", help="Re-run completed outputs (otherwise reuse them)")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if len(set(seeds)) != len(seeds) or len(seeds) < 5:
        raise ValueError("Phase 3 requires at least five distinct seeds")
    return seeds


def resolve_from_repository(value: str) -> Path:
    """Resolve user-relative study paths against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_cuda_available(device: str | None) -> None:
    if device is not None and not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the DINO study") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Phase 3 uses the cuda:0 study configs; rerun on the GPU environment, or explicitly pass a supported CPU device."
        )


def run_matrix(args: argparse.Namespace, seeds: list[int]) -> list[dict]:
    ensure_cuda_available(args.device)
    output_root = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    phase0_config = resolve_from_repository(args.phase0_config)
    attempts: list[dict] = []
    for seed in seeds:
        manifest_path = output_root / f"phase3_fold0_seed{seed}_paired_manifest.json"
        if not manifest_path.is_file():
            manifest_command = [
                args.python,
                "scripts/phase0_freeze_paired_inputs.py",
                "--config",
                str(phase0_config),
                "--fold",
                str(args.fold),
                "--seed",
                str(seed),
                "--output",
                str(manifest_path),
            ]
            print("Freezing paired inputs: " + " ".join(manifest_command), flush=True)
            subprocess.run(manifest_command, cwd=ROOT, check=True)
        for condition in PHASE3_CONDITIONS:
            run_dir = phase3_run_dir(output_root, condition, seed)
            metrics = run_dir / "metrics.json"
            metadata = run_dir / "reference_metadata.json"
            if not args.rerun and metrics.is_file() and metadata.is_file():
                print(f"Reusing completed {condition}, seed {seed}: {run_dir}", flush=True)
                attempts.append({"condition": condition, "seed": seed, "status": "reused", "run_dir": str(run_dir)})
                continue
            command = [
                args.python,
                "scripts/run_reference_composition_study.py",
                "--config",
                str(config_dir / f"{condition}.yaml"),
                "--fold",
                str(args.fold),
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--condition",
                condition,
                "--clean-shots",
                str(args.clean_shots),
                "--additional-shots",
                str(args.additional_shots),
                "--output-dir",
                str(run_dir),
                "--paired-manifest",
                str(manifest_path),
                "--skip-sam2",
            ]
            if args.device is not None:
                command.extend(("--device", args.device))
            print("Running: " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            attempts.append({"condition": condition, "seed": seed, "status": "completed", "run_dir": str(run_dir), "command": command})
    return attempts


def main() -> None:
    args = parse_args()
    args.output_dir = str(resolve_from_repository(args.output_dir))
    args.config_dir = str(resolve_from_repository(args.config_dir))
    seeds = parse_seeds(args.seeds)
    attempts = run_matrix(args, seeds) if args.run else []
    report, rows, deltas = build_phase3_report(args.output_dir, seeds)
    if attempts:
        report["execution"] = attempts
    paths = save_phase3_report(args.output_dir, report, rows, deltas)
    print(f"Wrote Phase 3 report: {paths['report']}")
    print(f"Wrote individual runs: {paths['individual_runs']}")
    print(f"Wrote paired deltas: {paths['paired_deltas']}")
    for comparison, metrics in report["paired_delta_sign_tracking"].items():
        auprc = metrics["auprc"]
        print(
            f"{comparison} AUPRC mean={auprc['mean']:.6f} "
            f"signs(+/-/0)={auprc['positive']}/{auprc['negative']}/{auprc['zero']}"
        )


if __name__ == "__main__":
    main()
