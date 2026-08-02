#!/usr/bin/env python3
"""Run and summarize Phase 4 compact purification controls on fold 0 only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_refbank/phase4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--phase3-clean-metrics",
        default="results_refbank/phase3/f0_clean_s42/metrics.json",
        help="Paired Phase 3 clean baseline used only for fixed-threshold comparison",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def specs() -> list[dict]:
    controls = []
    for percentile in (95.0, 97.5, 99.0, 99.5):
        tag = str(percentile).replace(".5", "_5").replace(".0", "")
        controls.append(
            {"name": f"auto_p{tag}", "condition": "auto_purified", "percentile": percentile}
        )
        controls.append(
            {"name": f"random_matched_p{tag}", "condition": "random_filtered", "percentile": percentile}
        )
    for trim_fraction in (0.05, 0.10, 0.20):
        controls.append(
            {
                "name": f"trim_{int(trim_fraction * 100)}pct",
                "condition": "fixed_ratio_trim",
                "trim_fraction": trim_fraction,
            }
        )
    return controls


def _ensure_cuda(device: str | None) -> None:
    if device is not None and not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run Phase 4 in the GPU screen session.")


def _run(command: list[str], label: str) -> None:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        try:
            code = process.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            print(f"  [{label}] still running ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)
    print(f"  [{label}] completed in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _metric_row(spec: dict, metrics_path: Path, clean_metrics: dict) -> dict:
    metrics = _load_json(metrics_path)
    patch = metrics["patch"]
    reference = metrics["reference"]
    clean_reference = clean_metrics["reference"]
    if metrics.get("fold") != 0 or metrics.get("seed") != 42:
        raise ValueError(f"Phase 4 result has wrong fold/seed: {metrics_path}")
    if metrics.get("split_seed") != clean_metrics.get("split_seed"):
        raise ValueError(f"Phase 4 result has a different split seed: {metrics_path}")
    if reference.get("paired_manifest_id") != clean_reference.get("paired_manifest_id"):
        raise ValueError(f"Phase 4 result does not use the paired clean manifest: {metrics_path}")
    bank = reference.get("bank_stats", {})
    fixed = patch.get("fixed_threshold", {})
    clean_fixed = clean_metrics["patch"].get("fixed_threshold", {})
    row = {
        **spec,
        "path": str(metrics_path),
        "auprc": patch.get("auprc"),
        "auroc": patch.get("auroc"),
        "fixed_f1": fixed.get("f1"),
        "fixed_precision": fixed.get("precision"),
        "fixed_recall": fixed.get("recall"),
        "f1_max": patch.get("f1_optimal", {}).get("f1"),
        "query_threshold": metrics.get("pred_score_threshold"),
        "query_threshold_delta_vs_clean": abs(
            float(metrics.get("pred_score_threshold", 0.0))
            - float(clean_metrics.get("pred_score_threshold", 0.0))
        ),
        "fixed_f1_delta_vs_clean": float(fixed.get("f1", 0.0))
        - float(clean_fixed.get("f1", 0.0)),
        "candidate_retained": bank.get("n_accepted_candidate_patches"),
        "candidate_rejected": bank.get("n_rejected_candidate_patches"),
        "candidate_retention_fraction": bank.get("acceptance_fraction"),
        "filter_extras": bank.get("extras", {}),
    }
    return row


def aggregate(output_dir: Path, clean_metrics_path: Path) -> Path:
    clean_metrics = _load_json(clean_metrics_path)
    if clean_metrics.get("fold") != 0 or clean_metrics.get("seed") != 42:
        raise ValueError("Phase 4 requires the saved fold-0, seed-42 clean baseline")
    rows = []
    for spec in specs():
        metrics_path = output_dir / f"phase4_f0_{spec['name']}_s42" / "metrics.json"
        if not metrics_path.is_file():
            raise ValueError(f"Missing Phase 4 result: {metrics_path}")
        rows.append(_metric_row(spec, metrics_path, clean_metrics))

    # Random filtering is a negative control, not a tuning candidate.  Rank
    # automatic and fixed-ratio settings by AUPRC first, then threshold
    # stability relative to the paired clean run, then fixed-threshold F1.
    candidates = [row for row in rows if row["condition"] != "random_filtered"]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["auprc"]),
            float(row["query_threshold_delta_vs_clean"]),
            -float(row["fixed_f1"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
    report = {
        "phase": "phase4_compact_purification_controls",
        "fold": 0,
        "seed": 42,
        "sam2_skipped": True,
        "spatial_cleanup": False,
        "clean_baseline_metrics": str(clean_metrics_path),
        "controls": rows,
        "selection_candidates_ranked": ranked,
        "recommended_setting": ranked[0],
        "selection_criterion": (
            "Rank non-random settings by AUPRC descending, then smaller query-threshold "
            "change versus paired clean, then fixed-threshold F1 descending. F1-max is "
            "reported but never used for selection."
        ),
    }
    path = output_dir / "phase4_compact_purification_controls_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace, output_dir: Path) -> None:
    _ensure_cuda(args.device)
    manifest = output_dir / f"phase4_fold0_seed{args.seed}_paired_manifest.json"
    if not manifest.is_file():
        command = [
            args.python, "-u", "scripts/phase0_freeze_paired_inputs.py", "--config",
            "configs/phase0_paired_reference_manifest.yaml", "--fold", "0", "--seed",
            str(args.seed), "--output", str(manifest),
        ]
        print("[Phase 4] freezing paired input manifest", flush=True)
        _run(command, "manifest")
    total = len(specs())
    for index, spec in enumerate(specs(), start=1):
        run_dir = output_dir / f"phase4_f0_{spec['name']}_s{args.seed}"
        if not args.rerun and (run_dir / "metrics.json").is_file():
            print(f"[run {index}/{total}: {spec['name']}] reusing completed output", flush=True)
            continue
        command = [
            args.python, "-u", "scripts/run_reference_composition_study.py", "--config",
            "configs/reference_bank/auto_purified.yaml", "--fold", "0", "--seed", str(args.seed),
            "--split-seed", str(args.split_seed), "--condition", spec["condition"],
            "--clean-shots", "2", "--additional-shots", "8", "--output-dir", str(run_dir),
            "--paired-manifest", str(manifest), "--skip-sam2",
        ]
        if "percentile" in spec:
            command.extend(("--acceptance-percentile", str(spec["percentile"])))
        if "trim_fraction" in spec:
            command.extend(("--fixed-trim-fraction", str(spec["trim_fraction"])))
        if args.device is not None:
            command.extend(("--device", args.device))
        print(f"[run {index}/{total}: {spec['name']}] starting", flush=True)
        _run(command, spec["name"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.seed != 42 or args.split_seed != 42:
        raise ValueError("Phase 4 is intentionally restricted to fold 0, seed 42, split seed 42")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = resolve(args.phase3_clean_metrics)
    if args.run:
        run(args, output_dir)
    report = aggregate(output_dir, baseline)
    print(f"Wrote Phase 4 report: {report}")


if __name__ == "__main__":
    main()
