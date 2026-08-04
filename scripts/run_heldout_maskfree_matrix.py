#!/usr/bin/env python3
"""Phase 12: frozen held-out mask-free matrix (folds 1–4)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.heldout_aggregation import (  # noqa: E402
    aggregate_heldout_matrix,
    paired_deltas,
)

FINAL_BUDGET = 51_200
FROZEN_TRIM = 0.20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase12_heldout_maskfree.yaml")
    p.add_argument("--output-dir", default="results_refbank/phase12")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default=None)
    p.add_argument("--run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-sam2", action="store_true")
    p.add_argument("--track", choices=("primary", "efficiency", "all"), default="all")
    return p.parse_args()


def frozen_primary_conditions() -> list[dict]:
    return [
        {"name": "clean_2", "condition": "clean", "clean_shots": 2, "additional_shots": 0, "budget": FINAL_BUDGET, "filter": "none"},
        {"name": "clean_8", "condition": "clean", "clean_shots": 8, "additional_shots": 0, "budget": FINAL_BUDGET, "filter": "none"},
        {"name": "naive_2plus8", "condition": "contaminated_all", "clean_shots": 2, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "none"},
        {"name": "random20_2plus8", "condition": "random_filtered", "clean_shots": 2, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "random20", "trim": FROZEN_TRIM},
        {"name": "proposed_distance20_2plus8", "condition": "fixed_ratio_trim", "clean_shots": 2, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "distance20", "trim": FROZEN_TRIM},
        {"name": "oracle_2plus8", "condition": "oracle_purified", "clean_shots": 2, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "oracle"},
    ]


def frozen_efficiency_conditions() -> list[dict]:
    return [
        {"name": "proposed_1plus8_natural", "condition": "fixed_ratio_trim", "clean_shots": 1, "additional_shots": 8, "budget": None, "filter": "distance20", "trim": FROZEN_TRIM},
        {"name": "proposed_2plus8_exact", "condition": "fixed_ratio_trim", "clean_shots": 2, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "distance20", "trim": FROZEN_TRIM},
        {"name": "proposed_4plus8_exact", "condition": "fixed_ratio_trim", "clean_shots": 4, "additional_shots": 8, "budget": FINAL_BUDGET, "filter": "distance20", "trim": FROZEN_TRIM},
        {"name": "clean_8", "condition": "clean", "clean_shots": 8, "additional_shots": 0, "budget": FINAL_BUDGET, "filter": "none"},
    ]


def validate_frozen_config(config: dict) -> None:
    """Reject keys that would retune the frozen fold-0 choices."""
    forbidden = {"retune_trim", "retune_percentile", "sweep_trim_fraction"}
    bad = forbidden.intersection(config.keys())
    if bad:
        raise ValueError(f"Frozen Phase-12 config rejects retune keys: {sorted(bad)}")
    phase12 = config.get("phase12", {})
    trim = phase12.get("trim_fraction", FROZEN_TRIM)
    if abs(float(trim) - FROZEN_TRIM) > 1e-12:
        raise ValueError(
            f"Phase-12 trim_fraction is frozen at {FROZEN_TRIM}, got {trim}"
        )
    if phase12.get("purification_mode", "fixed_ratio_trim") != "fixed_ratio_trim":
        raise ValueError("Phase-12 purification_mode is frozen to fixed_ratio_trim")


def _run(command: list[str], label: str) -> None:
    print(f"  [{label}] {' '.join(command)}", flush=True)
    subprocess.check_call(command, cwd=ROOT)


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_frozen_config(config)

    out_root = Path(args.output_dir)
    maskfree_root = out_root / "mask_free"
    oracle_root = out_root / "oracle"
    maskfree_root.mkdir(parents=True, exist_ok=True)
    oracle_root.mkdir(parents=True, exist_ok=True)

    conditions = []
    if args.track in ("primary", "all"):
        conditions.extend(frozen_primary_conditions())
    if args.track in ("efficiency", "all"):
        for row in frozen_efficiency_conditions():
            if row["name"] not in {c["name"] for c in conditions}:
                conditions.append(row)

    sam2_targets = {"clean_8", "naive_2plus8", "proposed_distance20_2plus8"}
    jobs = []
    for fold in args.folds:
        for seed in args.seeds:
            for spec in conditions:
                track_dir = oracle_root if spec["filter"] == "oracle" else maskfree_root
                run_dir = track_dir / f"f{fold}_s{seed}_{spec['name']}"
                jobs.append((fold, seed, spec, run_dir))

    if args.run:
        for fold, seed, spec, run_dir in jobs:
            if args.resume and (run_dir / "metrics.json").is_file():
                print(f"resume skip {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, "-u", "scripts/run_reference_composition_study.py",
                "--config", "configs/reference_bank/auto_purified.yaml",
                "--fold", str(fold), "--seed", str(seed),
                "--condition", spec["condition"],
                "--clean-shots", str(spec["clean_shots"]),
                "--additional-shots", str(spec["additional_shots"]),
                "--output-dir", str(run_dir),
            ]
            if not args.run_sam2 or spec["name"] not in sam2_targets:
                command.append("--skip-sam2")
            if spec.get("trim") is not None:
                command.extend(("--fixed-trim-fraction", str(spec["trim"])))
            if spec.get("budget") is not None:
                command.extend(
                    ("--coreset-size", str(spec["budget"]), "--budget-policy", "greedy_coreset")
                )
            if args.device:
                command.extend(("--device", args.device))
            _run(command, spec["name"])

    # Aggregate whatever metrics exist (GPU host can re-run after experiments).
    rows = []
    for fold, seed, spec, run_dir in jobs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "condition": spec["name"],
                "track": "oracle" if spec["filter"] == "oracle" else "mask_free",
                "auprc": metrics.get("patch", {}).get("auprc"),
                "auroc": metrics.get("patch", {}).get("auroc"),
                "fixed_f1": metrics.get("patch", {}).get("fixed_threshold", {}).get("f1"),
                "f1_max": metrics.get("patch", {}).get("f1_optimal", {}).get("f1"),
                "path": str(metrics_path),
            }
        )

    aggregate = aggregate_heldout_matrix(rows)
    # Example paired deltas: proposed vs naive when both present.
    proposed = [r for r in rows if r["condition"] == "proposed_distance20_2plus8"]
    naive = [r for r in rows if r["condition"] == "naive_2plus8"]
    report = {
        "phase": "phase12_heldout_maskfree",
        "frozen": {
            "purification_mode": "fixed_ratio_trim",
            "trim_fraction": FROZEN_TRIM,
            "budget": FINAL_BUDGET,
            "budget_policy": "greedy_coreset",
        },
        "folds": args.folds,
        "seeds": args.seeds,
        "run_sam2": bool(args.run_sam2),
        "n_rows_present": len(rows),
        "aggregate": aggregate,
        "paired_deltas_proposed_minus_naive": paired_deltas(naive, proposed),
        "tables": {
            "mask_free": [r for r in rows if r["track"] == "mask_free"],
            "oracle": [r for r in rows if r["track"] == "oracle"],
        },
        "notes": [
            "Do not retune trim/percentile on held-out folds.",
            "Bootstrap CIs use fold/seed pairs, not patch-IID samples.",
            "SAM2 only for clean-8 / naive-2+8 / proposed-2+8 when --run-sam2 is set.",
        ],
    }
    report_path = out_root / "phase12_heldout_maskfree_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report_path} ({len(rows)} completed rows)")


if __name__ == "__main__":
    main()
