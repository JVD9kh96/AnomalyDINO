#!/usr/bin/env python3
"""Aggregate reference-bank study metrics.json files into a summary table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Root directory of study runs")
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input-dir>/summary_table.csv)",
    )
    return p.parse_args()


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_metrics_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "contamination" in data and "patch" not in data:
            # contamination curve run — skip main table or flatten later
            continue
        if "patch" not in data:
            continue
        ref = data.get("reference") or {}
        bank = ref.get("bank_stats") or {}
        row = {
            "path": str(path.relative_to(root)),
            "method": data.get("condition") or ref.get("reference_mode"),
            "fold": data.get("fold"),
            "seed": data.get("seed"),
            "clean_shots": len(ref.get("clean_reference_ids") or []),
            "additional_images": len(ref.get("additional_reference_ids") or []),
            "gt_masks_used_in_fitting": data.get("gt_masks_used_in_fitting", False),
            "patch_auprc": _get(data, "patch", "auprc"),
            "patch_auroc": _get(data, "patch", "auroc"),
            "fixed_f1": _get(data, "patch", "fixed_threshold", "f1"),
            "f1_max": _get(data, "patch", "f1_optimal", "f1"),
            "fixed_precision": _get(data, "patch", "fixed_threshold", "precision"),
            "fixed_recall": _get(data, "patch", "fixed_threshold", "recall"),
            "sam2_dice": _get(data, "mask", "class_agnostic", "global", "dice"),
            "sam2_iou": _get(data, "mask", "class_agnostic", "global", "iou"),
            "memory_bank_size": data.get("memory_bank_size")
            or bank.get("final_memory_bank_size"),
            "fit_time_s": data.get("fit_time_s"),
            "mean_predict_time_s": data.get("mean_predict_time_s"),
            "acceptance_fraction": bank.get("acceptance_fraction"),
            "calibration_threshold": bank.get("calibration_threshold"),
        }
        # Per-class AUPRC if present
        per_class = _get(data, "patch", "per_class") or {}
        for class_id, block in per_class.items():
            row[f"class_{class_id}_auprc"] = block.get("auprc")
        rows.append(row)
    return rows


def compute_oracle_gaps(rows: list[dict]) -> list[dict]:
    """Pair oracle_purified vs auto_purified by fold+seed+shots."""
    gaps = []
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("fold"),
            row.get("seed"),
            row.get("clean_shots"),
            row.get("additional_images"),
            row.get("method"),
        )
        by_key[key] = row

    for row in rows:
        if row.get("method") != "auto_purified":
            continue
        oracle_key = (
            row.get("fold"),
            row.get("seed"),
            row.get("clean_shots"),
            row.get("additional_images"),
            "oracle_purified",
        )
        oracle = by_key.get(oracle_key)
        if not oracle:
            continue
        auto_auprc = row.get("patch_auprc")
        oracle_auprc = oracle.get("patch_auprc")
        if auto_auprc is None or oracle_auprc is None:
            continue
        gaps.append(
            {
                "fold": row.get("fold"),
                "seed": row.get("seed"),
                "clean_shots": row.get("clean_shots"),
                "additional_images": row.get("additional_images"),
                "oracle_auprc": oracle_auprc,
                "auto_auprc": auto_auprc,
                "oracle_gap_auprc": float(oracle_auprc) - float(auto_auprc),
                "oracle_f1_max": oracle.get("f1_max"),
                "auto_f1_max": row.get("f1_max"),
                "oracle_gap_f1_max": (
                    None
                    if oracle.get("f1_max") is None or row.get("f1_max") is None
                    else float(oracle["f1_max"]) - float(row["f1_max"])
                ),
            }
        )
    return gaps


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    out_csv = Path(args.output or root / "summary_table.csv")
    rows = load_metrics_files(root)
    if not rows:
        print(f"No metrics.json files found under {root}")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    # Prefer a readable column order
    preferred = [
        "method",
        "fold",
        "seed",
        "clean_shots",
        "additional_images",
        "gt_masks_used_in_fitting",
        "patch_auprc",
        "fixed_f1",
        "f1_max",
        "sam2_dice",
        "memory_bank_size",
        "path",
    ]
    ordered = [c for c in preferred if c in fieldnames] + [
        c for c in fieldnames if c not in preferred
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    gaps = compute_oracle_gaps(rows)
    gap_path = out_csv.with_name("oracle_gap.csv")
    if gaps:
        with open(gap_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(gaps[0].keys()))
            writer.writeheader()
            writer.writerows(gaps)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    if gaps:
        mean_gap = sum(g["oracle_gap_auprc"] for g in gaps) / len(gaps)
        print(f"Wrote {len(gaps)} oracle-gap rows to {gap_path} (mean AUPRC gap={mean_gap:.4f})")


if __name__ == "__main__":
    main()
