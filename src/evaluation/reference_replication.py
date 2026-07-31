"""Utilities for strict, paired reference-bank replication reports.

This module deliberately operates on the ``metrics.json`` and
``reference_metadata.json`` files produced by the study runner.  Keeping the
aggregation independent of model code makes the Phase 3 report reproducible
without re-running DINO feature extraction.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PHASE3_CONDITIONS = (
    "clean",
    "contaminated_all",
    "auto_purified",
    "oracle_purified",
)
EXPANDED_CONDITIONS = PHASE3_CONDITIONS[1:]
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "auprc": ("patch", "auprc"),
    "auroc": ("patch", "auroc"),
    "f1_max": ("patch", "f1_optimal", "f1"),
    "fixed_f1": ("patch", "fixed_threshold", "f1"),
    "fixed_precision": ("patch", "fixed_threshold", "precision"),
    "fixed_recall": ("patch", "fixed_threshold", "recall"),
    "memory_bank_size": ("memory_bank_size",),
}


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def phase3_run_dir(root: str | Path, condition: str, seed: int) -> Path:
    """Return the canonical Phase 3 output directory for one paired run."""
    if condition not in PHASE3_CONDITIONS:
        raise ValueError(f"Unsupported Phase 3 condition: {condition}")
    return Path(root) / f"f0_{condition}_s{int(seed)}"


def load_phase3_run(root: str | Path, condition: str, seed: int) -> dict[str, Any]:
    """Load one completed run and flatten its reportable values."""
    run_dir = phase3_run_dir(root, condition, seed)
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "reference_metadata.json"
    if not metrics_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"Missing Phase 3 outputs for {condition}, seed {seed}: "
            f"need {metrics_path.name} and {metadata_path.name} in {run_dir}"
        )
    metrics = _load_json(metrics_path)
    metadata = _load_json(metadata_path)
    if metrics.get("condition") != condition:
        raise ValueError(
            f"{metrics_path} records condition={metrics.get('condition')!r}, "
            f"expected {condition!r}"
        )
    if int(metrics.get("fold", -1)) != 0 or int(metrics.get("seed", -1)) != int(seed):
        raise ValueError(f"Unexpected fold/seed in {metrics_path}")
    if metrics.get("split_seed") is None:
        raise ValueError(f"Missing split_seed provenance in {metrics_path}")
    if metrics.get("sam2_skipped") is not True or metrics.get("mask") not in (None, {}):
        raise ValueError(
            f"{metrics_path} does not prove a SAM2-free run; Phase 3 early sweep "
            "must be run with --skip-sam2."
        )

    reference = metrics.get("reference")
    if reference != metadata:
        raise ValueError(
            f"Reference metadata mismatch between metrics.json and "
            f"reference_metadata.json for {condition}, seed {seed}"
        )
    manifest_id = metadata.get("paired_manifest_id")
    if not manifest_id:
        raise ValueError(f"Missing paired_manifest_id in {metadata_path}")
    row: dict[str, Any] = {
        "condition": condition,
        "fold": 0,
        "seed": int(seed),
        "split_seed": int(metrics["split_seed"]),
        "run_dir": str(run_dir),
        "clean_reference_ids": list(metadata.get("clean_reference_ids") or []),
        "additional_reference_ids": list(metadata.get("additional_reference_ids") or []),
        "n_clean_reference_images": len(metadata.get("clean_reference_ids") or []),
        "n_additional_reference_images": len(
            metadata.get("additional_reference_ids") or []
        ),
        "sam2_skipped": True,
        "paired_manifest_id": str(manifest_id),
    }
    for name, path in METRIC_PATHS.items():
        row[name] = _as_float(_nested_value(metrics, path))
    return row


def validate_paired_ids(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    """Assert shared clean/additional IDs for every requested seed.

    Clean IDs must match across all four conditions.  Additional IDs must
    match across the three expanded-bank conditions; clean intentionally has
    no additional IDs.
    """
    validations: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows = {row["condition"]: row for row in rows if row["seed"] == seed}
        missing = set(PHASE3_CONDITIONS) - set(seed_rows)
        if missing:
            raise ValueError(f"Seed {seed} lacks conditions: {sorted(missing)}")
        clean_ids = seed_rows["clean"]["clean_reference_ids"]
        for condition in PHASE3_CONDITIONS[1:]:
            if seed_rows[condition]["clean_reference_ids"] != clean_ids:
                raise ValueError(
                    f"Paired clean-reference ID failure for seed {seed}: clean and "
                    f"{condition} selected different IDs"
                )
        additional_ids = seed_rows["contaminated_all"]["additional_reference_ids"]
        manifest_ids = {seed_rows[condition]["paired_manifest_id"] for condition in PHASE3_CONDITIONS}
        if len(manifest_ids) != 1:
            raise ValueError(f"Paired-manifest ID failure for seed {seed}")
        for condition in EXPANDED_CONDITIONS[1:]:
            if seed_rows[condition]["additional_reference_ids"] != additional_ids:
                raise ValueError(
                    f"Paired additional-reference ID failure for seed {seed}: "
                    f"contaminated_all and {condition} selected different IDs"
                )
        if seed_rows["clean"]["additional_reference_ids"]:
            raise ValueError(f"Clean condition unexpectedly has additional IDs for seed {seed}")
        validations.append(
            {
                "seed": seed,
                "paired_clean_ids": True,
                "paired_additional_ids": True,
                "clean_reference_ids": clean_ids,
                "additional_reference_ids": additional_ids,
                "paired_manifest_id": next(iter(manifest_ids)),
            }
        )
    return validations


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "median": statistics.median(values) if values else None,
    }


def _delta_sign(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def build_phase3_report(root: str | Path, seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the complete Phase 3 report from the 4 × N run matrix."""
    unique_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique_seeds) < 5:
        raise ValueError("Phase 3 requires at least five reference seeds")
    rows = [
        load_phase3_run(root, condition, seed)
        for seed in unique_seeds
        for condition in PHASE3_CONDITIONS
    ]
    paired_validation = validate_paired_ids(rows, unique_seeds)
    split_seeds = {row["split_seed"] for row in rows}
    if len(split_seeds) != 1:
        raise ValueError(f"Phase 3 runs use different validation split seeds: {sorted(split_seeds)}")

    condition_summary: dict[str, dict[str, Any]] = {}
    for condition in PHASE3_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        metrics: dict[str, Any] = {}
        for metric in METRIC_PATHS:
            by_seed = {
                str(row["seed"]): row[metric]
                for row in condition_rows
                if row[metric] is not None
            }
            metrics[metric] = {**_summary(list(by_seed.values())), "values_by_seed": by_seed}
        condition_summary[condition] = metrics

    paired_deltas: list[dict[str, Any]] = []
    sign_tracking: dict[str, dict[str, Any]] = {}
    comparisons = {
        "naive_minus_clean": ("contaminated_all", "clean"),
        "auto_minus_naive": ("auto_purified", "contaminated_all"),
    }
    for comparison, (numerator, denominator) in comparisons.items():
        comparison_rows: list[dict[str, Any]] = []
        for seed in unique_seeds:
            left = next(row for row in rows if row["seed"] == seed and row["condition"] == numerator)
            right = next(row for row in rows if row["seed"] == seed and row["condition"] == denominator)
            for metric in METRIC_PATHS:
                if left[metric] is None or right[metric] is None:
                    continue
                delta = float(left[metric]) - float(right[metric])
                comparison_rows.append(
                    {
                        "comparison": comparison,
                        "condition_a": numerator,
                        "condition_b": denominator,
                        "metric": metric,
                        "seed": seed,
                        "value_a": left[metric],
                        "value_b": right[metric],
                        "delta": delta,
                        "sign": _delta_sign(delta),
                    }
                )
        paired_deltas.extend(comparison_rows)
        sign_tracking[comparison] = {}
        for metric in METRIC_PATHS:
            metric_rows = [row for row in comparison_rows if row["metric"] == metric]
            deltas = [row["delta"] for row in metric_rows]
            signs = [row["sign"] for row in metric_rows]
            sign_tracking[comparison][metric] = {
                **_summary(deltas),
                "signs_by_seed": {str(row["seed"]): row["sign"] for row in metric_rows},
                "positive": signs.count("positive"),
                "negative": signs.count("negative"),
                "zero": signs.count("zero"),
            }

    report = {
        "phase": "phase3_multiseed_fold0_replication",
        "fold": 0,
        "split_seed": next(iter(split_seeds)),
        "seeds": unique_seeds,
        "conditions": list(PHASE3_CONDITIONS),
        "n_runs": len(rows),
        "sam2_skipped": True,
        "paired_id_validation": paired_validation,
        "condition_summary": condition_summary,
        "paired_delta_sign_tracking": sign_tracking,
        "notes": [
            "Standard deviation is the sample standard deviation (n - 1).",
            "Fixed-threshold metrics retain the calibration used by each source run; inspect its Phase 1 calibration report before interpreting their absolute values.",
        ],
    }
    return report, rows, paired_deltas


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def save_phase3_report(
    output_dir: str | Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    paired_deltas: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write explicitly named Phase 3 JSON and CSV result artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out / "phase3_multiseed_fold0_replication_report.json",
        "individual_runs": out / "phase3_individual_runs.csv",
        "paired_deltas": out / "phase3_paired_deltas.csv",
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    _write_csv(paths["individual_runs"], rows)
    _write_csv(paths["paired_deltas"], paired_deltas)
    return paths
