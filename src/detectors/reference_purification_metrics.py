from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.severstal.dataset import SeverstalSample
from src.severstal.rle import union_masks
from src.severstal.transforms import (
    compute_processed_shape,
    mask_to_patch_overlap,
    resize_mask_like_model,
)


ORACLE_OVERLAP_RULES: dict[str, dict[str, Any]] = {
    "any_overlap": {
        "threshold": 0.0,
        "operator": ">",
        "description": "reject when defect overlap is greater than zero",
    },
    "at_least_10_percent": {
        "threshold": 0.10,
        "operator": ">=",
        "description": "reject when defect overlap is at least 10 percent",
    },
    "at_least_50_percent": {
        "threshold": 0.50,
        "operator": ">=",
        "description": "reject when defect overlap is at least 50 percent",
    },
}

DEFAULT_OVERLAP_QUANTILES = (
    0.0,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
    100.0,
)
DEFAULT_HISTOGRAM_BINS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class CandidatePatchOverlaps:
    image_id: str
    union_overlap: np.ndarray
    class_overlaps: dict[int, np.ndarray]


def compute_candidate_patch_overlaps(
    sample: SeverstalSample,
    *,
    resolution: int,
    patch_size: int,
    num_classes: int = 4,
) -> CandidatePatchOverlaps:
    """Map GT masks to exact per-patch overlap fractions on the DINO grid."""
    native_shape = sample.image.shape[:2]
    _, grid_size = compute_processed_shape(native_shape, resolution, patch_size)
    aligned_masks: dict[int, np.ndarray] = {}
    class_overlaps: dict[int, np.ndarray] = {}

    for class_id in range(1, num_classes + 1):
        mask = sample.masks_by_class.get(
            class_id,
            np.zeros(native_shape, dtype=bool),
        )
        aligned = resize_mask_like_model(
            mask,
            native_shape,
            resolution,
            patch_size,
        )
        aligned_masks[class_id] = aligned
        class_overlaps[class_id] = mask_to_patch_overlap(
            aligned,
            grid_size,
            patch_size,
        )

    union_overlap = mask_to_patch_overlap(
        union_masks(list(aligned_masks.values())),
        grid_size,
        patch_size,
    )
    return CandidatePatchOverlaps(
        image_id=sample.image_id,
        union_overlap=union_overlap,
        class_overlaps=class_overlaps,
    )


def oracle_rejection_mask(overlaps: np.ndarray, rule: str) -> np.ndarray:
    """Return the explicit oracle removal mask for one overlap rule."""
    if rule not in ORACLE_OVERLAP_RULES:
        raise ValueError(
            f"Unknown overlap rule {rule!r}; choose "
            f"{', '.join(ORACLE_OVERLAP_RULES)}"
        )
    values = np.asarray(overlaps, dtype=np.float64)
    config = ORACLE_OVERLAP_RULES[rule]
    threshold = float(config["threshold"])
    if config["operator"] == ">":
        return values > threshold
    return values >= threshold


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quality_metrics_for_positive_mask(
    *,
    positive_mask: np.ndarray,
    true_normal_mask: np.ndarray,
    rejected_mask: np.ndarray,
) -> dict[str, Any]:
    positive = np.asarray(positive_mask, dtype=bool).ravel()
    normal = np.asarray(true_normal_mask, dtype=bool).ravel()
    rejected = np.asarray(rejected_mask, dtype=bool).ravel()
    if positive.shape != rejected.shape or normal.shape != rejected.shape:
        raise ValueError("Overlap and rejection masks must have matching shapes")

    accepted = ~rejected
    n_positive = int(np.sum(positive))
    n_normal = int(np.sum(normal))
    n_rejected = int(np.sum(rejected))
    n_accepted = int(np.sum(accepted))
    rejected_positive = int(np.sum(rejected & positive))
    accepted_positive = int(np.sum(accepted & positive))
    rejected_normal = int(np.sum(rejected & normal))
    accepted_normal = int(np.sum(accepted & normal))

    return {
        "normal_retention": _safe_ratio(accepted_normal, n_normal),
        "anomalous_rejection_recall": _safe_ratio(
            rejected_positive,
            n_positive,
        ),
        "rejected_patch_precision": _safe_ratio(
            rejected_positive,
            n_rejected,
        ),
        "final_contamination": _safe_ratio(
            accepted_positive,
            n_accepted,
        ),
        "counts": {
            "n_candidate_patches": int(rejected.size),
            "n_anomalous_patches": n_positive,
            "n_true_normal_patches": n_normal,
            "n_rejected_patches": n_rejected,
            "n_accepted_patches": n_accepted,
            "n_rejected_anomalous_patches": rejected_positive,
            "n_accepted_anomalous_patches": accepted_positive,
            "n_rejected_normal_patches": rejected_normal,
            "n_accepted_normal_patches": accepted_normal,
        },
    }


def compute_purification_quality(
    *,
    union_overlaps: np.ndarray,
    class_overlaps: np.ndarray,
    rejected_mask: np.ndarray,
    truth_rule: str = "any_overlap",
) -> dict[str, Any]:
    """Compute overall and per-class patch purification quality.

    For each class, anomaly recall uses that class's positive patches. Normal
    retention always uses patches normal for every class, so rejecting a patch
    from another defect class is not incorrectly treated as loss of normal data.
    """
    union_values = np.asarray(union_overlaps, dtype=np.float64)
    class_values = np.asarray(class_overlaps, dtype=np.float64)
    rejected = np.asarray(rejected_mask, dtype=bool)
    if class_values.ndim < 2:
        raise ValueError("class_overlaps must include a class dimension")
    if union_values.shape != rejected.shape:
        raise ValueError("union_overlaps and rejected_mask must match")
    if class_values.shape[0] <= 0 or class_values.shape[1:] != union_values.shape:
        raise ValueError(
            "class_overlaps must have shape (num_classes, *union_overlaps.shape)"
        )

    global_positive = oracle_rejection_mask(union_values, truth_rule)
    true_normal = ~global_positive
    overall = _quality_metrics_for_positive_mask(
        positive_mask=global_positive,
        true_normal_mask=true_normal,
        rejected_mask=rejected,
    )

    by_class: dict[str, Any] = {}
    for class_index in range(class_values.shape[0]):
        class_positive = oracle_rejection_mask(
            class_values[class_index],
            truth_rule,
        )
        by_class[str(class_index + 1)] = _quality_metrics_for_positive_mask(
            positive_mask=class_positive,
            true_normal_mask=true_normal,
            rejected_mask=rejected,
        )

    return {
        "truth_rule": truth_rule,
        "truth_rule_definition": dict(ORACLE_OVERLAP_RULES[truth_rule]),
        "overall": overall,
        "by_defect_class": by_class,
        "class_metric_note": (
            "Per-class anomaly metrics use that class as positive; normal retention "
            "uses patches normal for all classes."
        ),
    }


def _stack_overlap_records(
    records: list[CandidatePatchOverlaps],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("At least one candidate overlap record is required")
    image_ids = [record.image_id for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Candidate overlap records contain duplicate image IDs")

    union_shape = records[0].union_overlap.shape
    class_ids = sorted(records[0].class_overlaps)
    if not class_ids:
        raise ValueError("Candidate records must contain class overlap maps")

    union_maps = []
    class_maps = []
    for record in records:
        if record.union_overlap.shape != union_shape:
            raise ValueError("All candidate overlap grids must have the same shape")
        if sorted(record.class_overlaps) != class_ids:
            raise ValueError("All candidate records must have the same defect classes")
        if any(record.class_overlaps[c].shape != union_shape for c in class_ids):
            raise ValueError("Class and union overlap grids must have matching shapes")
        union_maps.append(np.asarray(record.union_overlap, dtype=np.float32))
        class_maps.append(
            np.stack(
                [
                    np.asarray(record.class_overlaps[c], dtype=np.float32)
                    for c in class_ids
                ],
                axis=0,
            )
        )
    return image_ids, np.stack(union_maps, axis=0), np.stack(class_maps, axis=0)


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    positive = np.asarray(values, dtype=np.float64).ravel()
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return {
            "n_positive_overlap_patches": 0,
            "quantiles": {},
            "histogram": [],
        }

    histogram_counts, histogram_edges = np.histogram(
        positive,
        bins=np.asarray(DEFAULT_HISTOGRAM_BINS, dtype=np.float64),
    )
    histogram = [
        {
            "lower": float(histogram_edges[index]),
            "upper": float(histogram_edges[index + 1]),
            "count": int(histogram_counts[index]),
        }
        for index in range(histogram_counts.size)
    ]
    return {
        "n_positive_overlap_patches": int(positive.size),
        "quantiles": {
            f"p{quantile:g}": float(np.percentile(positive, quantile))
            for quantile in DEFAULT_OVERLAP_QUANTILES
        },
        "histogram": histogram,
    }


def build_phase2_purification_report(
    records: list[CandidatePatchOverlaps],
    *,
    auto_rejected_by_image: dict[str, np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    image_ids, union_maps, class_maps_by_image = _stack_overlap_records(records)
    class_maps = np.moveaxis(class_maps_by_image, 1, 0)
    n_candidate_patches = int(union_maps.size)

    oracle_masks = {
        rule: oracle_rejection_mask(union_maps, rule)
        for rule in ORACLE_OVERLAP_RULES
    }
    oracle_removal = {
        rule: {
            "rule": dict(ORACLE_OVERLAP_RULES[rule]),
            "n_removed_patches": int(np.sum(mask)),
            "removal_rate": _safe_ratio(int(np.sum(mask)), n_candidate_patches),
            "n_retained_patches": int(n_candidate_patches - np.sum(mask)),
        }
        for rule, mask in oracle_masks.items()
    }

    auto_mask: np.ndarray | None = None
    if auto_rejected_by_image is not None:
        expected = set(image_ids)
        actual = set(auto_rejected_by_image)
        if actual != expected:
            raise ValueError(
                "Auto-rejection image IDs do not match frozen candidates: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        auto_mask = np.stack(
            [np.asarray(auto_rejected_by_image[image_id], dtype=bool) for image_id in image_ids]
        )
        if auto_mask.shape != union_maps.shape:
            raise ValueError(
                f"Auto-rejection masks have shape {auto_mask.shape}, expected "
                f"{union_maps.shape}"
            )

    sensitivity: dict[str, Any] = {}
    for truth_rule in ORACLE_OVERLAP_RULES:
        oracle_quality = {
            oracle_rule: compute_purification_quality(
                union_overlaps=union_maps,
                class_overlaps=class_maps,
                rejected_mask=oracle_mask,
                truth_rule=truth_rule,
            )
            for oracle_rule, oracle_mask in oracle_masks.items()
        }
        if auto_mask is None:
            auto_quality: dict[str, Any] = {
                "status": "not_evaluated",
                "reason": "No auto-rejection mask bundle was supplied.",
            }
        else:
            auto_quality = compute_purification_quality(
                union_overlaps=union_maps,
                class_overlaps=class_maps,
                rejected_mask=auto_mask,
                truth_rule=truth_rule,
            )
        sensitivity[truth_rule] = {
            "oracle": oracle_quality,
            "auto": auto_quality,
        }

    overlap_distribution = {
        "definition": "positive overlap after resize/crop to the DINO patch grid",
        "union": _distribution_summary(union_maps),
        "by_defect_class": {
            str(class_index + 1): _distribution_summary(class_maps[class_index])
            for class_index in range(class_maps.shape[0])
        },
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "phase2",
        "canonical_ground_truth_rule": "any_overlap",
        "overlap_rules": ORACLE_OVERLAP_RULES,
        "n_candidate_images": len(image_ids),
        "n_candidate_patches": n_candidate_patches,
        "candidate_image_ids": image_ids,
        "oracle_removal": oracle_removal,
        "overlap_distribution": overlap_distribution,
        "purification_quality": {
            "canonical_any_overlap": sensitivity["any_overlap"],
            "sensitivity_by_ground_truth_rule": sensitivity,
        },
        "auto_status": (
            "evaluated" if auto_mask is not None else "missing_auto_rejection_masks"
        ),
        "metadata": dict(metadata or {}),
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    report["phase2_report_id"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    distribution_arrays = {
        "candidate_image_ids": np.asarray(image_ids),
        "union_overlap_maps": union_maps,
        "class_overlap_maps": class_maps_by_image,
        "union_anomalous_overlaps": union_maps[union_maps > 0.0],
    }
    for class_index in range(class_maps.shape[0]):
        values = class_maps[class_index]
        distribution_arrays[f"class_{class_index + 1}_anomalous_overlaps"] = values[
            values > 0.0
        ]
    return report, distribution_arrays


BANK_FILTER_NAMES = (
    "naive",
    "auto_purified",
    "distance_trim_20",
    "random_size_matched",
    "oracle",
)


def rejected_mask_from_keep_mask(keep_mask: np.ndarray) -> np.ndarray:
    return ~np.asarray(keep_mask, dtype=bool)


def count_overlap_patches_direct(
    union_overlaps: np.ndarray,
    *,
    threshold: float = 0.0,
    operator: str = ">",
) -> int:
    """Direct patch-overlap count used as the Phase-2 acceptance fixture oracle."""
    values = np.asarray(union_overlaps, dtype=np.float64)
    if operator == ">":
        return int(np.sum(values > threshold))
    if operator == ">=":
        return int(np.sum(values >= threshold))
    raise ValueError(f"Unsupported operator={operator!r}")


def build_multi_bank_purification_report(
    records: list[CandidatePatchOverlaps],
    *,
    bank_rejected_by_image: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any] | None = None,
    selected_oracle_rule: str = "any_overlap",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Compare purification quality across named banks after construction.

    ``bank_rejected_by_image`` maps bank_name -> {image_id -> rejected_mask}.
    Expected bank names include naive / auto_purified / distance_trim_20 /
    random_size_matched / oracle.
    """
    base_report, distribution_arrays = build_phase2_purification_report(
        records,
        auto_rejected_by_image=bank_rejected_by_image.get("auto_purified"),
        metadata=metadata,
    )
    image_ids, union_maps, class_maps_by_image = _stack_overlap_records(records)
    class_maps = np.moveaxis(class_maps_by_image, 1, 0)

    bank_quality: dict[str, Any] = {}
    for bank_name, rejected_by_image in bank_rejected_by_image.items():
        expected = set(image_ids)
        actual = set(rejected_by_image)
        if actual != expected:
            raise ValueError(
                f"Bank {bank_name!r} rejection IDs mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        rejected = np.stack(
            [np.asarray(rejected_by_image[image_id], dtype=bool) for image_id in image_ids]
        )
        if rejected.shape != union_maps.shape:
            raise ValueError(
                f"Bank {bank_name!r} rejection shape {rejected.shape} != {union_maps.shape}"
            )
        per_truth: dict[str, Any] = {}
        for truth_rule in ORACLE_OVERLAP_RULES:
            per_truth[truth_rule] = compute_purification_quality(
                union_overlaps=union_maps,
                class_overlaps=class_maps,
                rejected_mask=rejected,
                truth_rule=truth_rule,
            )
        bank_quality[bank_name] = {
            "n_rejected_patches": int(np.sum(rejected)),
            "n_retained_patches": int(rejected.size - np.sum(rejected)),
            "quality_by_truth_rule": per_truth,
            "canonical": per_truth[selected_oracle_rule],
        }

    # Direct overlap counts for acceptance tests / audits.
    direct_counts = {
        rule: {
            "n_positive_patches": count_overlap_patches_direct(
                union_maps,
                threshold=float(spec["threshold"]),
                operator=str(spec["operator"]),
            ),
            "by_image": {
                image_id: count_overlap_patches_direct(
                    union_maps[index],
                    threshold=float(spec["threshold"]),
                    operator=str(spec["operator"]),
                )
                for index, image_id in enumerate(image_ids)
            },
        }
        for rule, spec in ORACLE_OVERLAP_RULES.items()
    }

    base_report["multi_bank_purification_quality"] = bank_quality
    base_report["selected_oracle_rule"] = selected_oracle_rule
    base_report["direct_overlap_counts"] = direct_counts
    base_report["banks_evaluated"] = sorted(bank_quality)
    # Refresh report id after extensions.
    body = {key: value for key, value in base_report.items() if key != "phase2_report_id"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    base_report["phase2_report_id"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return base_report, distribution_arrays


def save_phase2_purification_artifacts(
    *,
    report: dict[str, Any],
    distribution_arrays: dict[str, np.ndarray],
    output_dir: str | Path,
    report_filename: str = "phase2_purification_quality_report.json",
    distribution_filename: str = "phase2_patch_overlap_distribution.npz",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_filename
    distribution_path = output_dir / distribution_filename
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    np.savez_compressed(distribution_path, **distribution_arrays)
    return report_path, distribution_path


def save_auto_rejection_bundle(
    path: str | Path,
    *,
    image_ids: list[str],
    rejected_masks: np.ndarray,
    method: str,
) -> Path:
    masks = np.asarray(rejected_masks, dtype=bool)
    if masks.ndim < 2 or masks.shape[0] != len(image_ids):
        raise ValueError("rejected_masks first dimension must match image_ids")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids contains duplicates")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_ids=np.asarray(image_ids),
        rejected_masks=masks,
        method=np.asarray(method),
    )
    return path


def load_auto_rejection_bundle(path: str | Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"image_ids", "rejected_masks", "method"}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"Auto-rejection bundle is missing: {sorted(missing)}")
        image_ids = [str(value) for value in bundle["image_ids"].tolist()]
        masks = np.asarray(bundle["rejected_masks"], dtype=bool)
        method = str(bundle["method"].item())
    if masks.shape[0] != len(image_ids):
        raise ValueError("Auto-rejection masks do not align with image IDs")
    return {
        image_id: masks[index]
        for index, image_id in enumerate(image_ids)
    }, method
