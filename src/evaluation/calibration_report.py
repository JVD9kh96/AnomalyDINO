from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1
CANDIDATE_SCORE_METHOD = (
    "leave_one_clean_reference_out_scores_against_clean_seed_bank"
)
QUERY_SCORE_METHOD = (
    "leave_one_clean_reference_out_scores_against_final_bank"
)
F1_MAX_METHOD = "validation_gt_exact_threshold_sweep_diagnostic_only"
DEFAULT_QUANTILES = (0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.5, 100.0)

CrossFitScorer = Callable[[str, list[str], str], np.ndarray]


def _finite_scores(values: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"{name} must contain at least one finite score")
    return scores


def _validate_percentile(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be in [0, 100], got {value}")
    return value


def _ids_sha256(image_ids: list[str]) -> str:
    payload = "\n".join(sorted(image_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_cross_fitted_clean_score_sets(
    *,
    clean_reference_ids: list[str],
    final_reference_ids: list[str],
    score_held_out: CrossFitScorer,
) -> dict[str, Any]:
    """Score each clean reference twice while excluding it from both banks.

    ``score_held_out`` receives ``(held_out_id, bank_reference_ids, bank_stage)``.
    The caller is responsible for applying the mode's patch filtering when
    ``bank_stage`` is ``"final"``.
    """
    clean_ids = list(clean_reference_ids)
    final_ids = list(final_reference_ids)
    if len(clean_ids) < 2:
        raise ValueError("Cross-fitting requires at least two clean references")
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("clean_reference_ids contains duplicates")
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("final_reference_ids contains duplicates")
    if not set(clean_ids).issubset(final_ids):
        raise ValueError("Every clean reference must be represented in the final bank")

    candidate_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    candidate_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}

    for held_out_id in clean_ids:
        clean_bank_ids = [image_id for image_id in clean_ids if image_id != held_out_id]
        final_bank_ids = [image_id for image_id in final_ids if image_id != held_out_id]
        if held_out_id in clean_bank_ids or held_out_id in final_bank_ids:
            raise AssertionError("Held-out clean reference was not excluded from a bank")

        candidate_scores = _finite_scores(
            score_held_out(
                held_out_id,
                clean_bank_ids,
                "candidate_acceptance_clean_bank",
            ),
            f"candidate acceptance scores for {held_out_id}",
        )
        query_scores = _finite_scores(
            score_held_out(
                held_out_id,
                final_bank_ids,
                "query_final_bank",
            ),
            f"final-bank query scores for {held_out_id}",
        )
        candidate_parts.append(candidate_scores)
        query_parts.append(query_scores)
        candidate_counts[held_out_id] = int(candidate_scores.size)
        query_counts[held_out_id] = int(query_scores.size)

    return {
        "held_out_clean_candidate_distances": np.concatenate(candidate_parts),
        "cross_fitted_clean_query_scores": np.concatenate(query_parts),
        "clean_reference_ids": clean_ids,
        "final_reference_ids": final_ids,
        "candidate_score_counts_by_image": candidate_counts,
        "query_score_counts_by_image": query_counts,
        "candidate_score_method": CANDIDATE_SCORE_METHOD,
        "query_score_method": QUERY_SCORE_METHOD,
        "candidate_self_exclusion": True,
        "query_self_exclusion": True,
        "clean_reference_ids_sha256": _ids_sha256(clean_ids),
        "final_reference_ids_sha256": _ids_sha256(final_ids),
    }


def score_quantiles(
    values: np.ndarray,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
) -> dict[str, float]:
    scores = _finite_scores(values, "score quantile input")
    checked = [_validate_percentile(value, "quantile") for value in quantiles]
    return {
        f"p{value:g}": float(np.percentile(scores, value))
        for value in checked
    }


def _flatten_validation(
    validation_scores: np.ndarray,
    validation_gt_labels: np.ndarray,
    validation_valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(validation_scores, dtype=np.float64).ravel()
    labels = np.asarray(validation_gt_labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(
            "validation_scores and validation_gt_labels must have identical shapes"
        )

    keep = np.isfinite(scores)
    if validation_valid_mask is not None:
        valid = np.asarray(validation_valid_mask, dtype=bool).ravel()
        if valid.shape != scores.shape:
            raise ValueError(
                "validation_valid_mask must match the flattened validation scores"
            )
        keep &= valid
    scores = scores[keep]
    labels = labels[keep]
    if scores.size == 0:
        raise ValueError("No finite, valid validation scores remain")
    return scores, labels


def _operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = scores >= float(threshold)
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_gt_positive_patches": int(np.sum(labels)),
        "n_pred_positive_patches": int(np.sum(predictions)),
    }


def exact_f1_max_operating_point(
    validation_scores: np.ndarray,
    validation_gt_labels: np.ndarray,
) -> dict[str, Any]:
    """Find F1-max over every distinct score; ties prefer the higher threshold."""
    scores = np.asarray(validation_scores, dtype=np.float64).ravel()
    labels = np.asarray(validation_gt_labels, dtype=bool).ravel()
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("F1-max inputs must be non-empty arrays with matching shapes")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    tp = cumulative_tp[group_ends]
    predicted = group_ends + 1
    fp = predicted - tp
    total_positive = int(np.sum(labels))
    fn = total_positive - tp
    precision = tp / predicted
    recall = tp / total_positive if total_positive else np.zeros_like(tp, dtype=float)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision, dtype=float),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    threshold = float(sorted_scores[group_ends[best_index]])
    result = _operating_point(scores, labels, threshold)
    result["method"] = F1_MAX_METHOD
    return result


def build_phase1_calibration_report(
    *,
    held_out_clean_candidate_distances: np.ndarray,
    cross_fitted_clean_query_scores: np.ndarray,
    validation_scores: np.ndarray,
    validation_gt_labels: np.ndarray,
    final_bank_id: str,
    clean_bank_id: str,
    candidate_acceptance_percentile: float = 99.0,
    query_percentile: float = 99.5,
    validation_valid_mask: np.ndarray | None = None,
    candidate_score_method: str = CANDIDATE_SCORE_METHOD,
    query_score_method: str = QUERY_SCORE_METHOD,
    candidate_self_exclusion: bool = True,
    query_self_exclusion: bool = True,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the single Phase 1 audit report for one completed final bank."""
    if not final_bank_id or not clean_bank_id:
        raise ValueError("clean_bank_id and final_bank_id are required")
    if candidate_score_method != CANDIDATE_SCORE_METHOD:
        raise ValueError("Candidate scores are not marked as clean-bank cross-fitted")
    if query_score_method != QUERY_SCORE_METHOD:
        raise ValueError("Query scores are not marked as final-bank cross-fitted")
    if not candidate_self_exclusion or not query_self_exclusion:
        raise ValueError("Both calibration score sets must exclude the held-out image")

    p_accept = _validate_percentile(
        candidate_acceptance_percentile, "candidate_acceptance_percentile"
    )
    p_query = _validate_percentile(query_percentile, "query_percentile")
    candidate_scores = _finite_scores(
        held_out_clean_candidate_distances,
        "held_out_clean_candidate_distances",
    )
    clean_query_scores = _finite_scores(
        cross_fitted_clean_query_scores,
        "cross_fitted_clean_query_scores",
    )
    val_scores, val_labels = _flatten_validation(
        validation_scores,
        validation_gt_labels,
        validation_valid_mask,
    )

    tau_accept = float(np.percentile(candidate_scores, p_accept))
    tau_query = float(np.percentile(clean_query_scores, p_query))
    fixed = _operating_point(val_scores, val_labels, tau_query)
    f1_max = exact_f1_max_operating_point(val_scores, val_labels)

    report: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase": "phase1",
        "candidate_acceptance": {
            "percentile": p_accept,
            "threshold": tau_accept,
            "method": CANDIDATE_SCORE_METHOD,
            "clean_bank_id": clean_bank_id,
            "n_scores": int(candidate_scores.size),
        },
        "query_operating_point": {
            "percentile": p_query,
            "threshold": tau_query,
            "method": QUERY_SCORE_METHOD,
            "final_bank_id": final_bank_id,
            "bank_stage": "final",
            "n_scores": int(clean_query_scores.size),
        },
        "f1_max_threshold": f1_max["threshold"],
        "f1_max_operating_point": f1_max,
        "threshold_methods": {
            "tau_accept": CANDIDATE_SCORE_METHOD,
            "tau_query": QUERY_SCORE_METHOD,
            "f1_max_threshold": F1_MAX_METHOD,
        },
        "score_quantiles": {
            "held_out_clean_candidate_distances": score_quantiles(
                candidate_scores, quantiles
            ),
            "cross_fitted_clean_query_scores_against_final_bank": score_quantiles(
                clean_query_scores, quantiles
            ),
            "validation_query_scores": score_quantiles(val_scores, quantiles),
        },
        "n_gt_positive_patches": fixed["n_gt_positive_patches"],
        "n_pred_positive_patches": fixed["n_pred_positive_patches"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "f1": fixed["f1"],
        "confusion": {
            "tp": fixed["tp"],
            "fp": fixed["fp"],
            "fn": fixed["fn"],
            "tn": fixed["tn"],
        },
        "safeguards": {
            "candidate_and_query_calibration_are_distinct": True,
            "query_threshold_uses_validation_gt": False,
            "f1_max_is_deployable": False,
            "query_threshold_bank_stage": "final",
            "candidate_self_exclusion": True,
            "query_self_exclusion": True,
        },
        "metadata": dict(metadata or {}),
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    report["calibration_report_id"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return report


def save_phase1_calibration_report(
    report: dict[str, Any],
    run_dir: str | Path,
    filename: str = "phase1_calibration_report.json",
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / filename
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def create_and_save_phase1_calibration_report(
    *,
    clean_reference_ids: list[str],
    final_reference_ids: list[str],
    score_held_out: CrossFitScorer,
    validation_scores: np.ndarray,
    validation_gt_labels: np.ndarray,
    final_bank_id: str,
    clean_bank_id: str,
    run_dir: str | Path,
    candidate_acceptance_percentile: float = 99.0,
    query_percentile: float = 99.5,
    validation_valid_mask: np.ndarray | None = None,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    metadata: dict[str, Any] | None = None,
    filename: str = "phase1_calibration_report.json",
) -> tuple[dict[str, Any], Path]:
    """Run cross-fitting and persist exactly one report for a mode run."""
    cross_fitted = collect_cross_fitted_clean_score_sets(
        clean_reference_ids=clean_reference_ids,
        final_reference_ids=final_reference_ids,
        score_held_out=score_held_out,
    )
    report = build_phase1_calibration_report(
        held_out_clean_candidate_distances=cross_fitted[
            "held_out_clean_candidate_distances"
        ],
        cross_fitted_clean_query_scores=cross_fitted[
            "cross_fitted_clean_query_scores"
        ],
        validation_scores=validation_scores,
        validation_gt_labels=validation_gt_labels,
        validation_valid_mask=validation_valid_mask,
        final_bank_id=final_bank_id,
        clean_bank_id=clean_bank_id,
        candidate_acceptance_percentile=candidate_acceptance_percentile,
        query_percentile=query_percentile,
        candidate_score_method=cross_fitted["candidate_score_method"],
        query_score_method=cross_fitted["query_score_method"],
        candidate_self_exclusion=cross_fitted["candidate_self_exclusion"],
        query_self_exclusion=cross_fitted["query_self_exclusion"],
        quantiles=quantiles,
        metadata=metadata,
    )
    output_path = save_phase1_calibration_report(
        report,
        run_dir,
        filename=filename,
    )
    return report, output_path


def save_phase1_score_bundle(
    path: str | Path,
    *,
    cross_fitted_scores: dict[str, Any],
    validation_scores: np.ndarray,
    validation_gt_labels: np.ndarray,
    final_bank_id: str,
    clean_bank_id: str,
    validation_valid_mask: np.ndarray | None = None,
) -> Path:
    """Persist mode-run evidence consumed by the phase-numbered report script."""
    if not final_bank_id or not clean_bank_id:
        raise ValueError("clean_bank_id and final_bank_id are required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "held_out_clean_candidate_distances": np.asarray(
            cross_fitted_scores["held_out_clean_candidate_distances"]
        ),
        "cross_fitted_clean_query_scores": np.asarray(
            cross_fitted_scores["cross_fitted_clean_query_scores"]
        ),
        "validation_scores": np.asarray(validation_scores),
        "validation_gt_labels": np.asarray(validation_gt_labels, dtype=bool),
        "final_bank_id": np.asarray(final_bank_id),
        "clean_bank_id": np.asarray(clean_bank_id),
        "candidate_score_method": np.asarray(
            cross_fitted_scores.get("candidate_score_method", "")
        ),
        "query_score_method": np.asarray(
            cross_fitted_scores.get("query_score_method", "")
        ),
        "candidate_self_exclusion": np.asarray(
            bool(cross_fitted_scores.get("candidate_self_exclusion", False))
        ),
        "query_self_exclusion": np.asarray(
            bool(cross_fitted_scores.get("query_self_exclusion", False))
        ),
    }
    if validation_valid_mask is not None:
        payload["validation_valid_mask"] = np.asarray(
            validation_valid_mask, dtype=bool
        )
    np.savez_compressed(path, **payload)
    return path


def load_phase1_score_bundle(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "held_out_clean_candidate_distances",
            "cross_fitted_clean_query_scores",
            "validation_scores",
            "validation_gt_labels",
            "final_bank_id",
            "clean_bank_id",
            "candidate_score_method",
            "query_score_method",
            "candidate_self_exclusion",
            "query_self_exclusion",
        }
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"Phase 1 score bundle is missing: {sorted(missing)}")
        result: dict[str, Any] = {
            key: np.asarray(bundle[key])
            for key in (
                "held_out_clean_candidate_distances",
                "cross_fitted_clean_query_scores",
                "validation_scores",
                "validation_gt_labels",
            )
        }
        result.update(
            {
                "final_bank_id": str(bundle["final_bank_id"].item()),
                "clean_bank_id": str(bundle["clean_bank_id"].item()),
                "candidate_score_method": str(
                    bundle["candidate_score_method"].item()
                ),
                "query_score_method": str(bundle["query_score_method"].item()),
                "candidate_self_exclusion": bool(
                    bundle["candidate_self_exclusion"].item()
                ),
                "query_self_exclusion": bool(
                    bundle["query_self_exclusion"].item()
                ),
            }
        )
        if "validation_valid_mask" in bundle.files:
            result["validation_valid_mask"] = np.asarray(
                bundle["validation_valid_mask"], dtype=bool
            )
    return result


def build_report_from_score_bundle(
    bundle: dict[str, Any],
    *,
    candidate_acceptance_percentile: float = 99.0,
    query_percentile: float = 99.5,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_phase1_calibration_report(
        held_out_clean_candidate_distances=bundle[
            "held_out_clean_candidate_distances"
        ],
        cross_fitted_clean_query_scores=bundle[
            "cross_fitted_clean_query_scores"
        ],
        validation_scores=bundle["validation_scores"],
        validation_gt_labels=bundle["validation_gt_labels"],
        validation_valid_mask=bundle.get("validation_valid_mask"),
        final_bank_id=bundle["final_bank_id"],
        clean_bank_id=bundle["clean_bank_id"],
        candidate_acceptance_percentile=candidate_acceptance_percentile,
        query_percentile=query_percentile,
        candidate_score_method=bundle["candidate_score_method"],
        query_score_method=bundle["query_score_method"],
        candidate_self_exclusion=bundle["candidate_self_exclusion"],
        query_self_exclusion=bundle["query_self_exclusion"],
        quantiles=quantiles,
        metadata=metadata,
    )
