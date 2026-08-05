"""Frozen fold-0 primary settings selected by Phase 4 / Phase 5.

Do not retune these on held-out folds. Controls (naive, random20, oracle, clean)
may differ only in the stated experimental factor.
"""

from __future__ import annotations

from typing import Any


# Phase 4 recommended_setting (results/phase4/..._report.json).
FROZEN_PURIFICATION_MODE = "fixed_ratio_trim"
FROZEN_TRIM_FRACTION = 0.20
FROZEN_PURIFICATION_STRATEGY = "fixed_ratio_distance_trim"

# Phase 5 exact budget (= 8 clean × 6,400 DINO patches at 448 px).
FROZEN_FINAL_PATCH_BUDGET = 51_200
FROZEN_BUDGET_POLICY = "greedy_coreset"

# Default development protocol used when selecting the freeze.
FROZEN_FOLD = 0
FROZEN_SEED = 42
FROZEN_SPLIT_SEED = 42
FROZEN_CLEAN_SHOTS_PRIMARY = 2
FROZEN_ADDITIONAL_SHOTS_PRIMARY = 8

# Artifact paths relative to repo root (synced from remote GPU runs).
PHASE4_REPORT = "results/phase4/phase4_compact_purification_controls_report.json"
PHASE5_REPORT = "results/phase5/phase5_exact_memory_budget_controls_report.json"


def frozen_primary_dict() -> dict[str, Any]:
    return {
        "purification_mode": FROZEN_PURIFICATION_MODE,
        "trim_fraction": FROZEN_TRIM_FRACTION,
        "purification_strategy": FROZEN_PURIFICATION_STRATEGY,
        "budget": FROZEN_FINAL_PATCH_BUDGET,
        "budget_policy": FROZEN_BUDGET_POLICY,
        "fold_selected_on": FROZEN_FOLD,
        "seed_selected_on": FROZEN_SEED,
        "split_seed": FROZEN_SPLIT_SEED,
        "primary_clean_shots": FROZEN_CLEAN_SHOTS_PRIMARY,
        "primary_additional_shots": FROZEN_ADDITIONAL_SHOTS_PRIMARY,
        "phase4_report": PHASE4_REPORT,
        "phase5_report": PHASE5_REPORT,
        "applies_to": [
            "phase5_proposed_rows",
            "phase12_proposed_mask_free",
            "phase12_reference_efficiency_proposed",
            "any_deployable_proposed_baseline",
        ],
        "does_not_apply_to": [
            "phase6_clean_bank_contamination_mechanism",
            "clean_only_baselines",
            "naive_contaminated_controls",
            "oracle_analysis_upper_bound",
            "gt_anomaly_memory_optional_extension",
        ],
    }


def apply_frozen_purification_to_detector_cfg(detector_cfg: dict[str, Any]) -> dict[str, Any]:
    """Mutate a detector config dict to the frozen proposed purification + budget."""
    out = dict(detector_cfg)
    out["reference_mode"] = FROZEN_PURIFICATION_MODE
    out["coreset_size"] = FROZEN_FINAL_PATCH_BUDGET
    out["budget_policy"] = FROZEN_BUDGET_POLICY
    pur = dict(out.get("reference_purification") or {})
    pur["fixed_trim_fraction"] = FROZEN_TRIM_FRACTION
    pur["spatial_cleanup"] = False
    out["reference_purification"] = pur
    return out


def assert_matches_frozen_primary(
    *,
    purification_mode: str,
    trim_fraction: float,
    budget: int | None = None,
    budget_policy: str | None = None,
) -> None:
    if purification_mode != FROZEN_PURIFICATION_MODE:
        raise ValueError(
            f"Expected frozen purification_mode={FROZEN_PURIFICATION_MODE!r}, "
            f"got {purification_mode!r}"
        )
    if abs(float(trim_fraction) - FROZEN_TRIM_FRACTION) > 1e-12:
        raise ValueError(
            f"Expected frozen trim_fraction={FROZEN_TRIM_FRACTION}, got {trim_fraction}"
        )
    if budget is not None and int(budget) != FROZEN_FINAL_PATCH_BUDGET:
        raise ValueError(
            f"Expected frozen budget={FROZEN_FINAL_PATCH_BUDGET}, got {budget}"
        )
    if budget_policy is not None and budget_policy != FROZEN_BUDGET_POLICY:
        raise ValueError(
            f"Expected frozen budget_policy={FROZEN_BUDGET_POLICY!r}, got {budget_policy!r}"
        )
