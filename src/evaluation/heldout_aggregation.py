"""Image-level / hierarchical paired bootstrap for held-out aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_deltas(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    key: str = "auprc",
    pair_keys: tuple[str, ...] = ("fold", "seed"),
) -> list[dict[str, Any]]:
    """Pair rows by fold/seed and compute metric deltas (b - a)."""
    index_a = {
        tuple(row[k] for k in pair_keys): row
        for row in rows_a
    }
    deltas = []
    for row_b in rows_b:
        pair = tuple(row_b[k] for k in pair_keys)
        if pair not in index_a:
            continue
        row_a = index_a[pair]
        va = row_a.get(key)
        vb = row_b.get(key)
        if va is None or vb is None:
            continue
        deltas.append(
            {
                **{k: row_b[k] for k in pair_keys},
                "metric": key,
                "value_a": float(va),
                "value_b": float(vb),
                "delta": float(vb) - float(va),
            }
        )
    return deltas


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of image-level (or paired) values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "n_boot": int(n_boot),
        }
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(values)),
        "ci_low": low,
        "ci_high": high,
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def hierarchical_bootstrap_mean_ci(
    groups: list[np.ndarray],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Hierarchical bootstrap: resample groups, then resample observations within groups.

    Use image-level groups of patch metrics (or fold-level groups of seed metrics).
    Do not treat patches as IID samples.
    """
    clean_groups = [
        np.asarray(g, dtype=np.float64)[np.isfinite(np.asarray(g, dtype=np.float64))]
        for g in groups
    ]
    clean_groups = [g for g in clean_groups if g.size > 0]
    if not clean_groups:
        return bootstrap_mean_ci(np.asarray([]), n_boot=n_boot, seed=seed, alpha=alpha)

    rng = np.random.default_rng(seed)
    n_groups = len(clean_groups)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        parts = []
        for gi in chosen:
            g = clean_groups[int(gi)]
            parts.append(g[rng.integers(0, g.size, size=g.size)])
        means[i] = float(np.mean(np.concatenate(parts)))
    flat = np.concatenate(clean_groups)
    return {
        "mean": float(np.mean(flat)),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n_groups": int(n_groups),
        "n_observations": int(flat.size),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def aggregate_heldout_matrix(
    rows: list[dict[str, Any]],
    *,
    condition_key: str = "condition",
    metric_keys: tuple[str, ...] = ("auprc", "auroc", "fixed_f1", "f1_max"),
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate per-condition paired fold/seed rows with bootstrap CIs."""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row[condition_key]), []).append(row)

    summary: dict[str, Any] = {}
    for condition, items in sorted(by_condition.items()):
        metric_summary = {}
        for metric in metric_keys:
            values = np.asarray(
                [item.get(metric) for item in items if item.get(metric) is not None],
                dtype=np.float64,
            )
            metric_summary[metric] = bootstrap_mean_ci(
                values, n_boot=n_boot, seed=seed
            )
        summary[condition] = {
            "n_rows": len(items),
            "metrics": metric_summary,
        }
    return {
        "n_rows": len(rows),
        "conditions": summary,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "unit": "fold_seed_pair"},
    }
