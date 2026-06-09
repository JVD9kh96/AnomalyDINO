from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def cohens_d(healthy: np.ndarray, anomaly: np.ndarray) -> float:
    if len(healthy) < 2 or len(anomaly) < 2:
        return float("nan")
    n1, n2 = len(healthy), len(anomaly)
    s1, s2 = np.var(healthy, ddof=1), np.var(anomaly, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(anomaly) - np.mean(healthy)) / np.sqrt(pooled))


def compute_separability_metrics(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    all_labels: np.ndarray | None = None,
    all_scores: np.ndarray | None = None,
) -> dict:
    healthy = healthy_scores.astype(np.float64)
    anomaly = anomaly_scores.astype(np.float64)

    result = {
        "ks_statistic": float(stats.ks_2samp(healthy, anomaly).statistic)
        if len(healthy) > 0 and len(anomaly) > 0
        else float("nan"),
        "wasserstein_distance": float(stats.wasserstein_distance(healthy, anomaly))
        if len(healthy) > 0 and len(anomaly) > 0
        else float("nan"),
        "cohens_d": cohens_d(healthy, anomaly),
    }

    if all_labels is not None and all_scores is not None:
        labels = all_labels.astype(int)
        scores = all_scores.astype(np.float64)
        if len(np.unique(labels)) > 1:
            result["auroc"] = float(roc_auc_score(labels, scores))
            result["auprc"] = float(average_precision_score(labels, scores))
        else:
            result["auroc"] = float("nan")
            result["auprc"] = float("nan")

    return result


def distribution_summary(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p5": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
    }
