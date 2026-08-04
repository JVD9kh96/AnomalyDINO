"""Calibrated dual-bank scoring modes (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.detectors.knn_index import pairwise_knn_distances
from src.detectors.reference_purification import dual_bank_scores as legacy_dual_bank_scores


ScoreMode = Literal[
    "normal",
    "anomaly_diagnostic",
    "margin",
    "ratio",
    "gated_hybrid",
]


@dataclass
class RobustZCalibration:
    """Robust z-score calibration fitted on reference/training scores only."""

    median: float
    iqr: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        scale = self.iqr if self.iqr > 1e-12 else 1.0
        return (np.asarray(values, dtype=np.float64) - self.median) / scale


def fit_robust_z(values: np.ndarray) -> RobustZCalibration:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return RobustZCalibration(median=0.0, iqr=1.0)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    iqr = float(q75 - q25)
    return RobustZCalibration(median=float(q50), iqr=iqr if iqr > 1e-12 else 1.0)


@dataclass
class DualBankScorer:
    normal_bank: np.ndarray
    anomaly_banks: dict[int, np.ndarray]
    mode: ScoreMode = "gated_hybrid"
    lambda_a: float = 1.0
    normal_gate_percentile: float = 95.0
    knn_metric: str = "L2_normalized"
    k_neighbors: int = 1
    d_normal_calibration: RobustZCalibration | None = None
    d_anomaly_calibration: dict[int, RobustZCalibration] | None = None
    gate_threshold: float | None = None

    def fit_calibration(self, reference_features: np.ndarray) -> None:
        """Fit robust z-score stats on reference/training features only."""
        d_n = pairwise_knn_distances(
            reference_features,
            self.normal_bank,
            self.knn_metric,
            self.k_neighbors,
            faiss_on_cpu=True,
        )
        self.d_normal_calibration = fit_robust_z(d_n)
        self.gate_threshold = float(
            np.percentile(d_n, self.normal_gate_percentile)
        )
        self.d_anomaly_calibration = {}
        for class_id, bank in self.anomaly_banks.items():
            if bank.size == 0:
                continue
            d_a = pairwise_knn_distances(
                reference_features,
                bank,
                self.knn_metric,
                self.k_neighbors,
                faiss_on_cpu=True,
            )
            # Higher anomaly score = more anomalous => calibrate -d_A.
            self.d_anomaly_calibration[class_id] = fit_robust_z(-d_a)

    def score(self, query_features: np.ndarray) -> dict[str, Any]:
        query = np.asarray(query_features, dtype=np.float32)
        d_normal = pairwise_knn_distances(
            query,
            self.normal_bank,
            self.knn_metric,
            self.k_neighbors,
            faiss_on_cpu=True,
        ).astype(np.float64)

        d_anomaly_by_class: dict[int, np.ndarray] = {}
        neg_d_anomaly_by_class: dict[int, np.ndarray] = {}
        for class_id, bank in self.anomaly_banks.items():
            if bank is None or len(bank) == 0:
                continue
            d_a = pairwise_knn_distances(
                query,
                bank,
                self.knn_metric,
                self.k_neighbors,
                faiss_on_cpu=True,
            ).astype(np.float64)
            d_anomaly_by_class[class_id] = d_a
            neg = -d_a
            if self.d_anomaly_calibration and class_id in self.d_anomaly_calibration:
                neg = self.d_anomaly_calibration[class_id].transform(neg)
            neg_d_anomaly_by_class[class_id] = neg

        if self.d_normal_calibration is not None:
            d_n_cal = self.d_normal_calibration.transform(d_normal)
        else:
            d_n_cal = d_normal

        if neg_d_anomaly_by_class:
            stacked = np.stack(list(neg_d_anomaly_by_class.values()), axis=0)
            best_idx = np.argmax(stacked, axis=0)
            class_ids = list(neg_d_anomaly_by_class.keys())
            predicted_class = np.asarray(
                [class_ids[int(i)] for i in best_idx], dtype=np.int32
            )
            d_a_best = stacked[best_idx, np.arange(stacked.shape[1])]
            raw_d_a_best = np.stack(
                [d_anomaly_by_class[c] for c in class_ids], axis=0
            )[best_idx, np.arange(stacked.shape[1])]
        else:
            predicted_class = np.zeros(query.shape[0], dtype=np.int32)
            d_a_best = np.zeros(query.shape[0], dtype=np.float64)
            raw_d_a_best = np.zeros(query.shape[0], dtype=np.float64)

        mode = self.mode
        if mode == "normal":
            scores = d_n_cal
        elif mode == "anomaly_diagnostic":
            scores = d_a_best
        elif mode == "margin":
            # Calibrated margin: z(d_N) + lambda * z(-d_A); higher = more anomalous.
            scores = d_n_cal + self.lambda_a * d_a_best
        elif mode == "ratio":
            scores = d_n_cal / np.maximum(raw_d_a_best, 1e-8)
        elif mode == "gated_hybrid":
            gate = self.gate_threshold if self.gate_threshold is not None else np.inf
            hybrid = d_n_cal + self.lambda_a * d_a_best
            scores = np.where(d_normal >= gate, hybrid, d_n_cal)
        else:
            raise ValueError(f"Unknown score mode={mode!r}")

        # Enforce higher = more anomalous polarity for all modes.
        scores = np.asarray(scores, dtype=np.float32)
        return {
            "scores": scores,
            "d_normal": d_normal.astype(np.float32),
            "d_anomaly_by_class": {
                int(k): v.astype(np.float32) for k, v in d_anomaly_by_class.items()
            },
            "predicted_nearest_anomaly_class": predicted_class,
            "mode": mode,
        }


def dual_bank_scores(
    query_features: np.ndarray,
    normal_bank: np.ndarray,
    defect_bank: np.ndarray,
    *,
    knn_metric: str,
    k_neighbors: int = 1,
    alpha: float = 1.0,
) -> np.ndarray:
    """Backward-compatible wrapper around the legacy margin score."""
    return legacy_dual_bank_scores(
        query_features,
        normal_bank,
        defect_bank,
        knn_metric=knn_metric,
        k_neighbors=k_neighbors,
        alpha=alpha,
    )


def assert_higher_is_anomalous(scores: np.ndarray, labels: np.ndarray) -> bool:
    """Soft polarity check: mean score on positives should exceed negatives."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if not labels.any() or labels.all():
        return True
    return float(np.mean(scores[labels])) >= float(np.mean(scores[~labels]))
