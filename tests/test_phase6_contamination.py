from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.anomaly_dino import AnomalyDINODetector, ReferenceNeighborTrace


def _enumerate_conditions():
    rates = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    compositions = (
        "uniform",
        "class_balanced",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
    )
    conditions = [{"rate": 0.0, "composition": "shared_zero", "name": "rate0_shared"}]
    for rate in rates:
        if rate == 0.0:
            continue
        for composition in compositions:
            conditions.append(
                {
                    "rate": float(rate),
                    "composition": composition,
                    "name": f"rate{rate:g}_{composition}",
                }
            )
    return conditions


class Phase6ContaminationTests(unittest.TestCase):
    def test_condition_count_shares_zero(self):
        conditions = _enumerate_conditions()
        self.assertEqual(sum(1 for c in conditions if c["rate"] == 0.0), 1)
        self.assertEqual(len(conditions), 1 + 6 * 6)

    def test_replacement_preserves_bank_size(self):
        detector = AnomalyDINODetector(
            device="cpu",
            faiss_on_cpu=True,
            coreset_size=None,
        )
        rng = np.random.default_rng(0)
        clean = rng.normal(size=(100, 8)).astype(np.float32)
        anom = rng.normal(size=(50, 8)).astype(np.float32) + 5.0
        classes = np.array([1] * 25 + [2] * 25, dtype=np.int32)
        meta = [
            {"image_id": f"a{i}", "grid_rc": (i // 10, i % 10)}
            for i in range(50)
        ]
        extras = detector.inject_contamination_replacement(
            clean,
            anom,
            0.10,
            seed=42,
            anomalous_classes=classes,
            anomalous_meta=meta,
            target_bank_size=100,
        )
        self.assertEqual(detector.last_bank_stats.final_memory_bank_size, 100)
        self.assertEqual(extras["n_injected"], 10)
        self.assertEqual(len(detector._bank_provenance), 100)
        injected = [
            p for p in detector._bank_provenance if p["source"] == "injected_anomaly"
        ]
        self.assertEqual(len(injected), 10)
        for entry in injected:
            self.assertIn(entry["bank_index"], extras["injected_bank_indices"])
            self.assertIsNotNone(entry["image_id"])

    def test_zero_rate_injects_nothing(self):
        detector = AnomalyDINODetector(device="cpu", faiss_on_cpu=True)
        clean = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
        anom = np.random.default_rng(1).normal(size=(64, 4)).astype(np.float32)
        extras = detector.inject_contamination_replacement(
            clean, anom, 0.0, seed=0, target_bank_size=64
        )
        self.assertEqual(extras["n_injected"], 0)
        self.assertTrue(all(p["source"] == "clean" for p in detector._bank_provenance))

    def test_neighbor_trace_dataclass(self):
        trace = ReferenceNeighborTrace(
            query_image_id="q",
            query_grid_rc=(1, 2),
            neighbor_image_id="n",
            neighbor_grid_rc=(3, 4),
            neighbor_source="injected_anomaly",
            neighbor_class=2,
            distance=0.5,
        )
        self.assertEqual(trace.neighbor_source, "injected_anomaly")


if __name__ == "__main__":
    unittest.main()
