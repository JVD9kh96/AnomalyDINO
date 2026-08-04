from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.anomaly_memory import (
    AnomalyFeatureGrid,
    require_gt_anomaly_memory,
    select_anomaly_patches,
)
from src.detectors.anomaly_dino import ReferenceFeatureGrid
from src.detectors.dual_bank import (
    DualBankScorer,
    assert_higher_is_anomalous,
    fit_robust_z,
)
from src.evaluation.heldout_aggregation import (
    aggregate_heldout_matrix,
    bootstrap_mean_ci,
    hierarchical_bootstrap_mean_ci,
    paired_deltas,
)

FROZEN_TRIM = 0.20


def validate_frozen_config(config: dict) -> None:
    """Local copy of Phase-12 freeze checks to avoid importing the GPU runner."""
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


class AnomalyMemoryTests(unittest.TestCase):
    def test_fail_closed_without_flag(self):
        with self.assertRaisesRegex(RuntimeError, "allow_gt_anomaly_memory"):
            require_gt_anomaly_memory(False)

    def test_loco_excludes_held_out_class_from_source_classes(self):
        sample = _FakeSample("img", class_present=1)
        grid = ReferenceFeatureGrid(
            image_id="img",
            features=np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32),
            grid_size=(2, 2),
        )
        # Monkeypatch overlap computation via direct AnomalyFeatureGrid.
        overlap = np.zeros((2, 2, 4), dtype=np.float32)
        overlap[:, :, 0] = 0.5
        overlap[:, :, 1] = 0.2
        ag = AnomalyFeatureGrid(
            image_id="img",
            features=grid.features,
            grid_size=(2, 2),
            overlap_by_class=overlap,
            d_normal=np.ones((2, 2), dtype=np.float32),
            source_classes=(1, 2),
        )
        # Zero held-out class channel as LOCO construction would.
        held_out = {2}
        ag.overlap_by_class[:, :, 1] = 0.0
        result = select_anomaly_patches(
            [ag],
            class_id=2,
            overlap_rule="any_overlap",
            exclude_classes=held_out,
        )
        self.assertEqual(result.n_patches, 0)
        self.assertTrue(result.extras.get("excluded"))

    def test_selection_reproducible(self):
        overlap = np.zeros((2, 2, 4), dtype=np.float32)
        overlap[0, 0, 0] = 0.8
        overlap[1, 1, 0] = 0.6
        ag = AnomalyFeatureGrid(
            image_id="img",
            features=np.arange(32, dtype=np.float32).reshape(4, 8),
            grid_size=(2, 2),
            overlap_by_class=overlap,
            d_normal=np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
            source_classes=(1,),
        )
        a = select_anomaly_patches([ag], class_id=1, overlap_rule="any_overlap", seed=0)
        b = select_anomaly_patches([ag], class_id=1, overlap_rule="any_overlap", seed=0)
        self.assertEqual(a.n_patches, b.n_patches)
        self.assertTrue(np.array_equal(a.selected_indices, b.selected_indices))


class DualBankTests(unittest.TestCase):
    def test_all_modes_higher_is_anomalous_polarity(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(size=(64, 8)).astype(np.float32)
        anomaly = (rng.normal(size=(32, 8)) + np.array([5, 0, 0, 0, 0, 0, 0, 0])).astype(
            np.float32
        )
        # Negatives near the normal bank; positives near the anomaly bank.
        query_neg = normal[:20] + rng.normal(scale=0.01, size=(20, 8)).astype(np.float32)
        query_pos = anomaly[:20] + rng.normal(scale=0.01, size=(20, 8)).astype(np.float32)
        query = np.concatenate([query_neg, query_pos], axis=0)
        labels = np.array([False] * 20 + [True] * 20)

        for mode in ("normal", "anomaly_diagnostic", "margin", "ratio", "gated_hybrid"):
            scorer = DualBankScorer(
                normal_bank=normal,
                anomaly_banks={1: anomaly},
                mode=mode,
                lambda_a=1.0,
                normal_gate_percentile=50.0,
                knn_metric="L2",
            )
            scorer.fit_calibration(normal)
            out = scorer.score(query)
            self.assertTrue(
                assert_higher_is_anomalous(out["scores"], labels),
                msg=f"polarity failed for mode={mode}",
            )
            self.assertEqual(out["scores"].shape[0], query.shape[0])
            self.assertIn(1, out["d_anomaly_by_class"])
            self.assertTrue(np.all(np.isfinite(out["scores"])))

    def test_robust_z_fit(self):
        cal = fit_robust_z(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
        self.assertGreater(cal.iqr, 0.0)


class HeldoutAggregationTests(unittest.TestCase):
    def test_bootstrap_and_paired_deltas(self):
        values = np.array([0.1, 0.2, 0.15, 0.25])
        ci = bootstrap_mean_ci(values, n_boot=200, seed=0)
        self.assertIn("ci_low", ci)
        self.assertLessEqual(ci["ci_low"], ci["mean"])
        self.assertGreaterEqual(ci["ci_high"], ci["mean"])

        groups = [np.array([0.1, 0.2]), np.array([0.15, 0.25])]
        hci = hierarchical_bootstrap_mean_ci(groups, n_boot=100, seed=0)
        self.assertEqual(hci["n_groups"], 2)

        deltas = paired_deltas(
            [{"fold": 1, "seed": 42, "auprc": 0.4}],
            [{"fold": 1, "seed": 42, "auprc": 0.5}],
        )
        self.assertAlmostEqual(deltas[0]["delta"], 0.1)

        agg = aggregate_heldout_matrix(
            [
                {"condition": "clean_8", "auprc": 0.4, "auroc": 0.7, "fixed_f1": 0.3, "f1_max": 0.35},
                {"condition": "clean_8", "auprc": 0.5, "auroc": 0.75, "fixed_f1": 0.32, "f1_max": 0.36},
            ],
            n_boot=50,
        )
        self.assertIn("clean_8", agg["conditions"])

    def test_frozen_config_rejects_retune(self):
        with self.assertRaisesRegex(ValueError, "retune"):
            validate_frozen_config({"retune_trim": True, "phase12": {"trim_fraction": 0.20}})
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_frozen_config({"phase12": {"trim_fraction": 0.10}})
        validate_frozen_config({"phase12": {"trim_fraction": 0.20, "purification_mode": "fixed_ratio_trim"}})


class _FakeSample:
    def __init__(self, image_id: str, class_present: int):
        self.image_id = image_id
        self.image = np.zeros((32, 64, 3), dtype=np.uint8)
        self.masks_by_class = {
            class_present: np.ones((32, 64), dtype=bool),
        }


if __name__ == "__main__":
    unittest.main()
