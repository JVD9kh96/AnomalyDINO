from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.reference_purification_metrics import (
    CandidatePatchOverlaps,
    build_phase2_purification_report,
    compute_purification_quality,
    load_auto_rejection_bundle,
    oracle_rejection_mask,
    save_auto_rejection_bundle,
    save_phase2_purification_artifacts,
)


def _synthetic_record() -> CandidatePatchOverlaps:
    return CandidatePatchOverlaps(
        image_id="candidate.jpg",
        union_overlap=np.array([[0.0, 0.05, 0.20, 0.60]], dtype=np.float32),
        class_overlaps={
            1: np.array([[0.0, 0.05, 0.20, 0.0]], dtype=np.float32),
            2: np.array([[0.0, 0.0, 0.0, 0.60]], dtype=np.float32),
        },
    )


class ReferencePurificationMetricTests(unittest.TestCase):
    def test_oracle_overlap_rules_are_explicit_for_thin_defects(self):
        overlaps = _synthetic_record().union_overlap
        self.assertEqual(int(np.sum(oracle_rejection_mask(overlaps, "any_overlap"))), 3)
        self.assertEqual(
            int(np.sum(oracle_rejection_mask(overlaps, "at_least_10_percent"))),
            2,
        )
        self.assertEqual(
            int(np.sum(oracle_rejection_mask(overlaps, "at_least_50_percent"))),
            1,
        )

    def test_quality_metrics_use_any_overlap_as_thin_defect_truth(self):
        record = _synthetic_record()
        class_maps = np.stack([record.class_overlaps[1], record.class_overlaps[2]])
        rejected = oracle_rejection_mask(
            record.union_overlap,
            "at_least_10_percent",
        )
        quality = compute_purification_quality(
            union_overlaps=record.union_overlap,
            class_overlaps=class_maps,
            rejected_mask=rejected,
            truth_rule="any_overlap",
        )

        overall = quality["overall"]
        self.assertEqual(overall["normal_retention"], 1.0)
        self.assertAlmostEqual(overall["anomalous_rejection_recall"], 2.0 / 3.0)
        self.assertEqual(overall["rejected_patch_precision"], 1.0)
        self.assertEqual(overall["final_contamination"], 0.5)
        self.assertIn("1", quality["by_defect_class"])
        self.assertIn("2", quality["by_defect_class"])

    def test_auto_metrics_capture_false_rejection_and_remaining_contamination(self):
        record = _synthetic_record()
        auto_rejected = {
            record.image_id: np.array([[True, True, False, True]], dtype=bool)
        }
        report, _ = build_phase2_purification_report(
            [record],
            auto_rejected_by_image=auto_rejected,
        )
        auto = report["purification_quality"]["canonical_any_overlap"]["auto"]

        self.assertEqual(auto["overall"]["normal_retention"], 0.0)
        self.assertAlmostEqual(
            auto["overall"]["anomalous_rejection_recall"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            auto["overall"]["rejected_patch_precision"],
            2.0 / 3.0,
        )
        self.assertEqual(auto["overall"]["final_contamination"], 1.0)

    def test_report_and_raw_distribution_are_saved(self):
        report, arrays = build_phase2_purification_report([_synthetic_record()])
        with tempfile.TemporaryDirectory() as tmp:
            report_path, distribution_path = save_phase2_purification_artifacts(
                report=report,
                distribution_arrays=arrays,
                output_dir=tmp,
            )
            self.assertTrue(report_path.exists())
            self.assertTrue(distribution_path.exists())
            with np.load(distribution_path, allow_pickle=False) as distribution:
                self.assertEqual(
                    distribution["union_anomalous_overlaps"].shape,
                    (3,),
                )

    def test_auto_rejection_bundle_round_trip(self):
        masks = np.array([[[True, False], [False, True]]], dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_auto_rejection_bundle(
                Path(tmp) / "phase2_auto_rejections.npz",
                image_ids=["candidate.jpg"],
                rejected_masks=masks,
                method="auto_purified",
            )
            loaded, method = load_auto_rejection_bundle(path)

        self.assertEqual(method, "auto_purified")
        self.assertTrue(np.array_equal(loaded["candidate.jpg"], masks[0]))


if __name__ == "__main__":
    unittest.main()
