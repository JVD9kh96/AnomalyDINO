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

    def test_direct_overlap_counts_match_aggregate_helper(self):
        from src.detectors.reference_purification_metrics import (
            build_multi_bank_purification_report,
            count_overlap_patches_direct,
        )

        record = _synthetic_record()
        rejected = {
            "naive": {record.image_id: np.zeros_like(record.union_overlap, dtype=bool)},
            "oracle": {
                record.image_id: oracle_rejection_mask(
                    record.union_overlap, "any_overlap"
                )
            },
            "distance_trim_20": {
                record.image_id: np.array([[False, False, True, True]], dtype=bool)
            },
            "random_size_matched": {
                record.image_id: np.array([[True, False, True, False]], dtype=bool)
            },
        }
        report, _ = build_multi_bank_purification_report(
            [record],
            bank_rejected_by_image=rejected,
        )
        direct = count_overlap_patches_direct(
            record.union_overlap, threshold=0.0, operator=">"
        )
        self.assertEqual(
            report["direct_overlap_counts"]["any_overlap"]["n_positive_patches"],
            direct,
        )
        self.assertEqual(
            report["oracle_removal"]["any_overlap"]["n_removed_patches"],
            direct,
        )
        self.assertIn("naive", report["multi_bank_purification_quality"])
        self.assertIn("oracle", report["multi_bank_purification_quality"])

    def test_oracle_rules_remove_expected_synthetic_grid_patches(self):
        overlaps = np.array(
            [[0.0, 0.0, 0.05, 0.15, 0.55]],
            dtype=np.float32,
        )
        self.assertEqual(int(np.sum(oracle_rejection_mask(overlaps, "any_overlap"))), 3)
        self.assertEqual(
            int(np.sum(oracle_rejection_mask(overlaps, "at_least_10_percent"))),
            2,
        )
        self.assertEqual(
            int(np.sum(oracle_rejection_mask(overlaps, "at_least_50_percent"))),
            1,
        )


class ReferencePurificationIndexTests(unittest.TestCase):
    def test_fixed_ratio_and_random_size_matched_indices(self):
        from src.detectors.reference_purification import (
            fixed_ratio_distance_trim_indices,
            random_size_matched_indices,
            selected_indices_to_keep_mask,
        )

        scores = np.array([0.5, 0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        selected = fixed_ratio_distance_trim_indices(scores, trim_fraction=0.20)
        # Keep 80% of 5 => 4 patches; lowest distances first.
        self.assertEqual(selected.size, 4)
        self.assertTrue(np.all(np.isin(selected, [1, 3, 0, 4])))
        keep = selected_indices_to_keep_mask(selected, scores.size)
        self.assertEqual(int(keep.sum()), 4)

        random_a = random_size_matched_indices(5, 4, seed=42)
        random_b = random_size_matched_indices(5, 4, seed=42)
        self.assertTrue(np.array_equal(random_a, random_b))
        self.assertEqual(random_a.size, selected.size)


if __name__ == "__main__":
    unittest.main()
