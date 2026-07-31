from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration_report import (
    CANDIDATE_SCORE_METHOD,
    QUERY_SCORE_METHOD,
    build_phase1_calibration_report,
    build_report_from_score_bundle,
    collect_cross_fitted_clean_score_sets,
    create_and_save_phase1_calibration_report,
    load_phase1_score_bundle,
    save_phase1_score_bundle,
)


class CalibrationReportTests(unittest.TestCase):
    def test_cross_fit_excludes_held_out_clean_image_from_both_banks(self):
        calls = []

        def scorer(held_out_id, bank_ids, bank_stage):
            calls.append((held_out_id, list(bank_ids), bank_stage))
            self.assertNotIn(held_out_id, bank_ids)
            return np.array([0.1, 0.2], dtype=np.float32)

        scores = collect_cross_fitted_clean_score_sets(
            clean_reference_ids=["clean_a", "clean_b"],
            final_reference_ids=["clean_a", "clean_b", "candidate_a"],
            score_held_out=scorer,
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(scores["candidate_score_method"], CANDIDATE_SCORE_METHOD)
        self.assertEqual(scores["query_score_method"], QUERY_SCORE_METHOD)
        self.assertEqual(
            scores["held_out_clean_candidate_distances"].shape, (4,)
        )
        self.assertEqual(scores["cross_fitted_clean_query_scores"].shape, (4,))

    def test_report_separates_deployable_and_f1_max_thresholds(self):
        report = build_phase1_calibration_report(
            held_out_clean_candidate_distances=np.array([1.0, 2.0, 3.0, 4.0]),
            cross_fitted_clean_query_scores=np.array([0.1, 0.2, 0.3, 0.4]),
            validation_scores=np.array([0.1, 0.2, 0.3, 0.4]),
            validation_gt_labels=np.array([0, 1, 0, 1]),
            clean_bank_id="clean-bank",
            final_bank_id="final-bank",
            candidate_acceptance_percentile=75.0,
            query_percentile=50.0,
        )

        self.assertAlmostEqual(report["candidate_acceptance"]["threshold"], 3.25)
        self.assertAlmostEqual(report["query_operating_point"]["threshold"], 0.25)
        self.assertAlmostEqual(report["f1_max_threshold"], 0.2)
        self.assertEqual(report["n_gt_positive_patches"], 2)
        self.assertEqual(report["n_pred_positive_patches"], 2)
        self.assertAlmostEqual(report["precision"], 0.5)
        self.assertAlmostEqual(report["recall"], 0.5)
        self.assertFalse(report["safeguards"]["query_threshold_uses_validation_gt"])
        self.assertFalse(report["safeguards"]["f1_max_is_deployable"])

    def test_valid_mask_controls_report_counts(self):
        report = build_phase1_calibration_report(
            held_out_clean_candidate_distances=np.array([0.1, 0.2]),
            cross_fitted_clean_query_scores=np.array([0.1, 0.2]),
            validation_scores=np.array([0.0, 0.3, 0.4]),
            validation_gt_labels=np.array([1, 1, 0]),
            validation_valid_mask=np.array([False, True, True]),
            clean_bank_id="clean-bank",
            final_bank_id="final-bank",
            query_percentile=50.0,
        )

        self.assertEqual(report["n_gt_positive_patches"], 1)
        self.assertEqual(report["n_pred_positive_patches"], 2)

    def test_score_bundle_round_trip(self):
        cross_fitted = {
            "held_out_clean_candidate_distances": np.array([1.0, 2.0]),
            "cross_fitted_clean_query_scores": np.array([0.1, 0.2]),
            "candidate_score_method": CANDIDATE_SCORE_METHOD,
            "query_score_method": QUERY_SCORE_METHOD,
            "candidate_self_exclusion": True,
            "query_self_exclusion": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = save_phase1_score_bundle(
                Path(tmp) / "phase1_scores.npz",
                cross_fitted_scores=cross_fitted,
                validation_scores=np.array([0.1, 0.2]),
                validation_gt_labels=np.array([0, 1]),
                final_bank_id="final-bank",
                clean_bank_id="clean-bank",
            )
            bundle = load_phase1_score_bundle(path)
            report = build_report_from_score_bundle(bundle)

        self.assertEqual(
            report["query_operating_point"]["final_bank_id"], "final-bank"
        )

    def test_report_rejects_non_cross_fitted_query_provenance(self):
        kwargs = {
            "held_out_clean_candidate_distances": np.array([0.1, 0.2]),
            "cross_fitted_clean_query_scores": np.array([0.1, 0.2]),
            "validation_scores": np.array([0.1, 0.2]),
            "validation_gt_labels": np.array([0, 1]),
            "clean_bank_id": "clean-bank",
            "final_bank_id": "final-bank",
        }
        invalid = copy.deepcopy(kwargs)
        invalid["query_score_method"] = "validation_f1_tuning"

        with self.assertRaisesRegex(ValueError, "final-bank cross-fitted"):
            build_phase1_calibration_report(**invalid)

    def test_one_call_hook_saves_one_report_per_run(self):
        def scorer(held_out_id, bank_ids, bank_stage):
            del held_out_id, bank_ids
            if bank_stage == "candidate_acceptance_clean_bank":
                return np.array([1.0, 2.0])
            return np.array([0.1, 0.2])

        with tempfile.TemporaryDirectory() as tmp:
            report, output_path = create_and_save_phase1_calibration_report(
                clean_reference_ids=["clean_a", "clean_b"],
                final_reference_ids=["clean_a", "clean_b", "candidate_a"],
                score_held_out=scorer,
                validation_scores=np.array([0.1, 0.2, 0.3]),
                validation_gt_labels=np.array([0, 1, 1]),
                final_bank_id="final-bank",
                clean_bank_id="clean-bank",
                run_dir=tmp,
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "phase1_calibration_report.json")
            self.assertEqual(report["phase"], "phase1")


if __name__ == "__main__":
    unittest.main()
