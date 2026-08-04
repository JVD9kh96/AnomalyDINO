from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase5_memory_budget_controls import (
    FINAL_BUDGET,
    active_specs,
    additive_specs,
    aggregate,
    specs,
)


class Phase5AdditiveBudgetTests(unittest.TestCase):
    def test_additive_rows_target_exact_budget(self):
        rows = additive_specs(include_optional_4plus8=True)
        names = {row["name"] for row in rows}
        self.assertIn("naive_greedy_budget_2plus8", names)
        self.assertIn("random20_greedy_budget_2plus8", names)
        self.assertIn("oracle_greedy_budget_2plus8", names)
        self.assertIn("naive_greedy_budget_4plus8", names)
        for row in rows:
            self.assertEqual(row["budget"], FINAL_BUDGET)
            self.assertEqual(row["budget_policy"], "greedy_coreset")

    def test_random20_and_distance20_share_candidate_retention_target(self):
        additive = {row["name"]: row for row in additive_specs()}
        random20 = additive["random20_greedy_budget_2plus8"]
        self.assertEqual(random20["filter"], "random_reject_20")
        self.assertAlmostEqual(float(random20["fixed_trim_fraction"]), 0.20)
        proposed = [
            row
            for row in specs("fixed_ratio_trim")
            if row["name"] == "purified_budget_2plus8"
        ][0]
        self.assertEqual(proposed["budget"], random20["budget"])
        self.assertEqual(proposed["clean_shots"], random20["clean_shots"])
        self.assertEqual(proposed["additional_shots"], random20["additional_shots"])

    def test_append_rows_merges_without_dropping_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prior = {
                "phase": "phase5_exact_memory_budget_controls",
                "rows": [
                    {
                        "name": "purified_budget_2plus8",
                        "budget": FINAL_BUDGET,
                        "n_memory_patches_final": FINAL_BUDGET,
                        "budget_exact": True,
                    }
                ],
                "notes": ["prior"],
            }
            report_path = output_dir / "phase5_exact_memory_budget_controls_report.json"
            report_path.write_text(json.dumps(prior), encoding="utf-8")

            # Fabricate additive metrics so aggregate can merge.
            for spec in additive_specs(include_optional_4plus8=False):
                run_dir = output_dir / f"phase5_f0_{spec['name']}_s42"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "fold": 0,
                            "seed": 42,
                            "split_seed": 42,
                            "patch": {
                                "auprc": 0.5,
                                "auroc": 0.6,
                                "fixed_threshold": {"f1": 0.4},
                                "f1_optimal": {"f1": 0.45},
                            },
                            "reference": {
                                "bank_stats": {
                                    "n_memory_patches_clean": 12800,
                                    "n_candidate_patches_before_filter": 51200,
                                    "n_candidate_patches_after_filter": 40960,
                                    "n_memory_patches_before_budget": 53760,
                                    "n_memory_patches_final": FINAL_BUDGET,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            out = aggregate(
                output_dir,
                {"condition": "fixed_ratio_trim", "trim_fraction": 0.20},
                append_rows=True,
                include_optional_4plus8=False,
            )
            merged = json.loads(out.read_text(encoding="utf-8"))
            names = [row["name"] for row in merged["rows"]]
            self.assertIn("purified_budget_2plus8", names)
            self.assertIn("naive_greedy_budget_2plus8", names)
            self.assertTrue(all(
                row.get("budget") in (None, FINAL_BUDGET)
                or row["name"] == "purified_budget_2plus8"
                for row in merged["rows"]
            ))
            for row in merged["rows"]:
                if row["name"].startswith(("naive_greedy", "random20", "oracle_greedy")):
                    self.assertEqual(row["n_memory_patches_final"], FINAL_BUDGET)
                    self.assertTrue(row["budget_exact"])

    def test_append_only_active_specs(self):
        append_only = active_specs(
            "fixed_ratio_trim",
            append_rows=True,
            include_optional_4plus8=False,
        )
        self.assertEqual(len(append_only), 3)
        full = active_specs(
            "fixed_ratio_trim",
            append_rows=False,
            include_optional_4plus8=False,
        )
        self.assertGreater(len(full), len(append_only))


if __name__ == "__main__":
    unittest.main()
