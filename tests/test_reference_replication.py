from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reference_replication import (
    PHASE3_CONDITIONS,
    build_phase3_report,
    phase3_run_dir,
    save_phase3_report,
)


def _write_run(root: Path, condition: str, seed: int, value: float, *, bad_clean: bool = False) -> None:
    run_dir = phase3_run_dir(root, condition, seed)
    run_dir.mkdir(parents=True)
    clean_ids = [f"clean-{seed}-a", f"clean-{seed}-b"]
    if bad_clean and condition == "auto_purified":
        clean_ids = ["not-paired"]
    metadata = {
        "clean_reference_ids": clean_ids,
        "additional_reference_ids": [] if condition == "clean" else [f"add-{seed}-{i}" for i in range(8)],
        "paired_manifest_id": f"manifest-{seed}",
    }
    metrics = {
        "condition": condition,
        "fold": 0,
        "seed": seed,
        "split_seed": 42,
        "reference": metadata,
        "memory_bank_size": 100 + value,
        "mask": None,
        "sam2_skipped": True,
        "patch": {
            "auprc": value,
            "auroc": value + 0.1,
            "f1_optimal": {"f1": value + 0.2},
            "fixed_threshold": {"f1": value + 0.3, "precision": value + 0.4, "recall": value + 0.5},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "reference_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class ReferenceReplicationTests(unittest.TestCase):
    def test_builds_all_summaries_and_tracks_paired_signs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (42, 43, 44, 45, 46):
                values = {"clean": 0.1, "contaminated_all": 0.12, "auto_purified": 0.11, "oracle_purified": 0.13}
                for condition in PHASE3_CONDITIONS:
                    _write_run(root, condition, seed, values[condition])
            report, rows, deltas = build_phase3_report(root, [42, 43, 44, 45, 46])
            self.assertEqual(report["n_runs"], 20)
            self.assertEqual(report["condition_summary"]["clean"]["auprc"]["mean"], 0.1)
            naive = report["paired_delta_sign_tracking"]["naive_minus_clean"]["auprc"]
            auto = report["paired_delta_sign_tracking"]["auto_minus_naive"]["auprc"]
            self.assertEqual((naive["positive"], naive["negative"], naive["zero"]), (5, 0, 0))
            self.assertEqual((auto["positive"], auto["negative"], auto["zero"]), (0, 5, 0))
            paths = save_phase3_report(root, report, rows, deltas)
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_rejects_unpaired_clean_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (42, 43, 44, 45, 46):
                for condition in PHASE3_CONDITIONS:
                    _write_run(root, condition, seed, 0.1, bad_clean=(seed == 42))
            with self.assertRaisesRegex(ValueError, "Paired clean-reference"):
                build_phase3_report(root, [42, 43, 44, 45, 46])

    def test_requires_five_seeds(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            build_phase3_report("unused", [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
