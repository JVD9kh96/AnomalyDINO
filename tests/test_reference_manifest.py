from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reference_manifest import (
    REFERENCE_MODES,
    build_paired_reference_manifest,
    load_paired_reference_manifest,
    reference_inputs_for_mode,
    save_paired_reference_manifest,
    validate_paired_reference_manifest,
)


class _FakeDataset:
    image_shape = (256, 1600)
    num_classes = 4

    def __init__(self) -> None:
        self.clean_ids = [f"clean_{index}.jpg" for index in range(6)]
        self.defect_ids = [
            f"class_{class_id}_{index}.jpg"
            for class_id in range(1, 5)
            for index in range(3)
        ]
        self.train_ids = self.clean_ids + self.defect_ids
        self.validation_ids = ["validation_0.jpg", "validation_1.jpg"]

    def get_fold_split(self, fold: int):
        assert fold == 0
        return list(self.train_ids), list(self.validation_ids)

    def select_reference_ids(
        self,
        fold: int,
        shots: int,
        seed: int,
        reference_sampling: str,
    ):
        assert fold == 0
        if reference_sampling == "defect_free":
            return self.clean_ids[:shots]
        if reference_sampling == "class_balanced":
            per_class = shots // self.num_classes
            return [
                f"class_{class_id}_{index}.jpg"
                for class_id in range(1, 5)
                for index in range(per_class)
            ]
        raise AssertionError(reference_sampling)

    def image_has_defect(self, image_id: str) -> bool:
        return image_id.startswith("class_")

    def get_image_classes(self, image_id: str) -> list[int]:
        if not self.image_has_defect(image_id):
            return []
        return [int(image_id.split("_")[1])]


def _build_manifest(additional_sampling: str = "class_balanced"):
    return build_paired_reference_manifest(
        _FakeDataset(),
        fold=0,
        seed=42,
        clean_shots=2,
        additional_shots=8,
        additional_sampling=additional_sampling,
        resolution=448,
        patch_size=14,
    )


class ReferenceManifestTests(unittest.TestCase):
    def test_phase0_manifest_is_deterministic_and_has_expected_patch_budget(self):
        first = _build_manifest()
        second = _build_manifest()

        self.assertEqual(first, second)
        self.assertEqual(len(first["clean_reference_ids"]), 2)
        self.assertEqual(len(first["additional_reference_ids"]), 8)
        self.assertEqual(first["geometry"]["grid_size"], [32, 200])
        self.assertEqual(first["candidate_patch_count_total"], 8 * 6400)
        self.assertTrue(all(first["additional_reference_has_defect"].values()))

    def test_unverified_selection_is_deterministic_without_requiring_defects(self):
        first = _build_manifest(additional_sampling="unverified")
        second = _build_manifest(additional_sampling="unverified")

        self.assertEqual(
            first["additional_reference_ids"], second["additional_reference_ids"]
        )
        self.assertTrue(
            set(first["clean_reference_ids"]).isdisjoint(
                first["additional_reference_ids"]
            )
        )

    def test_manifest_round_trip_and_mode_inputs(self):
        manifest = _build_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_paired_reference_manifest(
                manifest,
                Path(tmp) / "phase0_fold0_seed42_paired_manifest.json",
            )
            loaded = load_paired_reference_manifest(path, dataset=_FakeDataset())

        self.assertEqual(loaded, manifest)
        for mode in REFERENCE_MODES:
            inputs = reference_inputs_for_mode(loaded, mode)
            self.assertEqual(
                inputs["clean_reference_ids"], manifest["clean_reference_ids"]
            )
            if mode in {"clean", "dino_knn_rollout"}:
                self.assertEqual(inputs["additional_reference_ids"], [])
            else:
                self.assertEqual(
                    inputs["additional_reference_ids"],
                    manifest["additional_reference_ids"],
                )

    def test_manifest_rejects_validation_leakage(self):
        manifest = _build_manifest()
        manifest["clean_reference_ids"][0] = manifest["split"]["validation_ids"][0]

        with self.assertRaisesRegex(ValueError, "Validation IDs"):
            validate_paired_reference_manifest(manifest)

    def test_manifest_rejects_manual_edit(self):
        manifest = copy.deepcopy(_build_manifest())
        manifest["seed"] = 43

        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_paired_reference_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
