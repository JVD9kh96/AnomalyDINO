from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.severstal.dataset import SeverstalDataset
from src.severstal.transforms import compute_processed_shape


MANIFEST_SCHEMA_VERSION = 1

# These are input policies, not claims that every scoring implementation is present.
REFERENCE_MODES = (
    "clean",
    "contaminated_all",
    "class_balanced_all",
    "oracle_purified",
    "auto_purified",
    "random_filtered",
    "fixed_ratio_trim",
    "gt_anomaly_bank",
    "dino_knn_rollout",
    "normal_anomaly_rollout",
)

_CLEAN_ONLY_MODES = {"clean", "dino_knn_rollout"}
_ADDITIONAL_SAMPLING_MODES = {"class_balanced", "defect_only", "unverified"}


def _ids_sha256(image_ids: list[str]) -> str:
    payload = "\n".join(sorted(image_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _manifest_id(manifest_without_id: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest_without_id).encode("utf-8")).hexdigest()


def _stable_select(
    image_ids: list[str],
    count: int,
    seed: int,
    namespace: str,
) -> list[str]:
    """Select IDs by a stable seeded hash, independent of input ordering."""
    ranked = sorted(
        set(image_ids),
        key=lambda image_id: (
            hashlib.sha256(
                f"{namespace}\0{seed}\0{image_id}".encode("utf-8")
            ).hexdigest(),
            image_id,
        ),
    )
    return ranked[:count]


def build_paired_reference_manifest(
    dataset: SeverstalDataset,
    *,
    fold: int,
    seed: int,
    clean_shots: int,
    additional_shots: int,
    additional_sampling: str = "class_balanced",
    resolution: int = 448,
    patch_size: int = 14,
) -> dict[str, Any]:
    """Build the deterministic Phase 0 input manifest for one fold and seed.

    Ground-truth status and class metadata are recorded only after candidate IDs
    have been selected. The ``unverified`` policy therefore does not use masks or
    labels to choose additional images.
    """
    if clean_shots < 0 or additional_shots < 0:
        raise ValueError("clean_shots and additional_shots must be non-negative")
    if additional_sampling not in _ADDITIONAL_SAMPLING_MODES:
        choices = ", ".join(sorted(_ADDITIONAL_SAMPLING_MODES))
        raise ValueError(
            f"Unknown additional_sampling={additional_sampling!r}; choose {choices}"
        )
    if resolution <= 0 or patch_size <= 0:
        raise ValueError("resolution and patch_size must be positive")

    train_ids, validation_ids = dataset.get_fold_split(fold)
    train_set = set(train_ids)

    clean_reference_ids = dataset.select_reference_ids(
        fold,
        clean_shots,
        seed,
        reference_sampling="defect_free",
    )
    if len(clean_reference_ids) != clean_shots:
        raise ValueError(
            f"Requested {clean_shots} clean references, selected "
            f"{len(clean_reference_ids)}"
        )

    if additional_sampling == "class_balanced":
        additional_reference_ids = dataset.select_reference_ids(
            fold,
            additional_shots,
            seed,
            reference_sampling="class_balanced",
        )
    else:
        candidate_ids = [
            image_id
            for image_id in train_ids
            if image_id not in clean_reference_ids
            and (
                additional_sampling == "unverified"
                or dataset.image_has_defect(image_id)
            )
        ]
        additional_reference_ids = _stable_select(
            candidate_ids,
            additional_shots,
            seed,
            namespace=f"phase0-fold-{fold}-{additional_sampling}",
        )

    # This also protects future sampling implementations that do not naturally
    # separate normal seeds and additional candidates.
    additional_reference_ids = [
        image_id
        for image_id in additional_reference_ids
        if image_id not in clean_reference_ids
    ]
    if len(additional_reference_ids) != additional_shots:
        raise ValueError(
            f"Requested {additional_shots} additional references, selected "
            f"{len(additional_reference_ids)}"
        )

    reference_ids = clean_reference_ids + additional_reference_ids
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference selection contains duplicate image IDs")
    if not set(reference_ids).issubset(train_set):
        raise ValueError("Reference selection contains IDs outside the training split")

    native_shape = tuple(int(v) for v in dataset.image_shape)
    processed_shape, grid_size = compute_processed_shape(
        native_shape,
        resolution,
        patch_size,
    )
    patches_per_image = int(grid_size[0] * grid_size[1])

    additional_reference_classes = {
        image_id: dataset.get_image_classes(image_id)
        for image_id in additional_reference_ids
    }
    additional_reference_has_defect = {
        image_id: dataset.image_has_defect(image_id)
        for image_id in additional_reference_ids
    }
    candidate_patch_counts = {
        image_id: patches_per_image for image_id in additional_reference_ids
    }

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "phase0",
        "study": "clean_seed_reference_expansion",
        "fold": int(fold),
        "seed": int(seed),
        "clean_reference_ids": clean_reference_ids,
        "additional_reference_ids": additional_reference_ids,
        "additional_reference_classes": additional_reference_classes,
        "additional_reference_has_defect": additional_reference_has_defect,
        "candidate_patch_counts": candidate_patch_counts,
        "candidate_patch_count_total": int(sum(candidate_patch_counts.values())),
        "selection": {
            "clean_sampling": "defect_free",
            "clean_shots": int(clean_shots),
            "additional_sampling": additional_sampling,
            "additional_shots": int(additional_shots),
        },
        "geometry": {
            "native_shape": list(native_shape),
            "resolution": int(resolution),
            "patch_size": int(patch_size),
            "processed_shape": list(processed_shape),
            "grid_size": list(grid_size),
            "patches_per_image": patches_per_image,
        },
        "split": {
            "train_count": len(train_ids),
            "validation_count": len(validation_ids),
            "train_ids_sha256": _ids_sha256(train_ids),
            "validation_ids_sha256": _ids_sha256(validation_ids),
            "validation_ids": list(validation_ids),
        },
    }
    manifest["manifest_id"] = _manifest_id(manifest)
    validate_paired_reference_manifest(manifest, dataset=dataset)
    return manifest


def validate_paired_reference_manifest(
    manifest: dict[str, Any],
    *,
    dataset: SeverstalDataset | None = None,
) -> None:
    """Fail closed when a paired manifest is incomplete, edited, or leaks data."""
    required = {
        "schema_version",
        "phase",
        "fold",
        "seed",
        "clean_reference_ids",
        "additional_reference_ids",
        "additional_reference_classes",
        "additional_reference_has_defect",
        "candidate_patch_counts",
        "candidate_patch_count_total",
        "selection",
        "geometry",
        "split",
        "manifest_id",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={manifest['schema_version']!r}"
        )
    if manifest["phase"] != "phase0":
        raise ValueError(f"Expected phase='phase0', got {manifest['phase']!r}")

    clean_ids = list(manifest["clean_reference_ids"])
    additional_ids = list(manifest["additional_reference_ids"])
    reference_ids = clean_ids + additional_ids
    validation_ids = list(manifest["split"].get("validation_ids", []))

    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference IDs are duplicated within or across input pools")
    leaked = sorted(set(reference_ids) & set(validation_ids))
    if leaked:
        raise ValueError(f"Validation IDs appear in a reference pool: {leaked[:5]}")

    expected_additional_keys = set(additional_ids)
    for field in (
        "additional_reference_classes",
        "additional_reference_has_defect",
        "candidate_patch_counts",
    ):
        actual_keys = set(manifest[field])
        if actual_keys != expected_additional_keys:
            raise ValueError(
                f"{field} keys do not match additional_reference_ids: "
                f"missing={sorted(expected_additional_keys - actual_keys)[:5]}, "
                f"extra={sorted(actual_keys - expected_additional_keys)[:5]}"
            )

    patch_counts = manifest["candidate_patch_counts"]
    if any(not isinstance(value, int) or value <= 0 for value in patch_counts.values()):
        raise ValueError("Every candidate patch count must be a positive integer")
    if sum(patch_counts.values()) != manifest["candidate_patch_count_total"]:
        raise ValueError("candidate_patch_count_total does not match per-image counts")

    selection = manifest["selection"]
    if len(clean_ids) != selection.get("clean_shots"):
        raise ValueError("clean_reference_ids count does not match clean_shots")
    if len(additional_ids) != selection.get("additional_shots"):
        raise ValueError(
            "additional_reference_ids count does not match additional_shots"
        )

    expected_id = _manifest_id(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    if manifest["manifest_id"] != expected_id:
        raise ValueError("Manifest fingerprint mismatch; the file was modified")

    if dataset is None:
        return

    fold = int(manifest["fold"])
    train_ids, current_validation_ids = dataset.get_fold_split(fold)
    train_set = set(train_ids)
    if list(current_validation_ids) != validation_ids:
        raise ValueError("Manifest validation IDs do not match the current fold split")
    if manifest["split"].get("train_ids_sha256") != _ids_sha256(train_ids):
        raise ValueError("Manifest training split fingerprint does not match the dataset")
    if manifest["split"].get("validation_ids_sha256") != _ids_sha256(
        current_validation_ids
    ):
        raise ValueError(
            "Manifest validation split fingerprint does not match the dataset"
        )
    outside_train = sorted(set(reference_ids) - train_set)
    if outside_train:
        raise ValueError(f"Reference IDs are outside the training split: {outside_train[:5]}")
    dirty_clean = [image_id for image_id in clean_ids if dataset.image_has_defect(image_id)]
    if dirty_clean:
        raise ValueError(f"Clean reference IDs contain defects: {dirty_clean[:5]}")

    for image_id in additional_ids:
        if manifest["additional_reference_has_defect"][image_id] != (
            dataset.image_has_defect(image_id)
        ):
            raise ValueError(f"Defect-status drift for candidate {image_id}")
        if manifest["additional_reference_classes"][image_id] != (
            dataset.get_image_classes(image_id)
        ):
            raise ValueError(f"Class metadata drift for candidate {image_id}")


def save_paired_reference_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> Path:
    validate_paired_reference_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def load_paired_reference_manifest(
    path: str | Path,
    *,
    dataset: SeverstalDataset | None = None,
) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        manifest = json.load(file)
    validate_paired_reference_manifest(manifest, dataset=dataset)
    return manifest


def reference_inputs_for_mode(
    manifest: dict[str, Any],
    mode: str,
) -> dict[str, list[str]]:
    """Resolve immutable clean/candidate inputs for a reference-study mode."""
    validate_paired_reference_manifest(manifest)
    if mode not in REFERENCE_MODES:
        raise ValueError(
            f"Unknown reference mode {mode!r}; choose one of {', '.join(REFERENCE_MODES)}"
        )
    return {
        "clean_reference_ids": list(manifest["clean_reference_ids"]),
        "additional_reference_ids": (
            []
            if mode in _CLEAN_ONLY_MODES
            else list(manifest["additional_reference_ids"])
        ),
    }
