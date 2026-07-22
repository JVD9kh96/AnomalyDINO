from __future__ import annotations

from src.severstal.dataset import SeverstalDataset


def resolve_training_ids(
    dataset: SeverstalDataset,
    fold_idx: int,
    shots: int | None,
    seed: int,
    reference_sampling: str = "class_balanced",
) -> list[str]:
    """
    Resolve supervised training image IDs for a fold.

    - shots is None → all images in the fold train split (defect + defect-free)
    - shots is int → same deterministic selection as k-shot CV
      (including -1 = all class_balanced / defect_free eligible)
    """
    train_ids, _ = dataset.get_fold_split(fold_idx)
    if shots is None:
        return list(train_ids)
    return dataset.select_reference_ids(
        fold_idx,
        shots,
        seed,
        reference_sampling=reference_sampling,
    )
