"""Supervised linear probe training on frozen DINOv2 patch tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.sampling import resolve_training_ids
    from src.training.trainer import PatchClassifierTrainer, TrainConfig

__all__ = [
    "PatchClassifierTrainer",
    "TrainConfig",
    "resolve_training_ids",
]


def __getattr__(name: str):
    if name in ("PatchClassifierTrainer", "TrainConfig"):
        from src.training.trainer import PatchClassifierTrainer, TrainConfig

        return {
            "PatchClassifierTrainer": PatchClassifierTrainer,
            "TrainConfig": TrainConfig,
        }[name]
    if name == "resolve_training_ids":
        from src.training.sampling import resolve_training_ids

        return resolve_training_ids
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
