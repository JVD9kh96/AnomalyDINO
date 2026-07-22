from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.severstal.dataset import SeverstalDataset, SeverstalSample
from src.severstal.rle import union_masks
from src.severstal.transforms import (
    compute_processed_shape,
    mask_to_patch_overlap,
    resize_mask_like_model,
)


@dataclass
class PatchTargetBundle:
    binary: np.ndarray  # (H, W) float32 {0, 1}
    multiclass: np.ndarray  # (H, W) int64; -1 = non-anomalous
    grid_size: tuple[int, int]


def build_patch_targets(
    masks_by_class: dict[int, np.ndarray],
    native_shape: tuple[int, int],
    smaller_edge_size: int,
    patch_size: int,
    overlap_threshold: float,
    num_classes: int = 4,
) -> PatchTargetBundle:
    """
    Build binary and multiclass patch labels aligned to the model grid.

    Multiclass: among classes with overlap >= threshold, take argmax overlap.
    Non-anomalous patches get multiclass label -1.
    """
    _, grid_size = compute_processed_shape(native_shape, smaller_edge_size, patch_size)
    gh, gw = grid_size
    overlap_stack = np.zeros((num_classes, gh, gw), dtype=np.float32)

    for class_id in range(1, num_classes + 1):
        mask = masks_by_class.get(class_id, np.zeros(native_shape, dtype=bool))
        aligned = resize_mask_like_model(
            mask, native_shape, smaller_edge_size, patch_size
        )
        overlap_stack[class_id - 1] = mask_to_patch_overlap(
            aligned, grid_size, patch_size
        )

    union_mask = union_masks(
        [
            resize_mask_like_model(
                masks_by_class.get(c, np.zeros(native_shape, dtype=bool)),
                native_shape,
                smaller_edge_size,
                patch_size,
            )
            for c in range(1, num_classes + 1)
        ]
    )
    union_overlaps = mask_to_patch_overlap(union_mask, grid_size, patch_size)
    binary = (union_overlaps >= overlap_threshold).astype(np.float32)

    best_class = overlap_stack.argmax(axis=0).astype(np.int64)  # 0..C-1
    best_overlap = overlap_stack.max(axis=0)
    multiclass = np.full((gh, gw), -1, dtype=np.int64)
    anomalous = best_overlap >= overlap_threshold
    multiclass[anomalous] = best_class[anomalous]

    return PatchTargetBundle(binary=binary, multiclass=multiclass, grid_size=grid_size)


class PatchImageDataset(Dataset):
    """Yields one Severstal image + patch targets per item."""

    def __init__(
        self,
        samples: list[SeverstalSample] | Sequence[SeverstalSample],
        resolution: int,
        patch_size: int,
        gt_overlap_threshold: float,
        num_classes: int = 4,
    ):
        self.samples = samples
        self.resolution = resolution
        self.patch_size = patch_size
        self.gt_overlap_threshold = gt_overlap_threshold
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        native_shape = sample.image.shape[:2]
        targets = build_patch_targets(
            sample.masks_by_class,
            native_shape,
            self.resolution,
            self.patch_size,
            self.gt_overlap_threshold,
            num_classes=self.num_classes,
        )
        return {
            "image": sample.image,
            "image_id": sample.image_id,
            "binary": targets.binary,
            "multiclass": targets.multiclass,
            "grid_size": targets.grid_size,
        }


class LazySampleList:
    """Load Severstal samples on demand to avoid holding the full fold in RAM."""

    def __init__(self, dataset: SeverstalDataset, image_ids: list[str]):
        self._dataset = dataset
        self._image_ids = list(image_ids)

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> SeverstalSample:
        return self._dataset.load_sample(self._image_ids[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]



def collate_patch_images(batch: list[dict]) -> dict:
    """Keep images as a list (variable prep handled by backbone); stack labels later."""
    return {
        "images": [b["image"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "binary": [b["binary"] for b in batch],
        "multiclass": [b["multiclass"] for b in batch],
        "grid_sizes": [b["grid_size"] for b in batch],
    }


def flatten_batch_targets(
    binary_list: list[np.ndarray],
    multiclass_list: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    binary = torch.from_numpy(
        np.concatenate([b.ravel() for b in binary_list]).astype(np.float32)
    )
    multiclass = torch.from_numpy(
        np.concatenate([m.ravel() for m in multiclass_list]).astype(np.int64)
    )
    return binary, multiclass
