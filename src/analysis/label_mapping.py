from __future__ import annotations

import numpy as np

from src.severstal.transforms import (
    mask_to_patch_overlap,
    resize_mask_like_model,
)


def map_patch_labels(
    mask: np.ndarray,
    native_shape: tuple[int, int],
    grid_size: tuple[int, int],
    patch_size: int,
    smaller_edge_size: int,
    rule: str = "overlap_ratio_threshold",
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map pixel-level GT mask to patch-level boolean labels and coordinates.

    Returns:
        labels: (grid_h, grid_w) bool
        coords: (N, 6) array [row, col, y1, x1, y2, x2] in processed image space
    """
    aligned = resize_mask_like_model(
        mask, native_shape, smaller_edge_size, patch_size
    )
    overlaps = mask_to_patch_overlap(aligned, grid_size, patch_size)
    grid_h, grid_w = grid_size

    if rule == "center_point":
        labels = np.zeros((grid_h, grid_w), dtype=bool)
        for i in range(grid_h):
            for j in range(grid_w):
                cy = i * patch_size + patch_size // 2
                cx = j * patch_size + patch_size // 2
                if cy < aligned.shape[0] and cx < aligned.shape[1]:
                    labels[i, j] = aligned[cy, cx]
    elif rule == "any_overlap":
        labels = overlaps > 0
    elif rule == "overlap_ratio_threshold":
        labels = overlaps >= threshold
    elif rule == "majority_vote":
        labels = overlaps > 0.5
    else:
        raise ValueError(
            f"Unknown patch label rule: {rule!r}. "
            "Choose center_point, any_overlap, overlap_ratio_threshold, majority_vote."
        )

    coords = build_patch_coords(grid_size, patch_size)
    return labels, coords


def build_patch_coords(
    grid_size: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    grid_h, grid_w = grid_size
    coords = []
    for i in range(grid_h):
        for j in range(grid_w):
            y1 = i * patch_size
            x1 = j * patch_size
            y2 = (i + 1) * patch_size
            x2 = (j + 1) * patch_size
            coords.append([i, j, y1, x1, y2, x2])
    return np.asarray(coords, dtype=np.int32)
