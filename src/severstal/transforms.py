import cv2
import numpy as np
from scipy import ndimage

from src.severstal.rle import union_masks


def compute_processed_shape(
    native_shape: tuple[int, int],
    smaller_edge_size: int,
    patch_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Mirror DINOv2 prepare_image geometry: resize smaller edge, crop to patch multiple.

    Returns (processed_shape (H, W), grid_size (grid_h, grid_w)).
    """
    h, w = native_shape
    if h < w:
        new_h = smaller_edge_size
        new_w = int(round(w * smaller_edge_size / h))
    else:
        new_w = smaller_edge_size
        new_h = int(round(h * smaller_edge_size / w))

    cropped_h = new_h - new_h % patch_size
    cropped_w = new_w - new_w % patch_size
    grid_h = cropped_h // patch_size
    grid_w = cropped_w // patch_size
    return (cropped_h, cropped_w), (grid_h, grid_w)


def resize_mask_like_model(
    mask: np.ndarray,
    native_shape: tuple[int, int],
    smaller_edge_size: int,
    patch_size: int,
) -> np.ndarray:
    """Resize and crop a boolean mask to match model preprocessing."""
    h, w = native_shape
    if h < w:
        new_h = smaller_edge_size
        new_w = int(round(w * smaller_edge_size / h))
    else:
        new_w = smaller_edge_size
        new_h = int(round(h * smaller_edge_size / w))

    resized = cv2.resize(
        mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    cropped_h = new_h - new_h % patch_size
    cropped_w = new_w - new_w % patch_size
    return resized[:cropped_h, :cropped_w]


def mask_to_patch_overlap(
    mask: np.ndarray,
    grid_size: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Fraction of defective pixels per patch cell, shape (grid_h, grid_w)."""
    grid_h, grid_w = grid_size
    overlaps = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            patch = mask[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            overlaps[i, j] = patch.mean() if patch.size > 0 else 0.0
    return overlaps


def overlap_to_patch_labels(
    overlaps: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return overlaps >= threshold


def scores_to_patch_predictions(
    patch_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return patch_scores >= threshold


def build_gt_patch_labels(
    masks_by_class: dict[int, np.ndarray],
    native_shape: tuple[int, int],
    smaller_edge_size: int,
    patch_size: int,
    overlap_threshold: float,
    num_classes: int = 4,
) -> dict[str, np.ndarray]:
    """
    Build class-agnostic and per-class GT patch label grids aligned to model grid.
    """
    _, grid_size = compute_processed_shape(native_shape, smaller_edge_size, patch_size)

    class_labels = {}
    aligned_masks = {}
    for class_id in range(1, num_classes + 1):
        mask = masks_by_class.get(class_id, np.zeros(native_shape, dtype=bool))
        aligned = resize_mask_like_model(mask, native_shape, smaller_edge_size, patch_size)
        overlaps = mask_to_patch_overlap(aligned, grid_size, patch_size)
        class_labels[str(class_id)] = overlap_to_patch_labels(overlaps, overlap_threshold)
        aligned_masks[class_id] = aligned

    union_mask = union_masks(list(aligned_masks.values()))
    union_overlaps = mask_to_patch_overlap(union_mask, grid_size, patch_size)
    class_labels["agnostic"] = overlap_to_patch_labels(union_overlaps, overlap_threshold)
    return class_labels


def patches_to_bboxes(
    pred_patches: np.ndarray,
    patch_size: int,
    processed_shape: tuple[int, int],
    native_shape: tuple[int, int],
    min_prompt_area: int = 1,
    connectivity: int = 2,
) -> list[list[float]]:
    """
    Convert predicted anomalous patch grid to bounding boxes in native image coords.

    Returns list of [x1, y1, x2, y2] in native (W, H) pixel coordinates for SAM2.
    """
    labeled, num_features = ndimage.label(pred_patches, structure=np.ones((3, 3)))
    if num_features == 0:
        return []

    proc_h, proc_w = processed_shape
    native_h, native_w = native_shape
    scale_x = native_w / proc_w
    scale_y = native_h / proc_h

    bboxes = []
    for label_id in range(1, num_features + 1):
        component = labeled == label_id
        if component.sum() < min_prompt_area:
            continue
        rows, cols = np.where(component)
        y1 = rows.min() * patch_size
        x1 = cols.min() * patch_size
        y2 = (rows.max() + 1) * patch_size
        x2 = (cols.max() + 1) * patch_size

        # Scale to native coordinates (x, y order for SAM2)
        nx1 = x1 * scale_x
        ny1 = y1 * scale_y
        nx2 = x2 * scale_x
        ny2 = y2 * scale_y
        bboxes.append([float(nx1), float(ny1), float(nx2), float(ny2)])

    return bboxes


def patches_to_points(
    pred_patches: np.ndarray,
    patch_size: int,
    processed_shape: tuple[int, int],
    native_shape: tuple[int, int],
    min_prompt_area: int = 1,
) -> list[list[float]]:
    """Convert predicted anomalous patches to center points in native coords."""
    bboxes = patches_to_bboxes(
        pred_patches, patch_size, processed_shape, native_shape, min_prompt_area
    )
    points = []
    for x1, y1, x2, y2 in bboxes:
        points.append([(x1 + x2) / 2, (y1 + y2) / 2])
    return points
