from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from src.detectors.base import DetectorOutput
from src.evaluation.mask_metrics import compute_dice, compute_iou
from src.segmenters.base import SegmenterOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.rle import union_masks
from src.severstal.transforms import scores_to_patch_predictions


def _upsample_patch_grid(
    patch_grid: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Upsample patch grid to image resolution for overlay."""
    gh, gw = patch_grid.shape
    h, w = image_shape
    cell_h = max(1, h // gh)
    cell_w = max(1, w // gw)
    upsampled = np.repeat(np.repeat(patch_grid, cell_h, axis=0), cell_w, axis=1)
    return upsampled[:h, :w]


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4):
    overlay = image.copy().astype(np.float32) / 255.0
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 1)


def _confusion_colormap(
    gt: np.ndarray,
    pred: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """RGB image: TP=green, FP=red, FN=yellow, TN=transparent."""
    gt_up = _upsample_patch_grid(gt.astype(float), image_shape)
    pred_up = _upsample_patch_grid(pred.astype(float), image_shape)
    h, w = image_shape
    canvas = np.zeros((h, w, 4), dtype=np.float32)
    tp = (gt_up > 0.5) & (pred_up > 0.5)
    fp = (gt_up <= 0.5) & (pred_up > 0.5)
    fn = (gt_up > 0.5) & (pred_up <= 0.5)
    canvas[tp] = [0, 0.8, 0, 0.5]
    canvas[fp] = [0.9, 0, 0, 0.5]
    canvas[fn] = [0.9, 0.8, 0, 0.5]
    return canvas


def plot_sample_page(
    sample: SeverstalSample,
    det_out: DetectorOutput,
    gt_labels: dict[str, np.ndarray],
    pred_labels: np.ndarray,
    seg_out: SegmenterOutput,
    pred_threshold: float,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    image = sample.image
    h, w = image.shape[:2]
    gt_union = union_masks(list(sample.masks_by_class.values()))

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Row 0: patch level
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    gt_overlay = _overlay_mask(image, _upsample_patch_grid(gt_labels["agnostic"], (h, w)), (0, 0.8, 0))
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("GT patches")
    axes[0, 1].axis("off")

    pred_overlay = _overlay_mask(image, _upsample_patch_grid(pred_labels, (h, w)), (0.9, 0, 0))
    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title(f"Pred patches (thr={pred_threshold})")
    axes[0, 2].axis("off")

    conf = _confusion_colormap(gt_labels["agnostic"], pred_labels, (h, w))
    axes[0, 3].imshow(image)
    axes[0, 3].imshow(conf)
    axes[0, 3].set_title("TP/FP/FN (green/red/yellow)")
    axes[0, 3].axis("off")

    # Row 1: mask level
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    gt_mask_overlay = _overlay_mask(image, gt_union, (0, 0.8, 0))
    axes[1, 1].imshow(gt_mask_overlay)
    axes[1, 1].set_title("GT mask")
    axes[1, 1].axis("off")

    pred_mask_overlay = _overlay_mask(image, seg_out.mask, (0.9, 0, 0))
    axes[1, 2].imshow(pred_mask_overlay)
    iou = compute_iou(seg_out.mask, gt_union)
    dice = compute_dice(seg_out.mask, gt_union)
    axes[1, 2].set_title(f"SAM2 mask\nIoU={iou:.3f} Dice={dice:.3f}")
    axes[1, 2].axis("off")

    im = axes[1, 3].imshow(det_out.patch_scores, cmap="hot")
    axes[1, 3].set_title("Patch scores")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
    axes[1, 3].axis("off")

    fig.suptitle(f"{sample.image_id} | defect={sample.has_defect}", fontsize=12)
    plt.tight_layout()
    return fig


def save_visualizations_pdf(
    viz_data: list[dict],
    output_path: str | Path,
    pred_threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for item in viz_data:
            fig = plot_sample_page(
                sample=item["sample"],
                det_out=item["det_out"],
                gt_labels=item["gt_labels"],
                pred_labels=item["pred_labels"],
                seg_out=item["seg_out"],
                pred_threshold=pred_threshold,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
