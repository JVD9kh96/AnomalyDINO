"""Unit tests for Severstal evaluation components (no dataset/GPU required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation.mask_metrics import compute_dice, compute_iou
from src.evaluation.patch_metrics import (
    _compute_confusion,
    _metrics_from_confusion,
    aggregate_global_metrics,
    aggregate_image_mean_metrics,
)
from src.evaluation.reproducibility import create_fold_splits, seed_all
from src.severstal.dataset import SeverstalDataset
from src.severstal.rle import mask2rle, rle2mask, union_masks
from src.severstal.transforms import (
    build_gt_patch_labels,
    compute_processed_shape,
    patches_to_bboxes,
    scores_to_patch_predictions,
)


def test_rle_roundtrip():
    mask = np.zeros((256, 1600), dtype=bool)
    mask[10:20, 100:200] = True
    decoded = rle2mask(mask2rle(mask), (256, 1600))
    assert np.array_equal(decoded, mask)


def test_patch_confusion_metrics():
    gt = np.array([[1, 0], [1, 1]])
    pred = np.array([[1, 0], [0, 1]])
    counts = _compute_confusion(gt, pred)
    assert counts["tp"] == 2
    assert counts["fp"] == 0
    assert counts["fn"] == 1
    metrics = _metrics_from_confusion(counts)
    assert metrics["precision"] == 1.0
    assert abs(metrics["recall"] - 2 / 3) < 1e-6


def test_fold_splits_reproducible():
    ids = [f"img_{i:04d}" for i in range(20)]
    labels = [i % 2 for i in range(20)]
    splits_a = create_fold_splits(ids, labels, n_folds=5, seed=42)
    splits_b = create_fold_splits(ids, labels, n_folds=5, seed=42)
    assert splits_a == splits_b


def test_patches_to_bboxes():
    pred = np.zeros((4, 4), dtype=bool)
    pred[1:3, 2:4] = True
    bboxes = patches_to_bboxes(
        pred,
        patch_size=14,
        processed_shape=(56, 56),
        native_shape=(256, 1600),
    )
    assert len(bboxes) == 1
    x1, y1, x2, y2 = bboxes[0]
    assert x1 < x2 and y1 < y2


def test_class_balanced_pool_selection():
    pool = ["img_a.jpg", "img_b.jpg", "img_c.jpg", "img_d.jpg"]
    selected = SeverstalDataset._select_from_class_pool(
        pool, count=2, seed=0, already_selected=set()
    )
    assert len(selected) == 2
    assert len(set(selected)) == 2

    # Images already picked for another class are skipped
    selected2 = SeverstalDataset._select_from_class_pool(
        pool, count=2, seed=0, already_selected=set(selected)
    )
    assert len(selected2) == 2
    assert not set(selected2) & set(selected)


def test_class_balanced_shots_must_be_divisible():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train_images").mkdir()
        (root / "train.csv").write_text("ImageId,ClassId,EncodedPixels\n")
        for i in range(4):
            (root / "train_images" / f"{i}.jpg").write_bytes(b"")

        dataset = SeverstalDataset(
            data_root=root,
            n_folds=2,
            seed=0,
            stratify=False,
        )
        try:
            dataset._select_class_balanced_reference_ids(
                fold_idx=0, shots=7, seed=0
            )
            raise AssertionError("Expected ValueError for shots=7")
        except ValueError as e:
            assert "divisible" in str(e)


def test_gt_patch_labels():
    masks = {c: np.zeros((256, 1600), dtype=bool) for c in range(1, 5)}
    masks[1][0:50, 0:100] = True
    labels = build_gt_patch_labels(
        masks, (256, 1600), smaller_edge_size=448, patch_size=14, overlap_threshold=0.5
    )
    assert "agnostic" in labels
    assert labels["agnostic"].shape[0] > 0


if __name__ == "__main__":
    test_rle_roundtrip()
    test_patch_confusion_metrics()
    test_fold_splits_reproducible()
    test_patches_to_bboxes()
    test_class_balanced_pool_selection()
    test_class_balanced_shots_must_be_divisible()
    test_gt_patch_labels()
    seed_all(0)
    print("All unit tests passed.")
