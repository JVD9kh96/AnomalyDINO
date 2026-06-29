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
from src.detectors.sobel_features import (
    CalibrationStats,
    ScoreModeParams,
    apply_calibration,
    apply_score_mode,
    compute_calibration_stats,
    feature_sobel_norm,
    tokens_to_feature_map,
)
from src.detectors import build_detector
from src.detectors.dino_attention_rollout import DINOv2AttentionRolloutDetector
from src.detectors.dino_cls_cosine import DINOv2ClsPatchCosineDetector
from src.detectors.coreset import greedy_coreset
from src.detectors.dino_features import (
    patch_tokens_to_grid,
    resolve_layer_indices,
    spatial_neighbor_aggregate,
)
from src.detectors.dino_cls_cosine import prototype_anomaly_scores
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.ensemble import EnsembleDetector
from src.detectors.attention_features import compute_attention_rollout, normalize_attention
from src.evaluation.threshold_tuning import (
    default_threshold_grid,
    find_f1_optimal,
    find_recall_at,
    sweep_patch_thresholds,
    ThresholdSweepRow,
)
from src.severstal.transforms import (
    build_gt_patch_labels,
    compute_processed_shape,
    patches_to_bboxes,
    scores_to_patch_predictions,
)

try:
    import torch
except ImportError:
    torch = None


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


def test_tokens_to_feature_map_shape():
    tokens = np.arange(2 * 2 * 3, dtype=np.float32).reshape(4, 3)
    feat = tokens_to_feature_map(tokens, grid_size=(2, 2))
    assert feat.shape == (1, 3, 2, 2)


def test_feature_sobel_norm_shape():
    if torch is None:
        return
    feat = torch.randn(1, 4, 5, 5)
    norms = feature_sobel_norm(feat, norm_reduction="l2")
    assert norms.shape == (1, 5, 5)
    norms_mean = feature_sobel_norm(feat, norm_reduction="mean")
    assert norms_mean.shape == (1, 5, 5)


def test_score_modes():
    norms = np.array([[1.0, 2.0], [3.0, 100.0]], dtype=np.float32)
    calib = compute_calibration_stats(np.array([1.0, 2.0, 3.0, 4.0]))

    raw = apply_score_mode(norms, ScoreModeParams(score_mode="raw"), calib=None)
    assert raw.shape == norms.shape

    zscore = apply_score_mode(
        norms, ScoreModeParams(score_mode="per_image_zscore"), calib=None
    )
    assert np.abs(np.mean(zscore)) < 0.5

    zscore_cal = apply_score_mode(
        norms, ScoreModeParams(score_mode="per_image_zscore"), calib=calib
    )
    assert zscore_cal.shape == norms.shape

    iqr = apply_score_mode(
        norms, ScoreModeParams(score_mode="per_image_iqr", iqr_k=1.5), calib=None
    )
    assert iqr[1, 1] > iqr[0, 0]

    pct = apply_score_mode(
        norms, ScoreModeParams(score_mode="per_image_percentile", percentile=75),
        calib=None,
    )
    assert pct.min() >= 0.0


def test_calibration_stats_from_norms():
    stats = compute_calibration_stats(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert isinstance(stats, CalibrationStats)
    assert stats.ref_mean == 2.0


def test_apply_calibration_zero_shot():
    raw = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    calibrated = apply_calibration(raw, None)
    assert np.allclose(calibrated, raw)


def test_apply_calibration_few_shot():
    ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stats = compute_calibration_stats(ref)
    calibrated = apply_calibration(ref, stats)
    assert np.isclose(np.mean(calibrated), 0.0, atol=1e-5)
    assert np.isclose(np.std(calibrated), 1.0, atol=1e-5)


def test_build_detector_cls_cosine():
    det = build_detector({"name": "dino_cls_cosine", "layer": "last"})
    assert isinstance(det, DINOv2ClsPatchCosineDetector)
    assert det.layer == "last"


def test_build_detector_cls_cosine_prototype():
    det = build_detector(
        {
            "name": "dino_cls_cosine",
            "scoring_mode": "prototype",
            "prototype_reference_sampling": "defect_free",
        }
    )
    assert det.scoring_mode == "prototype"


def test_build_detector_mahalanobis():
    det = build_detector(
        {
            "name": "dino_mahalanobis",
            "layers": [4, 8, 11],
            "pca_components": 10,
        }
    )
    from src.detectors.dino_mahalanobis import DINOv2MahalanobisDetector

    assert isinstance(det, DINOv2MahalanobisDetector)
    assert det.layers == [4, 8, 11]


def test_build_detector_attention_rollout():
    det = build_detector(
        {
            "name": "dino_attention_rollout",
            "attention_rollout": {
                "average_heads": True,
                "include_residual": False,
                "discard_ratio": 0.1,
            },
        }
    )
    assert isinstance(det, DINOv2AttentionRolloutDetector)
    assert det.include_residual is False
    assert det.discard_ratio == 0.1


def test_detector_fit_empty_refs():
    det = DINOv2ClsPatchCosineDetector()
    det.fit([])
    assert det._calib_stats is None

    det2 = DINOv2AttentionRolloutDetector()
    det2.fit([])
    assert det2._calib_stats is None


def test_shots_zero_returns_empty_refs():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train_images").mkdir()
        (root / "train.csv").write_text("ImageId,ClassId,EncodedPixels\n")
        for i in range(8):
            (root / "train_images" / f"{i}.jpg").write_bytes(b"")

        dataset = SeverstalDataset(data_root=root, n_folds=2, seed=0, stratify=False)
        refs = dataset.select_reference_ids(
            fold_idx=0, shots=0, seed=0, reference_sampling="class_balanced"
        )
        assert refs == []


def test_gt_patch_labels():
    masks = {c: np.zeros((256, 1600), dtype=bool) for c in range(1, 5)}
    masks[1][0:50, 0:100] = True
    labels = build_gt_patch_labels(
        masks, (256, 1600), smaller_edge_size=448, patch_size=14, overlap_threshold=0.5
    )
    assert "agnostic" in labels
    assert labels["agnostic"].shape[0] > 0


def test_prototype_anomaly_scores_higher_for_dissimilar():
    ref_cls = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    similar = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    dissimilar = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    sim_scores = prototype_anomaly_scores(ref_cls, similar, (1, 2))
    dis_scores = prototype_anomaly_scores(ref_cls, dissimilar, (1, 2))
    assert dis_scores.mean() > sim_scores.mean()


def test_greedy_coreset_reduces_count():
    features = np.random.randn(100, 8).astype(np.float32)
    core = greedy_coreset(features, ratio=0.1, seed=0)
    assert core.shape[0] == 10
    assert core.shape[1] == 8


def test_spatial_neighbor_aggregate():
    grid = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    out = spatial_neighbor_aggregate(grid, kernel=3)
    assert out.shape == grid.shape


def test_resolve_layer_indices():
    assert resolve_layer_indices("last", 12) == [11]
    assert resolve_layer_indices([4, 8, 11], 12) == [4, 8, 11]


def test_threshold_sweep_and_operating_points():
    per_image = [
        {
            "patch_scores": np.array([0.1, 0.5, 0.9]),
            "gt_labels": {"agnostic": np.array([False, True, True])},
            "valid_mask": None,
        }
    ]
    rows = sweep_patch_thresholds(per_image, np.array([0.2, 0.6]))
    f1_best = find_f1_optimal(rows)
    recall_row = find_recall_at(rows, target_recall=0.5)
    assert isinstance(f1_best, ThresholdSweepRow)
    assert recall_row.recall >= 0.5 or recall_row.recall == max(r.recall for r in rows)


def test_attention_rollout_last_n_layers():
    attn = np.eye(5, dtype=np.float32)
    rollout_all = compute_attention_rollout([attn] * 6, include_residual=False)
    rollout_last2 = compute_attention_rollout(
        [attn] * 6, include_residual=False, last_n_layers=2
    )
    assert rollout_all.shape == rollout_last2.shape


def test_attention_head_reduction_max():
    attn_4d = np.ones((1, 2, 4, 4), dtype=np.float32) * 0.25
    attn_4d[0, 1] = 0.5
    norm_max = normalize_attention(attn_4d, head_reduction="max")
    assert norm_max.shape == (4, 4)


def test_mahalanobis_diagonal_scoring():
    from src.detectors.dino_mahalanobis import DINOv2MahalanobisDetector

    det = DINOv2MahalanobisDetector(pca_components=None)
    det._means = np.zeros((2, 2, 2), dtype=np.float32)
    det._variances = np.ones((2, 2, 2), dtype=np.float32)
    grid = np.zeros((2, 2, 2), dtype=np.float32)
    grid[0, 0] = [3.0, 0.0]
    scores = det._mahalanobis_scores(grid)
    assert scores[0, 0] > scores[1, 1]


def test_ensemble_detector_weighted_sum():
    class _Stub(BaseAnomalyDetector):
        def __init__(self, scores):
            self._scores = scores
            self._patch_size = 14
            self.resolution = 448

        def fit(self, reference_samples):
            pass

        def predict(self, sample):
            return DetectorOutput(
                image_id=sample.image_id,
                patch_scores=self._scores,
                grid_size=(1, 1),
                processed_shape=(14, 14),
                patch_size=14,
            )

        @property
        def supports_class_prediction(self):
            return False

    from src.severstal.dataset import SeverstalSample

    s1 = _Stub(np.array([[1.0]], dtype=np.float32))
    s2 = _Stub(np.array([[3.0]], dtype=np.float32))
    ens = EnsembleDetector([s1, s2], weights=[0.5, 0.5])
    sample = SeverstalSample(
        image_id="t",
        image_path=Path("t.jpg"),
        masks_by_class={},
        has_defect=False,
        image=np.zeros((14, 14, 3), dtype=np.uint8),
    )
    out = ens.predict(sample)
    assert out.patch_scores.shape == (1, 1)


if __name__ == "__main__":
    test_rle_roundtrip()
    test_patch_confusion_metrics()
    test_fold_splits_reproducible()
    test_patches_to_bboxes()
    test_class_balanced_pool_selection()
    test_class_balanced_shots_must_be_divisible()
    test_tokens_to_feature_map_shape()
    test_feature_sobel_norm_shape()
    test_score_modes()
    test_calibration_stats_from_norms()
    test_apply_calibration_zero_shot()
    test_apply_calibration_few_shot()
    test_build_detector_cls_cosine()
    test_build_detector_cls_cosine_prototype()
    test_build_detector_mahalanobis()
    test_build_detector_attention_rollout()
    test_detector_fit_empty_refs()
    test_shots_zero_returns_empty_refs()
    test_gt_patch_labels()
    test_prototype_anomaly_scores_higher_for_dissimilar()
    test_greedy_coreset_reduces_count()
    test_spatial_neighbor_aggregate()
    test_resolve_layer_indices()
    test_threshold_sweep_and_operating_points()
    test_attention_rollout_last_n_layers()
    test_attention_head_reduction_max()
    test_mahalanobis_diagonal_scoring()
    test_ensemble_detector_weighted_sum()
    seed_all(0)
    print("All unit tests passed.")
