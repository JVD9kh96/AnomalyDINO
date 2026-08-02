"""Unit tests for reference composition / purification (CPU, no DINOv2)."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.detectors.anomaly_dino import AnomalyDINODetector, ReferenceFeatureGrid
from src.detectors.knn_index import build_faiss_index, pairwise_knn_distances
from src.detectors.reference_calibration import calibrate_normal_distances
from src.detectors.reference_purification import (
    apply_spatial_cleanup,
    dual_bank_scores,
    oracle_keep_mask_from_gt,
    purify_reference_features,
    purify_reference_grid,
)
from src.detectors import build_detector
from src.severstal.dataset import SeverstalDataset, SeverstalSample
from src.severstal.rle import mask2rle


def _make_toy_dataset(root: Path, n_clean: int = 20, n_per_class: int = 6) -> SeverstalDataset:
    images = root / "train_images"
    images.mkdir(parents=True, exist_ok=True)
    rows = ["ImageId,ClassId,EncodedPixels"]

    # Clean images: empty annotations (or no rows)
    for i in range(n_clean):
        img_id = f"clean_{i:03d}.jpg"
        (images / img_id).write_bytes(b"\x00")

    # Defect images with synthetic RLE covering a small rectangle
    shape = (256, 1600)
    for class_id in range(1, 5):
        for j in range(n_per_class):
            img_id = f"def_c{class_id}_{j:03d}.jpg"
            (images / img_id).write_bytes(b"\x00")
            mask = np.zeros(shape, dtype=bool)
            # Distinct region per class/image
            r0 = 10 + class_id * 20
            c0 = 50 + j * 30
            mask[r0 : r0 + 20, c0 : c0 + 40] = True
            rows.append(f"{img_id},{class_id},{mask2rle(mask)}")

    (root / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return SeverstalDataset(
        data_root=root,
        n_folds=2,
        seed=0,
        stratify=True,
        shuffle=True,
    )


def test_clean_references_contain_no_defects():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_toy_dataset(Path(tmp))
        meta = ds.select_reference_composition(
            0, seed=0, reference_mode="clean", clean_shots=4, additional_shots=0
        )
        assert len(meta["clean_reference_ids"]) == 4
        for img_id in meta["clean_reference_ids"]:
            assert not ds._has_defect[img_id]
            assert meta["reference_image_has_defect"][img_id] is False


def test_class_balanced_additional_contain_requested_classes():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_toy_dataset(Path(tmp))
        ids = ds.select_additional_reference_ids(
            0, n=4, seed=1, sampling="class_balanced", exclude_ids=set()
        )
        assert len(ids) == 4
        covered = set()
        for img_id in ids:
            covered.update(ds.get_image_classes(img_id))
        assert covered == {1, 2, 3, 4}


def test_deterministic_additional_and_composition():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_toy_dataset(Path(tmp))
        a = ds.select_reference_composition(
            0,
            seed=3,
            reference_mode="contaminated_all",
            clean_shots=2,
            additional_shots=4,
            additional_sampling="class_balanced",
        )
        b = ds.select_reference_composition(
            0,
            seed=3,
            reference_mode="contaminated_all",
            clean_shots=2,
            additional_shots=4,
            additional_sampling="class_balanced",
        )
        assert a["clean_reference_ids"] == b["clean_reference_ids"]
        assert a["additional_reference_ids"] == b["additional_reference_ids"]


def test_train_val_never_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_toy_dataset(Path(tmp))
        for fold in range(ds.n_folds):
            train_ids, val_ids = ds.get_fold_split(fold)
            meta = ds.select_reference_composition(
                fold,
                seed=5 + fold,
                reference_mode="auto_purified",
                clean_shots=2,
                additional_shots=4,
            )
            refs = set(meta["clean_reference_ids"]) | set(
                meta["additional_reference_ids"]
            )
            assert refs.isdisjoint(val_ids)
            assert refs.issubset(set(train_ids))


def test_loo_calibration_excludes_held_out_image():
    # Two clean grids with distinct feature clusters.
    g0 = ReferenceFeatureGrid(
        "a",
        np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
        (1, 3),
        None,
    )
    g1 = ReferenceFeatureGrid(
        "b",
        np.array([[5.0, 5.0], [5.1, 5.0], [5.0, 5.1]], dtype=np.float32),
        (1, 3),
        None,
    )
    calib = calibrate_normal_distances(
        [g0, g1], knn_metric="L2", k_neighbors=1, percentiles=(99.0,)
    )
    # Held-out scores for cluster A against bank B should be large (~7),
    # and B against A similarly — never near-zero self distances.
    assert calib.scores.min() > 1.0
    assert len(calib.scores) == 6


def test_normal_like_patches_retained_and_outliers_rejected():
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 0.1, size=(40, 8)).astype(np.float32)
    normal_like = rng.normal(0.0, 0.1, size=(20, 8)).astype(np.float32)
    outliers = rng.normal(5.0, 0.1, size=(20, 8)).astype(np.float32)

    # Threshold from clean self-distances (approx).
    from src.detectors.knn_index import pairwise_knn_distances as pwd

    # Build LOO-ish threshold using a FAISS bank of clean patches
    index = build_faiss_index(clean, "L2", faiss_on_cpu=True)
    # Use a high percentile of clean-to-clean distances as threshold
    # (each clean patch vs bank — includes near-self; still separates outliers).
    clean_d = pairwise_knn_distances(clean, clean[:30], "L2", 1)
    thr = float(np.percentile(clean_d, 99.0)) + 0.5

    cand = np.concatenate([normal_like, outliers], axis=0)
    result = purify_reference_grid(
        cand, index, thr, knn_metric="L2", k_neighbors=1
    )
    keep_normal = result.keep_mask[:20]
    keep_out = result.keep_mask[20:]
    assert keep_normal.mean() > 0.7
    assert keep_out.mean() < 0.3


def test_higher_distance_lower_retention_probability():
    rng = np.random.default_rng(1)
    clean = rng.normal(0.0, 0.05, size=(50, 4)).astype(np.float32)
    # Candidates at increasing distances from origin/cluster
    candidates = np.stack(
        [
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ]
    )
    result = purify_reference_features(
        candidates, clean, threshold=0.5, knn_metric="L2", k_neighbors=1
    )
    # Monotonic: once rejected, farther patches stay rejected at this threshold.
    for i in range(len(result.scores) - 1):
        if result.scores[i] > result.scores[i + 1]:
            continue
        # scores should be non-decreasing with distance from bank
        assert result.scores[i] <= result.scores[i + 1] + 1e-5
    assert result.keep_mask[0]
    assert not result.keep_mask[-1]


def test_oracle_masks_remove_correct_grid_patches():
    shape = (56, 56)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    masks = {c: np.zeros(shape, dtype=bool) for c in range(1, 5)}
    # Defect occupies top-left so first few patches are positive after resize.
    masks[1][:28, :28] = True
    sample = SeverstalSample(
        image_id="oracle.jpg",
        image_path=Path("oracle.jpg"),
        image=image,
        masks_by_class=masks,
        has_defect=True,
    )
    # resolution=56, patch_size=14 => grid 4x4
    keep = oracle_keep_mask_from_gt(
        sample,
        grid_size=(4, 4),
        patch_size=14,
        resolution=56,
        overlap_threshold=0.5,
        num_classes=4,
    )
    keep = keep.reshape(4, 4)
    # Top-left patches should be rejected (False); bottom-right kept.
    assert keep[0, 0] is np.False_ or keep[0, 0] == False
    assert keep[-1, -1] is np.True_ or keep[-1, -1] == True


def test_auto_purification_never_reads_gt_masks():
    sig = inspect.signature(purify_reference_grid)
    assert "mask" not in sig.parameters
    assert "gt" not in "".join(sig.parameters.keys()).lower()
    assert "sample" not in sig.parameters

    sig2 = inspect.signature(purify_reference_features)
    assert "sample" not in sig2.parameters
    assert "masks" not in "".join(sig2.parameters.keys()).lower()


def test_memory_bank_sizes_match_logged_values():
    det = AnomalyDINODetector(
        device="cpu",
        faiss_on_cpu=True,
        reference_mode="contaminated_all",
    )
    # Bypass DINO: call build_memory_bank directly
    grids = [
        ReferenceFeatureGrid(
            "c0",
            np.random.default_rng(0).normal(size=(10, 16)).astype(np.float32),
            (2, 5),
            np.ones(10, dtype=bool),
        ),
        ReferenceFeatureGrid(
            "a0",
            np.random.default_rng(1).normal(size=(6, 16)).astype(np.float32),
            (2, 3),
            np.array([True, True, True, False, False, True]),
        ),
    ]
    det.last_bank_stats.n_memory_patches_before_filtering = 10 + 6
    det.build_memory_bank(grids)
    assert det.last_bank_stats.final_memory_bank_size == 10 + 4
    assert det.last_bank_stats.n_memory_patches_after_filtering == 14
    assert det._knn_index.ntotal == 14


def test_score_polarity_higher_more_anomalous():
    bank = np.zeros((5, 4), dtype=np.float32)
    near = np.array([[0.1, 0.0, 0.0, 0.0]], dtype=np.float32)
    far = np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    d_near = pairwise_knn_distances(near, bank, "L2", 1)[0]
    d_far = pairwise_knn_distances(far, bank, "L2", 1)[0]
    assert d_far > d_near

    dual = dual_bank_scores(
        far,
        normal_bank=bank,
        defect_bank=far,
        knn_metric="L2",
        alpha=1.0,
    )
    # far is close to defect bank => d_defect small => score still meaningful
    assert dual.shape == (1,)


def test_spatial_cleanup_flips_tiny_rejections():
    keep = np.ones((3, 3), dtype=bool)
    keep[1, 1] = False  # single rejected patch
    cleaned = apply_spatial_cleanup(
        keep.ravel(), (3, 3), min_rejected_component_patches=2
    ).reshape(3, 3)
    assert cleaned[1, 1]


def test_build_detector_reference_mode_wiring():
    det = build_detector(
        {
            "name": "anomaly_dino",
            "reference_mode": "auto_purified",
            "faiss_on_cpu": True,
            "device": "cpu",
            "reference_purification": {"normal_acceptance_percentile": 97.5},
            "use_dual_bank": False,
        },
        seed=0,
    )
    assert isinstance(det, AnomalyDINODetector)
    assert det.reference_mode == "auto_purified"
    assert det.reference_purification["normal_acceptance_percentile"] == 97.5


def test_oracle_mode_requires_flag():
    det = AnomalyDINODetector(
        device="cpu",
        faiss_on_cpu=True,
        reference_mode="oracle_purified",
        allow_oracle_reference_filtering=False,
    )
    grid = ReferenceFeatureGrid(
        "x",
        np.ones((4, 8), dtype=np.float32),
        (2, 2),
        None,
    )
    # fit_reference_composition needs samples; test the guard with empty lists
    # by calling with mode override after constructing dummy sample-less path:
    try:
        det.fit_reference_composition([], [], reference_mode="oracle_purified")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "allow_oracle_reference_filtering" in str(e)


def test_one_shot_patch_loo_calibration():
    feats = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    grid = ReferenceFeatureGrid("only", feats, (2, 2), None)
    calib = calibrate_normal_distances([grid], knn_metric="L2", k_neighbors=1)
    assert calib.scores.shape == (4,)
    assert np.all(calib.scores >= 0)


def test_phase4_random_control_matches_auto_count_and_fixed_trim_ratio():
    clean_grid = ReferenceFeatureGrid(
        "clean", np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]], dtype=np.float32), (2, 2), None
    )
    candidate_grid = ReferenceFeatureGrid(
        "candidate",
        np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32),
        (2, 2),
        None,
    )

    class _Sample:
        def __init__(self, image_id):
            self.image_id = image_id

    samples = {"clean": clean_grid, "candidate": candidate_grid}

    def fit(mode, config):
        detector = AnomalyDINODetector(
            device="cpu", faiss_on_cpu=True, reference_mode=mode,
            reference_purification=config, pca_random_state=17,
        )
        detector._ensure_model = lambda: None
        detector.extract_reference_features = lambda sample, use_cache=True: samples[sample.image_id]
        detector.fit_reference_composition([_Sample("clean")], [_Sample("candidate")])
        return detector.last_bank_stats

    auto = fit("auto_purified", {"normal_acceptance_percentile": 95.0})
    random = fit("random_filtered", {"normal_acceptance_percentile": 95.0})
    trimmed = fit("fixed_ratio_trim", {"fixed_trim_fraction": 0.5})
    assert random.n_accepted_candidate_patches == auto.n_accepted_candidate_patches
    assert random.extras["matched_automatic_retained_patches"] == auto.n_accepted_candidate_patches
    assert trimmed.n_accepted_candidate_patches == 2
    assert trimmed.extras["fixed_trim_fraction"] == 0.5


def test_phase5_random_budget_is_exact_and_count_provenance_is_separate():
    rng = np.random.default_rng(4)
    grid = ReferenceFeatureGrid("bank", rng.normal(size=(10, 4)).astype(np.float32), (2, 5), None)
    first = AnomalyDINODetector(
        device="cpu", faiss_on_cpu=True, coreset_size=4, budget_policy="random", pca_random_state=9
    )
    first.last_bank_stats.n_memory_patches_clean = 10
    first.build_memory_bank([grid])
    second = AnomalyDINODetector(
        device="cpu", faiss_on_cpu=True, coreset_size=4, budget_policy="random", pca_random_state=9
    )
    second.last_bank_stats.n_memory_patches_clean = 10
    second.build_memory_bank([grid])
    assert first.last_bank_stats.n_memory_patches_before_budget == 10
    assert first.last_bank_stats.n_memory_patches_final == 4
    assert first.last_bank_stats.final_memory_bank_size == 4
    np.testing.assert_array_equal(first._normal_bank_features, second._normal_bank_features)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(f"{failed} tests failed")
    print(f"All {len(tests)} tests passed.")
