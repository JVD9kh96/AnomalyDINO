"""Unit tests for DINO patch signal distribution analysis."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.analysis.aggregation import ScoreAggregator
from src.analysis.config import AnalysisConfig, load_config, save_config
from src.analysis.label_mapping import build_patch_coords, map_patch_labels
from src.analysis.metrics import cohens_d, compute_separability_metrics, distribution_summary
from src.analysis.plotting import plot_distribution_triptych
from src.analysis.scorers import (
    AttentionRolloutScorer,
    ClsPatchCosineScorer,
    PatchL2Scorer,
    SobelFeatureScorer,
)
from src.analysis.types import FeatureBundle
from src.detectors.attention_features import (
    compute_attention_rollout,
    normalize_attention,
    rollout_to_patch_scores,
)
from src.detectors.cls_patch_features import compute_cls_patch_cosine

try:
    import torch
except ImportError:
    torch = None


def _make_bundle(grid=(2, 2), dim=4):
    gh, gw = grid
    n = gh * gw
    patches = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    cls = np.ones(dim, dtype=np.float32)
    return FeatureBundle(
        layer_index=0,
        cls_token=cls,
        patch_tokens=patches,
        grid_size=grid,
        processed_shape=(gh * 14, gw * 14),
        patch_size=14,
    )


def test_patch_label_rules():
    mask = np.zeros((256, 1600), dtype=bool)
    mask[0:20, 0:20] = True
    native = (256, 1600)
    grid = (32, 200)
    ps = 14
    res = 448

    labels_any, _ = map_patch_labels(
        mask, native, grid, ps, res, rule="any_overlap", threshold=0.5
    )
    assert labels_any[0, 0]

    labels_thresh, coords = map_patch_labels(
        mask, native, grid, ps, res, rule="overlap_ratio_threshold", threshold=0.5
    )
    assert labels_thresh.shape == grid
    assert coords.shape[1] == 6


def test_cls_patch_cosine_scorer():
    bundle = _make_bundle()
    config = AnalysisConfig()
    scores = ClsPatchCosineScorer().score(bundle, config)
    assert scores.shape == bundle.grid_size
    assert np.all(np.isfinite(scores))

    direct = compute_cls_patch_cosine(
        bundle.cls_token, bundle.patch_tokens, bundle.grid_size
    )
    assert np.allclose(scores, direct)


def test_patch_l2_scorer():
    bundle = _make_bundle()
    config = AnalysisConfig()
    scores = PatchL2Scorer().score(bundle, config)
    assert scores.shape == bundle.grid_size
    assert np.all(scores >= 0)


def test_sobel_feature_scorer():
    if torch is None:
        return
    bundle = _make_bundle(grid=(4, 4), dim=8)
    config = AnalysisConfig()
    scores = SobelFeatureScorer().score(bundle, config)
    assert scores.shape == (4, 4)


def test_attention_rollout():
    attn = np.eye(5, dtype=np.float32) * 0.5
    rollout = compute_attention_rollout([attn, attn], include_residual=True)
    assert rollout.shape == (5, 5)
    bundle = FeatureBundle(
        layer_index=0,
        cls_token=np.ones(4, dtype=np.float32),
        patch_tokens=np.random.randn(4, 4).astype(np.float32),
        grid_size=(2, 2),
        processed_shape=(28, 28),
        patch_size=14,
        attention=attn,
        attentions_all_layers=[attn],
    )
    config = AnalysisConfig()
    scores = AttentionRolloutScorer().score(bundle, config)
    assert scores.shape == (2, 2)


def test_attention_rollout_4d_input():
    """Detector hooks return (batch, heads, tokens, tokens); rollout must handle this."""
    num_tokens = 5
    attn_4d = np.ones((1, 3, num_tokens, num_tokens), dtype=np.float32) / num_tokens
    rollout = compute_attention_rollout([attn_4d, attn_4d], include_residual=True)
    assert rollout.shape == (num_tokens, num_tokens)
    scores = rollout_to_patch_scores(rollout, grid_size=(2, 2))
    assert scores.shape == (2, 2)


def test_compute_attention_weights():
    if torch is None:
        return
    import torch.nn as nn

    from src.analysis.feature_extractors import DinoFeatureExtractor

    class FakeAttn(nn.Module):
        num_heads = 2

        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12, bias=False)
            self.scale = 2**-0.5

    module = FakeAttn()
    x = torch.randn(1, 3, 4)
    weights = DinoFeatureExtractor._compute_attention_weights(module, x)
    assert weights.shape == (1, 2, 3, 3)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_aggregation_split():
    agg = ScoreAggregator("patch_l2", 0, save_per_image=False)
    scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array([[False, True], [False, True]], dtype=bool)
    coords = build_patch_coords((2, 2), 14)
    agg.add("img1", scores, labels, coords)
    summary = agg.finalize()
    assert summary["healthy"]["count"] == 2
    assert summary["anomaly"]["count"] == 2


def test_metrics_cohens_d():
    healthy = np.array([0.0, 0.1, 0.2])
    anomaly = np.array([1.0, 1.1, 1.2])
    d = cohens_d(healthy, anomaly)
    assert d > 0


def test_plotting_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dist.png"
        plot_distribution_triptych(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.8, 0.9, 1.0]),
            "patch_l2",
            0,
            path,
        )
        assert path.exists()


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(
            "seed: 7\noutput_dir: out\nscorers: [patch_l2]\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_path)
        assert cfg.seed == 7
        assert cfg.scorers == ["patch_l2"]
        save_config(cfg, Path(tmp) / "saved.json")
        assert (Path(tmp) / "saved.json").exists()


def test_distribution_summary_empty():
    s = distribution_summary(np.array([], dtype=np.float32))
    assert s["count"] == 0


def test_separability_metrics():
    healthy = np.zeros(50)
    anomaly = np.ones(50)
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([healthy, anomaly])
    m = compute_separability_metrics(healthy, anomaly, labels, scores)
    assert m["auroc"] == 1.0


if __name__ == "__main__":
    test_patch_label_rules()
    test_cls_patch_cosine_scorer()
    test_patch_l2_scorer()
    test_sobel_feature_scorer()
    test_attention_rollout()
    test_attention_rollout_4d_input()
    test_compute_attention_weights()
    test_aggregation_split()
    test_metrics_cohens_d()
    test_plotting_creates_file()
    test_config_roundtrip()
    test_distribution_summary_empty()
    test_separability_metrics()
    print("All analysis unit tests passed.")
