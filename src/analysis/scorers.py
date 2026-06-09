from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from src.analysis.config import AnalysisConfig
from src.analysis.types import FeatureBundle
from src.detectors.sobel_features import feature_sobel_norm, tokens_to_feature_map


class BaseScorer(ABC):
    name: str

    @abstractmethod
    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        """Return patch score grid (grid_h, grid_w)."""


class ClsPatchCosineScorer(BaseScorer):
    name = "cls_patch_cosine"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        cls = bundle.cls_token.astype(np.float32)
        patches = bundle.patch_tokens.astype(np.float32)
        cls_norm = cls / (np.linalg.norm(cls) + 1e-8)
        patch_norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
        patches_normed = patches / patch_norms
        sims = patches_normed @ cls_norm
        return sims.reshape(bundle.grid_size).astype(np.float32)


class PatchL2Scorer(BaseScorer):
    name = "patch_l2"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        norms = np.linalg.norm(bundle.patch_tokens, axis=1)
        return norms.reshape(bundle.grid_size).astype(np.float32)


class SobelFeatureScorer(BaseScorer):
    name = "sobel_feature"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        feat_map = tokens_to_feature_map(bundle.patch_tokens, bundle.grid_size)
        norms = feature_sobel_norm(feat_map, config.sobel.norm_reduction)
        return norms.squeeze(0).detach().cpu().numpy().astype(np.float32)


class SobelImageScorer(BaseScorer):
    name = "sobel_image"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        if bundle.preprocessed_image is None:
            raise ValueError("preprocessed_image required for sobel_image scorer")
        gray = cv2.cvtColor(bundle.preprocessed_image, cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx * gx + gy * gy)

        grid_h, grid_w = bundle.grid_size
        ps = bundle.patch_size
        scores = np.zeros((grid_h, grid_w), dtype=np.float32)
        for i in range(grid_h):
            for j in range(grid_w):
                patch = magnitude[
                    i * ps : (i + 1) * ps,
                    j * ps : (j + 1) * ps,
                ]
                if config.sobel.image_reduction == "mean":
                    scores[i, j] = patch.mean()
                elif config.sobel.image_reduction == "l2":
                    scores[i, j] = np.linalg.norm(patch)
                else:
                    scores[i, j] = patch.max()
        return scores


class AttentionRolloutScorer(BaseScorer):
    name = "attention_rollout"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        attentions = bundle.attentions_all_layers
        if not attentions:
            if bundle.attention is not None:
                attentions = [bundle.attention]
            else:
                raise ValueError(
                    "No attention maps captured for attention_rollout scorer."
                )

        rollout = compute_attention_rollout(
            attentions,
            average_heads=config.attention_rollout.average_heads,
            include_residual=config.attention_rollout.include_residual,
            discard_ratio=config.attention_rollout.discard_ratio,
        )
        cls_to_patches = rollout[0, 1:]
        expected = bundle.grid_size[0] * bundle.grid_size[1]
        if cls_to_patches.shape[0] != expected:
            cls_to_patches = cls_to_patches[:expected]
        return cls_to_patches.reshape(bundle.grid_size).astype(np.float32)


def compute_attention_rollout(
    attentions: list[np.ndarray],
    average_heads: bool = True,
    include_residual: bool = True,
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """
    Compute attention rollout from per-layer (tokens, tokens) attention matrices.
    """
    result = None
    num_tokens = attentions[0].shape[-1]

    for attn in attentions:
        a = attn.astype(np.float64)
        if a.ndim == 3:
            a = a.mean(axis=0) if average_heads else a[0]
        if include_residual:
            a = a + np.eye(a.shape[0], dtype=np.float64)
        a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)

        if discard_ratio > 0:
            flat = a.reshape(-1)
            threshold = np.percentile(flat, 100 * (1 - discard_ratio))
            a = np.where(a < threshold, 0, a)
            a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)

        if result is None:
            result = a
        else:
            if result.shape[0] != a.shape[0]:
                min_t = min(result.shape[0], a.shape[0])
                result = result[:min_t, :min_t]
                a = a[:min_t, :min_t]
            result = result @ a

    if result is None:
        result = np.eye(num_tokens, dtype=np.float64)
    return result.astype(np.float32)


SCORER_REGISTRY: dict[str, BaseScorer] = {
    ClsPatchCosineScorer.name: ClsPatchCosineScorer(),
    PatchL2Scorer.name: PatchL2Scorer(),
    SobelFeatureScorer.name: SobelFeatureScorer(),
    SobelImageScorer.name: SobelImageScorer(),
    AttentionRolloutScorer.name: AttentionRolloutScorer(),
}


def get_scorer(name: str) -> BaseScorer:
    if name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer: {name!r}. Available: {list(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[name]


def score_bundle(
    bundle: FeatureBundle,
    scorer_name: str,
    config: AnalysisConfig,
) -> np.ndarray:
    return get_scorer(scorer_name).score(bundle, config)
