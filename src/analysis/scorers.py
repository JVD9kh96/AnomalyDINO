from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from src.analysis.config import AnalysisConfig
from src.analysis.types import FeatureBundle
from src.detectors.attention_features import compute_attention_rollout, rollout_to_patch_scores
from src.detectors.cls_patch_features import compute_cls_patch_cosine
from src.detectors.sobel_features import feature_sobel_norm, tokens_to_feature_map


class BaseScorer(ABC):
    name: str

    @abstractmethod
    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        """Return patch score grid (grid_h, grid_w)."""


class ClsPatchCosineScorer(BaseScorer):
    name = "cls_patch_cosine"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        return compute_cls_patch_cosine(
            bundle.cls_token,
            bundle.patch_tokens,
            bundle.grid_size,
        )


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
        out = norms.squeeze(0).detach().cpu().numpy().astype(np.float32)
        del feat_map, norms
        return out


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
            last_n_layers=config.attention_rollout.last_n_layers,
            head_reduction=config.attention_rollout.head_reduction,
        )
        return rollout_to_patch_scores(rollout, bundle.grid_size)


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
