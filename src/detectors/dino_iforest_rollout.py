from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.coreset import greedy_coreset
from src.detectors.dino_features import (
    RolloutConfig,
    compute_rollout_deviation,
    extract_knn_features_and_rollout,
    fit_rollout_stats,
    fuse_branch_scores,
    patch_tokens_to_grid,
    spatial_neighbor_aggregate,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape
from src.utils import augment_image


class DINOv2IForestRolloutDetector(BaseAnomalyDetector):
    """
    Few-shot detector combining IsolationForest patch anomaly scores with
    reference-anchored attention rollout deviation.

    IsolationForest branch uses per-patch anomaly score:
        if_score = -IsolationForest.score_samples(patch_features)
    (higher = more anomalous).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        masking: bool = False,
        mask_ref_images: bool = False,
        rotation: bool = False,
        pca_random_state: int = 42,
        coreset_ratio: float | None = None,
        neighbor_aggregate: bool = False,
        rollout_cfg: RolloutConfig | dict | None = None,
        fusion_mode: str = "weighted_sum",
        iforest_weight: float = 0.5,
        rollout_weight: float = 0.5,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        contamination: str | float = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int | None = -1,
    ):
        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.masking = masking
        self.mask_ref_images = mask_ref_images
        self.rotation = rotation
        self.pca_random_state = pca_random_state
        self.coreset_ratio = coreset_ratio
        self.neighbor_aggregate = neighbor_aggregate
        self.rollout_cfg = (
            rollout_cfg
            if isinstance(rollout_cfg, RolloutConfig)
            else RolloutConfig.from_dict(rollout_cfg or {})
        )
        self.fusion_mode = fusion_mode
        self.iforest_weight = iforest_weight
        self.rollout_weight = rollout_weight

        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = float(max_features)
        self.bootstrap = bool(bootstrap)
        self.n_jobs = n_jobs

        self._model = None
        self._patch_size = 14
        self._rollout_mean: np.ndarray | None = None
        self._rollout_std: np.ndarray | None = None
        self._iforest = None

    def _ensure_model(self) -> None:
        if self._model is None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device[-1]))
            self._model = get_model(
                self.model_name,
                "cuda" if "cuda" in self.device else "cpu",
                self.resolution,
            )
            self._patch_size = getattr(self._model.model, "patch_size", 14)

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def _prepare_features(
        self, features: np.ndarray, grid_size: tuple[int, int]
    ) -> np.ndarray:
        if not self.neighbor_aggregate:
            return features
        grid = patch_tokens_to_grid(features, grid_size)
        aggregated = spatial_neighbor_aggregate(grid)
        return aggregated.reshape(-1, aggregated.shape[-1])

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if not reference_samples:
            raise ValueError(
                "dino_iforest_rollout requires reference samples (shots > 0)."
            )

        # Local import so non-ML envs can still import the package.
        from sklearn.ensemble import IsolationForest

        self._ensure_model()

        features_ref: list[np.ndarray] = []
        rollout_maps: list[np.ndarray] = []

        with torch.inference_mode():
            for sample in tqdm(
                reference_samples, desc="Building IForest+rollout refs", leave=False
            ):
                images = (
                    augment_image(sample.image) if self.rotation else [sample.image]
                )
                for image in images:
                    features, rollout_map, grid_size = extract_knn_features_and_rollout(
                        self._model, image, self.rollout_cfg
                    )
                    features = self._prepare_features(features, grid_size)
                    mask_ref = self._model.compute_background_mask(
                        features,
                        grid_size,
                        threshold=10,
                        masking_type=(self.mask_ref_images and self.masking),
                        random_state=self.pca_random_state,
                    )
                    features_ref.append(features[mask_ref])
                    rollout_maps.append(rollout_map)

        if not features_ref:
            raise ValueError("No reference patch features extracted. Check reference images.")

        features_ref_arr = np.concatenate(features_ref, axis=0).astype(np.float32)

        if self.coreset_ratio is not None and 0 < self.coreset_ratio < 1.0:
            features_ref_arr = greedy_coreset(
                features_ref_arr, ratio=self.coreset_ratio, seed=self.pca_random_state
            )

        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.pca_random_state,
        )
        self._iforest.fit(features_ref_arr)

        self._rollout_mean, self._rollout_std = fit_rollout_stats(rollout_maps)

    def _iforest_scores(
        self, features: np.ndarray, grid_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._iforest is None:
            raise RuntimeError("IsolationForest must be fit in fit() before predict().")

        if self.masking:
            patch_valid = self._model.compute_background_mask(
                features,
                grid_size,
                threshold=10,
                masking_type=True,
                random_state=self.pca_random_state,
            )
        else:
            patch_valid = np.ones(features.shape[0], dtype=bool)

        masked = features[patch_valid].astype(np.float32)
        # score_samples: higher = more normal => negate for anomaly score
        scores = -self._iforest.score_samples(masked)

        out = np.zeros(features.shape[0], dtype=np.float32)
        out[patch_valid] = scores.astype(np.float32)
        return out.reshape(grid_size), patch_valid.reshape(grid_size)

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if self._iforest is None or self._rollout_mean is None or self._rollout_std is None:
            raise RuntimeError("Detector must be fit before predict.")

        self._ensure_model()
        native_shape = sample.image.shape[:2]

        with torch.inference_mode():
            features, rollout_map, grid_size = extract_knn_features_and_rollout(
                self._model, sample.image, self.rollout_cfg
            )
            features = self._prepare_features(features, grid_size)
            iforest_map, patch_valid = self._iforest_scores(features, grid_size)
            rollout_dev = compute_rollout_deviation(
                rollout_map, self._rollout_mean, self._rollout_std
            )
            patch_scores = fuse_branch_scores(
                iforest_map,
                rollout_dev,
                mode=self.fusion_mode,
                knn_weight=self.iforest_weight,
                rollout_weight=self.rollout_weight,
            )

        processed_shape, _ = compute_processed_shape(
            native_shape, self.resolution, self._patch_size
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=patch_valid,
            patch_class_scores=None,
        )

