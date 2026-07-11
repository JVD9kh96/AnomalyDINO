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


class DINOv2KnnRolloutDetector(BaseAnomalyDetector):
    """
    Few-shot detector combining AnomalyDINO kNN distances with
    reference-anchored attention rollout deviation.

    Both branches depend on reference images (shots > 0), so changing shots
    changes patch rankings unlike global z-score calibration alone.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        knn_metric: str = "L2_normalized",
        k_neighbors: int = 1,
        faiss_on_cpu: bool = False,
        masking: bool = False,
        mask_ref_images: bool = False,
        rotation: bool = False,
        pca_random_state: int = 42,
        coreset_ratio: float | None = None,
        neighbor_aggregate: bool = False,
        rollout_cfg: RolloutConfig | dict | None = None,
        fusion_mode: str = "weighted_sum",
        knn_weight: float = 0.5,
        rollout_weight: float = 0.5,
    ):
        assert knn_metric in ("L2", "L2_normalized")
        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.knn_metric = knn_metric
        self.k_neighbors = k_neighbors
        self.faiss_on_cpu = faiss_on_cpu
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
        self.knn_weight = knn_weight
        self.rollout_weight = rollout_weight

        self._model = None
        self._knn_index = None
        self._patch_size = 14
        self._rollout_mean: np.ndarray | None = None
        self._rollout_std: np.ndarray | None = None
        self._grid_size: tuple[int, int] | None = None

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

    def _knn_distances(
        self, features: np.ndarray, grid_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        import faiss

        if self._knn_index is None:
            raise RuntimeError("kNN index must be built in fit() before predict().")

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

        features_masked = features[patch_valid].astype(np.float32)

        if self.knn_metric == "L2":
            distances, _ = self._knn_index.search(features_masked, k=self.k_neighbors)
            if self.k_neighbors > 1:
                distances = distances.mean(axis=1)
            distances = np.sqrt(distances)
        else:
            faiss.normalize_L2(features_masked)
            distances, _ = self._knn_index.search(features_masked, k=self.k_neighbors)
            if self.k_neighbors > 1:
                distances = distances.mean(axis=1)
            distances = distances / 2

        output_distances = np.zeros(features.shape[0], dtype=np.float32)
        output_distances[patch_valid] = distances.squeeze()
        return output_distances.reshape(grid_size), patch_valid.reshape(grid_size)

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        import faiss

        if not reference_samples:
            raise ValueError(
                "dino_knn_rollout requires reference samples (shots > 0)."
            )

        self._ensure_model()
        features_ref: list[np.ndarray] = []
        rollout_maps: list[np.ndarray] = []

        with torch.inference_mode():
            for sample in tqdm(
                reference_samples, desc="Building kNN+rollout refs", leave=False
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

        features_ref_arr = np.concatenate(features_ref, axis=0).astype("float32")

        if self.coreset_ratio is not None and 0 < self.coreset_ratio < 1.0:
            features_ref_arr = greedy_coreset(
                features_ref_arr, ratio=self.coreset_ratio, seed=self.pca_random_state
            )

        if self.faiss_on_cpu:
            self._knn_index = faiss.IndexFlatL2(features_ref_arr.shape[1])
        else:
            res = faiss.StandardGpuResources()
            self._knn_index = faiss.GpuIndexFlatL2(res, features_ref_arr.shape[1])

        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref_arr)
        self._knn_index.add(features_ref_arr)

        self._rollout_mean, self._rollout_std = fit_rollout_stats(rollout_maps)
        self._grid_size = rollout_maps[0].shape

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if self._knn_index is None or self._rollout_mean is None or self._rollout_std is None:
            raise RuntimeError("Detector must be fit before predict.")

        self._ensure_model()
        native_shape = sample.image.shape[:2]

        with torch.inference_mode():
            features, rollout_map, grid_size = extract_knn_features_and_rollout(
                self._model, sample.image, self.rollout_cfg
            )
            features = self._prepare_features(features, grid_size)
            knn_map, patch_valid = self._knn_distances(features, grid_size)
            rollout_dev = compute_rollout_deviation(
                rollout_map, self._rollout_mean, self._rollout_std
            )
            patch_scores = fuse_branch_scores(
                knn_map,
                rollout_dev,
                mode=self.fusion_mode,
                knn_weight=self.knn_weight,
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
