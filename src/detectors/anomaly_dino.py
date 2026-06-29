from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.coreset import greedy_coreset
from src.detectors.dino_features import patch_tokens_to_grid, spatial_neighbor_aggregate
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape
from src.utils import augment_image


class AnomalyDINODetector(BaseAnomalyDetector):
    """AnomalyDINO few-shot patch-based detector for Severstal evaluation."""

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

        self._model = None
        self._knn_index = None
        self._patch_size = 14

    def _ensure_model(self) -> None:
        if self._model is None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device[-1]))
            self._model = get_model(
                self.model_name, "cuda" if "cuda" in self.device else "cpu", self.resolution
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
        import faiss

        self._ensure_model()
        features_ref = []

        with torch.inference_mode():
            for sample in tqdm(reference_samples, desc="Building memory bank", leave=False):
                images = augment_image(sample.image) if self.rotation else [sample.image]
                for image in images:
                    tensor, grid_size = self._model.prepare_image(image)
                    features = self._model.extract_features(tensor)
                    features = self._prepare_features(features, grid_size)
                    mask_ref = self._model.compute_background_mask(
                        features,
                        grid_size,
                        threshold=10,
                        masking_type=(self.mask_ref_images and self.masking),
                        random_state=self.pca_random_state,
                    )
                    features_ref.append(features[mask_ref])

        if not features_ref:
            raise ValueError("No reference patch features extracted. Check reference images.")

        features_ref = np.concatenate(features_ref, axis=0).astype("float32")

        if self.coreset_ratio is not None and 0 < self.coreset_ratio < 1.0:
            features_ref = greedy_coreset(
                features_ref, ratio=self.coreset_ratio, seed=self.pca_random_state
            )

        if self.faiss_on_cpu:
            self._knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            res = faiss.StandardGpuResources()
            self._knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])

        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
        self._knn_index.add(features_ref)

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        import faiss

        self._ensure_model()
        if self._knn_index is None:
            raise RuntimeError("Detector must be fit before predict.")

        native_shape = sample.image.shape[:2]
        with torch.inference_mode():
            tensor, grid_size = self._model.prepare_image(sample.image)
            features = self._model.extract_features(tensor)
            features = self._prepare_features(features, grid_size)

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

            features_masked = features[patch_valid]

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
            patch_scores = output_distances.reshape(grid_size)

        processed_shape, _ = compute_processed_shape(
            native_shape, self.resolution, self._patch_size
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=patch_valid.reshape(grid_size),
            patch_class_scores=None,
        )
