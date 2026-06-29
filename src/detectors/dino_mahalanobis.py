from __future__ import annotations

import os

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.dino_features import (
    extract_patch_tokens,
    patch_tokens_to_grid,
    resolve_layer_indices,
    spatial_neighbor_aggregate,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape

RIDGE = 1e-5


class DINOv2MahalanobisDetector(BaseAnomalyDetector):
    """
    PaDiM-style per-position Mahalanobis distance on DINOv2 patch features.

    Uses diagonal covariance per grid position with optional PCA for stability.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        layers: int | str | list[int] | None = "last",
        pca_components: int | None = 50,
        prototype_reference_sampling: str = "defect_free",
        neighbor_aggregate: bool = False,
        pca_random_state: int = 42,
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_mahalanobis requires a DINOv2 model name, got {model_name!r}."
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.layers = layers
        self.pca_components = pca_components
        self.prototype_reference_sampling = prototype_reference_sampling
        self.neighbor_aggregate = neighbor_aggregate
        self.pca_random_state = pca_random_state

        self._model = None
        self._patch_size = 14
        self._pca: PCA | None = None
        self._means: np.ndarray | None = None
        self._variances: np.ndarray | None = None
        self._grid_size: tuple[int, int] | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device[-1]))
            cuda = "cuda" in self.device
            self._model = get_model(
                self.model_name,
                "cuda" if cuda else "cpu",
                self.resolution,
            )
            self._patch_size = getattr(self._model.model, "patch_size", 14)

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def _extract_feature_grid(self, image: np.ndarray) -> np.ndarray:
        self._ensure_model()
        tensor, grid_size = self._model.prepare_image(image)
        features, _ = extract_patch_tokens(self._model, tensor, self.layers)
        grid = patch_tokens_to_grid(features, grid_size)
        if self.neighbor_aggregate:
            grid = spatial_neighbor_aggregate(grid)
        return grid.astype(np.float32)

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if not reference_samples:
            raise ValueError("dino_mahalanobis requires reference samples (shots > 0).")

        self._ensure_model()
        stacks: list[np.ndarray] = []

        for sample in tqdm(
            reference_samples, desc="Fitting Mahalanobis detector", leave=False
        ):
            stacks.append(self._extract_feature_grid(sample.image))

        stack = np.stack(stacks, axis=0)
        n_ref, gh, gw, dim = stack.shape
        self._grid_size = (gh, gw)

        flat = stack.reshape(-1, dim).astype(np.float64)
        if self.pca_components is not None and self.pca_components < dim:
            self._pca = PCA(
                n_components=self.pca_components,
                svd_solver="randomized",
                random_state=self.pca_random_state,
            )
            flat_reduced = self._pca.fit_transform(flat)
            stack = flat_reduced.reshape(n_ref, gh, gw, self.pca_components)
            dim = self.pca_components
        else:
            self._pca = None

        means = np.zeros((gh, gw, dim), dtype=np.float32)
        variances = np.zeros((gh, gw, dim), dtype=np.float32)

        for i in range(gh):
            for j in range(gw):
                vecs = stack[:, i, j, :].astype(np.float64)
                means[i, j] = vecs.mean(axis=0)
                if n_ref > 1:
                    variances[i, j] = vecs.var(axis=0, ddof=1) + RIDGE
                else:
                    variances[i, j] = RIDGE

        self._means = means
        self._variances = variances

    def _mahalanobis_scores(self, features_grid: np.ndarray) -> np.ndarray:
        assert self._means is not None and self._variances is not None

        gh, gw, dim = features_grid.shape
        scores = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                diff = features_grid[i, j].astype(np.float64) - self._means[i, j]
                var = self._variances[i, j]
                scores[i, j] = float(np.sqrt(np.sum((diff * diff) / var)))
        return scores

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if self._means is None or self._variances is None:
            raise RuntimeError("Detector must be fit before predict.")

        native_shape = sample.image.shape[:2]
        features_grid = self._extract_feature_grid(sample.image)

        if self._pca is not None:
            gh, gw, dim = features_grid.shape
            flat = features_grid.reshape(-1, dim)
            flat_reduced = self._pca.transform(flat)
            features_grid = flat_reduced.reshape(gh, gw, self._pca.n_components_)

        patch_scores = self._mahalanobis_scores(features_grid)
        grid_size = features_grid.shape[:2]

        processed_shape, _ = compute_processed_shape(
            native_shape, self.resolution, self._patch_size
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=None,
            patch_class_scores=None,
        )
