from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.sobel_features import (
    CalibrationStats,
    ScoreModeParams,
    apply_score_mode,
    compute_calibration_stats,
    feature_sobel_norm,
    norms_to_numpy,
    tokens_to_feature_map,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


class DINOv2SobelDetector(BaseAnomalyDetector):
    """
    DINOv2 feature-space Sobel anomaly detector.

    Zero-shot when fit receives no references (shots=0).
    Few-shot calibration when reference images are provided (shots>0).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        norm_reduction: str = "l2",
        score_mode: str = "raw",
        zscore_k: float = 2.0,
        iqr_k: float = 1.5,
        percentile: float = 95.0,
        masking: bool = False,
        pca_random_state: int = 42,
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_sobel requires a DINOv2 model name, got {model_name!r}."
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.norm_reduction = norm_reduction
        self.score_params = ScoreModeParams(
            score_mode=score_mode,
            zscore_k=zscore_k,
            iqr_k=iqr_k,
            percentile=percentile,
        )
        self.masking = masking
        self.pca_random_state = pca_random_state

        self._model = None
        self._patch_size = 14
        self._calib_stats: CalibrationStats | None = None
        self._torch_device: torch.device | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device[-1]))
            cuda = "cuda" in self.device
            self._torch_device = torch.device(
                "cuda" if cuda and torch.cuda.is_available() else "cpu"
            )
            self._model = get_model(
                self.model_name,
                "cuda" if cuda else "cpu",
                self.resolution,
            )
            self._patch_size = getattr(self._model.model, "patch_size", 14)

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def _extract_norms(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray | None]:
        self._ensure_model()
        assert self._torch_device is not None

        with torch.inference_mode():
            tensor, grid_size = self._model.prepare_image(image)
            features = self._model.extract_features(tensor)

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

            feat_map = tokens_to_feature_map(features, grid_size).to(self._torch_device)
            norms = feature_sobel_norm(feat_map, self.norm_reduction)
            norm_np = norms_to_numpy(norms)

            if self.masking:
                valid_grid = patch_valid.reshape(grid_size)
                norm_np = norm_np.copy()
                norm_np[~valid_grid] = 0.0
            else:
                valid_grid = None

        return norm_np, grid_size, valid_grid

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if not reference_samples:
            self._calib_stats = None
            return

        self._ensure_model()
        ref_norms: list[np.ndarray] = []

        for sample in tqdm(reference_samples, desc="Calibrating Sobel detector", leave=False):
            norms, _, valid_grid = self._extract_norms(sample.image)
            if valid_grid is not None:
                ref_norms.append(norms[valid_grid])
            else:
                ref_norms.append(norms.ravel())

        if not ref_norms:
            self._calib_stats = None
            return

        all_norms = np.concatenate([n.ravel() for n in ref_norms])
        self._calib_stats = compute_calibration_stats(all_norms)

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        native_shape = sample.image.shape[:2]
        norms, grid_size, valid_grid = self._extract_norms(sample.image)
        patch_scores = apply_score_mode(norms, self.score_params, self._calib_stats)

        processed_shape, _ = compute_processed_shape(
            native_shape, self.resolution, self._patch_size
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=valid_grid,
            patch_class_scores=None,
        )
