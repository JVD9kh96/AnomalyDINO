from __future__ import annotations

import os

import numpy as np

from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors import build_detector
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape

EPS = 1e-8


def _per_image_zscore(scores: np.ndarray) -> np.ndarray:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return ((scores - mean) / (std + EPS)).astype(np.float32)


class EnsembleDetector(BaseAnomalyDetector):
    """
    Weighted ensemble of sub-detectors with per-image z-score normalization.

    patch_scores = sum(w_i * zscore(sub_detector_i scores))
  """

    def __init__(
        self,
        sub_detectors: list[BaseAnomalyDetector],
        weights: list[float] | None = None,
        device: str = "cuda:0",
    ):
        if not sub_detectors:
            raise ValueError("EnsembleDetector requires at least one sub-detector.")
        self.sub_detectors = sub_detectors
        if weights is None:
            weights = [1.0 / len(sub_detectors)] * len(sub_detectors)
        if len(weights) != len(sub_detectors):
            raise ValueError("weights length must match sub_detectors length.")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.device = device

        self._patch_size = 14
        self._resolution = 448

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        for det in self.sub_detectors:
            det.fit(reference_samples)
        if self.sub_detectors:
            self._patch_size = getattr(
                self.sub_detectors[0], "_patch_size", self._patch_size
            )
            self._resolution = getattr(
                self.sub_detectors[0], "resolution", self._resolution
            )

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if not self.sub_detectors:
            raise RuntimeError("Ensemble has no sub-detectors.")

        combined: np.ndarray | None = None
        grid_size = None
        processed_shape = None
        patch_size = self._patch_size

        for weight, det in zip(self.weights, self.sub_detectors):
            out = det.predict(sample)
            z = _per_image_zscore(out.patch_scores)
            if combined is None:
                combined = weight * z
                grid_size = out.grid_size
                processed_shape = out.processed_shape
                patch_size = out.patch_size
            else:
                combined = combined + weight * z

        assert combined is not None and grid_size is not None and processed_shape is not None

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=combined.astype(np.float32),
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=patch_size,
            patch_valid_mask=None,
            patch_class_scores=None,
        )


def build_ensemble_detector(config: dict, seed: int = 42) -> EnsembleDetector:
    sub_cfgs = config.get("sub_detectors", [])
    if not sub_cfgs:
        raise ValueError("ensemble detector requires sub_detectors list in config.")

    sub_detectors = [build_detector(sub_cfg, seed=seed) for sub_cfg in sub_cfgs]
    weights = config.get("weights")
    return EnsembleDetector(
        sub_detectors=sub_detectors,
        weights=weights,
        device=config.get("device", "cuda:0"),
    )
