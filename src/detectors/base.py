from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.severstal.dataset import SeverstalSample


@dataclass
class DetectorOutput:
    image_id: str
    patch_scores: np.ndarray
    grid_size: tuple[int, int]
    processed_shape: tuple[int, int]
    patch_size: int
    patch_valid_mask: np.ndarray | None = None
    patch_class_scores: np.ndarray | None = None


class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        """Build model state (e.g. memory bank) from normal reference images."""

    @abstractmethod
    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        """Run anomaly detection on a single image."""

    @property
    @abstractmethod
    def supports_class_prediction(self) -> bool:
        """Whether per-class patch scores are produced."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
