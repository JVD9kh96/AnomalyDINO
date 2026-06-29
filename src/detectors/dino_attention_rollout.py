from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.attention_features import (
    capture_dino_attentions,
    compute_attention_rollout,
    rollout_to_patch_scores,
)
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.sobel_features import (
    CalibrationStats,
    apply_calibration,
    compute_calibration_stats,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


class DINOv2AttentionRolloutDetector(BaseAnomalyDetector):
    """
    DINOv2 CLS-to-patch attention rollout anomaly detector.

    Zero-shot when fit receives no references (shots=0).
    Few-shot calibration when reference images are provided (shots>0).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        average_heads: bool = True,
        include_residual: bool = True,
        discard_ratio: float = 0.0,
        last_n_layers: int | None = None,
        head_reduction: str | None = None,
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_attention_rollout requires a DINOv2 model name, got {model_name!r}."
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.average_heads = average_heads
        self.include_residual = include_residual
        self.discard_ratio = discard_ratio
        self.last_n_layers = last_n_layers
        self.head_reduction = head_reduction

        self._model = None
        self._patch_size = 14
        self._calib_stats: CalibrationStats | None = None

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

    def _extract_raw_scores(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        self._ensure_model()

        tensor, grid_size = self._model.prepare_image(image)
        attentions = capture_dino_attentions(self._model, tensor)
        rollout = compute_attention_rollout(
            attentions,
            average_heads=self.average_heads,
            include_residual=self.include_residual,
            discard_ratio=self.discard_ratio,
            last_n_layers=self.last_n_layers,
            head_reduction=self.head_reduction,
        )
        scores = rollout_to_patch_scores(rollout, grid_size)
        return scores, grid_size

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if not reference_samples:
            self._calib_stats = None
            return

        self._ensure_model()
        ref_scores: list[np.ndarray] = []

        for sample in tqdm(
            reference_samples, desc="Calibrating attention rollout detector", leave=False
        ):
            scores, _ = self._extract_raw_scores(sample.image)
            ref_scores.append(scores.ravel())

        if not ref_scores:
            self._calib_stats = None
            return

        all_scores = np.concatenate(ref_scores)
        self._calib_stats = compute_calibration_stats(all_scores)

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        native_shape = sample.image.shape[:2]
        raw_scores, grid_size = self._extract_raw_scores(sample.image)
        patch_scores = apply_calibration(raw_scores, self._calib_stats)

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
