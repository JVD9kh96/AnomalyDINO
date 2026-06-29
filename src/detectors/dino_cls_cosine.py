from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.cls_patch_features import compute_cls_patch_cosine, resolve_layer_index
from src.detectors.dino_features import extract_layer_outputs
from src.detectors.sobel_features import (
    CalibrationStats,
    apply_calibration,
    compute_calibration_stats,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape

EPS = 1e-8


def prototype_anomaly_scores(
    ref_cls: np.ndarray,
    patch_tokens: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Higher = more anomalous: 1 - cosine similarity to reference CLS prototype."""
    cosine = compute_cls_patch_cosine(ref_cls, patch_tokens, grid_size)
    return (1.0 - cosine).astype(np.float32)


class DINOv2ClsPatchCosineDetector(BaseAnomalyDetector):
    """
    DINOv2 CLS-to-patch cosine similarity anomaly detector.

    scoring_mode:
      - per_image: cos(cls_test, patch_test) — zero-shot analysis parity
      - prototype: 1 - cos(ref_cls_prototype, patch_test) — few-shot, higher=anomaly
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        layer: int | str = "last",
        scoring_mode: str = "per_image",
        prototype_reference_sampling: str = "defect_free",
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_cls_cosine requires a DINOv2 model name, got {model_name!r}."
            )
        if scoring_mode not in ("per_image", "prototype"):
            raise ValueError(
                f"scoring_mode must be 'per_image' or 'prototype', got {scoring_mode!r}."
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.layer = layer
        self.scoring_mode = scoring_mode
        self.prototype_reference_sampling = prototype_reference_sampling

        self._model = None
        self._patch_size = 14
        self._num_layers: int | None = None
        self._layer_idx: int | None = None
        self._calib_stats: CalibrationStats | None = None
        self._ref_cls: np.ndarray | None = None

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
            self._num_layers = len(self._model.model.blocks)
            self._layer_idx = resolve_layer_index(self.layer, self._num_layers)

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def _extract_tokens(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        self._ensure_model()
        assert self._layer_idx is not None

        tensor, grid_size = self._model.prepare_image(image)
        layer_outputs, grid_size = extract_layer_outputs(
            self._model, tensor, self._layer_idx
        )
        patch_np, cls_np = layer_outputs[0]
        return patch_np, cls_np, grid_size

    def _extract_raw_scores(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        patch_np, cls_np, grid_size = self._extract_tokens(image)

        if self.scoring_mode == "prototype":
            if self._ref_cls is None:
                raise RuntimeError(
                    "prototype scoring_mode requires fit() with reference samples."
                )
            scores = prototype_anomaly_scores(self._ref_cls, patch_np, grid_size)
        else:
            scores = compute_cls_patch_cosine(cls_np, patch_np, grid_size)

        return scores, grid_size

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if self.scoring_mode == "prototype":
            if not reference_samples:
                raise ValueError(
                    "prototype scoring_mode requires reference samples (shots > 0)."
                )
            self._ensure_model()
            cls_tokens: list[np.ndarray] = []

            for sample in tqdm(
                reference_samples,
                desc="Fitting CLS prototype detector",
                leave=False,
            ):
                _, cls_np, _ = self._extract_tokens(sample.image)
                cls_tokens.append(cls_np)

            self._ref_cls = np.mean(cls_tokens, axis=0).astype(np.float32)

            ref_scores: list[np.ndarray] = []
            for sample in tqdm(
                reference_samples,
                desc="Calibrating prototype scores",
                leave=False,
            ):
                patch_np, _, grid_size = self._extract_tokens(sample.image)
                ref_scores.append(
                    prototype_anomaly_scores(
                        self._ref_cls, patch_np, grid_size
                    ).ravel()
                )

            all_scores = np.concatenate(ref_scores)
            self._calib_stats = compute_calibration_stats(all_scores)
            return

        self._ref_cls = None
        if not reference_samples:
            self._calib_stats = None
            return

        self._ensure_model()
        ref_scores: list[np.ndarray] = []

        for sample in tqdm(
            reference_samples, desc="Calibrating CLS cosine detector", leave=False
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
