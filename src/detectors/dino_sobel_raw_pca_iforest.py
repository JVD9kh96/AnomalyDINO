from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.backbones import get_model
from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


EPS = 1e-8


class DINOv2SobelRawPCAIForestDetector(BaseAnomalyDetector):
    """
    IsolationForest detector over concatenated PCA-reduced DINOv2 patch tokens.

    Branches:
      1. DINOv2 patch tokens extracted from an image-space Sobel edge image.
      2. DINOv2 patch tokens extracted from the raw RGB image.

    Each branch gets its own PCA fitted on k-shot reference patches. The reduced
    branches are concatenated and used as patch-level IsolationForest features.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        pca_components: int = 3,
        pca_whiten: bool = False,
        standardize: bool = True,
        masking: bool = False,
        mask_ref_images: bool = False,
        sobel_kernel_size: int = 3,
        iforest_n_estimators: int = 200,
        iforest_max_samples: str | int | float = "auto",
        iforest_contamination: str | float = "auto",
        iforest_max_features: float = 1.0,
        iforest_bootstrap: bool = False,
        iforest_n_jobs: int | None = -1,
        random_state: int = 42,
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_sobel_raw_pca_iforest requires a DINOv2 model name, "
                f"got {model_name!r}."
            )
        if pca_components < 1:
            raise ValueError("pca_components must be >= 1.")
        if sobel_kernel_size not in (1, 3, 5, 7):
            raise ValueError("sobel_kernel_size must be one of 1, 3, 5, or 7.")

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.standardize = standardize
        self.masking = masking
        self.mask_ref_images = mask_ref_images
        self.sobel_kernel_size = sobel_kernel_size
        self.iforest_n_estimators = iforest_n_estimators
        self.iforest_max_samples = iforest_max_samples
        self.iforest_contamination = iforest_contamination
        self.iforest_max_features = iforest_max_features
        self.iforest_bootstrap = iforest_bootstrap
        self.iforest_n_jobs = iforest_n_jobs
        self.random_state = random_state

        self._model = None
        self._patch_size = 14
        self._raw_pca: PCA | None = None
        self._sobel_pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._iforest: IsolationForest | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            if "cuda" in self.device:
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

    def _sobel_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=self.sobel_kernel_size,
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=self.sobel_kernel_size,
        )
        magnitude = np.sqrt(gx * gx + gy * gy)
        scale = float(magnitude.max())
        if scale > EPS:
            magnitude = magnitude / scale
        sobel_u8 = np.clip(magnitude * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(sobel_u8[:, :, None], 3, axis=2)

    def _extract_branch_tokens(
        self,
        image: np.ndarray,
        *,
        mask_background: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
        self._ensure_model()
        assert self._model is not None

        with torch.inference_mode():
            raw_tensor, grid_size = self._model.prepare_image(image)
            raw_features = self._model.extract_features(raw_tensor).astype(np.float32)

            sobel_tensor, sobel_grid_size = self._model.prepare_image(
                self._sobel_image(image)
            )
            if sobel_grid_size != grid_size:
                raise RuntimeError(
                    f"Raw grid {grid_size} and Sobel grid {sobel_grid_size} differ."
                )
            sobel_features = self._model.extract_features(sobel_tensor).astype(np.float32)

            if mask_background:
                patch_valid = self._model.compute_background_mask(
                    raw_features,
                    grid_size,
                    threshold=10,
                    masking_type=True,
                    random_state=self.random_state,
                )
            else:
                patch_valid = np.ones(raw_features.shape[0], dtype=bool)

        return raw_features, sobel_features, grid_size, patch_valid.astype(bool)

    def _fit_pca(self, values: np.ndarray, *, salt: int) -> PCA:
        n_components = min(self.pca_components, values.shape[0], values.shape[1])
        if n_components < self.pca_components:
            print(
                "Warning: reducing pca_components from "
                f"{self.pca_components} to {n_components} because only "
                f"{values.shape[0]} samples and {values.shape[1]} features are available."
            )
        solver = "randomized" if n_components < min(values.shape) else "full"
        pca = PCA(
            n_components=n_components,
            whiten=self.pca_whiten,
            svd_solver=solver,
            random_state=self.random_state + salt,
        )
        pca.fit(values)
        return pca

    def _make_patch_features(
        self,
        raw_features: np.ndarray,
        sobel_features: np.ndarray,
    ) -> np.ndarray:
        if self._raw_pca is None or self._sobel_pca is None:
            raise RuntimeError("PCA models are not fit.")

        raw_reduced = self._raw_pca.transform(raw_features.astype(np.float32))
        sobel_reduced = self._sobel_pca.transform(sobel_features.astype(np.float32))
        features = np.concatenate([sobel_reduced, raw_reduced], axis=1).astype(np.float32)

        if self._scaler is not None:
            features = self._scaler.transform(features).astype(np.float32)
        return features

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        if not reference_samples:
            raise ValueError(
                "dino_sobel_raw_pca_iforest requires k-shot references; "
                "set detector.shots > 0."
            )

        self._raw_pca = None
        self._sobel_pca = None
        self._scaler = None
        self._iforest = None

        raw_refs: list[np.ndarray] = []
        sobel_refs: list[np.ndarray] = []

        for sample in tqdm(
            reference_samples,
            desc="Building Sobel/raw PCA IsolationForest bank",
            leave=False,
        ):
            raw, sobel, _, patch_valid = self._extract_branch_tokens(
                sample.image,
                mask_background=self.mask_ref_images and self.masking,
            )
            raw_refs.append(raw[patch_valid])
            sobel_refs.append(sobel[patch_valid])

        raw_matrix = np.concatenate(raw_refs, axis=0).astype(np.float32)
        sobel_matrix = np.concatenate(sobel_refs, axis=0).astype(np.float32)
        if raw_matrix.size == 0 or sobel_matrix.size == 0:
            raise ValueError("No reference patch features extracted.")

        self._raw_pca = self._fit_pca(raw_matrix, salt=0)
        self._sobel_pca = self._fit_pca(sobel_matrix, salt=17)

        train_features = self._make_patch_features(raw_matrix, sobel_matrix)
        if self.standardize:
            self._scaler = StandardScaler()
            train_features = self._scaler.fit_transform(train_features).astype(np.float32)

        self._iforest = IsolationForest(
            n_estimators=self.iforest_n_estimators,
            max_samples=self.iforest_max_samples,
            contamination=self.iforest_contamination,
            max_features=self.iforest_max_features,
            bootstrap=self.iforest_bootstrap,
            n_jobs=self.iforest_n_jobs,
            random_state=self.random_state,
        )
        self._iforest.fit(train_features)

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if self._iforest is None:
            raise RuntimeError("Detector must be fit before predict.")

        native_shape = sample.image.shape[:2]
        raw, sobel, grid_size, patch_valid = self._extract_branch_tokens(
            sample.image,
            mask_background=self.masking,
        )
        features = self._make_patch_features(raw, sobel)

        # sklearn score_samples is higher for normal patches; invert polarity.
        scores = -self._iforest.score_samples(features).astype(np.float32)
        output_scores = np.zeros(raw.shape[0], dtype=np.float32)
        output_scores[patch_valid] = scores[patch_valid]
        patch_scores = output_scores.reshape(grid_size)

        processed_shape, _ = compute_processed_shape(
            native_shape,
            self.resolution,
            self._patch_size,
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
