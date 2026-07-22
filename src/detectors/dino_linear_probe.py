from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape
from src.training.trainer import PatchClassifierTrainer, TrainConfig


class DINOv2LinearProbeDetector(BaseAnomalyDetector):
    """
    Frozen DINOv2 + learnable linear patch classifier.

    fit() trains the linear head(s) (or loads a checkpoint). predict() returns
    binary anomaly probabilities and, in binary_multiclass mode, gated
    per-class scores conditioned on the binary F1-optimal threshold.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        classification_mode: str = "binary",
        binary_loss: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_mc: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 50,
        num_workers: int = 0,
        num_classes: int = 4,
        gt_overlap_threshold: float = 0.5,
        threshold_num_steps: int = 101,
        checkpoint_path: str | None = None,
        save_plots: bool = True,
        log_format: str = "both",
        seed: int = 42,
        **kwargs: Any,
    ):
        if not model_name.startswith("dinov2"):
            raise ValueError(
                f"dino_linear_probe requires a DINOv2 model name, got {model_name!r}."
            )
        if classification_mode not in ("binary", "binary_multiclass"):
            raise ValueError(
                "classification_mode must be 'binary' or 'binary_multiclass', "
                f"got {classification_mode!r}."
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.classification_mode = classification_mode
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self._patch_size = 14

        self.train_config = TrainConfig(
            classification_mode=classification_mode,
            binary_loss=binary_loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            lambda_mc=lambda_mc,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            num_classes=num_classes,
            resolution=resolution,
            gt_overlap_threshold=gt_overlap_threshold,
            threshold_num_steps=threshold_num_steps,
            save_plots=save_plots,
            log_format=log_format,
            seed=seed,
            device=device,
            model_name=model_name,
        )
        self.trainer = PatchClassifierTrainer(self.train_config)
        self.optimal_threshold: float | None = None

    @property
    def supports_class_prediction(self) -> bool:
        return self.classification_mode == "binary_multiclass"

    def fit(
        self,
        reference_samples: Sequence[SeverstalSample],
        val_samples: Sequence[SeverstalSample] | None = None,
        output_dir: str | Path | None = None,
        train_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Train linear probe on ``reference_samples`` (training set for this fold).

        Optional ``val_samples`` are used for epoch metrics and F1 threshold search.
        If ``checkpoint_path`` is set, loads weights instead of training.
        """
        if train_cfg:
            if "save_plots" in train_cfg:
                self.train_config.save_plots = bool(train_cfg["save_plots"])
            if "log_format" in train_cfg:
                self.train_config.log_format = str(train_cfg["log_format"])

        if self.checkpoint_path:
            self.trainer.load_checkpoint(self.checkpoint_path)
            self.optimal_threshold = self.trainer.optimal_threshold
            if self.trainer._backbone is not None:
                self._patch_size = getattr(
                    self.trainer._backbone.model, "patch_size", self._patch_size
                )
            return

        if not reference_samples:
            raise ValueError(
                "dino_linear_probe.fit() requires non-empty training samples "
                "(set shots to null for full train fold, -1 for all eligible, "
                "or N for k-shot)."
            )

        result = self.trainer.fit(
            train_samples=reference_samples,
            val_samples=val_samples,
            output_dir=output_dir,
        )
        self.optimal_threshold = float(result["optimal_threshold"])
        if self.trainer._backbone is not None:
            self._patch_size = getattr(
                self.trainer._backbone.model, "patch_size", self._patch_size
            )

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        if self.trainer.classifier is None:
            raise RuntimeError(
                "DINOv2LinearProbeDetector.predict() called before fit()/checkpoint load."
            )

        native_shape = sample.image.shape[:2]
        processed_shape, _ = compute_processed_shape(
            native_shape,
            smaller_edge_size=self.resolution,
            patch_size=self._patch_size,
        )
        patch_scores, patch_class_scores, grid_size = self.trainer.predict_image(
            sample.image
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=None,
            patch_class_scores=patch_class_scores,
        )
