from __future__ import annotations

import numpy as np

from src.segmenters.base import BaseSegmenter, SegmenterOutput, SegmenterPrompts


class SAM2Segmenter(BaseSegmenter):
    """SAM2 segmenter via Ultralytics, prompted with bounding boxes or points."""

    def __init__(
        self,
        model_name: str = "sam2.1_b.pt",
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import SAM

            self._model = SAM(self.model_name)

    def segment(
        self,
        image: np.ndarray,
        prompts: SegmenterPrompts,
    ) -> SegmenterOutput:
        self._ensure_model()
        native_shape = image.shape[:2]

        if not prompts.bboxes and not prompts.points:
            return SegmenterOutput(
                mask=np.zeros(native_shape, dtype=bool),
                masks_by_class=None,
            )

        kwargs = {}
        if prompts.bboxes:
            flat_bboxes = []
            for bbox in prompts.bboxes:
                flat_bboxes.extend(bbox)
            kwargs["bboxes"] = flat_bboxes
        elif prompts.points:
            flat_points = []
            labels = prompts.point_labels or [1] * len(prompts.points)
            for pt in prompts.points:
                flat_points.extend(pt)
            kwargs["points"] = flat_points
            kwargs["labels"] = labels

        results = self._model.predict(
            source=image,
            device=self.device,
            verbose=False,
            **kwargs,
        )

        combined = np.zeros(native_shape, dtype=bool)
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data
            if hasattr(masks_data, "cpu"):
                masks_np = masks_data.cpu().numpy()
            else:
                masks_np = np.asarray(masks_data)

            for mask in masks_np:
                mask_bool = mask.astype(bool)
                if mask_bool.shape != native_shape:
                    import cv2

                    mask_bool = cv2.resize(
                        mask_bool.astype(np.uint8),
                        (native_shape[1], native_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                combined |= mask_bool

        return SegmenterOutput(mask=combined, masks_by_class=None)
