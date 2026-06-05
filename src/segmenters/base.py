from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmenterPrompts:
    bboxes: list[list[float]] = field(default_factory=list)
    points: list[list[float]] = field(default_factory=list)
    point_labels: list[int] = field(default_factory=list)


@dataclass
class SegmenterOutput:
    mask: np.ndarray
    masks_by_class: dict[int, np.ndarray] | None = None


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        prompts: SegmenterPrompts,
    ) -> SegmenterOutput:
        """Generate segmentation mask from image and prompts."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
