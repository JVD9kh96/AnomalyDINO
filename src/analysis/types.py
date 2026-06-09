from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnalysisSample:
    image_id: str
    image: np.ndarray
    mask: np.ndarray
    meta: dict | None = None


@dataclass
class FeatureBundle:
    layer_index: int
    cls_token: np.ndarray
    patch_tokens: np.ndarray
    grid_size: tuple[int, int]
    processed_shape: tuple[int, int]
    patch_size: int
    attention: np.ndarray | None = None
    attentions_all_layers: list[np.ndarray] | None = None
    preprocessed_image: np.ndarray | None = None


@dataclass
class PatchResult:
    image_id: str
    scores: np.ndarray
    labels: np.ndarray
    coords: np.ndarray
    layer_index: int
    scorer_name: str
