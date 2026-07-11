from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.detectors.attention_features import (
    capture_dino_attentions,
    compute_attention_rollout,
    rollout_to_patch_scores,
)
from src.detectors.cls_patch_features import resolve_layer_index

RIDGE = 1e-5
EPS = 1e-8


@dataclass
class RolloutConfig:
    average_heads: bool = True
    include_residual: bool = True
    discard_ratio: float = 0.0
    last_n_layers: int | None = None
    head_reduction: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RolloutConfig:
        if not data:
            return cls()
        return cls(
            average_heads=data.get("average_heads", True),
            include_residual=data.get("include_residual", True),
            discard_ratio=data.get("discard_ratio", 0.0),
            last_n_layers=data.get("last_n_layers"),
            head_reduction=data.get("head_reduction"),
        )


def rollout_map_from_tensor(
    model_wrapper,
    image_tensor: torch.Tensor,
    grid_size: tuple[int, int],
    rollout_cfg: RolloutConfig,
) -> np.ndarray:
    """Compute CLS-to-patch attention rollout scores on a preprocessed tensor."""
    attentions = capture_dino_attentions(model_wrapper, image_tensor)
    rollout = compute_attention_rollout(
        attentions,
        average_heads=rollout_cfg.average_heads,
        include_residual=rollout_cfg.include_residual,
        discard_ratio=rollout_cfg.discard_ratio,
        last_n_layers=rollout_cfg.last_n_layers,
        head_reduction=rollout_cfg.head_reduction,
    )
    return rollout_to_patch_scores(rollout, grid_size)


def extract_knn_features_and_rollout(
    model_wrapper,
    image: np.ndarray,
    rollout_cfg: RolloutConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Extract last-layer patch features (kNN) and attention rollout map from one image.

    Returns:
        features: (N_patches, D) flat patch tokens for kNN
        rollout_map: (H, W) CLS-to-patch rollout scores
        grid_size: (grid_h, grid_w)
    """
    if isinstance(rollout_cfg, dict):
        rollout_cfg = RolloutConfig.from_dict(rollout_cfg)
    elif rollout_cfg is None:
        rollout_cfg = RolloutConfig()

    tensor, grid_size = model_wrapper.prepare_image(image)
    with torch.inference_mode():
        features = model_wrapper.extract_features(tensor)
    rollout_map = rollout_map_from_tensor(model_wrapper, tensor, grid_size, rollout_cfg)
    return features.astype(np.float32), rollout_map.astype(np.float32), grid_size


def fit_rollout_stats(rollout_maps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean and std of rollout maps from reference images."""
    if not rollout_maps:
        raise ValueError("Cannot fit rollout stats from empty rollout_maps.")
    stack = np.stack(rollout_maps, axis=0).astype(np.float64)
    mean = stack.mean(axis=0).astype(np.float32)
    if stack.shape[0] > 1:
        std = stack.std(axis=0, ddof=1).astype(np.float32) + RIDGE
    else:
        std = np.full_like(mean, RIDGE, dtype=np.float32)
    return mean, std


def compute_rollout_deviation(
    rollout_map: np.ndarray,
    rollout_mean: np.ndarray,
    rollout_std: np.ndarray,
) -> np.ndarray:
    """Per-patch deviation from reference rollout: |x - mean| / std."""
    diff = np.abs(rollout_map.astype(np.float64) - rollout_mean.astype(np.float64))
    return (diff / rollout_std.astype(np.float64)).astype(np.float32)


def per_image_zscore(scores: np.ndarray) -> np.ndarray:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return ((scores - mean) / (std + EPS)).astype(np.float32)


def fuse_branch_scores(
    knn_map: np.ndarray,
    rollout_dev_map: np.ndarray,
    mode: str = "weighted_sum",
    knn_weight: float = 0.5,
    rollout_weight: float = 0.5,
) -> np.ndarray:
    """Fuse kNN and rollout-deviation maps after per-image z-scoring."""
    knn_z = per_image_zscore(knn_map)
    roll_z = per_image_zscore(rollout_dev_map)

    if mode == "weighted_sum":
        total = knn_weight + rollout_weight
        if total <= 0:
            raise ValueError("Fusion weights must sum to a positive value.")
        w_knn = knn_weight / total
        w_roll = rollout_weight / total
        return (w_knn * knn_z + w_roll * roll_z).astype(np.float32)
    if mode == "product":
        return (knn_z * roll_z).astype(np.float32)
    if mode == "max":
        return np.maximum(knn_z, roll_z).astype(np.float32)
    raise ValueError(
        f"Unknown fusion mode: {mode!r}. Choose weighted_sum, product, or max."
    )


def resolve_layer_indices(
    layers: int | str | list[int] | None,
    num_layers: int,
) -> list[int]:
    """Resolve layer config to sorted 0-based indices."""
    if layers is None or layers == "last":
        return [num_layers - 1]
    if layers == "all":
        return list(range(num_layers))
    if isinstance(layers, int):
        if layers < 0 or layers >= num_layers:
            raise ValueError(
                f"Layer index {layers} out of range for model with {num_layers} layers."
            )
        return [layers]
    if isinstance(layers, list):
        resolved = []
        for layer in layers:
            resolved.extend(resolve_layer_indices(layer, num_layers))
        return sorted(set(resolved))
    raise ValueError(f"Invalid layers config: {layers!r}")


def extract_layer_outputs(
    model_wrapper,
    image_tensor: torch.Tensor,
    layers: int | str | list[int] | None = "last",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    """
    Extract CLS and patch tokens from selected transformer layers.

    Returns:
        layer_outputs: list of (patch_tokens, cls_token) per layer, each (N_patches, D) / (D,)
        grid_size: (grid_h, grid_w)
    """
    num_layers = len(model_wrapper.model.blocks)
    layer_indices = resolve_layer_indices(layers, num_layers)

    batch = image_tensor.unsqueeze(0).to(model_wrapper.device)
    with torch.inference_mode():
        outputs = model_wrapper.model.get_intermediate_layers(
            batch,
            n=num_layers,
            return_class_token=True,
            norm=True,
        )

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for layer_idx in layer_indices:
        patch_tokens, cls_token = outputs[layer_idx]
        patch_np = patch_tokens.squeeze(0).cpu().numpy().astype(np.float32)
        cls_np = cls_token.squeeze(0).cpu().numpy().astype(np.float32)
        results.append((patch_np, cls_np))

    ps = model_wrapper.model.patch_size
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    grid_size = (h // ps, w // ps)

    return results, grid_size


def extract_patch_tokens(
    model_wrapper,
    image_tensor: torch.Tensor,
    layers: int | str | list[int] | None = "last",
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Extract patch tokens, concatenating across layers when multiple are requested.

    Returns:
        features: (N_patches, D) or (N_patches, D * n_layers)
        grid_size: (grid_h, grid_w)
    """
    layer_outputs, grid_size = extract_layer_outputs(model_wrapper, image_tensor, layers)
    if len(layer_outputs) == 1:
        return layer_outputs[0][0], grid_size
    return np.concatenate([patches for patches, _ in layer_outputs], axis=1), grid_size


def extract_cls_token(
    model_wrapper,
    image_tensor: torch.Tensor,
    layer: int | str = "last",
) -> np.ndarray:
    """Extract CLS token from a single layer."""
    num_layers = len(model_wrapper.model.blocks)
    layer_idx = resolve_layer_index(layer, num_layers)
    layer_outputs, _ = extract_layer_outputs(model_wrapper, image_tensor, layer_idx)
    return layer_outputs[0][1]


def patch_tokens_to_grid(
    patch_tokens: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Reshape flat patch tokens to (H, W, D)."""
    gh, gw = grid_size
    return patch_tokens.reshape(gh, gw, -1)


def spatial_neighbor_aggregate(
    features_grid: np.ndarray,
    kernel: int = 3,
) -> np.ndarray:
    """
    Replace each patch feature with the mean of its spatial kernel neighbors.

    Args:
        features_grid: (H, W, D)
        kernel: odd spatial kernel size (default 3)
    """
    if kernel <= 1:
        return features_grid.astype(np.float32)

    gh, gw, dim = features_grid.shape
    pad = kernel // 2
    padded = np.pad(features_grid, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(features_grid, dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            patch = padded[i : i + kernel, j : j + kernel, :]
            out[i, j] = patch.reshape(-1, dim).mean(axis=0)
    return out
