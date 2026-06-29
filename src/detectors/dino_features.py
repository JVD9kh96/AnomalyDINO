from __future__ import annotations

import numpy as np
import torch

from src.detectors.cls_patch_features import resolve_layer_index


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
