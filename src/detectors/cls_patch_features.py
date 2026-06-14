from __future__ import annotations

import numpy as np

EPS = 1e-8


def compute_cls_patch_cosine(
    cls_token: np.ndarray,
    patch_tokens: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Cosine similarity between normalized CLS token and each patch token."""
    cls = cls_token.astype(np.float32)
    patches = patch_tokens.astype(np.float32)
    cls_norm = cls / (np.linalg.norm(cls) + EPS)
    patch_norms = np.linalg.norm(patches, axis=1, keepdims=True) + EPS
    patches_normed = patches / patch_norms
    sims = patches_normed @ cls_norm
    return sims.reshape(grid_size).astype(np.float32)


def resolve_layer_index(layer: int | str, num_layers: int) -> int:
    """Resolve layer config ('last' or int index) to a 0-based layer index."""
    if layer == "last":
        return num_layers - 1
    if isinstance(layer, int):
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"Layer index {layer} out of range for model with {num_layers} layers."
            )
        return layer
    raise ValueError(f"Invalid layer config: {layer!r}. Use 'last' or an int index.")
