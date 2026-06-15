from __future__ import annotations

import numpy as np
import torch


def compute_attention_weights(attn_module: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Compute softmax(QK^T/sqrt(d)) from a DINOv2 Attention module input."""
    batch, num_tokens, channels = x.shape
    num_heads = attn_module.num_heads
    head_dim = channels // num_heads
    qkv = attn_module.qkv(x).reshape(batch, num_tokens, 3, num_heads, head_dim)
    q, k, _v = qkv.unbind(2)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    scale = getattr(attn_module, "scale", head_dim**-0.5)
    weights = (q @ k.transpose(-2, -1)) * scale
    weights = torch.softmax(weights, dim=-1)
    return weights.detach().cpu().numpy()


def normalize_attention(attn: np.ndarray, average_heads: bool = True) -> np.ndarray:
    """Reduce attention to (tokens, tokens), averaging batch and heads."""
    arr = attn
    if arr.ndim == 4:
        arr = arr.mean(axis=(0, 1)) if average_heads else arr[0, 0]
    elif arr.ndim == 3:
        arr = arr.mean(axis=0) if average_heads else arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D–4D attention array, got shape {arr.shape}")
    return arr.astype(np.float32)


def capture_dino_attentions(model_wrapper, image_tensor: torch.Tensor) -> list[np.ndarray]:
    """Run forward pass with hooks to capture per-layer attention weight matrices."""
    model = model_wrapper.model
    storage: list[np.ndarray] = []

    def make_hook():
        def hook_fn(module, inp, _out):
            if not inp:
                return
            x = inp[0]
            if not isinstance(x, torch.Tensor):
                return
            storage.append(compute_attention_weights(module, x))

        return hook_fn

    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.register_forward_hook(make_hook()))

    try:
        batch = image_tensor.unsqueeze(0).to(model_wrapper.device)
        with torch.inference_mode():
            model.forward_features(batch)
    finally:
        for h in hooks:
            h.remove()

    return storage


def compute_attention_rollout(
    attentions: list[np.ndarray],
    average_heads: bool = True,
    include_residual: bool = True,
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute attention rollout from per-layer (tokens, tokens) attention matrices."""
    result = None
    num_tokens = attentions[0].shape[-1]

    for attn in attentions:
        a = normalize_attention(attn, average_heads=average_heads).astype(np.float64)
        if include_residual:
            a = a + np.eye(a.shape[0], dtype=np.float64)
        a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)

        if discard_ratio > 0:
            flat = a.reshape(-1)
            threshold = np.percentile(flat, 100 * (1 - discard_ratio))
            a = np.where(a < threshold, 0, a)
            a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)

        if result is None:
            result = a
        else:
            if result.shape[0] != a.shape[0]:
                min_t = min(result.shape[0], a.shape[0])
                result = result[:min_t, :min_t]
                a = a[:min_t, :min_t]
            result = result @ a

    if result is None:
        result = np.eye(num_tokens, dtype=np.float64)
    return result.astype(np.float32)


def rollout_to_patch_scores(
    rollout: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Extract CLS-to-patch rollout weights and reshape to patch grid."""
    cls_to_patches = rollout[0, 1:]
    expected = grid_size[0] * grid_size[1]
    if cls_to_patches.shape[0] != expected:
        raise ValueError(
            f"Rollout patch count {cls_to_patches.shape[0]} does not match "
            f"grid {grid_size} ({expected}). Check attention capture/normalization."
        )
    return cls_to_patches.reshape(grid_size).astype(np.float32)
