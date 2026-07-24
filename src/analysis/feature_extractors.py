from __future__ import annotations

import os

import numpy as np
import torch

from src.analysis.config import AnalysisConfig
from src.analysis.types import AnalysisSample, FeatureBundle
from src.backbones import get_model
from src.evaluation.reproducibility import clear_cuda_memory


class DinoFeatureExtractor:
    """Extract DINOv2 CLS/patch tokens and per-layer attention maps."""

    def __init__(self, config: AnalysisConfig):
        if not config.model.name.startswith("dinov2"):
            raise ValueError(
                f"Analysis module v1 supports DINOv2 only, got {config.model.name!r}"
            )
        self.config = config
        self._wrapper = None
        self._num_layers: int | None = None

    def _ensure_model(self) -> None:
        if self._wrapper is not None:
            return
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.device[-1]))
        cuda = "cuda" in self.config.device
        self._wrapper = get_model(
            self.config.model.name,
            "cuda" if cuda else "cpu",
            self.config.model.resolution,
        )
        self._num_layers = len(self._wrapper.model.blocks)

    @property
    def num_layers(self) -> int:
        self._ensure_model()
        assert self._num_layers is not None
        return self._num_layers

    @property
    def patch_size(self) -> int:
        self._ensure_model()
        return getattr(self._wrapper.model, "patch_size", 14)

    def _needs_attentions(self) -> bool:
        return "attention_rollout" in self.config.scorers

    def _tensor_to_display_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert preprocessed CxHxW tensor to RGB uint8 for visualization."""
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = np.clip(img * std + mean, 0, 1)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _compute_attention_weights(
        attn_module: torch.nn.Module, x: torch.Tensor
    ) -> np.ndarray:
        """Compute softmax(QK^T/sqrt(d)) from a DINOv2 Attention module input."""
        batch, num_tokens, channels = x.shape
        num_heads = attn_module.num_heads
        head_dim = channels // num_heads
        qkv = attn_module.qkv(x).reshape(
            batch, num_tokens, 3, num_heads, head_dim
        )
        q, k, _v = qkv.unbind(2)
        del qkv, _v
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        scale = getattr(attn_module, "scale", head_dim**-0.5)
        weights = (q @ k.transpose(-2, -1)) * scale
        del q, k
        weights = torch.softmax(weights, dim=-1)
        out = weights.detach().cpu().numpy()
        del weights
        return out

    def _capture_attentions(self, image_tensor: torch.Tensor) -> list[np.ndarray]:
        assert self._wrapper is not None
        model = self._wrapper.model
        storage: list[np.ndarray] = []

        def make_hook():
            def hook_fn(module, inp, _out):
                if not inp:
                    return
                x = inp[0]
                if not isinstance(x, torch.Tensor):
                    return
                storage.append(self._compute_attention_weights(module, x))

            return hook_fn

        hooks = []
        for blk in model.blocks:
            hooks.append(blk.attn.register_forward_hook(make_hook()))

        batch = None
        try:
            batch = image_tensor.unsqueeze(0).to(self._wrapper.device)
            with torch.inference_mode():
                model.forward_features(batch)
        finally:
            for h in hooks:
                h.remove()
            if batch is not None:
                del batch
            clear_cuda_memory()

        return storage

    def extract(self, sample: AnalysisSample) -> dict[int, FeatureBundle]:
        self._ensure_model()
        assert self._wrapper is not None

        image_tensor, grid_size = self._wrapper.prepare_image(sample.image)
        processed_shape = (image_tensor.shape[1], image_tensor.shape[2])
        patch_size = self.patch_size
        layer_indices = sorted(self.config.resolve_layer_indices(self.num_layers))

        attentions: list[np.ndarray] = []
        if self._needs_attentions():
            attentions = self._capture_attentions(image_tensor)

        preprocessed_image = self._tensor_to_display_image(image_tensor)

        batch = image_tensor.unsqueeze(0).to(self._wrapper.device)
        try:
            with torch.inference_mode():
                layer_outputs = self._wrapper.model.get_intermediate_layers(
                    batch,
                    n=layer_indices,
                    return_class_token=True,
                    norm=True,
                )

            attn_all = [
                self._normalize_attention(a) for a in attentions if a is not None
            ] or None
            del attentions

            bundles: dict[int, FeatureBundle] = {}
            for layer_idx, (patch_tokens, cls_token) in zip(
                layer_indices, layer_outputs
            ):
                patch_np = patch_tokens.squeeze(0).detach().cpu().numpy()
                cls_np = cls_token.squeeze(0).detach().cpu().numpy()

                expected = grid_size[0] * grid_size[1]
                if patch_np.shape[0] != expected:
                    raise ValueError(
                        f"Layer {layer_idx}: patch count {patch_np.shape[0]} != "
                        f"grid {grid_size} ({expected})"
                    )

                attn_layer = (
                    attn_all[layer_idx]
                    if attn_all is not None and layer_idx < len(attn_all)
                    else None
                )

                bundles[layer_idx] = FeatureBundle(
                    layer_index=layer_idx,
                    cls_token=cls_np.astype(np.float32),
                    patch_tokens=patch_np.astype(np.float32),
                    grid_size=grid_size,
                    processed_shape=processed_shape,
                    patch_size=patch_size,
                    attention=attn_layer,
                    attentions_all_layers=attn_all,
                    preprocessed_image=preprocessed_image,
                )
        finally:
            del batch
            if "layer_outputs" in locals():
                del layer_outputs
            del image_tensor
            clear_cuda_memory()

        return bundles

    @staticmethod
    def _normalize_attention(attn: np.ndarray) -> np.ndarray:
        """Reduce attention to (tokens, tokens), averaging batch and heads."""
        arr = attn
        if arr.ndim == 4:
            arr = arr.mean(axis=(0, 1))
        elif arr.ndim == 3:
            arr = arr.mean(axis=0)
        return arr.astype(np.float32)
