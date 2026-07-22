from __future__ import annotations

import torch
import torch.nn as nn


class PatchLinearClassifier(nn.Module):
    """Linear probe(s) on frozen patch tokens."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 4,
        classification_mode: str = "binary",
    ):
        super().__init__()
        if classification_mode not in ("binary", "binary_multiclass"):
            raise ValueError(
                f"classification_mode must be 'binary' or 'binary_multiclass', "
                f"got {classification_mode!r}."
            )
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classification_mode = classification_mode

        self.binary_head = nn.Linear(feature_dim, 1)
        self.multiclass_head: nn.Linear | None
        if classification_mode == "binary_multiclass":
            self.multiclass_head = nn.Linear(feature_dim, num_classes)
        else:
            self.multiclass_head = None

    @property
    def uses_multiclass(self) -> bool:
        return self.multiclass_head is not None

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            patch_tokens: (..., D)

        Returns:
            binary_logits: (...,)
            multiclass_logits: (..., C) or None
        """
        binary_logits = self.binary_head(patch_tokens).squeeze(-1)
        multiclass_logits = None
        if self.multiclass_head is not None:
            multiclass_logits = self.multiclass_head(patch_tokens)
        return binary_logits, multiclass_logits
