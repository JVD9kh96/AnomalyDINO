from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """BCE with logits. logits/targets shape (N,)."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Focal loss for binary classification (Lin et al.).

    logits/targets shape (N,). alpha weights the positive class.
    """
    targets = targets.float()
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return loss.mean()


def binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "bce":
        return binary_bce_loss(logits, targets)
    if loss_type == "focal":
        return binary_focal_loss(
            logits, targets, gamma=focal_gamma, alpha=focal_alpha
        )
    raise ValueError(f"Unknown binary_loss: {loss_type!r}. Choose 'bce' or 'focal'.")


def multiclass_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Softmax cross-entropy on anomalous patches only.

    logits: (N, C), targets: (N,) int64 in [0, C).
    Returns 0 if N == 0.
    """
    if logits.numel() == 0 or targets.numel() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits, targets)
