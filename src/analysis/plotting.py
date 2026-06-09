from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _shared_bins(
    healthy: np.ndarray,
    anomaly: np.ndarray,
    num_bins: int = 50,
) -> np.ndarray:
    combined = np.concatenate([healthy, anomaly]) if healthy.size and anomaly.size else (
        healthy if healthy.size else anomaly
    )
    if combined.size == 0:
        return np.linspace(0, 1, num_bins + 1)
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return np.linspace(vmin, vmax, num_bins + 1)


def plot_distribution_triptych(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    scorer_name: str,
    layer_index: int,
    output_path: str | Path,
    num_bins: int = 50,
) -> None:
    healthy = healthy_scores[np.isfinite(healthy_scores)]
    anomaly = anomaly_scores[np.isfinite(anomaly_scores)]
    bins = _shared_bins(healthy, anomaly, num_bins=num_bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    title = f"{scorer_name} (layer {layer_index})"

    axes[0].hist(
        anomaly,
        bins=bins,
        density=False,
        color="tab:red",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_title("Anomalous patches")
    axes[0].set_xlabel(scorer_name)
    axes[0].set_ylabel("Count")

    axes[1].hist(
        healthy,
        bins=bins,
        density=False,
        color="tab:green",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_title("Healthy patches")
    axes[1].set_xlabel(scorer_name)
    axes[1].set_ylabel("Count")

    axes[2].hist(
        healthy,
        bins=bins,
        density=False,
        color="tab:green",
        alpha=0.5,
        label="Healthy",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[2].hist(
        anomaly,
        bins=bins,
        density=False,
        color="tab:red",
        alpha=0.5,
        label="Anomalous",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[2].set_title("Overlay")
    axes[2].set_xlabel(scorer_name)
    axes[2].set_ylabel("Count")
    axes[2].legend()

    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])

    fig.suptitle(title)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_heatmap(
    image: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    im = axes[1].imshow(scores, cmap="hot", interpolation="nearest")
    axes[1].set_title(title or "Patch scores")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
