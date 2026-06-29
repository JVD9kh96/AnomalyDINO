from __future__ import annotations

import numpy as np


def greedy_coreset(features: np.ndarray, ratio: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Greedy farthest-point coreset subsampling.

    Args:
        features: (N, D) feature matrix
        ratio: fraction of points to keep
        seed: RNG seed for initial point

    Returns:
        Subsampled features (n_keep, D)
    """
    n = features.shape[0]
    if n == 0:
        return features
    n_keep = max(1, int(n * ratio))
    if n_keep >= n:
        return features

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for _ in range(n_keep - 1):
        last = features[selected[-1]].astype(np.float64)
        dists = np.linalg.norm(features.astype(np.float64) - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))

    return features[selected]
