"""Shared kNN helpers for AnomalyDINO memory banks and purification.

Uses FAISS when available; falls back to NumPy brute-force (CPU tests).
"""

from __future__ import annotations

import numpy as np

try:
    import faiss as _faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _faiss = None
    _HAS_FAISS = False


class _NumpyFlatIndex:
    """Minimal Flat-L2 index compatible with knn_distances()."""

    def __init__(self, dim: int):
        self.dim = dim
        self._data = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._data.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype("float32", copy=False))
        if self._data.shape[0] == 0:
            self._data = vectors.copy()
        else:
            self._data = np.concatenate([self._data, vectors], axis=0)

    def search(self, queries: np.ndarray, k: int):
        queries = np.ascontiguousarray(queries.astype("float32", copy=False))
        # squared L2
        # (Q,1,D) - (1,N,D) -> (Q,N)
        diff = queries[:, None, :] - self._data[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        k = min(k, self._data.shape[0])
        idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row = np.arange(queries.shape[0])[:, None]
        part = dists[row, idx]
        order = np.argsort(part, axis=1)
        idx = idx[row, order]
        part = part[row, order]
        return part.astype(np.float32), idx.astype(np.int64)


def prepare_bank_features(features: np.ndarray, knn_metric: str) -> np.ndarray:
    """Copy features and L2-normalize when using cosine-like distances."""
    out = np.ascontiguousarray(features.astype("float32", copy=True))
    if knn_metric == "L2_normalized":
        if _HAS_FAISS:
            _faiss.normalize_L2(out)
        else:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)
    return out


def build_faiss_index(
    features: np.ndarray,
    knn_metric: str,
    *,
    faiss_on_cpu: bool = True,
):
    """Build a Flat L2 index from (N, D) features (FAISS or NumPy fallback)."""
    prepared = prepare_bank_features(features, knn_metric)
    if prepared.shape[0] == 0:
        raise ValueError("Cannot build FAISS index from empty feature set.")

    if not _HAS_FAISS:
        index = _NumpyFlatIndex(prepared.shape[1])
        index.add(prepared)
        return index

    if faiss_on_cpu:
        index = _faiss.IndexFlatL2(prepared.shape[1])
    else:
        try:
            res = _faiss.StandardGpuResources()
            index = _faiss.GpuIndexFlatL2(res, prepared.shape[1])
        except Exception:
            index = _faiss.IndexFlatL2(prepared.shape[1])
    index.add(prepared)
    return index


def knn_distances(
    query_features: np.ndarray,
    index,
    knn_metric: str,
    k_neighbors: int = 1,
) -> np.ndarray:
    """
    Query index; return distances with AnomalyDINO polarity.

    Higher distance = more anomalous / less normal.
    """
    if query_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    queries = np.ascontiguousarray(query_features.astype("float32", copy=True))
    k = min(k_neighbors, max(1, index.ntotal))

    if knn_metric == "L2_normalized":
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-12)
        if _HAS_FAISS and not isinstance(index, _NumpyFlatIndex):
            _faiss.normalize_L2(queries)

    distances, _ = index.search(queries, k=k)
    if k > 1:
        distances = distances.mean(axis=1)
    else:
        distances = distances.reshape(-1)

    if knn_metric == "L2":
        return np.sqrt(np.maximum(distances, 0.0)).astype(np.float32)
    return (distances / 2.0).astype(np.float32)


def pairwise_knn_distances(
    query_features: np.ndarray,
    bank_features: np.ndarray,
    knn_metric: str,
    k_neighbors: int = 1,
    *,
    faiss_on_cpu: bool = True,
) -> np.ndarray:
    """Build a temporary index from bank_features and score query_features."""
    if bank_features.shape[0] == 0:
        raise ValueError("bank_features must be non-empty")
    index = build_faiss_index(bank_features, knn_metric, faiss_on_cpu=faiss_on_cpu)
    return knn_distances(query_features, index, knn_metric, k_neighbors)
