from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.analysis.metrics import compute_separability_metrics, distribution_summary
from src.evaluation.reproducibility import save_json


@dataclass
class ScoreAggregator:
    scorer_name: str
    layer_index: int
    save_per_image: bool = True

    healthy_scores: list[float] = field(default_factory=list)
    anomaly_scores: list[float] = field(default_factory=list)
    all_scores: list[float] = field(default_factory=list)
    all_labels: list[int] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    image_scores: dict[str, np.ndarray] = field(default_factory=dict)
    image_labels: dict[str, np.ndarray] = field(default_factory=dict)
    patch_coords: np.ndarray | None = None

    def add(
        self,
        image_id: str,
        scores: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray | None = None,
    ) -> None:
        if self.patch_coords is None and coords is not None:
            self.patch_coords = coords

        scores_flat = scores.ravel().astype(np.float32)
        labels_flat = labels.ravel().astype(bool)

        valid = np.isfinite(scores_flat)
        scores_flat = scores_flat[valid]
        labels_flat = labels_flat[valid]

        healthy = scores_flat[~labels_flat]
        anomaly = scores_flat[labels_flat]

        self.healthy_scores.extend(healthy.tolist())
        self.anomaly_scores.extend(anomaly.tolist())
        self.all_scores.extend(scores_flat.tolist())
        self.all_labels.extend(labels_flat.astype(int).tolist())
        self.image_ids.append(image_id)

        if self.save_per_image:
            self.image_scores[image_id] = scores.astype(np.float32)
            self.image_labels[image_id] = labels.astype(bool)

    def finalize(self) -> dict:
        healthy_arr = np.asarray(self.healthy_scores, dtype=np.float32)
        anomaly_arr = np.asarray(self.anomaly_scores, dtype=np.float32)
        all_scores_arr = np.asarray(self.all_scores, dtype=np.float32)
        all_labels_arr = np.asarray(self.all_labels, dtype=np.int32)

        summary = {
            "scorer": self.scorer_name,
            "layer": self.layer_index,
            "healthy": distribution_summary(healthy_arr),
            "anomaly": distribution_summary(anomaly_arr),
            "separability": compute_separability_metrics(
                healthy_arr,
                anomaly_arr,
                all_labels_arr,
                all_scores_arr,
            ),
        }
        return summary

    def save(self, output_dir: str | Path, save_heatmaps: bool = False) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        healthy_arr = np.asarray(self.healthy_scores, dtype=np.float32)
        anomaly_arr = np.asarray(self.anomaly_scores, dtype=np.float32)
        all_labels_arr = np.asarray(self.all_labels, dtype=np.int32)

        np.save(output_dir / "healthy_scores.npy", healthy_arr)
        np.save(output_dir / "anomaly_scores.npy", anomaly_arr)
        np.save(output_dir / "patch_labels.npy", all_labels_arr)
        np.save(
            output_dir / "image_ids.npy",
            np.asarray(self.image_ids, dtype=object),
        )
        if self.patch_coords is not None:
            np.save(output_dir / "patch_coords.npy", self.patch_coords)

        summary = self.finalize()
        save_json(summary, output_dir / "summary.json")

        if self.save_per_image:
            scores_dir = output_dir / "per_image_scores"
            labels_dir = output_dir / "per_image_labels"
            scores_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            for image_id, arr in self.image_scores.items():
                safe_id = image_id.replace("/", "_")
                np.save(scores_dir / f"{safe_id}.npy", arr)
                np.save(labels_dir / f"{safe_id}.npy", self.image_labels[image_id])
