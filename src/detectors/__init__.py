from src.detectors.base import BaseAnomalyDetector, DetectorOutput


def build_detector(config: dict, seed: int = 42) -> BaseAnomalyDetector:
    name = config.get("name", "anomaly_dino")
    if name == "anomaly_dino":
        from src.detectors.anomaly_dino import AnomalyDINODetector

        return AnomalyDINODetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            knn_metric=config.get("knn_metric", "L2_normalized"),
            k_neighbors=config.get("k_neighbors", 1),
            faiss_on_cpu=config.get("faiss_on_cpu", False),
            masking=config.get("masking", False),
            mask_ref_images=config.get("mask_ref_images", False),
            rotation=config.get("rotation", False),
            pca_random_state=seed,
        )
    raise ValueError(f"Unknown detector: {name}")
