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
    if name == "dino_sobel":
        from src.detectors.dino_sobel import DINOv2SobelDetector

        sobel_cfg = config.get("sobel", {})
        return DINOv2SobelDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            norm_reduction=sobel_cfg.get("norm_reduction", "l2"),
            score_mode=config.get("score_mode", "raw"),
            zscore_k=config.get("zscore_k", 2.0),
            iqr_k=config.get("iqr_k", 1.5),
            percentile=config.get("percentile", 95.0),
            masking=config.get("masking", False),
            pca_random_state=seed,
        )
    if name == "dino_cls_cosine":
        from src.detectors.dino_cls_cosine import DINOv2ClsPatchCosineDetector

        return DINOv2ClsPatchCosineDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            layer=config.get("layer", "last"),
        )
    if name == "dino_attention_rollout":
        from src.detectors.dino_attention_rollout import DINOv2AttentionRolloutDetector

        rollout_cfg = config.get("attention_rollout", {})
        return DINOv2AttentionRolloutDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            average_heads=rollout_cfg.get("average_heads", True),
            include_residual=rollout_cfg.get("include_residual", True),
            discard_ratio=rollout_cfg.get("discard_ratio", 0.0),
        )
    raise ValueError(f"Unknown detector: {name}")
