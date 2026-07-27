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
            coreset_ratio=config.get("coreset_ratio"),
            neighbor_aggregate=config.get("neighbor_aggregate", False),
            reference_mode=config.get("reference_mode"),
            allow_oracle_reference_filtering=config.get(
                "allow_oracle_reference_filtering", False
            ),
            use_dual_bank=config.get("use_dual_bank", False),
            dual_bank_alpha=config.get("dual_bank_alpha", 1.0),
            defect_mining_percentile=config.get("defect_mining_percentile", 99.5),
            reference_purification=config.get("reference_purification"),
            gt_overlap_threshold=config.get("gt_overlap_threshold", 0.5),
            num_classes=config.get("num_classes", 4),
            coreset_size=config.get("coreset_size"),
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
            scoring_mode=config.get("scoring_mode", "per_image"),
            prototype_reference_sampling=config.get(
                "prototype_reference_sampling", "defect_free"
            ),
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
            last_n_layers=rollout_cfg.get("last_n_layers"),
            head_reduction=rollout_cfg.get("head_reduction"),
        )
    if name == "dino_mahalanobis":
        from src.detectors.dino_mahalanobis import DINOv2MahalanobisDetector

        return DINOv2MahalanobisDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            layers=config.get("layers", "last"),
            pca_components=config.get("pca_components", 50),
            prototype_reference_sampling=config.get(
                "prototype_reference_sampling", "defect_free"
            ),
            neighbor_aggregate=config.get("neighbor_aggregate", False),
            pca_random_state=seed,
        )
    if name == "ensemble":
        from src.detectors.ensemble import build_ensemble_detector

        return build_ensemble_detector(config, seed=seed)
    if name == "dino_knn_rollout":
        from src.detectors.dino_knn_rollout import DINOv2KnnRolloutDetector
        from src.detectors.dino_features import RolloutConfig

        rollout_cfg = RolloutConfig.from_dict(config.get("attention_rollout"))
        fusion_cfg = config.get("fusion", {})
        return DINOv2KnnRolloutDetector(
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
            coreset_ratio=config.get("coreset_ratio"),
            neighbor_aggregate=config.get("neighbor_aggregate", False),
            rollout_cfg=rollout_cfg,
            fusion_mode=fusion_cfg.get("mode", "weighted_sum"),
            knn_weight=fusion_cfg.get("knn_weight", 0.5),
            rollout_weight=fusion_cfg.get("rollout_weight", 0.5),
        )
    if name == "dino_iforest_rollout":
        from src.detectors.dino_iforest_rollout import DINOv2IForestRolloutDetector
        from src.detectors.dino_features import RolloutConfig

        rollout_cfg = RolloutConfig.from_dict(config.get("attention_rollout"))
        fusion_cfg = config.get("fusion", {})
        if_cfg = config.get("iforest", {})
        return DINOv2IForestRolloutDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            masking=config.get("masking", False),
            mask_ref_images=config.get("mask_ref_images", False),
            rotation=config.get("rotation", False),
            pca_random_state=seed,
            coreset_ratio=config.get("coreset_ratio"),
            neighbor_aggregate=config.get("neighbor_aggregate", False),
            rollout_cfg=rollout_cfg,
            fusion_mode=fusion_cfg.get("mode", "weighted_sum"),
            iforest_weight=fusion_cfg.get("iforest_weight", 0.5),
            rollout_weight=fusion_cfg.get("rollout_weight", 0.5),
            n_estimators=if_cfg.get("n_estimators", 200),
            max_samples=if_cfg.get("max_samples", "auto"),
            contamination=if_cfg.get("contamination", "auto"),
            max_features=if_cfg.get("max_features", 1.0),
            bootstrap=if_cfg.get("bootstrap", False),
            n_jobs=if_cfg.get("n_jobs", -1),
        )
    if name == "dino_linear_probe":
        from src.detectors.dino_linear_probe import DINOv2LinearProbeDetector

        train_block = config.get("_train_block") or {}
        return DINOv2LinearProbeDetector(
            model_name=config.get("model_name", "dinov2_vits14"),
            resolution=config.get("resolution", 448),
            device=config.get("device", "cuda:0"),
            classification_mode=config.get("classification_mode", "binary"),
            binary_loss=config.get("binary_loss", "bce"),
            focal_gamma=config.get("focal_gamma", 2.0),
            focal_alpha=config.get("focal_alpha", 0.25),
            lambda_mc=config.get("lambda_mc", 1.0),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
            batch_size=config.get("batch_size", 4),
            epochs=config.get("epochs", 50),
            num_workers=config.get("num_workers", 0),
            num_classes=config.get("num_classes", 4),
            gt_overlap_threshold=config.get("gt_overlap_threshold", 0.5),
            threshold_num_steps=config.get("threshold_num_steps", 101),
            checkpoint_path=config.get("checkpoint_path"),
            save_plots=train_block.get("save_plots", True),
            log_format=train_block.get("log_format", "both"),
            seed=seed,
        )
    raise ValueError(f"Unknown detector: {name}")
