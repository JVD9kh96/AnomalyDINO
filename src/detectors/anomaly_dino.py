"""AnomalyDINO few-shot patch detector with reference-composition modes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.detectors.coreset import greedy_coreset
from src.detectors.knn_index import build_faiss_index, knn_distances
from src.detectors.reference_calibration import (
    NormalDistanceCalibration,
    calibrate_normal_distances,
)
from src.detectors.reference_purification import (
    apply_spatial_cleanup,
    dual_bank_scores,
    mine_suspected_defect_mask,
    oracle_keep_mask_from_gt,
    purify_reference_grid,
)
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


REFERENCE_MODES = (
    "clean",
    "contaminated_all",
    "class_balanced_all",
    "oracle_purified",
    "auto_purified",
    "random_filtered",
    "fixed_ratio_trim",
)


@dataclass
class ReferenceFeatureGrid:
    image_id: str
    features: np.ndarray
    grid_size: tuple[int, int]
    patch_keep_mask: np.ndarray | None = None


@dataclass
class MemoryBankStats:
    n_memory_patches_before_filtering: int = 0
    n_memory_patches_after_filtering: int = 0
    n_memory_patches_clean: int = 0
    n_candidate_patches_before_filter: int = 0
    n_candidate_patches_after_filter: int = 0
    n_memory_patches_before_budget: int = 0
    n_memory_patches_final: int = 0
    n_clean_patches: int = 0
    n_candidate_patches: int = 0
    n_accepted_candidate_patches: int = 0
    n_rejected_candidate_patches: int = 0
    acceptance_fraction: float = 0.0
    calibration_percentile: float | None = None
    calibration_threshold: float | None = None
    final_memory_bank_size: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def greedy_coreset_absolute(
    features: np.ndarray, n_keep: int, seed: int = 42
) -> np.ndarray:
    """Keep exactly n_keep patches via greedy coreset (or all if smaller)."""
    n = features.shape[0]
    if n_keep <= 0 or n_keep >= n:
        return features
    ratio = n_keep / float(n)
    # greedy_coreset uses int(n * ratio); nudge to hit n_keep.
    selected = greedy_coreset(features, ratio=min(1.0, ratio), seed=seed)
    if selected.shape[0] == n_keep:
        return selected
    if selected.shape[0] > n_keep:
        return selected[:n_keep]
    out = greedy_coreset(features, ratio=min(1.0, (n_keep + 1) / float(n)), seed=seed)
    if out.shape[0] >= n_keep:
        return out[:n_keep]
    return out


class AnomalyDINODetector(BaseAnomalyDetector):
    """AnomalyDINO few-shot patch-based detector for Severstal evaluation."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resolution: int = 448,
        device: str = "cuda:0",
        knn_metric: str = "L2_normalized",
        k_neighbors: int = 1,
        faiss_on_cpu: bool = False,
        masking: bool = False,
        mask_ref_images: bool = False,
        rotation: bool = False,
        pca_random_state: int = 42,
        coreset_ratio: float | None = None,
        neighbor_aggregate: bool = False,
        reference_mode: str | None = None,
        allow_oracle_reference_filtering: bool = False,
        use_dual_bank: bool = False,
        dual_bank_alpha: float = 1.0,
        defect_mining_percentile: float = 99.5,
        reference_purification: dict | None = None,
        gt_overlap_threshold: float = 0.5,
        num_classes: int = 4,
        coreset_size: int | None = None,
        budget_policy: str = "greedy_coreset",
    ):
        assert knn_metric in ("L2", "L2_normalized")
        if reference_mode is not None and reference_mode not in REFERENCE_MODES:
            raise ValueError(
                f"Unknown reference_mode={reference_mode!r}; "
                f"expected one of {REFERENCE_MODES}"
            )
        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.knn_metric = knn_metric
        self.k_neighbors = k_neighbors
        self.faiss_on_cpu = faiss_on_cpu
        self.masking = masking
        self.mask_ref_images = mask_ref_images
        self.rotation = rotation
        self.pca_random_state = pca_random_state
        self.coreset_ratio = coreset_ratio
        self.neighbor_aggregate = neighbor_aggregate
        self.reference_mode = reference_mode
        self.allow_oracle_reference_filtering = allow_oracle_reference_filtering
        self.use_dual_bank = use_dual_bank
        self.dual_bank_alpha = dual_bank_alpha
        self.defect_mining_percentile = defect_mining_percentile
        self.reference_purification = dict(reference_purification or {})
        self.gt_overlap_threshold = gt_overlap_threshold
        self.num_classes = num_classes
        self.coreset_size = coreset_size
        if budget_policy not in {"greedy_coreset", "random"}:
            raise ValueError("budget_policy must be 'greedy_coreset' or 'random'")
        self.budget_policy = budget_policy

        self._model = None
        self._knn_index = None
        self._patch_size = 14
        self._normal_bank_features: np.ndarray | None = None
        self._defect_bank_features: np.ndarray | None = None
        self._calibration: NormalDistanceCalibration | None = None
        self.last_bank_stats: MemoryBankStats = MemoryBankStats()
        self._feature_cache: dict[str, ReferenceFeatureGrid] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            from src.backbones import get_model

            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device[-1]))
            self._model = get_model(
                self.model_name, "cuda" if "cuda" in self.device else "cpu", self.resolution
            )
            self._patch_size = getattr(self._model.model, "patch_size", 14)

    @property
    def supports_class_prediction(self) -> bool:
        return False

    def _prepare_features(
        self, features: np.ndarray, grid_size: tuple[int, int]
    ) -> np.ndarray:
        if not self.neighbor_aggregate:
            return features
        from src.detectors.dino_features import (
            patch_tokens_to_grid,
            spatial_neighbor_aggregate,
        )

        grid = patch_tokens_to_grid(features, grid_size)
        aggregated = spatial_neighbor_aggregate(grid)
        return aggregated.reshape(-1, aggregated.shape[-1])

    def extract_reference_features(
        self,
        sample: SeverstalSample,
        *,
        use_cache: bool = True,
    ) -> ReferenceFeatureGrid:
        """Extract patch features for one reference image (optionally cached)."""
        import torch

        self._ensure_model()
        cache_key = sample.image_id
        if use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        with torch.inference_mode():
            # Rotation expands the bank; cache stores the unrotated primary grid.
            # Composition fitting uses unrotated grids for purification; rotation
            # is applied in build_memory_bank when self.rotation is True via
            # re-extraction of rotated views only at bank-build time if needed.
            tensor, grid_size = self._model.prepare_image(sample.image)
            features = self._model.extract_features(tensor)
            features = self._prepare_features(features, grid_size)
            mask_ref = self._model.compute_background_mask(
                features,
                grid_size,
                threshold=10,
                masking_type=(self.mask_ref_images and self.masking),
                random_state=self.pca_random_state,
            )
            keep = np.asarray(mask_ref, dtype=bool).ravel()

        grid = ReferenceFeatureGrid(
            image_id=sample.image_id,
            features=np.asarray(features, dtype=np.float32),
            grid_size=tuple(grid_size),
            patch_keep_mask=keep,
        )
        if use_cache:
            self._feature_cache[cache_key] = grid
        return grid

    def _collect_features_from_grids(
        self, feature_grids: list[ReferenceFeatureGrid]
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for grid in feature_grids:
            feats = grid.features
            if grid.patch_keep_mask is not None:
                mask = np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
                feats = feats[mask]
            if feats.shape[0]:
                parts.append(feats.astype(np.float32, copy=False))
        if not parts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype("float32")

    def _maybe_coreset(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return features
        if self.coreset_size is not None and self.coreset_size > 0:
            if self.budget_policy == "random":
                n_keep = min(int(self.coreset_size), int(features.shape[0]))
                rng = np.random.default_rng(self.pca_random_state)
                selected = np.sort(rng.choice(features.shape[0], n_keep, replace=False))
                return features[selected]
            return greedy_coreset_absolute(
                features, int(self.coreset_size), seed=self.pca_random_state
            )
        if self.coreset_ratio is not None and 0 < self.coreset_ratio < 1.0:
            return greedy_coreset(
                features, ratio=self.coreset_ratio, seed=self.pca_random_state
            )
        return features

    def build_memory_bank(
        self,
        feature_grids: list[ReferenceFeatureGrid],
        *,
        defect_feature_grids: list[ReferenceFeatureGrid] | None = None,
    ) -> None:
        """Build FAISS index from feature grids (respecting patch_keep_mask)."""
        features_ref = self._collect_features_from_grids(feature_grids)
        before = int(features_ref.shape[0])
        features_ref = self._maybe_coreset(features_ref)
        after = int(features_ref.shape[0])
        if after == 0:
            raise ValueError("No reference patch features extracted. Check reference images.")

        self._normal_bank_features = features_ref
        self._knn_index = build_faiss_index(
            features_ref, self.knn_metric, faiss_on_cpu=self.faiss_on_cpu
        )

        self._defect_bank_features = None
        if self.use_dual_bank and defect_feature_grids:
            defect_feats = self._collect_features_from_grids(defect_feature_grids)
            if defect_feats.shape[0] > 0:
                self._defect_bank_features = defect_feats

        self.last_bank_stats.n_memory_patches_after_filtering = before
        self.last_bank_stats.n_memory_patches_before_budget = before
        self.last_bank_stats.n_memory_patches_final = after
        self.last_bank_stats.final_memory_bank_size = after
        if self.last_bank_stats.n_memory_patches_before_filtering <= 0:
            self.last_bank_stats.n_memory_patches_before_filtering = before

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        """Backward-compatible fit: all patches from each sample enter the bank."""
        import torch

        from src.utils import augment_image

        self._ensure_model()
        self._feature_cache.clear()
        grids: list[ReferenceFeatureGrid] = []

        with torch.inference_mode():
            for sample in tqdm(reference_samples, desc="Building memory bank", leave=False):
                images = augment_image(sample.image) if self.rotation else [sample.image]
                for rot_idx, image in enumerate(images):
                    # Use a synthetic id for rotated views so they are not collapsed.
                    if rot_idx == 0 and not self.rotation:
                        grid = self.extract_reference_features(sample, use_cache=True)
                        grids.append(grid)
                    else:
                        tensor, grid_size = self._model.prepare_image(image)
                        features = self._model.extract_features(tensor)
                        features = self._prepare_features(features, grid_size)
                        mask_ref = self._model.compute_background_mask(
                            features,
                            grid_size,
                            threshold=10,
                            masking_type=(self.mask_ref_images and self.masking),
                            random_state=self.pca_random_state,
                        )
                        grids.append(
                            ReferenceFeatureGrid(
                                image_id=f"{sample.image_id}#rot{rot_idx}",
                                features=np.asarray(features, dtype=np.float32),
                                grid_size=tuple(grid_size),
                                patch_keep_mask=np.asarray(mask_ref, dtype=bool).ravel(),
                            )
                        )

        before = sum(
            int(g.features.shape[0] if g.patch_keep_mask is None else g.patch_keep_mask.sum())
            for g in grids
        )
        self.last_bank_stats = MemoryBankStats(
            n_memory_patches_before_filtering=before,
            n_clean_patches=before,
            n_memory_patches_clean=before,
        )
        self.build_memory_bank(grids)
        self._log_bank_stats()

    def fit_reference_composition(
        self,
        clean_samples: list[SeverstalSample],
        additional_samples: list[SeverstalSample] | None = None,
        *,
        reference_mode: str | None = None,
    ) -> MemoryBankStats:
        """
        Fit memory bank under a reference composition mode.

        Extracts each unique image once and reuses grids for calibration /
        purification / final bank construction.
        """
        mode = reference_mode or self.reference_mode or "clean"
        if mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference_mode={mode!r}")
        if mode == "oracle_purified" and not self.allow_oracle_reference_filtering:
            raise RuntimeError(
                "oracle_purified requires allow_oracle_reference_filtering=true"
            )

        additional_samples = additional_samples or []
        self._ensure_model()
        self._feature_cache.clear()

        clean_grids = [
            self.extract_reference_features(s, use_cache=True) for s in clean_samples
        ]
        additional_grids = [
            self.extract_reference_features(s, use_cache=True) for s in additional_samples
        ]

        n_clean = int(sum(_count_active(g) for g in clean_grids))
        n_cand = int(sum(_count_active(g) for g in additional_grids))
        before = n_clean + n_cand

        purif_cfg = self.reference_purification
        acceptance_pct = float(purif_cfg.get("normal_acceptance_percentile", 99.0))
        spatial_cleanup = bool(purif_cfg.get("spatial_cleanup", False))
        min_rej = int(purif_cfg.get("min_rejected_component_patches", 2))

        calibration: NormalDistanceCalibration | None = None
        calib_threshold: float | None = None
        accepted = 0
        rejected = 0
        filter_extras: dict[str, Any] = {}
        final_grids: list[ReferenceFeatureGrid] = []
        defect_grids: list[ReferenceFeatureGrid] = []

        if mode == "clean":
            final_grids = list(clean_grids)

        elif mode in ("contaminated_all", "class_balanced_all"):
            final_grids = [*clean_grids, *additional_grids]
            accepted = n_cand
            rejected = 0

        elif mode == "oracle_purified":
            final_grids = list(clean_grids)
            sample_by_id = {s.image_id: s for s in additional_samples}
            for grid in additional_grids:
                sample = sample_by_id[grid.image_id]
                oracle_keep = oracle_keep_mask_from_gt(
                    sample,
                    grid.grid_size,
                    self._patch_size,
                    self.resolution,
                    overlap_threshold=self.gt_overlap_threshold,
                    num_classes=self.num_classes,
                )
                base_keep = (
                    np.ones(grid.features.shape[0], dtype=bool)
                    if grid.patch_keep_mask is None
                    else np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
                )
                keep = base_keep & oracle_keep
                accepted += int(keep.sum())
                rejected += int(base_keep.sum() - keep.sum())
                final_grids.append(
                    ReferenceFeatureGrid(
                        image_id=grid.image_id,
                        features=grid.features,
                        grid_size=grid.grid_size,
                        patch_keep_mask=keep,
                    )
                )

        elif mode in ("auto_purified", "random_filtered", "fixed_ratio_trim"):
            if not clean_grids:
                raise ValueError(f"{mode} requires at least one clean reference")
            calibration = calibrate_normal_distances(
                clean_grids,
                knn_metric=self.knn_metric,
                k_neighbors=self.k_neighbors,
            )
            self._calibration = calibration
            calib_threshold = calibration.threshold_at(acceptance_pct)
            clean_index = build_faiss_index(
                self._collect_features_from_grids(clean_grids),
                self.knn_metric,
                faiss_on_cpu=True,
            )
            final_grids = list(clean_grids)
            base_keeps: list[np.ndarray] = []
            distance_scores: list[np.ndarray] = []
            automatic_keeps: list[np.ndarray] = []
            for grid in additional_grids:
                base_keep = (
                    np.ones(grid.features.shape[0], dtype=bool)
                    if grid.patch_keep_mask is None
                    else np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
                )
                # Score all feature rows; then intersect with base_keep.
                result = purify_reference_grid(
                    grid.features,
                    clean_index,
                    calib_threshold,
                    knn_metric=self.knn_metric,
                    k_neighbors=self.k_neighbors,
                )
                base_keeps.append(base_keep)
                distance_scores.append(result.scores)
                automatic_keeps.append(result.keep_mask & base_keep)

            if mode == "auto_purified":
                final_keeps = automatic_keeps
                if spatial_cleanup:
                    final_keeps = [
                        apply_spatial_cleanup(
                            keep, grid.grid_size, min_rejected_component_patches=min_rej
                        )
                        & base_keep
                        for grid, keep, base_keep in zip(
                            additional_grids, automatic_keeps, base_keeps
                        )
                    ]
                filter_extras["purification_strategy"] = "automatic_threshold"
            elif mode == "random_filtered":
                if spatial_cleanup:
                    raise ValueError("random_filtered requires spatial_cleanup=false")
                n_match = int(sum(keep.sum() for keep in automatic_keeps))
                n_candidates = int(sum(keep.sum() for keep in base_keeps))
                selected = np.zeros(n_candidates, dtype=bool)
                if n_match:
                    rng = np.random.default_rng(self.pca_random_state)
                    selected[rng.choice(n_candidates, n_match, replace=False)] = True
                final_keeps = []
                offset = 0
                for grid, base_keep in zip(additional_grids, base_keeps):
                    active = np.flatnonzero(base_keep)
                    keep = np.zeros(grid.features.shape[0], dtype=bool)
                    keep[active] = selected[offset : offset + active.size]
                    offset += active.size
                    final_keeps.append(keep)
                filter_extras.update(
                    {
                        "purification_strategy": "random_matched_to_automatic",
                        "matched_automatic_retained_patches": n_match,
                        "random_filter_seed": self.pca_random_state,
                    }
                )
            else:
                if spatial_cleanup:
                    raise ValueError("fixed_ratio_trim requires spatial_cleanup=false")
                trim_fraction = float(purif_cfg.get("fixed_trim_fraction", 0.0))
                if not 0.0 <= trim_fraction < 1.0:
                    raise ValueError("fixed_trim_fraction must be in [0, 1)")
                n_candidates = int(sum(keep.sum() for keep in base_keeps))
                n_keep_target = int(round((1.0 - trim_fraction) * n_candidates))
                scores = np.concatenate(
                    [score[base_keep] for score, base_keep in zip(distance_scores, base_keeps)]
                ) if n_candidates else np.zeros((0,), dtype=np.float32)
                selected = np.zeros(n_candidates, dtype=bool)
                order = np.argsort(scores, kind="stable")
                selected[order[:n_keep_target]] = True
                final_keeps = []
                offset = 0
                for grid, base_keep in zip(additional_grids, base_keeps):
                    active = np.flatnonzero(base_keep)
                    keep = np.zeros(grid.features.shape[0], dtype=bool)
                    keep[active] = selected[offset : offset + active.size]
                    offset += active.size
                    final_keeps.append(keep)
                calib_threshold = float(scores[order[n_keep_target - 1]]) if n_keep_target else None
                filter_extras.update(
                    {
                        "purification_strategy": "fixed_ratio_distance_trim",
                        "fixed_trim_fraction": trim_fraction,
                        "fixed_trim_retained_patches": n_keep_target,
                    }
                )

            for grid, keep, base_keep in zip(additional_grids, final_keeps, base_keeps):
                n_base = int(base_keep.sum())
                n_keep = int(keep.sum())
                accepted += n_keep
                rejected += n_base - n_keep
                final_grids.append(
                    ReferenceFeatureGrid(
                        image_id=grid.image_id,
                        features=grid.features,
                        grid_size=grid.grid_size,
                        patch_keep_mask=keep,
                    )
                )
                if self.use_dual_bank and mode == "auto_purified":
                    suspected = (~keep) & base_keep
                    if suspected.any():
                        defect_grids.append(
                            ReferenceFeatureGrid(
                                image_id=grid.image_id,
                                features=grid.features,
                                grid_size=grid.grid_size,
                                patch_keep_mask=suspected,
                            )
                        )

        else:
            raise ValueError(f"Unhandled reference_mode={mode!r}")

        # Optional dual-bank mining when not already filled by auto_purified.
        if self.use_dual_bank and not defect_grids and additional_grids and clean_grids:
            clean_feats = self._collect_features_from_grids(clean_grids)
            for grid in additional_grids:
                base_keep = (
                    np.ones(grid.features.shape[0], dtype=bool)
                    if grid.patch_keep_mask is None
                    else np.asarray(grid.patch_keep_mask, dtype=bool).ravel()
                )
                feats = grid.features
                suspected, _, _ = mine_suspected_defect_mask(
                    feats,
                    clean_feats,
                    self.defect_mining_percentile,
                    knn_metric=self.knn_metric,
                    k_neighbors=self.k_neighbors,
                )
                suspected = suspected & base_keep
                if suspected.any():
                    defect_grids.append(
                        ReferenceFeatureGrid(
                            image_id=grid.image_id,
                            features=grid.features,
                            grid_size=grid.grid_size,
                            patch_keep_mask=suspected,
                        )
                    )

        acceptance_fraction = (
            float(accepted / n_cand) if n_cand > 0 else (1.0 if mode == "clean" else 0.0)
        )
        self.last_bank_stats = MemoryBankStats(
            n_memory_patches_before_filtering=before,
            n_clean_patches=n_clean,
            n_candidate_patches=n_cand,
            n_memory_patches_clean=n_clean,
            n_candidate_patches_before_filter=n_cand,
            n_candidate_patches_after_filter=accepted,
            n_accepted_candidate_patches=accepted,
            n_rejected_candidate_patches=rejected,
            acceptance_fraction=acceptance_fraction,
            calibration_percentile=(
                acceptance_pct if mode in ("auto_purified", "random_filtered") else None
            ),
            calibration_threshold=calib_threshold,
            extras=filter_extras,
        )
        self.build_memory_bank(final_grids, defect_feature_grids=defect_grids or None)
        self._log_bank_stats()
        return self.last_bank_stats

    def _log_bank_stats(self) -> None:
        s = self.last_bank_stats
        print(
            "  Memory bank: "
            f"clean={s.n_clean_patches}, candidates={s.n_candidate_patches}, "
            f"accepted={s.n_accepted_candidate_patches}, "
            f"rejected={s.n_rejected_candidate_patches}, "
            f"accept_frac={s.acceptance_fraction:.4f}, "
            f"calib_pct={s.calibration_percentile}, "
            f"calib_thr={s.calibration_threshold}, "
            f"final_size={s.final_memory_bank_size}"
        )

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        import torch

        self._ensure_model()
        if self._knn_index is None and self._normal_bank_features is None:
            raise RuntimeError("Detector must be fit before predict.")

        native_shape = sample.image.shape[:2]
        with torch.inference_mode():
            tensor, grid_size = self._model.prepare_image(sample.image)
            features = self._model.extract_features(tensor)
            features = self._prepare_features(features, grid_size)

            if self.masking:
                patch_valid = self._model.compute_background_mask(
                    features,
                    grid_size,
                    threshold=10,
                    masking_type=True,
                    random_state=self.pca_random_state,
                )
            else:
                patch_valid = np.ones(features.shape[0], dtype=bool)

            features_masked = features[patch_valid]

            if (
                self.use_dual_bank
                and self._normal_bank_features is not None
                and self._defect_bank_features is not None
                and len(self._defect_bank_features) > 0
            ):
                distances = dual_bank_scores(
                    features_masked,
                    self._normal_bank_features,
                    self._defect_bank_features,
                    knn_metric=self.knn_metric,
                    k_neighbors=self.k_neighbors,
                    alpha=self.dual_bank_alpha,
                )
            else:
                distances = knn_distances(
                    features_masked,
                    self._knn_index,
                    self.knn_metric,
                    self.k_neighbors,
                )

            output_distances = np.zeros(features.shape[0], dtype=np.float32)
            output_distances[patch_valid] = distances.squeeze()
            patch_scores = output_distances.reshape(grid_size)

        processed_shape, _ = compute_processed_shape(
            native_shape, self.resolution, self._patch_size
        )

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=self._patch_size,
            patch_valid_mask=patch_valid.reshape(grid_size),
            patch_class_scores=None,
        )

    def inject_contamination(
        self,
        clean_features: np.ndarray,
        anomalous_features: np.ndarray,
        contamination_ratio: float,
        *,
        seed: int = 42,
    ) -> None:
        """
        Build a bank from clean features plus a controlled fraction of anomalies.

        contamination_ratio is the fraction of the *final* bank that is anomalous.
        """
        clean_features = np.asarray(clean_features, dtype=np.float32)
        anomalous_features = np.asarray(anomalous_features, dtype=np.float32)
        if clean_features.shape[0] == 0:
            raise ValueError("clean_features must be non-empty")
        ratio = float(contamination_ratio)
        if ratio <= 0 or anomalous_features.shape[0] == 0:
            bank = clean_features
        else:
            # n_anom / (n_clean + n_anom) = ratio  => n_anom = ratio/(1-ratio)*n_clean
            if ratio >= 1.0:
                n_anom = anomalous_features.shape[0]
                bank = anomalous_features
            else:
                n_anom = int(round(ratio / (1.0 - ratio) * clean_features.shape[0]))
                n_anom = max(1, min(n_anom, anomalous_features.shape[0]))
                rng = np.random.default_rng(seed)
                idx = rng.choice(anomalous_features.shape[0], size=n_anom, replace=False)
                bank = np.concatenate(
                    [clean_features, anomalous_features[idx]], axis=0
                )

        bank = self._maybe_coreset(bank)
        self._normal_bank_features = bank
        self._defect_bank_features = None
        self._knn_index = build_faiss_index(
            bank, self.knn_metric, faiss_on_cpu=self.faiss_on_cpu
        )
        self.last_bank_stats = MemoryBankStats(
            n_memory_patches_before_filtering=int(bank.shape[0]),
            n_memory_patches_after_filtering=int(bank.shape[0]),
            final_memory_bank_size=int(bank.shape[0]),
            extras={"contamination_ratio": ratio},
        )


def _count_active(grid: ReferenceFeatureGrid) -> int:
    if grid.patch_keep_mask is None:
        return int(grid.features.shape[0])
    return int(np.asarray(grid.patch_keep_mask, dtype=bool).sum())
