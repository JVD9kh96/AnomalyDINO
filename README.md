# Steel Repository Documentation

This repository contains two related systems:

1. **AnomalyDINO** — the original WACV 2025 few-shot anomaly detection code for MVTec-AD / VisA.
2. **Severstal CV stack** — a modular patch-level anomaly detection + SAM2 mask refinement pipeline for the [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) dataset, with a separate **analysis** package for probing DINO patch signals.

If you want to plug in a **custom detector**, start with [Adding a custom detector](#adding-a-custom-detector). The rest of this document explains the layout, contracts, and how existing pieces fit together.

---

## Table of contents

1. [Repository structure](#1-repository-structure)
2. [Architecture overview](#2-architecture-overview)
3. [Data](#3-data)
4. [Detector system](#4-detector-system)
5. [Built-in detectors](#5-built-in-detectors)
6. [Cross-validation evaluation](#6-cross-validation-evaluation)
7. [Segmenters (SAM2)](#7-segmenters-sam2)
8. [Analysis package](#8-analysis-package)
9. [Configs and entry points](#9-configs-and-entry-points)
10. [Adding a custom detector](#adding-a-custom-detector)
11. [Adding a custom analysis scorer](#adding-a-custom-analysis-scorer)
12. [Tips and pitfalls](#tips-and-pitfalls)

---

## 1. Repository structure

```
Steel/
├── configs/                  # YAML configs (CV + analysis)
├── scripts/                  # Threshold / ensemble tuning CLIs
├── src/
│   ├── detectors/            # Pluggable anomaly detectors (Severstal)
│   ├── evaluation/           # K-fold CV, metrics, threshold tuning
│   ├── severstal/            # Dataset, RLE, patch geometry
│   ├── segmenters/           # SAM2 mask refinement
│   ├── analysis/             # Patch signal distribution probes
│   ├── visualization/        # Severstal PDF overlays
│   ├── backbones.py          # DINOv2 / ViT wrappers
│   ├── detection.py          # Legacy AnomalyDINO object loop (MVTec/VisA)
│   ├── post_eval.py          # Legacy MVTec/VisA metrics
│   ├── utils.py              # Augment, maps, dataset info
│   └── visualize.py          # Legacy sample plots
├── tests/                    # Unit / smoke tests
├── run_severstal_cv.py       # Primary Severstal CV entry point
├── run_analysis.py           # Patch distribution analysis
├── run_anomalydino.py        # Original MVTec/VisA few-shot eval
├── run_anomalydino_batched.py
├── requirements.txt
└── README.md                 # Paper repro + Severstal tutorial
```

### Module roles

| Package / module | Role |
|------------------|------|
| `src/detectors/` | Anomaly detectors implementing `BaseAnomalyDetector`. Factory: `build_detector()`. |
| `src/evaluation/` | Orchestrates K-fold CV (`cross_validation.py`), patch/mask metrics, threshold & ensemble tuning. |
| `src/severstal/` | Loads images + RLE masks, fold splits, reference sampling, patch grid / GT label / SAM2 prompt helpers. |
| `src/segmenters/` | Turns anomalous patches into masks (SAM2 via Ultralytics). |
| `src/analysis/` | Scores every patch with GT labels; reports separability (AUROC, KS, …). No CV, no SAM2. |
| `src/visualization/` | Per-fold PDF overlays of image / GT / pred patches / SAM2. |
| `src/backbones.py` | Shared DINOv2 loading (`get_model`). |
| `src/detection.py`, `post_eval.py`, `utils.py`, `visualize.py` | Original AnomalyDINO MVTec/VisA path (not the Severstal detector API). |

### Detectors package breakdown

| File | Purpose |
|------|---------|
| `base.py` | `BaseAnomalyDetector`, `DetectorOutput` |
| `__init__.py` | `build_detector(config, seed)` factory |
| `anomaly_dino.py` | FAISS kNN memory-bank detector |
| `dino_sobel.py` | Feature-space Sobel norms |
| `dino_cls_cosine.py` | CLS↔patch cosine / prototype |
| `dino_attention_rollout.py` | Attention rollout scores |
| `dino_mahalanobis.py` | PaDiM-style diagonal Mahalanobis + PCA |
| `dino_knn_rollout.py` | kNN + reference rollout deviation fusion |
| `dino_iforest_rollout.py` | IsolationForest + rollout fusion |
| `ensemble.py` | Weighted z-score ensemble of sub-detectors |
| `dino_features.py` | Shared token / rollout / fusion helpers |
| `attention_features.py` | Attention capture + rollout math |
| `cls_patch_features.py` | CLS–patch cosine helpers |
| `sobel_features.py` | Sobel norms + calibration / score modes |
| `coreset.py` | Greedy coreset for memory banks |

---

## 2. Architecture overview

### Severstal CV path (main path for custom detectors)

```
YAML config
    │
    ▼
run_severstal_cv.py
    │
    ▼
run_cross_validation()
    │
    ├─ SeverstalDataset ──► fold split + select_reference_ids
    │
    ├─ build_detector() ──► detector.fit(refs)
    │
    ├─ for each val sample:
    │     detector.predict(sample) ──► DetectorOutput (patch_scores)
    │           │
    │           ├─► patch metrics (P/R/F1 vs GT patches)
    │           └─► binarize ──► bboxes/points ──► SAM2 ──► IoU/Dice
    │
    └─ results_severstal/<timestamp>/
```

There is **no separate train/infer binary** for Severstal: “training” is `detector.fit(reference_samples)` inside each fold; inference and metrics happen in the same run.

### Analysis path (signal exploration, not evaluation)

```
configs/analysis_severstal.yaml
    │
    ▼
run_analysis.py → src.analysis.cli → run_analysis()
    │
    ├─ DinoFeatureExtractor (CLS / patch / attention per layer)
    ├─ map GT mask → patch labels
    ├─ scorers (cls_patch_cosine, sobel_feature, …)
    └─ distributions + AUROC/AUPRC/… under results_analysis/
```

Use analysis to decide *which signals look useful*. Then implement a CV detector and tune `pred_score_threshold` separately — analysis score scales are **not** interchangeable with kNN / Mahalanobis / IsolationForest scores.

---

## 3. Data

### Severstal layout

```
data/severstal/
├── train_images/     # *.jpg (256 × 1600)
└── train.csv         # ImageId, ClassId, EncodedPixels
```

- Images with no rows in `train.csv` are treated as defect-free.
- Four defect classes (`ClassId` 1–4), RLE-encoded masks.

### Core types (`src/severstal/dataset.py`)

```python
@dataclass
class SeverstalSample:
    image_id: str
    image_path: Path
    masks_by_class: dict[int, np.ndarray]  # class_id → binary mask
    has_defect: bool
    image: np.ndarray                      # RGB, (H, W, 3)
```

`SeverstalDataset` handles discovery, annotations, stratified K-fold splits, and reference selection:

| `reference_sampling` | Behavior |
|----------------------|----------|
| `class_balanced` | `shots` images evenly across the 4 defect classes (`shots` must be divisible by 4) |
| `defect_free` | Only normal (no-defect) images |

`shots: 0` → empty reference list (zero-shot detectors; `fit()` may be a no-op).

### Patch geometry (`src/severstal/transforms.py`)

Detectors and metrics share DINOv2-style preprocessing:

- Resize so the smaller edge equals `resolution` (default `448`).
- Crop so H/W are multiples of `patch_size` (14 for DINOv2).
- `compute_processed_shape(native_shape, smaller_edge_size, patch_size)` returns `(processed_shape, grid_size)`.

GT patch labels use `gt_overlap_threshold` (fraction of defective pixels in a patch). Predicted patches are binarized with `pred_score_threshold`, then converted to bboxes/points for SAM2.

### MVTec / VisA (legacy)

Used only by `run_anomalydino.py` / `run_anomalydino_batched.py` via `src/detection.py` and `src/utils.get_dataset_info`. That path does **not** use `BaseAnomalyDetector`.

---

## 4. Detector system

### Base class (`src/detectors/base.py`)

```python
@dataclass
class DetectorOutput:
    image_id: str
    patch_scores: np.ndarray              # (grid_h, grid_w), higher = more anomalous
    grid_size: tuple[int, int]
    processed_shape: tuple[int, int]
    patch_size: int
    patch_valid_mask: np.ndarray | None = None      # optional (H, W) bool
    patch_class_scores: np.ndarray | None = None    # optional (H, W, C)


class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, reference_samples: list[SeverstalSample]) -> None: ...

    @abstractmethod
    def predict(self, sample: SeverstalSample) -> DetectorOutput: ...

    @property
    @abstractmethod
    def supports_class_prediction(self) -> bool: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

**You must implement:** `fit`, `predict`, `supports_class_prediction`.

### `DetectorOutput` contract

| Field | Shape | Meaning |
|-------|-------|---------|
| `patch_scores` | `(grid_h, grid_w)` | Continuous anomaly scores; **higher = more anomalous** |
| `grid_size` | `(grid_h, grid_w)` | Patch grid after resize/crop |
| `processed_shape` | `(H, W)` | Image size after preprocessing |
| `patch_size` | `int` | Patch edge in pixels (14 for DINOv2) |
| `patch_valid_mask` | `(H, W)` or `None` | Patches included in metrics (e.g. exclude PCA background) |
| `patch_class_scores` | `(H, W, C)` or `None` | Per-class scores; only if `supports_class_prediction` is `True` |

`processed_shape` / `patch_size` **must** match the geometry used for GT alignment in `src/severstal/transforms.py`. Prefer calling `compute_processed_shape(...)` rather than inventing your own resize logic.

### Factory registration (`src/detectors/__init__.py`)

```python
def build_detector(config: dict, seed: int = 42) -> BaseAnomalyDetector:
    name = config.get("name", "anomaly_dino")
    if name == "anomaly_dino":
        ...
    # add your branch here
    raise ValueError(f"Unknown detector: {name}")
```

The CV loop calls:

```python
detector = build_detector(detector_cfg, seed=fold_seed)
detector.fit(ref_samples)
out = detector.predict(sample)  # → DetectorOutput
```

### Registered detector names

| `detector.name` | Class |
|-----------------|-------|
| `anomaly_dino` | `AnomalyDINODetector` |
| `dino_sobel` | `DINOv2SobelDetector` |
| `dino_cls_cosine` | `DINOv2ClsPatchCosineDetector` |
| `dino_attention_rollout` | `DINOv2AttentionRolloutDetector` |
| `dino_mahalanobis` | `DINOv2MahalanobisDetector` |
| `dino_knn_rollout` | `DINOv2KnnRolloutDetector` |
| `dino_iforest_rollout` | `DINOv2IForestRolloutDetector` |
| `ensemble` | `EnsembleDetector` (via `build_ensemble_detector`) |

---

## 5. Built-in detectors

Shared YAML knobs for most detectors:

```yaml
detector:
  name: <factory_key>
  model_name: dinov2_vits14   # dinov2_vitb14, dinov2_vitl14, …
  resolution: 448
  device: cuda:0
  shots: 8                   # 0 = zero-shot where supported
  reference_sampling: class_balanced  # or defect_free
```

Always pair with:

```yaml
patch_eval:
  gt_overlap_threshold: 0.5
  pred_score_threshold: <tuned>   # use scripts/tune_patch_threshold.py
```

### `anomaly_dino` — FAISS kNN memory bank

- **Fit:** build a patch feature memory bank from references (optional coreset, rotation, masking).
- **Predict:** kNN distance grid (`knn_metric`, `k_neighbors`).
- **Requires** `shots > 0` for a meaningful bank.
- Config: `configs/severstal.yaml`

| Extra options | Description |
|---------------|-------------|
| `knn_metric` | `L2_normalized` (default) or `L2` |
| `k_neighbors` | Neighbors for scoring (default `1`) |
| `faiss_on_cpu` | Force CPU FAISS |
| `coreset_ratio` | Greedy subsample of bank (`null` = keep all) |
| `neighbor_aggregate` | 3×3 mean on features before indexing |
| `masking` / `mask_ref_images` / `rotation` | Preprocessing / augment options |

```shell
python run_severstal_cv.py --config configs/severstal.yaml
python scripts/tune_patch_threshold.py --config configs/severstal.yaml --fold 0
```

### `dino_sobel` — feature-space Sobel

- High Sobel norms in DINOv2 embedding space as anomaly cues.
- `shots: 0` → zero-shot; `shots > 0` → global calibration from refs.
- Config: `configs/severstal_dino_sobel.yaml`
- Analysis counterpart: `sobel_feature` scorer.

| Option | Values | Notes |
|--------|--------|-------|
| `sobel.norm_reduction` | `l2`, `mean`, `max` | Aggregate across embedding dims |
| `score_mode` | `raw`, `per_image_zscore`, `per_image_iqr`, `per_image_percentile` | How norms become scores |

Suggested thresholds: `raw` ~0.35; `per_image_zscore` ~2.0 (with `zscore_k: 2.0`).

### `dino_cls_cosine` — CLS↔patch cosine

| `scoring_mode` | Behavior |
|----------------|----------|
| `per_image` | `cos(cls_test, patch_test)` — zero-shot OK (`shots: 0`) |
| `prototype` | `1 - cos(mean_cls_defect_free, patch_test)` — needs `shots > 0`, `defect_free` refs |

Configs: `configs/severstal_dino_cls_cosine.yaml`, `configs/severstal_dino_cls_cosine_prototype.yaml`  
Analysis counterpart: `cls_patch_cosine`.

### `dino_attention_rollout` — attention rollout

- CLS→patch attention rollout across layers.
- Zero-shot with `shots: 0`; `shots > 0` only applies global z-score calibration (rankings largely unchanged).
- Config: `configs/severstal_dino_attention_rollout.yaml`
- Analysis counterpart: `attention_rollout`.

| `attention_rollout.*` | Typical |
|-----------------------|---------|
| `average_heads` | `true` |
| `include_residual` | `true` |
| `discard_ratio` | `0.0` or `0.7` |
| `last_n_layers` | `null` or `4` |
| `head_reduction` | `mean` / `max` |

For few-shot attention that **actually changes with shots**, use `dino_knn_rollout` or `dino_iforest_rollout`.

### `dino_knn_rollout` — kNN + rollout deviation

- Memory-bank kNN **plus** per-cell `|rollout - ref_mean| / ref_std`.
- **Requires `shots > 0`.** Changing shots changes both bank and rollout baseline.
- Config: `configs/severstal_dino_knn_rollout.yaml`

```yaml
fusion:
  mode: weighted_sum   # weighted_sum | product | max
  knn_weight: 0.5
  rollout_weight: 0.5
```

### `dino_iforest_rollout` — IsolationForest + rollout

- Same fusion idea as kNN+rollout, but the first branch is sklearn `IsolationForest` on reference patch features (`-score_samples` → higher = more anomalous).
- **Requires `shots > 0`.**
- Config: `configs/severstal_dino_iforest_rollout.yaml`

```yaml
iforest:
  n_estimators: 200
  max_samples: auto
  contamination: auto
  max_features: 1.0
  bootstrap: false
  n_jobs: -1
fusion:
  mode: weighted_sum
  iforest_weight: 0.5
  rollout_weight: 0.5
```

### `dino_mahalanobis` — PaDiM-style Mahalanobis

- Per-position diagonal Mahalanobis on patch features with optional PCA.
- **Requires `shots > 0`**; defaults to `prototype_reference_sampling: defect_free`.
- Configs: `configs/severstal_dino_mahalanobis.yaml`, `configs/severstal_dino_mahalanobis_multilayer.yaml`

| Option | Description |
|--------|-------------|
| `layers` | `last`, int, or list e.g. `[4, 8, 11]` |
| `pca_components` | PCA dim before fit (default `50`) |
| `neighbor_aggregate` | 3×3 spatial mean before fit/predict |

### `ensemble` — weighted z-score fusion

- Fits each sub-detector; at predict time, z-scores each map per image and takes a weighted sum.
- Tune weights on **fold 0 only**; report on folds 1–4.
- Config: `configs/severstal_dino_ensemble.yaml`

```shell
python scripts/tune_ensemble_weights.py --config configs/severstal_dino_ensemble.yaml
python run_severstal_cv.py --config configs/severstal_dino_ensemble.yaml
```

### Detector ↔ analysis mapping

| CV detector | Analysis scorer | Notes |
|-------------|-----------------|-------|
| `anomaly_dino` | — | Tune threshold via script |
| `dino_cls_cosine` (`per_image`) | `cls_patch_cosine` | Same raw signal |
| `dino_attention_rollout` | `attention_rollout` | Same raw signal |
| `dino_sobel` | `sobel_feature` | Exploration; weak AUROC alone |
| `dino_knn_rollout` / `dino_iforest_rollout` / `dino_mahalanobis` / `ensemble` | — | Fit-dependent; always tune CV threshold |

---

## 6. Cross-validation evaluation

**Entry:** `python run_severstal_cv.py --config configs/<name>.yaml [--fold N]`

**Orchestrator:** `src/evaluation/cross_validation.py::run_cross_validation`

Per fold:

1. Stratified train/val split; select `shots` reference IDs.
2. `build_detector` → `fit(refs)`.
3. For each val image: `predict` → continuous patch scores.
4. Binarize with `pred_score_threshold`; compare to GT patches (`gt_overlap_threshold`) → precision / recall / F1.
5. Anomalous patches → bboxes (or points) → SAM2 → IoU / Dice.
6. Write `fold_*/metrics.json` and optional `visualizations.pdf`.
7. Aggregate mean±std → `summary.json`.

### Output layout

```
results_severstal/<timestamp>/
├── config.yaml
├── folds.json
├── summary.json
└── fold_0/
    ├── metrics.json
    └── visualizations.pdf
```

### Metrics

| Level | Metrics | Module |
|-------|---------|--------|
| Patch | TP/FP/FN → P/R/F1 (global + image-mean); class-wise if detector supports it | `src/evaluation/patch_metrics.py` |
| Mask | IoU, Dice vs SAM2 | `src/evaluation/mask_metrics.py` |

Report **both** patch F1 and mask IoU/Dice when comparing methods.

### Threshold tuning

```shell
python scripts/tune_patch_threshold.py --config configs/<detector>.yaml --fold 0
python scripts/tune_patch_threshold.py --config configs/<detector>.yaml --fold 0 --with-sam2
```

Outputs go to `results_threshold_tuning/<timestamp>/`. Under heavy healthy:anomaly imbalance, prefer operating points like **recall@0.7** when SAM2 recall matters more than precision.

---

## 7. Segmenters (SAM2)

Interface (`src/segmenters/base.py`):

```python
class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, prompts: SegmenterPrompts) -> SegmenterOutput: ...
```

Factory: `build_segmenter()` in `src/segmenters/__init__.py`.  
Implementation: `src/segmenters/sam2_ultralytics.py` (Ultralytics downloads weights such as `sam2.1_b.pt` on first use).

Typical config:

```yaml
segmenter:
  name: sam2
  model: sam2.1_b.pt
  prompt_mode: bbox    # or point-style prompts from transforms
  min_prompt_area: 1
  device: cuda:0
```

---

## 8. Analysis package

**Purpose:** Test whether DINO-derived patch signals separate healthy vs anomalous patches using GT masks — **without** memory banks, CV folds, or SAM2.

**Entry:**

```shell
python run_analysis.py --config configs/analysis_severstal.yaml
python run_analysis.py --config configs/analysis_severstal.yaml --all-scorers
python run_analysis.py --config configs/analysis_severstal.yaml --scorer cls_patch_cosine --max_images 50
```

**Flow (`src/analysis/anomaly_distribution.py::run_analysis`):**

1. Load typed `AnalysisConfig` from YAML.
2. Build `DinoFeatureExtractor`; resolve layers (`last` / `all` / indices).
3. Iterate Severstal images via adapters → `AnalysisSample`.
4. Extract CLS / patch tokens / attentions → `FeatureBundle` per layer.
5. Map GT mask → patch labels (`overlap_ratio_threshold`, etc.).
6. Run each scorer → aggregate healthy vs anomaly scores.
7. Export `.npy`, `summary.json` (AUROC, AUPRC, KS, Wasserstein, Cohen’s d), `distribution.png`, optional heatmaps.

### Built-in scorers (`src/analysis/scorers.py`)

| Name | Signal |
|------|--------|
| `cls_patch_cosine` | Cosine(CLS, patch) |
| `patch_l2` | L2 norm of patch token |
| `sobel_feature` | Sobel on feature map |
| `sobel_image` | Image-space Sobel pooled per patch |
| `attention_rollout` | CLS→patch attention rollout |

### Analysis config sketch

```yaml
seed: 42
device: cuda:0
output_dir: results_analysis
model: { name: dinov2_vits14, resolution: 448 }
dataset: { name: severstal, root: data/severstal, image_shape: [256, 1600] }
layers: last
patch_label: { rule: overlap_ratio_threshold, threshold: 0.5 }
scorers: [cls_patch_cosine, patch_l2, sobel_feature, attention_rollout]
```

### Analysis vs CV

| | Analysis | CV (`run_severstal_cv`) |
|--|----------|-------------------------|
| Goal | Signal existence / separability | End-to-end detector quality |
| Fit / refs | No | Yes (`fit` + `shots`) |
| Threshold | N/A (ranking metrics) | `pred_score_threshold` |
| SAM2 | No | Yes |
| Custom extension | `BaseScorer` | `BaseAnomalyDetector` |

---

## 9. Configs and entry points

### CV config skeleton

```yaml
seed: 42

data:
  root: data/severstal
  image_shape: [256, 1600]
  num_classes: 4

cv:
  n_folds: 5
  stratify: true
  shuffle: true

patch_eval:
  gt_overlap_threshold: 0.5
  pred_score_threshold: 0.35

detector:
  name: anomaly_dino
  # … detector-specific keys …

segmenter:
  name: sam2
  model: sam2.1_b.pt
  prompt_mode: bbox
  device: cuda:0

output:
  dir: results_severstal
  save_visualizations: true
  max_viz_images_per_fold: 20
```

Ready-made configs live under `configs/severstal*.yaml` and `configs/analysis_severstal.yaml`.

### CLI cheat sheet

| Command | Role |
|---------|------|
| `python run_severstal_cv.py --config ... [--fold N]` | Full / single-fold CV |
| `python run_analysis.py --config configs/analysis_severstal.yaml` | Patch signal analysis |
| `python scripts/tune_patch_threshold.py --config ... --fold 0` | Sweep `pred_score_threshold` |
| `python scripts/tune_ensemble_weights.py --config ...` | Ensemble weight search |
| `python run_anomalydino.py ...` | Legacy MVTec/VisA few-shot |
| `python tests/test_severstal_unit.py` | Smoke tests (no GPU/data required for basic checks) |

CLI overrides for CV: `--data_root`, `--detector`, `--fold`.

---

## Adding a custom detector

### 1. Subclass `BaseAnomalyDetector`

Create e.g. `src/detectors/my_detector.py`:

```python
from __future__ import annotations

import numpy as np

from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


class MyDetector(BaseAnomalyDetector):
    def __init__(self, device: str = "cuda:0", resolution: int = 448, **kwargs):
        self.device = device
        self.resolution = resolution
        # store hyperparameters / model handles here

    @property
    def supports_class_prediction(self) -> bool:
        return False  # True only if you fill patch_class_scores

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        # Build memory bank / stats / forest / etc.
        # reference_samples[i].image is RGB (H, W, 3)
        # May be empty when shots == 0
        pass

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        native_shape = sample.image.shape[:2]
        patch_size = 14  # must match backbone
        processed_shape, grid_size = compute_processed_shape(
            native_shape,
            smaller_edge_size=self.resolution,
            patch_size=patch_size,
        )

        # (grid_h, grid_w), higher = more anomalous
        patch_scores = np.zeros(grid_size, dtype=np.float32)

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=patch_size,
            patch_valid_mask=None,
            patch_class_scores=None,
        )
```

Use `src/detectors/anomaly_dino.py` or `dino_iforest_rollout.py` as full reference implementations. Shared DINO helpers live in `dino_features.py`, `attention_features.py`, `cls_patch_features.py`, and `sobel_features.py`.

### 2. Register in the factory

In `src/detectors/__init__.py`, inside `build_detector`:

```python
if name == "my_detector":
    from src.detectors.my_detector import MyDetector
    return MyDetector(
        device=config.get("device", "cuda:0"),
        resolution=config.get("resolution", 448),
        # map YAML keys → constructor args
        # use `seed` for any RNG (PCA, forests, …)
    )
```

### 3. Add a YAML config

```yaml
# configs/severstal_my_detector.yaml
seed: 42

data:
  root: data/severstal
  image_shape: [256, 1600]
  num_classes: 4

cv:
  n_folds: 5
  stratify: true
  shuffle: true

patch_eval:
  gt_overlap_threshold: 0.5
  pred_score_threshold: 0.5   # tune after first dry run

detector:
  name: my_detector
  device: cuda:0
  resolution: 448
  shots: 8
  reference_sampling: class_balanced
  # your hyperparameters …

segmenter:
  name: sam2
  model: sam2.1_b.pt
  prompt_mode: bbox
  device: cuda:0

output:
  dir: results_severstal
  save_visualizations: true
  max_viz_images_per_fold: 20
```

### 4. Run and tune

```shell
# Smoke: single fold
python run_severstal_cv.py --config configs/severstal_my_detector.yaml --fold 0

# Tune decision threshold on fold 0 val
python scripts/tune_patch_threshold.py --config configs/severstal_my_detector.yaml --fold 0

# Full CV after updating pred_score_threshold in the YAML
python run_severstal_cv.py --config configs/severstal_my_detector.yaml
```

Or override the name without a new file:

```shell
python run_severstal_cv.py --config configs/severstal.yaml --detector my_detector
```

### Checklist for a working plug-in

- [ ] Subclasses `BaseAnomalyDetector` with `fit` / `predict` / `supports_class_prediction`
- [ ] Returns `DetectorOutput` with **higher scores = more anomalous**
- [ ] `processed_shape` / `patch_size` / `grid_size` match `compute_processed_shape`
- [ ] Registered under a unique string in `build_detector`
- [ ] YAML has `detector.name` plus any custom keys you read in the factory
- [ ] Threshold tuned with `tune_patch_threshold.py` (do not copy thresholds from other detectors)
- [ ] Optional: class-aware metrics via `patch_class_scores` + `supports_class_prediction=True`
- [ ] Optional: probe the raw signal first with a custom analysis scorer (below)

---

## Adding a custom analysis scorer

For early signal checks only (not a substitute for a CV detector):

1. Subclass `BaseScorer` in `src/analysis/scorers.py`:

```python
class MyScorer(BaseScorer):
    name = "my_scorer"

    def score(self, bundle: FeatureBundle, config: AnalysisConfig) -> np.ndarray:
        # return (grid_h, grid_w) float scores
        ...
```

2. Register in `SCORER_REGISTRY`.
3. Add the name under `scorers:` in `configs/analysis_severstal.yaml` (or pass `--scorer my_scorer`).

---

## Tips and pitfalls

- **Geometry first.** Misaligned `processed_shape` / `patch_size` silently breaks GT labels and SAM2 prompts. Prefer `compute_processed_shape` and the same `resolution` as DINOv2 preprocessing.
- **Score polarity.** Metrics treat high scores as anomalous. If your method naturally outputs “similarity to normal,” invert it (as `prototype` cosine does with `1 - cos`).
- **Zero-shot vs few-shot.** Empty refs when `shots: 0` — `fit` must tolerate that if you claim zero-shot support.
- **`class_balanced` shots.** Must be divisible by `num_classes` (4).
- **Reference policy overrides.** Some detectors force `defect_free` refs (e.g. Mahalanobis, CLS prototype, ensemble defaults) inside `run_cross_validation` — check that function if refs look wrong.
- **Do not reuse analysis thresholds** for kNN / Mahalanobis / IsolationForest / fused detectors.
- **Reproducibility.** Each fold uses `seed_all(seed + fold_idx)`; pass `seed` from `build_detector` into any randomized fit (PCA, forests, coreset).
- **Legacy path.** MVTec/VisA scripts do not go through `BaseAnomalyDetector`; custom Severstal detectors only need the factory + CV config path.

---

## Quick mental model

| Layer | Abstraction | Extend by |
|-------|-------------|-----------|
| Signal probe | `BaseScorer` | `SCORER_REGISTRY` |
| Anomaly map | `BaseAnomalyDetector` | `build_detector` |
| Mask refine | `BaseSegmenter` | `build_segmenter` |
| Eval loop | `run_cross_validation` | YAML + scripts |

Custom methods almost always mean: **implement `BaseAnomalyDetector` → register → YAML → tune threshold → `run_severstal_cv.py`.**
