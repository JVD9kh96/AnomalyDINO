# [WACV2025] AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 <img align="right" src="media/AnomalyDINO.png" style="height: 84px; max-width: 100%;">

*Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer*

This is the official code to reproduce the experiments in the paper [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529), accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025).

## Prerequisits

1. Create a virtual environment (e.g., `python -m venv .venvAnomalyDINO`), activate it (e.g., `source .venvAnomalyDINO/bin/activate`) and install the required dependencies for AnomalyDINO:
    ```shell
    pip install -r requirements.txt
    ```
    Info: If you want to use `faiss` with GPU-acceleration we recommend setting up a conda environment with the required packages instead (only conda installation is supported, see, e.g., [here](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss#why-dont-you-support-installing-via-xxx-)). To perform similarity search on CPU set the additional flag `--faiss_on_cpu`.

2. Download and prepare the datasets [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their official sources.
For VisA, follow the instruction in the official repo to organize the data in the official 1-class splits. 
The default data roots are `data/mvtec_anomaly_detection` for MVTec-AD, and `data/VisA_pytorch/1cls/` for VisA. 
Please adapt the function calls below if necessary. 
Alternatively, prepare your own dataset accordingly:
    ```
    your_data_root
    ├── object1
    │   ├── ground_truth        # anomaly annotations per anomaly type (optional)
    │   │   ├── anomaly_type1
    │   │   ├── ...
    │   ├── test                # test images per anomaly type & 'good' (if applicable)
    │   │   ├── anomaly_type1    
    │   │   ├── ...
    │   │   └── good
    │   └── train               # train/reference images (without anomalies)
    │       └── good
    ├── object2
    │   ├── ...
    ```
When no 'good' test set is available, just inference is performed (no evaluation of detection/segmentation metrics possible).

## Usage

### Short Demo
Get started with the minimal demo to perform few-shot anomaly detection (`demo_AD_DINO.ipynb`).

### Few-shot anomaly detection

For the full evaluation, run the script `run_anomalydino.py` on the selected dataset for a given number of shots and repetitions (seeds).
The preprocessing to your dataset can be specified in `src/utils.py` in `get_dataset_info`, default is "agnostic" (apply masking whenever PCA-based masking works well & augment reference samples by rotations, see the paper).

The results for the default setting, i.e., all considered shots, three repetitions, and agnostic preprocessing, can be reproduced by calling:
```shell
python run_anomalydino.py --dataset MVTec --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/mvtec_anomaly_detection
```

```shell
python run_anomalydino.py --dataset VisA --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/VisA_pytorch/1cls/
```

For a faster inspection use, e.g.,
```shell
python run_anomalydino.py --dataset MVTec --shots 1 --num_seeds 1 --preprocess informed --data_root data/mvtec_anomaly_detection
```

The script automatically creates some example plots, plots some anomaly maps for each object, and automatically evaluates each run (activate evaluation of segementation with `--eval_segm` if applicable).

Evaluation results are saved in the respective results directory as `metrics_seed={seed}.json` for each seed.


### Batched-Zero-Shot Anomay Detection
To reproduce the results in the *batched* zero-shot scenario, run `run_anomalydino_batched.py` with appropriate arguments:

```shell
python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
```
```shell
python run_anomalydino_batched.py --dataset VisA --data_root data/VisA_pytorch/1cls/
```

---

## Tutorial: Severstal Steel Defect Detection Evaluation

This repository extends AnomalyDINO with a modular evaluation protocol for the [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) dataset. The pipeline runs K-fold cross-validation, reports patch-level detection metrics (precision / recall / F1) and SAM2 mask-level metrics (IoU / Dice), and saves results as JSON plus visualization PDFs.

### Environment setup

1. **Create and activate a virtual environment** (recommended):

    ```shell
    python -m venv .venv
    # Linux / macOS
    source .venv/bin/activate
    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    ```

2. **Install PyTorch** for your CUDA version from [pytorch.org](https://pytorch.org/) if not already installed.

3. **Install dependencies**:

    ```shell
    pip install -r requirements.txt
    ```

    Notes:
    - **FAISS**: `faiss-gpu` is listed in `requirements.txt`. If GPU FAISS is unavailable, install `faiss-cpu` instead and set `detector.faiss_on_cpu: true` in the config.
    - **SAM2**: Ultralytics downloads the SAM2 weights (e.g. `sam2.1_b.pt`) automatically on first use.
    - **DINOv2**: Backbone weights are fetched via `torch.hub` on first run.

4. **Verify the install** (no dataset or GPU required):

    ```shell
    python tests/test_severstal_unit.py
    ```

### Dataset preparation

Download the Severstal competition data and place it under `data/severstal/`:

```
data/severstal/
├── train_images/          # all training images (.jpg)
└── train.csv              # annotations: ImageId, ClassId, EncodedPixels
```

- Images are **256 × 1600** pixels.
- `train.csv` lists defect segments per class (`ClassId` 1–4) in run-length-encoded (RLE) format.
- Images with no rows in `train.csv` are treated as defect-free (normal).

### Configuration

All experiment settings live in [`configs/severstal.yaml`](configs/severstal.yaml). Key options:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `cv` | `n_folds` | Number of cross-validation folds (default: 5) |
| `patch_eval` | `gt_overlap_threshold` | Fraction of patch pixels that must be defective for a GT patch to be positive |
| `patch_eval` | `pred_score_threshold` | Fixed absolute threshold on patch anomaly scores |
| `detector` | `shots` | Number of reference images per fold; with `class_balanced`, must be divisible by 4 (e.g. 8 → 2 per class) |
| `detector` | `reference_sampling` | `class_balanced` (default): equal shots per defect class; `defect_free`: legacy normal-only images |
| `detector` | `model_name` | DINOv2 backbone (e.g. `dinov2_vits14`) |
| `segmenter` | `model` | SAM2 checkpoint for Ultralytics (e.g. `sam2.1_b.pt`) |
| `output` | `dir` | Root directory for results |

### Running an experiment

**Full 5-fold cross-validation:**

```shell
python run_severstal_cv.py --config configs/severstal.yaml
```

**Single fold (useful for debugging):**

```shell
python run_severstal_cv.py --config configs/severstal.yaml --fold 0
```

**Override paths or detector from the command line:**

```shell
python run_severstal_cv.py --config configs/severstal.yaml --data_root data/severstal --detector anomaly_dino
```

#### What happens during a run

For each fold:

1. Training images are split into train / validation (stratified by defect presence).
2. A **memory bank** is built from reference images in the train split. By default (`reference_sampling: class_balanced`), `detector.shots` images are chosen evenly across the 4 defect classes (e.g. 8 shots → 2 train images containing class 1, 2 with class 2, etc.). Set `reference_sampling: defect_free` to use only normal (defect-free) images instead.
3. Each validation image is scored at **patch level** by the anomaly detector.
4. Predicted anomalous patches are passed as **bounding-box prompts** to SAM2 for mask refinement.
5. Metrics and visualizations are saved.

#### Output structure

Results are written to `results_severstal/<timestamp>/`:

```
results_severstal/<timestamp>/
├── config.yaml              # resolved config for reproducibility
├── folds.json               # image → validation fold assignment
├── summary.json             # mean ± std across folds
├── fold_0/
│   ├── metrics.json         # patch + mask metrics for this fold
│   └── visualizations.pdf   # GT/pred overlays (patch + mask level)
├── fold_1/
│   └── ...
└── ...
```

**Metrics reported:**

- **Patch level** — TP / FP / FN at patch granularity; precision, recall, F1 aggregated globally (sum over all patches) and as image-level means. Class-agnostic metrics are always computed; class-wise metrics require a class-aware detector.
- **Mask level** — IoU and Dice between SAM2-predicted masks and GT, aggregated globally and per-image.

Report **both** patch F1 and mask IoU/Dice when comparing detectors — patch F1 alone does not always predict SAM2 quality.

### Threshold tuning

Do **not** reuse analysis score distributions to set `pred_score_threshold` for kNN or Mahalanobis detectors (different score scales). Use the tuning scripts on a validation fold:

```shell
# Patch threshold sweep with PR curve + F1-optimal and recall@0.7 operating points
python scripts/tune_patch_threshold.py --config configs/severstal.yaml --fold 0

# Optional SAM2 preview on a val subset
python scripts/tune_patch_threshold.py --config configs/severstal.yaml --fold 0 --with-sam2

# Ensemble weights: tune on fold 0, benchmark on folds 1-4
python scripts/tune_ensemble_weights.py --config configs/severstal_dino_ensemble.yaml --tune-fold 0
```

Outputs land in `results_threshold_tuning/<timestamp>/` (`pr_curve.png`, `operating_points.png`, `threshold_tuning.json`) or `results_ensemble_tuning/<timestamp>/`.

Under patch imbalance (~30:1 healthy:anomaly), prefer the **recall@0.7** threshold for SAM2 downstream when recall matters more than precision.

### Detector ↔ analysis mapping

| CV detector | Config | Analysis scorer | Notes |
|-------------|--------|-----------------|-------|
| `anomaly_dino` | [`severstal.yaml`](configs/severstal.yaml) | — | kNN distance; tune threshold via script |
| `dino_cls_cosine` | [`severstal_dino_cls_cosine.yaml`](configs/severstal_dino_cls_cosine.yaml) | `cls_patch_cosine` | `scoring_mode: per_image` (zero-shot) |
| `dino_cls_cosine` (prototype) | [`severstal_dino_cls_cosine_prototype.yaml`](configs/severstal_dino_cls_cosine_prototype.yaml) | — | `scoring_mode: prototype`, `defect_free` refs |
| `dino_attention_rollout` | [`severstal_dino_attention_rollout.yaml`](configs/severstal_dino_attention_rollout.yaml) | `attention_rollout` | Zero-shot; `shots>0` only global z-score (rankings unchanged) |
| `dino_knn_rollout` | [`severstal_dino_knn_rollout.yaml`](configs/severstal_dino_knn_rollout.yaml) | — | kNN + reference rollout deviation; **shots changes rankings** |
| `dino_iforest_rollout` | [`severstal_dino_iforest_rollout.yaml`](configs/severstal_dino_iforest_rollout.yaml) | — | IsolationForest + reference rollout deviation; **shots changes rankings** |
| `dino_mahalanobis` | [`severstal_dino_mahalanobis.yaml`](configs/severstal_dino_mahalanobis.yaml) | — | PaDiM-style diagonal Mahalanobis + PCA |
| `dino_mahalanobis` (multi-layer) | [`severstal_dino_mahalanobis_multilayer.yaml`](configs/severstal_dino_mahalanobis_multilayer.yaml) | — | `layers: [4, 8, 11]` |
| `ensemble` | [`severstal_dino_ensemble.yaml`](configs/severstal_dino_ensemble.yaml) | — | Weighted z-score fusion; tune on fold 0 only |
| `dino_sobel` | [`severstal_dino_sobel.yaml`](configs/severstal_dino_sobel.yaml) | `sobel_feature` | `sobel_feature` AUROC ~0.44 — analysis only, not primary CV |

### DINOv2 Sobel detector (`dino_sobel`)

A built-in, self-supervised detector that applies Sobel edge detection in DINOv2 patch embedding space. High gradient norms in feature space are treated as anomaly cues.

**Run with the dedicated config:**

```shell
python run_severstal_cv.py --config configs/severstal_dino_sobel.yaml
```

Or switch detector in any config:

```yaml
detector:
  name: dino_sobel
```

#### Zero-shot vs few-shot calibration

| `shots` | Behavior |
|---------|----------|
| `0` | **Zero-shot** — no reference images; `fit()` is a no-op. Scoring uses only the test image (and per-image `score_mode` stats). |
| `> 0` | **Few-shot calibration** — reference images from the train fold (via `reference_sampling`) are used to compute global Sobel-norm statistics (`ref_mean`, `ref_std`, …). Test norms are adjusted as `(norm - ref_mean) / ref_std` before `score_mode` is applied. With `class_balanced`, `shots` must be divisible by 4. |

#### Detector config options

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model_name` | `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, … | DINOv2 backbone |
| `sobel.norm_reduction` | `l2`, `mean`, `max` | How to aggregate Sobel magnitude across embedding dimensions |
| `score_mode` | see table below | How Sobel norms become `patch_scores` |
| `zscore_k` | float (default `2.0`) | Reference for thresholding z-scores (pairs with `pred_score_threshold`) |
| `iqr_k` | float (default `1.5`) | Divisor for `per_image_iqr` scores |
| `percentile` | float (default `95`) | Percentile cutoff for `per_image_percentile` |
| `masking` | `true` / `false` | Optional DINOv2 PCA background mask (excluded patches scored as 0) |

#### `score_mode` options

All modes output continuous `patch_scores`; the evaluation pipeline still binarizes them with `patch_eval.pred_score_threshold` unless scores are already on a known scale.

| `score_mode` | What it does | Suggested `pred_score_threshold` |
|--------------|--------------|----------------------------------|
| `raw` | Sobel norm (calibrated if `shots > 0`) | Tune empirically (e.g. `0.35`) |
| `per_image_zscore` | Per-image z-score: `(norm - mean) / std`. With calibration, reference z-scoring replaces per-image mean/std | `2.0` (aligns with `zscore_k`) |
| `per_image_iqr` | Distance above Q3 relative to IQR: `(norm - Q3) / IQR`, scaled by `iqr_k` | `0.5`–`1.0` |
| `per_image_percentile` | Positive part above the per-image `percentile` threshold, max-normalized to [0, 1] | `0.1`–`0.5` |

**Example — zero-shot with z-score scoring:**

```yaml
detector:
  name: dino_sobel
  shots: 0
  score_mode: per_image_zscore

patch_eval:
  pred_score_threshold: 2.0
```

**Example — few-shot calibrated raw scores:**

```yaml
detector:
  name: dino_sobel
  shots: 8
  reference_sampling: class_balanced
  score_mode: raw

patch_eval:
  pred_score_threshold: 0.35
```

### DINOv2 CLS cosine detector (`dino_cls_cosine`)

Patch anomaly scores from CLS-to-patch cosine similarity at a chosen transformer layer.

| `scoring_mode` | Behavior |
|----------------|----------|
| `per_image` (default) | `cos(cls_test, patch_test)` — matches `cls_patch_cosine` analysis scorer |
| `prototype` | `1 - cos(mean_cls_defect_free, patch_test)` — higher = more anomalous; uses `prototype_reference_sampling: defect_free` |

**Run:**

```shell
python run_severstal_cv.py --config configs/severstal_dino_cls_cosine.yaml
python run_severstal_cv.py --config configs/severstal_dino_cls_cosine_prototype.yaml
```

#### Zero-shot vs few-shot calibration

| `shots` | `per_image` | `prototype` |
|---------|-------------|-------------|
| `0` | Zero-shot raw cosine (tune from analysis) | Not supported |
| `> 0` | Score z-score calibration | CLS prototype from defect-free refs + score calibration |

#### Detector config options

| Parameter | Values | Description |
|-----------|--------|-------------|
| `scoring_mode` | `per_image`, `prototype` | How CLS reference is chosen |
| `prototype_reference_sampling` | `defect_free` (default) | Reference images for prototype CLS |
| `layer` | `last` or int | Transformer layer |
| `shots` | int | Reference images; `0` = zero-shot (`per_image` only) |

**Example — prototype few-shot:**

```yaml
detector:
  name: dino_cls_cosine
  scoring_mode: prototype
  prototype_reference_sampling: defect_free
  shots: 8

patch_eval:
  pred_score_threshold: 0.15  # tune via scripts/tune_patch_threshold.py
```

### DINOv2 attention rollout detector (`dino_attention_rollout`)

Patch anomaly scores from CLS-to-patch attention rollout across all transformer layers. Matches the `attention_rollout` analysis scorer.

**Run with the dedicated config:**

```shell
python run_severstal_cv.py --config configs/severstal_dino_attention_rollout.yaml
```

#### Zero-shot vs few-shot calibration

Same `shots` / `reference_sampling` behavior as `dino_cls_cosine` and `dino_sobel`. When `shots > 0`, scores are reference-normalized z-scores and require separate threshold tuning.

#### Detector config options

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model_name` | `dinov2_vits14`, … | DINOv2 backbone |
| `attention_rollout.average_heads` | bool (default `true`) | Average attention heads before rollout |
| `attention_rollout.include_residual` | bool (default `true`) | Add identity to each layer attention |
| `attention_rollout.discard_ratio` | float (default `0.0`) | Sparsify low-attention weights (try `0.7`) |
| `attention_rollout.last_n_layers` | int or null | Rollout over last N layers only (try `4`) |
| `attention_rollout.head_reduction` | `mean`, `max` | Aggregate heads before rollout |
| `shots` | int (default `0`) | Reference images for calibration; `0` = off |
| `reference_sampling` | `class_balanced`, `defect_free` | How reference images are chosen when `shots > 0` |

Tune zero-shot thresholds from analysis distributions or `scripts/tune_patch_threshold.py`.

For few-shot attention signal that **actually changes** with `shots`, use [`dino_knn_rollout`](#dino_knn_rollout-detector-dino_knn_rollout) instead.

### DINO kNN + rollout detector (`dino_knn_rollout`)

Combines AnomalyDINO kNN patch distances with **reference-anchored rollout deviation** (`|rollout - ref_mean| / ref_std` per grid cell). Unlike standalone `dino_attention_rollout`, changing `shots` changes both the memory bank and the normal rollout baseline, so threshold tuning metrics should differ across `shots: 8` vs `16`.

**Requires `shots > 0`** and uses `reference_sampling: class_balanced` by default (same as [`severstal.yaml`](configs/severstal.yaml)).

```shell
# Tune threshold on fold 0 (verify metrics change when you edit shots)
python scripts/tune_patch_threshold.py --config configs/severstal_dino_knn_rollout.yaml --fold 0

# CV after setting pred_score_threshold from tuning output
python run_severstal_cv.py --config configs/severstal_dino_knn_rollout.yaml --fold 0
```

| Parameter | Description |
|-----------|-------------|
| `knn_metric` | `L2_normalized` (default) or `L2` |
| `k_neighbors` | kNN neighbors (default `1`) |
| `attention_rollout.*` | Same as `dino_attention_rollout` (`last_n_layers`, `discard_ratio`, `head_reduction`, …) |
| `fusion.mode` | `weighted_sum` (default), `product`, or `max` |
| `fusion.knn_weight` / `fusion.rollout_weight` | Branch weights for `weighted_sum` |
| `coreset_ratio` | Optional memory-bank subsampling |
| `neighbor_aggregate` | 3×3 neighbor mean on patch features before kNN |

### DINO IsolationForest + rollout detector (`dino_iforest_rollout`)

Same idea as `dino_knn_rollout`, but replaces the kNN branch with an **IsolationForest** fitted on reference patch features. The IsolationForest per-patch anomaly score is `-score_samples(X)` (higher = more anomalous). Changing `shots` changes the fitted forest and rollout baseline, so tuning metrics should differ for `shots: 8` vs `16`.

```shell
# Tune threshold on fold 0
python scripts/tune_patch_threshold.py --config configs/severstal_dino_iforest_rollout.yaml --fold 0

# CV after setting pred_score_threshold from tuning output
python run_severstal_cv.py --config configs/severstal_dino_iforest_rollout.yaml --fold 0
```

| Parameter | Description |
|-----------|-------------|
| `iforest.n_estimators` | Number of trees (default `200`) |
| `iforest.max_samples` | Subsample size per tree (`auto` by default) |
| `iforest.contamination` | Used internally by sklearn (we still tune `pred_score_threshold`) |
| `iforest.max_features` | Feature subsampling fraction (default `1.0`) |
| `fusion.iforest_weight` / `fusion.rollout_weight` | Branch weights for `weighted_sum` |

### DINOv2 Mahalanobis detector (`dino_mahalanobis`)

PaDiM-style per-position diagonal Mahalanobis distance on DINOv2 patch features with optional PCA (`pca_components: 50`). Uses `prototype_reference_sampling: defect_free` by default.

```shell
python run_severstal_cv.py --config configs/severstal_dino_mahalanobis.yaml
python run_severstal_cv.py --config configs/severstal_dino_mahalanobis_multilayer.yaml
```

| Parameter | Description |
|-----------|-------------|
| `layers` | `last`, int, or list e.g. `[4, 8, 11]` |
| `pca_components` | PCA dim before fitting (default `50`) |
| `neighbor_aggregate` | 3×3 spatial mean pooling on features before fit/predict |
| `shots` | Must be `> 0` |

### Ensemble detector (`ensemble`)

Weighted sum of sub-detectors after per-image z-score normalization. Tune weights on **fold 0 val only**; report metrics on **folds 1–4** to avoid circular evaluation.

```shell
python scripts/tune_ensemble_weights.py --config configs/severstal_dino_ensemble.yaml
python run_severstal_cv.py --config configs/severstal_dino_ensemble.yaml
```

### AnomalyDINO (`anomaly_dino`) options

| Parameter | Description |
|-----------|-------------|
| `coreset_ratio` | Greedy coreset subsampling of memory bank (e.g. `0.1`); `null` = keep all |
| `neighbor_aggregate` | 3×3 neighbor mean on patch features before indexing |

Tune `pred_score_threshold` with `scripts/tune_patch_threshold.py` — do not copy thresholds from cosine/Sobel configs.

Use [`run_analysis.py`](run_analysis.py) with `configs/analysis_severstal.yaml` to compare analysis scorers (AUROC). Analysis validates signal existence; CV detectors need per-detector threshold tuning.

### Patch signal distribution analysis

The `src/analysis/` package compares patch-level DINO signals between healthy and anomalous regions using ground-truth masks. It is separate from the CV evaluation pipeline and useful for understanding which features separate defects.

**Run on Severstal:**

```shell
python run_analysis.py --config configs/analysis_severstal.yaml
python run_analysis.py --config configs/analysis_severstal.yaml --all-scorers
python run_analysis.py --config configs/analysis_severstal.yaml --scorer cls_patch_cosine --max_images 50
```

**Built-in scorers:**

| Scorer | Description |
|--------|-------------|
| `cls_patch_cosine` | Cosine similarity between CLS token and each patch token |
| `patch_l2` | L2 norm of each patch token |
| `sobel_feature` | Feature-space Sobel norm (default; reuses `sobel_features.py`) |
| `sobel_image` | Image-space Sobel magnitude pooled per patch |
| `attention_rollout` | CLS-to-patch attention rollout score |

**Key config options** ([`configs/analysis_severstal.yaml`](configs/analysis_severstal.yaml)):

| Parameter | Description |
|-----------|-------------|
| `layers` | `last`, `all`, or list of layer indices |
| `patch_label.rule` | `center_point`, `any_overlap`, `overlap_ratio_threshold` (default), `majority_vote` |
| `patch_label.threshold` | Overlap fraction for `overlap_ratio_threshold` (default 0.5) |
| `sobel.norm_reduction` | `l2`, `mean`, or `max` for feature-space Sobel |

**Output** (`results_analysis/<timestamp>/`):

```
{scorer}/layer_{k}/
  healthy_scores.npy
  anomaly_scores.npy
  patch_labels.npy
  summary.json          # means, KS, Wasserstein, Cohen's d, AUROC, AUPRC
  distribution.png      # 3-panel raw-count histogram
  per_image_scores/
  heatmaps/
```

**Adding a new scorer:** subclass `BaseScorer` in [`src/analysis/scorers.py`](src/analysis/scorers.py) and register in `SCORER_REGISTRY`.

### Adding a new anomaly detector

The evaluation framework is built around a pluggable detector interface. To add your own method:

#### 1. Subclass `BaseAnomalyDetector`

Create a new file, e.g. `src/detectors/my_detector.py`:

```python
from __future__ import annotations

import numpy as np

from src.detectors.base import BaseAnomalyDetector, DetectorOutput
from src.severstal.dataset import SeverstalSample
from src.severstal.transforms import compute_processed_shape


class MyDetector(BaseAnomalyDetector):
    def __init__(self, device: str = "cuda:0", **kwargs):
        self.device = device
        # your init here

    @property
    def supports_class_prediction(self) -> bool:
        # Return True if you produce per-class patch scores
        return False

    def fit(self, reference_samples: list[SeverstalSample]) -> None:
        # Build state from defect-free reference images
        # reference_samples[i].image is an RGB numpy array (H, W, 3)
        pass

    def predict(self, sample: SeverstalSample) -> DetectorOutput:
        native_shape = sample.image.shape[:2]
        patch_size = 14  # must match your backbone's patch size
        processed_shape, grid_size = compute_processed_shape(
            native_shape, smaller_edge_size=448, patch_size=patch_size
        )

        # patch_scores: (grid_h, grid_w), higher = more anomalous
        patch_scores = np.zeros(grid_size, dtype=np.float32)

        return DetectorOutput(
            image_id=sample.image_id,
            patch_scores=patch_scores,
            grid_size=grid_size,
            processed_shape=processed_shape,
            patch_size=patch_size,
            patch_valid_mask=None,          # optional (H_grid, W_grid) bool mask
            patch_class_scores=None,        # optional (H, W, C) if class-aware
        )
```

**Contract for `DetectorOutput`:**

| Field | Shape | Description |
|-------|-------|-------------|
| `patch_scores` | `(grid_h, grid_w)` | Continuous anomaly score per patch |
| `grid_size` | `(grid_h, grid_w)` | Patch grid dimensions after model resize/crop |
| `processed_shape` | `(H, W)` | Image size after preprocessing (before patchification) |
| `patch_size` | int | Patch edge length in pixels (14 for DINOv2) |
| `patch_valid_mask` | `(grid_h, grid_w)` or `None` | Patches to include in metrics (exclude background) |
| `patch_class_scores` | `(grid_h, grid_w, C)` or `None` | Per-class scores; set only if `supports_class_prediction` is `True` |

Use [`src/detectors/anomaly_dino.py`](src/detectors/anomaly_dino.py) as a reference implementation.

#### 2. Register the detector in the factory

Add your detector to [`src/detectors/__init__.py`](src/detectors/__init__.py):

```python
def build_detector(config: dict, seed: int = 42) -> BaseAnomalyDetector:
    name = config.get("name", "anomaly_dino")
    if name == "anomaly_dino":
        ...
    elif name == "my_detector":
        from src.detectors.my_detector import MyDetector
        return MyDetector(device=config.get("device", "cuda:0"))
    raise ValueError(f"Unknown detector: {name}")
```

#### 3. Add config entries

In `configs/severstal.yaml` (or your own config file):

```yaml
detector:
  name: my_detector
  device: cuda:0
  # any custom hyperparameters for your detector
```

Then run:

```shell
python run_severstal_cv.py --config configs/severstal.yaml --detector my_detector
```

#### Tips

- **Preprocessing alignment**: GT masks are aligned to your patch grid in [`src/severstal/transforms.py`](src/severstal/transforms.py). If your detector uses different resize/crop logic, update `compute_processed_shape` / `resize_mask_like_model` accordingly, or ensure your `processed_shape` and `patch_size` match the actual preprocessing.
- **Class-aware detectors**: Set `supports_class_prediction = True` and populate `patch_class_scores` to enable class-wise patch and mask metrics.
- **Reproducibility**: The orchestrator calls `seed_all(seed + fold_idx)` at the start of each fold. Use the passed `seed` in `build_detector` for any randomized components.

---

This work uses the following ressources and datasets:
- [DINOv2](https://github.com/facebookresearch/dinov2), code and model available under Apache 2.0 license.
- The [MVTec-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), available under the CC BY-NC-SA 4.0 license.
- The [VisA dataset](https://github.com/amazon-science/spot-diff), available under the CC BY 4.0 license.

---

If you find this repository useful in your research/project, please consider citing the paper:

```
@inproceedings{damm2024anomalydino,
      title={AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2}, 
      author={Simon Damm and Mike Laszkiewicz and Johannes Lederer and Asja Fischer},
      booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV 2025)},
      year={2025},
      url={https://arxiv.org/abs/2405.14529}, 
}
```