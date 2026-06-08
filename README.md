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