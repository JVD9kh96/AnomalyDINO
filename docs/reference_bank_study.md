# Reference Composition and Contamination-Aware Memory Banks

Study of how reference composition affects DINOv2 patch-memory defect localization
(`AnomalyDINODetector`), and a lightweight method for mining normal patches from
defect-containing or unverified reference images.

**Primary question:** Can additional defect-containing images improve few-shot
anomaly localization when their likely anomalous patches are removed automatically?

**Primary detector:** existing `AnomalyDINODetector` only. Do not introduce a new
backbone or broad detector ensemble before this study completes.

---

## Operational phase status (0–12)

Status-aware tracker for the reference-composition campaign. Completed fold-0
artifacts are immutable inputs; do not regenerate Phase 0/3/4/5 evidence.

| Phase | Status | Code / evidence notes |
|------|--------|------------------------|
| 0 | COMPLETE | Paired manifests (`scripts/phase0_freeze_paired_inputs.py`) |
| 1 | CODE COMPLETE | `calibration_report.json` schema + study-runner emission + backfill CLI; GPU backfill from cached scores still needed on host |
| 2 | CODE COMPLETE | Multi-bank purification metrics + selected-index helpers; run analysis on fold-0 seed-42 pool on GPU host |
| 3 | COMPLETE | [`results/phase3/`](../results/phase3/) — see [`report.md`](report.md) |
| 4 | COMPLETE | [`results/phase4/`](../results/phase4/); frozen `fixed_ratio_trim` trim=0.20 — see [`report.md`](report.md) |
| 5 | COMPLETE (base rows) | [`results/phase5/`](../results/phase5/); additive naive/random20/oracle via `--append-rows` still pending |
| 6 | CODE COMPLETE | Replacement contamination + neighbor traces (`scripts/run_controlled_contamination_study.py`) |
| 7–10 | CODE COMPLETE | `anomaly_memory.py`, `dual_bank.py`, `scripts/run_anomaly_memory_study.py` (optional branch; stop/go on GPU) |
| 11 | CODE COMPLETE | Attention plan/runner reusing `dino_knn_rollout` (`scripts/run_attention_auxiliary_study.py`) |
| 12 | CODE COMPLETE | Frozen held-out harness (`scripts/run_heldout_maskfree_matrix.py` + bootstrap aggregation) |

**Frozen fold-0 primary setting:** `fixed_ratio_distance_trim` / `fixed_ratio_trim` with `trim_fraction=0.20`, exact budget `51200`, greedy coreset.

Canonical configs: [`configs/frozen_primary.yaml`](../configs/frozen_primary.yaml), [`configs/reference_bank/proposed_distance20.yaml`](../configs/reference_bank/proposed_distance20.yaml), [`src/evaluation/frozen_settings.py`](../src/evaluation/frozen_settings.py).

**Where it applies:** Phase-5 proposed rows, Phase-12 proposed arms, any paper “proposed” baseline.  
**Where it does not replace the factor:** clean/naive/oracle/random20 controls; Phase-6 clean-bank contamination (budget only); optional GT anomaly-memory / attention.
**Publication-critical path:** Phase 12 primary mask-free matrix on folds 1–4. Optional GT anomaly-memory and attention must not block it.

**GPU note:** This repository ships runners, schemas, and CPU unit tests. Feature extraction / fold experiments require a remote GPU host.

Completed fold-0 evidence lives under [`results/`](../results/) (phases 3–5). See [`docs/report.md`](report.md) for tabulated results. Remaining phases write under `results/` (preferred) or `results_refbank/`.

---

## Step-by-step GPU campaign commands (phases 0–12)

Run from the repo root on a machine with CUDA, dataset at `data/severstal`, and the Python env activated. Prefer `--skip-sam2` until patch thresholds are frozen. Do **not** regenerate immutable Phase 3/4/5 reports under `results/`.

### Prerequisites

```bash
cd /path/to/Steel
# optional: export CUDA_VISIBLE_DEVICES=0
python -c "import torch; assert torch.cuda.is_available()"
```

### P0 — make existing results interpretable

**Phase 0** (only if a needed paired manifest is missing; reuse `results/phase3/*_paired_manifest.json` and `results/phase5/*_manifest.json` when present):

```bash
python scripts/phase0_freeze_paired_inputs.py \
  --config configs/phase0_paired_reference_manifest.yaml \
  --fold 0 --seed 42 --clean-shots 2 --additional-shots 8 \
  --output results/phase0/phase0_fold0_seed42_paired_manifest.json
```

**Phase 1** — calibration audit (analysis-only backfill when score NPZs exist; otherwise emitted by new study runs):

```bash
# Backfill from any cached phase1_scores.npz under results/
python scripts/phase1_build_calibration_report.py \
  --config configs/phase1_fixed_threshold_calibration.yaml \
  --backfill-root results

# Or build one report from an explicit score bundle:
python scripts/phase1_build_calibration_report.py \
  --config configs/phase1_fixed_threshold_calibration.yaml \
  --scores results/phase5/phase5_f0_purified_budget_2plus8_s42/phase1_scores.npz \
  --output-dir results/phase1
```

**Phase 2** — overlap / purification quality (analysis-only; uses Phase 0/3 manifest):

```bash
python scripts/phase2_verify_oracle_purification.py \
  --config configs/phase2_oracle_purification_quality.yaml \
  --manifest results/phase3/phase3_fold0_seed42_paired_manifest.json \
  --banks naive oracle distance_trim_20 random_size_matched \
  --trim-fraction 0.20 --selected-oracle-rule any_overlap \
  --output-dir results/phase2
```

**Phase 5 additive rows** (do not rerun existing nine rows; merge into report):

```bash
# Full additive matrix in one job (server / long session — omit --only-names):
python scripts/phase5_memory_budget_controls.py \
  --output-dir results/phase5 \
  --phase4-report results/phase4/phase4_compact_purification_controls_report.json \
  --purification-mode fixed_ratio_trim --fixed-trim-fraction 0.20 \
  --append-rows --include-optional-4plus8 --run

# Kaggle: one heavy row per session, then re-aggregate without --run:
#   --only-names naive_greedy_budget_4plus8 --run
```

### Full server run (no 12h limit)

Omit `--only-names`, `--max-jobs`, `--skip-aggregate`, and single-seed filters.
Defaults already schedule the full matrix; `--resume` (where available) skips
completed `metrics.json` dirs if you restart.

```bash
# Phase 3 — all 5 seeds × 4 conditions, then report
python scripts/phase3_multiseed_fold0_replication.py \
  --output-dir results/phase3 --run --seeds 42,43,44,45,46 --device cuda:0

# Phase 4 — all compact purification controls
python scripts/phase4_compact_purification_controls.py \
  --output-dir results/phase4 --run --device cuda:0

# Phase 5 — full additive append (or drop --append-rows for the entire Phase 5 matrix)
python scripts/phase5_memory_budget_controls.py \
  --output-dir results/phase5 \
  --phase4-report results/phase4/phase4_compact_purification_controls_report.json \
  --purification-mode fixed_ratio_trim --fixed-trim-fraction 0.20 \
  --append-rows --include-optional-4plus8 --run

# Phase 6 — all contamination rates × compositions
python scripts/run_controlled_contamination_study.py \
  --config configs/phase6_controlled_contamination.yaml \
  --fold 0 --seed 42 --output-dir results/phase6 --device cuda:0 --resume

# Phase 12 — folds 1–4 × seeds 42–44 × all tracks
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 \
  --folds 1 2 3 4 --seeds 42 43 44 \
  --track all --run --resume --device cuda:0
```

### Kaggle 12h sharding (multi-seed / multi-row)

Heavy matrices exceed the 12h limit if launched whole. Prefer one seed, one
row, or `--max-jobs 1` per session; keep `--resume` / reuse of completed
`metrics.json` on.

**Phase 3** — one reference seed per session, then aggregate:

```bash
python scripts/phase3_multiseed_fold0_replication.py \
  --output-dir results/phase3 --run --seeds 42 --skip-aggregate --device cuda:0
# later: --seeds 43, then 44, 45, 46

python scripts/phase3_multiseed_fold0_replication.py \
  --output-dir results/phase3 --aggregate-seeds 42,43,44,45,46
```

**Phase 12** — one fold×seed×condition (or a small batch):

```bash
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 \
  --folds 1 --seeds 42 --only-names proposed_distance20_2plus8 \
  --track primary --run --resume --device cuda:0

# Or advance through the queue one incomplete job at a time:
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 \
  --folds 1 2 3 4 --seeds 42 43 44 \
  --track all --run --resume --max-jobs 1 --device cuda:0
```

**Phase 6** — one contamination condition / batch:

```bash
python scripts/run_controlled_contamination_study.py \
  --config configs/phase6_controlled_contamination.yaml \
  --fold 0 --seed 42 --output-dir results/phase6 \
  --only-names rate0.05_uniform --resume --device cuda:0
# or: --resume --max-jobs 1
```

### P1 — mechanism study + publication-critical held-out matrix

**Phase 6** — controlled contamination (replacement, constant 51,200 bank):

```bash
# Dry-run condition plan (CPU-safe)
python scripts/run_controlled_contamination_study.py \
  --config configs/phase6_controlled_contamination.yaml \
  --output-dir results/phase6 --dry-run

python scripts/run_controlled_contamination_study.py \
  --config configs/phase6_controlled_contamination.yaml \
  --fold 0 --seed 42 --output-dir results/phase6 --device cuda:0 --resume
```

**Phase 12** — frozen held-out primary mask-free matrix (folds 1–4, ≥3 seeds):

```bash
# Aggregate only (if runs already exist)
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 --resume

# Full GPU matrix (proposed arms load configs/reference_bank/proposed_distance20.yaml)
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 \
  --folds 1 2 3 4 --seeds 42 43 44 \
  --track all --run --resume --device cuda:0
```

After patch thresholds are frozen, SAM2 for selected conditions only:

```bash
python scripts/run_heldout_maskfree_matrix.py \
  --config configs/phase12_heldout_maskfree.yaml \
  --output-dir results/phase12 \
  --folds 1 2 3 4 --seeds 42 43 44 \
  --run --resume --run-sam2 --device cuda:0
```

Single proposed run (manual):

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/proposed_distance20.yaml \
  --fold 1 --seed 42 --condition fixed_ratio_trim \
  --clean-shots 2 --additional-shots 8 \
  --fixed-trim-fraction 0.20 --coreset-size 51200 --budget-policy greedy_coreset \
  --output-dir results/phase12/manual_proposed_f1_s42 --skip-sam2
```

Frozen settings (do not retune on held-out folds): `fixed_ratio_trim`, `trim_fraction=0.20`, budget `51200`, `greedy_coreset`.

### P2 — optional anomaly-memory branch (must not block Phase 12)

```bash
# Fail-closed without the flag; enable explicitly:
python scripts/run_anomaly_memory_study.py \
  --config configs/phase8_anomaly_memory_study.yaml \
  --allow-gt-anomaly-memory --stage all --fold 0 --seed 42 \
  --output-dir results/anomaly_memory --dry-run

python scripts/run_anomaly_memory_study.py \
  --config configs/phase8_anomaly_memory_study.yaml \
  --allow-gt-anomaly-memory --stage all --fold 0 --seed 42 \
  --output-dir results/anomaly_memory
```

Include in the paper only if Phase-10 stop/go passes (seen-class gains without material unseen-class regression).

### P3 — attention auxiliary (after thresholds frozen)

```bash
python scripts/run_attention_auxiliary_study.py \
  --config configs/phase11_attention_auxiliary.yaml \
  --output-dir results/phase11 --fold 0 --seed 42

# GPU beta sweep (writes sidecars; enable three-signal only if both simpler additions help):
python scripts/run_attention_auxiliary_study.py \
  --config configs/phase11_attention_auxiliary.yaml \
  --output-dir results/phase11 --fold 0 --seed 42 --run
```

### Immutable completed phases (do not regenerate)

| Phase | Artifact under `results/` |
|------|---------------------------|
| 3 | `phase3/phase3_multiseed_fold0_replication_report.json` |
| 4 | `phase4/phase4_compact_purification_controls_report.json` |
| 5 | `phase5/phase5_exact_memory_budget_controls_report.json` |

---

## Experimental settings (reference modes)

| Mode | Config key | Setting type | GT masks in fitting? |
|------|------------|--------------|----------------------|
| Clean normal bank | `clean` | Clean one-class few-shot baseline | No |
| Naive contaminated | `contaminated_all` | Contaminated-reference | No |
| Class-balanced all patches | `class_balanced_all` | Weakly supervised (class labels) | No |
| Oracle purified | `oracle_purified` | Oracle mask-filtered upper bound | Yes |
| Auto purified | `auto_purified` | Proposed deployable method | No |

### Clean one-class setting (`clean`)

Use only patches from defect-free images (`clean_shots`). Standard one-class
few-shot anomaly-detection baseline.

### Contaminated-reference setting (`contaminated_all`)

Clean seed set plus **all** patches from additional images that may contain
defects. Tests uncontrolled contamination.

### Weakly supervised class-balanced setting (`class_balanced_all`)

Reference images sampled evenly across defect classes; **all** patches enter the
memory bank. This is a defect-enriched, weakly supervised stress test — **not**
an unsupervised anomaly-detection baseline.

### Oracle mask-filtered setting (`oracle_purified`)

Defect-containing references with patches overlapping GT defect masks excluded.
**Analysis upper bound only** — not the deployable proposed method. Requires
`allow_oracle_reference_filtering: true`.

### Automatically purified setting (`auto_purified`)

Clean seed bank identifies and retains normal-looking patches from additional
defect-containing or unverified images. This is the proposed method.

---

## Phase 1 audit: current reference handling (pre-study)

### Defect-free selection

`SeverstalDataset._select_defect_free_reference_ids`:

- Pool: train-fold IDs with no non-empty RLE in `train.csv`.
- `shots > 0`: deterministic wrap-around selection from the sorted pool
  (start index `(seed * shots) % len(pool)`), so large fold seeds still return
  references. Legacy contiguous `seed * shots` slicing could undershoot for large seeds.
- `shots == -1`: all defect-free train IDs; `shots == 0`: empty.

### Class-balanced selection

`SeverstalDataset._select_class_balanced_reference_ids`:

- `shots` must be divisible by `num_classes` (4).
- `per_class = shots // num_classes` images per class via deterministic circular walk.
- Selected images are **defective** by construction.

### Do all patches enter the memory bank?

**Default yes.** `AnomalyDINODetector.fit` concatenates every patch from each
reference (and optional rotations). Optional reductions: PCA background mask
(`masking` + `mask_ref_images`) and `greedy_coreset`. **No GT filtering** existed
before this study.

### Are GT masks available during fitting?

Yes on `SeverstalSample.masks_by_class` (loaded in CV). AnomalyDINO historically
ignored them. Oracle mode now uses them when explicitly enabled.

### Patch coordinates ↔ native masks

Shared geometry in `src/severstal/transforms.py`:

1. Resize smaller edge → `resolution`, crop to patch multiple.
2. `mask_to_patch_overlap` → mean defective pixels per cell.
3. Label if overlap ≥ `gt_overlap_threshold` (default 0.5).

### Feature caching (before this study)

**None.** Features were re-extracted on every fit. The study adds in-memory
`ReferenceFeatureGrid` caching so LOO calibration and purification reuse one
DINO pass per reference image.

---

## Configuration example

```yaml
detector:
  name: anomaly_dino
  reference_mode: auto_purified
  clean_shots: 2
  additional_shots: 8
  additional_sampling: class_balanced  # class_balanced | random_train | mixed
  allow_oracle_reference_filtering: false
  use_dual_bank: false
  dual_bank_alpha: 1.0
  defect_mining_percentile: 99.5
  reference_purification:
    normal_acceptance_percentile: 99.0
    spatial_cleanup: false
    min_rejected_component_patches: 2
```

Legacy path (unchanged): omit `reference_mode` and use `shots` + `reference_sampling`.

**Calibration note:** with `clean_shots >= 2`, leave-one-**image**-out; with
`clean_shots == 1`, leave-one-**patch**-out within that image.

Acceptance rule: candidate patch kept iff `distance <= calibration.percentile_p`
(higher distance = less normal).

---

## Kaggle experiment runbook

Do **not** treat oracle as the deployable method. Freeze fold-0 choices before
evaluating folds 1–4.

### 1. Sanity — clean vs naive contaminated (fold 0)

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/clean.yaml --fold 0 --seed 42 --condition clean \
  --clean-shots 2 --additional-shots 8 --output-dir results_refbank/clean_f0

python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/contaminated_all.yaml --fold 0 --seed 42 \
  --condition contaminated_all --clean-shots 2 --additional-shots 8 \
  --output-dir results_refbank/contam_f0
```

Expect: contamination measurably hurts clean-bank AUPRC / F1.

### 2. Oracle upper bound (fold 0)

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/oracle_purified.yaml --fold 0 --seed 42 \
  --condition oracle_purified --clean-shots 2 --additional-shots 8 \
  --output-dir results_refbank/oracle_f0
```

### 3. Auto purify (fold 0)

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/auto_purified.yaml --fold 0 --seed 42 \
  --condition auto_purified --clean-shots 2 --additional-shots 8 \
  --output-dir results_refbank/auto_f0
```

### 4. Fold-0 ablations (freeze after this)

Acceptance percentiles:

```bash
for p in 95 97.5 99 99.5; do
  python scripts/run_reference_composition_study.py \
    --config configs/reference_bank/auto_purified.yaml --fold 0 --seed 42 \
    --condition auto_purified --clean-shots 2 --additional-shots 8 \
    --acceptance-percentile $p \
    --output-dir results_refbank/ablate_pct_${p}
done
```

Clean seed × additional counts × candidate source:

```bash
for cs in 1 2 4 8; do
  for as in 4 8 16; do
    for src in class_balanced random_train mixed; do
      python scripts/run_reference_composition_study.py \
        --config configs/reference_bank/auto_purified.yaml --fold 0 --seed 42 \
        --condition auto_purified --clean-shots $cs --additional-shots $as \
        --additional-sampling $src \
        --output-dir results_refbank/ablate_cs${cs}_as${as}_${src}
    done
  done
done
```

Also run `class_balanced_all` (weakly supervised stress):

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/class_balanced_all.yaml --fold 0 --seed 42 \
  --condition class_balanced_all --clean-shots 0 --additional-shots 8 \
  --output-dir results_refbank/cb_all_f0
```

### 5. Held-out folds 1–4 × ≥3 seeds (frozen settings)

Replace `CLEAN`, `ADD`, `PCT`, `SRC` with fold-0 choices:

```bash
for fold in 1 2 3 4; do
  for seed in 42 43 44; do
    for cond in clean contaminated_all auto_purified oracle_purified class_balanced_all; do
      python scripts/run_reference_composition_study.py \
        --config configs/reference_bank/${cond}.yaml --fold $fold --seed $seed \
        --condition $cond --clean-shots CLEAN --additional-shots ADD \
        --acceptance-percentile PCT --additional-sampling SRC \
        --output-dir results_refbank/holdout_${cond}_f${fold}_s${seed}
    done
  done
done
```

### 6. Size-matched memory-bank control

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/clean.yaml --fold 0 --seed 42 \
  --condition size_matched_clean --clean-shots 2 --coreset-size N \
  --output-dir results_refbank/size_clean_N

python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/auto_purified.yaml --fold 0 --seed 42 \
  --condition size_matched_purified --clean-shots 2 --additional-shots 8 \
  --coreset-size N --output-dir results_refbank/size_auto_N
```

### 7. Synthetic contamination curve

```bash
python scripts/run_reference_composition_study.py \
  --config configs/reference_bank/clean.yaml --fold 0 --seed 42 \
  --condition synthetic_contamination --clean-shots 8 \
  --output-dir results_refbank/contam_curve
```

Writes `contamination_curve.csv`, `contamination_vs_auprc.png`,
`contamination_vs_f1.png`, `memory_bank_statistics.json`.

### 8. SAM2 downstream

Keep `segmenter` enabled in the YAML (default in reference_bank configs). Metrics
include SAM2 Dice / IoU in each run’s `metrics.json`.

### 9. Optional dual-bank (only if auto-purify is strong)

Set `use_dual_bank: true` in config or pass through a modified YAML; do not replace
auto-purification as the primary method unless held-out gains are clear.

### Aggregate results

```bash
python scripts/aggregate_reference_bank_results.py \
  --input-dir results_refbank --output results_refbank/summary_table.csv
```

---

## Main result table columns

| Method | Clean shots | Additional images | GT masks used in fitting | Patch AUPRC | Fixed F1 | F1-max | SAM2 Dice |
|--------|-------------|-------------------|--------------------------|-------------|----------|--------|-----------|
| Clean bank | … | 0 | No | | | | |
| Naive contaminated | … | … | No | | | | |
| Class-balanced all | 0 or specified | … | No (class labels) | | | | |
| Oracle purified | … | … | Yes | | | | |
| Auto purified | … | … | No | | | | |

Report globally and by defect class.

---

## Stop / go criteria

Continue with the proposed method when:

1. Naive contamination degrades clean-bank performance.
2. Oracle purification recovers a meaningful portion of that loss.
3. Automatic purification closes a substantial part of the clean-to-oracle gap.
4. Gains remain after size-matched memory-bank control.
5. Results are consistent on ≥3 held-out folds.

Useful success targets:

- Auto purified > clean-only by ~2–3 AUPRC or F1 points, **or**
- Auto purified ≈ clean-only while using far fewer clean reference images
  (e.g. 1 clean + mined normals ≈ 8-clean baseline).

**Oracle gap** = oracle AUPRC − auto AUPRC (room left for better purification).

---

## Execution order (code complete; run on GPU)

Operational campaign (preferred):

1. P0: Phase 1 calibration audit + Phase 2 purification analysis + Phase 5 `--append-rows`
2. P1: Phase 6 mechanism study + Phase 12 primary mask-free held-out matrix
3. P2: Phases 7–10 anomaly-memory branch (include in paper only if stop/go passes)
4. P3: Phase 11 attention + SAM2 after patch thresholds are frozen

Legacy checklist (implementation complete earlier; still useful for configs):

1. Audit reference handling — done (this doc)
2. Baseline configs under `configs/reference_bank/`
3. Reference metadata in fold results
4. Cached reference feature grids
5. Clean vs naive contaminated
6. LOO normal calibration
7. Automatic patch purification
8. Oracle upper bound
9. Fold-0 threshold / shot ablations
10. Freeze settings
11. Folds 1–4 × multiple seeds — use `scripts/run_heldout_maskfree_matrix.py`
12. Memory-size-matched controls
13. SAM2 downstream — gated behind Phase-12 `--run-sam2`
14. Dual-bank / anomaly-memory only if stop/go passes

Do **not** implement axis-conditioned subspaces before this study shows a strong,
consistent result.
