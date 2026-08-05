# Reference-composition study report

Tabulated results from completed fold-0 experiments under [`results/`](../results/).  
Code status and remaining GPU commands: [`reference_bank_study.md`](reference_bank_study.md).

**Protocol (completed phases):** fold 0, split seed 42, SAM2 skipped, DINOv2 ViT-S/14 @ 448px.  
**Frozen primary setting (from Phase 4):** `fixed_ratio_trim` with `trim_fraction=0.20`, then Phase-5 exact budget `51_200` via greedy coreset.

### Does the recommended setting apply to upcoming experiments?

**Yes — for the deployable / “proposed” method and Phase-12 proposed arms.**  
**No — as a replacement for control arms or the Phase-6 mechanism study.**

| Upcoming work | Apply freeze? | What is frozen |
|---------------|:-------------:|----------------|
| Phase 5 additive `purified_*` / proposed rows | Yes | trim 0.20 + budget 51,200 greedy |
| Phase 12 `proposed_distance20_*` / efficiency proposed | Yes | same (enforced in runner) |
| Phase 12 clean / naive / random20 / oracle | Partial | only shared **budget 51,200** where exact-budget; filter differs by design |
| Phase 6 controlled contamination | Partial | **budget 51,200 only**; bank stays clean (no distance trim) |
| Phases 7–11 optional extensions | No (own grids) | compare against a normal bank built with the freeze when reporting safety |

Canonical copies:
- [`configs/frozen_primary.yaml`](../configs/frozen_primary.yaml)
- [`configs/reference_bank/proposed_distance20.yaml`](../configs/reference_bank/proposed_distance20.yaml)
- [`src/evaluation/frozen_settings.py`](../src/evaluation/frozen_settings.py)

| Phase | Status in `results/` | Report file |
|------|----------------------|-------------|
| 0 | Manifests embedded in phase3/5 | paired `*_manifest.json` |
| 1–2 | Not yet filled as standalone reports | run P0 commands on GPU host |
| 3 | Complete | `results/phase3/phase3_multiseed_fold0_replication_report.json` |
| 4 | Complete | `results/phase4/phase4_compact_purification_controls_report.json` |
| 5 | Complete (base 9 rows; additive naive/random20/oracle not yet appended) | `results/phase5/phase5_exact_memory_budget_controls_report.json` |
| 6–12 | Not run | harnesses ready |

---

## Phase 3 — multiseed fold-0 replication

**Design:** 5 paired seeds `{42…46}` × 4 conditions (`clean`, `contaminated_all`, `auto_purified`, `oracle_purified`) = 20 runs. Clean shots = 2, additional = 8. Paired clean/additional IDs validated per seed.

### Aggregate metrics (mean ± std over seeds)

| Condition | AUPRC | AUROC | F1-max | Fixed F1 | Bank size |
|-----------|------:|------:|-------:|---------:|----------:|
| clean | 0.102 ± 0.017 | 0.784 ± 0.030 | 0.189 ± 0.027 | 0.098 ± 0.099 | 12,800 |
| contaminated_all (naive) | 0.101 ± 0.011 | 0.796 ± 0.020 | 0.198 ± 0.017 | 0.110 ± 0.025 | 64,000 |
| auto_purified | 0.103 ± 0.012 | 0.797 ± 0.020 | 0.200 ± 0.019 | 0.027 ± 0.037 | 63,204 ± 1,088 |
| oracle_purified | 0.110 ± 0.011 | 0.804 ± 0.020 | 0.211 ± 0.017 | 0.142 ± 0.021 | 60,595 ± 971 |

### Per-seed AUPRC

| Seed | clean | naive | auto | oracle |
|-----:|------:|------:|-----:|-------:|
| 42 | 0.102 | 0.095 | 0.098 | 0.107 |
| 43 | 0.128 | 0.108 | 0.115 | 0.119 |
| 44 | 0.088 | 0.112 | 0.112 | 0.118 |
| 45 | 0.107 | 0.104 | 0.104 | 0.111 |
| 46 | 0.085 | 0.084 | 0.084 | 0.092 |

### Paired deltas (mean over seeds)

| Contrast | Δ AUPRC | Sign (AUPRC) | Δ Fixed F1 | Sign (Fixed F1) |
|----------|--------:|--------------|-----------:|-----------------|
| naive − clean | −0.0011 | 4 neg / 1 pos | +0.012 | 3 pos / 2 neg |
| auto − naive | +0.0018 | 4 pos / 1 zero | −0.084 | 5 neg |

**Read-out:** Ranking metrics (AUPRC/AUROC/F1-max) show a small oracle advantage and a tiny auto-vs-naive lift. Fixed-threshold F1 is unstable and collapses under auto purification relative to naive — consistent with Phase-1 notes that fixed operating points need final-bank `tau_query` revalidation before interpreting absolute Fixed F1.

---

## Phase 4 — compact purification controls (fold 0, seed 42)

**Baseline clean (2-shot) Fixed F1** used for deltas: 0.181 (from paired clean metrics).  
**Selection rule:** rank non-random settings by AUPRC ↓, then smaller `|Δ query threshold|` vs clean, then Fixed F1 ↓. F1-max is reported only.

### Recommended setting (frozen for later phases)

| Field | Value |
|-------|------:|
| Condition | `fixed_ratio_trim` |
| Trim fraction | **0.20** |
| AUPRC | **0.1084** |
| AUROC | 0.8132 |
| F1-max | 0.2151 |
| Fixed F1 | 0.0620 |
| Candidates retained | 40,960 / 51,200 (80%) |
| Query threshold | 0.4095 (Δ vs clean = 0) |

This setting is now the default for proposed runs via `configs/reference_bank/proposed_distance20.yaml` and is enforced by `scripts/run_heldout_maskfree_matrix.py` (rejects retuned trim/budget).

### All controls

| Name | Filter | Retain % | AUPRC | AUROC | F1-max | Fixed F1 | Δ Fixed F1 vs clean | Rank |
|------|--------|--------:|------:|------:|-------:|---------:|--------------------:|-----:|
| trim_20pct | distance trim 20% | 80.0 | **0.1084** | 0.8132 | 0.2151 | 0.0620 | −0.119 | **1** |
| trim_10pct | distance trim 10% | 90.0 | 0.1022 | 0.8062 | 0.2039 | 0.0479 | −0.133 | 2 |
| auto_p95 | auto @ 95th | 92.2 | 0.1007 | 0.8045 | 0.2015 | 0.0702 | −0.111 | 3 |
| auto_p97_5 | auto @ 97.5th | 94.3 | 0.0991 | 0.8026 | 0.1990 | 0.0512 | −0.130 | 4 |
| trim_5pct | distance trim 5% | 95.0 | 0.0987 | 0.8020 | 0.1986 | 0.0391 | −0.142 | 5 |
| auto_p99 | auto @ 99th | 95.9 | 0.0980 | 0.8014 | 0.1976 | 0.0348 | −0.146 | 6 |
| auto_p99_5 | auto @ 99.5th | 96.7 | 0.0975 | 0.8006 | 0.1969 | 0.0272 | −0.154 | 7 |
| random_matched_p95 | random size-matched | 92.2 | 0.0951 | 0.7971 | 0.1935 | 0.0513 | −0.130 | — |
| random_matched_p97_5 | random size-matched | 94.3 | 0.0951 | 0.7971 | 0.1935 | 0.0371 | −0.144 | — |
| random_matched_p99 | random size-matched | 95.9 | 0.0951 | 0.7970 | 0.1934 | 0.0269 | −0.154 | — |
| random_matched_p99_5 | random size-matched | 96.7 | 0.0951 | 0.7970 | 0.1934 | 0.0215 | −0.160 | — |

**Read-out:** Fixed-ratio 20% distance trim beats auto percentile settings on AUPRC and beats all random size-matched controls. This setting is frozen for Phase 5 / Phase 12.

---

## Phase 5 — exact memory-budget controls (fold 0, seed 42)

**Target budget:** 51,200 patches (= 8 clean × 6,400).  
**Selected purification:** Phase-4 `fixed_ratio_trim` @ 0.20.

### Clean-shot ladder

| Row | Clean | Add | Final bank | Exact 51,200? | AUPRC | AUROC | F1-max | Fixed F1 |
|-----|------:|----:|-----------:|:-------------:|------:|------:|-------:|---------:|
| clean_1 | 1 | 0 | 6,400 | no (scarce) | 0.070 | 0.703 | 0.135 | 0.062 |
| clean_2 | 2 | 0 | 12,800 | no | 0.102 | 0.818 | 0.191 | 0.181 |
| clean_4 | 4 | 0 | 25,600 | no | 0.105 | 0.813 | 0.200 | 0.177 |
| clean_8 | 8 | 0 | 51,200 | **yes** | 0.103 | 0.795 | 0.199 | 0.147 |

### Expansion / purification at or near budget

| Row | Filter | Final bank | Exact? | AUPRC | AUROC | F1-max | Fixed F1 | Cand. after filter |
|-----|--------|-----------:|:------:|------:|------:|-------:|---------:|-------------------:|
| expanded_full_2plus8 | none (full bank) | 64,000 | n/a | 0.095 | 0.797 | 0.193 | 0.078 | 51,200 |
| expanded_random_budget_2plus8 | none + random coreset | 51,200 | **yes** | 0.095 | 0.797 | 0.193 | 0.080 | 51,200 |
| purified_budget_1plus8 | distance20 | 47,360 | no (scarcity) | 0.091 | 0.766 | 0.181 | 0.065 | 40,960 |
| purified_budget_2plus8 | distance20 + greedy | 51,200 | **yes** | **0.108** | 0.813 | **0.215** | 0.062 | 40,960 |
| purified_budget_4plus8 | distance20 + greedy | 51,200 | **yes** | **0.114** | 0.815 | **0.218** | 0.136 | 40,960 |

### Head-to-head at exact 51,200 (2+8 family)

| Method | AUPRC | vs clean_8 AUPRC | vs clean_2 AUPRC |
|--------|------:|-----------------:|-----------------:|
| clean_8 | 0.1030 | — | +0.001 |
| expanded_random_budget_2plus8 (naive@budget) | 0.0950 | −0.008 | −0.007 |
| purified_budget_2plus8 (proposed@budget) | **0.1084** | **+0.005** | **+0.007** |

**Read-out:** At matched budget, proposed 20% distance trim + greedy coreset (2 clean + 8 candidates) beats both clean-8 and naive expansion. 4+8 purified is strongest AUPRC so far (0.114). 1+8 remains under-budget (47,360) and should be reported as scarcity, not exact-budget.

**Still missing in this artifact (code ready via `--append-rows`):**  
`naive_greedy_budget_2plus8`, `random20_greedy_budget_2plus8`, `oracle_greedy_budget_2plus8`, optional `naive_greedy_budget_4plus8`.

---

## Interim conclusions (fold 0 only)

1. **Oracle** is the ranking upper bound (Phase 3 AUPRC 0.110).
2. **Naive contamination** does not clearly destroy AUPRC vs clean on average, but bank size explodes; budget-matched naive underperforms clean-8 (Phase 5).
3. **Auto percentile purification** is weak vs **fixed 20% distance trim** on fold-0 seed 42 (Phase 4).
4. **Proposed setting** (`fixed_ratio_trim` 0.20 + greedy 51,200) is the frozen primary for held-out evaluation.
5. **Fixed-threshold F1** remains fragile across seeds/modes — complete Phase 1 final-bank calibration audit before claiming Fixed F1 gains.

---

## Next experiments (not in `results/` yet)

1. Phase 5 `--append-rows` policy-matched controls.  
2. Phase 6 replacement contamination curve.  
3. Phase 12 primary mask-free matrix on folds 1–4 × seeds 42/43/44.  
4. Optional Phases 7–11 only if schedule allows and stop/go passes.

Command copy-paste: see **Step-by-step GPU campaign commands** in [`reference_bank_study.md`](reference_bank_study.md).
