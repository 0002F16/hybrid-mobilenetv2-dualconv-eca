# Analysis & Visualization PRD
## Hybrid MobileNetV2 (DualConv + ECA) — Phase 10–12 Implementation Spec

**Scope**: This document is the implementation-ready PRD for Phases 10–12 of the thesis experiment pipeline:
data aggregation, efficiency measurement, statistical analysis, and all thesis-ready tables and figures.
It is written for Cursor / coding agents. All analysis must be reproducible from saved artifacts only.

**Companion document**: `implementation_spec.md` (architecture + training PRD).

---

## 0. Ground Truth: What Data Exists

Before implementing anything, the analysis system must parse and validate the actual artifact layout.

### 0.1 Experiment Matrix Status

Trained models are stored under `Trained Models/` with the following confirmed structure:

```
Trained Models/
├── CIFAR10/content/outputs/cifar100/          ← CIFAR-100 runs (stored inside CIFAR10 Colab session)
│   ├── baseline/  seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
│   ├── dualconv/  seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
│   ├── eca/       seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
│   └── hybrid/    seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
├── CIFAR100/content/outputs/cifar100/         ← Duplicate CIFAR-100 run set (same data, second Colab session)
│   └── [same 4 variants × 5 seeds]
└── TInyImageNet/content/outputs/tiny_imagenet/
    ├── baseline/  seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
    ├── dualconv/  seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
    ├── eca/       seed_{42,123,777,2024,3407}/metrics.json + logs/epochs.jsonl
    └── hybrid/    seed_{3407}/metrics.json + logs/epochs.jsonl    ← only 1 seed
```

**Exclude** from analysis: `_smoke_metrics/`, `_test_metrics/`, `_tmp_runs/`, `_tmp_runs_scheduler_none/`, `_tmp_smoke/` — these are debug/sanity runs.

**Deduplication rule**: When the same `(dataset, model, seed)` triple appears in both `CIFAR10/` and `CIFAR100/` sessions, treat them as the canonical same run. Verify they are byte-equal and keep one. If they differ, flag the discrepancy and use the `CIFAR100/` session as the authoritative source.

### 0.2 Per-Run Artifact Schema

**`metrics.json`** (final run summary):
```json
{
  "dataset": "cifar100",
  "model": "hybrid",
  "seed": 42,
  "model_profile": {
    "params": 2184881,
    "flops": 20520736,
    "macs": 10260368,
    "size_mb": 8.518371,
    "input_size_chw": [3, 32, 32],
    "flops_method": "fvcore"
  },
  "best_val": { "epoch": 38, "val_acc": 0.3482, "val_loss": 2.449 },
  "test": { "acc": 0.3472, "loss": 2.471, "top1_pp": 34.72, "top5_pp": 68.53 },
  "stopped_epoch": 58
}
```

**`logs/epochs.jsonl`** (one JSON object per line, per epoch):
```json
{"epoch": 1, "train_loss": 4.209, "val_acc": 0.0962, "val_loss": 3.822, "val_top1_pp": 9.62, "lr": 0.09999, "is_best": true, "early_stop_counter": 0}
```

**`logs/config.json`** (training hyperparameters):
```json
{
  "batch_size": 64, "dataset": "cifar100", "epochs": 200, "learning_rate": 0.1,
  "momentum": 0.9, "scheduler": "cosine", "seed": 2024,
  "early_stopping": {"enabled": true, "min_delta_pp": 0.1, "patience_epochs": 20, "warmup_epochs": 30},
  "weight_decay": 0.0005, "width_multiplier": 1.0
}
```

### 0.3 Known Data Issues to Handle

- **Tiny ImageNet hybrid**: only 1 seed (3407) is present. All per-variant aggregations must handle `n=1` gracefully (no std, no Wilcoxon, mark as "n=1" in tables).
- **CIFAR-100 seed bimodality**: some seeds triggered early stopping at ~50–70 epochs with ~31–35% test accuracy; others ran to ~195 epochs with ~59–63% accuracy. This bimodal distribution is a **real experimental finding**, not an outlier to remove. Report it explicitly.
- **CIFAR-10**: no full training runs exist (only smoke/debug). Do not produce CIFAR-10 accuracy tables.
- The `stopped_epoch` field is `null` when training ran to the epoch ceiling.

---

## 1. Data Aggregation Layer

### 1.1 Module: `analysis/data_loader.py`

**Purpose**: Scan the `Trained Models/` directory tree, parse all valid `metrics.json` and `epochs.jsonl` files, and return a deduplicated, validated set of records.

**Function**: `load_all_runs(trained_models_root: str) -> tuple[pd.DataFrame, pd.DataFrame]`

Returns:
- `runs_df`: one row per `(dataset, model, seed)`, columns below
- `epochs_df`: one row per `(dataset, model, seed, epoch)`, columns below

**`runs_df` columns**:
| Column | Type | Source |
|---|---|---|
| `dataset` | str | metrics.json |
| `model` | str | metrics.json — canonical: `baseline`, `dualconv`, `eca`, `hybrid` |
| `seed` | int | metrics.json |
| `test_top1` | float | `test.acc × 100` (percentage) |
| `test_top5` | float | `test.top5_pp` |
| `best_val_acc` | float | `best_val.val_acc × 100` |
| `best_epoch` | int | `best_val.epoch` |
| `stopped_epoch` | int or None | null → ran to ceiling |
| `early_stopped` | bool | `stopped_epoch is not None` |
| `params` | int | `model_profile.params` |
| `flops` | int | `model_profile.flops` |
| `macs` | int | `model_profile.macs` |
| `size_mb` | float | `model_profile.size_mb` |
| `source_path` | str | path to metrics.json |

**`epochs_df` columns**:
| Column | Type | Source |
|---|---|---|
| `dataset` | str | — |
| `model` | str | — |
| `seed` | int | — |
| `epoch` | int | epochs.jsonl |
| `train_loss` | float | — |
| `val_acc` | float | `val_acc × 100` |
| `val_loss` | float | — |
| `lr` | float | — |
| `is_best` | bool | — |
| `early_stop_counter` | int | — |

**Deduplication logic**:
1. Group by `(dataset, model, seed)`.
2. If multiple source paths exist, verify `test_top1` is identical across them (tolerance ±0.001). If identical, keep one. If not, log a warning and keep the `CIFAR100/` session version.

**Validation assertions** (fail loudly):
- All `model` values are in `{baseline, dualconv, eca, hybrid}`.
- All `dataset` values are in `{cifar100, tiny_imagenet}` after excluding debug paths.
- `test_top1` is in `[0, 100]`.
- `params > 0`, `flops > 0`.
- No duplicate `(dataset, model, seed)` rows after deduplication.

---

## 2. Model Efficiency Analysis

### 2.1 Efficiency Summary Table

**Module**: `analysis/efficiency.py`
**Function**: `build_efficiency_table(runs_df) -> pd.DataFrame`

Compute per-variant (aggregated across seeds and datasets, since model architecture is dataset-independent):

For each `model` in `{baseline, dualconv, eca, hybrid}`:
- `params` (integer, pick the modal value across seeds — should be constant per model)
- `params_M` (params / 1e6, 2 decimal places)
- `flops` (integer)
- `flops_M` (flops / 1e6, 2 decimal places)
- `macs_M` (macs / 1e6, 2 decimal places)
- `size_mb` (float)
- `delta_params_pct` (% change vs baseline)
- `delta_flops_pct` (% change vs baseline)
- `delta_size_pct` (% change vs baseline)
- `within_budget` (bool: |delta_params_pct| ≤ 10 AND |delta_flops_pct| ≤ 10)

**Note on hybrid FLOPs**: The hybrid variant has `params ≈ 2,184,881` and `flops ≈ 20,520,736` for CIFAR-100 32×32 inputs, representing approximately −7.5% params and −19.7% FLOPs vs baseline. This is counterintuitive (adding ECA reduces params/FLOPs vs baseline) because DualConv's grouped convolution is more efficient than the standard depthwise. Report this explicitly.

**Output table** (markdown + CSV):
```
| Model    | Params (M) | FLOPs (M) | MACs (M) | Size (MB) | ΔParams% | ΔFLOPs% | Budget OK |
|----------|-----------|----------|---------|---------|---------|--------|-----------|
| baseline | 2.35       | 25.67     | 12.84    | 9.20     | —        | —       | ✓         |
| dualconv | 2.18       | 20.51     | 10.25    | 8.51     | −7.4%    | −20.2%  | ✓         |
| eca      | 2.35       | 25.68     | 12.84    | 9.21     | +0.0%    | +0.0%   | ✓         |
| hybrid   | 2.18       | 20.52     | 10.26    | 8.52     | −7.4%    | −20.1%  | ✓         |
```

### 2.2 Efficiency Bar Charts

**Figure: `figures/efficiency_bars.png`**

Four grouped bar charts in a 2×2 layout:
- Top-left: Params (M) by variant
- Top-right: FLOPs (M) by variant
- Bottom-left: MACs (M) by variant
- Bottom-right: Model size (MB) by variant

Style:
- One bar per variant, consistent color palette across all figures:
  - `baseline` → gray (`#6B7280`)
  - `dualconv` → blue (`#3B82F6`)
  - `eca` → green (`#10B981`)
  - `hybrid` → orange (`#F59E0B`)
- Annotate each bar with the value and the delta-% vs baseline (except baseline itself)
- Horizontal dashed lines at the ±10% budget thresholds relative to baseline
- Figure title: "Model Efficiency Profile (32×32 input)"

---

## 3. Predictive Performance Analysis

### 3.1 Module: `analysis/performance.py`
**Function**: `build_performance_summary(runs_df) -> pd.DataFrame`

For each `(dataset, model)` combination:
- `n_seeds`: count of seeds
- `test_top1_mean`: mean of `test_top1` across seeds
- `test_top1_std`: std of `test_top1`
- `test_top1_min`, `test_top1_max`
- `test_top5_mean`, `test_top5_std` (for cifar100 and tiny_imagenet only)
- `best_epoch_mean`, `best_epoch_std`
- `early_stopped_count`: how many seeds triggered early stopping
- `early_stopped_frac`: `early_stopped_count / n_seeds`

### 3.2 Master Performance Table

**File**: `results/tables/performance_summary.csv` and `results/tables/performance_summary.md`

Columns: `dataset`, `model`, `n`, `top1_mean ± std`, `top5_mean ± std`, `early_stopped`

Format for `top1_mean ± std` cells: `"59.91 ± 14.23"` (2 decimal places).

**Important**: CIFAR-100 will show high std (~14–15pp) due to the bimodal seed distribution (seeds that converge vs. stop early). Do NOT omit the std — it is a thesis finding.

### 3.3 Per-Dataset Performance Bar Charts

**Figures**: `figures/top1_bars_cifar100.png`, `figures/top1_bars_tiny_imagenet.png`

For each dataset:
- Grouped bar chart of mean Test Top-1 (%) by model
- Error bars = ±1 standard deviation across seeds
- Annotate bars with mean value
- Y-axis: start from `max(0, global_min - 5)` to `min(100, global_max + 5)`
- Include a horizontal dashed line at the baseline mean for reference
- Caption note: for CIFAR-100, state that high std reflects bimodal convergence (see Section 5)

### 3.4 Per-Dataset Top-5 Bar Charts

**Figure**: `figures/top5_bars_cifar100.png`, `figures/top5_bars_tiny_imagenet.png`

Same structure as Top-1 charts but for Top-5 accuracy.

### 3.5 Violin / Box Plots of Seed Distribution

**Figures**: `figures/top1_violin_cifar100.png`, `figures/top1_violin_tiny_imagenet.png`

For each dataset:
- Violin plot (or box + strip plot if n=5 per group) of `test_top1` values per model
- Overlay individual seed points as dots, jittered
- This makes the bimodal distribution in CIFAR-100 visually explicit
- Color code dots by seed value (5 fixed colors for the 5 seeds)

### 3.6 Per-Seed Scatter Plot

**Figure**: `figures/per_seed_scatter_cifar100.png`

X-axis: seed (categorical: 42, 123, 777, 2024, 3407)
Y-axis: test_top1 (%)
One line per model variant (4 lines)
Shows how each seed performs across models simultaneously — reveals correlated convergence failures.

---

## 4. Accuracy–Efficiency Trade-off Analysis

### 4.1 Accuracy vs FLOPs Scatter

**Figure**: `figures/pareto_flops_cifar100.png`, `figures/pareto_flops_tiny_imagenet.png`

For each dataset:
- X-axis: FLOPs (M)
- Y-axis: Test Top-1 mean (%)
- One point per model variant
- Error bars: ±1 std in Y direction
- Label each point with model name
- Draw the Pareto frontier (lower-left dominated points are non-Pareto; highlight Pareto-efficient points with a star or bold outline)
- Annotate which models are Pareto-efficient in accuracy vs. compute

### 4.2 Accuracy vs Params Scatter

**Figure**: `figures/pareto_params_cifar100.png`, `figures/pareto_params_tiny_imagenet.png`

Same as 4.1 but X-axis = Params (M).

### 4.3 Accuracy vs Model Size Scatter

**Figure**: `figures/pareto_size_cifar100.png`, `figures/pareto_size_tiny_imagenet.png`

Same as 4.1 but X-axis = Model size (MB).

### 4.4 Combined Efficiency–Accuracy Summary Table

**File**: `results/tables/efficiency_accuracy.md`

| Model | Params (M) | FLOPs (M) | CIFAR-100 Top-1 | TinyImageNet Top-1 | Pareto (CIFAR-100) | Pareto (TinyIN) |
|-------|-----------|----------|----------------|-------------------|------------------|----|
| ...   |           |          |                |                   |                  |    |

Pareto = "Yes" if a model is not dominated on both accuracy AND FLOPs axes simultaneously.

---

## 5. Training Dynamics Analysis

### 5.1 Module: `analysis/dynamics.py`

**Function**: `build_learning_curves(epochs_df) -> dict`

For each `(dataset, model)`:
- Aggregate `train_loss` and `val_acc` across seeds:
  - At each epoch, compute mean ± std across seeds that have reached that epoch
  - Handle ragged sequences (early-stopped runs have fewer epochs) using only seeds present at each epoch, and mark the divergence point

**Function**: `detect_convergence_groups(runs_df) -> pd.DataFrame`

For CIFAR-100 specifically:
- Classify each run as `converged` (test_top1 > 50%) or `not_converged` (test_top1 ≤ 50%)
- Report per-model: count of converged vs. not-converged, which seeds are in each group
- Compute mean metrics separately for each group

### 5.2 Learning Curve Plots

**Figures**: `figures/learning_curves_val_acc_cifar100.png`, `figures/learning_curves_val_acc_tiny_imagenet.png`

For each dataset:
- 2×2 subplot grid (one per model variant)
- Each subplot: X = epoch, Y = val_acc mean (%), with ±1 std shaded band
- Color: same variant-consistent palette
- Mark the mean best_epoch with a vertical dashed line
- For CIFAR-100: separate the shaded bands for converged vs. non-converged seed groups to avoid misleading averages. Plot two separate mean lines per subplot (converged seeds, non-converged seeds) in lighter/darker shade of the variant color.

**Figure**: `figures/learning_curves_train_loss_cifar100.png`, `figures/learning_curves_train_loss_tiny_imagenet.png`

Same structure as above but Y = train_loss. Do not split by convergence group (train loss is informative regardless).

**Figure**: `figures/learning_curves_combined_cifar100.png`

Combined 4-variant overlay (not subplots) — one chart with 4 mean val_acc lines, one per variant, for a high-level visual comparison.

### 5.3 Early Stopping Analysis Table

**File**: `results/tables/early_stopping_analysis.md`

For CIFAR-100:
| Model | Seeds Stopped Early | Stopped at Epoch | Seeds Converged | Converged Test Top-1 (mean) | Not-Converged Test Top-1 (mean) |
|-------|--------------------|-----------------|-----------------|-----------------------------|--------------------------------|

This table is a direct explanation of the high variance in CIFAR-100 results and must appear in the thesis.

### 5.4 Best Epoch Distribution

**Figure**: `figures/best_epoch_dist.png`

Scatter or strip plot:
- X-axis: model variant
- Y-axis: best_epoch value
- One dot per seed per model per dataset (two datasets side by side)
- Reveals whether different variants converge at different points in training

---

## 6. Statistical Analysis

### 6.1 Module: `analysis/stats.py`

Dependencies: `scipy`, `numpy`

**Function**: `paired_wilcoxon(runs_df, dataset, variant_a, variant_b, metric='test_top1') -> dict`

- Filter `runs_df` to `(dataset, model in {variant_a, variant_b})` and join on `seed`
- Only use seeds present in both variants (paired)
- Run `scipy.stats.wilcoxon(a_scores, b_scores, alternative='two-sided')`
- Return: `{'stat': float, 'p_value': float, 'n_pairs': int, 'mean_diff': float, 'median_diff': float}`

**Function**: `holm_bonferroni(p_values: list[float], alpha=0.05) -> list[float]`

Standard Holm-Bonferroni correction. Return corrected p-values in same order as input.

**Function**: `bootstrap_ci(a_scores, b_scores, n_bootstrap=10000, ci=0.95, seed=42) -> tuple[float, float]`

- Compute paired differences `d = a - b`
- Bootstrap resample `d` with replacement `n_bootstrap` times
- Return `(lower_bound, upper_bound)` of the `ci` confidence interval on the mean difference

**Function**: `run_all_comparisons(runs_df) -> pd.DataFrame`

Run the following comparisons for each dataset where n_pairs ≥ 3:

Primary (required for thesis):
1. baseline vs dualconv
2. baseline vs eca
3. baseline vs hybrid

Secondary (report but do not overclaim):
4. dualconv vs hybrid
5. eca vs hybrid

Apply Holm-Bonferroni correction across the 3 primary comparisons per dataset.

Return a DataFrame with columns:
`dataset`, `comparison`, `n_pairs`, `mean_diff_pp`, `median_diff_pp`, `ci_lower`, `ci_upper`, `wilcoxon_stat`, `p_value`, `p_corrected`, `significant_at_0.05`

### 6.2 Statistical Results Table

**File**: `results/tables/stats_primary.md`

For each dataset × primary comparison:
| Dataset | Comparison | N pairs | Mean Δ Top-1 (pp) | 95% CI | p (raw) | p (Holm) | Significant |
|---------|-----------|---------|------------------|--------|---------|---------|-------------|

**Note on small n**: With n=5 seeds, Wilcoxon has very low power. Report results but include the caveat that with n=5, only large effect sizes will reach significance. Do not over-interpret non-significant results as confirmation of equivalence.

**Note on Tiny ImageNet hybrid**: With n=1 for hybrid on Tiny ImageNet, skip Wilcoxon for any comparison involving hybrid×TinyImageNet. Report the single value with "n=1, no test possible".

### 6.3 Effect Size Table

**File**: `results/tables/stats_effect_sizes.md`

Report for each comparison:
- `mean_diff_pp`: positive = first variant is better
- `ci_lower`, `ci_upper`: 95% bootstrap CI
- Interpretation label: "no difference", "small (<1pp)", "moderate (1–3pp)", "large (>3pp)"

---

## 7. Cross-Dataset Summary

### 7.1 Master Summary Table

**File**: `results/tables/master_summary.md`

One master table covering all completed runs:

| Model | Params (M) | FLOPs (M) | CIFAR-100 Top-1 | CIFAR-100 Top-5 | TinyImageNet Top-1 | TinyImageNet Top-5 | n (C100) | n (TIN) |
|-------|-----------|----------|----------------|----------------|-------------------|-------------------|---------|---------|

Values: `mean ± std` format. Use `—` for unavailable (e.g., TinyImageNet hybrid with n=1 is reported as the single value with "(n=1)" annotation).

### 7.2 Heatmap: Test Top-1 by Variant × Dataset

**Figure**: `figures/heatmap_top1.png`

- Rows = model variants
- Columns = datasets
- Cell value = mean test Top-1 (%)
- Color scale: sequential (higher = darker green or blues)
- Annotate each cell with mean ± std
- Makes cross-dataset patterns immediately visible

---

## 8. Implementation Requirements

### 8.1 Directory Structure

```
analysis/
├── data_loader.py         ← Section 1: raw data ingestion
├── efficiency.py          ← Section 2: efficiency metrics
├── performance.py         ← Section 3: accuracy aggregation
├── dynamics.py            ← Section 5: learning curves, convergence
├── stats.py               ← Section 6: statistical tests
├── plots.py               ← All figure generation
├── tables.py              ← All markdown/CSV table generation
└── run_analysis.py        ← Top-level orchestrator

results/
├── tables/
│   ├── efficiency_summary.md
│   ├── efficiency_summary.csv
│   ├── performance_summary.md
│   ├── performance_summary.csv
│   ├── efficiency_accuracy.md
│   ├── early_stopping_analysis.md
│   ├── stats_primary.md
│   ├── stats_effect_sizes.md
│   └── master_summary.md
└── figures/
    ├── efficiency_bars.png
    ├── top1_bars_cifar100.png
    ├── top1_bars_tiny_imagenet.png
    ├── top5_bars_cifar100.png
    ├── top5_bars_tiny_imagenet.png
    ├── top1_violin_cifar100.png
    ├── top1_violin_tiny_imagenet.png
    ├── per_seed_scatter_cifar100.png
    ├── pareto_flops_cifar100.png
    ├── pareto_flops_tiny_imagenet.png
    ├── pareto_params_cifar100.png
    ├── pareto_params_tiny_imagenet.png
    ├── pareto_size_cifar100.png
    ├── pareto_size_tiny_imagenet.png
    ├── learning_curves_val_acc_cifar100.png
    ├── learning_curves_val_acc_tiny_imagenet.png
    ├── learning_curves_train_loss_cifar100.png
    ├── learning_curves_train_loss_tiny_imagenet.png
    ├── learning_curves_combined_cifar100.png
    ├── best_epoch_dist.png
    └── heatmap_top1.png
```

### 8.2 Top-Level Orchestrator

**File**: `analysis/run_analysis.py`

CLI usage:
```bash
python analysis/run_analysis.py \
  --trained_models_root "Trained Models/" \
  --output_dir results/ \
  --figures_dpi 300 \
  --figures_format png
```

Execution order:
1. `data_loader.load_all_runs()` → validate and deduplicate
2. `efficiency.build_efficiency_table()` → save CSV + MD
3. `performance.build_performance_summary()` → save CSV + MD
4. `stats.run_all_comparisons()` → save stats tables
5. `dynamics.*` → compute learning curves and convergence groups
6. `tables.*` → generate all markdown tables
7. `plots.*` → generate all figures
8. Print a completion report: how many runs loaded, how many missing from the planned 60-run matrix, any validation warnings

### 8.3 Plotting Standards

All figures must:
- Use `matplotlib` with `seaborn` style (`seaborn-v0_8-whitegrid` or equivalent)
- Font: 11pt axis labels, 9pt tick labels, 10pt legend
- DPI: 300 for print, saved as PNG
- Figure size: single-panel 6×4 in, 2×2 grid 9×7 in
- Consistent variant color palette (hardcoded dict, imported from one shared `constants.py`):
  ```python
  VARIANT_COLORS = {
      'baseline': '#6B7280',   # gray
      'dualconv': '#3B82F6',   # blue
      'eca':      '#10B981',   # green
      'hybrid':   '#F59E0B',   # orange/amber
  }
  VARIANT_LABELS = {
      'baseline': 'Baseline',
      'dualconv': 'DualConv',
      'eca':      'ECA',
      'hybrid':   'Hybrid (DualConv+ECA)',
  }
  DATASET_LABELS = {
      'cifar100':      'CIFAR-100',
      'tiny_imagenet': 'Tiny ImageNet',
  }
  ```
- All legends use `VARIANT_LABELS` display names, not raw keys
- All axis labels include units where applicable (%, M params, M FLOPs, MB)
- No hardcoded paths in plotting functions — accept `output_dir` as argument

### 8.4 Table Standards

All markdown tables must:
- Be valid GitHub-Flavored Markdown
- Use `—` for missing/inapplicable values (not empty cells)
- Include a one-line caption immediately below the table
- Float values: 2 decimal places for accuracy/%, 3 decimal places for p-values
- Bold the best (highest) accuracy per row in performance tables

### 8.5 Dependencies

Add to `requirements.txt` (or a separate `requirements_analysis.txt`):
```
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
```

No `torch` or `torchvision` required in the analysis layer.

---

## 9. Key Analysis Findings to Surface (Known From Data Audit)

These findings are pre-confirmed from data inspection and must be explicitly surfaced in the analysis output:

### 9.1 CIFAR-100 Bimodal Convergence
Some seeds (42, 777, 3407) triggered early stopping at ~50–70 epochs with ~31–35% Top-1. Seeds 123 and 2024 ran to ~195–200 epochs and reached ~59–63% Top-1. This pattern appears consistently across **all four model variants**, suggesting it is driven by training dynamics (learning rate schedule, early stopping sensitivity) not by model architecture. The analysis must:
- Detect and report this split automatically
- Compute mean Top-1 for converged vs. non-converged groups separately
- Flag this in the master summary table

### 9.2 DualConv + Hybrid Efficiency Reduction
Counterintuitively, `dualconv` and `hybrid` variants have **fewer params and FLOPs than baseline** (~−7.5% params, ~−20% FLOPs) because DualConv's grouped branch (groups = C_exp // 2) is more parameter-efficient than the standard depthwise conv at the bottleneck channel widths used. ECA adds only 29 parameters (+0.001%). The analysis must report this clearly and verify it from the `model_profile` fields.

### 9.3 Tiny ImageNet Hybrid Incompleteness
The `hybrid` variant on Tiny ImageNet has only 1 seed (3407). This should be flagged in all tables and all Tiny ImageNet comparisons involving hybrid should be annotated as preliminary.

### 9.4 ECA Overhead Is Negligible
ECA adds +29 params and +11,808 FLOPs over baseline (+0.001% each). All efficiency comparisons must reflect this. ECA is effectively parameter-neutral.

---

## 10. Agent Instructions for Cursor

- Implement `data_loader.py` first and validate it returns the correct number of rows before building any analysis.
- Expected row counts after deduplication: CIFAR-100: 4 variants × 5 seeds = 20 rows. Tiny ImageNet: 3 variants × 5 seeds + 1 (hybrid) = 16 rows. Total = 36 rows in `runs_df`.
- Do not hardcode dataset or model names — derive them from the JSON files and validate against the canonical sets.
- Every function that produces a figure must also save a corresponding `.csv` of the underlying aggregated data (e.g., `figures/top1_bars_cifar100.csv` with the mean/std values used). This ensures reproducibility.
- The `run_analysis.py` orchestrator must print a completion checklist at the end: which figures and tables were written, which were skipped due to missing data, and any validation warnings encountered.
- Do not filter out or suppress the bimodal CIFAR-100 result. It is real data.
- Use `pathlib.Path` throughout — no string path concatenation.
- Write at least one `pytest`-compatible test in `tests/test_data_loader.py` that loads from a minimal fixture (2 runs) and asserts the expected `runs_df` schema and row count.
