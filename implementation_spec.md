# Product Requirements Document
## Hybrid MobileNetV2 (DualConv + ECA) Thesis Implementation PRD

**Purpose:** This document is an implementation-ready PRD for building, training, evaluating, and reporting a thesis project on a hybrid MobileNetV2 architecture that integrates **DualConv** and **Efficient Channel Attention (ECA)** for complex image classification under resource constraints.


---

## 0. Non-Negotiables

The following rules are mandatory unless the user explicitly overrides them:

- modify only the bottleneck range corresponding to **B4–B10**
- preserve a **shared training regime** across all four variants
- keep all modified variants within **±10%** of the experimental baseline in both parameter count and FLOPs
- use one fixed split policy per dataset and reuse it across all seeds and variants
- use the fixed seed set: `42`, `123`, `3407`, `2024`, `777`
- evaluate final predictive metrics from the **best validation checkpoint**, not the last epoch
- do not silently change architectural scope, insertion points, or bottleneck targets
- do not tune different variants differently unless explicitly requested
- save machine-readable outputs first, then generate tables/figures from those outputs
- document every thesis-specific assumption in code comments, config, or run metadata

---

## 1. Project Summary

### 1.1 Goal
Build a reproducible research codebase that implements and evaluates four MobileNetV2-based model variants:

1. **Baseline MobileNetV2**
2. **MobileNetV2 + DualConv**
3. **MobileNetV2 + ECA**
4. **MobileNetV2 + DualConv + ECA (Hybrid)**

The system must support controlled comparison across:
- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

The project must preserve a **lightweight efficiency budget** while testing whether hybridizing DualConv and ECA improves classification performance on complex small-image benchmarks.

### 1.2 Success Criteria
The implementation is successful if it:
- correctly builds all four variants
- modifies only the intended MobileNetV2 bottleneck range
- enforces reproducible training/evaluation
- logs metrics consistently across seeds/datasets/variants
- computes efficiency metrics and statistical comparisons
- outputs thesis-ready tables and figures

### 1.3 Core Constraint
All modified variants must stay within **±10%** of the baseline MobileNetV2 in:
- parameter count
- FLOPs

If a variant exceeds the budget, it must be flagged before full training.

---

## 2. Scope

### 2.1 In Scope
- PyTorch implementation of all four variants
- dataset preparation and transforms
- deterministic training runs
- evaluation on CIFAR-10, CIFAR-100, Tiny-ImageNet
- ablation-ready experiment orchestration
- latency, FLOPs, parameter, and model-size measurement
- statistical testing and bootstrap confidence intervals
- plots and summary tables for thesis/manuscript use

### 2.2 Out of Scope
- web app or inference UI
- deployment to mobile devices
- neural architecture search
- automated hyperparameter optimization per variant
- broader benchmark expansion beyond the three target datasets
- model compression techniques beyond the stated architecture changes

---

## 3. Required Architectural Design

### 3.1 Base Backbone
Use **TorchVision MobileNetV2** as the canonical baseline source.

Implementation must explicitly document:
- pinned `torch` and `torchvision` versions
- exact mapping between thesis bottleneck labels **B1–B17** and TorchVision block indices
- any CIFAR-specific stride adjustments

### 3.2 Variants to Implement

#### Variant A: Baseline
The **experimental baseline** is TorchVision MobileNetV2 plus any **shared dataset-resolution adaptations** required for the thesis protocol. It is not necessarily byte-for-byte vanilla TorchVision if CIFAR/Tiny-ImageNet resolution handling requires a shared stride policy.

Definition:
- all shared input-resolution and stride adaptations belong to the baseline
- DualConv and ECA are **not** part of the baseline
- all non-baseline variants must inherit the exact same shared adaptations before architectural comparison begins

#### Variant B: DualConv-only
Replace the standard depthwise convolution stage with **DualConv** in the target bottleneck range.

#### Variant C: ECA-only
Insert **ECA** in the target bottleneck range while keeping the original convolution operator.

#### Variant D: Hybrid
Combine **DualConv** and **ECA** in the same target bottlenecks.

### 3.3 Modification Range
Architectural modifications should be applied only to the **mid-depth bottleneck range corresponding to thesis blocks B4–B10**.

Implementation must include:
- a documented block mapping utility
- assertions/tests verifying that only the intended blocks are modified

### 3.4 DualConv Requirements
DualConv replaces the bottleneck’s depthwise stage and must:
- accept the **expanded bottleneck feature map** as input
- contain two branches:
  - **Branch A:** grouped 3×3 convolution
  - **Branch B:** 1×1 convolution
- fuse branches by element-wise addition
- apply BatchNorm and ReLU6 after fusion

Locked implementation rules:
- let `C_exp` be the expanded channel width entering the original depthwise stage
- **Branch A** must use `Conv2d(C_exp, C_exp, kernel_size=3, stride=s, padding=1, groups=C_exp // 2, bias=False)`
- **Branch B** must use `Conv2d(C_exp, C_exp, kernel_size=1, stride=1, padding=0, bias=False)`
- branch outputs must be fused as `A(x) + B(x_or_aligned)` followed by `BatchNorm2d(C_exp)` and `ReLU6`
- if Branch A uses `stride=2`, Branch B alignment behavior must be explicit and consistent across all variants; the implementation must not silently create shape-mismatched fusion
- if exact thesis code is unavailable, the default implementation policy is:
  - apply the bottleneck stride to **Branch A**
  - keep **Branch B** at stride 1
  - use an explicit alignment strategy for Branch B before summation, and document that strategy in code/comments/tests
- residual connection legality must remain governed by standard MobileNetV2 rules after the modified operator

Implementation details:
- add tests that verify the grouped branch uses `groups=C_exp // 2`
- add tests for both stride-1 and stride-2 bottlenecks
- do not change channels before or after DualConv except through the standard bottleneck structure

### 3.5 ECA Requirements
ECA must:
- operate **after projection convolution + batch norm**
- be applied **before residual addition** when a residual path exists
- use global average pooling + 1D convolution + sigmoid-based channel reweighting
- remain lightweight enough to preserve the budget

Locked implementation rules:
- let `C_out` be the bottleneck output channel count after projection
- ECA must follow: `AdaptiveAvgPool2d(1) -> reshape to [B, 1, C_out] -> Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False) -> Sigmoid -> channel-wise scaling`
- the default kernel sizing rule must follow the manuscript/PRD intent for lightweight adaptive ECA sizing
- if the exact adaptive rule is ambiguous in source material, the implementation must centralize the kernel-size function in one helper and document the chosen formula
- ECA placement is locked to **after projection BN and before residual addition/output emission**
- stride-2 / no-residual blocks must still apply ECA to the projected output before returning the block output

Implementation details:
- add tests verifying the insertion point is after projection BN
- add tests for both residual and non-residual bottlenecks
- document the exact ECA kernel function in code and experiment metadata

### 3.6 Hybrid Bottleneck Requirements
The hybrid bottleneck sequence must conceptually be:
1. expansion 1×1 conv + BN + ReLU6
2. DualConv replacing the depthwise stage
3. projection 1×1 conv + BN
4. ECA on projected output
5. residual addition when legal under MobileNetV2 rules

### 3.7 Shared Stride Policy for CIFAR/Tiny-ImageNet
Stride handling must be treated as a **shared baseline adaptation**, not as a per-variant design choice.

Required policy:
- Phase 3 must generate a **feature-map audit** for 32×32 and 64×64 inputs showing spatial resolution after the initial conv and every bottleneck stage
- based on that audit, the implementation must lock one shared stride policy for the experimental baseline
- once selected, the same stride policy must be reused unchanged across `baseline`, `dualconv`, `eca`, and `hybrid`
- the chosen policy must be written to:
  - code comments near the baseline implementation
  - config or metadata
  - an artifact file such as `artifacts/block_mapping_and_stride_policy.md`

Minimum required audit output:
- stage/block name
- input resolution
- output resolution
- whether stride differs from vanilla TorchVision
- rationale for the final shared policy

---

## 4. Dataset and Data Pipeline Requirements

### 4.1 Target Datasets
The training system must support:
- **CIFAR-10**
- **CIFAR-100**
- **Tiny-ImageNet**

### 4.2 Split Policy
Use the following policy consistently across all runs:

#### CIFAR-10 / CIFAR-100
- official training set
- fixed 10% validation split carved from training
- official test set for final testing

#### Tiny-ImageNet
- official training set
- fixed 10% validation split carved from training
- official validation set used as **test set**

This must be stated explicitly in code comments and experiment metadata to avoid ambiguity.

### 4.3 Validation Split Requirements
- split once per dataset using a fixed split seed
- reuse the exact same split across all variants and all training seeds
- avoid leakage between train/val/test
- store split metadata in machine-readable form

Required split metadata fields:
- dataset name
- split seed
- train count
- validation count
- test count
- for Tiny-ImageNet: explicit note that the official validation set is used as the test set

### 4.4 Preprocessing and Augmentation
Implementation must support dataset-specific transforms:

#### CIFAR-10 / CIFAR-100
- pad
- random crop to 32×32
- random horizontal flip
- normalization using dataset-specific mean/std computed from the training subset only

#### Tiny-ImageNet
- resize to target resolution
- random crop
- random horizontal flip
- normalization using dataset-specific mean/std computed from training subset only

### 4.5 Normalization Rules
- mean/std must be computed from the **post-split training subset only**, not from validation or test data
- computed values should be stored in code/config for reproducibility

### 4.6 Data Loader Requirements
The codebase must include tests/assertions for:
- input shapes
- label range validity
- split sizes
- deterministic split reproducibility
- Tiny-ImageNet folder organization compatibility

---

## 5. Training Requirements

### 5.1 Shared Training Regime
All variants must be trained under the **same optimization configuration** unless a change is explicitly approved and documented.

### 5.2 Optimizer and Schedule
Use:
- **SGD with momentum 0.9**
- base learning rate: **0.01**
- weight decay: **1e-4**
- cosine annealing schedule over the dataset-specific full epoch ceiling

### 5.3 Batch Size
- batch size = **64** across all datasets and variants unless hardware constraints force a documented exception

### 5.4 Loss Function
- use standard multi-class cross-entropy from raw logits
- do not place softmax inside the model

### 5.5 Epoch Ceilings
Target defaults:
- CIFAR-10: **150** epochs
- CIFAR-100: **200** epochs
- Tiny-ImageNet: **100** epochs

### 5.6 Early Stopping
Use:
- monitor: validation Top-1 accuracy
- warm-up: **30 epochs**
- patience: **20 epochs**
- minimum improvement threshold: **0.1 percentage points**

The best validation checkpoint must be used for final test evaluation.

### 5.7 Seed Policy
Use the same training seeds across all variant-dataset combinations:
- `42`
- `123`
- `3407`
- `2024`
- `777`

### 5.8 Determinism Requirements
Provide a seed utility that sets:
- Python random seed
- NumPy seed
- PyTorch CPU seed
- PyTorch CUDA seed(s)
- cuDNN determinism flags as appropriate

The code should reduce nondeterminism as much as practical and document any remaining backend variability.

### 5.9 Checkpointing and Logging
Each run must log at minimum:
- dataset
- variant
- seed
- train loss per epoch
- validation Top-1 per epoch
- best validation score
- best epoch
- test Top-1 from best checkpoint
- total executed epochs
- early stopping trigger status

Checkpointing must save:
- model weights
- config snapshot
- seed
- dataset
- variant
- best epoch metadata

Required artifact layout:
- `results/raw/{dataset}/{variant}/seed_{seed}.json`
- `results/raw/{dataset}/{variant}/seed_{seed}.ckpt` or equivalent checkpoint
- `results/processed/metrics_summary.csv`
- `results/processed/stats_summary.csv`
- `results/figures/`
- `results/tables/`
- `artifacts/block_mapping_and_stride_policy.md`
- `artifacts/split_metadata/{dataset}.json`

Minimum per-run JSON schema:
- `dataset`
- `variant`
- `seed`
- `split_seed`
- `train_count`
- `val_count`
- `test_count`
- `train_epochs_executed`
- `best_epoch`
- `best_val_top1`
- `test_top1`
- `test_top5`
- `params`
- `flops`
- `model_size_mb`
- `latency_ms`
- `early_stopped`
- `commit_hash`
- `torch_version`
- `torchvision_version`
- `device`
- `notes`

---

## 6. Testing and Verification Requirements

### 6.1 Unit-Level Checks
Before full training, implement tests for:
- module construction
- forward-pass output shapes
- correct insertion/replacement of modules
- residual compatibility
- stride-2 behavior
- budget verification calls

### 6.2 Sanity Runs
Before large-scale training, run short sanity experiments to verify:
- loss decreases from initialization
- accuracy exceeds random guessing
- model outputs are numerically stable
- no shape mismatch across datasets

### 6.3 Budget Verification
Create a verification step that computes for each variant:
- trainable parameter count
- FLOPs at dataset-native resolution

This must run before full training and fail loudly if a variant exceeds the ±10% budget.

### 6.4 Reproducibility Checks
Add at least one reproducibility test that repeats the same short run with the same seed/config and verifies that outputs or checkpoints are acceptably consistent.

---

## 7. Experiment Execution Requirements

### 7.1 Required Experiment Matrix
Execute the full controlled experiment matrix:
- 4 variants
- 3 datasets
- 5 seeds

Total planned training runs: **60**.

### 7.2 Variant Labels
Use stable canonical labels throughout logs/files/plots:
- `baseline`
- `dualconv`
- `eca`
- `hybrid`

### 7.3 Dataset Labels
Use stable canonical labels:
- `cifar10`
- `cifar100`
- `tiny_imagenet`

### 7.4 Run Orchestration
Provide a script or command interface to:
- launch single runs
- launch all seeds for a variant/dataset
- launch the full experiment matrix
- resume interrupted experiments safely

### 7.5 Run Metadata
Every run must emit a machine-readable record containing:
- git commit hash if available
- timestamp
- versions of torch/torchvision/fvcore/scipy/etc.
- hardware info
- dataset
- variant
- seed
- config values
- split seed and split counts
- baseline/stride policy identifier
- any thesis-specific implementation note needed to reproduce the run

---

## 8. Evaluation Requirements

### 8.1 Predictive Metrics
Measure:
- **Top-1 accuracy** for all datasets
- **Top-5 accuracy** for CIFAR-100 and Tiny-ImageNet

Top-5 for CIFAR-10 should be omitted or clearly marked as non-informative.

### 8.2 Efficiency Metrics
Measure:
- parameter count
- FLOPs
- model size on disk
- inference latency

### 8.3 FLOPs Profiling
Use a consistent FLOPs tool and document:
- input shape used
- whether unsupported ops are excluded
- whether activations/BN/attention-specific ops are counted

### 8.4 Latency Measurement
Latency measurement must:
- be separated from training
- use fixed hardware when possible
- use FP32
- use batch size 1
- include warm-up iterations before timed iterations
- report mean latency, and optionally dispersion statistics

### 8.5 Evaluation Output Format
For each completed run, save a structured record containing all predictive and efficiency metrics needed for aggregation.

---

## 9. Statistical Analysis Requirements

### 9.1 Primary Comparison Logic
Statistical testing should compare each variant against the baseline using paired seeds.

Primary comparisons:
1. baseline vs dualconv
2. baseline vs eca
3. baseline vs hybrid

Optional secondary comparisons:
4. dualconv vs hybrid
5. eca vs hybrid

### 9.2 Statistical Tests
Use:
- **Wilcoxon signed-rank test** for paired comparisons
- **Holm-Bonferroni correction** for multiple comparisons

### 9.3 Effect Size and Uncertainty
Report:
- paired performance difference
- effect size summary (mean or median paired difference)
- **95% bootstrap confidence intervals**

### 9.4 Claim Criteria
A performance gain should only be described as meaningful if supported by:
- directional consistency across seeds
- corrected statistical significance where applicable
- non-trivial effect size relative to run variability

### 9.5 Statistical Deliverables
Produce per dataset:
- pairwise comparison table
- corrected p-values
- effect sizes
- confidence intervals

---

## 10. Data Visualization Requirements

### 10.1 Must-Have Plots
Generate thesis-ready plots for each dataset including at minimum:

1. **Accuracy vs FLOPs scatter plot**
2. **Accuracy vs parameter count scatter plot**
3. **Bar chart of Top-1 accuracy by variant**
4. **Bar chart of latency by variant**
5. optional: **Top-5 accuracy plot** where applicable

### 10.2 Plot Standards
Plots must:
- use consistent variant naming and colors
- include axis labels and units
- include error bars where appropriate
- be exportable in high-resolution formats for thesis use
- be reproducible from saved results only

### 10.3 Summary Tables
Generate tables for:
- predictive metrics per dataset and variant
- efficiency metrics per variant
- statistical test results
- full master summary across datasets and variants

### 10.4 Pareto Reporting
Where relevant, identify Pareto-efficient models in accuracy-efficiency plots.

---

## 11. Repository / Codebase Requirements

### 11.1 Recommended Structure
```text
project_root/
├── models/
│   ├── baseline.py
│   ├── dualconv.py
│   ├── eca.py
│   ├── hybrid_bottleneck.py
│   ├── blocks.py
│   └── factory.py
├── data/
│   ├── datasets.py
│   ├── transforms.py
│   ├── splits.py
│   ├── stats.py
│   └── tiny_imagenet_prep.py
├── training/
│   ├── trainer.py
│   ├── scheduler.py
│   ├── checkpointing.py
│   └── seeds.py
├── evaluation/
│   ├── metrics.py
│   ├── efficiency.py
│   ├── latency.py
│   └── stats_tests.py
├── experiments/
│   ├── configs/
│   ├── run_one.py
│   ├── run_matrix.py
│   └── verify_budget.py
├── viz/
│   ├── plots.py
│   └── tables.py
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── tables/
├── tests/
├── requirements.txt
├── README.md
└── PRD.md
```

### 11.2 Code Quality Requirements
- modular implementation
- typed/configurable where practical
- no hardcoded path assumptions without config override
- clear docstrings for architectural modules
- comments around block mapping and thesis-specific assumptions

### 11.3 Config Requirements
The experiment system should use explicit configs for:
- dataset
- input resolution
- variant
- optimizer settings
- scheduler settings
- early stopping settings
- seed
- output paths

---

## 12. Phased Execution Plan

## Phase 1 — Environment and Project Skeleton
### Objective
Create a clean, reproducible repository skeleton and lock core dependencies.

### Tasks
- initialize repository structure
- pin dependency versions
- create base config system
- add README and reproduction notes
- add version logging utility

### Deliverables
- runnable project skeleton
- requirements file
- base experiment config files

### Exit Criteria
- environment installs successfully
- imports pass
- config loading works

---

## Phase 2 — Data Pipeline Implementation
### Objective
Implement dataset loading, split creation, transforms, and normalization.

### Tasks
- build CIFAR-10 loader
- build CIFAR-100 loader
- build Tiny-ImageNet loader
- implement Tiny-ImageNet reorganization helper if needed
- implement deterministic train/val split generation
- compute/store normalization statistics
- add shape and split validation tests

### Deliverables
- `datasets.py`
- `transforms.py`
- `splits.py`
- `stats.py`
- Tiny-ImageNet prep script

### Exit Criteria
- all datasets load correctly
- split sizes are correct
- normalization values are reproducible
- one batch from each dataset passes validation assertions

---

## Phase 3 — Baseline Model Implementation
### Objective
Implement the baseline MobileNetV2 and establish the canonical block mapping.

### Tasks
- wrap TorchVision MobileNetV2
- document bottleneck index mapping B1–B17
- apply dataset-resolution adaptation consistently if required
- verify output shape across datasets/class counts
- compute baseline params/FLOPs

### Deliverables
- `baseline.py`
- mapping documentation/comments
- baseline verification outputs

### Exit Criteria
- baseline forward pass works on all target resolutions
- mapping between thesis blocks and code indices is documented
- baseline efficiency metrics are recorded

---

## Phase 4 — DualConv Implementation
### Objective
Implement DualConv and integrate it into target bottlenecks.

### Tasks
- implement standalone DualConv module
- integrate replacement logic into target blocks only
- handle stride-1 and stride-2 cases
- verify shape correctness and residual behavior
- compute budget overhead vs baseline

### Deliverables
- `dualconv.py`
- target block replacement utility
- budget verification report for DualConv variant

### Exit Criteria
- DualConv variant builds and runs forward passes
- target blocks only are modified
- budget remains within threshold or violation is clearly reported

---

## Phase 5 — ECA Implementation
### Objective
Implement ECA and integrate it into target bottlenecks.

### Tasks
- implement ECA module
- insert ECA after projection BN in target blocks only
- support residual and non-residual bottlenecks
- verify output correctness
- compute budget overhead vs baseline

### Deliverables
- `eca.py`
- insertion utility
- budget verification report for ECA variant

### Exit Criteria
- ECA variant builds and runs forward passes
- ECA placement is correct and documented
- budget remains within threshold or violation is clearly reported

---

## Phase 6 — Hybrid Assembly
### Objective
Build the full hybrid variant and verify all four modes through one common factory.

### Tasks
- implement common variant factory
- assemble baseline / dualconv / eca / hybrid paths
- verify all output shapes and target modifications
- compute params/FLOPs for all variants at all required resolutions

### Deliverables
- `factory.py`
- all-variant verification table

### Exit Criteria
- all four variants build from one entry point
- all pass forward tests
- all have recorded efficiency metrics

---

## Phase 7 — Training Infrastructure
### Objective
Create the reusable training/evaluation utilities, checkpointing, logging artifacts, and determinism utilities.

### Tasks
- implement reusable epoch training step (`train_one_epoch`)
- implement evaluation utility (`evaluate`) for val/test loss + accuracy
- implement checkpoint save/load utilities
- implement config loading utilities (YAML)
- implement canonical experiment runner that:
  - applies seed control deterministically
  - configures optimizer + scheduler
  - saves best-validation checkpoint
  - evaluates test metrics from best-validation checkpoint
  - writes machine-readable run outputs
- implement early stopping in the canonical runner (warm-up + patience + min-delta)
- add a short smoke/reproducibility test (may be synthetic) to validate the pipeline

### Deliverables
- `training/train.py` (epoch step)
- `training/evaluate.py` (val/test evaluation)
- `training/utils.py` (checkpointing + YAML config loading)
- `data/preprocessing.py` (seed control utility)
- `experiments/run_train_eval.py` (canonical training+evaluation runner)
- `training/smoke_test.py` (pipeline smoke test)

### Strict gaps / follow-ups (must be addressed before Phase 9)
- **Trainer abstraction**: no reusable `Trainer` class/module currently exists; early stopping and per-epoch orchestration live in experiment runners.
- **Resume/retry support**: resume-from-checkpoint and robust retry handling are not implemented yet; Phase 9 requires this for the 60-run matrix.
- **Structured per-epoch logs**: run-level JSON artifacts exist, but there is no centralized per-epoch event log (e.g., JSONL/CSV) or unified logger API.
- **End-to-end reproducibility mini-run**: existing smoke test is synthetic and may instantiate a concrete model directly; add a true mini-run that exercises `models.factory.build_model` and the canonical runner path.

### Exit Criteria
- running a short training job completes end-to-end via the canonical runner
- checkpoints save/load correctly (best-validation checkpoint exists and can be loaded)
- test metrics are computed from the best-validation checkpoint (not the last epoch)
- machine-readable artifacts are written for each run:
  - `outputs/<dataset>/<model>/seed_<seed>/checkpoints/best.pt`
  - `outputs/<dataset>/<model>/seed_<seed>/logs/env.json`
  - `outputs/<dataset>/<model>/seed_<seed>/logs/config.json`
  - `outputs/<dataset>/<model>/seed_<seed>/metrics.json`

---

## Phase 8 — Sanity Training and Debugging
### Objective
Validate that each variant can train without crashing and produce plausible learning behavior.

### Tasks
- run short CIFAR-10 sanity training for each variant
- confirm decreasing loss
- confirm above-random accuracy
- inspect for NaNs, shape issues, or unstable gradients
- validate best-checkpoint testing flow

### Deliverables
- sanity run logs
- bug fixes from observed failures

### Exit Criteria
- each variant completes a sanity run successfully
- no major numerical or pipeline issues remain

---

## Phase 9 — Full Training Matrix
### Objective
Execute the complete 60-run experiment matrix.

### Tasks
- prerequisite: implement resume-from-checkpoint and retry-safe runs (see Phase 7 strict gaps)
- run all variants across all datasets and seeds
- support resume/retry for interrupted runs
- monitor run completion and output integrity
- aggregate raw metrics after each run

### Deliverables
- complete raw results for all run combinations
- saved checkpoints and metadata

### Exit Criteria
- all planned runs are completed or explicitly marked failed
- missing/incomplete runs are identified clearly

---

## Phase 10 — Evaluation and Efficiency Measurement
### Objective
Compute predictive and efficiency metrics consistently for the completed run set.

### Tasks
- compute Top-1 / Top-5 metrics
- compute parameter counts and FLOPs
- compute checkpoint sizes
- run latency benchmarking on fixed hardware
- store processed per-run and per-variant results

### Deliverables
- processed metrics tables
- latency outputs
- efficiency reports

### Exit Criteria
- all required metrics are available for every completed run/variant/dataset

---

## Phase 11 — Statistical Analysis
### Objective
Produce the statistical comparisons needed for the thesis.

### Tasks
- assemble paired seed comparisons
- run Wilcoxon signed-rank tests
- apply Holm-Bonferroni correction
- compute bootstrap confidence intervals
- summarize effect sizes

### Deliverables
- statistical results tables
- machine-readable stats outputs

### Exit Criteria
- all primary comparisons are computed for each dataset
- corrected p-values and confidence intervals are available

---

## Phase 12 — Data Visualization and Reporting Assets
### Objective
Generate thesis-ready figures and tables from the processed results.

### Tasks
- generate all required plots
- generate per-dataset summary tables
- generate master cross-dataset table
- export publication/thesis-friendly figures
- document any caveats directly in captions or accompanying notes

### Deliverables
- figures directory
- tables directory
- final summary artifacts for manuscript insertion

### Exit Criteria
- all core plots/tables can be regenerated from saved results
- outputs are presentable for thesis use

---

## 13. Risks and Implementation Notes

### 13.1 High-Risk Areas
- incorrect thesis block mapping to TorchVision indices
- inconsistent CIFAR stride handling across variants
- Tiny-ImageNet split misuse or poor documentation
- hidden budget overrun from operator replacement
- nondeterministic behavior across hardware sessions
- FLOPs tool counting inconsistencies

### 13.2 Required Mitigations
- add explicit mapping assertions/tests
- centralize stride-fix logic and reuse across all variants
- encode dataset split policy in code/config
- run budget verification before full training
- record library/hardware versions in run metadata
- document FLOPs counting assumptions in results output

---

## 14. Final Deliverables

The completed project must produce:

1. a reproducible codebase
2. four implemented MobileNetV2 variants
3. complete experiment logs and checkpoints
4. processed evaluation results
5. statistical comparison outputs
6. thesis-ready plots and tables
7. documentation of architecture assumptions, split policy, and reproducibility details

---

## 15. Agent Instructions for Cursor / Coding Agents

When implementing this PRD:
- prefer small, testable commits/patches
- verify shapes and block mappings before training
- do not silently change architecture scope
- do not tune different variants differently unless explicitly asked
- preserve one shared training regime for fair comparison
- fail loudly on budget violations or split inconsistencies
- store outputs in machine-readable formats first, then generate figures/tables from those outputs
- document every thesis-specific assumption in code comments or config

If any ambiguity remains, resolve it by:
1. preserving fairness across variants
2. maximizing reproducibility
3. minimizing deviation from the thesis/manuscript intent
4. documenting the decision clearly
