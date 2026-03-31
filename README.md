# mobilenetv2-dualconv-eca

Hybrid MobileNetV2 with efficient convolution and lightweight attention for complex image classification.

## Execution environment

- **Training, FLOPs/parameter profiling, and model-size profiling** can be run on **Google Colab** (or any single-GPU environment).
- **Inference latency** should be measured on **local hardware** (e.g. a fixed local GPU) for stable, comparable timings. See thesis Section 3.7.1.

## Reproducibility notes

- **Dependency pinning**: install from the pinned `requirements.txt` (exact versions).
- **Seeds**: training is intended to run with fixed seeds (see configs). The thesis seed set is: `42`, `123`, `3407`, `2024`, `777`.
- **Determinism**: this repo sets seeds (Python/NumPy/PyTorch) via `data.preprocessing.set_seed`. GPU kernels can still have nondeterminism depending on CUDA/cuDNN and operator choices; record `outputs/logs/env.json` for every run.
- **Version logging**: training writes `outputs/logs/env.json` with `python`, `torch`, `torchvision`, CUDA/cuDNN availability, and git commit hash.

## Structure

```
├── configs/           # Experiment configs (cifar10, cifar100, tiny_imagenet)
├── data/              # Preprocessing and dataset loaders
├── models/            # Backbone, efficient conv, attention, hybrid model
├── training/          # Train, evaluate, checkpoint utilities
├── experiments/       # Run scripts
├── outputs/           # Checkpoints, logs, figures
├── notebooks/         # Exploratory analysis
└── docs/              # Methodology notes
```

## Streamlit experiment dashboard

Visualize per-seed runs, efficiency, training curves, and statistical tables for the four MobileNetV2 variants across CIFAR-10, CIFAR-100, and Tiny-ImageNet.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first load, the app reads metrics from `Trained Models v3/` (preferred) and `Trained Models v2/` for **CIFAR-10** and **CIFAR-100** (`metrics.json` + `logs/epochs.jsonl`). You can also select `Trained Models v1/` from the sidebar if present. **Tiny-ImageNet** is not present in those folders; the dashboard fills it with **placeholder** data until you upload CSVs. If `epochs.jsonl` is shorter in v3 than in v2 for the same run, the loader keeps the file with **more lines**. **Latency** in the efficiency table uses the thesis static values (not read from disk).

**Version indicators in the UI:** `df_runs` gains **`metrics_source`** (`v2`, `v3`, or `placeholder`) and **`curves_source`** (which folder supplied `epochs.jsonl`, or `placeholder` if the log was too short). `df_curves` includes **`curves_source`** per row. `df_efficiency` includes **`metrics_json_from`** (e.g. `v3` or `v2+v3`) describing which `metrics.json` versions fed the averaged profile. The sidebar summarizes row counts per version; some runs can show **mixed** v2 vs v3 when test metrics come from one tree and the longer training log from the other.

### CSV upload schemas (optional)

Replace all session data by uploading three files together and clicking **Apply CSV uploads**.

**`df_runs.csv`**

| seed | variant      | dataset       | top1_acc | top5_acc |
|------|--------------|---------------|----------|----------|
| 42   | Baseline     | CIFAR-10      | 92.1     | 96.5     |
| 42   | DualConv-only| CIFAR-10      | 92.4     | 96.8     |

Optional columns: `metrics_source`, `curves_source` (otherwise the app sets **CSV upload**).

**`df_efficiency.csv`** (one row per variant)

| variant       | params_M | flops_M | size_mb | latency_ms |
|---------------|----------|---------|---------|------------|
| Baseline      | 3.40     | 300.0   | 13.0    | 8.2        |
| DualConv-only | 3.51     | 309.0   | 13.4    | 8.9        |

Optional: `metrics_json_from` (otherwise **CSV upload**).

**`df_curves.csv`**

| variant | dataset  | seed | epoch | train_loss | val_loss | val_top1 |
|---------|----------|------|-------|------------|----------|----------|
| Baseline| CIFAR-10 | 42   | 1     | 4.5        | 4.4      | 0.5      |

Optional: `curves_source` per row (otherwise **CSV upload**).

Use **Reset to placeholder + disk merge** to restore the default merge of disk artifacts and synthetic Tiny-ImageNet.

## Quick Start

```bash
pip install -r requirements.txt
python3 example_usage.py
```

### Version logging (standalone)

If you want to capture environment metadata without running training:

```bash
python3 -c "from utils.versioning import write_env_info_json; write_env_info_json('outputs/logs/env.json')"
```

### Baseline profiling

Profile MobileNetV2 baseline (params, FLOPs/MACs, model size, latency):

```bash
python scripts/profile_baseline.py --input_res 32
python scripts/profile_baseline.py --input_res 64 --num_classes 200
```

Requires `fvcore` (or `thop`) for FLOPs. Latency is intended to be run on local hardware (see above).

## Training

```bash
# CIFAR-10 (default config)
python experiments/train_cifar10.py

# CIFAR-100
python experiments/train_cifar10.py --config configs/cifar100.yaml

# Run all configs
python experiments/run_all.py
```

## Smoke test (no real training)

Run a single synthetic batch through model + loss + backward + optimizer step:

```bash
python training/smoke_test.py --config configs/cifar10.yaml
```

Training policy (config-driven):

- Max epochs: CIFAR-10 = 150, CIFAR-100 = 200, Tiny-ImageNet = 100
- Validation: every epoch
- Summary logging: every 10 epochs
- Early stopping monitor: validation Top-1 accuracy
- Early stopping rules: warm-up = 30 epochs, patience = 20 epochs, minimum improvement = 0.1 percentage points
- Training stops early only when the patience rule is met after warm-up, otherwise it runs until the dataset-specific epoch ceiling

## Datasets

- **CIFAR-10 / CIFAR-100**: Auto-downloaded to `./data`
- **Tiny-ImageNet**: Set `dataset_root` in `configs/tiny_imagenet.yaml`

# dualconv-eca-mobilenetv2
