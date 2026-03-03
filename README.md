# mobilenetv2-dualconv-eca

Hybrid MobileNetV2 with efficient convolution and lightweight attention for complex image classification.

## Execution environment

- **Training, FLOPs/parameter profiling, and model-size profiling** can be run on **Google Colab** (or any single-GPU environment).
- **Inference latency** should be measured on **local hardware** (e.g. a fixed local GPU) for stable, comparable timings. See thesis Section 3.7.1.

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

## Quick Start

```bash
pip install -r requirements.txt
python example_usage.py
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
