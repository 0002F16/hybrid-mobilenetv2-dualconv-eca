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

## Datasets

- **CIFAR-10 / CIFAR-100**: Auto-downloaded to `./data`
- **Tiny-ImageNet**: Set `dataset_root` in `configs/tiny_imagenet.yaml`
# dualconv-eca-mobilenetv2
