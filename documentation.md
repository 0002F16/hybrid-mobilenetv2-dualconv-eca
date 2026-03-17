# DualConv MobileNetV2 Variants

This project includes **DualConv-based MobileNetV2 variants** inspired by:
**“DualConv: Dual Convolutional Kernels for Lightweight Deep Neural Networks” (Zhong et al., 2022)**.

The goal is to compare parameter count, FLOPs, and serialized model size against the baseline MobileNetV2 implementation in this repo.

## What was implemented

### DualConv building blocks

File: `models/dualconv.py`

- **`DualConv2d`**: parallel convolutions on the same input and sum the outputs
  - \(3\times3\) **group conv** (groups = `G`)
  - \(1\times1\) **pointwise conv**
  - Output: `conv3x3(x) + conv1x1(x)`
- **`DualConvBlock`**: paper-style MobileNetV2 replacement block
  - `DualConv2d → BatchNorm2d → ReLU6`
  - Optional residual connection when `stride == 1` and `in_channels == out_channels`

### Model variants

File: `models/mobilenetv2_dualconv_variants.py`

All variants reuse the repo’s baseline topology and naming scheme:
`B1..B17` with the same channel schedule/strides, and the same stem/head/classifier pattern as `models/mobilenetv2_baseline.py`.

- **`MobileNetV2DualConvAll`** (variant “DualConv-all”)
  - Replaces **all** `B1..B17` blocks with `DualConvBlock`.
- **`MobileNetV2DualConvB4B10`** (variant “DualConv-B4B10”)
  - Replaces **only** `B4..B10` with `DualConvBlock`.
- **`MobileNetV2DualConvB4B7`** (variant “DualConv-B4B7”)
  - Replaces **only** `B4..B7` with `DualConvBlock`.

Exports updated in: `models/__init__.py`

### Profiling script

File: `scripts/profile_dualconv_variants.py`

Profiles:
- **Params**: `utils.profiling.count_parameters` (trainable params)
- **FLOPs/MACs**: `utils.profiling.compute_flops` (fvcore-first; FLOPs = 2 × MACs)
- **Model size (MB)**: `utils.profiling.measure_model_size_mb` (size of saved `state_dict`)
- Also prints **% change vs baseline** for Params/FLOPs/size

## Results (CIFAR-style run)

Command used:

```powershell
python scripts/profile_dualconv_variants.py --input_res 32 --small_input --dualconv_groups 4
```

Environment notes:
- FLOPs counted by `fvcore` (`FlopCountAnalysis`), as used by this repo’s profiling utilities.
- You may see messages like “Unsupported operator …” from fvcore (e.g. `aten::hardtanh_` from `ReLU6`, `aten::add` from residuals / DualConv sum). These ops are not included in fvcore’s FLOP total, but conv FLOPs dominate.

### Summary table (input 1×3×32×32, width_mult=1.0, G=4)

Baseline reference:
- **MobileNetV2 Baseline**
  - Params: **2,236,682** (2.24M)
  - FLOPs: **25,556,736**
  - Model size: **8.7601 MB**

Variants:

| Model | Params | Δ Params vs baseline | FLOPs | Δ FLOPs vs baseline | Model size (MB) | Δ Size vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| DualConv-all (`MobileNetV2DualConvAll`) | 952,026 | -57.44% | 9,352,960 | -63.40% | 3.6963 | -57.81% |
| DualConv-B4B10 (`MobileNetV2DualConvB4B10`) | 2,069,562 | -7.47% | 20,393,728 | -20.20% | 8.0709 | -7.87% |
| DualConv-B4B7 (`MobileNetV2DualConvB4B7`) | 2,192,058 | -2.00% | 22,353,664 | -12.53% | 8.5657 | -2.22% |

## How to run

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run profiling (recommended)

```powershell
# CIFAR-like (32x32)
python scripts/profile_dualconv_variants.py --input_res 32 --small_input --dualconv_groups 4

# Tiny-ImageNet-like (64x64), example
python scripts/profile_dualconv_variants.py --input_res 64 --num_classes 200 --dualconv_groups 4
```

Useful flags:
- `--width_mult 1.0`: width multiplier (same convention as baseline)
- `--dualconv_groups 4`: DualConv group count \(G\)
- `--small_input`: CIFAR-style stem stride=1 (matches existing baseline profiling usage)
- `--output outputs/profile_report.txt`: write the report to a file

### Baseline-only profiling (existing script)

```powershell
python scripts/profile_baseline.py --input_res 32 --small_input
```

