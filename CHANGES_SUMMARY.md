# Changes summary (LR warmup, early stopping, best-accuracy knobs)

This file summarizes the code/config changes made in this workspace to support:

- **LR warmup** (linear warmup → cosine)
- **Early stopping** (warmup + patience) with **percentage-based defaults**
- **“Best-accuracy” recipe knobs** (RandAugment, RandomErasing, Mixup/CutMix, label smoothing)

All tests were run after changes and passed.

---

## 1) LR warmup (new)

### What

Added support for a top-level YAML key:

- `lr_warmup_epochs` (int, default `0`)

When `scheduler: cosine` and `lr_warmup_epochs > 0`, the schedule becomes:

1. **Linear warmup** from `start_factor=1e-6` → `1.0` for `lr_warmup_epochs` epochs
2. **Cosine annealing** for the remaining `epochs - lr_warmup_epochs` epochs

Validation:

- `lr_warmup_epochs >= 0`
- `lr_warmup_epochs < epochs`

### Files

- Updated: `training/utils.py`
- Updated tests: `tests/test_scheduler_config.py`

---

## 2) Early stopping parameters (existing feature, updated defaults)

### What

Early stopping already existed and is configured via YAML:

```yaml
early_stopping:
  enabled: true
  warmup_epochs: ...
  patience_epochs: ...
  min_delta_pp: ...
```

We updated dataset configs so that:

- `early_stopping.warmup_epochs` = **30%** of total `epochs`
- `patience_epochs` = **20%** of total `epochs`

### Resulting numbers per dataset

- **CIFAR-10** (`epochs: 150`)
  - `warmup_epochs: 45` (30%)
  - `patience_epochs: 30` (20%)
- **CIFAR-100** (`epochs: 200`)
  - `warmup_epochs: 60` (30%)
  - `patience_epochs: 40` (20%)
- **Tiny-ImageNet** (`epochs: 100`)
  - `warmup_epochs: 30` (30%)
  - `patience_epochs: 20` (20%)

### Files

- Updated: `configs/cifar10.yaml`
- Updated: `configs/cifar100.yaml`
- Updated: `configs/tiny_imagenet.yaml`

---

## 3) “Best-accuracy knob set” (YAML → data + training)

### Goal knobs (added to all dataset YAMLs)

```yaml
randaugment_num_ops: 2
randaugment_magnitude: 9
random_erasing_p: 0.25
mix_prob: 1.0
mixup_alpha: 1.0
cutmix_alpha: 1.0
label_smoothing: 0.1
```

These knobs are applied uniformly across **all model variants** through the shared runner.

---

## 4) Data augmentation pipeline changes

### What

Extended `get_transforms(...)` to optionally include:

- **RandAugment** (train only; inserted before `ToTensor()`)
- **RandomErasing** (train only; appended after `Normalize(...)`)

Threaded these options through:

- `get_cifar10_loaders(...)`
- `get_cifar100_loaders(...)`
- `get_tiny_imagenet_loaders(...)`
- `experiments/run_train_eval.py` now passes YAML values into `get_dataset_loaders(...)`

### Files

- Updated: `data/preprocessing.py`
- Updated: `experiments/run_train_eval.py`

---

## 5) Training loop changes (Mixup/CutMix + label smoothing)

### Label smoothing

The main runner now builds:

- `nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))`

### Mixup/CutMix

Added a helper module and wired mixing into training:

- Mix is applied per batch with probability `mix_prob`
- If mixing is applied: 50/50 selection between Mixup and CutMix
- Loss uses the “index-based” mixture:
  - `lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)`

### Wiring

- `Trainer` stores `mix_prob`, `mixup_alpha`, `cutmix_alpha`
- `Trainer.fit` passes these to `train_one_epoch(...)`

### Files

- Added: `training/mix.py`
- Updated: `training/train.py`
- Updated: `training/trainer.py`
- Updated: `experiments/run_train_eval.py`

---

## 6) Smoke + unit tests

### What

- Smoke mini-run now:
  - uses `label_smoothing` from config
  - passes mixing knobs into `Trainer`
  - still uses `build_scheduler` (and clamps warmup to 0 when mini-epochs is too small)

- Added minimal unit coverage to ensure:
  - transforms contain RandAugment + RandomErasing when enabled
  - `train_one_epoch` runs with mixing enabled

### Files

- Updated: `training/smoke_test.py`
- Added: `tests/test_best_accuracy_knobs.py`

---

## 7) Test status

- `pytest` run after these changes: **all tests passed**.

