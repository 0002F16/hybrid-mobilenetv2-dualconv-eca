## Dataset YAML configuration guide

This repo uses one YAML file per dataset under `configs/`:
- `configs/cifar10.yaml`
- `configs/cifar100.yaml`
- `configs/tiny_imagenet.yaml`

Each file describes **what to train**, **how to train**, and **how to reproduce** runs. The training entrypoints (e.g. `experiments/run_train_eval.py`) read these keys into a config dict and pass them through the model factory + training loop.

---

## Common keys (all three files)

### Dataset
- **`dataset`**: Dataset identifier used by the data-loading code.
  - Values here: `cifar10`, `cifar100`, `tiny_imagenet`
- **`dataset_root`**: Root directory containing the dataset on disk.
  - CIFAR configs default to `./data`
  - Tiny-ImageNet config is a placeholder like `/path/to/tiny-imagenet-200` (you must change it when running locally/Colab).

### Model
- **`model`**: Which model variant to build (e.g. `baseline`, `dualconv`, `eca`, `hybrid`).
  - In all three configs it is currently set to `hybrid` by default.
- **`num_classes`**: Number of output classes for the classifier head.
  - CIFAR-10: `10`
  - CIFAR-100: `100`
  - Tiny-ImageNet: `200`
- **`width_multiplier`**: MobileNetV2 width scaling factor. `1.0` is the standard width.

### Training (optimizer hyperparameters)
- **`batch_size`**: Mini-batch size.
- **`epochs`**: Maximum number of epochs to train (early stopping may stop sooner).
- **`learning_rate`**: Base LR used by the optimizer before scheduling/warmup.
- **`momentum`**: SGD momentum.
- **`weight_decay`**: L2 regularization strength.

### Learning-rate warmup and schedule
- **`lr_warmup_epochs`**: Number of warmup epochs for the LR schedule (linear warmup before cosine).
  - This is **only** about the optimizer LR schedule.
- **`scheduler`**: Scheduler type; all three use `cosine` (cosine annealing).

### Validation / logging cadence
- **`val_interval_epochs`**: How often to run validation (in epochs).
- **`summary_log_interval_epochs`**: How often to print/log summary metrics (in epochs).

### Early stopping
The `early_stopping` block controls when training halts early based on validation improvements.

- **`early_stopping.enabled`**: Turn early stopping on/off.
- **`early_stopping.warmup_epochs`**: Epochs to wait before patience starts counting.
  - This is **separate** from `lr_warmup_epochs`.
- **`early_stopping.patience_epochs`**: Stop if no improvement for this many epochs (after warmup).
- **`early_stopping.min_delta_pp`**: Minimum improvement (in **percentage points**) needed to reset patience.

### Regularization / augmentation knobs
- **`randaugment_num_ops`**, **`randaugment_magnitude`**: RandAugment strength.
- **`random_erasing_p`**: Probability of Random Erasing.
- **`mix_prob`**: Probability of applying MixUp/CutMix (implementation-dependent, but used as “mixing enabled” knob).
- **`mixup_alpha`**, **`cutmix_alpha`**: Beta distribution parameters controlling MixUp/CutMix strength.
- **`label_smoothing`**: Label smoothing value (e.g. 0.1).

### Reproducibility / dataloader
- **`seed`**: Random seed for training.
- **`split_seed`**: Seed controlling any dataset split/shuffle logic (keeps train/val splits consistent).
- **`num_workers`**: DataLoader workers.

---

## Dataset-specific differences (what changes across the three YAMLs)

### `configs/cifar10.yaml`
- **`num_classes`**: `10`
- **`epochs`**: `150`
- **`early_stopping.warmup_epochs`**: `45`
- **`early_stopping.patience_epochs`**: `30`
- **`learning_rate`**: `0.1`
- **`batch_size`**: `64`

Interpretation: slightly shorter training than CIFAR-100, with earlier early-stopping warmup.

### `configs/cifar100.yaml`
- **`num_classes`**: `100`
- **`epochs`**: `200`
- **`early_stopping.warmup_epochs`**: `60`
- **`early_stopping.patience_epochs`**: `40`
- **`learning_rate`**: `0.1`
- **`batch_size`**: `64`

Interpretation: longer run and more patience, reflecting the harder 100-class task.

### `configs/tiny_imagenet.yaml`
- **`dataset_root`**: placeholder path (must be set to your Tiny-ImageNet directory).
- **`num_classes`**: `200`
- **`epochs`**: `200`
- **`learning_rate`**: `0.2` (higher than CIFAR configs)
- **`batch_size`**: `128` (larger than CIFAR configs)
- **`early_stopping`**: same warmup/patience as CIFAR-100 (`60` / `40`)

Interpretation: larger batch + larger LR, consistent with the larger dataset and different scale.

---

## Practical tips

- **Switching variants**: change `model:` in the YAML (e.g. `baseline`, `dualconv`, `eca`, `hybrid`) to run the same training recipe on different architectures.
- **Budgeted comparisons**: keep all knobs except `model` (and dataset-specific necessities like `num_classes`) the same when reporting thesis comparisons.
- **Warmup terminology**: `lr_warmup_epochs` affects LR only; `early_stopping.warmup_epochs` affects when “patience” starts counting.

