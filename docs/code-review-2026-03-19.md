# Comprehensive Code Review — `hybrid-mobilenetv2-dualconv-eca`

Date: 2026-03-19
Reviewer: David

## Executive summary

This repository is in **pretty solid shape structurally** for a thesis/research codebase:

- the project layout is clear and readable
- model construction is reasonably modular
- reproducibility got real attention (seed setting, split metadata, env logging)
- the training/eval runner works end-to-end
- the current automated test suite passes locally (`5 passed`)
- the code compiles cleanly with `python3 -m compileall .`

That said, there are several issues that matter if the goal is **research-grade correctness + maintainability + defensible experimental results**.

### Overall verdict

- **Architecture / organization:** Good
- **Reproducibility posture:** Good
- **Documentation quality:** Decent, but some drift exists
- **Training/evaluation correctness:** Mostly good, with a few important caveats
- **Test coverage:** Too narrow for the repo’s scope
- **Production readiness:** Not production code; acceptable for research, but should be tightened before final thesis submission or wider reuse

## What I validated

### Static / runtime checks

- `python3 -m pytest -q` → **5 passed**
- `python3 -m compileall .` → **passed**

### Repo-level observations

- project has clean topical separation: `models/`, `data/`, `training/`, `experiments/`, `scripts/`
- run artifacts are being written under `outputs/`
- the repo includes generated output files under `outputs/_tmp_runs/...` and `outputs/_tmp_smoke/...`

---

## Strengths

## 1) Good project decomposition

The code is split along sensible boundaries:

- `models/` for architecture variants
- `data/` for dataset loading and preprocessing
- `training/` for train/eval/checkpoint mechanics
- `experiments/` for orchestration
- `utils/versioning.py` for environment capture

That’s a strong baseline for a research repo because it reduces coupling and makes experimental plumbing easier to reason about.

## 2) Reproducibility work is better than average

The repo does more than the usual “set one seed and hope” approach:

- fixed split seed support in `data/preprocessing.py`
- split metadata written with hashes of train/val indices
- environment metadata logged to `env.json`
- config snapshot logged per run

Those are exactly the kinds of details that make later auditability much better.

## 3) Model factory is centralized and readable

`models/factory.py` gives a single dispatch point for:

- baseline
- dualconv variants
- hybrid

That’s the right pattern for experiments. It keeps training code from filling up with model-specific branching.

## 4) The baseline implementation is explicit

`models/mobilenetv2_baseline.py` is easy to inspect and maps well to a thesis narrative because the B1–B17 blocks are explicit and named. That makes targeted ablation/replacement work much easier to defend academically.

## 5) Resume/checkpoint support exists

The trainer supports:

- `best.pt`
- `last.pt`
- optional resume path
- scheduler state restore attempts

That’s good and already more robust than many student repos.

---

## Findings

## High-priority findings

### 1) Loss averaging is mathematically biased by batch count instead of sample count

**Severity:** High

#### Evidence

`training/train.py`:

```python
21     for batch_idx, (data, target) in enumerate(train_loader):
...
28         total_loss += loss.item()
29         num_batches += 1
31     return total_loss / max(num_batches, 1)
```

`training/evaluate.py`:

```python
23         for data, target in loader:
...
26             total_loss += criterion(output, target).item()
31     avg_loss = total_loss / max(len(loader), 1)
```

#### Why this matters

With `CrossEntropyLoss(reduction="mean")` (the default), `loss.item()` is already the mean over the current batch. Averaging those means equally across batches gives **batch-weighted loss**, not **sample-weighted loss**.

That becomes wrong whenever the last batch is smaller than the others, or when batch sizes vary for any reason.

Accuracy is sample-weighted. Loss should be too, especially if you want training curves and reported metrics to be defensible.

#### Recommended fix

Accumulate:

- `total_loss += loss.item() * batch_size`
- `total_examples += batch_size`

Then return:

- `total_loss / total_examples`

Apply this in both training and evaluation.

---

### 2) Scheduler configuration is declared in YAML but effectively ignored

**Severity:** High

#### Evidence

`configs/cifar10.yaml`:

```yaml
scheduler: cosine
```

`experiments/run_train_eval.py`:

```python
133     scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
```

There is no branching on `cfg["scheduler"]`; the code always instantiates `CosineAnnealingLR`.

#### Why this matters

This creates **config drift**:

- the config implies scheduler selection is configurable
- the implementation hardcodes one scheduler regardless of the config value

That’s a reproducibility/documentation problem. Anyone reading the config would assume it controls behavior when it currently does not.

#### Recommended fix

Either:

1. actually honor the config field, or
2. remove the scheduler key from configs and explicitly document that cosine annealing is fixed by design.

If this is for a thesis, I’d prefer option 1 for clarity and future-proofing.

---

### 3) `best.pt` trainer metadata can be stale relative to the epoch that created it

**Severity:** Medium-High

#### Evidence

In `training/trainer.py`, best checkpoint saving happens **before** the meaningful-improvement / early-stopping counters are updated:

```python
139                 if float(val_acc) > float(best.get("val_acc", -1.0)):
140                     best = {"val_acc": float(val_acc), "val_loss": float(val_loss), "epoch": int(epoch)}
141                     is_best = True
142                     save_checkpoint(
...
149                         extra={"trainer_state": {
150                             "best_val": best,
151                             "best_top1_pp": best_top1_pp,
152                             "epochs_without_meaningful_improve": epochs_without_meaningful_improve,
153                         }},
154                     )
```

Then later:

```python
205             meaningful_improvement = (float(val_top1_pp) - float(best_top1_pp)) > float(self.early.min_delta_pp)
206             if meaningful_improvement:
207                 best_top1_pp = float(val_top1_pp)
208                 epochs_without_meaningful_improve = 0
209             else:
210                 epochs_without_meaningful_improve += 1
```

#### Why this matters

The model weights in `best.pt` are correct, but the embedded trainer state may not reflect the exact post-epoch early-stopping state. That can make resume behavior subtly inconsistent if someone resumes from `best.pt` rather than `last.pt`.

#### Recommended fix

Update all early-stopping state first, then save both `best.pt` and `last.pt` using the finalized trainer state for that epoch.

---

### 4) Epoch logs record the early-stop counter before it is updated for the current epoch

**Severity:** Medium

#### Evidence

`training/trainer.py`:

```python
160             self._append_epoch_log(
...
169                     "early_stop_counter": int(epochs_without_meaningful_improve),
170                 }
171             )
```

This happens before:

```python
205             meaningful_improvement = ...
206             if meaningful_improvement:
207                 best_top1_pp = float(val_top1_pp)
208                 epochs_without_meaningful_improve = 0
209             else:
210                 epochs_without_meaningful_improve += 1
```

#### Why this matters

Your `epochs.jsonl` is slightly misleading: the recorded counter is really the **pre-update** value for that epoch.

Not catastrophic, but if you later analyze training dynamics, this creates off-by-one confusion.

#### Recommended fix

Move epoch log writing until after the early-stopping state is updated, or log both pre/post values explicitly.

---

## Medium-priority findings

### 5) `.gitignore` does not ignore nested run artifacts under `outputs/`

**Severity:** Medium

#### Evidence

`.gitignore`:

```gitignore
1  # Outputs
2  outputs/checkpoints/*.pt
3  outputs/logs/*
4  outputs/figures/*
```

But the repo currently contains tracked/generated files such as:

- `outputs/_tmp_runs/cifar10/hybrid/seed_42/checkpoints/best.pt`
- `outputs/_tmp_runs/cifar10/hybrid/seed_42/logs/config.json`
- `outputs/_tmp_smoke/cifar10/hybrid/seed_42/metrics.json`

#### Why this matters

This is artifact leakage into the repo. For a research repo, committed machine-generated outputs can:

- bloat diffs
- create confusion about canonical results
- accidentally publish stale runs
- make the repo look dirtier / less reproducible

#### Recommended fix

Replace the narrow ignores with something like:

```gitignore
outputs/**
!outputs/checkpoints/.gitkeep
!outputs/logs/.gitkeep
!outputs/figures/.gitkeep
```

If you intentionally want selected result files versioned, use a dedicated `results/` or `reports/` folder instead of mixing transient outputs into `outputs/`.

---

### 6) Attention module naming/documentation overstates what is actually implemented

**Severity:** Medium

#### Evidence

`models/attention.py` header/docstrings say:

```python
1  """
2  Lightweight attention modules: SE, ECA, CBAM-style.
```

and

```python
57 class LightweightAttention(nn.Module):
58     """
59     Lightweight attention combining channel and spatial cues.
60 
61     Uses SE for channels and a minimal spatial gate.
62     """
```

But implementation is only:

```python
66         self.channel_attn = SqueezeExcitation(channels, reduction)
68     def forward(self, x: torch.Tensor) -> torch.Tensor:
69         return self.channel_attn(x)
```

There is no spatial gate in `LightweightAttention`.

#### Why this matters

This is not a runtime bug, but it is a **research communication bug**. If this code backs a thesis or paper, naming drift is dangerous because it can make the implementation differ from the written method description.

#### Recommended fix

Choose one:

1. rename/document it honestly as SE-only attention, or
2. implement the missing spatial gate.

Right now, the code says more than it does.

---

### 7) Global SSL monkey-patch happens at import time in data preprocessing

**Severity:** Medium

#### Evidence

`data/preprocessing.py`:

```python
4  # Fix SSL certificate verification on macOS (Python.org installs)
5  try:
6      import certifi
7  
8      ssl._create_default_https_context = lambda: ssl.create_default_context(
9          cafile=certifi.where()
10     )
11 except ImportError:
12     pass
```

#### Why this matters

This mutates global interpreter SSL behavior as a side effect of importing a data module. That is surprising and creates hidden coupling.

In a small personal repo it may “work,” but architecturally it’s the wrong layer.

#### Recommended fix

Move SSL environment setup into:

- a download-specific utility, or
- an explicit CLI/bootstrap step

Do not patch interpreter-global SSL behavior inside a general preprocessing module.

---

### 8) The repo has duplicate core building-block implementations

**Severity:** Medium

#### Evidence

There are duplicated concepts across files, including:

- `_make_divisible`
- `InvertedResidual`

For example these are defined separately in:

- `models/backbone.py`
- `models/hybrid.py`
- `models/mobilenetv2_baseline.py`
- `models/efficient_conv.py`

#### Why this matters

Duplication increases drift risk. If one variant gets corrected or tuned later, others can silently diverge.

For a thesis codebase with multiple ablations, that drift can become painful.

#### Recommended fix

Extract shared primitives into one canonical module, e.g.:

- `models/common.py`
- `models/blocks.py`

Then import them everywhere.

---

## Low-priority findings

### 9) `epoch` parameter in `train_one_epoch` is unused

**Severity:** Low

#### Evidence

`training/train.py` accepts `epoch`, but it is not used.

#### Why this matters

This is harmless, but it suggests either:

- logging was intended but never added, or
- the parameter is unnecessary API surface

#### Recommended fix

Remove it or use it for structured logging.

---

### 10) Some docstrings/comments are slightly outdated or overly broad

Examples:

- `LightweightAttention` claims channel + spatial cues, but only SE exists
- README title/footer has naming duplication / drift (`mobilenetv2-dualconv-eca` vs `dualconv-eca-mobilenetv2`)
- training comments imply configurability where implementation is more fixed than advertised

This won’t break execution, but it weakens the repo’s credibility.

---

## Testing assessment

## What’s good

The existing tests do validate some useful reproducibility/data behavior:

- split reproducibility
- split-seed sensitivity
- Tiny-ImageNet annotation parsing
- invalid fraction guardrails
- transform output shapes

## What’s missing

Test coverage is **far too narrow** for the actual system.

Currently only one real test module exists:

- `tests/test_data_pipeline.py`

### Important missing test categories

#### 1) Model construction smoke tests

You should have tests that instantiate:

- baseline
- dualconv_all
- dualconv_b4b10
- dualconv_b4b7
- hybrid

and run a forward pass for:

- CIFAR-sized inputs (32x32)
- Tiny-ImageNet-sized inputs (64x64)

#### 2) Output shape tests

Assert that logits match `num_classes` for all variants.

#### 3) Training loop smoke test

A synthetic dataset test should validate:

- forward
- backward
- optimizer step
- scheduler step
- no shape/device errors

#### 4) Checkpoint/resume tests

You should verify that:

- `last.pt` is created
- `best.pt` is created
- resume restores epoch correctly
- optimizer state restores
- scheduler state restores

#### 5) Metrics/logging tests

You should verify that a run writes:

- `logs/config.json`
- `logs/env.json`
- `logs/epochs.jsonl`
- `metrics.json`

#### 6) Early stopping tests

These are especially important because the logic is stateful and easy to get subtly wrong.

---

## Design review by subsystem

## Models

### Positives

- factory pattern is clean
- baseline is explicit and thesis-friendly
- dualconv replacement scopes are readable
- hybrid model wiring is easy to follow

### Concerns

- duplication of shared MobileNetV2 logic across files
- architectural naming drift around attention
- no tests proving all model variants produce valid outputs

### Recommendation

Consolidate common MobileNetV2 primitives and add model-shape smoke tests immediately.

---

## Data pipeline

### Positives

- split generation is deterministic
- split metadata hashing is a strong touch
- dataset dispatcher is simple and readable
- Tiny-ImageNet official val usage is explicitly encoded

### Concerns

- import-time SSL patch is a layering smell
- preprocessing module is doing a lot (seeding, splitting, transforms, stats, SSL behavior, metadata)
- the module is starting to become a “god file”

### Recommendation

Split responsibilities over time:

- `seed.py`
- `splits.py`
- `transforms.py`
- `stats.py`
- `tiny_imagenet.py`

Not required immediately, but it would improve maintainability.

---

## Training pipeline

### Positives

- trainer abstraction is decent
- resume support exists
- early stopping is configurable
- metrics are persisted in machine-readable form

### Concerns

- incorrect loss averaging
- early-stopping state/log ordering is awkward
- scheduler is hardcoded despite config field

### Recommendation

This is the area I would tighten first, because it directly affects experiment correctness and interpretability.

---

## Documentation / repo hygiene

### Positives

- README gives quick-start and reproducibility notes
- thesis-style framing is present
- profiling and experiment scripts are discoverable

### Concerns

- generated outputs are leaking into versioned repo content
- naming drift across files and descriptions
- a few comments/docstrings overclaim behavior

### Recommendation

Treat documentation consistency as part of experimental correctness, not just polish.

---

## Suggested priority order for fixes

## Priority 1 — fix now

1. **Correct loss averaging** in `training/train.py` and `training/evaluate.py`
2. **Make scheduler config truthful**: either honor `cfg["scheduler"]` or remove it
3. **Fix `.gitignore`** so nested `outputs/**` artifacts are excluded
4. **Align attention docs with actual implementation**

## Priority 2 — next

1. Reorder trainer logging/checkpoint state updates for consistency
2. Add model smoke tests and checkpoint/resume tests
3. Remove or relocate the SSL monkey-patch

## Priority 3 — cleanup

1. Deduplicate shared model primitives
2. Reduce docstring drift / naming inconsistency
3. Break up oversized preprocessing responsibilities

---

## Final assessment

If I were grading this as a serious research codebase, I’d say:

> **The repo is thoughtfully organized and already demonstrates above-average reproducibility discipline, but it still has a few correctness and maintainability issues that should be fixed before treating its outputs as fully thesis-polished.**

The biggest real issue is the **loss accounting**. After that, the next most important thing is making sure the **configuration and documentation match what the code actually does**.

Once those are fixed and a few more tests are added, this becomes a much stronger and more defensible repo.

---

## Short status note

- **Last completed step:** full repo review + validation pass
- **Current step:** report written to Markdown
- **Blocker status:** no blocker
- **Visible output:** `docs/code-review-2026-03-19.md`

