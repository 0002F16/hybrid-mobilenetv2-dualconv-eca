# Prep external CIFAR-10 eval assets for Colab

This repo now includes a Colab notebook for evaluating the external curated CIFAR-10 ImageFolder dataset in the cloud:

- `colab/Eval_External_CIFAR10_Curated.ipynb`

## What Colab needs
Colab will **not** have your local files by default, so you need to provide:

1. **Repo source**
   - default: cloned from GitHub inside Colab
2. **Trained checkpoints**
   - local source: `Trained Models/cifar10`
3. **Curated external dataset**
   - local source: `/Users/macbookm1/.openclaw/workspace/curated-image-datasets/data/cifar10`

## Recommended: put zip files in Google Drive
Create these two zip files locally, then upload them to Drive:

```bash
cd /Users/macbookm1/Documents/hybrid-mobilenetv2-dualconv-eca
zip -r cifar10_checkpoints_for_colab.zip 'Trained Models/cifar10'

cd /Users/macbookm1/.openclaw/workspace/curated-image-datasets/data
zip -r cifar10_curated_dataset_for_colab.zip cifar10
```

Suggested Drive folder:

```text
MyDrive/hybrid-mobilenetv2-dualconv-eca/colab_inputs/
```

Put these there:

- `cifar10_checkpoints_for_colab.zip`
- `cifar10_curated_dataset_for_colab.zip`

## What the notebook does
- mounts Google Drive
- clones the repo into `/content`
- installs pinned requirements
- unzips checkpoints + curated dataset from Drive (or uses extracted folders directly)
- runs external CIFAR-10 evaluation
- writes reports into the repo working copy
- copies/archives results back to Drive for download/export

## Notes on no-resize
True no-resize evaluation is memory-heavy because some images are very large. The notebook exposes both:

- `resize=32` full-dataset run
- optional `resize=0` no-resize run

For the no-resize run, use a **high-RAM Colab runtime** if possible.
