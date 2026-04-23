# Prep external CIFAR-10 eval assets for Colab

This repo includes a Colab notebook for evaluating the external curated CIFAR-10 ImageFolder dataset in the cloud:

- `colab/Eval_External_CIFAR10_Curated.ipynb`

## Current asset setup
The full curated dataset is now published as a **GitHub Release asset**:

- Release page: `https://github.com/0002F16/hybrid-mobilenetv2-dualconv-eca/releases/tag/external-cifar10-curated-v1`
- Asset name: `cifar10_curated_dataset_for_colab.zip`
- SHA-256: `8e96a85d84ac7a3800e1be6a094e74b9c421bf816260f69e124277727d09ee2c`

That means the Colab notebook can download the dataset directly from GitHub.

## What Colab still needs from you
Colab still needs your **trained checkpoints**, because those are not stored in the repo or release by default.

Local source:

- `Trained Models/cifar10`

## Recommended: put checkpoints zip in Google Drive
Create this zip locally, then upload it to Drive:

```bash
cd /Users/macbookm1/Documents/hybrid-mobilenetv2-dualconv-eca
zip -r cifar10_checkpoints_for_colab.zip 'Trained Models/cifar10'
```

Suggested Drive folder:

```text
MyDrive/hybrid-mobilenetv2-dualconv-eca/colab_inputs/
```

Put this there:

- `cifar10_checkpoints_for_colab.zip`

## Optional override
If you want, you can also put the dataset zip or extracted dataset folder in Drive, but it is no longer required because the notebook defaults to the GitHub Release asset.

## What the notebook does
- mounts Google Drive
- clones the repo into `/content`
- installs pinned requirements
- downloads the curated dataset from the GitHub Release asset by default
- unzips checkpoints from Drive (or uses extracted checkpoints already in Drive)
- runs external CIFAR-10 evaluation
- writes reports into the repo working copy
- copies/archives results back to Drive for download/export

## Notes on no-resize
True no-resize evaluation is memory-heavy because some images are very large. The notebook exposes both:

- `resize=32` full-dataset run
- optional `resize=0` no-resize run

For the no-resize run, use a **high-RAM Colab runtime** if possible.
