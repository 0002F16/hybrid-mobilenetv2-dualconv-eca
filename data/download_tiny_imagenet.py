from __future__ import annotations

import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def _download(url: str, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {pct:6.2f}% ({mb:,.1f}/{total_mb:,.1f} MiB)")
        sys.stdout.flush()

    if dst_path.exists() and dst_path.stat().st_size > 0:
        return dst_path

    urllib.request.urlretrieve(url, str(dst_path), reporthook=_reporthook)  # noqa: S310
    sys.stdout.write("\n")
    sys.stdout.flush()
    return dst_path


def _extract(archive_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive_path.suffixes).lower()

    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        return extract_to

    if suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar"):
        mode = "r:gz" if suffix.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(extract_to)  # noqa: S202
        return extract_to

    raise ValueError(f"Unsupported archive type: {archive_path.name}")


def _find_tiny_root(root: Path) -> Path:
    candidates = [
        root / "tiny-imagenet-200",
        root / "tiny_imagenet_200",
        root / "Tiny-ImageNet-200",
        root,
    ]
    for c in candidates:
        if (c / "train").exists() and (c / "val").exists():
            return c
    for p in root.rglob("tiny-imagenet-200"):
        if (p / "train").exists() and (p / "val").exists():
            return p
    raise FileNotFoundError(
        f"Could not locate tiny-imagenet-200 under {root}. "
        "Expected a folder containing train/ and val/."
    )


def ensure_tiny_imagenet(
    *,
    url: str,
    data_dir: str | Path = "data",
    force_redownload: bool = False,
) -> Path:
    """
    Download + extract Tiny-ImageNet (URL) and return dataset root path.

    The returned path is compatible with `data.preprocessing.get_tiny_imagenet_loaders`,
    which expects:
      - root/train/<wnid>/... (ImageFolder)
      - root/val/images/*.JPEG
      - root/val/val_annotations.txt
    """
    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)

    download_dir = data_dir_p / "downloads"
    extract_dir = data_dir_p / "tiny_imagenet"

    if force_redownload:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if download_dir.exists():
            shutil.rmtree(download_dir)

    archive_name = os.path.basename(url.split("?", 1)[0]) or "tiny-imagenet-200.zip"
    archive_path = download_dir / archive_name

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        archive_path = _download(url, archive_path)
        _extract(archive_path, extract_dir)

    tiny_root = _find_tiny_root(extract_dir)

    ann = tiny_root / "val" / "val_annotations.txt"
    images = tiny_root / "val" / "images"
    train_dir = tiny_root / "train"
    if not (ann.exists() and images.exists() and train_dir.exists()):
        raise FileNotFoundError(
            f"Tiny-ImageNet structure incomplete at {tiny_root}. "
            f"Missing one of: {train_dir}, {images}, {ann}"
        )

    return tiny_root


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct download URL to tiny-imagenet-200 archive")
    p.add_argument("--data_dir", default="data", help="Where to store downloads/extracted data")
    p.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = p.parse_args()

    out = ensure_tiny_imagenet(url=args.url, data_dir=args.data_dir, force_redownload=args.force)
    print(str(out))

