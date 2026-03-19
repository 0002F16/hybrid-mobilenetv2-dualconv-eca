from __future__ import annotations

from pathlib import Path

import pytest

from data.preprocessing import (
    _parse_tiny_imagenet_val_annotations,
    get_transforms,
    make_train_val_split_indices,
)


def test_split_indices_reproducible_across_training_seeds() -> None:
    # training seed must not affect split membership; only split_seed matters
    n = 50_000
    split_seed = 1337
    t1, v1 = make_train_val_split_indices(n=n, val_fraction=0.1, split_seed=split_seed)
    t2, v2 = make_train_val_split_indices(n=n, val_fraction=0.1, split_seed=split_seed)
    assert t1 == t2
    assert v1 == v2
    assert len(set(t1).intersection(v1)) == 0
    assert len(t1) + len(v1) == n


def test_split_indices_change_when_split_seed_changes() -> None:
    n = 10_000
    t1, v1 = make_train_val_split_indices(n=n, val_fraction=0.1, split_seed=1337)
    t2, v2 = make_train_val_split_indices(n=n, val_fraction=0.1, split_seed=1338)
    assert (t1 != t2) or (v1 != v2)


def test_tiny_imagenet_val_annotations_parser(tmp_path: Path) -> None:
    ann = tmp_path / "val_annotations.txt"
    ann.write_text(
        "\n".join(
            [
                "val_00000001.JPEG\tn01443537\t0\t0\t64\t64",
                "val_00000002.JPEG\tn01629819\t0\t0\t64\t64",
                "",
            ]
        )
        + "\n"
    )
    mapping = _parse_tiny_imagenet_val_annotations(ann)
    assert mapping["val_00000001.JPEG"] == "n01443537"
    assert mapping["val_00000002.JPEG"] == "n01629819"


def test_split_indices_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError):
        make_train_val_split_indices(n=100, val_fraction=0.0, split_seed=1)
    with pytest.raises(ValueError):
        make_train_val_split_indices(n=100, val_fraction=1.0, split_seed=1)


def test_transforms_output_shapes_no_download() -> None:
    from PIL import Image

    mean = (0.5, 0.5, 0.5)
    std = (0.2, 0.2, 0.2)

    # CIFAR: random crop to 32
    cifar_train_t, cifar_test_t = get_transforms("cifar10", mean=mean, std=std)
    img = Image.new("RGB", (40, 40), color=(128, 128, 128))
    x_train = cifar_train_t(img)
    x_test = cifar_test_t(img)
    assert tuple(x_train.shape) == (3, 32, 32)
    assert tuple(x_test.shape) == (3, 40, 40)

    # Tiny-ImageNet: resize/crop to 64
    tiny_train_t, tiny_test_t = get_transforms("tiny_imagenet", mean=mean, std=std)
    big_img = Image.new("RGB", (96, 96), color=(128, 128, 128))
    xt_train = tiny_train_t(big_img)
    xt_test = tiny_test_t(big_img)
    assert tuple(xt_train.shape) == (3, 64, 64)
    assert tuple(xt_test.shape) == (3, 64, 64)

