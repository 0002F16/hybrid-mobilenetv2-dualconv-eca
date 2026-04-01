# Run from repository root:
#   streamlit run demo/streamlit_app.py
"""Streamlit UI: pick a trained run, upload an image, show top-1 and top-5 predictions."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from demo.inference import (
    TrainedRun,
    discover_runs,
    load_model_for_run,
    preprocess_image,
    predict_topk,
    repo_root,
    resolve_label,
)


@st.cache_resource
def _cached_load_model(run_path_str: str, device_str: str) -> tuple:
    """Load model + cfg + test transform; keyed by run path and device."""
    path = Path(run_path_str)
    run = TrainedRun(path=path.resolve(), dataset="", variant="", seed="")
    device = torch.device(device_str)
    model, cfg, _mean, _std, test_transform = load_model_for_run(run, device=device)
    return model, cfg, test_transform


def _default_trained_root() -> Path:
    return repo_root() / "Trained Models"


def main() -> None:
    st.set_page_config(page_title="Trained model demo", layout="centered")
    st.title("Image classification demo")
    st.caption("Uses checkpoints under your trained-models folder (best.pt + config + split metadata).")

    sidebar = st.sidebar
    trained_root = Path(
        sidebar.text_input(
            "Trained models root",
            value=str(_default_trained_root()),
            help="Folder containing <dataset>/<variant>/seed_<n>/ runs.",
        )
    )
    cifar_data_root = Path(
        sidebar.text_input(
            "CIFAR data root (for class names)",
            value=str(repo_root() / "data"),
            help="CIFAR-10/100 metadata download location for label strings.",
        )
    )
    tiny_root_str = sidebar.text_input(
        "Tiny ImageNet root (optional, for class names)",
        value="",
        help="Path to tiny-imagenet-200 (must contain train/ with class folders). Leave empty to show class indices.",
    )
    tiny_imagenet_root = Path(tiny_root_str).resolve() if tiny_root_str.strip() else None

    cuda_available = torch.cuda.is_available()
    device_choice = sidebar.radio(
        "Device",
        options=["auto", "cpu", "cuda"],
        index=0,
    )
    if device_choice == "auto":
        device_str = "cuda" if cuda_available else "cpu"
    else:
        device_str = device_choice
        if device_str == "cuda" and not cuda_available:
            st.warning("CUDA not available; using CPU.")
            device_str = "cpu"
    device = torch.device(device_str)

    runs = discover_runs(trained_root)
    if not runs:
        st.error(
            f"No runnable checkpoints found under `{trained_root}`. "
            "Each run needs `checkpoints/best.pt`, `logs/config.json`, and "
            "`artifacts/split_metadata/<dataset>.json`."
        )
        st.stop()

    run = st.selectbox("Model run", runs, format_func=lambda r: r.label)

    if run.dataset == "cifar100":
        st.info(
            "**CIFAR-100 limits:** These checkpoints are trained on **32×32** CIFAR images. "
            "Your upload is resized to 32×32, so fine detail is lost and arbitrary photos are "
            "**out of distribution**. Also, CIFAR-100 has only **100 fixed labels** — there is "
            "**no “dog” class** (closest might be *wolf*, *fox*, *tiger*, etc.), so pet or "
            "studio photos often get confusing labels and **low confidence**. "
            "For more natural photo categories, try a **Tiny ImageNet** run; to sanity-check "
            "accuracy, use images similar to the CIFAR test set."
        )
    elif run.dataset == "cifar10":
        st.info(
            "**CIFAR-10:** Trained on **32×32** images with **10 coarse classes** only. "
            "Random high-resolution uploads are out of distribution after resizing; "
            "expect weaker, low-confidence predictions than on in-distribution CIFAR test images."
        )

    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if uploaded is None:
        st.info("Upload an image to see predictions.")
        st.stop()

    image = Image.open(uploaded)

    try:
        model, cfg, test_transform = _cached_load_model(str(run.path), device_str)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    batch = preprocess_image(image, test_transform)
    try:
        top = predict_topk(model, batch, device, k=5)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    dataset = str(cfg.get("dataset", "unknown")).lower()
    rows = []
    for rank, (idx, prob) in enumerate(top, start=1):
        label = resolve_label(
            idx,
            dataset=dataset,
            cifar_data_root=cifar_data_root,
            tiny_imagenet_root=tiny_imagenet_root,
        )
        rows.append({"rank": rank, "class_id": idx, "label": label, "probability": prob})

    df = pd.DataFrame(rows)

    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Input", use_container_width=True)
    with col_pred:
        top1 = rows[0]
        st.metric("Top-1", f"{top1['label']}")
        st.caption(f"class_id={top1['class_id']} · p={top1['probability']:.4f}")

    st.subheader("Top-5")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability": st.column_config.NumberColumn(format="%.4f"),
        },
    )


if __name__ == "__main__":
    main()
