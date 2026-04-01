"""
Local latency benchmark for trained checkpoints under `Trained Models/`.

Matches thesis-style measurement:
- FP32 forward pass latency
- batch size 1
- 50 warm-up iterations + 200 timed iterations
- CUDA-synchronized timing when on GPU (delegated to utils.profiling.measure_latency)

This is intentionally a *local* tool; do not rely on it for CI stability.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.factory import build_model
from utils.profiling import measure_latency


Variant = Literal["baseline", "dualconv", "eca", "hybrid"]


@dataclass(frozen=True)
class RunCandidate:
    dataset: str
    variant: str
    seed_dir: str  # e.g. "seed_42"
    run_dir: Path
    metrics_path: Path
    config_path: Path

    @property
    def seed_int(self) -> int | None:
        try:
            return int(self.seed_dir.split("_", 1)[1])
        except Exception:
            return None

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / "best.pt"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_candidates(trained_root: Path) -> list[RunCandidate]:
    root = Path(trained_root)
    if not root.is_dir():
        return []

    out: list[RunCandidate] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name.lower()
        for variant_dir in sorted(dataset_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name.lower()
            for run_dir in sorted(variant_dir.glob("seed_*")):
                if not run_dir.is_dir():
                    continue
                metrics_path = run_dir / "metrics.json"
                config_path = run_dir / "logs" / "config.json"
                if not metrics_path.is_file() or not config_path.is_file():
                    continue
                out.append(
                    RunCandidate(
                        dataset=dataset,
                        variant=variant,
                        seed_dir=run_dir.name,
                        run_dir=run_dir.resolve(),
                        metrics_path=metrics_path,
                        config_path=config_path,
                    )
                )
    return out


def choose_dataset_for_one_per_variant(candidates: list[RunCandidate], *, requested: str | None) -> str | None:
    if requested and requested.lower() != "auto":
        return requested.lower()
    if not candidates:
        return None
    # Pick dataset with most candidates; tie-break lexicographically for determinism.
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.dataset] = counts.get(c.dataset, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best


def _score_from_metrics(metrics: dict[str, Any]) -> tuple[float, float]:
    """
    Higher is better score tuple for sorting.
    Primary: best_val.val_acc (descending)
    Secondary: best_val.val_loss (ascending)
    """
    best_val = metrics.get("best_val", {}) if isinstance(metrics.get("best_val", {}), dict) else {}
    acc = _safe_float(best_val.get("val_acc"))
    loss = _safe_float(best_val.get("val_loss"))
    # Missing fields sink to the bottom.
    if acc is None:
        acc = float("-inf")
    if loss is None:
        loss = float("inf")
    return acc, -loss  # negate loss so higher is better in a descending sort


def select_best_seed_per_variant(
    candidates: list[RunCandidate],
    *,
    dataset: str,
    variants: tuple[str, ...] = ("baseline", "dualconv", "eca", "hybrid"),
) -> dict[str, RunCandidate | None]:
    by_variant: dict[str, list[tuple[RunCandidate, tuple[float, float], float, int]]] = {v: [] for v in variants}
    for c in candidates:
        if c.dataset != dataset:
            continue
        if c.variant not in by_variant:
            continue
        try:
            metrics = _read_json(c.metrics_path)
        except Exception:
            continue
        acc_desc_loss = _score_from_metrics(metrics)
        # Tie-breakers: lower val_loss (already encoded), then lower seed int, then path.
        seed_int = c.seed_int if c.seed_int is not None else 10**9
        # Keep the original (non-negated) loss for the JSON.
        best_val = metrics.get("best_val", {}) if isinstance(metrics.get("best_val", {}), dict) else {}
        val_loss = _safe_float(best_val.get("val_loss"))
        if val_loss is None:
            val_loss = float("inf")
        by_variant[c.variant].append((c, acc_desc_loss, val_loss, seed_int))

    selected: dict[str, RunCandidate | None] = {}
    for v in variants:
        items = by_variant.get(v, [])
        if not items:
            selected[v] = None
            continue
        items.sort(key=lambda t: (-t[1][0], -t[1][1], t[3], str(t[0].run_dir)))
        selected[v] = items[0][0]
    return selected


def _input_size_chw_for_run(candidate: RunCandidate) -> tuple[int, int, int]:
    # Prefer the recorded profile shape from metrics.json.
    try:
        metrics = _read_json(candidate.metrics_path)
        prof = metrics.get("model_profile", {}) if isinstance(metrics.get("model_profile", {}), dict) else {}
        chw = prof.get("input_size_chw")
        if (
            isinstance(chw, list)
            and len(chw) == 3
            and all(isinstance(x, (int, float)) for x in chw)
        ):
            c, h, w = (int(chw[0]), int(chw[1]), int(chw[2]))
            if c > 0 and h > 0 and w > 0:
                return (c, h, w)
    except Exception:
        pass

    # Fallback: infer from dataset.
    ds = candidate.dataset.lower()
    if ds in {"cifar10", "cifar100"}:
        return (3, 32, 32)
    if ds in {"tiny_imagenet", "tiny-imagenet", "tinyimagenet"}:
        return (3, 64, 64)
    return (3, 32, 32)


def resolve_device(choice: str) -> torch.device:
    c = choice.lower()
    if c == "cpu":
        return torch.device("cpu")
    if c == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device 'cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_env_meta(device: torch.device) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        meta["cuda"] = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
        }
        try:
            p = torch.cuda.get_device_properties(0)
            meta["cuda"]["device_properties"] = {
                "total_memory_bytes": int(getattr(p, "total_memory", 0)),
                "major": int(getattr(p, "major", 0)),
                "minor": int(getattr(p, "minor", 0)),
                "multi_processor_count": int(getattr(p, "multi_processor_count", 0)),
            }
        except Exception:
            pass
    # Best-effort git SHA (optional)
    git_head = _REPO_ROOT / ".git" / "HEAD"
    if git_head.is_file():
        try:
            head = git_head.read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_path = _REPO_ROOT / ".git" / ref
                if ref_path.is_file():
                    meta["git_commit"] = ref_path.read_text(encoding="utf-8").strip()
            else:
                meta["git_commit"] = head
        except Exception:
            pass
    return meta


def run_latency_benchmark(
    *,
    trained_root: Path,
    dataset_choice: str,
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int,
    latency_fn: Callable[..., float] = measure_latency,
) -> dict[str, Any]:
    candidates = discover_run_candidates(trained_root)
    dataset = choose_dataset_for_one_per_variant(candidates, requested=dataset_choice)
    variants = ("baseline", "dualconv", "eca", "hybrid")
    selected = select_best_seed_per_variant(candidates, dataset=dataset or "", variants=variants)

    results: list[dict[str, Any]] = []
    for variant in variants:
        cand = selected.get(variant)
        if cand is None:
            results.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "status": "missing_run",
                    "reason": "No run candidate found (missing metrics.json or logs/config.json).",
                }
            )
            continue

        entry: dict[str, Any] = {
            "dataset": cand.dataset,
            "variant": cand.variant,
            "seed": cand.seed_dir,
            "run_dir": str(cand.run_dir),
        }

        if not cand.checkpoint_path.is_file():
            entry.update(
                {
                    "status": "skipped",
                    "reason": f"Missing checkpoint: {cand.checkpoint_path}",
                }
            )
            results.append(entry)
            continue

        try:
            cfg = _read_json(cand.config_path)
        except Exception as e:
            entry.update({"status": "skipped", "reason": f"Failed to read config.json: {e}"})
            results.append(entry)
            continue

        input_size_chw = _input_size_chw_for_run(cand)
        entry["input_size_chw"] = list(input_size_chw)
        entry["batch_size"] = int(batch_size)
        entry["warmup"] = int(warmup)
        entry["iters"] = int(iters)
        entry["dtype"] = "fp32"

        try:
            model = build_model(cfg).to(device)
            ckpt = torch.load(cand.checkpoint_path, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                raise RuntimeError("Unexpected checkpoint format (expected dict with model_state_dict).")
            model.load_state_dict(state, strict=True)
            model.eval()
        except Exception as e:
            entry.update({"status": "skipped", "reason": f"Failed to load model/checkpoint: {e}"})
            results.append(entry)
            continue

        try:
            latency_ms = latency_fn(
                model=model,
                input_size=input_size_chw,
                device=device,
                warmup=int(warmup),
                iters=int(iters),
                batch_size=int(batch_size),
            )
            entry.update(
                {
                    "status": "ok",
                    "latency_ms_per_image": float(latency_ms),
                }
            )
        except Exception as e:
            entry.update({"status": "error", "reason": f"Latency measurement failed: {e}"})

        results.append(entry)

    payload = {
        "meta": collect_env_meta(device),
        "settings": {
            "trained_root": str(Path(trained_root).resolve()),
            "dataset_choice": dataset_choice,
            "selected_dataset": dataset,
            "variants": list(variants),
            "dtype": "fp32",
            "batch_size": int(batch_size),
            "warmup": int(warmup),
            "iters": int(iters),
        },
        "results": results,
    }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local latency benchmark for trained checkpoints")
    p.add_argument(
        "--trained_root",
        type=str,
        default=str(_REPO_ROOT / "Trained Models"),
        help="Root containing <dataset>/<variant>/seed_<n>/ folders.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=str(_REPO_ROOT / "Trained Models" / "latency_results.json"),
        help="Write JSON report to this path.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="Dataset to benchmark (e.g. cifar100, tiny_imagenet). Use 'auto' to pick one.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for latency measurement.",
    )
    p.add_argument("--warmup", type=int, default=50, help="Warm-up iterations.")
    p.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size (thesis uses 1).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = resolve_device(args.device)

    payload = run_latency_benchmark(
        trained_root=Path(args.trained_root),
        dataset_choice=str(args.dataset),
        device=device,
        warmup=int(args.warmup),
        iters=int(args.iters),
        batch_size=int(args.batch_size),
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Print minimal summary for terminal usage.
    ok = [r for r in payload["results"] if r.get("status") == "ok"]
    skipped = [r for r in payload["results"] if r.get("status") == "skipped"]
    errors = [r for r in payload["results"] if r.get("status") == "error"]
    print(f"Wrote: {out_path}")
    print(f"Selected dataset: {payload['settings'].get('selected_dataset')}")
    print(f"ok={len(ok)} skipped={len(skipped)} error={len(errors)}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

