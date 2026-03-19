"""Environment/version collection utilities for reproducible runs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvInfo:
    created_at_utc: str
    python_version: str
    platform: str
    machine: str
    processor: str
    torch_version: str | None
    torchvision_version: str | None
    cuda_available: bool | None
    cuda_version: str | None
    cudnn_version: int | None
    device_count: int | None
    git_commit: str | None


def _safe_import_versions() -> tuple[str | None, str | None, dict[str, Any]]:
    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cudnn_version = (
            int(torch.backends.cudnn.version())
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available()
            else None
        )
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        torch_version = None
        cuda_available = None
        cuda_version = None
        cudnn_version = None
        device_count = None

    try:
        import torchvision  # type: ignore

        torchvision_version = getattr(torchvision, "__version__", None)
    except Exception:
        torchvision_version = None

    extra: dict[str, Any] = {}
    return torch_version, torchvision_version, {
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "device_count": device_count,
        **extra,
    }


def _git_commit_hash(repo_root: str | Path | None = None) -> str | None:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(
            cmd,
            cwd=str(Path(repo_root)) if repo_root is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def collect_env_info(repo_root: str | Path | None = None) -> EnvInfo:
    torch_version, torchvision_version, torch_bits = _safe_import_versions()
    now = datetime.now(timezone.utc).isoformat()
    return EnvInfo(
        created_at_utc=now,
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        cuda_available=torch_bits["cuda_available"],
        cuda_version=torch_bits["cuda_version"],
        cudnn_version=torch_bits["cudnn_version"],
        device_count=torch_bits["device_count"],
        git_commit=_git_commit_hash(repo_root=repo_root),
    )


def env_info_as_dict(repo_root: str | Path | None = None) -> dict[str, Any]:
    return asdict(collect_env_info(repo_root=repo_root))


def write_env_info_json(path: str | Path, repo_root: str | Path | None = None) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = env_info_as_dict(repo_root=repo_root)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path

