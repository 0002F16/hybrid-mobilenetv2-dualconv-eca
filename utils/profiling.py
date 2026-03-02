"""
Profiling utilities: parameters, FLOPs/MACs, latency, model size.

Thesis 3.7.3: FLOPs computed with fvcore; params = trainable only.
All functions are deterministic and do not depend on global state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(
    model: nn.Module,
    input_size: tuple[int, ...],
    device: torch.device,
    method: str = "fvcore",
) -> dict[str, Any]:
    """
    Compute FLOPs and MACs for one forward pass.

    input_size: (C, H, W), e.g. (3, 32, 32). Batch size is 1.
    method: "fvcore" (thesis default) or "thop". Tries primary first, then fallback.
    Returns dict with macs, flops, method_used. FLOPs = 2 * MACs (multiply-add counts as 2 ops).
    """
    model = model.to(device)
    model.eval()
    dummy = torch.zeros(1, *input_size, device=device, dtype=torch.float32)

    macs = 0
    flops = 0
    method_used = "none"

    # Thesis 3.7.3: FLOPs computed using fvcore. Use fvcore first, thop as fallback.
    if method == "thop":
        try:
            import thop
            macs, flops = thop.profile(model, inputs=(dummy,), verbose=False)
            macs = int(macs)
            flops = int(flops)
            # thop returns FLOPs; MACs ≈ flops/2 for multiply-add
            if flops and not macs:
                macs = flops // 2
            elif macs and not flops:
                flops = 2 * macs
            method_used = "thop"
        except Exception:
            pass
        if method_used == "none":
            try:
                from fvcore.nn import FlopCountAnalysis
                flop_counter = FlopCountAnalysis(model, dummy)
                flops = int(flop_counter.total())
                macs = flops // 2  # FLOPs = 2 * MACs
                method_used = "fvcore"
            except Exception:
                pass
    else:
        # Default: fvcore first (thesis), then thop
        try:
            from fvcore.nn import FlopCountAnalysis
            flop_counter = FlopCountAnalysis(model, dummy)
            flops = int(flop_counter.total())
            macs = flops // 2  # FLOPs = 2 * MACs for multiply-add
            method_used = "fvcore"
        except Exception:
            try:
                import thop
                macs, flops = thop.profile(model, inputs=(dummy,), verbose=False)
                macs = int(macs)
                flops = int(flops)
                if flops and not macs:
                    macs = flops // 2
                elif macs and not flops:
                    flops = 2 * macs
                method_used = "thop"
            except Exception:
                pass

    if method_used == "none":
        raise RuntimeError(
            "FLOPs profiling requires fvcore or thop. Install with: pip install fvcore  # or pip install thop"
        )

    return {"macs": macs, "flops": flops, "method_used": method_used}


def measure_latency(
    model: nn.Module,
    input_size: tuple[int, ...],
    device: torch.device,
    warmup: int = 30,
    iters: int = 200,
    batch_size: int = 1,
) -> float:
    """
    Measure mean inference latency in ms per image.

    Uses torch.no_grad() and torch.cuda.synchronize() when device is CUDA.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.zeros(batch_size, *input_size, device=device, dtype=torch.float32)
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if is_cuda:
                torch.cuda.synchronize()

        if is_cuda:
            torch.cuda.synchronize()

        import time
        if is_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        else:
            t0 = time.perf_counter()

        for _ in range(iters):
            _ = model(dummy)
            if is_cuda:
                torch.cuda.synchronize()

        if is_cuda:
            end_ev.record()
            torch.cuda.synchronize()
            total_ms = start_ev.elapsed_time(end_ev)
        else:
            total_ms = (time.perf_counter() - t0) * 1000.0

    num_images = iters * batch_size
    return total_ms / num_images


def measure_model_size_mb(model: nn.Module, tmp_path: str | Path | None = None) -> float:
    """
    Save state_dict to disk and return size in MB.

    If tmp_path is None, uses a temporary file (caller does not need to pass path).
    """
    if tmp_path is None:
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_path = fd.name
        fd.close()
        try:
            torch.save(model.state_dict(), tmp_path)
            size_bytes = Path(tmp_path).stat().st_size
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        path = Path(tmp_path)
        torch.save(model.state_dict(), path)
        size_bytes = path.stat().st_size
    return size_bytes / (1024.0 * 1024.0)
