import os
import json
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_LATENCY_TEST", "").strip() not in {"1", "true", "TRUE", "yes", "YES"},
    reason="Local-only latency benchmark (set RUN_LOCAL_LATENCY_TEST=1 to enable).",
)
def test_local_latency_benchmark_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    This test is intentionally skipped by default because performance tests are noisy
    and hardware-dependent. When enabled, it only asserts that the benchmark runs and
    emits a JSON with the expected shape.
    """
    import torch
    import torch.nn as nn

    import scripts.latency_local_test as latency_mod

    # Build a tiny synthetic trained-models tree so the test does not depend on
    # real local checkpoints and does not need to run a real forward pass.
    trained_root = tmp_path / "Trained Models"
    dataset = "cifar10"
    variants = ["baseline", "dualconv", "eca", "hybrid"]
    seeds = {"baseline": "seed_42", "dualconv": "seed_123", "eca": "seed_3407", "hybrid": "seed_42"}

    for v in variants:
        run_dir = trained_root / dataset / v / seeds[v]
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        # metrics.json used for best-seed selection + input_size_chw
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "model": v,
                    "seed": int(seeds[v].split("_", 1)[1]),
                    "best_val": {"epoch": 1, "val_acc": 0.5, "val_loss": 1.0},
                    "model_profile": {"input_size_chw": [3, 32, 32]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # config.json is required by the runner but we will monkeypatch build_model.
        (run_dir / "logs" / "config.json").write_text(
            json.dumps({"dataset": dataset, "model": v, "num_classes": 10, "width_multiplier": 1.0}) + "\n",
            encoding="utf-8",
        )
        # Touch a checkpoint file so the runner doesn't skip.
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"not-a-real-checkpoint")

    class DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def fake_build_model(_cfg: dict) -> nn.Module:
        return DummyModel()

    def fake_torch_load(_path: Path, *args, **kwargs):
        _ = args, kwargs
        # Empty state dict matches DummyModel (no params/buffers).
        return {"model_state_dict": {}}

    called: list[dict] = []

    def fake_latency_fn(*, model, input_size, device, warmup, iters, batch_size) -> float:
        # Assert the thesis-style parameters were threaded through correctly.
        assert warmup == 50
        assert iters == 200
        assert batch_size == 1
        assert input_size == (3, 32, 32)
        assert str(device) == "cpu"
        called.append({"ok": True})
        return 12.34

    monkeypatch.setattr(latency_mod, "build_model", fake_build_model)
    monkeypatch.setattr(latency_mod.torch, "load", fake_torch_load)

    payload = latency_mod.run_latency_benchmark(
        trained_root=trained_root,
        dataset_choice=dataset,
        device=torch.device("cpu"),
        warmup=50,
        iters=200,
        batch_size=1,
        latency_fn=fake_latency_fn,
    )

    assert "meta" in payload and isinstance(payload["meta"], dict)
    assert "settings" in payload and isinstance(payload["settings"], dict)
    assert "results" in payload and isinstance(payload["results"], list)
    assert payload["settings"]["dtype"] == "fp32"
    assert payload["settings"]["batch_size"] == 1

    assert called, "Expected fake latency function to be called at least once."

    variants = {r.get("variant") for r in payload["results"]}
    assert {"baseline", "dualconv", "eca", "hybrid"}.issubset(variants)

    # Write a json artifact for debugging when running locally.
    out = tmp_path / "latency_results.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

