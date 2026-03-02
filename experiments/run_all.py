"""
Run training for all dataset configs.

Run from project root:
    python experiments/run_all.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    configs = sorted(Path(project_root / "configs").glob("*.yaml"))

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg.name}")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, "experiments/train_cifar10.py", "--config", str(cfg)],
            cwd=project_root,
        )
        if result.returncode != 0:
            print(f"Failed: {cfg.name}")
            sys.exit(result.returncode)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
