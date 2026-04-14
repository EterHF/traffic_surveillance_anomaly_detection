from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import load_settings


def main() -> None:
    cfg = load_settings("configs/debug.yaml")
    required = [
        "src/schemas.py",
        "src/perception/detector_tracker.py",
        "src/vlm/model_loader.py",
        "src/pipeline/offline.py",
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    print("smoke test passed: structure and config load are OK")
    print(f"debug tracker: {cfg.perception.tracker}")


if __name__ == "__main__":
    main()
