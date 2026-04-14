import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.online import OnlinePipeline
from src.settings import load_settings


if __name__ == "__main__":
    cfg = load_settings("configs/default.yaml")
    pipe = OnlinePipeline(cfg)
    pipe.run(cfg.input_video)
