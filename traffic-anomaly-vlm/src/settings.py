from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# DEBUG
# import sys
# if __package__ in (None, ""):
#     project_root = Path(__file__).resolve().parent.parent
#     if str(project_root) not in sys.path:
#         sys.path.insert(0, str(project_root))

# import all config class
from src.features.feature_components.scene import SceneFeatureConfig
from src.proposals.tree_pipeline_components import NodeSelectorConfig, TreeBuildConfig
from src.proposals.boundary_detector import BoundaryDetectorConfig
from src.perception.detector_tracker import DetectorTrackerConfig
from src.vlm.model_loader import VLMConfig


cls_map = {
    "scene_features": SceneFeatureConfig,
    "boundary_detector": BoundaryDetectorConfig,
    "tree_build_config": TreeBuildConfig,
    "node_selector": NodeSelectorConfig,
    "perception": DetectorTrackerConfig,
    "vlm": VLMConfig,
}

class Settings:
    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        for k, v in raw.items():
            if isinstance(v, dict):
                setattr(self, k, Settings(v))
            else:
                setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return self._raw


def load_settings(config_path: str = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _merge_dict(raw, overrides)

    return Settings(raw)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-video", default="")
    return parser.parse_args()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def instantiate_from_config(config: Settings) -> dict[str, Any]:
    instances = {}
    for key, cls in cls_map.items():
        if hasattr(config, key):
            section = getattr(config, key)
            if hasattr(section, "to_dict"):
                payload = section.to_dict()
            else:
                payload = {k: v for k, v in vars(section).items() if not k.startswith("_")}
            instances[key] = cls(**payload)
    return instances


if __name__ == "__main__":
    args = parse_cli()
    settings = load_settings(args.config)
    print(settings.to_dict())
    config_instances = instantiate_from_config(settings)
    for k, v in config_instances.items():
        print(f"{k}: {type(v)}\n")
