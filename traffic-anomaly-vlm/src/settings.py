from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


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
