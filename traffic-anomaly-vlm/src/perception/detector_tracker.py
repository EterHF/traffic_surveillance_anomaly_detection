from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from ultralytics import YOLO
except Exception as _e:
    print(f"Error importing YOLO from ultralytics: {_e}")


@dataclass
class DetectorTrackerConfig:
    """Runtime options for Ultralytics detection + tracking."""

    model_path: str
    tracker: str = "bytetrack.yaml"
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] | None = None
    device: str | None = None


class DetectorTracker:
    """Thin wrapper around `YOLO.track` with a stable local API."""

    def __init__(
        self,
        cfg: DetectorTrackerConfig | None = None,
    ): 
        if cfg is None:
            raise ValueError("DetectorTracker requires a DetectorTrackerConfig")
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    def _track_kwargs(self, persist: bool) -> dict:
        kwargs = {
            "persist": persist,
            "tracker": self.cfg.tracker,
            "conf": self.cfg.conf,
            "iou": self.cfg.iou,
            "classes": self.cfg.classes,
            "verbose": False,
        }
        if self.cfg.device:
            kwargs["device"] = self.cfg.device
        return kwargs

    def track_video(self, source: str) -> Iterable:
        return self.model.track(
            source=source,
            stream=True,
            **self._track_kwargs(persist=True),
        )

    def track_frame(self, frame: np.ndarray, persist: bool = True):
        return self.model.track(
            source=frame,
            **self._track_kwargs(persist=bool(persist)),
        )
