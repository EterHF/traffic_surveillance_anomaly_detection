from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from ultralytics import YOLO
except Exception as _e:  # pragma: no cover
    YOLO = None
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None


class DetectorTracker:
    def __init__(
        self,
        model_path: str,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
        classes: list[int] | None = None,
    ):
        if YOLO is None:
            raise ImportError(
                f"Original error: {_IMPORT_ERR}"
            )
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.classes = classes

    def track_video(self, source: str) -> Iterable:
        return self.model.track(
            source=source,
            stream=True,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )

    def track_frame(self, frame: np.ndarray, persist: bool = True):
        return self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )
