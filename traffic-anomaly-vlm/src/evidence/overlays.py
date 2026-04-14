from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.schemas import TrackObject


def draw_overlay(frame: np.ndarray, tracks: list[TrackObject], main_id: int | None) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox_xyxy)
        color = (0, 0, 255) if main_id is not None and t.track_id == main_id else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"id={t.track_id}:{t.cls_name}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def save_image(path: str, image: np.ndarray) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)
    return path
