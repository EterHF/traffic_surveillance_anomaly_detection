from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from src.schemas import TrackObject


def plot_trajectories(tracks: list[TrackObject], width: int = 1280, height: int = 720) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for t in tracks:
        by_id[t.track_id].append(t)

    for tid, seq in by_id.items():
        color = ((tid * 37) % 255, (tid * 67) % 255, (tid * 97) % 255)
        for i in range(1, len(seq)):
            p1 = (int(seq[i - 1].cx), int(seq[i - 1].cy))
            p2 = (int(seq[i].cx), int(seq[i].cy))
            cv2.line(canvas, p1, p2, color, 2)
    return canvas
