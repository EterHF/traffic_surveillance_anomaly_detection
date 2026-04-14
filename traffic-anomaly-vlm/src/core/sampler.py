from __future__ import annotations

from typing import Iterator

import numpy as np


class FrameSampler:
    def __init__(self, source_fps: float, target_fps: float):
        self.source_fps = max(source_fps, 1.0)
        self.target_fps = max(target_fps, 1.0)
        self.stride = max(1, int(round(self.source_fps / self.target_fps)))

    def sample(self, iterator: Iterator[tuple[int, np.ndarray]]) -> Iterator[tuple[int, np.ndarray]]:
        for frame_id, frame in iterator:
            if frame_id % self.stride == 0:
                yield frame_id, frame
