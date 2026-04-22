from typing import Iterator
import numpy as np

class FrameSampler:
    """
    Downsamples a video stream from source_fps to target_fps.
    """
    def __init__(self, source_fps: float, target_fps: float):
        self.source_fps = source_fps
        self.target_fps = target_fps
        # Simple interval-based downsampling
        self.stride = max(1, int(round(self.source_fps / self.target_fps)))

    def sample(self, frame_iterator: Iterator[tuple[int, np.ndarray]]) -> Iterator[tuple[int, np.ndarray]]:
        """
        Takes an iterator that yields (frame_id, frame) and yields sampled frames.
        Returns original frame_id so downstream tasks can map back to video timeline.
        """
        for frame_id, frame in frame_iterator:
            if frame_id % self.stride == 0:
                yield frame_id, frame
