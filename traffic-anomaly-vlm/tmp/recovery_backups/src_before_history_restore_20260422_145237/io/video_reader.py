import cv2
import numpy as np
from typing import Iterator

class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yields (frame_id, frame_bgr)
        """
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1
            
        # Optional: reset for subsequent iterations
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
