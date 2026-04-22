import cv2
import numpy as np
from src.schemas import TrackObject

class TrajectorySketcher:
    def __init__(self):
        pass
        
    def sketch(self, tracks: List[TrackObject], bg_image: np.ndarray = None) -> np.ndarray:
        """
        Draw a sketch of object trajectories over a background.
        """
        if bg_image is None:
            bg_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            
        sketch = bg_image.copy()
        for i in range(1, len(tracks)):
            pt1 = (int(tracks[i-1].cx), int(tracks[i-1].cy))
            pt2 = (int(tracks[i].cx), int(tracks[i].cy))
            cv2.line(sketch, pt1, pt2, (0, 0, 255), 2)
            
        return sketch
