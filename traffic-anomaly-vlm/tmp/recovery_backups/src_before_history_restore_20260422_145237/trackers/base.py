from abc import ABC, abstractmethod
from src.schemas import Detection, TrackObject

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame_id: int) -> list[TrackObject]:
        """
        Update the tracker with new detections for the current frame.
        Args:
            detections: List of Detection objects for the current frame.
            frame_id: Original or sampled frame index.
        Returns:
            A list of TrackObject containing tracked identities and bounding boxes.
        """
        raise NotImplementedError
