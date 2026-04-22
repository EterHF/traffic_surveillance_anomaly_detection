from abc import ABC, abstractmethod
from typing import Any
from src.schemas import Detection

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> list[Detection]:
        """
        Run object detection on a single frame.
        Args:
            frame: Typically a numpy ndarray (BGR image from OpenCV)
        Returns:
            A list of Detection objects.
        """
        raise NotImplementedError
