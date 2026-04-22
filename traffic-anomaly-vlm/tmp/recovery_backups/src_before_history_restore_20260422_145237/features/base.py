from abc import ABC, abstractmethod
from src.schemas import WindowFeature

class BaseFeatureBuilder(ABC):
    @abstractmethod
    def ready(self, frame_buffer: list, track_buffer: list) -> bool:
        """
        Check if the feature builder has enough data to build a feature.
        """
        pass

    @abstractmethod
    def build(self, frame_buffer: list, track_buffer: list) -> WindowFeature:
        """
        Build the feature object for the current window.
        """
        pass
