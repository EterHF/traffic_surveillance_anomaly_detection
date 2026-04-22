from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas import WindowFeature


class BaseTrigger(ABC):
    @abstractmethod
    def score(self, window_feature: WindowFeature) -> float:
        raise NotImplementedError
