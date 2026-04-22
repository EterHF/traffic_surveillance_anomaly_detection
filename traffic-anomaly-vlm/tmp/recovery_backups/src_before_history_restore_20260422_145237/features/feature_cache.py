from __future__ import annotations

from collections import deque
from typing import Optional

from src.schemas import WindowFeature


class FeatureCache:
    def __init__(self, max_windows: Optional[int] = None):
        # max_windows=None means keep all windows for full-video post analysis.
        self._max_windows = max_windows
        if max_windows is None:
            self._cache: list[WindowFeature] | deque[WindowFeature] = []
        else:
            self._cache = deque(maxlen=max_windows)

    def add(self, wf: WindowFeature) -> None:
        self._cache.append(wf)

    def all(self) -> list[WindowFeature]:
        return list(self._cache)
