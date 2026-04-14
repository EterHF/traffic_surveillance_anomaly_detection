from __future__ import annotations


class BoundaryDetector:
    def __init__(self, high: float = 1.0, low: float = 0.5):
        self.high = high
        self.low = low

    def detect(self, scores: list[float]) -> list[tuple[int, int]]:
        events: list[tuple[int, int]] = []
        in_event = False
        start = 0
        for i, s in enumerate(scores):
            if not in_event and s >= self.high:
                in_event = True
                start = i
            elif in_event and s < self.low:
                in_event = False
                events.append((start, i - 1))
        if in_event:
            events.append((start, len(scores) - 1))
        return events
