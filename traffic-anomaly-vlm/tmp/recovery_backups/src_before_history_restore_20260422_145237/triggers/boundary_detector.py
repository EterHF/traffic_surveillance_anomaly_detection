from typing import List, Tuple

class BoundaryDetector:
    def __init__(self, high_thresh: float = 2.5, low_thresh: float = 1.0):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.in_event = False
        self.start_idx = -1
        
    def detect(self, scores: List[float]) -> List[Tuple[int, int]]:
        """
        Takes a list of scores over time and returns (start_idx, end_idx) tuples for events
        using simple hysteresis thresholding.
        """
        events = []
        in_event = False
        start_idx = -1
        
        for i, score in enumerate(scores):
            if not in_event and score >= self.high_thresh:
                in_event = True
                start_idx = i
            elif in_event and score < self.low_thresh:
                in_event = False
                events.append((start_idx, i - 1))
                
        if in_event:
            events.append((start_idx, len(scores) - 1))
            
        return events
