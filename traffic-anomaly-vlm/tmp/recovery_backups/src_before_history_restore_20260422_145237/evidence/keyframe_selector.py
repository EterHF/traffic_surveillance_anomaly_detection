from typing import List
from src.schemas import EventProposal

class KeyframeSelector:
    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames
        
    def select(self, proposal: EventProposal) -> List[int]:
        """
        Select N keyframes from the proposal duration.
        Typically: start, end, peak, and one intermediate.
        """
        duration = proposal.end_frame - proposal.start_frame
        
        if duration <= self.num_frames:
            return list(range(proposal.start_frame, proposal.end_frame + 1))
            
        step = max(1, duration // self.num_frames)
        keyframes = [proposal.start_frame + i * step for i in range(self.num_frames)]
        
        # Ensure peak is included if relevant
        if proposal.peak_frame not in keyframes:
            keyframes[-1] = proposal.peak_frame
            
        return sorted(keyframes)
