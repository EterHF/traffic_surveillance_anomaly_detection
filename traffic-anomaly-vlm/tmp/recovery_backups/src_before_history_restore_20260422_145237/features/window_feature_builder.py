from src.features.base import BaseFeatureBuilder
from src.schemas import WindowFeature, TrackObject
from typing import List, Dict, Any

class WindowFeatureBuilder(BaseFeatureBuilder):
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.window_idx = 0

    def ready(self, frame_buffer: List[Any], track_buffer: List[List[TrackObject]]) -> bool:
        return len(frame_buffer) >= self.window_size

    def build(self, frame_buffer: List[Any], track_buffer: List[List[TrackObject]]) -> WindowFeature:
        # Simplistic feature building for the skeleton
        
        start_frame = track_buffer[0][0].frame_id if track_buffer and track_buffer[0] else 0
        end_frame = track_buffer[-1][0].frame_id if track_buffer and track_buffer[-1] else 0
        
        feature_dict = {
            "num_frames": len(frame_buffer),
            "num_tracks": sum(len(tb) for tb in track_buffer)
        }
        
        feat = WindowFeature(
            window_id=self.window_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            feature_dict=feature_dict,
            trigger_score=0.0 # Placeholder
        )
        self.window_idx += 1
        return feat
