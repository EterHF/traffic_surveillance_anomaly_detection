from __future__ import annotations

from src.features.scene_features import build_scene_features
from src.features.track_features import build_track_features
from src.schemas import WindowFeature


class FeatureBuilder:
    def __init__(self, window_size: int, window_step: int):
        self.window_size = window_size
        self.window_step = window_step
        self.window_id = 0

    def ready(self, frame_ids: list[int], track_window: list[list]) -> bool:
        return len(frame_ids) >= self.window_size and len(track_window) >= self.window_size

    def build(self, frame_ids: list[int], track_window: list[list]) -> WindowFeature:
        track_feats = build_track_features(track_window)
        scene_feats = build_scene_features(track_window)
        merged = {**track_feats, **scene_feats}

        trigger_score = merged["speed_mean"] * 0.01 + merged["density_proxy"] * 0.00001

        wf = WindowFeature(
            window_id=self.window_id,
            start_frame=frame_ids[0],
            end_frame=frame_ids[-1],
            feature_dict=merged,
            trigger_score=float(trigger_score),
        )
        self.window_id += 1
        return wf
