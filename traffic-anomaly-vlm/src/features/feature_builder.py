from __future__ import annotations

from src.features.scene_features import build_scene_features
from src.features.track_features import build_track_features
from src.schemas import WindowFeature


class FeatureBuilder:
    def __init__(
        self,
        window_size: int,
        window_step: int,
        track_fit_degree: int = 2,
        track_history_len: int | None = None,
    ):
        self.window_size = window_size
        self.window_step = window_step
        self.track_fit_degree = int(track_fit_degree)
        self.track_history_len = track_history_len
        self.window_id = 0

    def ready(self, frame_ids: list[int], track_window: list[list]) -> bool:
        return len(frame_ids) >= self.window_size and len(track_window) >= self.window_size

    def build(self, frame_ids: list[int], track_window: list[list]) -> WindowFeature:
        track_feats = build_track_features(
            track_window,
            fit_degree=self.track_fit_degree,
            history_len=self.track_history_len,
        )
        scene_feats = build_scene_features(track_window)
        merged = {**track_feats, **scene_feats}

        selected_keys = [
            "dir_turn_rates_sum",
            "pred_center_sum_err",
            "pred_iou_sum_err",
            "speed_turn_rates_sum",
            "turn_active_ratio"
        ]
        weights = {
            "dir_turn_rates_sum": 0.25,
            "pred_center_sum_err": 0.25,
            "pred_iou_sum_err": 0.25,
            "speed_turn_rates_sum": 0.25,
            "turn_active_ratio": 0.25,
        }
        trigger_score = 0.0
        for key in selected_keys:
            val = merged.get(key, 0.0)
            weight = weights.get(key, 0.0)
            trigger_score += val * weight

        wf = WindowFeature(
            window_id=self.window_id,
            start_frame=frame_ids[0],
            end_frame=frame_ids[-1],
            feature_dict=merged,
            trigger_score=float(trigger_score),
        )
        self.window_id += 1
        return wf
