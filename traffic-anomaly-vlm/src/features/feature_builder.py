from __future__ import annotations

from src.features.scene_features import build_scene_features
from src.features.track_features import build_track_features
from src.schemas import WindowFeature


TRACK_RISK_WEIGHTS = {
    "dir_turn_rates_sum": 0.25,
    "pred_center_sum_err": 0.25,
    "pred_iou_sum_err": 0.25,
    "speed_turn_rates_sum": 0.25,
    "turn_active_ratio": 0.25,
}

OBJECT_PRIOR_WEIGHTS = {
    "collision_warning": 0.45,
    "pred_center_max_err": 0.20,
    "pred_iou_max_err": 0.15,
    "turn_active_ratio": 0.10,
    "disappear_far_sum": 0.10,
}

SCENE_CONTEXT_WEIGHTS = {
    "count_std": 0.30,
    "count_delta": 0.20,
    "density_std": 0.30,
    "density_delta": 0.20,
}


def _weighted_sum(features: dict[str, float], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += float(features.get(key, 0.0)) * float(weight)
    return float(score)


def compute_track_risk_score(features: dict[str, float]) -> float:
    return _weighted_sum(features, TRACK_RISK_WEIGHTS)


def compute_object_prior_score(features: dict[str, float]) -> float:
    return _weighted_sum(features, OBJECT_PRIOR_WEIGHTS)


def compute_scene_context_score(features: dict[str, float]) -> float:
    return _weighted_sum(features, SCENE_CONTEXT_WEIGHTS)


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
        track_risk_score = compute_track_risk_score(merged)
        object_prior_score = compute_object_prior_score(merged)
        scene_context_score = compute_scene_context_score(merged)
        merged["track_risk_score"] = float(track_risk_score)
        merged["object_prior_score"] = float(object_prior_score)
        merged["scene_context_score"] = float(scene_context_score)

        wf = WindowFeature(
            window_id=self.window_id,
            start_frame=frame_ids[0],
            end_frame=frame_ids[-1],
            feature_dict=merged,
            # Keep trigger_score backward-compatible for the existing track branch.
            trigger_score=float(track_risk_score),
        )
        self.window_id += 1
        return wf
