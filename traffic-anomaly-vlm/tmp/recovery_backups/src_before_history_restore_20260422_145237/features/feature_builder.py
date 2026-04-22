from __future__ import annotations

import numpy as np

from src.features.feature_components.scene import SceneFeatureConfig, SceneFeatureExtractor
from src.features.feature_components.track import build_track_features
from src.schemas import WindowFeature


OBJECT_FEATURE_WEIGHTS = {
    "dir_turn_rates_sum": 0.25,
    "pred_center_sum_err": 0.25,
    "pred_iou_sum_err": 0.25,
    "speed_turn_rates_sum": 0.25,
    "turn_active_ratio": 0.25,
    "collision_warning": 0.45,
    "pred_center_max_err": 0.20,
    "pred_iou_max_err": 0.15,
    "disappear_far_sum": 0.10,
}

SCENE_FEATURE_WEIGHTS = {
    "count_std": 0.15,
    "count_delta": 0.10,
    "density_std": 0.15,
    "density_delta": 0.10,
    "lowres_eventness": 0.20,
    "track_layout_eventness": 0.20,
    "clip_eventness": 0.20,
    "raft_eventness": 0.20,
}

TRIGGER_BRANCH_WEIGHTS = {
    "object_score": 0.7,
    "scene_score": 0.3,
}


def _normalized_weighted_sum(features: dict[str, float], weights: dict[str, float]) -> float:
    valid = {key: float(weight) for key, weight in weights.items() if float(weight) > 0.0}
    if not valid:
        return 0.0
    weight_sum = sum(valid.values())
    score = sum(float(features.get(key, 0.0)) * weight for key, weight in valid.items())
    return float(score / max(1e-6, weight_sum))


def compute_object_score(features: dict[str, float]) -> float:
    return _normalized_weighted_sum(features, OBJECT_FEATURE_WEIGHTS)


def compute_scene_score(features: dict[str, float], scene_cfg: SceneFeatureConfig) -> float:
    return _normalized_weighted_sum(features, SCENE_FEATURE_WEIGHTS)


class FeatureBuilder:
    """Build flat object/scene trigger scores for each offline window."""

    def __init__(
        self,
        window_size: int,
        window_step: int,
        track_fit_degree: int = 2,
        track_history_len: int | None = None,
        scene_cfg: SceneFeatureConfig | None = None,
    ):
        self.window_size = window_size
        self.window_step = window_step
        self.track_fit_degree = int(track_fit_degree)
        self.track_history_len = track_history_len
        self.scene_cfg = scene_cfg or SceneFeatureConfig()
        self.scene_extractor = SceneFeatureExtractor(self.scene_cfg)
        self.window_id = 0

    def ready(
        self,
        frame_ids: list[int],
        track_window: list[list],
        image_window: list[np.ndarray] | None = None,
    ) -> bool:
        if image_window is not None and len(image_window) < self.window_size:
            return False
        return len(frame_ids) >= self.window_size and len(track_window) >= self.window_size

    def build(
        self,
        frame_ids: list[int],
        track_window: list[list],
        image_window: list[np.ndarray] | None = None,
    ) -> WindowFeature:
        track_feats = build_track_features(
            track_window,
            fit_degree=self.track_fit_degree,
            history_len=self.track_history_len,
        )
        scene_feats = self.scene_extractor.compute(track_window, images_bgr=image_window)
        merged = {**track_feats, **scene_feats}
        object_score = compute_object_score(merged)
        scene_score = compute_scene_score(merged, self.scene_cfg)
        trigger_score = _normalized_weighted_sum(
            {
                "object_score": object_score,
                "scene_score": scene_score,
            },
            TRIGGER_BRANCH_WEIGHTS,
        )
        merged["object_score"] = float(object_score)
        merged["scene_score"] = float(scene_score)
        merged["trigger_score"] = float(trigger_score)

        wf = WindowFeature(
            window_id=self.window_id,
            start_frame=frame_ids[0],
            end_frame=frame_ids[-1],
            feature_dict=merged,
            trigger_score=float(trigger_score),
        )
        self.window_id += 1
        return wf
