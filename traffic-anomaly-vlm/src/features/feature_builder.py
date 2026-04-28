from __future__ import annotations

import numpy as np

from src.features.feature_components.scene import SceneFeatureConfig, SceneFeatureExtractor
from src.features.feature_components.track import build_track_features
from src.schemas import WindowFeature


OBJECT_FEATURE_WEIGHTS = {
    "dir_turn_rates_sum": 1.0,
    "speed_turn_rates_sum": 1.0,
    "turn_active_ratio": 0.2,
    "pred_center_sum_err": 1.0,
    "pred_iou_sum_err": 1.0,
    "collision_warning": 0.2,
    "pred_center_max_err": 0.0,
    "pred_iou_max_err": 0.0,
    "disappear_far_sum": 0.2,
}

SCENE_FEATURE_WEIGHTS = {
    "count_std": 0.0,
    "count_delta": 0.0,
    "density_std": 0.0,
    "density_delta": 0.0,
    "lowres_eventness": 0.0,
    "track_layout_eventness": 0.0,
    "clip_eventness": 0.0,
    "raft_eventness": 0.0,
}

FEATURE_Z_DEADZONE = 1.0
FEATURE_Z_CAP = 8.0
FEATURE_HIGH_PERCENTILE = 98.0
GATE_EXPERT_THRESHOLD = 0.25
GATE_EVIDENCE_THRESHOLD = 0.5
GATE_SIGMOID_K = 6.0
GATE_TOP_K = 3
GATE_MEDIAN_WINDOW = 3
GATE_MIN_EXPERT_CONFIDENCE = 0.05
GATE_MIN_ACTIVE_EXPERTS = 2
GATE_STRONG_EVIDENCE = 2.0
GATE_SINGLE_EXPERT_SCALE = 0.25
GATE_BACKGROUND_PERCENTILE = 50.0
GATE_SHOULDER_PERCENTILE = 80.0
GATE_PEAK_PROMINENCE = 0.5
GATE_PEAK_DOMINANCE_RATIO = 2.0
GATE_ACTIVE_PROMINENCE_RATIO = 0.5
GATE_MAX_ACTIVE_RATIO = 0.35


def _weighted_sum(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(
        sum(
            float(features.get(key, 0.0)) * float(weight)
            for key, weight in weights.items()
            if float(weight) > 0.0
        )
    )


def _empty_score_curves(length: int) -> dict[str, np.ndarray]:
    length = max(0, int(length))

    def zeros() -> np.ndarray:
        return np.zeros((length,), dtype=np.float32)

    return {
        "trigger_score": zeros(),
        "gate_global_evidence": zeros(),
        "gate_null_weight": np.ones((length,), dtype=np.float32),
        "gate_active_experts": zeros(),
    }


def _median_filter_1d(values: np.ndarray, window: int = GATE_MEDIAN_WINDOW) -> np.ndarray:
    window = max(1, int(window))
    if values.size == 0 or window <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(values.size):
        out[idx] = float(np.median(padded[idx : idx + window]))
    return out


def _scale_salience(salience: np.ndarray) -> np.ndarray:
    high = max(float(np.percentile(salience, FEATURE_HIGH_PERCENTILE)), 1e-6)
    return np.clip(salience / high, 0.0, 1.0).astype(np.float32)


def _evidence_curves(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = values.astype(np.float32, copy=True)
    if values.size == 0:
        return values, values, values, values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1e-6, 1.4826 * mad)
    # z_raw is the expert evidence after a dead zone; short normal jitter stays 0.
    z_raw = np.clip(
        np.maximum((values - median) / sigma - FEATURE_Z_DEADZONE, 0.0),
        0.0,
        FEATURE_Z_CAP,
    ).astype(np.float32)
    z_persistent = _median_filter_1d(z_raw, GATE_MEDIAN_WINDOW)
    return (
        z_raw,
        z_persistent,
        _scale_salience(z_raw),
        _scale_salience(z_persistent),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _salience_peak_stats(salience: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    peak = float(np.percentile(salience, FEATURE_HIGH_PERCENTILE))
    background = float(np.percentile(salience, GATE_BACKGROUND_PERCENTILE))
    shoulder = float(np.percentile(salience, GATE_SHOULDER_PERCENTILE))
    peak_prominence = max(0.0, peak - background)
    shoulder_prominence = max(shoulder - background, 0.0)
    peak_dominance = peak_prominence / max(shoulder_prominence, 1e-6)
    active_threshold = max(
        GATE_EXPERT_THRESHOLD,
        background + GATE_ACTIVE_PROMINENCE_RATIO * peak_prominence,
    )
    active = salience > active_threshold
    return peak, background, peak_prominence, peak_dominance, active


def _is_sparse_peak_response(salience: np.ndarray) -> bool:
    if salience.size == 0:
        return False

    _, _, peak_prominence, peak_dominance, _ = _salience_peak_stats(salience)
    if peak_prominence <= GATE_PEAK_PROMINENCE:
        return False
    return peak_dominance >= GATE_PEAK_DOMINANCE_RATIO


def _expert_video_confidence(salience: np.ndarray, base_weight: float, *, sparse_peak: bool = False) -> float:
    """Score whether one expert is representative for the whole video."""
    if salience.size == 0 or float(base_weight) <= 0.0:
        return 0.0

    peak, _, peak_prominence, peak_dominance, active = _salience_peak_stats(salience)
    if peak_prominence <= GATE_PEAK_PROMINENCE:
        return 0.0

    active_count = int(np.sum(active))
    if active_count <= 0:
        return 0.0

    mean_active = float(np.mean(salience[active]))
    active_ratio = float(active_count) / max(1.0, float(salience.size))
    if sparse_peak:
        # For sparse peak experts, do not hard-reject by active ratio; the
        # active mask can be sensitive to threshold choice. Rely on peak
        # dominance instead, and use coverage only as a mild penalty.
        coverage = max(0.2, 1.0 - 0.5 * active_ratio)
        dominance_bonus = min(2.0, peak_dominance / GATE_PEAK_DOMINANCE_RATIO)
    else:
        coverage = max(0.1, 1.0 - active_ratio / max(1e-6, GATE_MAX_ACTIVE_RATIO))
        dominance_bonus = 1.0
    peakiness = np.clip(peak_prominence / max(peak, 1e-6), 0.0, 1.0)

    return (
        float(base_weight)
        * peak_prominence
        * (0.5 * peak_prominence + 0.5 * mean_active)
        * coverage
        * float(peakiness)
        * float(dominance_bonus)
    )


def build_window_score_curves(windows: list[WindowFeature]) -> dict[str, np.ndarray]:
    """Aggregate raw per-window features into a gated trigger score curve."""

    if not windows:
        return _empty_score_curves(0)

    feature_rows = [window.feature_dict for window in windows]
    num_windows = len(feature_rows)

    expert_weights = {
        **{key: float(weight) for key, weight in OBJECT_FEATURE_WEIGHTS.items() if float(weight) > 0.0},
        **{key: float(weight) for key, weight in SCENE_FEATURE_WEIGHTS.items() if float(weight) > 0.0},
    }
    if not expert_weights:
        return _empty_score_curves(num_windows)

    expert_keys = list(expert_weights.keys())
    salience_curves: list[np.ndarray] = []
    expert_scores: list[np.ndarray] = []
    sparse_peak_experts: list[bool] = []
    for key in expert_keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in feature_rows], dtype=np.float32)
        raw_salience, persistent_salience, raw_score, persistent_score = _evidence_curves(values)
        sparse_peak = _is_sparse_peak_response(raw_salience)
        salience_curves.append(raw_salience if sparse_peak else persistent_salience)
        expert_scores.append(raw_score if sparse_peak else persistent_score)
        sparse_peak_experts.append(bool(sparse_peak))

    salience_mat = np.stack(salience_curves, axis=0)  # [N, T]
    score_mat = np.stack(expert_scores, axis=0)  # [N, T]
    base_weights = np.asarray([expert_weights[key] for key in expert_keys], dtype=np.float32)[:, None]
    sparse_peak_mask = np.asarray(sparse_peak_experts, dtype=bool)
    expert_confidence = np.asarray(
        [
            _expert_video_confidence(
                salience_mat[idx],
                float(base_weights[idx, 0]),
                sparse_peak=bool(sparse_peak_mask[idx]),
            )
            for idx in range(len(expert_keys))
        ],
        dtype=np.float32,
    )
    selected_indices = [
        int(idx)
        for idx in np.argsort(expert_confidence)[::-1]
        if float(expert_confidence[int(idx)]) >= GATE_MIN_EXPERT_CONFIDENCE
    ]
    if GATE_TOP_K > 0:
        selected_indices = selected_indices[:GATE_TOP_K]

    if not selected_indices:
        curves = _empty_score_curves(num_windows)
        curves["gate_global_evidence"] = np.max(salience_mat, axis=0).astype(np.float32)
        return curves

    selected_salience = salience_mat[selected_indices, :]
    selected_scores = score_mat[selected_indices, :]
    selected_base_weights = base_weights[selected_indices, :]
    selected_sparse_peak_mask = sparse_peak_mask[selected_indices]

    # Sparse gate is now applied only inside representative experts selected at
    # video level, instead of re-routing to arbitrary top-k experts every frame.
    activations = np.maximum(selected_salience - GATE_EXPERT_THRESHOLD, 0.0) * selected_base_weights
    activation_sum = np.sum(activations, axis=0)
    weights = activations / np.maximum(activation_sum[None, :], 1e-6)
    global_evidence = np.max(selected_salience, axis=0)
    active_experts = np.sum(activations > 0.0, axis=0).astype(np.float32)
    # Null-aware evidence gate: when no expert has persistent evidence, rho is
    # near zero and the null expert implicitly owns the remaining probability.
    rho = _sigmoid(GATE_SIGMOID_K * (global_evidence - GATE_EVIDENCE_THRESHOLD))
    rho = np.where(global_evidence >= GATE_EXPERT_THRESHOLD, rho, 0.0).astype(np.float32)
    # Isolated single-expert spikes are often detector/tracker noise. Keep them
    # only when the evidence is very strong; otherwise require cross-support.
    sparse_peak_active = (
        np.any(activations[selected_sparse_peak_mask, :] > 0.0, axis=0)
        if np.any(selected_sparse_peak_mask)
        else np.zeros((num_windows,), dtype=bool)
    )
    weak_single_expert = (active_experts < GATE_MIN_ACTIVE_EXPERTS) & (global_evidence < GATE_STRONG_EVIDENCE)
    weak_single_expert = weak_single_expert & ~sparse_peak_active
    rho = np.where(weak_single_expert, rho * GATE_SINGLE_EXPERT_SCALE, rho).astype(np.float32)

    trigger_score = (rho * np.sum(weights * selected_scores, axis=0)).astype(np.float32)
    return {
        "trigger_score": trigger_score,
        "gate_global_evidence": global_evidence.astype(np.float32),
        "gate_null_weight": (1.0 - rho).astype(np.float32),
        "gate_active_experts": active_experts,
    }


class FeatureBuilder:
    """Build flat trigger features for each offline window."""

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
        scene_pair_slices: dict[str, np.ndarray] | None = None,
    ) -> WindowFeature:
        track_feats = build_track_features(
            track_window,
            fit_degree=self.track_fit_degree,
            history_len=self.track_history_len,
        )
        scene_feats = self.scene_extractor.compute(
            track_window,
            images_bgr=image_window,
            lowres_weight=SCENE_FEATURE_WEIGHTS["lowres_eventness"],
            layout_weight=SCENE_FEATURE_WEIGHTS["track_layout_eventness"],
            clip_weight=SCENE_FEATURE_WEIGHTS["clip_eventness"],
            raft_weight=SCENE_FEATURE_WEIGHTS["raft_eventness"],
            lowres_pair=(scene_pair_slices or {}).get("lowres_pair"),
            layout_pair=(scene_pair_slices or {}).get("layout_pair"),
            clip_pair=(scene_pair_slices or {}).get("clip_pair"),
            raft_pair=(scene_pair_slices or {}).get("raft_pair"),
        )
        merged = {**track_feats, **scene_feats}
        trigger_score = _weighted_sum(
            merged,
            {
                **OBJECT_FEATURE_WEIGHTS,
                **SCENE_FEATURE_WEIGHTS,
            },
        )
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
