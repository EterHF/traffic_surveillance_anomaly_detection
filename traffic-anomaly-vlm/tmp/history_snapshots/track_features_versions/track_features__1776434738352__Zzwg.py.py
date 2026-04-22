from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from src.schemas import TrackObject


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def _frame_dims_from_window(window_tracks: list[list[TrackObject]]) -> tuple[float, float]:
    all_tracks = [t for f in window_tracks for t in f]
    if not all_tracks:
        return 1.0, 1.0
    fw = max((float(t.frame_w) for t in all_tracks), default=0.0)
    fh = max((float(t.frame_h) for t in all_tracks), default=0.0)
    if fw > 0.0 and fh > 0.0:
        return fw, fh
    x2_max = max((float(t.bbox_xyxy[2]) for t in all_tracks), default=1.0)
    y2_max = max((float(t.bbox_xyxy[3]) for t in all_tracks), default=1.0)
    return max(1.0, x2_max), max(1.0, y2_max)


def _is_static_full_window(seq: list[TrackObject], window_len: int) -> bool:
    # Ignore tracks that stay through the whole window and barely move.
    if len(seq) < max(3, window_len):
        return False
    seq = sorted(seq, key=lambda x: x.frame_id)
    d = math.hypot(seq[-1].cx - seq[0].cx, seq[-1].cy - seq[0].cy)
    scale = math.sqrt(max(1.0, sum(t.area for t in seq) / max(1, len(seq))))
    return (d / scale) < 0.05


def _predict_next_center(seq: list[TrackObject]) -> tuple[float, float] | None:
    if len(seq) < 2:
        return None
    seq = sorted(seq, key=lambda x: x.frame_id)
    max_hist = min(len(seq), 8)
    # Use median frame gap in recent history for stable extrapolation.
    dts = [
        max(1, int(seq[i].frame_id - seq[i - 1].frame_id))
        for i in range(max(1, len(seq) - max_hist + 1), len(seq))
    ]
    dt = sorted(dts)[len(dts) // 2] if dts else 1
    return _predict_with_history(seq, dt=dt, lookback=max_hist)


def _distance_to_nearest_edge(x: float, y: float, frame_w: float, frame_h: float) -> float:
    x_clamped = min(max(0.0, x), frame_w)
    y_clamped = min(max(0.0, y), frame_h)
    return float(min(x_clamped, frame_w - x_clamped, y_clamped, frame_h - y_clamped))


def _build_time_axis(seq: list[TrackObject]) -> np.ndarray:
    t = np.array([float(p.frame_id) for p in seq], dtype=np.float64)
    t = t - float(t[-1])
    return t


def _fit_poly_1d(t: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    deg = max(1, min(int(degree), int(len(t) - 1)))
    # Degrade degree when unique timestamps are insufficient.
    unique_t = np.unique(t)
    deg = min(deg, max(1, int(len(unique_t) - 1)))
    return np.polyfit(t, y, deg=deg)


def _poly_velocity_and_position(seq: list[TrackObject], degree: int = 2) -> tuple[float, float, float, float] | None:
    if len(seq) < 2:
        return None
    t = _build_time_axis(seq)
    x = np.array([float(p.cx) for p in seq], dtype=np.float64)
    y = np.array([float(p.cy) for p in seq], dtype=np.float64)

    try:
        px = _fit_poly_1d(t, x, degree)
        py = _fit_poly_1d(t, y, degree)
    except Exception:
        return None

    dpx = np.polyder(px)
    dpy = np.polyder(py)

    vx = float(np.polyval(dpx, 0.0))
    vy = float(np.polyval(dpy, 0.0))
    cx = float(np.polyval(px, 0.0))
    cy = float(np.polyval(py, 0.0))
    return vx, vy, cx, cy


def _velocity_at(seq: list[TrackObject], end_idx: int, span: int) -> tuple[float, float]:
    start_idx = max(0, end_idx - span + 1)
    sub = seq[start_idx : end_idx + 1]
    fitted = _poly_velocity_and_position(sub, degree=2)
    if fitted is None:
        return 0.0, 0.0
    vx, vy, _, _ = fitted
    return vx, vy


def _predict_with_history(history: list[TrackObject], dt: int, lookback: int) -> tuple[float, float] | None:
    if len(history) < 2:
        return None
    hist = sorted(history, key=lambda x: x.frame_id)
    max_hist = min(len(hist), max(3, lookback))
    sub = hist[-max_hist:]
    fitted = _poly_velocity_and_position(sub, degree=2)
    if fitted is None:
        return None
    _, _, cx0, cy0 = fitted

    # Time axis is relative to the last frame (t=0), so prediction at +dt.
    t = _build_time_axis(sub)
    x = np.array([float(p.cx) for p in sub], dtype=np.float64)
    y = np.array([float(p.cy) for p in sub], dtype=np.float64)
    try:
        px = _fit_poly_1d(t, x, degree=2)
        py = _fit_poly_1d(t, y, degree=2)
    except Exception:
        return None
    pred_cx = float(np.polyval(px, float(dt)))
    pred_cy = float(np.polyval(py, float(dt)))

    # Keep continuity around last observed point in degenerate fit cases.
    if not np.isfinite(pred_cx) or not np.isfinite(pred_cy):
        return cx0, cy0
    return pred_cx, pred_cy


def build_track_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for frame_tracks in window_tracks:
        for t in frame_tracks:
            by_id[t.track_id].append(t)

    frame_w, frame_h = _frame_dims_from_window(window_tracks)
    frame_diag = max(1.0, math.hypot(frame_w, frame_h))
    window_len = len(window_tracks)
    speed_span = max(2, min(12, window_len))
    pred_lookback = max(3, min(16, window_len))

    speeds_by_id: dict[int, list[float]] = defaultdict(list)
    pred_center_errs_by_id: dict[int, list[float]] = defaultdict(list)
    pred_iou_errs_by_id: dict[int, list[float]] = defaultdict(list)
    speed_turn_rates: list[float] = []
    dir_turn_rates: list[float] = []
    turn_candidates = 0
    turn_active = 0
    # track_lens: list[float] = []

    # first_ids = {t.track_id for t in (window_tracks[0] if window_tracks else [])}
    # last_ids = {t.track_id for t in (window_tracks[-1] if window_tracks else [])}
    # disappear_step_ratios: list[float] = []
    disappear_edge_dists: list[float] = []
    disappear_far_flags: list[float] = []

    # Static objects
    static_ids = {
        tid
        for tid, seq in by_id.items()
        if _is_static_full_window(seq, window_len)
    }

    # Check if there are tracks that disappear in the middle of the window, and if so, whether they disappear near the edge.
    for i in range(max(0, len(window_tracks) - 1)):
        cur_map = {t.track_id: t for t in window_tracks[i] if t.track_id not in static_ids}
        nxt_ids = {t.track_id for t in window_tracks[i + 1] if t.track_id not in static_ids}
        cur_ids = set(cur_map.keys())
        if not cur_ids:
            continue
        disappeared = cur_ids - nxt_ids
        for tid in disappeared:
            last_t = cur_map[tid]
            d_edge = _distance_to_nearest_edge(last_t.cx, last_t.cy, frame_w, frame_h)
            d_norm = d_edge / frame_diag
            disappear_edge_dists.append(d_norm)
            disappear_far_flags.append(1.0 if d_norm > 0.08 else 0.0)

    # Precompute per-track sequences without static full-window objects.
    filtered_by_id = {tid: seq for tid, seq in by_id.items() if tid not in static_ids}

    for seq in filtered_by_id.values():
        seq = sorted(seq, key=lambda x: x.frame_id)
        # track_lens.append(float(len(seq)))
        if len(seq) < 2:
            continue
        # Speed estimation from multi-frame velocity for better robustness.
        for i in range(1, len(seq)):
            vx_i, vy_i = _velocity_at(seq, i, speed_span)
            scale = math.sqrt(max(seq[i].area, 1.0))
            v = math.hypot(vx_i, vy_i) / scale
            speeds_by_id[seq[i].track_id].append(v)

        # Prediction error using multi-frame motion history.
        for i in range(2, len(seq)):
            p0 = seq[i - 2]
            p1 = seq[i - 1]
            p2 = seq[i]

            dt2 = max(1, int(p2.frame_id - p1.frame_id))

            pred = _predict_with_history(seq[:i], dt=dt2, lookback=pred_lookback)
            if pred is None:
                continue
            pred_cx, pred_cy = pred

            scale = math.sqrt(max(p2.area, 1.0))
            pred_center_errs_by_id[p2.track_id].append(math.hypot(p2.cx - pred_cx, p2.cy - pred_cy) / scale)

            pred_box = [
                p1.cx - p1.w / 2.0 + (pred_cx - p1.cx),
                p1.cy - p1.h / 2.0 + (pred_cy - p1.cy),
                p1.cx + p1.w / 2.0 + (pred_cx - p1.cx),
                p1.cy + p1.h / 2.0 + (pred_cy - p1.cy),
            ]
            pred_iou_errs_by_id[p2.track_id].append(1.0 - _iou_xyxy(pred_box, p2.bbox_xyxy))

            # Turn activity from multi-frame smoothed velocities.
            v1x, v1y = _velocity_at(seq, i - 1, speed_span)
            v2x, v2y = _velocity_at(seq, i, speed_span)
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            dir1 = math.atan2(v1y, v1x) if n1 > 0 else 0.0
            dir2 = math.atan2(v2y, v2x) if n2 > 0 else 0.0
            turn_candidates += 1
            speed_change_ratio = abs(n1 - n2) / max(1e-3, (n1 + n2) * 0.5)
            speed_turn_hit = speed_change_ratio > 0.4
            dir_turn_hit = abs(math.sin(dir2 - dir1)) > 0.35
            if speed_turn_hit:
                speed_turn_rates.append(speed_change_ratio)
            if dir_turn_hit:
                dir_turn_rates.append(abs(math.sin(dir2 - dir1)))
            if speed_turn_hit or dir_turn_hit:
                turn_active += 1

    # first_ids = first_ids - static_ids
    # last_ids = last_ids - static_ids
    # new_track_ratio = float(len(last_ids - first_ids) / max(1, len(last_ids)))
    # lost_track_ratio = float(len(first_ids - last_ids) / max(1, len(first_ids)))

    # Current-frame ids
    current_ids = set(t.track_id for t in (window_tracks[-1] if window_tracks else [])) - static_ids
    # curr_speeds = [speeds_by_id[tid][-1] for tid in current_ids if tid in speeds_by_id and speeds_by_id[tid]]
    # curr_pred_center_errs = [pred_center_errs_by_id[tid][-1] for tid in current_ids if tid in pred_center_errs_by_id and pred_center_errs_by_id[tid]]
    # curr_pred_iou_errs = [pred_iou_errs_by_id[tid][-1] for tid in current_ids if tid in pred_iou_errs_by_id and pred_iou_errs_by_id[tid]]

    # Predict next-frame trajectories from current track states and estimate collision warning.
    pred_next: dict[int, tuple[float, float]] = {}
    for tid in current_ids:
        seq = sorted(filtered_by_id.get(tid, []), key=lambda x: x.frame_id)
        pred = _predict_next_center(seq)
        if pred is not None:
            pred_next[tid] = pred
    # If any two predicted centers are very close in the next frame, it's a strong signal of potential collision or interaction, which may indicate anomaly in traffic scenarios. The threshold and decay function can be tuned based on validation data.
    min_next_dist = 1.0
    pred_ids = list(pred_next.keys())
    for i in range(len(pred_ids)):
        for j in range(i + 1, len(pred_ids)):
            a = pred_next[pred_ids[i]]
            b = pred_next[pred_ids[j]]
            d = math.hypot(a[0] - b[0], a[1] - b[1]) / frame_diag
            if d < min_next_dist:
                min_next_dist = d

    if len(pred_ids) < 2:
        min_next_dist = 1.0
    collision_warning = math.exp(-min_next_dist / 0.03)

    speeds_mean = sum((v for speeds in speeds_by_id.values() for v in speeds)) / max(1, sum(len(s) for s in speeds_by_id.values()))
    pred_center_max_err = max((v for errs in pred_center_errs_by_id.values() for v in errs), default=0.0)
    pred_iou_max_err = max((v for errs in pred_iou_errs_by_id.values() for v in errs), default=0.0)
    speed_turn_rates_max = max(speed_turn_rates) if speed_turn_rates else 0.0
    dir_turn_rates_max = max(dir_turn_rates) if dir_turn_rates else 0.0
    disappear_far_sum = sum(disappear_far_flags)

    return {
        "track_count": float(len(filtered_by_id)),
        "speeds_mean": speeds_mean,
        "pred_center_max_err": pred_center_max_err,
        "pred_iou_max_err": pred_iou_max_err,
        "pred_center_sum_err": sum((v for errs in pred_center_errs_by_id.values() for v in errs)),
        "pred_iou_sum_err": sum((v for errs in pred_iou_errs_by_id.values() for v in errs)),
        "speed_turn_rates_max": speed_turn_rates_max,
        "dir_turn_rates_max": dir_turn_rates_max,
        "turn_active_ratio": float(turn_active / max(1, turn_candidates)),
        "collision_warning": collision_warning,
        "disappear_far_sum": disappear_far_sum,
        # "new_track_ratio": new_track_ratio,
        # "lost_track_ratio": lost_track_ratio,
        # "disappear_near_edge_ratio": sum(disappear_far_flags) / max(1, len(disappear_far_flags)) if disappear_far_flags else 0.0,
        # "disappear_edge_dist_mean": sum(disappear_edge_dists) / max(1, len(disappear_edge_dists)) if disappear_edge_dists else 0.0,
        # "collision_warning": collision_warning,
    }
