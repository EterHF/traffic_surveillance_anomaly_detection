from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from src.schemas import TrackObject


STATIC_MOVE_RATIO = 0.10
DISAPPEAR_FAR_EDGE_RATIO = 0.20
TURN_CHANGE_RATIO = 0.5
SPEED_SCALING_FACTOR = 15.0
# Motion thresholds are normalized by object size, so near/far objects share a
# comparable turn prior. The pixel floor still filters detector jitter.
MIN_TURN_SPEED_RATIO = 0.03
MIN_DIR_MOTION_RATIO = 0.05
MIN_DIR_MOTION_PX = 2.0
# Sum-like priors are aggregated by top-k mean to reduce object-count bias.
TOPK_AGG_COUNT = 3
# Ignore motion priors when the box is clipped, strongly resized, or jumps too
# far for its scale; these are usually tracker artifacts rather than behavior.
EDGE_UNSTABLE_RATIO = 0.05
MAX_MOTION_AREA_RATIO = 1.5
MAX_STEP_RATIO = 1.2
REPLACEMENT_CENTER_RATIO = 0.06
REPLACEMENT_IOU = 0.05
# Pure bottom-center is sensitive to truncated boxes near the image edge. Use a
# lower-center anchor that still follows ground motion better than bbox center.
TRACK_POINT_Y_RATIO = 0.70

MotionFit = tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None]


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


def _track_point(track: TrackObject) -> tuple[float, float]:
    x1, y1, x2, y2 = track.bbox_xyxy
    return (
        float(x1 + x2) * 0.5,
        float(y1) + float(y2 - y1) * TRACK_POINT_Y_RATIO,
    )


def _track_scale(track: TrackObject) -> float:
    return math.sqrt(max(float(track.area), 1.0))


def _box_area_ratio(a: TrackObject, b: TrackObject) -> float:
    area_a = max(float(a.area), 1.0)
    area_b = max(float(b.area), 1.0)
    return max(area_a, area_b) / min(area_a, area_b)


def _is_edge_unstable(track: TrackObject, frame_w: float, frame_h: float) -> bool:
    x1, y1, x2, y2 = track.bbox_xyxy
    margin = EDGE_UNSTABLE_RATIO * max(1.0, math.hypot(frame_w, frame_h))
    return (
        float(x1) <= margin
        or float(y1) <= margin
        or float(frame_w) - float(x2) <= margin
        or float(frame_h) - float(y2) <= margin
    )


def _is_motion_stable_pair(
    prev: TrackObject,
    cur: TrackObject,
    frame_w: float,
    frame_h: float,
) -> bool:
    if _is_edge_unstable(prev, frame_w, frame_h) or _is_edge_unstable(cur, frame_w, frame_h):
        return False
    if _box_area_ratio(prev, cur) > MAX_MOTION_AREA_RATIO:
        return False
    px, py = _track_point(prev)
    cx, cy = _track_point(cur)
    step = math.hypot(cx - px, cy - py)
    scale = max(1.0, 0.5 * (_track_scale(prev) + _track_scale(cur)))
    return (step / scale) <= MAX_STEP_RATIO


def _has_near_replacement(
    last_track: TrackObject,
    next_tracks: list[TrackObject],
    frame_diag: float,
) -> bool:
    lx, ly = _track_point(last_track)
    for candidate in next_tracks:
        if int(candidate.track_id) == int(last_track.track_id):
            continue
        if int(candidate.cls_id) != int(last_track.cls_id):
            continue
        cx, cy = _track_point(candidate)
        center_dist = math.hypot(cx - lx, cy - ly) / max(1.0, frame_diag)
        if center_dist <= REPLACEMENT_CENTER_RATIO:
            return True
        if _iou_xyxy(last_track.bbox_xyxy, candidate.bbox_xyxy) >= REPLACEMENT_IOU:
            return True
    return False


def _box_from_track_point(track: TrackObject, point: tuple[float, float]) -> list[float]:
    px, py = point
    top_offset = track.h * TRACK_POINT_Y_RATIO
    return [
        float(px - track.w / 2.0),
        float(py - top_offset),
        float(px + track.w / 2.0),
        float(py + track.h - top_offset),
    ]


def _topk_mean(values: list[float], k: int = TOPK_AGG_COUNT) -> float:
    if not values:
        return 0.0
    top = sorted((float(v) for v in values), reverse=True)[: max(1, int(k))]
    return float(sum(top) / len(top))


def _is_static_full_window(seq: list[TrackObject], window_len: int) -> bool:
    # Ignore tracks that stay through the whole window and barely move.
    if len(seq) < max(3, window_len):
        return False
    seq = sorted(seq, key=lambda x: x.frame_id)
    x0, y0 = _track_point(seq[0])
    x1, y1 = _track_point(seq[-1])
    d = math.hypot(x1 - x0, y1 - y0)
    scale = math.sqrt(max(1.0, sum(t.area for t in seq) / max(1, len(seq))))
    return (d / scale) < STATIC_MOVE_RATIO


def _fit_track_motion(
    seq: list[TrackObject],
    *,
    fit_degree: int = 3,
    history_len: int | None = None,
    end_idx: int | None = None,
    predict_next: bool = False,
    predict_dt: int | None = None,
) -> MotionFit | None:
    """Fit image-space motion and optionally extrapolate a future point.

    The fitted velocity is pixels per original frame_id. Callers multiply by
    sampled-frame dt before comparing motion magnitude.
    """
    if len(seq) < 2:
        return None

    points = sorted(seq, key=lambda x: x.frame_id)
    if end_idx is not None:
        end_idx = min(max(0, int(end_idx)), len(points) - 1)
        points = points[: end_idx + 1]
        if len(points) < 2:
            return None

    max_hist = min(len(points), max(3, int(history_len or len(points))))
    points = points[-max_hist:]
    t = np.array([float(p.frame_id) for p in points], dtype=np.float64)
    t = t - float(t[-1])
    track_points = [_track_point(p) for p in points]
    x = np.array([float(p[0]) for p in track_points], dtype=np.float64)
    y = np.array([float(p[1]) for p in track_points], dtype=np.float64)

    deg = max(1, min(int(fit_degree), len(points) - 1))
    unique_t = np.unique(t)
    deg = min(deg, max(1, int(len(unique_t) - 1)))

    try:
        px = np.polyfit(t, x, deg=deg)
        py = np.polyfit(t, y, deg=deg)
    except Exception:
        return None

    vx = float(np.polyval(np.polyder(px), 0.0))
    vy = float(np.polyval(np.polyder(py), 0.0))
    center = (float(np.polyval(px, 0.0)), float(np.polyval(py, 0.0)))

    pred_center: tuple[float, float] | None = None
    if predict_next or predict_dt is not None:
        if predict_dt is None:
            dts = [
                max(1, int(points[i].frame_id - points[i - 1].frame_id))
                for i in range(1, len(points))
            ]
            predict_dt = sorted(dts)[len(dts) // 2] if dts else 1
        pred_cx = float(np.polyval(px, float(predict_dt)))
        pred_cy = float(np.polyval(py, float(predict_dt)))
        pred_center = (
            (pred_cx, pred_cy)
            if np.isfinite(pred_cx) and np.isfinite(pred_cy)
            else center
        )

    return (vx, vy), center, pred_center


def _distance_to_nearest_edge(x: float, y: float, frame_w: float, frame_h: float) -> float:
    x_clamped = min(max(0.0, x), frame_w)
    y_clamped = min(max(0.0, y), frame_h)
    return float(min(x_clamped, frame_w - x_clamped, y_clamped, frame_h - y_clamped))


def build_track_features(
    window_tracks: list[list[TrackObject]],
    fit_degree: int = 3,
    history_len: int | None = None,
) -> dict[str, float]:
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for frame_tracks in window_tracks:
        for t in frame_tracks:
            by_id[t.track_id].append(t)

    frame_w, frame_h = window_tracks[-1][0].frame_w, window_tracks[-1][0].frame_h if window_tracks and window_tracks[-1] else (640, 480)   

    frame_diag = max(1.0, math.hypot(frame_w, frame_h))
    window_len = len(window_tracks)
    # If history_len is not provided, default to current window length.
    fit_hist = min(window_len, max(3, int(history_len))) if history_len is not None else window_len
    fit_degree = max(1, fit_degree)

    current_speeds: list[float] = []
    current_pred_center_errs: list[float] = []
    current_pred_iou_errs: list[float] = []
    speed_turn_rates: list[float] = []
    dir_turn_rates: list[float] = []
    turn_candidates = 0
    turn_active = 0
    disappear_far_flags: list[float] = []

    # Static objects
    static_ids = {
        tid
        for tid, seq in by_id.items()
        if _is_static_full_window(seq, window_len)
    }

    # Tracks that disappear away from the frame edge are stronger anomaly signals.
    for i in range(max(0, len(window_tracks) - 1)):
        cur_map = {t.track_id: t for t in window_tracks[i] if t.track_id not in static_ids}
        nxt_tracks = [t for t in window_tracks[i + 1] if t.track_id not in static_ids]
        nxt_ids = {t.track_id for t in nxt_tracks}
        cur_ids = set(cur_map.keys())
        if not cur_ids:
            continue
        disappeared = cur_ids - nxt_ids
        for tid in disappeared:
            last_t = cur_map[tid]
            if len(by_id.get(tid, [])) < fit_hist:
                continue
            if _has_near_replacement(last_t, nxt_tracks, frame_diag):
                continue
            px, py = _track_point(last_t)
            d_edge = _distance_to_nearest_edge(px, py, frame_w, frame_h)
            d_norm = d_edge / frame_diag
            disappear_far_flags.append(1.0 if d_norm > DISAPPEAR_FAR_EDGE_RATIO else 0.0)

    filtered_by_id = {
        tid: sorted(seq, key=lambda x: x.frame_id)
        for tid, seq in by_id.items()
        if tid not in static_ids
    }
    current_ids = {
        t.track_id
        for t in (window_tracks[-1] if window_tracks else [])
    } - static_ids
    pred_next_boxes: dict[int, list[float]] = {}

    # Compute only current-frame metrics for each visible track.
    for tid in current_ids:
        seq = filtered_by_id.get(tid, [])
        if len(seq) < 3:
            continue
        current_pair_stable = _is_motion_stable_pair(seq[-2], seq[-1], frame_w, frame_h)
        motion_degree = 1 if len(seq) < fit_hist else fit_degree
        motion = _fit_track_motion(
            seq,
            fit_degree=motion_degree,
            history_len=fit_hist,
            predict_next=current_pair_stable,
        )

        if motion is None:
            v2x = v2y = 0.0
            next_center = None
        else:
            (v2x, v2y), _, next_center = motion
        last = seq[-1]
        if next_center is not None:
            pred_next_boxes[tid] = _box_from_track_point(last, next_center)

        scale_cur = _track_scale(seq[-1])
        n2_motion = math.hypot(v2x, v2y)
        dt_cur = max(1, int(seq[-1].frame_id - seq[-2].frame_id))
        current_speeds.append(n2_motion * dt_cur / scale_cur * SPEED_SCALING_FACTOR)

        if len(seq) >= fit_hist and current_pair_stable:
            p1 = seq[-2]
            p2 = seq[-1]
            dt2 = max(1, int(p2.frame_id - p1.frame_id))
            pred_degree = 1 if len(seq[:-1]) < fit_hist else fit_degree
            prev_fit = _fit_track_motion(
                seq[:-1],
                fit_degree=pred_degree,
                history_len=fit_hist,
                predict_dt=dt2,
            )

            if prev_fit is None:
                v1x = v1y = 0.0
                pred_center = None
            else:
                (v1x, v1y), _, pred_center = prev_fit

            if pred_center is not None:
                pred_x, pred_y = pred_center
                cur_x, cur_y = _track_point(p2)
                center_err = math.hypot(cur_x - pred_x, cur_y - pred_y) / scale_cur
                current_pred_center_errs.append(center_err)

                pred_box = _box_from_track_point(p1, pred_center)
                current_pred_iou_errs.append(1.0 - _iou_xyxy(pred_box, p2.bbox_xyxy))

            # turn_active per track: compare last two speed magnitudes.
            turn_candidates += 1
            n1_motion = math.hypot(v1x, v1y)
            n2_motion = math.hypot(v2x, v2y)
            motion_scale = max(1.0, 0.5 * (_track_scale(p1) + _track_scale(p2)))
            n1_ratio = n1_motion * dt2 / motion_scale
            n2_ratio = n2_motion * dt2 / motion_scale

            # Symmetric denominator avoids huge ratios when the previous
            # normalized speed is nearly zero.
            speed_change_ratio = abs(n1_ratio - n2_ratio) / max(
                MIN_TURN_SPEED_RATIO,
                0.5 * (n1_ratio + n2_ratio),
            )
            speed_turn_rates.append(speed_change_ratio)

            # Direction is unreliable for nearly static tracks, where detector jitter dominates.
            min_motion_px = min(n1_motion, n2_motion) * dt2
            min_dir_motion_px = max(MIN_DIR_MOTION_PX, MIN_DIR_MOTION_RATIO * motion_scale)
            if min_motion_px >= min_dir_motion_px:
                dir1 = math.atan2(v1y, v1x)
                dir2 = math.atan2(v2y, v2x)
                dir_change_ratio = abs(math.sin(dir2 - dir1))
            else:
                dir_change_ratio = 0.0
            dir_turn_rates.append(dir_change_ratio)

            if speed_change_ratio > TURN_CHANGE_RATIO or dir_change_ratio > TURN_CHANGE_RATIO:
                turn_active += 1

    # Overlapping predicted boxes suggest a possible collision or interaction.
    max_next_iou = 0.0
    pred_ids = list(pred_next_boxes.keys())
    for i in range(len(pred_ids)):
        for j in range(i + 1, len(pred_ids)):
            iou = _iou_xyxy(pred_next_boxes[pred_ids[i]], pred_next_boxes[pred_ids[j]])
            if iou > max_next_iou:
                max_next_iou = iou
    collision_warning = max_next_iou

    speeds_mean = sum(current_speeds) / max(1, len(current_speeds))
    pred_center_max_err = max(current_pred_center_errs, default=0.0)
    pred_iou_max_err = max(current_pred_iou_errs, default=0.0)
    speed_turn_rates_max = max(speed_turn_rates) if speed_turn_rates else 0.0
    dir_turn_rates_max = max(dir_turn_rates) if dir_turn_rates else 0.0
    disappear_far_sum = sum(disappear_far_flags)

    return {
        "track_count": float(len(filtered_by_id)),
        "speeds_mean": speeds_mean,

        "pred_center_max_err": pred_center_max_err,
        "pred_iou_max_err": pred_iou_max_err,

        "pred_center_sum_err": _topk_mean(current_pred_center_errs),
        "pred_iou_sum_err": _topk_mean(current_pred_iou_errs),

        "speed_turn_rates_max": speed_turn_rates_max,
        "speed_turn_rates_sum": _topk_mean(speed_turn_rates),

        "dir_turn_rates_max": dir_turn_rates_max,
        "dir_turn_rates_sum": _topk_mean(dir_turn_rates),

        "turn_active_ratio": float(turn_active / max(1, turn_candidates)),

        "collision_warning": collision_warning,
        "disappear_far_sum": disappear_far_sum,
    }
