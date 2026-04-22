from __future__ import annotations

from collections import defaultdict
import math

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
    p0 = seq[-2]
    p1 = seq[-1]
    dt = max(1, int(p1.frame_id - p0.frame_id))
    vx = (p1.cx - p0.cx) / float(dt)
    vy = (p1.cy - p0.cy) / float(dt)
    return (p1.cx + vx, p1.cy + vy)


def _distance_to_nearest_edge(x: float, y: float, frame_w: float, frame_h: float) -> float:
    return float(min(x, frame_w - x, y, frame_h - y))


def build_track_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for frame_tracks in window_tracks:
        for t in frame_tracks:
            by_id[t.track_id].append(t)

    frame_w, frame_h = _frame_dims_from_window(window_tracks)
    frame_diag = max(1.0, math.hypot(frame_w, frame_h))
    window_len = len(window_tracks)

    speeds: list[float] = []
    accs: list[float] = []
    acc_abs: list[float] = []
    areas: list[float] = []
    pred_center_errs: list[float] = []
    pred_iou_errs: list[float] = []
    turn_rates: list[float] = []
    turn_candidates = 0
    turn_active = 0
    track_lens: list[float] = []

    first_ids = {t.track_id for t in (window_tracks[0] if window_tracks else [])}
    last_ids = {t.track_id for t in (window_tracks[-1] if window_tracks else [])}
    disappear_step_ratios: list[float] = []
    disappear_edge_dists: list[float] = []
    disappear_far_flags: list[float] = []

    static_ids = {
        tid
        for tid, seq in by_id.items()
        if _is_static_full_window(seq, window_len)
    }

    for i in range(max(0, len(window_tracks) - 1)):
        cur_map = {t.track_id: t for t in window_tracks[i] if t.track_id not in static_ids}
        nxt_ids = {t.track_id for t in window_tracks[i + 1] if t.track_id not in static_ids}
        cur_ids = set(cur_map.keys())
        if not cur_ids:
            continue
        disappeared = cur_ids - nxt_ids
        disappear_step_ratios.append(float(len(disappeared) / max(1, len(cur_ids))))
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
        track_lens.append(float(len(seq)))
        if len(seq) < 2:
            continue
        v_last = 0.0
        for i in range(1, len(seq)):
            dx = seq[i].cx - seq[i - 1].cx
            dy = seq[i].cy - seq[i - 1].cy
            scale = math.sqrt(max(seq[i].area, 1.0))
            v = ((dx * dx + dy * dy) ** 0.5) / scale
            speeds.append(v)
            a = v - v_last
            accs.append(a)
            acc_abs.append(abs(a))
            v_last = v
            areas.append(seq[i].area)

        for i in range(2, len(seq)):
            p0 = seq[i - 2]
            p1 = seq[i - 1]
            p2 = seq[i]

            dt1 = max(1, int(p1.frame_id - p0.frame_id))
            dt2 = max(1, int(p2.frame_id - p1.frame_id))

            vx = (p1.cx - p0.cx) / float(dt1)
            vy = (p1.cy - p0.cy) / float(dt1)

            pred_cx = p1.cx + vx * dt2
            pred_cy = p1.cy + vy * dt2
            scale = math.sqrt(max(p2.area, 1.0))
            pred_center_errs.append(math.hypot(p2.cx - pred_cx, p2.cy - pred_cy) / scale)

            pred_box = [
                p1.cx - p1.w / 2.0 + vx * dt2,
                p1.cy - p1.h / 2.0 + vy * dt2,
                p1.cx + p1.w / 2.0 + vx * dt2,
                p1.cy + p1.h / 2.0 + vy * dt2,
            ]
            pred_iou_errs.append(1.0 - _iou_xyxy(pred_box, p2.bbox_xyxy))

            v1x, v1y = p1.cx - p0.cx, p1.cy - p0.cy
            v2x, v2y = p2.cx - p1.cx, p2.cy - p1.cy
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            turn_candidates += 1
            if n1 > 1e-6 and n2 > 1e-6:
                # Gate out near-static micro jitter; turn rate should represent meaningful motion.
                s1 = n1 / math.sqrt(max(p1.area, 1.0))
                s2 = n2 / math.sqrt(max(p2.area, 1.0))
                if min(s1, s2) < 0.02:
                    continue
                cosv = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                turn_rates.append(math.acos(cosv) / math.pi)
                turn_active += 1

    first_ids = first_ids - static_ids
    last_ids = last_ids - static_ids
    new_track_ratio = float(len(last_ids - first_ids) / max(1, len(last_ids)))
    lost_track_ratio = float(len(first_ids - last_ids) / max(1, len(first_ids)))

    # Current-frame speed/acc from tracks that are present in the current frame.
    current_ids = {t.track_id for t in (window_tracks[-1] if window_tracks else [])} - static_ids
    curr_speeds: list[float] = []
    curr_accs: list[float] = []
    for tid in current_ids:
        seq = sorted(filtered_by_id.get(tid, []), key=lambda x: x.frame_id)
        if len(seq) < 2:
            continue
        p0 = seq[-2]
        p1 = seq[-1]
        dt = max(1, int(p1.frame_id - p0.frame_id))
        v_now = math.hypot(p1.cx - p0.cx, p1.cy - p0.cy) / (math.sqrt(max(p1.area, 1.0)) * dt)
        curr_speeds.append(v_now)
        if len(seq) >= 3:
            p_1 = seq[-3]
            dt_prev = max(1, int(p0.frame_id - p_1.frame_id))
            v_prev = math.hypot(p0.cx - p_1.cx, p0.cy - p_1.cy) / (math.sqrt(max(p0.area, 1.0)) * dt_prev)
            curr_accs.append(v_now - v_prev)

    # Current-frame prediction error from previous frames only.
    curr_pred_center_errs: list[float] = []
    curr_pred_iou_errs: list[float] = []
    curr_map = {t.track_id: t for t in (window_tracks[-1] if window_tracks else []) if t.track_id not in static_ids}
    for tid, cur_t in curr_map.items():
        seq = sorted(filtered_by_id.get(tid, []), key=lambda x: x.frame_id)
        if len(seq) < 3:
            continue
        p0 = seq[-3]
        p1 = seq[-2]
        p2 = seq[-1]
        if p2.frame_id != cur_t.frame_id:
            continue
        dt1 = max(1, int(p1.frame_id - p0.frame_id))
        dt2 = max(1, int(p2.frame_id - p1.frame_id))
        vx = (p1.cx - p0.cx) / float(dt1)
        vy = (p1.cy - p0.cy) / float(dt1)
        pred_cx = p1.cx + vx * dt2
        pred_cy = p1.cy + vy * dt2
        scale = math.sqrt(max(p2.area, 1.0))
        curr_pred_center_errs.append(math.hypot(p2.cx - pred_cx, p2.cy - pred_cy) / scale)
        pred_box = [
            p1.cx - p1.w / 2.0 + vx * dt2,
            p1.cy - p1.h / 2.0 + vy * dt2,
            p1.cx + p1.w / 2.0 + vx * dt2,
            p1.cy + p1.h / 2.0 + vy * dt2,
        ]
        curr_pred_iou_errs.append(1.0 - _iou_xyxy(pred_box, p2.bbox_xyxy))

    # Predict next-frame trajectories from current track states and estimate collision warning.
    pred_next: dict[int, tuple[float, float]] = {}
    for tid in current_ids:
        seq = sorted(filtered_by_id.get(tid, []), key=lambda x: x.frame_id)
        pred = _predict_next_center(seq)
        if pred is not None:
            pred_next[tid] = pred

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

    pred_p95 = sorted(pred_center_errs)[int(0.95 * (len(pred_center_errs) - 1))] if pred_center_errs else 0.0
    speed_p95 = sorted(speeds)[int(0.95 * (len(speeds) - 1))] if speeds else 0.0
    disappear_mean = float(sum(disappear_step_ratios) / len(disappear_step_ratios)) if disappear_step_ratios else 0.0
    disappear_max = float(max(disappear_step_ratios)) if disappear_step_ratios else 0.0
    disappear_edge_mean = float(sum(disappear_edge_dists) / len(disappear_edge_dists)) if disappear_edge_dists else 0.0
    disappear_edge_max = float(max(disappear_edge_dists)) if disappear_edge_dists else 0.0
    disappear_far_ratio = float(sum(disappear_far_flags) / len(disappear_far_flags)) if disappear_far_flags else 0.0

    return {
        "track_count": float(len(filtered_by_id)),
        "track_count_current": float(len(current_ids)),
        "track_count_static_ignored": float(len(static_ids)),
        "speed_mean": float(sum(speeds) / len(speeds)) if speeds else 0.0,
        "acc_mean": float(sum(accs) / len(accs)) if accs else 0.0,
        "acc_abs_mean": float(sum(acc_abs) / len(acc_abs)) if acc_abs else 0.0,
        "speed_current": float(sum(curr_speeds) / len(curr_speeds)) if curr_speeds else 0.0,
        "acc_current": float(sum(curr_accs) / len(curr_accs)) if curr_accs else 0.0,
        "area_mean": float(sum(areas) / len(areas)) if areas else 0.0,
        "track_age_mean": float(sum(track_lens) / len(track_lens)) if track_lens else 0.0,
        "speed_p95": float(speed_p95),
        "pred_center_err_mean": float(sum(pred_center_errs) / len(pred_center_errs)) if pred_center_errs else 0.0,
        "pred_center_err_p95": float(pred_p95),
        "pred_iou_err_mean": float(sum(pred_iou_errs) / len(pred_iou_errs)) if pred_iou_errs else 0.0,
        "pred_center_err_current": float(sum(curr_pred_center_errs) / len(curr_pred_center_errs)) if curr_pred_center_errs else 0.0,
        "pred_iou_err_current": float(sum(curr_pred_iou_errs) / len(curr_pred_iou_errs)) if curr_pred_iou_errs else 0.0,
        "turn_rate_mean": float(sum(turn_rates) / len(turn_rates)) if turn_rates else 0.0,
        "turn_rate_active_ratio": float(turn_active / max(1, turn_candidates)),
        "next_frame_collision_warning": float(collision_warning),
        "next_frame_min_pred_dist": float(min_next_dist),
        "new_track_ratio": new_track_ratio,
        "lost_track_ratio": lost_track_ratio,
        "disappear_ratio_mean": disappear_mean,
        "disappear_ratio_max": disappear_max,
        "disappear_edge_dist_mean": disappear_edge_mean,
        "disappear_edge_dist_max": disappear_edge_max,
        "disappear_far_from_edge_ratio": disappear_far_ratio,
    }
