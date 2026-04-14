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


def build_track_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for frame_tracks in window_tracks:
        for t in frame_tracks:
            by_id[t.track_id].append(t)

    speeds: list[float] = []
    accs: list[float] = []
    acc_abs: list[float] = []
    areas: list[float] = []
    pred_center_errs: list[float] = []
    pred_iou_errs: list[float] = []
    turn_rates: list[float] = []
    track_lens: list[float] = []

    first_ids = {t.track_id for t in (window_tracks[0] if window_tracks else [])}
    last_ids = {t.track_id for t in (window_tracks[-1] if window_tracks else [])}

    for seq in by_id.values():
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
            if n1 > 1e-6 and n2 > 1e-6:
                cosv = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                turn_rates.append(math.acos(cosv) / math.pi)

    new_track_ratio = float(len(last_ids - first_ids) / max(1, len(last_ids)))
    lost_track_ratio = float(len(first_ids - last_ids) / max(1, len(first_ids)))

    pred_p95 = sorted(pred_center_errs)[int(0.95 * (len(pred_center_errs) - 1))] if pred_center_errs else 0.0
    speed_p95 = sorted(speeds)[int(0.95 * (len(speeds) - 1))] if speeds else 0.0

    return {
        "track_count": float(len(by_id)),
        "speed_mean": float(sum(speeds) / len(speeds)) if speeds else 0.0,
        "acc_mean": float(sum(accs) / len(accs)) if accs else 0.0,
        "acc_abs_mean": float(sum(acc_abs) / len(acc_abs)) if acc_abs else 0.0,
        "area_mean": float(sum(areas) / len(areas)) if areas else 0.0,
        "track_age_mean": float(sum(track_lens) / len(track_lens)) if track_lens else 0.0,
        "speed_p95": float(speed_p95),
        "pred_center_err_mean": float(sum(pred_center_errs) / len(pred_center_errs)) if pred_center_errs else 0.0,
        "pred_center_err_p95": float(pred_p95),
        "pred_iou_err_mean": float(sum(pred_iou_errs) / len(pred_iou_errs)) if pred_iou_errs else 0.0,
        "turn_rate_mean": float(sum(turn_rates) / len(turn_rates)) if turn_rates else 0.0,
        "new_track_ratio": new_track_ratio,
        "lost_track_ratio": lost_track_ratio,
    }
