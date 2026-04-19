from __future__ import annotations

from src.schemas import TrackObject


def _frame_dims_from_tracks(frame_tracks: list[TrackObject]) -> tuple[float, float]:
    if not frame_tracks:
        return 1.0, 1.0
    fw = max((float(t.frame_w) for t in frame_tracks), default=0.0)
    fh = max((float(t.frame_h) for t in frame_tracks), default=0.0)
    if fw > 0.0 and fh > 0.0:
        return fw, fh
    x2_max = max((float(t.bbox_xyxy[2]) for t in frame_tracks), default=1.0)
    y2_max = max((float(t.bbox_xyxy[3]) for t in frame_tracks), default=1.0)
    return max(1.0, x2_max), max(1.0, y2_max)


def _frame_density(frame_tracks: list[TrackObject]) -> float:
    if not frame_tracks:
        return 0.0
    fw, fh = _frame_dims_from_tracks(frame_tracks)
    frame_area = max(1.0, fw * fh)
    area_sum = sum(float(t.area) for t in frame_tracks)
    density = float(area_sum / frame_area)
    return density


def _current_dir_bins(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    # Direction bins for current frame motion using previous frame as reference.
    bins = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}
    if len(window_tracks) < 2:
        return bins
    prev = {t.track_id: t for t in window_tracks[-2]}
    curr = {t.track_id: t for t in window_tracks[-1]}
    for tid, tc in curr.items():
        tp = prev.get(tid)
        if tp is None:
            continue
        dx = tc.cx - tp.cx
        dy = tc.cy - tp.cy
        key = f"{int(dx >= 0)}{int(dy >= 0)}"
        bins[key] += 1.0
    return bins


def build_scene_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    all_tracks: list[TrackObject] = [t for frame_tracks in window_tracks for t in frame_tracks]
    if not all_tracks:
        return {
            "cur_count": 0.0,
            "count_mean": 0.0,
            "count_std": 0.0,
            "count_delta": 0.0, 
            "cur_density": 0.0,
            "density_mean": 0.0,
            "density_std": 0.0,
            "density_delta": 0.0,
            # "bin_l1_delta": 0.0,
            # "interaction_proxy": 0.0,
        }

    # Window-level density.
    density_mean = 0.0
    density_std = 0.0
    density_delta = 0.0
    if window_tracks:
        density_vals = [_frame_density(f) for f in window_tracks]
        density_mean = float(sum(density_vals) / max(1, len(density_vals)))
        density_std = float((sum((d - density_mean) ** 2 for d in density_vals) / max(1, len(density_vals))) ** 0.5)
        density_delta = float(density_vals[-1] - density_vals[0]) if len(density_vals) >= 2 else 0.0

    # Window-level object count.
    per_frame_counts = [len(f) for f in window_tracks]
    count_mean = float(sum(per_frame_counts) / max(1, len(per_frame_counts)))
    count_std = float(
        (sum((c - count_mean) ** 2 for c in per_frame_counts) / max(1, len(per_frame_counts))) ** 0.5
    )
    count_delta = float(per_frame_counts[-1] - per_frame_counts[0]) if len(per_frame_counts) >= 2 else 0.0

    # DEPRECATED

    # Window-level dir_bins.
    # cur_bins = _current_dir_bins(window_tracks)
    # prev_bins = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}
    # if len(window_tracks) >= 3:
    #     prev_like = window_tracks[0:2]
    #     prev_bins = _current_dir_bins(prev_like)
    # bin_l1_delta = float(
    #     abs(cur_bins["00"] - prev_bins["00"])
    #     + abs(cur_bins["01"] - prev_bins["01"])
    #     + abs(cur_bins["10"] - prev_bins["10"])
    #     + abs(cur_bins["11"] - prev_bins["11"])
    # )

    # Interaction proxy: average inverse nearest-neighbor distance of object centers per frame.
    # inv_nn_vals: list[float] = []
    # for frame_tracks in window_tracks:
    #     if len(frame_tracks) < 2:
    #         continue
    #     for i, t in enumerate(frame_tracks):
    #         min_d = float("inf")
    #         for j, s in enumerate(frame_tracks):
    #             if i == j:
    #                 continue
    #             d = math.hypot(t.cx - s.cx, t.cy - s.cy)
    #             if d < min_d:
    #                 min_d = d
    #         if min_d < float("inf"):
    #             inv_nn_vals.append(1.0 / (min_d + 1e-6))

    # interaction_proxy = float(sum(inv_nn_vals) / len(inv_nn_vals)) if inv_nn_vals else 0.0

    return {
        "cur_count": float(per_frame_counts[-1]) if per_frame_counts else 0.0,
        "count_mean": count_mean,
        "count_std": count_std,
        "count_delta": count_delta,
        
        "cur_density": float(density_vals[-1]) if window_tracks else 0.0,
        "density_mean": density_mean,
        "density_std": density_std,
        "density_delta": density_delta,
        # "bin_l1_delta": bin_l1_delta,
        # "interaction_proxy": interaction_proxy,
    }
