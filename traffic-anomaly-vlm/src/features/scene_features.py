from __future__ import annotations

from collections import Counter
import math

from src.schemas import TrackObject


def build_scene_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    all_tracks: list[TrackObject] = [t for frame_tracks in window_tracks for t in frame_tracks]
    if not all_tracks:
        return {
            "object_count": 0.0,
            "class_diversity": 0.0,
            "density_proxy": 0.0,
            "direction_entropy_proxy": 0.0,
        }

    class_counter = Counter([t.cls_name for t in all_tracks])
    obj_count = len(all_tracks)
    diversity = float(len(class_counter))

    # Density proxy: average area occupancy over all boxes.
    density_proxy = sum(t.area for t in all_tracks) / max(obj_count, 1)

    # Direction entropy proxy: use sign of x/y movement aggregated by track.
    dir_bins = Counter()
    by_id: dict[int, list[TrackObject]] = {}
    for t in all_tracks:
        by_id.setdefault(t.track_id, []).append(t)
    for seq in by_id.values():
        if len(seq) < 2:
            continue
        dx = seq[-1].cx - seq[0].cx
        dy = seq[-1].cy - seq[0].cy
        key = f"{int(dx >= 0)}_{int(dy >= 0)}"
        dir_bins[key] += 1

    total_dirs = max(sum(dir_bins.values()), 1)
    probs = [v / total_dirs for v in dir_bins.values()]
    entropy = -sum(p * (0 if p == 0 else __import__("math").log(p + 1e-12)) for p in probs)

    per_frame_counts = [len(f) for f in window_tracks]
    count_mean = float(sum(per_frame_counts) / max(1, len(per_frame_counts)))
    count_std = float(
        (sum((c - count_mean) ** 2 for c in per_frame_counts) / max(1, len(per_frame_counts))) ** 0.5
    )
    count_delta = float(per_frame_counts[-1] - per_frame_counts[0]) if per_frame_counts else 0.0

    first_density = float(sum(t.area for t in window_tracks[0]) / max(1, len(window_tracks[0]))) if window_tracks else 0.0
    last_density = float(sum(t.area for t in window_tracks[-1]) / max(1, len(window_tracks[-1]))) if window_tracks else 0.0
    density_delta = float(last_density - first_density)

    # Interaction proxy: average inverse nearest-neighbor distance of object centers per frame.
    inv_nn_vals: list[float] = []
    for frame_tracks in window_tracks:
        if len(frame_tracks) < 2:
            continue
        for i, t in enumerate(frame_tracks):
            min_d = float("inf")
            for j, s in enumerate(frame_tracks):
                if i == j:
                    continue
                d = math.hypot(t.cx - s.cx, t.cy - s.cy)
                if d < min_d:
                    min_d = d
            if min_d < float("inf"):
                inv_nn_vals.append(1.0 / (min_d + 1e-6))

    interaction_proxy = float(sum(inv_nn_vals) / len(inv_nn_vals)) if inv_nn_vals else 0.0

    return {
        "object_count": float(obj_count),
        "class_diversity": diversity,
        "density_proxy": float(density_proxy),
        "direction_entropy_proxy": float(entropy),
        "object_count_std": count_std,
        "object_count_delta": count_delta,
        "density_delta": density_delta,
        "interaction_inv_nn_proxy": interaction_proxy,
    }
