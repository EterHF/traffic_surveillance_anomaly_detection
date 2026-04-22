from __future__ import annotations

from collections import defaultdict

from src.schemas import TrackObject


def build_track_features(window_tracks: list[list[TrackObject]]) -> dict[str, float]:
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for frame_tracks in window_tracks:
        for t in frame_tracks:
            by_id[t.track_id].append(t)

    speeds: list[float] = []
    accs: list[float] = []
    areas: list[float] = []

    for seq in by_id.values():
        if len(seq) < 2:
            continue
        v_last = 0.0
        for i in range(1, len(seq)):
            dx = seq[i].cx - seq[i - 1].cx
            dy = seq[i].cy - seq[i - 1].cy
            v = (dx * dx + dy * dy) ** 0.5
            speeds.append(v)
            accs.append(v - v_last)
            v_last = v
            areas.append(seq[i].area)

    return {
        "track_count": float(len(by_id)),
        "speed_mean": float(sum(speeds) / len(speeds)) if speeds else 0.0,
        "acc_mean": float(sum(accs) / len(accs)) if accs else 0.0,
        "area_mean": float(sum(areas) / len(areas)) if areas else 0.0,
    }
