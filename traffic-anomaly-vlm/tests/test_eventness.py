from __future__ import annotations

import numpy as np

from src.features.eventness_features import EventnessFeatureConfig, EventnessFeatureExtractor
from src.schemas import TrackObject


def _make_track(frame_id: int, track_id: int, cx: float, cy: float) -> TrackObject:
    w = 20.0
    h = 12.0
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return TrackObject(
        frame_id=frame_id,
        track_id=track_id,
        cls_id=2,
        cls_name="car",
        score=0.9,
        bbox_xyxy=[x1, y1, x2, y2],
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        area=w * h,
        frame_w=96.0,
        frame_h=64.0,
    )


def test_eventness_lowres_and_layout_detect_change():
    images: list[np.ndarray] = []
    tracks: list[list[TrackObject]] = []
    for idx in range(6):
        img = np.zeros((64, 96, 3), dtype=np.uint8)
        if idx >= 3:
            img[:, :] = 220
            tracks.append([_make_track(idx, 1, 72.0, 32.0)])
        else:
            tracks.append([_make_track(idx, 1, 24.0, 32.0)])
        images.append(img)

    extractor = EventnessFeatureExtractor(
        EventnessFeatureConfig(
            use_clip=False,
            use_raft=False,
            use_lowres=True,
            use_track_layout=True,
            lowres_size=16,
            occupancy_grid=4,
            smooth_window=5,
            smooth_polyorder=2,
        )
    )
    result = extractor.compute(images, tracks)

    assert result.frame_scores.shape == (6,)
    assert "lowres_eventness" in result.cue_scores
    assert "track_layout_eventness" in result.cue_scores
    assert float(np.max(result.frame_scores)) > 0.1
    assert int(np.argmax(result.frame_scores)) in {3, 4}
