from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.schemas import SpanProposal, TrackObject


def load_frame_ref(frame_ref: Any) -> np.ndarray | None:
    if isinstance(frame_ref, str):
        return cv2.imread(frame_ref)
    return frame_ref


def save_image(path: str, image: np.ndarray) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)
    return path


def select_keyframes(span: SpanProposal, n: int = 4) -> list[int]:
    if span.end_frame <= span.start_frame:
        return [span.start_frame]
    frame_span = span.end_frame - span.start_frame
    step = max(1, frame_span // max(n - 1, 1))
    picks = [span.start_frame + i * step for i in range(n)]
    picks[-1] = span.end_frame
    if span.peak_frame not in picks:
        picks[min(len(picks) - 1, n // 2)] = span.peak_frame
    return sorted(set(picks))


def build_event_summary(span: SpanProposal) -> dict:
    return {
        "event_id": span.span_id,
        "start_frame": span.start_frame,
        "end_frame": span.end_frame,
        "peak_frame": span.peak_frame,
        "eventness_peak": float(span.eventness_peak),
        "eventness_mean": float(span.eventness_mean),
        "span_prior_score": float(span.prior_score),
        # "mean_trigger": mean_trigger,
        # "window_count": len(related),
    }


def plot_trajectories(tracks: list[TrackObject], width: int = 1280, height: int = 720) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    by_id: dict[int, list[TrackObject]] = defaultdict(list)
    for track in tracks:
        by_id[track.track_id].append(track)

    for track_id, seq in by_id.items():
        color = ((track_id * 37) % 255, (track_id * 67) % 255, (track_id * 97) % 255)
        for idx in range(1, len(seq)):
            p1 = (int(seq[idx - 1].cx), int(seq[idx - 1].cy))
            p2 = (int(seq[idx].cx), int(seq[idx].cy))
            cv2.line(canvas, p1, p2, color, 2)
    return canvas


def resolve_crop_box(frame: np.ndarray, tracks: list[TrackObject], focus_track_id: int | None) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    focus_tracks = [track for track in tracks if focus_track_id is not None and int(track.track_id) == int(focus_track_id)]
    if not focus_tracks:
        focus_tracks = list(tracks)
    if not focus_tracks:
        return 0, 0, max(1, w), max(1, h)

    x1 = min(float(track.bbox_xyxy[0]) for track in focus_tracks)
    y1 = min(float(track.bbox_xyxy[1]) for track in focus_tracks)
    x2 = max(float(track.bbox_xyxy[2]) for track in focus_tracks)
    y2 = max(float(track.bbox_xyxy[3]) for track in focus_tracks)

    pad_x = max(12.0, 0.15 * (x2 - x1 + 1.0))
    pad_y = max(12.0, 0.15 * (y2 - y1 + 1.0))
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, max(1, w), max(1, h)
    return x1, y1, x2, y2


def crop_focus_region(frame: np.ndarray, tracks: list[TrackObject], focus_track_id: int | None) -> np.ndarray:
    x1, y1, x2, y2 = resolve_crop_box(frame, tracks, focus_track_id)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame.copy()
    return crop


def build_labeled_montage(tiles: list[np.ndarray], labels: list[str], thumb_size: tuple[int, int] = (280, 180)) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for tile, label in zip(tiles, labels):
        view = cv2.resize(tile, thumb_size)
        cv2.putText(
            view,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(view)
    while len(rendered) < 4:
        rendered.append(np.full((thumb_size[1], thumb_size[0], 3), 235, dtype=np.uint8))
    top = np.hstack(rendered[:2])
    bottom = np.hstack(rendered[2:4])
    return np.vstack([top, bottom])


def filter_tracks_by_frame(
    tracks: list[TrackObject],
    frame_id: int,
) -> list[TrackObject]:
    return [track for track in tracks if int(track.frame_id) == int(frame_id)]


def draw_track_annotations(
    frame: np.ndarray,
    frame_tracks: list[TrackObject],
    span_tracks: list[TrackObject] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    span_tracks = span_tracks or []

    history_by_id: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for track in span_tracks:
        history_by_id[int(track.track_id)].append((int(track.cx), int(track.cy)))

    for track_id, points in history_by_id.items():
        if len(points) < 2:
            continue
        color = ((track_id * 37) % 255, (track_id * 67) % 255, (track_id * 97) % 255)
        for idx in range(1, len(points)):
            cv2.line(canvas, points[idx - 1], points[idx], color, 2, cv2.LINE_AA)

    for track in frame_tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]
        track_id = int(track.track_id)
        color = ((track_id * 37) % 255, (track_id * 67) % 255, (track_id * 97) % 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"id={track_id} {track.cls_name}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (int(track.cx), int(track.cy)), 3, color, -1)
    return canvas
