import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

from src.features.feature_components.track import _fit_track_motion, _track_point
from src.schemas import TrackObject


TRACKS_JSON = Path("v1_tracks_by_id.json")
TRACK_ID = "39"
OUTPUT_VIDEO = Path("debug_prediction_error.mp4")
OUTPUT_LAST_FRAME = Path("debug_prediction_error_last.jpg")

FIT_DEGREE = 2
HISTORY_LEN = 8
FPS = 25
PLOT_H = 120
TAIL_LEN = 64


def _transcode_for_vscode(source: Path, target: Path) -> bool:
    """Convert OpenCV output to H.264/yuv420p for VS Code/browser preview."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        source.replace(target)
        return False

    source.unlink(missing_ok=True)
    return True


def _make_track_object(frame: dict) -> TrackObject:
    return TrackObject(
        track_id=int(frame["track_id"]),
        frame_id=int(frame["frame_id"]),
        cls_id=int(frame["cls_id"]),
        cls_name=str(frame["cls_name"]),
        score=float(frame["score"]),
        bbox_xyxy=[float(x) for x in frame["bbox_xyxy"]],
        cx=float(frame["cx"]),
        cy=float(frame["cy"]),
        w=float(frame["w"]),
        h=float(frame["h"]),
        area=float(frame["area"]),
        frame_w=float(frame["frame_w"]),
        frame_h=float(frame["frame_h"]),
    )


def _load_track(track_id: str) -> list[TrackObject]:
    with TRACKS_JSON.open("r") as f:
        tracks_by_id = json.load(f)
    if track_id not in tracks_by_id:
        raise KeyError(f"track_id={track_id!r} not found in {TRACKS_JSON}")
    return sorted(
        [_make_track_object(frame) for frame in tracks_by_id[track_id]],
        key=lambda x: x.frame_id,
    )


def _predict_current(
    track_objects: list[TrackObject],
    index: int,
) -> tuple[tuple[float, float], float, float] | None:
    if index < 2:
        return None

    prev = track_objects[index - 1]
    curr = track_objects[index]
    dt = max(1, int(curr.frame_id - prev.frame_id))
    fit = _fit_track_motion(
        track_objects[:index],
        fit_degree=FIT_DEGREE,
        history_len=HISTORY_LEN,
        predict_dt=dt,
    )
    if fit is None or fit[2] is None:
        return None

    pred_cx, pred_cy = fit[2]
    curr_cx, curr_cy = _track_point(curr)
    err_px = math.hypot(curr_cx - pred_cx, curr_cy - pred_cy)
    err_norm = err_px / math.sqrt(max(curr.area, 1.0))
    return (pred_cx, pred_cy), err_px, err_norm


def _point(pt: tuple[float, float]) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def _draw_error_curve(
    canvas: np.ndarray,
    errors: list[float | None],
    current_index: int,
    top: int,
    width: int,
) -> None:
    cv2.rectangle(canvas, (0, top), (width - 1, top + PLOT_H - 1), (250, 250, 250), -1)
    cv2.line(canvas, (0, top), (width - 1, top), (210, 210, 210), 1)

    valid = [e for e in errors[: current_index + 1] if e is not None]
    max_err = max(1.0, max(valid, default=1.0))
    left, right = 42, width - 12
    bottom = top + PLOT_H - 24
    plot_top = top + 18

    cv2.putText(
        canvas,
        f"prediction error px  max={max_err:.1f}",
        (left, top + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (left, bottom), (right, bottom), (180, 180, 180), 1)
    cv2.line(canvas, (left, plot_top), (left, bottom), (180, 180, 180), 1)

    if len(errors) < 2:
        return

    points: list[tuple[int, int]] = []
    denom = max(1, len(errors) - 1)
    for i, err in enumerate(errors):
        if err is None or i > current_index:
            continue
        x = int(left + (right - left) * i / denom)
        y = int(bottom - (bottom - plot_top) * min(err, max_err) / max_err)
        points.append((x, y))

    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (80, 120, 240), 2)
    if points:
        cv2.circle(canvas, points[-1], 4, (20, 70, 230), -1)


def _draw_frame(
    track_objects: list[TrackObject],
    index: int,
    prediction: tuple[tuple[float, float], float, float] | None,
    errors_px: list[float | None],
) -> np.ndarray:
    curr = track_objects[index]
    curr_cx, curr_cy = _track_point(curr)
    frame_w = int(max(1, round(curr.frame_w)))
    frame_h = int(max(1, round(curr.frame_h)))
    canvas = np.full((frame_h + PLOT_H, frame_w, 3), 245, dtype=np.uint8)

    start = max(0, index - TAIL_LEN)
    tail = track_objects[start : index + 1]
    for a, b in zip(tail, tail[1:]):
        alpha = (b.frame_id - tail[0].frame_id + 1) / max(1, tail[-1].frame_id - tail[0].frame_id + 1)
        color = (int(170 - 90 * alpha), int(170 - 70 * alpha), int(170 + 60 * alpha))
        a_cx, a_cy = _track_point(a)
        b_cx, b_cy = _track_point(b)
        cv2.line(canvas, _point((a_cx, a_cy)), _point((b_cx, b_cy)), color, 2, cv2.LINE_AA)

    for old in tail[:-1]:
        old_cx, old_cy = _track_point(old)
        cv2.circle(canvas, _point((old_cx, old_cy)), 2, (165, 165, 165), -1)

    x1, y1, x2, y2 = [int(round(x)) for x in curr.bbox_xyxy]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (55, 95, 230), 2)
    cv2.circle(canvas, _point((curr_cx, curr_cy)), 5, (40, 70, 235), -1)

    status = "warming up"
    if prediction is not None:
        pred_center, err_px, err_norm = prediction
        pred_pt = _point(pred_center)
        true_pt = _point((curr_cx, curr_cy))
        cv2.circle(canvas, pred_pt, 6, (45, 170, 40), 2, cv2.LINE_AA)
        cv2.line(canvas, true_pt, pred_pt, (40, 150, 220), 2, cv2.LINE_AA)
        status = f"err={err_px:.1f}px  norm={err_norm:.3f}"

    cv2.rectangle(canvas, (8, 8), (390, 76), (255, 255, 255), -1)
    cv2.rectangle(canvas, (8, 8), (390, 76), (210, 210, 210), 1)
    lines = [
        f"track {curr.track_id}  frame {curr.frame_id}  idx {index + 1}/{len(track_objects)}",
        status,
        "red: actual center / green: predicted center",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (18, 30 + row * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    _draw_error_curve(canvas, errors_px, index, frame_h, frame_w)
    return canvas


def main() -> None:
    track_objects = _load_track(TRACK_ID)
    if not track_objects:
        raise RuntimeError(f"track_id={TRACK_ID!r} is empty")

    predictions = [_predict_current(track_objects, i) for i in range(len(track_objects))]
    errors_px = [None if pred is None else pred[1] for pred in predictions]

    frame_w = int(max(1, round(track_objects[0].frame_w)))
    frame_h = int(max(1, round(track_objects[0].frame_h))) + PLOT_H
    temp_video = OUTPUT_VIDEO.with_name(f"{OUTPUT_VIDEO.stem}_raw{OUTPUT_VIDEO.suffix}")
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {temp_video}")

    last_frame = None
    for index, prediction in enumerate(predictions):
        frame = _draw_frame(track_objects, index, prediction, errors_px)
        writer.write(frame)
        last_frame = frame
    writer.release()
    vscode_ready = _transcode_for_vscode(temp_video, OUTPUT_VIDEO)

    if last_frame is not None:
        cv2.imwrite(str(OUTPUT_LAST_FRAME), last_frame)

    valid_errors = [e for e in errors_px if e is not None]
    codec = "h264/yuv420p" if vscode_ready else "opencv fallback"
    print(f"wrote {OUTPUT_VIDEO} ({len(track_objects)} frames, {codec})")
    print(f"wrote {OUTPUT_LAST_FRAME}")
    if valid_errors:
        print(
            "error px: "
            f"mean={np.mean(valid_errors):.2f}, "
            f"max={np.max(valid_errors):.2f}, "
            f"last={valid_errors[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
