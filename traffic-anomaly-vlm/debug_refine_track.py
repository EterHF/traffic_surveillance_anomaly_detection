from pathlib import Path
import subprocess

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.schemas import TrackObject


CONFIG_PATH = "configs/default.yaml"
VIDEO_ID = "v1"
FRAMES_PATH = Path(f"/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal/{VIDEO_ID}")
OUTPUT_VIDEO = Path(f"refined_tracks_{VIDEO_ID}.mp4")
OUTPUT_FPS = 25

# Set to an int when you only want a quick parameter smoke test.
MAX_FRAMES: int | None = None
TAIL_LEN = 48

# TRACKER = "configs/botsort.yaml"
TRACK_PARAMS = {
    "max_frame_gap": 8,
    "max_center_dist": 0.06,
    "max_size_ratio": 2.5,
    "min_direction_cos": -0.1,
    "max_speed_ratio": 3.5,
    "gap_relax_factor": 0.20,
}


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


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 9176 + 13)
    color = rng.integers(70, 235, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def _frame_files(frames_path: Path) -> list[Path]:
    files = sorted(
        p for p in frames_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if MAX_FRAMES is not None:
        files = files[:MAX_FRAMES]
    if not files:
        raise FileNotFoundError(f"No image frames found in {frames_path}")
    return files


def _load_detector():
    from src.perception.detector_tracker import DetectorTracker, DetectorTrackerConfig

    with Path(CONFIG_PATH).open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    perception_cfg = dict(raw_cfg.get("perception", {}))
    # tracker_path = TRACKER
    # perception_cfg["tracker"] = tracker_path
    print(f"loaded perception config: {perception_cfg}")

    return DetectorTracker(DetectorTrackerConfig(**perception_cfg))


def _track_frames(frame_files: list[Path]) -> tuple[list[np.ndarray], list[list[TrackObject]]]:
    detector_tracker = _load_detector()
    frames: list[np.ndarray] = []
    tracks_per_frame: list[list[TrackObject]] = []

    for frame_path in tqdm(frame_files, desc="track frames", dynamic_ncols=True):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        frame_id = int(frame_path.stem)
        results = detector_tracker.track_frame(frame, persist=True)
        tracks = parse_ultralytics_results(results, frame_id)
        frames.append(frame)
        tracks_per_frame.append(tracks)

    return frames, tracks_per_frame


def _draw_tracks(
    frame: np.ndarray,
    raw_tracks: list[TrackObject],
    refined_tracks: list[TrackObject],
    history: dict[int, list[tuple[int, int]]],
    frame_index: int,
    changed_count: int,
) -> np.ndarray:
    canvas = frame.copy()

    for raw, refined in zip(raw_tracks, refined_tracks):
        tid = int(refined.track_id)
        old_tid = int(raw.track_id)
        color = _color_for_id(tid)

        cx, cy = int(round(refined.cx)), int(round(refined.cy))
        points = history.setdefault(tid, [])
        points.append((cx, cy))
        if len(points) > TAIL_LEN:
            del points[:-TAIL_LEN]

        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 4, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(round(v)) for v in refined.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, (cx, cy), 3, color, -1, cv2.LINE_AA)

        label = f"id {tid}"
        if old_tid != tid:
            label = f"id {tid} <- {old_tid}"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (8, 8), (370, 68), (255, 255, 255), -1)
    cv2.rectangle(canvas, (8, 8), (370, 68), (210, 210, 210), 1)
    cv2.putText(
        canvas,
        f"{VIDEO_ID} refined tracks  frame_idx={frame_index}",
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"objects={len(refined_tracks)}  remapped_ids={changed_count}",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_video(
    frames: list[np.ndarray],
    raw_tracks_per_frame: list[list[TrackObject]],
    refined_tracks_per_frame: list[list[TrackObject]],
    id_map: dict[int, int],
) -> None:
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    temp_video = OUTPUT_VIDEO.with_name(f"{OUTPUT_VIDEO.stem}_raw{OUTPUT_VIDEO.suffix}")
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {temp_video}")

    history: dict[int, list[tuple[int, int]]] = {}
    changed_count = sum(1 for old, new in id_map.items() if int(old) != int(new))

    for idx, (frame, raw_tracks, refined_tracks) in enumerate(
        tqdm(
            zip(frames, raw_tracks_per_frame, refined_tracks_per_frame),
            total=len(frames),
            desc="write video",
            dynamic_ncols=True,
        )
    ):
        vis = _draw_tracks(frame, raw_tracks, refined_tracks, history, idx, changed_count)
        writer.write(vis)

    writer.release()
    vscode_ready = _transcode_for_vscode(temp_video, OUTPUT_VIDEO)
    codec = "h264/yuv420p" if vscode_ready else "opencv fallback"
    print(f"video codec: {codec}")


def main() -> None:
    frame_files = _frame_files(FRAMES_PATH)
    frames, raw_tracks_per_frame = _track_frames(frame_files)
    refined_tracks_per_frame, id_map = refine_track_ids(raw_tracks_per_frame, **TRACK_PARAMS)
    # refined_tracks_per_frame, id_map = raw_tracks_per_frame, {}

    changed = {old: new for old, new in id_map.items() if int(old) != int(new)}
    print(f"wrote {OUTPUT_VIDEO}")
    print(f"frames={len(frames)} raw_ids={len(id_map)} remapped_ids={len(changed)}")
    if changed:
        print(f"remap preview: {dict(list(changed.items())[:20])}")


if __name__ == "__main__":
    main()
