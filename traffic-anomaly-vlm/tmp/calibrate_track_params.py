from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.sampler import FrameSampler
from src.core.video_io import build_reader
from src.eval.utils import parse_manifest_intervals
from src.features.feature_components import track as track_features
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.schemas import TrackObject
from src.settings import instantiate_from_config, load_settings


def _track_point(obj: TrackObject, y_ratio: float = track_features.TRACK_POINT_Y_RATIO) -> tuple[float, float]:
    x1, y1, x2, y2 = obj.bbox_xyxy
    return float(x1 + x2) * 0.5, float(y1) + float(y2 - y1) * float(y_ratio)


def _track_scale(obj: TrackObject) -> float:
    return math.sqrt(max(float(obj.area), 1.0))


def _box_area_ratio(a: TrackObject, b: TrackObject) -> float:
    area_a = max(float(a.area), 1.0)
    area_b = max(float(b.area), 1.0)
    return max(area_a, area_b) / min(area_a, area_b)


def _edge_distance_norm(obj: TrackObject) -> float:
    x, y = _track_point(obj)
    fw = max(1.0, float(obj.frame_w))
    fh = max(1.0, float(obj.frame_h))
    x = min(max(0.0, x), fw)
    y = min(max(0.0, y), fh)
    return min(x, fw - x, y, fh - y) / max(1.0, math.hypot(fw, fh))


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
    return float(inter / max(1e-6, area_a + area_b - inter))


def _has_near_replacement(last_track: TrackObject, next_tracks: list[TrackObject], frame_diag: float) -> bool:
    lx, ly = _track_point(last_track)
    for candidate in next_tracks:
        if int(candidate.track_id) == int(last_track.track_id):
            continue
        if int(candidate.cls_id) != int(last_track.cls_id):
            continue
        cx, cy = _track_point(candidate)
        center_dist = math.hypot(cx - lx, cy - ly) / max(1.0, frame_diag)
        if center_dist <= track_features.REPLACEMENT_CENTER_RATIO:
            return True
        if _iou_xyxy(last_track.bbox_xyxy, candidate.bbox_xyxy) >= track_features.REPLACEMENT_IOU:
            return True
    return False


def _is_gt_frame(frame_id: int, intervals: list[list[int]]) -> bool:
    return any(int(start) <= int(frame_id) <= int(end) for start, end in intervals)


def _percentiles(values: list[float], qs: Iterable[float]) -> dict[str, float]:
    if not values:
        return {f"p{q:g}": 0.0 for q in qs}
    arr = np.asarray(values, dtype=np.float32)
    return {f"p{q:g}": float(np.percentile(arr, q)) for q in qs}


def _load_cached_tracks(cache_path: Path) -> tuple[list[int], list[list[TrackObject]]] | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text())
    frame_ids = [int(v) for v in payload["frame_ids"]]
    tracks = [[TrackObject(**item) for item in row] for row in payload["tracks_per_frame"]]
    return frame_ids, tracks


def _save_cached_tracks(cache_path: Path, frame_ids: list[int], tracks: list[list[TrackObject]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "frame_ids": [int(v) for v in frame_ids],
                "tracks_per_frame": [[t.model_dump() for t in row] for row in tracks],
            },
            indent=2,
        )
    )


def _track_video(video_path: Path, cfg, detector_tracker: DetectorTracker, cache_path: Path) -> tuple[list[int], list[list[TrackObject]]]:
    cached = _load_cached_tracks(cache_path)
    if cached is not None:
        return cached

    reader = build_reader(
        str(video_path),
        input_fps=float(cfg.video.input_fps),
        resize_max_side=int(cfg.video.resize_max_side),
        resize_interpolation=str(cfg.video.resize_interpolation),
    )
    sampler = FrameSampler(reader.fps(), float(cfg.video.fps_sample))
    frame_ids: list[int] = []
    tracks_raw: list[list[TrackObject]] = []
    try:
        total_frames = int(reader.frame_count())
        total_sampled = None if total_frames <= 0 else (total_frames + sampler.stride - 1) // sampler.stride
        for frame_id, frame in tqdm(
            sampler.sample(iter(reader)),
            total=total_sampled,
            desc=f"{video_path.stem}: track",
            dynamic_ncols=True,
        ):
            frame_ids.append(int(frame_id))
            results = detector_tracker.track_frame(frame, persist=True)
            tracks_raw.append(parse_ultralytics_results(results, frame_id))
    finally:
        reader.release()

    tracks = tracks_raw
    if bool(getattr(cfg.track_refiner, "enable", False)):
        tracks, _ = refine_track_ids(
            tracks_raw,
            max_frame_gap=cfg.track_refiner.max_frame_gap,
            max_center_dist=cfg.track_refiner.max_center_dist,
            max_size_ratio=cfg.track_refiner.max_size_ratio,
            min_direction_cos=cfg.track_refiner.min_direction_cos,
            max_speed_ratio=cfg.track_refiner.max_speed_ratio,
            gap_relax_factor=cfg.track_refiner.gap_relax_factor,
            min_track_len=getattr(cfg.track_refiner, "min_track_len", 3),
            min_track_span=getattr(cfg.track_refiner, "min_track_span", 4),
            min_box_area=getattr(cfg.track_refiner, "min_box_area", 20.0),
            min_area_ratio=getattr(cfg.track_refiner, "min_area_ratio", 0.0005),
        )
    _save_cached_tracks(cache_path, frame_ids, tracks)
    return frame_ids, tracks


def _collect_video_stats(frame_ids: list[int], tracks_per_frame: list[list[TrackObject]], gt_intervals: list[list[int]], window_size: int) -> dict[str, list[float]]:
    by_id: dict[int, list[TrackObject]] = {}
    frame_to_tracks = dict(zip(frame_ids, tracks_per_frame))
    for frame_tracks in tracks_per_frame:
        for obj in frame_tracks:
            if not _is_gt_frame(int(obj.frame_id), gt_intervals):
                by_id.setdefault(int(obj.track_id), []).append(obj)

    stats: dict[str, list[float]] = {
        "track_disp_ratio": [],
        "edge_distance_norm": [],
        "area_ratio": [],
        "step_ratio": [],
        "step_px": [],
        "low_disp_step_ratio": [],
        "low_disp_step_px": [],
        "triplet_min_step_ratio": [],
        "triplet_min_step_px": [],
        "disappear_edge_norm": [],
    }

    for seq in by_id.values():
        seq = sorted(seq, key=lambda x: x.frame_id)
        if len(seq) >= max(3, int(window_size)):
            x0, y0 = _track_point(seq[0])
            x1, y1 = _track_point(seq[-1])
            avg_scale = math.sqrt(max(1.0, sum(float(t.area) for t in seq) / len(seq)))
            disp_ratio = math.hypot(x1 - x0, y1 - y0) / avg_scale
            stats["track_disp_ratio"].append(float(disp_ratio))

        low_disp_track = False
        if len(seq) >= 3:
            x0, y0 = _track_point(seq[0])
            x1, y1 = _track_point(seq[-1])
            avg_scale = math.sqrt(max(1.0, sum(float(t.area) for t in seq) / len(seq)))
            low_disp_track = (math.hypot(x1 - x0, y1 - y0) / avg_scale) < 0.5

        for obj in seq:
            stats["edge_distance_norm"].append(_edge_distance_norm(obj))

        pair_steps_ratio: list[float] = []
        pair_steps_px: list[float] = []
        for prev, cur in zip(seq[:-1], seq[1:]):
            if _is_gt_frame(int(prev.frame_id), gt_intervals) or _is_gt_frame(int(cur.frame_id), gt_intervals):
                continue
            px, py = _track_point(prev)
            cx, cy = _track_point(cur)
            dt = max(1, int(cur.frame_id - prev.frame_id))
            scale = max(1.0, 0.5 * (_track_scale(prev) + _track_scale(cur)))
            step_px = math.hypot(cx - px, cy - py)
            step_ratio = step_px / scale
            stats["area_ratio"].append(_box_area_ratio(prev, cur))
            stats["step_px"].append(step_px)
            stats["step_ratio"].append(step_ratio)
            pair_steps_px.append(step_px)
            pair_steps_ratio.append(step_ratio)
            if low_disp_track:
                stats["low_disp_step_px"].append(step_px)
                stats["low_disp_step_ratio"].append(step_ratio)

        for i in range(1, len(seq) - 1):
            p0, p1, p2 = seq[i - 1], seq[i], seq[i + 1]
            if any(_is_gt_frame(int(p.frame_id), gt_intervals) for p in (p0, p1, p2)):
                continue
            if i - 1 >= len(pair_steps_px) or i >= len(pair_steps_px):
                continue
            stats["triplet_min_step_px"].append(min(pair_steps_px[i - 1], pair_steps_px[i]))
            stats["triplet_min_step_ratio"].append(min(pair_steps_ratio[i - 1], pair_steps_ratio[i]))

    for idx, frame_id in enumerate(frame_ids[:-1]):
        if _is_gt_frame(frame_id, gt_intervals):
            continue
        cur_tracks = [t for t in tracks_per_frame[idx] if not _is_gt_frame(int(t.frame_id), gt_intervals)]
        if not cur_tracks:
            continue
        lookahead_tracks = [
            t
            for row in tracks_per_frame[idx + 1 : idx + 1 + track_features.DISAPPEAR_LOOKAHEAD]
            for t in row
            if not _is_gt_frame(int(t.frame_id), gt_intervals)
        ]
        future_ids = {int(t.track_id) for t in lookahead_tracks}
        frame_diag = max(
            1.0,
            math.hypot(
                max((float(t.frame_w) for t in cur_tracks), default=640.0),
                max((float(t.frame_h) for t in cur_tracks), default=480.0),
            ),
        )
        for obj in cur_tracks:
            tid = int(obj.track_id)
            if tid in future_ids:
                continue
            observed_len = sum(1 for t in by_id.get(tid, []) if int(t.frame_id) <= int(obj.frame_id))
            if observed_len < track_features.DISAPPEAR_MIN_TRACK_LEN:
                continue
            if _has_near_replacement(obj, lookahead_tracks, frame_diag):
                continue
            stats["disappear_edge_norm"].append(_edge_distance_norm(obj))

    return stats


def _extend_stats(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, []).extend(float(v) for v in values)


def _recommend(stats: dict[str, list[float]]) -> dict[str, float]:
    low_step_ratio = stats.get("low_disp_step_ratio", [])
    low_step_px = stats.get("low_disp_step_px", [])
    step_ratio = stats.get("step_ratio", [])
    area_ratio = stats.get("area_ratio", [])
    track_disp = stats.get("track_disp_ratio", [])
    disappear_edge = stats.get("disappear_edge_norm", [])

    return {
        "STATIC_MOVE_RATIO": float(np.clip(np.percentile(track_disp, 35) if track_disp else 0.15, 0.10, 0.30)),
        "DISAPPEAR_FAR_EDGE_RATIO": float(np.clip(np.percentile(disappear_edge, 90) if disappear_edge else 0.18, 0.15, 0.25)),
        "DISAPPEAR_LOOKAHEAD": float(track_features.DISAPPEAR_LOOKAHEAD),
        "DISAPPEAR_MIN_TRACK_LEN": float(track_features.DISAPPEAR_MIN_TRACK_LEN),
        "TURN_CHANGE_RATIO": float(track_features.TURN_CHANGE_RATIO),
        "SPEED_SCALING_FACTOR": float(track_features.SPEED_SCALING_FACTOR),
        "MIN_TURN_SPEED_RATIO": float(np.clip(np.percentile(low_step_ratio, 90) if low_step_ratio else 0.05, 0.03, 0.10)),
        "MIN_DIR_MOTION_RATIO": float(np.clip(np.percentile(low_step_ratio, 85) if low_step_ratio else 0.04, 0.03, 0.10)),
        "MIN_DIR_MOTION_PX": float(np.clip(np.percentile(low_step_px, 90) if low_step_px else 3.0, 2.0, 8.0)),
        "TOPK_AGG_COUNT": float(track_features.TOPK_AGG_COUNT),
        "EDGE_UNSTABLE_RATIO": float(track_features.EDGE_UNSTABLE_RATIO),
        "MAX_MOTION_AREA_RATIO": float(np.clip(np.percentile(area_ratio, 99) if area_ratio else 1.5, 1.5, 3.0)),
        "MAX_STEP_RATIO": float(np.clip(np.percentile(step_ratio, 98) if step_ratio else 1.5, 1.0, 3.0)),
        "REPLACEMENT_CENTER_RATIO": float(track_features.REPLACEMENT_CENTER_RATIO),
        "REPLACEMENT_IOU": float(track_features.REPLACEMENT_IOU),
        "TRACK_POINT_Y_RATIO": float(track_features.TRACK_POINT_Y_RATIO),
    }


def _write_report(out_dir: Path, stats: dict[str, list[float]], recommendation: dict[str, float], video_ids: list[str]) -> None:
    lines = [
        "# Track Parameter Calibration",
        "",
        f"Videos: {', '.join(video_ids)}",
        "",
        "Statistics are computed on sampled/refined tracks outside manifest anomaly intervals.",
        "",
        "## Distributions",
        "",
    ]
    for key in sorted(stats):
        values = stats[key]
        pct = _percentiles(values, [5, 25, 50, 75, 85, 90, 95, 98, 99])
        lines.append(f"### {key}")
        lines.append(f"count: {len(values)}")
        lines.append("")
        lines.append("| percentile | value |")
        lines.append("|---:|---:|")
        for name, value in pct.items():
            lines.append(f"| {name} | {value:.6f} |")
        lines.append("")

    lines.extend(["## Recommendation", "", "| parameter | value |", "|---|---:|"])
    for key, value in recommendation.items():
        if value.is_integer():
            lines.append(f"| {key} | {int(value)} |")
        else:
            lines.append(f"| {key} | {value:.6f} |")
    lines.append("")

    out_dir.joinpath("track_param_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--videos", nargs="*", default=["v1", "v3", "v10", "v15", "v20"])
    parser.add_argument("--video-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/videos/abnormal")
    parser.add_argument("--out-dir", default="tmp/track_param_calibration")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_settings(args.config)
    detector_tracker = DetectorTracker(instantiate_from_config(cfg)["perception"])
    manifest_path = str(getattr(getattr(cfg, "evaluation", None), "manifest_path", "") or "")

    all_stats: dict[str, list[float]] = {}
    used_videos: list[str] = []
    for video_id in args.videos:
        video_path = Path(args.video_root) / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"skip missing video: {video_path}")
            continue
        frame_ids, tracks = _track_video(video_path, cfg, detector_tracker, out_dir / "cache" / f"{video_id}_tracks.json")
        gt_intervals = parse_manifest_intervals(manifest_path, video_id) if manifest_path else []
        video_stats = _collect_video_stats(frame_ids, tracks, gt_intervals, int(cfg.video.window_size))
        _extend_stats(all_stats, video_stats)
        used_videos.append(video_id)

    recommendation = _recommend(all_stats)
    out_dir.joinpath("track_param_stats.json").write_text(
        json.dumps(
            {
                "videos": used_videos,
                "counts": {key: len(values) for key, values in all_stats.items()},
                "percentiles": {
                    key: _percentiles(values, [5, 25, 50, 75, 85, 90, 95, 98, 99])
                    for key, values in all_stats.items()
                },
                "recommendation": recommendation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(out_dir, all_stats, recommendation, used_videos)
    print(json.dumps({"videos": used_videos, "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
