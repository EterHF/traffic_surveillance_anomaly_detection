from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.schemas import TrackObject
from src.settings import load_settings


def list_frame_paths(frame_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in Path(frame_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return [str(p) for p in files]


def parse_frame_id(path: str) -> int:
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def collect_tracks_per_frame(args: argparse.Namespace) -> list[list[TrackObject]]:
    cfg = load_settings(args.config)
    frames_dir = Path(args.frames_root) / args.video_id
    frame_paths = list_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    frame_paths = frame_paths[:: max(1, int(args.frame_stride))]
    if int(args.max_frames) > 0:
        frame_paths = frame_paths[: int(args.max_frames)]

    detector = DetectorTracker(
        model_path=cfg.perception.yolo_model,
        tracker=cfg.perception.tracker,
        conf=cfg.perception.conf,
        iou=cfg.perception.iou,
        classes=list(cfg.perception.classes),
    )

    all_tracks_per_frame: list[list[TrackObject]] = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            all_tracks_per_frame.append([])
            continue
        fid = parse_frame_id(p)
        results = detector.track_frame(img, persist=True)
        tracks = parse_ultralytics_results(results, fid)
        all_tracks_per_frame.append(tracks)

    return all_tracks_per_frame


def _stats(tracks_per_frame: list[list[TrackObject]]) -> dict[str, float]:
    by_id: dict[int, list[int]] = {}
    total_boxes = 0
    for frame_tracks in tracks_per_frame:
        total_boxes += len(frame_tracks)
        for t in frame_tracks:
            by_id.setdefault(int(t.track_id), []).append(int(t.frame_id))

    lengths = [len(v) for v in by_id.values()]
    lengths_sorted = sorted(lengths)

    gap_count = 0
    max_gap = 0
    for fids in by_id.values():
        f = sorted(fids)
        for i in range(1, len(f)):
            g = f[i] - f[i - 1]
            if g > 1:
                gap_count += 1
                if g > max_gap:
                    max_gap = g

    return {
        "num_ids": float(len(by_id)),
        "total_boxes": float(total_boxes),
        "mean_track_len": float(statistics.mean(lengths)) if lengths else 0.0,
        "median_track_len": float(statistics.median(lengths_sorted)) if lengths_sorted else 0.0,
        "short_tracks_le_3": float(sum(1 for x in lengths if x <= 3)),
        "single_frame_tracks": float(sum(1 for x in lengths if x <= 1)),
        "ids_with_internal_gaps": float(gap_count),
        "max_internal_gap": float(max_gap),
    }


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick check before/after track refine effectiveness")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--video-id", default="v1")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use all frames")

    parser.add_argument("--refine-max-gap", type=int, default=None)
    parser.add_argument("--refine-max-center-dist", type=float, default=None)
    parser.add_argument("--refine-max-size-ratio", type=float, default=None)

    parser.add_argument("--out-dir", default="outputs/test/refine_check")
    return parser.parse_args()


def main() -> None:
    args = build_args()
    cfg = load_settings(args.config)
    ref_cfg = getattr(getattr(cfg, "perception", None), "track_refiner", None)
    refine_max_gap = int(args.refine_max_gap) if args.refine_max_gap is not None else int(getattr(ref_cfg, "max_gap", 8))
    refine_max_center_dist = (
        float(args.refine_max_center_dist)
        if args.refine_max_center_dist is not None
        else float(getattr(ref_cfg, "max_center_dist", 0.06))
    )
    refine_max_size_ratio = (
        float(args.refine_max_size_ratio)
        if args.refine_max_size_ratio is not None
        else float(getattr(ref_cfg, "max_size_ratio", 2.5))
    )

    raw = collect_tracks_per_frame(args)
    refined, id_map = refine_track_ids(
        raw,
        max_frame_gap=refine_max_gap,
        max_center_dist=refine_max_center_dist,
        max_size_ratio=refine_max_size_ratio,
    )

    raw_stats = _stats(raw)
    refined_stats = _stats(refined)

    merged = {int(k): int(v) for k, v in id_map.items() if int(k) != int(v)}
    summary = {
        "video_id": args.video_id,
        "num_frames": len(raw),
        "refine_params": {
            "max_gap": int(refine_max_gap),
            "max_center_dist": float(refine_max_center_dist),
            "max_size_ratio": float(refine_max_size_ratio),
        },
        "raw_stats": raw_stats,
        "refined_stats": refined_stats,
        "deltas": {
            "num_ids_delta": refined_stats["num_ids"] - raw_stats["num_ids"],
            "mean_track_len_delta": refined_stats["mean_track_len"] - raw_stats["mean_track_len"],
            "short_tracks_le_3_delta": refined_stats["short_tracks_le_3"] - raw_stats["short_tracks_le_3"],
            "single_frame_tracks_delta": refined_stats["single_frame_tracks"] - raw_stats["single_frame_tracks"],
        },
        "merged_count": len(merged),
        "merged_id_map": merged,
    }

    out_dir = Path(args.out_dir) / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quick_refine_check.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved: {out_path}")
    print("raw ids:", int(raw_stats["num_ids"]), "-> refined ids:", int(refined_stats["num_ids"]))
    print("short<=3:", int(raw_stats["short_tracks_le_3"]), "->", int(refined_stats["short_tracks_le_3"]))
    print("mean len:", f"{raw_stats['mean_track_len']:.3f}", "->", f"{refined_stats['mean_track_len']:.3f}")
    print("merged id count:", len(merged))


if __name__ == "__main__":
    main()
