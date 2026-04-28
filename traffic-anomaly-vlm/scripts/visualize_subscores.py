from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.utils import parse_manifest_intervals
from src.eval.visualize import plot_all_subscores
from tmp.orchestrator_raw_score import RawScoreOrchestrator


TRACK_KEYS = [
    "track_count",
    "speeds_mean",
    "pred_center_max_err",
    "pred_iou_max_err",
    "pred_center_sum_err",
    "pred_iou_sum_err",
    "speed_turn_rates_max",
    "speed_turn_rates_sum",
    "dir_turn_rates_max",
    "dir_turn_rates_sum",
    "turn_active_ratio",
    "collision_warning",
    "disappear_far_sum",
]

SCENE_KEYS = [
    "cur_count",
    "count_mean",
    "count_std",
    "count_delta",
    "cur_density",
    "density_mean",
    "density_std",
    "density_delta",
    "lowres_eventness",
    "track_layout_eventness",
    "clip_eventness",
    "raft_eventness",
]


def args_main() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw track/scene subscores for all videos.")
    parser.add_argument("--cfg", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--frames-root",
        type=str,
        default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs/subscores"),
    )
    parser.add_argument(
        "--reuse-cache-dir",
        type=str,
        default=str(ROOT / "tmp/raw_score_search/cache"),
    )
    parser.add_argument("--precompute-clip", action="store_true")
    parser.add_argument("--precompute-raft", action="store_true")
    return parser.parse_args()


def _load_or_extract_payload(
    extractor: RawScoreOrchestrator,
    video_dir: Path,
    cache_dir: Path,
    precompute_clip: bool,
    precompute_raft: bool,
) -> dict:
    video_id = video_dir.name
    cached_feature_path = cache_dir / video_id / "feature_rows.json"
    if cached_feature_path.exists():
        return json.loads(cached_feature_path.read_text(encoding="utf-8"))

    video_cache_dir = cache_dir / video_id
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    extractor.precompute_clip = bool(precompute_clip)
    extractor.precompute_raft = bool(precompute_raft)
    return extractor.extract_video(
        frames_dir=str(video_dir),
        output_dir=str(video_cache_dir),
        video_id=video_id,
    )


def _dense_series(values: list[float], num_frames: int, window_size: int) -> list[float]:
    if num_frames <= 0:
        return []
    if not values:
        return [0.0] * num_frames
    prefix_len = max(0, int(window_size) - 1)
    dense = [float(values[0])] * prefix_len + [float(v) for v in values]
    if len(dense) < num_frames:
        dense.extend([float(dense[-1])] * (num_frames - len(dense)))
    return dense[:num_frames]


def _build_dense_subscores(payload: dict, keys: list[str]) -> dict[str, list[float]]:
    rows = [row["feature_dict"] for row in payload.get("window_rows", [])]
    frame_ids = payload.get("sampled_frame_ids", [])
    window_size = int(payload.get("window_size", 16))
    curves: dict[str, list[float]] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        curves[key] = _dense_series(values, len(frame_ids), window_size)
    return curves


def _save_csv(path: Path, frame_ids: list[int], curves: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["frame_id", *curves.keys()]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for index, frame_id in enumerate(frame_ids):
            row = [int(frame_id)]
            for key in curves:
                values = curves[key]
                row.append(float(values[index]) if index < len(values) else 0.0)
            writer.writerow(row)


def main() -> None:
    args = args_main()
    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.reuse_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = RawScoreOrchestrator(
        args.cfg,
        precompute_clip=bool(args.precompute_clip),
        precompute_raft=bool(args.precompute_raft),
    )

    summaries: list[dict] = []
    video_dirs = sorted(path for path in frames_root.iterdir() if path.is_dir())
    for video_dir in video_dirs:
        video_id = video_dir.name
        payload = _load_or_extract_payload(
            extractor=extractor,
            video_dir=video_dir,
            cache_dir=cache_dir,
            precompute_clip=bool(args.precompute_clip),
            precompute_raft=bool(args.precompute_raft),
        )
        frame_ids = [int(v) for v in payload.get("sampled_frame_ids", [])]
        gt_intervals = parse_manifest_intervals(args.manifest, video_id)
        track_curves = _build_dense_subscores(payload, TRACK_KEYS)
        scene_curves = _build_dense_subscores(payload, SCENE_KEYS)

        video_out_dir = output_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(video_out_dir / "track_subscores.csv", frame_ids, track_curves)
        _save_csv(video_out_dir / "scene_subscores.csv", frame_ids, scene_curves)

        plot_all_subscores(
            frame_ids=frame_ids,
            subscores=track_curves,
            gt_intervals=gt_intervals,
            title=f"{video_id} Track Subscores",
            output_dir=str(video_out_dir),
            filename="track_subscores.png",
        )
        plot_all_subscores(
            frame_ids=frame_ids,
            subscores=scene_curves,
            gt_intervals=gt_intervals,
            title=f"{video_id} Scene Subscores",
            output_dir=str(video_out_dir),
            filename="scene_subscores.png",
        )

        summaries.append(
            {
                "video_id": video_id,
                "num_sampled_frames": int(len(frame_ids)),
                "num_track_components": int(len(track_curves)),
                "num_scene_components": int(len(scene_curves)),
                "gt_intervals": [[int(a), int(b)] for a, b in gt_intervals],
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames_root": str(frames_root),
                "num_videos": int(len(summaries)),
                "videos": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
