from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.scene_features import build_scene_features
from src.features.track_features import build_track_features
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.proposals.boundary_detector import BoundaryDetector
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


def parse_manifest_intervals(manifest_path: str, video_id: str) -> list[list[int]]:
    p = Path(manifest_path)
    if not p.exists() or not video_id:
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if not parts or parts[0] != video_id:
                continue
            nums = [int(x) for x in parts[2:] if x.lstrip("-").isdigit()]
            intervals: list[list[int]] = []
            for i in range(0, len(nums) - 1, 2):
                a, b = nums[i], nums[i + 1]
                if b < a:
                    a, b = b, a
                intervals.append([a, b])
            return intervals
    return []


def _plot_component_group(
    frame_ids: np.ndarray,
    feats: list[dict[str, float]],
    out_dir: Path,
    title_prefix: str,
    gt_intervals: list[list[int]],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    keys: set[str] = set()
    for f in feats:
        keys.update(f.keys())

    paths: dict[str, str] = {}
    for key in sorted(keys):
        y = np.array([float(f.get(key, 0.0)) for f in feats], dtype=np.float32)
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(frame_ids, y, linewidth=1.2)
        for s, e in gt_intervals:
            ax.axvspan(s, e, color="red", alpha=0.12)
        ax.set_title(f"{title_prefix}:{key}")
        ax.set_xlabel("frame_id")
        ax.set_ylabel(key)
        fig.tight_layout()

        p = out_dir / f"{key}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths[key] = str(p)

    return paths


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_settings(args.config)
    frames_dir = Path(args.frames_root) / args.video_id
    frame_paths = list_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    frame_paths = frame_paths[:: max(1, int(args.frame_stride))]

    images: list[np.ndarray] = []
    frame_ids: list[int] = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        images.append(img)
        frame_ids.append(parse_frame_id(p))

    if not images:
        raise ValueError("No readable images after loading")

    frame_ids_np = np.array(frame_ids, dtype=np.int32)
    gt_intervals = parse_manifest_intervals(args.manifest, args.video_id)

    detector = DetectorTracker(
        model_path=cfg.perception.yolo_model,
        tracker=cfg.perception.tracker,
        conf=cfg.perception.conf,
        iou=cfg.perception.iou,
        classes=list(cfg.perception.classes),
    )

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

    all_tracks_per_frame: list[list] = []
    track_feats: list[dict[str, float]] = []
    scene_feats: list[dict[str, float]] = []

    window_size = max(2, int(args.window_size))
    track_fit_degree = max(1, int(args.track_fit_degree))
    track_history_len = int(args.track_history_len) if int(args.track_history_len) > 0 else None
    for i, img in enumerate(images):
        results = detector.track_frame(img, persist=True)
        tracks = parse_ultralytics_results(results, int(frame_ids_np[i]))
        all_tracks_per_frame.append(tracks)

    # Refine fragmented track ids before computing features.
    refined_tracks_per_frame, id_mapping = refine_track_ids(
        all_tracks_per_frame,
        max_frame_gap=refine_max_gap,
        max_center_dist=refine_max_center_dist,
        max_size_ratio=refine_max_size_ratio,
    )

    for i in range(len(images)):
        tracks = refined_tracks_per_frame[i]

        l = max(0, i - window_size + 1)
        win_tracks = refined_tracks_per_frame[l : i + 1]
        track_feats.append(
            build_track_features(
                win_tracks,
                fit_degree=track_fit_degree,
                history_len=track_history_len,
            )
        )
        scene_feats.append(build_scene_features(win_tracks))

    all_tracks_by_id = {}
    for frame_tracks in refined_tracks_per_frame:
        for t in frame_tracks:
            all_tracks_by_id.setdefault(t.track_id, []).append([t.frame_id, t.cx, t.cy])
    all_tracks_path = out_dir / "all_tracks.json"
    with all_tracks_path.open("w", encoding="utf-8") as f:
        json.dump(all_tracks_by_id, f, ensure_ascii=False, indent=2)

    refine_map_path = out_dir / "track_refine_map.json"
    with refine_map_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in id_mapping.items()}, f, ensure_ascii=False, indent=2)

    track_plot_paths = _plot_component_group(
        frame_ids=frame_ids_np,
        feats=track_feats,
        out_dir=out_dir / "track_components",
        title_prefix="track",
        gt_intervals=gt_intervals,
    )
    scene_plot_paths = _plot_component_group(
        frame_ids=frame_ids_np,
        feats=scene_feats,
        out_dir=out_dir / "scene_components",
        title_prefix="scene",
        gt_intervals=gt_intervals,
    )

    track_scores_selected_keys = [
        "dir_turn_rates_sum",
        "pred_center_sum_err",
        "pred_iou_sum_err",
        "speed_turn_rates_sum",
        "turn_active_ratio"
    ]
    track_score_weights = {
        "dir_turn_rates_sum": 0.25,
        "pred_center_sum_err": 0.25,
        "pred_iou_sum_err": 0.25,
        "speed_turn_rates_sum": 0.25,
        "turn_active_ratio": 0.25,
    }
    track_score_selected = np.array([[f.get(k, 0.0) for k in track_scores_selected_keys] for f in track_feats], dtype=np.float32)
    track_score_weights_arr = np.array([track_score_weights.get(k, 0.0) for k in track_scores_selected_keys], dtype=np.float32)
    track_score_final = np.dot(track_score_selected, track_score_weights_arr)

    boundary_cfg = getattr(getattr(cfg, "components", None), "boundary_detector", None)
    detector = BoundaryDetector(
        high=getattr(boundary_cfg, "high", None) if boundary_cfg is not None else None,
        low=getattr(boundary_cfg, "low", None) if boundary_cfg is not None else None,
        online=bool(getattr(boundary_cfg, "online", False)) if boundary_cfg is not None else False,
        use_savgol_filter=bool(getattr(boundary_cfg, "use_savgol_filter", False)) if boundary_cfg is not None else False,
        method=str(getattr(boundary_cfg, "method", "by_peeks")) if boundary_cfg is not None else "by_peeks",
        high_z=float(getattr(boundary_cfg, "high_z", 1.0)) if boundary_cfg is not None else 1.0,
        low_z=float(getattr(boundary_cfg, "low_z", 0.5)) if boundary_cfg is not None else 0.5,
        factor_online=float(getattr(boundary_cfg, "factor_online", 0.3)) if boundary_cfg is not None else 0.3,
        peak_gap=int(getattr(boundary_cfg, "peak_gap", 5)) if boundary_cfg is not None else 5,
        peak_expand=tuple(getattr(boundary_cfg, "peak_expand", [12, 25])) if boundary_cfg is not None else (12, 25),
        min_span_len=int(getattr(boundary_cfg, "min_span_len", 12)) if boundary_cfg is not None else 12,
        merge_gap=int(getattr(boundary_cfg, "merge_gap", 25)) if boundary_cfg is not None else 25,
    )
    det_raw = detector.detect(track_score_final.astype(np.float32).tolist())
    det_intervals = det_raw if isinstance(det_raw, list) else []
    det_peaks: list[tuple[int, int, float]] = []
    for s, e in det_intervals:
        if s > e or s < 0 or e >= int(frame_ids_np.size):
            continue
        rel_idx = int(np.argmax(track_score_final[s : e + 1]))
        p_idx = int(s + rel_idx)
        det_peaks.append((int(s), int(p_idx), float(track_score_final[p_idx])))

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(frame_ids_np, track_score_final, linewidth=1.2)
    for s, e in gt_intervals:
        ax.axvspan(s, e, color="red", alpha=0.12)
    for s, e in det_intervals:
        ax.axvspan(int(frame_ids_np[s]), int(frame_ids_np[e]), color="orange", alpha=0.10)
    if det_peaks:
        peak_x = [int(frame_ids_np[p]) for _, p, _ in det_peaks]
        peak_y = [float(v) for _, _, v in det_peaks]
        ax.scatter(peak_x, peak_y, color="black", s=20, zorder=5, label="detected_peaks")
        ax.legend(loc="upper right")
    ax.set_title(
        "Final Track Anomaly Score "
        f"(boundary={str(getattr(boundary_cfg, 'method', 'by_peeks'))}, "
        f"savgol={bool(getattr(boundary_cfg, 'use_savgol_filter', False)) if boundary_cfg is not None else False})"
    )
    ax.set_xlabel("frame_id")
    ax.set_ylabel("anomaly_score")
    fig.tight_layout()
    final_score_path = out_dir / "final_track_anomaly_score.png"
    fig.savefig(final_score_path, dpi=160)
    plt.close(fig)

    summary = {
        "video_id": args.video_id,
        "num_frames": int(frame_ids_np.size),
        "window_size": int(window_size),
        "track_fit_degree": int(track_fit_degree),
        "track_history_len": int(track_history_len) if track_history_len is not None else int(window_size),
        "track_refine": {
            "max_gap": int(refine_max_gap),
            "max_center_dist": float(refine_max_center_dist),
            "max_size_ratio": float(refine_max_size_ratio),
            "refined_id_count": int(sum(1 for k, v in id_mapping.items() if int(k) != int(v))),
            "mapping_path": str(refine_map_path),
        },
        "gt_intervals": gt_intervals,
        "detected_intervals": [
            [int(frame_ids_np[s]), int(frame_ids_np[e])] for s, e in det_intervals if 0 <= s <= e < int(frame_ids_np.size)
        ],
        "detected_peaks": [
            {
                "start_frame": int(frame_ids_np[s]),
                "peak_frame": int(frame_ids_np[p]),
                "peak_score": float(v),
            }
            for s, p, v in det_peaks
        ],
        "boundary_detector": {
            "method": str(getattr(boundary_cfg, "method", "by_peeks")) if boundary_cfg is not None else "by_peeks",
            "high": getattr(boundary_cfg, "high", None) if boundary_cfg is not None else None,
            "low": getattr(boundary_cfg, "low", None) if boundary_cfg is not None else None,
            "online": bool(getattr(boundary_cfg, "online", False)) if boundary_cfg is not None else False,
            "use_savgol_filter": bool(getattr(boundary_cfg, "use_savgol_filter", False)) if boundary_cfg is not None else False,
            "high_z": float(getattr(boundary_cfg, "high_z", 1.0)) if boundary_cfg is not None else 1.0,
            "low_z": float(getattr(boundary_cfg, "low_z", 0.5)) if boundary_cfg is not None else 0.5,
            "factor_online": float(getattr(boundary_cfg, "factor_online", 0.3)) if boundary_cfg is not None else 0.3,
            "peak_gap": int(getattr(boundary_cfg, "peak_gap", 5)) if boundary_cfg is not None else 5,
            "peak_expand": list(getattr(boundary_cfg, "peak_expand", [12, 25])) if boundary_cfg is not None else [12, 25],
            "min_span_len": int(getattr(boundary_cfg, "min_span_len", 12)) if boundary_cfg is not None else 12,
            "merge_gap": int(getattr(boundary_cfg, "merge_gap", 25)) if boundary_cfg is not None else 25,
        },
        "paths": {
            "track_components": track_plot_paths,
            "scene_components": scene_plot_paths,
            "final_score": str(final_score_path),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Done. Raw component plots saved at: %s", out_dir)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw scene/track feature components only")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--frames-root",
        default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal",
        help="Path to root directory containing video frame directories",
    )
    parser.add_argument("--video-id", default="v1")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--track-fit-degree", type=int, default=2)
    parser.add_argument("--track-history-len", type=int, default=0, help="0 means use window-size")
    parser.add_argument("--refine-max-gap", type=int, default=None)
    parser.add_argument("--refine-max-center-dist", type=float, default=None)
    parser.add_argument("--refine-max-size-ratio", type=float, default=None)
    parser.add_argument("--out-dir", default="outputs/debug/raw_components")
    return parser.parse_args()


if __name__ == "__main__":
    run(build_args())
