from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
	
from src.features.feature_components.track import build_track_features
from src.core.video_io import build_reader
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.settings import load_settings


def _color_from_track_id(track_id: int) -> tuple[int, int, int]:
	# Stable pseudo-color for each track id.
	return (
		int((track_id * 37) % 255),
		int((track_id * 67) % 255),
		int((track_id * 97) % 255),
	)


def _open_writer(save_path: str, fps: float, frame_w: int, frame_h: int, codec: str) -> cv2.VideoWriter:
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*codec)
	writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open VideoWriter: {save_path}")
	return writer


def _draw_feature_panel(
	frame_h: int,
	panel_w: int,
	track_feats: dict[str, float],
	feat_histories: dict[str, deque[float]],
	history_points: int,
) -> np.ndarray:
	panel = np.full((frame_h, panel_w, 3), 22, dtype=np.uint8)
	cv2.putText(panel, "track_features", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 230, 255), 2, cv2.LINE_AA)

	keys = sorted(track_feats.keys())
	row_h = 34
	y = 42
	for key in keys:
		if y + row_h > frame_h - 4:
			break
		v = float(track_feats.get(key, 0.0))
		cv2.putText(
			panel,
			f"{key}: {v:.4f}",
			(10, y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.44,
			(235, 235, 235),
			1,
			cv2.LINE_AA,
		)

		hist = list(feat_histories.get(key, []))[-max(8, history_points):]
		x0, x1 = 10, panel_w - 10
		y0, y1 = y + 5, y + row_h - 6
		cv2.rectangle(panel, (x0, y0), (x1, y1), (60, 60, 60), 1)
		if len(hist) >= 2:
			h_arr = np.array(hist, dtype=np.float32)
			h_min = float(np.min(h_arr))
			h_max = float(np.max(h_arr))
			if h_max - h_min < 1e-8:
				h_max = h_min + 1e-8

			xs = np.linspace(x0 + 1, x1 - 1, num=len(h_arr)).astype(np.int32)
			ys = (y1 - 1 - (h_arr - h_min) / (h_max - h_min) * max(1, (y1 - y0 - 2))).astype(np.int32)
			pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
			cv2.polylines(panel, [pts], False, (85, 200, 255), 1, cv2.LINE_AA)

		y += row_h

	return panel


def visualize_tracking_to_video(
	input_source: str,
	output_video: str,
	model_path: str,
	tracker_cfg: str,
	conf: float,
	iou: float,
	classes: list[int] | None,
	input_fps: float,
	output_fps: float,
	trail_len: int,
	window_size: int,
	track_fit_degree: int,
	track_history_len: int,
	refine_max_gap: int,
	refine_max_center_dist: float,
	refine_max_size_ratio: float,
	feature_panel_width: int,
	feature_history_points: int,
	resize_max_side: int,
	resize_interpolation: str,
	codec: str
) -> str:
	detector = DetectorTracker(
		model_path=model_path,
		tracker=tracker_cfg,
		conf=conf,
		iou=iou,
		classes=classes,
	)
	reader = build_reader(
		input_source,
		input_fps=input_fps,
		resize_max_side=resize_max_side,
		resize_interpolation=resize_interpolation,
	)

	total = reader.frame_count()

	history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_len))
	window_tracks: deque[list] = deque(maxlen=max(2, int(window_size)))
	feat_histories: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(8, int(feature_history_points))))
	writer: cv2.VideoWriter | None = None

	fps = float(output_fps if output_fps > 0 else (reader.fps() or input_fps or 30.0))

	try:
		# Stage 1: detect and collect raw tracks.
		raw_frames: list[np.ndarray] = []
		raw_frame_ids: list[int] = []
		raw_tracks_per_frame: list[list] = []
		collect_bar = tqdm(total=total if total > 0 else None, desc="Collecting tracks", unit="frame")
		for frame_id, frame in reader:
			results = detector.track_frame(frame, persist=True)
			tracks = parse_ultralytics_results(results, frame_id)
			raw_frames.append(frame)
			raw_frame_ids.append(frame_id)
			raw_tracks_per_frame.append(tracks)
			collect_bar.update(1)
		collect_bar.close()

		# Stage 2: refine fragmented IDs.
		refined_tracks_per_frame, id_map = refine_track_ids(
			raw_tracks_per_frame,
			max_frame_gap=max(1, int(refine_max_gap)),
			max_center_dist=float(refine_max_center_dist),
			max_size_ratio=float(refine_max_size_ratio),
		)
		merged_count = sum(1 for k, v in id_map.items() if int(k) != int(v))

		# Stage 3: render with refined tracks and feature panel.
		render_bar = tqdm(total=len(raw_frames), desc="Rendering video", unit="frame")
		written = 0
		for idx in range(len(raw_frames)):
			frame = raw_frames[idx]
			frame_id = int(raw_frame_ids[idx])
			tracks = refined_tracks_per_frame[idx]
			window_tracks.append(tracks)
			track_feats = build_track_features(
				list(window_tracks),
				fit_degree=max(1, int(track_fit_degree)),
				history_len=max(2, int(track_history_len)),
			)
			for k, v in track_feats.items():
				feat_histories[k].append(float(v))

			vis = frame.copy()
			for t in tracks:
				tid = int(t.track_id)
				x1, y1, x2, y2 = map(int, t.bbox_xyxy)
				cx, cy = int(t.cx), int(t.cy)
				color = _color_from_track_id(tid)
				history[tid].append((cx, cy))

				cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
				label = f"id={tid} {t.cls_name} {t.score:.2f}"
				cv2.putText(
					vis,
					label,
					(x1, max(0, y1 - 6)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.45,
					color,
					1,
					cv2.LINE_AA,
				)

			for tid, pts in history.items():
				if len(pts) < 2:
					continue
				color = _color_from_track_id(tid)
				p_prev = pts[0]
				for p_cur in list(pts)[1:]:
					cv2.line(vis, p_prev, p_cur, color, 2)
					p_prev = p_cur

			cv2.putText(
				vis,
				f"frame={frame_id} active_tracks={len(tracks)} fit_deg={track_fit_degree} hist={track_history_len} refined={merged_count}",
				(10, 24),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(20, 220, 20),
				2,
				cv2.LINE_AA,
			)

			panel = _draw_feature_panel(
				frame_h=vis.shape[0],
				panel_w=max(300, int(feature_panel_width)),
				track_feats=track_feats,
				feat_histories=feat_histories,
				history_points=max(8, int(feature_history_points)),
			)
			canvas = np.concatenate([vis, panel], axis=1)

			if writer is None:
				h, w = canvas.shape[:2]
				writer = _open_writer(output_video, fps=fps, frame_w=w, frame_h=h, codec=codec)

			writer.write(canvas)
			written += 1
			render_bar.update(1)

		render_bar.close()
	finally:
		reader.release()
		if writer is not None:
			writer.release()

	return output_video


def _default_output_path(input_source: str) -> str:
	src_name = Path(input_source).stem or "track_vis"
	return str(Path("outputs") / "debug" / f"{src_name}_track_vis.mp4")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize tracking result and save as video")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--frame-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
	parser.add_argument("--output-root", default="outputs/test/track_vis")
	parser.add_argument("--video-id", default="")
	parser.add_argument("--trail-len", type=int, default=100, help="Number of previous positions to show as trail")
	parser.add_argument("--window-size", type=int, default=16)
	parser.add_argument("--track-fit-degree", type=int, default=-1, help="<=0 means use config/default")
	parser.add_argument("--track-history-len", type=int, default=0, help="<=0 means use window-size")
	parser.add_argument("--refine-max-gap", type=int, default=None)
	parser.add_argument("--refine-max-center-dist", type=float, default=None)
	parser.add_argument("--refine-max-size-ratio", type=float, default=None)
	parser.add_argument("--feature-panel-width", type=int, default=420)
	parser.add_argument("--feature-history-points", type=int, default=120)
	parser.add_argument("--output-fps", type=float, default=25.0)
	parser.add_argument("--codec", default="mp4v")
	parser.add_argument("--resize-max-side", type=int, default=640)
	parser.add_argument("--resize-interpolation", default="area")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = load_settings(args.config)

	input_source = f"{args.frame_root}/{args.video_id}"
	if not input_source:
		raise ValueError("input source is empty, please provide --input-video or set input_video in config")

	output_video = f"{args.output_root}/{args.video_id}_track_vis.mp4"

	resize_max_side = int(
		args.resize_max_side
		if args.resize_max_side >= 0
		else int(getattr(cfg.video, "resize_max_side", 0) or 0)
	)
	resize_interpolation = args.resize_interpolation or str(
		getattr(cfg.video, "resize_interpolation", "area")
	)

	classes = list(cfg.perception.classes) if getattr(cfg.perception, "classes", None) is not None else None
	track_cfg = getattr(getattr(cfg, "features", None), "track", None)
	track_fit_degree = int(args.track_fit_degree) if int(args.track_fit_degree) > 0 else int(getattr(track_cfg, "fit_degree", 2)) if track_cfg is not None else 2
	track_history_len = int(args.track_history_len) if int(args.track_history_len) > 0 else int(args.window_size)
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

	saved_path = visualize_tracking_to_video(
		input_source=input_source,
		output_video=output_video,
		model_path=str(cfg.perception.yolo_model),
		tracker_cfg=str(cfg.perception.tracker),
		conf=float(cfg.perception.conf),
		iou=float(cfg.perception.iou),
		classes=classes,
		input_fps=float(getattr(cfg.video, "input_fps", args.output_fps)),
		output_fps=float(args.output_fps),
		trail_len=max(1, int(args.trail_len)),
		window_size=max(2, int(args.window_size)),
		track_fit_degree=max(1, track_fit_degree),
		track_history_len=max(2, track_history_len),
		refine_max_gap=max(1, int(refine_max_gap)),
		refine_max_center_dist=float(refine_max_center_dist),
		refine_max_size_ratio=float(refine_max_size_ratio),
		feature_panel_width=max(300, int(args.feature_panel_width)),
		feature_history_points=max(8, int(args.feature_history_points)),
		resize_max_side=resize_max_side,
		resize_interpolation=resize_interpolation,
		codec=str(args.codec)
	)
	print(f"saved visualization video: {saved_path}")


if __name__ == "__main__":
	main()
