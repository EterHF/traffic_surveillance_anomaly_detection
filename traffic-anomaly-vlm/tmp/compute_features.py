from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.features.clip_features import CLIPFeatureConfig, CLIPFeatureExtractor
from src.features.raft_features import RAFTFeatureConfig, RAFTFeatureExtractor
from src.features.scene_features import build_scene_features
from src.features.track_features import build_track_features
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
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


def robust_zscore(x: np.ndarray, win: int = 31) -> np.ndarray:
	z = np.zeros_like(x, dtype=np.float32)
	for i in range(x.size):
		l = max(0, i - win + 1)
		seg = x[l : i + 1]
		seg = seg[np.isfinite(seg)]
		if seg.size == 0:
			z[i] = 0.0
			continue
		med = np.median(seg)
		mad = np.median(np.abs(seg - med))
		scale = 1.4826 * mad + 1e-6
		z[i] = float((x[i] - med) / scale)
	return z


def normalize_01(x: np.ndarray) -> np.ndarray:
	m = np.isfinite(x)
	if not np.any(m):
		return np.zeros_like(x, dtype=np.float32)
	v = x.copy().astype(np.float32)
	# Use robust percentile range to avoid a few outliers flattening the whole curve.
	lo = float(np.percentile(v[m], 5.0))
	hi = float(np.percentile(v[m], 95.0))
	if hi - lo < 1e-6:
		out = np.zeros_like(v, dtype=np.float32)
		out[m] = 0.0
		return out
	v = np.clip(v, lo, hi)
	out = np.zeros_like(v, dtype=np.float32)
	out[m] = (v[m] - lo) / (hi - lo)
	return out


def temporal_change_ratio(x: np.ndarray, ema_window: int = 31) -> np.ndarray:
	"""Frame-wise relative change against causal EMA baseline."""
	v = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if v.size == 0:
		return v
	win = max(3, int(ema_window))
	alpha = 2.0 / float(win + 1)
	ema = np.zeros_like(v, dtype=np.float32)
	ema[0] = v[0]
	for i in range(1, v.size):
		ema[i] = alpha * v[i] + (1.0 - alpha) * ema[i - 1]
	ratio = (v - ema) / (np.abs(ema) + 1e-6)
	# Suppress warm-up instability at the beginning.
	warm = max(2, win // 3)
	ratio[:warm] = 0.0
	return ratio.astype(np.float32)


def baseline_deviation_score(x: np.ndarray, baseline_len: int) -> np.ndarray:
	"""Two-sided robust deviation from an early-video baseline (median/MAD)."""
	v = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if v.size == 0:
		return v
	b = max(5, min(int(baseline_len), v.size))
	base = v[:b]
	med = float(np.median(base))
	mad = float(np.median(np.abs(base - med)))
	dev = np.abs(v - med) / (1.4826 * mad + 1e-6)
	return normalize_01(dev)


def early_late_drift(x: np.ndarray, frac: float = 0.2) -> float:
	v = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if v.size == 0:
		return 0.0
	h = max(5, int(round(v.size * frac)))
	h = min(h, v.size)
	return float(np.mean(v[:h]) - np.mean(v[-h:]))


def peak_center_confidence(x: np.ndarray, q: float = 90.0) -> float:
	v = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if v.size <= 1:
		return 0.5
	th = float(np.percentile(v, q))
	idx = np.where(v >= th)[0]
	if idx.size == 0:
		return 0.5
	pos = float(np.mean(idx) / float(v.size - 1))
	# 1.0 at center, 0.0 near edges.
	return max(0.0, 1.0 - 2.0 * abs(pos - 0.5))


def reliability_from_drift(score_sig: np.ndarray) -> float:
	"""Reliability prior from temporal behavior of a component score."""
	score_drift = max(0.0, early_late_drift(score_sig, frac=0.2))
	center_conf = peak_center_confidence(score_sig, q=90.0)
	center_conf = center_conf * center_conf
	# Keep a floor so no component is fully disabled.
	return float(0.10 + 0.45 * score_drift + 0.45 * center_conf)


def intervals_to_binary(frame_ids: np.ndarray, intervals: list[list[int]]) -> np.ndarray:
	y = np.zeros(frame_ids.shape[0], dtype=np.int32)
	for i, fid in enumerate(frame_ids.tolist()):
		for s, e in intervals:
			if s <= fid <= e:
				y[i] = 1
				break
	return y

# Align pairwise features (which have length n-1) to frame ids (length n) by padding with NaN at the start.
def align_pair(arr: np.ndarray, n: int) -> np.ndarray:
	out = np.full(n, np.nan, dtype=np.float32)
	if arr.size == 0:
		return out
	m = min(n - 1, arr.size)
	out[1 : 1 + m] = arr[:m]
	return out


def plot_signal(ax, frame_ids: np.ndarray, y: np.ndarray, label: str, gt_intervals: list[list[int]], color: str) -> None:
	ax.plot(frame_ids, y, label=label, linewidth=1.2, color=color)
	for s, e in gt_intervals:
		ax.axvspan(s, e, color="red", alpha=0.12)


def flow_to_color(flow_hw2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Convert flow [H, W, 2] to BGR color map and magnitude map."""
	fx = flow_hw2[..., 0]
	fy = flow_hw2[..., 1]
	mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

	hsv = np.zeros((flow_hw2.shape[0], flow_hw2.shape[1], 3), dtype=np.uint8)
	hsv[..., 0] = ((ang / 2.0) % 180).astype(np.uint8)
	hsv[..., 1] = 255
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return bgr, mag.astype(np.float32)


def run(args: argparse.Namespace) -> None:
	out_dir = Path(args.out_dir) / args.video_id
	out_dir.mkdir(parents=True, exist_ok=True)

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

	cfg = load_settings(args.config)
	logging.info("Config loaded successfully")

	# get frames_dirs using root + video_id
	frames_dir = Path(args.frames_root) / args.video_id
	logging.info(f"Resolving frame paths from {frames_dir}...")
	frame_paths = list_frame_paths(frames_dir)
	if not frame_paths:
		raise ValueError(f"No frames found in {frames_dir}")
	frame_paths = frame_paths[:: max(1, int(args.frame_stride))]

	images: list[np.ndarray] = []
	frame_ids_list: list[int] = []
	for p in frame_paths:
		img = cv2.imread(p)
		if img is None:
			continue
		images.append(img)
		frame_ids_list.append(parse_frame_id(p))
	if not images:
		raise ValueError("No readable images after loading")

	frame_ids = np.array(frame_ids_list, dtype=np.int32)
	n = frame_ids.shape[0]

	gt_intervals = parse_manifest_intervals(args.manifest, args.video_id)
	y_true = intervals_to_binary(frame_ids, gt_intervals)
	logging.info(f"Loaded {n} frames with ids from {frame_ids[0]} to {frame_ids[-1]}")
	logging.info(f"Parsed GT intervals: {gt_intervals}")

	detector = DetectorTracker(
		model_path=cfg.perception.yolo_model,
		tracker=cfg.perception.tracker,
		conf=cfg.perception.conf,
		iou=cfg.perception.iou,
		classes=list(cfg.perception.classes),
	)

	clip_cfg = getattr(getattr(cfg, "features", None), "clip", None)
	if clip_cfg is None:
		raise ValueError("Missing features.clip in config")
	raft_cfg = getattr(getattr(cfg, "features", None), "raft", None)
	if raft_cfg is None:
		raise ValueError("Missing features.raft in config")

	clip_extractor = CLIPFeatureExtractor(
		CLIPFeatureConfig(
			model_name=str(getattr(clip_cfg, "model_name", "openai/clip-vit-base-patch32")),
			model_revision=(getattr(clip_cfg, "model_revision", None) or None),
			device=str(getattr(clip_cfg, "device", args.device)),
			batch_size=int(getattr(clip_cfg, "batch_size", 24)),
			resize_long_side=int(getattr(clip_cfg, "resize_long_side", 0)),
			resize_interpolation=str(getattr(clip_cfg, "resize_interpolation", "area")),
		)
	)
	logging.info("CLIP feature extractor initialized successfully")
	raft_extractor = RAFTFeatureExtractor(
		RAFTFeatureConfig(
			raft_root=str(getattr(raft_cfg, "root", "src/features/RAFT")),
			weights_path=str(getattr(raft_cfg, "weights", "weights/raft/raft-things.pth")),
			device=str(getattr(raft_cfg, "device", args.device)),
			iters=int(getattr(raft_cfg, "iters", 12)),
			resize_long_side=int(getattr(raft_cfg, "resize_long_side", 640)),
			resize_interpolation=str(getattr(raft_cfg, "resize_interpolation", "area")),
			flow_proj_dim=int(getattr(raft_cfg, "flow_proj_dim", 128)),
			flow_proj_seed=int(getattr(raft_cfg, "flow_proj_seed", 42)),
		)
	)
	logging.info("RAFT feature extractor initialized successfully")

	window_size = max(2, int(args.window_size))
	all_tracks_per_frame = []
	track_feats = []
	scene_feats = []

	logging.info(f"Processing {n} frames with window size {window_size}...")
	for i, img in enumerate(images):
		results = detector.track_frame(img, persist=True)
		tracks = parse_ultralytics_results(results, int(frame_ids[i]))
		all_tracks_per_frame.append(tracks)

		l = max(0, i - window_size + 1)
		win_tracks = all_tracks_per_frame[l : i + 1]
		track_feats.append(build_track_features(win_tracks))
		scene_feats.append(build_scene_features(win_tracks))

	clip_pair = clip_extractor.compute_pair_errors(images)
	raft_pair = raft_extractor.compute_pair_errors(images)

	clip_cos = align_pair(clip_pair["cosine_error"], n)
	clip_l2 = align_pair(clip_pair["l2_error"], n)
	clip_evt = align_pair(clip_pair.get("eventvad_combined_error", clip_pair["l2_error"]), n)
	raft_mag = align_pair(raft_pair["flow_mag_mean"], n)
	raft_evt = align_pair(raft_pair.get("eventvad_combined_error", raft_pair["flow_mag_mean"]), n)

	track_count = np.array([f["track_count"] for f in track_feats], dtype=np.float32)
	speed_mean = np.array([f["speed_mean"] for f in track_feats], dtype=np.float32)
	acc_mean = np.array([abs(f["acc_mean"]) for f in track_feats], dtype=np.float32)
	acc_abs_mean = np.array([f.get("acc_abs_mean", 0.0) for f in track_feats], dtype=np.float32)
	pred_center_err = np.array([f.get("pred_center_err_mean", 0.0) for f in track_feats], dtype=np.float32)
	pred_iou_err = np.array([f.get("pred_iou_err_mean", 0.0) for f in track_feats], dtype=np.float32)
	turn_rate = np.array([f.get("turn_rate_mean", 0.0) for f in track_feats], dtype=np.float32)
	new_track_ratio = np.array([f.get("new_track_ratio", 0.0) for f in track_feats], dtype=np.float32)
	lost_track_ratio = np.array([f.get("lost_track_ratio", 0.0) for f in track_feats], dtype=np.float32)
	density = np.array([f["density_proxy"] for f in scene_feats], dtype=np.float32)
	obj_count_std = np.array([f.get("object_count_std", 0.0) for f in scene_feats], dtype=np.float32)
	density_delta = np.array([abs(f.get("density_delta", 0.0)) for f in scene_feats], dtype=np.float32)
	interaction_proxy = np.array([f.get("interaction_inv_nn_proxy", 0.0) for f in scene_feats], dtype=np.float32)

	# Track score: trajectory prediction and motion-instability cues.
	track_core = (
		0.35 * pred_center_err
		+ 0.25 * pred_iou_err
		+ 0.15 * acc_abs_mean
		+ 0.10 * turn_rate
		+ 0.10 * new_track_ratio
		+ 0.05 * lost_track_ratio
	)

	baseline_len = max(20, int(0.20 * n))
	track_score = baseline_deviation_score(track_core, baseline_len)

	# Scene score: crowd density + directional complexity (simple and interpretable).
	density_score = normalize_01(density)
	turn_score = normalize_01(turn_rate)
	scene_core = 0.60 * density_score + 0.25 * turn_score + 0.15 * normalize_01(interaction_proxy)
	scene_score = baseline_deviation_score(scene_core, baseline_len)

	clip_score = baseline_deviation_score(np.nan_to_num(clip_evt, nan=0.0), baseline_len)
	raft_score = baseline_deviation_score(np.nan_to_num(raft_evt, nan=0.0), baseline_len)

	# Reliability-aware interpretable fusion.
	track_available = float(np.mean(track_count > 0.0))
	if track_available < 0.20:
		w_track, w_scene, w_clip, w_raft = 0.10, 0.55, 0.20, 0.15
	elif track_available < 0.50:
		w_track, w_scene, w_clip, w_raft = 0.30, 0.45, 0.15, 0.10
	else:
		w_track, w_scene, w_clip, w_raft = 0.45, 0.35, 0.10, 0.10

	rel_track = reliability_from_drift(track_score)
	rel_scene = reliability_from_drift(scene_score)
	rel_clip = reliability_from_drift(clip_score)
	rel_raft = reliability_from_drift(raft_score)

	w_track *= rel_track
	w_scene *= rel_scene
	w_clip *= rel_clip
	w_raft *= rel_raft
	w_sum = w_track + w_scene + w_clip + w_raft
	if w_sum <= 1e-6:
		w_track, w_scene, w_clip, w_raft = 0.40, 0.35, 0.15, 0.10
	else:
		w_track /= w_sum
		w_scene /= w_sum
		w_clip /= w_sum
		w_raft /= w_sum

	fused_score = w_track * track_score + w_scene * scene_score + w_clip * clip_score + w_raft * raft_score

	csv_path = out_dir / "timeseries.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"frame_id",
				"gt",
				"track_count",
				"speed_mean",
				"acc_mean_abs",
				"pred_center_err_mean",
				"pred_iou_err_mean",
				"turn_rate_mean",
				"new_track_ratio",
				"lost_track_ratio",
				"scene_density",
				"scene_object_count_std",
				"scene_density_delta_abs",
				"scene_interaction_inv_nn",
				"clip_cosine_error",
				"clip_l2_error",
				"clip_eventvad_combined_error",
				"raft_flow_mag_mean",
				"raft_eventvad_combined_error",
				"score_track",
				"score_scene",
				"score_clip",
				"score_raft",
				"score_fused",
			]
		)
		for i in range(n):
			writer.writerow(
				[
					int(frame_ids[i]),
					int(y_true[i]),
					float(track_count[i]),
					float(speed_mean[i]),
					float(acc_mean[i]),
					float(pred_center_err[i]),
					float(pred_iou_err[i]),
					float(turn_rate[i]),
					float(new_track_ratio[i]),
					float(lost_track_ratio[i]),
					float(density[i]),
					float(obj_count_std[i]),
					float(density_delta[i]),
					float(interaction_proxy[i]),
					float(clip_cos[i]) if np.isfinite(clip_cos[i]) else "",
					float(clip_l2[i]) if np.isfinite(clip_l2[i]) else "",
					float(clip_evt[i]) if np.isfinite(clip_evt[i]) else "",
					float(raft_mag[i]) if np.isfinite(raft_mag[i]) else "",
					float(raft_evt[i]) if np.isfinite(raft_evt[i]) else "",
					float(track_score[i]),
					float(scene_score[i]),
					float(clip_score[i]),
					float(raft_score[i]),
					float(fused_score[i]),
				]
			)

	fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
	plot_signal(axs[0], frame_ids, track_score, "track_score", gt_intervals, "tab:blue")
	plot_signal(axs[0], frame_ids, scene_score, "scene_score", gt_intervals, "tab:orange")
	plot_signal(axs[0], frame_ids, clip_score, "clip_score", gt_intervals, "tab:purple")
	plot_signal(axs[0], frame_ids, raft_score, "raft_score", gt_intervals, "tab:green")
	axs[0].set_ylabel("component score")
	axs[0].legend(loc="upper left")

	plot_signal(axs[1], frame_ids, fused_score, "fused_score", gt_intervals, "tab:red")
	axs[1].set_ylabel("fused")
	axs[1].legend(loc="upper left")

	fig.tight_layout()
	fig.savefig(out_dir / "scores_plot.png", dpi=160)
	plt.close(fig)

	# Visualize one raft flow map as sanity check (also save original images for reference).
	flow_vis_path = ""
	flow_mag_path = ""
	frame_a_path = ""
	frame_b_path = ""
	if n >= 2:
		pair_idx = gt_intervals[0][0] - frame_ids[0] if gt_intervals else 0
		pair_idx = max(0, min(pair_idx, n - 2))
		flow_pair = raft_extractor.extract_features(images[pair_idx : pair_idx + 2], channel_last=True)
		if flow_pair.shape[0] > 0:
			flow_hw2 = flow_pair[0]
			flow_bgr, flow_mag = flow_to_color(flow_hw2)

			frame_a = images[pair_idx]
			frame_b = images[pair_idx + 1]

			frame_a_path = str(out_dir / f"flow_pair_{pair_idx:04d}_frame_a.jpg")
			frame_b_path = str(out_dir / f"flow_pair_{pair_idx:04d}_frame_b.jpg")
			flow_vis_path = str(out_dir / f"flow_pair_{pair_idx:04d}_color.jpg")
			flow_mag_path = str(out_dir / f"flow_pair_{pair_idx:04d}_magnitude.png")

			cv2.imwrite(frame_a_path, frame_a)
			cv2.imwrite(frame_b_path, frame_b)
			cv2.imwrite(flow_vis_path, flow_bgr)

			fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
			im = ax2.imshow(flow_mag, cmap="magma")
			ax2.set_title(f"RAFT flow magnitude (pair index {pair_idx})")
			ax2.set_axis_off()
			plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
			fig2.tight_layout()
			fig2.savefig(flow_mag_path, dpi=160)
			plt.close(fig2)
	

	summary = {
		"video_id": args.video_id,
		"num_frames": int(n),
		"frame_stride": int(args.frame_stride),
		"window_size": int(window_size),
		"gt_intervals": gt_intervals,
		"paths": {
			"timeseries_csv": str(csv_path),
			"plot": str(out_dir / "scores_plot.png"),
			"flow_color": flow_vis_path,
			"flow_magnitude": flow_mag_path,
			"flow_frame_a": frame_a_path,
			"flow_frame_b": frame_b_path,
		},
		"score_means": {
			"track": float(np.mean(track_score)),
			"scene": float(np.mean(scene_score)),
			"clip": float(np.mean(clip_score)),
			"raft": float(np.mean(raft_score)),
			"fused": float(np.mean(fused_score)),
		},
		"fusion_info": {
			"track_available_ratio": track_available,
			"weights": {
				"track": w_track,
				"scene": w_scene,
				"clip": w_clip,
				"raft": w_raft,
			},
		},
		"score_contrast": {
			"track_anomaly_over_normal": float(
				(np.mean(track_score[y_true == 1]) / (np.mean(track_score[y_true == 0]) + 1e-6))
				if np.any(y_true == 1) and np.any(y_true == 0)
				else 0.0
			),
			"scene_anomaly_over_normal": float(
				(np.mean(scene_score[y_true == 1]) / (np.mean(scene_score[y_true == 0]) + 1e-6))
				if np.any(y_true == 1) and np.any(y_true == 0)
				else 0.0
			),
			"clip_anomaly_over_normal": float(
				(np.mean(clip_score[y_true == 1]) / (np.mean(clip_score[y_true == 0]) + 1e-6))
				if np.any(y_true == 1) and np.any(y_true == 0)
				else 0.0
			),
			"raft_anomaly_over_normal": float(
				(np.mean(raft_score[y_true == 1]) / (np.mean(raft_score[y_true == 0]) + 1e-6))
				if np.any(y_true == 1) and np.any(y_true == 0)
				else 0.0
			),
			"fused_anomaly_over_normal": float(
				(np.mean(fused_score[y_true == 1]) / (np.mean(fused_score[y_true == 0]) + 1e-6))
				if np.any(y_true == 1) and np.any(y_true == 0)
				else 0.0
			),
		},
		"drift": {
			"track_score_early_minus_late": early_late_drift(track_score),
			"scene_score_early_minus_late": early_late_drift(scene_score),
			"clip_score_early_minus_late": early_late_drift(clip_score),
			"raft_score_early_minus_late": early_late_drift(raft_score),
		},
	}
	with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	logging.info("Done. Outputs at: %s", out_dir)


def build_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Feature exploration scaffold for track/scene/CLIP/RAFT + visualization")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal", help="Path to root directory containing video frame directories")
	parser.add_argument("--video-id", default="v1", help="Video id used to query manifest GT intervals")
	parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
	parser.add_argument("--frame-stride", type=int, default=1)
	parser.add_argument("--window-size", type=int, default=16)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--out-dir", default="outputs/debug/features_scaffold")
	return parser.parse_args()


if __name__ == "__main__":
	run(build_args())