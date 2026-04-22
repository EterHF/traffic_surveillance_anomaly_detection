from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.utils import list_frame_paths, parse_frame_id
from src.features.feature_components.scene import SceneFeatureConfig
from src.features.feature_components.scene import SceneFeatureExtractor
from src.features.feature_components.scene import align_pair_signal
from src.features.feature_builder import SCENE_FEATURE_WEIGHTS
from src.features.feature_components.track import build_track_features
from src.perception.detector_tracker import DetectorTracker
from src.perception.detector_tracker import DetectorTrackerConfig
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids


@dataclass
class RawWindowRecord:
    window_id: int
    start_frame: int
    end_frame: int
    feature_dict: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": int(self.window_id),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "feature_dict": {str(k): float(v) for k, v in self.feature_dict.items()},
        }


class RawScoreOrchestrator:
    """Offline raw-feature extractor for score-composition experiments."""

    def __init__(self, cfg_path: str, *, precompute_clip: bool = False, precompute_raft: bool = False):
        with Path(cfg_path).open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.detector_tracker = DetectorTracker(
            DetectorTrackerConfig(**dict(self.cfg.get("perception", {}))),
        )
        self.scene_extractor = SceneFeatureExtractor(
            SceneFeatureConfig(**dict(self.cfg.get("scene_features", {}))),
        )

        video_cfg = dict(self.cfg.get("video", {}))
        track_cfg = dict(self.cfg.get("track_features", {}))
        refine_cfg = dict(self.cfg.get("track_refiner", {}))

        self.window_size = max(1, int(video_cfg.get("window_size", 16)))
        self.window_step = max(1, int(video_cfg.get("window_step", 1)))
        self.track_fit_degree = int(track_cfg.get("fit_degree", 2))
        self.track_history_len = track_cfg.get("history_len", None)
        self.precompute_clip = bool(precompute_clip)
        self.precompute_raft = bool(precompute_raft)

        self.enable_track_refine = bool(refine_cfg.get("enable", False))
        self.track_refine_params = {
            "max_frame_gap": int(refine_cfg.get("max_frame_gap", 8)),
            "max_center_dist": float(refine_cfg.get("max_center_dist", 0.06)),
            "max_size_ratio": float(refine_cfg.get("max_size_ratio", 2.5)),
            "min_direction_cos": float(refine_cfg.get("min_direction_cos", -0.1)),
            "max_speed_ratio": float(refine_cfg.get("max_speed_ratio", 3.5)),
            "gap_relax_factor": float(refine_cfg.get("gap_relax_factor", 0.20)),
        }

        fps_in = float(video_cfg.get("input_fps", 25))
        fps_sample = float(video_cfg.get("fps_sample", fps_in))
        self.frame_stride = max(1, int(round(max(1.0, fps_in) / max(1.0, fps_sample))))

    @staticmethod
    def _frame_density(frame_tracks: list[Any]) -> float:
        if not frame_tracks:
            return 0.0
        frame_w = max((float(track.frame_w) for track in frame_tracks), default=0.0)
        frame_h = max((float(track.frame_h) for track in frame_tracks), default=0.0)
        if frame_w <= 0.0 or frame_h <= 0.0:
            frame_w = max((float(track.bbox_xyxy[2]) for track in frame_tracks), default=1.0)
            frame_h = max((float(track.bbox_xyxy[3]) for track in frame_tracks), default=1.0)
        frame_area = max(1.0, frame_w * frame_h)
        return float(sum(float(track.area) for track in frame_tracks) / frame_area)

    @staticmethod
    def _top_mean(values: np.ndarray, ratio: float = 0.25) -> float:
        if values.size == 0:
            return 0.0
        keep = max(1, int(np.ceil(values.size * float(ratio))))
        return float(np.mean(np.sort(values)[-keep:]))

    def _window_pair_summary(self, pair_values: np.ndarray, window_len: int) -> float:
        signal = align_pair_signal(pair_values.astype(np.float32), window_len)
        return self._top_mean(self._robust_scale(signal), ratio=0.25)

    @staticmethod
    def _robust_scale(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=True)
        if values.size == 0:
            return values
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = max(1e-6, 1.4826 * mad)
        z = np.maximum((values - median) / sigma, 0.0)
        hi = max(float(np.percentile(z, 95)), 1e-6)
        return np.clip(z / hi, 0.0, 1.0).astype(np.float32)

    def extract_video(
        self,
        frames_dir: str,
        output_dir: str,
        video_id: str | None = None,
    ) -> dict[str, Any]:
        frames_path = Path(frames_dir)
        video_name = video_id or frames_path.name
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = list_frame_paths(str(frames_path))
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_dir}")

        sampled_frame_paths = frame_paths[:: self.frame_stride]
        sampled_frame_ids = [parse_frame_id(path) for path in sampled_frame_paths]

        sampled_tracks_raw: list[list[Any]] = []
        sampled_images: list[np.ndarray] = []
        for frame_path in tqdm(sampled_frame_paths, desc=f"{video_name}: track", dynamic_ncols=True):
            frame = cv2.imread(frame_path)
            if frame is None:
                raise FileNotFoundError(f"Failed to read frame: {frame_path}")
            results = self.detector_tracker.track_frame(frame, persist=True)
            sampled_tracks_raw.append(parse_ultralytics_results(results, parse_frame_id(frame_path)))
            sampled_images.append(frame)

        sampled_tracks = sampled_tracks_raw
        refine_map: dict[int, int] = {}
        if self.enable_track_refine and sampled_tracks_raw:
            sampled_tracks, refine_map = refine_track_ids(sampled_tracks_raw, **self.track_refine_params)

        count_series = np.asarray([float(len(frame_tracks)) for frame_tracks in sampled_tracks], dtype=np.float32)
        density_series = np.asarray([self._frame_density(frame_tracks) for frame_tracks in sampled_tracks], dtype=np.float32)
        pair_signals = self.scene_extractor.compute_pair_signals(
            window_tracks=sampled_tracks,
            images_bgr=sampled_images,
            lowres_weight=SCENE_FEATURE_WEIGHTS.get("lowres_eventness", 0.0),
            layout_weight=SCENE_FEATURE_WEIGHTS.get("track_layout_eventness", 0.0),
            clip_weight=1.0 if self.precompute_clip else 0.0,
            raft_weight=1.0 if self.precompute_raft else 0.0,
        )
        lowres_pair = pair_signals["lowres_pair"]
        layout_pair = pair_signals["layout_pair"]
        clip_pair = pair_signals["clip_pair"]
        raft_pair = pair_signals["raft_pair"]

        frame_window: deque[int] = deque(maxlen=self.window_size)
        track_window: deque[list[Any]] = deque(maxlen=self.window_size)

        window_rows: list[RawWindowRecord] = []
        for index, (frame_id, frame_path, tracks) in enumerate(
            zip(sampled_frame_ids, sampled_frame_paths, sampled_tracks)
        ):
            frame_window.append(int(frame_id))
            track_window.append(tracks)

            if len(frame_window) < self.window_size:
                continue
            window_start = index + 1 - self.window_size
            if window_start % self.window_step != 0:
                continue

            track_feats = build_track_features(
                list(track_window),
                fit_degree=self.track_fit_degree,
                history_len=self.track_history_len,
            )
            window_end = index
            count_seg = count_series[window_start : window_end + 1]
            density_seg = density_series[window_start : window_end + 1]
            lowres_seg = lowres_pair[window_start:window_end]
            layout_seg = layout_pair[window_start:window_end]
            clip_seg = clip_pair[window_start:window_end]
            raft_seg = raft_pair[window_start:window_end]
            scene_feats = {
                "cur_count": float(count_seg[-1]) if count_seg.size else 0.0,
                "count_mean": float(np.mean(count_seg)) if count_seg.size else 0.0,
                "count_std": float(np.std(count_seg)) if count_seg.size else 0.0,
                "count_delta": float(count_seg[-1] - count_seg[0]) if count_seg.size >= 2 else 0.0,
                "cur_density": float(density_seg[-1]) if density_seg.size else 0.0,
                "density_mean": float(np.mean(density_seg)) if density_seg.size else 0.0,
                "density_std": float(np.std(density_seg)) if density_seg.size else 0.0,
                "density_delta": float(density_seg[-1] - density_seg[0]) if density_seg.size >= 2 else 0.0,
                "lowres_eventness": self._window_pair_summary(lowres_seg, len(frame_window)) if lowres_seg.size else 0.0,
                "track_layout_eventness": self._window_pair_summary(layout_seg, len(frame_window)) if layout_seg.size else 0.0,
                "clip_eventness": self._window_pair_summary(clip_seg, len(frame_window)) if clip_seg.size else 0.0,
                "raft_eventness": self._window_pair_summary(raft_seg, len(frame_window)) if raft_seg.size else 0.0,
            }
            window_rows.append(
                RawWindowRecord(
                    window_id=len(window_rows),
                    start_frame=int(frame_window[0]),
                    end_frame=int(frame_window[-1]),
                    feature_dict={**track_feats, **scene_feats},
                )
            )

        payload = {
            "video_id": str(video_name),
            "frames_dir": str(frames_path),
            "frame_stride": int(self.frame_stride),
            "window_size": int(self.window_size),
            "window_step": int(self.window_step),
            "track_fit_degree": int(self.track_fit_degree),
            "track_history_len": (
                None if self.track_history_len is None else int(self.track_history_len)
            ),
            "sampled_frame_ids": [int(v) for v in sampled_frame_ids],
            "sampled_frame_paths": [str(v) for v in sampled_frame_paths],
            "track_refine_enabled": bool(self.enable_track_refine),
            "track_refine_map": {str(k): int(v) for k, v in refine_map.items()},
            "window_rows": [row.to_dict() for row in window_rows],
        }

        with (out_dir / "feature_rows.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload


__all__ = ["RawScoreOrchestrator", "RawWindowRecord"]
