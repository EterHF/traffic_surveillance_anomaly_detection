from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from tqdm.auto import tqdm

from src.schemas import EventNode, TrackObject, WindowFeature
from src.settings import load_settings, instantiate_from_config
from src.core.utils import get_logger
from src.core.video_io import build_reader
from src.core.sampler import FrameSampler
from src.perception.track_parser import parse_ultralytics_results

from src.perception.detector_tracker import DetectorTracker
from src.perception.track_refiner import refine_track_ids
from src.features.feature_builder import FeatureBuilder, SCENE_FEATURE_WEIGHTS
from src.vlm.model_loader import LocalVLM
from src.proposals.boundary_detector import BoundaryDetector
from src.proposals.tree_pipeline import TreePipeline

from src.eval.visualize import plot_scores


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        value = value.strip().rstrip(",")
    return float(value)


class Orchestrator:
    def __init__(self, cfg_path: str):
        # Load settings
        self.cfg = load_settings(cfg_path)
        self.cfg_instances = instantiate_from_config(self.cfg)
        self.logger = get_logger()

        # Output dir
        self.output_dir = Path(self.cfg.output.output_dir)
        self.assets_dir = self.output_dir / self.cfg.evidence_builder.assets_subdir
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        detector_cfg = self.cfg_instances.get("perception") or self.cfg_instances.get("detector_tracker")
        self.detector_tracker = DetectorTracker(detector_cfg)
        self.feature_builder = FeatureBuilder(
            self.cfg.video.window_size,
            self.cfg.video.window_step,
            self.cfg.track_features.fit_degree,
            self.cfg.track_features.history_len,
            self.cfg_instances.get("scene_features"),
        )
        self.enable_track_refine = bool(getattr(self.cfg.track_refiner, "enable", False))
        self.track_refiner_params: dict[str, Any] = {}
        if self.enable_track_refine:
            self.track_refiner_params = {
                "max_frame_gap": self.cfg.track_refiner.max_frame_gap,
                "max_center_dist": self.cfg.track_refiner.max_center_dist,
                "max_size_ratio": self.cfg.track_refiner.max_size_ratio,
                "min_direction_cos": self.cfg.track_refiner.min_direction_cos,
                "max_speed_ratio": self.cfg.track_refiner.max_speed_ratio,
                "gap_relax_factor": self.cfg.track_refiner.gap_relax_factor,
            }

        self.boundary_detector_cfg = self.cfg_instances.get("boundary_detector")
        self.tree_build_cfg = self.cfg_instances.get("tree_build_config") or self.cfg_instances.get("tree_build")
        self.vlm = LocalVLM(self.cfg_instances.get("vlm"))
        self.tree_method = str(getattr(self.cfg.evidence_builder, "method", "montage"))
        self.tree_pipeline = TreePipeline(
            self.vlm,
            self.tree_build_cfg,
            self.boundary_detector_cfg,
            _safe_float(getattr(self.cfg.tree_refine_config, "min_confidence", 0.35), 0.35),
            _safe_float(getattr(self.cfg.tree_refine_config, "prior_weight", 0.4), 0.4),
            _safe_float(getattr(self.cfg.tree_refine_config, "vlm_weight", 0.6), 0.6),
            str(self.assets_dir),
        )
        self.logger.info("Orchestrator initialized with config: %s", cfg_path)

    def _build_window_features(
        self,
        sampled_frame_ids: list[int],
        sampled_frame_paths: list[str],
        sampled_tracks: list[list[TrackObject]],
    ) -> tuple[list[Any], list[float]]:
        window_size = max(1, int(self.cfg.video.window_size))
        window_step = max(1, int(self.cfg.video.window_step))

        frame_window: deque[int] = deque(maxlen=window_size)
        track_window: deque[list[TrackObject]] = deque(maxlen=window_size)
        image_window: deque[Any] = deque(maxlen=window_size)
        scene_pair_cache = self._precompute_scene_pair_cache(sampled_frame_paths, sampled_tracks)
        need_window_images = False

        windows = []
        for idx, (frame_id, frame_path, tracks) in enumerate(
            zip(sampled_frame_ids, sampled_frame_paths, sampled_tracks)
        ):
            frame_window.append(int(frame_id))
            track_window.append(tracks)
            if len(frame_window) < window_size:
                continue

            window_start = idx + 1 - window_size
            if window_start % window_step != 0:
                continue

            wf = self.feature_builder.build(
                frame_ids=list(frame_window),
                track_window=list(track_window),
                image_window=list(image_window) if need_window_images else None,
                scene_pair_slices={
                    "lowres_pair": scene_pair_cache["lowres_pair"][window_start:idx],
                    "layout_pair": scene_pair_cache["layout_pair"][window_start:idx],
                    "clip_pair": scene_pair_cache["clip_pair"][window_start:idx],
                    "raft_pair": scene_pair_cache["raft_pair"][window_start:idx],
                },
            )
            windows.append(wf)

        # FIXME: Assume window_step == 1
        scores = [windows[0].trigger_score] * (window_size - 1) # FIXME: temporarily pad first window score
        scores += [float(window.trigger_score) for window in windows]
        return windows, scores

    def _precompute_scene_pair_cache(
        self,
        sampled_frame_paths: list[str],
        sampled_tracks: list[list[TrackObject]],
    ) -> dict[str, np.ndarray]:
        need_images = bool(
            SCENE_FEATURE_WEIGHTS.get("lowres_eventness", 0.0) > 0.0
            or SCENE_FEATURE_WEIGHTS.get("clip_eventness", 0.0) > 0.0
            or SCENE_FEATURE_WEIGHTS.get("raft_eventness", 0.0) > 0.0
        )
        images_bgr: list[np.ndarray] | None = None
        if need_images and sampled_frame_paths:
            images_bgr = []
            for frame_path in sampled_frame_paths:
                image = cv2.imread(frame_path)
                if image is None:
                    raise FileNotFoundError(f"Failed to read sampled frame from cache: {frame_path}")
                images_bgr.append(image)

        return self.feature_builder.scene_extractor.compute_pair_signals(
            window_tracks=sampled_tracks,
            images_bgr=images_bgr,
            lowres_weight=SCENE_FEATURE_WEIGHTS.get("lowres_eventness", 0.0),
            layout_weight=SCENE_FEATURE_WEIGHTS.get("track_layout_eventness", 0.0),
            clip_weight=SCENE_FEATURE_WEIGHTS.get("clip_eventness", 0.0),
            raft_weight=SCENE_FEATURE_WEIGHTS.get("raft_eventness", 0.0),
        )

    def run_offline(self, input_video: str):
        if not input_video:
            raise ValueError("input_video is empty")
        # Make subdirs for current video
        video_id = Path(input_video).stem
        cur_output_dir = self.results_dir / video_id
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        cur_assets_dir = self.assets_dir / video_id
        cur_assets_dir.mkdir(parents=True, exist_ok=True)

        reader = build_reader(
            input_video,
            input_fps=self.cfg.video.input_fps,
            resize_max_side=self.cfg.video.resize_max_side,
            resize_interpolation=self.cfg.video.resize_interpolation,
        )
        sampler = FrameSampler(reader.fps(), self.cfg.video.fps_sample)
        self.logger.info(f"Video opened: {input_video}, original_fps={self.cfg.video.input_fps}, sample_fps={self.cfg.video.fps_sample}")
        self.logger.info(f"Processing total {reader.frame_count()} frames")

        frame_cache_dir = Path(self.output_dir) / "cache" / video_id / "sampled_frames"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)

        sampled_frame_ids: list[int] = []
        sampled_frame_paths: list[str] = []
        sampled_tracks_raw: list[list[TrackObject]] = []
        sampled_frames: dict[int, Any] = {}

        try:
            self.logger.info("Start sampling frames and tracking...")
            total_frames = int(reader.frame_count())
            total_sampled = None if total_frames <= 0 else (total_frames + sampler.stride - 1) // sampler.stride
            for frame_id, frame in tqdm(
                sampler.sample(iter(reader)),
                total=total_sampled,
                desc=f"{video_id}: sample+track",
                dynamic_ncols=True,
            ):
                sampled_frame_ids.append(frame_id)
                frame_path = frame_cache_dir / f"{frame_id:08d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                sampled_frames[frame_id] = str(frame_path)
                sampled_frame_paths.append(str(frame_path))

                results = self.detector_tracker.track_frame(frame, persist=True)
                tracks = parse_ultralytics_results(results, frame_id)
                sampled_tracks_raw.append(tracks)
        finally:
            reader.release()
            self.logger.info(f"Finished tracking and caching sampled frames. Total sampled frames: {len(sampled_frame_ids)}")

        sampled_tracks = sampled_tracks_raw
        if self.enable_track_refine and sampled_tracks_raw:
            sampled_tracks, track_refine_map = refine_track_ids(
                sampled_tracks_raw,
                **self.track_refiner_params
            )

        # Build window features
        # Keep only one sliding window of decoded images in memory.
        windows, scores = self._build_window_features(
            sampled_frame_ids=sampled_frame_ids,
            sampled_frame_paths=sampled_frame_paths,
            sampled_tracks=sampled_tracks,
        )
        assert len(scores) == len(sampled_frame_ids), f"Length of scores ({len(scores)}) should match length of sampled frames ({len(sampled_frame_ids)})"
        plot_scores(scores, frame_ids=sampled_frame_ids, output_dir=cur_output_dir)
        self.logger.info(f"Score plot saved to {cur_output_dir}")

        coarse_spans = BoundaryDetector(self.boundary_detector_cfg).detect(scores)

        # # Run tree pipeline
        # all_tracks = [track for frame_tracks in sampled_tracks for track in frame_tracks]
        # all_nodes: list[EventNode] = []

        # if not windows:
        #     self.logger.warning(
        #         "No valid windows were built. The video is likely shorter than window_size=%d.",
        #         int(self.cfg.video.window_size),
        #     )
        # elif max(scores, default=0.0) <= 0.0:
        #     self.logger.warning("All dense scores are non-positive; skipping tree pipeline.")
        # else:
        #     all_nodes = self.tree_pipeline.process(
        #         images=sampled_frame_paths,
        #         frame_ids=sampled_frame_ids,
        #         scores=scores,
        #         all_tracks=all_tracks,
        #         windows=windows,
        #         method=self.tree_method,
        #     )
        #     self.logger.info(f"Tree pipeline finished with {len(all_nodes)} event nodes.")


        # return all_nodes
