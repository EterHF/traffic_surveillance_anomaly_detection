from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import cv2
import numpy as np
from tqdm.auto import tqdm

from src.schemas import EventNode, TrackObject, WindowFeature
from src.settings import load_settings, instantiate_from_config
from src.core.utils import get_logger, write_json
from src.core.video_io import build_reader
from src.core.sampler import FrameSampler
from src.perception.track_parser import parse_ultralytics_results

from src.perception.detector_tracker import DetectorTracker
from src.perception.track_refiner import refine_track_ids
from src.features.feature_builder import FeatureBuilder, SCENE_FEATURE_WEIGHTS, build_window_score_curves
from src.vlm.model_loader import LocalVLM
from src.proposals.boundary_detector import BoundaryDetector
from src.proposals.tree_pipeline import TreePipeline

from src.eval.metrics import compute_gt_span_metrics
from src.eval.visualize import plot_all_subscores, plot_peaks, plot_scores
from src.eval.utils import parse_manifest_intervals



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
        score_postprocess = getattr(self.cfg, "score_postprocess", None)
        self.score_smooth_window = max(0, int(getattr(score_postprocess, "smooth_window", 0) or 0))
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
                "min_track_len": getattr(self.cfg.track_refiner, "min_track_len", 3),
                "min_track_span": getattr(self.cfg.track_refiner, "min_track_span", 4),
                "min_box_area": getattr(self.cfg.track_refiner, "min_box_area", 20.0),
                "min_area_ratio": getattr(self.cfg.track_refiner, "min_area_ratio", 0.0005),
            }

        self.boundary_detector_cfg = self.cfg_instances.get("boundary_detector")
        # self.vlm = LocalVLM(self.cfg_instances["vlm"])
        # self.tree_pipeline = TreePipeline(
        #     vlm=self.vlm,
        #     tree_build_cfg=self.cfg_instances["tree_build_config"],
        #     boundary_cfg=self.boundary_detector_cfg,
        #     min_confidence=float(self.cfg.tree_refine_config.min_confidence),
        #     prior_weight=float(self.cfg.tree_refine_config.prior_weight),
        #     vlm_weight=float(self.cfg.tree_refine_config.vlm_weight),
        #     positive_threshold=float(self.cfg.tree_refine_config.positive_threshold),
        #     output_assets_dir=str(self.assets_dir),
        # )
        self.logger.info("Orchestrator initialized with config: %s", cfg_path)

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or values.size == 0:
            return values.astype(np.float32, copy=True)
        pad = max(0, int(window) // 2)
        padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
        kernel = np.ones((int(window),), dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    def _dense_score_curves(self, windows: list[WindowFeature], num_frames: int) -> dict[str, np.ndarray]:
        if num_frames <= 0:
            empty = np.zeros((0,), dtype=np.float32)
            return {
                "dense_trigger_raw": empty,
                "dense_trigger": empty,
            }

        if not windows:
            empty = np.zeros((num_frames,), dtype=np.float32)
            return {
                "dense_trigger_raw": empty,
                "dense_trigger": empty,
            }

        prefix_len = max(0, int(self.cfg.video.window_size) - 1)
        window_curves = build_window_score_curves(windows)
        trigger_values = window_curves["trigger_score"].astype(np.float32).tolist()

        def _to_dense(values: list[float]) -> np.ndarray:
            prefix = [float(values[0])] * prefix_len
            dense = np.asarray(prefix + values, dtype=np.float32)
            if dense.size < num_frames:
                dense = np.pad(dense, (0, num_frames - dense.size), mode="edge")
            return dense[:num_frames].astype(np.float32)

        dense_trigger_raw = _to_dense(trigger_values)
        dense_trigger = self._moving_average(dense_trigger_raw, self.score_smooth_window)

        dense_curves = {
            "dense_trigger_raw": dense_trigger_raw,
            "dense_trigger": dense_trigger.astype(np.float32),
        }
        for key in ("gate_global_evidence", "gate_null_weight", "gate_active_experts"):
            if key in window_curves:
                dense_curves[f"dense_{key}"] = _to_dense(window_curves[key].astype(np.float32).tolist())
        return dense_curves

    @staticmethod
    def _frame_spans_from_indices(
        frame_ids: list[int],
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        frame_spans: list[tuple[int, int]] = []
        for start_idx, end_idx in spans:
            if 0 <= start_idx <= end_idx < len(frame_ids):
                frame_spans.append((int(frame_ids[start_idx]), int(frame_ids[end_idx])))
        return frame_spans

    def _build_window_features(
        self,
        sampled_frame_ids: list[int],
        sampled_frame_paths: list[str],
        sampled_tracks: list[list[TrackObject]],
    ) -> tuple[list[WindowFeature], dict[str, np.ndarray]]:
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

        return windows, self._dense_score_curves(windows, len(sampled_frame_ids))

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
        start_time = time.perf_counter()
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
        sampled_detection_results: list[Any] = []

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
                if self.cfg.video.SAVE_TRACK_VIDEOS:
                    sampled_detection_results.append(results)
                tracks = parse_ultralytics_results(results, frame_id)
                sampled_tracks_raw.append(tracks)
        finally:
            reader.release()
            self.logger.info(f"Finished tracking and caching sampled frames. Total sampled frames: {len(sampled_frame_ids)}")

        sampled_tracks = sampled_tracks_raw
        track_refine_map: dict[int, int] = {}
        if self.enable_track_refine and sampled_tracks_raw:
            sampled_tracks, track_refine_map = refine_track_ids(
                sampled_tracks_raw,
                **self.track_refiner_params
            )

        # DEBUG
        # Save all tracks by id for debugging and visualization
        if self.cfg.video.SAVE_ALL_TRACKS:
            track_frames_payload = {
                "video_id": video_id,
                "frame_ids": [int(frame_id) for frame_id in sampled_frame_ids],
                "tracks_per_frame": [
                    {
                        "frame_id": int(frame_id),
                        "tracks": [track.model_dump() for track in frame_tracks],
                    }
                    for frame_id, frame_tracks in zip(sampled_frame_ids, sampled_tracks)
                ],
            }
            raw_track_frames_payload = {
                "video_id": video_id,
                "frame_ids": [int(frame_id) for frame_id in sampled_frame_ids],
                "tracks_per_frame": [
                    {
                        "frame_id": int(frame_id),
                        "tracks": [track.model_dump() for track in frame_tracks],
                    }
                    for frame_id, frame_tracks in zip(sampled_frame_ids, sampled_tracks_raw)
                ],
            }
            write_json(str(cur_output_dir / "sampled_tracks.json"), track_frames_payload)
            write_json(str(cur_output_dir / "sampled_tracks_raw.json"), raw_track_frames_payload)

            all_tracks_by_id: dict[int, list[TrackObject]] = {}
            for frame_tracks in sampled_tracks:
                for track in frame_tracks:
                    all_tracks_by_id.setdefault(track.track_id, []).append(track)
            tracks_by_id_payload = {str(track_id): [track.model_dump() for track in tracks] for track_id, tracks in all_tracks_by_id.items()}
            write_json(str(cur_output_dir / "tracks_by_id.json"), tracks_by_id_payload)

        # DEBUG: visualize sampled/refined tracks as a VS Code-readable MP4.
        if self.cfg.video.SAVE_TRACK_VIDEOS:
            if sampled_frame_paths and sampled_tracks:
                debug_video = cur_output_dir / f"debug_tracks_{video_id}.mp4"
                debug_video_raw = debug_video.with_name(f"{debug_video.stem}_raw{debug_video.suffix}")
                first_frame = cv2.imread(sampled_frame_paths[0])
                if first_frame is not None:
                    height, width = first_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(debug_video_raw),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(getattr(self.cfg.video, "fps_sample", 12) or 12),
                        (width, height),
                    )
                    if writer.isOpened():
                        for frame_path, result in zip(sampled_frame_paths, sampled_detection_results):
                            if isinstance(result, (list, tuple)) and result:
                                plot_im = result[0].plot()
                            elif hasattr(result, "plot"):
                                plot_im = result.plot()
                            else:
                                plot_im = cv2.imread(frame_path)
                            if plot_im is not None:
                                writer.write(plot_im)
                        writer.release()

                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-i",
                            str(debug_video_raw),
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
                            str(debug_video),
                        ]
                        try:
                            subprocess.run(cmd, check=True)
                            debug_video_raw.unlink(missing_ok=True)
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            debug_video_raw.replace(debug_video)
                        self.logger.info(f"Debug track video saved to {debug_video}")
                    else:
                        self.logger.warning(f"Failed to open debug track video writer: {debug_video_raw}")

        # Build window features
        # Keep only one sliding window of decoded images in memory.
        windows, curves = self._build_window_features(
            sampled_frame_ids=sampled_frame_ids,
            sampled_frame_paths=sampled_frame_paths,
            sampled_tracks=sampled_tracks,
        )
        scores = curves["dense_trigger"].tolist()
        assert len(scores) == len(sampled_frame_ids), f"Length of scores ({len(scores)}) should match length of sampled frames ({len(sampled_frame_ids)})"

        manifest_path = str(getattr(getattr(self.cfg, "evaluation", None), "manifest_path", "") or "")
        gt_intervals = parse_manifest_intervals(manifest_path, video_id) if manifest_path else []

        # plot all subscores
        if self.cfg.video.PLOT_ALL_SUBSCORES:
            all_subscores = {}
            if windows and sampled_frame_ids:
                subscore_keys = sorted({key for window in windows for key in window.feature_dict.keys()})
                prefix_len = max(0, int(self.cfg.video.window_size) - 1)
                for key in subscore_keys:
                    values = [float(window.feature_dict.get(key, 0.0) or 0.0) for window in windows]
                    if not values:
                        continue
                    dense = np.asarray([values[0]] * prefix_len + values, dtype=np.float32)
                    if dense.size < len(sampled_frame_ids):
                        dense = np.pad(dense, (0, len(sampled_frame_ids) - dense.size), mode="edge")
                    all_subscores[key] = dense[: len(sampled_frame_ids)].astype(np.float32).tolist()
                for key in (
                    "dense_gate_global_evidence",
                    "dense_gate_null_weight",
                    "dense_gate_active_experts",
                ):
                    if key in curves:
                        all_subscores[key] = curves[key].astype(np.float32).tolist()

                write_json(
                    str(cur_output_dir / "all_subscores.json"),
                    {
                        "frame_ids": [int(frame_id) for frame_id in sampled_frame_ids],
                        "subscores": all_subscores,
                        "window_rows": [
                            {
                                "window_id": int(window.window_id),
                                "start_frame": int(window.start_frame),
                                "end_frame": int(window.end_frame),
                                "feature_dict": {
                                    key: float(value)
                                    for key, value in window.feature_dict.items()
                                },
                            }
                            for window in windows
                        ],
                    },
                )
                plot_all_subscores(
                    frame_ids=sampled_frame_ids,
                    subscores=all_subscores,
                    gt_intervals=gt_intervals,
                    title="All Subscores",
                    output_dir=str(cur_output_dir),
                    filename="all_subscores.png",
                )

        boundary_result = BoundaryDetector(self.boundary_detector_cfg).detect_details(scores)
        peaks = boundary_result.peaks
        coarse_spans = boundary_result.spans
        coarse_frame_spans = self._frame_spans_from_indices(sampled_frame_ids, coarse_spans)

        plot_peaks(
            scores=scores,
            frame_ids=sampled_frame_ids,
            peaks=peaks,
            title="Dense Trigger Scores with Detected Peaks",
            output_dir=str(cur_output_dir),
        )
        plot_scores(
            scores=curves["dense_trigger"],
            frame_ids=sampled_frame_ids,
            pred_spans=coarse_frame_spans,
            gt_intervals=gt_intervals,
            output_dir=str(cur_output_dir),
        )
        self.logger.info(f"Score plot saved to {cur_output_dir}")

        # Delete cached sampled frames to save space
        for frame_path in sampled_frame_paths:
            try:
                Path(frame_path).unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete cached frame {frame_path}: {e}")

        runtime_sec = float(time.perf_counter() - start_time)
        result = {
            "video_id": video_id,
            "num_sampled_frames": int(len(sampled_frame_ids)),
            "num_windows": int(len(windows)),
            "num_pred_spans": int(len(coarse_frame_spans)),
            "peaks_idx": [int(v) for v in peaks],
            "peaks_frame": [int(sampled_frame_ids[v]) for v in peaks if 0 <= int(v) < len(sampled_frame_ids)],
            "core_spans_idx": [[int(s), int(e)] for s, e in coarse_spans],
            "core_spans_frame": [[int(s), int(e)] for s, e in coarse_frame_spans],
            "thresholds": {str(k): float(v) for k, v in boundary_result.thresholds.items()},
            "gt_intervals": [[int(s), int(e)] for s, e in gt_intervals],
            "gt_metrics": compute_gt_span_metrics(coarse_frame_spans, gt_intervals),
            "runtime_sec": runtime_sec,
        }
        (cur_output_dir / "boundary_spans.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        return result
