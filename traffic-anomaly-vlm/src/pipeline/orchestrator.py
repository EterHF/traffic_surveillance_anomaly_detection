from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.core.logger import get_logger
from src.core.sampler import FrameSampler
from src.core.video_io import build_reader
from src.evidence.builder import EvidenceBuilder
from src.features.feature_builder import FeatureBuilder
from src.features.feature_cache import FeatureCache
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_refiner import refine_track_ids
from src.perception.track_cache import TrackCache
from src.perception.track_parser import parse_ultralytics_results
from src.proposals.main_object import select_main_object
from src.proposals.proposal_refiner import ProposalRefiner
from src.proposals.start_time import refine_start_time
from src.schemas import FinalResult, TrackObject, VLMResult
from src.triggers.boundary import BoundaryDetector
from src.triggers.event_builder import EventBuilder


class Orchestrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger()

        components = getattr(cfg, "components", None)
        track_cache_cfg = getattr(components, "track_cache", None) if components else None
        feature_cache_cfg = getattr(components, "feature_cache", None) if components else None
        boundary_cfg = getattr(components, "boundary_detector", None) if components else None
        refiner_cfg = getattr(components, "proposal_refiner", None) if components else None
        evidence_cfg = getattr(components, "evidence_builder", None) if components else None

        self.detector_tracker = DetectorTracker(
            model_path=cfg.perception.yolo_model,
            tracker=cfg.perception.tracker,
            conf=cfg.perception.conf,
            iou=cfg.perception.iou,
            classes=list(cfg.perception.classes),
        )

        auto_track_frames = max(64, int(cfg.video.window_size * 4))
        track_max_frames = getattr(track_cache_cfg, "max_frames", None)
        if track_max_frames is None:
            track_max_frames = auto_track_frames
        else:
            track_max_frames = max(int(track_max_frames), int(cfg.video.window_size))
        self.track_cache_max_frames = int(track_max_frames)
        self.track_cache = TrackCache(max_frames=track_max_frames)

        track_feat_cfg = getattr(getattr(cfg, "features", None), "track", None)
        track_fit_degree = int(getattr(track_feat_cfg, "fit_degree", 2)) if track_feat_cfg is not None else 2
        track_history_len_raw = getattr(track_feat_cfg, "history_len", None) if track_feat_cfg is not None else None
        track_history_len = int(track_history_len_raw) if track_history_len_raw is not None else None
        self.feature_builder = FeatureBuilder(
            cfg.video.window_size,
            cfg.video.window_step,
            track_fit_degree=track_fit_degree,
            track_history_len=track_history_len,
        )
        feature_max_windows = getattr(feature_cache_cfg, "max_windows", None)
        if feature_max_windows is not None:
            feature_max_windows = int(feature_max_windows)
            if feature_max_windows <= 0:
                feature_max_windows = None
        self.feature_cache_max_windows = feature_max_windows
        self.feature_cache = FeatureCache(max_windows=feature_max_windows)

        track_refiner_cfg = getattr(getattr(cfg, "perception", None), "track_refiner", None)
        self.track_refiner_enable = bool(getattr(track_refiner_cfg, "enable", track_refiner_cfg is not None))
        self.track_refiner_max_gap = int(getattr(track_refiner_cfg, "max_gap", 8)) if track_refiner_cfg is not None else 8
        self.track_refiner_max_center_dist = (
            float(getattr(track_refiner_cfg, "max_center_dist", 0.06)) if track_refiner_cfg is not None else 0.06
        )
        self.track_refiner_max_size_ratio = (
            float(getattr(track_refiner_cfg, "max_size_ratio", 2.5)) if track_refiner_cfg is not None else 2.5
        )
        self.track_refiner_min_direction_cos = (
            float(getattr(track_refiner_cfg, "min_direction_cos", -0.1)) if track_refiner_cfg is not None else -0.1
        )
        self.track_refiner_max_speed_ratio = (
            float(getattr(track_refiner_cfg, "max_speed_ratio", 3.5)) if track_refiner_cfg is not None else 3.5
        )
        self.track_refiner_gap_relax_factor = (
            float(getattr(track_refiner_cfg, "gap_relax_factor", 0.2)) if track_refiner_cfg is not None else 0.2
        )

        boundary_high = float(getattr(boundary_cfg, "high", 1.0))
        boundary_low = float(getattr(boundary_cfg, "low", 0.5))
        self.boundary_detector = BoundaryDetector(high=boundary_high, low=boundary_low)
        self.event_builder = EventBuilder()
        self.refiner = ProposalRefiner(
            min_len_frames=int(getattr(refiner_cfg, "min_len_frames", 8)),
            merge_gap_frames=int(getattr(refiner_cfg, "merge_gap_frames", 8)),
            buffer_frames=int(getattr(refiner_cfg, "buffer_frames", 2)),
        )

        assets_subdir = str(getattr(evidence_cfg, "assets_subdir", "assets"))
        assets_dir = str(Path(cfg.output.output_dir) / assets_subdir)
        self.evidence_builder = EvidenceBuilder(assets_dir)
        self.use_vlm = bool(getattr(cfg.vlm, "enable", False))

    def run_offline(self, input_video: str) -> list[FinalResult]:
        if not input_video:
            raise ValueError("input_video is empty")

        input_fps = float(getattr(self.cfg.video, "input_fps", 25.0))
        reader = build_reader(
            input_video,
            input_fps=input_fps,
            resize_max_side=int(getattr(self.cfg.video, "resize_max_side", 0) or 0),
            resize_interpolation=str(getattr(self.cfg.video, "resize_interpolation", "area")),
        )
        self.logger.info(f"Video {input_video} opened with FPS={reader.fps()} and total_frames={reader.frame_count()}")
        sampler = FrameSampler(reader.fps() or 25.0, self.cfg.video.fps_sample)

        frame_cache_dir = Path(self.cfg.output.output_dir) / "cache" / "sampled_frames"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)

        sampled_frame_ids: list[int] = []
        sampled_tracks_raw: list[list[TrackObject]] = []
        sampled_frames: dict[int, Any] = {}

        try:
            for frame_id, frame in sampler.sample(iter(reader)):
                sampled_frame_ids.append(frame_id)
                frame_path = frame_cache_dir / f"{frame_id:08d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                sampled_frames[frame_id] = str(frame_path)

                results = self.detector_tracker.track_frame(frame, persist=True)
                tracks = parse_ultralytics_results(results, frame_id)
                sampled_tracks_raw.append(tracks)
        finally:
            reader.release()

        sampled_tracks = sampled_tracks_raw
        track_refine_map: dict[int, int] = {}
        if self.track_refiner_enable and sampled_tracks_raw:
            sampled_tracks, track_refine_map = refine_track_ids(
                sampled_tracks_raw,
                max_frame_gap=self.track_refiner_max_gap,
                max_center_dist=self.track_refiner_max_center_dist,
                max_size_ratio=self.track_refiner_max_size_ratio,
                min_direction_cos=self.track_refiner_min_direction_cos,
                max_speed_ratio=self.track_refiner_max_speed_ratio,
                gap_relax_factor=self.track_refiner_gap_relax_factor,
            )

        # Rebuild caches from refined tracks so trigger/proposal/evidence use consistent ids.
        self.track_cache = TrackCache(max_frames=self.track_cache_max_frames)
        self.feature_cache = FeatureCache(max_windows=self.feature_cache_max_windows)
        frame_prefix: list[int] = []
        track_prefix: list[list[TrackObject]] = [] # cache in order to simulate online processing, so only use tracks within bounded cache tail for feature/windows build, but keep all sampled tracks for final proposal/evidence correctness.
        for frame_id, tracks in zip(sampled_frame_ids, sampled_tracks):
            frame_prefix.append(frame_id)
            track_prefix.append(tracks)
            self.track_cache.add(tracks)
            if self.feature_builder.ready(frame_prefix, track_prefix):
                frame_win = frame_prefix[-self.cfg.video.window_size :]
                track_win = self.track_cache.get_window(self.cfg.video.window_size)
                wf = self.feature_builder.build(frame_win, track_win)
                self.feature_cache.add(wf)

        windows = self.feature_cache.all()
        scores = [w.trigger_score for w in windows]
        boundaries = self.boundary_detector.detect(scores)
        proposals = self.event_builder.build(boundaries, windows)
        proposals = self.refiner.refine(proposals)

        self.last_debug = {
            "windows": windows,
            "scores": scores,
            "proposals": proposals,
            "sampled_frame_ids": sampled_frame_ids,
            "sampled_frames": sampled_frames,
            "sampled_tracks": sampled_tracks,
            "sampled_tracks_raw": sampled_tracks_raw,
            "track_refine_map": track_refine_map,
        }

        # Use all sampled tracks (not bounded cache tail) for full-video proposal/evidence correctness.
        all_tracks = [t for tracks in sampled_tracks for t in tracks]
        final_results: list[FinalResult] = []
        for p in proposals:
            p = refine_start_time(select_main_object(p, all_tracks))
            evidence = self.evidence_builder.build(p, windows, sampled_frames, all_tracks)

            if self.use_vlm:
                from src.vlm.infer import run_inference
                from src.vlm.model_loader import LocalVLM
                from src.vlm.parser import parse_vlm_output
                from src.vlm.prompt_stage1 import build_stage1_prompt
                from src.vlm.prompt_stage2 import build_stage2_prompt

                local_vlm = LocalVLM(
                    model_path=self.cfg.vlm.model_path,
                    device=self.cfg.vlm.device,
                    dtype=self.cfg.vlm.dtype,
                )
                p1 = build_stage1_prompt(evidence.summary)
                stage1_text = run_inference(local_vlm, p1, evidence.keyframe_paths, self.cfg.vlm.max_new_tokens)
                p2 = build_stage2_prompt(stage1_text)
                stage2_text = run_inference(local_vlm, p2, evidence.overlay_paths, self.cfg.vlm.max_new_tokens)
                vlm_result = parse_vlm_output(stage2_text)
            else:
                vlm_result = VLMResult(
                    is_anomaly=False,
                    event_type="vlm_disabled",
                    confidence=0.0,
                    summary="VLM inference is disabled. Detection/tracking and visual debug outputs are available.",
                    supporting_evidence=[],
                    counter_evidence=[],
                )

            final_results.append(FinalResult(event_id=p.event_id, proposal=p, evidence=evidence, vlm_result=vlm_result))

        return final_results
