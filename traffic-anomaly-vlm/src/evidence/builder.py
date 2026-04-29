from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.evidence.evidence_utils import (
    build_event_summary,
    build_labeled_montage,
    draw_track_annotations,
    filter_tracks_by_frame,
    load_frame_ref,
    save_image,
)
from src.schemas import EvidencePack, SpanProposal, TrackObject


class EvidenceFactory:
    """Unified evidence factory for span-level VLM evidence."""

    def __init__(self, default_output_dir: str | None = None):
        self.default_output_dir = str(default_output_dir) if default_output_dir else None
        if self.default_output_dir:
            Path(self.default_output_dir).mkdir(parents=True, exist_ok=True)

    def build(
        self,
        subject: SpanProposal,
        *,
        images: list[str],
        frame_ids: list[int],
        output_dir: str | None = None,
        all_tracks: list[TrackObject] | None = None,
        method: Literal["montage", "frames", "enhanced"] = "montage",
    ) -> EvidencePack:

        resolved_output_dir = str(output_dir or self.default_output_dir or "").strip()
        if not resolved_output_dir:
            raise ValueError("output_dir is required unless default_output_dir is set")
        if method == "montage":
            return self._build_montage_evidence(
                span=subject,
                images=images,
                frame_ids=frame_ids,
                output_dir=resolved_output_dir,
            )
        if method == "frames":
            return self._build_frames_evidence(
                span=subject,
                images=images,
                frame_ids=frame_ids,
            )
        if method == "enhanced":
            return self._build_enhanced_evidence(
                span=subject,
                images=images,
                frame_ids=frame_ids,
                all_tracks=all_tracks or [],
                output_dir=resolved_output_dir,
            )
        raise NotImplementedError(f"Unsupported evidence build method: {method}")

    @staticmethod
    def _sample_span_indices(
        start_idx: int,
        peak_idx: int,
        end_idx: int,
        frame_ids: list[int] | None = None,
    ) -> list[int]:
        if end_idx <= start_idx:
            return [int(start_idx)]
        if frame_ids:
            last_idx = len(frame_ids) - 1
            start_idx = max(0, min(int(start_idx), last_idx))
            end_idx = max(0, min(int(end_idx), last_idx))
            target = (int(frame_ids[start_idx]) + int(frame_ids[end_idx])) * 0.5
            local_frames = np.asarray(frame_ids[start_idx : end_idx + 1], dtype=np.float64)
            midpoint = int(start_idx + int(np.argmin(np.abs(local_frames - target))))
        else:
            midpoint = int(round((int(start_idx) + int(end_idx)) * 0.5))
        picks = [int(start_idx), midpoint, int(peak_idx), int(end_idx)]
        ordered: list[int] = []
        seen: set[int] = set()
        for idx in picks:
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)
        return ordered

    def _build_montage_evidence(
        self,
        span: SpanProposal,
        images: list[str],
        frame_ids: list[int],
        output_dir: str,
    ) -> EvidencePack:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        start_idx = int(span.start_idx)
        end_idx = int(span.end_idx)
        peak_idx = int(span.peak_idx)
        span_id = str(span.span_id)
        sample_indices = self._sample_span_indices(start_idx, peak_idx, end_idx, frame_ids=frame_ids)

        tiles: list[np.ndarray] = []
        labels: list[str] = []
        keyframe_ids: list[int] = []
        for idx in sample_indices:
            if idx < 0 or idx >= len(images):
                continue
            frame = load_frame_ref(images[idx])
            if frame is None:
                continue
            frame_id = int(frame_ids[idx])
            keyframe_ids.append(frame_id)
            tiles.append(frame.copy())
            labels.append(f"frame={frame_id}")

        if not tiles:
            tiles = [np.full((180, 280, 3), 235, dtype=np.uint8)]
            labels = ["empty"]

        montage = build_labeled_montage(tiles, labels)
        montage_path = str(out_dir / f"{span_id}_montage.jpg")
        save_image(montage_path, montage)

        summary = build_event_summary(span)
        summary["keyframe_ids"] = [int(v) for v in keyframe_ids]
        summary["evidence_method"] = "montage"
        return EvidencePack(
            event_id=span_id,
            keyframe_paths=[montage_path] if montage_path else [],
            summary=summary,
        )

    def _build_frames_evidence(
        self,
        span: SpanProposal,
        images: list[str],
        frame_ids: list[int],
    ) -> EvidencePack:
        sample_indices = self._sample_span_indices(
            int(span.start_idx),
            int(span.peak_idx),
            int(span.end_idx),
            frame_ids=frame_ids,
        )
        keyframe_paths: list[str] = []
        keyframe_ids: list[int] = []
        for idx in sample_indices:
            if 0 <= idx < len(images):
                keyframe_paths.append(str(images[idx]))
                keyframe_ids.append(int(frame_ids[idx]))

        summary = build_event_summary(span)
        summary["keyframe_ids"] = [int(v) for v in keyframe_ids]
        summary["evidence_method"] = "frames"
        summary["num_images"] = int(len(keyframe_paths))
        return EvidencePack(
            event_id=str(span.span_id),
            keyframe_paths=keyframe_paths,
            summary=summary,
        )

    def _build_enhanced_evidence(
        self,
        span: SpanProposal,
        images: list[str],
        frame_ids: list[int],
        all_tracks: list[TrackObject],
        output_dir: str,
    ) -> EvidencePack:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_indices = self._sample_span_indices(
            int(span.start_idx),
            int(span.peak_idx),
            int(span.end_idx),
            frame_ids=frame_ids,
        )
        span_tracks = [
            track
            for track in all_tracks
            if int(span.start_frame) <= int(track.frame_id) <= int(span.end_frame)
        ]
        keyframe_paths: list[str] = []
        keyframe_ids: list[int] = []

        for idx in sample_indices:
            if idx < 0 or idx >= len(images):
                continue
            frame = load_frame_ref(images[idx])
            if frame is None:
                continue
            frame_id = int(frame_ids[idx])
            annotated = draw_track_annotations(
                frame,
                frame_tracks=filter_tracks_by_frame(span_tracks, frame_id),
                span_tracks=span_tracks,
            )
            save_path = str(out_dir / f"{span.span_id}_frame_{frame_id}.jpg")
            save_image(save_path, annotated)
            keyframe_paths.append(save_path)
            keyframe_ids.append(frame_id)

        summary = build_event_summary(span)
        summary["keyframe_ids"] = [int(v) for v in keyframe_ids]
        summary["evidence_method"] = "enhanced"
        summary["num_images"] = int(len(keyframe_paths))
        return EvidencePack(
            event_id=str(span.span_id),
            keyframe_paths=keyframe_paths,
            summary=summary,
        )


class EvidenceBuilder:
    def __init__(self, output_assets_dir: str):
        self._factory = EvidenceFactory(default_output_dir=output_assets_dir)

    def build(
        self,
        proposal: SpanProposal,
        images: list[str],
        frame_ids: list[int],
        all_tracks: list[TrackObject],
        method: Literal["montage", "frames", "enhanced"] = "montage",
        output_dir: str | None = None,
    ) -> EvidencePack:
        return self._factory.build(
            proposal,
            images=images,
            frame_ids=frame_ids,
            all_tracks=all_tracks,
            method=method,
            output_dir=output_dir,
        )
