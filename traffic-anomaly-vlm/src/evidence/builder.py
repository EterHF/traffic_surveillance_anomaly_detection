from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.evidence.evidence_utils import (
    build_event_summary,
    build_labeled_montage,
    load_frame_ref,
    save_image
)
from src.schemas import EvidencePack, EventNode, TrackObject, WindowFeature


class EvidenceFactory:
    """Unified evidence factory for EventNode-based evidence."""

    def __init__(self, default_output_dir: str | None = None):
        self.default_output_dir = str(default_output_dir) if default_output_dir else None
        if self.default_output_dir:
            Path(self.default_output_dir).mkdir(parents=True, exist_ok=True)

    def build(
        self,
        subject: EventNode,
        *,
        images: list[str],
        frame_ids: list[int],
        output_dir: str | None = None,
        windows: list[WindowFeature] | None = None,
        all_tracks: list[TrackObject] | None = None,
        method: Literal["montage", "enhanced"] = "montage",
    ) -> EvidencePack:

        resolved_output_dir = str(output_dir or self.default_output_dir or "").strip()
        if not resolved_output_dir:
            raise ValueError("output_dir is required unless default_output_dir is set")
        if method == "montage":
            return self._build_montage_evidence(
                node=subject,
                images=images,
                frame_ids=frame_ids,
                output_dir=resolved_output_dir,
            )
        else:
            raise NotImplementedError(f"Unsupported evidence build method: {method}")

    @staticmethod
    def _sample_span_indices(start_idx: int, peak_idx: int, end_idx: int) -> list[int]:
        if end_idx <= start_idx:
            return [int(start_idx)]
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
        node: EventNode,
        images: list[str],
        frame_ids: list[int],
        output_dir: str,
    ) -> EvidencePack:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        start_idx = int(node.start_idx)
        end_idx = int(node.end_idx)
        peak_idx = int(node.peak_idx)
        span_id = str(node.node_id)
        sample_indices = self._sample_span_indices(start_idx, peak_idx, end_idx)

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

        summary = build_event_summary(node)
        summary["keyframe_ids"] = [int(v) for v in keyframe_ids]
        summary["evidence_method"] = "montage"
        return EvidencePack(
            event_id=span_id,
            keyframe_paths=[montage_path] if montage_path else [],
            summary=summary,
        )

    def _build_enhanced_evidence(
        self,
        node: EventNode,
        images: list[str],
        frame_ids: list[int],
        windows: list[WindowFeature] | None,
        all_tracks: list[TrackObject],
        output_dir: str,
    ) -> EvidencePack:
        raise NotImplementedError("Enhanced evidence build is not implemented yet")


class EvidenceBuilder:
    def __init__(self, output_assets_dir: str):
        self._factory = EvidenceFactory(default_output_dir=output_assets_dir)

    def build(
        self,
        proposal: EventNode,
        images: list[str],
        frame_ids: list[int],
        all_tracks: list[TrackObject],
        windows: list[WindowFeature] | None = None,
        method: Literal["montage", "enhanced"] = "montage",
        output_dir: str | None = None,
    ) -> EvidencePack:
        return self._factory.build(
            proposal,
            images=images,
            frame_ids=frame_ids,
            windows=windows,
            all_tracks=all_tracks,
            method=method,
            output_dir=output_dir,
        )
