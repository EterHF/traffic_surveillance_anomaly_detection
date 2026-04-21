from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.evidence.overlays import draw_overlay, save_image
from src.proposals.coarse_proposal import CoarseProposal
from src.schemas import TrackObject


@dataclass
class CoarseEvidencePack:
    proposal_id: str
    keyframe_ids: list[int] = field(default_factory=list)
    raw_montage_path: str = ""
    overlay_montage_path: str = ""
    crop_montage_path: str = ""
    summary: dict = field(default_factory=dict)

    def vlm_image_paths(self) -> list[str]:
        return [p for p in [self.raw_montage_path, self.overlay_montage_path, self.crop_montage_path] if p]


def _select_focus_track(tracks: list[TrackObject]) -> int | None:
    if not tracks:
        return None
    focus = max(tracks, key=lambda t: float(t.area))
    return int(focus.track_id)


def _sample_indices(proposal: CoarseProposal) -> list[int]:
    picks = [proposal.start_idx, int(round((proposal.start_idx + proposal.end_idx) * 0.5)), proposal.peak_idx, proposal.end_idx]
    out: list[int] = []
    seen = set()
    for idx in picks:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _resolve_crop_box(frame: np.ndarray, tracks: list[TrackObject], focus_track_id: int | None) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    focus_tracks = [t for t in tracks if focus_track_id is not None and int(t.track_id) == int(focus_track_id)]
    if not focus_tracks:
        focus_tracks = list(tracks)
    if not focus_tracks:
        return 0, 0, max(1, w), max(1, h)

    x1 = min(float(t.bbox_xyxy[0]) for t in focus_tracks)
    y1 = min(float(t.bbox_xyxy[1]) for t in focus_tracks)
    x2 = max(float(t.bbox_xyxy[2]) for t in focus_tracks)
    y2 = max(float(t.bbox_xyxy[3]) for t in focus_tracks)

    pad_x = max(12.0, 0.15 * (x2 - x1 + 1.0))
    pad_y = max(12.0, 0.15 * (y2 - y1 + 1.0))
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, max(1, w), max(1, h)
    return x1, y1, x2, y2


def _crop_focus_region(frame: np.ndarray, tracks: list[TrackObject], focus_track_id: int | None) -> np.ndarray:
    x1, y1, x2, y2 = _resolve_crop_box(frame, tracks, focus_track_id)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame.copy()
    return crop


def _build_montage(tiles: list[np.ndarray], labels: list[str], thumb_size: tuple[int, int] = (280, 180)) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for tile, label in zip(tiles, labels):
        view = cv2.resize(tile, thumb_size)
        cv2.putText(
            view,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(view)
    while len(rendered) < 4:
        rendered.append(np.full((thumb_size[1], thumb_size[0], 3), 235, dtype=np.uint8))
    top = np.hstack(rendered[:2])
    bottom = np.hstack(rendered[2:4])
    return np.vstack([top, bottom])


def build_coarse_evidence(
    proposal: CoarseProposal,
    images: list[np.ndarray],
    frame_ids: list[int],
    tracks_per_frame: list[list[TrackObject]],
    output_dir: str,
) -> CoarseEvidencePack:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_by_frame = {int(fid): idx for idx, fid in enumerate(frame_ids)}

    sample_indices = _sample_indices(proposal)
    raw_tiles: list[np.ndarray] = []
    overlay_tiles: list[np.ndarray] = []
    crop_tiles: list[np.ndarray] = []
    labels: list[str] = []
    keyframe_ids: list[int] = []

    peak_tracks = tracks_per_frame[proposal.peak_idx] if 0 <= proposal.peak_idx < len(tracks_per_frame) else []
    focus_track_id = _select_focus_track(peak_tracks)

    for idx in sample_indices:
        if idx < 0 or idx >= len(images):
            continue
        frame_id = int(frame_ids[idx])
        keyframe_ids.append(frame_id)
        frame = images[idx]
        tracks = tracks_per_frame[idx]
        raw_tiles.append(frame.copy())
        overlay_tiles.append(draw_overlay(frame, tracks, focus_track_id))
        crop_tiles.append(_crop_focus_region(frame, tracks, focus_track_id))
        labels.append(f"frame={frame_id}")

    if not raw_tiles:
        blank = np.full((180, 280, 3), 235, dtype=np.uint8)
        raw_tiles = [blank]
        overlay_tiles = [blank]
        crop_tiles = [blank]
        labels = ["empty"]

    raw_montage = _build_montage(raw_tiles, labels)
    overlay_montage = _build_montage(overlay_tiles, labels)
    crop_montage = _build_montage(crop_tiles, labels)

    raw_montage_path = str(out_dir / f"{proposal.proposal_id}_raw.jpg")
    overlay_montage_path = str(out_dir / f"{proposal.proposal_id}_overlay.jpg")
    crop_montage_path = str(out_dir / f"{proposal.proposal_id}_crop.jpg")
    save_image(raw_montage_path, raw_montage)
    save_image(overlay_montage_path, overlay_montage)
    save_image(crop_montage_path, crop_montage)

    total_frames = max(1, len(frame_ids))
    summary = {
        "proposal_id": proposal.proposal_id,
        "start_frame": int(proposal.start_frame),
        "end_frame": int(proposal.end_frame),
        "peak_frame": int(proposal.peak_frame),
        "span_len": int(proposal.span_len()),
        "span_ratio": float(proposal.span_ratio(total_frames)),
        "peak_offset_ratio": float(proposal.peak_offset_ratio()),
        "sources": list(proposal.sources),
        "coarse_score": float(proposal.coarse_score),
        "seed_peak": float(proposal.seed_peak),
        "seed_mean": float(proposal.seed_mean),
        "track_peak": float(proposal.track_peak),
        "object_peak": float(proposal.object_peak),
        "focus_track_id": int(focus_track_id) if focus_track_id is not None else None,
        "keyframe_ids": [int(v) for v in keyframe_ids],
    }
    return CoarseEvidencePack(
        proposal_id=proposal.proposal_id,
        keyframe_ids=keyframe_ids,
        raw_montage_path=raw_montage_path,
        overlay_montage_path=overlay_montage_path,
        crop_montage_path=crop_montage_path,
        summary=summary,
    )
