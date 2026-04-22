from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.schemas import EventProposal, TrackObject, WindowFeature


@dataclass
class LegacyWindowProposalConfig:
    """Settings for the legacy window-based offline proposal conversion."""

    min_len_frames: int = 8
    merge_gap_frames: int = 8
    buffer_frames: int = 2
    peak_tolerance_frames: int = 2


def build_window_event_proposals(
    boundaries: list[tuple[int, int]],
    windows: list[WindowFeature],
    cfg: LegacyWindowProposalConfig | None = None,
) -> list[EventProposal]:
    """Convert window-index spans into buffered frame-level event proposals."""

    cfg = cfg or LegacyWindowProposalConfig()
    if not boundaries or not windows:
        return []

    proposals: list[EventProposal] = []
    for start_idx, end_idx in boundaries:
        proposal = _build_single_window_proposal(start_idx, end_idx, windows, cfg)
        if proposal is not None:
            proposals.append(proposal)

    proposals = _merge_close_window_proposals(proposals, cfg)
    proposals = [proposal for proposal in proposals if _proposal_span_len(proposal) >= int(cfg.min_len_frames)]
    return [_rename_proposal(proposal, idx) for idx, proposal in enumerate(proposals, start=1)]


def annotate_window_event_proposals(
    proposals: list[EventProposal],
    tracks: list[TrackObject],
    peak_tolerance_frames: int = 2,
) -> list[EventProposal]:
    """Attach main/related track ids using only tracks inside each proposal span."""

    if not proposals or not tracks:
        return proposals

    annotated: list[EventProposal] = []
    tolerance = max(0, int(peak_tolerance_frames))
    for proposal in proposals:
        span_tracks = [track for track in tracks if proposal.start_frame <= track.frame_id <= proposal.end_frame]
        if not span_tracks:
            annotated.append(proposal)
            continue

        counts = Counter(track.track_id for track in span_tracks)
        peak_counts = Counter(
            track.track_id
            for track in span_tracks
            if abs(int(track.frame_id) - int(proposal.peak_frame)) <= tolerance
        )
        max_area_by_track: dict[int, float] = defaultdict(float)
        for track in span_tracks:
            max_area_by_track[track.track_id] = max(max_area_by_track[track.track_id], float(track.area))

        ranked_ids = sorted(
            counts,
            key=lambda track_id: (
                peak_counts.get(track_id, 0),
                counts[track_id],
                max_area_by_track.get(track_id, 0.0),
                -int(track_id),
            ),
            reverse=True,
        )
        main_track_id = int(ranked_ids[0]) if ranked_ids else None
        annotated.append(
            proposal.model_copy(
                update={
                    "main_track_id": main_track_id,
                    "related_track_ids": [int(track_id) for track_id in ranked_ids],
                }
            )
        )
    return annotated


def _build_single_window_proposal(
    start_idx: int,
    end_idx: int,
    windows: list[WindowFeature],
    cfg: LegacyWindowProposalConfig,
) -> EventProposal | None:
    start_idx = max(0, int(start_idx))
    end_idx = min(len(windows) - 1, int(end_idx))
    if end_idx < start_idx:
        return None

    window_slice = windows[start_idx : end_idx + 1]
    if not window_slice:
        return None

    peak_window = max(window_slice, key=lambda window: float(window.trigger_score))
    start_frame = max(0, int(window_slice[0].start_frame) - int(cfg.buffer_frames))
    end_frame = int(window_slice[-1].end_frame) + int(cfg.buffer_frames)
    mean_trigger = sum(float(window.trigger_score) for window in window_slice) / max(1, len(window_slice))
    return EventProposal(
        event_id="event.pending",
        start_frame=start_frame,
        end_frame=end_frame,
        peak_frame=int(peak_window.end_frame),
        scores={
            "peak_trigger": float(peak_window.trigger_score),
            "mean_trigger": float(mean_trigger),
            "window_count": float(len(window_slice)),
        },
    )


def _merge_close_window_proposals(
    proposals: list[EventProposal],
    cfg: LegacyWindowProposalConfig,
) -> list[EventProposal]:
    if not proposals:
        return []
    ordered = sorted(proposals, key=lambda proposal: (int(proposal.start_frame), int(proposal.end_frame)))
    merged = [ordered[0]]
    for proposal in ordered[1:]:
        prev = merged[-1]
        if int(proposal.start_frame) - int(prev.end_frame) > int(cfg.merge_gap_frames):
            merged.append(proposal)
            continue
        merged[-1] = _merge_two_proposals(prev, proposal)
    return merged


def _merge_two_proposals(left: EventProposal, right: EventProposal) -> EventProposal:
    left_peak = float(left.scores.get("peak_trigger", 0.0))
    right_peak = float(right.scores.get("peak_trigger", 0.0))
    peak_frame = int(left.peak_frame if left_peak >= right_peak else right.peak_frame)
    total_windows = float(left.scores.get("window_count", 0.0) + right.scores.get("window_count", 0.0))
    weighted_mean = (
        float(left.scores.get("mean_trigger", 0.0)) * float(left.scores.get("window_count", 0.0))
        + float(right.scores.get("mean_trigger", 0.0)) * float(right.scores.get("window_count", 0.0))
    ) / max(1.0, total_windows)
    return EventProposal(
        event_id=left.event_id,
        start_frame=min(int(left.start_frame), int(right.start_frame)),
        end_frame=max(int(left.end_frame), int(right.end_frame)),
        peak_frame=peak_frame,
        scores={
            "peak_trigger": max(left_peak, right_peak),
            "mean_trigger": float(weighted_mean),
            "window_count": float(total_windows),
        },
    )


def _proposal_span_len(proposal: EventProposal) -> int:
    return int(proposal.end_frame - proposal.start_frame + 1)


def _rename_proposal(proposal: EventProposal, idx: int) -> EventProposal:
    return proposal.model_copy(update={"event_id": f"event_{idx:03d}"})

