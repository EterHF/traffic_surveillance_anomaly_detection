from __future__ import annotations

from src.schemas import EventProposal, WindowFeature


def build_summary(proposal: EventProposal, windows: list[WindowFeature]) -> dict:
    related = [w for w in windows if w.start_frame >= proposal.start_frame and w.end_frame <= proposal.end_frame]
    mean_trigger = sum(w.trigger_score for w in related) / max(len(related), 1)
    return {
        "event_id": proposal.event_id,
        "start_frame": proposal.start_frame,
        "end_frame": proposal.end_frame,
        "peak_frame": proposal.peak_frame,
        "main_track_id": proposal.main_track_id,
        "related_track_ids": proposal.related_track_ids,
        "peak_trigger": proposal.scores.get("peak_trigger", 0.0),
        "mean_trigger": mean_trigger,
        "window_count": len(related),
    }
