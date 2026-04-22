from __future__ import annotations

from src.schemas import EventProposal, WindowFeature


class EventBuilder:
    def __init__(self):
        self._idx = 0

    def build(self, boundaries: list[tuple[int, int]], windows: list[WindowFeature]) -> list[EventProposal]:
        proposals: list[EventProposal] = []
        for s, e in boundaries:
            segment = windows[s : e + 1]
            if not segment:
                continue
            peak = max(segment, key=lambda x: x.trigger_score)
            eid = f"evt_{self._idx:05d}"
            self._idx += 1
            proposals.append(
                EventProposal(
                    event_id=eid,
                    start_frame=segment[0].start_frame,
                    end_frame=segment[-1].end_frame,
                    peak_frame=(peak.start_frame + peak.end_frame) // 2,
                    scores={"peak_trigger": peak.trigger_score},
                )
            )
        return proposals
