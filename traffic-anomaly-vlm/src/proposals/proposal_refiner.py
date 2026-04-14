from __future__ import annotations

from src.schemas import EventProposal


class ProposalRefiner:
    def __init__(self, min_len_frames: int = 8, merge_gap_frames: int = 8, buffer_frames: int = 2):
        self.min_len_frames = min_len_frames
        self.merge_gap_frames = merge_gap_frames
        self.buffer_frames = buffer_frames

    def refine(self, proposals: list[EventProposal]) -> list[EventProposal]:
        if not proposals:
            return []

        proposals = sorted(proposals, key=lambda p: p.start_frame)
        merged: list[EventProposal] = [proposals[0]]

        for p in proposals[1:]:
            prev = merged[-1]
            if p.start_frame - prev.end_frame <= self.merge_gap_frames:
                prev.end_frame = max(prev.end_frame, p.end_frame)
                if p.scores.get("peak_trigger", 0.0) > prev.scores.get("peak_trigger", 0.0):
                    prev.peak_frame = p.peak_frame
                    prev.scores = dict(p.scores)
            else:
                merged.append(p)

        out: list[EventProposal] = []
        for p in merged:
            p.start_frame = max(0, p.start_frame - self.buffer_frames)
            p.end_frame += self.buffer_frames
            if (p.end_frame - p.start_frame) >= self.min_len_frames:
                out.append(p)
        return out
