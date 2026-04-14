from __future__ import annotations

from src.schemas import EventProposal


def select_keyframes(proposal: EventProposal, n: int = 4) -> list[int]:
    if proposal.end_frame <= proposal.start_frame:
        return [proposal.start_frame]
    span = proposal.end_frame - proposal.start_frame
    step = max(1, span // max(n - 1, 1))
    picks = [proposal.start_frame + i * step for i in range(n)]
    picks[-1] = proposal.end_frame
    if proposal.peak_frame not in picks:
        picks[min(len(picks) - 1, n // 2)] = proposal.peak_frame
    return sorted(set(picks))
