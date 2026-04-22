from __future__ import annotations

from collections import Counter

from src.schemas import EventProposal, TrackObject


def select_main_object(proposal: EventProposal, tracks: list[TrackObject]) -> EventProposal:
    event_tracks = [t for t in tracks if proposal.start_frame <= t.frame_id <= proposal.end_frame]
    if not event_tracks:
        return proposal

    cnt = Counter([t.track_id for t in event_tracks])
    main_id, _ = cnt.most_common(1)[0]
    proposal.main_track_id = int(main_id)

    related = [tid for tid, _ in cnt.most_common(5) if tid != main_id]
    proposal.related_track_ids = [int(x) for x in related]
    return proposal
