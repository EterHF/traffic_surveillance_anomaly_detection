from __future__ import annotations

from collections import deque

from src.schemas import TrackObject


class TrackCache:
    def __init__(self, max_frames: int = 64):
        self.frames: deque[list[TrackObject]] = deque(maxlen=max_frames)

    def add(self, tracks: list[TrackObject]) -> None:
        self.frames.append(tracks)

    def get_window(self, window_size: int) -> list[list[TrackObject]]:
        return list(self.frames)[-window_size:]

    def all_tracks_flat(self) -> list[TrackObject]:
        out: list[TrackObject] = []
        for tracks in self.frames:
            out.extend(tracks)
        return out
