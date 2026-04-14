from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.evidence.keyframes import select_keyframes
from src.evidence.overlays import draw_overlay, save_image
from src.evidence.summary import build_summary
from src.evidence.trajectory_plot import plot_trajectories
from src.schemas import EvidencePack, EventProposal, TrackObject, WindowFeature


class EvidenceBuilder:
    def __init__(self, output_assets_dir: str):
        self.output_assets_dir = output_assets_dir
        Path(output_assets_dir).mkdir(parents=True, exist_ok=True)

    def build(
        self,
        proposal: EventProposal,
        windows: list[WindowFeature],
        frames: dict[int, Any],
        all_tracks: list[TrackObject],
    ) -> EvidencePack:
        key_ids = select_keyframes(proposal)
        keyframe_paths: list[str] = []
        overlay_paths: list[str] = []

        for fid in key_ids:
            frame_ref = frames.get(fid)
            if isinstance(frame_ref, str):
                frame = cv2.imread(frame_ref)
            else:
                frame = frame_ref
            if frame is None:
                continue
            raw_path = str(Path(self.output_assets_dir) / f"{proposal.event_id}_frame_{fid}.jpg")
            save_image(raw_path, frame)
            keyframe_paths.append(raw_path)

            tracks_on_frame = [t for t in all_tracks if t.frame_id == fid]
            over = draw_overlay(frame, tracks_on_frame, proposal.main_track_id)
            overlay_path = str(Path(self.output_assets_dir) / f"{proposal.event_id}_overlay_{fid}.jpg")
            save_image(overlay_path, over)
            overlay_paths.append(overlay_path)

        event_tracks = [t for t in all_tracks if proposal.start_frame <= t.frame_id <= proposal.end_frame]
        traj = plot_trajectories(event_tracks)
        traj_path = str(Path(self.output_assets_dir) / f"{proposal.event_id}_trajectory.jpg")
        cv2.imwrite(traj_path, traj)

        summary = build_summary(proposal, windows)

        return EvidencePack(
            event_id=proposal.event_id,
            keyframe_paths=keyframe_paths,
            overlay_paths=overlay_paths,
            trajectory_plot_path=traj_path,
            summary=summary,
        )
