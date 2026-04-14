from __future__ import annotations

from pathlib import Path

from src.core.utils import ensure_dir, write_json
from src.eval.visual_debug import (
    save_detection_preview_montage,
    save_event_timeline,
    save_keyframe_montage,
    save_window_score_curve,
)
from src.pipeline.orchestrator import Orchestrator


class OfflinePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.orch = Orchestrator(cfg)

    def run(self, input_video: str):
        results = self.orch.run_offline(input_video)
        out_dir = ensure_dir(self.cfg.output.output_dir)
        if self.cfg.output.save_json:
            write_json(f"{out_dir}/results/offline_results.json", [r.model_dump() for r in results])

        debug = getattr(self.orch, "last_debug", {})
        windows = debug.get("windows", [])
        proposals = debug.get("proposals", [])
        sampled_ids = debug.get("sampled_frame_ids", [])
        sampled_frames = debug.get("sampled_frames", {})
        sampled_tracks = debug.get("sampled_tracks", [])
        total_frames = (max(sampled_ids) + 1) if sampled_ids else 1

        debug_dir = Path(out_dir) / "debug"
        save_window_score_curve(windows, str(debug_dir / "window_score_curve.png"))
        save_event_timeline(total_frames, proposals, str(debug_dir / "event_timeline.png"))
        save_detection_preview_montage(
            sampled_ids,
            sampled_frames,
            sampled_tracks,
            str(debug_dir / "detection_preview_montage.jpg"),
        )
        save_keyframe_montage(results, str(debug_dir / "keyframe_montage.jpg"))
        return results
