from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrackObject(BaseModel):
    frame_id: int
    track_id: int
    cls_id: int
    cls_name: str
    score: float
    bbox_xyxy: list[float]
    cx: float
    cy: float
    w: float
    h: float
    area: float


class WindowFeature(BaseModel):
    window_id: int
    start_frame: int
    end_frame: int
    feature_dict: dict[str, Any] = Field(default_factory=dict)
    trigger_score: float = 0.0


class EventProposal(BaseModel):
    event_id: str
    start_frame: int
    end_frame: int
    peak_frame: int
    main_track_id: int | None = None
    related_track_ids: list[int] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    event_id: str
    keyframe_paths: list[str] = Field(default_factory=list)
    overlay_paths: list[str] = Field(default_factory=list)
    trajectory_plot_path: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class VLMResult(BaseModel):
    is_anomaly: bool
    event_type: str
    confidence: float
    summary: str
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    event_id: str
    proposal: EventProposal
    evidence: EvidencePack
    vlm_result: VLMResult
