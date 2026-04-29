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
    # Optional frame size, filled by parser when available.
    frame_w: float = 0.0
    frame_h: float = 0.0


class WindowFeature(BaseModel):
    window_id: int
    start_frame: int
    end_frame: int
    feature_dict: dict[str, Any] = Field(default_factory=dict)
    trigger_score: float = 0.0


class SpanProposal(BaseModel):
    """Flat candidate span with optional VLM scoring fields."""

    span_id: str
    start_idx: int
    end_idx: int
    peak_idx: int
    start_frame: int
    end_frame: int
    peak_frame: int
    eventness_peak: float
    eventness_mean: float
    left_gap: int = 0
    right_gap: int = 0
    prior_score: float = 0.0
    vlm_score: float = 0.0
    vlm_confidence: float = 0.0
    fused_score: float = 0.0
    is_positive: bool = False


class EvidencePack(BaseModel):
    event_id: str
    keyframe_paths: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
