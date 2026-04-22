from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import EvidencePack


@dataclass
class SpanScore:
    """Semantic score for one span after optional VLM refinement."""

    proposal_id: str
    prior_score: float
    vlm_score: float
    fused_score: float
    confidence: float
    reason: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict:
        payload = {
            "proposal_id": str(self.proposal_id),
            "prior_score": float(self.prior_score),
            "vlm_score": float(self.vlm_score),
            "fused_score": float(self.fused_score),
            "confidence": float(self.confidence),
            "reason": str(self.reason),
        }
        if self.raw_output:
            payload["raw_output"] = str(self.raw_output)
        return payload


@dataclass
class EventNode:
    """One node in the coarse-to-fine event tree."""

    node_id: str
    level: int
    start_idx: int
    end_idx: int
    peak_idx: int
    start_frame: int
    end_frame: int
    peak_frame: int
    eventness_peak: float
    eventness_mean: float
    span_prior_score: float = 0.0
    vlm_score: float = 0.0
    vlm_confidence: float = 0.0
    fused_score: float = 0.0
    children: list["EventNode"] = field(default_factory=list)

    @property
    def proposal_id(self) -> str:
        return str(self.node_id)

    def span_len(self) -> int:
        return int(self.end_idx - self.start_idx + 1)

    def to_dict(self) -> dict:
        return {
            "node_id": str(self.node_id),
            "level": int(self.level),
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "peak_idx": int(self.peak_idx),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "peak_frame": int(self.peak_frame),
            "eventness_peak": float(self.eventness_peak),
            "eventness_mean": float(self.eventness_mean),
            "span_prior_score": float(self.span_prior_score),
            "vlm_score": float(self.vlm_score),
            "vlm_confidence": float(self.vlm_confidence),
            "fused_score": float(self.fused_score),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class TreePipelineResult:
    """Outputs from the full tree-based anomaly proposal pipeline."""

    coarse_spans: list[EventNode] = field(default_factory=list)
    selected_coarse_spans: list[EventNode] = field(default_factory=list)
    coarse_evidences: dict[str, EvidencePack] = field(default_factory=dict)
    coarse_scores: dict[str, SpanScore] = field(default_factory=dict)
    candidate_nodes: list[EventNode] = field(default_factory=list)
    fine_evidences: dict[str, EvidencePack] = field(default_factory=dict)
    fine_scores: dict[str, SpanScore] = field(default_factory=dict)
    selected_nodes: list[EventNode] = field(default_factory=list)

    @property
    def coarse_proposals(self) -> list[EventNode]:
        return self.coarse_spans

    @property
    def coarse_selected(self) -> list[EventNode]:
        return self.selected_coarse_spans

    @property
    def fine_nodes(self) -> list[EventNode]:
        return self.candidate_nodes

    @property
    def route_evidences(self) -> dict[str, EvidencePack]:
        return self.coarse_evidences

    @property
    def route_decisions(self) -> dict[str, SpanScore]:
        return self.coarse_scores
