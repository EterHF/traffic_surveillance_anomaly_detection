from __future__ import annotations

import numpy as np

from .config import NodeSelectorConfig
from .helpers import clip01, interval_iou, node_signal_summary
from src.schemas import EventNode
from .tree_builder import flatten_event_tree


def score_event_nodes(
    nodes: list[EventNode],
    eventness_scores: list[float] | np.ndarray,
    track_risk_scores: list[float] | np.ndarray,
    object_prior_scores: list[float] | np.ndarray | None = None,
    gate_floor: float = 0.35,
    reference_span: int = 32,
) -> list[EventNode]:
    """Fuse eventness, track risk, and semantic priors into final node scores."""

    eventness = np.asarray(eventness_scores, dtype=np.float32)
    track_risk = np.asarray(track_risk_scores, dtype=np.float32)
    object_prior = np.asarray(object_prior_scores, dtype=np.float32) if object_prior_scores is not None else None

    scored: list[EventNode] = []
    gate_floor = clip01(gate_floor)
    for node in nodes:
        event_peak, event_top_mean = node_signal_summary(eventness, node.start_idx, node.end_idx)
        track_peak, track_top_mean = node_signal_summary(track_risk, node.start_idx, node.end_idx)
        prior_peak = 0.0
        prior_top_mean = 0.0
        if object_prior is not None:
            prior_peak, prior_top_mean = node_signal_summary(object_prior, node.start_idx, node.end_idx)

        event_strength = 0.7 * event_peak + 0.3 * event_top_mean
        risk_strength = max(track_peak, 0.5 * (track_peak + prior_peak), prior_top_mean, track_top_mean)
        node.eventness_peak = float(event_peak)
        node.eventness_mean = float(event_top_mean)
        span_len = max(1, int(node.span_len()))
        compactness = min(1.0, np.sqrt(max(1.0, float(reference_span)) / float(span_len)))
        semantic_score = max(float(node.span_prior_score), float(node.vlm_score))
        semantic_gate = 1.0 if semantic_score <= 0.0 else gate_floor + (1.0 - gate_floor) * semantic_score
        confidence_gate = 1.0 if float(node.vlm_confidence) <= 0.0 else 0.75 + 0.25 * float(node.vlm_confidence)

        node.fused_score = float(
            event_strength * (gate_floor + (1.0 - gate_floor) * risk_strength) * compactness * semantic_gate * confidence_gate
        )
        scored.append(node)
    return scored


def select_nodes_from_list(
    nodes: list[EventNode],
    eventness_scores: list[float] | np.ndarray,
    track_risk_scores: list[float] | np.ndarray,
    object_prior_scores: list[float] | np.ndarray | None = None,
    cfg: NodeSelectorConfig | None = None,
) -> list[EventNode]:
    """Keep a compact set of salient, non-overlapping candidate nodes."""

    cfg = cfg or NodeSelectorConfig()
    nodes = [node for node in nodes if node.span_len() >= int(cfg.min_node_len)]
    if not nodes:
        return []

    scored = score_event_nodes(
        nodes,
        eventness_scores=eventness_scores,
        track_risk_scores=track_risk_scores,
        object_prior_scores=object_prior_scores,
        gate_floor=cfg.gate_floor,
        reference_span=cfg.reference_span,
    )
    scored = sorted(
        scored,
        key=lambda node: (float(node.fused_score), float(node.eventness_peak), -int(node.level)),
        reverse=True,
    )

    selected: list[EventNode] = []
    for node in scored:
        if any(interval_iou((node.start_idx, node.end_idx), (chosen.start_idx, chosen.end_idx)) >= float(cfg.overlap_iou) for chosen in selected):
            continue
        selected.append(node)
        if len(selected) >= int(cfg.top_k):
            break
    return selected


def select_salient_nodes(
    root: EventNode,
    eventness_scores: list[float] | np.ndarray,
    track_risk_scores: list[float] | np.ndarray,
    object_prior_scores: list[float] | np.ndarray | None = None,
    cfg: NodeSelectorConfig | None = None,
) -> list[EventNode]:
    """Flatten one event tree and keep the highest-value nodes."""

    return select_nodes_from_list(
        flatten_event_tree(root, include_root=False),
        eventness_scores=eventness_scores,
        track_risk_scores=track_risk_scores,
        object_prior_scores=object_prior_scores,
        cfg=cfg,
    )
