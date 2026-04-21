from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.proposals.event_tree import EventNode, flatten_event_tree


@dataclass
class NodeSelectorConfig:
    top_k: int = 6
    min_node_len: int = 8
    overlap_iou: float = 0.85
    gate_floor: float = 0.55
    reference_span: int = 96


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s + 1)
    union = max(1, (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)
    return float(inter / union)


def _top_mean(values: np.ndarray, ratio: float = 0.25) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * float(ratio))))
    topk = np.sort(values)[-k:]
    return float(np.mean(topk))


def _node_signal_summary(values: np.ndarray, start_idx: int, end_idx: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lo = max(0, int(start_idx))
    hi = min(values.size - 1, int(end_idx))
    if hi < lo:
        return 0.0, 0.0
    seg = values[lo : hi + 1]
    if seg.size == 0:
        return 0.0, 0.0
    return float(np.max(seg)), _top_mean(seg, ratio=0.25)


def score_event_nodes(
    nodes: list[EventNode],
    eventness_scores: list[float] | np.ndarray,
    track_risk_scores: list[float] | np.ndarray,
    object_prior_scores: list[float] | np.ndarray | None = None,
    gate_floor: float = 0.35,
    reference_span: int = 32,
) -> list[EventNode]:
    eventness = np.asarray(eventness_scores, dtype=np.float32)
    track_risk = np.asarray(track_risk_scores, dtype=np.float32)
    object_prior = np.asarray(object_prior_scores, dtype=np.float32) if object_prior_scores is not None else None

    scored: list[EventNode] = []
    gate_floor = min(max(float(gate_floor), 0.0), 1.0)
    for node in nodes:
        event_peak, event_top_mean = _node_signal_summary(eventness, node.start_idx, node.end_idx)
        track_peak, track_top_mean = _node_signal_summary(track_risk, node.start_idx, node.end_idx)
        prior_peak = 0.0
        prior_top_mean = 0.0
        if object_prior is not None:
            prior_peak, prior_top_mean = _node_signal_summary(object_prior, node.start_idx, node.end_idx)

        event_strength = 0.7 * event_peak + 0.3 * event_top_mean
        risk_strength = max(track_peak, 0.5 * (track_peak + prior_peak), prior_top_mean, track_top_mean)
        node.eventness_peak = float(event_peak)
        node.eventness_mean = float(event_top_mean)
        span_len = max(1, int(node.span_len()))
        compactness = min(1.0, np.sqrt(max(1.0, float(reference_span)) / float(span_len)))
        node.fused_score = float(event_strength * (gate_floor + (1.0 - gate_floor) * risk_strength) * compactness)
        scored.append(node)
    return scored


def select_salient_nodes(
    root: EventNode,
    eventness_scores: list[float] | np.ndarray,
    track_risk_scores: list[float] | np.ndarray,
    object_prior_scores: list[float] | np.ndarray | None = None,
    cfg: NodeSelectorConfig | None = None,
) -> list[EventNode]:
    cfg = cfg or NodeSelectorConfig()
    nodes = flatten_event_tree(root, include_root=False)
    nodes = [n for n in nodes if n.span_len() >= int(cfg.min_node_len)]
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
        key=lambda n: (float(n.fused_score), float(n.eventness_peak), -int(n.level)),
        reverse=True,
    )

    selected: list[EventNode] = []
    for node in scored:
        overlap = False
        for chosen in selected:
            iou = _interval_iou((node.start_idx, node.end_idx), (chosen.start_idx, chosen.end_idx))
            if iou >= float(cfg.overlap_iou):
                overlap = True
                break
        if overlap:
            continue
        selected.append(node)
        if len(selected) >= int(cfg.top_k):
            break
    return selected
