from __future__ import annotations

import numpy as np

from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig

from .config import TreeBuildConfig
from src.schemas import EventNode


def _node_from_span(
    node_id: str,
    level: int,
    start_idx: int,
    end_idx: int,
    frame_ids: list[int],
    eventness_scores: np.ndarray,
) -> EventNode:
    local = eventness_scores[start_idx : end_idx + 1]
    if local.size == 0:
        peak_idx = start_idx
        peak_val = 0.0
        mean_val = 0.0
    else:
        rel_peak = int(np.argmax(local))
        peak_idx = int(start_idx + rel_peak)
        peak_val = float(local[rel_peak])
        mean_val = float(np.mean(local))
    return EventNode(
        node_id=node_id,
        level=level,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        peak_idx=int(peak_idx),
        start_frame=int(frame_ids[start_idx]),
        end_frame=int(frame_ids[end_idx]),
        peak_frame=int(frame_ids[peak_idx]),
        eventness_peak=float(peak_val),
        eventness_mean=float(mean_val),
    )


def _choose_high_z(cfg: TreeBuildConfig, level: int) -> float:
    if not cfg.per_level_high_z:
        return 1.0
    idx = min(max(0, int(level - 1)), len(cfg.per_level_high_z) - 1)
    return float(cfg.per_level_high_z[idx])


def _should_stop(local_scores: np.ndarray, span_len: int, cfg: TreeBuildConfig) -> bool:
    if span_len < max(int(cfg.split_min_len), int(cfg.min_span_len)):
        return True
    if local_scores.size == 0:
        return True
    peak = float(np.max(local_scores))
    mean = float(np.mean(local_scores))
    return (peak - mean) < 0.08 # FIXME: tune this threshold or make it adaptive based on cfg and level


def _detect_child_spans(local_scores: np.ndarray, cfg: TreeBuildConfig, level: int) -> list[tuple[int, int]]:
    if local_scores.size == 0:
        return []

    shrink = max(1, int(level))
    detector = BoundaryDetector(
        cfg=BoundaryDetectorConfig(
            use_savgol_filter=bool(cfg.use_savgol_filter),
            method="by_peeks",
            high_z=_choose_high_z(cfg, level),
            peak_gap=max(1, int(round(cfg.peak_gap / shrink))),
            peak_expand=(
                max(2, int(round(cfg.peak_expand[0] / shrink))),
                max(3, int(round(cfg.peak_expand[1] / shrink))),
            ),
            min_span_len=max(2, int(cfg.min_span_len)),
            merge_gap=max(0, int(round(cfg.merge_gap / shrink))),
        )
    )
    spans = detector.detect(local_scores.astype(np.float32).tolist())
    return spans if isinstance(spans, list) else []


def _split_children(
    parent: EventNode,
    frame_ids: list[int],
    eventness_scores: np.ndarray,
    cfg: TreeBuildConfig,
    level: int,
) -> list[EventNode]:
    if level > int(cfg.max_depth):
        return []

    local_scores = eventness_scores[parent.start_idx : parent.end_idx + 1]
    if _should_stop(local_scores, parent.span_len(), cfg):
        return []

    local_spans = _detect_child_spans(local_scores, cfg, level)
    if not local_spans:
        return []

    abs_spans: list[tuple[int, int]] = []
    for start_idx, end_idx in local_spans:
        abs_start = int(parent.start_idx + start_idx)
        abs_end = int(parent.start_idx + end_idx)
        if abs_end < abs_start:
            continue
        if (abs_end - abs_start + 1) < int(cfg.min_span_len):
            continue
        abs_spans.append((abs_start, abs_end))

    if not abs_spans:
        return []

    parent_cover = float(sum((end - start + 1) for start, end in abs_spans)) / max(1.0, float(parent.span_len()))
    if len(abs_spans) == 1 and parent_cover >= 0.92:
        return []

    children: list[EventNode] = []
    for child_idx, (start_idx, end_idx) in enumerate(abs_spans, start=1):
        child = _node_from_span(
            node_id=f"{parent.node_id}.{child_idx}",
            level=level,
            start_idx=start_idx,
            end_idx=end_idx,
            frame_ids=frame_ids,
            eventness_scores=eventness_scores,
        )
        child.children = _split_children(child, frame_ids, eventness_scores, cfg, level + 1)
        children.append(child)
    return children


def build_event_tree_from_root(
    frame_ids: list[int],
    eventness_scores: list[float] | np.ndarray,
    cfg: TreeBuildConfig | None = None,
) -> EventNode:
    """Recursively split an eventness curve into a coarse-to-fine event tree."""

    scores = np.asarray(eventness_scores, dtype=np.float32)
    if scores.size != len(frame_ids):
        raise ValueError("frame_ids and eventness_scores must have the same length")

    root = _node_from_span(
        node_id="root",
        level=0,
        start_idx=0,
        end_idx=len(frame_ids) - 1,
        frame_ids=frame_ids,
        eventness_scores=scores,
    )
    root.children = _split_children(root, frame_ids, scores, cfg, level=1)
    return root


def build_event_tree_from_coarse_nodes(
    coarse_nodes: list[EventNode],
    frame_ids: list[int],
    eventness_scores: list[float] | np.ndarray,
    cfg: TreeBuildConfig | None = None,
) -> EventNode:
    """Recursively split a list of coarse nodes into a coarse-to-fine event tree."""
    scores = np.asarray(eventness_scores, dtype=np.float32)
    if scores.size != len(frame_ids):
        raise ValueError("frame_ids and eventness_scores must have the same length")
    
    cfg = cfg
    root = _node_from_span(
        node_id="root",
        level=0,
        start_idx=0,
        end_idx=len(frame_ids) - 1,
        frame_ids=frame_ids,
        eventness_scores=scores,
    )
    for node in coarse_nodes:
        node.children = _split_children(node, frame_ids, np.asarray(eventness_scores, dtype=np.float32), cfg, level=2)
        root.children.append(node)
    return root 


def flatten_event_tree(root: EventNode, include_root: bool = False) -> list[EventNode]:
    out: list[EventNode] = []

    def _walk(node: EventNode) -> None:
        if include_root or node.level > 0:
            out.append(node)
        for child in node.children:
            _walk(child)

    _walk(root)
    return out
