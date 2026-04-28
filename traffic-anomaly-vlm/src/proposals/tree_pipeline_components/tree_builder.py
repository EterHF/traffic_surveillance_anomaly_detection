from __future__ import annotations

import numpy as np

from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from src.schemas import EventNode

from .config import TreeBuildConfig


def _node_from_span(
    node_id: str,
    level: int,
    start_idx: int,
    end_idx: int,
    frame_ids: list[int],
    eventness_scores: np.ndarray | list[float],
) -> EventNode:
    scores = np.asarray(eventness_scores, dtype=np.float32)
    local = scores[start_idx : end_idx + 1]
    if local.size == 0:
        peak_idx = int(start_idx)
        peak_val = 0.0
        mean_val = 0.0
    else:
        rel_peak = int(np.argmax(local))
        peak_idx = int(start_idx + rel_peak)
        peak_val = float(local[rel_peak])
        mean_val = float(np.mean(local))

    return EventNode(
        node_id=str(node_id),
        level=int(level),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        peak_idx=int(peak_idx),
        start_frame=int(frame_ids[start_idx]),
        end_frame=int(frame_ids[end_idx]),
        peak_frame=int(frame_ids[peak_idx]),
        eventness_peak=float(peak_val),
        eventness_mean=float(mean_val),
    )


def _node_seed_score(node: EventNode) -> float:
    return float(
        max(
            float(node.fused_score),
            float(node.vlm_score),
            float(node.span_prior_score),
            0.7 * float(node.eventness_peak) + 0.3 * float(node.eventness_mean),
        )
    )


def _leaf_detector_cfg(cfg: TreeBuildConfig) -> BoundaryDetectorConfig:
    high_z = min(tuple(float(v) for v in cfg.per_level_high_z)) if cfg.per_level_high_z else 1.0
    return BoundaryDetectorConfig(
        high_z=float(high_z),
        peak_gap=max(1, int(cfg.peak_gap)),
        peak_expand=tuple(int(v) for v in cfg.peak_expand),
        merge_gap=0,
    )


def _build_leaf_nodes(
    frame_ids: list[int],
    eventness_scores: np.ndarray,
    cfg: TreeBuildConfig,
) -> list[EventNode]:
    spans = BoundaryDetector(_leaf_detector_cfg(cfg)).detect(eventness_scores.astype(np.float32).tolist())
    min_keep = max(int(cfg.min_span_len), int(cfg.min_output_span))
    leaves: list[EventNode] = []
    for index, (start_idx, end_idx) in enumerate(spans, start=1):
        if (int(end_idx) - int(start_idx) + 1) < min_keep:
            continue
        node = _node_from_span(
            node_id=f"leaf.{index}",
            level=0,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            frame_ids=frame_ids,
            eventness_scores=eventness_scores,
        )
        node.span_prior_score = 0.7 * float(node.eventness_peak) + 0.3 * float(node.eventness_mean)
        leaves.append(node)
    return leaves


def _normalize_leaf_nodes(
    coarse_nodes: list[EventNode],
    frame_ids: list[int],
    eventness_scores: np.ndarray,
) -> list[EventNode]:
    leaves: list[EventNode] = []
    for index, node in enumerate(sorted(coarse_nodes, key=lambda item: (int(item.start_idx), int(item.end_idx))), start=1):
        leaf = _node_from_span(
            node_id=str(node.node_id or f"leaf.{index}"),
            level=0,
            start_idx=int(node.start_idx),
            end_idx=int(node.end_idx),
            frame_ids=frame_ids,
            eventness_scores=eventness_scores,
        )
        leaf.span_prior_score = float(
            max(
                float(node.span_prior_score),
                float(node.fused_score),
                0.7 * float(leaf.eventness_peak) + 0.3 * float(leaf.eventness_mean),
            )
        )
        leaf.vlm_score = float(node.vlm_score)
        leaf.vlm_confidence = float(node.vlm_confidence)
        leaf.fused_score = float(node.fused_score)
        leaf.children = list(node.children)
        leaves.append(leaf)
    return leaves


def _group_adjacent_nodes(nodes: list[EventNode], gap_limit: int) -> list[list[EventNode]]:
    if not nodes:
        return []
    groups: list[list[EventNode]] = [[nodes[0]]]
    for node in nodes[1:]:
        prev = groups[-1][-1]
        gap = int(node.start_idx) - int(prev.end_idx) - 1
        if gap <= int(gap_limit):
            groups[-1].append(node)
        else:
            groups.append([node])
    return groups


def _build_parent_node(
    leaves: list[EventNode],
    frame_ids: list[int],
    eventness_scores: np.ndarray,
    parent_id: str,
) -> EventNode:
    start_idx = min(int(node.start_idx) for node in leaves)
    end_idx = max(int(node.end_idx) for node in leaves)
    parent = _node_from_span(
        node_id=parent_id,
        level=1,
        start_idx=start_idx,
        end_idx=end_idx,
        frame_ids=frame_ids,
        eventness_scores=eventness_scores,
    )
    parent.children = list(leaves)
    parent.span_prior_score = float(
        max(
            0.7 * float(parent.eventness_peak) + 0.3 * float(parent.eventness_mean),
            max(_node_seed_score(node) for node in leaves),
        )
    )
    parent.vlm_score = float(max(float(node.vlm_score) for node in leaves))
    parent.vlm_confidence = float(max(float(node.vlm_confidence) for node in leaves))
    parent.fused_score = float(max(_node_seed_score(node) for node in leaves))
    return parent


def _assign_levels(node: EventNode, level: int = 0) -> None:
    node.level = int(level)
    for child in node.children:
        _assign_levels(child, level + 1)


def build_event_tree_from_root(
    frame_ids: list[int],
    eventness_scores: list[float] | np.ndarray,
    cfg: TreeBuildConfig | None = None,
) -> EventNode:
    cfg = cfg or TreeBuildConfig()
    scores = np.asarray(eventness_scores, dtype=np.float32)
    if scores.size != len(frame_ids):
        raise ValueError("frame_ids and eventness_scores must have the same length")
    leaves = _build_leaf_nodes(frame_ids, scores, cfg)
    return build_event_tree_from_coarse_nodes(leaves, frame_ids, scores, cfg)


def build_event_tree_from_coarse_nodes(
    coarse_nodes: list[EventNode],
    frame_ids: list[int],
    eventness_scores: list[float] | np.ndarray,
    cfg: TreeBuildConfig | None = None,
) -> EventNode:
    """Build a minimal bottom-up event tree from positive leaf nodes."""

    cfg = cfg or TreeBuildConfig()
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
    if not coarse_nodes:
        return root

    leaves = _normalize_leaf_nodes(coarse_nodes, frame_ids, scores)
    gap_limit = max(0, int(cfg.merge_gap))
    groups = _group_adjacent_nodes(leaves, gap_limit=gap_limit)
    parents: list[EventNode] = []
    for parent_index, group in enumerate(groups, start=1):
        parents.append(
            _build_parent_node(
                leaves=group,
                frame_ids=frame_ids,
                eventness_scores=scores,
                parent_id=f"event.{parent_index}",
            )
        )

    root.children = parents
    _assign_levels(root, level=0)
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
