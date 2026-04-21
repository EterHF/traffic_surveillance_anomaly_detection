from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.proposals.boundary_detector import BoundaryDetector


@dataclass
class EventTreeConfig:
    max_depth: int = 2
    min_span_len: int = 8
    split_min_len: int = 20
    peak_gap: int = 4
    peak_expand: tuple[int, int] = (4, 8)
    merge_gap: int = 4
    per_level_high_z: tuple[float, ...] = (0.9, 0.55)
    use_savgol_filter: bool = False


@dataclass
class EventNode:
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
    fused_score: float = 0.0
    children: list["EventNode"] = field(default_factory=list)

    def span_len(self) -> int:
        return int(self.end_idx - self.start_idx + 1)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "level": int(self.level),
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "peak_idx": int(self.peak_idx),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "peak_frame": int(self.peak_frame),
            "eventness_peak": float(self.eventness_peak),
            "eventness_mean": float(self.eventness_mean),
            "fused_score": float(self.fused_score),
            "children": [child.to_dict() for child in self.children],
        }


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


def _choose_high_z(cfg: EventTreeConfig, level: int) -> float:
    if not cfg.per_level_high_z:
        return 1.0
    idx = min(max(0, int(level - 1)), len(cfg.per_level_high_z) - 1)
    return float(cfg.per_level_high_z[idx])


def _should_stop(local_scores: np.ndarray, span_len: int, cfg: EventTreeConfig) -> bool:
    if span_len < max(int(cfg.split_min_len), int(cfg.min_span_len)):
        return True
    if local_scores.size == 0:
        return True
    peak = float(np.max(local_scores))
    mean = float(np.mean(local_scores))
    return (peak - mean) < 0.08


def _detect_child_spans(
    local_scores: np.ndarray,
    cfg: EventTreeConfig,
    level: int,
) -> list[tuple[int, int]]:
    if local_scores.size == 0:
        return []

    shrink = max(1, int(level))
    detector = BoundaryDetector(
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
    spans = detector.detect(local_scores.astype(np.float32).tolist())
    return spans if isinstance(spans, list) else []


def _split_children(
    parent: EventNode,
    frame_ids: list[int],
    eventness_scores: np.ndarray,
    cfg: EventTreeConfig,
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
    for s, e in local_spans:
        abs_s = int(parent.start_idx + s)
        abs_e = int(parent.start_idx + e)
        if abs_e < abs_s:
            continue
        if (abs_e - abs_s + 1) < int(cfg.min_span_len):
            continue
        abs_spans.append((abs_s, abs_e))

    if not abs_spans:
        return []

    parent_cover = float(sum((e - s + 1) for s, e in abs_spans)) / max(1.0, float(parent.span_len()))
    if len(abs_spans) == 1 and parent_cover >= 0.92:
        return []

    children: list[EventNode] = []
    for child_idx, (s, e) in enumerate(abs_spans, start=1):
        child = _node_from_span(
            node_id=f"{parent.node_id}.{child_idx}",
            level=level,
            start_idx=s,
            end_idx=e,
            frame_ids=frame_ids,
            eventness_scores=eventness_scores,
        )
        child.children = _split_children(child, frame_ids, eventness_scores, cfg, level + 1)
        children.append(child)
    return children


def build_event_tree(
    frame_ids: list[int],
    eventness_scores: list[float] | np.ndarray,
    cfg: EventTreeConfig | None = None,
) -> EventNode:
    cfg = cfg or EventTreeConfig()
    if not frame_ids:
        return EventNode(
            node_id="root",
            level=0,
            start_idx=0,
            end_idx=0,
            peak_idx=0,
            start_frame=0,
            end_frame=0,
            peak_frame=0,
            eventness_peak=0.0,
            eventness_mean=0.0,
        )

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


def flatten_event_tree(root: EventNode, include_root: bool = False) -> list[EventNode]:
    out: list[EventNode] = []

    def _walk(node: EventNode) -> None:
        if include_root or node.level > 0:
            out.append(node)
        for child in node.children:
            _walk(child)

    _walk(root)
    return out
