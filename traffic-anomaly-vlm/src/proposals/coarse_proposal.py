from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.proposals.boundary_detector import BoundaryDetector
from src.proposals.event_tree import EventTreeConfig, EventNode, build_event_tree, flatten_event_tree


@dataclass
class CoarseProposalConfig:
    interval_high_z: float = 0.85
    interval_peak_gap: int = 4
    interval_peak_expand: tuple[int, int] = (10, 18)
    interval_min_span_len: int = 12
    interval_merge_gap: int = 10
    tree_max_depth: int = 1
    tree_min_span_len: int = 8
    tree_split_min_len: int = 20
    tree_peak_gap: int = 4
    tree_peak_expand: tuple[int, int] = (4, 8)
    tree_merge_gap: int = 4
    tree_high_z: tuple[float, ...] = (0.9,)
    merge_iou: float = 0.40
    merge_center_gap: int = 20
    max_proposals: int = 8


@dataclass
class CoarseProposal:
    proposal_id: str
    start_idx: int
    end_idx: int
    peak_idx: int
    start_frame: int
    end_frame: int
    peak_frame: int
    sources: list[str] = field(default_factory=list)
    coarse_score: float = 0.0
    seed_peak: float = 0.0
    seed_mean: float = 0.0
    track_peak: float = 0.0
    object_peak: float = 0.0

    def span_len(self) -> int:
        return int(self.end_idx - self.start_idx + 1)

    def center_idx(self) -> float:
        return 0.5 * float(self.start_idx + self.end_idx)

    def span_ratio(self, total_len: int) -> float:
        return float(self.span_len() / max(1, int(total_len)))

    def peak_offset_ratio(self) -> float:
        span = max(1, self.span_len())
        return float((self.peak_idx - self.start_idx) / span)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "peak_idx": int(self.peak_idx),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "peak_frame": int(self.peak_frame),
            "sources": list(self.sources),
            "coarse_score": float(self.coarse_score),
            "seed_peak": float(self.seed_peak),
            "seed_mean": float(self.seed_mean),
            "track_peak": float(self.track_peak),
            "object_peak": float(self.object_peak),
        }


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


def _peak_idx_for_span(seed_scores: np.ndarray, start_idx: int, end_idx: int) -> int:
    local = seed_scores[start_idx : end_idx + 1]
    if local.size == 0:
        return int(start_idx)
    rel_peak = int(np.argmax(local))
    return int(start_idx + rel_peak)


def _make_proposal(
    proposal_id: str,
    start_idx: int,
    end_idx: int,
    frame_ids: list[int],
    seed_scores: np.ndarray,
    track_scores: np.ndarray,
    object_scores: np.ndarray,
    source_name: str,
) -> CoarseProposal:
    peak_idx = _peak_idx_for_span(seed_scores, start_idx, end_idx)
    seed_seg = seed_scores[start_idx : end_idx + 1]
    track_seg = track_scores[start_idx : end_idx + 1]
    object_seg = object_scores[start_idx : end_idx + 1]
    seed_peak = float(np.max(seed_seg)) if seed_seg.size else 0.0
    seed_mean = float(_top_mean(seed_seg, ratio=0.25)) if seed_seg.size else 0.0
    track_peak = float(np.max(track_seg)) if track_seg.size else 0.0
    object_peak = float(np.max(object_seg)) if object_seg.size else 0.0
    coarse_score = float(0.6 * seed_peak + 0.25 * seed_mean + 0.15 * max(track_peak, object_peak))
    return CoarseProposal(
        proposal_id=proposal_id,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        peak_idx=int(peak_idx),
        start_frame=int(frame_ids[start_idx]),
        end_frame=int(frame_ids[end_idx]),
        peak_frame=int(frame_ids[peak_idx]),
        sources=[source_name],
        coarse_score=coarse_score,
        seed_peak=seed_peak,
        seed_mean=seed_mean,
        track_peak=track_peak,
        object_peak=object_peak,
    )


def build_interval_proposals(
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    cfg: CoarseProposalConfig | None = None,
) -> list[CoarseProposal]:
    cfg = cfg or CoarseProposalConfig()
    seed_arr = np.asarray(seed_scores, dtype=np.float32)
    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)

    detector = BoundaryDetector(
        use_savgol_filter=False,
        method="by_peeks",
        high_z=float(cfg.interval_high_z),
        peak_gap=int(cfg.interval_peak_gap),
        peak_expand=tuple(int(v) for v in cfg.interval_peak_expand),
        min_span_len=int(cfg.interval_min_span_len),
        merge_gap=int(cfg.interval_merge_gap),
    )
    spans = detector.detect(seed_arr.astype(np.float32).tolist())
    if not isinstance(spans, list):
        return []

    proposals: list[CoarseProposal] = []
    for idx, (start_idx, end_idx) in enumerate(spans, start=1):
        if end_idx < start_idx:
            continue
        proposals.append(
            _make_proposal(
                proposal_id=f"coarse.interval.{idx}",
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                frame_ids=frame_ids,
                seed_scores=seed_arr,
                track_scores=track_arr,
                object_scores=object_arr,
                source_name="interval",
            )
        )
    return proposals


def build_tree_proposals(
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    cfg: CoarseProposalConfig | None = None,
) -> list[CoarseProposal]:
    cfg = cfg or CoarseProposalConfig()
    seed_arr = np.asarray(seed_scores, dtype=np.float32)
    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)

    tree_cfg = EventTreeConfig(
        max_depth=int(cfg.tree_max_depth),
        min_span_len=int(cfg.tree_min_span_len),
        split_min_len=int(cfg.tree_split_min_len),
        peak_gap=int(cfg.tree_peak_gap),
        peak_expand=tuple(int(v) for v in cfg.tree_peak_expand),
        merge_gap=int(cfg.tree_merge_gap),
        per_level_high_z=tuple(float(v) for v in cfg.tree_high_z),
        use_savgol_filter=False,
    )
    root = build_event_tree(frame_ids=frame_ids, eventness_scores=seed_arr, cfg=tree_cfg)
    nodes = [node for node in flatten_event_tree(root, include_root=False) if int(node.level) == 1]

    proposals: list[CoarseProposal] = []
    for idx, node in enumerate(nodes, start=1):
        proposals.append(
            _make_proposal(
                proposal_id=f"coarse.tree.{idx}",
                start_idx=int(node.start_idx),
                end_idx=int(node.end_idx),
                frame_ids=frame_ids,
                seed_scores=seed_arr,
                track_scores=track_arr,
                object_scores=object_arr,
                source_name="tree",
            )
        )
    return proposals


def merge_coarse_proposals(
    proposals: list[CoarseProposal],
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    cfg: CoarseProposalConfig | None = None,
) -> list[CoarseProposal]:
    cfg = cfg or CoarseProposalConfig()
    seed_arr = np.asarray(seed_scores, dtype=np.float32)
    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)

    if not proposals:
        return []

    ordered = sorted(proposals, key=lambda p: (int(p.start_idx), int(p.end_idx)))
    merged: list[CoarseProposal] = []
    for proposal in ordered:
        if not merged:
            merged.append(proposal)
            continue

        prev = merged[-1]
        iou = _interval_iou((prev.start_idx, prev.end_idx), (proposal.start_idx, proposal.end_idx))
        center_gap = abs(prev.center_idx() - proposal.center_idx())
        if iou < float(cfg.merge_iou) and center_gap > float(cfg.merge_center_gap):
            merged.append(proposal)
            continue

        start_idx = min(prev.start_idx, proposal.start_idx)
        end_idx = max(prev.end_idx, proposal.end_idx)
        combined = _make_proposal(
            proposal_id=prev.proposal_id,
            start_idx=start_idx,
            end_idx=end_idx,
            frame_ids=frame_ids,
            seed_scores=seed_arr,
            track_scores=track_arr,
            object_scores=object_arr,
            source_name="merged",
        )
        combined.sources = sorted(set(list(prev.sources) + list(proposal.sources)))
        merged[-1] = combined

    merged = sorted(
        merged,
        key=lambda p: (float(p.coarse_score), -float(p.span_len()), -float(len(p.sources))),
        reverse=True,
    )
    if int(cfg.max_proposals) > 0:
        merged = merged[: int(cfg.max_proposals)]

    out: list[CoarseProposal] = []
    for idx, proposal in enumerate(merged, start=1):
        proposal.proposal_id = f"coarse.{idx}"
        out.append(proposal)
    return out


def build_coarse_proposals(
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    cfg: CoarseProposalConfig | None = None,
) -> list[CoarseProposal]:
    cfg = cfg or CoarseProposalConfig()
    interval_props = build_interval_proposals(frame_ids, seed_scores, track_scores, object_scores, cfg=cfg)
    tree_props = build_tree_proposals(frame_ids, seed_scores, track_scores, object_scores, cfg=cfg)
    proposals = interval_props + tree_props
    return merge_coarse_proposals(
        proposals,
        frame_ids=frame_ids,
        seed_scores=seed_scores,
        track_scores=track_scores,
        object_scores=object_scores,
        cfg=cfg,
    )
