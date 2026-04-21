from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.proposals.coarse_proposal import CoarseProposal
from src.proposals.event_tree import EventNode, EventTreeConfig, build_event_tree, flatten_event_tree, make_event_node_from_span
from src.vlm.coarse_router import CoarseRouteDecision


@dataclass
class AdaptiveFineTreeConfig:
    min_output_span: int = 8
    focus_region_pad: float = 0.10
    localized_high_z: tuple[float, ...] = (0.8, 0.45)
    localized_split_min_len: int = 12
    localized_peak_gap: int = 3
    localized_peak_expand: tuple[int, int] = (2, 5)
    localized_merge_gap: int = 3
    multi_stage_high_z: tuple[float, ...] = (0.75, 0.42)
    multi_stage_split_min_len: int = 10
    multi_stage_peak_gap: int = 3
    multi_stage_peak_expand: tuple[int, int] = (2, 4)
    multi_stage_merge_gap: int = 2
    uncertain_high_z: tuple[float, ...] = (0.9, 0.55)
    uncertain_split_min_len: int = 16
    uncertain_peak_gap: int = 4
    uncertain_peak_expand: tuple[int, int] = (4, 8)
    uncertain_merge_gap: int = 4


def _clamp_span(start_idx: int, end_idx: int, total_len: int) -> tuple[int, int]:
    if total_len <= 0:
        return 0, 0
    start_idx = min(max(0, int(start_idx)), total_len - 1)
    end_idx = min(max(int(start_idx), int(end_idx)), total_len - 1)
    return start_idx, end_idx


def _apply_boundary_adjust(
    proposal: CoarseProposal,
    decision: CoarseRouteDecision,
    total_len: int,
) -> tuple[int, int]:
    start_idx = int(proposal.start_idx + decision.boundary_adjust[0])
    end_idx = int(proposal.end_idx + decision.boundary_adjust[1])
    return _clamp_span(start_idx, end_idx, total_len)


def _focus_subspan(
    start_idx: int,
    end_idx: int,
    focus_region: tuple[float, float],
    pad: float,
    total_len: int,
) -> tuple[int, int]:
    span_len = max(1, end_idx - start_idx + 1)
    focus_start = max(0.0, min(1.0, float(focus_region[0]) - float(pad)))
    focus_end = max(focus_start, min(1.0, float(focus_region[1]) + float(pad)))
    sub_start = start_idx + int(round(focus_start * span_len))
    sub_end = start_idx + int(round(focus_end * span_len))
    return _clamp_span(sub_start, sub_end, total_len)


def _cfg_for_decision(cfg: AdaptiveFineTreeConfig, decision: CoarseRouteDecision) -> EventTreeConfig:
    if decision.temporal_pattern == "localized":
        return EventTreeConfig(
            max_depth=2,
            min_span_len=int(cfg.min_output_span),
            split_min_len=int(cfg.localized_split_min_len),
            peak_gap=int(cfg.localized_peak_gap),
            peak_expand=tuple(int(v) for v in cfg.localized_peak_expand),
            merge_gap=int(cfg.localized_merge_gap),
            per_level_high_z=tuple(float(v) for v in cfg.localized_high_z),
            use_savgol_filter=False,
        )
    if decision.temporal_pattern == "multi_stage":
        return EventTreeConfig(
            max_depth=2,
            min_span_len=int(cfg.min_output_span),
            split_min_len=int(cfg.multi_stage_split_min_len),
            peak_gap=int(cfg.multi_stage_peak_gap),
            peak_expand=tuple(int(v) for v in cfg.multi_stage_peak_expand),
            merge_gap=int(cfg.multi_stage_merge_gap),
            per_level_high_z=tuple(float(v) for v in cfg.multi_stage_high_z),
            use_savgol_filter=False,
        )
    return EventTreeConfig(
        max_depth=1,
        min_span_len=int(cfg.min_output_span),
        split_min_len=int(cfg.uncertain_split_min_len),
        peak_gap=int(cfg.uncertain_peak_gap),
        peak_expand=tuple(int(v) for v in cfg.uncertain_peak_expand),
        merge_gap=int(cfg.uncertain_merge_gap),
        per_level_high_z=tuple(float(v) for v in cfg.uncertain_high_z),
        use_savgol_filter=False,
    )


def _annotate_nodes(nodes: list[EventNode], decision: CoarseRouteDecision) -> list[EventNode]:
    for node in nodes:
        node.route_role = str(decision.role)
        node.route_temporal_pattern = str(decision.temporal_pattern)
        node.route_confidence = float(decision.confidence)
    return nodes


def build_fine_nodes_for_proposal(
    proposal: CoarseProposal,
    decision: CoarseRouteDecision,
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    cfg: AdaptiveFineTreeConfig | None = None,
) -> list[EventNode]:
    cfg = cfg or AdaptiveFineTreeConfig()
    total_len = len(frame_ids)
    seed_arr = np.asarray(seed_scores, dtype=np.float32)
    if total_len == 0 or seed_arr.size == 0:
        return []

    if not bool(decision.contains_core_anomaly) and str(decision.role) == "context":
        return []

    adjusted_start, adjusted_end = _apply_boundary_adjust(proposal, decision, total_len)
    if decision.temporal_pattern == "sustained" or decision.split_recommendation == "keep_coarse":
        node = make_event_node_from_span(
            node_id=f"{proposal.proposal_id}.coarse",
            level=1,
            start_idx=adjusted_start,
            end_idx=adjusted_end,
            frame_ids=frame_ids,
            eventness_scores=seed_arr,
        )
        return _annotate_nodes([node], decision)

    focus_start, focus_end = _focus_subspan(
        adjusted_start,
        adjusted_end,
        decision.focus_region,
        pad=float(cfg.focus_region_pad),
        total_len=total_len,
    )
    local_cfg = _cfg_for_decision(cfg, decision)
    local_frame_ids = frame_ids[focus_start : focus_end + 1]
    local_seed = seed_arr[focus_start : focus_end + 1]
    if len(local_frame_ids) < int(cfg.min_output_span):
        node = make_event_node_from_span(
            node_id=f"{proposal.proposal_id}.focus",
            level=1,
            start_idx=focus_start,
            end_idx=focus_end,
            frame_ids=frame_ids,
            eventness_scores=seed_arr,
        )
        return _annotate_nodes([node], decision)

    root = build_event_tree(local_frame_ids, local_seed, cfg=local_cfg)
    local_nodes = flatten_event_tree(root, include_root=False)

    global_nodes: list[EventNode] = []
    for idx, node in enumerate(local_nodes, start=1):
        start_idx = int(focus_start + node.start_idx)
        end_idx = int(focus_start + node.end_idx)
        if (end_idx - start_idx + 1) < int(cfg.min_output_span):
            continue
        global_nodes.append(
            make_event_node_from_span(
                node_id=f"{proposal.proposal_id}.fine.{idx}",
                level=1,
                start_idx=start_idx,
                end_idx=end_idx,
                frame_ids=frame_ids,
                eventness_scores=seed_arr,
            )
        )

    if not global_nodes:
        fallback = make_event_node_from_span(
            node_id=f"{proposal.proposal_id}.focus",
            level=1,
            start_idx=focus_start,
            end_idx=focus_end,
            frame_ids=frame_ids,
            eventness_scores=seed_arr,
        )
        global_nodes = [fallback]

    return _annotate_nodes(global_nodes, decision)
