from __future__ import annotations

import numpy as np

from src.evidence.coarse_router_evidence import CoarseEvidencePack
from src.proposals.adaptive_fine_tree import AdaptiveFineTreeConfig, build_fine_nodes_for_proposal
from src.proposals.coarse_proposal import CoarseProposal, CoarseProposalConfig, build_coarse_proposals
from src.proposals.node_selector import NodeSelectorConfig
from src.proposals.routed_tree import build_routed_event_tree
from src.vlm.coarse_router import CoarseRouteDecision, CoarseRouter, CoarseRouterConfig


def test_coarse_proposals_merge_interval_and_tree_sources():
    frame_ids = list(range(40))
    seed = np.zeros((40,), dtype=np.float32)
    seed[10:16] = [0.2, 0.5, 0.9, 1.0, 0.7, 0.3]
    track = seed.copy()
    obj = 0.4 * seed

    proposals = build_coarse_proposals(
        frame_ids=frame_ids,
        seed_scores=seed,
        track_scores=track,
        object_scores=obj,
        cfg=CoarseProposalConfig(
            interval_high_z=0.3,
            interval_peak_gap=3,
            interval_peak_expand=(2, 4),
            interval_min_span_len=4,
            interval_merge_gap=2,
            tree_max_depth=1,
            tree_min_span_len=4,
            tree_split_min_len=8,
            tree_peak_gap=3,
            tree_peak_expand=(2, 4),
            tree_merge_gap=2,
            tree_high_z=(0.3,),
            merge_iou=0.2,
            merge_center_gap=8,
            max_proposals=4,
        ),
    )

    assert proposals
    best = proposals[0]
    assert best.start_frame <= 12 <= best.end_frame
    assert "interval" in best.sources or "tree" in best.sources


def test_coarse_router_heuristic_outputs_localized_route():
    proposal = CoarseProposal(
        proposal_id="coarse.1",
        start_idx=10,
        end_idx=20,
        peak_idx=14,
        start_frame=10,
        end_frame=20,
        peak_frame=14,
        sources=["interval", "tree"],
        coarse_score=0.8,
        seed_peak=1.0,
        seed_mean=0.4,
        track_peak=0.9,
        object_peak=0.5,
    )
    evidence = CoarseEvidencePack(
        proposal_id="coarse.1",
        summary={
            "span_ratio": 0.22,
            "peak_offset_ratio": 0.35,
            "sources": ["interval", "tree"],
            "seed_peak": 1.0,
            "seed_mean": 0.4,
        },
    )
    router = CoarseRouter(CoarseRouterConfig(enabled=False, mode="heuristic", max_nodes=2))
    decision = router.route([proposal], {"coarse.1": evidence})["coarse.1"]

    assert decision.role == "core"
    assert decision.temporal_pattern in {"localized", "multi_stage"}
    assert decision.split_recommendation in {"split_2", "split_3"}


def test_adaptive_fine_tree_respects_context_pruning_and_route_annotations():
    frame_ids = list(range(30))
    seed = np.zeros((30,), dtype=np.float32)
    seed[8:15] = [0.2, 0.5, 0.9, 1.0, 0.6, 0.3, 0.1]
    proposal = CoarseProposal(
        proposal_id="coarse.2",
        start_idx=6,
        end_idx=18,
        peak_idx=11,
        start_frame=6,
        end_frame=18,
        peak_frame=11,
        sources=["interval"],
        coarse_score=0.75,
        seed_peak=1.0,
        seed_mean=0.45,
        track_peak=0.9,
        object_peak=0.4,
    )

    context_nodes = build_fine_nodes_for_proposal(
        proposal,
        CoarseRouteDecision(
            proposal_id="coarse.2",
            contains_core_anomaly=False,
            role="context",
            temporal_pattern="uncertain",
            focus_region=(0.15, 0.85),
            split_recommendation="keep_coarse",
            boundary_adjust=(0, 0),
            confidence=0.5,
            reason="context",
        ),
        frame_ids=frame_ids,
        seed_scores=seed,
    )
    assert context_nodes == []

    fine_nodes = build_fine_nodes_for_proposal(
        proposal,
        CoarseRouteDecision(
            proposal_id="coarse.2",
            contains_core_anomaly=True,
            role="core",
            temporal_pattern="localized",
            focus_region=(0.2, 0.7),
            split_recommendation="split_2",
            boundary_adjust=(0, 0),
            confidence=0.8,
            reason="localized",
        ),
        frame_ids=frame_ids,
        seed_scores=seed,
        cfg=AdaptiveFineTreeConfig(min_output_span=4),
    )

    assert fine_nodes
    assert all(node.route_role == "core" for node in fine_nodes)
    assert any(8 <= node.peak_frame <= 13 for node in fine_nodes)


def test_routed_tree_builds_selected_nodes_on_synthetic_clip(tmp_path):
    frame_ids = list(range(36))
    images = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in frame_ids]
    tracks_per_frame = [[] for _ in frame_ids]
    track = np.zeros((36,), dtype=np.float32)
    track[12:19] = [0.1, 0.4, 0.8, 1.0, 0.9, 0.5, 0.2]
    obj = np.zeros((36,), dtype=np.float32)
    obj[13:18] = [0.05, 0.2, 0.35, 0.3, 0.1]
    seed = np.clip(track + 0.5 * obj, 0.0, 1.0)

    result = build_routed_event_tree(
        frame_ids=frame_ids,
        images=images,
        tracks_per_frame=tracks_per_frame,
        track_scores=track,
        object_scores=obj,
        seed_scores=seed,
        output_dir=str(tmp_path / "coarse"),
        coarse_cfg=CoarseProposalConfig(
            interval_high_z=0.3,
            interval_peak_gap=3,
            interval_peak_expand=(2, 4),
            interval_min_span_len=4,
            interval_merge_gap=2,
            tree_max_depth=1,
            tree_min_span_len=4,
            tree_split_min_len=8,
            tree_peak_gap=3,
            tree_peak_expand=(2, 4),
            tree_merge_gap=2,
            tree_high_z=(0.3,),
            merge_iou=0.2,
            merge_center_gap=8,
            max_proposals=4,
        ),
        router_cfg=CoarseRouterConfig(enabled=False, mode="heuristic", max_nodes=2),
        selector_cfg=NodeSelectorConfig(top_k=3, min_node_len=4, overlap_iou=0.9, gate_floor=0.4, reference_span=24),
    )

    assert result.coarse_proposals
    assert result.fine_nodes
    assert result.selected_nodes
    assert any(12 <= node.peak_frame <= 18 for node in result.selected_nodes)
