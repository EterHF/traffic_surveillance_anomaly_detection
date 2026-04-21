from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.evidence.coarse_router_evidence import CoarseEvidencePack, build_coarse_evidence
from src.proposals.adaptive_fine_tree import AdaptiveFineTreeConfig, build_fine_nodes_for_proposal
from src.proposals.coarse_proposal import CoarseProposal, CoarseProposalConfig, build_coarse_proposals
from src.proposals.event_tree import EventNode
from src.proposals.node_selector import NodeSelectorConfig, select_nodes_from_list
from src.schemas import TrackObject
from src.vlm.coarse_router import CoarseRouteDecision, CoarseRouter, CoarseRouterConfig


@dataclass
class RoutedTreeResult:
    coarse_proposals: list[CoarseProposal] = field(default_factory=list)
    route_evidences: dict[str, CoarseEvidencePack] = field(default_factory=dict)
    route_decisions: dict[str, CoarseRouteDecision] = field(default_factory=dict)
    fine_nodes: list[EventNode] = field(default_factory=list)
    selected_nodes: list[EventNode] = field(default_factory=list)


def build_routed_event_tree(
    frame_ids: list[int],
    images: list[np.ndarray],
    tracks_per_frame: list[list[TrackObject]],
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    seed_scores: list[float] | np.ndarray,
    output_dir: str,
    coarse_cfg: CoarseProposalConfig | None = None,
    router_cfg: CoarseRouterConfig | None = None,
    fine_cfg: AdaptiveFineTreeConfig | None = None,
    selector_cfg: NodeSelectorConfig | None = None,
) -> RoutedTreeResult:
    coarse_cfg = coarse_cfg or CoarseProposalConfig()
    router_cfg = router_cfg or CoarseRouterConfig()
    fine_cfg = fine_cfg or AdaptiveFineTreeConfig()
    selector_cfg = selector_cfg or NodeSelectorConfig()

    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)
    seed_arr = np.asarray(seed_scores, dtype=np.float32)

    proposals = build_coarse_proposals(
        frame_ids=frame_ids,
        seed_scores=seed_arr,
        track_scores=track_arr,
        object_scores=object_arr,
        cfg=coarse_cfg,
    )

    evidences: dict[str, CoarseEvidencePack] = {}
    for proposal in proposals:
        evidences[proposal.proposal_id] = build_coarse_evidence(
            proposal=proposal,
            images=images,
            frame_ids=frame_ids,
            tracks_per_frame=tracks_per_frame,
            output_dir=output_dir,
        )

    decisions = CoarseRouter(router_cfg).route(proposals, evidences)

    fine_nodes: list[EventNode] = []
    for proposal in proposals:
        decision = decisions.get(proposal.proposal_id)
        if decision is None:
            continue
        fine_nodes.extend(
            build_fine_nodes_for_proposal(
                proposal=proposal,
                decision=decision,
                frame_ids=frame_ids,
                seed_scores=seed_arr,
                cfg=fine_cfg,
            )
        )

    selected_nodes = select_nodes_from_list(
        fine_nodes,
        eventness_scores=seed_arr,
        track_risk_scores=track_arr,
        object_prior_scores=object_arr,
        cfg=selector_cfg,
    )
    return RoutedTreeResult(
        coarse_proposals=proposals,
        route_evidences=evidences,
        route_decisions=decisions,
        fine_nodes=fine_nodes,
        selected_nodes=selected_nodes,
    )
