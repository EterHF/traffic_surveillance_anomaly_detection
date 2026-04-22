from __future__ import annotations

from typing import Any

import numpy as np

from src.evidence.builder import build_span_evidence

from .config import NodeSelectorConfig, TreeBuildConfig, TreePipelineConfig, VLMRefineConfig
from .models import EventNode, SpanScore, TreePipelineResult
from .refiner import SpanVLMRefiner, router_uses_vlm
from .selector import select_nodes_from_list
from .tree_builder import build_tree_nodes_from_span, coarse_spans_as_nodes, make_event_node_from_span, set_node_score


def _bootstrap_coarse_nodes(frame_ids: list[int], seed_arr: np.ndarray) -> list[EventNode]:
    """Temporary coarse bootstrap: use full clip as one seed node."""

    if not frame_ids:
        return []
    node = make_event_node_from_span(
        node_id="coarse.1",
        level=1,
        start_idx=0,
        end_idx=max(0, len(frame_ids) - 1),
        frame_ids=frame_ids,
        eventness_scores=seed_arr,
    )
    node.span_prior_score = float(node.eventness_peak)
    return [node]


def _decision_fused_score(decision: SpanScore | None) -> float:
    return float(decision.fused_score) if decision is not None else 0.0


def _select_top_coarse_spans(
    coarse_spans: list[EventNode],
    coarse_scores: dict[str, SpanScore],
    cfg: TreePipelineConfig,
) -> list[EventNode]:
    ranked = sorted(
        coarse_spans,
        key=lambda span: _decision_fused_score(coarse_scores.get(span.proposal_id)),
        reverse=True,
    )
    selected = [
        span
        for span in ranked
        if _decision_fused_score(coarse_scores.get(span.proposal_id)) >= float(cfg.coarse_min_fused_score)
    ]
    if not selected and ranked:
        selected = [ranked[0]]
    if int(cfg.coarse_keep_top_k) > 0:
        selected = selected[: int(cfg.coarse_keep_top_k)]
    return selected


def _apply_span_scores(nodes: list[EventNode], decisions: dict[str, SpanScore]) -> None:
    for node in nodes:
        decision = decisions.get(node.node_id)
        if decision is None:
            continue
        set_node_score(node, decision, use_local_prior=False)


def _build_candidate_nodes(
    selected_coarse_spans: list[EventNode],
    coarse_scores: dict[str, SpanScore],
    frame_ids: list[int],
    seed_arr: np.ndarray,
    enable_tree: bool,
    tree_cfg: TreeBuildConfig,
) -> list[EventNode]:
    if not enable_tree:
        return coarse_spans_as_nodes(selected_coarse_spans, coarse_scores, frame_ids, seed_arr)

    candidate_nodes: list[EventNode] = []
    for coarse_span in selected_coarse_spans:
        coarse_score = coarse_scores.get(coarse_span.proposal_id)
        if coarse_score is None:
            continue
        candidate_nodes.extend(
            build_tree_nodes_from_span(
                coarse_span=coarse_span,
                coarse_score=coarse_score,
                frame_ids=frame_ids,
                seed_scores=seed_arr,
                cfg=tree_cfg,
            )
        )
    return candidate_nodes


def run_tree_pipeline(
    frame_ids: list[int],
    images: list[np.ndarray],
    tracks_per_frame: list[list[Any]],
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    seed_scores: list[float] | np.ndarray,
    output_dir: str,
    pipeline_cfg: TreePipelineConfig | None = None,
    tree_cfg: TreeBuildConfig | None = None,
    selector_cfg: NodeSelectorConfig | None = None,
    router_cfg: VLMRefineConfig | None = None,
) -> TreePipelineResult:
    """Minimal pipeline wrapper that keeps tree components runnable.

    Coarse mining logic is intentionally deferred; for now we bootstrap one coarse node
    from the full clip and focus on tree/refine/select flow.
    """

    del tracks_per_frame
    pipeline_cfg = pipeline_cfg or TreePipelineConfig()
    tree_cfg = tree_cfg or TreeBuildConfig()
    selector_cfg = selector_cfg or NodeSelectorConfig()
    router_cfg = router_cfg or VLMRefineConfig()

    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)
    seed_arr = np.asarray(seed_scores, dtype=np.float32)

    coarse_spans = _bootstrap_coarse_nodes(frame_ids=frame_ids, seed_arr=seed_arr)
    coarse_evidences = {
        span.proposal_id: build_span_evidence(
            span=span,
            images=images,
            frame_ids=frame_ids,
            output_dir=f"{output_dir}/coarse",
        )
        for span in coarse_spans
    }
    refiner = SpanVLMRefiner(router_cfg)
    coarse_scores = refiner.score(
        coarse_spans,
        coarse_evidences,
        use_vlm=router_uses_vlm(router_cfg, pipeline_cfg.enable_coarse_vlm_refine),
    )
    selected_coarse_spans = _select_top_coarse_spans(coarse_spans, coarse_scores, pipeline_cfg)

    candidate_nodes = _build_candidate_nodes(
        selected_coarse_spans=selected_coarse_spans,
        coarse_scores=coarse_scores,
        frame_ids=frame_ids,
        seed_arr=seed_arr,
        enable_tree=bool(pipeline_cfg.enable_tree),
        tree_cfg=tree_cfg,
    )

    fine_evidences = {
        node.node_id: build_span_evidence(
            span=node,
            images=images,
            frame_ids=frame_ids,
            output_dir=f"{output_dir}/fine",
        )
        for node in candidate_nodes
    }
    fine_scores = (
        refiner.score(
            candidate_nodes,
            fine_evidences,
            use_vlm=router_uses_vlm(router_cfg, pipeline_cfg.enable_fine_vlm_refine),
        )
        if candidate_nodes
        else {}
    )
    _apply_span_scores(candidate_nodes, fine_scores)

    selected_nodes = select_nodes_from_list(
        candidate_nodes,
        eventness_scores=seed_arr,
        track_risk_scores=track_arr,
        object_prior_scores=object_arr,
        cfg=selector_cfg,
    )
    return TreePipelineResult(
        coarse_spans=coarse_spans,
        selected_coarse_spans=selected_coarse_spans,
        coarse_evidences=coarse_evidences,
        coarse_scores=coarse_scores,
        candidate_nodes=candidate_nodes,
        fine_evidences=fine_evidences,
        fine_scores=fine_scores,
        selected_nodes=selected_nodes,
    )


def build_routed_event_tree(
    frame_ids: list[int],
    images: list[np.ndarray],
    tracks_per_frame: list[list[Any]],
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    seed_scores: list[float] | np.ndarray,
    output_dir: str,
    tree_cfg: TreeBuildConfig | None = None,
    selector_cfg: NodeSelectorConfig | None = None,
    pipeline_cfg: TreePipelineConfig | None = None,
    router_cfg: VLMRefineConfig | None = None,
) -> TreePipelineResult:
    return run_tree_pipeline(
        frame_ids=frame_ids,
        images=images,
        tracks_per_frame=tracks_per_frame,
        track_scores=track_scores,
        object_scores=object_scores,
        seed_scores=seed_scores,
        output_dir=output_dir,
        pipeline_cfg=pipeline_cfg,
        tree_cfg=tree_cfg,
        selector_cfg=selector_cfg,
        router_cfg=router_cfg,
    )
