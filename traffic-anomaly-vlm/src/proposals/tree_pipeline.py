from __future__ import annotations

from typing import Literal

import numpy as np

from src.evidence.builder import EvidenceBuilder
from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from src.proposals.tree_pipeline_components import (
    TreeBuildConfig,
    _node_from_span,
    build_event_tree_from_coarse_nodes,
    clip01,
)
from src.schemas import EventNode, EvidencePack, TrackObject, WindowFeature
from src.vlm.infer import run_inference
from src.vlm.model_loader import LocalVLM
from src.vlm.parser import (
    parse_span_score_output,
    parse_span_score_output_for_stage1,
    parse_span_score_output_for_stage2,
)
from src.vlm.prompts import build_span_score_prompt, build_stage1_prompt, build_stage2_prompt


class NodeVLMInterface:
    """Unified node-level VLM scoring with single-stage and two-stage prompts."""

    def __init__(
        self,
        vlm: LocalVLM,
        *,
        prompt_method: str = "single_stage",
        min_confidence: float = 0.35,
        prior_weight: float = 0.4,
        vlm_weight: float = 0.6,
        positive_threshold: float = 0.5,
    ):
        self.vlm = vlm
        self.prompt_method = str(prompt_method)
        self.min_confidence = float(min_confidence)
        self.prior_weight = float(prior_weight)
        self.vlm_weight = float(vlm_weight)
        self.positive_threshold = float(positive_threshold)

    def score_node(self, node: EventNode, evidence: EvidencePack) -> EventNode:
        prior = clip01(float(node.span_prior_score))
        if self.prompt_method == "two_stage":
            stage1_prompt = build_stage1_prompt(evidence.summary, evidence_method=str(evidence.summary.get("evidence_method", "montage")))
            stage1_raw = run_inference(self.vlm, stage1_prompt, evidence.keyframe_paths)
            stage1_output = parse_span_score_output_for_stage1(stage1_raw)
            stage2_prompt = build_stage2_prompt(
                stage1_output,
                summary=evidence.summary,
                evidence_method=str(evidence.summary.get("evidence_method", "montage")),
            )
            stage2_raw = run_inference(self.vlm, stage2_prompt, evidence.keyframe_paths)
            parsed = parse_span_score_output_for_stage2(stage2_raw)
        else:
            prompt = build_span_score_prompt(
                evidence.summary,
                evidence_method=str(evidence.summary.get("evidence_method", "montage")),
            )
            raw = run_inference(self.vlm, prompt, evidence.keyframe_paths)
            parsed = parse_span_score_output(raw)

        confidence = clip01(float(parsed.get("confidence", 0.0)))
        if confidence < self.min_confidence:
            node.vlm_score = prior
            node.vlm_confidence = confidence
            node.fused_score = prior
            return node

        vlm_score = clip01(float(parsed.get("anomaly_score", prior)))
        weight_sum = max(1e-6, self.prior_weight + self.vlm_weight)
        node.vlm_score = vlm_score
        node.vlm_confidence = confidence
        node.fused_score = clip01(
            (self.prior_weight * prior + self.vlm_weight * vlm_score) / weight_sum
        )
        return node

    def is_positive(self, node: EventNode) -> bool:
        prior = clip01(float(node.span_prior_score))
        vlm_score = clip01(float(node.vlm_score if node.vlm_score > 0.0 else prior))
        confidence = clip01(float(node.vlm_confidence))
        decision_score = clip01((1.0 - confidence) * prior + confidence * vlm_score)
        return decision_score >= self.positive_threshold


class TreePipeline:
    """Minimal pipeline: detect leaf spans, VLM-score leaves, merge positive leaves."""

    def __init__(
        self,
        vlm: LocalVLM,
        tree_build_cfg: TreeBuildConfig | None = None,
        boundary_cfg: BoundaryDetectorConfig | None = None,
        min_confidence: float = 0.35,
        prior_weight: float = 0.4,
        vlm_weight: float = 0.6,
        positive_threshold: float = 0.5,
        output_assets_dir: str = "./assets/evidence",
    ):
        self.tree_build_cfg = tree_build_cfg or TreeBuildConfig()
        self.boundary_detector = BoundaryDetector(boundary_cfg or BoundaryDetectorConfig())
        self.evidence_builder = EvidenceBuilder(output_assets_dir=output_assets_dir)
        self.vlm_interface = NodeVLMInterface(
            vlm,
            prompt_method=str(self.tree_build_cfg.prompt_method),
            min_confidence=min_confidence,
            prior_weight=prior_weight,
            vlm_weight=vlm_weight,
            positive_threshold=positive_threshold,
        )

    def _build_leaf_nodes(
        self,
        frame_ids: list[int],
        scores: np.ndarray,
    ) -> list[EventNode]:
        spans = self.boundary_detector.detect(scores.astype(np.float32).tolist())
        leaves: list[EventNode] = []
        for index, (start_idx, end_idx) in enumerate(spans, start=1):
            node = _node_from_span(
                node_id=f"leaf.{index}",
                level=0,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                frame_ids=frame_ids,
                eventness_scores=scores,
            )
            node.span_prior_score = 0.7 * float(node.eventness_peak) + 0.3 * float(node.eventness_mean)
            leaves.append(node)
        return leaves

    def _build_leaf_evidences(
        self,
        leaves: list[EventNode],
        *,
        images: list[str],
        frame_ids: list[int],
        all_tracks: list[TrackObject],
        windows: list[WindowFeature] | None,
        evidence_method: Literal["montage", "frames", "enhanced"],
        output_dir: str | None,
    ) -> dict[str, EvidencePack]:
        evidences: dict[str, EvidencePack] = {}
        for leaf in leaves:
            evidences[leaf.node_id] = self.evidence_builder.build(
                proposal=leaf,
                images=images,
                frame_ids=frame_ids,
                all_tracks=all_tracks,
                windows=windows,
                method=evidence_method,
                output_dir=output_dir,
            )
        return evidences

    def _score_leaf_nodes(
        self,
        leaves: list[EventNode],
        evidences: dict[str, EvidencePack],
    ) -> list[EventNode]:
        scored: list[EventNode] = []
        for leaf in leaves:
            evidence = evidences.get(leaf.node_id)
            if evidence is None:
                leaf.vlm_score = clip01(float(leaf.span_prior_score))
                leaf.fused_score = clip01(float(leaf.span_prior_score))
                scored.append(leaf)
                continue
            scored.append(self.vlm_interface.score_node(leaf, evidence))
        return scored

    def process(
        self,
        images: list[str],
        frame_ids: list[int],
        scores: list[float] | np.ndarray,
        all_tracks: list[TrackObject],
        windows: list[WindowFeature] | None = None,
        method: Literal["montage", "frames", "enhanced"] = "montage",
        output_dir: str | None = None,
    ) -> list[EventNode]:
        eventness = np.asarray(scores, dtype=np.float32)
        leaves = self._build_leaf_nodes(frame_ids, eventness)
        if not leaves:
            return []

        evidences = self._build_leaf_evidences(
            leaves,
            images=images,
            frame_ids=frame_ids,
            all_tracks=all_tracks,
            windows=windows,
            evidence_method=method,
            output_dir=output_dir,
        )
        scored_leaves = self._score_leaf_nodes(leaves, evidences)
        positive_leaves = [leaf for leaf in scored_leaves if self.vlm_interface.is_positive(leaf)]
        if not positive_leaves:
            return []

        root = build_event_tree_from_coarse_nodes(
            coarse_nodes=positive_leaves,
            frame_ids=frame_ids,
            eventness_scores=eventness,
            cfg=self.tree_build_cfg,
        )
        return list(root.children)
