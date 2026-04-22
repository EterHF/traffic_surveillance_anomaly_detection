from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from src.evidence.builder import EvidenceBuilder
from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from src.schemas import EventNode, EvidencePack, TrackObject, WindowFeature

from src.proposals.tree_pipeline_components import (
	NodeSelectorConfig,	
	TreeBuildConfig,
	_node_from_span,
	# build_event_tree_from_root,
    build_event_tree_from_coarse_nodes,
	flatten_event_tree,
    clip01
)

from src.vlm.infer import run_inference
from src.vlm.model_loader import LocalVLM
from src.vlm.parser import parse_span_score_output
from src.vlm.prompts import build_span_score_prompt, build_stage1_prompt, build_stage2_prompt


class UnifiedVLMInterface:
	"""Single VLM entry for node scoring with selectable prompts in src.vlm.prompts."""

	def __init__(
		self,
		vlm: LocalVLM,
		min_confidence: float = 0.35,
		prior_weight: float = 0.4,
		vlm_weight: float = 0.6,
	):
		self.vlm = vlm
		self.min_confidence = float(min_confidence)
		self.prior_weight = float(prior_weight)
		self.vlm_weight = float(vlm_weight)

	def _get_vlm(self) -> LocalVLM:
		return self.vlm

	def _prompt_builder(self, prompt_type: str) -> Callable:
		if prompt_type == "span":
			return build_span_score_prompt
		elif prompt_type == "stage1":
			return build_stage1_prompt
		elif prompt_type == "stage2":
			return build_stage2_prompt
		raise ValueError(f"unsupported prompt_type: {prompt_type}")

	def score_node(self, node: EventNode, evidence: EvidencePack, prompt_method: str = "single_stage") -> EventNode:
		prior = clip01(float(node.span_prior_score))

		if prompt_method == "single_stage":
			prompt = self._prompt_builder(prompt_type="span")(evidence.summary)
			raw = run_inference(
				vlm=self._get_vlm(),
				prompt=prompt,
				image_paths=evidence.keyframe_paths
			)
			parsed = parse_span_score_output(raw)
			confidence = clip01(float(parsed.get("confidence", 0.0)))
			if confidence < float(self.min_confidence): # FIXME
				node.vlm_score = prior
				node.vlm_confidence = confidence
				node.fused_score = prior
				return node

			vlm_score = clip01(float(parsed.get("anomaly_score", prior)))
			pw = max(0.0, float(self.prior_weight))
			vw = max(0.0, float(self.vlm_weight))
			fused = prior if (pw + vw) <= 1e-6 else clip01((pw * prior + vw * vlm_score) / (pw + vw))

			node.vlm_score = vlm_score
			node.vlm_confidence = confidence
			node.fused_score = fused
			return node
		else:
			raise NotImplementedError(f"Unsupported prompt_method: {prompt_method}")


def refine_event_nodes_with_vlm(
	nodes: list[EventNode],
	evidences: dict[str, EvidencePack],
	vlm: UnifiedVLMInterface,
	prompt_method: str = "single_stage"
) -> list[EventNode]:
	refined: list[EventNode] = []
	for node in nodes:
		evidence = evidences.get(node.node_id)
		if evidence is None:
			node.vlm_score = clip01(node.span_prior_score)
			node.fused_score = clip01(node.span_prior_score)
			refined.append(node)
			continue
		refined.append(vlm.score_node(node, evidence, prompt_method=prompt_method))
	return refined


class TreePipeline:
	def __init__(self, 
			  vlm: LocalVLM,
			  tree_build_cfg: TreeBuildConfig | None = None,
			  boundary_cfg: BoundaryDetectorConfig | None = None,
			  min_confidence: float = 0.35,
			  prior_weight: float = 0.4,
			  vlm_weight: float = 0.6,
			  output_assets_dir: str = "./assets/evidence",
			  ):
		self.tree_build_cfg = tree_build_cfg
		self.vlm_interface = UnifiedVLMInterface(vlm=vlm, min_confidence=min_confidence, prior_weight=prior_weight, vlm_weight=vlm_weight)
		self.boundary_detector = BoundaryDetector(cfg=boundary_cfg)
		self.evidence_builder = EvidenceBuilder(output_assets_dir=output_assets_dir)

	def process(self, 
			 images: list[str],
			 frame_ids: list[int], 
			 scores: list[float] | np.ndarray,
			 all_tracks: list[TrackObject],
			 windows: list[WindowFeature] | None = None,
			 method: Literal["montage", "overlay"] = "montage",
			 output_dir: str | None = None,
			 ) -> list[EventNode]:
		raw_spans = self.boundary_detector.detect(np.asarray(scores, dtype=np.float32).tolist())
		coarse_nodes = []
		for idx, (start_idx, end_idx) in enumerate(raw_spans, start=1):
			node = _node_from_span(
				node_id=str(idx),
				level=1,
				start_idx=start_idx,
				end_idx=end_idx,
				frame_ids=frame_ids,
				eventness_scores=scores,
			)
			node.span_prior_score = 0.7 * node.eventness_peak + 0.3 * node.eventness_mean # FIXME
			coarse_nodes.append(node)

		if self.tree_build_cfg.refine_coarse:
			evidences = {node.node_id: self.evidence_builder.build(
				proposal=node,
				images=images,
				frame_ids=frame_ids,
				all_tracks=all_tracks,
				windows=windows,
				method=method,
				output_dir=output_dir,
			) for node in coarse_nodes}
			
			refined_coarse = refine_event_nodes_with_vlm(coarse_nodes, 
												evidences, 
												self.vlm_interface,
												self.tree_build_cfg.prompt_method)
		else:
			refined_coarse = coarse_nodes
		
		tree = build_event_tree_from_coarse_nodes(refined_coarse, frame_ids, scores, self.tree_build_cfg)
		all_nodes = flatten_event_tree(tree)

		return all_nodes
