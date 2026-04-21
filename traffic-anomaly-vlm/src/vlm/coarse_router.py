from __future__ import annotations

from dataclasses import dataclass

from src.evidence.coarse_router_evidence import CoarseEvidencePack
from src.proposals.coarse_proposal import CoarseProposal
from src.vlm.infer import run_inference
from src.vlm.model_loader import LocalVLM
from src.vlm.parser import parse_coarse_router_output
from src.vlm.prompt_coarse_router import build_coarse_router_prompt


@dataclass
class CoarseRouterConfig:
    enabled: bool = False
    mode: str = "heuristic"
    max_nodes: int = 4
    max_new_tokens: int = 256
    max_image_size: int = 640
    min_confidence: float = 0.35
    model_path: str = ""
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class CoarseRouteDecision:
    proposal_id: str
    contains_core_anomaly: bool
    role: str
    temporal_pattern: str
    focus_region: tuple[float, float]
    split_recommendation: str
    boundary_adjust: tuple[int, int]
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "contains_core_anomaly": bool(self.contains_core_anomaly),
            "role": str(self.role),
            "temporal_pattern": str(self.temporal_pattern),
            "focus_region": [float(self.focus_region[0]), float(self.focus_region[1])],
            "split_recommendation": str(self.split_recommendation),
            "boundary_adjust": {"left": int(self.boundary_adjust[0]), "right": int(self.boundary_adjust[1])},
            "confidence": float(self.confidence),
            "reason": str(self.reason),
        }


class CoarseRouter:
    def __init__(self, cfg: CoarseRouterConfig):
        self.cfg = cfg
        self._vlm: LocalVLM | None = None

    def route(
        self,
        proposals: list[CoarseProposal],
        evidences: dict[str, CoarseEvidencePack],
    ) -> dict[str, CoarseRouteDecision]:
        ordered = sorted(proposals, key=lambda p: float(p.coarse_score), reverse=True)
        results: dict[str, CoarseRouteDecision] = {}
        routed_ids = {p.proposal_id for p in ordered[: max(0, int(self.cfg.max_nodes))]}

        for proposal in ordered:
            evidence = evidences.get(proposal.proposal_id)
            if proposal.proposal_id in routed_ids and self.cfg.enabled and str(self.cfg.mode).lower() == "vlm" and evidence is not None:
                decision = self._route_with_vlm(proposal, evidence)
            else:
                decision = self._route_with_heuristics(proposal, evidence)
            results[proposal.proposal_id] = decision
        return results

    def _route_with_vlm(self, proposal: CoarseProposal, evidence: CoarseEvidencePack) -> CoarseRouteDecision:
        try:
            vlm = self._get_or_load_vlm()
            prompt = build_coarse_router_prompt(evidence.summary)
            raw = run_inference(
                vlm=vlm,
                prompt=prompt,
                image_paths=evidence.vlm_image_paths(),
                max_new_tokens=int(self.cfg.max_new_tokens),
                max_image_size=int(self.cfg.max_image_size),
            )
            parsed = parse_coarse_router_output(raw)
            decision = self._parsed_to_decision(proposal.proposal_id, parsed)
            if float(decision.confidence) < float(self.cfg.min_confidence):
                return self._route_with_heuristics(proposal, evidence)
            return decision
        except Exception as exc:
            heuristic = self._route_with_heuristics(proposal, evidence)
            heuristic.reason = f"heuristic fallback after vlm error: {exc}"
            return heuristic

    def _get_or_load_vlm(self) -> LocalVLM:
        if self._vlm is None:
            self._vlm = LocalVLM(
                model_path=str(self.cfg.model_path),
                device=str(self.cfg.device),
                dtype=str(self.cfg.dtype),
            )
        return self._vlm

    @staticmethod
    def _parsed_to_decision(proposal_id: str, parsed: dict) -> CoarseRouteDecision:
        boundary_adjust = parsed.get("boundary_adjust", {})
        return CoarseRouteDecision(
            proposal_id=proposal_id,
            contains_core_anomaly=bool(parsed.get("contains_core_anomaly", False)),
            role=str(parsed.get("role", "uncertain")),
            temporal_pattern=str(parsed.get("temporal_pattern", "uncertain")),
            focus_region=(
                float(parsed.get("focus_region", [0.15, 0.85])[0]),
                float(parsed.get("focus_region", [0.15, 0.85])[1]),
            ),
            split_recommendation=str(parsed.get("split_recommendation", "keep_coarse")),
            boundary_adjust=(int(boundary_adjust.get("left", 0)), int(boundary_adjust.get("right", 0))),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=str(parsed.get("reason", "")),
        )

    @staticmethod
    def _route_with_heuristics(proposal: CoarseProposal, evidence: CoarseEvidencePack | None) -> CoarseRouteDecision:
        summary = evidence.summary if evidence is not None else {}
        span_ratio = float(summary.get("span_ratio", 0.0))
        peak_ratio = float(summary.get("peak_offset_ratio", 0.5))
        source_count = len(summary.get("sources", []))
        seed_peak = float(summary.get("seed_peak", proposal.seed_peak))
        seed_mean = max(1e-6, float(summary.get("seed_mean", proposal.seed_mean)))
        peak_contrast = float(seed_peak / seed_mean)

        role = "core"
        contains_core_anomaly = True
        temporal_pattern = "sustained"
        split_recommendation = "keep_coarse"
        focus_region = (0.08, 0.92)
        confidence = 0.62
        reason = "heuristic route from span ratio and score contrast"

        if peak_contrast >= 1.5 and span_ratio <= 0.35:
            temporal_pattern = "localized"
            split_recommendation = "split_2"
            focus_half_width = 0.18 if span_ratio > 0.18 else 0.28
            focus_region = (max(0.0, peak_ratio - focus_half_width), min(1.0, peak_ratio + focus_half_width))
            confidence = 0.70
        elif source_count >= 2 and span_ratio <= 0.55:
            temporal_pattern = "multi_stage"
            split_recommendation = "split_3"
            focus_region = (0.12, 0.88)
            confidence = 0.66
        elif span_ratio >= 0.55:
            temporal_pattern = "sustained"
            split_recommendation = "keep_coarse"
            focus_region = (0.05, 0.95)
            confidence = 0.74
        else:
            temporal_pattern = "uncertain"
            split_recommendation = "split_2"
            focus_region = (0.12, 0.88)

        if proposal.coarse_score < 0.32 and max(proposal.track_peak, proposal.object_peak) < 0.28:
            role = "context"
            contains_core_anomaly = False
            temporal_pattern = "uncertain"
            split_recommendation = "keep_coarse"
            focus_region = (0.15, 0.85)
            confidence = 0.45
            reason = "low coarse score and weak object/track evidence"

        left_adjust = -4 if peak_ratio < 0.25 else 0
        right_adjust = 4 if peak_ratio > 0.75 else 0
        if temporal_pattern == "sustained":
            left_adjust = min(left_adjust, -2)
            right_adjust = max(right_adjust, 2)

        return CoarseRouteDecision(
            proposal_id=proposal.proposal_id,
            contains_core_anomaly=contains_core_anomaly,
            role=role,
            temporal_pattern=temporal_pattern,
            focus_region=(float(focus_region[0]), float(focus_region[1])),
            split_recommendation=split_recommendation,
            boundary_adjust=(int(left_adjust), int(right_adjust)),
            confidence=float(confidence),
            reason=reason,
        )
