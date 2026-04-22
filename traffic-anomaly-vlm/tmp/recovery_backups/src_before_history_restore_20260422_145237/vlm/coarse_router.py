from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.evidence.builder import evidence_image_paths
from src.schemas import EvidencePack
from src.vlm.prompts import build_span_score_prompt
from src.vlm.parser import parse_coarse_router_output

if TYPE_CHECKING:
    from src.vlm.model_loader import LocalVLM


@dataclass
class CoarseRouterConfig:
    """Reusable VLM span scorer configuration for coarse spans and fine nodes."""

    enabled: bool = False
    mode: str = "heuristic"
    max_nodes: int = 4
    max_new_tokens: int = 256
    max_image_size: int = 640
    min_confidence: float = 0.35
    prior_weight: float = 0.4
    vlm_weight: float = 0.6
    model_path: str = ""
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class CoarseRouteDecision:
    """Score fusion result for one candidate span."""

    proposal_id: str
    prior_score: float
    vlm_score: float
    fused_score: float
    confidence: float
    reason: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict:
        payload = {
            "proposal_id": str(self.proposal_id),
            "prior_score": float(self.prior_score),
            "vlm_score": float(self.vlm_score),
            "fused_score": float(self.fused_score),
            "confidence": float(self.confidence),
            "reason": str(self.reason),
        }
        if self.raw_output:
            payload["raw_output"] = str(self.raw_output)
        return payload


def _span_id(span: Any) -> str:
    return str(getattr(span, "proposal_id", getattr(span, "node_id", getattr(span, "span_id", "span"))))


def _prior_score(span: Any) -> float:
    for key in ("coarse_score", "span_prior_score", "fused_score", "eventness_peak"):
        if hasattr(span, key):
            try:
                return float(getattr(span, key))
            except Exception:
                return 0.0
    return 0.0


def _clip01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


class CoarseRouter:
    """Small wrapper that turns span evidence into scalar semantic scores."""

    def __init__(self, cfg: CoarseRouterConfig):
        self.cfg = cfg
        self._vlm: LocalVLM | None = None

    def route(
        self,
        proposals: list[Any],
        evidences: dict[str, EvidencePack],
    ) -> dict[str, CoarseRouteDecision]:
        ordered = sorted(proposals, key=_prior_score, reverse=True)
        results: dict[str, CoarseRouteDecision] = {}
        routed_ids = {_span_id(span) for span in ordered[: max(0, int(self.cfg.max_nodes))]}

        for proposal in ordered:
            proposal_id = _span_id(proposal)
            evidence = evidences.get(proposal_id)
            if proposal_id in routed_ids and self.cfg.enabled and str(self.cfg.mode).lower() == "vlm" and evidence is not None:
                decision = self._score_with_vlm(proposal, evidence)
            else:
                decision = self._prior_only_decision(proposal, reason="prior score only")
            results[proposal_id] = decision
        return results

    def _score_with_vlm(self, proposal: Any, evidence: EvidencePack) -> CoarseRouteDecision:
        prior = _prior_score(proposal)
        try:
            vlm = self._get_or_load_vlm()
            prompt = build_span_score_prompt(evidence.summary)
            from src.vlm.infer import run_inference

            raw = run_inference(
                vlm=vlm,
                prompt=prompt,
                image_paths=evidence_image_paths(evidence),
                max_new_tokens=int(self.cfg.max_new_tokens),
                max_image_size=int(self.cfg.max_image_size),
            )
            parsed = parse_coarse_router_output(raw)
            confidence = float(parsed.get("confidence", 0.0))
            if confidence < float(self.cfg.min_confidence):
                fallback = self._prior_only_decision(proposal, reason="vlm confidence too low, fallback to prior")
                fallback.raw_output = str(raw)
                return fallback

            vlm_score = _clip01(float(parsed.get("anomaly_score", prior)))
            fused = self._fuse_scores(prior, vlm_score)
            return CoarseRouteDecision(
                proposal_id=_span_id(proposal),
                prior_score=_clip01(prior),
                vlm_score=vlm_score,
                fused_score=fused,
                confidence=confidence,
                reason=str(parsed.get("reason", "")),
                raw_output=str(raw),
            )
        except Exception as exc:
            fallback = self._prior_only_decision(proposal, reason=f"vlm error fallback: {exc}")
            return fallback

    def _get_or_load_vlm(self) -> LocalVLM:
        if self._vlm is None:
            from src.vlm.model_loader import LocalVLM

            self._vlm = LocalVLM(
                model_path=str(self.cfg.model_path),
                device=str(self.cfg.device),
                dtype=str(self.cfg.dtype),
            )
        return self._vlm

    def _fuse_scores(self, prior_score: float, vlm_score: float) -> float:
        prior_score = _clip01(prior_score)
        vlm_score = _clip01(vlm_score)
        prior_weight = max(0.0, float(self.cfg.prior_weight))
        vlm_weight = max(0.0, float(self.cfg.vlm_weight))
        if prior_weight + vlm_weight <= 1e-6:
            return prior_score
        return _clip01((prior_weight * prior_score + vlm_weight * vlm_score) / (prior_weight + vlm_weight))

    @staticmethod
    def _prior_only_decision(proposal: Any, reason: str = "") -> CoarseRouteDecision:
        prior = _clip01(_prior_score(proposal))
        return CoarseRouteDecision(
            proposal_id=_span_id(proposal),
            prior_score=prior,
            vlm_score=prior,
            fused_score=prior,
            confidence=0.0,
            reason=reason,
        )
