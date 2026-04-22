from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.evidence.builder import evidence_image_paths
from src.schemas import EvidencePack

from .config import VLMRefineConfig
from .helpers import clip01
from .models import SpanScore

if TYPE_CHECKING:
    from src.vlm.model_loader import LocalVLM


class SpanVLMRefiner:
    """Internal reusable span scorer that optionally queries the VLM."""

    def __init__(self, cfg: VLMRefineConfig):
        self.cfg = cfg
        self._vlm: LocalVLM | None = None

    def score(
        self,
        spans: list[Any],
        evidences: dict[str, EvidencePack],
        use_vlm: bool,
    ) -> dict[str, SpanScore]:
        ordered = sorted(spans, key=self._prior_score, reverse=True)
        results: dict[str, SpanScore] = {}
        refined_ids = {self._span_id(span) for span in ordered[: max(0, int(self.cfg.max_nodes))]}

        for span in ordered:
            span_id = self._span_id(span)
            evidence = evidences.get(span_id)
            if span_id in refined_ids and use_vlm and evidence is not None:
                decision = self._score_with_vlm(span, evidence)
            else:
                decision = self._prior_only_score(span, reason="prior score only")
            results[span_id] = decision
        return results

    def route(
        self,
        proposals: list[Any],
        evidences: dict[str, EvidencePack],
    ) -> dict[str, SpanScore]:
        """Compatibility wrapper for the previous coarse-router method name."""

        return self.score(
            spans=proposals,
            evidences=evidences,
            use_vlm=router_uses_vlm(self.cfg, enabled_flag=True),
        )

    def _score_with_vlm(self, span: Any, evidence: EvidencePack) -> SpanScore:
        prior = self._prior_score(span)
        try:
            vlm = self._get_or_load_vlm()
            from src.vlm.infer import run_inference
            from src.vlm.parser import parse_coarse_router_output
            from src.vlm.prompts import build_span_score_prompt

            raw = run_inference(
                vlm=vlm,
                prompt=build_span_score_prompt(evidence.summary),
                image_paths=evidence_image_paths(evidence),
                max_new_tokens=int(self.cfg.max_new_tokens),
                max_image_size=int(self.cfg.max_image_size),
            )
            parsed = parse_coarse_router_output(raw)
            confidence = float(parsed.get("confidence", 0.0))
            if confidence < float(self.cfg.min_confidence):
                fallback = self._prior_only_score(span, reason="vlm confidence too low, fallback to prior")
                fallback.raw_output = str(raw)
                return fallback

            vlm_score = clip01(float(parsed.get("anomaly_score", prior)))
            fused_score = self._fuse_scores(prior, vlm_score)
            return SpanScore(
                proposal_id=self._span_id(span),
                prior_score=clip01(prior),
                vlm_score=vlm_score,
                fused_score=fused_score,
                confidence=confidence,
                reason=str(parsed.get("reason", "")),
                raw_output=str(raw),
            )
        except Exception as exc:
            return self._prior_only_score(span, reason=f"vlm error fallback: {exc}")

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
        prior_weight = max(0.0, float(self.cfg.prior_weight))
        vlm_weight = max(0.0, float(self.cfg.vlm_weight))
        if prior_weight + vlm_weight <= 1e-6:
            return clip01(prior_score)
        return clip01((prior_weight * clip01(prior_score) + vlm_weight * clip01(vlm_score)) / (prior_weight + vlm_weight))

    @staticmethod
    def _span_id(span: Any) -> str:
        return str(getattr(span, "proposal_id", getattr(span, "node_id", getattr(span, "span_id", "span"))))

    @staticmethod
    def _prior_score(span: Any) -> float:
        for key in ("prior_score", "coarse_score", "span_prior_score", "fused_score", "eventness_peak"):
            if hasattr(span, key):
                try:
                    return float(getattr(span, key))
                except Exception:
                    return 0.0
        return 0.0

    @classmethod
    def _prior_only_score(cls, span: Any, reason: str = "") -> SpanScore:
        prior = clip01(cls._prior_score(span))
        return SpanScore(
            proposal_id=cls._span_id(span),
            prior_score=prior,
            vlm_score=prior,
            fused_score=prior,
            confidence=0.0,
            reason=reason,
        )


def router_uses_vlm(cfg: VLMRefineConfig, enabled_flag: bool) -> bool:
    return bool(cfg.enabled and enabled_flag and str(cfg.mode).lower() == "vlm")
